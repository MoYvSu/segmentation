# -*- coding: utf-8 -*-
"""固定N1后端，在完整边界拼接后接入Region1000 NAF；审计和候选缓存不进入results。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from models.boundary_naf import BoundaryNAFNet
from models.boundary_restore import state_sha256
from tools import infer_semifinal as entry
from tools.boundary_naf_monitor import predict_boundary_tiled
from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from tools.semantic_marker_diagnostic import marker_stages
from utils.marker_seed_support import filter_unsupported_seeds

FORMAT = 'boundary_naf_region_inference_v1'
CHECKPOINT_FORMAT = 'boundary_region_training_v1'
INPUT_CONTRACT = dict(input_mode='skeleton', input_threshold=.5, tile_size=256,
                      overlap=64, reflect_pad=32, batch_size=16, iterations=1,
                      revert_unconnected=False)
MAX_PARAMETERS = 500_000_000


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def validate_input_contract(contract):
    if (not isinstance(contract, dict) or set(contract) != set(INPUT_CONTRACT)
            or any(type(contract[k]) is not type(v) or contract[k] != v
                   for k, v in INPUT_CONTRACT.items())):
        raise ValueError('Unknown Region1000 NAF input contract')
    return deepcopy(contract)


def load_naf_checkpoint(path, device='cuda'):
    """只加载G，不恢复训练优化器/RNG；构造器封存并恢复CPU随机流。"""
    path = Path(path).resolve()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('format') != CHECKPOINT_FORMAT
            or type(saved.get('global_step')) is not int or saved['global_step'] != 1000):
        raise ValueError('Expected formal boundary_region_training_v1 step1000')
    cfg = saved.get('config')
    architecture = saved.get('architecture')
    if (not isinstance(cfg, dict) or not isinstance(cfg.get('model'), dict)
            or not isinstance(architecture, dict)
            or architecture != dict(cfg['model'], seed=cfg.get('seed'))):
        raise ValueError('NAF checkpoint architecture/config mismatch')
    contract = validate_input_contract(saved.get('input_contract'))
    if contract != cfg.get('inference'):
        raise ValueError('NAF checkpoint input_contract/config mismatch')
    digest = saved.get('model_sha256')
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise ValueError('NAF checkpoint model SHA256 is missing')
    model = BoundaryNAFNet(**architecture)
    if model.architecture != architecture:
        raise ValueError('NAF checkpoint architecture changed during construction')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != digest:
        raise RuntimeError('NAF loaded state SHA256 differs')
    model = model.to(device).eval().requires_grad_(False)
    provenance = dict(format=FORMAT, checkpoint_format=CHECKPOINT_FORMAT,
        checkpoint=str(path), checkpoint_sha256=entry.sha(path),
        checkpoint_bytes=path.stat().st_size, global_step=1000,
        architecture=deepcopy(architecture), model_sha256=digest,
        parameter_count=sum(p.numel() for p in model.parameters()),
        input_contract=contract, no_optimizer_or_rng_restored=True)
    return model, provenance


def _sources(*, include_relation=False):
    names = ['tools/infer_boundary_naf.py', 'tools/infer_semifinal.py',
             'tools/boundary_naf_monitor.py', 'models/boundary_naf.py',
             'models/boundary_restore.py', 'tools/probe_affinity_marker_replay.py',
             'tools/semantic_marker_diagnostic.py', 'tools/probe_affinity_partition.py',
             'tools/affinity_connectivity_views.py', 'utils/marker_seed_support.py',
             'utils/post_process.py', 'tools/affinity_native_views.py',
             'tools/semantic_crossover.py']
    if include_relation:
        names += ['models/boundary_relation.py', 'utils/boundary_relation_inference.py',
                  'utils/boundary_relation_merge.py', 'utils/boundary_relation_supervision.py']
    return {name: entry.sha(ROOT / name) for name in names}


def _relation_options(args, config):
    """关系头只在显式开启时接到C终态之后；默认原链完全不变。"""
    options = config.get('boundary_relation', {})
    if not isinstance(options, dict):
        raise ValueError('boundary_relation配置必须为对象')
    checkpoint = getattr(args, 'relation_checkpoint', None)
    enabled = checkpoint is not None or options.get('enabled', False)
    if not enabled:
        if any(getattr(args, name, None) is not None for name in
               ('relation_threshold', 'relation_batch_size', 'relation_expected_model_sha')):
            raise ValueError('未启用关系头时不能传关系超参数')
        return None
    if checkpoint is None:
        filename = options.get('checkpoint_file')
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError('关系权重须是model目录内的单个文件名')
        model_dir = Path(args.model_dir).resolve() if args.model_dir else ROOT/'model'
        checkpoint = model_dir/filename
    def value(name, key, default):
        override = getattr(args, name, None)
        return options.get(key, default) if override is None else override
    return dict(checkpoint=checkpoint,
        threshold=value('relation_threshold', 'threshold', .1),
        batch_size=value('relation_batch_size', 'batch_size', 64),
        expected_model_sha=value('relation_expected_model_sha', 'expected_model_sha', None))


def _apply_relation_result(result, rgb, before, after, processor):
    """在raw语义投票与final200之后合并，不重新投票或清理。"""
    ids, classes, audit = processor.process(rgb, before, after, result['instances'], result['classes'])
    entry.verify_prediction(ids, classes, rgb.shape[:2])
    output = dict(result, instances=ids, classes=classes)
    output['phases'] = entry.phase_description(ids, classes)
    if 'instance_scores' in output:
        output['instance_scores'] = {key: result['instance_scores'][key] for key in classes}
        output['instance_score_policy'] = 'relation_survivor_original_score_inherited_not_recomputed'
    return output, audit


def _frozen(model):
    return all(not module.training for module in model.modules()) and all(
        not parameter.requires_grad for parameter in model.parameters())


def _synchronize(device):
    """CUDA工作是异步提交的，计时前后等待完成；CPU不调用CUDA。"""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _timing_summary(records, key):
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    return dict(count=int(values.size), sum_seconds=float(values.sum()),
        mean_seconds=float(values.mean()) if values.size else None,
        median_seconds=float(np.median(values)) if values.size else None,
        p90_seconds=float(np.percentile(values, 90)) if values.size else None)


def _supported_readout(semantic, after, reference, config):
    """只去掉在原N1低响应区零支撑的实际reflect32种子；不修改连续热图。"""
    stages = marker_stages(after, config)
    reflected = build_marker_variant(stages, 'reflect', 32)
    original_markers = reflected['marker_labels']
    threshold = float(config['inference'].get('marker_boundary_low_threshold', .45))
    filtered, audit = filter_unsupported_seeds(original_markers, reference,
                                               low_threshold=threshold)
    # 仅允许删除完整种子；保留ID/位置，不增添或改写任何概率和真实障碍。
    retained = np.isin(original_markers, audit['retained_ids']) & (original_markers > 0)
    if (not np.array_equal(filtered[retained], original_markers[retained])
            or np.any(filtered[~retained] != 0)):
        raise RuntimeError('Seed filter changed retained IDs/positions or introduced seed pixels')
    replay = decode_marker_replay(semantic, torch.from_numpy(after)[None, None],
                                  config, filtered)
    capture = replay['capture_audit']
    if (not np.array_equal(replay['actual_markers'], filtered)
            or capture.get('actual_watershed_calls') != 1
            or not capture.get('original_marker_exact')
            or not capture.get('original_elevation_exact')
            or capture.get('real_barrier_changed') is not False):
        raise RuntimeError('Filtered seed replay changed frozen watershed contract')
    audit.update(mode='original-zero', reference='original_N1_pre_NAF_boundary',
        reference_raw_array_sha256=array_sha(reference),
        marker_source='NAF_probability_actual_reflect32_markers',
        marker_reflection_padding=32, readout_audit=capture,
        readout_hashes=replay['hashes'])
    return (replay['instances'], replay['classes']), audit


def run(args, *, device='cuda'):
    """沿用原150/100入口及读出，只替换传给reflect解码的完整边界。"""
    run_started = time.perf_counter()
    if getattr(args, 'save_analysis', False):
        raise ValueError('Use --capture-names for bounded NAF captures, not --save-analysis')
    args = argparse.Namespace(**vars(args))
    args.save_analysis = False
    seed_support = getattr(args, 'seed_support', 'none')
    if seed_support not in ('none', 'original-zero'):
        raise ValueError('Unknown --seed-support mode')
    args.seed_support = seed_support
    config = entry.load_config(str(Path(args.config).resolve()))
    if getattr(args, 'dataset', None):
        config['submission']['dataset'] = args.dataset
    entry.validate_config(config)
    if entry.local_restoration(config) != dict(mode='zero', model='restorer', candidate='N1'):
        raise ValueError('Region1000 candidate requires the fixed N1 backend')
    if config['submission']['models'].get('terminal_semantic') is not None:
        raise ValueError('Region1000 candidate keeps the original semantic head')
    relation_options = _relation_options(args, config)
    relation = None
    if relation_options is not None:
        area = entry.resolve_instance_area_policy(config['inference'])
        if (seed_support != 'original-zero' or area['final_min_instance_area'] != 200
                or area['final_small_instance_action'] != 'neighbor_merge_v1'):
            raise ValueError('关系头须接在原N1/旧NAF/S0/raw类别/final200的C终态之后')
        from utils.boundary_relation_inference import BoundaryRelationPostprocessor
        relation = BoundaryRelationPostprocessor(device=device, **relation_options)
    files = entry.input_files(Path(args.input_dir).resolve(), args.only,
                              config['submission'].get('dataset'))
    capture_names = getattr(args, 'capture_names', None) or []
    capture_all_maps = bool(getattr(args, 'capture_all_maps', False))
    if capture_all_maps and capture_names:
        raise ValueError('--capture-all-maps and --capture-names are mutually exclusive')
    if len(set(capture_names)) != len(capture_names) or not set(capture_names) <= {p.stem for p in files}:
        raise ValueError('--capture-names must be distinct names selected for this run')
    captures = {p.stem for p in files} if capture_all_maps else set(capture_names)
    output = Path(args.output_dir).resolve()
    _synchronize(device)
    model_started = time.perf_counter()
    naf, provenance = load_naf_checkpoint(args.naf_checkpoint, device)
    if relation is not None and provenance['model_sha256'] != '20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8':
        raise ValueError('关系头训练特征固定旧20a59 NAF，不能静默接其他修复器')
    _synchronize(device)
    naf_load_seconds = time.perf_counter() - model_started
    if not _frozen(naf):
        raise RuntimeError('NAF must remain frozen/eval')
    source_hashes = _sources(include_relation=relation is not None)
    initial_state = state_sha256(naf)
    original_predict = entry.predict
    original_reflect = entry.decode_reflect
    original_load_models = entry.load_models
    original_revote = entry.views.revote_instances
    current = dict(source=None, calls=0, record=None, capture=None, raw_calls=0, raw_probability=None, relation_maps=None)
    records = []
    base_parameter_counts = []
    backend_load_seconds = []

    def revote(instances, raw_probability, *values, **options):
        if current['source'] is None or current['raw_calls']:
            raise RuntimeError('Expected exactly one raw probability vote per source')
        current['raw_calls'] += 1
        raw = np.array(raw_probability, copy=True)
        if (raw.dtype != np.float32 or raw.ndim != 2 or not np.isfinite(raw).all()
                or np.any((raw < 0) | (raw > 1))):
            raise ValueError('Raw vote probability must be finite native FP32 HW in [0,1]')
        result = original_revote(instances, raw_probability, *values, **options)
        if not np.array_equal(raw, raw_probability):
            raise RuntimeError('Original raw vote modified its probability map')
        current['raw_probability'] = raw
        return result

    def load_models(config, paths, actual_device):
        _synchronize(actual_device)
        started = time.perf_counter()
        models = original_load_models(config, paths, actual_device)
        _synchronize(actual_device)
        backend_load_seconds.append(time.perf_counter() - started)
        count = sum(p.numel() for module in models if module is not None
                    for p in module.parameters())
        relation_count = relation.parameter_count if relation is not None else 0
        if count + provenance['parameter_count'] + relation_count >= MAX_PARAMETERS:
            raise RuntimeError('Backend plus NAF parameters exceed competition limit')
        base_parameter_counts.append(count)
        return models

    def decode(semantic, boundary, config, *, original_decode=None):
        if current['source'] is None or current['calls']:
            raise RuntimeError('Expected exactly one final boundary decode per source')
        current['calls'] += 1
        if (not torch.is_tensor(boundary) or boundary.dtype != torch.float32
                or boundary.ndim != 4 or tuple(boundary.shape[:2]) != (1, 1)):
            raise ValueError('NAF decode requires original FP32 B1HW boundary')
        before = np.array(boundary.detach().cpu()[0, 0].numpy(), copy=True)
        before_hash = array_sha(before)
        semantic_hash = array_sha(semantic.detach().cpu().contiguous().numpy())
        _synchronize(device)
        naf_started = time.perf_counter()
        after, inference = predict_boundary_tiled(
            before, naf, provenance['input_contract'], device=device)
        _synchronize(device)
        naf_seconds = time.perf_counter() - naf_started
        if array_sha(before) != before_hash:
            raise RuntimeError('NAF inference modified its input boundary')
        if (not isinstance(after, np.ndarray) or after.dtype != np.float32
                or after.shape != before.shape or not np.isfinite(after).all()
                or np.any((after < 0) | (after > 1))):
            raise RuntimeError('NAF inference output contract differs')
        after_hash = array_sha(after)
        if inference.get('output_sha256') != after_hash:
            raise RuntimeError('NAF inference audit output SHA differs')
        if relation is not None:
            current['relation_maps'] = (before, after)
        record = dict(source=current['source'], shape=list(before.shape),
            input_boundary_sha256=before_hash, output_boundary_sha256=after_hash,
            semantic_sha256=semantic_hash, inference=inference, naf_seconds=naf_seconds)
        if current['source'] in captures:
            folder = output / 'audit/naf_captures'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (current['source'] + '.npz')
            if path.exists():
                raise FileExistsError('Refuse capture overwrite: ' + str(path))
            current['capture'] = (path, dict(boundary_before=before, boundary_after=after,
                                semantic=semantic.detach().cpu().contiguous().numpy()))
            record['capture'] = path.relative_to(output).as_posix()
        # 直接调用保存的reflect函数；不能再经evaluate_frozen_readout查回当前hook。
        _synchronize(device)
        readout_started = time.perf_counter()
        if seed_support == 'original-zero':
            result, record['seed_support'] = _supported_readout(semantic, after, before, config)
        else:
            result = original_reflect(semantic, torch.from_numpy(after)[None, None], config,
                                      original_decode=original_decode)
            record['seed_support'] = dict(mode='none', removed_seed_count=0,
                                         reference=None, unchanged_saved_reflect=True)
        _synchronize(device)
        record['readout_seconds'] = time.perf_counter() - readout_started
        if (array_sha(after) != after_hash
                or array_sha(boundary.detach().cpu()[0, 0].numpy()) != before_hash
                or array_sha(semantic.detach().cpu().contiguous().numpy()) != semantic_hash):
            raise RuntimeError('Final decode modified frozen boundary/semantic input')
        current['record'] = record
        return result

    def predict(model, restorer, terminal, path, config, actual_device, **kwargs):
        if current['source'] is not None:
            raise RuntimeError('Nested source prediction is not supported')
        current.update(source=Path(path).stem, calls=0, record=None, capture=None,
                       raw_calls=0, raw_probability=None, relation_maps=None)
        try:
            _synchronize(actual_device)
            predict_started = time.perf_counter()
            rgb, result = original_predict(model, restorer, terminal, path, config, actual_device, **kwargs)
            _synchronize(actual_device)
            predict_seconds = time.perf_counter() - predict_started
            if current['calls'] != 1 or current['record'] is None:
                raise RuntimeError('Source did not perform exactly one NAF boundary decode')
            # 原predict_views仍返回上游B，实际IDs来自上面替换后的边界。
            if array_sha(result['boundary']) != current['record']['input_boundary_sha256']:
                raise RuntimeError('Unexpected upstream result boundary changed')
            current['record']['predict_seconds'] = predict_seconds
            if relation is not None:
                if current['relation_maps'] is None:
                    raise RuntimeError('关系头缺少原N1/旧NAF连续输入')
                _synchronize(actual_device)
                relation_started = time.perf_counter()
                result, relation_audit = _apply_relation_result(
                    result, rgb, *current['relation_maps'], relation)
                _synchronize(actual_device)
                relation_seconds = time.perf_counter() - relation_started
                folder = output/'audit/relation'; folder.mkdir(parents=True, exist_ok=True)
                relation_path = folder/(current['source']+'.json')
                if relation_path.exists():
                    raise FileExistsError('拒绝覆盖关系审计：'+str(relation_path))
                entry.write(relation_path, relation_audit)
                current['record']['boundary_relation'] = dict(
                    audit=relation_path.relative_to(output).as_posix(), audit_sha256=entry.sha(relation_path),
                    seconds=relation_seconds, threshold=relation.threshold,
                    scored_pairs=len(relation_audit['pairs']),
                    selected_pairs=relation_audit['merge']['selected_count'],
                    filled_seam_pixels=relation_audit['merge']['filled_seam_pixels'])
            if capture_all_maps and (current['raw_calls'] != 1 or current['raw_probability'] is None):
                raise RuntimeError('Source did not perform exactly one raw vote for full-map capture')
            if current['capture'] is not None:
                capture_path, values = current['capture']
                if capture_all_maps:
                    raw = current['raw_probability']
                    if raw.shape != values['boundary_before'].shape:
                        raise RuntimeError('Raw probability/native boundary shape differs')
                    values['raw_probability'] = raw
                    current['record']['raw_probability_sha256'] = array_sha(raw)
                # 不依赖压缩率预估，写完仍预留至少512MiB；不主动删除任何文件。
                required = sum(value.nbytes for value in values.values()) + 513 * 1024**2
                if shutil.disk_usage(capture_path.parent).free < required:
                    raise OSError('Insufficient capture disk space: need arrays plus 512MiB reserve')
                np.savez_compressed(capture_path, **values)
                current['record']['capture_arrays'] = sorted(values)
            records.append(current['record'])
            return rgb, result
        finally:
            current.update(source=None, calls=0, record=None, capture=None,
                           raw_calls=0, raw_probability=None, relation_maps=None)

    try:
        entry_started = time.perf_counter()
        with patch.object(entry, 'load_models', side_effect=load_models), \
                patch.object(entry, 'decode_reflect', side_effect=decode), \
                patch.object(entry, 'predict', side_effect=predict), \
                (patch.object(entry.views, 'revote_instances', side_effect=revote)
                 if capture_all_maps else nullcontext()):
            entry.run(args)
        _synchronize(device)
        entry_run_wall_seconds = time.perf_counter() - entry_started
        if (len(base_parameter_counts) != 1 or len(records) != len(files)
                or [r['source'] for r in records] != [p.stem for p in files]):
            raise RuntimeError('NAF/model invocation/source coverage differs')
        if (not _frozen(naf) or state_sha256(naf) != initial_state
                or entry.sha(provenance['checkpoint']) != provenance['checkpoint_sha256']
                or _sources(include_relation=relation is not None) != source_hashes):
            raise RuntimeError('NAF checkpoint/model/source changed during inference')
        audit = output / 'audit'
        report_path = audit / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if (not report.get('complete') or not report.get('frozen_models_unchanged')
                or report['parameter_count'] != base_parameter_counts[0]
                or 'boundary_naf' in report['models']
                or len(report['images']) != len(records)):
            raise RuntimeError('Original inference receipt differs or NAF was counted twice')
        for image, record in zip(report['images'], records):
            if image['source'] != record['source']:
                raise RuntimeError('NAF/source report order differs')
            image['boundary_naf'] = record
        if relation is not None:
            relation.verify_unchanged()
            relation_provenance = relation.provenance()
            if (relation_provenance.get('loaded_parameter_count', 0)
                    and entry.sha(relation_provenance['checkpoint']) != relation_provenance['checkpoint_sha256']):
                raise RuntimeError('关系checkpoint文件在推理期间改变')
        total_parameters = base_parameter_counts[0] + provenance['parameter_count'] + (
            relation.parameter_count if relation is not None else 0)
        timing = dict(device=str(device), precision='FP32',
            upstream_runtime=deepcopy(report.get('runtime')),
            predict_scope='original_predict including upstream models, NAF, seed filter, CPU watershed and raw vote; excludes PNG/JSON/capture saves and packaging; not pure GPU time',
            readout_scope='NAF output to decoded instances, including seed filtering and CPU watershed; raw voting remains in original_predict',
            cuda_synchronized=torch.device(device).type == 'cuda',
            predict=_timing_summary(records, 'predict_seconds'),
            naf=_timing_summary(records, 'naf_seconds'),
            readout=_timing_summary(records, 'readout_seconds'),
            naf_load_seconds=naf_load_seconds, backend_load_seconds=sum(backend_load_seconds),
            model_load_seconds=naf_load_seconds + sum(backend_load_seconds),
            entry_run_wall_seconds=entry_run_wall_seconds,
            run_wall_seconds=time.perf_counter() - run_started,
            run_wall_scope='wrapper start through upstream output saves and final integrity checks; excludes final audit JSON serialization and external packaging')
        if relation is not None:
            durations = [dict(relation_seconds=r['boundary_relation']['seconds'],
                              integrated_seconds=r['predict_seconds']+r['boundary_relation']['seconds']) for r in records]
            timing['relation'] = _timing_summary(durations, 'relation_seconds')
            timing['predict_plus_relation'] = _timing_summary(durations, 'integrated_seconds')
            timing['relation_scope'] = 'relation scores and ID merge; first used image includes lazy checkpoint load; excludes audit serialization'
            timing['predict_plus_relation_scope'] = 'original predict plus relation scoring and merge; excludes PNG/JSON/audit saves and packaging'
        receipt = dict(complete=True, format=FORMAT, training=False,
            naf=provenance, naf_frozen_unchanged=True, imported_sources=source_hashes,
            upstream_parameter_count=base_parameter_counts[0], parameter_count=total_parameters,
            images=records, captures=sorted(captures), capture_all_maps=capture_all_maps,
            seed_support=seed_support, timing=timing,
            returned_boundary_is_upstream=True,
            geometry='N1 stitched boundary -> global skeleton -> Region1000 NAF -> unchanged reflect32/WS/raw vote')
        if seed_support == 'original-zero':
            receipt['geometry'] = ('N1 stitched boundary -> global skeleton -> Region1000 NAF '
                '-> reflect32 markers -> original N1 zero-low-support seed deletion '
                '-> unchanged NAF real barrier/WS/raw vote')
        report['upstream_geometry'] = report['geometry']
        report['geometry'] = receipt['geometry']
        report['upstream_parameter_count'] = base_parameter_counts[0]
        report['parameter_count'] = total_parameters
        report['parameter_count_includes_naf'] = True
        report['naf_frozen_unchanged'] = True
        report['naf_complete'] = True
        report['models']['boundary_naf'] = provenance
        report['model_states']['boundary_naf'] = initial_state
        report['returned_boundary_is_upstream'] = True
        report['seed_support'] = seed_support
        report['naf_timing'] = timing
        if relation is not None:
            relation_receipt = dict(format='boundary_relation_entry_v1', complete=True,
                provenance=relation.provenance(), stage='after_raw_vote_and_final200',
                parameter_count=relation.parameter_count,
                images=[dict(source=r['source'], **r['boundary_relation']) for r in records],
                no_revote=True, no_second_area_cleanup=True, no_WS_rerun=True,
                all_scores_from_original_C_geometry=True, original_raw_classes_inherited=True)
            receipt['boundary_relation'] = relation_receipt
            receipt['geometry'] += ' -> one-pass same-predicted-class non-overlapping relation merges'
            report['geometry'] = receipt['geometry']
            report['boundary_relation_complete'] = True
            report['models']['boundary_relation'] = relation.provenance()
            if relation_provenance.get('loaded_parameter_count', 0):
                report['model_states']['boundary_relation'] = relation_provenance['model_sha256']
            report['parameter_count_includes_relation'] = True
            report['relation_frozen_unchanged'] = True
            entry.write(audit/'relation_report.json', relation_receipt)
        entry.write(audit / 'naf_report.json', receipt)
        entry.write(report_path, report)
        status_path = audit / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        status.update(naf_complete=True, parameter_count=total_parameters)
        if relation is not None:
            status['boundary_relation_complete'] = True
        entry.write(status_path, status)
        return receipt
    except Exception as exc:
        # 原run完成后若包装层验收失败，也不能留下可被误当作完整候选的状态。
        audit = output / 'audit'
        if audit.is_dir():
            entry.write(audit / 'naf_failed.json', dict(complete=False, format=FORMAT,
                error=repr(exc), completed_naf_images=len(records)))
            status_path = audit / 'status.json'
            if status_path.is_file():
                status = json.loads(status_path.read_text(encoding='utf-8'))
                status.update(complete=False, status='failed', naf_complete=False, naf_error=repr(exc))
                entry.write(status_path, status)
            report_path = audit / 'report.json'
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding='utf-8'))
                report['base_inference_complete_before_naf_check'] = report.get('complete', False)
                report.update(complete=False, naf_complete=False, naf_error=repr(exc))
                entry.write(report_path, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default=str(ROOT / 'config/inference/semifinal_n1.yaml'))
    parser.add_argument('--model-dir')
    parser.add_argument('--dataset', choices=('quarter_final_100', 'semifinal_150'))
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--naf-checkpoint', required=True)
    parser.add_argument('--seed-support', choices=('none', 'original-zero'), default='none',
                        help='original-zero deletes only reflect32 seeds with zero original N1 low-boundary support')
    parser.add_argument('--capture-names', nargs='+',
                        help='Only these selected sources get private before/after/semantic NPZ captures')
    parser.add_argument('--capture-all-maps', action='store_true',
                        help='Capture all sources with before/after/semantic/raw probability after timed prediction')
    parser.add_argument('--relation-checkpoint', help='显式启用冻结1000步关系头，接在C终态之后')
    parser.add_argument('--relation-threshold', type=float, default=None)
    parser.add_argument('--relation-batch-size', type=int, default=None)
    parser.add_argument('--relation-expected-model-sha', default=None)
    receipt = run(parser.parse_args(argv))
    print(json.dumps(dict(complete=True, format=FORMAT, images=len(receipt['images']),
                         parameter_count=receipt['parameter_count'])), flush=True)


if __name__ == '__main__':
    main()
