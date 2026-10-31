# -*- coding: utf-8 -*-
"""固定N1后端，接入单通道、作者式输入线宽的NAF终态；审计不进入results。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.boundary_rgb import BoundaryRGBNAFNet
from models.boundary_restore import state_sha256
from tools import infer_semifinal as entry
from tools.boundary_rgb_monitor import predict_boundary_rgb_tiled
from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from tools.semantic_marker_diagnostic import marker_stages
from utils.marker_seed_support import filter_unsupported_seeds

FORMAT = 'boundary_width_inference_v1'
CHECKPOINT_FORMAT = 'boundary_rgb_training_v1'
EXPECTED_MODEL_SHA = '20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8'
NOGAN_RIDGE_MODEL_SHA = '8eb7c8da49f9e04f6c7f5537f064e4cb7317bec047e0e0411d9c18ca655be283'
INPUT_CONTRACT = dict(input_mode='boundary', input_threshold=.5, tile_size=256,
    overlap=64, reflect_pad=32, batch_size=16, iterations=1, revert_unconnected=False,
    boundary_representation='binary', output_stitch='probability',
    input_normalization=dict(mode='skeleton_dilate', kernel_size=8))
NUMERICAL_CONTRACT = dict(precision='float32', cudnn_tf32=False, matmul_tf32=False,
                         cudnn_benchmark=False, cudnn_deterministic=True)
MAX_PARAMETERS = 500_000_000


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _exact_contract(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(_exact_contract(value[k], v) for k, v in expected.items())
    return value == expected


def validate_input_contract(contract):
    if not _exact_contract(contract, INPUT_CONTRACT):
        raise ValueError('Unknown width-normalized NAF input contract')
    return deepcopy(contract)


def load_naf_checkpoint(path, device='cuda', *, expected_model_sha=None):
    """只严格加载G；不恢复优化器、训练RNG或判别器。"""
    path = Path(path).resolve()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('format') != CHECKPOINT_FORMAT or type(saved.get('global_step')) is not int
            or saved['global_step'] != 1000):
        raise ValueError('Expected formal boundary_rgb_training_v1 step1000')
    cfg, architecture = saved.get('config'), saved.get('architecture')
    if (not isinstance(cfg, dict) or not isinstance(cfg.get('model'), dict)
            or cfg.get('input_mode') != 'boundary'
            or architecture != dict(cfg['model'], seed=cfg.get('seed'), input_mode='boundary')):
        raise ValueError('NAF checkpoint architecture/config mismatch')
    contract = validate_input_contract(saved.get('input_contract'))
    if (contract != cfg.get('inference')
            or cfg.get('input_normalization') != INPUT_CONTRACT['input_normalization']):
        raise ValueError('NAF checkpoint input_contract/config mismatch')
    numerical = saved.get('identity', {}).get('numerical_contract')
    if not _exact_contract(numerical, NUMERICAL_CONTRACT):
        raise ValueError('NAF checkpoint numerical contract mismatch')
    digest = saved.get('model_sha256')
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise ValueError('NAF checkpoint model SHA256 missing')
    if expected_model_sha is not None and digest != expected_model_sha:
        raise ValueError('NAF model SHA256 is not the selected candidate')
    model = BoundaryRGBNAFNet(**architecture)
    if model.architecture != architecture:
        raise ValueError('NAF architecture changed during construction')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != digest:
        raise RuntimeError('NAF loaded state SHA256 differs')
    model = model.to(device).eval().requires_grad_(False)
    provenance = dict(format=FORMAT, checkpoint_format=CHECKPOINT_FORMAT,
        checkpoint=str(path), checkpoint_sha256=entry.sha(path), checkpoint_bytes=path.stat().st_size,
        global_step=1000, architecture=deepcopy(architecture), model_sha256=digest,
        parameter_count=sum(p.numel() for p in model.parameters()), input_contract=contract,
        numerical_contract=deepcopy(numerical), no_optimizer_or_rng_restored=True)
    return model, provenance


@contextmanager
def naf_runtime_context():
    """仅NAF匹配训练精度；退出后恢复N1前端设置，不强开其他确定性算法。"""
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = old


def _sources():
    names = ['tools/infer_boundary_rgb.py', 'tools/infer_semifinal.py',
             'tools/boundary_rgb_monitor.py', 'models/boundary_rgb.py', 'models/boundary_naf.py',
             'models/boundary_restore.py', 'utils/boundary_input.py',
             'tools/probe_affinity_marker_replay.py', 'tools/semantic_marker_diagnostic.py',
             'tools/probe_affinity_partition.py', 'tools/affinity_connectivity_views.py',
             'utils/post_process.py', 'utils/marker_seed_support.py',
             'utils/instance_area_policy.py', 'utils/final_instance_cleanup.py',
             'tools/affinity_native_views.py', 'tools/semantic_crossover.py']
    return {name: entry.sha(ROOT / name) for name in names}


def _frozen(model):
    return all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters())


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _timing_summary(records, key):
    values = np.asarray([r[key] for r in records], np.float64)
    return dict(count=int(values.size), sum_seconds=float(values.sum()),
        mean_seconds=float(values.mean()) if values.size else None,
        median_seconds=float(np.median(values)) if values.size else None,
        p90_seconds=float(np.percentile(values, 90)) if values.size else None)


def _s0_expected_model_sha(expected):
    """旧缺省仍锁20a59；新候选只能显式指定已核验的完整G哈希。"""
    selected = EXPECTED_MODEL_SHA if expected is None else expected
    if selected not in (EXPECTED_MODEL_SHA, NOGAN_RIDGE_MODEL_SHA):
        raise ValueError('S0 requires the fixed 20a59 or verified NoGAN 8eb selected model SHA')
    return selected


def _supported_readout(semantic, after, reference, config):
    """仅撤销原N1<=.45支撑为零的reflect种子，不改概率/高程或末端raw票。"""
    stages = marker_stages(after, config)
    reflected = build_marker_variant(stages, 'reflect', 32)
    original_markers = reflected['marker_labels']
    filtered, audit = filter_unsupported_seeds(original_markers, reference, low_threshold=.45)
    retained = np.isin(original_markers, audit['retained_ids']) & (original_markers > 0)
    if (not np.array_equal(filtered[retained], original_markers[retained])
            or np.any(filtered[~retained] != 0)):
        raise RuntimeError('S0 changed retained marker IDs/positions or introduced seed pixels')
    replay = decode_marker_replay(semantic, torch.from_numpy(after)[None, None], config, filtered)
    capture = replay['capture_audit']
    if (not np.array_equal(replay['actual_markers'], filtered)
            or capture.get('actual_watershed_calls') != 1
            or not capture.get('original_marker_exact')
            or not capture.get('original_elevation_exact')
            or capture.get('real_barrier_changed') is not False):
        raise RuntimeError('S0 replay changed the frozen watershed contract')
    audit.update(mode='original-zero', reference='original_N1_pre_NAF_boundary',
        reference_raw_array_sha256=array_sha(reference), marker_source='NAF_probability_actual_reflect32_markers',
        marker_reflection_padding=32, retained_marker_pixels_and_IDs_exact=True,
        readout_audit=capture, readout_hashes=replay['hashes'])
    return (replay['instances'], replay['classes']), audit


def run(args, *, device='cuda'):
    started = time.perf_counter()
    if getattr(args, 'save_analysis', False) or getattr(args, 'capture_all_maps', False):
        raise ValueError('Use bounded --capture-names, not full-map analysis')
    args = argparse.Namespace(**vars(args))
    args.save_analysis = False
    seed_support = getattr(args, 'seed_support', 'none')
    if seed_support not in ('none', 'original-zero'):
        raise ValueError('--seed-support must be none or original-zero')
    args.seed_support = seed_support
    expected_model_sha = getattr(args, 'expected_model_sha', None)
    if seed_support == 'original-zero':
        expected_model_sha = _s0_expected_model_sha(expected_model_sha)
    cfg = entry.load_config(str(Path(args.config).resolve()))
    if getattr(args, 'dataset', None):
        cfg['submission']['dataset'] = args.dataset
    entry.validate_config(cfg)
    if entry.local_restoration(cfg) != dict(mode='zero', model='restorer', candidate='N1'):
        raise ValueError('Width candidate requires fixed N1 backend')
    if cfg['submission']['models'].get('terminal_semantic') is not None:
        raise ValueError('Width candidate keeps original semantic head')
    files = entry.input_files(Path(args.input_dir).resolve(), args.only, cfg['submission'].get('dataset'))
    names = getattr(args, 'capture_names', None) or []
    if len(set(names)) != len(names) or not set(names) <= {p.stem for p in files}:
        raise ValueError('--capture-names must be distinct selected sources')
    if len(names) > 4:
        raise ValueError('Bounded captures allow at most four fixed sources')
    captures = set(names)
    output = Path(args.output_dir).resolve()
    # 否则entry拒绝已有目录后，外围失败标记可能误写上一轮审计文件。
    if output.exists():
        raise FileExistsError('Use a fresh output directory; existing artifacts remain untouched')
    _synchronize(device)
    then = time.perf_counter()
    naf, provenance = load_naf_checkpoint(args.naf_checkpoint, device,
        expected_model_sha=expected_model_sha)
    _synchronize(device)
    naf_load_seconds = time.perf_counter() - then
    if not _frozen(naf):
        raise RuntimeError('NAF must remain frozen/eval')
    source_hashes, initial_state = _sources(), state_sha256(naf)
    original_predict, original_reflect, original_load_models = entry.predict, entry.decode_reflect, entry.load_models
    original_revote = entry.views.revote_instances
    current = dict(source=None, rgb=None, calls=0, raw_calls=0, record=None, capture=None)
    records, base_counts, load_times = [], [], []

    def revote(instances, raw_probability, *values, **options):
        if current['source'] is None or current['raw_calls']:
            raise RuntimeError('Expected exactly one raw probability vote per source')
        current['raw_calls'] += 1
        digest = array_sha(raw_probability)
        result = original_revote(instances, raw_probability, *values, **options)
        if array_sha(raw_probability) != digest:
            raise RuntimeError('Raw vote modified frozen probabilities')
        current['record']['raw_probability_sha256'] = digest
        if current['capture'] is not None:
            current['capture'][1]['raw_probability'] = np.array(raw_probability, copy=True)
        return result

    def load_models(config, paths, actual_device):
        _synchronize(actual_device)
        then = time.perf_counter()
        models = original_load_models(config, paths, actual_device)
        _synchronize(actual_device)
        load_times.append(time.perf_counter() - then)
        count = sum(p.numel() for m in models if m is not None for p in m.parameters())
        if count + provenance['parameter_count'] >= MAX_PARAMETERS:
            raise RuntimeError('Backend plus NAF parameters exceed competition limit')
        base_counts.append(count)
        return models

    def decode(semantic, boundary, config, *, original_decode=None):
        if current['source'] is None or current['calls']:
            raise RuntimeError('Expected exactly one final NAF decode per source')
        current['calls'] += 1
        if (not torch.is_tensor(boundary) or boundary.dtype != torch.float32
                or boundary.ndim != 4 or tuple(boundary.shape[:2]) != (1, 1)):
            raise ValueError('NAF decode requires FP32 B1HW boundary')
        before = np.array(boundary.detach().cpu()[0, 0].numpy(), copy=True)
        rgb = current['rgb']
        if rgb.shape != (*before.shape, 3):
            raise ValueError('Preloaded native RGB shape differs from stitched boundary')
        before_sha, rgb_sha = array_sha(before), array_sha(rgb)
        semantic_sha = array_sha(semantic.detach().cpu().contiguous().numpy())
        _synchronize(device)
        then = time.perf_counter()
        with naf_runtime_context():
            after, infer = predict_boundary_rgb_tiled(before, rgb, naf, provenance['input_contract'], device=device)
            _synchronize(device)
        naf_seconds = time.perf_counter() - then
        if array_sha(before) != before_sha or array_sha(rgb) != rgb_sha:
            raise RuntimeError('NAF inference modified its input boundary/RGB')
        if (not isinstance(after, np.ndarray) or after.dtype != np.float32 or after.shape != before.shape
                or not np.isfinite(after).all() or np.any((after < 0) | (after > 1))):
            raise RuntimeError('NAF output probability contract differs')
        after_sha = array_sha(after)
        if infer.get('output_sha256') != after_sha:
            raise RuntimeError('NAF output SHA audit differs')
        record = dict(source=current['source'], shape=list(before.shape),
            input_boundary_sha256=before_sha, output_boundary_sha256=after_sha,
            input_rgb_sha256=rgb_sha, semantic_sha256=semantic_sha, inference=infer,
            naf_seconds=naf_seconds, seeds_and_barriers_regenerated=True, seed_support='none')
        current['record'] = record
        if current['source'] in captures:
            folder = output / 'audit/naf_captures'
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (current['source'] + '.npz')
            if path.exists():
                raise FileExistsError('Refuse capture overwrite')
            current['capture'] = (path, dict(boundary_before=before, boundary_after=after,
                semantic=semantic.detach().cpu().contiguous().numpy()))
            record['capture'] = path.relative_to(output).as_posix()
        then = time.perf_counter()
        if seed_support == 'original-zero':
            result, record['seed_support'] = _supported_readout(semantic, after, before, config)
        else:
            result = original_reflect(semantic, torch.from_numpy(after)[None, None], config,
                                      original_decode=original_decode)
        _synchronize(device)
        record['readout_seconds'] = time.perf_counter() - then
        if (array_sha(after) != after_sha or array_sha(boundary.detach().cpu()[0, 0].numpy()) != before_sha
                or array_sha(semantic.detach().cpu().contiguous().numpy()) != semantic_sha):
            raise RuntimeError('Decode modified frozen probability input')
        return result

    def predict(model, restorer, terminal, path, config, actual_device, **kwargs):
        if current['source'] is not None:
            raise RuntimeError('Nested source prediction unsupported')
        # decode在原predict内部发生，所以必须先加载RGB，而非等待原predict返回。
        bgr = entry.read_image(path, cv2.IMREAD_COLOR)
        if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('Native RGB image cannot be read')
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        file_sha = entry.sha(path)
        current.update(source=Path(path).stem, rgb=rgb, calls=0, raw_calls=0, record=None, capture=None)
        try:
            _synchronize(actual_device)
            then = time.perf_counter()
            actual_rgb, result = original_predict(model, restorer, terminal, path, config, actual_device, **kwargs)
            _synchronize(actual_device)
            if current['calls'] != 1 or current['raw_calls'] != 1 or current['record'] is None:
                raise RuntimeError('Source must perform exactly one NAF decode and raw vote')
            if not np.array_equal(actual_rgb, rgb) or entry.sha(path) != file_sha:
                raise RuntimeError('Native source RGB/file changed during prediction')
            if array_sha(result['boundary']) != current['record']['input_boundary_sha256']:
                raise RuntimeError('Returned upstream boundary differs')
            if 'final_area_cleanup' in result:
                current['record']['final_area_cleanup'] = result['final_area_cleanup']
            current['record'].update(predict_seconds=time.perf_counter()-then, input_file_sha256=file_sha)
            if current['capture'] is not None:
                path, values = current['capture']
                required = sum(v.nbytes for v in values.values()) + 3 * 1024**3
                if shutil.disk_usage(path.parent).free < required:
                    raise OSError('Capture needs arrays plus 3GiB fast-disk reserve')
                np.savez_compressed(path, **values)
                current['record']['capture_arrays'] = sorted(values)
            records.append(current['record'])
            return actual_rgb, result
        finally:
            current.update(source=None, rgb=None, calls=0, raw_calls=0, record=None, capture=None)

    try:
        then = time.perf_counter()
        with patch.object(entry, 'load_models', side_effect=load_models), \
             patch.object(entry, 'decode_reflect', side_effect=decode), \
             patch.object(entry, 'predict', side_effect=predict), \
             patch.object(entry.views, 'revote_instances', side_effect=revote):
            entry.run(args)
        _synchronize(device)
        entry_seconds = time.perf_counter() - then
        if (len(base_counts) != 1 or len(records) != len(files)
                or [r['source'] for r in records] != [p.stem for p in files]):
            raise RuntimeError('NAF source/model invocation count differs')
        if (not _frozen(naf) or state_sha256(naf) != initial_state
                or entry.sha(provenance['checkpoint']) != provenance['checkpoint_sha256']
                or _sources() != source_hashes):
            raise RuntimeError('NAF checkpoint/model/source changed')
        audit = output / 'audit'
        report_path = audit / 'report.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if (not report.get('complete') or not report.get('frozen_models_unchanged')
                or report['parameter_count'] != base_counts[0] or 'boundary_naf' in report['models']
                or len(report['images']) != len(records)):
            raise RuntimeError('Base inference report differs or NAF parameters counted twice')
        for image, record in zip(report['images'], records):
            if image['source'] != record['source']:
                raise RuntimeError('Image report source order differs')
            image['boundary_naf'] = record
        total = base_counts[0] + provenance['parameter_count']
        removed_seed_count = sum(r['seed_support']['removed_seed_count'] for r in records
                                 if isinstance(r['seed_support'], dict))
        timing = dict(device=str(device), precision='FP32', upstream_runtime=deepcopy(report.get('runtime')),
            predict=_timing_summary(records, 'predict_seconds'), naf=_timing_summary(records, 'naf_seconds'),
            readout=_timing_summary(records, 'readout_seconds'), naf_load_seconds=naf_load_seconds,
            backend_load_seconds=sum(load_times), entry_run_wall_seconds=entry_seconds,
            run_wall_seconds=time.perf_counter()-started,
            scope='predict includes upstream, NAF and CPU readout/raw vote; excludes saves and packaging')
        receipt = dict(complete=True, format=FORMAT, training=False, naf=provenance,
            naf_frozen_unchanged=True, imported_sources=source_hashes, upstream_parameter_count=base_counts[0],
            parameter_count=total, total_parameters=total, images=records, captures=sorted(captures), timing=timing,
            returned_boundary_is_upstream=True, seed_support=seed_support, removed_seed_count=removed_seed_count,
            geometry='N1 stitched B -> whole threshold/skeleton/ellipse8 -> single-channel NAF -> fresh reflect32 seeds/barriers -> unchanged WS/raw vote')
        if seed_support == 'original-zero':
            selected_label = 'fixed20a59' if initial_state == EXPECTED_MODEL_SHA else 'fixed' + initial_state[:8]
            receipt['geometry'] = ('N1 stitched B -> whole threshold/skeleton/ellipse8 -> ' + selected_label + ' NAF -> '
                'fresh reflect32 seeds/barriers -> remove only seeds with zero original_N1<=.45 support -> unchanged WS/raw vote')
        report.update(upstream_geometry=report['geometry'], geometry=receipt['geometry'],
            upstream_parameter_count=base_counts[0], parameter_count=total, parameter_count_includes_naf=True,
            naf_frozen_unchanged=True, naf_complete=True, returned_boundary_is_upstream=True, naf_timing=timing,
            seed_support=seed_support, removed_seed_count=removed_seed_count)
        report['models']['boundary_naf'] = provenance
        report['model_states']['boundary_naf'] = initial_state
        entry.write(audit / 'naf_report.json', receipt)
        entry.write(report_path, report)
        status_path = audit / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        status.update(naf_complete=True, parameter_count=total, parameter_count_includes_naf=True,
                      seed_support=seed_support, removed_seed_count=removed_seed_count)
        entry.write(status_path, status)
        return receipt
    except Exception as error:
        audit = output / 'audit'
        if audit.is_dir():
            entry.write(audit / 'naf_failed.json', dict(complete=False, format=FORMAT,
                error=repr(error), completed_naf_images=len(records)))
            for name in ('status.json', 'report.json'):
                p = audit / name
                if p.is_file():
                    document = json.loads(p.read_text(encoding='utf-8'))
                    document.update(complete=False, naf_complete=False, naf_error=repr(error))
                    if name == 'status.json':
                        document['status'] = 'failed'
                    entry.write(p, document)
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
    parser.add_argument('--expected-model-sha', default=EXPECTED_MODEL_SHA)
    parser.add_argument('--capture-names', nargs='+')
    parser.add_argument('--seed-support', choices=('none', 'original-zero'), default='none',
                        help='Explicit S0: remove only unsupported reflect seeds using original N1<=.45')
    result = run(parser.parse_args(argv))
    print(json.dumps(dict(complete=True, format=FORMAT, images=len(result['images']),
                         parameter_count=result['parameter_count'])), flush=True)


if __name__ == '__main__':
    main()
