# -*- coding: utf-8 -*-
"""固定train659弱界的只读原49推理捕获；GT干预仅供定位，不能提交。

只各执行一次R1/coverage native1024正式predict_views。Hook和wrapper只观察
原返回值；短/全关系反事实使用原CUDA logits的stride与融合/裁剪函数，
不再前向、不改参数、不改实际预测。缓存可在同sam2_env进行CPU重放。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_affinity_coverage import (CAPTURE_SHA, REPORT_SHA, EXPECTED_CANDIDATE,
    Predictions, capture_config_projection, normalized_config, pin_files, quantiles, read, sha, write)
from tools.probe_affinity_partition import correct_local_relations, representative
from train_backend_adaptation import fusion_options, tensor_digest
from utils.affinity_deployment import probability_to_logit
from utils.affinity_fusion import affinity_boundary_probability, _soft_dilated_support
from utils.offset_letterbox import letterbox_instance_geometry
from utils.patch_diagnostic import BlendMap, tile_boxes

R1_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
SOURCE = 'train_659.jpg'
SEED = 314184
PAIR = (69, 70)
SUBARMS = ('baseline', 'short_correct', 'all_correct')


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def contact_mask(gt):
    """仅GT69/70原生四邻直接接触的两个已知端点。"""
    gt = np.asarray(gt)
    if gt.ndim != 2 or not np.issubdtype(gt.dtype, np.integer):
        raise ValueError('GT must be an integer HW map')
    result = np.zeros(gt.shape, bool)
    for a, b, ta, tb in ((gt[:, :-1], gt[:, 1:], result[:, :-1], result[:, 1:]),
                         (gt[:-1], gt[1:], result[:-1], result[1:])):
        selected = ((a == PAIR[0]) & (b == PAIR[1])) | ((a == PAIR[1]) & (b == PAIR[0]))
        ta |= selected; tb |= selected
    return result


def threshold_stats(boundary, mask, threshold):
    """统计实际部署CPU logit→sigmoid严格>；旧>=回执单独保留。"""
    values = np.asarray(boundary, np.float32)[mask]
    deployed = torch.sigmoid(probability_to_logit(torch.from_numpy(values.copy()))).numpy()
    return dict(pixels=int(values.size), boundary_quantiles=quantiles(values),
        deployed_quantiles=quantiles(deployed), strict_above=int((deployed > threshold).sum()),
        equal=int((deployed == threshold).sum()), below=int((deployed < threshold).sum()),
        historical_raw_greater_equal=int((values >= threshold).sum()),
        rule='CPU probability_to_logit then sigmoid; strict >')


def component_maps(logits, config):
    mode, kwargs = fusion_options(config)
    if mode != 'gated' or kwargs['short_reduction'] != 'top2':
        raise RuntimeError('Only fixed original gated/top2 fusion accepted')
    negative = 1. - logits.sigmoid()
    short = negative[:, :4].topk(k=2, dim=1).values.mean(1, keepdim=True)
    d2 = negative[:, 4:6].mean(1, keepdim=True)
    d4 = negative[:, 6:8].mean(1, keepdim=True)
    w2 = kwargs['distance2_weight'] * _soft_dilated_support(short, 2, kwargs['support_threshold'], kwargs['support_temperature'])
    w4 = kwargs['distance4_weight'] * _soft_dilated_support(short, 4, kwargs['support_threshold'], kwargs['support_temperature'])
    return dict(short=short, distance2=d2, distance4=d4, weighted2=w2, weighted4=w4)


def relation_stats(logits, chosen):
    probability = (1. - logits.sigmoid())[chosen]
    return dict(selected_edges=int(chosen.sum()), original_disconnection_probability=quantiles(probability.numpy()),
        original_disconnection_gt_half=int((probability > .5).sum()),
        per_channel=[dict(channel=c, selected=int(chosen[0, c].sum()),
            original_disconnection_probability=quantiles((1. - logits[0, c].sigmoid())[chosen[0, c]].numpy()))
            for c in range(8)])


def verify_protected(protected):
    for name, digest in protected.items():
        if sha(name) != digest:
            raise RuntimeError('Protected input/source changed: ' + name)


def capture_once(core, model, restorer, path, config, gt, output, arm):
    """原正式推理一次；CUDA融合反事实保持原logits内存stride。"""
    views = core.coverage_original_views
    boxes = tile_boxes(gt.shape, 1024, config['affinity_native']['overlap'])
    if len(boxes) != 12 or tuple(gt.shape) != (1936, 2584):
        raise RuntimeError('Fixed original 12-tile train659 geometry differs')
    folder = output / arm; folder.mkdir()
    records, pending, observed, watershed = [], {}, {}, []
    original_grid, original_crop, original_maps = views.boundary_grid, views.crop_letterbox_output, views.prediction_maps
    original_watershed = cv2.watershed
    mode, kwargs = fusion_options(config)
    case = dict(kind='merge', gt_ids=list(PAIR))

    def head_hook(_module, _inputs, result):
        if pending.get('active'):
            logits = result['affinity_logits']
            if tuple(logits.shape) != (1, 8, 512, 512) or logits.dtype != torch.float32:
                raise RuntimeError('Actual affinity logits must be FP32 [1,8,512,512]')
            if 'logits' in pending:
                raise RuntimeError('More than one affinity forward in actual tile')
            pending['logits'] = logits.detach()
        # Returning None preserves the original PyTorch forward result.

    def maps_wrapper(*args, **kw):
        value = original_maps(*args, **kw)
        observed['semantic'] = value[1].detach().cpu().numpy().copy()
        return value

    def watershed_wrapper(image, markers):
        before = markers.copy()
        value = original_watershed(image, markers)
        watershed.append(dict(markers=before, watershed=np.maximum(value, 0).copy()))
        return value

    def grid_wrapper(*args, **kw):
        index = len(records)
        if index >= len(boxes) or pending.get('active'):
            raise RuntimeError('Actual tile order differs')
        pending.clear(); pending['active'] = True
        try:
            grid = original_grid(*args, **kw)
        finally:
            pending['active'] = False
        logits = pending.pop('logits')
        # torch.where preserves the actual layout; establish an exact empty intervention gate.
        empty = torch.zeros_like(logits, dtype=torch.bool)
        zero = torch.where(empty, logits.new_tensor(-16.), logits)
        recomputed = affinity_boundary_probability(zero.float(), mode=mode, **kwargs)
        if not torch.equal(recomputed, grid):
            raise RuntimeError('CUDA empty-mask fusion is not exact; refusing counterfactual')
        y, x, y1, x1 = boxes[index]
        grid_gt, valid, metadata = letterbox_instance_geometry(gt[y:y1, x:x1], 1024, 512)
        potential = bool(np.isin(gt[y:y1, x:x1], PAIR).any())
        row = dict(index=index, box=list(boxes[index]), seed=SEED + 100000 + 1024 * 1000 + index,
            relevant=False, logits_shape=list(logits.shape), logits_stride=list(logits.stride()),
            empty_mask_grid_exact=True, geometry=metadata.to_dict(), map_file=f'{arm}/tile_{index:02d}.npz')
        arrays = dict(grid=grid.detach().cpu().numpy().copy(), grid_gt=grid_gt, valid=valid)
        pending.update(row=row, arrays=arrays, grid=grid, zero_grid=recomputed, counter_grids={})
        if potential:
            cpu = logits.detach().cpu().float()
            selections = {}
            for channels, name in (('short', 'short_correct'), ('all', 'all_correct')):
                _corrected, chosen, counts = correct_local_relations(cpu, grid_gt, valid, case, channels)
                selections[name] = (chosen, counts)
            # Seeing one GT in a tile is not interface coverage. Persist logits only
            # when an actual legal, entirely-known GT69/70 relation is selectable.
            row['relevant'] = any(counts['selected_edges'] > 0 for chosen, counts in selections.values())
            row['interface_selection_counts'] = {name: counts for name, (chosen, counts) in selections.items()}
            if row['relevant']:
                arrays['logits'] = cpu.numpy().copy()
            for name, (chosen, counts) in selections.items():
                if not row['relevant']:
                    continue
                selected = chosen.to(logits.device)
                corrected = torch.where(selected, logits.new_tensor(-16.), logits)
                if not torch.equal(corrected[~selected], logits[~selected]):
                    raise RuntimeError('Intervention touched unselected logits')
                fused = affinity_boundary_probability(corrected.float(), mode=mode, **kwargs)
                pending['counter_grids'][name] = fused
                arrays['grid_' + name] = fused.detach().cpu().numpy().copy()
                row[name] = dict(**counts, relation=relation_stats(cpu, chosen),
                    corrected_stride=list(corrected.stride()))
            # Scalar short/d2/d4 maps are small; preserve actual CUDA components for the crossing.
            if row['relevant']:
                for name, tensor in component_maps(logits.float(), config).items():
                    arrays[name] = tensor.detach().cpu().numpy().copy()
        return grid

    def crop_wrapper(*args, **kw):
        value = original_crop(*args, **kw)
        if not pending.get('row'):
            if 'raw_probability' in observed:
                raise RuntimeError('Unexpected crop before actual tiles')
            observed['raw_probability'] = value.detach().float().cpu().sigmoid()[0, 0].numpy().copy()
            return value
        row, arrays = pending['row'], pending['arrays']
        if args[0] is not pending['grid']:
            raise RuntimeError('Actual cropped tile differs from captured grid')
        arrays['native'] = value.detach().cpu().numpy().copy()
        empty_native = original_crop(pending['zero_grid'], *args[1:], **kw)
        if not torch.equal(empty_native, value):
            raise RuntimeError('Empty-intervention native crop is not exact')
        arrays['native_empty'] = empty_native.detach().cpu().numpy().copy()
        for name, grid in pending['counter_grids'].items():
            arrays['native_' + name] = original_crop(grid, *args[1:], **kw).detach().cpu().numpy().copy()
        row['empty_mask_native_exact'] = True
        row['pad_h'], row['pad_w'] = int(args[2]), int(args[3])
        geometry = row['geometry']
        if (row['pad_h'] != 1024 - geometry['resized_height']
                or row['pad_w'] != 1024 - geometry['resized_width']):
            raise RuntimeError('Actual RGB letterbox padding and GT geometry differ')
        np.savez_compressed(output / row['map_file'], **arrays)
        records.append(row); pending.clear()
        print(json.dumps(dict(stage='tile', arm=arm, index=len(records), total=len(boxes),
            relevant=row['relevant'])), flush=True)
        return value

    fixed = {}
    hook = model.affinity_decoder.register_forward_hook(head_hook)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(views, 'boundary_grid', side_effect=grid_wrapper))
            stack.enter_context(patch.object(views, 'crop_letterbox_output', side_effect=crop_wrapper))
            stack.enter_context(patch.object(views, 'prediction_maps', side_effect=maps_wrapper))
            stack.enter_context(patch('utils.post_process.cv2.watershed', side_effect=watershed_wrapper))
            with torch.no_grad():
                rgb, actual = core.predict_views(model, restorer, path, config, 'cuda', SEED, [1024], fixed_audit=fixed)
    finally:
        hook.remove()
    if len(records) != 12 or sum(r['relevant'] for r in records) != 2 or set(actual) != {'patch1024'}:
        raise RuntimeError('Actual single-source 12 tiles / two relevant tiles differ')
    if set(observed) != {'semantic', 'raw_probability'}:
        raise RuntimeError('Both original-path fixed whole maps must be observed')
    if len(watershed) != 1:
        raise RuntimeError('Expected one actual formal watershed call')
    if fixed != dict(geometry_semantic_sha256=array_sha(observed['semantic']),
                     raw_probability_sha256=array_sha(observed['raw_probability'])):
        raise RuntimeError('Observed maps differ from actual fixed audit')
    value = actual['patch1024']; stitched, blends = {}, {}
    for name in SUBARMS:
        blend = BlendMap(gt.shape)
        for row in records:
            with np.load(output / row['map_file'], allow_pickle=False) as arrays:
                key = 'native' if name == 'baseline' or not row['relevant'] else 'native_' + name
                blend.add(row['box'], arrays[key][0, 0])
        stitched[name], blends[name] = blend.finish()
        blends[name]['tiles'] = len(records)
    if not np.array_equal(stitched['baseline'], value['boundary']) or blends['baseline'] != value['blend']:
        raise RuntimeError('Actual original-order baseline BlendMap is not exact')
    empty_blend = BlendMap(gt.shape)
    for row in records:
        with np.load(output / row['map_file'], allow_pickle=False) as arrays:
            empty_blend.add(row['box'], arrays['native_empty'][0, 0])
    empty_boundary, empty_audit = empty_blend.finish(); empty_audit['tiles'] = len(records)
    if not np.array_equal(empty_boundary, value['boundary']) or empty_audit != value['blend']:
        raise RuntimeError('Empty-intervention original-order BlendMap is not exact')
    if blends['baseline']['tiles_max_overlap'] != 9:
        raise RuntimeError('Original max-overlap9 differs')
    np.savez_compressed(folder / 'maps.npz', semantic=observed['semantic'], raw_probability=observed['raw_probability'],
                        **watershed[0],
                        **{name: boundary for name, boundary in stitched.items()})
    if not cv2.imwrite(str(folder / 'baseline_inst.png'), value['instances']):
        raise RuntimeError('Cannot save actual uint16 baseline')
    write(folder / 'baseline_class.json', value['classes'])
    return value, dict(arm=arm, epoch=20, source=SOURCE, seed=SEED, tiles=records,
        fixed_whole_semantics=fixed, blend=blends, single_predict_views_call=True,
        no_hook_return_change=True, original_order_baseline_stitch_exact=True,
        empty_mask_grid_native_and_stitch_exact=True,
        intervention_fusion_device='cuda', output_shape=list(gt.shape), actual_watershed_calls=1,
        actual_marker_sha256=array_sha(watershed[0]['markers']),
        actual_watershed_sha256=array_sha(watershed[0]['watershed']))


def capture(args):
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend
    from tools.analyze_affinity_tail_gt import existing_project_sources
    from tools.run_affinity_sweep import runtime_contract
    from tools.run_affinity_short import precision_record
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_affinity_coverage import load_original_core, reference_config
    from utils.config import load_config, project_path

    output, run, cache = Path(args.output).resolve(), Path(args.run_root).resolve(), Path(args.reference_cache).resolve()
    if not output.is_relative_to(ROOT / 'outputs/affinity_weak') or output.exists():
        raise RuntimeError('Fresh private outputs/affinity_weak output required')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual new-server sam2_env GPU required')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    pipeline, status = read(run / 'pipeline_status.json'), read(run / 'candidate/status.json')
    if (pipeline.get('status') != 'complete' or pipeline.get('stage') != 'done'
            or status.get('status') != 'complete' or status.get('epoch') != 20
            or status.get('final_checkpoint_sha256') != EXPECTED_CANDIDATE
            or any(status.get(k) is not True for k in ('frozen_unchanged', 'restorer_unchanged', 'strict_reload_equal'))):
        raise RuntimeError('Completed fixed coveragee20 receipt required')
    config = status['config']; base = reference_config(config)
    if status['config'] != pipeline['config']:
        raise RuntimeError('Training config receipt differs')
    protected = {str(Path(__file__).resolve()): sha(__file__)}
    if len(pipeline['sources']) != 157 or pipeline['sources'] != pipeline['contract']['sources']:
        raise RuntimeError('Exactly registered source157 required')
    for name, digest in pipeline['sources'].items():
        for path in (ROOT / name, run / 'source' / name):
            protected[str(path)] = digest
    for path in (run / 'pipeline_status.json', run / 'candidate/status.json', cache / 'capture_manifest.json',
                 cache / 'capture.json', cache / 'report.json'):
        protected[str(path)] = sha(path)
    pin_files(cache, read(cache / 'capture_manifest.json'))
    if sha(cache / 'capture.json') != CAPTURE_SHA or sha(cache / 'report.json') != REPORT_SHA:
        raise RuntimeError('Full original32 R1 cache identity differs')
    old = read(cache / 'capture.json')
    if (not old['complete'] or len(old['rows']) != 32 or len(old['seed_map']) != 32
            or old['checkpoint_sha256'] != R1_SHA
            or capture_config_projection(old['config'], load_config(str(ROOT / 'config/train/affinity_tail.yaml')))
               != normalized_config(base)):
        raise RuntimeError('Full original32 R1 native contract differs')
    row = next(r for r in old['rows'] if r['source'] == SOURCE)
    if row['seed'] != SEED or old['seed_map'][SOURCE] != SEED:
        raise RuntimeError('Fixed original source seed314184 differs')
    path, gt_path = (Path(project_path(config, row[key])) for key in ('source_path', 'gt_path'))
    protected[str(path)], protected[str(gt_path)] = row['source_file_sha256'], row['gt_file_sha256']
    path_receipt_file = Path(args.path_receipt).resolve()
    path_receipt = read(path_receipt_file)
    if (path_receipt.get('complete') is not True or path_receipt.get('actual_marker_exact') is not True
            or path_receipt.get('source') != SOURCE or path_receipt.get('gt_ids') != list(PAIR)
            or path_receipt.get('source_file_sha256') != protected[str(path)]
            or path_receipt.get('gt_file_sha256') != protected[str(gt_path)]
            or path_receipt.get('cache_file_sha256') != sha(cache / row['map_file'])
            or len(path_receipt.get('path_crossings', [])) != 1):
        raise RuntimeError('Private fixed marker-path receipt identity differs')
    protected[str(path_receipt_file)] = sha(path_receipt_file)
    fixed_points = path_receipt['points']
    crossing = [path_receipt['path_crossings'][0][key] for key in ('a', 'b')]
    rgb = read_rgb(str(path)); gt, lookup = load_completed_semantic_source(gt_path, rgb.shape[:2])
    gt_classes = {int(g): int(lookup[g]) for g in np.unique(gt) if g > 0}
    if [representative((gt == g) & (cv2.imread(str(cache / row['mask_file']), cv2.IMREAD_UNCHANGED) == 25))
            for g in PAIR] != fixed_points:
        raise RuntimeError('Original sharedpred25 deep representative points differ')
    if contact_mask(gt).sum() != 244 or {int(gt[tuple(p)]) for p in crossing} != set(PAIR):
        raise RuntimeError('Original known native interface/crossing differs')
    newgt = audit_new_gt(config)
    if newgt != old['newgt'] or newgt != pipeline['contract']['newgt']:
        raise RuntimeError('Original processed32 newGT including filled differs')
    for sample in newgt['samples']:
        protected.update({str(ROOT / name): digest for name, digest in sample['files_sha256'].items()})
    candidates = Path(args.candidate_cache).resolve()
    candidate_report, weak = read(candidates / 'report.json'), read(candidates / 'weak659.json')
    if not candidate_report['complete'] or candidate_report['sources'] != 32 or candidate_report['newgt'] != newgt:
        raise RuntimeError('Completed coverage canonical32 audit required')
    candidate_row = next(r for r in candidate_report['rows'] if r['source'] == SOURCE)
    for path0 in (candidates / 'report.json', candidates / 'weak659.json',
                  candidates / 'masks/train_659_inst.png', candidates / 'masks/train_659_class.json'):
        protected[str(path0)] = sha(path0)
    checkpoints = dict(r1=Path(args.reference).resolve() / 'final.pt', coverage=run / 'candidate/final.pt')
    for arm, checkpoint in checkpoints.items():
        expected = R1_SHA if arm == 'r1' else EXPECTED_CANDIDATE
        protected[str(checkpoint)] = expected
    restoration = config['backend_adaptation']['restoration']
    d5a_path = Path(project_path(config, restoration['checkpoint']))
    protected[str(d5a_path)] = restoration['checkpoint_sha256']
    verify_protected(protected)
    runtime = runtime_contract(config)
    precision = precision_record(torch)
    core = load_original_core(config)
    imported = existing_project_sources(list(sys.modules.values()), ROOT)
    protected.update({str(ROOT / name): digest for name, digest in imported['files'].items()})
    restorer = get_restorer(config, 'cuda').eval().requires_grad_(False)
    d5a_state = tensor_digest(restorer.state_dict().items())
    if d5a_state != status['restorer_state_sha256']:
        raise RuntimeError('Actual frozen D5a state differs')
    output.mkdir(parents=True); arms = {}
    for arm, checkpoint in checkpoints.items():
        arm_config = base if arm == 'r1' else config
        model = load_backend(checkpoint, arm_config, 'cuda')[0].eval().requires_grad_(False)
        state = tensor_digest(model.state_dict().items())
        if frozen_digest(model) != status['frozen_state_sha256'] or (arm == 'r1' and state != old['frozen_states_sha256']['model']):
            raise RuntimeError('Actual frozen encoder/LoRA/semantic or R1 state differs')
        value, record = capture_once(core, model, restorer, path, arm_config, gt, output, arm)
        previous = Predictions(cache / 'masks' if arm == 'r1' else candidates / 'masks')
        try:
            prior = previous.load('train_659', gt.shape)
        finally:
            previous.close()
        if not np.array_equal(value['instances'], prior[0]) or value['classes'] != prior[1]:
            raise RuntimeError('Actual final instance/class cache parity failed: ' + arm)
        if record['fixed_whole_semantics'] != row['fixed_whole_semantics']:
            raise RuntimeError('Frozen whole semantics differ from original R1')
        if arm == 'r1':
            with np.load(cache / row['map_file'], allow_pickle=False) as maps:
                if not np.array_equal(value['boundary'], maps['boundary']):
                    raise RuntimeError('Actual full native R1 boundary cache parity failed')
            if value['blend'] != row['blend']:
                raise RuntimeError('Actual original R1 blend audit differs')
            if any(record[key] != row[key] for key in ('actual_marker_sha256', 'actual_watershed_sha256')):
                raise RuntimeError('Actual original R1 marker/watershed cache parity failed')
        elif record['fixed_whole_semantics'] != candidate_row['fixed_whole_semantics']:
            raise RuntimeError('Frozen coverage whole semantics differ from completed candidate cache')
        threshold = float(config['inference']['boundary_threshold'])
        stats = threshold_stats(value['boundary'], contact_mask(gt), threshold)
        saved = weak['arms'][arm]
        if (stats['boundary_quantiles'] != saved['native_contact_boundary_quantiles']
                or stats['historical_raw_greater_equal'] != saved['above_fixed_threshold']):
            raise RuntimeError('Actual continuous weak-boundary statistics differ from completed candidate audit: ' + arm)
        if tensor_digest(model.state_dict().items()) != state or tensor_digest(restorer.state_dict().items()) != d5a_state:
            raise RuntimeError('Read-only capture changed model/restorer state')
        record.update(model_state_sha256=state, checkpoint=str(checkpoint), checkpoint_sha256=protected[str(checkpoint)],
            final_ids_and_classes_cache_exact=True, fixed_semantics_cache_exact=True, weak_boundary_stats_cache_exact=True,
            contact=stats, config=arm_config)
        arms[arm] = record
        del model; gc.collect(); torch.cuda.empty_cache()
        print('ARM_CAPTURED', arm, flush=True)
    verify_protected(protected)
    if runtime_contract(config) != runtime or audit_new_gt(config) != newgt or precision_record(torch) != precision:
        raise RuntimeError('Runtime/newGT changed during frozen capture')
    write(output / 'capture.json', dict(complete=True, format='affinity_weak_capture_v1', no_training=True,
        original49_binding=True, source157_verified=True, source=SOURCE, seed=SEED, points=fixed_points,
        crossing=crossing, gt_classes=gt_classes, source_path=str(path), gt_path=str(gt_path), config=config,
        arms=arms, newgt=newgt, current_runtime=runtime, precision=precision,
        migrated_runtime_delta={k: dict(old=old['sealed_runtime'].get(k), current=v)
            for k, v in runtime.items() if old['sealed_runtime'].get(k) != v},
        restorer_checkpoint=str(d5a_path), restorer_checkpoint_sha256=restoration['checkpoint_sha256'],
        restorer_state_sha256=d5a_state, protected_sha256=protected, protected_before_after_exact=True,
        caveat='Frozen GT69/70 output counterfactual on a seen training source; not training, validation, an upper bound or submission.'))
    write(output / 'capture_manifest.json', dict(complete=True, files={p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file()}))
    print('CAPTURE_COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('capture',), default='capture')
    parser.add_argument('--run-root', default='outputs/affinity_coverage')
    parser.add_argument('--reference', default='outputs/affinity_balance/mixed')
    parser.add_argument('--reference-cache', default='outputs/ferrite_native/analysis')
    parser.add_argument('--candidate-cache', default='outputs/affinity_coverage/analysis/gt')
    parser.add_argument('--path-receipt', default='output/affinity_weak/marker_path/report.json',
                        help='Private fixed-path receipt; no per-image coordinates embedded in public code')
    parser.add_argument('--output', default='outputs/affinity_weak')
    capture(parser.parse_args())


if __name__ == '__main__':
    main()
