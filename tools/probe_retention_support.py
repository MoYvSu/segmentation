# -*- coding: utf-8 -*-
"""本地CPU门槛：filled外围负保留是否覆盖351退化界面；不训练/不前向。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.affinity_spatial_trace import edge_roi_mask, native_grid_footprint
from utils.affinity_structure import mask_provenance
from utils.interior_support_masks import interface_support_factor
from utils.offset_letterbox import letterbox_instance_geometry
from utils.retention_support import array_sha, build_processed_negative_retention, processed_native_support


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sealed_hash(record, suffix, path):
    expected = [value for key, value in record['protected_inputs'].items() if key.endswith(suffix)]
    if len(expected) != 1 or _sha(path) != expected[0]:
        raise ValueError('local file does not match sealed protected input: '+suffix)


def _tile(pin, gt):
    box = list(pin['box']); y, x, y1, x1 = box
    labels, valid, geometry = letterbox_instance_geometry(gt[y:y1, x:x1], 1024, 512)
    return dict(index=pin['index'], box=box, grid_gt=labels, valid=valid,
                geometry=geometry.to_dict(), pad_h=1024-geometry.resized_height,
                pad_w=1024-geometry.resized_width)


def frozen_cache_contract(fit, source, gt_sha, supervision_pins):
    """最小私有冻结采集合同；只需数组，不保存模型features或运行优化器。"""
    return dict(
        format='retention_support_capture_v1', source=source, gt_sha256=gt_sha,
        checkpoint_sha256=fit['initial_checkpoint_sha256'],
        restorer_sha256=fit['restorer_checkpoint_sha256'],
        initial_head_sha256=fit['initial_head_sha256'],
        original_boundary_sha256=fit['captures'][source]['original_boundary_sha256'],
        producer='capture_frozen_features at fixed R1/D5a, build_interior_supervision unchanged',
        actual_capture_total_tiles=fit['captures'][source]['actual']['tile_count'],
        complete_original_stitch_exact_required=True, no_training=True, no_optimizer=True,
        files_private_ignored_only=True, no_features_serialized=True,
        tile_fields=['index', 'box', 'geometry', 'pad_h', 'pad_w', 'file', 'file_sha256'],
        npz_arrays=['teacher_logits', 'grid', 'grid_gt', 'valid', 'targets',
                    'full_legal', 'band', 'native_legal', 'retention'],
        instruction='Capture all actual native windows once; export only fixed supervised windows. Save CPU copies of actual frozen arrays, not reconstructed teacher logits.',
        tiles=[dict(index=pin['index'], box=pin['box'],
                    teacher_sha256=pin['audit']['band']['teacher_logits_sha256'],
                    grid_sha256=pin['audit']['band']['captured_fused_grid_sha256'],
                    band_sha256=pin['audit']['band']['band_sha256'],
                    native_legal_sha256=pin['audit']['masks']['native_legal']['mask_sha256'],
                    retention_sha256=pin['audit']['masks']['retention']['mask_sha256'])
               for pin in supervision_pins])


def _resolve_cache_file(manifest_path, entry):
    """明确支持manifest相对路径与固定生产器项目相对声明，不猜测目录。"""
    manifest_path = Path(manifest_path)
    declared = entry.get('file')
    if not isinstance(declared, str) or not declared:
        raise ValueError('cache file must be an explicit nonempty path string')
    prefix = 'outputs/gt_source/retention_cache/'
    if declared.startswith(prefix):
        index = entry.get('index')
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError('project-relative cache file must have a nonnegative integer tile index')
        basename = f'tile_{index:02d}.npz'
        if ('..' in PurePosixPath(declared).parts or declared != prefix+basename):
            raise ValueError('project-relative cache declaration must match exact fixed prefix and tile index')
        file = (manifest_path.parent/basename).resolve()
        rule = 'exact outputs/gt_source/retention_cache/ prefix and matching tile basename -> manifest directory'
        convention = 'explicit_producer_project_relative'
    else:
        file = (manifest_path.parent/declared).resolve()
        rule = 'manifest-relative path, resolved inside manifest directory'
        convention = 'manifest_relative'
    if not file.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError('cache file must stay inside the manifest directory')
    return file, dict(declared_file=declared, resolved_file=str(file),
                      convention=convention, normalization_rule=rule,
                      filename_or_array_data_changed=False)


def _load_cache(path, contract):
    """只读显式manifest，不猜测其他源logits或从boundary逆推出8通道关系。"""
    path = Path(path)
    if not path.is_file():
        return None
    manifest = _read(path)
    for key in ('format', 'source', 'gt_sha256', 'checkpoint_sha256', 'restorer_sha256',
                'initial_head_sha256', 'original_boundary_sha256'):
        if manifest.get(key) != contract[key]:
            raise ValueError('frozen cache identity differs: '+key)
    if manifest.get('complete_original_stitch_exact') is not True or manifest.get('training') is not False:
        raise ValueError('cache must certify complete original stitch and no training')
    expected = {pin['index']: pin for pin in contract['tiles']}
    entries = manifest.get('tiles', [])
    if len(entries) != len(expected) or {entry['index'] for entry in entries} != set(expected):
        raise ValueError('cache must contain each fixed supervised window exactly once')
    loaded = {}
    for entry in entries:
        pin = expected[entry['index']]
        if entry['box'] != pin['box']:
            raise ValueError('cache native box differs from fixed supervised window')
        file, resolution = _resolve_cache_file(path, entry)
        if _sha(file) != entry['file_sha256']:
            raise ValueError('cache file is outside manifest folder or differs from sealed file SHA')
        with np.load(file, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in contract['npz_arrays']}
        for key, digest_key in [('teacher_logits', 'teacher_sha256'), ('grid', 'grid_sha256'),
                                ('band', 'band_sha256'), ('native_legal', 'native_legal_sha256'),
                                ('retention', 'retention_sha256')]:
            if array_sha(arrays[key]) != pin[digest_key]:
                raise ValueError('cache frozen array SHA differs: '+key)
        loaded[entry['index']] = (dict(entry, cache_file_resolution=resolution), arrays)
    return loaded


def run(args):
    torch.set_num_threads(4)
    support_dir, fit_path, gt_path, detail_path = [Path(value).resolve() for value in
        (args.support, args.fit, args.gt, args.detail)]
    support_path = support_dir/'report.json'
    support, fit = _read(support_path), _read(fit_path)
    if not support['complete'] or support['training'] or not fit['complete']:
        raise ValueError('completed frozen support/fit receipts required')
    _sealed_hash(support, '/outputs/interior_fit/report.json', fit_path)
    _sealed_hash(support, '/train_351_gt.npz', gt_path)
    _sealed_hash(support, '/outputs/interior_trace/train_351/detail.npz', detail_path)
    protected = {str(p): _sha(p) for p in (support_path, fit_path, gt_path, detail_path,
                                          support_dir/'train_351/base_maps.npz')}
    role = support['sources']['train_351']['role']
    pair, gid = role['pair'], role['gt_id']
    with np.load(gt_path, allow_pickle=False) as archive:
        gt = archive['instance_map'].copy()
        covered = archive['original_covered'].copy()
    if not np.isin(covered, [0, 1]).all():
        raise ValueError('original coverage must be binary')
    original = covered.astype(bool)
    masks, factor_audit = interface_support_factor(gt, original, pair, radius=2)
    with np.load(support_dir/'train_351/base_maps.npz', allow_pickle=False) as archive:
        connect = archive['boundary'].copy()
    if connect.shape != gt.shape or connect.dtype != np.float32 or not np.isfinite(connect).all():
        raise ValueError('native boundary must retain GT-aligned finite FP32')
    with np.load(detail_path, allow_pickle=False) as archive:
        y, x, y1, x1 = map(int, archive['box'])
        control_crop, connect_crop = archive['control_boundary'].copy(), archive['connect_boundary'].copy()
    if not np.array_equal(connect_crop, connect[y:y1, x:x1]):
        raise ValueError('sealed support boundary does not match sealed detail crop')
    if (control_crop.shape != connect_crop.shape or not np.isfinite(control_crop).all()
            or not (0 <= y < y1 <= gt.shape[0] and 0 <= x < x1 <= gt.shape[1])):
        raise ValueError('control detail must retain finite native geometry')
    decreased = np.zeros(gt.shape, bool)
    decreased[y:y1, x:x1] = control_crop > connect_crop
    roi = masks['completed'] & ~original & decreased
    pins = fit['captures']['train_351']['supervision']
    contract = frozen_cache_contract(fit, 'train_351', _sha(gt_path), pins)
    cache_path = (Path(args.cache_root)/'cache.json' if getattr(args, 'cache_root', None)
                  else Path(args.cache))
    cache = _load_cache(cache_path, contract)
    if cache is not None:
        protected[str(cache_path.resolve())] = _sha(cache_path)
        for entry, _ in cache.values():
            cache_file, _ = _resolve_cache_file(cache_path, entry)
            protected[str(cache_file)] = _sha(cache_file)
    report = dict(
        format='retention_support_CPU_gate_v1', complete=False, training=False,
        model_forward=False, optimizer_created=False, source='train_351', pair=pair,
        ROI=dict(rule='filled full interface within sealed control>connect continuous-response crop',
                 provenance=mask_provenance(gt, original, roi), sha256=array_sha(roi),
                 caveat='Observed decreased interface support domain, not an identified unique topological bottleneck'),
        interface_factors=factor_audit, tiles=[], protected_inputs=protected,
        frozen_capture_required=contract,
        CPU_operator_only=True, actual_GPU_forward_not_replayed=True,
        full_training_loss_or_blackbox_gain_not_tested=True)
    for pin in pins:
        tile = _tile(pin, gt)
        footprint, footprint_audit = native_grid_footprint(tile, roi)
        raw, raw_audit = edge_roi_mask(tile, gt, original, footprint, pair, 'negative')
        original_path, known_path, path_audit = processed_native_support(tile, gt, original)
        upper = raw & known_path & ~original_path
        record = dict(index=pin['index'], box=pin['box'], footprint=footprint_audit,
                      raw=raw_audit, native_support=path_audit,
                      spatial_new_negative_upper_bound_edges=int(upper.sum()),
                      upper_bound_per_channel=[int(upper[0, c].sum()) for c in range(8)],
                      teacher_and_fixed_band_applied=False,
                      actual_new_highconfidence_edges=None)
        if cache is not None:
            entry, arrays = cache[pin['index']]
            for key in ('geometry', 'pad_h', 'pad_w'):
                if entry[key] != tile[key]:
                    raise ValueError('cache geometry differs from sealed native window')
            for key in ('grid_gt', 'valid'):
                if not np.array_equal(arrays[key], tile[key]):
                    raise ValueError('cache GT/valid differs from processed source')
            teacher = torch.from_numpy(arrays['teacher_logits'])
            sup = {key: torch.from_numpy(arrays[key]) for key in
                   ('targets', 'full_legal', 'band', 'native_legal', 'retention')}
            expanded, audit = build_processed_negative_retention(tile, gt, original, teacher, sup)
            selected = expanded['added'].numpy() & raw
            record.update(expansion=audit, cache_file_resolution=entry['cache_file_resolution'],
                          actual_new_highconfidence_edges=int(selected.sum()),
                          new_highconfidence_per_channel=[int(selected[0, c].sum()) for c in range(8)],
                          teacher_and_fixed_band_applied=True)
        report['tiles'].append(record)
    count = sum(row['spatial_new_negative_upper_bound_edges'] for row in report['tiles'])
    report.update(spatial_upper_bound_edges=count, cache_available=cache is not None,
                  actual_gate='unknown_missing_frozen_teacher_and_band' if cache is None else
                      ('eligible_edges_present' if any(row['actual_new_highconfidence_edges'] for row in report['tiles'])
                       else 'no_eligible_edges_in_observed_support_domain'),
                  actual_eligible_edge_count=None if cache is None else
                      sum(row['actual_new_highconfidence_edges'] for row in report['tiles']),
                  conclusion=('Known-path spatial eligibility is only an upper bound. Actual confidence/band intersection requires the frozen cache; do not start training on this result.'
                              if cache is None else 'Actual mask eligibility only; no training/topology/blackbox improvement inferred.'))
    if any(_sha(path) != digest for path, digest in protected.items()):
        raise RuntimeError('read-only gate changed a protected input')
    report.update(complete=True, protected_inputs_exact=True)
    output = Path(args.output).resolve()
    if not (output.is_relative_to(ROOT/'output') or output.is_relative_to(ROOT/'outputs')):
        raise ValueError('diagnostic artifacts must remain in ignored output/outputS')
    if output.exists():
        raise FileExistsError('fresh ignored output directory required')
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'capture_required.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), actual_gate=report['actual_gate'],
                          spatial_upper_bound_edges=count, actual_eligible_edge_count=report['actual_eligible_edge_count']),
                     ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support', default='output/interior_support')
    parser.add_argument('--fit', default='output/interior_fit/report.json')
    parser.add_argument('--gt', default='outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps/train_351_gt.npz')
    parser.add_argument('--detail', default='output/interior_trace/train_351/detail.npz')
    parser.add_argument('--cache', default='output/retention_support_cache/capture.json')
    parser.add_argument('--cache-root', help='明确冻结目录；读取其cache.json，优先于--cache')
    parser.add_argument('--output', default='output/retention_support')
    run(parser.parse_args())
