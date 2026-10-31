# -*- coding: utf-8 -*-
"""冻结N0全尺寸训练热图缓存；不使用GT生成响应，不运行语义或最终划分。

源图身份按内容SHA匹配训练池，完整原生窗口保留旧索引/噪声/拼接顺序。
已封存中心ROI必须与新全图裁片逐元素相同；缓存保留原概率幅值。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def matched_sources(gt, identity, bridge, *, expected_count=32, pool_size=1000):
    """名称只用于GT自身登记；跨数据目录匹配始终以完整文件SHA为依据。"""
    rows = gt.get('rows', [])
    pool = identity.get('pool_files', [])
    if (gt.get('all_identities_match') is not True or len(rows) != expected_count
            or len({r['source'] for r in rows}) != expected_count
            or len(pool) != pool_size
            or {r['full_pool_index'] for r in pool} != set(range(pool_size))):
        raise ValueError('Incomplete GT or full-pool identity registry')
    by_hash = defaultdict(list)
    for row in pool:
        by_hash[row['sha256']].append(row)
    refs = [r for r in bridge.get('files', []) if r['domain'] == 'train_gt']
    if (bridge.get('complete') is not True or len(refs) != expected_count
            or {r['source'] for r in refs} != {r['source'] for r in rows}):
        raise ValueError('Exact complete prior GT-source bridge required')
    refs = {r['source']: r for r in refs}
    result = []
    for row in sorted(rows, key=lambda r: r['source']):
        source = row['source']
        if not source or Path(source).name != source or '/' in source or '\\' in source:
            raise ValueError('Unsafe source identity')
        digest = row['hashes']['image_sha256']
        matches = by_hash[digest]
        if len(matches) != 1:
            raise ValueError('Training image requires unique exact content match: '+source)
        match, reference = matches[0], refs[source]
        if (reference['input_sha256'] != digest or reference['source_index'] != match['full_pool_index']
                or reference['source_shape'] != row['shape']):
            raise ValueError('Prior bridge image/seed/shape differs: '+source)
        result.append(dict(source=source, domain='train_gt', path=row['image'],
            source_index=match['full_pool_index'], image_sha256=digest, shape=row['shape'],
            pool_match=dict(source=match['source'], path=match['path'], sha256=digest,
                            full_pool_index=match['full_pool_index']),
            processed_gt=dict(path=row['gt'], sha256=row['hashes']['gt_sha256'],
                              classes_path=row['classes'], classes_sha256=row['hashes']['classes_sha256']),
            reference=reference))
    return result


def exact_roi_bridge(boundary, reference, roi_box):
    """只接受完全一致，不用数值容差掩盖前向或拼接路径变化。"""
    value = np.asarray(boundary)
    if (value.ndim != 2 or value.dtype != np.float32 or not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1))):
        raise ValueError('Native boundary must retain finite float32 probabilities')
    y, x, y1, x1 = roi_box
    if not (0 <= y < y1 <= value.shape[0] and 0 <= x < x1 <= value.shape[1]):
        raise ValueError('Bridge ROI outside full boundary')
    actual = value[y:y1, x:x1]
    if reference.shape != actual.shape or reference.dtype != np.float32:
        raise ValueError('Bridge shape/dtype differs')
    error = float(np.max(np.abs(actual-reference)))
    if not np.array_equal(actual, reference):
        raise RuntimeError(f'Full-image crop differs from prior N0 ROI: maxdiff={error}')
    return dict(exact=True, max_absolute_difference=error, roi_box=list(roi_box), roi_box_format='yxyx')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('code-root', 'baseline-audit', 'model-dir', 'gt-manifest', 'identity-map',
                'bridge-manifest', 'window-tool', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    args.code_root, args.output = args.code_root.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Full heatmap cache requires a fresh output directory')
    gt, identity, bridge = read(args.gt_manifest), read(args.identity_map), read(args.bridge_manifest)
    if sha(args.gt_manifest) != identity['gt_manifest_sha256']:
        raise RuntimeError('Processed-GT identity registry changed')
    if sha(args.window_tool) != bridge['capture_script_sha256']:
        raise RuntimeError('Window helper is not the previously executed frozen helper')
    rows = matched_sources(gt, identity, bridge)
    for row in rows:
        gt_source = row['processed_gt']
        checks = [(row['path'], row['image_sha256']),
                  (row['pool_match']['path'], row['image_sha256']),
                  (gt_source['path'], gt_source['sha256']),
                  (gt_source['classes_path'], gt_source['classes_sha256']),
                  (row['reference']['path'], row['reference']['sha256'])]
        for path, expected in checks:
            if sha(path) != expected:
                raise RuntimeError('Registered input changed: '+str(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    estimate = sum(int(np.prod(row['shape']))*4 for row in rows)
    if shutil.disk_usage(args.output.parent).free < 2*estimate+1024**3:
        raise RuntimeError('Insufficient free space for immutable full-map cache')
    sys.path.insert(0, str(args.code_root))
    import cv2
    from tools import infer_semifinal as entry
    from tools.affinity_connectivity_views import _inference_state
    from train_backend_adaptation import tensor_digest
    from utils.image_io import read_image
    from utils.patch_diagnostic import BlendMap, tile_boxes
    spec = importlib.util.spec_from_file_location('_frozen_domain_window', args.window_tool)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    old = read(args.baseline_audit)
    config = old['config']
    if (Path(config['paths']['project_root']).resolve() != args.code_root
            or config != bridge['config'] or old['model_states'] != bridge['model_states']):
        raise RuntimeError('N0 protocol differs from original ROI capture')
    for name, expected in old['imported_sources'].items():
        if sha(args.code_root/name) != expected:
            raise RuntimeError('Sealed source changed: '+name)
    entry.validate_config(config)
    entry.configure_runtime()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                     '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is occupied: '+active)
    paths = entry.model_paths(config, args.model_dir.resolve())
    model, restorer, terminal = entry.load_models(config, paths, 'cuda')
    if terminal is not None or entry.local_restoration(config)['candidate'] != 'N0':
        raise RuntimeError('Only unchanged N0 local inference is allowed')
    state = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if state != old['model_states']:
        raise RuntimeError('Frozen model states differ')
    runtime = entry.runtime_record()
    if runtime != bridge['runtime']:
        raise RuntimeError('Runtime differs from the exact ROI reference')
    args.output.mkdir()
    sources = args.output/'source'
    sources.mkdir()
    for name, path in [('capture_boundary_real.py', Path(__file__)),
                       ('capture_boundary_domain.py', args.window_tool),
                       ('processed_gt32.json', args.gt_manifest), ('gt_pool_identity.json', args.identity_map),
                       ('roi_capture_manifest.json', args.bridge_manifest)]:
        shutil.copyfile(path, sources/name)
    write(args.output/'frozen_config.json', config)
    manifest = dict(format='boundary_real_heatmaps_v1', complete=False, training=False,
        semantic_forward=False, final_decode=False, normalization='none; original N0 float32 [0,1]',
        source_root=str(args.code_root), source_manifest=entry.verify_source_manifest(args.code_root),
        capture_script_sha256=sha(__file__), window_helper_sha256=sha(args.window_tool),
        processed_gt_manifest_sha256=sha(args.gt_manifest), identity_map_sha256=sha(args.identity_map),
        roi_bridge_manifest_sha256=sha(args.bridge_manifest), baseline_audit_sha256=sha(args.baseline_audit),
        config=config, models=old['models'], model_states=state, runtime=runtime, files=[])
    started = time.time()
    for number, row in enumerate(rows):
        rgb = cv2.cvtColor(read_image(row['path']), cv2.COLOR_BGR2RGB)
        if list(rgb.shape[:2]) != row['shape']:
            raise RuntimeError('Source RGB shape differs')
        boxes = tile_boxes(rgb.shape[:2], 1024, config['affinity_native']['overlap'])
        blend = BlendMap(rgb.shape[:2])
        prior_windows = {r['window']: r for r in row['reference']['windows']}
        window_audits = []
        with _inference_state(model, restorer, 'cuda'):
            for index, box in enumerate(boxes):
                seed = helper.seed_for(row, index)
                arrays, audit = helper.capture_one(model, restorer, rgb, box, config, seed['local_seed'])
                reference = prior_windows.get(index)
                if reference:
                    for field in ('input_tensor_sha256', 'restored_tensor_sha256', 'grid512_sha256', 'boundary1024_sha256'):
                        if audit[field] != reference[field]:
                            raise RuntimeError('Previously captured N0 window differs: '+field)
                    if seed != reference['seed_audit'] or list(box) != reference['box']:
                        raise RuntimeError('Prior window seed or geometry differs')
                blend.add(box, arrays['boundary1024'])
                window_audits.append(dict(window=index, box=list(box), seed_audit=seed,
                                         previous_window_exact=reference is not None, **audit))
        boundary, blend_audit = blend.finish()
        with np.load(row['reference']['path'], allow_pickle=False) as prior:
            bridge_result = exact_roi_bridge(boundary, prior['stitched_boundary512'], row['reference']['roi_box'])
        target = args.output/(row['source']+'.npz')
        np.savez_compressed(target, boundary=boundary)
        digest = sha(target)
        manifest['files'].append(dict(source=row['source'], heat_path=str(target), heat_sha256=digest,
            map_path=str(target), map_sha256=digest, image_sha256=row['image_sha256'], input_sha256=row['image_sha256'],
            image_path=row['path'], shape=row['shape'], source_index=row['source_index'],
            pool_match=row['pool_match'], processed_gt=row['processed_gt'], windows=window_audits,
            blend_audit=blend_audit, bridge=bridge_result,
            boundary_sha256=helper.array_sha(boundary), reference_cache_sha256=row['reference']['sha256']))
        write(args.output/'manifest.json', manifest)
        print(f'CAPTURED {number+1}/{len(rows)} {row["source"]} windows={len(boxes)} bridge=exact', flush=True)
    after = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if after != state:
        raise RuntimeError('Frozen models changed')
    for name, expected in old['imported_sources'].items():
        if sha(args.code_root/name) != expected:
            raise RuntimeError('Sealed source changed during capture')
    manifest.update(complete=True, frozen_models_unchanged=True, sealed_source_unchanged=True,
                    elapsed_seconds=time.time()-started, total_cache_bytes=sum(Path(r['heat_path']).stat().st_size for r in manifest['files']))
    write(args.output/'manifest.json', manifest)
    print('COMPLETE', len(manifest['files']), manifest['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
