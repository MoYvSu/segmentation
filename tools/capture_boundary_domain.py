# -*- coding: utf-8 -*-
"""冻结N0的原生局部热图采集，不训练、不运行全图语义或最终分割。

样本与窗口由私有manifest预先登记，测试数据只用于无标签域差诊断。
复用封存源码、模型和Philox噪声；中心512按相交原窗的完整权重与顺序拼接。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


FORMAT = 'boundary_domain_samples_v1'
TRAIN_SEED_OFFSET = 2_000_000
NATIVE_SIZE = 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_samples(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if (manifest.get('format') != FORMAT or manifest.get('complete') is not True
            or not isinstance(manifest.get('files'), list) or not manifest['files']):
        raise ValueError('Expected complete boundary_domain_samples_v1 manifest')
    registered = {}
    for group, rows in [('files', manifest['files']), ('anchors', manifest.get('anchors', []))]:
      seen = set()
      for row in rows:
        source = row.get('source')
        if (not isinstance(source, str) or not source or Path(source).name != source
                or '/' in source or '\\' in source or source in seen):
            raise ValueError('Source names must be unique safe stems')
        seen.add(source)
        if source in registered:
            identity = ('source', 'path', 'file_stem', 'domain', 'source_index', 'input_sha256')
            if any(row.get(key) != registered[source].get(key) for key in identity):
                raise ValueError('Repeated anchor and sample identities differ')
        registered[source] = row
        if row.get('domain') not in ('train_gt', 'train_extra', 'test'):
            raise ValueError('Unknown source domain')
        if (type(row.get('source_index')) is not int or row['source_index'] < 0
                or not isinstance(row.get('path'), str)
                or not isinstance(row.get('input_sha256'), str) or len(row['input_sha256']) != 64):
            raise ValueError('Missing stable source index/path/input SHA')
    for row in manifest.get('anchors', []):
        if row.get('outside_statistical_sample') is not True or row['domain'] != 'test':
            raise ValueError('Anchors must be explicitly excluded test references')
    return manifest


def seed_for(row, window):
    """测试沿用旧完整100图索引；训练使用独立的固定数值命名空间。"""
    if type(window) is not int or window < 0:
        raise ValueError('Window index must be nonnegative')
    index = row['source_index']
    if type(index) is not int or index < 0:
        raise ValueError('Source index must be nonnegative')
    if row['domain'] == 'test':
        source = row['source']
        if not source.startswith('test_') or not source[5:].isdigit():
            raise ValueError('Test source must use its original test_XXX identity')
        number = int(source[5:])
        if not 69 <= number <= 168 or index != number-69:
            raise ValueError('Test source index must match the original complete100 ordering')
        offset = 0
    elif row['domain'] in ('train_gt', 'train_extra'):
        if index >= 1000:
            raise ValueError('Training source index must refer to the fixed1000 list')
        offset = TRAIN_SEED_OFFSET
    else:
        raise ValueError('Unknown source domain')
    source_seed = 314159+offset+index
    return dict(base_seed=314159, source_index=index, train_namespace_offset=offset,
        source_seed=source_seed, local_offset=100000, native_size=NATIVE_SIZE,
        window_index=window, local_seed=source_seed+100000+NATIVE_SIZE*1000+window)


def select_window(shape, row, overlap=.25):
    from utils.patch_diagnostic import tile_boxes
    if len(shape) != 2 or min(shape) < NATIVE_SIZE:
        raise ValueError('A true native1024 window requires both source dimensions >=1024')
    boxes = tile_boxes(shape, NATIVE_SIZE, overlap)
    if 'window' in row:
        index = row['window']
        if type(index) is not int or not 0 <= index < len(boxes):
            raise ValueError('Window index is outside the original tile sequence')
    elif 'box' in row:
        target = tuple(row['box'])
        if target not in boxes:
            raise ValueError('Selected box is not an original deployment window')
        index = boxes.index(target)
    else:
        cy, cx = shape[0]/2, shape[1]/2
        index = min(range(len(boxes)), key=lambda i: (((boxes[i][0]+boxes[i][2])/2-cy)**2
                                                     +((boxes[i][1]+boxes[i][3])/2-cx)**2, i))
    box = boxes[index]
    if 'box' in row and tuple(row['box']) != tuple(box):
        raise ValueError('Manifest box differs from its original window index')
    if box[2]-box[0] != NATIVE_SIZE or box[3]-box[1] != NATIVE_SIZE:
        raise ValueError('Window must be native1024 without rescaling')
    return index, box


def roi_window_plan(shape, row):
    """保留完整tile序列索引，只省略与中心ROI完全无交集的窗口。"""
    from utils.patch_diagnostic import tile_boxes
    if 'source_shape' in row and list(row['source_shape']) != list(shape):
        raise ValueError('Registered source shape differs')
    index, box = select_window(shape, row)
    y, x, _, _ = box
    roi = (y+256, x+256, y+768, x+768)
    if 'roi_box' in row:
        specified = tuple(row['roi_box'])
        convention = row.get('roi_box_format')
        if convention == 'xyxy':
            specified = (specified[1], specified[0], specified[3], specified[2])
        elif convention != 'yxyx':
            raise ValueError('Manifest ROI must explicitly declare yxyx or xyxy')
        if specified != roi:
            raise ValueError('Registered ROI differs from primary window center512')
    windows = [(i, b) for i, b in enumerate(tile_boxes(shape, NATIVE_SIZE, .25))
               if max(b[0], roi[0]) < min(b[2], roi[2])
               and max(b[1], roi[1]) < min(b[3], roi[3])]
    return index, box, roi, windows


class RoiBlendMap:
    """原BlendMap的精确ROI裁片：权重按完整窗口生成，不能在ROI内重建权重。"""
    def __init__(self, shape, roi):
        y, x, y1, x1 = roi
        if not (0 <= y < y1 <= shape[0] and 0 <= x < x1 <= shape[1]):
            raise ValueError('ROI outside source')
        self.roi = tuple(roi)
        self.total = np.zeros((y1-y, x1-x), np.float32)
        self.weight = np.zeros_like(self.total)
        self.count = np.zeros(self.total.shape, np.uint16)

    def add(self, box, value):
        y, x, y1, x1 = box
        value = np.asarray(value, np.float32)
        if value.shape != (y1-y, x1-x) or not np.isfinite(value).all():
            raise ValueError('Invalid patch map')
        ry, rx, ry1, rx1 = self.roi
        top, left, bottom, right = max(y, ry), max(x, rx), min(y1, ry1), min(x1, rx1)
        if bottom <= top or right <= left:
            return
        h, w = value.shape
        wy = np.maximum(.05, 1-np.abs(np.linspace(-1, 1, h, dtype=np.float32)))
        wx = np.maximum(.05, 1-np.abs(np.linspace(-1, 1, w, dtype=np.float32)))
        weights = wy[:, None]*wx[None, :]
        source = np.s_[top-y:bottom-y, left-x:right-x]
        target = np.s_[top-ry:bottom-ry, left-rx:right-rx]
        self.total[target] += value[source]*weights[source]
        self.weight[target] += weights[source]
        self.count[target] += 1

    def finish(self):
        if not np.all(self.weight > 0):
            raise ValueError('Uncovered pixels in ROI')
        return self.total/self.weight, dict(tiles_max_overlap=int(self.count.max()),
            tiles_min_overlap=int(self.count.min()), weight_sha256=array_sha(self.weight),
            count_sha256=array_sha(self.count))


def capture_one(model, restorer, rgb, box, config, seed, *, device='cuda', operators=None):
    """只调用旧部署的局部恢复/融合函数，operators只供轻量CPU契约测试。"""
    import torch
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3):
        raise ValueError('Source RGB must be uint8 HWC with three channels')
    y, x, y1, x1 = box
    if not (0 <= y < y1 <= rgb.shape[0] and 0 <= x < x1 <= rgb.shape[1]):
        raise ValueError('Window box is outside RGB')
    selected = np.ascontiguousarray(rgb[y:y1, x:x1])
    if selected.shape != (NATIVE_SIZE, NATIVE_SIZE, 3):
        raise ValueError('Window RGB must be native1024')
    if operators is None:
        from models.backend_adaptation import restore_first
        from tools.probe_affinity_patch import boundary_grid, tensor_from_rgb
        from utils.affinity_deployment import crop_letterbox_output
        operators = tensor_from_rgb, restore_first, boundary_grid, crop_letterbox_output
    tensor_from_rgb, restore_first, boundary_grid, crop_letterbox_output = operators
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        image, ph, pw = tensor_from_rgb(selected, device)
        if tuple(image.shape) != (1, 3, NATIVE_SIZE, NATIVE_SIZE) or ph != 0 or pw != 0:
            raise RuntimeError('Native1024 input unexpectedly resized or padded')
        restored = restore_first(restorer, image, seed)
        if restored.shape != image.shape:
            raise RuntimeError('Restorer changed local shape')
        grid = boundary_grid(model, restored, config)
        if tuple(grid.shape) != (1, 1, 512, 512) or grid.dtype != torch.float32:
            raise RuntimeError('Frozen N0 must produce the actual512 fused probability grid')
        native = crop_letterbox_output(grid, NATIVE_SIZE, ph, pw, (NATIVE_SIZE, NATIVE_SIZE))
    grid_array = grid.detach().cpu()[0, 0].numpy()
    native_array = native.detach().cpu()[0, 0].numpy()
    for name, value, shape in [('grid512', grid_array, (512, 512)),
                               ('boundary1024', native_array, (1024, 1024))]:
        if (value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all()
                or np.any((value < 0) | (value > 1))):
            raise RuntimeError(name+' violated continuous probability contract')
    arrays = dict(rgb1024=selected, boundary1024=native_array, grid512=grid_array)
    audit = dict(input_tensor_sha256=array_sha(image.detach().cpu().numpy()),
        restored_tensor_sha256=array_sha(restored.detach().cpu().numpy()),
        rgb1024_sha256=array_sha(selected), boundary1024_sha256=array_sha(native_array),
        grid512_sha256=array_sha(grid_array), input_shape=[1, 3, 1024, 1024],
        grid_shape=[1, 1, 512, 512], native_shape=[1024, 1024], pad_h=0, pad_w=0,
        interpolation='bilinear_align_corners_true', crop512_in_window_xyxy=[256, 256, 768, 768])
    return arrays, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--baseline-audit', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-window-root', type=Path)
    parser.add_argument('--control-manifest', type=Path)
    args = parser.parse_args()
    args.code_root = args.code_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Use a fresh capture directory; prior evidence cannot be overwritten')
    manifest = load_samples(args.manifest)
    # 先读完全部来源并核验摘要，避免半套源清单进入GPU阶段。
    for row in manifest['files']+manifest.get('anchors', []):
        path = Path(row['path'])
        if path.stem != row.get('file_stem', row['source']) or sha(path) != row['input_sha256']:
            raise RuntimeError('Source image differs from the registered manifest: '+row['source'])
    sys.path.insert(0, str(args.code_root))
    import cv2
    import torch
    from tools import infer_semifinal as entry
    from tools.affinity_connectivity_views import _inference_state
    from train_backend_adaptation import tensor_digest
    from utils.image_io import read_image
    old = json.loads(args.baseline_audit.read_text(encoding='utf-8'))
    config = old['config']
    if Path(config['paths']['project_root']).resolve() != args.code_root:
        raise RuntimeError('Source snapshot differs from original N0 config')
    if (config['affinity_native']['overlap'] != .25
            or config['affinity_deployment']['fusion_mode'] != 'gated'
            or config['affinity_deployment']['short_reduction'] != 'top2'):
        raise RuntimeError('Expected original gated/top2/native1024 deployment')
    for name, expected in old['imported_sources'].items():
        if sha(args.code_root/name) != expected:
            raise RuntimeError('Sealed source differs: '+name)
    entry.validate_config(config)
    entry.configure_runtime()
    control_rows = {}
    controls = None
    if args.control_manifest:
        controls = json.loads(args.control_manifest.read_text(encoding='utf-8'))
        if (controls.get('complete') is not True or controls['model_states'] != old['model_states']
                or controls['source_root'] != str(args.code_root)):
            raise RuntimeError('Control bridge source or models differ')
        control_rows = {row['source']: row for row in controls['files']}
    planned = {}
    for row in manifest['files']+manifest.get('anchors', []):
        shape = read_image(row['path']).shape[:2]
        plan = roi_window_plan(shape, row)
        for index, _ in plan[3]:
            seed_for(row, index)
        if row['source'] in planned and planned[row['source']] != plan:
            raise ValueError('Repeated anchor and sample crop plans differ')
        planned[row['source']] = plan
        control = control_rows.get(row['source'])
        reference = row.get('reference_cache_path', control['path'] if control else None)
        expected = row.get('reference_cache_sha256', control['sha256'] if control else None)
        if reference and (not expected or sha(reference) != expected):
            raise RuntimeError('Reference cache changed: '+row['source'])
        if row.get('outside_statistical_sample') and not reference:
            raise ValueError('Anchor must have a registered actual N0 cache')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                     '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is occupied: '+active)
    paths = entry.model_paths(config, args.model_dir.resolve())
    model, restorer, terminal = entry.load_models(config, paths, 'cuda')
    if terminal is not None or entry.local_restoration(config)['candidate'] != 'N0':
        raise RuntimeError('Only the frozen N0 local path is allowed')
    states = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if states != old['model_states']:
        raise RuntimeError('Loaded model states differ from N0')
    args.output.mkdir(parents=True)
    (args.output/'source').mkdir()
    shutil.copyfile(__file__, args.output/'source'/Path(__file__).name)
    shutil.copyfile(args.manifest, args.output/'sample_manifest.json')
    write(args.output/'frozen_config.json', config)
    receipt = dict(format='boundary_domain_capture_v1', complete=False, training=False,
        no_global_semantic_forward=True, no_final_decode=True,
        capture_scope='primary native1024 plus original overlap blend for center512 ROI',
        roi_box_format='yxyx', window_order='original full-image index ascending',
        blend_weights='original full1024 triangular weights cropped only after construction',
        source_root=str(args.code_root), source_manifest=entry.verify_source_manifest(args.code_root),
        capture_script_sha256=sha(__file__), sample_manifest_sha256=sha(args.manifest),
        config=config, model_states=states, models=old['models'], runtime=entry.runtime_record(),
        baseline_audit_sha256=sha(args.baseline_audit), files=[], anchors=[])
    if controls:
        receipt['control_manifest_sha256'] = sha(args.control_manifest)
    old_rows = {row['source']: row for row in old['images']}
    started = time.time()
    # 先验收固定anchor；不通过时不继续80图统计采集。
    ordered = [('anchors', row) for row in manifest.get('anchors', [])]
    ordered += [('files', row) for row in manifest['files']]
    for number, (group, row) in enumerate(ordered):
        rgb = cv2.cvtColor(read_image(row['path']), cv2.COLOR_BGR2RGB)
        index, box, roi, windows = planned[row['source']]
        blend = RoiBlendMap(rgb.shape[:2], roi)
        window_audits = []
        primary_arrays = primary_audit = None
        with _inference_state(model, restorer, 'cuda'):
            for window, window_box in windows:
                seeds = seed_for(row, window)
                arrays, audit = capture_one(model, restorer, rgb, window_box, config, seeds['local_seed'])
                exact = {}
                if row['source'] in old_rows:
                    old_window = old_rows[row['source']]['local_windows'][window]
                    if (old_window['box'] != list(window_box) or old_window['seed'] != seeds['local_seed']
                            or old_window['input_sha256'] != audit['input_tensor_sha256']
                            or old_window['restored_sha256'] != audit['restored_tensor_sha256']):
                        raise RuntimeError('Known N0 window input/restoration/seed changed')
                    exact['input_restoration_exact'] = True
                if args.reference_window_root:
                    reference = args.reference_window_root/row['source']/f'window_{window:02d}.npz'
                    if reference.is_file():
                        with np.load(reference, allow_pickle=False) as value:
                            if (not np.array_equal(value['native_fused'], arrays['boundary1024'])
                                    or not np.array_equal(value['fused'][0], arrays['grid512'])
                                    or value['box'].tolist() != list(window_box)):
                                raise RuntimeError('Previously captured N0 grid/native boundary differs')
                        exact.update(grid_native_exact=True, reference_sha256=sha(reference))
                blend.add(window_box, arrays['boundary1024'])
                window_audits.append(dict(window=window, box=list(window_box), seed_audit=seeds,
                                          reference_exact=exact, **audit))
                if window == index:
                    primary_arrays, primary_audit = arrays, audit
        if primary_arrays is None:
            raise RuntimeError('Primary window absent from ROI plan')
        stitched, blend_audit = blend.finish()
        arrays, audit = primary_arrays, primary_audit
        arrays.update(stitched_boundary512=stitched, roi_box=np.asarray(roi, np.int32))
        control = control_rows.get(row['source'])
        reference = row.get('reference_cache_path', control['path'] if control else None)
        reference_exact = None
        if reference:
            with np.load(reference, allow_pickle=False) as cached:
                ry, rx, ry1, rx1 = roi
                expected = cached['boundary'][ry:ry1, rx:rx1]
                equal = np.array_equal(stitched, expected)
                error = float(np.max(np.abs(stitched-expected)))
            reference_exact = dict(stitched_roi_exact=equal, max_absolute_difference=error,
                                   reference_cache_path=reference, reference_cache_sha256=sha(reference))
            if not equal:
                write(args.output/'bridge_failure.json', dict(source=row['source'], roi_box=list(roi),
                    reference_exact=reference_exact, windows=window_audits))
                raise RuntimeError('Actual N0 ROI blend differs from full-image cache; stopping')
        target_dir = args.output/'anchors' if group == 'anchors' else args.output
        target_dir.mkdir(exist_ok=True)
        target = target_dir/(row['source']+'.npz')
        np.savez_compressed(target, **arrays)
        receipt[group].append(dict(source=row['source'], domain=row['domain'],
            source_index=row['source_index'], source_path=row['path'], input_sha256=row['input_sha256'],
            source_shape=list(rgb.shape[:2]), window=index, box=list(box), seed_audit=seed_for(row, index),
            roi_box=list(roi), roi_box_format='yxyx', stitched_boundary512_sha256=array_sha(stitched),
            windows=window_audits, blend_audit=blend_audit, outside_statistical_sample=group == 'anchors',
            path=str(target), sha256=sha(target), reference_exact=reference_exact, **audit))
        write(args.output/'capture_manifest.json', receipt)
        print(f'CAPTURED {number+1}/{len(ordered)} {row["source"]} primary={index} windows={len(windows)}', flush=True)
    after = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if states != after:
        raise RuntimeError('Frozen model states changed during capture')
    for name, expected in old['imported_sources'].items():
        if sha(args.code_root/name) != expected:
            raise RuntimeError('Sealed source changed during capture: '+name)
    receipt.update(complete=True, frozen_models_unchanged=True, sealed_source_unchanged=True,
                   elapsed_seconds=time.time()-started)
    write(args.output/'capture_manifest.json', receipt)
    print('COMPLETE', len(receipt['files']), receipt['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
