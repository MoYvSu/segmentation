# -*- coding: utf-8 -*-
"""冻结旧D5a/N1，将整图RGB空间模糊转换成NAF训练用二值边界缓存。

该脚本须在独立进程运行。项目模块只从--code-root封存源码加载，额外三个
无命名空间helper用spec直载。只接收原32张训练源，不读取测试图或产生测试GT。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import cv2
import numpy as np


FORMAT = 'boundary_naf_perturbed_heatmaps_v1'
WINDOW_HELPER_SHA = '9db09e7b802f45c01dbc082ab604c54e5e44143f50f7d0706478dfd66adaecce'
N1_HELPER_SHA = '0121d9ea8db103a9e3a0d7be689c376aba880f00df5310155c8abcaeba3926bc'
AUGMENTATION_SHA = 'efc3d306055b531635c1a3545362695d1125c2a52c05781822ca5d3323a631c7'
MINIMUM_FREE_BYTES = 10*1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def file_path(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base/path).resolve()


def register_training_rows(clean, processed, *, only=None):
    """只允许同一32源；GT仅核身份，不把其数组送入前端。"""
    if (clean.get('format') != 'boundary_naf_real_heatmaps_v1' or clean.get('complete') is not True
            or clean.get('local_restoration') != dict(mode='zero', model='restorer')):
        raise ValueError('Expected completed frozen N1 training heatmaps')
    originals, labels = clean.get('files', []), processed.get('rows', [])
    if len(originals) != 32 or len(labels) != 32:
        raise ValueError('Exactly32 clean heat sources and processed-GT rows required')
    old = {r['source']: r for r in originals}
    gt = {r['source']: r for r in labels}
    if len(old) != 32 or len(gt) != 32 or set(old) != set(gt):
        raise ValueError('Clean and processed-GT source identities differ')
    chosen = sorted(old) if only is None else list(only)
    if (not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(old)
            or any(not isinstance(n, str) or not n.startswith('train_') or not n[6:].isdigit() for n in chosen)):
        raise ValueError('Only distinct registered train_XXX sources may be selected')
    result = []
    for name in chosen:
        source, target = old[name], gt[name]
        if (source.get('image_sha256') != target['hashes']['image_sha256']
                or source.get('gt_sha256') != target['hashes']['gt_sha256']
                or source.get('classes_sha256') != target['hashes']['classes_sha256']
                or source.get('shape') != target['shape']
                or type(source.get('source_index')) is not int or not 0 <= source['source_index'] < 1000
                or not isinstance(source.get('windows'), list) or not source['windows']):
            raise ValueError('Source content/GT/shape/window identity differs: '+name)
        result.append(dict(source=name, clean=source, processed_gt=target))
    return result


def binary_record(boundary):
    """只返回阈值.5的bool；连续图只登记SHA和描述统计，不落大数组。"""
    if (not isinstance(boundary, np.ndarray) or boundary.ndim != 2 or boundary.dtype != np.float32
            or not np.isfinite(boundary).all() or np.any((boundary < 0) | (boundary > 1))):
        raise ValueError('Expected finite float32 continuous boundary in [0,1]')
    binary = np.ascontiguousarray(boundary >= .5, dtype=bool)
    return binary, dict(representation='thresholded_binary', threshold=.5,
        continuous_boundary_sha256=array_sha(boundary), binary_sha256=array_sha(binary),
        continuous_statistics=dict(min=float(boundary.min()), max=float(boundary.max()),
            mean=float(boundary.mean()), quantiles={str(q): float(np.quantile(boundary, q)) for q in (.05, .5, .95)}),
        binary_positive_fraction=float(binary.mean()), continuous_boundary_saved=False)


def _load_helper(path, name, expected):
    path = Path(path).resolve()
    if sha(path) != expected:
        raise RuntimeError('Helper SHA differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_frozen_imports(code_root):
    """拒绝从工作checkout混入同名项目模块；三个spec helper不在这些包名下。"""
    root = Path(code_root).resolve()
    checked = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in ('tools', 'utils', 'models', 'data') and not name.startswith('train_'):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix == '.py':
            if not path.is_relative_to(root):
                raise RuntimeError('Project module escaped sealed source root: '+name+'='+str(path))
            checked[str(path.relative_to(root))] = sha(path)
    return checked


def save_thumbnail(path, clean_rgb, degraded_rgb, clean_boundary, boundary, sigma, title):
    """只保留统一尺寸的小图；不保存整幅RGB或连续B。"""
    h, w = clean_rgb.shape[:2]
    scale = min(320/w, 256/h)
    size = max(1, round(w*scale)), max(1, round(h*scale))
    tiles = [cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)]
    for value in (clean_boundary, boundary):
        gray = np.rint(np.clip(value, 0, 1)*255).astype(np.uint8)
        tiles.append(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS))
    maximum = max(1., float(sigma.max()))
    tiles.append(cv2.applyColorMap(np.rint(sigma/maximum*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS))
    canvas = np.full((size[1]+55, size[0]*5, 3), 245, np.uint8)
    cv2.putText(canvas, title, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
    for index, (tile, label) in enumerate(zip(tiles, ('clean RGB', 'perturbed RGB', 'clean N1 B', 'perturbed N1 B', 'nominal sigma'))):
        canvas[55:, index*size[0]:(index+1)*size[0]] = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, label, (index*size[0]+4, 43), cv2.FONT_HERSHEY_SIMPLEX, .40, (20, 20, 20), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('Thumbnail encoding failed')
    Path(path).write_bytes(encoded.tobytes())


def run(args):
    code_root = Path(args.code_root).resolve()
    output = Path(args.output_root).resolve()
    clean_path, processed_path = Path(args.clean_manifest).resolve(), Path(args.processed_gt_manifest).resolve()
    clean, processed = read(clean_path), read(processed_path)
    rows = register_training_rows(clean, processed, only=args.only)
    if len(set(args.profiles)) != len(args.profiles) or not set(args.profiles) <= {'moderate', 'strong'}:
        raise ValueError('Distinct moderate/strong perturbation profiles required')
    if clean.get('processed_gt_manifest_sha256') != sha(processed_path):
        raise RuntimeError('Processed-GT manifest does not match original clean cache')
    config = deepcopy(clean['config'])
    if (Path(config['paths']['project_root']).resolve() != code_root
            or config['submission'].get('local_restoration') != dict(mode='zero', model='restorer')):
        raise RuntimeError('Sealed root/local-zero N1 contract differs')
    if output.exists():
        raise FileExistsError('Perturbed cache output must be fresh: '+str(output))
    if shutil.disk_usage(output.parent).free <= MINIMUM_FREE_BYTES+64*1024**2:
        raise RuntimeError('Preserve10GiB fast-disk free before capture')
    pins = {str(clean_path): sha(clean_path), str(processed_path): sha(processed_path),
            str(Path(__file__).resolve()): sha(__file__)}
    for row in rows:
        target, old = row['processed_gt'], row['clean']
        for key in ('image', 'gt', 'classes'):
            path = file_path(target[key], processed_path.parent)
            if sha(path) != target['hashes'][key+'_sha256']:
                raise RuntimeError('Original training source changed: '+row['source']+'/'+key)
            pins[str(path)] = target['hashes'][key+'_sha256']
        path = file_path(old['heat_path'], clean_path.parent)
        if sha(path) != old['heat_sha256']:
            raise RuntimeError('Original clean heatmap changed: '+row['source'])
        pins[str(path)] = old['heat_sha256']

    # 清除启动脚本目录/checkout的优先级，独立进程只能导入封存项目依赖。
    checkout = Path(__file__).resolve().parents[1]
    sys.path[:] = [str(code_root)]+[p for p in sys.path if p and Path(p).resolve() not in (code_root, checkout, checkout/'tools')]
    assert_frozen_imports(code_root)
    import torch
    from tools import infer_semifinal as entry
    from tools.affinity_connectivity_views import _inference_state
    from tools.probe_affinity_patch import tensor_from_rgb, boundary_grid
    from utils.affinity_deployment import crop_letterbox_output
    from utils.patch_diagnostic import BlendMap, tile_boxes
    from utils.restoration_probe import restore_probe
    from train_backend_adaptation import tensor_digest
    helper = _load_helper(args.window_helper, '_naf_perturb_window', WINDOW_HELPER_SHA)
    n1 = _load_helper(args.n1_helper, '_naf_perturb_n1', N1_HELPER_SHA)
    augmentation = _load_helper(args.augmentation_module, '_naf_perturb_rgb', AUGMENTATION_SHA)
    for path in (args.window_helper, args.n1_helper, args.augmentation_module):
        pins[str(Path(path).resolve())] = sha(path)
    entry.validate_config(config)
    entry.configure_runtime()  # 沿用旧TF32/runtime，不能为了新实验改变clean重放。
    source_receipt = entry.verify_source_manifest(code_root)
    if source_receipt.get('manifest_present') is not True:
        raise RuntimeError('Sealed code manifest is required')
    paths = entry.model_paths(config, Path(args.model_dir).resolve())
    for path in paths.values():
        pins[str(path)] = sha(path)
    model, restorer, terminal = entry.load_models(config, paths, 'cuda')
    if terminal is not None or entry.local_restoration(config)['candidate'] != 'N1':
        raise RuntimeError('Expected original shared D5a and local-zero N1 only')
    states = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if states != clean['model_states']:
        raise RuntimeError('Frozen model state identities differ')
    imports_before = assert_frozen_imports(code_root)
    output.mkdir(parents=True)
    (output/'gallery').mkdir()
    manifest = dict(format=FORMAT, complete=False, training=False,
        clean_real_input_manifest=dict(path=str(clean_path), sha256=sha(clean_path)),
        clean_real_input_manifest_sha256=sha(clean_path),
        processed_gt_manifest=dict(path=str(processed_path), sha256=sha(processed_path)),
        processed_gt_manifest_sha256=sha(processed_path),
        source_root=str(code_root), source_manifest=source_receipt, models=deepcopy(clean['models']), model_states=states,
        local_restoration=deepcopy(clean['local_restoration']), config=config, runtime=entry.runtime_record(),
        source_count=len(rows), full32_source_count=32, only_training_sources=True, no_test_training=True,
        no_GT_as_model_input=True, representation='thresholded_binary', threshold=.5,
        capture_script_sha256=sha(__file__), augmentation_module_sha256=sha(args.augmentation_module),
        window_helper_sha256=sha(args.window_helper), n1_helper_sha256=sha(args.n1_helper),
        clean_verified_sources=[], variants=[dict(id=p, augmentation=dict(profile=p, config=augmentation.profile_settings(p)), rows=[])
            for p in args.profiles])
    operators = tensor_from_rgb, boundary_grid, crop_letterbox_output

    def capture(rgb, prior, sigma=None):
        expected_boxes = tile_boxes(rgb.shape[:2], 1024, config['affinity_native']['overlap'])
        old_windows = prior['windows']
        if len(expected_boxes) != len(old_windows) or any(list(box) != old['box'] for box, old in zip(expected_boxes, old_windows)):
            raise RuntimeError('Original full-image window order changed')
        blend, audits = BlendMap(rgb.shape[:2]), []
        with _inference_state(model, restorer, 'cuda'):
            for index, (box, old) in enumerate(zip(expected_boxes, old_windows)):
                seed = old['seed']
                if type(seed) is not int:
                    raise ValueError('Original local window seed missing')
                arrays, audit = n1.capture_n1_window(helper, model, restorer, rgb, box, config, seed,
                    operators=operators, zero_probe=restore_probe)
                if sigma is None and (audit['input_tensor_sha256'] != old['input_tensor_sha256']
                        or audit['restored_tensor_sha256'] != old['restored_tensor_sha256']
                        or audit['grid512_sha256'] != old['grid512_sha256']
                        or audit['boundary1024_sha256'] != old['boundary1024_sha256']):
                    raise RuntimeError('Clean tensor/restoration/grid/native replay differs: '+str(index))
                blend.add(box, arrays['boundary1024'])
                item = dict(window=index, box=list(box), seed=seed, **audit)
                if sigma is not None:
                    y, x, y1, x1 = box
                    crop = sigma[y:y1, x:x1]
                    item['augmentation_crop'] = dict(sigma_min=float(crop.min()), sigma_max=float(crop.max()),
                        sigma_mean_square=float(np.mean(crop.astype(np.float64)**2)),
                        weak_fraction=float(np.mean(crop <= 1.5)), strong_fraction=float(np.mean(crop >= 4.)))
                audits.append(item)
        boundary, blend_audit = blend.finish()
        return boundary, blend_audit, audits

    started = time.monotonic()
    try:
        for number, registered in enumerate(rows, 1):
            name, old, target = registered['source'], registered['clean'], registered['processed_gt']
            rgb_path = file_path(target['image'], processed_path.parent)
            rgb_bgr = cv2.imdecode(np.frombuffer(rgb_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise ValueError('Cannot read registered training RGB: '+name)
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            if list(rgb.shape[:2]) != old['shape']:
                raise RuntimeError('Original RGB native shape differs')
            with np.load(file_path(old['heat_path'], clean_path.parent), allow_pickle=False) as archive:
                clean_boundary = archive['boundary'].copy()
            if array_sha(clean_boundary) != old['boundary_sha256']:
                raise RuntimeError('Original clean boundary array SHA differs')
            if args.verify_clean:
                replay, _, _ = capture(rgb, old)
                if not np.array_equal(replay, clean_boundary):
                    raise RuntimeError('Clean complete stitched boundary replay differs: '+name)
                manifest['clean_verified_sources'].append(name)
                print(json.dumps(dict(stage='clean_exact', source=name, index=number, total=len(rows))), flush=True)
            for variant in manifest['variants']:
                if shutil.disk_usage(output).free <= MINIMUM_FREE_BYTES+rgb.shape[0]*rgb.shape[1]:
                    raise RuntimeError('Stop before consuming10GiB free fast-disk reserve')
                profile = variant['id']
                seed = 20261030+old['source_index']
                degraded, sigma, augmentation_receipt = augmentation.perturb_native_rgb(rgb, profile=profile, seed=seed)
                boundary, blend_audit, windows = capture(degraded, old, sigma)
                binary, representation = binary_record(boundary)
                folder = output/profile
                folder.mkdir(exist_ok=True)
                path = folder/(name+'.npz')
                if path.exists():
                    raise FileExistsError(path)
                np.savez_compressed(path, boundary=binary)
                with np.load(path, allow_pickle=False) as reread:
                    if set(reread.files) != {'boundary'} or reread['boundary'].dtype != bool or array_sha(reread['boundary']) != representation['binary_sha256']:
                        raise RuntimeError('Written binary heatmap reload differs')
                row = deepcopy(old)
                row.update(heat_path=str(path), heat_sha256=sha(path), boundary_sha256=representation['binary_sha256'],
                    clean_boundary_sha256=old['boundary_sha256'], augmentation_receipt=augmentation_receipt,
                    augmentation_seed=seed, windows=windows, blend_audit=blend_audit, **representation)
                variant['rows'].append(row)
                gallery = output/'gallery'/(name+'_'+profile+'.jpg')
                save_thumbnail(gallery, rgb, degraded, clean_boundary, boundary, sigma, name+' '+profile)
                row['thumbnail'] = dict(path=str(gallery), sha256=sha(gallery))
                atomic_json(output/'manifest.partial.json', manifest)
                print(json.dumps(dict(stage='captured', source=name, profile=profile, index=number, total=len(rows),
                    windows=len(windows), binary_positive_fraction=representation['binary_positive_fraction'],
                    clean_binary_positive_fraction=float(np.mean(clean_boundary >= .5)))), flush=True)
        after = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
        if after != states or entry.verify_source_manifest(code_root) != source_receipt:
            raise RuntimeError('Frozen model state or sealed source changed')
        imports_after = assert_frozen_imports(code_root)
        if any(imports_after.get(name) != digest for name, digest in imports_before.items()):
            raise RuntimeError('Imported frozen source changed')
        for path, digest in pins.items():
            if sha(path) != digest:
                raise RuntimeError('Protected source changed: '+path)
        manifest.update(complete=True, frozen_models_unchanged=True, sealed_source_unchanged=True,
            protected_sources_unchanged=True, protected_sources=pins, imported_sources=imports_after,
            verify_clean=bool(args.verify_clean), elapsed_seconds=time.monotonic()-started,
            total_cache_bytes=sum(Path(r['heat_path']).stat().st_size for v in manifest['variants'] for r in v['rows']),
            free_fast_disk_bytes=shutil.disk_usage(output).free)
        atomic_json(output/'manifest.json', manifest)
        atomic_json(output/'complete.json', dict(complete=True, manifest=str(output/'manifest.json'),
            manifest_sha256=sha(output/'manifest.json'), source_count=len(rows), profiles=list(args.profiles),
            verify_clean=bool(args.verify_clean), frozen_models_unchanged=True, no_training=True))
        print(json.dumps(dict(stage='complete', source_count=len(rows), profiles=list(args.profiles),
            seconds=manifest['elapsed_seconds'], cache_bytes=manifest['total_cache_bytes'])), flush=True)
        return manifest
    except Exception as error:
        atomic_json(output/'failed.json', dict(complete=False, error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), completed_variants={v['id']:len(v['rows']) for v in manifest['variants']}))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clean-manifest', 'processed-gt-manifest', 'code-root', 'model-dir', 'output-root',
                 'augmentation-module', 'window-helper', 'n1-helper'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--profiles', nargs='+', choices=('moderate', 'strong'), default=['moderate', 'strong'])
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--verify-clean', action='store_true')
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
