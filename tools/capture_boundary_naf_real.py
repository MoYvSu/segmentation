# -*- coding: utf-8 -*-
"""缓存N1真实训练热图与固定控制；只局部零噪声，不训练或修改后端。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def n1_config(n0):
    config = copy.deepcopy(n0)
    if config['submission'].get('local_restoration') != dict(mode='single', model='restorer'):
        raise ValueError('Expected the unchanged N0 input protocol')
    config['submission']['local_restoration'] = dict(mode='zero', model='restorer')
    return config


def training_sources(gt, n0, expected_count=32):
    rows = gt.get('rows', [])
    if (not n0.get('complete') or len(rows) != expected_count
            or len({r['source'] for r in rows}) != expected_count):
        raise ValueError('Complete distinct processed-GT sources required')
    old = {r['source']: r for r in n0.get('files', [])}
    if set(old) != {r['source'] for r in rows}:
        raise ValueError('N0 cache source set differs')
    result = []
    for row in sorted(rows, key=lambda r: r['source']):
        name = row['source']
        if not name.startswith('train_') or not name[6:].isdigit():
            raise ValueError('Only training sources are allowed')
        prior = old[name]
        if (prior['image_sha256'] != row['hashes']['image_sha256']
                or prior['processed_gt']['sha256'] != row['hashes']['gt_sha256']
                or prior['processed_gt']['classes_sha256'] != row['hashes']['classes_sha256']
                or prior['shape'] != row['shape']
                or prior['pool_match']['sha256'] != row['hashes']['image_sha256']
                or prior['pool_match']['full_pool_index'] != prior['source_index']
                or type(prior['source_index']) is not int or not 0 <= prior['source_index'] < 1000):
            raise ValueError('Training content/GT/source-index identity differs: ' + name)
        result.append(dict(source=name, domain='train_gt', source_index=prior['source_index'],
                           gt=row, n0=prior, shape=row['shape']))
    return result


def capture_n1_window(helper, model, restorer, rgb, box, config, seed, *, operators, zero_probe):
    audits = []

    def zero_restore(actual_restorer, image, actual_seed):
        audit = {}
        restored, _ = zero_probe(actual_restorer, image, actual_seed, 'zero', audit=audit)
        if (audit.get('noise_nonzero') != 0 or audit.get('initial_equals_condition') is not True
                or audit.get('q_sample_calls') != 1):
            raise RuntimeError('N1 local inference did not use one actual zero-noise initial state')
        audits.append(audit)
        return restored

    supplied = (operators[0], zero_restore, operators[1], operators[2])
    arrays, audit = helper.capture_one(model, restorer, rgb, box, config, seed, operators=supplied)
    if len(audits) != 1:
        raise RuntimeError('Expected exactly one local restoration call per window')
    return arrays, dict(**audit, actual_local_zero=audits[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['code-root', 'model-dir', 'gt-manifest', 'n0-heat-manifest', 'window-tool',
                 'n1-old-audit', 'n1-new-audit', 'n0-old-controls', 'n0-new-controls', 'output-root']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.code_root = args.code_root.resolve()
    args.output_root = args.output_root.resolve()
    heat_output, control_output = args.output_root / 'heatmaps', args.output_root / 'controls'
    if heat_output.exists() or control_output.exists():
        raise FileExistsError('New N1 cache/control directories must be fresh')
    n0, gt = read(args.n0_heat_manifest), read(args.gt_manifest)
    rows = training_sources(gt, n0)
    config = n1_config(n0['config'])
    if Path(config['paths']['project_root']).resolve() != args.code_root:
        raise RuntimeError('Sealed source root differs')
    for row in rows:
        for key in ['image', 'gt', 'classes']:
            if sha(row['gt'][key]) != row['gt']['hashes'][key + '_sha256']:
                raise RuntimeError('Registered source changed: ' + row['source'] + '/' + key)
        if sha(row['n0']['heat_path']) != row['n0']['heat_sha256']:
            raise RuntimeError('Preserved N0 heatmap changed')
    old_report, new_report = read(args.n1_old_audit), read(args.n1_new_audit)
    for report in [old_report, new_report]:
        if report['local_restoration'] != dict(mode='zero', model='restorer', candidate='N1'):
            raise RuntimeError('Reference report is not N1')
        for relative, digest in report['imported_sources'].items():
            if sha(args.code_root / relative) != digest:
                raise RuntimeError('Reference source changed: ' + relative)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output_root).free < sum(int(r['shape'][0])*int(r['shape'][1])*8 for r in rows) + 1024**3:
        raise RuntimeError('Insufficient fast-disk space')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is occupied: ' + active)
    sys.path.insert(0, str(args.code_root))
    import cv2
    import numpy as np
    import torch
    from tools import infer_semifinal as entry
    from tools.affinity_connectivity_views import _inference_state
    from tools.probe_affinity_patch import tensor_from_rgb, boundary_grid
    from tools.semantic_crossover import revote_instances
    from train_backend_adaptation import tensor_digest
    from utils.affinity_deployment import crop_letterbox_output
    from utils.image_io import read_image
    from utils.patch_diagnostic import BlendMap, tile_boxes
    from utils.restoration_probe import restore_probe
    spec = importlib.util.spec_from_file_location('_frozen_training_window', args.window_tool)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    entry.validate_config(config); entry.configure_runtime()
    paths = entry.model_paths(config, args.model_dir.resolve())
    model, restorer, terminal = entry.load_models(config, paths, 'cuda')
    if terminal is not None or entry.local_restoration(config)['candidate'] != 'N1':
        raise RuntimeError('Only original restorer with local zero mode is supported')
    state = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if state != n0['model_states'] or state != old_report['model_states'] or state != new_report['model_states']:
        raise RuntimeError('Frozen model identities differ')
    operators = tensor_from_rgb, boundary_grid, crop_letterbox_output
    source_digest = entry.verify_source_manifest(args.code_root)

    def capture_map(rgb, source_seed, prior_windows=None):
        boxes = tile_boxes(rgb.shape[:2], 1024, config['affinity_native']['overlap'])
        blend = BlendMap(rgb.shape[:2]); windows = []
        with _inference_state(model, restorer, 'cuda'):
            for index, box in enumerate(boxes):
                seed = source_seed + 100000 + 1024*1000 + index
                arrays, audit = capture_n1_window(helper, model, restorer, rgb, box, config, seed,
                    operators=operators, zero_probe=restore_probe)
                if prior_windows is not None:
                    old_window = prior_windows[index]
                    if (list(box) != old_window['box'] or seed != old_window['seed']
                            or audit['input_tensor_sha256'] != old_window['input_sha256']
                            or audit['restored_tensor_sha256'] != old_window['restored_sha256']):
                        raise RuntimeError('N1 local replay differs from official-entry reference')
                blend.add(box, arrays['boundary1024'])
                windows.append(dict(window=index, box=list(box), seed=seed, **audit))
        boundary, blend_audit = blend.finish()
        return boundary, blend_audit, windows

    # 固定四图：全图语义/原图末端语义复用N0缓存，逐项核与N1报告相同。
    control_output.mkdir(); config_path = control_output/'frozen_config.json'; write(config_path, config)
    controls = dict(format='boundary_restore_controls_v1', complete=False, training=False,
        config=str(config_path), config_sha256=sha(config_path), source_root=str(args.code_root),
        source_manifest=source_digest, models=n0['models'], model_states=state, local_restoration=dict(mode='zero', model='restorer'),
        capture_script_sha256=sha(__file__), test_gt_used=False, global_noise_unchanged=True, files=[])
    old_n0, new_n0 = read(args.n0_old_controls), read(args.n0_new_controls)
    for name in ['test_101', 'test_116', 'test_166', 'test_169']:
        reference_report = new_report if name == 'test_169' else old_report
        reference_audit_path = args.n1_new_audit if name == 'test_169' else args.n1_old_audit
        reference_root = reference_audit_path.parent.parent
        old = next(r for r in reference_report['images'] if r['source'] == name)
        n0_row = next(r for r in (new_n0 if name == 'test_169' else old_n0)['files'] if r['source'] == name)
        if sha(n0_row['path']) != n0_row['sha256'] or n0_row['fixed_paths'] != old['fixed_paths']:
            raise RuntimeError('Frozen whole-image paths differ from N1 reference')
        with np.load(n0_row['path'], allow_pickle=False) as z:
            rgb = z['rgb'].copy(); semantic = z['semantic'].copy(); probability = z['raw_probability'].copy()
        if hashlib.sha256(semantic.tobytes()).hexdigest() != old['fixed_paths']['geometry_semantic_sha256']:
            raise RuntimeError('Cached whole semantic differs')
        if hashlib.sha256(probability.tobytes()).hexdigest() != old['fixed_paths']['raw_probability_sha256']:
            raise RuntimeError('Cached terminal raw probability differs')
        if name == 'test_169':
            boundary, blend_audit, windows = capture_map(rgb, old['seed'], old['local_windows'])
            cache_reuse = 'native windows recomputed; all local input/restored SHA exact'
        else:
            native_path = reference_root / old['analysis']
            with np.load(native_path, allow_pickle=False) as z: boundary = z['boundary'].copy()
            blend_audit, windows = old['blend'], old['local_windows']
            replay, _, _ = capture_map(rgb, old['seed'], old['local_windows'])
            if not np.array_equal(replay, boundary):
                raise RuntimeError('New N1 capture operator does not reproduce reference boundary')
            cache_reuse = 'existing N1 native cache; independent local replay and reference window SHA exact'
        ids, _ = entry.decode_reflect(torch.from_numpy(semantic), torch.from_numpy(boundary)[None,None], config)
        classes, _ = revote_instances(ids, probability)
        png = reference_root/'results'/(name+'_inst.png'); cls_path = reference_root/'results'/(name+'_class.json')
        if sha(png) != old['png_sha256'] or sha(cls_path) != old['classes_sha256']:
            raise RuntimeError('Reference final outputs changed')
        expected_ids = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if not np.array_equal(ids, expected_ids) or classes != read(cls_path):
            raise RuntimeError('N1 fixed controls do not reproduce original final output')
        target = control_output/(name+'.npz')
        np.savez_compressed(target, rgb=rgb, boundary=boundary, semantic=semantic, raw_probability=probability, instances=ids)
        dest_png = control_output/png.name; dest_cls = control_output/cls_path.name
        shutil.copyfile(png, dest_png); shutil.copyfile(cls_path, dest_cls)
        controls['files'].append(dict(source=name, path=str(target), sha256=sha(target),
            control_png=str(dest_png), control_png_sha256=sha(dest_png), control_classes=str(dest_cls),
            control_classes_sha256=sha(dest_cls), input_sha256=old['input_sha256'], shape=list(boundary.shape),
            fixed_paths=old['fixed_paths'], source_boundary_cache_sha256=hashlib.sha256(boundary.tobytes()).hexdigest(),
            seed=old['seed'], reference_final_exact=True, decoder_ids_exact=True, raw_classes_exact=True,
            cache_reuse=cache_reuse, local_windows=windows))
        write(control_output/'control_manifest.json', controls)
        print('CONTROL_EXACT', name, flush=True)
    controls['complete'] = True; write(control_output/'control_manifest.json', controls)

    heat_output.mkdir()
    manifest = dict(format='boundary_naf_real_heatmaps_v1', complete=False, training=False, no_test_training=True,
        local_restoration=dict(mode='zero', model='restorer'), global_noise_unchanged=True,
        models=n0['models'], model_states=state, config=config, runtime=entry.runtime_record(),
        source_root=str(args.code_root), source_manifest=source_digest,
        processed_gt_manifest_sha256=sha(args.gt_manifest), n0_heat_manifest_sha256=sha(args.n0_heat_manifest),
        capture_script_sha256=sha(__file__), window_helper_sha256=sha(args.window_tool), files=[])
    started = time.time()
    for number, row in enumerate(rows, 1):
        rgb = cv2.cvtColor(read_image(row['gt']['image']), cv2.COLOR_BGR2RGB)
        source_seed = 314159 + 2000000 + row['source_index']
        boundary, blend_audit, windows = capture_map(rgb, source_seed)
        for audit, prior in zip(windows, row['n0']['windows']):
            if (audit['box'] != prior['box'] or audit['seed'] != prior['seed_audit']['local_seed']
                    or audit['input_tensor_sha256'] != prior['input_tensor_sha256']):
                raise RuntimeError('Training N1 input/order/seed differs from N0 capture')
        target = heat_output/(row['source']+'.npz'); np.savez_compressed(target, boundary=boundary)
        manifest['files'].append(dict(source=row['source'], heat_path=str(target), heat_sha256=sha(target),
            image_sha256=row['gt']['hashes']['image_sha256'], gt_sha256=row['gt']['hashes']['gt_sha256'],
            classes_sha256=row['gt']['hashes']['classes_sha256'], shape=row['shape'], source_index=row['source_index'],
            pool_match=row['n0']['pool_match'], train_namespace_offset=2000000,
            windows=windows, blend_audit=blend_audit, boundary_sha256=hashlib.sha256(boundary.tobytes()).hexdigest()))
        write(heat_output/'manifest.json', manifest)
        print('CAPTURED', number, len(rows), row['source'], flush=True)
    after = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if after != state or entry.verify_source_manifest(args.code_root) != source_digest:
        raise RuntimeError('Frozen model or sealed source changed')
    for report in [old_report, new_report]:
        for relative, digest in report['imported_sources'].items():
            if sha(args.code_root/relative) != digest:
                raise RuntimeError('Imported source changed during capture')
    manifest.update(complete=True, frozen_models_unchanged=True, sealed_source_unchanged=True,
                    elapsed_seconds=time.time()-started, total_cache_bytes=sum(Path(r['heat_path']).stat().st_size for r in manifest['files']))
    write(heat_output/'manifest.json', manifest)
    write(args.output_root/'capture_complete.json', dict(complete=True, training=False, source_count=len(rows),
        heat_manifest=str(heat_output/'manifest.json'), heat_manifest_sha256=sha(heat_output/'manifest.json'),
        control_manifest=str(control_output/'control_manifest.json'), control_manifest_sha256=sha(control_output/'control_manifest.json'),
        frozen_models_unchanged=True, original_sources_preserved=True, no_full_test_inference=True))
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
