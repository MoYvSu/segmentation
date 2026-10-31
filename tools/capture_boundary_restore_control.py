# -*- coding: utf-8 -*-
"""补边对照的最小全图缓存：复用旧 native 边界，仅补齐冻结两路语义。

不运行 native 窗口、不训练、不改既有输出。三个全图摘要以及原始最终
PNG/JSON 必须精确一致，才发布 control_manifest.json 的 complete 状态。
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


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', nargs='+', required=True)
    args = parser.parse_args()
    args.code_root = args.code_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Control evidence requires a fresh output directory')
    if len(args.images) != len(set(args.images)):
        raise ValueError('Duplicate source selection')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                     '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU is occupied; do not overlap capture with training: '+active)
    sys.path.insert(0, str(args.code_root))
    import cv2
    import numpy as np
    import torch
    from tools import infer_semifinal as entry
    from tools.affinity_connectivity_views import _inference_state
    from tools.semantic_crossover import revote_instances
    from train_backend_adaptation import prediction_maps, tensor_digest
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    original = json.loads((args.baseline / 'audit/report.json').read_text(encoding='utf-8'))
    config = original['config']
    if Path(config['paths']['project_root']).resolve() != args.code_root:
        raise RuntimeError('Original source root differs')
    for name, expected in original['imported_sources'].items():
        if sha(args.code_root / name) != expected:
            raise RuntimeError('Sealed source differs: '+name)
    entry.validate_config(config)
    entry.configure_runtime()
    paths = entry.model_paths(config, args.model_dir.resolve())
    model, restorer, terminal = entry.load_models(config, paths, 'cuda')
    if terminal is not None or entry.local_restoration(config) != dict(mode='single', model='restorer', candidate='N0'):
        raise RuntimeError('Capture requires unchanged N0')
    state = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if state != original['model_states']:
        raise RuntimeError('Original frozen model state differs')
    args.output.mkdir(parents=True)
    (args.output / 'source').mkdir()
    shutil.copyfile(__file__, args.output / 'source' / Path(__file__).name)
    config_path = args.output / 'frozen_config.json'
    write(config_path, config)
    manifest = dict(format='boundary_restore_controls_v1', complete=False, training=False,
        config=str(config_path), config_sha256=sha(config_path), source_root=str(args.code_root),
        source_manifest=entry.verify_source_manifest(args.code_root), model_states=state,
        models=original['models'], baseline=str(args.baseline.resolve()),
        baseline_audit_sha256=sha(args.baseline / 'audit/report.json'),
        capture_script_sha256=sha(__file__), runtime=entry.runtime_record(),
        native_windows_recomputed=False, test_gt_used=False, files=[])
    old_rows = {row['source']: row for row in original['images']}
    start = time.time()
    for stem in args.images:
        old = old_rows[stem]
        matches = [p for p in args.input_dir.iterdir() if p.is_file() and p.stem == stem]
        if len(matches) != 1 or sha(matches[0]) != old['input_sha256']:
            raise RuntimeError('Original image identity differs: '+stem)
        image_path = matches[0]
        old_png = args.baseline / 'results' / (stem+'_inst.png')
        old_classes = args.baseline / 'results' / (stem+'_class.json')
        if sha(old_png) != old['png_sha256'] or sha(old_classes) != old['classes_sha256']:
            raise RuntimeError('Original final results changed')
        print('CAPTURE', stem, flush=True)
        seed = entry.source_seed(stem, config['submission'].get('dataset'))
        with _inference_state(model, restorer, 'cuda'):
            rgb, semantic, global_boundary = prediction_maps(model, image_path, config, 'cuda', restorer, seed)
            _, tensor, ph, pw = prepare_image(image_path, 1024, 'cuda')
            raw = model.semantic_decoder(model.encoder(tensor.float()), tensor.float())
            if isinstance(raw, dict):
                raw = raw['semantic_logits']
            probability = crop_letterbox_output(raw.float(), 1024, ph, pw, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()
        fixed_paths = dict(
            geometry_semantic_sha256=hashlib.sha256(semantic.contiguous().numpy().tobytes()).hexdigest(),
            raw_probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
            global_boundary_sha256=hashlib.sha256(global_boundary.contiguous().numpy().tobytes()).hexdigest())
        if fixed_paths != old['fixed_paths']:
            raise RuntimeError('Frozen whole-image maps differ from N0: '+stem)
        cached_path = args.baseline / old['analysis']
        with np.load(cached_path, allow_pickle=False) as cache:
            boundary = cache['boundary'].copy()
        if boundary.dtype != np.float32 or boundary.shape != tuple(rgb.shape[:2]):
            raise RuntimeError('Native boundary cache contract differs')
        ids, _ = entry.decode_reflect(semantic, torch.from_numpy(boundary)[None, None], config)
        classes, _ = revote_instances(ids, probability)
        expected_ids = cv2.imread(str(old_png), cv2.IMREAD_UNCHANGED)
        expected_classes = json.loads(old_classes.read_text(encoding='utf-8'))
        entry.verify_prediction(ids, classes, boundary.shape)
        if not np.array_equal(ids, expected_ids) or classes != expected_classes:
            raise RuntimeError('Captured native decoder or raw vote does not reproduce N0')
        destination = args.output / (stem+'.npz')
        np.savez_compressed(destination, rgb=rgb, boundary=boundary,
                            semantic=semantic.contiguous().numpy(), raw_probability=probability, instances=ids)
        png = args.output / (stem+'_inst.png')
        class_path = args.output / (stem+'_class.json')
        shutil.copyfile(old_png, png)
        shutil.copyfile(old_classes, class_path)
        manifest['files'].append(dict(source=stem, path=str(destination), sha256=sha(destination),
            control_png=str(png), control_png_sha256=sha(png),
            control_classes=str(class_path), control_classes_sha256=sha(class_path),
            input_sha256=old['input_sha256'], source_boundary_cache_sha256=sha(cached_path),
            seed=seed, shape=list(boundary.shape), fixed_paths=fixed_paths,
            decoder_ids_exact=True, raw_classes_exact=True))
        write(args.output / 'control_manifest.json', manifest)
        print('EXACT', stem, flush=True)
    after = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if state != after:
        raise RuntimeError('Frozen model state changed during capture')
    for name, expected in original['imported_sources'].items():
        if sha(args.code_root / name) != expected:
            raise RuntimeError('Sealed source changed during capture: '+name)
    manifest.update(complete=True, source_unchanged=True, frozen_models_unchanged=True,
                    elapsed_seconds=time.time()-start)
    write(args.output / 'control_manifest.json', manifest)
    print('COMPLETE', manifest['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
