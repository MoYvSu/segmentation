# -*- coding: utf-8 -*-
"""冻结已部署模型，隔离全图/局部恢复噪声的变化；仅诊断，不训练或选测试最优种子。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.rgb_spatial_blur import apply_sigma_map_blur
from models.backend_adaptation import restore_first
from tools import infer_semifinal as deployment
from tools.probe_affinity_patch import boundary_grid, tensor_from_rgb
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import tensor_digest
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.config import load_config
from utils.image_io import read_image, write_image
from utils.patch_diagnostic import BlendMap, phase_description, reconstruction_metrics, tile_boxes
from utils.restoration_noise import restoration_noise
from utils.restoration_probe import restore_probe

SOURCES = ('test_088', 'test_101', 'test_116', 'test_141', 'test_166', 'test_167')
OFFSETS = (0, 1, 17, 101, 1009, 10007, 104729, 130363)
VARIANTS = [(f's{i}', 'single', offset) for i, offset in enumerate(OFFSETS)] + [
    ('zero', 'zero', 0), ('pair', 'pair', 0)]
CASES = ['control'] + [f'{group}_{suffix}' for suffix, _, _ in VARIANTS
                        for group in ('local', 'global', 'both')] + ['local_mean']


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def restore_difference(value, reference):
    diff = value - reference
    return dict(mae=float(diff.abs().mean()), mse=float((diff * diff).mean()),
                max_abs=float(diff.abs().max()))


def thumbnail(tensor):
    rgb = tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    return cv2.resize(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)


def save_restore_strip(path, panels):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bodies = []
    for name, rgb in panels:
        body = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        body = cv2.copyMakeBorder(body, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(body, name, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (10, 10, 10), 1)
        bodies.append(body)
    write_image(path, np.concatenate(bodies, axis=1))


def save_case(output, source, case, boundary, semantic, probability, config):
    ids, _ = deployment.decode_reflect(torch.from_numpy(semantic)[None, None],
                                       torch.from_numpy(boundary)[None, None], config)
    classes, _ = revote_instances(ids, probability)
    deployment.verify_prediction(ids, classes, boundary.shape)
    directory = output / 'results' / case
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / (source + '_inst.png')
    write_image(png, ids)
    write(directory / (source + '_class.json'), classes)
    if not np.array_equal(read_image(png, cv2.IMREAD_UNCHANGED), ids):
        raise RuntimeError('Saved uint16 PNG differs')
    maps = output / 'maps' / source
    maps.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(maps / (case + '.npz'), boundary=boundary.astype(np.float32),
                        geometry_semantic=semantic.astype(np.float32), raw_probability=probability,
                        instances=ids)
    return dict(phases=phase_description(ids, classes), boundary_sha256=digest(boundary),
                geometry_semantic_sha256=digest(semantic), raw_probability_sha256=digest(probability)), ids, classes


@torch.no_grad()
def image_probe(path, model, restorer, config, output, *, formal_check):
    seed = deployment.source_seed(path.stem)
    rgb, image, ph, pw = prepare_image(path, 1024, 'cuda')
    shape = rgb.shape[:2]
    raw = model.semantic_decoder(model.encoder(image.float()), image.float())
    if isinstance(raw, dict):
        raw = raw['semantic_logits']
    probability = crop_letterbox_output(raw.float(), 1024, ph, pw, shape).cpu().sigmoid()[0, 0].numpy()
    del raw
    global_semantic, stats = {}, dict(global_restore={}, local_restore={})
    restored0 = None; positive0 = None
    global_panels = [('input', thumbnail(image))]
    for suffix, mode, offset in VARIANTS:
        restored, x0 = restore_probe(restorer, image, seed + offset, mode,
                                     positive_raw=positive0 if mode == 'pair' else None)
        if suffix == 's0':
            restored0, positive0 = restored, x0
            official_restore = restore_first(restorer, image, seed)
            if not torch.equal(restored, official_restore):
                raise RuntimeError('Probe single restoration differs from formal restore_first')
            del official_restore
        stats['global_restore'][suffix] = restore_difference(restored, restored0)
        # 与正式 prediction_maps 一样调用完整后端，避免改变 decoder 执行契约。
        logits = model(restored)['semantic_logits']
        global_semantic[suffix] = crop_letterbox_output(logits, 1024, ph, pw, shape).cpu()[0, 0].numpy()
        if suffix in ('s0', 's1', 'zero', 'pair'):
            global_panels.append((suffix, thumbnail(restored)))
        del restored, x0, logits
    save_restore_strip(output / 'restoration' / (path.stem + '_global.png'), global_panels)
    del image, restored0, positive0
    boxes = tile_boxes(shape, 1024, config['affinity_native']['overlap'])
    blends = {suffix: BlendMap(shape) for suffix, _, _ in VARIANTS}
    for index, box in enumerate(boxes):
        y, x, y1, x1 = box
        image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], 'cuda')
        tile_seed = seed + 100000 + 1024 * 1000 + index
        restored0 = None; positive0 = None
        panels = [('input', thumbnail(image))]
        for suffix, mode, offset in VARIANTS:
            restored, x0 = restore_probe(restorer, image, tile_seed + offset, mode,
                                         positive_raw=positive0 if mode == 'pair' else None)
            if suffix == 's0':
                restored0, positive0 = restored, x0
            stats['local_restore'].setdefault(suffix, []).append(restore_difference(restored, restored0))
            grid = boundary_grid(model, restored, config)
            native = crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
            blends[suffix].add(box, native)
            if index == 0 and suffix in ('s0', 's1', 'zero', 'pair'):
                panels.append((suffix, thumbnail(restored)))
            del restored, x0, grid, native
        if index == 0:
            save_restore_strip(output / 'restoration' / (path.stem + '_tile0.png'), panels)
        del image, restored0, positive0
        print(json.dumps(dict(stage='tile', source=path.stem, completed=index+1, tiles=len(boxes))), flush=True)
    boundaries = {suffix: blend.finish()[0] for suffix, blend in blends.items()}
    del blends
    fixed_semantic = global_semantic['s0']
    row, ids, classes = save_case(output, path.stem, 'control', boundaries['s0'], fixed_semantic,
                                 probability, config)
    if formal_check:
        _, official = deployment.predict(model, restorer, None, path, config, 'cuda')
        if (not np.array_equal(official['boundary'], boundaries['s0'])
                or not np.array_equal(official['instances'], ids) or official['classes'] != classes):
            raise RuntimeError('Control differs from complete formal deployment: ' + path.stem)
        row['formal_control_exact'] = True
        del official
    result = dict(control=row)
    for suffix, _, _ in VARIANTS:
        for group in ('local', 'global', 'both'):
            case = group + '_' + suffix
            boundary = boundaries['s0'] if group == 'global' else boundaries[suffix]
            semantic = fixed_semantic if group == 'local' else global_semantic[suffix]
            result[case] = save_case(output, path.stem, case, boundary, semantic, probability, config)[0]
            print(json.dumps(dict(stage='decoded', source=path.stem, case=case,
                                  ferrite=result[case]['phases']['ferrite'])), flush=True)
    mean_boundary = np.mean(np.stack([boundaries[f's{i}'] for i in range(len(OFFSETS))]), axis=0, dtype=np.float32)
    result['local_mean'] = save_case(output, path.stem, 'local_mean', mean_boundary,
                                     fixed_semantic, probability, config)[0]
    # 明确验证隔离：local只改边界，global只改geometry语义，所有case的末端语义相同。
    for name, value in result.items():
        if value['raw_probability_sha256'] != row['raw_probability_sha256']:
            raise RuntimeError('Raw final-vote semantic changed')
        if name.startswith('local_') and value['geometry_semantic_sha256'] != row['geometry_semantic_sha256']:
            raise RuntimeError('Local case changed geometry semantic')
        if name.startswith('global_') and value['boundary_sha256'] != row['boundary_sha256']:
            raise RuntimeError('Global case changed native boundary')
    return dict(input_sha256=deployment.sha(path), source_seed=seed, boxes=boxes,
                cases=result, restoration=stats, isolation_passed=True)


def weak_gradient_metrics(reference, prediction):
    def gradient(image):
        gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        return (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8,
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8)
    rx, ry = gradient(reference); px, py = gradient(prediction)
    rg, pg = np.hypot(rx, ry), np.hypot(px, py)
    # 排除裁块外缘；分位数完全由清晰参照确定，不依候选调整。
    valid = np.ones(rg.shape, bool); valid[:8] = False; valid[-8:] = False
    valid[:, :8] = False; valid[:, -8:] = False
    lo, hi = np.quantile(rg[valid], [.5, .85])
    weak = valid & (rg >= max(float(lo), .003)) & (rg < hi)
    strong = valid & (rg >= hi)
    flat = valid & (rg < max(float(lo), .003))
    directional = np.maximum(0, (rx*px + ry*py) / np.maximum(rg, 1e-8))
    return dict(weak_reference_pixels=int(weak.sum()),
                weak_gradient_l1=float(np.mean(np.abs(pg[weak] - rg[weak]))) if weak.any() else None,
                weak_gradient_ratio=float(pg[weak].mean() / max(1e-8, rg[weak].mean())) if weak.any() else None,
                weak_directional_retention=float(directional[weak].sum() / max(1e-8, rg[weak].sum())) if weak.any() else None,
                strong_directional_retention=float(directional[strong].sum() / max(1e-8, rg[strong].sum())) if strong.any() else None,
                flat_gradient_mean=float(pg[flat].mean()) if flat.any() else None)


@torch.no_grad()
def quality_probe(directory, restorer, output):
    files = sorted(p for p in Path(directory).iterdir()
                   if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
    if len(files) != 1000:
        raise RuntimeError('Expected exactly 1000 allowed original training images')
    rng = np.random.default_rng(20261008)
    selected = sorted(int(i) for i in rng.choice(len(files), 4, replace=False))
    plan = []; images = []
    for i in selected:
        path = files[i]
        bgr = read_image(path); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if min(h, w) < 1024:
            raise RuntimeError('Quality plan needs native1024 crops; no padding supervision')
        y = int(rng.integers(h-1024+1)); x = int(rng.integers(w-1024+1))
        images.append((path.stem, rgb[y:y+1024, x:x+1024].astype(np.float32) / 255))
        plan.append(dict(source=path.stem, input_sha256=deployment.sha(path), box=[y, x, y+1024, x+1024]))
    write(output / 'quality' / 'plan.json', dict(selection_seed=20261008, images=plan,
          scope='Training originals; mechanism only, no held-out generalization or instance GT'))
    rows = []
    for index, (source, clear) in enumerate(images):
        phase = np.linspace(0, 1, 1024, dtype=np.float32)[None, :]
        phase = phase * phase * (3 - 2 * phase)
        sigma = np.broadcast_to(.8 + (4.5-.8) * phase, (1024, 1024)).copy()
        conditions = dict(identity=clear, uniform=cv2.GaussianBlur(clear, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT),
                          spatial=apply_sigma_map_blur(clear, sigma, (.8, 4.5), levels=8))
        for condition, degraded in conditions.items():
            image = torch.from_numpy(degraded).permute(2, 0, 1)[None].to('cuda')
            input_metrics = {**reconstruction_metrics(clear, degraded), **weak_gradient_metrics(clear, degraded)}
            panels = [('clear', thumbnail(torch.from_numpy(clear).permute(2, 0, 1)[None])),
                      (condition, thumbnail(image))]
            for offset in OFFSETS[:3]:
                seed = 730000 + index * 100 + offset
                single, raw = restore_probe(restorer, image, seed)
                for mode in ('single', 'pair'):
                    restored = single if mode == 'single' else restore_probe(restorer, image, seed, 'pair', positive_raw=raw)[0]
                    pred = restored[0].cpu().permute(1, 2, 0).numpy()
                    rows.append(dict(source=source, condition=condition, mode=mode, offset=offset,
                        input_metrics=input_metrics, **reconstruction_metrics(clear, pred), **weak_gradient_metrics(clear, pred)))
                    if offset == 0:
                        panels.append((mode, thumbnail(restored)))
            zero = restore_probe(restorer, image, 730000 + index*100, 'zero')[0]
            pred = zero[0].cpu().permute(1, 2, 0).numpy()
            rows.append(dict(source=source, condition=condition, mode='zero', offset=0,
                             input_metrics=input_metrics, **reconstruction_metrics(clear, pred), **weak_gradient_metrics(clear, pred)))
            panels.append(('zero', thumbnail(zero)))
            save_restore_strip(output / 'quality' / f'{source}_{condition}.png', panels)
            write(output / 'quality' / 'metrics.json', dict(complete=False, plan=plan, rows=rows))
            print(json.dumps(dict(stage='quality', source=source, condition=condition)), flush=True)
    write(output / 'quality' / 'metrics.json', dict(complete=True, plan=plan, rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--train-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--only', nargs='+', choices=SOURCES, help='Smoke only; full default is the predeclared six')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError('Use a fresh experiment directory')
    output.mkdir(parents=True)
    started = time.time(); deployment.configure_runtime()
    config = load_config(str(args.config.resolve())); config['paths']['project_root'] = str(ROOT)
    deployment.validate_config(config)
    paths = deployment.model_paths(config, args.model_dir.resolve())
    sources = list(SOURCES if args.only is None else args.only)
    metadata = dict(format='restoration_noise_probe_v1', complete=False, training=False, optimizer_steps=0,
        sources=sources, expected_cases=CASES, offsets=list(OFFSETS), selection_seed=20261007,
        selection='fixed101/116 plus four random of remaining98 before predictions',
        config=config, config_sha256=deployment.sha(args.config), runtime=deployment.runtime_record(),
        models={key: dict(file=value.name, sha256=deployment.sha(value)) for key, value in paths.items()},
        policy='raw final semantic fixed; zero still denoises; pair averages unclipped x0; mean averages continuous boundaries',
        images={})
    write(output / 'record.json', metadata)
    try:
        model, restorer, terminal = deployment.load_models(config, paths, 'cuda')
        if terminal is not None:
            raise RuntimeError('Terminal semantic must be original fixed head')
        states = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
        if any(m.training or any(p.requires_grad for p in m.parameters()) for root in (model, restorer) for m in root.modules()):
            raise RuntimeError('All model parameters must be frozen/eval')
        # 正式4090原噪声与便携固定映射必须逐像素相同。
        test_seed = deployment.source_seed('test_101')
        reference = torch.randn((1, 3, 1024, 1024), device='cuda',
                                generator=torch.Generator(device='cuda').manual_seed(test_seed))
        portable = restoration_noise(reference.shape, test_seed, 'cuda')
        if torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 4090' and not torch.equal(reference, portable):
            raise RuntimeError('4090 portable noise differs from original CUDA generator')
        metadata['noise_protocol_vs_current_cuda'] = dict(exact=torch.equal(reference, portable),
            max_abs=float((reference-portable).abs().max()), shape=list(reference.shape))
        del reference, portable
        for path in deployment.input_files(args.input_dir, sources):
            metadata['images'][path.stem] = image_probe(path, model, restorer, config, output, formal_check=True)
            write(output / 'record.json', metadata)
            print(json.dumps(dict(stage='image_complete', source=path.stem, seconds=time.time()-started)), flush=True)
        quality_probe(args.train_dir, restorer, output)
        after = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
        if states != after:
            raise RuntimeError('Model state changed during inference')
        for key, path in paths.items():
            if deployment.sha(path) != metadata['models'][key]['sha256']:
                raise RuntimeError('Model file changed')
        metadata.update(complete=True, frozen_models_unchanged=True, model_states=states,
                        imported_sources=deployment.local_module_sources(), elapsed_seconds=time.time()-started)
        write(output / 'record.json', metadata)
    except Exception as error:
        metadata.update(error=repr(error), elapsed_seconds=time.time()-started)
        write(output / 'record.json', metadata)
        raise


if __name__ == '__main__':
    main()
