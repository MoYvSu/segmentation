# -*- coding: utf-8 -*-
"""冻结 D5a 最终实例划分，探查三套语义头对数字亮度变化的分类敏感性。

这里只重跑 encoder 和语义头，不运行 affinity 或分水岭，不产出提交包。
RGB 加常量在未裁剪区域保持通道差及空间差；它不等同于物理光照变化。
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.semantic_crossover import revote_instances


def luminance(image):
    """输入 RGB 张量在当前数字编码空间的加权亮度，非物理辐照度。"""
    weights = image.new_tensor((.2126, .7152, .0722)).view(1, 3, 1, 1)
    return (image * weights).sum(dim=1, keepdim=True)


def adjust_brightness(image, kind, value):
    """保持 FP32 输入；恒等分支直接返回同一个张量，禁止 uint8 往返。"""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Brightness parameter must be finite')
    if kind == 'offset':
        if value == 0:
            return image
        return (image + value).clamp(0, 1)
    if kind == 'gamma':
        if value <= 0:
            raise ValueError('Gamma must be positive')
        if value == 1:
            return image
        light = luminance(image)
        return (image + light.clamp(0, 1).pow(value) - light).clamp(0, 1)
    raise ValueError(f'Unknown brightness transform: {kind}')


def unclipped_adjustment(image, kind, value):
    if kind == 'offset':
        return image + float(value)
    light = luminance(image)
    return image + light.clamp(0, 1).pow(float(value)) - light


def variant_key(kind, value):
    if kind == 'offset':
        return 'offset_0' if value == 0 else f'offset_{"p" if value > 0 else "m"}{round(abs(value) * 100):03d}'
    return f'gamma_{round(value * 100):03d}'


def variant_specs(offsets, gammas):
    offsets = list(map(float, offsets))
    if 0.0 not in offsets:
        offsets.append(0.0)
    specs = [{'key': variant_key(kind, value), 'kind': kind, 'value': value}
             for kind, values in (('offset', offsets), ('gamma', gammas)) for value in values]
    keys = [row['key'] for row in specs]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate variant keys; parameters must differ at two decimal places')
    if any(not np.isfinite(row['value']) or (row['kind'] == 'gamma' and row['value'] <= 0) for row in specs):
        raise ValueError('Invalid brightness parameters')
    return specs


def input_metrics(original, changed, kind, value, content_shape):
    height, width = content_shape
    original = original[:, :, :height, :width]
    changed = changed[:, :, :height, :width]
    before_clip = unclipped_adjustment(original, kind, value)
    clipped = (before_clip < 0) | (before_clip > 1)
    difference = luminance(changed) - luminance(original)
    return {
        'mean_y': float(luminance(changed).mean()),
        'mean_y_change': float(difference.mean()),
        'mean_absolute_y_change': float(difference.abs().mean()),
        'y_change_std': float(difference.std(unbiased=False)),
        'clipped_pixel_fraction': float(clipped.any(dim=1).float().mean()),
        'clipped_channel_fraction': float(clipped.float().mean()),
        'valid_pixels': int(height * width),
        'interpretation': ('Digital RGB offset; outside clipping, chroma and spatial differences are preserved.'
                           if kind == 'offset' else 'Digital luminance gamma; local contrast also changes.'),
    }


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _read_prediction(directory, stem, expected_shape):
    directory = Path(directory)
    instances = cv2.imread(str(directory / f'{stem}_inst.png'), cv2.IMREAD_UNCHANGED)
    if instances is None or instances.dtype != np.uint16 or instances.shape != expected_shape:
        raise ValueError(f'Invalid original-sized uint16 prediction: {directory}/{stem}')
    classes = json.loads((directory / f'{stem}_class.json').read_text(encoding='utf-8'))
    ids = {str(int(value)) for value in np.unique(instances) if value > 0}
    if set(classes) != ids or any(type(value) is not int or value not in (0, 1) for value in classes.values()):
        raise ValueError(f'Invalid class mapping: {directory}/{stem}')
    return instances, classes


def _geometry(instances):
    result = {}
    for value in np.unique(instances):
        if value == 0:
            continue
        ys, xs = np.nonzero(instances == value)
        result[str(int(value))] = {'area': int(len(xs)),
            'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            'centroid': [float(xs.mean()), float(ys.mean())]}
    return result


def _vote_summary(classes, scores, base_classes, geometry):
    ids = list(classes)
    changed = [key for key in ids if classes[key] != base_classes[key]]
    ferrite = [key for key in ids if classes[key] == 1]
    ferrite_area = sum(geometry[key]['area'] for key in ferrite)
    return {'classes': classes, 'vote_scores': scores, 'instances': len(ids),
        'ferrite': len(ferrite), 'pearlite': len(ids) - len(ferrite),
        'ferrite_pixels': ferrite_area, 'covered_pixels': sum(row['area'] for row in geometry.values()),
        'ferrite_mean_area': ferrite_area / len(ferrite) if ferrite else None,
        'changed_instances': len(changed), 'changed_pixels': sum(geometry[key]['area'] for key in changed),
        'ferrite_to_pearlite_ids': [key for key in changed if classes[key] == 0],
        'pearlite_to_ferrite_ids': [key for key in changed if classes[key] == 1]}


def _resize(array, shape, *, nearest=False):
    if tuple(array.shape[:2]) == tuple(shape):
        return array.copy()
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def _input_preview(tensor, content_shape, preview_shape):
    height, width = content_shape
    image = tensor[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
    image = np.rint(image.clip(0, 1) * 255).astype(np.uint8)
    return _resize(image, preview_shape)


@torch.no_grad()
def run_probe(config_path, output_dir, images, offsets, gammas, preview_width=1280):
    # 延迟模型导入，CPU 测试只需亮度变换，不加载 SAM2 依赖或任何权重。
    from data.mim_dataset import list_images
    from models.backend_adaptation import build_backend, load_backend, restore_first
    from train_backend_adaptation import sha
    from train_semantic_d5a import get_restorer, frozen_digest
    from utils.affinity_deployment import prepare_image, crop_letterbox_output
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Run the probe in sam2_env on the GPU server')
    if preview_width <= 0:
        raise ValueError('preview_width must be positive')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    config = load_config(config_path)
    cfg = config['semantic_adaptation']
    voting = config['inference']
    if (voting.get('semantic_vote_mode'), voting.get('semantic_vote_erode_width'),
            voting.get('semantic_vote_threshold')) != ('probability_mean', 0, .5):
        raise ValueError('Probe requires deployment probability_mean, no erosion, strict > 0.5')
    variants = variant_specs(offsets, gammas)
    files = [Path(path) for path in list_images(project_path(config, config['inference']['test_dir']))]
    if len({path.stem for path in files}) != len(files):
        raise ValueError('Duplicate input stems')
    selected = images or cfg['monitor']['images']
    if len(set(selected)) != len(selected) or not set(selected).issubset({path.stem for path in files}):
        raise ValueError('Requested images missing or duplicated')
    output = Path(project_path(config, output_dir))
    output.mkdir(parents=True, exist_ok=False)
    preview = output / 'previews'
    preview.mkdir()
    train_root = Path(project_path(config, cfg['output_dir']))
    epoch = int(cfg['epochs'])
    deployment = {'original': Path(project_path(config, cfg['d5a_prediction_dir'])),
                  'full': train_root / 'full' / 'deployment', 'simple': train_root / 'simple' / 'deployment'}
    checkpoints = {arm: train_root / arm / f'epoch_{epoch:03d}.pt' for arm in ('full', 'simple')}
    report = {'status': 'running', 'precision': 'FP32', 'variants': variants, 'models': {}, 'images': {},
        'config': str(Path(config_path)), 'fixed_instance_source': str(deployment['original']),
        'voting': {'mode': 'probability_mean', 'erode_width': 0, 'threshold': .5, 'strict_greater': True},
        'scope': 'Digital brightness sensitivity. Frozen original D5a final instances; no affinity forward, no watershed, no submission.',
        'limitations': ['No test GT; class flips do not establish correctness.',
                       'Digital RGB offset is not physical illumination recovery.',
                       'Gamma changes local contrast as well as brightness; clipping can also change detail/chroma.'],
        'restoration': {'checkpoint': cfg['restoration']['checkpoint'],
                        'sha256': cfg['restoration']['checkpoint_sha256'], 'epoch': 60,
                        'mode': 'first_x0', 'computed_once_per_image': True,
                        'seed_rule': 'inference_seed + index in full sorted test file list'},
    }
    report_path = output / 'results.json'
    started = time.time()
    cached = {}
    try:
        restorer = get_restorer(config, device)
        for index, path in enumerate(files):
            if path.stem not in selected:
                continue
            rgb, tensor, pad_h, pad_w = prepare_image(path, 1024, device)
            seed = int(cfg['restoration']['inference_seed']) + index
            restored = restore_first(restorer, tensor, seed).float()
            content_shape = (1024 - pad_h, 1024 - pad_w)
            native_shape = rgb.shape[:2]
            scale = min(1.0, float(preview_width) / native_shape[1])
            preview_shape = (max(1, round(native_shape[0] * scale)), max(1, round(native_shape[1] * scale)))
            fixed, _ = _read_prediction(deployment['original'], path.stem, native_shape)
            geometry = _geometry(fixed)
            inputs = {'raw_rgb': _resize(rgb, preview_shape), 'fixed_instances': _resize(fixed, preview_shape, nearest=True)}
            metrics = {}
            for spec in variants:
                changed = adjust_brightness(restored, spec['kind'], spec['value'])
                metrics[spec['key']] = input_metrics(restored, changed, spec['kind'], spec['value'], content_shape)
                inputs[spec['key']] = _input_preview(changed, content_shape, preview_shape)
                if not cv2.imwrite(str(preview / f'{path.stem}_{spec["key"]}.png'),
                                   cv2.cvtColor(inputs[spec['key']], cv2.COLOR_RGB2BGR)):
                    raise IOError('Could not save private input preview')
            cache_name = f'previews/{path.stem}_inputs.npz'
            np.savez_compressed(output / cache_name, **inputs)
            report['images'][path.stem] = {'stem': path.stem, 'source_image': str(path),
                'native_shape': list(native_shape), 'preview_shape': list(preview_shape),
                'input_content_shape': list(content_shape), 'restoration_seed': seed,
                'input_cache': cache_name, 'instance_geometry': geometry, 'input_metrics': metrics, 'models': {}}
            cached[path.stem] = {'restored': restored.cpu(), 'pad_h': pad_h, 'pad_w': pad_w,
                'native_shape': native_shape, 'preview_shape': preview_shape, 'fixed_instances': fixed}
            print(f'restored once: {path.stem}, seed={seed}', flush=True)
            _write_json(report_path, report)
        del restorer, tensor, restored, changed
        gc.collect()
        torch.cuda.empty_cache()

        reference_frozen_digest = None
        for name in ('original', 'full', 'simple'):
            if name == 'original':
                model, metadata = build_backend(config, device)
                model_info = {'sources': metadata['sources'], 'epoch': metadata['sources']['semantic'].get('epoch'),
                              'initialization': 'existing_original_D5a_semantic'}
            else:
                model, bundle = load_backend(str(checkpoints[name]), config, device)
                saved_arm = bundle['backend_adaptation'].get('semantic_adaptation', {}).get('arm')
                if saved_arm != name or int(bundle['epoch']) != epoch:
                    raise ValueError(f'{name}: checkpoint arm or epoch differs')
                model_info = {'checkpoint': str(checkpoints[name]), 'sha256': sha(checkpoints[name]),
                              'epoch': int(bundle['epoch']), 'architecture': bundle['architecture']['semantic_decoder']}
                del bundle
            model.eval().requires_grad_(False)
            model.encoder.trainable_lora = False
            digest = frozen_digest(model)
            if reference_frozen_digest is None:
                reference_frozen_digest = digest
            elif digest != reference_frozen_digest:
                raise RuntimeError(f'{name}: encoder/LoRA/affinity do not match original')
            model_info.update(nonsemantic_state_sha256=digest, nonsemantic_exact_across_models=True,
                              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                              all_modules_eval=all(not module.training for module in model.modules()),
                              deployment_directory=str(deployment[name]))
            report['models'][name] = model_info
            for stem, cache in cached.items():
                restored = cache['restored'].to(device)
                geometry = report['images'][stem]['instance_geometry']
                own_instances, own_classes = _read_prediction(deployment[name], stem, cache['native_shape'])
                all_votes, probabilities = {}, {}
                check = None
                # 先零偏移，确保所有扫描都只相对于各自未经调节的语义结果计数。
                ordered = sorted(variants, key=lambda row: row['key'] != 'offset_0')
                for spec in ordered:
                    changed = adjust_brightness(restored, spec['kind'], spec['value'])
                    # 不调用 model.forward：它会多跑 affinity，而此处固定所有最终实例。
                    features = model.encoder(changed)
                    logits = model.semantic_decoder(features, changed)
                    native_logits = crop_letterbox_output(logits, 1024, cache['pad_h'], cache['pad_w'], cache['native_shape']).cpu()
                    probability = torch.sigmoid(native_logits)[0, 0].numpy().astype(np.float32, copy=False)
                    classes, scores = revote_instances(cache['fixed_instances'], probability)
                    if spec['key'] == 'offset_0':
                        base_classes = classes
                        reproduced, reproduced_scores = revote_instances(own_instances, probability)
                        mismatch = [key for key in own_classes if own_classes[key] != reproduced[key]]
                        check = {'exact': not mismatch, 'instances': len(own_classes), 'mismatch_ids': mismatch,
                                 'prediction_directory': str(deployment[name]),
                                 'mismatch_scores': {key: reproduced_scores[key] for key in mismatch}}
                        if mismatch:
                            report['images'][stem]['models'][name] = {'zero_deployment_check': check}
                            raise RuntimeError(f'{name}/{stem}: zero brightness does not reproduce original deployment classes')
                    all_votes[spec['key']] = {**spec, **_vote_summary(classes, scores, base_classes, geometry)}
                    probabilities[spec['key']] = _resize(probability, cache['preview_shape']).astype(np.float32, copy=False)
                    print(f'probe {name}/{stem}/{spec["key"]}: flips={all_votes[spec["key"]]["changed_instances"]}', flush=True)
                    del features, logits, native_logits, probability, changed
                cache_name = f'previews/{stem}_{name}.npz'
                np.savez_compressed(output / cache_name, **probabilities)
                report['images'][stem]['models'][name] = {'zero_deployment_check': check,
                    'variants': all_votes, 'probability_cache': cache_name}
                _write_json(report_path, report)
                del restored, probabilities, own_instances
            if frozen_digest(model) != digest:
                raise RuntimeError(f'{name}: frozen state changed during inference')
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report['summary'] = {}
        for name in report['models']:
            report['summary'][name] = {}
            for spec in variants:
                rows = [image['models'][name]['variants'][spec['key']] for image in report['images'].values()]
                report['summary'][name][spec['key']] = {
                    key: sum(row[key] for row in rows) for key in
                    ('instances', 'ferrite', 'pearlite', 'changed_instances', 'changed_pixels', 'ferrite_pixels', 'covered_pixels')}
                report['summary'][name][spec['key']].update(
                    ferrite_to_pearlite=sum(len(row['ferrite_to_pearlite_ids']) for row in rows),
                    pearlite_to_ferrite=sum(len(row['pearlite_to_ferrite_ids']) for row in rows))
        report.update(status='completed', elapsed_seconds=time.time() - started,
                      zero_deployment_exact=all(image['models'][name]['zero_deployment_check']['exact']
                                                for image in report['images'].values() for name in report['models']))
    except BaseException as error:
        report.update(status='failed', error=repr(error), elapsed_seconds=time.time() - started)
        raise
    finally:
        _write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_d5a.yaml')
    parser.add_argument('--output-dir', default='outputs/brightness_probe')
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--offsets', type=float, nargs='+', default=[-.20, -.10, -.05, 0, .05, .10, .20])
    parser.add_argument('--gammas', type=float, nargs='*', default=[.8, 1.25])
    parser.add_argument('--preview-width', type=int, default=1280)
    args = parser.parse_args()
    report = run_probe(args.config, args.output_dir, args.images, args.offsets, args.gammas, args.preview_width)
    print(json.dumps({key: report[key] for key in ('status', 'zero_deployment_exact', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
