# -*- coding: utf-8 -*-
"""双噪声恢复短训的固定私有缩略图；不构建验证集或执行分割。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from data.rgb_restoration_dataset import prepare_rgb, read_rgb
from train_rgb_restoration import reconstruction_errors, save_checkpoint, write_json
from utils.config import project_path
from utils.image_io import write_image
from utils.restoration_noise import NOISE_MODE, restoration_noise
from utils.restoration_probe import finish, first_raw


DEFAULT_TEST_IMAGES = ('test_088', 'test_101', 'test_116', 'test_141', 'test_166', 'test_167')
CONDITIONS = ('identity', 'uniform', 'endpoint')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _roi(target, valid, size):
    """清晰参考中的弱图像梯度决定窗口；不根据恢复结果挑选，也不是晶界GT。"""
    gray = target.float().mean(0).numpy()
    mask = valid[0].numpy() > 0
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    samples = magnitude[mask & (magnitude > 1e-5)]
    if len(samples):
        low, high = np.quantile(samples, [.25, .65])
        weak = ((magnitude >= low) & (magnitude <= high) & mask).astype(np.float32)
    else:
        weak = mask.astype(np.float32)
    height, width = gray.shape
    side = min(int(size), height, width)
    # 按真实窗口内的弱梯度覆盖找固定窗口；离散网格使选择不依赖OpenCV滤波边缘方式。
    ys = sorted(set([0, max(0, height-side), *range(0, max(1, height-side+1), max(1, side//2))]))
    xs = sorted(set([0, max(0, width-side), *range(0, max(1, width-side+1), max(1, side//2))]))
    y, x = max(((y, x) for y in ys for x in xs),
               key=lambda pos: float(weak[pos[0]:pos[0]+side, pos[1]:pos[1]+side].sum()))
    return [int(y), int(x), int(side), int(side)]


def prepare_monitor(config, dataset, output_root):
    """预随机选训练源，保存配对、有效区和回执；不修改传入数据集及全局随机状态。"""
    settings = copy.deepcopy(config.get('restore_consistency', {}).get('monitor', {}))
    settings.setdefault('seed', 20261008)
    settings.setdefault('samples', 4)
    settings.setdefault('thumbnail_size', 192)
    settings.setdefault('roi_size', 128)
    if (int(settings['samples']) < 1 or int(settings['thumbnail_size']) < 16
            or int(settings['roi_size']) < 4):
        raise ValueError('monitor requires positive samples, thumbnail_size>=16 and roi_size>=4')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(range(len(dataset.samples)), key=lambda index: str(dataset.samples[index]))
    generator = np.random.default_rng(int(settings['seed']))
    positions = generator.choice(len(ordered), min(int(settings['samples']), len(ordered)), replace=False)
    indices = sorted(ordered[int(position)] for position in positions)
    if not indices:
        raise ValueError('monitor has no training sources')
    source_epoch = int(settings.get('source_epoch', config.get('restore_consistency', {}).get('data_epoch_offset', 60)))
    uniform, endpoint = copy.deepcopy(dataset), copy.deepcopy(dataset)
    for fixed in (uniform, endpoint):
        fixed.seed = int(settings['seed'])
        fixed.set_epoch(source_epoch)
        fixed.masked_pretraining = None
        fixed.training_noise = None
        fixed.probabilities = np.asarray([0., 1., 0., 0.])
        fixed.degradation['profile_probabilities'] = [0., 1., 0., 0.]
    if 'spatial_blur' not in endpoint.degradation:
        raise ValueError('endpoint monitor requires the D5a spatial_blur recipe')
    uniform.degradation['spatial_blur']['probability'] = 0.
    uniform.degradation['spatial_blur'].pop('endpoint_transition', None)
    endpoint.degradation['spatial_blur'].update(enabled=True, probability=1.)
    transition = endpoint.degradation['spatial_blur'].get('endpoint_transition')
    if not transition:
        raise ValueError('endpoint monitor requires the D5a endpoint_transition recipe')
    transition.update(enabled=True, probability=1.)
    pairs, receipts = [], []
    for position, index in enumerate(indices):
        original, spatial = uniform[index], endpoint[index]
        if (not torch.equal(original['target'], spatial['target'])
                or not torch.equal(original['valid'], spatial['valid'])):
            raise RuntimeError('uniform and endpoint monitor crop geometry differs')
        if not spatial.get('endpoint_blur_applied') or not spatial.get('sigma_crop_coexist'):
            raise RuntimeError('endpoint monitor did not exercise weak and strong regions')
        source = Path(dataset.samples[index])
        for condition, item in (('identity', original), ('uniform', original), ('endpoint', spatial)):
            target, valid = item['target'].clone(), item['valid'].clone()
            value = target.clone() if condition == 'identity' else item['input'].clone()
            receipt = dict(source=source.name, source_index=int(index), source_sha256=_sha(source),
                           condition=condition, dataset_seed=int(settings['seed']), dataset_epoch=source_epoch,
                           noise_seeds=[int(settings['seed'])+100000+position*2,
                                        int(settings['seed'])+100001+position*2],
                           valid_pixels=int(valid.sum()), shape=list(value.shape),
                           weak_reference_gradient_roi=_roi(target, valid, settings['roi_size']),
                           masked_pretraining=False, extra_training_noise=False)
            if condition != 'identity':
                receipt['degradation'] = {key: _plain(value) for key, value in item.items()
                                           if key not in ('input', 'target', 'valid', 'source')}
            pairs.append(dict(input=value, target=target, valid=valid, receipt=receipt))
            receipts.append(receipt)
    tests = []
    directory = settings.get('test_dir')
    if directory:
        directory = Path(project_path(config, str(directory)))
        available = [path for path in directory.rglob('*') if path.is_file()
                     and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}]
        for position, name in enumerate(settings.get('test_images', DEFAULT_TEST_IMAGES)):
            matches = [path for path in available if path.stem == name or path.name == name]
            if len(matches) != 1:
                raise ValueError(f'monitor requires exactly one test image {name}, found {len(matches)}')
            path = matches[0]
            boxed, valid = prepare_rgb(read_rgb(path), 1024)
            tensor = torch.from_numpy(boxed.transpose(2, 0, 1).copy())
            receipt = dict(source=path.name, source_sha256=_sha(path), has_ground_truth=False,
                           noise_seeds=[int(settings['seed'])+200000+position*2,
                                        int(settings['seed'])+200001+position*2],
                           geometry='letterbox1024_reflect', valid_pixels=int(valid.sum()))
            tests.append(dict(input=tensor, valid=torch.from_numpy(valid[None].copy()), receipt=receipt))
    manifest = dict(purpose='seen_training_mechanism_diagnostic_and_unlabeled_test_monitor_only',
                    has_holdout=False, seed=int(settings['seed']), source_indices=indices,
                    source_epoch=source_epoch, training_geometry='existing_dataset_letterbox_then_crop',
                    conditions=list(CONDITIONS), roi_definition='clear_reference_weak_image_gradient_not_boundary_GT',
                    settings=settings, pairs=receipts, tests=[item['receipt'] for item in tests],
                    degradation=copy.deepcopy(endpoint.degradation),
                    noise='two_independent_standard_normals; CUDA portable protocol, CPU dedicated generators')
    context = dict(settings=settings, pairs=pairs, tests=tests, manifest=manifest)
    save_checkpoint(output_root/'monitor_fixed.pt', context)
    write_json(output_root/'monitor_manifest.json', manifest)
    return context


def _noise(shape, seed, device):
    if device.type == 'cuda':
        return restoration_noise(shape, int(seed), device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(shape, device=device, dtype=torch.float32, generator=generator)


def _errors(left, right, valid):
    return {key: float(value[0]) for key, value in reconstruction_errors(left, right, valid).items()}


def _rgb(tensor):
    return tensor.detach().float().cpu().permute(1, 2, 0).numpy()


def _panel(arrays, labels, size):
    panels = []
    for array, label in zip(arrays, labels):
        image = np.round(np.clip(array, 0., 1.)*255).astype(np.uint8)
        height, width = image.shape[:2]
        scale = size/max(height, width)
        image = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))),
                           interpolation=cv2.INTER_AREA)
        canvas = np.full((size+26, size, 3), 245, dtype=np.uint8)
        y, x = 26+(size-image.shape[0])//2, (size-image.shape[1])//2
        canvas[y:y+image.shape[0], x:x+image.shape[1]] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(canvas, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1, cv2.LINE_AA)
        panels.append(canvas)
    return np.concatenate(panels, axis=1)


def _save_training(path, item, output_a, output_b, settings, epoch, *, noise_mode='random'):
    arrays = [_rgb(item['target']), _rgb(item['input']), _rgb(output_a[0]), _rgb(output_b[0])]
    arrays.append(np.abs(arrays[2]-arrays[3])*16.)
    size = int(settings['thumbnail_size'])
    restored_labels = (['Zero restored A', 'Zero restored B'] if noise_mode == 'zero'
                       else ['Noise A restored', 'Noise B restored'])
    roi_labels = ['Zero A ROI', 'Zero B ROI'] if noise_mode == 'zero' else ['Noise A ROI', 'Noise B ROI']
    full = _panel(arrays, ['Clean reference', 'Degraded input', *restored_labels, '|A-B| x16'], size)
    y, x, h, w = item['receipt']['weak_reference_gradient_roi']
    roi = _panel([value[y:y+h, x:x+w] for value in arrays],
                 ['Weak ref. gradient', 'Same input ROI', *roi_labels, 'ROI |A-B| x16'], size)
    title = np.full((32, full.shape[1], 3), 245, dtype=np.uint8)
    label = f"e{epoch} | {item['receipt']['source']} | {item['receipt']['condition']} | fixed scale"
    cv2.putText(title, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1, cv2.LINE_AA)
    write_image(path, np.concatenate((title, full, roi), axis=0))


@torch.no_grad()
def run_monitor(model, context, output_dir, epoch, updates, device, *, noise_mode='random'):
    """实际T16首步、FP32和固定噪声；异常传播，退出时恢复模型原train/eval状态。"""
    if noise_mode not in ('random', 'zero'):
        raise ValueError('monitor noise_mode must be random or zero')
    device = torch.device(device)
    if int(model.num_steps) != 16 or not bool(torch.isclose(model.eta[-1].float(), torch.tensor(1., device=model.eta.device))):
        raise ValueError('consistency monitor requires the deployed T16 eta_T=1 schedule')
    previous_training = model.training
    destination = Path(output_dir)/'monitor'/f'e{int(epoch):02d}'
    destination.mkdir(parents=True, exist_ok=True)
    settings, records, tests = context['settings'], [], []
    try:
        model.eval()
        for position, item in enumerate(context['pairs']):
            image, target, valid = [item[key][None].to(device).float() for key in ('input', 'target', 'valid')]
            outputs, raw = [], []
            for seed in item['receipt']['noise_seeds']:
                noise = (torch.zeros_like(image) if noise_mode == 'zero'
                         else _noise(image.shape, seed, device))
                prediction = first_raw(model, image, noise)
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError('non-finite restoration monitor output')
                raw.append(prediction)
                outputs.append(finish(image, prediction))
            a, b = outputs
            row = dict(**item['receipt'], input_errors=_errors(image, target, valid),
                       output_a_errors=_errors(a, target, valid), output_b_errors=_errors(b, target, valid),
                       noise_difference=_errors(a, b, valid), raw_noise_difference=_errors(raw[0], raw[1], valid),
                       clear_input_drift=_errors(a, image, valid) if item['receipt']['condition'] == 'identity' else None)
            records.append(row)
            _save_training(destination/f'{position:02d}_{Path(item["receipt"]["source"]).stem}_{item["receipt"]["condition"]}.png',
                           item, a, b, settings, epoch, noise_mode=noise_mode)
        for position, item in enumerate(context['tests']):
            image, valid = [item[key][None].to(device).float() for key in ('input', 'valid')]
            outputs = []
            for seed in item['receipt']['noise_seeds']:
                noise = (torch.zeros_like(image) if noise_mode == 'zero'
                         else _noise(image.shape, seed, device))
                raw = first_raw(model, image, noise)
                if not bool(torch.isfinite(raw).all()):
                    raise FloatingPointError('non-finite unlabeled test monitor output')
                outputs.append(finish(image, raw))
            a, b = outputs
            arrays = [_rgb(image[0]), _rgb(a[0]), _rgb(b[0]), _rgb((a-b).abs()[0])*16.]
            restored_labels = (['Zero restored A', 'Zero restored B'] if noise_mode == 'zero'
                               else ['Noise A restored', 'Noise B restored'])
            write_image(destination/f'test_{position:02d}_{Path(item["receipt"]["source"]).stem}.png',
                        _panel(arrays, ['Original no GT', *restored_labels, '|A-B| x16'],
                               int(settings['thumbnail_size'])))
            tests.append(dict(**item['receipt'], noise_difference=_errors(a, b, valid),
                              input_drift=_errors(a, image, valid)))
        groups = {}
        for condition in CONDITIONS:
            selected = [row for row in records if row['condition'] == condition]
            groups[condition] = {name: {key: float(np.mean([row[name][key] for row in selected]))
                                       for key in ('rgb_l1', 'gradient_l1', 'mse')}
                                 for name in ('input_errors', 'output_a_errors', 'output_b_errors', 'noise_difference')}
        report = dict(epoch=int(epoch), updates=int(updates), actual_step=16, fp32=True,
                      noise_mode=('zero' if noise_mode == 'zero' else
                                  NOISE_MODE if device.type == 'cuda' else 'dedicated_cpu_generator'),
                      has_test_ground_truth=False, purpose=context['manifest']['purpose'],
                      training=records, groups=groups, tests=tests)
        write_json(destination/'metrics.json', report)
        return report
    finally:
        model.train(previous_training)
