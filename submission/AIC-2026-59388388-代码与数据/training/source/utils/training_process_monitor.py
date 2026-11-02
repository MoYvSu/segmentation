# -*- coding: utf-8 -*-
"""固定训练源过程缩略图；不计算验证指标，不改变后续训练随机流或模式。"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import re
import sys

import cv2
import numpy as np
import torch


@contextmanager
def preserve_rng():
    """不为CPU观察初始化CUDA；已有CUDA上下文才保存其随机状态。"""
    python, numpy = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def observation_state(model):
    """逐子模块恢复混合train/eval；异常同样恢复，BN等buffer不得漂移。"""
    modules = list(model.modules())
    modes = [(module, module.training) for module in modules]
    lora_modes = [(module, module.trainable_lora) for module in modules if hasattr(module, 'trainable_lora')]
    buffers = [(name, value, value.detach().cpu().clone()) for name, value in model.named_buffers()]
    with preserve_rng():
        try:
            model.eval()
            for module, _ in lora_modes:
                module.trainable_lora = False
            with torch.no_grad():
                yield
        finally:
            failed = sys.exc_info()[0] is not None
            changed = []
            try:
                with torch.no_grad():
                    for name, value, original in buffers:
                        if not torch.equal(value.detach().cpu(), original):
                            changed.append(name)
                            value.copy_(original.to(device=value.device, dtype=value.dtype))
            finally:
                # 不调用train()递归覆盖已经恢复的子模块模式。
                for module, training in modes:
                    module.training = training
                for module, flag in lora_modes:
                    module.trainable_lora = flag
            if changed and not failed:
                raise RuntimeError('Process monitor unexpectedly changed model buffers: ' + ', '.join(changed))


def monitor_due(epoch, total_epochs, every_epochs=10):
    if type(epoch) is not int or type(total_epochs) is not int or type(every_epochs) is not int:
        raise ValueError('Monitor epoch values must be integers')
    if not 0 <= epoch <= total_epochs or total_epochs < 1 or every_epochs < 1:
        raise ValueError('Invalid process monitor epoch schedule')
    return epoch in (0, 1, total_epochs) or epoch % every_epochs == 0


def _probability(value):
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError('Monitor probabilities must be finite and in [0,1]')
    intensity = np.rint(value * 255).astype(np.uint8)
    return cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)


def _target(value, *, semantic=False):
    valid = np.isfinite(value) & (value >= 0)
    result = np.full((*value.shape, 3), 128, np.uint8)
    if semantic:
        result[valid & (value <= .5)] = (220, 120, 50)  # 珠光体蓝色，OpenCV BGR。
        result[valid & (value > .5)] = (40, 180, 245)  # 铁素体金色。
    else:
        result[valid] = np.rint(np.clip(value[valid], 0, 1)[:, None] * 255).astype(np.uint8)
    return result


def _panel(image, title, size, *, nearest=False):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    body = np.full((size, size, 3), 255, np.uint8)
    body[:resized.shape[0], :resized.shape[1]] = resized
    body = cv2.copyMakeBorder(body, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(body, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 0, 0), 1, cv2.LINE_AA)
    return body


class TrainingProcessMonitor:
    """复制训练Dataset，固定其前两个源；完整letterbox前向，只保存小图。"""
    def __init__(self, dataset, output_dir, options):
        self.every_epochs = options.get('every_epochs', 10)
        self.thumbnail_size = options.get('thumbnail_size', 224)
        if (type(self.every_epochs) is not int or self.every_epochs < 1
                or type(self.thumbnail_size) is not int or not 64 <= self.thumbnail_size <= 512):
            raise ValueError('Process monitor requires positive interval and thumbnail size 64..512')
        if len(dataset) < 1:
            raise ValueError('Process monitor needs at least one training source')
        with preserve_rng():
            self.dataset = deepcopy(dataset)
        self.dataset.augment = False
        self.dataset.crop_size = 0
        self.indices = tuple(range(min(2, len(self.dataset))))
        self.output_dir = Path(output_dir) / 'process_monitor'

    def maybe_save(self, model, epoch, total_epochs, device):
        if not monitor_due(epoch, total_epochs, self.every_epochs):
            return None
        destination = self.output_dir / f'epoch_{epoch:03d}'
        if destination.exists():
            summary = destination / 'summary.json'
            if not summary.is_file() or json.loads(summary.read_text(encoding='utf-8')).get('complete') is not True:
                raise FileExistsError('Incomplete process monitor output exists: ' + str(destination))
            return dict(skipped_existing=True, epoch=epoch, path=str(destination))
        with observation_state(model):
            destination.mkdir(parents=True)
            rows = []
            for index in self.indices:
                sample = self.dataset[index]
                image, target = sample['image'], sample['target']
                if (image.ndim != 3 or image.shape[0] != 3 or target.ndim != 3 or target.shape[0] < 2
                        or image.shape[-2:] != target.shape[-2:]):
                    raise ValueError('Monitor requires aligned CHW RGB and semantic/boundary targets')
                inputs = image[None].to(device=device, dtype=torch.float32)
                logits = model(inputs, output_size=tuple(target.shape[-2:]))
                if (not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[0] != 1
                        or logits.shape[1] < 2 or logits.shape[-2:] != target.shape[-2:]):
                    raise ValueError('Monitor model must return aligned semantic/boundary logits')
                probabilities = logits[0, :2].float().sigmoid().cpu().numpy()
                rgb = image.detach().float().cpu().permute(1, 2, 0).numpy()
                reference = target[:2].detach().float().cpu().numpy()
                if not np.isfinite(rgb).all() or not np.isfinite(reference).all():
                    raise ValueError('Nonfinite process monitor input/target')
                rgb = cv2.cvtColor(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                teacher = bool(getattr(self.dataset, 'teacher_gt_dir', None))
                label = 'Teacher' if teacher else 'Training GT'
                panels = [_panel(rgb, 'Fixed training input', self.thumbnail_size),
                    _panel(_probability(probabilities[0]), 'Pred P(F): 0..1', self.thumbnail_size),
                    _panel(_probability(probabilities[1]), 'Pred boundary: 0..1', self.thumbnail_size),
                    _panel(_target(reference[0], semantic=True), label + ' F/P; ignore grey', self.thumbnail_size, nearest=True),
                    _panel(_target(reference[1]), label + ' boundary; ignore grey', self.thumbnail_size, nearest=True)]
                source = str(sample.get('image_path', 'training_' + str(index)))
                name = re.sub(r'[^A-Za-z0-9_.-]', '_', Path(source).stem)
                path = destination / f'{index:02d}_{name}.png'
                success, encoded = cv2.imencode('.png', np.concatenate(panels, axis=1))
                if not success:
                    raise OSError('Could not encode process thumbnail: ' + str(path))
                path.write_bytes(encoded.tobytes())
                rows.append(dict(dataset_index=index, source=source, file=path.name,
                                 input_shape=list(image.shape), target_source='model_teacher' if teacher else 'training_GT'))
            report = dict(format='training_process_thumbnail_v1', complete=True, epoch=epoch,
                selection='first two existing training-dataset sources; no augmentation or crop',
                thumbnail_size=self.thumbnail_size, samples=rows, full_arrays_saved=False,
                validation_metrics_computed=False, watershed_computed=False, checkpoint_selection=False,
                probability_range=[0, 1], ignore_colour='grey')
        # 只有模式/buffer/RNG隔离检查退出成功，才把本次观察标记为完成。
        (destination / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                           allow_nan=False), encoding='utf-8')
        return report


def build_process_monitor(options, dataset, output_dir):
    """缺省或{}完全关闭；只有enabled=true才读取Dataset或创建任何文件。"""
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError('train.process_monitor must be an object')
    if set(options) - {'enabled', 'every_epochs', 'thumbnail_size'}:
        raise ValueError('Unknown train.process_monitor options')
    if type(options.get('enabled', False)) is not bool:
        raise ValueError('train.process_monitor.enabled must be true/false')
    if not options.get('enabled', False):
        return None
    return TrainingProcessMonitor(dataset, output_dir, options)
