# -*- coding: utf-8 -*-
"""零噪声适配的固定过程图；同时保留同权重、原带噪协议的独立观察。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools import restore_consistency_monitor as base
from train_rgb_restoration import write_json
from utils.image_io import read_image, write_image


def prepare_monitor(config, dataset, output_root):
    """直接复用已有固定配对，避免把新一轮采样差异算给零噪声。"""
    context = base.prepare_monitor(config, dataset, output_root)
    write_json(Path(output_root)/'zero_monitor_protocol.json', dict(
        fixed_manifest='monitor_manifest.json',
        zero_view='T16 first x0 with exactly zero q_sample noise; seeds inactive',
        noisy_view='unchanged two fixed independent noises from original monitor',
        has_holdout=False, has_test_ground_truth=False))
    return context


def _column(image, column, size, label, *, title_rows=0):
    """复用相同缩略图像素；只改列名，不再缩放或选择窗口。"""
    value = image[:, column*size:(column+1)*size].copy()
    for start in (title_rows, title_rows+size+26):
        if start >= len(value):
            break
        value[start:start+26] = 245
        cv2.putText(value, label if start == title_rows else label+' ROI',
                    (4, start+17), cv2.FONT_HERSHEY_SIMPLEX, .36, (20, 20, 20), 1, cv2.LINE_AA)
    return value


def _comparison_gallery(context, output_dir, epoch):
    destination = Path(output_dir)/'monitor'/f'e{int(epoch):02d}'
    zero_root = Path(output_dir)/'zero_view'/'monitor'/f'e{int(epoch):02d}'
    noisy_root = Path(output_dir)/'noisy_view'/'monitor'/f'e{int(epoch):02d}'
    size = int(context['settings']['thumbnail_size'])
    for position, item in enumerate(context['pairs']):
        name = f'{position:02d}_{Path(item["receipt"]["source"]).stem}_{item["receipt"]["condition"]}.png'
        zero, noisy = read_image(zero_root/name), read_image(noisy_root/name)
        # 32像素标题独立重写，避免截取原跨列标题后留下半句话。
        columns = [_column(zero, 0, size, 'Clean reference', title_rows=32),
                   _column(zero, 1, size, 'Degraded input', title_rows=32),
                   _column(zero, 2, size, 'Zero restored', title_rows=32),
                   _column(noisy, 2, size, 'Noisy restored', title_rows=32)]
        picture = np.concatenate(columns, axis=1)
        picture[:32] = 245
        label = f'e{int(epoch)} | {item["receipt"]["source"]} | {item["receipt"]["condition"]} | same weight'
        cv2.putText(picture, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1, cv2.LINE_AA)
        write_image(destination/name, picture)
    for position, item in enumerate(context['tests']):
        name = f'test_{position:02d}_{Path(item["receipt"]["source"]).stem}.png'
        zero, noisy = read_image(zero_root/name), read_image(noisy_root/name)
        picture = np.concatenate([_column(zero, 0, size, 'Original no GT'),
                                  _column(zero, 1, size, 'Zero restored'),
                                  _column(noisy, 1, size, 'Noisy restored')], axis=1)
        write_image(destination/name, picture)


def run_monitor(model, context, output_dir, epoch, updates, device):
    """同权重两种输入，零噪声仍重复两次forward；原噪声结果不参与训练。"""
    output_dir = Path(output_dir)
    zero = base.run_monitor(model, context, output_dir/'zero_view', epoch, updates, device,
                            noise_mode='zero')
    noisy = base.run_monitor(model, context, output_dir/'noisy_view', epoch, updates, device)
    destination = output_dir/'monitor'/f'e{int(epoch):02d}'
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(epoch=int(epoch), updates=int(updates), actual_step=16, fp32=True,
                  noise_mode='zero_and_original_fixed_noisy_views', has_test_ground_truth=False,
                  purpose=context['manifest']['purpose'], groups=zero['groups'],
                  zero_view=zero, noisy_view=noisy,
                  comparison_columns=dict(training=['clear', 'input', 'zero', 'noisy'],
                                          test=['input_no_GT', 'zero', 'noisy']))
    _comparison_gallery(context, output_dir, epoch)
    write_json(destination/'metrics.json', report)
    return report
