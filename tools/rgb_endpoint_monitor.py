# -*- coding: utf-8 -*-
"""D5a固定端点训练配对及真实图过程缩略图；不作验证或模型选择。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from data.rgb_restoration_dataset import prepare_rgb, read_rgb
from data.rgb_spatial_blur import make_endpoint_sigma_map, apply_sigma_map_blur
from tools.rgb_spatial_monitor import clean_prediction_trajectory
from train_rgb_restoration import reconstruction_errors, write_json
from utils.config import project_path


def build_endpoint_fixed_samples(config, train_set, fixed_manifest):
    cfg = config['rgb_restoration']
    settings = cfg['monitor']['endpoint_blur']
    count = int(settings.get('samples', 4))
    if not 1 <= count <= len(fixed_manifest['sources']):
        raise ValueError('endpoint monitor requires original fixed training sources')
    bounds = cfg['degradation']['blur_sigma']
    collected = {key: [] for key in ('input', 'target', 'valid', 'sigma')}
    sources = []
    for index in fixed_manifest['source_indices'][:count]:
        path = train_set.samples[index]
        boxed, valid = prepare_rgb(read_rgb(path), cfg['image_size'])
        vh, vw = int(valid[:, 0].sum()), int(valid[0].sum())
        size = int(cfg['crop_size'])
        top, left = max(0, (vh-size)//2), max(0, (vw-size)//2)
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], index, 1707]))
        field = make_endpoint_sigma_map((vh, vw), (top, left, min(size, vh), min(size, vw)),
                                        settings, rng, bounds)
        degraded = apply_sigma_map_blur(boxed[:vh, :vw], field, bounds,
                                       levels=cfg['degradation']['spatial_blur'].get('levels', 8))
        padding = (0, cfg['image_size']-vh, 0, cfg['image_size']-vw)
        degraded = cv2.copyMakeBorder(degraded, *padding, cv2.BORDER_REFLECT)
        field = cv2.copyMakeBorder(field, *padding, cv2.BORDER_REFLECT)
        crop = np.s_[top:top+size, left:left+size]
        collected['input'].append(torch.from_numpy(degraded[crop].transpose(2, 0, 1).copy()))
        collected['target'].append(torch.from_numpy(boxed[crop].transpose(2, 0, 1).copy()))
        collected['valid'].append(torch.from_numpy(valid[crop][None].copy()))
        collected['sigma'].append(torch.from_numpy(field[crop][None].copy()))
        sources.append(Path(path).name)
    if sources != fixed_manifest['sources'][:count]:
        raise ValueError('endpoint monitor sources differ from original training manifest')
    fixed = {key: torch.stack(value) for key, value in collected.items()}
    fixed.update(source=sources, sigma_bounds=list(bounds))
    manifest = dict(purpose='seen_training_diagnostic_not_validation_or_selection', sources=sources,
                    source_indices=fixed_manifest['source_indices'][:count], seed=cfg['seed'],
                    endpoint_blur=settings, sigma_bounds=list(bounds),
                    image_size=cfg['image_size'], crop_size=cfg['crop_size'],
                    geometry='valid_full_image_endpoint_blur_then_center_crop',
                    additional_resampling=False, additional_local_damage=False,
                    region_definition='nominal_sigma_le_1.5_and_ge_4_not_optical_ground_truth')
    return fixed, manifest


def build_real_fixed_samples(config):
    cfg = config['rgb_restoration']
    settings = cfg['monitor']['real']
    directory = Path(project_path(config, settings['data_dir']))
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    available = [p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    inputs, masks, sources = [], [], []
    for name in settings['images']:
        matches = [p for p in available if p.stem == name or p.name == name]
        if len(matches) != 1:
            raise ValueError(f'real monitor requires one image for {name}, found {len(matches)}')
        boxed, valid = prepare_rgb(read_rgb(matches[0]), cfg['image_size'])
        inputs.append(torch.from_numpy(boxed.transpose(2, 0, 1).copy()))
        masks.append(torch.from_numpy(valid[None].copy()))
        sources.append(matches[0].name)
    fixed = dict(input=torch.stack(inputs), valid=torch.stack(masks), source=sources)
    manifest = dict(purpose='unlabeled_visual_monitor_only_no_training_or_selection',
                    sources=sources, data_dir=settings['data_dir'], image_size=cfg['image_size'],
                    regions='valid_content_horizontal_quarters_at_vertical_center', has_ground_truth=False)
    return fixed, manifest


def _save_rows(path, tensors, labels, title, boxes, size):
    """统一[0,1]色阶；ROI只依赖预定几何位置，不依赖输出。"""
    arrays = [np.round(t.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()*255)
              .astype(np.uint8) for t in tensors]
    rows = []
    for region, box in boxes:
        panels = []
        for array, label in zip(arrays, labels):
            if box is not None:
                y, x, h, w = box
                array = array[y:y+h, x:x+w]
            panel = np.full((size+24, size, 3), 245, np.uint8)
            # 缩略图保持宽高比，空白补齐不拉伸。
            ah, aw = array.shape[:2]
            ratio = size/max(ah, aw)
            thumb = cv2.resize(array, (max(1, round(aw*ratio)), max(1, round(ah*ratio))),
                               interpolation=cv2.INTER_AREA)
            th, tw = thumb.shape[:2]
            oy, ox = (size-th)//2+24, (size-tw)//2
            panel[oy:oy+th, ox:ox+tw] = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
            cv2.putText(panel, f'{region} {label}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        .36, (20, 20, 20), 1, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    header = np.full((32, size*len(tensors), 3), 245, np.uint8)
    cv2.putText(header, title, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.png', np.concatenate([header, *rows], axis=0))
    if not ok:
        raise OSError(f'cannot encode {path}')
    encoded.tofile(str(path))


def _quarter_boxes(valid, roi_size):
    mask = valid.squeeze().bool()
    ys, xs = torch.where(mask)
    h, w = int(ys.max())+1, int(xs.max())+1
    roi = min(int(roi_size), h, w)
    y = max(0, min(h-roi, h//2-roi//2))
    return [('Full', (0, 0, h, w)),
            ('Left', (y, max(0, min(w-roi, w//4-roi//2)), roi, roi)),
            ('Right', (y, max(0, min(w-roi, 3*w//4-roi//2)), roi, roi))]


@torch.no_grad()
def monitor_endpoint(model, fixed, output_dir, *, device, epoch, updates, monitor_cfg, use_amp, amp_dtype):
    previous_training = model.training
    model.eval()
    destination = Path(output_dir)/'endpoint_monitor'/f'epoch_{epoch:03d}_update_{updates:06d}'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    wanted = {1, 2, 4, 8, model.num_steps}
    try:
        for index, source in enumerate(fixed['source']):
            inputs, target, valid, sigma = [fixed[k][index:index+1].to(device)
                                            for k in ('input', 'target', 'valid', 'sigma')]
            regions = {'all': valid, 'weak': valid*(sigma<=1.5), 'strong': valid*(sigma>=4.0)}
            if any(float(mask.sum()) < 1 for mask in regions.values()):
                raise ValueError('fixed endpoint pair lacks weak or strong content')
            baseline = {name: {k: float(v.mean()) for k, v in reconstruction_errors(inputs, target, mask).items()}
                        for name, mask in regions.items()}
            predictions = {}
            generator = torch.Generator(device=device).manual_seed(int(monitor_cfg.get('sampling_seed', 314159))+index)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                for calls, prediction in clean_prediction_trajectory(model, inputs, generator):
                    if not bool(torch.isfinite(prediction).all()):
                        raise FloatingPointError(f'non-finite endpoint monitor: {source}')
                    row = dict(source=source, calls=calls, regions={})
                    for name, mask in regions.items():
                        errors = {k: float(v.mean()) for k, v in reconstruction_errors(prediction, target, mask).items()}
                        row['regions'][name] = dict(input=baseline[name], output=errors, pixels=int(mask.sum()))
                    records.append(row)
                    if calls in wanted:
                        predictions[calls] = prediction[0].float().cpu()
            low, high = fixed['sigma_bounds']
            colors = cv2.applyColorMap(np.round(np.clip((fixed['sigma'][index, 0].numpy()-low)/(high-low), 0, 1)*255)
                                       .astype(np.uint8), cv2.COLORMAP_TURBO)
            field = torch.from_numpy(cv2.cvtColor(colors, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float()/255
            steps = sorted(predictions)
            tensors = [fixed['target'][index], fixed['input'][index], field, *[predictions[k] for k in steps]]
            labels = ['Clean', 'Blur', 'Sigma', *[f'x0 #{k}' for k in steps]]
            _save_rows(destination/f'{Path(source).stem}.png', tensors, labels,
                       f'Epoch {epoch} | update {updates} | endpoint seen-training probe | fixed scale',
                       _quarter_boxes(fixed['valid'][index], monitor_cfg.get('roi_size', 256)),
                       int(monitor_cfg.get('thumbnail_size', 224)))
        report = dict(epoch=epoch, updates=updates, records=records,
                      precision=str(amp_dtype) if use_amp else 'FP32',
                      scope='seen_training_diagnostic_not_validation_or_selection',
                      region_definition='nominal_sigma_le_1.5_and_ge_4_not_optical_ground_truth')
        write_json(destination/'errors.json', report)
        return report
    finally:
        model.train(previous_training)


@torch.no_grad()
def monitor_real(model, fixed, output_dir, *, device, epoch, updates, monitor_cfg, use_amp, amp_dtype):
    previous_training = model.training
    model.eval()
    destination = Path(output_dir)/'real_monitor'/f'epoch_{epoch:03d}_update_{updates:06d}'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    wanted = {1, 2, 4, 8, model.num_steps}
    try:
        for index, source in enumerate(fixed['source']):
            inputs = fixed['input'][index:index+1].to(device)
            generator = torch.Generator(device=device).manual_seed(int(monitor_cfg.get('sampling_seed', 314159))+index)
            predictions = {}
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                for calls, prediction in clean_prediction_trajectory(model, inputs, generator):
                    if not bool(torch.isfinite(prediction).all()):
                        raise FloatingPointError(f'non-finite real monitor: {source}')
                    if calls in wanted:
                        predictions[calls] = prediction[0].float().cpu()
            steps = sorted(predictions)
            _save_rows(destination/f'{Path(source).stem}.png',
                       [fixed['input'][index], *[predictions[k] for k in steps]],
                       ['Input', *[f'x0 #{k}' for k in steps]],
                       f'Epoch {epoch} | update {updates} | {source} | unlabeled view only | fixed scale',
                       _quarter_boxes(fixed['valid'][index], monitor_cfg.get('roi_size', 256)),
                       int(monitor_cfg.get('thumbnail_size', 224)))
            records.append(dict(source=source, displayed_calls=steps))
        report = dict(epoch=epoch, updates=updates, records=records, has_ground_truth=False,
                      precision=str(amp_dtype) if use_amp else 'FP32',
                      scope='unlabeled_visual_monitor_only_no_training_or_selection')
        write_json(destination/'manifest.json', report)
        return report
    finally:
        model.train(previous_training)
