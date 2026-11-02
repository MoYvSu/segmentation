# -*- coding: utf-8 -*-
"""训练内固定空间模糊观察：过程图、逐步误差和清晰输入漂移，不选权重。"""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import torch

from data.rgb_restoration_dataset import prepare_rgb, read_rgb
from data.rgb_spatial_blur import apply_spatial_blur
from train_rgb_restoration import reconstruction_errors, write_json


def build_spatial_fixed_samples(config, train_set, fixed_manifest):
    cfg = config['rgb_restoration']
    settings = cfg['monitor']['spatial_blur']
    count = int(settings.get('samples', len(fixed_manifest['sources'])))
    if not 1 <= count <= len(fixed_manifest['sources']):
        raise ValueError('spatial monitor samples must be drawn from the original fixed sources')
    spatial_cfg = copy.deepcopy(cfg['degradation']['spatial_blur'])
    low, high = cfg['degradation']['blur_sigma']
    base_sigma = float(settings.get('base_sigma', 3.0))
    if not low < base_sigma < high:
        raise ValueError('spatial monitor base_sigma must be inside training blur bounds')
    sources, inputs, targets, masks, sigmas = [], [], [], [], []
    for index in fixed_manifest['source_indices'][:count]:
        path = train_set.samples[index]
        boxed, valid = prepare_rgb(read_rgb(path), cfg['image_size'])
        vh, vw = int(valid[:, 0].sum()), int(valid[0].sum())
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], index, 607]))
        degraded, sigma = apply_spatial_blur(boxed[:vh, :vw], base_sigma, spatial_cfg, rng, (low, high))
        padding = (0, cfg['image_size'] - vh, 0, cfg['image_size'] - vw)
        degraded = cv2.copyMakeBorder(degraded, *padding, cv2.BORDER_REFLECT)
        sigma = cv2.copyMakeBorder(sigma, *padding, cv2.BORDER_REFLECT)
        size = cfg['crop_size']
        top, left = max(0, (vh-size)//2), max(0, (vw-size)//2)
        region = np.s_[top:top+size, left:left+size]
        targets.append(torch.from_numpy(boxed[region].transpose(2, 0, 1).copy()))
        inputs.append(torch.from_numpy(degraded[region].transpose(2, 0, 1).copy()))
        masks.append(torch.from_numpy(valid[region][None].copy()))
        sigmas.append(torch.from_numpy(sigma[region][None].copy()))
        sources.append(Path(path).name)
    if sources != fixed_manifest['sources'][:count]:
        raise ValueError('spatial monitor sources differ from the parent fixed manifest')
    fixed = dict(input=torch.stack(inputs), target=torch.stack(targets), valid=torch.stack(masks),
                 sigma=torch.stack(sigmas), source=sources, sigma_bounds=[float(low), float(high)])
    manifest = dict(purpose='training_only_not_validation_or_selection', sources=sources,
                    source_indices=fixed_manifest['source_indices'][:count], seed=cfg['seed'],
                    spatial_blur=spatial_cfg, base_sigma=base_sigma, sigma_bounds=[low, high],
                    geometry='letterbox_valid_content_blur_then_center_crop',
                    additional_resampling=False, additional_local_damage=False,
                    region_definition='lower_and_upper_sigma_tertiles_within_valid_crop',
                    image_size=cfg['image_size'], crop_size=cfg['crop_size'])
    return fixed, manifest


def clean_prediction_trajectory(model, inputs, generator):
    """原采样链每次的清晰图估计；末项严格复现forward，未改变时间表。"""
    condition = inputs.float()
    state = condition + model.kappa * torch.randn(inputs.shape, device=inputs.device,
                                                   dtype=torch.float32, generator=generator)
    for calls, step in enumerate(range(model.num_steps, 0, -1), 1):
        t_index = torch.full((len(inputs),), step, device=inputs.device, dtype=torch.long)
        prediction = model.denoise(state, condition, t_index)
        state, variance = model.posterior_mean_variance(state, prediction, t_index)
        if step > 1:
            state = state + variance.sqrt() * torch.randn(inputs.shape, device=inputs.device,
                                                         dtype=torch.float32, generator=generator)
        output = ((condition + (state-condition)).clamp(0, 1) if step == 1
                  else prediction.clamp(0, 1))
        yield calls, output


def _save_panels(path, tensors, labels, title, size, roi_size):
    rows = []
    h, w = tensors[0].shape[-2:]
    roi = min(roi_size, h, w)
    top, left = (h-roi)//2, (w-roi)//2
    for crop in (False, True):
        panels = []
        for tensor, label in zip(tensors, labels):
            if crop:
                tensor = tensor[:, top:top+roi, left:left+roi]
            array = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
            array = np.round(np.clip(array, 0, 1)*255).astype(np.uint8)
            panel = np.full((size+24, size, 3), 245, np.uint8)
            panel[24:] = cv2.cvtColor(cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            cv2.putText(panel, label + (' ROI' if crop else ''), (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        .38, (20, 20, 20), 1, cv2.LINE_AA)
            panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    header = np.full((34, size*len(tensors), 3), 245, np.uint8)
    cv2.putText(header, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.png', np.concatenate([header]+rows, axis=0))
    if not ok:
        raise OSError(f'cannot encode {path}')
    encoded.tofile(str(path))


def _save_curves(path, rows, epoch):
    # 已有依赖；只画训练内诊断，不参与损失或checkpoint选择。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, region in zip(axes, ('all', 'weak', 'strong')):
        for key, label in [('rgb_l1', 'RGB L1'), ('gradient_l1', 'Gradient L1')]:
            steps = sorted({r['calls'] for r in rows})
            ratios = []
            for step in steps:
                matched = [r for r in rows if r['calls'] == step and region in r['regions']]
                baseline = np.mean([r['regions'][region]['input'][key] for r in matched])
                output = np.mean([r['regions'][region]['output'][key] for r in matched])
                ratios.append(output/max(baseline, 1e-12))
            ax.plot(steps, ratios, marker='.', label=label)
        ax.axhline(1, ls='--', color='gray', lw=1)
        ax.set(title=region + ' region', xlabel='Calls in original chain', ylabel='Error / input error')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'Epoch {epoch}: fixed seen training pairs; relative sigma thirds; no validation')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def monitor_spatial(model, fixed, output_dir, *, device, epoch, updates, monitor_cfg, use_amp, amp_dtype):
    previous_training = model.training
    model.eval()
    destination = Path(output_dir)/'spatial_monitor'/f'epoch_{epoch:03d}_update_{updates:06d}'
    destination.mkdir(parents=True, exist_ok=True)
    rows, identity = [], []
    seed = int(monitor_cfg.get('sampling_seed', 314159))
    size, roi = int(monitor_cfg.get('thumbnail_size', 224)), int(monitor_cfg.get('roi_size', 256))
    try:
        for index, source in enumerate(fixed['source']):
            inputs, target, valid, sigma = [fixed[k][index:index+1].to(device) for k in ('input','target','valid','sigma')]
            cuts = torch.quantile(sigma[valid.bool()], torch.tensor([1/3,2/3], device=device))
            regions = {'all':valid, 'weak':valid*(sigma<=cuts[0]), 'strong':valid*(sigma>=cuts[1])}
            baseline = {name:{k:float(v.mean()) for k,v in reconstruction_errors(inputs,target,mask).items()}
                        for name,mask in regions.items()}
            first, last = None, None
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                for calls, prediction in clean_prediction_trajectory(model, inputs, torch.Generator(device=device).manual_seed(seed+index)):
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError(f'non-finite spatial monitor: {source} call {calls}')
                    row = dict(source=source, calls=calls, t=model.num_steps-calls+1, regions={})
                    for name, mask in regions.items():
                        errors = {k:float(v.mean()) for k,v in reconstruction_errors(prediction,target,mask).items()}
                        row['regions'][name] = dict(input=baseline[name], output=errors, pixels=int(mask.sum()))
                    rows.append(row)
                    if calls == 1:
                        first = prediction[0].cpu()
                    last = prediction[0].cpu()
                clean_first, clean_last = None, None
                for calls, prediction in clean_prediction_trajectory(model, target, torch.Generator(device=device).manual_seed(seed+index)):
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError(f'non-finite identity monitor: {source}')
                    if calls in (1, model.num_steps):
                        identity.append(dict(source=source, calls=calls, errors={k:float(v.mean()) for k,v in
                                        reconstruction_errors(prediction,target,valid).items()}))
                    if calls == 1:
                        clean_first = prediction[0].cpu()
                    clean_last = prediction[0].cpu()
            low, high = fixed['sigma_bounds']
            normalized = (fixed['sigma'][index,0].numpy()-low)/(high-low)
            colors = cv2.applyColorMap(np.round(np.clip(normalized,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
            field = torch.from_numpy(cv2.cvtColor(colors,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()/255
            _save_panels(destination/f'{index:02d}_{Path(source).stem}.png',
                         [fixed['target'][index],fixed['input'][index],field,first,last],
                         ['Clean target','Spatial blur',f'Base sigma {low:g}-{high:g}','First prediction','Full sampling'],
                         f'Epoch {epoch} | update {updates} | spatial training probe | fixed scale',size,roi)
            _save_panels(destination/f'{index:02d}_{Path(source).stem}_identity.png',
                         [fixed['target'][index],clean_first,clean_last],['Clean input','First prediction','Full sampling'],
                         f'Epoch {epoch} | update {updates} | clean input drift',size,roi)
        report = dict(epoch=epoch, updates=updates, scope='seen_training_diagnostic_not_validation_or_selection',
                      region_definition='relative_lower_upper_sigma_tertiles',
                      precision=str(amp_dtype) if use_amp else 'FP32', records=rows, clean_input_drift=identity)
        write_json(destination/'errors.json', report)
        _save_curves(destination/'trajectory_curves.png', rows, epoch)
        return report
    finally:
        model.train(previous_training)
