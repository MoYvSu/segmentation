# -*- coding: utf-8 -*-
"""RGB条件NAF的原生分块推理与固定缩略monitor，不重跑冻结后端。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.boundary_naf_monitor import _fixed_controls, original_view, real_pair_metrics
from tools.boundary_restore_monitor import panels, _heat, _overlay, paired_metrics, zoom_box
from tools.boundary_restore_readout import tiles, _weights, _probability, evaluate_frozen_readout
from utils.boundary_input import input_normalization_settings, normalize_boundary_input
from utils.config import project_path


@contextmanager
def _evaluation(model):
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, training in flags:
            module.training = training


def _mode(model, settings):
    mode = getattr(model, 'input_mode', None)
    if mode is None:
        mode = getattr(model, 'architecture', {}).get('input_mode')
    configured = settings.get('input_mode', mode)
    if mode is None:
        mode = configured
    if mode not in ('rgb_boundary', 'boundary') or configured != mode:
        raise ValueError('RGB NAF input_mode must match model: rgb_boundary or boundary')
    return mode


def _rgb_array(rgb, shape, rgb_input_format='uint8'):
    if rgb_input_format not in ('uint8', 'float32_unit'):
        raise ValueError('rgb_input_format必须明确为uint8或float32_unit')
    dtype = np.dtype('uint8') if rgb_input_format == 'uint8' else np.dtype('float32')
    if (not isinstance(rgb, np.ndarray) or rgb.dtype not in (np.dtype('uint8'), np.dtype('float32'))
            or rgb.dtype != dtype or rgb.shape != (*shape, 3)):
        raise ValueError('RGB must be native aligned HWC uint8 or float32 with three channels')
    if rgb.dtype == np.float32 and (not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1))):
        raise ValueError('Restored float32 RGB must be finite and in [0,1]')
    return rgb


def predict_boundary_rgb_tiled(boundary, rgb, model, settings, *, device='cpu', enabled=True):
    """对齐的RGB和二值T一次前向，直接拼接输出概率，不把原B再次加回。

    所有窗口保持原生像素；只作同步reflect填充。关闭模块原样返回输入B，
    开启时T-only也是独立表示候选，不能称为原部署的恒等路径。
    """
    boundary = _probability(boundary, 'boundary')
    if not enabled:
        return boundary, dict(bypass=True, boundary_exact=True, reason='module_disabled')
    rgb = _rgb_array(rgb, boundary.shape, settings.get('rgb_input_format', 'uint8'))
    if settings.get('iterations', 1) != 1 or settings.get('revert_unconnected', False):
        raise ValueError('RGB NAF uses one pass without post-restoration revert')
    mode = _mode(model, settings)
    threshold = settings.get('input_threshold', .5)
    if type(threshold) not in (int, float) or float(threshold) != .5:
        raise ValueError('RGB NAF binary input threshold must be 0.5')
    size = settings.get('tile_size', 256)
    overlap = settings.get('overlap', 64)
    pad = settings.get('reflect_pad', 32)
    batch_size = settings.get('batch_size', 16)
    if type(pad) is not int or pad < 0 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Invalid RGB NAF tile padding or batch size')
    before_sha = hashlib.sha256(boundary.tobytes()).hexdigest()
    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    binary, normalization = normalize_boundary_input(boundary, threshold=threshold,
        settings=settings.get('input_normalization'))
    source = binary.astype(np.float32)
    padded = np.pad(source, pad, mode='reflect') if pad else source
    padded_rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode='reflect') if pad else rgb
    boxes = tiles(padded.shape, size, overlap)
    total = np.zeros(padded.shape, np.float32)
    weights = np.zeros_like(total)
    target = torch.device(device)
    with _evaluation(model), torch.autocast(device_type=target.type, enabled=False):
        for start in range(0, len(boxes), batch_size):
            chunk = boxes[start:start + batch_size]
            inputs = []
            for y, x, y1, x1 in chunk:
                a = padded[y:y1, x:x1]
                image = padded_rgb[y:y1, x:x1]
                h, w = a.shape
                if (h, w) != (size, size):
                    a = np.pad(a, ((0, size-h), (0, size-w)), mode='reflect')
                    image = np.pad(image, ((0, size-h), (0, size-w), (0, 0)), mode='reflect')
                value = a[None]
                if mode == 'rgb_boundary':
                    rgb_value = image.transpose(2, 0, 1).astype(np.float32)
                    if rgb.dtype == np.uint8:
                        rgb_value = rgb_value / 255.
                    value = np.concatenate((rgb_value, value), axis=0)
                inputs.append(value)
            tensor = torch.from_numpy(np.stack(inputs).copy()).to(target)
            prediction = model(tensor)
            if (not torch.is_tensor(prediction) or not prediction.is_floating_point()
                    or tuple(prediction.shape) != (len(chunk), 1, size, size)
                    or not bool(torch.isfinite(prediction).all())
                    or bool(((prediction < 0) | (prediction > 1)).any())):
                raise RuntimeError('RGB NAF output must be finite B1HW probability in [0,1]')
            arrays = prediction.detach().float().cpu().numpy()[:, 0]
            for output, (y, x, y1, x1) in zip(arrays, chunk):
                h, w = y1-y, x1-x
                weight = _weights(h, w)
                total[y:y1, x:x1] += output[:h, :w] * weight
                weights[y:y1, x:x1] += weight
    if not np.all(weights > 0):
        raise RuntimeError('RGB NAF tiling missed native pixels')
    result = total / weights
    if pad:
        result = result[pad:-pad, pad:-pad]
    result = np.clip(result, 0, 1).astype(np.float32)
    if (hashlib.sha256(boundary.tobytes()).hexdigest() != before_sha
            or hashlib.sha256(rgb.tobytes()).hexdigest() != rgb_sha):
        raise RuntimeError('RGB NAF tiling changed frozen inputs')
    receipt = dict(bypass=False, input_mode=mode, input_threshold=.5,
        input_boundary_sha256=before_sha, input_rgb_sha256=rgb_sha,
        output_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        tiles=len(boxes), tile_size=size, overlap=overlap, reflect_pad=pad,
        batch_size=batch_size, iterations=1, revert_unconnected=False,
        stitching='direct_probability_weighted_mean', native_resize=False,
        absolute_change_mean=float(np.abs(result-boundary).mean()))
    if normalization['mode'] != 'none':
        receipt['input_normalization'] = normalization
    if rgb.dtype == np.float32:
        receipt.update(input_rgb_dtype='float32', input_rgb_scale='unit_interval_no_requantization')
    return result, receipt


def _sample_rgb(sample):
    rgb = sample['rgb']
    if torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if rgb.dtype != np.uint8:
        if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError('Training monitor RGB must be uint8 or float [0,1]')
        rgb = np.rint(rgb * 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _erasure_pair_diagnostics(sample, prediction, clean_prediction, target, valid):
    """固定分母比较受损/干净输入的预测，不把人工缺口混入天然缺口。"""
    clean = sample['clean_input_boundary'][0].numpy()
    actual = sample['input_boundary'][0].numpy()
    mask = sample['erasure_mask'][0].numpy() > .5
    artificial = sample['artificial_gap_mask'][0].numpy() > .5
    if not (clean.shape == actual.shape == mask.shape == artificial.shape == target.shape == valid.shape):
        raise ValueError('擦断monitor的输入/mask/GT坐标不一致')
    if not np.array_equal(actual, np.where(mask, 0., clean)):
        raise ValueError('擦断monitor的请求mask不能重建实际损伤输入')
    positive = (target > .5) & valid
    near_clean = cv2.dilate((clean > .5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    near_actual = cv2.dilate((actual > .5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    expected_artificial = positive & near_clean & ~near_actual
    if not np.array_equal(artificial, expected_artificial):
        raise ValueError('人工缺口mask不等于原支持且擦后r2缺失的共同GT点')
    damage_context = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    masks = {
        'artificial_missing': artificial,
        'normalized_natural_missing': sample['gap_mask'][0].numpy() > .5,
        'raw_natural_missing': sample.get('raw_gap_mask', sample['gap_mask'])[0].numpy() > .5,
        'original_supported': positive & near_clean,
        'untouched_supported': positive & near_clean & ~damage_context,
    }
    peaks = {name: cv2.dilate(value, np.ones((5, 5), np.uint8))
             for name, value in [('damaged_output', prediction), ('clean_output', clean_prediction)]}
    rows = {}
    for name, chosen in masks.items():
        chosen = chosen & positive
        count = int(chosen.sum())
        rows[name] = dict(points=count)
        for label, peak in peaks.items():
            rows[name][label + '_probability'] = float(peak[chosen].mean()) if count else None
            rows[name][label + '_above065'] = float((peak[chosen] > .65).mean()) if count else None
    removed = (clean > .5) & mask
    if np.any(removed & ~valid):
        raise ValueError('实际擦除进入unknown/frame/crop rim/padding，不能作为本实验输入')
    return dict(cohorts=rows, requested_mask_pixels=int(mask.sum()),
        deleted_input_pixels=int(removed.sum()),
        damaged_vs_clean_output_valid_mae=float(np.abs(prediction-clean_prediction)[valid].mean()) if valid.any() else None,
        denominator='GT positive and same valid native view; r2 L-infinity peaks; not final connectivity or accuracy',
        source=sample['meta']['erasure'])


def _erasure_picture(clean, request_mask):
    """灰色为请求擦除范围，红色为范围内实际删掉的T像素。"""
    request = np.asarray(request_mask) > .5
    picture = np.repeat((request.astype(np.uint8) * 70)[..., None], 3, axis=2)
    picture[(np.asarray(clean) > .5) & request] = (255, 30, 30)
    return picture


def _shift_monitor_picture(model, folder, index, sample, clean_input, probability, settings, width):
    """仅两块固定样本增加图；颜色尺度固定，不用重新归一化掩盖幅值下降。"""
    from utils.boundary_rgb_training import build_tolerant_targets
    from utils.boundary_shift_consistency import build_shift_masks, shift_integer_reflect
    device = next(model.parameters()).device
    target = sample['target'][None].to(device)
    valid = sample['valid'][None].to(device)
    prepared = build_tolerant_targets(target, valid)
    common_inner = (build_shift_masks(prepared, settings, 1, 0)['inner']
                    & build_shift_masks(prepared, settings, 0, 1)['inner'])
    chosen = (common_inner & prepared['valid'])[0, 0].detach().cpu().numpy()
    shifted_maps = []
    for dy, dx in ((1, 0), (0, 1)):
        predicted = model(shift_integer_reflect(clean_input, dy, dx))
        aligned = shift_integer_reflect(predicted, -dy, -dx)[0, 0].detach().float().cpu().numpy()
        shifted_maps.append(aligned)
    arrays = [
        ('Clean T8', _heat(clean_input[0, 0].detach().cpu().numpy())),
        ('Processed GT', _heat(sample['target'][0].numpy())),
        ('Base output', _heat(probability)),
        ('Shift y+1 / inverse aligned', _heat(shifted_maps[0])),
        ('Shift x+1 / inverse aligned', _heat(shifted_maps[1])),
        ('Max abs difference x4 / fixed scale', _heat(np.maximum(
            np.abs(shifted_maps[0] - probability), np.abs(shifted_maps[1] - probability)) * 4.)),
    ]
    for _, array in arrays:
        array[~chosen] = 96
    panels(folder / f'pair_{int(index):03d}_shift.jpg',
        f'{sample["source"]} | draw {index} | clean T8 / phase diagnostic', arrays, width,
        'Gray: outside central valid area. Error display is abs probability difference x4, fixed scale; not GT accuracy.')


def save_monitor(model, config, output, step, dataset, *, force_full=False, source_limit=None):
    """保存固定已见GT配对；仅预定步骤作四张无标签整图最终读出。"""
    cfg = config['boundary_naf']
    settings, inf = cfg['monitor'], cfg['inference']
    normalization = input_normalization_settings(cfg.get('input_normalization'))
    if normalization != input_normalization_settings(inf.get('input_normalization')):
        raise ValueError('Training and inference input_normalization must match')
    normalized = normalization['mode'] != 'none'
    erased_training = isinstance(cfg.get('erasure'), dict) and cfg['erasure'].get('enabled') is True
    shift_training = isinstance(cfg.get('shift_consistency'), dict) and cfg['shift_consistency'].get('enabled') is True
    shift_settings = None
    if shift_training:
        from utils.boundary_shift_consistency import shift_consistency_settings
        shift_settings = shift_consistency_settings(cfg['shift_consistency'])
        if inf['input_mode'] != 'boundary' or not erased_training:
            raise ValueError('平移monitor仅用于无RGB擦断实验')
    folder = Path(output) / 'monitor' / f's{int(step):06d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'summary.json').exists():
        raise ValueError('过程图已有完整回执，不覆盖既有结果')
    device = next(model.parameters()).device
    width = int(settings.get('thumbnail_width', 280))
    rows = []
    baseline_name = settings.get('baseline_name', 'N1')
    with _evaluation(model):
        pair_indices = list(settings['pair_indices'])
        extra_indices = []
        if erased_training:
            extra_indices = settings.get('erasure_pair_indices', [])
            if (not isinstance(extra_indices, list) or len(extra_indices) > 2
                    or any(type(index) is not int or index < 0 for index in extra_indices)
                    or len(set(extra_indices)) != len(extra_indices)):
                raise ValueError('额外擦断monitor最多两个确定的非负draw，不按模型效果改选')
            pair_indices += [index for index in extra_indices if index not in pair_indices]
        for index in pair_indices:
            sample = dataset[int(index)]
            source = sample['input_boundary'][0].numpy()
            raw_source = sample.get('raw_input_boundary', sample['input_boundary'])[0].numpy()
            target = sample['target'][0].numpy()
            valid = sample['valid'][0].numpy() > 0
            gap = sample.get('raw_gap_mask', sample['gap_mask'])[0].numpy() > 0
            prediction = model(sample['input'][None].to(device))[0, 0].detach().float().cpu().numpy()
            clean_source = source
            clean_prediction = None
            erasure_diagnostic = None
            if erased_training:
                required = {'clean_input_boundary', 'erasure_mask', 'artificial_gap_mask'}
                if not required.issubset(sample) or 'erasure' not in sample['meta']:
                    raise ValueError('已启用擦断但固定monitor缺少实际输入与mask回执')
                clean_source = sample['clean_input_boundary'][0].numpy()
                from utils.boundary_rgb_inputs import clean_model_input
                clean_input = clean_model_input(sample, input_mode=inf['input_mode'])
                clean_prediction = model(clean_input[None].to(device))[0, 0].detach().float().cpu().numpy()
                erasure_diagnostic = _erasure_pair_diagnostics(sample, prediction, clean_prediction, target, valid)
            diagnostic, noise = real_pair_metrics(raw_source, target, prediction, valid,
                                                   original_view(dataset, sample))
            if normalized:
                normalization_diagnostic, _ = real_pair_metrics(clean_source, target, prediction, valid,
                                                                 original_view(dataset, sample))
                diagnostic.update({'normalized_' + key: value for key, value in normalization_diagnostic.items()})
            rgb = _sample_rgb({'rgb': sample.get('image', sample['rgb'])})
            cells = [('RGB / training source', rgb)]
            conditioning_rgb = _sample_rgb(sample) if cfg.get('restored_rgb_manifest') else None
            if conditioning_rgb is not None:
                cells.append(('D5a RGB / model condition', conditioning_rgb))
            cells.extend([('Processed GT', _heat(target)),
                     ('Real N1 binary T', _heat(raw_source))])
            if erased_training:
                cells.extend([('Clean standardized T', _heat(clean_source)),
                    ('Erased model input T', _heat(source)),
                    ('Request gray / deleted T red', _erasure_picture(clean_source, sample['erasure_mask'][0].numpy()))])
            elif normalized:
                cells.append(('Standardized input T', _heat(source)))
            cells.append(('NAF restored', _heat(prediction)))
            if erased_training:
                cells.append(('Same model / clean T', _heat(clean_prediction)))
            for _, array in cells[1:]:
                array[~valid] = 96
            panels(folder / f'pair_{int(index):03d}.jpg',
                   f'{sample["source"]} | step {step} | fixed draw {index}', cells, width,
                   'Seen training GT; gray ignored; native pixels and aligned RGB/T; no independent accuracy claim.')
            artificial_focus = sample['artificial_gap_mask'][0].numpy() > .5 if erased_training else None
            focus = artificial_focus if erased_training and np.any(artificial_focus & valid) else gap
            (x, y, x1, y1), label = zoom_box(target, valid, focus, noise, side=128)
            if erased_training and np.any(artificial_focus & valid):
                label = 'Artificial gap / fixed data-only location'
            local = [('RGB', rgb[y:y1, x:x1]), ('GT', _heat(target[y:y1, x:x1])),
                     ('Raw input T' if normalized else 'Input T', _heat(raw_source[y:y1, x:x1]))]
            if conditioning_rgb is not None:
                local.insert(1, ('D5a RGB', conditioning_rgb[y:y1, x:x1]))
            if erased_training:
                local.extend([('Clean T8', _heat(clean_source[y:y1, x:x1])),
                    ('Erased T8', _heat(source[y:y1, x:x1])),
                    ('Erase mask', _erasure_picture(clean_source, sample['erasure_mask'][0].numpy())[y:y1, x:x1])])
            elif normalized:
                local.append(('Standardized T', _heat(source[y:y1, x:x1])))
            local.append(('Output', _heat(prediction[y:y1, x:x1])))
            if erased_training:
                local.append(('Clean output', _heat(clean_prediction[y:y1, x:x1])))
            panels(folder / f'pair_{int(index):03d}_local.jpg',
                f'{label} | {sample["source"]} | step {step}', local, width)
            rows.append(dict(kind='fixed_training_pair', source=sample['source'], draw=int(index),
                meta=sample['meta'], **diagnostic, **paired_metrics(raw_source, target, prediction, valid, gap, noise)))
            if erased_training:
                rows[-1].update(erasure_training=True, erasure=erasure_diagnostic,
                    monitor_group='common_control' if index in settings['pair_indices'] else 'erasure_extra',
                    clean_prediction_sha256=hashlib.sha256(clean_prediction.tobytes()).hexdigest(),
                    damaged_prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest())
            if conditioning_rgb is not None:
                rows[-1].update(rgb_conditioning='restored_float32_cache',
                    model_RGB_tensor_sha256=hashlib.sha256(sample['rgb'].detach().cpu().contiguous().numpy().tobytes()).hexdigest())
            if shift_training:
                from utils.boundary_rgb_inputs import clean_model_input
                from utils.boundary_shift_consistency import translation_probe
                probe_input = clean_model_input(sample, input_mode='boundary')[None].to(device)
                base_probability = torch.from_numpy(clean_prediction)[None, None].to(device)
                phase = translation_probe(model, probe_input, sample['target'][None].to(device),
                    sample['valid'][None].to(device), settings=shift_settings,
                    base_probability=base_probability)
                rows[-1].update(shift_consistency=dict(input='same_clean_T8_before_erasure', metrics=phase,
                    warning='Low aligned MAE is not accuracy; inspect original support and actual readout as well'))
                if index in settings.get('shift_pair_indices', []):
                    _shift_monitor_picture(model, folder, index, sample, probe_input,
                                           clean_prediction, shift_settings, width)

        full = bool(force_full or step in settings.get('full_readout_steps', []))
        if full:
            restored_rgb_cache = None
            if cfg.get('restored_rgb_manifest'):
                from data.boundary_rgb import load_restored_rgb_manifest
                restored_rgb_cache = load_restored_rgb_manifest(
                    project_path(config, settings['restored_rgb_manifest']))
                if restored_rgb_cache['domain'] != 'monitor_test':
                    raise ValueError('完整monitor仅可读取隔离的monitor_test修复RGB清单')
            controls = _fixed_controls(config, settings)
            if source_limit is not None:
                controls = controls[:source_limit]
            rois = {}
            if settings.get('roi_spec'):
                rois = json.loads(Path(project_path(config, settings['roi_spec'])).read_text(encoding='utf-8')).get('images', {})
            receipt_path = Path(output) / 'fixed_baseline_check.json'
            previous = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
            checks = previous.get('sources', {})
            for row, frozen_config in controls:
                with np.load(row['path'], allow_pickle=False) as arrays:
                    a = {k: arrays[k].copy() for k in arrays.files}
                classes = json.loads(Path(row['control_classes']).read_text(encoding='utf-8'))
                name, boundary, rgb = row['source'], a['boundary'], a['rgb']
                model_rgb, conditioning_receipt = rgb, None
                if restored_rgb_cache is not None:
                    from data.boundary_rgb import restored_rgb_for_source
                    model_rgb, conditioning_receipt = restored_rgb_for_source(restored_rgb_cache, name, rgb)
                if name not in checks:
                    baseline = evaluate_frozen_readout(a['semantic'], a['raw_probability'], boundary,
                        frozen_config, reference=dict(instances=a['instances'], classes=classes))
                    adapted = (boundary >= .5).astype(np.float32)
                    adapter_ids, _, adapter_readout = evaluate_frozen_readout(
                        a['semantic'], a['raw_probability'], adapted, frozen_config)
                    checks[name] = dict(cache_sha256=row['sha256'], reference_final_exact=True,
                        readout=baseline[2], adapter_only_instances=int(adapter_ids.max(initial=0)),
                        adapter_readout=adapter_readout, adapter_monitor_step=int(step))
                    adapter_cells = [('RGB', rgb), ('Binary T', _heat(adapted)), ('Binary outlines', _overlay(rgb, adapter_ids))]
                    if normalized:
                        norm_input, norm_receipt = normalize_boundary_input(boundary, settings=normalization)
                        normalized_ids, _, normalized_readout = evaluate_frozen_readout(
                            a['semantic'], a['raw_probability'], norm_input.astype(np.float32), frozen_config)
                        checks[name].update(input_normalization=dict(normalization),
                            normalized_adapter_only_instances=int(normalized_ids.max(initial=0)),
                            normalized_adapter_readout=normalized_readout,
                            normalized_adapter_input_receipt=norm_receipt)
                        adapter_cells.extend([('Standardized T', _heat(norm_input)),
                                              ('Standardized outlines', _overlay(rgb, normalized_ids))])
                    panels(folder / f'{name}_adapter.jpg', f'{name} | binary input representation only',
                        adapter_cells, width,
                        'Independent input adaptation control; computed once, never reused as candidate seeds.')
                elif checks[name]['cache_sha256'] != row['sha256']:
                    raise RuntimeError('原图控制缓存被替换')
                if normalization != checks[name].get('input_normalization', {'mode': 'none'}):
                    raise RuntimeError('Input representation differs from saved adapter diagnostic')
                restored, inference = predict_boundary_rgb_tiled(boundary, model_rgb, model, inf, device=device)
                # 此入口从修复概率重新生成reflect种子与真实障碍，不注入旧markers。
                ids, candidate_classes, readout = evaluate_frozen_readout(
                    a['semantic'], a['raw_probability'], restored, frozen_config)
                whole_cells = [('RGB / no test GT', rgb), (baseline_name + ' heatmap', _heat(boundary))]
                if conditioning_receipt is not None:
                    whole_cells.insert(1, ('D5a RGB / model condition', _sample_rgb({'rgb': model_rgb})))
                normalized_source = None
                if normalized:
                    normalized_source, _ = normalize_boundary_input(boundary, settings=normalization)
                    whole_cells.append(('Standardized input T', _heat(normalized_source)))
                whole_cells.extend([('NAF heatmap', _heat(restored)),
                    (baseline_name + ' outlines', _overlay(rgb, a['instances'])), ('NAF outlines', _overlay(rgb, ids))])
                panels(folder / f'{name}_whole.jpg', f'{name} | step {step} | {inf["input_mode"]}',
                    whole_cells, width,
                    'Frozen backend/semantic; regenerate seeds and barriers from restored map; no test accuracy claim.')
                roi = rois.get(name, {}).get('roi')
                if roi:
                    x, y, x1, y1 = map(int, roi)
                    cut = np.s_[y:y1, x:x1]
                    local = [('RGB', rgb[cut]), (baseline_name, _heat(boundary[cut]))]
                    if conditioning_receipt is not None:
                        local.insert(1, ('D5a RGB', _sample_rgb({'rgb': model_rgb})[cut]))
                    if normalized:
                        local.append(('Standardized T', _heat(normalized_source[cut])))
                    local.extend([('NAF', _heat(restored[cut])),
                        (baseline_name + ' outlines', _overlay(rgb[cut], a['instances'][cut])),
                        ('NAF outlines', _overlay(rgb[cut], ids[cut]))])
                    panels(folder / f'{name}_local.jpg', f'{name} | fixed ROI | step {step}', local, width)
                rows.append(dict(kind='unlabelled_fixed_monitor', source=name, no_test_gt=True,
                    original_instances=int(a['instances'].max(initial=0)), candidate_instances=int(ids.max(initial=0)),
                    adapter_only_instances=checks[name]['adapter_only_instances'],
                    classes_changed=candidate_classes != classes, inference=inference, readout=readout,
                    reference_final_exact=checks[name]['reference_final_exact'],
                    seeds_and_barriers_regenerated=True, old_markers_reused=False))
                if normalized:
                    rows[-1].update(input_normalization=dict(normalization),
                        normalized_adapter_only_instances=checks[name]['normalized_adapter_only_instances'])
                if conditioning_receipt is not None:
                    rows[-1].update(rgb_conditioning=conditioning_receipt)
            receipt_path.write_text(json.dumps(dict(complete=True, sources=checks),
                ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    result = dict(format='boundary_rgb_monitor_v1', global_step=int(step), rows=rows,
        full_readout=full, no_test_training=True, random_e0_is_not_identity=True,
        input_mode=inf['input_mode'], baseline_name=baseline_name, no_revert_unconnected=True,
        interpretation='Seen training mechanisms or unlabelled changes, not blackbox score or epoch selection')
    if normalized:
        result['input_normalization'] = dict(normalization)
    if erased_training:
        result.update(erasure_training=True, erasure_recipe=dict(cfg['erasure']),
            common_pair_indices=list(settings['pair_indices']), erasure_pair_indices=list(extra_indices),
            fixed_pair_epoch=getattr(dataset, 'epoch', None),
            erasure_rng_policy='stateless by source/draw/epoch; monitor access cannot consume training stream')
    if cfg.get('restored_rgb_manifest'):
        result.update(rgb_conditioning='frozen_N1_local_zero_float32_RGB',
            restored_training_manifest=str(cfg['restored_rgb_manifest']),
            restored_monitor_manifest=str(settings['restored_rgb_manifest']))
    if shift_training:
        result.update(shift_consistency=dict(shift_settings),
                      shift_probe_input='clean_T8_all_fixed_pairs',
                      shift_picture_indices=list(settings.get('shift_pair_indices', [])))
    (folder / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                   allow_nan=False), encoding='utf-8')
    return result
