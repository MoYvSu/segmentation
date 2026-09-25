# -*- coding: utf-8 -*-
"""在全部人工训练源上诊断语义头的光照敏感性；不产生验证集或竞赛预测。

固定新 GT 实例，用部署相同的原尺寸概率均值投票。只运行 encoder 和语义头，
光照在冻结 D5a 后施加；像素和实例统计分别记录，不能称作官方指标。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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


def classification_metrics(target, probability):
    """0/1 以外的目标忽略；每个输入元素等权，可用于像素或实例。"""
    target, probability = np.asarray(target), np.asarray(probability)
    if target.shape != probability.shape:
        raise ValueError('Target and probability shapes differ')
    known = (target == 0) | (target == 1)
    truth = target[known].astype(np.int64)
    scores = probability[known].astype(np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Known-target probabilities must be finite within [0,1]')
    prediction = scores > .5  # 保留部署严格阈值；等于 .5 是珠光体。
    confusion = np.bincount(2 * truth + prediction, minlength=4).reshape(2, 2)
    scores = scores.clip(1e-7, 1 - 1e-7)
    bce_sum = float(-(truth * np.log(scores) + (1 - truth) * np.log1p(-scores)).sum())
    return metrics_from_counts(confusion, bce_sum)


def metrics_from_counts(confusion, bce_sum):
    confusion = np.asarray(confusion, dtype=np.int64)
    n = int(confusion.sum())
    recalls, ious = [], []
    for cls in (0, 1):
        count = int(confusion[cls].sum())
        union = int(confusion[cls].sum() + confusion[:, cls].sum() - confusion[cls, cls])
        recalls.append(float(confusion[cls, cls] / count) if count else None)
        ious.append(float(confusion[cls, cls] / union) if union else None)
    return {'n': n, 'confusion': confusion.tolist(), 'confusion_axes': 'rows GT P/F; columns predicted P/F',
            'accuracy': float(np.trace(confusion) / n) if n else None,
            'balanced_accuracy': float(np.mean([x for x in recalls if x is not None])) if n else None,
            'pearlite_recall': recalls[0], 'ferrite_recall': recalls[1],
            'pearlite_iou': ious[0], 'ferrite_iou': ious[1],
            'mean_iou': float(np.mean([x for x in ious if x is not None])) if n else None,
            'bce_sum': float(bce_sum), 'bce': float(bce_sum / n) if n else None}


def error_transitions(target, baseline_probability, probability):
    """只相对同一图/模型/条件的零扰动判断纠错或新增错误。"""
    target, baseline, current = map(np.asarray, (target, baseline_probability, probability))
    if not (target.shape == baseline.shape == current.shape):
        raise ValueError('Transition shapes differ')
    known = (target == 0) | (target == 1)
    truth, before, after = target[known], baseline[known] > .5, current[known] > .5
    was_correct, is_correct = before == truth, after == truth
    broken = int((was_correct & ~is_correct).sum())
    fixed = int((~was_correct & is_correct).sum())
    return {'n': int(known.sum()), 'changed': int((before != after).sum()),
            'correct_to_wrong': broken, 'wrong_to_correct': fixed, 'net_new_errors': broken - fixed,
            'pearlite_to_ferrite': int((~before & after).sum()),
            'ferrite_to_pearlite': int((before & ~after).sum())}


def make_core_instances(instances, radius, minimum_pixels=12):
    """按原尺寸等效半径腐蚀每个实例；太细实例不回退到边界区域。"""
    if radius < 0 or minimum_pixels < 1:
        raise ValueError('Invalid core settings')
    result = np.zeros_like(instances)
    for value in np.unique(instances):
        if value == 0:
            continue
        ys, xs = np.nonzero(instances == value)
        region = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        local = (instances[region] == value).astype(np.uint8)
        padded = cv2.copyMakeBorder(local, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distances = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        core = distances > float(radius)
        if int(core.sum()) >= int(minimum_pixels):
            result[region][core] = value
    return result


def prepare_regions(instances, core_instances):
    """每幅图只扫描一次 ID 包围框，后续21种推理统计复用局部掩码。"""
    regions = []
    for value in np.unique(instances):
        if value == 0:
            continue
        ys, xs = np.nonzero(instances == value)
        region = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        mask = instances[region] == value
        core_mask = core_instances[region] == value
        regions.append({'id': int(value), 'region': region, 'mask': mask, 'core_mask': core_mask,
                        'area': int(mask.sum()), 'core_area': int(core_mask.sum())})
    return regions


def region_votes(probability, regions):
    """与revote_instances相同的C顺序float32均值，不改投票精度或阈值。"""
    scores, core_scores = {}, {}
    for item in regions:
        local = probability[item['region']]
        scores[item['id']] = float(local[item['mask']].mean())
        if item['core_area']:
            core_scores[item['id']] = float(local[item['core_mask']].mean())
    return scores, core_scores


def evaluate_native(probability, instances, class_lookup, core_instances, *,
                    baseline_probability=None, clipped=None, regions=None, baseline_votes=None):
    """原尺寸 GT 固定；核心和无截断核心是防止补缝边界混杂的附加诊断。"""
    if instances.dtype != np.uint16 or core_instances.dtype != np.uint16:
        raise ValueError('Native GT and core maps must be uint16')
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != instances.shape or core_instances.shape != instances.shape:
        raise ValueError('Native maps must have matching shapes')
    if clipped is None:
        clipped = np.zeros(instances.shape, dtype=bool)
    if clipped.shape != instances.shape:
        raise ValueError('Clipping mask shape differs')
    target = class_lookup[instances]
    regions = prepare_regions(instances, core_instances) if regions is None else regions
    scores, core_scores = region_votes(probability, regions)
    if baseline_probability is None:
        baseline_probability = probability
    base_scores, base_core_scores = (region_votes(np.asarray(baseline_probability, dtype=np.float32), regions)
                                     if baseline_votes is None else baseline_votes)
    records = []
    for item in regions:
        value = item['id']
        if class_lookup[value] not in (0, 1):
            continue
        score, core_area = scores[value], item['core_area']
        clipped_core = clipped[item['region']][item['core_mask']]
        records.append({'id': value, 'gt': int(class_lookup[value]), 'p_f': score,
            'pred': int(score > .5), 'baseline_p_f': base_scores[value],
            'core_p_f': core_scores.get(value), 'core_pred': int(core_scores[value] > .5) if value in core_scores else None,
            'baseline_core_p_f': base_core_scores.get(value), 'area': item['area'],
            'core_area': core_area,
            'unclipped_core_fraction': float((~clipped_core).mean()) if core_area else None})

    def instance_subset(rows, core=False):
        field, base_field = ('core_p_f', 'baseline_core_p_f') if core else ('p_f', 'baseline_p_f')
        truth = np.asarray([row['gt'] for row in rows], dtype=np.int64)
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        base_values = np.asarray([row[base_field] for row in rows], dtype=np.float64)
        return {**classification_metrics(truth, values),
                'transitions': error_transitions(truth, base_values, values)}

    core_rows = [row for row in records if row['core_area'] > 0]
    no_clip_rows = [row for row in core_rows if row['unclipped_core_fraction'] == 1.0]
    core_target = np.where(core_instances > 0, class_lookup[core_instances], -1)
    return {'pixels': {**classification_metrics(target, probability),
                       'transitions': error_transitions(target, baseline_probability, probability)},
            'instances': instance_subset(records), 'core_instances': instance_subset(core_rows, True),
            'unclipped_core_instances': instance_subset(no_clip_rows, True),
            'core_pixels': classification_metrics(core_target, probability),
            'core_excluded': {'instances': len(records) - len(core_rows),
                'pearlite': sum(row['gt'] == 0 and row['core_area'] == 0 for row in records),
                'ferrite': sum(row['gt'] == 1 and row['core_area'] == 0 for row in records)},
            'instance_records': records}


def inverse_geometry(tensor, sample):
    """PairedDegradationDataset先双翻转后旋转，这里严格逆序恢复原朝向。"""
    turns = int(sample['rotation_k'])
    result = torch.rot90(tensor, -turns, dims=(-2, -1)) if turns else tensor
    if sample['vertical_flip']:
        result = result.flip(-2)
    if sample['horizontal_flip']:
        result = result.flip(-1)
    return result.contiguous()


def paired_bootstrap(rows, metric, draws=1000, seed=20260925):
    """以整张源图重采样，报告逐图等权差值；不把实例当独立重复。"""
    differences = np.asarray([row['current'][metric] - row['baseline'][metric]
                              for row in rows if row['current'][metric] is not None
                              and row['baseline'][metric] is not None], dtype=np.float64)
    if len(differences) == 0:
        return {'images': 0, 'mean_delta': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(len(differences), size=(draws, len(differences)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {'images': len(differences), 'mean_delta': float(differences.mean()),
            'ci95': [float(low), float(high)], 'positive_images': int((differences > 1e-12).sum()),
            'negative_images': int((differences < -1e-12).sum()),
            'unchanged_images': int((np.abs(differences) <= 1e-12).sum()),
            'bootstrap_draws': draws, 'bootstrap_unit': 'source_image', 'weighting': 'equal_image'}


def aggregate_report(report):
    summary = {}
    for condition in ('clean', 'blur'):
        summary[condition] = {}
        for model in report['models']:
            summary[condition][model] = {}
            for variant in report['variants']:
                key = variant['key']
                images = [row['conditions'][condition]['models'][model]
                          for row in report['images'].values()]
                result = {}
                for level in ('pixels', 'instances', 'core_instances', 'unclipped_core_instances', 'core_pixels'):
                    values = [row[key][level] for row in images]
                    confusion = sum((np.asarray(v['confusion'], dtype=np.int64) for v in values), np.zeros((2, 2), dtype=np.int64))
                    result[level] = metrics_from_counts(confusion, sum(v['bce_sum'] for v in values))
                    if 'transitions' in values[0]:
                        result[level]['transitions'] = {name: sum(v['transitions'][name] for v in values)
                                                       for name in values[0]['transitions']}
                    if level != 'unclipped_core_instances':
                        paired = [{'current': row[key][level], 'baseline': row['offset_0'][level]} for row in images]
                        result[level]['paired_image_accuracy'] = paired_bootstrap(paired, 'accuracy')
                        result[level]['paired_image_bce'] = paired_bootstrap(paired, 'bce')
                summary[condition][model][key] = result
    return summary


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _resize(array, shape, nearest=False):
    if tuple(array.shape[:2]) == tuple(shape):
        return array.copy()
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)


def _preview_rgb(image, content_shape, preview_shape):
    height, width = content_shape
    rgb = image[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
    return _resize(np.rint(rgb.clip(0, 1) * 255).astype(np.uint8), preview_shape)


def _variant_specs():
    return [{'key': 'offset_0', 'kind': 'offset', 'value': 0.0},
            {'key': 'offset_m010', 'kind': 'offset', 'value': -.1},
            {'key': 'offset_p010', 'kind': 'offset', 'value': .1},
            {'key': 'local_m010', 'kind': 'local', 'value': -.1},
            {'key': 'local_p010', 'kind': 'local', 'value': .1},
            {'key': 'gamma_080', 'kind': 'gamma', 'value': .8},
            {'key': 'gamma_125', 'kind': 'gamma', 'value': 1.25}]


def illuminated_input(image, spec, content_shape):
    from data.semantic_illumination import apply_illumination, fixed_smooth_field
    field = fixed_smooth_field(image, content_shape=content_shape) if spec['kind'] == 'local' else None
    before_clip = apply_illumination(image, spec['kind'], spec['value'], field=field, clamp=False)
    changed = apply_illumination(image, spec['kind'], spec['value'], field=field)
    clipped = ((before_clip < 0) | (before_clip > 1)).any(dim=1, keepdim=True)
    height, width = content_shape
    weights = image.new_tensor((.2126, .7152, .0722)).view(1, 3, 1, 1)
    difference = ((changed - image) * weights).sum(1, keepdim=True)[:, :, :height, :width]
    valid_clip = clipped[:, :, :height, :width]
    metrics = {'mean_y_change': float(difference.mean()), 'mean_absolute_y_change': float(difference.abs().mean()),
               'y_change_std': float(difference.std(unbiased=False)),
               'clipped_pixel_fraction': float(valid_clip.float().mean()), 'valid_pixels': int(height * width),
               'kind': spec['kind'], 'value': spec['value']}
    return changed, clipped, metrics


@torch.no_grad()
def run_probe(config_path, output_dir, images_limit=None):
    # 延迟模型导入，CPU测试只读取统计函数。
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import build_backend, load_backend, restore_first
    from train_backend_adaptation import sha
    from train_semantic_d5a import build_dataset, frozen_digest, get_restorer
    from utils.affinity_deployment import crop_letterbox_output
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env on the GPU server')
    if images_limit is not None and images_limit < 1:
        raise ValueError('images-limit must be positive')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    config = load_config(config_path)
    cfg = config['semantic_adaptation']
    voting = config['inference']
    if (voting.get('semantic_vote_mode'), voting.get('semantic_vote_erode_width'),
            voting.get('semantic_vote_threshold')) != ('probability_mean', 0, .5):
        raise ValueError('Diagnostic requires the fixed deployed mean vote and strict .5 threshold')
    dataset, data_metadata = build_dataset(config)
    base = dataset.base_dataset
    dataset.probabilities = np.asarray([0., 1., 0., 0.])
    dataset.degradation['profile_probabilities'] = [0., 1., 0., 0.]
    dataset.set_epoch(0)
    files = base.samples[:images_limit] if images_limit else base.samples
    output = Path(project_path(config, output_dir))
    output.mkdir(parents=True, exist_ok=False)
    preview_dir = output / 'previews'
    preview_dir.mkdir()
    variants = _variant_specs()
    report = {'status': 'running', 'config': str(config_path), 'variants': variants, 'models': {}, 'images': {},
        'protocol': {'cohort': 'all_manual_training_sources_no_holdout', 'expected_sources': len(base),
            'selected_sources': len(files), 'smoke_subset': images_limit is not None,
            'scope': 'Training-source robustness diagnosis, not held-out generalization or official competition evaluation.',
            'conditions': {'clean': 'original RGB letterbox -> frozen D5a first_x0',
                'blur': 'fixed-seed D5a training degradation conditioned on blur + optional achromatic noise -> frozen D5a first_x0'},
            'illumination_position': 'after D5a, before encoder and semantic head',
            'precision': 'FP32', 'affinity_forward': False, 'watershed': False,
            'ground_truth': 'same native completed GT instance_map and class lookup for all variants',
            'voting': {'mode': 'probability_mean', 'strict_threshold': .5},
            'core_radius_model_pixels': 3, 'core_minimum_native_pixels': 12,
            'unclipped_core_rule': 'Retain only instances whose entire retained core has zero clipped pixels.',
            'primary_variants': ['offset_m010', 'offset_p010', 'local_m010', 'local_p010'],
            'gamma_is_secondary': 'Gamma changes local contrast as well as brightness.',
            'preview_selection': 'first three sorted source filenames, independent of outcome',
            'restoration_seed_rule': 'inference_seed + full sorted source index; shared by clean and blur',
            'degradation_epoch': 0, 'degradation_profile_probabilities_override': [0., 1., 0., 0.],
            'limitations': ['All sources were used during training; this cannot establish test generalization.',
                'Completed GT can contain propagated labels; core diagnostics reduce boundary sensitivity, not label uncertainty.',
                'Digital brightness transforms do not simulate all physical lighting processes.']},
        'data': data_metadata, 'restoration': deepcopy(cfg['restoration'])}
    report_path = output / 'results.json'
    cache = {}
    started = time.time()

    # D5a每个图/条件仅执行一次，CPU缓存完整FP32输入供三模型共同使用。
    restorer = get_restorer(config, device)
    for index, path in enumerate(files):
        sample = base[index]
        rgb = read_rgb(str(path))
        instances, lookup = load_completed_semantic_source(base.completed_gt_dir / (path.stem + '_gt.npz'), rgb.shape[:2])
        if int(instances.max()) > 65535:
            raise ValueError('Native instance IDs exceed the deployment voting uint16 contract')
        instances = instances.astype(np.uint16)
        height, width = map(int, sample['input_content_shape'].tolist())
        native_shape = instances.shape
        radius = 3 * max(native_shape[0] / height, native_shape[1] / width)
        core = make_core_instances(instances, radius)
        scale = min(1., 768 / native_shape[1])
        preview_shape = (max(1, round(native_shape[0] * scale)), max(1, round(native_shape[1] * scale)))
        row = {'source_image': str(path), 'native_shape': list(native_shape),
            'content_shape': [height, width], 'core_radius_native_pixels': radius,
            'gt_instance_path': str(base.completed_gt_dir / (path.stem + '_gt.npz')),
            'gt_class_path': str(base.completed_gt_dir / (path.stem + '_class.json')),
            'conditions': {}, 'preview_shape': list(preview_shape) if index < 3 else None}
        degraded = dataset[index]
        undo = inverse_geometry(degraded['semantic_instance_map'], degraded)
        if not torch.equal(undo, sample['semantic_instance_map']):
            raise RuntimeError('Inverse augmentation failed to recover native GT orientation')
        blur = inverse_geometry(degraded['image'], degraded)
        cache[path.stem] = {'instances': instances, 'lookup': lookup, 'core': core,
            'regions': prepare_regions(instances, core),
            'native_shape': native_shape, 'content_shape': (height, width),
            'preview_shape': preview_shape if index < 3 else None, 'conditions': {}}
        for condition_index, (condition, input_tensor) in enumerate((('clean', sample['image']), ('blur', blur))):
            seed = int(cfg['restoration']['inference_seed']) + index
            restored = restore_first(restorer, input_tensor.unsqueeze(0).to(device), seed).float()
            c_row = {'restoration_seed': seed, 'input_metrics': {}, 'models': {}}
            if condition == 'blur':
                c_row['degradation'] = {key: degraded[key] for key in
                    ('profile', 'is_spatial', 'endpoint_applied', 'noise_sigma', 'base_sigma')}
                c_row['geometry_inverse_exact'] = True
            previews = {'raw_rgb': _resize(rgb, preview_shape),
                        'condition_rgb': _preview_rgb(input_tensor.unsqueeze(0), (height, width), preview_shape),
                        'gt_instances': _resize(instances, preview_shape, True),
                        'gt_semantic': _resize(lookup[instances], preview_shape, True).astype(np.int8)} if index < 3 else {}
            for variant in variants:
                changed, clipped, metrics = illuminated_input(restored, variant, (height, width))
                if variant['key'] == 'offset_0' and not torch.equal(changed, restored):
                    raise RuntimeError('Zero illumination transform is not exact identity')
                c_row['input_metrics'][variant['key']] = metrics
                if index < 3:
                    previews[variant['key']] = _preview_rgb(changed, (height, width), preview_shape)
            cache[path.stem]['conditions'][condition] = restored.cpu()
            if index < 3:
                filename = f'previews/{path.stem}_{condition}_inputs.npz'
                np.savez_compressed(output / filename, **previews)
                c_row['preview_inputs'] = filename
            row['conditions'][condition] = c_row
            print(f'D5a cached once: {path.stem}/{condition}', flush=True)
        report['images'][path.stem] = row
        _write_json(report_path, report)
    del restorer, restored, changed, clipped
    gc.collect()
    torch.cuda.empty_cache()

    reference_frozen = None
    heads, shared_encoder = {}, None
    train_root = Path(project_path(config, cfg['output_dir']))
    epoch = int(cfg['epochs'])
    for name in ('original', 'full', 'simple'):
        if name == 'original':
            model, metadata = build_backend(config, device)
            model_info = {'sources': metadata['sources'], 'epoch': metadata['sources']['semantic'].get('epoch')}
        else:
            checkpoint = train_root / name / f'epoch_{epoch:03d}.pt'
            model, bundle = load_backend(str(checkpoint), config, device)
            if (bundle['backend_adaptation'].get('semantic_adaptation', {}).get('arm') != name
                    or int(bundle['epoch']) != epoch):
                raise ValueError('Checkpoint arm or epoch differs from configuration')
            model_info = {'checkpoint': str(checkpoint), 'sha256': sha(checkpoint), 'epoch': epoch,
                          'architecture': bundle['architecture']['semantic_decoder']}
            del bundle
        model.eval().requires_grad_(False)
        model.encoder.trainable_lora = False
        digest = frozen_digest(model)
        if reference_frozen is None:
            reference_frozen = digest
        if digest != reference_frozen:
            raise RuntimeError('Frozen encoder/LoRA/affinity differ across models')
        model_info.update(nonsemantic_state_sha256=digest, nonsemantic_exact_across_models=True,
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        report['models'][name] = model_info
        heads[name] = model.semantic_decoder
        if shared_encoder is None:
            shared_encoder = model.encoder
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report['protocol']['shared_encoder_forward'] = 'One FP32 encoder forward per image/condition/variant; three heads share features after exact nonsemantic state checks.'
    for stem, item in cache.items():
        height, width = item['content_shape']
        for condition, cached_input in item['conditions'].items():
            restored = cached_input.to(device)
            predictions = {name: {} for name in heads}
            all_metrics = {name: {} for name in heads}
            baselines, baseline_votes = {}, {}
            for variant in variants:
                changed, clipped, input_metrics = illuminated_input(restored, variant, (height, width))
                clipped_native = _resize(clipped[0, 0, :height, :width].cpu().numpy().astype(np.uint8),
                                         item['native_shape'], True).astype(bool)
                features = shared_encoder(changed)
                for name, head in heads.items():
                    logits = head(features, changed)
                    native = crop_letterbox_output(logits, 1024, 1024 - height, 1024 - width, item['native_shape']).cpu()
                    probability = torch.sigmoid(native)[0, 0].numpy().astype(np.float32, copy=False)
                    if variant['key'] == 'offset_0':
                        baselines[name] = probability.copy()
                        baseline_votes[name] = region_votes(baselines[name], item['regions'])
                    if name not in baselines:
                        raise RuntimeError('The exact zero variant must run first')
                    evaluated = evaluate_native(probability, item['instances'], item['lookup'], item['core'],
                        baseline_probability=baselines[name], clipped=clipped_native,
                        regions=item['regions'], baseline_votes=baseline_votes[name])
                    evaluated['zero_input_identity_exact'] = bool(torch.equal(changed, restored)) if variant['key'] == 'offset_0' else None
                    all_metrics[name][variant['key']] = evaluated
                    if item['preview_shape']:
                        predictions[name][variant['key']] = _resize(probability, item['preview_shape']).astype(np.float32)
                    print(f'probe {name}/{stem}/{condition}/{variant["key"]}: '
                          f'accuracy={evaluated["instances"]["accuracy"]:.4f}, '
                          f'new_wrong={evaluated["instances"]["transitions"]["correct_to_wrong"]}, '
                          f'fixed={evaluated["instances"]["transitions"]["wrong_to_correct"]}', flush=True)
                    del logits, native, probability
                del features, changed, clipped
            c_row = report['images'][stem]['conditions'][condition]
            c_row['models'] = all_metrics
            for name in heads:
                if item['preview_shape']:
                    filename = f'previews/{stem}_{condition}_{name}.npz'
                    np.savez_compressed(output / filename, **predictions[name])
                    c_row.setdefault('preview_probabilities', {})[name] = filename
        _write_json(report_path, report)
    del heads, shared_encoder, restored
    gc.collect()
    torch.cuda.empty_cache()
    report['summary'] = aggregate_report(report)
    report['status'] = 'completed'
    report['elapsed_seconds'] = time.time() - started
    _write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_d5a.yaml')
    parser.add_argument('--output-dir', default='outputs/labelled_light')
    parser.add_argument('--images-limit', type=int, default=None, help='仅用于明确标记的 CPU/GPU smoke；主实验不设置')
    args = parser.parse_args()
    report = run_probe(args.config, args.output_dir, args.images_limit)
    print(json.dumps({'status': report['status'], 'images': len(report['images']),
                      'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
