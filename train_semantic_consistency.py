# -*- coding: utf-8 -*-
"""保留完整simple权重的语义接续；共享几何、固定教师或独立灰度先验。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from data.mim_dataset import list_images
from data.semantic_consistency import (SemanticConsistencyDataset, apply_acquisition_appearance,
    build_reliable_semantic_targets, full_content_mask, masked_semantic_consistency)
from models.backend_adaptation import load_backend, restore_first, save_backend
from models.semantic_lora import SemanticLoRA, semantic_features
from train_backend_adaptation import prediction_maps, save_prediction, sha, tensor_digest, write_json
from train_direct_semantic_affinity import build_semantic_criterion, compute_semantic_loss, move_batch, set_seed
from train_semantic_d5a import (aligned_semantic_logits, build_dataset, frozen_digest,
    get_restorer, learning_rate_factor, monitor, semantic_train_mode)
from utils.affinity_deployment import prepare_image
from utils.config import load_config, project_path


def continue_simple(model):
    """只改训练开关，不调用会重置简化头的历史configure_arm。"""
    before = tensor_digest(model.semantic_decoder.state_dict().items())
    if model.semantic_decoder.semantic_residual is not None:
        raise ValueError('This experiment requires the original simple head without RGB residual')
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.semantic_decoder.modules()):
        raise ValueError('Extra forwards would alter BatchNorm state; expected the existing GroupNorm head')
    model.eval().requires_grad_(False)
    model.encoder.trainable_lora = False
    model.semantic_decoder.requires_grad_(True)
    if before != tensor_digest(model.semantic_decoder.state_dict().items()):
        raise RuntimeError('Warmstart unexpectedly changed semantic weights')
    return before


def verify_contract(config, saved):
    """路径允许迁移；冻结模型与实际部署参数从原simple继承。"""
    previous = saved['config']
    for key in ('affinity_deployment', 'post_process'):
        if config[key] != previous[key]:
            raise ValueError(f'Deployment contract changed: {key}')
    left, right = deepcopy(config['inference']), deepcopy(previous['inference'])
    left.pop('output_dir', None)
    right.pop('output_dir', None)
    if left != right:
        raise ValueError('Inference contract changed beyond output directory')
    if config['semantic_adaptation']['restoration'] != previous['semantic_adaptation']['restoration']:
        raise ValueError('D5a identity, sampling or inference seed changed')
    if saved['backend_adaptation'].get('semantic_adaptation', {}).get('arm') != 'simple':
        raise ValueError('Source checkpoint is not the original simple arm')
    if config['sam2'].get('input_normalization', 'legacy_none') != 'legacy_none':
        raise ValueError('Expected legacy_none encoder normalization')


def consistency_weight(epoch, maximum, ramp_epochs):
    return float(maximum) * min(1., epoch / max(1, int(ramp_epochs)))


def candidate_arm(config):
    mode = config['semantic_consistency'].get('target_mode', 'teacher')
    if mode not in ('teacher', 'gray_prior'):
        raise ValueError(f'Unknown semantic target mode: {mode}')
    return 'prior' if mode == 'gray_prior' else 'consistency'


@contextmanager
def unlabeled_rng(seed, device):
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(int(seed))
        yield


def unbatch_value(value):
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: unbatch_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return unbatch_value(value[0]) if len(value) == 1 else [unbatch_value(v) for v in value]
    return value


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def gradient_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def lora_comparison_config(config):
    """本轮只允许语义LoRA与输出位置不同；gray4损失、数据和预算均保留。"""
    result = deepcopy(config)
    result.pop('semantic_lora', None)
    result.pop('semantic_scale', None)
    result['semantic_adaptation'].pop('output_dir', None)
    return json.loads(json.dumps(result, allow_nan=False))


def load_lora_reference(config):
    lcfg = config.get('semantic_lora', {})
    if not lcfg.get('enabled', False):
        return None, None
    directory = Path(project_path(config, lcfg['reference_dir']))
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    count = (config['semantic_adaptation']['epochs'] * config['semantic_adaptation']['expected_manual_sources']
             * config['semantic_adaptation']['manual_repeats'])
    if (status['status'] != 'completed' or status['smoke'] or status['arm'] != 'prior'
            or status['updates'] != count or status['failed_updates']
            or lora_comparison_config(status['config']) != lora_comparison_config(config)):
        raise RuntimeError('Semantic LoRA requires the complete, identically configured gray4 reference')
    if sha(Path(status['final_checkpoint'])) != lcfg['reference_sha256']:
        raise RuntimeError('Gray4 reference checkpoint changed')
    rows = [json.loads(line) for line in (directory / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != count or any(row['updates'] != i + 1 for i, row in enumerate(rows)):
        raise RuntimeError('Incomplete gray4 view replay')
    return status, rows


def shared_frozen_digest(model):
    if getattr(model, 'semantic_lora', None) is None:
        return frozen_digest(model)
    return tensor_digest((name, value) for name, value in model.state_dict().items()
                         if not name.startswith(('semantic_decoder.', 'semantic_lora.')))


def verify_lora_input(reference, current, *, unlabeled=False):
    fields = ('name', 'global_draw', 'source_index', 'profile', 'is_spatial', 'endpoint_applied',
              'noise_sigma', 'horizontal_flip', 'vertical_flip', 'rotation_k', 'appearance',
              'strong_restoration_seed', 'accepted_pixels', 'accepted_ferrite', 'accepted_pearlite',
              'valid_pixels', 'known_gt_excluded_pixels') if unlabeled else (
              'epoch', 'step', 'updates', 'name', 'seed', 'draw_index', 'input_sha256',
              'appearance_input_sha256', 'appearance', 'restoration_seed', 'learning_rate')
    for key in fields:
        # 尺度候选对照的是裁切前原先验；实际参与loss的裁后计数另行记录。
        value = current.get('scale_parent_target', {}).get(key, current[key]) if unlabeled else current[key]
        if value != reference[key]:
            raise RuntimeError(f'Gray4 replay input/target differs: {key}')
    if unlabeled and current['hard_views']['selected_index'] != reference['hard_views']['selected_index']:
        raise RuntimeError('Gray4 selected view changed')


def gray_unlabelled_loss(model, restorer, raw, config, device, seed, preview_path=None, reference=None):
    from data.semantic_gray import build_gray_targets, gray_prior_loss, save_prior_preview
    cfg, gcfg = config['semantic_consistency'], config['semantic_gray']
    # 目标在清晰原图上独立生成，强退化与D5a不会修改目标；GT已知区排除。
    target, mask, info = build_gray_targets(raw['weak_image'], raw['valid_content'],
        raw['known_gt_mask'], gcfg, grid=gcfg['comparison_grid'])
    raw = move_batch(raw, device)
    target, mask = target.to(device), mask.to(device)
    scale_spec = None
    if config.get('semantic_scale', {}).get('enabled', False):
        from data.semantic_scale import choose_scale_view
        scale_spec = choose_scale_view(raw['strong_image'], raw['valid_content'],
                                       config['semantic_scale'], seed + 9100000)
    hard = config.get('semantic_hard', {}).get('enabled', False)
    if hard:
        from data.semantic_hard import hard_gray_forward
        logits, strong, appearance, selection = hard_gray_forward(model, restorer, raw, target, mask, config, seed,
            replay=reference['hard_views'] if reference is not None else None, scale_spec=scale_spec)
        info['hard_views'] = selection
    else:
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            strong, appearance = apply_acquisition_appearance(
                raw['strong_image'], raw['valid_content'], cfg['appearance'], seed + 3000000)
            strong = restore_first(restorer, strong, seed + 4000000)
            features = model.encoder(strong.float())
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.semantic_decoder(features, strong)
    if scale_spec is not None:
        from data.semantic_scale import apply_scale_target
        info['scale_view'] = scale_spec
        info['scale_parent_target'] = {key: info[key] for key in (
            'accepted_pixels', 'accepted_ferrite', 'accepted_pearlite', 'valid_pixels', 'known_gt_excluded_pixels')}
        if scale_spec['mode'] == 'local':
            grid = target.shape[-2:]
            valid = torch.nn.functional.interpolate(raw['valid_content'].float(), grid, mode='area') >= 1
            known = torch.nn.functional.interpolate(raw['known_gt_mask'].float(), grid, mode='area') > 0
            target, mask = apply_scale_target(target, scale_spec), apply_scale_target(mask, scale_spec)
            valid, known = apply_scale_target(valid, scale_spec), apply_scale_target(known, scale_spec)
            info.update(accepted_pixels=int(mask.sum()), accepted_ferrite=int((mask & (target > .5)).sum()),
                accepted_pearlite=int((mask & (target <= .5)).sum()), valid_pixels=int(valid.sum()),
                known_gt_excluded_pixels=int((valid & known).sum()),
                accepted_fraction=float(mask.sum() / valid.sum().clamp_min(1)))
    loss, activity = gray_prior_loss(logits, target, mask,
        mode=gcfg.get('loss_mode', 'soft_bce'), minimum_confidence=gcfg.get('minimum_confidence', .9),
        maximum_normalization_gain=gcfg.get('maximum_normalization_gain', 1.),
        return_info=True)
    probability = logits.detach().float().sigmoid()
    accepted = int(mask.sum())
    info.update(**activity, appearance=appearance, name=raw['name'][0],
        prior_student_abs_gap=float((probability - target).abs()[mask].mean()) if accepted else 0.,
        class_disagreement=int((((probability > .5) != (target > .5)) & mask).sum()),
        prior_f_student_p=int(((target > .5) & (probability <= .5) & mask).sum()),
        prior_p_student_f=int(((target <= .5) & (probability > .5) & mask).sum()),
        strong_restoration_seed=seed + 4000000)
    for key in ('global_draw', 'source_index', 'profile', 'is_spatial', 'endpoint_applied',
                'noise_sigma', 'horizontal_flip', 'vertical_flip', 'rotation_k'):
        info[key] = unbatch_value(raw[key])
    if reference is not None:
        verify_lora_input(reference, info, unlabeled=True)
    if preview_path is not None:
        clean, valid = raw['weak_image'], raw['valid_content']
        if scale_spec is not None:
            from data.semantic_scale import apply_scale_image, apply_scale_target
            clean, valid = apply_scale_image(clean, scale_spec), apply_scale_target(valid, scale_spec)
        save_prior_preview(preview_path, clean, strong, target, mask, logits, valid)
    return loss, info


def unlabelled_loss(model, teacher, restorer, raw, config, device, seed, preview_path=None, reference=None):
    """仅学生强视图建图；教师与编码器冻结。调用方负责隔离Dropout随机流。"""
    if candidate_arm(config) == 'prior':
        return gray_unlabelled_loss(model, restorer, raw, config, device, seed, preview_path, reference)
    cfg = config['semantic_consistency']
    raw = move_batch(raw, device)
    weak_seed = seed + 2000000
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        weak = restore_first(restorer, raw['weak_image'], weak_seed)
        weak_logits = teacher(model.encoder(weak.float()), weak)
        check = restore_first(restorer, raw['weak_check_image'], weak_seed)
        check_logits = teacher(model.encoder(check.float()), check)
        target, mask, info = build_reliable_semantic_targets(
            weak_logits, check_logits, raw['valid_content'], confidence=cfg['confidence'],
            max_probability_gap=cfg['max_probability_gap'], erosion_radius=cfg['erosion_radius'])
        strong, appearance = apply_acquisition_appearance(
            raw['strong_image'], raw['valid_content'], cfg['appearance'], seed + 3000000)
        strong = restore_first(restorer, strong, seed + 4000000)
        features = model.encoder(strong.float())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model.semantic_decoder(features, strong)
    loss, loss_info = masked_semantic_consistency(logits, target, mask)
    # 统一运行记录的计数含义：可靠像素均是教师预测，不是真实组织占比。
    accepted = int(mask.sum())
    ferrite = int((mask.bool() & (target >= .5)).sum())
    info = {**info, **loss_info, 'accepted_pixels': accepted,
            'accepted_ferrite': ferrite, 'accepted_pearlite': accepted - ferrite,
            'appearance': appearance, 'name': raw['name'][0],
            'teacher_student_abs_gap': float((torch.sigmoid(logits.detach().float()) - target).abs()[mask.bool()].mean()) if accepted else 0.}
    for key in ('global_draw', 'source_index', 'weak_stats', 'profile', 'is_spatial',
                'endpoint_applied', 'noise_sigma', 'horizontal_flip', 'vertical_flip', 'rotation_k'):
        info[key] = unbatch_value(raw[key])
    info.update(weak_restoration_seed=weak_seed, strong_restoration_seed=seed + 4000000)
    return loss, info


def train(config, arm, output_dir, smoke=False, zero_weight=False, freeze_semantic_lora=False):
    if config['semantic_consistency'].get('deterministic_training', False):
        # 零权重重放发现CUDA归约抖动；本轮两臂共同启用确定性实现，不能只放宽比较阈值。
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run with sam2_env and a BF16-capable GPU')
    candidate = candidate_arm(config)
    if arm not in ('control', candidate) or zero_weight and arm != candidate:
        raise ValueError('Invalid comparison arm')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    cfg, ucfg = config['semantic_adaptation'], config['semantic_consistency']
    lora_reference, replay_rows = load_lora_reference(config)
    use_lora = replay_rows is not None
    use_scale = config.get('semantic_scale', {}).get('enabled', False)
    if use_scale and (not use_lora or arm != 'prior'):
        raise ValueError('Scale experiment requires the existing private-LoRA prior training path')
    if use_lora and (arm != 'prior' or zero_weight or not config.get('semantic_hard', {}).get('enabled')):
        raise ValueError('Independent semantic LoRA requires the unchanged gray4 prior path')
    if freeze_semantic_lora and (not use_lora or not smoke):
        raise ValueError('Freeze-LoRA replay is only the short regression check')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    source = Path(project_path(config, ucfg['initial_checkpoint']))
    if sha(source) != ucfg['initial_sha256']:
        raise ValueError('Original simple e60 checkpoint identity changed')
    set_seed(cfg['seed'])
    model, saved = load_backend(source, config, device)
    verify_contract(config, saved)
    metadata = deepcopy(saved['backend_adaptation'])
    source_epoch = int(saved['epoch'])
    frozen_initial = frozen_digest(model)
    semantic_initial = continue_simple(model)
    parameters = list(model.semantic_decoder.parameters())
    before_head = {name: value.detach().cpu().clone() for name, value in model.semantic_decoder.named_parameters()}
    before_lora, lora_parameters = None, []
    if use_lora:
        if getattr(model, 'semantic_lora', None) is not None:
            raise ValueError('Initial simple checkpoint must not already contain an independent semantic LoRA')
        model.semantic_lora = SemanticLoRA(model.encoder.trunk,
            gradient_checkpointing=config['semantic_lora']['gradient_checkpointing']).to(device)
        model.semantic_lora.requires_grad_(not freeze_semantic_lora)
        before_lora = {name: value.detach().cpu().clone() for name, value in model.semantic_lora.state_dict().items()}
        lora_parameters = [p for p in model.semantic_lora.parameters() if p.requires_grad]
        parameters += lora_parameters
        metadata['architecture']['semantic_lora'] = model.semantic_lora.architecture()
        metadata['semantic_adaptation']['frozen'] = ['D5a', 'SAM2_base', 'affinity_LoRA', 'affinity']
    restorer = get_restorer(config, device)
    restoration_initial = tensor_digest(restorer.state_dict().items())
    dataset, data_info = build_dataset(config)
    if data_info != saved['extra']['data']:
        raise ValueError('Labeled source/GT or existing blur/noise recipe changed')
    metadata['semantic_consistency'] = {'version': 2 if candidate == 'prior' else 1, 'arm': arm,
        'initialization': 'continue_original_simple_e60', 'source_checkpoint': ucfg['initial_checkpoint'],
        'source_sha256': ucfg['initial_sha256'], 'source_epoch': source_epoch,
        'initial_semantic_sha256': semantic_initial,
        'teacher': 'clean_raw_gray_prior' if candidate == 'prior' else 'fixed_initial_simple_head',
        'shared_encoder_frozen': True, 'extra_unlabeled_stream': arm == candidate,
        'common_acquisition_appearance': ucfg['appearance'], 'zero_weight_replay': zero_weight}
    if use_lora:
        metadata['semantic_lora'] = {'version': SemanticLoRA.VERSION, 'initialization': 'clone_existing_lora',
            'trainable': not freeze_semantic_lora, 'gray4_views_replayed': True,
            'reference_sha256': config['semantic_lora']['reference_sha256'],
            'shared_base_and_affinity_lora_frozen': True, 'encoder_precision': 'FP32'}
    if candidate == 'prior':
        metadata['semantic_consistency']['gray_loss_mode'] = config['semantic_gray'].get('loss_mode', 'soft_bce')
        metadata['semantic_consistency']['gray_minimum_confidence'] = config['semantic_gray'].get('minimum_confidence', .9)
        metadata['semantic_consistency']['gray_maximum_normalization_gain'] = config['semantic_gray'].get('maximum_normalization_gain', 1.)
        if config.get('semantic_hard', {}).get('enabled', False):
            metadata['semantic_consistency']['hard_views'] = deepcopy(config['semantic_hard'])
    del saved
    teacher = deepcopy(model.semantic_decoder).eval().requires_grad_(False) if arm == 'consistency' else None
    teacher_initial = tensor_digest(teacher.state_dict().items()) if teacher is not None else None
    unlabeled = None
    if arm == candidate:
        pool = Path(project_path(config, ucfg['unlabeled_dir'])).resolve()
        if pool == Path(project_path(config, config['inference']['test_dir'])).resolve():
            raise ValueError('Test images must not enter training')
        dataset_type, extra_args = SemanticConsistencyDataset, {}
        if candidate == 'prior':
            from data.semantic_gray import GrayPriorDataset
            dataset_type, extra_args = GrayPriorDataset, {'manual_dataset': dataset.base_dataset}
        unlabeled = dataset_type(pool, data_info['degradation'], cfg['noise'], cfg['seed'] + 271,
            draws_per_epoch=len(dataset), expected_sources=ucfg['expected_sources'], image_size=1024,
            weak_config=ucfg['weak_check'], **extra_args)
        if candidate == 'prior':
            write_json(output / 'manual_source_mapping.json', unlabeled.manual_mapping)
    total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in restorer.parameters())
    training_total = total + (sum(p.numel() for p in teacher.parameters()) if teacher is not None else 0)
    if training_total >= 500000000:
        raise ValueError('Parameter budget exceeded')
    if any(p.requires_grad for name, p in model.named_parameters()
           if not name.startswith(('semantic_decoder.', 'semantic_lora.') if use_lora else ('semantic_decoder.',))):
        raise RuntimeError('A frozen backend parameter became trainable')
    status = {'status': 'preflight', 'arm': arm, 'smoke': smoke, 'zero_weight': zero_weight,
        'initialization': metadata['semantic_consistency'], 'sources': metadata['sources'],
        'data': data_info, 'config': config, 'frozen_state_sha256': frozen_initial,
        'd5a_state_sha256': restoration_initial, 'initial_semantic_sha256': semantic_initial,
        'teacher_state_sha256': teacher_initial, 'total_parameters': total,
        'training_total_parameters': training_total, 'trainable_parameters': sum(p.numel() for p in parameters),
        'selection': 'prespecified_epoch20_no_holdout_no_test_selection', 'updates': 0, 'failed_updates': 0}
    if use_lora:
        status['semantic_lora'] = metadata['semantic_lora']
        status['semantic_lora_parameters'] = sum(p.numel() for p in model.semantic_lora.parameters())
    if use_scale:
        metadata['semantic_scale'] = deepcopy(config['semantic_scale'])
        status['semantic_scale'] = metadata['semantic_scale']
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    first = files[0]
    image, semantic, boundary = prediction_maps(model, first, config, device, restorer, cfg['restoration']['inference_seed'])
    probe_dir = output / 'initial_probe'
    probe_dir.mkdir()
    instances, classes = save_prediction(image, semantic, boundary, config, probe_dir, first.stem)
    reference = Path(project_path(config, ucfg['original_prediction_dir']))
    old_instances = cv2.imread(str(reference / f'{first.stem}_inst.png'), cv2.IMREAD_UNCHANGED)
    old_classes = json.loads((reference / f'{first.stem}_class.json').read_text(encoding='utf-8'))
    if (not np.array_equal(instances, old_instances) or
            {str(k): int(v) for k, v in classes.items()} != old_classes or instances.dtype != np.uint16):
        raise RuntimeError('Warmstart final output does not reproduce original simple deployment')
    status['initial_final_output_equal'] = True
    write_json(output / 'status.json', status)
    _, probe, _, _ = prepare_image(first, 1024, device)
    probe = restore_first(restorer, probe, cfg['restoration']['inference_seed'])
    with torch.no_grad():
        initial_affinity = model(probe)['affinity_logits'].cpu()
    monitor(model, restorer, config, output, 0)
    if use_scale:
        from tools.semantic_scale_monitor import scale_monitor
        scale_monitor(model, restorer, dataset, config, output, 0)
    use_hard = unlabeled is not None and config.get('semantic_hard', {}).get('enabled', False)
    if use_hard:
        from data.semantic_hard import hard_monitor
        hard_monitor(model, restorer, unlabeled, config, output, 0)
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rates']['simple'],
                                  weight_decay=cfg['weight_decay'], eps=1e-4)
    criterion = build_semantic_criterion(config, device)
    epochs = 1 if smoke else int(cfg['epochs'])
    visited, effective = set(), set()
    accepted_total = [0, 0]
    measured_nonzero_u = False
    steps, started = 0, time.time()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        if unlabeled is not None:
            unlabeled.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(cfg['seed'] + epoch * 2), pin_memory=True)
        lr = cfg['learning_rates']['simple'] * learning_rate_factor(epoch, cfg['epochs'], cfg['warmup_epochs'], cfg['minimum_lr_ratio'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        weight = 0. if zero_weight or arm == 'control' else consistency_weight(epoch, ucfg['maximum_weight'], ucfg['ramp_epochs'])
        semantic_train_mode(model)
        epoch_rows = []
        for index, raw in enumerate(loader):
            if smoke and index >= int(ucfg['smoke_steps']):
                break
            step_seed = int(cfg['seed']) + epoch * 10000 + index
            set_seed(step_seed)
            input_sha = tensor_digest([('image', raw['image'])])
            batch = move_batch(raw, device)
            batch['image'], appearance = apply_acquisition_appearance(
                batch['image'], full_content_mask(batch, batch['image']), ucfg['appearance'], step_seed + 5000000)
            appearance_sha = tensor_digest([('image', batch['image'])])
            batch['image'] = restore_first(restorer, batch['image'], step_seed + 1000000)
            scale_spec = None
            if use_scale:
                from data.semantic_scale import choose_scale_view, apply_scale_batch
                scale_spec = choose_scale_view(batch['image'], full_content_mask(batch, batch['image']),
                    config['semantic_scale'], step_seed + 9000000)
                batch = apply_scale_batch(batch, scale_spec)
            optimizer.zero_grad(set_to_none=True)
            if use_lora:
                with torch.autocast('cuda', enabled=False):
                    features = semantic_features(model, batch['image'].float())
            else:
                with torch.no_grad(), torch.autocast('cuda', enabled=False):
                    features = model.encoder(batch['image'].float())
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model.semantic_decoder(features, batch['image'])
            loss = compute_semantic_loss(criterion, aligned_semantic_logits(logits, batch), batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite supervised loss')
            loss.backward()
            supervised_value = float(loss.detach())
            supervised_norm = gradient_norm(parameters) if use_hard or smoke or steps == 0 or (
                candidate == 'prior' and index == 0) else None
            lora_supervised_norm = gradient_norm(lora_parameters) if use_lora else None
            del features, logits, loss, batch
            uvalue, unorm, uinfo, effective_weight = 0., None, None, weight
            if unlabeled is not None:
                # 额外视图与Dropout不能推进下一次监督前向的随机流。
                with unlabeled_rng(step_seed + 6000000, device):
                    uraw = default_collate([unlabeled[index]])
                    preview = output / 'prior_examples' / f'step_{steps:04d}.png' if candidate == 'prior' and steps < 3 else None
                    uloss, uinfo = unlabelled_loss(model, teacher, restorer, uraw, config, device, step_seed, preview,
                        reference=replay_rows[steps]['unlabeled'] if use_lora else None)
                    if not bool(torch.isfinite(uloss)):
                        raise FloatingPointError('Non-finite unlabeled loss')
                    if use_hard or smoke or (not measured_nonzero_u and uinfo['accepted_pixels'] > 0) or (
                            candidate == 'prior' and index == 0 and uinfo['accepted_pixels'] > 0):
                        grads = torch.autograd.grad(uloss, parameters, retain_graph=True)
                        unorm = float(torch.stack([g.detach().float().square().sum() for g in grads]).sum().sqrt())
                        measured_nonzero_u = measured_nonzero_u or unorm > 0
                        del grads
                    if use_hard:
                        from data.semantic_hard import gradient_budget_weight
                        effective_weight = gradient_budget_weight(weight, supervised_norm, unorm,
                            config['semantic_hard']['maximum_gradient_ratio'])
                    (effective_weight * uloss).backward()
                    uvalue = float(uloss.detach())
                    del uloss, uraw
                visited.add(uinfo['name'])
                if uinfo['accepted_pixels']:
                    effective.add(uinfo['name'])
                accepted_total[0] += uinfo['accepted_pearlite']
                accepted_total[1] += uinfo['accepted_ferrite']
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg['grad_clip'], error_if_nonfinite=True))
            if not math.isfinite(norm) or norm <= 0:
                raise FloatingPointError(f'Invalid total gradient: {norm}')
            optimizer.step()
            steps += 1
            row = {'epoch': epoch, 'step': index + 1, 'updates': steps, 'name': raw['name'][0],
                'seed': step_seed, 'draw_index': int(raw['draw_index']), 'input_sha256': input_sha,
                'appearance_input_sha256': appearance_sha, 'appearance': appearance,
                'restoration_seed': step_seed + 1000000, 'semantic_loss': supervised_value,
                'unlabeled_loss': uvalue, 'unlabeled_weight': effective_weight, 'unlabeled': uinfo,
                'supervised_grad_norm': supervised_norm, 'unlabeled_grad_norm': unorm,
                'weighted_unlabeled_grad_norm': effective_weight * unorm if unorm is not None else None, 'grad_norm': norm,
                'learning_rate': lr}
            if use_lora:
                verify_lora_input(replay_rows[steps - 1], row)
                row['semantic_lora_supervised_grad_norm'] = lora_supervised_norm
            if use_scale:
                row['scale_view'] = scale_spec
            if use_hard:
                row.update(nominal_unlabeled_weight=weight,
                           gradient_budget_applied=effective_weight < weight,
                           gradient_ratio=(effective_weight * unorm / supervised_norm) if supervised_norm > 0 else 0.)
            append_json(output / 'steps.jsonl', row)
            epoch_rows.append(row)
            if steps == 1:
                status.update(status='running', updates=steps, epoch=epoch, first_update=row)
                write_json(output / 'status.json', status)
                print(json.dumps({'first_update': row, 'arm': arm}, ensure_ascii=False), flush=True)
        expected = int(ucfg['smoke_steps']) if smoke else len(dataset)
        if len(epoch_rows) != expected:
            raise RuntimeError('Incomplete epoch')
        summary = {'epoch': epoch, 'updates': steps, 'failed_updates': 0, 'learning_rate': lr,
            'semantic_loss': float(np.mean([r['semantic_loss'] for r in epoch_rows])),
            'unlabeled_loss': float(np.mean([r['unlabeled_loss'] for r in epoch_rows])),
            'unlabeled_weight': weight, 'visited_sources': len(visited), 'effective_sources': len(effective),
            'accepted_pearlite_pixels': accepted_total[0], 'accepted_ferrite_pixels': accepted_total[1],
            'elapsed_seconds': time.time() - started, 'peak_cuda_mib': torch.cuda.max_memory_allocated() / 1024**2}
        if use_scale:
            summary['scale_views'] = {stream: {mode: sum(
                (r if stream == 'supervised' else r['unlabeled'])['scale_view']['mode'] == mode for r in epoch_rows)
                for mode in ('full', 'local')} for stream in ('supervised', 'unlabeled')}
        if candidate == 'prior' and unlabeled is not None:
            active = [r['unlabeled'] for r in epoch_rows]
            summary['gray_prior'] = {
                'accepted_fraction': sum(r['accepted_pixels'] for r in active) / max(1, sum(r['valid_pixels'] for r in active)),
                'loss_mode': config['semantic_gray'].get('loss_mode', 'soft_bce'),
                'maximum_normalization_gain': config['semantic_gray'].get('maximum_normalization_gain', 1.),
                'zero_active_draws': sum(r['active_pixels'] == 0 for r in active),
                'normalization_gain_mean': {label: float(np.mean([
                    r['class_normalization'][label]['gain'] for r in active
                    if r['class_normalization'][label]['accepted']]))
                    if any(r['class_normalization'][label]['accepted'] for r in active) else 0.
                    for label in ('ferrite', 'pearlite')},
                'active_pixels': sum(r['active_pixels'] for r in active),
                'satisfied_pixels': sum(r['satisfied_pixels'] for r in active),
                'active_fraction': sum(r['active_pixels'] for r in active) / max(1, sum(r['accepted_pixels'] for r in active)),
                'active_ferrite': sum(r['active_ferrite'] for r in active),
                'active_pearlite': sum(r['active_pearlite'] for r in active),
                'confidence_only_active_pixels': sum(r['confidence_only_active_pixels'] for r in active),
                'class_disagreement': sum(r['class_disagreement'] for r in active),
                'prior_f_student_p': sum(r['prior_f_student_p'] for r in active),
                'prior_p_student_f': sum(r['prior_p_student_f'] for r in active),
                'mean_probability_gap': sum(r['prior_student_abs_gap'] * r['accepted_pixels'] for r in active) / max(1, sum(r['accepted_pixels'] for r in active)),
                'known_gt_excluded_pixels': sum(r['known_gt_excluded_pixels'] for r in active),
                'empty_target_draws': sum(r['accepted_pixels'] == 0 for r in active)}
            if use_hard:
                selections = [r['hard_views'] for r in active]
                summary['hard_views'] = {
                    'selected_modes': {mode: sum(r['selected_mode'] == mode for r in selections)
                                       for mode in ('base', 'global', 'local')},
                    'forced_base_draws': sum(r['keep_base_draw'] for r in selections),
                    'gradient_budget_steps': sum(r['gradient_budget_applied'] for r in epoch_rows),
                    'actual_weight_mean': float(np.mean([r['unlabeled_weight'] for r in epoch_rows])),
                    'gradient_ratio_mean': float(np.mean([r['gradient_ratio'] for r in epoch_rows])),
                    'gradient_ratio_max': max(r['gradient_ratio'] for r in epoch_rows),
                    'base_region_conflicts': sum(r['views'][0]['diagnostic']['disagree_regions'] for r in selections),
                    'selected_region_conflicts': sum(r['views'][r['selected_index']]['diagnostic']['disagree_regions'] for r in selections)}
        append_json(output / 'epochs.jsonl', summary)
        print(json.dumps({'arm': arm, **summary}), flush=True)
        if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == epochs:
            monitor(model, restorer, config, output, epoch)
            if use_scale:
                scale_monitor(model, restorer, dataset, config, output, epoch)
            if use_hard:
                hard_monitor(model, restorer, unlabeled, config, output, epoch)
        extra = {'arm': arm, 'updates': steps, 'failed_updates': 0, 'smoke': smoke,
                 'data': data_info, 'optimizer': optimizer.state_dict()}
        if not smoke:
            save_backend(model, config, metadata, output / 'last.pt', epoch, extra=extra)
        if epoch == epochs:
            save_backend(model, config, metadata, output / f'epoch_{epoch:03d}.pt', epoch,
                extra={k: v for k, v in extra.items() if k != 'optimizer'})
        status.update(status='running', **summary)
        write_json(output / 'status.json', status)
    if shared_frozen_digest(model) != frozen_initial or tensor_digest(restorer.state_dict().items()) != restoration_initial:
        raise RuntimeError('Frozen backend or D5a changed')
    if teacher is not None and tensor_digest(teacher.state_dict().items()) != teacher_initial:
        raise RuntimeError('Fixed teacher changed')
    if unlabeled is not None:
        expected_visited = min(steps, ucfg['expected_sources'])
        if len(visited) != expected_visited or min(accepted_total) <= 0:
            raise RuntimeError('Unlabeled coverage incomplete or one class has no reliable supervision')
    deltas = {name: float((p.detach().cpu() - before_head[name]).abs().max())
              for name, p in model.semantic_decoder.named_parameters()}
    if max(deltas.values()) <= 0:
        raise RuntimeError('Semantic head did not update')
    if use_lora:
        lora_delta = max(float((value.detach().cpu() - before_lora[name]).abs().max())
                         for name, value in model.semantic_lora.state_dict().items())
        if (lora_delta == 0) != freeze_semantic_lora:
            raise RuntimeError('Independent semantic LoRA update/freeze contract failed')
        status.update(semantic_lora_max_delta=lora_delta, semantic_lora_changed=lora_delta > 0,
                      gray4_input_replay_passed=True, gray4_replayed_steps=steps)
    model.eval()
    with torch.no_grad():
        expected = {key: value.cpu() for key, value in model(probe).items()}
    if not torch.equal(expected['affinity_logits'], initial_affinity):
        raise RuntimeError('Frozen affinity prediction changed')
    final = output / f'epoch_{epochs:03d}.pt'
    del optimizer, parameters, before_head, model, teacher
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(final, config, device)
    with torch.no_grad():
        actual = reloaded(probe)
    differences = {k: float((actual[k].cpu() - expected[k]).abs().max()) for k in expected}
    if max(differences.values()) > 1e-6:
        raise RuntimeError(f'Checkpoint reload changed outputs: {differences}')
    status.update(status='completed', frozen_weights_unchanged=True, d5a_unchanged=True,
        fixed_teacher_unchanged=True, frozen_affinity_output_equal=True, semantic_weights_changed=True,
        strict_reload_passed=True, reload_logit_max_delta=differences,
        semantic_max_delta=max(deltas.values()), final_checkpoint=str(final),
        visited_source_names=sorted(visited), effective_source_names=sorted(effective))
    write_json(output / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_consistency.yaml')
    parser.add_argument('--arm', choices=('control', 'consistency', 'prior'), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--zero-weight', action='store_true')
    parser.add_argument('--freeze-semantic-lora', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(project_path(config, args.output_dir))
    try:
        train(config, args.arm, output, args.smoke, args.zero_weight, args.freeze_semantic_lora)
    except BaseException as error:
        if output.exists():
            write_json(output / 'failure.json', {'error': repr(error), 'arm': args.arm})
        raise


if __name__ == '__main__':
    main()
