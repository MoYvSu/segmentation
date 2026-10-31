# -*- coding: utf-8 -*-
"""原 RGB 上下文补边：旧真实热图配对与损失不变，独立随机初始化五轮。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from models.boundary_context import FORMAT, BoundaryContextNet, load_boundary_context
from models.boundary_restore import boundary_restore_loss, region_masks, state_sha256
from train_boundary_restore import (
    canonical_sha, capture_rng, file_sha, learning_rate, reconcile_log, restore_rng,
    run_monitor, write_json, _optimizer, validate_recipe as validate_real_recipe,
)
from utils.config import load_config, project_path

DATA_MODE = 'real_n0_rgb_context'
STRUCTURE_MODE = 'real_n0_rgb_structure'
SUPERVISION_VERSION = 'boundary_structure_v1'


def deterministic_flags():
    return dict(enabled=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


@contextmanager
def deterministic_resume_probe():
    """只让续训等价核验采用确定性算法；不改变正式训练及TF32配置。"""
    original = deterministic_flags()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(original['enabled'], warn_only=original['warn_only'])


def validate_recipe(config):
    cfg = config['boundary_restore']
    if cfg.get('data_mode') not in (DATA_MODE, STRUCTURE_MODE):
        raise ValueError('RGB context requires explicit real_n0_rgb_context or real_n0_rgb_structure data_mode')
    compatible = deepcopy(config)
    compatible['boundary_restore']['data_mode'] = 'real_n0_heatmaps'
    validate_real_recipe(compatible)
    if not cfg.get('shared_e0_checkpoint'):
        raise ValueError('A verified common random epoch-zero checkpoint is required')
    if cfg.get('context_normalization') != 'uint8_RGB_div255':
        raise ValueError('Original RGB context must use uint8_RGB_div255')
    if cfg.get('micro_batch_size') != 4:
        raise ValueError('This controlled run uses microbatch4/effective batch4')
    if cfg.get('data_mode') == STRUCTURE_MODE:
        from data.boundary_structure import validated_structure_geometry
        from models.boundary_structure_loss import validated_structure_loss_recipe
        validated_structure_geometry(cfg.get('structure'))
        validated_structure_loss_recipe(cfg.get('structure'))
    return cfg


def build_dataset(config, seed_offset=0):
    from data.boundary_context import BoundaryContextDataset
    cfg = config['boundary_restore']
    mode = cfg.get('data_mode')
    if mode not in (DATA_MODE, STRUCTURE_MODE):
        raise ValueError('Wrong context dataset mode')
    factory, extra = BoundaryContextDataset, {}
    if mode == STRUCTURE_MODE:
        from data.boundary_structure import BoundaryStructureDataset
        factory, extra = BoundaryStructureDataset, dict(structure_recipe=deepcopy(cfg['structure']))
    return factory(project_path(config, cfg['manifest']),
        real_manifest_path=project_path(config, cfg['real_manifest']), recipe=deepcopy(cfg['real_recipe']),
        crop_size=int(cfg['crop_size']), views_per_source=int(cfg['views_per_source']),
        seed=int(cfg['seed'])+int(seed_offset), identity_probability=float(cfg['identity_probability']),
        gap_probability=float(cfg['gap_probability']), noise_probability=float(cfg['noise_probability']),
        valid_margin=int(cfg['valid_margin']), expected_sources=int(cfg['expected_sources']), **extra)


def collate_samples(samples):
    from data.boundary_context import collate_boundary_context
    has_structure = ['structure_paths' in sample for sample in samples]
    if any(has_structure) and not all(has_structure):
        raise ValueError('Cannot mix context and structural supervision in one batch')
    if all(has_structure):
        from data.boundary_structure import collate_boundary_structure
        batch = collate_boundary_structure(samples)
    else:
        batch = collate_boundary_context(samples)
    batch['views'] = [{key:row[key] for key in ('source', 'meta')} for row in samples]
    return batch


def build_model(config, device='cpu'):
    cfg = config['boundary_restore']
    model = BoundaryContextNet(**cfg['model'], seed=int(cfg['seed']))
    model.initialize_from_e0(project_path(config, cfg['shared_e0_checkpoint']))
    return model.to(device)


def build_identity(model, dataset, config):
    cfg = config['boundary_restore']
    mode = cfg.get('data_mode')
    if mode not in (DATA_MODE, STRUCTURE_MODE) or dataset.identity.get('data_mode') != mode:
        raise ValueError('Context dataset provenance differs')
    if (model.initialization_receipt.get('source') != 'sealed_old_random_e0'
        or model.initialization_receipt.get('checkpoint_sha256') !=
            file_sha(project_path(config, cfg['shared_e0_checkpoint']))):
        raise ValueError('Missing checked common epoch-zero initialization')
    root = Path(__file__).resolve().parent
    sources = ('train_boundary_context.py', 'models/boundary_context.py', 'data/boundary_context.py',
        'tools/boundary_context_monitor.py', 'train_boundary_restore.py', 'models/boundary_restore.py',
        'data/boundary_restore.py', 'data/boundary_real.py', 'data/dino_semantic.py',
        'models/dino_semantic.py', 'tools/boundary_restore_monitor.py',
        'tools/boundary_restore_readout.py', 'tools/semantic_dino_monitor.py', 'utils/config.py')
    keys = ('seed', 'crop_size', 'views_per_source', 'expected_sources', 'updates_per_epoch',
        'effective_batch_size', 'micro_batch_size', 'schedule_epochs', 'precision', 'learning_rate',
        'minimum_lr_ratio', 'weight_decay', 'adam_eps', 'grad_clip', 'identity_probability',
        'gap_probability', 'noise_probability', 'valid_margin', 'loss', 'monitor_epochs',
        'real_recipe', 'context_normalization')
    settings = cfg.get('monitor', {})
    asset_keys = ('control_manifest', 'additional_control_manifest', 'roi_spec')
    assets = {key:file_sha(project_path(config, settings[key])) for key in asset_keys if settings.get(key)}
    if mode == STRUCTURE_MODE:
        sources += ('data/boundary_structure.py', 'data/boundary_correct.py', 'models/boundary_structure_loss.py')
        keys += ('structure',)
    identity = dict(format=FORMAT, identity_version=1, data_mode=mode,
        architecture=deepcopy(model.architecture), data=deepcopy(dataset.identity),
        initial_state_sha256=model.initial_state_sha256,
        shared_initial_state_sha256=model.shared_initial_state_sha256,
        initialization_receipt=deepcopy(model.initialization_receipt),
        shared_e0_checkpoint_sha256=file_sha(project_path(config, cfg['shared_e0_checkpoint'])),
        recipe={key:deepcopy(cfg[key]) for key in keys},
        runtime_sources={name:file_sha(root/name) for name in sources},
        monitor_recipe={key:deepcopy(value) for key,value in settings.items() if key not in asset_keys},
        monitor_asset_sha256=assets,
        dataset_manifest_sha256=file_sha(project_path(config, cfg['manifest'])),
        real_manifest_sha256=file_sha(project_path(config, cfg['real_manifest'])),
        trained_checkpoint_inherited=False, test_gt_used=False, deployment_replaced=False,
        experiment_factor='aligned_original_RGB_context_only; original_heat_pairs_and_loss_unchanged')
    if mode == STRUCTURE_MODE:
        identity.update(supervision_version=SUPERVISION_VERSION,
            experiment_factor='trusted_instance_separation_in_legal_corridors; all_other_training_conditions_fixed',
            model_format_unchanged=True, existing_control_reused=True)
    return identity


def save_checkpoint(path, model, optimizer, identity, step, cfg, monitors):
    path = Path(path)
    payload = dict(format=FORMAT, architecture=model.architecture, model_state_dict=model.state_dict(),
        model_sha256=state_sha256(model), initial_state_sha256=model.initial_state_sha256,
        shared_initial_state_sha256=model.shared_initial_state_sha256,
        initialization_receipt=deepcopy(model.initialization_receipt),
        identity=identity, identity_sha256=canonical_sha(identity), optimizer=optimizer.state_dict(),
        rng=capture_rng(), global_step=int(step), epoch=step//int(cfg['updates_per_epoch']),
        step_in_epoch=step%int(cfg['updates_per_epoch']), monitor_epochs=list(monitors),
        scheduler=dict(kind='cosine_global_step', next_step=step,
                       total_updates=int(cfg['schedule_epochs'])*int(cfg['updates_per_epoch'])),
        stage_epochs=int(cfg['epochs']))
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume(path, model, optimizer, identity, cfg):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('format') != FORMAT or saved.get('identity') != identity
        or saved.get('identity_sha256') != canonical_sha(identity)
        or saved.get('architecture') != model.architecture
        or saved.get('initial_state_sha256') != model.initial_state_sha256
        or saved.get('shared_initial_state_sha256') != model.shared_initial_state_sha256
        or saved.get('initialization_receipt') != model.initialization_receipt):
        raise ValueError('Resume source/data/recipe/common initialization identity differs')
    step = saved.get('global_step')
    updates = int(cfg['updates_per_epoch'])
    if type(step) is not int or not 0 <= step <= int(cfg['epochs'])*updates:
        raise ValueError('Resume step outside requested stage')
    if (saved.get('epoch') != step//updates or saved.get('step_in_epoch') != step%updates
        or saved.get('scheduler') != dict(kind='cosine_global_step', next_step=step,
                                         total_updates=int(cfg['schedule_epochs'])*updates)):
        raise ValueError('Resume optimizer step and cosine schedule disagree')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != saved['model_sha256']:
        raise ValueError('Restored model state digest differs')
    optimizer_state = saved['optimizer']
    states = optimizer_state.get('state', {})
    parameter_ids = [value for group in optimizer_state.get('param_groups', []) for value in group['params']]
    if (len(parameter_ids) != len(list(model.parameters()))
        or len(set(parameter_ids)) != len(parameter_ids)
        or (step == 0 and states)
        or (step > 0 and set(states) != set(parameter_ids))):
        raise ValueError('Resume optimizer parameter coverage differs from global step')
    for state in states.values():
        optimizer_step = state.get('step')
        if (not torch.is_tensor(optimizer_step) or optimizer_step.numel() != 1
            or not bool(torch.isfinite(optimizer_step)) or float(optimizer_step) != step):
            raise ValueError('Resume optimizer internal step differs from global step')
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    return step, list(saved['monitor_epochs'])


def structure_active(cfg):
    return cfg.get('data_mode') == STRUCTURE_MODE and float(cfg['structure']['weight']) > 0


def base_supervision_batch(batch, cfg):
    """只在合法补边自由区释放旧 N0 逐像素目标；不原地修改旧配对。"""
    if not structure_active(cfg):
        return batch
    free = batch.get('structure_free_mask')
    if (not torch.is_tensor(free) or free.shape != batch['valid'].shape
        or free.dtype != torch.bool):
        raise ValueError('Structural free mask must be a matching boolean B1HW tensor')
    if bool((free & (batch['target'] >= float(cfg['structure']['teacher_release_threshold']))).any()):
        raise ValueError('Structural free mask cannot release original strong N0 responses')
    if bool((free & ~batch['valid'].bool()).any()):
        raise ValueError('Structural free mask cannot release unknown or padded pixels')
    if not bool(free.any()):
        return batch
    result = dict(batch)
    valid = batch['valid'].bool() & ~free
    result['valid'] = valid.to(batch['valid'].dtype)
    for key in ('gap_mask', 'noise_mask'):
        result[key] = (batch[key].bool() & valid).to(batch[key].dtype)
    return result


def combined_loss(prediction, batch, cfg, *, normalizers=None, structure_denominators=None):
    """旧配对目标与独立结构关系；权重为零时执行完全相同的旧损失路径。"""
    base_batch = base_supervision_batch(batch, cfg)
    loss, stats = boundary_restore_loss(prediction, base_batch, normalizers=normalizers, **cfg['loss'])
    if not structure_active(cfg):
        return loss, stats
    from models.boundary_structure_loss import structural_boundary_loss
    auxiliary, structural_stats = structural_boundary_loss(prediction, batch, cfg['structure'],
                                                           normalizers=structure_denominators)
    total = loss + auxiliary
    stats = dict(stats, loss=float(total.detach()), base_loss=float(loss.detach()), structure=structural_stats)
    return total, stats


def probe_structural_gradient(model, batches, cfg):
    """独立核验附加结构项有真实梯度；不写参数、不执行优化器更新。"""
    if not structure_active(cfg):
        return dict(active=False, optimizer_updates=0)
    from models.boundary_structure_loss import structural_boundary_loss, structure_normalizers
    denominators = structure_normalizers(batches, cfg['structure'])
    if denominators['points'] == 0:
        raise RuntimeError('Structural smoke requires at least one nonempty legal path')
    device = next(model.parameters()).device
    before, training = state_sha256(model), model.training
    squared_norm, losses = 0., []
    try:
        model.eval()
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            for batch in batches:
                moved = {key:value.to(device) if torch.is_tensor(value) else value for key,value in batch.items()}
                prediction = model(moved['input'], moved['context_rgb'])
                auxiliary, stats = structural_boundary_loss(prediction, moved, cfg['structure'],
                                                            normalizers=denominators)
                gradients = torch.autograd.grad(auxiliary, tuple(model.parameters()), allow_unused=True)
                for gradient in gradients:
                    if gradient is not None:
                        if not bool(torch.isfinite(gradient).all()):
                            raise FloatingPointError('Non-finite structural-only smoke gradient')
                        squared_norm += float(gradient.double().square().sum())
                losses.append(stats)
    finally:
        model.train(training)
    if state_sha256(model) != before:
        raise RuntimeError('Structural gradient probe changed model weights')
    if not squared_norm > 0:
        raise RuntimeError('Structural supervision did not produce a nonzero model gradient')
    return dict(active=True, optimizer_updates=0, normalizers=denominators,
                gradient_norm=squared_norm**.5, model_unchanged=True, rows=losses)


def optimization_step(model, optimizer, batches, cfg, step):
    """有效 batch 统一分母；结构项不改变模型、优化器或原配对随机流。"""
    count = sum(int(batch['input'].shape[0]) for batch in batches)
    if count != int(cfg['effective_batch_size']):
        raise ValueError('Update does not contain the declared effective batch')
    loss_cfg = cfg['loss']
    names = ('valid', 'gap', 'noise', 'preserve')
    base_batches = [base_supervision_batch(batch, cfg) for batch in batches]
    normalizers = {name:sum(int(region_masks(b, float(loss_cfg['preserve_threshold']))[name].sum())
                            for b in base_batches) for name in names}
    structure_denominators = None
    if structure_active(cfg):
        from models.boundary_structure_loss import structure_normalizers
        structure_denominators = structure_normalizers(batches, cfg['structure'])
    device = next(model.parameters()).device
    model.train(); optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    rows = []
    for batch in batches:
        batch = {key:value.to(device) if torch.is_tensor(value) else value for key,value in batch.items()}
        with torch.autocast(device_type=device.type, enabled=False):
            prediction = model(batch['input'], batch['context_rgb'])
            loss, stats = combined_loss(prediction, batch, cfg, normalizers=normalizers,
                                        structure_denominators=structure_denominators)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite RGB context restoration loss')
        loss.backward(); rows.append(stats)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip']))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Non-finite RGB context gradients')
    optimizer.step()
    regions = {}
    for name in rows[0]['regions']:
        pixels = sum(row['regions'][name]['pixels'] for row in rows)
        error = sum(row['regions'][name]['absolute_error'] for row in rows)
        regions[name] = dict(pixels=pixels, absolute_error=error, mae=error/pixels if pixels else None)
    result = dict(global_step=step+1, loss=sum(row['loss'] for row in rows), regions=regions,
        region_denominators=normalizers, microbatches=rows, learning_rate=lr,
        grad_norm_before_clip=float(norm), nonzero_gradient=bool(float(norm)>0),
        effective_batch_size=count, identity_samples=sum(row['identity_samples'] for row in rows),
        unknown_pixels=sum(row['unknown_pixels'] for row in rows), context_input='aligned_original_RGB_div255')
    if structure_active(cfg):
        result.update(base_loss=sum(row['base_loss'] for row in rows),
            structure_loss=sum(row['structure']['loss'] for row in rows),
            structure_denominators=structure_denominators,
            structure_path_count=sum(row['structure']['path_count'] for row in rows),
            structure_point_count=sum(row['structure']['point_count'] for row in rows),
            structure_free_pixels=sum(row['structure']['free_pixels'] for row in rows),
            released_original_target_pixels=sum(int(batch['structure_free_mask'].sum()) for batch in batches))
    return result


def _batches(dataset, draw, cfg):
    count, micro = int(cfg['effective_batch_size']), int(cfg['micro_batch_size'])
    samples = [dataset[i] for i in range(draw, draw+count)]
    return [collate_samples(samples[i:i+micro]) for i in range(0, count, micro)]


def train(model, dataset, config, output, callback, resume=None):
    cfg, output = config['boundary_restore'], Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise ValueError('Fresh training requires an empty output directory')
    if resume is None and state_sha256(model) != model.initial_state_sha256:
        raise ValueError('Fresh training requires exact common random e0; trained weights are forbidden')
    if callback is None:
        raise ValueError('Fixed process monitor callback is required')
    identity = build_identity(model, dataset, config)
    optimizer = _optimizer(model, cfg)
    step, monitors = (0, []) if resume is None else load_resume(resume, model, optimizer, identity, cfg)
    if resume:
        reconcile_log(output, step)
    dataset.verify_sources()
    updates, batch = int(cfg['updates_per_epoch']), int(cfg['effective_batch_size'])
    if len(dataset) != updates*batch:
        raise ValueError('Source sampler length does not match the exact update budget')
    write_json(output/'identity.json', identity)
    write_json(output/'resolved_config.json', config)
    write_json(output/'initialization.json', dict(random_initialization=True,
        shared_initial_state_sha256=model.shared_initial_state_sha256,
        initial_state_sha256=model.initial_state_sha256, initialization_receipt=model.initialization_receipt,
        parameters=sum(p.numel() for p in model.parameters()), resume=resume is not None,
        trained_checkpoint_inherited=False, added_RGB_weights_zero=True, residual_output_zero=True))
    def save(path):
        save_checkpoint(path, model, optimizer, identity, step, cfg, monitors)
    def render(epoch):
        if epoch in cfg['monitor_epochs'] and epoch not in monitors:
            run_monitor(model, config, output, epoch, next(model.parameters()).device, callback)
            monitors.append(epoch); save(output/'last.pt')
    if step == 0:
        save(output/'epoch_000.pt'); render(0)
    elif step%updates == 0:
        render(step//updates)
    active_epoch = None
    while step < int(cfg['epochs'])*updates:
        epoch = step//updates+1
        if epoch != active_epoch:
            dataset.set_epoch(epoch); active_epoch = epoch
            write_json(output/f'sampling_e{epoch:03d}.json', dict(receipt=dataset.plan_receipt(), plan=dataset.plan))
        batches = _batches(dataset, (step%updates)*batch, cfg)
        started = time.monotonic()
        row = optimization_step(model, optimizer, batches, cfg, step)
        row.update(epoch=epoch, update_in_epoch=step%updates+1,
                   views=[view for b in batches for view in b['views']], step_seconds=time.monotonic()-started)
        step += 1
        with (output/'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        if step%int(cfg['checkpoint_every_updates']) == 0 or step%updates == 0:
            save(output/'last.pt')
        if step%updates == 0:
            dataset.verify_sources(); save(output/f'epoch_{epoch:03d}.pt'); render(epoch)
            receipt = dict(epoch=epoch, global_step=step, model_sha256=state_sha256(model))
            write_json(output/f'epoch_{epoch:03d}.json', receipt); print(json.dumps(receipt), flush=True)
    save(output/'last.pt')
    write_json(output/'complete.json', dict(format=FORMAT, complete=True, epochs=int(cfg['epochs']),
        optimizer_updates=step, windows=step*batch, monitor_epochs=monitors,
        model_changed=state_sha256(model)!=model.initial_state_sha256, model_sha256=state_sha256(model),
        identity_sha256=canonical_sha(identity), official_score=None, convergence_claimed=False, baseline_replaced=False))
    return output


def smoke(model, dataset, config, output, callback):
    """两更新、首步完整读出、严格重载及实际下一步续训一致性；不用于正式初始化。"""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Smoke output must be empty')
    cfg, device = config['boundary_restore'], next(model.parameters()).device
    dataset.verify_sources(); dataset.set_epoch(1)
    identity = build_identity(model, dataset, config)
    optimizer = _optimizer(model, cfg)
    sample = collate_samples([dataset[0]])
    heat, rgb = sample['input'].to(device), sample['context_rgb'].to(device)
    model.eval()
    with torch.no_grad():
        if not torch.equal(model(heat, rgb), heat) or torch.count_nonzero(model.residual(heat, rgb)):
            raise RuntimeError('Common random initialization is not exact boundary identity')
    run_monitor(model, config, output, 0, device, callback)
    structure_probe = None
    if structure_active(cfg):
        structure_probe = probe_structural_gradient(model,
            _batches(dataset, 0, cfg)+_batches(dataset, int(cfg['effective_batch_size']), cfg), cfg)
    rows = [optimization_step(model, optimizer, _batches(dataset, i*int(cfg['effective_batch_size']), cfg), cfg, i)
            for i in range(2)]
    if not any(row['nonzero_gradient'] for row in rows):
        raise RuntimeError('Smoke did not exercise a nonzero training gradient')
    path = output/'smoke.pt'
    save_checkpoint(path, model, optimizer, identity, 2, cfg, [0])
    model.eval()
    with torch.no_grad():
        before = model(heat, rgb).cpu()
        after = load_boundary_context(path, device)(heat, rgb).cpu()
    if not torch.equal(before, after):
        raise RuntimeError('Strict reload changed actual output')
    resumed = build_model(config, device)
    resumed_optimizer = _optimizer(resumed, cfg)
    saved_step, _ = load_resume(path, resumed, resumed_optimizer, identity, cfg)
    if saved_step != 2 or state_sha256(resumed)!=state_sha256(model):
        raise RuntimeError('Checkpoint could not resume exactly')
    # 独立副本的第3步仅用于续训验收，不把它计入两步smoke或正式预算。
    probe_batches = _batches(dataset, 2*int(cfg['effective_batch_size']), cfg)
    saved_rng = capture_rng()
    probe_flags_before = deterministic_flags()
    # CUDA双线性插值/卷积反传可非逐位确定；核验隔离此噪声，正式两步与五轮不变。
    with deterministic_resume_probe():
        optimization_step(model, optimizer, probe_batches, cfg, 2)
        restore_rng(saved_rng)
        optimization_step(resumed, resumed_optimizer, probe_batches, cfg, 2)
        next_step_exact = state_sha256(model)==state_sha256(resumed)
    probe_flags_after = deterministic_flags()
    if probe_flags_after != probe_flags_before:
        raise RuntimeError('Resume probe leaked deterministic flags into training')
    if not next_step_exact:
        raise RuntimeError('Resumed next optimizer update differs from uninterrupted update')
    load_resume(path, model, optimizer, identity, cfg)
    run_monitor(model, config, output, 1, device, callback)
    report = dict(format='boundary_context_smoke_v1', complete=True, optimizer_updates=2,
        resume_probe_updates=1, rows=rows, initial_identity_exact=True,
        strict_reload_prediction_equal=True, optimizer_resume_next_step_exact=next_step_exact,
        probe_deterministic=True, probe_flags_restored=True,
        probe_flags_before=probe_flags_before, probe_flags_after=probe_flags_after,
        formal_training_complete=False, shared_initial_state_sha256=model.shared_initial_state_sha256,
        model_sha256=state_sha256(model), identity_sha256=canonical_sha(identity))
    if structure_probe is not None:
        report.update(supervision_version=SUPERVISION_VERSION, structural_gradient_probe=structure_probe)
    write_json(output/'smoke.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/boundary_context.yaml')
    parser.add_argument('--output'); parser.add_argument('--resume'); parser.add_argument('--device', default='cuda')
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ('preflight', 'smoke', 'run'):
        modes.add_argument('--'+name, action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config); cfg = validate_recipe(config)
    random.seed(int(cfg['seed'])); np.random.seed(int(cfg['seed'])); torch.manual_seed(int(cfg['seed']))
    torch.set_num_threads(int(cfg['torch_threads'])); cv2.setNumThreads(int(cfg['opencv_threads']))
    dataset = build_dataset(config); model = build_model(config)
    if args.preflight:
        if args.resume:
            raise ValueError('Preflight does not load a trained checkpoint')
        dataset.verify_sources()
        batch = collate_samples([dataset[0]])
        with torch.no_grad():
            prediction = model(batch['input'], batch['context_rgb'])
            loss, stats = combined_loss(prediction, batch, cfg)
        if not torch.equal(prediction, batch['input']):
            raise RuntimeError('CPU e0 identity preflight failed')
        print(json.dumps(dict(format='boundary_context_preflight_v1', training=False, gpu_used=False,
            identity=build_identity(model, dataset, config), plan=dataset.plan_receipt(),
            shape=list(prediction.shape), context_shape=list(batch['context_rgb'].shape),
            e0_identity_exact=True, sample_loss=float(loss), sample_coverage=stats,
            parameters=sum(p.numel() for p in model.parameters()),
            planned_optimizer_updates=int(cfg['epochs'])*int(cfg['updates_per_epoch'])), ensure_ascii=False, indent=2))
        return
    from tools.boundary_context_monitor import monitor
    output = project_path(config, args.output or cfg['output_dir'])
    model.to(args.device)
    if args.smoke:
        if args.resume:
            raise ValueError('Smoke always starts from verified common random e0')
        smoke(model, dataset, config, output, monitor)
    else:
        train(model, dataset, config, output, monitor, args.resume)


if __name__ == '__main__':
    main()
