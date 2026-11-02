# -*- coding: utf-8 -*-
"""显式选择合成GT或真实N0热图的补边短训；无测试GT、无基线替换。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from models.boundary_restore import (
    FORMAT, BoundaryRestoreNet, boundary_restore_loss, load_boundary_restorer, region_masks, state_sha256,
)
from utils.config import load_config, project_path


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(value):
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda']:
        if not torch.cuda.is_available() or len(value['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA random-state topology differs')
        torch.cuda.set_rng_state_all([state.cpu() for state in value['cuda']])


def data_mode(config):
    """旧配置不新增隐式行为；真实热图配方必须明确选择。"""
    mode = config['boundary_restore'].get('data_mode', 'synthetic_gt')
    if mode not in ('synthetic_gt', 'real_n0_heatmaps'):
        raise ValueError('Unknown boundary restoration data_mode: '+repr(mode))
    return mode


def build_dataset(config, seed_offset=0):
    """训练与固定配对过程图共用工厂，避免监控误读另一种监督。"""
    cfg = config['boundary_restore']
    mode = data_mode(config)
    common = dict(crop_size=int(cfg.get('crop_size', 512)),
        views_per_source=int(cfg.get('views_per_source', 8)),
        seed=int(cfg.get('seed', 20261025))+int(seed_offset),
        identity_probability=float(cfg.get('identity_probability', .2)),
        gap_probability=float(cfg.get('gap_probability', .6)),
        noise_probability=float(cfg.get('noise_probability', .2)),
        valid_margin=int(cfg.get('valid_margin', 8)), expected_sources=int(cfg.get('expected_sources', 32)))
    manifest = project_path(config, cfg['manifest'])
    if mode == 'synthetic_gt':
        from data.boundary_restore import BoundaryRestoreDataset
        return BoundaryRestoreDataset(manifest, target_sigma=float(cfg.get('target_sigma', 2.)), **common)
    from data.boundary_real import BoundaryRealDataset
    if not cfg.get('real_manifest') or not isinstance(cfg.get('real_recipe'), dict):
        raise ValueError('real_n0_heatmaps requires real_manifest and an explicit real_recipe')
    return BoundaryRealDataset(manifest, real_manifest_path=project_path(config, cfg['real_manifest']),
                               recipe=deepcopy(cfg['real_recipe']), **common)


def collate_samples(samples):
    tensor_keys = ('input', 'target', 'valid', 'gap_mask', 'noise_mask', 'identity')
    return {**{k: torch.stack([row[k] for row in samples]) for k in tensor_keys},
            'views': [{k: row[k] for k in ('source', 'meta')} for row in samples]}


def validate_recipe(config):
    cfg = config['boundary_restore']
    mode = data_mode(config)
    if mode == 'real_n0_heatmaps' and (not cfg.get('real_manifest') or not isinstance(cfg.get('real_recipe'), dict)):
        raise ValueError('real_n0_heatmaps requires real_manifest and an explicit real_recipe')
    for key, value in dict(crop_size=512, views_per_source=8, expected_sources=32,
                           updates_per_epoch=64, effective_batch_size=4, schedule_epochs=20,
                           precision='float32').items():
        if cfg.get(key) != value:
            raise ValueError(f'This controlled boundary experiment fixes {key}={value}')
    if cfg['micro_batch_size'] not in (1, 2, 4) or cfg['effective_batch_size'] % cfg['micro_batch_size']:
        raise ValueError('Microbatch must divide effective batch 4')
    if not 1 <= int(cfg['epochs']) <= cfg['schedule_epochs']:
        raise ValueError('Stage epochs outside the fixed 20-epoch schedule')
    if cfg['monitor_epochs'] != [0, 1, 3, 5]:
        raise ValueError('Fixed e0/e1/e3/e5 process monitors are required')
    for key in ('learning_rate', 'adam_eps', 'grad_clip'):
        if not 0 < float(cfg[key]) < float('inf'):
            raise ValueError(f'{key} must be positive and finite')
    if not 0 <= float(cfg['weight_decay']) < float('inf') or not 0 <= float(cfg['minimum_lr_ratio']) <= 1:
        raise ValueError('Invalid weight decay or LR floor')
    probs = [float(cfg[k]) for k in ('identity_probability', 'gap_probability', 'noise_probability')]
    if not all(0 <= p <= 1 for p in probs) or not math.isclose(sum(probs), 1., abs_tol=1e-8):
        raise ValueError('Identity/gap/noise are mutually exclusive modes whose probabilities sum to 1')
    return cfg


def learning_rate(step, cfg):
    total = int(cfg['schedule_epochs']) * int(cfg['updates_per_epoch'])
    if not 0 <= step < total:
        raise ValueError('Optimizer step outside fixed cosine schedule')
    floor = float(cfg['minimum_lr_ratio'])
    return float(cfg['learning_rate']) * (floor + (1-floor)*.5*(1+math.cos(math.pi*step/max(1, total-1))))


def build_identity(model, dataset, config):
    cfg = config['boundary_restore']
    mode = data_mode(config)
    if mode == 'real_n0_heatmaps' and dataset.identity.get('data_mode') != mode:
        raise ValueError('Real heatmap configuration requires the matching dataset provenance')
    keys = ('seed', 'crop_size', 'views_per_source', 'expected_sources', 'updates_per_epoch',
            'effective_batch_size', 'micro_batch_size', 'schedule_epochs', 'precision', 'learning_rate',
            'minimum_lr_ratio', 'weight_decay', 'adam_eps', 'grad_clip', 'identity_probability',
            'gap_probability', 'noise_probability', 'valid_margin', 'loss', 'monitor_epochs')
    keys += ('target_sigma',) if mode == 'synthetic_gt' else ('real_recipe',)
    root = Path(__file__).resolve().parent
    source_names = ('models/boundary_restore.py', 'data/boundary_restore.py', 'train_boundary_restore.py',
                    'data/dino_semantic.py', 'models/dino_semantic.py',
                    'tools/boundary_restore_monitor.py', 'tools/boundary_restore_readout.py',
                    'tools/semantic_dino_monitor.py', 'utils/config.py')
    if mode == 'real_n0_heatmaps':
        source_names += ('data/boundary_real.py',)
    monitor = cfg.get('monitor', {})
    assets = {k: file_sha(project_path(config, monitor[k])) for k in ('control_manifest', 'roi_spec') if monitor.get(k)}
    extra = {}
    if mode == 'real_n0_heatmaps':
        extra.update(identity_version=2, data_mode=mode,
                     real_manifest_sha256=file_sha(project_path(config, cfg['real_manifest'])))
    return dict(format=FORMAT, architecture=model.architecture, data=deepcopy(dataset.identity),
                initial_state_sha256=model.initial_state_sha256,
                recipe={key: deepcopy(cfg[key]) for key in keys},
                runtime_sources={name: file_sha(root/name) for name in source_names},
                monitor_recipe={k: v for k, v in monitor.items() if k not in ('control_manifest', 'roi_spec')},
                monitor_asset_sha256=assets, dataset_manifest_sha256=file_sha(project_path(config, cfg['manifest'])),
                initialization='random_hidden_layers_zero_residual_output',
                test_gt_used=False, deployment_replaced=False, scoring_implemented=False, **extra)


def save_checkpoint(path, model, optimizer, identity, step, cfg, monitors):
    payload = dict(format=FORMAT, architecture=model.architecture, model_state_dict=model.state_dict(),
                   model_sha256=state_sha256(model), initial_state_sha256=model.initial_state_sha256,
                   identity=identity, identity_sha256=canonical_sha(identity), optimizer=optimizer.state_dict(),
                   rng=capture_rng(), global_step=int(step), epoch=step//int(cfg['updates_per_epoch']),
                   step_in_epoch=step % int(cfg['updates_per_epoch']), monitor_epochs=list(monitors),
                   scheduler=dict(kind='cosine_global_step', next_step=step,
                                  total_updates=int(cfg['schedule_epochs'])*int(cfg['updates_per_epoch'])),
                   stage_epochs=int(cfg['epochs']))
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_resume(path, model, optimizer, identity, cfg):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved.get('format') != FORMAT or saved.get('identity') != identity
            or saved.get('identity_sha256') != canonical_sha(identity)
            or saved.get('architecture') != model.architecture
            or saved.get('initial_state_sha256') != model.initial_state_sha256):
        raise ValueError('Resume source/data/recipe/random initialization identity differs')
    step = saved.get('global_step')
    if type(step) is not int or not 0 <= step <= int(cfg['epochs'])*int(cfg['updates_per_epoch']):
        raise ValueError('Resume step outside requested stage')
    total = int(cfg['schedule_epochs'])*int(cfg['updates_per_epoch'])
    if (saved.get('epoch') != step//int(cfg['updates_per_epoch'])
            or saved.get('step_in_epoch') != step % int(cfg['updates_per_epoch'])
            or saved.get('scheduler') != dict(kind='cosine_global_step', next_step=step, total_updates=total)):
        raise ValueError('Resume optimizer step and cosine schedule disagree')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    if state_sha256(model) != saved['model_sha256']:
        raise ValueError('Restored model state digest differs')
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    return step, list(saved['monitor_epochs'])


def reconcile_log(output, step):
    path = Path(output)/'steps.jsonl'
    if not path.is_file():
        return
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    values = [json.loads(line)['global_step'] for line in lines]
    if values and (any(type(v) is not int for v in values)
                   or values != list(range(values[0], values[0]+len(values)))):
        raise ValueError('Step log contains duplicate or discontinuous updates')
    if values and values[-1] < step:
        raise ValueError('Existing log is behind checkpoint; resume into an isolated directory')
    tail = [line for value, line in zip(values, lines) if value > step]
    if tail:
        backup = Path(output)/f'steps_after_s{step}_{time.time_ns()}.jsonl'
        with backup.open('x', encoding='utf-8') as stream:
            stream.writelines(tail)
        temp = path.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(line for value, line in zip(values, lines) if value <= step), encoding='utf-8')
        temp.replace(path)
        write_json(Path(output)/'resume_log.json', dict(step=step, preserved_tail=backup.name,
                   replayed_updates=len(tail), duplicate_updates_counted=False))


def run_monitor(model, config, output, epoch, device, callback):
    rng, mode, before = capture_rng(), model.training, state_sha256(model)
    try:
        model.eval()
        with torch.no_grad():
            result = callback(model, config, Path(output), int(epoch), device)
        if state_sha256(model) != before:
            raise RuntimeError('Monitor changed model weights or buffers')
        return result
    finally:
        restore_rng(rng)
        model.train(mode)


def optimization_step(model, optimizer, batches, cfg, step):
    """各区域分母取整个有效 batch，microbatch 切分不改变监督权重。"""
    count = sum(int(b['input'].shape[0]) for b in batches)
    if count != int(cfg['effective_batch_size']):
        raise ValueError('Update does not contain the declared effective batch')
    loss_cfg = cfg['loss']
    threshold = float(loss_cfg['preserve_threshold'])
    names = ('valid', 'gap', 'noise', 'preserve')
    normalizers = {name: sum(int(region_masks(b, threshold)[name].sum()) for b in batches) for name in names}
    device = next(model.parameters()).device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    rows = []
    for batch in batches:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        prediction = model(batch['input'])
        loss, stats = boundary_restore_loss(prediction, batch, normalizers=normalizers, **loss_cfg)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite boundary restoration loss')
        loss.backward()
        rows.append(stats)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['grad_clip']))
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Non-finite restoration gradients')
    # e0 的纯恒等批次可合法产生零梯度；明确记录，不当作训练成功的证据。
    optimizer.step()
    regions = {}
    for name in rows[0]['regions']:
        pixels = sum(r['regions'][name]['pixels'] for r in rows)
        error = sum(r['regions'][name]['absolute_error'] for r in rows)
        regions[name] = dict(pixels=pixels, absolute_error=error, mae=error/pixels if pixels else None)
    return dict(global_step=step+1, loss=sum(r['loss'] for r in rows), regions=regions,
                region_denominators=normalizers, microbatches=rows, learning_rate=lr,
                grad_norm_before_clip=float(norm), nonzero_gradient=bool(float(norm) > 0),
                effective_batch_size=count, identity_samples=sum(r['identity_samples'] for r in rows),
                unknown_pixels=sum(r['unknown_pixels'] for r in rows))


def _batches(dataset, draw, cfg):
    count, micro = int(cfg['effective_batch_size']), int(cfg['micro_batch_size'])
    samples = [dataset[i] for i in range(draw, draw+count)]
    return [collate_samples(samples[i:i+micro]) for i in range(0, count, micro)]


def _optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']),
                            weight_decay=float(cfg['weight_decay']), eps=float(cfg['adam_eps']))


def train(model, dataset, config, output, callback, resume=None):
    cfg, output = config['boundary_restore'], Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise ValueError('Fresh training requires an empty output directory')
    if resume is None and state_sha256(model) != model.initial_state_sha256:
        raise ValueError('Fresh training requires the exact random initial model; previous weights are forbidden')
    if callback is None:
        raise ValueError('Fixed process monitor callback is required')
    device = next(model.parameters()).device
    identity = build_identity(model, dataset, config)
    optimizer = _optimizer(model, cfg)
    step, monitors = (0, []) if resume is None else load_resume(resume, model, optimizer, identity, cfg)
    if resume:
        reconcile_log(output, step)
    dataset.verify_sources()
    updates, batch = int(cfg['updates_per_epoch']), int(cfg['effective_batch_size'])
    if len(dataset) != updates * batch:
        raise ValueError('Source sampler length does not match exact 64-update budget')
    write_json(output/'identity.json', identity)
    write_json(output/'resolved_config.json', config)
    write_json(output/'initialization.json', dict(random_initialization=True, residual_output_zero=True,
               initial_state_sha256=model.initial_state_sha256, parameters=sum(p.numel() for p in model.parameters()),
               resume=resume is not None, pretrained_weights_used=False))
    def save(path):
        save_checkpoint(path, model, optimizer, identity, step, cfg, monitors)
    def monitor(epoch):
        if epoch in cfg['monitor_epochs'] and epoch not in monitors:
            run_monitor(model, config, output, epoch, device, callback)
            monitors.append(epoch)
            save(output/'last.pt')
    if step == 0:
        save(output/'epoch_000.pt')
        monitor(0)
    elif step % updates == 0:
        monitor(step//updates)
    active_epoch = None
    while step < int(cfg['epochs'])*updates:
        epoch = step//updates+1
        if epoch != active_epoch:
            dataset.set_epoch(epoch)
            active_epoch = epoch
            write_json(output/f'sampling_e{epoch:03d}.json', dict(receipt=dataset.plan_receipt(), plan=dataset.plan))
        batches = _batches(dataset, (step % updates)*batch, cfg)
        started = time.monotonic()
        row = optimization_step(model, optimizer, batches, cfg, step)
        row.update(epoch=epoch, update_in_epoch=step % updates+1,
                   views=[v for b in batches for v in b['views']], step_seconds=time.monotonic()-started)
        step += 1
        with (output/'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        if step % int(cfg['checkpoint_every_updates']) == 0 or step % updates == 0:
            save(output/'last.pt')
        if step % updates == 0:
            dataset.verify_sources()
            save(output/f'epoch_{epoch:03d}.pt')
            monitor(epoch)
            receipt = dict(epoch=epoch, global_step=step, model_sha256=state_sha256(model))
            write_json(output/f'epoch_{epoch:03d}.json', receipt)
            print(json.dumps(receipt), flush=True)
    save(output/'last.pt')
    write_json(output/'complete.json', dict(format=FORMAT, complete=True, epochs=int(cfg['epochs']),
               optimizer_updates=step, windows=step*batch, monitor_epochs=monitors,
               model_changed=state_sha256(model) != model.initial_state_sha256,
               model_sha256=state_sha256(model), identity_sha256=canonical_sha(identity),
               official_score=None, convergence_claimed=False, baseline_replaced=False))
    return output


def smoke(model, dataset, config, output, callback):
    """两次真实优化 + 严格重载 + 固定监控；单独目录，不冒充正式训练。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Smoke output must be empty')
    cfg, device = config['boundary_restore'], next(model.parameters()).device
    dataset.verify_sources()
    identity = build_identity(model, dataset, config)
    optimizer = _optimizer(model, cfg)
    sample = dataset[0]['input'][None].to(device)
    model.eval()
    with torch.no_grad():
        if not torch.equal(model(sample), sample) or torch.count_nonzero(model.residual(sample)):
            raise RuntimeError('Random initialization is not exact boundary identity')
    run_monitor(model, config, output, 0, device, callback)
    rows = [optimization_step(model, optimizer, _batches(dataset, i*int(cfg['effective_batch_size']), cfg), cfg, i)
            for i in range(2)]
    if not any(r['nonzero_gradient'] for r in rows):
        raise RuntimeError('Two-step smoke did not exercise a nonzero training gradient')
    path = output/'smoke.pt'
    save_checkpoint(path, model, optimizer, identity, 2, cfg, [0])
    model.eval()
    with torch.no_grad():
        before = model(sample).cpu()
    reloaded = load_boundary_restorer(path, device)
    with torch.no_grad():
        after = reloaded(sample).cpu()
    if not torch.equal(before, after):
        raise RuntimeError('Strict reload changed actual probability output')
    resumed = BoundaryRestoreNet(**cfg['model'], seed=int(cfg['seed'])).to(device)
    resumed_optimizer = _optimizer(resumed, cfg)
    saved_step, _ = load_resume(path, resumed, resumed_optimizer, identity, cfg)
    if saved_step != 2 or state_sha256(resumed) != state_sha256(model):
        raise RuntimeError('Training checkpoint could not resume exactly')
    run_monitor(model, config, output, 1, device, callback)
    report = dict(format='boundary_restore_smoke_v1', complete=True, optimizer_updates=2,
                  rows=rows, initial_identity_exact=True, strict_reload_prediction_equal=True,
                  optimizer_resume_checked=True, formal_training_complete=False,
                  model_sha256=state_sha256(model), identity_sha256=canonical_sha(identity))
    write_json(output/'smoke.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/boundary_restore.yaml')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cfg = validate_recipe(config)
    random.seed(int(cfg['seed'])); np.random.seed(int(cfg['seed'])); torch.manual_seed(int(cfg['seed']))
    torch.set_num_threads(int(cfg['torch_threads'])); cv2.setNumThreads(int(cfg['opencv_threads']))
    dataset = build_dataset(config)
    model = BoundaryRestoreNet(**cfg['model'], seed=int(cfg['seed']))
    if args.preflight:
        if args.resume:
            raise ValueError('Preflight does not load training checkpoints')
        batch = collate_samples([dataset[0]])
        with torch.no_grad():
            prediction = model(batch['input'])
            loss, stats = boundary_restore_loss(prediction, batch, **cfg['loss'])
        if not torch.equal(prediction, batch['input']):
            raise RuntimeError('CPU e0 identity preflight failed')
        print(json.dumps(dict(format='boundary_restore_preflight_v1', training=False, gpu_used=False,
              identity=build_identity(model, dataset, config), plan=dataset.plan_receipt(),
              shape=list(prediction.shape), e0_identity_exact=True, sample_loss=float(loss),
              sample_coverage=stats, parameters=sum(p.numel() for p in model.parameters()),
              planned_optimizer_updates=int(cfg['epochs'])*int(cfg['updates_per_epoch'])), ensure_ascii=False, indent=2))
        return
    from tools.boundary_restore_monitor import monitor
    output = Path(project_path(config, args.output or cfg['output_dir']))
    model.to(args.device)
    if args.smoke:
        if args.resume:
            raise ValueError('Smoke always starts from random initialization')
        smoke(model, dataset, config, output, monitor)
    else:
        train(model, dataset, config, output, monitor, resume=args.resume)


if __name__ == '__main__':
    main()
