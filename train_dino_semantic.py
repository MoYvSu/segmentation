# -*- coding: utf-8 -*-
"""DINO 通用骨干冻结、随机末端语义头五轮短训；不创建或更改几何预测。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch

from data.dino_semantic import ProcessedSemanticDataset, canonical_sha, collate_semantic
from models.dino_semantic import FrozenDinoSemantic, file_sha256, load_dino_backbone, state_sha256
from train_direct_semantic_affinity import build_semantic_criterion, compute_semantic_loss
from utils.config import load_config, project_path

FORMAT = 'frozen_dino_semantic_v1'


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Cannot restore the recorded CUDA RNG topology')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def learning_rate(step, cfg):
    """日程按预设 20 轮，不在五轮短训结束时把学习率衰减到底。"""
    total = int(cfg['schedule_epochs']) * int(cfg['updates_per_epoch'])
    if not 0 <= step < total:
        raise ValueError('Step outside the declared cosine schedule')
    floor = float(cfg.get('minimum_lr_ratio', .1))
    factor = floor + (1 - floor) * .5 * (1 + math.cos(math.pi * step / max(1, total - 1)))
    return float(cfg['learning_rate']) * factor


def validate_recipe(config):
    cfg = config['semantic_dino']
    expected = dict(expected_sources=32, input_size=1024, native_crop_size=1024,
                    views_per_source=8, updates_per_epoch=64, effective_batch_size=4,
                    schedule_epochs=20, precision='float32')
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'This controlled short run fixes {key}={value}')
    if cfg['micro_batch_size'] not in (1, 2, 4) or cfg['effective_batch_size'] % cfg['micro_batch_size']:
        raise ValueError('Microbatch must divide effective batch size 4')
    if not 1 <= int(cfg['epochs']) <= int(cfg['schedule_epochs']):
        raise ValueError('Stage length must be within the fixed 20-epoch schedule')
    if cfg.get('monitor_epochs') != [0, 1, 3, 5]:
        raise ValueError('This stage requires fixed e0/e1/e3/e5 monitors')
    if cfg['backbone'].get('architecture') != 'dinov3_vitb16':
        raise ValueError('Only official DINOv3 ViT-B/16 is configured')
    for key in ('learning_rate', 'weight_decay', 'adam_eps', 'grad_clip'):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) < 0:
            raise ValueError(f'Invalid optimization setting {key}')
    if float(cfg['learning_rate']) <= 0 or float(cfg['adam_eps']) <= 0 or float(cfg['grad_clip']) <= 0:
        raise ValueError('Positive learning rate, epsilon and gradient clip required')
    if not 0 <= float(cfg['minimum_lr_ratio']) <= 1 or not math.isfinite(float(cfg['minimum_lr_ratio'])):
        raise ValueError('minimum_lr_ratio must be finite and within [0,1]')
    return cfg


def build_identity(model, backbone_identity, dataset, config):
    cfg = config['semantic_dino']
    recipe_keys = ('seed', 'input_size', 'native_crop_size', 'views_per_source', 'updates_per_epoch',
                   'effective_batch_size', 'micro_batch_size', 'schedule_epochs', 'learning_rate',
                   'minimum_lr_ratio', 'weight_decay', 'adam_eps', 'grad_clip', 'precision')
    source_files = ('models/dino_semantic.py', 'data/dino_semantic.py', 'train_dino_semantic.py',
                    'utils/loss.py', 'utils/semantic_training.py', 'train_direct_semantic_affinity.py',
                    'data/dataset.py', 'data/semantic_targets.py', 'utils/offset_letterbox.py',
                    'tools/semantic_dino_monitor.py', 'utils/config.py')
    root = Path(__file__).resolve().parent
    monitor = cfg['monitor']
    control_files = {}
    for name in monitor.get('test_sources', []):
        control_dir = Path(project_path(config, monitor['control_dir']))
        for suffix in ('_inst.png', '_class.json'):
            path = control_dir / (name + suffix)
            control_files[path.name] = file_sha256(path)
        test_dir = Path(project_path(config, monitor['test_dir']))
        inputs = [p for p in test_dir.iterdir() if p.stem == name and p.suffix.lower() in
                  {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]
        if len(inputs) != 1:
            raise ValueError(f'Unique fixed monitor input required: {name}')
        control_files['input/' + inputs[0].name] = file_sha256(inputs[0])
    return dict(format=FORMAT, architecture=model.architecture, backbone=backbone_identity,
                data=dataset.identity, recipe={k: cfg[k] for k in recipe_keys},
                semantic_loss=deepcopy(config['direct_semantic_affinity']['semantic_loss']),
                runtime_sources={p: file_sha256(root / p) for p in source_files},
                control_manifest_sha256=cfg.get('control_manifest_sha256'),
                monitor_recipe={k: v for k, v in monitor.items() if k not in ('test_dir', 'control_dir')},
                monitor_epochs=cfg['monitor_epochs'], monitor_input_sha256=control_files,
                protocol='random_head; raw_RGB; frozen_backbone; terminal_semantic_only')


def save_checkpoint(path, model, optimizer, identity, step, cfg, monitor_epochs):
    path = Path(path)
    payload = dict(format=FORMAT, architecture=model.architecture, identity=identity,
                   identity_sha256=canonical_sha(identity), head_state_dict=model.head.state_dict(),
                   optimizer=optimizer.state_dict(), rng=capture_rng(), global_step=int(step),
                   epoch=int(step) // int(cfg['updates_per_epoch']),
                   step_in_epoch=int(step) % int(cfg['updates_per_epoch']),
                   initial_head_sha256=model.initial_head_sha256,
                   head_sha256=state_sha256(model.head), monitor_epochs=list(monitor_epochs),
                   scheduler=dict(kind='cosine_by_global_step', next_step=int(step),
                                  total_updates=int(cfg['schedule_epochs']) * int(cfg['updates_per_epoch'])),
                   stage_epochs=int(cfg['epochs']), backbone_weights_saved=False,
                   backbone_state_sha256=identity['backbone']['backbone_state_sha256'])
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume(path, model, optimizer, identity, cfg):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('format') != FORMAT or payload.get('identity') != identity
            or payload.get('identity_sha256') != canonical_sha(identity)):
        raise ValueError('Resume data/model/source/recipe identity differs')
    if payload.get('initial_head_sha256') != model.initial_head_sha256:
        raise ValueError('Resume did not start from the identical random head')
    step = payload.get('global_step')
    if type(step) is not int or not 0 <= step <= int(cfg['epochs']) * int(cfg['updates_per_epoch']):
        raise ValueError('Resume step exceeds the requested stage')
    if (payload.get('epoch') != step // int(cfg['updates_per_epoch'])
            or payload.get('step_in_epoch') != step % int(cfg['updates_per_epoch'])
            or payload.get('scheduler', {}).get('next_step') != step):
        raise ValueError('Resume step/schedule receipt disagrees')
    if state_sha256(model.backbone) != identity['backbone']['backbone_state_sha256']:
        raise ValueError('Frozen backbone differs before resume')
    model.head.load_state_dict(payload['head_state_dict'], strict=True)
    if state_sha256(model.head) != payload['head_sha256']:
        raise ValueError('Restored head tensor digest differs')
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'])
    return step, list(payload['monitor_epochs'])


def run_monitor(callback, model, epoch, output):
    """监控绝不参与 loss、梯度、随机流或 epoch 选择。"""
    state, mode = capture_rng(), model.training
    before = state_sha256(model.head)
    try:
        model.eval()
        with torch.no_grad():
            result = callback(model, int(epoch), Path(output))
        if state_sha256(model.head) != before:
            raise RuntimeError('Monitor modified semantic head state')
        return result
    finally:
        restore_rng(state)
        model.train(mode)


def optimization_step(model, optimizer, criterion, batches, cfg, step):
    """显式梯度累积；记录真实未裁剪范数和逐 microbatch loss。"""
    model.train()
    device = next(model.head.parameters()).device
    count = sum(int(b['image'].shape[0]) for b in batches)
    if count != int(cfg['effective_batch_size']):
        raise ValueError('Optimizer update did not contain the declared effective batch')
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    values, valid_pixels = [], 0
    for batch in batches:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        if not bool((batch['semantic_target'] >= 0).any()):
            raise ValueError('An empty-supervision microbatch would waste the fixed update budget')
        logits = model(batch['image'])
        loss = compute_semantic_loss(criterion, logits, batch)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite semantic loss')
        fraction = batch['image'].shape[0] / count
        (loss * fraction).backward()
        values.append(dict(loss=float(loss.detach()), images=int(batch['image'].shape[0])))
        valid_pixels += int((batch['semantic_target'] >= 0).sum())
    norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), float(cfg['grad_clip']))
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise FloatingPointError(f'Invalid semantic head gradient: {norm}')
    if any(p.grad is not None or p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError('Frozen backbone received a gradient')
    optimizer.step()
    return dict(global_step=step + 1, loss=sum(v['loss'] * v['images'] / count for v in values),
                microbatch_losses=values, grad_norm_before_clip=float(norm), learning_rate=lr,
                effective_batch_size=count, valid_pixels=valid_pixels, backbone_has_grad=False)


def reconcile_step_log(output, step):
    """checkpoint 落后日志时保留尾部备份，再从已保存更新恢复；不叠加重复 step。"""
    path = Path(output) / 'steps.jsonl'
    if not path.is_file():
        return
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    entries = [json.loads(line) for line in lines if line.strip()]
    if len(entries) != len(lines):
        raise ValueError('Step log contains unexpected empty records')
    values = [r.get('global_step') for r in entries]
    if values and (any(type(v) is not int for v in values)
                   or values != list(range(values[0], values[0] + len(values)))):
        raise ValueError('Step log is not a unique consecutive update sequence')
    if values and values[-1] < step:
        raise ValueError('Existing step log is behind the resume checkpoint; use an isolated output directory')
    prefix = [line for line, value in zip(lines, values) if value <= step]
    tail = [line for line, value in zip(lines, values) if value > step]
    if tail:
        backup = Path(output) / f'steps_after_s{step}_{time.time_ns()}.jsonl'
        with backup.open('x', encoding='utf-8') as stream:
            stream.writelines(tail)
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(prefix), encoding='utf-8')
        temporary.replace(path)
        write_json(Path(output) / 'resume_log_repair.json', dict(
            resumed_from_step=step, preserved_tail_file=backup.name, preserved_updates=len(tail),
            reason='Log entries after saved optimizer/RNG state will be replayed; tail preserved, not counted twice'))


def train(model, dataset, config, output, backbone_identity, callback, resume=None):
    cfg = config['semantic_dino']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and any(output.iterdir()):
        raise ValueError('Fresh training requires an empty output directory')
    if callback is None:
        raise ValueError('Fixed e0/e1/e3/e5 process monitor is required')
    device = next(model.head.parameters()).device
    identity = build_identity(model, backbone_identity, dataset, config)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(cfg['learning_rate']),
                                  weight_decay=float(cfg['weight_decay']), eps=float(cfg['adam_eps']))
    criterion = build_semantic_criterion(config, device)
    step, done_monitors = (0, []) if resume is None else load_resume(resume, model, optimizer, identity, cfg)
    if resume is not None:
        reconcile_step_log(output, step)
    initial_backbone = backbone_identity['backbone_state_sha256']
    if state_sha256(model.backbone) != initial_backbone:
        raise RuntimeError('Backbone differs from the loaded official source')
    dataset.verify_sources()
    write_json(output / 'identity.json', identity)
    write_json(output / 'resolved_config.json', config)
    write_json(output / 'sources.json', dataset.records)
    write_json(output / 'initialization.json', dict(
        initial_head_sha256=model.initial_head_sha256, backbone_state_sha256=initial_backbone,
        trainable_parameters=sum(p.numel() for p in model.head.parameters()),
        frozen_parameters=sum(p.numel() for p in model.backbone.parameters()),
        task_weights_loaded=False, random_head=True, resume=resume is not None))
    updates, batch, micro = int(cfg['updates_per_epoch']), int(cfg['effective_batch_size']), int(cfg['micro_batch_size'])
    target_steps = int(cfg['epochs']) * updates
    if len(dataset) != updates * batch:
        raise ValueError('Sampler length does not match the exact optimizer budget')
    def save(path):
        save_checkpoint(path, model, optimizer, identity, step, cfg, done_monitors)
    def monitor(epoch):
        if epoch in cfg['monitor_epochs'] and epoch not in done_monitors:
            run_monitor(callback, model, epoch, output)
            done_monitors.append(epoch)
            save(output / 'last.pt')
    if step == 0:
        save(output / 'epoch_000.pt')
        monitor(0)
    elif step % updates == 0:
        monitor(step // updates)
    started = time.monotonic()
    while step < target_steps:
        epoch = step // updates + 1
        if dataset.epoch != epoch or step % updates == 0:
            dataset.set_epoch(epoch)
            write_json(output / f'sampling_e{epoch:03d}.json', dict(**dataset.plan_receipt(), plan=dataset.plan))
        first = (step % updates) * batch
        samples = [dataset[i] for i in range(first, first + batch)]
        batches = [collate_semantic(samples[i:i+micro]) for i in range(0, batch, micro)]
        row = optimization_step(model, optimizer, criterion, batches, cfg, step)
        row.update(epoch=epoch, update_in_epoch=step % updates + 1,
                   views=[v for b in batches for v in b['views']],
                   elapsed_seconds=time.monotonic() - started)
        step += 1
        with (output / 'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if step % int(cfg.get('checkpoint_every_updates', 16)) == 0 or step % updates == 0:
            save(output / 'last.pt')
        if step % updates == 0:
            dataset.verify_sources()
            current_backbone = state_sha256(model.backbone)
            if current_backbone != initial_backbone:
                raise RuntimeError('Frozen backbone parameters or buffers changed')
            save(output / f'epoch_{epoch:03d}.pt')
            monitor(epoch)
            epoch_row = dict(epoch=epoch, global_step=step, head_sha256=state_sha256(model.head),
                             backbone_state_sha256=current_backbone, frozen_unchanged=True)
            write_json(output / f'epoch_{epoch:03d}.json', epoch_row)
            print(json.dumps(epoch_row), flush=True)
    save(output / 'last.pt')
    if state_sha256(model.head) == model.initial_head_sha256:
        raise RuntimeError('Random semantic head did not change')
    write_json(output / 'complete.json', dict(format=FORMAT, complete=True, epochs=int(cfg['epochs']),
               updates=step, effective_batch_size=batch, windows=step*batch, monitor_epochs=done_monitors,
               backbone_frozen=state_sha256(model.backbone) == initial_backbone,
               head_sha256=state_sha256(model.head), identity_sha256=canonical_sha(identity),
               training_data='processed_32_sources_only', official_score=None, accuracy_claimed=False,
               geometry_modified=False, control_retrained=False, convergence_claimed=False))
    return output


def smoke(model, dataset, config, output, backbone_identity):
    """真实授权骨干的两更新短测；独立目录，严格重载，不作为正式五轮训练。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Smoke requires an empty output directory')
    cfg = config['semantic_dino']
    identity = build_identity(model, backbone_identity, dataset, config)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=float(cfg['learning_rate']),
                                  weight_decay=float(cfg['weight_decay']), eps=float(cfg['adam_eps']))
    device = next(model.parameters()).device
    criterion = build_semantic_criterion(config, device)
    batch, micro = int(cfg['effective_batch_size']), int(cfg['micro_batch_size'])
    rows = []
    for step in range(2):
        samples = [dataset[i] for i in range(step * batch, (step + 1) * batch)]
        batches = [collate_semantic(samples[i:i+micro]) for i in range(0, batch, micro)]
        rows.append(optimization_step(model, optimizer, criterion, batches, cfg, step))
    if state_sha256(model.backbone) != backbone_identity['backbone_state_sha256']:
        raise RuntimeError('Backbone changed during smoke')
    checkpoint = output / 'smoke.pt'
    save_checkpoint(checkpoint, model, optimizer, identity, 2, cfg, [])
    sample = dataset[0]['image'][None].to(device)
    model.eval()
    with torch.no_grad():
        before = model(sample).cpu()
    # 冻结骨干只读复用；新的随机头和优化器严格从自己的 smoke 权重重载。
    reloaded = FrozenDinoSemantic(model.backbone, head_width=int(cfg['head_width']),
                                  head_seed=int(cfg['seed'])).to(device)
    loaded_optimizer = torch.optim.AdamW(reloaded.head.parameters(), lr=float(cfg['learning_rate']))
    loaded_step, _ = load_resume(checkpoint, reloaded, loaded_optimizer, identity, cfg)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(sample).cpu()
    if not torch.equal(before, after) or loaded_step != 2:
        raise RuntimeError('Strict smoke reload changed actual semantic prediction')
    report = dict(format='dino_semantic_smoke_v1', complete=True, optimizer_updates=2, rows=rows,
                  official_backbone_identity=backbone_identity, strict_reload_prediction_equal=True,
                  frozen_backbone_unchanged=True, formal_training_complete=False,
                  official_score=None, convergence_claimed=False)
    write_json(output / 'smoke.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_dino.yaml')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--data-check-only', '--preflight', dest='preflight', action='store_true',
                       help='Only verify processed GT, CPU sampler and monitor inputs; do not load weights')
    modes.add_argument('--smoke', action='store_true', help='Two updates with an approved official backbone and strict reload')
    modes.add_argument('--run', action='store_true', help='Run the declared five-epoch stage with approved backbone weights')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cfg = validate_recipe(config)
    output = Path(project_path(config, args.output or cfg['output_dir']))
    random.seed(int(cfg['seed']))
    np.random.seed(int(cfg['seed']))
    torch.manual_seed(int(cfg['seed']))
    torch.set_num_threads(int(cfg.get('torch_threads', 4)))
    cv2.setNumThreads(int(cfg.get('opencv_threads', 2)))
    dataset = ProcessedSemanticDataset(config)
    if args.preflight:
        if args.resume:
            raise ValueError('Preflight does not load training checkpoints')
        from tools.semantic_dino_monitor import build_monitor
        # 只构建固定输入并检查 N0 PNG/JSON，不运行模型或渲染图。
        build_monitor(config)
        result = dict(identity=dataset.identity, plan=dataset.plan_receipt(), training=False,
                      weights_loaded=False, monitor_input_contract_checked=True,
                      optimizer_updates=int(cfg['epochs']) * int(cfg['updates_per_epoch']),
                      effective_batch_size=int(cfg['effective_batch_size']))
        for index in (0, len(dataset.sources)):
            sample = dataset[index]
            result.setdefault('shape_checks', []).append(dict(
                source=sample['source'], mode=sample['mode'], image=list(sample['image'].shape),
                valid_pixels=int((sample['semantic_target'] >= 0).sum()), geometry=sample['geometry']))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    bcfg = cfg['backbone']
    backbone, identity = load_dino_backbone(project_path(config, bcfg['source_dir']),
                                           project_path(config, bcfg['checkpoint']), bcfg['sha256'])
    if bcfg.get('source_version'):
        identity['source_version'] = bcfg['source_version']
    model = FrozenDinoSemantic(backbone, head_width=int(cfg['head_width']), head_seed=int(cfg['seed'])).to(args.device)
    if args.smoke:
        if args.resume:
            raise ValueError('Smoke starts from a random head; no resume')
        smoke(model, dataset, config, output, identity)
        return
    from tools.semantic_dino_monitor import build_monitor
    callback = build_monitor(config)
    train(model, dataset, config, output, identity, callback, resume=args.resume)


if __name__ == '__main__':
    main()
