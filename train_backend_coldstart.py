# -*- coding: utf-8 -*-
"""固定D5a/SAM2，纯SSL起点冷启动双头；全人工和既有SAM2数据，无留出。"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.mim_dataset import list_images
from models.backend_adaptation import load_backend, restore_first, save_backend
from models.backend_coldstart import build_cold_backend
from models.direct_semantic_affinity import configure_direct_training_phase, direct_parameter_groups
from models.lora import enable_trunk_gradient_checkpointing
from models.rgb_restoration import load_rgb_restorer
from train_backend_adaptation import (
    build_datasets, frozen_digest, monitor, prediction_maps, save_prediction,
    sha, tensor_digest, write_json,
)
from train_direct_semantic_affinity import (
    build_semantic_criterion, compute_affinity_loss, compute_semantic_loss, move_batch, set_seed,
)
from train_semantic_d5a import aligned_semantic_logits
from utils.affinity_deployment import prepare_image
from utils.config import load_config, project_path


def phase_plan(config, smoke=False):
    cfg = config['backend_adaptation']
    warm, joint = int(cfg['head_warmup_epochs']), int(cfg['joint_epochs'])
    if min(warm, joint) < 1 or warm + joint != int(cfg['epochs']):
        raise ValueError('Cold start requires positive warmup/joint phases matching epochs')
    return [('head_warmup', 1 if smoke else warm, False), ('joint', 1 if smoke else joint, True)]


def get_restorer(config, device):
    cfg = config['backend_adaptation']['restoration']
    path = Path(project_path(config, cfg['checkpoint']))
    if cfg['mode'] != 'first_x0' or sha(path) != cfg['checkpoint_sha256']:
        raise ValueError('Expected the designated frozen D5a epoch 60 first_x0 model')
    return load_rgb_restorer(str(path), device).eval().requires_grad_(False)


def lora_digest(model):
    return tensor_digest((n, p) for n, p in model.encoder.named_parameters()
                         if 'lora_A' in n or 'lora_B' in n)


def phase_rates(config, train_lora):
    rates = config['backend_adaptation']['learning_rates']
    prefix = 'joint' if train_lora else 'warmup'
    result = {name: rates[f'{prefix}_{name}'] for name in ('semantic', 'affinity')}
    if train_lora:
        result['lora'] = rates['joint_lora']
    return result


def append_row(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def simple_semantic_loss(criterion, logits, batch):
    # simple输出为stride4，沿用原simple的logits插值，不缩小GT及细实例。
    return compute_semantic_loss(criterion, aligned_semantic_logits(logits, batch), batch)


def train(config, output_dir, smoke=False):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Use the sam2_env BF16 GPU environment')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    cfg = config['backend_adaptation']
    if cfg['amp_dtype'] != 'bfloat16':
        raise ValueError('Cold-start recipe requires BF16')
    plan = phase_plan(config, smoke)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    set_seed(cfg['seed'])
    model, metadata = build_cold_backend(config, device)
    restorer = get_restorer(config, device)
    manual, pseudo, data_info = build_datasets(config)
    data_info.update(semantic_direct_sources=data_info['manual_sources'],
                     affinity_direct_sources={'manual': data_info['manual_sources'],
                                              'sam2': data_info['sam2_sources']},
                     extra_unlabeled_stream=False)
    initial_frozen = frozen_digest(model)
    initial_lora = lora_digest(model)
    initial_d5a = tensor_digest(restorer.state_dict().items())
    total = model.parameter_summary()['total'] + sum(p.numel() for p in restorer.parameters())
    if total >= 500_000_000:
        raise ValueError('Combined model exceeds competition parameter limit')
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
              if n.startswith(('semantic_decoder.', 'affinity_decoder.')) or 'lora_A' in n or 'lora_B' in n}
    status = {'status': 'preflight', 'config': config, 'smoke': smoke, 'sources': metadata,
              'data': data_info, 'total_parameters': total, 'phases': {}, 'failed_updates': 0,
              'selection': 'prespecified_final_epoch_no_validation_no_phase_rewind',
              'initial_frozen_sha256': initial_frozen, 'initial_lora_sha256': initial_lora,
              'initial_d5a_state_sha256': initial_d5a}
    write_json(output / 'status.json', status)
    enable_trunk_gradient_checkpointing(model.encoder.trunk)
    criterion = build_semantic_criterion(config, device)
    loss_cfg = config['direct_semantic_affinity']['affinity_loss']
    monitor(model, restorer, config, output, 0)
    steps, epoch = 0, 0
    started = time.time()
    planned_epochs = sum(item[1] for item in plan)
    phase_updates = 2 if smoke else data_info['updates_per_epoch']
    for phase, duration, train_lora in plan:
        configure_direct_training_phase(model, train_lora=train_lora)
        groups = direct_parameter_groups(model, phase_rates(config, train_lora))
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e, length=duration:
            cfg['minimum_lr_ratio'] + (1 - cfg['minimum_lr_ratio']) * (1 + math.cos(math.pi * e / length)) / 2)
        status['phases'][phase] = {'epochs': 0, 'updates': 0}
        status.setdefault('trainable_parameters_by_phase', {})[phase] = {
            group['name']: sum(p.numel() for p in group['params']) for group in groups}
        for phase_epoch in range(1, duration + 1):
            epoch += 1
            manual.set_epoch(epoch)
            pseudo.set_epoch(epoch)
            loaders = [DataLoader(dataset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'],
                       generator=torch.Generator().manual_seed(cfg['seed'] + epoch * 2 + i), pin_memory=True)
                       for i, dataset in enumerate((manual, pseudo))]
            model.eval()
            model.semantic_decoder.train()
            model.affinity_decoder.train()
            totals = np.zeros(3, dtype=np.float64)
            count = 0
            learning_rates = {g['name']: g['lr'] for g in groups}
            for index, (raw, raw_pseudo) in enumerate(zip(*loaders)):
                if index >= phase_updates:
                    break
                step_seed = cfg['seed'] + epoch * 10000 + index
                set_seed(step_seed)
                batch, pb = move_batch(raw, device), move_batch(raw_pseudo, device)
                optimizer.zero_grad(set_to_none=True)
                batch['image'] = restore_first(restorer, batch['image'], step_seed + 1000000)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(batch['image'])
                    sem_loss = simple_semantic_loss(criterion, prediction['semantic_logits'], batch)
                    aff_loss, _ = compute_affinity_loss(prediction['affinity_logits'].float(), batch, loss_cfg, pseudo=False)
                    manual_loss = sem_loss + aff_loss
                if not bool(torch.isfinite(manual_loss)):
                    raise FloatingPointError('Non-finite manual loss')
                manual_loss.backward()
                del prediction, manual_loss
                # SAM2没有类别，第二次前向只监督affinity，不能伪装成语义看过64图。
                pb['image'] = restore_first(restorer, pb['image'], step_seed + 2000000)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    features = model.encoder(pb['image'])
                    logits = model.affinity_decoder(features)['affinity_logits'].float()
                    pseudo_loss, _ = compute_affinity_loss(logits, pb, loss_cfg, pseudo=True)
                if not bool(torch.isfinite(pseudo_loss)):
                    raise FloatingPointError('Non-finite SAM2 affinity loss')
                (cfg['pseudo_weight'] * pseudo_loss).backward()
                del features, logits
                missing_gradients = [n for n, p in model.named_parameters()
                                     if p.requires_grad and p.grad is None]
                if missing_gradients:
                    raise RuntimeError(f'Trainable tensors have no gradients: {missing_gradients}')
                grad_norms = {}
                for group in groups:
                    norms = [p.grad.float().norm() for p in group['params'] if p.grad is not None]
                    norm = float(torch.stack(norms).norm()) if norms else 0.0
                    if not math.isfinite(norm) or norm <= 0:
                        raise FloatingPointError(f'Invalid gradient group {group["name"]}: {norm}')
                    grad_norms[group['name']] = norm
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              cfg['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                count += 1
                values = [float(sem_loss.detach()), float(aff_loss.detach()), float(pseudo_loss.detach())]
                totals += values
                append_row(output / 'steps.jsonl', {'epoch': epoch, 'phase': phase, 'phase_epoch': phase_epoch,
                    'step': index + 1, 'updates': steps, 'manual': raw['name'], 'pseudo': raw_pseudo['name'],
                    'semantic_loss': values[0], 'affinity_loss': values[1], 'pseudo_loss': values[2],
                    'grad_norms': grad_norms, 'seed': step_seed})
            if count != phase_updates:
                raise RuntimeError('Incomplete cold-start training epoch')
            scheduler.step()
            status['phases'][phase]['epochs'] += 1
            status['phases'][phase]['updates'] += count
            row = {'epoch': epoch, 'phase': phase, 'phase_epoch': phase_epoch, 'updates': steps,
                   'failed_updates': 0, 'semantic_loss': totals[0] / count,
                   'affinity_loss': totals[1] / count, 'pseudo_loss': totals[2] / count,
                   'learning_rates': learning_rates, 'elapsed_seconds': time.time() - started,
                   'peak_cuda_mib': torch.cuda.max_memory_allocated() / 1024**2}
            append_row(output / 'epochs.jsonl', row)
            print(json.dumps(row), flush=True)
            if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == planned_epochs:
                monitor(model, restorer, config, output, epoch)
            if not train_lora and phase_epoch == duration:
                status['warmup_lora_unchanged'] = lora_digest(model) == initial_lora
                if not status['warmup_lora_unchanged']:
                    raise RuntimeError('SSL LoRA changed during head-only warmup')
            extra = {'phase': phase, 'phase_epoch': phase_epoch, 'updates': steps, 'failed_updates': 0,
                     'smoke': smoke, 'data': data_info, 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict()}
            save_backend(model, config, metadata, output / 'last.pt', epoch, extra=extra)
            if phase_epoch == duration:
                save_backend(model, config, metadata, output / f'epoch_{epoch:03d}.pt', epoch,
                             extra={k: v for k, v in extra.items() if k not in ('optimizer', 'scheduler')})
            status.update(status='running', **row)
            write_json(output / 'status.json', status)
        del optimizer, scheduler, groups
    status['frozen_encoder_unchanged'] = frozen_digest(model) == initial_frozen
    status['d5a_unchanged'] = tensor_digest(restorer.state_dict().items()) == initial_d5a
    if not status['frozen_encoder_unchanged'] or not status['d5a_unchanged']:
        raise RuntimeError('Frozen SAM2 base or D5a changed')
    changes = {'semantic': 0.0, 'affinity': 0.0, 'lora': 0.0}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        group = 'semantic' if name.startswith('semantic_decoder.') else (
            'affinity' if name.startswith('affinity_decoder.') else 'lora')
        changes[group] = max(changes[group], float((parameter.detach().cpu() - before[name]).abs().max()))
    if any(not math.isfinite(delta) or delta <= 0 for delta in changes.values()):
        raise RuntimeError(f'A trainable component failed to update: {changes}')
    status['trainable_max_changes'] = changes
    # 严格重建fused模型，验证新头和新LoRA成套部署，不混回旧几何权重。
    probe_path = manual.base_dataset.samples[0]
    _, probe, _, _ = prepare_image(probe_path, 1024, device)
    probe = restore_first(restorer, probe, cfg['restoration']['inference_seed'])
    model.eval()
    with torch.no_grad():
        expected = {k: v.cpu() for k, v in model(probe).items()}
    del before, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(output / f'epoch_{epoch:03d}.pt', config, device)
    with torch.no_grad():
        actual = {k: v.cpu() for k, v in reloaded(probe).items()}
    differences = {k: float((expected[k] - actual[k]).abs().max()) for k in expected}
    if any(delta != 0 for delta in differences.values()):
        raise RuntimeError(f'Fused reload changed outputs: {differences}')
    status.update(status='completed', strict_reload_passed=True, reload_max_differences=differences,
                  checkpoint=str(output / f'epoch_{epoch:03d}.pt'),
                  checkpoint_sha256=sha(output / f'epoch_{epoch:03d}.pt'))
    write_json(output / 'status.json', status)


def inference(config, checkpoint, output_dir, smoke=False):
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    cfg = config['backend_adaptation']
    model, payload = load_backend(checkpoint, config, device)
    restorer = get_restorer(config, device)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, name in enumerate(list_images(project_path(config, config['inference']['test_dir']))):
        path = Path(name)
        if smoke and path.stem not in cfg['monitor']['images']:
            continue
        image, semantic, boundary = prediction_maps(model, path, config, device, restorer,
                                                    cfg['restoration']['inference_seed'] + index)
        instances, classes = save_prediction(image, semantic, boundary, config, output, path.stem)
        rows.append({'image': path.name, 'instances': len(classes), 'max_instance_id': int(instances.max()),
                     'ferrite': sum(int(v) == 1 for v in classes.values())})
        print(f'cold inference: {path.name}', flush=True)
    write_json(output / 'manifest.json', {'epoch': payload['epoch'], 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha(checkpoint), 'sampling_mode': 'first_x0', 'precision': 'FP32',
        'images': rows, 'config': config, 'official_score': None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/backend_cold.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(project_path(config, args.output_dir))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    try:
        if args.checkpoint:
            inference(config, args.checkpoint, output, args.smoke)
        else:
            train(config, output, args.smoke)
    except BaseException as error:
        if output.exists():
            write_json(output / 'failure.json', {'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
