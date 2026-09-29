# -*- coding: utf-8 -*-
"""固定D5a/编码器/语义，只适配affinity；完整部署关系指导双向困难边监督。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.affinity_connectivity import build_datasets
from data.mim_dataset import list_images
from models.backend_adaptation import build_backend, load_backend, restore_first, save_backend
from models.rgb_restoration import load_rgb_restorer
from tools.affinity_connectivity_views import predict_cross, save_monitor, training_partition
from tools.semantic_crossover import _read_prediction
from train_backend_adaptation import sha, tensor_digest, write_json
from train_direct_semantic_affinity import compute_affinity_loss, move_batch, set_seed
from utils.affinity_connectivity import deployment_connectivity_loss
from utils.affinity_interface import interface_loss
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path


def configure_training(model):
    model.eval().requires_grad_(False)
    model.encoder.trainable_lora = False
    model.affinity_decoder.train().requires_grad_(True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any(not n.startswith('affinity_decoder.') for n in names):
        raise RuntimeError('Only affinity_decoder may train')


def frozen_digest(model):
    return tensor_digest((n, v) for n, v in model.state_dict().items()
                         if not n.startswith('affinity_decoder.'))


def get_restorer(config, device):
    cfg = config['backend_adaptation']['restoration']
    path = project_path(config, cfg['checkpoint'])
    if sha(path) != cfg['checkpoint_sha256'] or cfg['mode'] != 'first_x0':
        raise ValueError('D5a checkpoint or sampling contract differs')
    return load_rgb_restorer(path, device).eval().requires_grad_(False)


def capped_weight(base, auxiliary, logits, requested, ratio):
    """限制附加项对输出logits的梯度；不冒充参数空间梯度保证。"""
    if requested <= 0:
        return 0.0, {'base_logit_grad': None, 'aux_logit_grad': None}
    base_norm = float(torch.autograd.grad(base, logits, retain_graph=True)[0].norm())
    aux_norm = float(torch.autograd.grad(auxiliary, logits, retain_graph=True)[0].norm())
    weight = min(float(requested), ratio * base_norm / max(aux_norm, 1e-12))
    return weight, {'base_logit_grad': base_norm, 'aux_logit_grad': aux_norm,
                    'weighted_logit_grad_ratio': weight * aux_norm / max(base_norm, 1e-12)}


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def auxiliary_loss(config, logits, target, valid, partition, instances, trusted):
    extra = config['affinity_connectivity']
    options = {k: extra[k] for k in ('max_groups', 'minimum_group_edges', 'negative_margin', 'positive_margin')}
    interface = config.get('affinity_interface', {})
    if interface.get('enabled', False):
        return interface_loss(logits, target, valid, partition, instances, trusted_pixels=trusted,
                              band_radius=interface['band_radius'], **options)
    return deployment_connectivity_loss(logits, target, valid, partition, instances,
        trusted_pixels=trusted, edges_per_group=extra['edges_per_group'], **options)


def common_config(config):
    """仅排除本轮显式独立的选位配置及输出位置，其余共同配置必须一致。"""
    result = deepcopy(config)
    result.pop('affinity_interface', None)
    result['backend_adaptation'].pop('output_dir', None)
    return json.loads(json.dumps(result))


def reused_control(config, initial_status):
    options = config.get('affinity_interface', {})
    if not options.get('enabled', False):
        return None
    root = Path(project_path(config, options['reused_control']))
    status = json.loads((root / 'status.json').read_text(encoding='utf8'))
    if (status['status'] != 'complete' or status['updates'] != 1280
            or sha(root / 'final.pt') != options['control_sha256']):
        raise RuntimeError('Reused control is incomplete or its checkpoint differs')
    if common_config(config) != common_config(status['config']):
        raise RuntimeError('Common training/inference configuration differs from control')
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data'):
        if initial_status[key] != status[key]:
            raise RuntimeError(f'Reused control dependency differs: {key}')
    rows = [json.loads(line) for line in (root / 'steps.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != 1280 or any(r['auxiliary_weight'] != 0 for r in rows):
        raise RuntimeError('Invalid control step receipts')
    return rows


def compare_draw(row, control):
    for key in ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw'):
        if row[key] != control[key]:
            raise RuntimeError(f'Existing control draw differs: {key}')


def parameter_effect(base, auxiliary, parameters, weight):
    """诊断实际参数空间的力度，不改变既有输出梯度预算或optimizer。"""
    parameters = list(parameters)
    a = torch.autograd.grad(base, parameters, retain_graph=True)
    b = torch.autograd.grad(auxiliary, parameters, retain_graph=True)
    an = torch.stack([v.float().square().sum() for v in a]).sum().sqrt()
    bn = torch.stack([v.float().square().sum() for v in b]).sum().sqrt()
    dot = torch.stack([(x.float()*y.float()).sum() for x,y in zip(a,b)]).sum()
    return dict(parameter_base_norm=float(an), parameter_aux_norm=float(bn),
                parameter_aux_ratio=float(weight*bn/an.clamp_min(1e-12)),
                parameter_gradient_cosine=float(dot/(an*bn).clamp_min(1e-12)))


def draw_receipt(batch):
    keys = ('source_index', 'draw_index', 'profile', 'is_spatial', 'endpoint_applied',
            'noise_sigma', 'base_sigma', 'horizontal_flip', 'vertical_flip', 'rotation_k')
    return {key: (batch[key].tolist() if torch.is_tensor(batch[key]) else batch[key])
            for key in keys if key in batch}


@torch.no_grad()
def initial_parity(model, restorer, config, device):
    files = list_images(project_path(config, config['inference']['test_dir']))
    names = set(config['affinity_connectivity']['parity_images'])
    results = []
    with zipfile.ZipFile(project_path(config, config['affinity_connectivity']['reference_zip'])) as archive:
        for index, filename in enumerate(files):
            path = Path(filename)
            if path.stem not in names:
                continue
            output = predict_cross(model, restorer, path, config, device,
                                   config['backend_adaptation']['restoration']['inference_seed'] + index)
            expected, classes, _ = _read_prediction(archive, path.stem)
            equal = np.array_equal(output['instances'], expected) and output['classes'] == classes
            results.append({'image': path.stem, 'final_output_equal': equal,
                            'instances': len(classes)})
            if not equal:
                raise RuntimeError(f'{path.stem}: e0 fails exact deployed PNG/class reproduction')
    if {r['image'] for r in results} != names:
        raise RuntimeError('Missing initial parity images')
    return results


def train(config, arm, output, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Use GPU server sam2_env')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    cfg, extra = config['backend_adaptation'], config['affinity_connectivity']
    if config.get('affinity_interface', {}).get('enabled') and arm != 'candidate':
        raise ValueError('Interface experiment reuses control; do not retrain it')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    set_seed(cfg['seed'])
    model, initialization = build_backend(config, device)
    restorer = get_restorer(config, device)
    total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in restorer.parameters())
    if total >= 500_000_000:
        raise RuntimeError('Combined parameter limit exceeded')
    initial_state = tensor_digest(model.state_dict().items())
    frozen_start = frozen_digest(model)
    restorer_start = tensor_digest(restorer.state_dict().items())
    initial_head = tensor_digest(model.affinity_decoder.state_dict().items())
    manual, pseudo, data_info = build_datasets(config)
    status = dict(status='preflight', arm=arm, smoke=smoke, config=config,
                  initialization=initialization, initial_state_sha256=initial_state,
                  frozen_state_sha256=frozen_start, restorer_state_sha256=restorer_start,
                  total_parameters=total, data=data_info, precision='FP32',
                  selection='fixed_final_epoch_no_holdout', official_score=None)
    write_json(output / 'status.json', status)
    reference_steps = reused_control(config, status)
    status['reused_control_verified'] = reference_steps is not None
    status['initial_parity'] = initial_parity(model, restorer, config, device)
    configure_training(model)
    optimizer = torch.optim.AdamW(model.affinity_decoder.parameters(),
                                 lr=cfg['learning_rates']['affinity'],
                                 weight_decay=cfg['weight_decay'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e:
        cfg['minimum_lr_ratio'] + (1 - cfg['minimum_lr_ratio']) *
        (1 + math.cos(math.pi * e / cfg['epochs'])) / 2)
    names = cfg['monitor']['images'][:2] if smoke else cfg['monitor']['images']
    save_monitor(model, restorer, config, device, output, 0, names)
    loss_cfg = config['direct_semantic_affinity']['affinity_loss']
    epochs, updates = (1 if smoke else cfg['epochs']), 0
    active = {'negative': 0, 'positive': 0}
    started = time.time()
    for epoch in range(1, epochs + 1):
        manual.set_epoch(epoch)
        pseudo.set_epoch(epoch)
        loaders = [DataLoader(ds, batch_size=1, shuffle=True, num_workers=cfg['num_workers'],
                             generator=torch.Generator().manual_seed(cfg['seed'] + epoch * 2 + i),
                             pin_memory=True) for i, ds in enumerate((manual, pseudo))]
        configure_training(model)
        sums, count = np.zeros(3), 0
        for index, (raw, raw_pseudo) in enumerate(zip(*loaders)):
            if smoke and index >= extra['smoke_updates']:
                break
            seed = cfg['seed'] + epoch * 10000 + index
            set_seed(seed)
            batch, pb = move_batch(raw, device), move_batch(raw_pseudo, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                restored = restore_first(restorer, batch['image'], seed + 1000000)
                features = model.encoder(restored)
            prediction = model.affinity_decoder(features)
            logits = prediction['affinity_logits']
            base, _ = compute_affinity_loss(logits, batch, loss_cfg, pseudo=False)
            auxiliary = logits.sum() * 0
            details, weight, norms = None, 0.0, {}
            if index % extra['every_updates'] == 0:
                with torch.no_grad():
                    sem = model.semantic_decoder(features, restored)
                    # semantic decoder returns the same mapping as fused forward.
                    prediction['semantic_logits'] = sem['semantic_logits'] if isinstance(sem, dict) else sem
                    partition = training_partition(prediction, batch, config)
                target, edge_valid = build_affinity_targets_torch(
                    batch['affinity_instance_map'], batch['affinity_valid_content'])
                auxiliary, details = auxiliary_loss(config, logits, target, edge_valid, partition,
                    batch['affinity_instance_map'], batch['trusted_pixels'])
                requested = (extra['weight'] * min(1.0, epoch / extra['warmup_epochs'])
                             if arm == 'candidate' else 0.0)
                weight, norms = capped_weight(base, auxiliary, logits, requested,
                                               extra['logit_gradient_ratio'])
                if reference_steps is not None and index == 0:
                    norms.update(parameter_effect(base, auxiliary, model.affinity_decoder.parameters(), weight))
                for direction in active:
                    active[direction] += details[direction]['selected_edges']
            manual_loss = base + weight * auxiliary
            if not bool(torch.isfinite(manual_loss)):
                raise FloatingPointError('Nonfinite manual loss')
            manual_loss.backward()
            losses = [float(base.detach()), float(auxiliary.detach())]
            del prediction, logits, base, auxiliary, manual_loss, features, restored
            with torch.no_grad():
                restored = restore_first(restorer, pb['image'], seed + 2000000)
                features = model.encoder(restored)
            plogits = model.affinity_decoder(features)['affinity_logits']
            pseudo_loss, _ = compute_affinity_loss(plogits, pb, loss_cfg, pseudo=True)
            if not bool(torch.isfinite(pseudo_loss)):
                raise FloatingPointError('Nonfinite pseudo loss')
            (cfg['pseudo_weight'] * pseudo_loss).backward()
            losses.append(float(pseudo_loss.detach()))
            grad = float(torch.nn.utils.clip_grad_norm_(model.affinity_decoder.parameters(),
                                                       cfg['grad_clip'], error_if_nonfinite=True))
            if grad <= 0:
                raise RuntimeError('Affinity has no gradient')
            if any(p.grad is not None for n, p in model.named_parameters()
                   if not n.startswith('affinity_decoder.')):
                raise RuntimeError('Frozen model has gradient')
            optimizer.step()
            updates += 1
            count += 1
            sums += losses
            row = dict(epoch=epoch, step=index + 1, updates=updates, seed=seed,
                       manual=raw['name'], pseudo=raw_pseudo['name'],
                       manual_draw=draw_receipt(raw), pseudo_draw=draw_receipt(raw_pseudo),
                       manual_bce=losses[0], auxiliary=losses[1], pseudo_bce=losses[2],
                       auxiliary_weight=weight, grad_norm=grad, connectivity=details, **norms)
            if reference_steps is not None:
                compare_draw(row, reference_steps[updates - 1])
            append_json(output / 'steps.jsonl', row)
            if index == 0 or smoke:
                print(json.dumps({'arm': arm, **row}), flush=True)
            del features, restored, plogits, pseudo_loss
        if not smoke and count != data_info['updates_per_epoch']:
            raise RuntimeError('Incomplete epoch')
        scheduler.step()
        row = dict(epoch=epoch, updates=updates, manual_bce=sums[0] / count,
                   auxiliary=sums[1] / count, pseudo_bce=sums[2] / count,
                   next_lr=optimizer.param_groups[0]['lr'], active_edges=active.copy(),
                   elapsed_seconds=time.time() - started,
                   peak_cuda_mib=torch.cuda.max_memory_allocated() / 1024**2)
        append_json(output / 'epochs.jsonl', row)
        print(json.dumps({'arm': arm, **row}), flush=True)
        # 小型续训状态覆盖保存；过程权重仅含affinity，不重复写冻结基座。
        checkpoint = dict(format='affinity_connectivity_head_v1', epoch=epoch, arm=arm,
                          geometry_state_dict=model.affinity_decoder.state_dict(),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          initialization=initialization, config=config)
        torch.save(checkpoint, output / 'last_head.pt')
        if epoch % 5 == 0:
            torch.save({k: v for k, v in checkpoint.items() if k not in ('optimizer', 'scheduler')},
                       output / f'head_{epoch:03d}.pt')
        if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == epochs:
            save_monitor(model, restorer, config, device, output, epoch, names)
        status.update(status='running', **row)
        write_json(output / 'status.json', status)
    frozen_ok = frozen_digest(model) == frozen_start
    restorer_ok = tensor_digest(restorer.state_dict().items()) == restorer_start
    head_changed = tensor_digest(model.affinity_decoder.state_dict().items()) != initial_head
    if not frozen_ok or not restorer_ok or not head_changed:
        raise RuntimeError('Frozen-state or trainable-state audit failed')
    final_state = tensor_digest(model.state_dict().items())
    final_path = output / 'final.pt'
    save_backend(model, config, initialization, final_path, epochs,
                 extra={'arm': arm, 'updates': updates, 'smoke': smoke,
                        'deployment': 'restored_geometry_raw_final_vote', 'data': data_info})
    del checkpoint, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(final_path, config, device)
    if tensor_digest(reloaded.state_dict().items()) != final_state:
        raise RuntimeError('Strict save/reload changed weights')
    status.update(status='complete', frozen_unchanged=frozen_ok, restorer_unchanged=restorer_ok,
                  affinity_changed=head_changed, strict_reload_equal=True,
                  final_checkpoint=str(final_path), final_state_sha256=final_state,
                  active_edges=active, both_directions_active=all(v > 0 for v in active.values()))
    if reference_steps is not None:
        status['paired_control_updates'] = updates
    write_json(output / 'status.json', status)
    print(json.dumps({'status': 'complete', 'arm': arm, 'active_edges': active}), flush=True)
    if smoke:
        # 短测已做完整严格重载，保留小head与审计记录即可。
        del reloaded
        final_path.unlink()


@torch.no_grad()
def inference(config, checkpoint, output):
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    model, bundle = load_backend(checkpoint, config, device)
    restorer = get_restorer(config, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    files = list_images(project_path(config, config['inference']['test_dir']))
    if len(files) != 100:
        raise RuntimeError('Expected the complete 100-image semifinal cohort')
    rows = []
    for index, filename in enumerate(files):
        path = Path(filename)
        prediction = predict_cross(model, restorer, path, config, device,
            config['backend_adaptation']['restoration']['inference_seed'] + index)
        instances, classes = prediction['instances'], prediction['classes']
        if instances.dtype != np.uint16 or instances.ndim != 2 or int(instances.max()) > 65535:
            raise RuntimeError('Invalid final instance format')
        if {str(int(i)) for i in np.unique(instances) if i} != set(classes):
            raise RuntimeError('Instance/class IDs differ')
        if not cv2.imwrite(str(output / f'{path.stem}_inst.png'), instances):
            raise RuntimeError('Failed to write instance PNG')
        write_json(output / f'{path.stem}_class.json', classes)
        rows.append(dict(image=path.name, instances=len(classes),
                         ferrite=sum(v == 1 for v in classes.values()),
                         max_instance_id=int(instances.max())))
        print(f'inference {path.stem}: {len(classes)}', flush=True)
    write_json(output / 'manifest.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        epoch=bundle['epoch'], config=config, deployment='restored_geometry_raw_final_vote',
        sampling_mode='first_x0', precision='FP32', images=rows, official_score=None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train/affinity_connectivity.yaml')
    parser.add_argument('--arm', choices=('control', 'candidate'), required=True)
    parser.add_argument('--output')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or project_path(config, config['backend_adaptation']['output_dir'], args.arm)
    if args.infer:
        if not args.checkpoint or not args.output:
            parser.error('--infer requires --checkpoint and --output')
        inference(config, args.checkpoint, output)
    else:
        train(config, args.arm, output, smoke=args.smoke)


if __name__ == '__main__':
    main()
