# -*- coding: utf-8 -*-
"""固定 D4 首步开关的后端配对微调；全人工训练，无验证集与代理选优。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.mim_dataset import list_images
from data.sam2_geometry_dataset import SAM2GeometryDataset
from models.backend_adaptation import build_backend, load_backend, restore_first, save_backend
from models.direct_semantic_affinity import configure_direct_training_phase, direct_parameter_groups
from models.lora import enable_trunk_gradient_checkpointing
from models.rgb_restoration import load_rgb_restorer
from train_direct_semantic_affinity import build_semantic_criterion, compute_affinity_loss, compute_semantic_loss, move_batch, set_seed
from utils.affinity_deployment import crop_affinity_boundary_output, crop_letterbox_output, postprocess, prepare_image, probability_to_logit
from utils.config import load_config, project_path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    for name, value in sorted(named_tensors):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(model):
    return tensor_digest((n, p) for n, p in model.encoder.named_parameters()
                         if 'lora_A' not in n and 'lora_B' not in n)


def get_restorer(config, device):
    cfg = config['backend_adaptation']['restoration']
    path = Path(project_path(config, cfg['checkpoint']))
    if sha(path) != cfg['checkpoint_sha256']:
        raise RuntimeError('D4 checkpoint differs from the submitted e60 model')
    model = load_rgb_restorer(str(path), device).eval().requires_grad_(False)
    if model.num_steps != 16 or model.kappa != 0.03 or cfg['mode'] != 'first_x0':
        raise ValueError('Expected the D4 first-x0 sampling contract')
    return model


def build_datasets(config):
    cfg = config['backend_adaptation']
    manual = CanonicalBackendDataset(
        project_path(config, config['paths']['raw_data_dir']),
        project_path(config, cfg['completed_gt_dir']), image_size=1024, affinity_grid=512)
    if len(manual) != cfg['expected_manual_sources']:
        raise ValueError(f'Expected all {cfg["expected_manual_sources"]} manual sources, got {len(manual)}')
    sam_cfg = cfg['sam2_geometry']
    directory = Path(project_path(config, sam_cfg['dataset_dir']))
    manifest = directory / 'manifest.jsonl'
    # 复用既有用户授权的完全相同64源清单；不虚构缺失的历史approval.json。
    if sha(manifest) != sam_cfg['manifest_sha256']:
        raise ValueError('SAM2 source manifest no longer matches the historically authorized set')
    if not Path(project_path(config, sam_cfg['reuse_authorization_doc'])).is_file():
        raise FileNotFoundError('Missing existing authorization evidence')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if (len(rows) != sam_cfg['expected_sources'] or
            len({r['source_sha256'] for r in rows}) != len(rows) or
            any(r.get('class_label') is not None for r in rows)):
        raise ValueError('SAM2 sources must remain the same unique class-agnostic set')
    pseudo = SAM2GeometryDataset(directory, project_path(config, sam_cfg['source_dir']),
                                image_size=1024, output_grid=512,
                                use_eroded_interiors=False, cache_in_memory=False)
    if len(pseudo) != len(rows):
        raise ValueError('SAM2 mask count differs from its manifest')
    degradation = load_config(project_path(config, cfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
    common = dict(degradation=degradation, noise=cfg['noise'])
    augmented_manual = PairedDegradationDataset(manual, seed=cfg['seed'], repeats=cfg['manual_repeats'], **common)
    augmented_pseudo = PairedDegradationDataset(pseudo, seed=cfg['seed'] + 17, **common)
    if len(augmented_manual) != len(augmented_pseudo):
        raise ValueError('Each update requires one manual and one distinct pseudo draw')
    return augmented_manual, augmented_pseudo, {
        'manual_sources': len(manual), 'sam2_sources': len(pseudo),
        'manual_repeats': cfg['manual_repeats'], 'updates_per_epoch': len(augmented_manual),
        'split_policy': 'all_manual_no_holdout',
        'canonical_targets': 'completed_instance_map_and_class_lookup_for_all_manual_targets',
        'sam2_manifest_sha256': sam_cfg['manifest_sha256'],
        'sam2_reuse_authorization_doc': sam_cfg['reuse_authorization_doc'],
        'degradation': degradation, 'noise': cfg['noise'],
    }


def fusion_options(config):
    deploy = config['affinity_deployment']
    return str(deploy.get('fusion_mode', 'gated')), {key: deploy.get(key, value) for key, value in {
        'distance2_weight': .5, 'distance4_weight': .25, 'support_threshold': .2,
        'support_temperature': .05, 'short_reduction': 'top2', 'short_softmax_temperature': .15}.items()}


@torch.no_grad()
def prediction_maps(model, path, config, device, restorer=None, seed=314159):
    image, tensor, ph, pw = prepare_image(path, 1024, device)
    if restorer is not None:
        tensor = restore_first(restorer, tensor, seed)
    output = model(tensor)
    semantic = crop_letterbox_output(output['semantic_logits'], 1024, ph, pw, image.shape[:2]).cpu()
    mode, kwargs = fusion_options(config)
    boundary = crop_affinity_boundary_output(output, 1024, ph, pw, image.shape[:2], mode, kwargs).cpu()
    return image, semantic, boundary


def save_prediction(image, semantic, boundary, config, directory, stem):
    result = torch.cat([semantic, probability_to_logit(boundary)], dim=1)
    _, instances, classes = postprocess(result, image.shape[:2], directory, stem, config['inference'],
        float(config['inference']['boundary_threshold']), False, image_rgb=image, semantic_challenger_logits=None)
    return instances, classes


@torch.no_grad()
def monitor(model, restorer, config, output_dir, epoch):
    """固定真实图仅做缩略观察，不算loss、不选权重、不影响训练随机序列。"""
    cfg = config['backend_adaptation']
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    target = output_dir / 'monitor' / f'epoch_{epoch:03d}'
    target.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    with torch.random.fork_rng(devices=[device.index or 0]):
        for index, path in enumerate(files):
            if path.stem not in cfg['monitor']['images']:
                continue
            image, semantic, boundary = prediction_maps(model, path, config, device, restorer,
                                                        cfg['restoration']['inference_seed'] + index)
            sem = torch.sigmoid(semantic)[0, 0].numpy()
            edge = boundary[0, 0].numpy()
            colours = np.where((sem >= .5)[..., None], np.array([64, 190, 244]), np.array([226, 125, 70])).astype(np.uint8)
            overlay = (.62 * image + .38 * colours).astype(np.uint8)
            heat = cv2.cvtColor(cv2.applyColorMap(np.rint(edge * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            panels = []
            for content, title in ((image, f'{path.stem} e{epoch}'), (overlay, 'Semantic probability >= 0.5'), (heat, 'Boundary probability')):
                thumb = cv2.resize(content, (320, round(content.shape[0] * 320 / content.shape[1])), interpolation=cv2.INTER_AREA)
                thumb = cv2.copyMakeBorder(thumb, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(thumb, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1, cv2.LINE_AA)
                panels.append(thumb)
            cv2.imwrite(str(target / f'{path.stem}.png'), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR))


def inference(config, checkpoint, arm, output_dir, smoke=False):
    device = torch.device('cuda')
    model, payload = load_backend(checkpoint, config, device)
    restorer = get_restorer(config, device) if arm == 'd4' else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    results = []
    for index, path in enumerate(files):
        if smoke and path.stem not in config['backend_adaptation']['monitor']['images']:
            continue
        image, semantic, boundary = prediction_maps(model, path, config, device, restorer,
            config['backend_adaptation']['restoration']['inference_seed'] + index)
        instances, classes = save_prediction(image, semantic, boundary, config, output_dir, path.stem)
        results.append({'image': path.name, 'instances': len(classes), 'max_instance_id': int(instances.max()),
                        'ferrite': sum(int(c) == 1 for c in classes.values()),
                        'pearlite': sum(int(c) == 0 for c in classes.values())})
        print(f'inference {arm}: {path.name}', flush=True)
    write_json(output_dir / 'manifest.json', {'arm': arm, 'epoch': payload['epoch'], 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha(checkpoint), 'sampling_mode': 'first_x0' if restorer is not None else 'none',
        'precision': 'FP32', 'images': results, 'config': config, 'official_score': None})


def train(config, arm, output_dir, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env on the GPU server')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    cfg = config['backend_adaptation']
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    set_seed(cfg['seed'])
    model, initialization = build_backend(config, device)
    initial_digest = tensor_digest(model.state_dict().items())
    frozen_initial = frozen_digest(model)
    restorer = get_restorer(config, device) if arm == 'd4' else None
    restoration_initial = tensor_digest(restorer.state_dict().items()) if restorer is not None else None
    manual, pseudo, data_info = build_datasets(config)
    sources = {'initialization': initialization, 'initial_state_sha256': initial_digest,
               'frozen_encoder_sha256': frozen_initial, 'restorer_state_sha256': restoration_initial,
               'arm': arm, 'data': data_info}
    expected_parameter_count = model.parameter_summary()['total'] + (sum(p.numel() for p in restorer.parameters()) if restorer is not None else 0)
    if expected_parameter_count >= 500_000_000:
        raise RuntimeError('Model parameter limit exceeded')
    status = {'status': 'preflight', 'arm': arm, 'smoke': smoke, 'sources': sources, 'config': config,
              'total_parameters': expected_parameter_count, 'selection': 'prespecified_final_epoch_no_validation'}
    write_json(output / 'status.json', status)
    # 真实首图的完整输出必须复现各自既有baseline，而不只检查网络张量。
    all_files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    first = all_files[0]
    image, sem, boundary = prediction_maps(model, first, config, device, restorer, cfg['restoration']['inference_seed'])
    probe_dir = output / 'initial_probe'
    inst, classes = save_prediction(image, sem, boundary, config, probe_dir, first.stem)
    old_dir = Path(project_path(config, cfg['d4_prediction_dir' if arm == 'd4' else 'baseline_prediction_dir']))
    old_inst = cv2.imread(str(old_dir / f'{first.stem}_inst.png'), -1)
    old_classes = json.loads((old_dir / f'{first.stem}_class.json').read_text(encoding='utf-8'))
    if not np.array_equal(inst, old_inst) or {str(k): int(v) for k, v in classes.items()} != old_classes:
        raise RuntimeError('Initial fused backend does not reproduce the deployed final prediction')
    status['initial_final_output_equal'] = True
    configure_direct_training_phase(model, train_lora=True)
    enable_trunk_gradient_checkpointing(model.encoder.trunk)
    groups = direct_parameter_groups(model, cfg['learning_rates'])
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda epoch: cfg['minimum_lr_ratio'] + (1 - cfg['minimum_lr_ratio']) * (1 + math.cos(math.pi * epoch / cfg['epochs'])) / 2)
    criterion = build_semantic_criterion(config, device)
    loss_cfg = config['direct_semantic_affinity']['affinity_loss']
    if cfg['amp_dtype'] != 'bfloat16' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This paired run requires BF16 support; no silent precision fallback')
    # 记录实际参数变化；基座与修复器完整摘要在训练末尾再次核验。
    group_names = {g['name']: {id(p) for p in g['params']} for g in groups}
    before_trainable = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    status['trainable_parameters'] = {g['name']: sum(p.numel() for p in g['params']) for g in groups}
    monitor(model, restorer, config, output, 0)
    steps, failed = 0, 0
    started = time.time()
    epochs = 1 if smoke else cfg['epochs']
    for epoch in range(1, epochs + 1):
        manual.set_epoch(epoch)
        pseudo.set_epoch(epoch)
        loaders = [DataLoader(dataset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'],
            generator=torch.Generator().manual_seed(cfg['seed'] + epoch * 2 + i), pin_memory=True)
            for i, dataset in enumerate((manual, pseudo))]
        model.train()
        model.encoder.trunk.eval()
        totals = np.zeros(3, np.float64)
        count = 0
        for index, (raw, raw_pseudo) in enumerate(zip(*loaders)):
            if smoke and index >= 2:
                break
            # 每个训练步骤重新固定dropout流；D4与数据增强均不消耗该全局流。
            step_seed = cfg['seed'] + epoch * 10000 + index
            set_seed(step_seed)
            batch, pb = move_batch(raw, device), move_batch(raw_pseudo, device)
            optimizer.zero_grad(set_to_none=True)
            if restorer is not None:
                batch['image'] = restore_first(restorer, batch['image'], step_seed + 1000000)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(batch['image'])
                sem_loss = compute_semantic_loss(criterion, prediction['semantic_logits'].float(), batch)
                aff_loss, _ = compute_affinity_loss(prediction['affinity_logits'].float(), batch, loss_cfg, pseudo=False)
                manual_loss = sem_loss + aff_loss
            if not bool(torch.isfinite(manual_loss)):
                raise FloatingPointError('Non-finite manual loss')
            manual_loss.backward()
            del prediction, manual_loss
            # 两张图先后反向，避免同时保留两份encoder激活。
            if restorer is not None:
                pb['image'] = restore_first(restorer, pb['image'], step_seed + 2000000)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = model.encoder(pb['image'])
                logits = model.affinity_decoder(features)['affinity_logits'].float()
                pseudo_loss, _ = compute_affinity_loss(logits, pb, loss_cfg, pseudo=True)
            if not bool(torch.isfinite(pseudo_loss)):
                raise FloatingPointError('Non-finite pseudo loss')
            (cfg['pseudo_weight'] * pseudo_loss).backward()
            del features, logits
            grad_norms = {g['name']: float(torch.stack([p.grad.float().norm() for p in g['params'] if p.grad is not None]).norm()) for g in groups}
            if any(not math.isfinite(v) or v <= 0 for v in grad_norms.values()):
                raise FloatingPointError(f'Invalid gradient groups: {grad_norms}')
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses = [float(sem_loss.detach()), float(aff_loss.detach()), float(pseudo_loss.detach())]
            totals += losses
            count += 1
            row = {'epoch': epoch, 'step': index + 1, 'updates': steps,
                   'manual': raw['name'], 'pseudo': raw_pseudo['name'], 'semantic_loss': losses[0],
                   'affinity_loss': losses[1], 'pseudo_loss': losses[2], 'grad_norms': grad_norms,
                   'seed': step_seed}
            with (output / 'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        if not smoke and count != data_info['updates_per_epoch']:
            raise RuntimeError('Incomplete training epoch')
        scheduler.step()
        row = {'epoch': epoch, 'updates': steps, 'failed_updates': failed,
               'semantic_loss': totals[0] / count, 'affinity_loss': totals[1] / count,
               'pseudo_loss': totals[2] / count, 'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
               'elapsed_seconds': time.time() - started, 'peak_cuda_mib': torch.cuda.max_memory_allocated() / 1024**2}
        with (output / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'arm': arm, **row}), flush=True)
        if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == epochs:
            monitor(model, restorer, config, output, epoch)
        extra = {'arm': arm, 'updates': steps, 'failed_updates': failed, 'smoke': smoke,
                 'data': data_info, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        save_backend(model, config, initialization, output / 'last.pt', epoch, extra=extra)
        if epoch in (20, 40, epochs):
            save_backend(model, config, initialization, output / f'epoch_{epoch:03d}.pt', epoch,
                         extra={k: v for k, v in extra.items() if k not in ('optimizer', 'scheduler')})
        status.update(status='running', **row)
        write_json(output / 'status.json', status)
    frozen_after = frozen_digest(model)
    restoration_after = tensor_digest(restorer.state_dict().items()) if restorer is not None else None
    if frozen_after != frozen_initial or restoration_after != restoration_initial:
        raise RuntimeError('Frozen backbone or D4 changed during finetuning')
    deltas = {group: 0.0 for group in group_names}
    for name, parameter in model.named_parameters():
        if name in before_trainable:
            delta = float((parameter.detach().cpu() - before_trainable[name]).abs().max())
            for group, ids in group_names.items():
                if id(parameter) in ids:
                    deltas[group] = max(deltas[group], delta)
    if any(v <= 0 for v in deltas.values()):
        raise RuntimeError('A trainable group did not change')
    # 真正重建模型严格读取保存后的完整权重，并比较FP32 logits。
    model.eval()
    _, probe, _, _ = prepare_image(first, 1024, device)
    with torch.no_grad():
        if restorer is not None:
            probe = restore_first(restorer, probe, cfg['restoration']['inference_seed'])
        expected = {k: v.cpu() for k, v in model(probe).items()}
    del optimizer, groups, before_trainable, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(output / f'epoch_{epochs:03d}.pt', config, device)
    with torch.no_grad():
        actual = reloaded(probe)
    maximum = {k: float((actual[k].cpu() - expected[k]).abs().max()) for k in expected}
    if any(v > 1e-6 for v in maximum.values()):
        raise RuntimeError(f'Checkpoint reload changed logits: {maximum}')
    status.update(status='completed', frozen_encoder_unchanged=True, d4_unchanged=True,
                  trainable_max_deltas=deltas, reload_logit_max_delta=maximum,
                  final_checkpoint=str(output / f'epoch_{epochs:03d}.pt'))
    write_json(output / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/backend_d4.yaml')
    parser.add_argument('--arm', choices=('d4', 'raw'), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--checkpoint', help='有此参数则只做完整部署推理')
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(project_path(config, args.output_dir))
    try:
        if args.checkpoint:
            inference(config, project_path(config, args.checkpoint), args.arm, output, args.smoke)
        else:
            train(config, args.arm, output, args.smoke)
    except BaseException as error:
        if output.exists():
            write_json(output / 'failure.json', {'error': repr(error), 'arm': args.arm})
        raise


if __name__ == '__main__':
    main()
