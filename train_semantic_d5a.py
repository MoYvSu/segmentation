# -*- coding: utf-8 -*-
"""冻结 D5a/SAM2/LoRA/affinity，仅训练完整或从零初始化的简化语义头。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.mim_dataset import list_images
from models.backend_adaptation import build_backend, load_backend, restore_first, save_backend
from models.fpn_decoder import FPNDecoder
from models.rgb_restoration import load_rgb_restorer
from train_backend_adaptation import prediction_maps, save_prediction, sha, tensor_digest, write_json
from train_direct_semantic_affinity import build_semantic_criterion, compute_semantic_loss, move_batch, set_seed
from utils.affinity_deployment import prepare_image
from utils.config import load_config, project_path


def frozen_digest(model):
    """包括 LoRA 和 running statistics；语义头之外全部保持不变。"""
    return tensor_digest((name, value) for name, value in model.state_dict().items()
                         if not name.startswith('semantic_decoder.'))


def configure_arm(model, metadata, arm, seed):
    if arm not in ('full', 'simple'):
        raise ValueError(f'Unknown semantic arm: {arm}')
    before = tensor_digest(model.semantic_decoder.state_dict().items())
    if arm == 'simple':
        # 仅重置目标模块，避免全模型重新初始化意外改动冻结 encoder/affinity。
        device = next(model.parameters()).device
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            model.semantic_decoder.semantic_residual = None
            model.semantic_decoder.semantic_residual_version = 'none'
            modules = [model.semantic_decoder.seg_fpn, model.semantic_decoder.seg_branch]
            FPNDecoder._reset_modules(modules)
            for module in modules:
                for layer in module.modules():
                    if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                        layer.reset_running_stats()
        decoder_cfg = metadata['architecture']['semantic_decoder']
        decoder_cfg.update(semantic_residual=False, semantic_residual_version='none')
    elif model.semantic_decoder.semantic_residual is None:
        raise ValueError('Full arm requires the existing high-resolution RGB residual')
    model.eval().requires_grad_(False)
    model.encoder.trainable_lora = False
    model.semantic_decoder.requires_grad_(True)
    after = tensor_digest(model.semantic_decoder.state_dict().items())
    if arm == 'full' and before != after:
        raise RuntimeError('Full semantic initialization changed')
    if arm == 'simple' and before == after:
        raise RuntimeError('Simple semantic initialization was not reset')
    metadata['semantic_adaptation'] = {
        'arm': arm, 'initialization': 'existing_full_head' if arm == 'full' else 'random_fpn_and_classifier',
        'source_semantic_sha256': before, 'initial_semantic_sha256': after,
        'random_initialization_seed': int(seed) if arm == 'simple' else None,
        'frozen': ['D5a', 'SAM2', 'LoRA', 'affinity'],
        'direct_rgb_residual': arm == 'full',
    }
    return metadata['semantic_adaptation']


def semantic_train_mode(model):
    model.eval()
    model.semantic_decoder.train()


def get_restorer(config, device):
    cfg = config['semantic_adaptation']['restoration']
    path = Path(project_path(config, cfg['checkpoint']))
    if sha(path) != cfg['checkpoint_sha256']:
        raise RuntimeError('D5a checkpoint differs from the designated epoch 60 model')
    if cfg['mode'] != 'first_x0':
        raise ValueError('This comparison requires frozen D5a first_x0 output')
    return load_rgb_restorer(str(path), device).eval().requires_grad_(False)


def build_dataset(config):
    cfg = config['semantic_adaptation']
    base = CanonicalBackendDataset(project_path(config, config['paths']['raw_data_dir']),
        project_path(config, cfg['completed_gt_dir']), image_size=1024, affinity_grid=512)
    if len(base) != cfg['expected_manual_sources']:
        raise ValueError(f'Expected all {cfg["expected_manual_sources"]} sources, got {len(base)}')
    degradation = load_config(project_path(config, cfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
    dataset = PairedDegradationDataset(base, degradation, cfg['noise'], cfg['seed'],
                                      repeats=cfg['manual_repeats'])
    target_digest = hashlib.sha256()
    for image in base.samples:
        for suffix in ('_gt.npz', '_class.json'):
            path = base.completed_gt_dir / (image.stem + suffix)
            target_digest.update(path.name.encode('utf-8'))
            target_digest.update(bytes.fromhex(sha(path)))
    return dataset, {
        'manual_sources': len(base), 'source_names': [path.name for path in base.samples],
        'completed_gt_sha256': target_digest.hexdigest(),
        'completed_gt_dir': cfg['completed_gt_dir'],
        'manual_repeats': cfg['manual_repeats'], 'updates_per_epoch': len(dataset),
        'split_policy': 'all_manual_no_holdout', 'sam2_pseudo_sources': 0,
        'canonical_targets': 'completed_instance_map_and_class_lookup',
        'degradation': degradation, 'noise': cfg['noise'],
    }


def learning_rate_factor(epoch, total_epochs, warmup_epochs, minimum_ratio):
    if epoch <= warmup_epochs:
        return epoch / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs - 1) / max(1, total_epochs - warmup_epochs - 1)
    return minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2


def aligned_semantic_logits(logits, batch):
    size = batch['semantic_target'].shape[-2:]
    if logits.shape[-2:] != size:
        logits = F.interpolate(logits.float(), size=size, mode='bilinear', align_corners=True)
    return logits.float()


@torch.no_grad()
def monitor(model, restorer, config, output, epoch):
    cfg = config['semantic_adaptation']
    files = [Path(path) for path in list_images(project_path(config, config['inference']['test_dir']))]
    target = Path(output) / 'monitor' / f'epoch_{epoch:03d}'
    target.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    devices = [device.index or 0] if device.type == 'cuda' else []
    found = []
    with torch.random.fork_rng(devices=devices):
        for index, path in enumerate(files):
            if path.stem not in cfg['monitor']['images']:
                continue
            image, semantic, boundary = prediction_maps(model, path, config, device, restorer,
                cfg['restoration']['inference_seed'] + index)
            sem = torch.sigmoid(semantic)[0, 0].numpy()
            edge = boundary[0, 0].numpy()
            colours = np.where((sem >= .5)[..., None], np.array([64, 190, 244]), np.array([226, 125, 70])).astype(np.uint8)
            overlay = (.62 * image + .38 * colours).astype(np.uint8)
            probability = cv2.cvtColor(cv2.applyColorMap(np.rint(sem * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            heat = cv2.cvtColor(cv2.applyColorMap(np.rint(edge * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            panels = []
            for content, title in ((image, f'{path.stem} e{epoch}'), (overlay, 'Semantic P(F) >= 0.5'),
                                   (probability, 'P(F): fixed 0..1'), (heat, 'Frozen affinity: 0..1')):
                thumb = cv2.resize(content, (256, round(content.shape[0] * 256 / content.shape[1])), interpolation=cv2.INTER_AREA)
                thumb = cv2.copyMakeBorder(thumb, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(thumb, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
                panels.append(thumb)
            if not cv2.imwrite(str(target / f'{path.stem}.png'), cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)):
                raise IOError(f'Failed to write process monitor: {path.stem}')
            found.append(path.stem)
    if set(found) != set(cfg['monitor']['images']):
        raise ValueError('A configured process monitor image is missing')


def inference(config, checkpoint, arm, output_dir, smoke=False):
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    model, payload = load_backend(checkpoint, config, device)
    saved_arm = payload['backend_adaptation'].get('semantic_adaptation', {}).get('arm')
    if saved_arm != arm:
        raise ValueError(f'Checkpoint arm {saved_arm!r} differs from requested {arm!r}')
    restorer = get_restorer(config, device)
    cfg = config['semantic_adaptation']
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = [Path(path) for path in list_images(project_path(config, config['inference']['test_dir']))]
    results = []
    for index, path in enumerate(files):
        if smoke and path.stem not in cfg['monitor']['images']:
            continue
        image, semantic, boundary = prediction_maps(model, path, config, device, restorer,
                                                    cfg['restoration']['inference_seed'] + index)
        instances, classes = save_prediction(image, semantic, boundary, config, output, path.stem)
        if instances.dtype != np.uint16 or int(instances.max()) > 65535:
            raise ValueError('Final instance output violates the uint16 contract')
        results.append({'image': path.name, 'instances': len(classes), 'max_instance_id': int(instances.max()),
            'ferrite': sum(int(c) == 1 for c in classes.values()),
            'pearlite': sum(int(c) == 0 for c in classes.values())})
        print(f'inference {arm}: {path.name}', flush=True)
    write_json(output / 'manifest.json', {'arm': arm, 'epoch': payload['epoch'], 'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha(checkpoint), 'd5a_sha256': cfg['restoration']['checkpoint_sha256'],
        'sampling_mode': 'first_x0', 'precision': 'FP32', 'images': results, 'config': config, 'official_score': None})


def train(config, arm, output_dir, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env on the GPU server')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    cfg = config['semantic_adaptation']
    if cfg['amp_dtype'] != 'bfloat16' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This comparison requires BF16 semantic training')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    set_seed(cfg['seed'])
    model, metadata = build_backend(config, device)
    frozen_initial = frozen_digest(model)
    initialization = configure_arm(model, metadata, arm, cfg['seed'])
    if frozen_digest(model) != frozen_initial:
        raise RuntimeError('Semantic initialization changed frozen weights')
    restorer = get_restorer(config, device)
    restoration_initial = tensor_digest(restorer.state_dict().items())
    dataset, data_info = build_dataset(config)
    total_parameters = model.parameter_summary()['total'] + sum(p.numel() for p in restorer.parameters())
    if total_parameters >= 500_000_000:
        raise ValueError('Combined D5a and segmentation parameters exceed the competition limit')
    before_trainable = {name: parameter.detach().cpu().clone()
                        for name, parameter in model.semantic_decoder.named_parameters()}
    parameters = list(model.semantic_decoder.parameters())
    if any(p.requires_grad for name, p in model.named_parameters() if not name.startswith('semantic_decoder.')):
        raise RuntimeError('Only the semantic decoder may be trainable')
    status = {'status': 'preflight', 'arm': arm, 'smoke': smoke, 'init': initialization,
        'sources': metadata['sources'], 'data': data_info, 'config': config,
        'frozen_state_sha256': frozen_initial, 'd5a_sha256': cfg['restoration']['checkpoint_sha256'],
        'd5a_state_sha256': restoration_initial, 'parameter_summary': model.parameter_summary(),
        'total_parameters': total_parameters, 'trainable_parameters': sum(p.numel() for p in parameters),
        'selection': 'prespecified_final_epoch_no_validation', 'updates': 0, 'failed_updates': 0,
        'precision': {'restorer': 'FP32', 'encoder': 'FP32', 'semantic_training': 'BF16', 'inference': 'FP32'}}
    write_json(output / 'status.json', status)
    all_files = [Path(path) for path in list_images(project_path(config, config['inference']['test_dir']))]
    first = all_files[0]
    image, semantic, boundary = prediction_maps(model, first, config, device, restorer, cfg['restoration']['inference_seed'])
    probe_dir = output / 'initial_probe'
    probe_dir.mkdir(parents=True, exist_ok=False)
    instances, classes = save_prediction(image, semantic, boundary, config, probe_dir, first.stem)
    if arm == 'full':
        original = Path(project_path(config, cfg['d5a_prediction_dir']))
        old_instances = cv2.imread(str(original / f'{first.stem}_inst.png'), cv2.IMREAD_UNCHANGED)
        old_classes = json.loads((original / f'{first.stem}_class.json').read_text(encoding='utf-8'))
        if (not np.array_equal(instances, old_instances)
                or {str(key): int(value) for key, value in classes.items()} != old_classes):
            raise RuntimeError('Full semantic initialization differs from the original D5a final prediction')
        status['initial_final_output_equal'] = True
    else:
        status['initial_final_output_equal'] = None
    status['initial_probe_image'] = first.name
    write_json(output / 'status.json', status)
    np.savez_compressed(probe_dir / 'maps.npz', semantic=semantic.numpy(), boundary=boundary.numpy())
    _, probe, _, _ = prepare_image(first, 1024, device)
    probe = restore_first(restorer, probe, cfg['restoration']['inference_seed'])
    with torch.no_grad():
        initial_affinity = model(probe)['affinity_logits'].cpu()
    monitor(model, restorer, config, output, 0)
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rates'][arm], weight_decay=cfg['weight_decay'], eps=1e-4)
    criterion = build_semantic_criterion(config, device)
    steps, failed = 0, 0
    epochs = 1 if smoke else int(cfg['epochs'])
    started = time.time()
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=cfg['num_workers'],
            generator=torch.Generator().manual_seed(cfg['seed'] + epoch * 2), pin_memory=True)
        lr = cfg['learning_rates'][arm] * learning_rate_factor(epoch, cfg['epochs'], cfg['warmup_epochs'], cfg['minimum_lr_ratio'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        semantic_train_mode(model)
        loss_total, count, endpoint_count = 0.0, 0, 0
        for index, raw in enumerate(loader):
            if smoke and index >= 2:
                break
            step_seed = cfg['seed'] + epoch * 10000 + index
            set_seed(step_seed)
            input_digest = tensor_digest([('image', raw['image'])])
            batch = move_batch(raw, device)
            restoration_seed = step_seed + 1000000
            batch['image'] = restore_first(restorer, batch['image'], restoration_seed)
            # 冻结 encoder 用部署相同 FP32 产生特征，仅语义头启用 BF16 训练。
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                features = model.encoder(batch['image'].float())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model.semantic_decoder(features, batch['image'])
            loss = compute_semantic_loss(criterion, aligned_semantic_logits(logits, batch), batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite semantic loss; run stopped without skipping data')
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, cfg['grad_clip'], error_if_nonfinite=True))
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise FloatingPointError(f'Invalid semantic gradient norm: {grad_norm}')
            optimizer.step()
            steps += 1
            loss_value = float(loss.detach())
            loss_total += loss_value
            count += 1
            endpoint_applied = bool(raw['endpoint_applied'].item())
            endpoint_count += int(endpoint_applied)
            row = {'epoch': epoch, 'step': index + 1, 'updates': steps, 'name': raw['name'][0],
                'draw_index': int(raw['draw_index'].item()), 'input_sha256': input_digest,
                'restoration_seed': restoration_seed, 'seed': step_seed, 'semantic_loss': loss_value,
                'grad_norm': grad_norm, 'profile': raw['profile'][0],
                'is_spatial': bool(raw['is_spatial'].item()), 'endpoint_applied': endpoint_applied,
                'noise_sigma': float(raw['noise_sigma'].item())}
            with (output / 'steps.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            del features, logits, loss, batch
        if count != (2 if smoke else data_info['updates_per_epoch']):
            raise RuntimeError('Incomplete training epoch')
        row = {'epoch': epoch, 'updates': steps, 'failed_updates': failed, 'semantic_loss': loss_total / count,
            'learning_rate': lr, 'endpoint_draws': endpoint_count, 'elapsed_seconds': time.time() - started,
            'peak_cuda_mib': torch.cuda.max_memory_allocated() / 1024**2}
        with (output / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({'arm': arm, **row}), flush=True)
        if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == epochs:
            monitor(model, restorer, config, output, epoch)
        extra = {'arm': arm, 'updates': steps, 'failed_updates': failed, 'smoke': smoke,
            'data': data_info, 'optimizer': optimizer.state_dict()}
        save_backend(model, config, metadata, output / 'last.pt', epoch, extra=extra)
        if epoch == epochs:
            save_backend(model, config, metadata, output / f'epoch_{epoch:03d}.pt', epoch,
                         extra={key: value for key, value in extra.items() if key != 'optimizer'})
        status.update(status='running', **row)
        write_json(output / 'status.json', status)
    if frozen_digest(model) != frozen_initial or tensor_digest(restorer.state_dict().items()) != restoration_initial:
        raise RuntimeError('Frozen encoder/LoRA/affinity/D5a changed during semantic training')
    deltas = {}
    for name, parameter in model.semantic_decoder.named_parameters():
        group = name.split('.')[0]
        delta = float((parameter.detach().cpu() - before_trainable[name]).abs().max())
        deltas[group] = max(deltas.get(group, 0.0), delta)
    if any(value <= 0 for value in deltas.values()):
        raise RuntimeError(f'A semantic parameter group did not change: {deltas}')
    model.eval()
    with torch.no_grad():
        expected = {key: value.cpu() for key, value in model(probe).items()}
    if not torch.equal(expected['affinity_logits'], initial_affinity):
        raise RuntimeError('Frozen affinity inference changed')
    del optimizer, parameters, before_trainable, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(output / f'epoch_{epochs:03d}.pt', config, device)
    with torch.no_grad():
        actual = reloaded(probe)
    maximum = {key: float((actual[key].cpu() - expected[key]).abs().max()) for key in expected}
    if any(value > 1e-6 for value in maximum.values()):
        raise RuntimeError(f'Strict checkpoint reload changed outputs: {maximum}')
    status.update(status='completed', frozen_weights_unchanged=True, d5a_unchanged=True,
        frozen_affinity_output_equal=True, strict_reload_passed=True, semantic_weights_changed=True,
        trainable_max_deltas=deltas, reload_logit_max_delta=maximum,
        final_checkpoint=str(output / f'epoch_{epochs:03d}.pt'))
    write_json(output / 'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_d5a.yaml')
    parser.add_argument('--arm', choices=('full', 'simple'), required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=('train', 'infer'), default='train')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--prediction-dir')
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(project_path(config, args.prediction_dir or args.output_dir))
    try:
        if args.mode == 'infer':
            if not args.checkpoint:
                parser.error('--mode infer requires --checkpoint')
            inference(config, project_path(config, args.checkpoint), args.arm, output, args.smoke)
        else:
            if args.checkpoint or args.prediction_dir:
                parser.error('--checkpoint/--prediction-dir require --mode infer')
            train(config, args.arm, output, args.smoke)
    except BaseException as error:
        if output.exists():
            write_json(output / 'failure.json', {'error': repr(error), 'arm': args.arm})
        raise


if __name__ == '__main__':
    main()
