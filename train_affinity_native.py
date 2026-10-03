# -*- coding: utf-8 -*-
"""复用全图control的初始化和预算，独立训练两种原生裁块affinity。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.affinity_native import build_datasets
from data.mim_dataset import list_images
from models.backend_adaptation import build_backend, load_backend, restore_first, save_backend
from tools.affinity_native_views import predict_views, save_monitor, save_training_draw, verify_prediction
from tools.analyze_affinity_connectivity import load_directory
from train_affinity_connectivity import configure_training, frozen_digest, get_restorer, initial_parity, draw_receipt, append_json
from train_backend_adaptation import sha, tensor_digest, write_json
from train_direct_semantic_affinity import compute_affinity_loss, move_batch, set_seed
from utils.config import load_config, project_path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def common_config(config):
    result = deepcopy(config)
    result.pop('affinity_native', None)
    result['backend_adaptation'].pop('output_dir', None)
    return json.loads(json.dumps(result))


def verify_control(config, state=None):
    opt = config['affinity_native']
    root = Path(project_path(config, opt['reused_control']))
    prior = read(root / 'status.json')
    expected = config['backend_adaptation']['epochs'] * 64
    if prior['status'] != 'complete' or prior['updates'] != expected or sha(root/'final.pt') != opt['control_sha256']:
        raise RuntimeError('Reused full-image control is incomplete or has changed')
    if common_config(config) != common_config(prior['config']):
        raise RuntimeError('Non-crop configuration differs from reused control')
    for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'):
        if prior.get(key) is not True:
            raise RuntimeError('Control audit failed: ' + key)
    steps = [json.loads(line) for line in (root/'steps.jsonl').read_text(encoding='utf8').splitlines()]
    if len(steps) != expected or any(r['auxiliary_weight'] != 0 for r in steps):
        raise RuntimeError('Invalid full-image control receipts')
    if state is not None:
        for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256'):
            if prior[key] != state[key]:
                raise RuntimeError('Control initialization or frozen state differs: ' + key)
        data = dict(prior['data'])
        data.pop('topology_supervision', None)
        if data != state['data']['parent_data']:
            raise RuntimeError('Original sources, losses or degradation configuration differ')
    return prior, steps


def compare_draw(row, control):
    # 裁窗位置与像素是预定变量；源图顺序、退化参数和随机噪声种子仍需相同。
    for key in ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw'):
        if row[key] != control[key]:
            raise RuntimeError('Control source/augmentation receipt differs: ' + key)


def audit_control(config, output):
    """确认新局部入口逐像素复现上轮冻结检查，不重训control。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prior, _ = verify_control(config)
    opt = config['affinity_native']
    model, _ = load_backend(project_path(config, opt['reused_control'], 'final.pt'), config, 'cuda')
    restorer = get_restorer(config, 'cuda')
    before = tensor_digest(model.state_dict().items())
    rows = []
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    for index, path in enumerate(files):
        if path.stem not in config['affinity_connectivity']['parity_images']:
            continue
        seed = config['backend_adaptation']['restoration']['inference_seed'] + index
        _, views = predict_views(model, restorer, path, config, 'cuda', seed, [0, *opt['sizes']])
        for name, view in views.items():
            directory = (project_path(config, opt['reused_control'], 'deployment') if name == 'global' else
                         project_path(config, opt['prior_probe'], path.stem, name + '_d5a'))
            ids, classes = load_directory(directory, path.stem)
            if not np.array_equal(view['instances'], ids) or view['classes'] != classes:
                raise RuntimeError(f'Frozen deployment parity failed: {path.stem}/{name}')
            rows.append(dict(source=path.stem, view=name, final_output_equal=True))
    expected = len(config['affinity_connectivity']['parity_images']) * (1+len(opt['sizes']))
    if len(rows) != expected or tensor_digest(model.state_dict().items()) != before:
        raise RuntimeError('Incomplete frozen deployment audit')
    write_json(output/'report.json', dict(passed=True, control_retrained=False, rows=rows,
               control_initial_state=prior['initial_state_sha256']))


def train(config, size, output, smoke=False, *, dataset_builder=None, control_verifier=None,
          keep_smoke_checkpoint=False, checkpoint_extras=None, monitor_saver=None):
    if not torch.cuda.is_available():
        raise RuntimeError('Use GPU server sam2_env')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    cfg, opt = config['backend_adaptation'], config['affinity_native']
    monitor = monitor_saver or save_monitor
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if cfg['num_workers'] != 0:
        raise ValueError('Native draw state currently requires num_workers=0')
    set_seed(cfg['seed'])
    device = torch.device('cuda')
    model, initialization = build_backend(config, device)
    restorer = get_restorer(config, device)
    total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in restorer.parameters())
    if total >= 500_000_000:
        raise RuntimeError('Model parameter limit exceeded')
    manual, pseudo, data = (dataset_builder or build_datasets)(config, size)
    initial_head = tensor_digest(model.affinity_decoder.state_dict().items())
    status = dict(status='preflight', size=size, smoke=smoke, config=config, initialization=initialization,
                  initial_state_sha256=tensor_digest(model.state_dict().items()),
                  frozen_state_sha256=frozen_digest(model),
                  restorer_state_sha256=tensor_digest(restorer.state_dict().items()),
                  data=data, total_parameters=total, precision='FP32',
                  selection='fixed_e20_all_sources_no_holdout', official_score=None)
    write_json(output/'status.json', status)
    _, reference = (control_verifier or verify_control)(config, status)
    status['reused_control_verified'] = True
    status['initial_parity'] = initial_parity(model, restorer, config, device)
    configure_training(model)
    optimizer = torch.optim.AdamW(model.affinity_decoder.parameters(),
                                 lr=cfg['learning_rates']['affinity'], weight_decay=cfg['weight_decay'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: cfg['minimum_lr_ratio'] +
        (1-cfg['minimum_lr_ratio'])*(1+math.cos(math.pi*e/cfg['epochs']))/2)
    monitor(model, restorer, config, device, output, 0, size, smoke=smoke)
    status['status'] = 'running'
    write_json(output/'status.json', status)
    loss_cfg = config['direct_semantic_affinity']['affinity_loss']
    updates, started = 0, time.time()
    epochs = 1 if smoke else cfg['epochs']
    for epoch in range(1, epochs+1):
        for ds in (manual, pseudo):
            ds.set_epoch(epoch)
        loaders = [DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
                             generator=torch.Generator().manual_seed(cfg['seed']+epoch*2+i))
                   for i, ds in enumerate((manual, pseudo))]
        configure_training(model)
        sums, count = np.zeros(2), 0
        for index, (raw, raw_pseudo) in enumerate(zip(*loaders)):
            if smoke and index >= opt['smoke_updates']:
                break
            seed = cfg['seed'] + epoch*10000 + index
            set_seed(seed)
            row = dict(epoch=epoch, step=index+1, seed=seed, manual=raw['name'], pseudo=raw_pseudo['name'],
                       manual_draw=draw_receipt(raw), pseudo_draw=draw_receipt(raw_pseudo),
                       manual_crop=raw['crop_box'].tolist(), pseudo_crop=raw_pseudo['crop_box'].tolist(),
                       crop_attempts=[raw['crop_attempt'].tolist(), raw_pseudo['crop_attempt'].tolist()])
            if 'training_view' in raw:
                if raw['training_view'] != raw_pseudo['training_view']:
                    raise RuntimeError('Manual and pseudo training views differ within an update')
                row['training_view'] = raw['training_view'][0]
            compare_draw(row, reference[updates])
            optimizer.zero_grad(set_to_none=True)
            losses, supervision = [], []
            for is_pseudo, raw_batch in [(False, raw), (True, raw_pseudo)]:
                batch = move_batch(raw_batch, device)
                with torch.no_grad():
                    restored = restore_first(restorer, batch['image'], seed+(2000000 if is_pseudo else 1000000))
                    if not is_pseudo and index < 2 and (smoke or epoch == 1 or epoch % 5 == 0):
                        save_training_draw(output, batch, restored, epoch, index+1)
                    features = model.encoder(restored)
                logits = model.affinity_decoder(features)['affinity_logits']
                loss, details = compute_affinity_loss(logits, batch, loss_cfg, pseudo=is_pseudo)
                if not torch.isfinite(loss) or details['positive_edges'] + details['negative_edges'] <= 0:
                    raise FloatingPointError('Nonfinite or unsupervised crop loss')
                (loss*(cfg['pseudo_weight'] if is_pseudo else 1.)).backward()
                losses.append(float(loss.detach()))
                supervision.append({k: details[k] for k in ('positive_edges', 'negative_edges', 'precision', 'recall', 'specificity')})
                del loss, logits, restored, features, batch
            grad = float(torch.nn.utils.clip_grad_norm_(model.affinity_decoder.parameters(), cfg['grad_clip'], error_if_nonfinite=True))
            if grad <= 0 or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('affinity_decoder.')):
                raise RuntimeError('Missing affinity gradient or unexpected frozen gradient')
            optimizer.step()
            updates += 1
            count += 1
            sums += losses
            row.update(updates=updates, manual_bce=losses[0], pseudo_bce=losses[1],
                       grad_norm=grad, supervision=supervision, auxiliary_weight=0.)
            append_json(output/'steps.jsonl', row)
            if index == 0 or smoke:
                print(json.dumps(dict(size=size, **row)), flush=True)
        if not smoke and count != data['updates_per_epoch']:
            raise RuntimeError('Incomplete epoch')
        scheduler.step()
        epoch_row = dict(epoch=epoch, updates=updates, manual_bce=sums[0]/count, pseudo_bce=sums[1]/count,
                         next_lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.time()-started,
                         peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2)
        append_json(output/'epochs.jsonl', epoch_row)
        checkpoint = dict(format='affinity_native_head_v1', epoch=epoch, size=size,
                          geometry_state_dict=model.affinity_decoder.state_dict(), config=config,
                          initialization=initialization, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        if checkpoint_extras is not None:
            extra = checkpoint_extras(epoch)
            if not isinstance(extra, dict) or set(extra).intersection(checkpoint):
                raise ValueError('Checkpoint extras must be a dict without reserved keys')
            checkpoint.update(extra)
        torch.save(checkpoint, output/'last_head.pt')
        if epoch % 5 == 0:
            torch.save({k:v for k,v in checkpoint.items() if k not in ('optimizer','scheduler')}, output/f'head_{epoch:03d}.pt')
        if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == epochs:
            monitor(model, restorer, config, device, output, epoch, size, smoke=smoke)
        status.update(status='running', **epoch_row)
        write_json(output/'status.json', status)
        print(json.dumps(dict(size=size, **epoch_row)), flush=True)
    status.update(frozen_unchanged=frozen_digest(model)==status['frozen_state_sha256'],
                  restorer_unchanged=tensor_digest(restorer.state_dict().items())==status['restorer_state_sha256'],
                  affinity_changed=tensor_digest(model.affinity_decoder.state_dict().items())!=initial_head,
                  paired_control_updates=updates)
    if not all(status[k] for k in ('frozen_unchanged','restorer_unchanged','affinity_changed')):
        raise RuntimeError('Training state audit failed')
    final_digest = tensor_digest(model.state_dict().items())
    save_backend(model, config, initialization, output/'final.pt', epochs,
                 extra=dict(native_size=size, updates=updates, smoke=smoke,
                            deployment='frozen_whole_semantic_stitched_native_affinity', data=data))
    del checkpoint, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(output/'final.pt', config, device)
    if tensor_digest(reloaded.state_dict().items()) != final_digest:
        raise RuntimeError('Strict reload differs')
    status.update(status='complete', strict_reload_equal=True, final_state_sha256=final_digest,
                  final_checkpoint_sha256=sha(output/'final.pt'))
    write_json(output/'status.json', status)
    print(f'COMPLETE native {size}, {updates} updates', flush=True)
    if smoke and not keep_smoke_checkpoint:
        del reloaded
        (output/'final.pt').unlink()


@torch.no_grad()
def inference(config, checkpoint, output, sizes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, bundle = load_backend(checkpoint, config, 'cuda')
    restorer = get_restorer(config, 'cuda')
    before = tensor_digest(model.state_dict().items())
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    if len(files) != config['affinity_native']['expected_test_images']:
        raise RuntimeError('Incomplete semifinal test cohort')
    rows = []
    for index, path in enumerate(files):
        rgb, views = predict_views(model, restorer, path, config, 'cuda',
            config['backend_adaptation']['restoration']['inference_seed']+index, sizes)
        for name, view in views.items():
            folder = output/name
            folder.mkdir(exist_ok=True)
            ids, classes = view['instances'], view['classes']
            verify_prediction(ids, classes, rgb.shape[:2])
            if not cv2.imwrite(str(folder/(path.stem+'_inst.png')), ids):
                raise RuntimeError('Could not save uint16 PNG')
            write_json(folder/(path.stem+'_class.json'), classes)
            row = dict(source=path.stem, view=name, phases=view['phases'], blend=view['blend'])
            rows.append(row)
            append_json(output/'images.jsonl', row)
        print('inference', path.stem, flush=True)
    if tensor_digest(model.state_dict().items()) != before:
        raise RuntimeError('Inference changed model')
    write_json(output/'manifest.json', dict(complete=True, checkpoint=str(checkpoint), epoch=bundle['epoch'],
               checkpoint_sha256=sha(checkpoint), sizes=sizes, images=rows, official_score=None,
               semantic='fixed full-image D5a geometry plus raw final vote', precision='FP32'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/train/affinity_native.yaml')
    p.add_argument('--mode', choices=['train','audit','infer'], default='train')
    p.add_argument('--size', type=int, choices=[1024,512])
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--checkpoint')
    p.add_argument('--views', type=int, nargs='+')
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = load_config(args.config)
    if args.mode == 'audit':
        audit_control(config, args.output)
    elif args.mode == 'infer':
        if not args.checkpoint or not args.views:
            p.error('infer needs checkpoint and views')
        inference(config, args.checkpoint, args.output, args.views)
    else:
        if args.size is None:
            p.error('train needs size')
        train(config, args.size, args.output, args.smoke)
