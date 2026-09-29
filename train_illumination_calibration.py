# -*- coding: utf-8 -*-
"""独立校光训练；全部1000源图，无留出、无测试监督、预定60轮终点。"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.illumination_pairs import IlluminationPairs
from models.illumination_calibration import (FORMAT, IlluminationCalibrator, calibration_errors,
                                           calibration_loss, load_calibrator)
from train_backend_adaptation import write_json
from utils.config import load_config, project_path


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def dataset_for(config):
    cfg = config['illumination_calibration']
    degradation = load_config(project_path(config, config['semantic_adaptation']['restoration']['degradation_config']))['rgb_restoration']['degradation']
    return IlluminationPairs(project_path(config, cfg['data_dir']), cfg, degradation)


def to_device(batch):
    return [batch[k].cuda(non_blocking=True) for k in ('input', 'target', 'valid')]


def checkpoint(model, optimizer, path, epoch, updates, cfg):
    torch.save(dict(format=FORMAT, model_config=model.config,
                    state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch, updates=updates, config=cfg), path)


@torch.no_grad()
def reconstruction_monitor(model, batch, directory, epoch):
    from tools.run_illumination_calibration import thumb, save_rgb
    directory.mkdir(parents=True, exist_ok=True)
    inp, target, valid = to_device(batch)
    restored = model(inp, valid)
    for i in range(inp.shape[0]):
        cells = []
        for value, label in ((inp, 'Perturbed'), (restored, f'Calibrated e{epoch}'), (target, 'Same blur / original light')):
            rgb = np.rint(value[i].permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            cells.append(thumb(rgb, label, 224))
        save_rgb(directory / f'{batch["source"][i]}.png', np.concatenate(cells, 1))
    errors = calibration_errors(restored, target, valid)
    baseline = calibration_errors(inp, target, valid)
    write_json(directory / 'errors.json', dict(epoch=epoch,
        input={k: float(v) for k, v in baseline.items()}, restored={k: float(v) for k, v in errors.items()},
        scope='Fixed training-source synthetic diagnostic, not holdout'))


def run_short_test(config, output):
    cfg = config['illumination_calibration']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_all(cfg['seed'])
    dataset = dataset_for(config)
    # 同4个训练源，同时学习原样和光照逆变换；短测权重不进入正式训练。
    batches = []
    from torch.utils.data._utils.collate import default_collate
    for identity in (False, True):
        local_cfg = copy.deepcopy(cfg)
        local_cfg['identity_probability'] = float(identity)
        temporary = IlluminationPairs(project_path(config, cfg['data_dir']), local_cfg, dataset.degradation)
        items = [temporary[i] for i in range(cfg['short_test']['sources'])]
        batches.append(default_collate(items))
    tensors = [to_device(batch) for batch in batches]
    model = IlluminationCalibrator(**cfg['model']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    started = time.time()
    for step in range(cfg['short_test']['steps']):
        inp, target, valid = tensors[int(step % 4 == 3)]
        optimizer.zero_grad(set_to_none=True)
        prediction, fields = model(inp, valid, return_fields=True)
        loss, _ = calibration_loss(prediction, target, valid, fields)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite short-test loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite short-test gradients')
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f'short {step+1}/{cfg["short_test"]["steps"]} loss={float(loss.detach()):.6f}', flush=True)
    reports = []
    model.eval()
    with torch.no_grad():
        for batch, (inp, target, valid) in zip(batches, tensors):
            pred = model(inp, valid)
            reports.append(dict(input={k: float(v) for k, v in calibration_errors(inp, target, valid).items()},
                                restored={k: float(v) for k, v in calibration_errors(pred, target, valid).items()}))
            reconstruction_monitor(model, batch, output / ('identity' if batch['identity'][0] else 'light'), -1)
    passed = (reports[0]['restored']['rgb_l1'] <= cfg['short_test']['rgb_ratio'] * reports[0]['input']['rgb_l1']
              and reports[0]['restored']['low_l1'] <= cfg['short_test']['rgb_ratio'] * reports[0]['input']['low_l1']
              and reports[1]['restored']['rgb_l1'] <= cfg['short_test']['identity_mae'])
    checkpoint(model, optimizer, output / 'model.pt', -1, cfg['short_test']['steps'], cfg)
    loaded, _ = load_calibrator(output / 'model.pt', 'cuda')
    with torch.no_grad():
        same = torch.equal(model(tensors[0][0], tensors[0][2]), loaded(tensors[0][0], tensors[0][2]))
    report = dict(passed=bool(passed and same), synthetic=reports[0], identity=reports[1],
                  strict_reload_equal=same, steps=cfg['short_test']['steps'],
                  sources=[p.name for p in dataset.samples[:cfg['short_test']['sources']]],
                  seconds=time.time() - started, scope='Training-source overfit test; fresh formal initialization')
    write_json(output / 'report.json', report)
    if not report['passed']:
        raise RuntimeError(f'Short-test gate failed: {report}')
    print(json.dumps(report), flush=True)


def train(config, output, backend, restorer):
    from tools.run_illumination_calibration import render_comparison
    from torch.utils.data._utils.collate import default_collate
    cfg = config['illumination_calibration']
    output = Path(output)
    model_dir = output / 'model'
    model_dir.mkdir(exist_ok=False)
    seed_all(cfg['seed'])
    dataset = dataset_for(config)
    fixed = default_collate([dataset[i] for i in range(4)])
    model = IlluminationCalibrator(**cfg['model']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
    updates, started = 0, time.time()
    write_json(output / 'training_sources.json', dict(sources=[p.name for p in dataset.samples],
        count=len(dataset), split_policy='all_train_no_holdout', target='same clean/blur content before synthetic illumination'))
    checkpoint(model, optimizer, model_dir / 'epoch_000.pt', 0, 0, cfg)
    reconstruction_monitor(model, fixed, output / 'synthetic' / 'epoch_000', 0)
    for epoch in range(1, cfg['epochs'] + 1):
        dataset.epoch = epoch
        generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],
                            worker_init_fn=worker_init, generator=generator, pin_memory=True,
                            multiprocessing_context='spawn' if cfg['num_workers'] else None)
        lr = cfg['minimum_lr'] + .5 * (cfg['learning_rate'] - cfg['minimum_lr']) * (1 + math.cos(math.pi * (epoch - 1) / max(1, cfg['epochs'] - 1)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        totals = dict(loss=0., rgb_l1=0., low_l1=0., gradient_l1=0., clipped_fraction=0.,
                      log_exposure=0., gamma_delta=0., identity=0., blurred=0., actual_nonzero=0.)
        count, seen = 0, set()
        for batch in loader:
            inp, target, valid = to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            pred, fields = model(inp, valid, return_fields=True)
            loss, errors = calibration_loss(pred, target, valid, fields)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite training gradient')
            optimizer.step()
            n = inp.shape[0]
            count += n; updates += 1; seen.update(batch['source'])
            if updates == 1:
                print(f'first_update completed: samples={n}, loss={float(loss.detach()):.6f}', flush=True)
            totals['loss'] += float(loss.detach()) * n
            for k, value in errors.items():
                totals[k] += float(value.detach()) * n
            for key in ('clipped_fraction', 'identity', 'blurred'):
                totals[key] += float(batch[key].sum())
            totals['log_exposure'] += float(batch['log_exposure'].abs().sum())
            totals['gamma_delta'] += float((batch['gamma'] - 1).abs().sum())
            totals['actual_nonzero'] += float(((batch['gamma'] - 1).abs() + batch['log_exposure'].abs() + batch['field_strength'] > 1e-4).sum())
        if count != cfg['expected_sources'] or len(seen) != count:
            raise RuntimeError('Not all training sources were used exactly once this epoch')
        record = dict(epoch=epoch, updates=updates, sources=count, learning_rate=lr,
                      elapsed_seconds=time.time() - started, **{k: v / count for k, v in totals.items()})
        with (output / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        checkpoint(model, optimizer, model_dir / 'last.pt', epoch, updates, cfg)
        if epoch == 1 or epoch % cfg['monitor']['every_epochs'] == 0 or epoch == cfg['epochs']:
            model.eval()
            checkpoint(model, optimizer, model_dir / f'epoch_{epoch:03d}.pt', epoch, updates, cfg)
            reconstruction_monitor(model, fixed, output / 'synthetic' / f'epoch_{epoch:03d}', epoch)
            render_comparison(config, model, backend, restorer, output / 'monitor' / f'epoch_{epoch:03d}', epoch)
    if updates != cfg['epochs'] * math.ceil(len(dataset) / cfg['batch_size']):
        raise RuntimeError('Incomplete training budget')
    loaded, _ = load_calibrator(model_dir / f'epoch_{cfg["epochs"]:03d}.pt', 'cuda')
    inp, _, valid = to_device(fixed)
    with torch.no_grad():
        if not torch.equal(model(inp, valid), loaded(inp, valid)):
            raise RuntimeError('Final strict reload changed output')
    write_json(output / 'training_complete.json', dict(epoch=cfg['epochs'], updates=updates,
        strict_reload_equal=True, source_count=len(dataset), failed_updates=0,
        trainable_parameters=sum(p.numel() for p in model.parameters())))
    return loaded
