# -*- coding: utf-8 -*-
"""只恢复定时关机中断的P025 e16→e20；旧源码、旧run和控制组保持不动。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.affinity_mixed import build_datasets
from models.backend_adaptation import build_backend, load_backend, restore_first, save_backend
from tools import run_affinity_source as original_runner
from tools.affinity_native_views import save_monitor, save_training_draw
from train_affinity_connectivity import append_json, configure_training, draw_receipt, frozen_digest, get_restorer
import train_affinity_native as native
from train_affinity_source import (SIDECAR, SOURCE_FORMAT, checkpoint_extras, source_coefficients,
                                   verify_reference, verify_run, _verify_sidecar)
from train_affinity_sweep import _check_rows, _rows, _verify_rng
from train_backend_adaptation import sha, tensor_digest, write_json
from train_direct_semantic_affinity import move_batch, set_seed
from utils.config import load_config, project_path

ORIGINAL = 'outputs/affinity_source'
CHECKPOINT_SHA256 = 'a8be7bc5902388316873fa419427b476b2d294e128e76fda057364e6a3bd3460'
PREFIX_EPOCHS, PREFIX_UPDATES, FINAL_UPDATES = 16, 1024, 1280
PREFIX_COUNTS = {'steps.jsonl': 1024, 'epochs.jsonl': 16, SIDECAR: 2048}
RESUME_NAME = 'tools/resume_affinity_source.py'


def retained_bytes(path, count):
    """原始行bytes照搬；不重新格式化旧数值或换行。"""
    lines = Path(path).read_bytes().splitlines(keepends=True)
    if len(lines) < count or any(not line.endswith(b'\n') for line in lines[:count]):
        raise RuntimeError('Incomplete committed log prefix: ' + str(path))
    return b''.join(lines[:count]), len(lines) - count


def restore_rng(rng):
    _verify_rng(rng)
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu'])
    if len(rng['torch_cuda']) != torch.cuda.device_count():
        raise RuntimeError('Saved CUDA random state device count differs')
    torch.cuda.set_rng_state_all(rng['torch_cuda'])


def expected_lr(config, epoch):
    cfg = config['backend_adaptation']
    return cfg['learning_rates']['affinity'] * (cfg['minimum_lr_ratio'] +
        (1 - cfg['minimum_lr_ratio']) * (1 + math.cos(math.pi * epoch / cfg['epochs'])) / 2)


def check_head(config, head):
    """不是通用续训：必须是已核实的原e16，原20轮cosine与1024次AdamW更新。"""
    weights = source_coefficients(config)
    if (weights['beta'] != .25 or head['format'] != 'affinity_native_head_v1'
            or head['epoch'] != 16 or head['size'] != 1024
            or json.loads(json.dumps(head['config'])) != json.loads(json.dumps(config))):
        raise RuntimeError('Only the registered P025 epoch16 checkpoint may resume')
    extra = head['source_state']
    if (extra['format'] != SOURCE_FORMAT or extra['version'] != 1 or extra['epoch'] != 16
            or extra['updates'] != 1024 or extra.get('raw_bce_logs_unscaled') is not True
            or extra['reference_sha256'] != config['affinity_source']['reference_sha256']
            or any(extra[key] != value for key, value in weights.items())):
        raise RuntimeError('Source-ratio checkpoint metadata differs')
    _verify_rng(extra['rng'])
    state = head['geometry_state_dict']
    if not state or any(not torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError('Nonfinite or empty saved affinity head')
    optimizer, scheduler = head['optimizer'], head['scheduler']
    if (not optimizer['state'] or len(optimizer['param_groups']) != 1
            or scheduler['last_epoch'] != 16 or scheduler['_step_count'] != 17
            or scheduler['base_lrs'] != [config['backend_adaptation']['learning_rates']['affinity']]
            or not math.isclose(optimizer['param_groups'][0]['lr'], expected_lr(config, 16), rel_tol=1e-12)
            or scheduler['_last_lr'] != [optimizer['param_groups'][0]['lr']]):
        raise RuntimeError('Original 20-epoch optimizer/scheduler state differs')
    for value in optimizer['state'].values():
        if float(value['step']) != 1024 or any(not torch.isfinite(value[key]).all()
                for key in ('exp_avg', 'exp_avg_sq')):
            raise RuntimeError('AdamW state is incomplete or not the epoch16 state')
    group = optimizer['param_groups'][0]
    cfg = config['backend_adaptation']
    if group['eps'] != 1e-4 or group['weight_decay'] != cfg['weight_decay'] or group['betas'] != (0.9, 0.999):
        raise RuntimeError('AdamW settings differ from the original run')


def validate_original(config):
    folder = ROOT / ORIGINAL
    pipeline = native.read(folder / 'pipeline_status.json')
    state = native.read(folder / 'p025/status.json')
    runtime = original_runner.prior_runner.runtime_contract(config)
    if (pipeline['config'] != json.loads(json.dumps(config)) or pipeline['smoke'] is not False
            or pipeline['control_retrained'] is not False or pipeline['stage'] != 'p025'
            or pipeline['status'] != 'running' or pipeline['runtime'] != runtime
            or len(pipeline['sources']) != 122 or set(pipeline['sources']) != set(original_runner.source_names())):
        raise RuntimeError('The original interrupted P025 pipeline identity differs')
    for name, digest in pipeline['sources'].items():
        if sha(ROOT / name) != digest or sha(folder / 'source' / name) != digest:
            raise RuntimeError('Original runtime source or snapshot changed: ' + name)
    original_runner.verify_historical(tuple(pipeline['sources']), runtime)
    if (state['config'] != pipeline['config'] or state['epoch'] != 16 or state['updates'] != 1024
            or state['status'] != 'running' or state['smoke'] is not False
            or state['size'] != 1024 or state['precision'] != 'FP32'):
        raise RuntimeError('Original completed epoch is not exactly e16/1024')
    _, control = verify_reference(config, state)
    head_path = folder / 'p025/last_head.pt'
    if sha(head_path) != CHECKPOINT_SHA256:
        raise RuntimeError('Registered epoch16 checkpoint SHA changed')
    head = torch.load(head_path, map_location='cpu', weights_only=False)
    check_head(config, head)
    if head['initialization'] != state['initialization']:
        raise RuntimeError('Saved initialization metadata differs')
    prefix = {}
    extra_rows = {}
    for name, count in PREFIX_COUNTS.items():
        content, extra_rows[name] = retained_bytes(folder / 'p025' / name, count)
        prefix[name] = content
    steps = [json.loads(line) for line in prefix['steps.jsonl'].splitlines()]
    _check_rows(steps, 1024, config)
    for actual, expected in zip(steps, control):
        native.compare_draw(actual, expected)
        for key in ('manual_crop', 'pseudo_crop', 'crop_attempts', 'training_view'):
            if actual[key] != expected[key]:
                raise RuntimeError('Original draw/crop/view differs: ' + key)
        for a, b in zip(actual['supervision'], expected['supervision']):
            if any(a[key] != b[key] for key in ('positive_edges', 'negative_edges')):
                raise RuntimeError('Original supervised edge counts differ')
    epoch_rows = [json.loads(line) for line in prefix['epochs.jsonl'].splitlines()]
    if [(row['epoch'], row['updates']) for row in epoch_rows] != [(e, 64 * e) for e in range(1, 17)]:
        raise RuntimeError('Original complete epoch receipts differ')
    side_rows = [json.loads(line) for line in prefix[SIDECAR].splitlines()]
    for index, row in enumerate(side_rows):
        expected = steps[index // 2]
        is_pseudo = bool(index % 2)
        source = 'pseudo' if is_pseudo else 'manual'
        coefficient = source_coefficients(config)['pseudo_coefficient' if is_pseudo else 'manual_coefficient']
        if (row['call'] != index + 1 or row['update'] != index // 2 + 1 or row['epoch'] != expected['epoch']
                or row['step'] != expected['step'] or row['source'] != expected[source]
                or row['raw_bce'] != expected[source + '_bce'] or row['source_coefficient'] != coefficient
                or row['source_kind'] != ('sam2' if is_pseudo else 'manual')):
            raise RuntimeError('Original completed source-gradient receipt differs')
    return pipeline, state, head, control, prefix, extra_rows, runtime


def original_identity(config, pipeline, runtime):
    root = ROOT / ORIGINAL
    return dict(original_run=ORIGINAL, original_pipeline_sha256=sha(root / 'pipeline_status.json'),
        original_status_sha256=sha(root / 'p025/status.json'), original_head_path=ORIGINAL + '/p025/last_head.pt',
        original_head_sha256=sha(root / 'p025/last_head.pt'), original_source_sha256=pipeline['sources'],
        original_log_sha256={name: sha(root / 'p025' / name) for name in PREFIX_COUNTS},
        resume_source_sha256={RESUME_NAME: sha(ROOT / RESUME_NAME)},
        epoch=16, prefix_updates=1024, new_updates=256, final_updates=1280,
        config=config, runtime=runtime, control_retrained=False)


def copy_prefix(root, prefix):
    candidate = root / 'p025'
    candidate.mkdir()
    for name, data in prefix.items():
        (candidate / name).write_bytes(data)
    previous = ROOT / ORIGINAL / 'p025'
    copied = {}
    for category, epochs in (('monitor', (0, 1, 5, 10, 15)), ('draws', (1, 5, 10, 15))):
        for epoch in epochs:
            source = previous / category / f'epoch_{epoch:03d}'
            if not source.is_dir():
                raise RuntimeError('Missing original observability: ' + str(source))
            dest = candidate / category / source.name
            dest.mkdir(parents=True)
            for item in source.iterdir():
                if not item.is_file() or item.suffix not in ('.png', '.json'):
                    raise RuntimeError('Unexpected original observability artifact: ' + str(item))
                shutil.copy2(item, dest / item.name)
                copied[(Path(category) / source.name / item.name).as_posix()] = sha(item)
    for epoch in (5, 10, 15):
        source = previous / f'head_{epoch:03d}.pt'
        shutil.copy2(source, candidate / source.name)
        copied[source.name] = sha(source)
    return dict(retained_prefix_sha256={name: sha(candidate / name) for name in prefix},
                copied_observability_and_head_sha256=copied)


def smoke_gate(config, identity):
    root = ROOT / 'outputs/affinity_source_resume_smoke'
    gate = native.read(root / 'pipeline_status.json')
    receipt = native.read(root / 'resume_receipt.json')
    probe = native.read(root / 'p025/smoke_receipt.json')
    if (gate['status'] != 'complete' or gate['smoke'] is not True or probe['passed'] is not True
            or receipt.get('identity') != json.loads(json.dumps(identity))
            or gate['resume_sources'] != identity['resume_source_sha256']
            or gate['sources'] != identity['original_source_sha256']):
        raise RuntimeError('Exact one-update restoration smoke must pass before continuation')
    for name, digest in identity['resume_source_sha256'].items():
        if sha(ROOT / name) != digest or sha(root / 'resume_source' / name) != digest:
            raise RuntimeError('Resume implementation changed after the technical gate')
    probe_head = torch.load(root / 'p025/probe_head.pt', map_location='cpu', weights_only=False)
    if (sha(root / 'p025/probe_head.pt') != probe['probe_head_sha256']
            or probe_head['updates'] != 1025 or probe_head['scheduler']['last_epoch'] != 16):
        raise RuntimeError('Restoration smoke checkpoint differs')
    return dict(path='outputs/affinity_source_resume_smoke/resume_receipt.json',
                sha256=sha(root / 'resume_receipt.json'), probe_sha256=sha(root / 'p025/smoke_receipt.json'))


def compare_interrupted_step(actual):
    rows = _rows(ROOT / ORIGINAL / 'p025/steps.jsonl')
    if len(rows) <= 1024:
        return dict(available=False, scope='No committed epoch17 step1 exists')
    previous = rows[1024]
    native.compare_draw(actual, previous)
    for key in ('manual_crop', 'pseudo_crop', 'crop_attempts', 'training_view', 'updates'):
        if actual[key] != previous[key]:
            raise RuntimeError('Restored e17 step1 draw differs from the interrupted run: ' + key)
    differences = {}
    for key in ('manual_bce', 'pseudo_bce', 'grad_norm'):
        differences[key] = dict(previous=previous[key], resumed=actual[key],
            absolute=abs(actual[key] - previous[key]),
            relative=abs(actual[key] - previous[key]) / max(abs(previous[key]), 1e-12))
        if not math.isclose(actual[key], previous[key], rel_tol=1e-4, abs_tol=1e-6):
            raise RuntimeError('Restored e17 step1 numerical replay differs: ' + key)
    for a, b in zip(actual['supervision'], previous['supervision']):
        if any(a[key] != b[key] for key in ('positive_edges', 'negative_edges')):
            raise RuntimeError('Restored e17 step1 edge counts differ')
    return dict(available=True, passed=True, differences=differences,
                numerical_policy='FP32 relative1e-4/absolute1e-6; no bitwise CUDA requirement')


def weighted_loss(logits, batch, loss_cfg, *, pseudo, config, output, update, epoch, step):
    """沿用原beta乘法与alpha scalar-hook顺序，记录实际logits梯度。"""
    loss, details = native.compute_affinity_loss(logits, batch, loss_cfg, pseudo=pseudo)
    weights = source_coefficients(config)
    coefficient = weights['pseudo_coefficient' if pseudo else 'manual_coefficient']
    loss.register_hook(lambda incoming: incoming * weights['alpha'])
    row = dict(call=(update - 1) * 2 + 1 + int(pseudo), epoch=epoch, step=step, update=update,
        source_kind='sam2' if pseudo else 'manual', source=batch['name'], raw_bce=float(loss.detach()),
        source_coefficient=coefficient, **weights)
    def record(gradient):
        norm = float(gradient.detach().float().norm())
        if not math.isfinite(norm):
            raise FloatingPointError('Nonfinite weighted logits gradient')
        append_json(Path(output) / SIDECAR, dict(**row, weighted_logits_grad_norm=norm,
            raw_logits_grad_norm=norm / coefficient,
            scope='loss-gradient diagnostic at logits, not parameter gradient or source importance'))
    logits.register_hook(record)
    return loss, details


def train_remaining(config, output, old_state, head, reference, *, smoke=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Use the original sam2_env CUDA runtime')
    cfg, size = config['backend_adaptation'], 1024
    if cfg['num_workers'] != 0:
        raise RuntimeError('The original mixed dataset requires num_workers=0')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    set_seed(cfg['seed'])
    model, initialization = build_backend(config, device)
    restorer = get_restorer(config, device)
    manual, pseudo, data = build_datasets(config, size)
    initial_head = tensor_digest(model.affinity_decoder.state_dict().items())
    if (tensor_digest(model.state_dict().items()) != old_state['initial_state_sha256']
            or frozen_digest(model) != old_state['frozen_state_sha256']
            or tensor_digest(restorer.state_dict().items()) != old_state['restorer_state_sha256']
            or initialization != old_state['initialization'] or data != old_state['data']):
        raise RuntimeError('Rebuilt original initialization/frozen state/restorer/data differs')
    model.affinity_decoder.load_state_dict(head['geometry_state_dict'], strict=True)
    configure_training(model)
    optimizer = torch.optim.AdamW(model.affinity_decoder.parameters(),
        lr=cfg['learning_rates']['affinity'], weight_decay=cfg['weight_decay'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: cfg['minimum_lr_ratio'] +
        (1 - cfg['minimum_lr_ratio']) * (1 + math.cos(math.pi * e / cfg['epochs'])) / 2)
    optimizer.load_state_dict(head['optimizer'])
    scheduler.load_state_dict(head['scheduler'])
    restore_rng(head['source_state']['rng'])
    if scheduler.last_epoch != 16 or optimizer.param_groups[0]['lr'] != expected_lr(config, 16):
        raise RuntimeError('Restored original learning-rate state differs')
    output = Path(output)
    status = deepcopy(old_state)
    status.update(status='running', resumed_from_epoch=16, restored_optimizer_scheduler_rng=True)
    write_json(output / 'status.json', status)
    loss_cfg = config['direct_semantic_affinity']['affinity_loss']
    monitor = partial(save_monitor,
        reference_folder=project_path(config, config['affinity_source']['reference'], 'deployment/patch1024'),
        reference_label='mixed balance e20 r1 h1 beta0.5 native1024')
    updates, started, first_replay = 1024, time.time(), None
    for epoch in range(17, 21):
        for ds in (manual, pseudo):
            ds.set_epoch(epoch)
        loaders = [DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(cfg['seed'] + epoch * 2 + i))
            for i, ds in enumerate((manual, pseudo))]
        configure_training(model)
        sums, count = np.zeros(2), 0
        for index, (raw, raw_pseudo) in enumerate(zip(*loaders)):
            seed = cfg['seed'] + epoch * 10000 + index
            set_seed(seed)
            row = dict(epoch=epoch, step=index + 1, seed=seed, manual=raw['name'], pseudo=raw_pseudo['name'],
                manual_draw=draw_receipt(raw), pseudo_draw=draw_receipt(raw_pseudo),
                manual_crop=raw['crop_box'].tolist(), pseudo_crop=raw_pseudo['crop_box'].tolist(),
                crop_attempts=[raw['crop_attempt'].tolist(), raw_pseudo['crop_attempt'].tolist()])
            if raw['training_view'] != raw_pseudo['training_view']:
                raise RuntimeError('Manual and SAM2 mixed views differ')
            row['training_view'] = raw['training_view'][0]
            native.compare_draw(row, reference[updates])
            for key in ('manual_crop', 'pseudo_crop', 'crop_attempts', 'training_view'):
                if row[key] != reference[updates][key]:
                    raise RuntimeError('Restored original mixed draw differs: ' + key)
            optimizer.zero_grad(set_to_none=True)
            losses, supervision = [], []
            for is_pseudo, raw_batch in ((False, raw), (True, raw_pseudo)):
                batch = move_batch(raw_batch, device)
                with torch.no_grad():
                    restored = restore_first(restorer, batch['image'], seed + (2000000 if is_pseudo else 1000000))
                    if not is_pseudo and index < 2 and (smoke or epoch % 5 == 0):
                        save_training_draw(output, batch, restored, epoch, index + 1)
                    features = model.encoder(restored)
                logits = model.affinity_decoder(features)['affinity_logits']
                loss, details = weighted_loss(logits, batch, loss_cfg, pseudo=is_pseudo, config=config,
                    output=output, update=updates + 1, epoch=epoch, step=index + 1)
                if not torch.isfinite(loss) or details['positive_edges'] + details['negative_edges'] <= 0:
                    raise FloatingPointError('Nonfinite or unsupervised restored loss')
                (loss * (cfg['pseudo_weight'] if is_pseudo else 1.)).backward()
                losses.append(float(loss.detach()))
                supervision.append({k: details[k] for k in ('positive_edges', 'negative_edges', 'precision', 'recall', 'specificity')})
                del loss, logits, restored, features, batch
            gradient = float(torch.nn.utils.clip_grad_norm_(model.affinity_decoder.parameters(),
                cfg['grad_clip'], error_if_nonfinite=True))
            if gradient <= 0 or any(p.grad is not None for name, p in model.named_parameters()
                    if not name.startswith('affinity_decoder.')):
                raise RuntimeError('Invalid affinity or frozen parameter gradient')
            optimizer.step()
            updates += 1
            count += 1
            sums += losses
            row.update(updates=updates, manual_bce=losses[0], pseudo_bce=losses[1], grad_norm=gradient,
                       supervision=supervision, auxiliary_weight=0.)
            append_json(output / 'steps.jsonl', row)
            if updates == 1025:
                first_replay = compare_interrupted_step(row)
            if index == 0:
                print(json.dumps(dict(size=size, **row)), flush=True)
            if smoke:
                state = dict(format='affinity_source_resume_smoke_v1', updates=1025, epoch=17, step=1,
                    geometry_state_dict=model.affinity_decoder.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), **checkpoint_extras(config, 16))
                state['source_state'].update(updates=1025, epoch=17, continuation_step=1)
                torch.save(state, output / 'probe_head.pt')
                reload = torch.load(output / 'probe_head.pt', map_location='cpu', weights_only=False)
                if tensor_digest(reload['geometry_state_dict'].items()) != tensor_digest(model.affinity_decoder.state_dict().items()):
                    raise RuntimeError('Smoke checkpoint head reload differs')
                _verify_rng(reload['source_state']['rng'])
                _verify_sidecar(config, output, _rows(output / 'steps.jsonl'))
                receipt = dict(passed=True, epoch=17, step=1, updates=1025, new_updates=1,
                    original_total_epochs=20, scheduler_not_restarted=True, scheduler_last_epoch=scheduler.last_epoch,
                    restored_optimizer_scheduler_rng=True, strict_head_reload_equal=True,
                    frozen_unchanged=frozen_digest(model) == old_state['frozen_state_sha256'],
                    restorer_unchanged=tensor_digest(restorer.state_dict().items()) == old_state['restorer_state_sha256'],
                    probe_head_sha256=sha(output / 'probe_head.pt'), interrupted_step_replay=first_replay)
                if not receipt['frozen_unchanged'] or not receipt['restorer_unchanged']:
                    raise RuntimeError('Restored smoke changed a frozen dependency')
                write_json(output / 'smoke_receipt.json', receipt)
                return receipt
        if count != data['updates_per_epoch'] or count != 64:
            raise RuntimeError('Restored epoch is incomplete')
        scheduler.step()
        epoch_row = dict(epoch=epoch, updates=updates, manual_bce=sums[0] / count, pseudo_bce=sums[1] / count,
            next_lr=optimizer.param_groups[0]['lr'],
            elapsed_seconds=old_state['elapsed_seconds'] + time.time() - started,
            resumed_elapsed_seconds=time.time() - started,
            peak_cuda_mib=torch.cuda.max_memory_allocated() / 1024**2)
        append_json(output / 'epochs.jsonl', epoch_row)
        checkpoint = dict(format='affinity_native_head_v1', epoch=epoch, size=size,
            geometry_state_dict=model.affinity_decoder.state_dict(), config=config, initialization=initialization,
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), **checkpoint_extras(config, epoch))
        torch.save(checkpoint, output / 'last_head.pt')
        if epoch % 5 == 0:
            torch.save({k: v for k, v in checkpoint.items() if k not in ('optimizer', 'scheduler')},
                output / f'head_{epoch:03d}.pt')
        if epoch % cfg['monitor']['every_epochs'] == 0 or epoch == 20:
            monitor(model, restorer, config, device, output, epoch, size)
        status.update(status='running', **epoch_row)
        write_json(output / 'status.json', status)
        print(json.dumps(dict(size=size, **epoch_row)), flush=True)
    status.update(frozen_unchanged=frozen_digest(model) == old_state['frozen_state_sha256'],
        restorer_unchanged=tensor_digest(restorer.state_dict().items()) == old_state['restorer_state_sha256'],
        affinity_changed=tensor_digest(model.affinity_decoder.state_dict().items()) != initial_head,
        paired_control_updates=updates)
    if updates != 1280 or not all(status[key] for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed')):
        raise RuntimeError('Restored full-budget training audit failed')
    digest = tensor_digest(model.state_dict().items())
    save_backend(model, config, initialization, output / 'final.pt', 20,
        extra=dict(native_size=1024, updates=1280, smoke=False,
            deployment='frozen_whole_semantic_stitched_native_affinity', data=data,
            resumed_from_epoch=16, new_updates=256))
    del checkpoint, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _ = load_backend(output / 'final.pt', config, device)
    if tensor_digest(reloaded.state_dict().items()) != digest:
        raise RuntimeError('Restored final strict reload differs')
    status.update(status='complete', strict_reload_equal=True, final_state_sha256=digest,
                  final_checkpoint_sha256=sha(output / 'final.pt'), new_updates=256)
    write_json(output / 'status.json', status)
    audit = verify_run(config, output, 1280)
    write_json(output / 'source_receipt.json', audit)
    write_json(output / 'resume_training_receipt.json', dict(new_updates=256,
        interrupted_step_replay=first_replay, total_updates=1280, original_total_epochs=20,
        restored_optimizer_scheduler_rng=True, scheduler_not_restarted=True))
    return audit


def run(*, smoke=False):
    config = load_config(original_runner.CONFIG_PATH)
    pipeline, old_state, head, reference, prefix, discarded, runtime = validate_original(config)
    identity = original_identity(config, pipeline, runtime)
    gate = None if smoke else smoke_gate(config, identity)
    root = ROOT / 'outputs' / ('affinity_source_resume_smoke' if smoke else 'affinity_source_resume')
    if root.exists():
        raise RuntimeError('Refusing to overwrite an existing recovery run: ' + str(root))
    if shutil.disk_usage(ROOT).free < config['affinity_native']['minimum_free_gib'] * 1024**3:
        raise RuntimeError('Insufficient fast-disk space')
    root.mkdir(parents=True)
    for name in pipeline['sources']:
        destination = root / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    for name in identity['resume_source_sha256']:
        destination = root / 'resume_source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    receipt = dict(**identity, identity=identity, **copy_prefix(root, prefix), discarded_partial=discarded,
        smoke_gate=gate, resumed_new_updates=0, original_run_preserved=True)
    write_json(root / 'resume_receipt.json', receipt)
    status = dict(status='running', stage='p025', pid=os.getpid(), smoke=smoke, config=config,
        sources=pipeline['sources'], resume_sources=identity['resume_source_sha256'], runtime=runtime,
        historical=pipeline['historical'], completed=[], audits={}, control_retrained=False,
        automatic_extension=False, competition_submission=False, automatic_package=False,
        resume_receipt='resume_receipt.json')
    write_json(root / 'pipeline_status.json', status)
    try:
        status['audits']['p025'] = train_remaining(config, root / 'p025', old_state, head, reference, smoke=smoke)
        receipt['resumed_new_updates'] = 1 if smoke else 256
        receipt['finished_training'] = not smoke
        if not smoke:
            receipt['training'] = native.read(root / 'p025/resume_training_receipt.json')
        write_json(root / 'resume_receipt.json', receipt)
        if not smoke:
            status['completed'].append('p025')
            status['stage'] = 'p025_infer'
            write_json(root / 'pipeline_status.json', status)
            native.inference(config, root / 'p025/final.pt', root / 'p025/deployment', [1024])
            status['completed'].append('p025_infer')
            status['stage'] = 'render'
            write_json(root / 'pipeline_status.json', status)
            original_runner.render_final(config, root)
        else:
            status['completed'].append('resume_smoke')
        # 原run逐byte保持；恢复产物不把partial旧run伪装成完成。
        check_pipeline, _, _, _, _, _, check_runtime = validate_original(config)
        if original_identity(config, check_pipeline, check_runtime) != identity:
            raise RuntimeError('Original interrupted run changed during recovery')
        status.update(status='complete', stage='done')
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(root / 'pipeline_status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(smoke=args.smoke)
