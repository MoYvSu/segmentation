# -*- coding: utf-8 -*-
"""复用mixed balance控制，仅扫描连接／断开监督的两个预定权重。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
import torch

from data.affinity_mixed import build_datasets as mixed_data
from models.fused_deployment import FUSED_DEPLOYMENT_FORMAT
from tools.affinity_native_views import save_monitor
from train_affinity_balance import comparable, verify_reference as verify_balance_reference
from train_affinity_native import compare_draw, read, train as native_train
from train_backend_adaptation import sha, tensor_digest, write_json
from utils.config import load_config, project_path


REFERENCE = 'outputs/affinity_balance/mixed'
REFERENCE_SHA256 = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
SWEEP_FORMAT = 'affinity_supervision_sweep_v1'
ALLOWED_WEIGHTS = (.75, 1.25)
AUDIT_FLAGS = ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal')
DEPENDENCY_KEYS = ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data')


def reference_config(config):
    """只剥离实验元数据与授权的r；其余差异由完整配置比较拒绝。"""
    result = deepcopy(config)
    options = result.pop('affinity_sweep')
    if (set(options) != {'reference', 'reference_sha256'}
            or options['reference'] != REFERENCE or options['reference_sha256'] != REFERENCE_SHA256):
        raise ValueError('The sweep must reuse the registered mixed balance control')
    weight = result['direct_semantic_affinity']['affinity_loss']['negative_weight']
    if isinstance(weight, bool) or weight not in ALLOWED_WEIGHTS:
        raise ValueError('Only the authorized weights 0.75 and 1.25 are allowed')
    if (result['backend_adaptation']['epochs'] != 20
            or result['backend_adaptation']['manual_repeats'] != 2
            or result['affinity_native']['penalty_sampling'] != 'mixed'):
        raise ValueError('The sweep retains mixed sampling and 20 x 64 updates')
    result['direct_semantic_affinity']['affinity_loss']['negative_weight'] = 1.0
    return result


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines()]


def _check_rows(rows, updates, config):
    """检查实际迭代和数值；不同r的loss/权重不要求逐值相等。"""
    if len(rows) != updates:
        raise RuntimeError('Missing or extra step receipts')
    for index, row in enumerate(rows):
        epoch, step = index // 64 + 1, index % 64 + 1
        seed = config['backend_adaptation']['seed'] + epoch * 10000 + step - 1
        if (row['epoch'], row['step'], row['updates'], row['seed']) != (epoch, step, index + 1, seed):
            raise RuntimeError('Step order, update budget or actual random seed differs')
        if row['auxiliary_weight'] != 0 or row['training_view'] not in ('whole', 'native1024'):
            raise RuntimeError('Unexpected auxiliary supervision or training view')
        if any(not math.isfinite(row[key]) or row[key] < 0 for key in ('manual_bce', 'pseudo_bce')):
            raise RuntimeError('Nonfinite or negative source loss')
        if not math.isfinite(row['grad_norm']) or row['grad_norm'] <= 0:
            raise RuntimeError('Nonfinite or missing affinity gradient')
        if len(row['supervision']) != 2:
            raise RuntimeError('Both manual and SAM2 supervision receipts are required')
        for source in row['supervision']:
            counts = [source[key] for key in ('positive_edges', 'negative_edges')]
            if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts)
                    or sum(counts) == 0):
                raise RuntimeError('Invalid supervised edge counts')
            if any(not math.isfinite(source[key]) or not 0 <= source[key] <= 1
                   for key in ('precision', 'recall', 'specificity')):
                raise RuntimeError('Invalid threshold diagnostic metrics')
    for start in range(0, updates, 64):
        epoch_rows = rows[start:start + 64]
        if len(epoch_rows) == 64 and sum(row['training_view'] == 'whole' for row in epoch_rows) != 32:
            raise RuntimeError('The mixed per-epoch view budget differs')


def verify_reference(config, state=None):
    """先执行既有balance祖先核验，再核验直接r1控制；不重训控制。"""
    base = reference_config(config)
    verify_balance_reference(base, state)
    folder = Path(project_path(config, config['affinity_sweep']['reference']))
    prior = read(folder / 'status.json')
    if comparable(base) != comparable(prior['config']):
        raise RuntimeError('A configuration changed beyond the authorized loss weight')
    if (prior['status'] != 'complete' or prior['updates'] != 1280 or prior['epoch'] != 20
            or prior.get('smoke') is not False or prior.get('size') != 1024
            or sha(folder / 'final.pt') != config['affinity_sweep']['reference_sha256']):
        raise RuntimeError('Mixed balance control is incomplete or changed')
    for key in AUDIT_FLAGS:
        if prior.get(key) is not True:
            raise RuntimeError('Mixed balance control audit failed: ' + key)
    if state is not None:
        for key in DEPENDENCY_KEYS:
            if prior[key] != state[key]:
                raise RuntimeError('Shared initialization, frozen dependency or data differs: ' + key)
    rows = _rows(folder / 'steps.jsonl')
    _check_rows(rows, 1280, base)
    return prior, rows


def capture_rng_state():
    """保存可重载的随机状态，不改变或消耗任何随机流。"""
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def checkpoint_extras(config, epoch, *, smoke=False):
    return {'sweep_state': dict(version=1, format=SWEEP_FORMAT, epoch=int(epoch),
        updates=int(epoch) * (8 if smoke else 64),
        negative_weight=config['direct_semantic_affinity']['affinity_loss']['negative_weight'],
        reference_sha256=config['affinity_sweep']['reference_sha256'], rng=capture_rng_state(),
        continuation='optimizer/scheduler/RNG saved; no resume CLI is provided')}


def _verify_rng(rng):
    if set(rng) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
        raise RuntimeError('Incomplete continuation random state')
    # 用临时生成器检查结构，既不恢复也不消耗训练/调用者的全局随机流。
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    if (not isinstance(rng['torch_cuda'], list)
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                   or value.ndim != 1 or value.numel() == 0
                   for value in [rng['torch_cpu'], *rng['torch_cuda']])):
        raise RuntimeError('Invalid Torch continuation random state')
    torch.Generator().set_state(rng['torch_cpu'])


def _verify_checkpoints(config, output, state, updates):
    path = output / 'final.pt'
    actual_sha = sha(path)
    if actual_sha != state['final_checkpoint_sha256']:
        raise RuntimeError('Candidate final checkpoint hash differs')
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    epoch = 1 if updates == 8 else 20
    if (bundle['format'] != FUSED_DEPLOYMENT_FORMAT or bundle['epoch'] != epoch
            or json.loads(json.dumps(bundle['config'])) != json.loads(json.dumps(config))
            or bundle['extra']['updates'] != updates or bundle['extra']['native_size'] != 1024
            or bundle['extra']['smoke'] is not (updates == 8)):
        raise RuntimeError('Candidate backend metadata or deployment format differs')
    tensors = bundle['model_state_dict']
    if (not tensors or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                           for value in tensors.values())
            or tensor_digest(tensors.items()) != state['final_state_sha256']
            or tensor_digest((name, value) for name, value in tensors.items()
                             if not name.startswith('affinity_decoder.')) != state['frozen_state_sha256']):
        raise RuntimeError('Saved model state is nonfinite or changed a frozen tensor')
    head = torch.load(output / 'last_head.pt', map_location='cpu', weights_only=False)
    if (head['format'] != 'affinity_native_head_v1' or head['epoch'] != epoch or head['size'] != 1024
            or json.loads(json.dumps(head['config'])) != json.loads(json.dumps(config))
            or not head.get('optimizer') or not head.get('scheduler')):
        raise RuntimeError('Head checkpoint is missing training state or differs')
    geometry = {name.removeprefix('affinity_decoder.'): value for name, value in tensors.items()
                if name.startswith('affinity_decoder.')}
    if (set(geometry) != set(head['geometry_state_dict']) or not geometry
            or any(not torch.equal(value, head['geometry_state_dict'][name]) for name, value in geometry.items())):
        raise RuntimeError('Final deployment and last training head differ')
    extra = head['sweep_state']
    if (extra['version'] != 1 or extra['format'] != SWEEP_FORMAT or extra['epoch'] != epoch
            or extra['updates'] != updates
            or extra['negative_weight'] != config['direct_semantic_affinity']['affinity_loss']['negative_weight']
            or extra['reference_sha256'] != config['affinity_sweep']['reference_sha256']):
        raise RuntimeError('Sweep continuation metadata differs')
    _verify_rng(extra['rng'])
    return dict(final_checkpoint_sha256=actual_sha, last_head_sha256=sha(output / 'last_head.pt'),
                frozen_saved_tensors_exact=True, head_bundle_exact=True, continuation_state_saved=True)


def _verify_observability(config, output, updates, steps):
    epochs = (0, 1) if updates == 8 else (0, 1, 5, 10, 15, 20)
    names = config['backend_adaptation']['monitor']['images'][:2] if updates == 8 else config['backend_adaptation']['monitor']['images']
    for epoch in epochs:
        folder = output / 'monitor' / f'epoch_{epoch:03d}'
        summary = read(folder / 'summary.json')
        if (summary['epoch'] != epoch or len(summary['images']) != len(names)
                or {row['source'] for row in summary['images']} != set(names)):
            raise RuntimeError('Incomplete fixed process monitor')
        for name in names:
            for suffix in ('full', 'detail', 'boundary'):
                image = cv2.imread(str(folder / f'{name}_{suffix}.png'), cv2.IMREAD_UNCHANGED)
                if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise RuntimeError('Missing or invalid process monitor PNG')
    for epoch in epochs[1:]:
        for step in (1, 2):
            folder = output / 'draws' / f'epoch_{epoch:03d}'
            receipt = read(folder / f'draw_{step:03d}.json')
            row = steps[(epoch - 1) * 64 + step - 1]
            if receipt['source'] != row['manual'] or receipt['crop_box'] != row['manual_crop']:
                raise RuntimeError('Training thumbnail differs from the actual draw')
            image = cv2.imread(str(folder / f'draw_{step:03d}.png'), cv2.IMREAD_UNCHANGED)
            if image is None or image.dtype != np.uint8 or image.shape != (282, 1024, 3):
                raise RuntimeError('Missing or invalid four-column training thumbnail')
    return dict(monitor_epochs=list(epochs), process_monitor_complete=True, actual_training_draws_complete=True)


def verify_run(config, output, updates=1280):
    if updates not in (8, 1280):
        raise ValueError('Only the eight-update smoke or complete 1280-update run may pass')
    output = Path(output)
    state = read(output / 'status.json')
    _, reference = verify_reference(config, state)
    if (state['config'] != json.loads(json.dumps(config)) or state['status'] != 'complete'
            or state['updates'] != updates or state['epoch'] != (1 if updates == 8 else 20)
            or state.get('smoke') is not (updates == 8) or state.get('size') != 1024
            or state.get('precision') != 'FP32'):
        raise RuntimeError('Candidate is incomplete or actual configuration/precision differs')
    for key in AUDIT_FLAGS:
        if state.get(key) is not True:
            raise RuntimeError('Candidate audit failed: ' + key)
    rows = _rows(output / 'steps.jsonl')
    _check_rows(rows, updates, config)
    for actual, expected in zip(rows, reference):
        compare_draw(actual, expected)
        for key in ('manual_crop', 'pseudo_crop', 'crop_attempts', 'training_view'):
            if actual[key] != expected[key]:
                raise RuntimeError('Sampling/crop/view differs from mixed balance: ' + key)
        for actual_source, expected_source in zip(actual['supervision'], expected['supervision']):
            for key in ('positive_edges', 'negative_edges'):
                if actual_source[key] != expected_source[key]:
                    raise RuntimeError('Supervised edge counts differ: ' + key)
    checkpoint = _verify_checkpoints(config, output, state, updates)
    observation = _verify_observability(config, output, updates, rows)
    return dict(format=SWEEP_FORMAT, passed=True, updates=updates, epoch=state['epoch'],
        negative_weight=config['direct_semantic_affinity']['affinity_loss']['negative_weight'],
        reference_negative_weight=1.0, reference_sha256=config['affinity_sweep']['reference_sha256'],
        control_retrained=False, paired_sources_crops_augmentation_views_and_counts_exact=True,
        initialization_sha256=state['initial_state_sha256'], frozen_state_sha256=state['frozen_state_sha256'],
        restorer_state_sha256=state['restorer_state_sha256'],
        numerical_policy='finite gradients; different objectives/FP32 runs need not be bitwise equal',
        diagnostic_metrics_scope='precision/recall/specificity are threshold diagnostics, not output parity',
        **checkpoint, **observation)


def train(config, output, smoke=False):
    verify_reference(config)
    monitor = partial(save_monitor,
        reference_folder=project_path(config, config['affinity_sweep']['reference'], 'deployment', 'patch1024'),
        reference_label='mixed balance e20 r1 native1024')
    native_train(config, 1024, output, smoke=smoke, dataset_builder=mixed_data,
                 control_verifier=verify_reference, keep_smoke_checkpoint=True,
                 checkpoint_extras=partial(checkpoint_extras, config, smoke=smoke), monitor_saver=monitor)
    audit = verify_run(config, output, 8 if smoke else 1280)
    write_json(Path(output) / 'sweep_receipt.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('train', 'verify'), default='train')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    configuration = load_config(args.config)
    if args.mode == 'verify':
        print(json.dumps(verify_run(configuration, args.output, 8 if args.smoke else 1280), ensure_ascii=False))
    else:
        train(configuration, args.output, args.smoke)
