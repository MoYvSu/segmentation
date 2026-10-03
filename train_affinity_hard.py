# -*- coding: utf-8 -*-
"""复用mixed balance控制，只关闭困难异实例关系重加权；不改历史扫参代码。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path

import torch

from data.affinity_mixed import build_datasets as mixed_data
from models.fused_deployment import FUSED_DEPLOYMENT_FORMAT
from tools.affinity_native_views import save_monitor
from train_affinity_balance import comparable, verify_reference as verify_balance_reference
from train_affinity_native import compare_draw, read, train as native_train
from train_affinity_sweep import (AUDIT_FLAGS, DEPENDENCY_KEYS, _check_rows, _rows,
                                 _verify_observability, _verify_rng, capture_rng_state)
from train_backend_adaptation import sha, tensor_digest, write_json
from utils.config import load_config, project_path


REFERENCE = 'outputs/affinity_balance/mixed'
REFERENCE_SHA256 = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
HARD_FORMAT = 'affinity_hard_negative_ablation_v1'


def reference_config(config):
    """剥离新实验元数据并恢复h1；完整控制比较继续拒绝其余变化。"""
    result = deepcopy(config)
    options = result.pop('affinity_hard')
    if (set(options) != {'reference', 'reference_sha256'} or options['reference'] != REFERENCE
            or options['reference_sha256'] != REFERENCE_SHA256):
        raise ValueError('H0 must reuse the registered mixed balance control')
    loss = result['direct_semantic_affinity']['affinity_loss']
    if isinstance(loss['hard_negative_weight'], bool) or loss['hard_negative_weight'] != 0.0:
        raise ValueError('Only the authorized hard_negative_weight=0 is allowed')
    cfg = result['backend_adaptation']
    if (loss['negative_weight'] != 1.0 or loss['hard_negative_gamma'] != 2.0 or cfg['pseudo_weight'] != .5
            or cfg['epochs'] != 20 or cfg['manual_repeats'] != 2
            or result['affinity_native']['penalty_sampling'] != 'mixed'):
        raise ValueError('H0 retains r1/gamma2/beta0.5, mixed sampling and 20 x 64 updates')
    loss['hard_negative_weight'] = 1.0
    return result


def verify_reference(config, state=None):
    """保留balance及祖先控制核验，并复用已完成的h1控制，不重训。"""
    base = reference_config(config)
    verify_balance_reference(base, state)
    folder = Path(project_path(config, config['affinity_hard']['reference']))
    prior = read(folder / 'status.json')
    if comparable(base) != comparable(prior['config']):
        raise RuntimeError('A configuration changed beyond the authorized hard-negative weight')
    if (prior['status'] != 'complete' or prior['updates'] != 1280 or prior['epoch'] != 20
            or prior.get('smoke') is not False or prior.get('size') != 1024
            or prior.get('precision') != 'FP32' or sha(folder / 'final.pt') != REFERENCE_SHA256):
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


def checkpoint_extras(config, epoch, *, smoke=False):
    loss = config['direct_semantic_affinity']['affinity_loss']
    return {'hard_state': dict(version=1, format=HARD_FORMAT, epoch=int(epoch),
        updates=int(epoch) * (8 if smoke else 64),
        negative_weight=loss['negative_weight'], hard_negative_weight=loss['hard_negative_weight'],
        hard_negative_gamma=loss['hard_negative_gamma'], pseudo_weight=config['backend_adaptation']['pseudo_weight'],
        reference_sha256=REFERENCE_SHA256, rng=capture_rng_state(),
        continuation='optimizer/scheduler/RNG saved; no resume CLI is provided')}


def _verify_checkpoints(config, output, state, updates):
    """部署模型、训练头及保存状态逐项核验，不借用旧扫参的格式标识。"""
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
    extra = head['hard_state']
    loss = config['direct_semantic_affinity']['affinity_loss']
    if (extra['version'] != 1 or extra['format'] != HARD_FORMAT or extra['epoch'] != epoch
            or extra['updates'] != updates or extra['negative_weight'] != loss['negative_weight']
            or extra['hard_negative_weight'] != loss['hard_negative_weight']
            or extra['hard_negative_gamma'] != loss['hard_negative_gamma']
            or extra['pseudo_weight'] != config['backend_adaptation']['pseudo_weight']
            or extra['reference_sha256'] != REFERENCE_SHA256):
        raise RuntimeError('Hard-negative continuation metadata differs')
    _verify_rng(extra['rng'])
    return dict(final_checkpoint_sha256=actual_sha, last_head_sha256=sha(output / 'last_head.pt'),
                frozen_saved_tensors_exact=True, head_bundle_exact=True, continuation_state_saved=True)


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
    return dict(format=HARD_FORMAT, passed=True, updates=updates, epoch=state['epoch'],
        negative_weight=1.0, hard_negative_weight=0.0, reference_hard_negative_weight=1.0,
        hard_negative_gamma=2.0, pseudo_weight=.5, reference_sha256=REFERENCE_SHA256,
        control_retrained=False, paired_sources_crops_augmentation_views_and_counts_exact=True,
        initialization_sha256=state['initial_state_sha256'], frozen_state_sha256=state['frozen_state_sha256'],
        restorer_state_sha256=state['restorer_state_sha256'],
        numerical_policy='finite gradients; different objectives/FP32 runs need not be bitwise equal',
        diagnostic_metrics_scope='recall is inside connection; specificity is true separation; not held-out accuracy',
        **checkpoint, **observation)


def train(config, output, smoke=False):
    verify_reference(config)
    monitor = partial(save_monitor,
        reference_folder=project_path(config, REFERENCE, 'deployment', 'patch1024'),
        reference_label='mixed balance e20 r1 h1 native1024')
    # native_train重新调用build_backend，复用原共同初态；控制e20只作对照，不是训练初始化。
    native_train(config, 1024, output, smoke=smoke, dataset_builder=mixed_data,
                 control_verifier=verify_reference, keep_smoke_checkpoint=True,
                 checkpoint_extras=partial(checkpoint_extras, config, smoke=smoke), monitor_saver=monitor)
    audit = verify_run(config, output, 8 if smoke else 1280)
    write_json(Path(output) / 'hard_receipt.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train/affinity_hard_h0.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('train', 'verify'), default='train')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    configuration = load_config(args.config)
    if args.mode == 'verify':
        print(json.dumps(verify_run(configuration, args.output, 8 if args.smoke else 1280), ensure_ascii=False))
    else:
        train(configuration, args.output, args.smoke)
