# -*- coding: utf-8 -*-
"""来源权重单变量：独立进程内缩放loss反传，保持旧trainer与原BCE日志不变。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path
import threading

import torch

from data.affinity_mixed import build_datasets as mixed_data
from models.fused_deployment import FUSED_DEPLOYMENT_FORMAT
from tools.affinity_native_views import save_monitor
from train_affinity_balance import comparable, verify_reference as verify_balance_reference
from train_affinity_connectivity import append_json
import train_affinity_native as native
from train_affinity_native import compare_draw, read
from train_affinity_sweep import (AUDIT_FLAGS, DEPENDENCY_KEYS, _check_rows, _rows,
                                 _verify_observability, _verify_rng, capture_rng_state)
from train_backend_adaptation import sha, tensor_digest, write_json
from utils.config import load_config, project_path


REFERENCE = 'outputs/affinity_balance/mixed'
REFERENCE_SHA256 = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
SOURCE_FORMAT = 'affinity_source_ratio_v1'
NOMINAL_COEFFICIENT_SUM = 1.5
SIDECAR = 'weighted_logits_grads.jsonl'


def source_coefficients(config):
    beta = config['backend_adaptation']['pseudo_weight']
    if isinstance(beta, bool) or beta not in (.25, .5):
        raise ValueError('Only beta=.25 or beta=.5 technical control is allowed')
    alpha = NOMINAL_COEFFICIENT_SUM / (1.0 + beta)
    return dict(beta=float(beta), alpha=alpha, manual_coefficient=alpha,
                pseudo_coefficient=alpha * beta, nominal_coefficient_sum=NOMINAL_COEFFICIENT_SUM)


def reference_config(config):
    result = deepcopy(config)
    options = result.pop('affinity_source')
    if (set(options) != {'reference', 'reference_sha256'} or options['reference'] != REFERENCE
            or options['reference_sha256'] != REFERENCE_SHA256):
        raise ValueError('P025 must reuse the registered mixed balance control')
    source_coefficients(result)
    loss, cfg = result['direct_semantic_affinity']['affinity_loss'], result['backend_adaptation']
    if (loss['negative_weight'] != 1.0 or loss['hard_negative_weight'] != 1.0
            or loss['hard_negative_gamma'] != 2.0 or cfg['epochs'] != 20 or cfg['manual_repeats'] != 2
            or result['affinity_native']['penalty_sampling'] != 'mixed'):
        raise ValueError('P025 retains r1/h1/gamma2, mixed sampling and 20 x 64 updates')
    cfg['pseudo_weight'] = .5
    return result


def verify_reference(config, state=None):
    base = reference_config(config)
    verify_balance_reference(base, state)
    folder = Path(project_path(config, REFERENCE))
    prior = read(folder / 'status.json')
    if comparable(base) != comparable(prior['config']):
        raise RuntimeError('A configuration changed beyond the authorized source ratio')
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


@contextmanager
def source_loss_scaling(config, output):
    """仅独立训练进程主线程使用；作用于反传而非LR，退出时恢复原函数。

    loss hook只把进入原loss的梯度乘alpha，原loss值及details完全不变。
    logits hook观察已经乘过alpha和来源系数的梯度，不修改梯度。
    该范数不能解释为参数梯度贡献；人工与SAM2有不同Jacobian和有效关系数。
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Source scaling requires an isolated training process main thread')
    original = native.compute_affinity_loss
    if getattr(original, '_affinity_source_scaler', False):
        raise RuntimeError('Nested source scaling contexts are not allowed')
    weights = source_coefficients(config)
    sidecar = Path(output) / SIDECAR
    counter = {'calls': 0, 'backwards': 0}

    def wrapped(logits, batch, loss_cfg, *, pseudo):
        call = counter['calls']
        if bool(pseudo) != bool(call % 2):
            raise RuntimeError('Expected exactly one manual then one SAM2 backward per update')
        loss, details = original(logits, batch, loss_cfg, pseudo=pseudo)
        if not loss.requires_grad or not logits.requires_grad:
            raise RuntimeError('Training source loss/logits must have gradients')
        if weights['alpha'] != 1.0:
            loss.register_hook(lambda incoming: incoming * weights['alpha'])
        update = call // 2 + 1
        epoch, step = (update - 1) // 64 + 1, (update - 1) % 64 + 1
        coefficient = weights['pseudo_coefficient'] if pseudo else weights['manual_coefficient']
        row = dict(call=call + 1, epoch=epoch, step=step, update=update,
                   source_kind='sam2' if pseudo else 'manual', source=batch['name'],
                   raw_bce=float(loss.detach()), source_coefficient=coefficient, **weights)

        def record_logits_gradient(gradient):
            norm = float(gradient.detach().float().norm())
            if not math.isfinite(norm):
                raise FloatingPointError('Nonfinite weighted logits gradient')
            append_json(sidecar, dict(**row, weighted_logits_grad_norm=norm,
                raw_logits_grad_norm=norm / coefficient,
                scope='loss-gradient diagnostic at logits, not parameter gradient or source importance'))
            counter['backwards'] += 1
            # 返回None保持gradient原样。
        logits.register_hook(record_logits_gradient)
        counter['calls'] += 1
        return loss, details

    wrapped._affinity_source_scaler = True
    native.compute_affinity_loss = wrapped
    try:
        yield counter
    finally:
        native.compute_affinity_loss = original


def checkpoint_extras(config, epoch, *, smoke=False):
    return {'source_state': dict(version=1, format=SOURCE_FORMAT, epoch=int(epoch),
        updates=int(epoch) * (8 if smoke else 64), **source_coefficients(config),
        reference_sha256=REFERENCE_SHA256, rng=capture_rng_state(), raw_bce_logs_unscaled=True,
        implementation='scalar loss-gradient alpha; existing trainer pseudo multiplier beta',
        continuation='optimizer/scheduler/RNG saved; no resume CLI is provided')}


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
    extra = head['source_state']
    if (extra['version'] != 1 or extra['format'] != SOURCE_FORMAT or extra['epoch'] != epoch
            or extra['updates'] != updates or extra['reference_sha256'] != REFERENCE_SHA256
            or extra.get('raw_bce_logs_unscaled') is not True
            or any(extra[k] != v for k, v in source_coefficients(config).items())):
        raise RuntimeError('Source-ratio continuation metadata differs')
    _verify_rng(extra['rng'])
    return dict(final_checkpoint_sha256=actual_sha, last_head_sha256=sha(output / 'last_head.pt'),
                frozen_saved_tensors_exact=True, head_bundle_exact=True, continuation_state_saved=True)


def _verify_sidecar(config, output, steps):
    receipts = _rows(output / SIDECAR)
    if len(receipts) != 2 * len(steps):
        raise RuntimeError('Missing or extra weighted-logits gradient receipts')
    weights = source_coefficients(config)
    for index, receipt in enumerate(receipts):
        row, pseudo = steps[index // 2], bool(index % 2)
        source = 'pseudo' if pseudo else 'manual'
        coefficient = weights['pseudo_coefficient'] if pseudo else weights['manual_coefficient']
        if (receipt['call'] != index + 1 or receipt['update'] != row['updates']
                or receipt['epoch'] != row['epoch'] or receipt['step'] != row['step']
                or receipt['source_kind'] != ('sam2' if pseudo else 'manual')
                or receipt['source'] != row[source] or receipt['raw_bce'] != row[source + '_bce']
                or receipt['source_coefficient'] != coefficient
                or any(receipt[k] != v for k, v in weights.items())):
            raise RuntimeError('Source gradient receipt differs from the actual draw/loss/coefficient')
        weighted, raw = receipt['weighted_logits_grad_norm'], receipt['raw_logits_grad_norm']
        if (not math.isfinite(weighted) or weighted < 0 or not math.isfinite(raw) or raw < 0
                or not math.isclose(weighted, coefficient * raw, rel_tol=1e-6, abs_tol=1e-12)):
            raise RuntimeError('Invalid weighted-logits gradient receipt')
    threshold = config['backend_adaptation']['grad_clip']
    clipped = sum(row['grad_norm'] > threshold for row in steps)
    return dict(weighted_logits_receipts=len(receipts), raw_bce_logs_unscaled=True,
                logits_gradient_scope='per-source loss gradient at logits; not parameter gradient or contribution ratio',
                parameter_clip=dict(threshold=threshold, triggered_updates=clipped,
                                    updates=len(steps), fraction=clipped / len(steps)))


def verify_run(config, output, updates=1280):
    if updates not in (8, 1280) or (source_coefficients(config)['beta'] == .5 and updates != 8):
        raise ValueError('Only eight-update technical control or P025 eight/1280 updates may pass')
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
    steps = _rows(output / 'steps.jsonl')
    _check_rows(steps, updates, config)
    for actual, expected in zip(steps, reference):
        compare_draw(actual, expected)
        for key in ('manual_crop', 'pseudo_crop', 'crop_attempts', 'training_view'):
            if actual[key] != expected[key]:
                raise RuntimeError('Sampling/crop/view differs from mixed balance: ' + key)
        for actual_source, expected_source in zip(actual['supervision'], expected['supervision']):
            for key in ('positive_edges', 'negative_edges'):
                if actual_source[key] != expected_source[key]:
                    raise RuntimeError('Supervised edge counts differ: ' + key)
    return dict(format=SOURCE_FORMAT, passed=True, updates=updates, epoch=state['epoch'],
        **source_coefficients(config), reference_beta=.5, reference_sha256=REFERENCE_SHA256,
        control_retrained=False, technical_control=source_coefficients(config)['beta'] == .5,
        paired_sources_crops_augmentation_views_and_counts_exact=True,
        initialization_sha256=state['initial_state_sha256'], frozen_state_sha256=state['frozen_state_sha256'],
        restorer_state_sha256=state['restorer_state_sha256'],
        **_verify_checkpoints(config, output, state, updates),
        **_verify_observability(config, output, updates, steps), **_verify_sidecar(config, output, steps))


def train(config, output, smoke=False):
    if source_coefficients(config)['beta'] == .5 and not smoke:
        raise ValueError('Beta=.5 is a short technical check; the complete control is already available')
    verify_reference(config)
    monitor = partial(save_monitor, reference_folder=project_path(config, REFERENCE, 'deployment', 'patch1024'),
                      reference_label='mixed balance e20 r1 h1 beta0.5 native1024')
    with source_loss_scaling(config, output) as counter:
        native.train(config, 1024, output, smoke=smoke, dataset_builder=mixed_data,
                     control_verifier=verify_reference, keep_smoke_checkpoint=True,
                     checkpoint_extras=partial(checkpoint_extras, config, smoke=smoke), monitor_saver=monitor)
    expected = 16 if smoke else 2560
    if counter != {'calls': expected, 'backwards': expected}:
        raise RuntimeError('Source loss and backward counts differ from the authorized update budget')
    audit = verify_run(config, output, 8 if smoke else 1280)
    write_json(Path(output) / 'source_receipt.json', audit)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_source_p025.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('train', 'verify'), default='train')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--technical-control', action='store_true')
    args = parser.parse_args()
    if args.technical_control and not args.smoke:
        parser.error('--technical-control requires --smoke; no complete control retraining')
    configuration = load_config(args.config)
    if args.technical_control:
        configuration['backend_adaptation']['pseudo_weight'] = .5
    if args.mode == 'verify':
        print(json.dumps(verify_run(configuration, args.output, 8 if args.smoke else 1280), ensure_ascii=False))
    else:
        train(configuration, args.output, args.smoke)
