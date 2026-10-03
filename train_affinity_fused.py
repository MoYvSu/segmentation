# -*- coding: utf-8 -*-
"""新GT融合排序实验；隔离原训练循环，不修改其封存源码或模块全局状态。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
from types import ModuleType

import numpy as np
import torch

from data.affinity_mixed import build_datasets
from train_affinity_balance import comparable, verify_reference as verify_balance
from train_backend_adaptation import fusion_options, sha, write_json
from tools.verify_affinity_newgt import audit_new_gt
from utils.affinity_fused_loss import fused_boundary_ranking_loss
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = 'config/train/affinity_fused.yaml'
REFERENCE = 'outputs/affinity_balance/mixed'
REFERENCE_SHA256 = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
FORMAT = 'affinity_fused_ranking_v1'
EXPECTED_OPTIONS = dict(format=FORMAT, reference=REFERENCE, reference_sha256=REFERENCE_SHA256,
    gt_npz_cohort_sha256='e7e34544fbb792c0e300d086689d28739882edd4b7aecd847f9e07499b49fe06',
    gt_class_cohort_sha256='1905ee57eec60c930fb75a5c1a8a605816a682262fa88e2c979d9a4fb87ce020',
    weight=.10, margin=.10, tail_fraction=.10, max_per_group=256, support_radius=4, manual_only=True)

# 与旧入口共用函数实现，但函数globals属于独立模块；退出后不污染native入口。
_path = ROOT / 'train_affinity_native.py'
_bytes = _path.read_bytes()
NATIVE_SHA256 = hashlib.sha256(_bytes).hexdigest()
_core = ModuleType(__name__ + '._native_core')
_core.__file__, _core.__package__ = str(_path), ''
exec(compile(_bytes, str(_path), 'exec'), _core.__dict__)


def _identity(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def reference_config(config):
    result = deepcopy(config)
    options = result.pop('affinity_fused', None)
    if not isinstance(options, dict) or set(options) != set(EXPECTED_OPTIONS):
        raise ValueError('The single registered fused-ranking recipe is required')
    for key, value in EXPECTED_OPTIONS.items():
        actual = options[key]
        if actual != value or (isinstance(value, bool) != isinstance(actual, bool)):
            raise ValueError('Unexpected fused-ranking option: ' + key)
    if _identity(comparable(result)) != _identity(comparable(load_config('config/train/affinity_balance_mixed.yaml'))):
        raise ValueError('A variable besides the additional ranking loss changed')
    return result


def verify_reference(config, state=None):
    base = reference_config(config)
    verify_balance(base, state)
    folder = Path(project_path(config, REFERENCE))
    prior = _core.read(folder / 'status.json')
    if (_identity(comparable(base)) != _identity(comparable(prior['config'])) or prior['status'] != 'complete'
            or prior['epoch'] != 20 or prior['updates'] != 1280 or prior['size'] != 1024
            or prior['smoke'] is not False or sha(folder/'final.pt') != REFERENCE_SHA256):
        raise RuntimeError('Registered mixed balance e20 control differs')
    for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'):
        if prior.get(key) is not True:
            raise RuntimeError('Reference audit failed: ' + key)
    if state is not None:
        for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data'):
            if state[key] != prior[key]:
                raise RuntimeError('Original initialization/frozen components/newGT sampling differ: ' + key)
    return prior, [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _gradient_audit(base, auxiliary, logits, shared, target, valid, fusion, weight):
    """只读诊断共享参数和输出受力；不改已有.grad、参数、模式或随机数。"""
    left = torch.autograd.grad(base, shared, retain_graph=True)[0].detach().float()
    right = torch.autograd.grad(auxiliary, shared, retain_graph=True)[0].detach().float() * weight
    ln, rn = float(left.norm()), float(right.norm())
    cos = float((left*right).sum())/(ln*rn) if ln and rn else None
    grad = torch.autograd.grad(auxiliary, logits, retain_graph=True)[0].detach()
    invalid = grad[~valid]
    if invalid.numel() and bool((invalid != 0).any()):
        raise RuntimeError('Added ranking gradient reached an unknown relation')
    _, before = fused_boundary_ranking_loss(logits.detach(), target, valid, fusion_kwargs=fusion)
    scale = .01/max(float(grad.abs().max()), 1.e-12)
    _, after = fused_boundary_ranking_loss(logits.detach()-scale*grad, target, valid, fusion_kwargs=fusion)
    if after['loss'] > before['loss'] + 1.e-6:
        raise RuntimeError('Small ranking descent increases its own objective')
    return dict(shared_parameter='affinity_decoder.affinity_head.0.weight',
        base_norm=ln, weighted_auxiliary_norm=rn, weighted_parameter_ratio=rn/max(ln,1.e-12),
        cosine=cos, unknown_logit_gradient_max=float(invalid.abs().max()) if invalid.numel() else 0.,
        directional_loss_before=before['loss'], directional_loss_after=after['loss'],
        selected_true_delta=(after['selected_true_mean']-before['selected_true_mean'])
            if before['selected_true_mean'] is not None else None,
        selected_interior_delta=(after['selected_interior_mean']-before['selected_interior_mean'])
            if before['selected_interior_mean'] is not None else None)


@contextmanager
def ranking_objective(config, output, *, smoke):
    original_loss, original_configure, original_append = (_core.compute_affinity_loss,
        _core.configure_training, _core.append_json)
    options = config['affinity_fused']
    mode, kwargs = fusion_options(config)
    if mode != 'gated':
        raise ValueError('This experiment retains gated deployment')
    fusion = dict(mode=mode, **kwargs)
    records, state = [], dict(shared=None)

    def configure(model):
        original_configure(model)
        state['shared'] = model.affinity_decoder.affinity_head[0].weight

    def loss_fn(logits, batch, loss_cfg, *, pseudo):
        base, details = original_loss(logits, batch, loss_cfg, pseudo=pseudo)
        if pseudo:
            return base, details
        if bool(batch['uncovered_boundary_source'].any()) or loss_cfg['manual_uncovered_as_boundary']:
            raise RuntimeError('Fused supervision requires canonical newGT with unknown pairs ignored')
        target, valid = build_affinity_targets_torch(batch['affinity_instance_map'],
            batch['affinity_valid_content'])
        auxiliary, metrics = fused_boundary_ranking_loss(logits, target, valid,
            fusion_kwargs=fusion, margin=options['margin'], tail_fraction=options['tail_fraction'],
            max_per_group=options['max_per_group'])
        index = len(records)
        row = dict(update=index+1, epoch=index//64+1, step=index%64+1,
                   source=batch['name'], manual_bce=float(base.detach()), ranking=metrics,
                   weighted_ranking=float(auxiliary.detach())*options['weight'])
        if smoke or index % 64 == 0:
            if state['shared'] is None:
                raise RuntimeError('Shared head parameter was not captured')
            row['gradient'] = _gradient_audit(base, auxiliary, logits, state['shared'],
                                             target, valid, fusion, options['weight'])
        records.append(row)
        original_append(Path(output)/'fused_loss.jsonl', row)
        return base + options['weight']*auxiliary, details

    def append(path, row):
        if Path(path).name == 'steps.jsonl':
            record = records[-1]
            if row['updates'] != record['update']:
                raise RuntimeError('Missing or duplicated manual ranking receipt')
            row['manual_total'] = row['manual_bce']
            row['manual_bce'] = record['manual_bce']
            row.update(auxiliary_weight=options['weight'], weighted_ranking=record['weighted_ranking'],
                       ranking_active_pairs=record['ranking']['active_pairs'],
                       ranking_valid_groups=record['ranking']['valid_groups'])
        elif Path(path).name == 'epochs.jsonl':
            epoch_records = [r for r in records if r['epoch'] == row['epoch']]
            row['manual_total'] = row['manual_bce']
            row['manual_bce'] = float(np.mean([r['manual_bce'] for r in epoch_records]))
            row['weighted_ranking'] = float(np.mean([r['weighted_ranking'] for r in epoch_records]))
            row['ranking_active_updates'] = sum(r['ranking']['active_pairs'] > 0 for r in epoch_records)
        original_append(path, row)

    def checkpoint_extras(epoch):
        return dict(fused_experiment=dict(format=FORMAT, options=options, manual_updates=len(records),
            last_epoch=epoch, native_source_sha256=NATIVE_SHA256), rng_state=_rng_state())

    _core.compute_affinity_loss, _core.configure_training, _core.append_json = loss_fn, configure, append
    try:
        yield records, checkpoint_extras
    finally:
        _core.compute_affinity_loss, _core.configure_training, _core.append_json = (
            original_loss, original_configure, original_append)


def monitor(model, restorer, config, device, output, epoch, size, *, smoke=False):
    # 只替换比较图中的参照位置，模型输入、种子与原monitor完全复用。
    _core.save_monitor(model, restorer, config, device, output, epoch, size, smoke=smoke,
        reference_folder=project_path(config, REFERENCE, 'deployment/patch1024'),
        reference_label='mixed balance e20 gated')


def verify_run(config, output, expected_updates):
    folder = Path(output)
    status = _core.read(folder/'status.json')
    _, reference = verify_reference(config, status)
    if (status['status'] != 'complete' or status['updates'] != expected_updates
            or status['config'] != json.loads(json.dumps(config))):
        raise RuntimeError('Fused-ranking run is incomplete or configuration differs')
    for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'):
        if status.get(key) is not True:
            raise RuntimeError('Candidate state audit failed: ' + key)
    rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    fused = [json.loads(s) for s in (folder/'fused_loss.jsonl').read_text().splitlines()]
    if len(rows) != expected_updates or len(fused) != expected_updates:
        raise RuntimeError('Incomplete update receipts')
    for index, (row, old, extra) in enumerate(zip(rows, reference, fused)):
        _core.compare_draw(row, old)
        for key in ('manual_crop', 'pseudo_crop', 'crop_attempts', 'training_view'):
            if row.get(key) != old.get(key):
                raise RuntimeError('Crop/view differs: ' + key)
        if row['auxiliary_weight'] != config['affinity_fused']['weight']:
            raise RuntimeError('Ranking coefficient changed')
        if (extra['update'] != index+1 or extra['epoch'] != row['epoch'] or extra['step'] != row['step']
                or extra['source'] != row['manual'] or extra['manual_bce'] != row['manual_bce']
                or extra['weighted_ranking'] != row['weighted_ranking']
                or extra['ranking']['active_pairs'] != row['ranking_active_pairs']
                or extra['ranking']['valid_groups'] != row['ranking_valid_groups']):
            raise RuntimeError('Ranking and original BCE sidecar differ')
        if (abs(extra['weighted_ranking'] - config['affinity_fused']['weight']*extra['ranking']['loss']) > 1.e-7
                or abs(row['manual_total']-row['manual_bce']-row['weighted_ranking']) > 1.e-6):
            raise RuntimeError('Logged manual objective differs from the registered sum')
    if sha(folder/'final.pt') != status['final_checkpoint_sha256']:
        raise RuntimeError('Completed fused bundle changed')
    head = torch.load(folder/'last_head.pt', map_location='cpu', weights_only=False)
    if (not all(k in head for k in ('optimizer','scheduler','rng_state','fused_experiment'))
            or head['fused_experiment']['manual_updates'] != expected_updates):
        raise RuntimeError('Continuation state is incomplete')
    gradients = [r['gradient'] for r in fused if 'gradient' in r]
    active = sum(r['ranking']['active_pairs'] > 0 for r in fused)
    if not active or not any(g['weighted_auxiliary_norm'] > 0 for g in gradients):
        raise RuntimeError('The added supervision has no measured effect')
    return dict(passed=True, updates=expected_updates, control_retrained=False,
        new_gt=True, identical_initialization_and_draws=True, frozen_unchanged=True,
        strict_reload_equal=True, active_ranking_updates=active, gradient_audits=gradients,
        checkpoint_sha256=status['final_checkpoint_sha256'], continuation_state_saved=True)


def train(config, output, *, smoke=False):
    reference_config(config)
    new_gt = audit_new_gt(config)
    if (new_gt['npz_cohort_sha256'] != config['affinity_fused']['gt_npz_cohort_sha256']
            or new_gt['class_cohort_sha256'] != config['affinity_fused']['gt_class_cohort_sha256']):
        raise RuntimeError('Registered processed newGT files differ')
    with ranking_objective(config, output, smoke=smoke) as (_, extras):
        _core.train(config, 1024, output, smoke=smoke, dataset_builder=build_datasets,
            control_verifier=verify_reference, keep_smoke_checkpoint=True,
            checkpoint_extras=extras, monitor_saver=monitor)
    if audit_new_gt(config) != new_gt:
        raise RuntimeError('NewGT or source identity changed during training')
    write_json(Path(output)/'newgt_receipt.json', new_gt)
    return verify_run(config, output, 8 if smoke else 1280)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    train(load_config(args.config), args.output, smoke=args.smoke)
