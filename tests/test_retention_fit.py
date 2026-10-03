# -*- coding: utf-8 -*-
"""虚构CPU短测门禁：旧控制复用、概率回拉梯度和原起点隔离。"""
from copy import deepcopy
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from utils.affinity_interior_loss import _native_masks, interior_objective
from utils.affinity_loss import build_affinity_targets_torch
from utils.affinity_retain_loss import balanced_affinity_retention_kl, build_reliable_retention_mask
from utils.offset_letterbox import letterbox_instance_geometry
from utils.retention_fit import (actual_added_gradient_gate, canonical_sha, capture_rng,
                                 expand_supervision, frozen_tensors_sha, restore_rng,
                                 retention_guard_objective, validate_peer, validate_resume_state)


def _case():
    gt = np.ones((6, 6), np.int32)
    gt[:, 3:] = 2
    original = gt > 0
    original[:, 3] = False
    original[0] = True  # 旧已覆盖断开边和新增filled断开边同时存在。
    grid_gt, valid, geometry = letterbox_instance_geometry(gt, 6, 6)
    tile = dict(box=[0, 0, 6, 6], grid_gt=grid_gt, valid=valid,
                geometry=geometry.to_dict(), pad_h=0, pad_w=0)
    target, legal = build_affinity_targets_torch(torch.from_numpy(grid_gt)[None],
                                                torch.from_numpy(valid)[None, None])
    teacher = torch.where(target == 1, 3., -3.).float()
    band = torch.zeros((6, 6), dtype=torch.bool)
    native, _, _, _, _ = _native_masks(tile, gt, original, np.zeros(gt.shape, bool))
    native = torch.from_numpy(native)
    retained, _ = build_reliable_retention_mask(teacher, target, legal, band,
                                                instance_map=grid_gt, valid_content=valid)
    local = legal & native & (target == 1)
    local[:, 4:] = False
    masks = dict(targets=target, full_legal=legal, band=band,
                 native_legal=native, retention=retained & native,
                 interior=local.clone(), local=local, outer=legal & native)
    tile['logits'] = teacher
    tile['features'] = (torch.zeros((1, 1, 6, 6)),)
    return tile, gt, original, masks


class _Head(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits.clone(), requires_grad=False)

    def forward(self, features):
        return {'affinity_logits': self.logits}


def test_new_guard_keeps_every_old_target_and_old_retention_selection():
    tile, gt, original, old = _case()
    before = deepcopy(old)
    masks, support, audit = expand_supervision(tile, gt, original, tile['logits'], old)
    assert support['added'].any()
    for key in old:
        assert torch.equal(old[key], before[key])
        assert torch.equal(masks[key], before[key])
    assert torch.equal(masks['retention_added'], support['added'])
    assert torch.equal(masks['retention_expanded'], support['expanded'])
    assert not (old['retention'] & ~masks['retention_expanded']).any()
    assert torch.all(old['targets'][support['added']] == 0)
    assert audit['old_retention_gradients_identical']


def test_guard_old_objective_values_and_gradients_are_exact_and_increment_has_no_other_support():
    tile, gt, original, old = _case()
    masks, support, _ = expand_supervision(tile, gt, original, tile['logits'], old)
    student = (tile['logits'].clone() + .2).requires_grad_(True)
    previous, previous_terms, _ = interior_objective(
        student, tile['logits'], old, local_weight=.25)
    guarded, terms, _ = retention_guard_objective(
        student, tile['logits'], masks, local_weight=.25)
    for key in previous_terms:
        assert torch.equal(terms[key], previous_terms[key])
        old_gradient = torch.autograd.grad(previous_terms[key], student, retain_graph=True)[0]
        kept_gradient = torch.autograd.grad(terms[key], student, retain_graph=True)[0]
        assert torch.equal(old_gradient, kept_gradient)
    added, _ = balanced_affinity_retention_kl(
        student, tile['logits'], old['targets'], support['added'])
    assert torch.equal(added, terms['retention_added'])
    assert torch.equal(guarded, previous + added)
    old_gradient = torch.autograd.grad(previous, student, retain_graph=True)[0]
    add_gradient = torch.autograd.grad(added, student, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(guarded, student, retain_graph=True)[0]
    assert torch.allclose(new_gradient, old_gradient + add_gradient, rtol=1e-6, atol=1e-8)
    assert not add_gradient[~support['added']].count_nonzero()
    assert add_gradient[support['added']].count_nonzero()


def test_empty_increment_is_exact_old_kl_and_cannot_pass_nonempty_gradient_gate():
    tile, gt, original, old = _case()
    original[:] = True
    native, _, _, _, _ = _native_masks(tile, gt, original, np.zeros(gt.shape, bool))
    old['native_legal'] = torch.from_numpy(native)
    retained, _ = build_reliable_retention_mask(tile['logits'], old['targets'], old['full_legal'],
        old['band'], instance_map=tile['grid_gt'], valid_content=tile['valid'])
    old['retention'] = retained & old['native_legal']
    masks, support, _ = expand_supervision(tile, gt, original, tile['logits'], old)
    assert not support['added'].any()
    student = tile['logits'].clone().requires_grad_(True) + .2
    previous, _ = balanced_affinity_retention_kl(student, tile['logits'], old['targets'], old['retention'])
    expanded, _ = balanced_affinity_retention_kl(student, tile['logits'], masks['targets'], masks['retention'])
    assert torch.equal(previous, expanded)
    old_total, _, _ = interior_objective(student, tile['logits'], old, local_weight=.25)
    guard_total, _, _ = retention_guard_objective(student, tile['logits'], masks, local_weight=.25)
    assert torch.equal(old_total, guard_total)
    old_gradient = torch.autograd.grad(old_total, student, retain_graph=True)[0]
    new_gradient = torch.autograd.grad(guard_total, student, retain_graph=True)[0]
    assert torch.equal(old_gradient, new_gradient)
    with pytest.raises(ValueError, match='newly eligible'):
        actual_added_gradient_gate(_Head(tile['logits']), tile, old, masks, {}, dict(lr=2e-5))


def test_disposable_parameter_gradient_probe_does_not_change_formal_start_or_random_stream():
    tile, gt, original, old = _case()
    masks, support, _ = expand_supervision(tile, gt, original, tile['logits'], old)
    head = _Head(tile['logits']).eval()
    before = frozen_tensors_sha(head, tile, old, masks)
    state = capture_rng()
    expected = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng(state)
    receipt = actual_added_gradient_gate(head, tile, old, masks, {},
                                        dict(lr=2e-5, eps=1e-4, weight_decay=1e-4))
    actual = (random.random(), float(np.random.random()), torch.rand(3))
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert receipt['first_R1_KL_gradients_exact_zero']
    assert receipt['actual_added_parameter_gradient_norm'] > 0
    assert 0 < receipt['nonzero_logit_gradient_edges'] <= int(support['added'].sum())
    assert receipt['outside_added_logit_gradient_max'] == 0
    assert receipt['probability_pullback_direction_verified']
    assert frozen_tensors_sha(head, tile, old, masks) == before
    assert head.logits.grad is None and not head.logits.requires_grad
    assert tile['logits'].grad is None


@pytest.mark.parametrize('mutation', ['positive', 'illegal', 'old_overlap', 'nonbool', 'wrong_shape'])
def test_added_objective_cannot_include_positive_unknown_padding_or_already_retained_edges(mutation):
    tile, gt, original, old = _case()
    masks, _, _ = expand_supervision(tile, gt, original, tile['logits'], old)
    masks = deepcopy(masks)
    if mutation in ('positive', 'illegal', 'old_overlap'):
        allowed = {'positive': old['full_legal'] & (old['targets'] == 1),
                   'illegal': ~old['full_legal'], 'old_overlap': old['retention']}
        index = tuple(torch.nonzero(allowed[mutation], as_tuple=False)[0].tolist())
        masks['retention_added'][index] = True
    elif mutation == 'nonbool':
        masks['retention_added'] = masks['retention_added'].float()
    else:
        masks['retention_added'] = masks['retention_added'][:, :, :-1]
    student = (tile['logits'].clone() + .2).requires_grad_(True)
    with pytest.raises(ValueError):
        retention_guard_objective(student, tile['logits'], masks, local_weight=.25)


def _peer():
    opts = dict(steps=32, accumulation=3, local_weight=.25, seed=8)
    schedule = [[dict(source='fiction', tile=0)]]
    peer = dict(format='affinity_interior_fit_v1', complete=True,
                optimizer_steps_per_arm=32, microbatches_per_update=3, snapshots=[0, 8, 16, 32],
                protected_inputs_exact=True, protected_code_exact=True,
                config=dict(affinity_interior_fit=opts), schedule=schedule,
                arms=dict(connect=dict(steps=32, microbatches=96, local_weight=.25,
                    strict_reload_equal=True, non_head_frozen_exact=True, feature_cache_exact=True,
                    final_head_sha256='fictional_final_head')))
    return peer, opts, schedule


def test_paired_control_is_reused_and_never_the_new_initialization():
    peer, opts, schedule = _peer()
    receipt = validate_peer(peer, opts, schedule)
    assert receipt['control_reused'] and not receipt['control_retrained']
    assert receipt['start_from_R1'] and not receipt['copied_connect_weights']


@pytest.mark.parametrize('mutation', ['incomplete', 'changed_recipe', 'changed_source_order',
                                     'changed_window', 'unfrozen', 'no_strict_reload', 'extra_updates'])
def test_peer_strong_identity_or_actual_exposure_mismatch_rejected(mutation):
    peer, opts, schedule = _peer()
    opts = deepcopy(opts)
    schedule = deepcopy(schedule)
    if mutation == 'incomplete':
        peer['complete'] = False
    elif mutation == 'changed_recipe':
        opts['seed'] += 1
    elif mutation == 'changed_source_order':
        schedule[0][0]['source'] = 'other_fiction'
    elif mutation == 'changed_window':
        schedule[0][0]['tile'] = 1
    elif mutation == 'unfrozen':
        peer['arms']['connect']['non_head_frozen_exact'] = False
    elif mutation == 'no_strict_reload':
        peer['arms']['connect']['strict_reload_equal'] = False
    else:
        peer['arms']['connect']['steps'] = 33
    with pytest.raises(ValueError):
        validate_peer(peer, opts, schedule)


def _resume():
    identity = dict(checkpoint_sha256='fiction_model', recipe_sha256='fiction_recipe')
    report = dict(schedule=[[dict(source='fiction', tile=0)]], captures=dict(fiction={}),
                  initial_head_sha256='fiction_head', runtime={}, precision={}, CPU_runtime={})
    state = dict(format='affinity_retention_guard_head_v1', arm='guard', identity=identity,
                 schedule=report['schedule'], capture_sha256=canonical_sha(report['captures']),
                 step=1, steps=[dict(step=1)], local_weight=.25,
                 geometry_state_dict={}, optimizer={}, rng={}, snapshots={})
    state.update({key: report[key] for key in ('initial_head_sha256', 'runtime', 'precision', 'CPU_runtime')})
    return state, identity, report


@pytest.mark.parametrize('mutation', ['none', 'wrong_format', 'wrong_identity', 'wrong_capture',
                                     'wrong_schedule', 'duplicate_step', 'uncommitted_step',
                                     'wrong_init', 'wrong_precision', 'missing_optimizer', 'missing_rng',
                                     'boolean_step', 'boolean_step_row'])
def test_resume_requires_committed_prefix_frozen_capture_and_optimizer_random_state(mutation):
    state, identity, report = _resume()
    state = deepcopy(state)
    if mutation == 'none':
        assert validate_resume_state(state, identity, report) == 1
        return
    if mutation == 'wrong_format':
        state['format'] = 'old_interior_fit_format'
    elif mutation == 'wrong_identity':
        state['identity']['checkpoint_sha256'] = 'other'
    elif mutation == 'wrong_capture':
        state['capture_sha256'] = 'other'
    elif mutation == 'wrong_schedule':
        state['schedule'][0][0]['tile'] = 1
    elif mutation == 'duplicate_step':
        state['step'] = 2
        state['steps'] = [dict(step=1), dict(step=1)]
    elif mutation == 'uncommitted_step':
        state['step'] = 2
    elif mutation == 'wrong_init':
        state['initial_head_sha256'] = 'other'
    elif mutation == 'wrong_precision':
        state['precision'] = {'dtype': 'fiction_changed'}
    elif mutation == 'missing_optimizer':
        state.pop('optimizer')
    elif mutation == 'boolean_step':
        state['step'] = True
    elif mutation == 'boolean_step_row':
        state['steps'][0]['step'] = True
    else:
        state.pop('rng')
    with pytest.raises(ValueError):
        validate_resume_state(state, identity, report)


@pytest.mark.parametrize('key,value', [('added_retention_weight', 2.),
                                     ('loss_contract', 'fictional_other_loss'),
                                     ('retention_objective', 'expanded_mask_mean')])
def test_registered_loss_version_and_fixed_added_weight_cannot_be_silently_overridden(tmp_path, key, value):
    from tools.probe_affinity_retention_guard import run
    base = Path(__file__).resolve().parents[1] / 'config/train/affinity_retention_guard.yaml'
    config = tmp_path / 'changed.yaml'
    config.write_text(yaml.safe_dump({'_base': str(base), 'affinity_retention_guard': {key: value}}),
                      encoding='utf8')
    args = SimpleNamespace(config=str(config), go32=False, resume=False, preflight_only=False)
    with pytest.raises(ValueError, match='old-plus-added'):
        run(args)


def _assert_exact_nested(left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_exact_nested(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_exact_nested(a, b)
    else:
        assert left == right


def test_empty_added_full_32_update_accumulation_and_optimizer_pipeline_matches_old_connect_exactly():
    tile, gt, original, old = _case()
    original[:] = True
    native, _, _, _, _ = _native_masks(tile, gt, original, np.zeros(gt.shape, bool))
    old['native_legal'] = torch.from_numpy(native)
    retained, _ = build_reliable_retention_mask(tile['logits'], old['targets'], old['full_legal'],
        old['band'], instance_map=tile['grid_gt'], valid_content=tile['valid'])
    old['retention'] = retained & old['native_legal']
    guarded_masks, support, _ = expand_supervision(tile, gt, original, tile['logits'], old)
    assert not support['added'].any()
    old_head = _Head(tile['logits']).eval().requires_grad_(True)
    guard_head = deepcopy(old_head)
    opts = dict(lr=2e-5, eps=1e-4, weight_decay=1e-4)
    old_optimizer = torch.optim.AdamW(old_head.parameters(), **opts)
    guard_optimizer = torch.optim.AdamW(guard_head.parameters(), **opts)
    for _ in range(32):
        old_optimizer.zero_grad(set_to_none=True)
        guard_optimizer.zero_grad(set_to_none=True)
        for _ in range(3):
            old_loss, _, _ = interior_objective(old_head(tile['features'])['affinity_logits'],
                tile['logits'], old, local_weight=.25)
            guard_loss, _, _ = retention_guard_objective(guard_head(tile['features'])['affinity_logits'],
                tile['logits'], guarded_masks, local_weight=.25)
            assert torch.equal(old_loss, guard_loss)
            (old_loss / 3).backward()
            (guard_loss / 3).backward()
        assert torch.equal(old_head.logits.grad, guard_head.logits.grad)
        old_norm = torch.nn.utils.clip_grad_norm_(old_head.parameters(), 1., error_if_nonfinite=True)
        guard_norm = torch.nn.utils.clip_grad_norm_(guard_head.parameters(), 1., error_if_nonfinite=True)
        assert torch.equal(old_norm, guard_norm)
        old_optimizer.step()
        guard_optimizer.step()
        _assert_exact_nested(old_head.state_dict(), guard_head.state_dict())
        _assert_exact_nested(old_optimizer.state_dict(), guard_optimizer.state_dict())
    reloaded = _Head(tile['logits']).eval()
    reloaded.load_state_dict(guard_head.state_dict(), strict=True)
    assert torch.equal(reloaded(tile['features'])['affinity_logits'],
                       old_head(tile['features'])['affinity_logits'])
