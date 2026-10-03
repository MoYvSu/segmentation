# -*- coding: utf-8 -*-
"""独立保留域短测的配对、实际梯度及续训合同；不修改封存旧配方。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import torch

from utils.affinity_interior_loss import interior_objective
from utils.affinity_retain_loss import balanced_affinity_retention_kl
from utils.retention_support import array_sha, build_processed_negative_retention


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False, separators=(',', ':')).encode('utf8')).hexdigest()


def frozen_tensors_sha(head, tile, old_masks, new_masks):
    values = dict(head={k:array_sha(v) for k,v in head.state_dict().items()},
                  features=[array_sha(v) for v in tile['features']],
                  teacher=array_sha(tile['logits']),
                  old={k:array_sha(v) for k,v in old_masks.items()},
                  new={k:array_sha(v) for k,v in new_masks.items()})
    return canonical_sha(values)


def validate_peer(peer, opts, schedule=None):
    """固定唯一已完成connect，不重跑control，也不继承connect权重。"""
    if (peer.get('format') != 'affinity_interior_fit_v1' or not peer.get('complete')
            or peer.get('optimizer_steps_per_arm') != 32 or peer.get('microbatches_per_update') != 3
            or peer.get('snapshots') != [0, 8, 16, 32]
            or not peer.get('protected_inputs_exact') or not peer.get('protected_code_exact')):
        raise ValueError('Completed sealed interior-fit peer required')
    previous = peer['config']['affinity_interior_fit']
    if canonical_sha(previous) != canonical_sha(opts):
        raise ValueError('Inherited original connect recipe differs')
    arm = peer['arms']['connect']
    if (arm.get('steps') != 32 or arm.get('microbatches') != 96 or arm.get('local_weight') != .25
            or not arm.get('strict_reload_equal') or not arm.get('non_head_frozen_exact')
            or not arm.get('feature_cache_exact')):
        raise ValueError('Completed paired connect state required')
    if schedule is not None and schedule != peer['schedule']:
        raise ValueError('Actual source order or deployment-window exposure differs')
    return dict(control_reused=True, control_retrained=False, start_from_R1=True,
                copied_connect_weights=False, control_final_head_sha256=arm['final_head_sha256'],
                steps=32, microbatches=96, local_weight=.25,
                previous_recipe_sha256=canonical_sha(previous))


def expand_supervision(tile, gt, original, teacher, old_masks, *, confidence=.9):
    expanded, audit = build_processed_negative_retention(
        tile, gt, original, teacher, old_masks, confidence=confidence)
    masks = dict(old_masks)
    masks['retention_added'] = expanded['added']
    masks['retention_expanded'] = expanded['expanded']
    # expanded仅诊断；正式旧KL保留原mask及原分母，新增负边独立归一。
    target = old_masks['targets']
    normalization = []
    for channel in range(8):
        row = dict(channel=channel)
        for phase, value in (('positive', 1), ('negative', 0)):
            old = int((expanded['old'][0, channel] & (target[0, channel] == value)).sum())
            new = int((expanded['expanded'][0, channel] & (target[0, channel] == value)).sum())
            row[phase] = dict(old_edges=old, expanded_edges=new,
                              old_per_edge_relative_factor=1. if old else None,
                              hypothetical_expanded_relative_factor=old/new if old and new else None)
        normalization.append(row)
    audit.update(normalization=normalization, all_old_masks_preserved=True,
                 old_retention_selection_preserved=True, old_retention_gradients_identical=True,
                 KL_recipe_unchanged=True,
                 objective_contract='retention_guard_loss_v1',
                 normalization_caveat='Old KL keeps exact original channel/class means. Added-only negative KL has independent unchanged KL expression and coefficient 1; expanded means are not used in training.')
    return masks, expanded, audit


def retention_guard_objective(logits, teacher_logits, supervision, *, local_weight, loss_options=None):
    """旧目标逐项保留，加独立新增负关系KL，系数固定1；空新增完全退回旧目标。"""
    added = supervision['retention_added']
    if (not torch.is_tensor(added) or added.dtype != torch.bool or added.shape != logits.shape
            or added.device != logits.device or added.requires_grad
            or bool((added & supervision['retention']).any())
            or bool((added & ~supervision['full_legal']).any())
            or bool((supervision['targets'][added] != 0).any())):
        raise ValueError('Added retention must be disjoint legal negative relations')
    total, terms, details = interior_objective(logits, teacher_logits, supervision,
                                               local_weight=local_weight, loss_options=loss_options)
    term, count = balanced_affinity_retention_kl(logits, teacher_logits, supervision['targets'], added)
    terms['retention_added'] = term; details['retention_added'] = count
    return total+term, terms, details


def retention_gradient_summary(terms, head, local_weight):
    """分开报告旧共同梯度、新保护及加入保护后的共同梯度。"""
    parameters = tuple(head.parameters())
    grads = {}
    for name, loss in terms.items():
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        grads[name] = [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,values)]
    def norm(values):
        return sum(float(g.double().square().sum()) for g in values)**.5
    old = [a+.5*b+c for a,b,c in zip(grads['full'],grads['outer'],grads['retention'])]
    guarded = [a+b for a,b in zip(old,grads['retention_added'])]
    total = [a+float(local_weight)*b for a,b in zip(guarded,grads['local'])]
    norms = {name:norm(values) for name,values in grads.items()}
    old_norm,guard_norm = norm(old),norm(guarded)
    dot = sum(float((a.double()*b.double()).sum()) for a,b in zip(grads['retention_added'],grads['local']))
    return dict(unweighted_parameter_gradient_norms=norms,
                old_common_parameter_gradient_norm=old_norm,
                guarded_common_parameter_gradient_norm=guard_norm,
                total_parameter_gradient_norm=norm(total),
                added_to_old_common=norms['retention_added']/old_norm if old_norm else None,
                weighted_local_to_old_common=local_weight*norms['local']/old_norm if old_norm else None,
                weighted_local_to_guarded_common=local_weight*norms['local']/guard_norm if guard_norm else None,
                added_local_cosine=dot/(norms['retention_added']*norms['local'])
                    if norms['retention_added'] and norms['local'] else None)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume CUDA random-state device count differs')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def actual_added_gradient_gate(head, tile, old_masks, new_masks, loss_kwargs, optimizer_options):
    """一次可丢弃旧connect真实更新后验证新增KL会回拉，不污染正式起点。

    R1对R1的KL初态为零是正确行为。必须在真实head上更新一次再检查，
    而不是通过人工编辑logits或拿非空mask冒充参数保护有效。
    """
    added = new_masks['retention_added']
    if not bool(added.any()):
        raise ValueError('Actual gate requires newly eligible negative edges')
    rng = capture_rng()
    frozen_before = frozen_tensors_sha(head, tile, old_masks, new_masks)
    probe = deepcopy(head).eval().requires_grad_(True)
    try:
        optimizer = torch.optim.AdamW(probe.parameters(), **optimizer_options)
        optimizer.zero_grad(set_to_none=True)
        logits = probe(tile['features'])['affinity_logits'].float()
        initial_term, _ = balanced_affinity_retention_kl(logits, tile['logits'], old_masks['targets'], added)
        initial_grads = torch.autograd.grad(initial_term, (*probe.parameters(), logits), retain_graph=True,
                                            allow_unused=True)
        if float(initial_term.detach()) != 0. or any(
                bool(torch.count_nonzero(g)) for g in initial_grads if g is not None):
            raise RuntimeError('Actual copied R1 initial added KL or gradient is not zero')
        loss, _, _ = interior_objective(logits, tile['logits'], old_masks,
                                         local_weight=.25, loss_options=loss_kwargs)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(probe.parameters(), 1., error_if_nonfinite=True))
        if norm <= 0:
            raise RuntimeError('Disposable actual connect update has no parameter gradient')
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        actual = probe(tile['features'])['affinity_logits'].float()
        term, _ = balanced_affinity_retention_kl(actual, tile['logits'], old_masks['targets'], added)
        parameters = tuple(probe.parameters())
        gradients = torch.autograd.grad(term, (*parameters, actual), allow_unused=True)
        parameter_norm = sum(float(g.detach().double().square().sum()) for g in gradients[:-1] if g is not None)**.5
        logit_grad = gradients[-1]
        if (not torch.isfinite(term) or not parameter_norm > 0 or logit_grad is None
                or not bool(torch.isfinite(logit_grad).all())
                or bool(torch.count_nonzero(logit_grad[~added]))):
            raise RuntimeError('New retention does not produce finite real-head-only selected gradients')
        # dKL/dlogit与p_student-p_teacher同号；梯度下降把它拉回R1。
        drift = actual.detach().sigmoid()-tile['logits'].detach().sigmoid()
        nonzero = added & (logit_grad != 0)
        if not bool(nonzero.any()) or bool((logit_grad[nonzero]*drift[nonzero] < -1e-10).any()):
            raise RuntimeError('Added KL gradient does not pull actual probabilities toward frozen R1')
        return dict(format='retention_actual_gradient_v1', passed=True,
                    disposable_probe_only=True, formal_start_unchanged=True, input_mask_sha256=array_sha(added),
                    actual_selected_edges=int(added.sum()), nonzero_logit_gradient_edges=int(nonzero.sum()),
                    disposable_old_connect_gradient_norm=norm, added_KL=float(term.detach()),
                    actual_added_parameter_gradient_norm=parameter_norm,
                    outside_added_logit_gradient_max=0., probability_pullback_direction_verified=True,
                    first_R1_KL_exact_zero=True, first_R1_KL_gradients_exact_zero=True,
                    frozen_input_sha256=frozen_before, source_head_features_masks_unchanged=True,
                    no_synthetic_logit_edit=True)
    finally:
        restore_rng(rng)
        if frozen_tensors_sha(head, tile, old_masks, new_masks) != frozen_before:
            raise RuntimeError('Actual gradient gate mutated source head, frozen features, teacher or supervision')
        del probe


def validate_resume_state(state, identity, report):
    if (state.get('format') != 'affinity_retention_guard_head_v1' or state.get('arm') != 'guard'
            or state.get('identity') != identity or state.get('schedule') != report['schedule']
            or state.get('capture_sha256') != canonical_sha(report['captures'])
            or type(state.get('step')) is not int or state['step'] not in range(33)
            or len(state.get('steps', [])) != state['step']
            or [row['step'] for row in state['steps']] != list(range(1, state['step']+1))
            or any(type(row.get('step')) is not int for row in state['steps'])
            or state.get('local_weight') != .25):
        raise ValueError('Resume format, frozen identity, capture, schedule, or committed step prefix differs')
    if any(state[key] != report[key] for key in ('initial_head_sha256', 'runtime', 'precision', 'CPU_runtime')):
        raise ValueError('Resume head or runtime contract differs')
    for key in ('geometry_state_dict', 'optimizer', 'rng', 'snapshots'):
        if key not in state:
            raise ValueError('Incomplete optimizer/RNG/snapshot resume state: '+key)
    return state['step']
