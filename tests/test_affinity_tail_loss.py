# -*- coding: utf-8 -*-
"""完整困难尾部的代表性选位：原损失兼容、覆盖、真实梯度及可信域检查。"""
import json
import random

import numpy as np
import pytest
import torch

from utils.affinity_fused_loss import fused_boundary_ranking_loss, fused_ranking_domains
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_loss import build_affinity_targets_torch
from utils.affinity_tail_loss import SAMPLING_FORMAT, tail_boundary_ranking_loss, tail_rank_positions

OPTIONS = dict(mode='gated', short_reduction='top2', distance2_weight=.5,
               distance4_weight=.25, support_threshold=.2, support_temperature=.05)


def geometry(kind='straight', size=48):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    ids = torch.ones((size, size), dtype=torch.long)
    if kind == 'straight':
        ids[:, size//2:] = 2
    elif kind == 'diagonal':
        ids[xx > yy] = 2
    elif kind == 'tee':
        ids[:, size//2:] = 2
        ids[size//2:, size//2:] = 3
    elif kind in ('loop', 'unknown_hole'):
        ids[14:34, 14:34] = 2
        if kind == 'unknown_hole':
            ids[21:27, 21:27] = 0
    elif kind == 'many':
        ids = yy.div(12, rounding_mode='floor') * ((size+11)//12) + xx.div(12, rounding_mode='floor') + 1
    else:
        raise ValueError(kind)
    return build_affinity_targets_torch(ids[None], torch.ones((1, 1, size, size), dtype=torch.bool))


def test_uncapped_loss_and_all_logit_gradients_equal_original_exactly():
    target, valid = geometry()
    target, valid = target.repeat(2, 1, 1, 1), valid.repeat(2, 1, 1, 1)
    logits = torch.randn(target.shape, generator=torch.Generator().manual_seed(77),
                         dtype=torch.float64).requires_grad_()
    old, old_info = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    new, info = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert all(row['tail_true_boundary'] <= 256 and row['tail_deep_interior'] <= 256 for row in info['groups'])
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, logits)[0], torch.autograd.grad(new, logits)[0])
    assert all(info[key] == old_info[key] for key in old_info if key != 'groups')


def test_capped_samples_cover_full_tail_and_hinge_includes_inactive_pairs():
    target, valid = geometry('many', 180)
    q = torch.linspace(.01, .99, 180*180, dtype=torch.float64).reshape(1, 1, 180, 180)
    logits = torch.logit(1-q.expand(-1, 8, -1, -1)).clone().requires_grad_()
    loss, info = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert info['sampling_format'] == SAMPLING_FORMAT
    row = info['groups'][0]
    assert row['tail_true_boundary'] > 256 and row['tail_deep_interior'] > 256
    for name, count_key in [('true', 'tail_true_boundary'), ('interior', 'tail_deep_interior')]:
        ranks = row['selected_'+name+'_ranks']
        assert len(ranks) == 256 and ranks[0] == 0 and ranks[-1] == row[count_key]-1
        gaps = [b-a for a, b in zip(ranks, ranks[1:])]
        assert min(gaps) > 0 and max(gaps)-min(gaps) <= 1
    boundary, interior = fused_ranking_domains(target, valid)
    scores = affinity_boundary_probability(logits, **OPTIONS)[:, 0]
    p = scores[boundary].sort().values[:row['tail_true_boundary']][row['selected_true_ranks']]
    n = scores[interior].sort(descending=True).values[:row['tail_deep_interior']][row['selected_interior_ranks']]
    violations = .1 + n[None, :] - p[:, None]
    assert torch.allclose(loss, violations.relu().mean(), rtol=0, atol=1.e-14)
    assert info['active_pairs'] == int((violations > 0).sum())
    old, original = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert info['selected_true_mean'] > original['selected_true_mean']
    assert info['selected_interior_mean'] < original['selected_interior_mean']
    assert new_finite_gradient(loss, logits).abs().sum() > 0
    json.dumps(info, allow_nan=False)


def new_finite_gradient(loss, logits):
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.isfinite(gradient).all()
    return gradient


def test_strong_rare_errors_are_diluted_but_remain_supervised():
    target, valid = geometry('many', 180)
    boundary, interior = fused_ranking_domains(target, valid)
    q = torch.full(boundary.shape, .2, dtype=torch.float64)
    q[boundary] = .7
    ipos = interior[0].nonzero()
    strong = ipos[:128]
    q[0, strong[:, 0], strong[:, 1]] = .95
    logits = torch.logit(1-q[:, None].expand(-1, 8, -1, -1)).clone().requires_grad_()
    old, old_info = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    new, info = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert 0 < new < old
    assert 0 < info['active_pairs'] < info['total_pairs']
    # 非活动组合仍在分母内；本例活动组合的间隔缺口均为.1+.95-.7。
    assert new.item() == pytest.approx(.35 * info['active_pairs']/info['total_pairs'], abs=1.e-12)
    assert info['selected_interior_mean'] < old_info['selected_interior_mean']
    old_grad = torch.autograd.grad(old, logits)[0]
    new_grad = new_finite_gradient(new, logits)
    rare_old = old_grad[:, :, strong[:, 0], strong[:, 1]].abs().sum()
    rare_new = new_grad[:, :, strong[:, 0], strong[:, 1]].abs().sum()
    assert 0 < rare_new < rare_old
    assert info['groups'][0]['selected_interior_ranks'][0] == 0


def test_valid_zero_loss_group_still_has_equal_sample_weight():
    target, valid = geometry('many', 96)
    logits = torch.zeros_like(target, requires_grad=True)
    one, _ = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    mixed, info = tail_boundary_ranking_loss(torch.cat([logits, target*32-16]),
        target.repeat(2, 1, 1, 1), valid.repeat(2, 1, 1, 1), fusion_kwargs=OPTIONS)
    assert info['valid_groups'] == 2 and info['groups'][1]['loss'] == 0
    assert mixed.item() == pytest.approx(one.item()/2)


@pytest.mark.parametrize('kind', ['straight', 'diagonal', 'tee', 'loop', 'unknown_hole'])
def test_perfect_geometries_have_zero_loss_and_gradient(kind):
    target, valid = geometry(kind)
    logits = (target * 32 - 16).requires_grad_()
    loss, info = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert loss.item() == 0 and info['active_pairs'] == 0
    assert new_finite_gradient(loss, logits).count_nonzero() == 0


def test_unknown_padding_and_gated_support_keep_original_protection():
    target, valid = geometry('unknown_hole')
    valid[:, 6, 20, 11] = False
    logits = torch.randn(target.shape, generator=torch.Generator().manual_seed(4)).requires_grad_()
    loss, _ = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS, max_per_group=8)
    gradient = new_finite_gradient(loss, logits)
    assert gradient[~valid].count_nonzero() == 0
    assert not fused_ranking_domains(target, valid)[1][0, 20, 11]


def test_function_consumes_no_rng_and_does_not_mutate_inputs():
    target, valid = geometry('many', 96)
    logits = torch.zeros_like(target, requires_grad=True)
    before = [v.detach().clone() for v in (logits, target, valid)]
    py, np_state, state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    a, ai = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    b, bi = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert torch.equal(a, b) and ai == bi
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), state)
    after_np = np.random.get_state()
    assert np_state[0] == after_np[0] and np.array_equal(np_state[1], after_np[1]) and np_state[2:] == after_np[2:]
    assert all(torch.equal(a, b) for a, b in zip((logits, target, valid), before))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_reduced_precision_inputs_use_float32_finite_loss_and_gradient(dtype):
    target, valid = geometry('many', 96)
    logits = torch.zeros_like(target, dtype=dtype, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, _ = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert new_finite_gradient(loss, logits).dtype == dtype


@pytest.mark.parametrize('kind', ['no_valid', 'no_boundary', 'no_interior', 'tiny'])
def test_empty_supervision_is_finite_differentiable_zero(kind):
    target, valid = geometry()
    if kind == 'no_valid': valid.zero_()
    elif kind == 'no_boundary': target.fill_(1)
    elif kind == 'no_interior': target.zero_()
    else: target, valid = target[:, :, :5, :5], valid[:, :, :5, :5]
    logits = torch.full_like(target, 1.e38, requires_grad=True)
    loss, info = tail_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert loss.item() == 0 and info['valid_groups'] == 0
    assert new_finite_gradient(loss, logits).count_nonzero() == 0


def test_rank_endpoints_singleton_and_large_integer_counts():
    assert tail_rank_positions(0).numel() == 0
    assert tail_rank_positions(1).tolist() == [0]
    assert tail_rank_positions(9, 1).tolist() == [4]
    assert tail_rank_positions(13, 20).tolist() == list(range(13))
    ranks = tail_rank_positions(2**62, 256)
    assert ranks[0] == 0 and ranks[-1] == 2**62-1
    assert bool((ranks[1:] > ranks[:-1]).all())


@pytest.mark.parametrize('case', ['nan', 'target', 'valid', 'shape', 'fusion', 'margin', 'fraction', 'cap'])
def test_malformed_contracts_are_rejected(case):
    target, valid = geometry()
    logits = torch.zeros_like(target, requires_grad=True)
    kw = dict(fusion_kwargs=OPTIONS)
    if case == 'nan': logits = torch.full_like(logits, float('nan'))
    elif case == 'target': target[0, 0, 0, 0] = .5
    elif case == 'valid': valid = valid.float()
    elif case == 'shape': logits = logits[:, :4]
    elif case == 'fusion': kw['fusion_kwargs'] = dict(OPTIONS, short_reduction='mean')
    elif case == 'margin': kw['margin'] = float('nan')
    elif case == 'fraction': kw['tail_fraction'] = True
    else: kw['max_per_group'] = 257
    with pytest.raises(ValueError):
        tail_boundary_ranking_loss(logits, target, valid, **kw)
