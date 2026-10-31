# -*- coding: utf-8 -*-
"""融合排序附加项的理想关系、方向、未知支持域与浮点安全检查。"""
from copy import deepcopy
import json

import pytest
import torch

from utils.affinity_fused_loss import fused_boundary_ranking_loss, fused_ranking_domains
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_loss import build_affinity_targets_torch

OPTIONS = dict(mode='gated', short_reduction='top2', distance2_weight=.5, distance4_weight=.25,
               support_threshold=.2, support_temperature=.05)


def geometry(kind='straight', size=48):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    ids = torch.ones((size, size), dtype=torch.int64)
    if kind == 'straight':
        ids[:, size // 2:] = 2
    elif kind == 'diagonal':
        ids[xx > yy] = 2
    elif kind == 'tee':
        ids[:, size // 2:] = 2
        ids[size // 2:, size // 2:] = 3
    elif kind in ('loop', 'unknown_hole'):
        ids[14:34, 14:34] = 2
        if kind == 'unknown_hole':
            ids[21:27, 21:27] = 0
    else:
        raise ValueError(kind)
    target, valid = build_affinity_targets_torch(ids[None], torch.ones((1, 1, size, size), dtype=torch.bool))
    return target, valid


@pytest.mark.parametrize('kind', ['straight', 'diagonal', 'tee', 'loop', 'unknown_hole'])
def test_ideal_new_gt_relations_have_zero_ranking_penalty_and_zero_gradient(kind):
    target, valid = geometry(kind)
    logits = (target * 32 - 16).requires_grad_()
    before = [v.clone() for v in (logits, target, valid)]
    loss, details = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    gradient, = torch.autograd.grad(loss, logits)
    assert details['valid_groups'] == 1 and details['total_pairs'] > 0
    assert loss.item() == 0 and details['active_pairs'] == 0
    assert torch.count_nonzero(gradient) == 0
    assert all(torch.equal(a, b) for a, b in zip((logits, target, valid), before))
    json.dumps(details, allow_nan=False)


def disturbed():
    target, valid = geometry()
    logits = target * 32 - 16
    # 一个真实分界锚点所有关系输出连接；深内部错误输出强分界。
    logits[:, :, 20, 23] = 2.
    logits[:, :, 20, 9] = -2.
    return logits.requires_grad_(), target, valid


def test_false_interior_peak_goes_down_and_weak_true_boundary_goes_up():
    logits, target, valid = disturbed()
    loss, details = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    gradient, = torch.autograd.grad(loss, logits)
    assert loss > 0 and details['active_pairs'] > 0
    before = affinity_boundary_probability(logits.detach(), **OPTIONS)
    after = affinity_boundary_probability(logits.detach() - gradient, **OPTIONS)
    assert after[0, 0, 20, 23] > before[0, 0, 20, 23]
    assert after[0, 0, 20, 9] < before[0, 0, 20, 9]
    # 中央长程及非中央支持池的短程都须保留真实梯度，不能detach gate。
    assert gradient[0, 4:, 20, 9].abs().sum() > 0
    elsewhere = gradient[:, :4].clone()
    elsewhere[:, :, 20, 9] = 0
    elsewhere[:, :, 20, 23] = 0
    assert elsewhere.abs().sum() > 0


def test_unknown_relations_and_pool_padding_never_receive_auxiliary_gradient():
    logits, target, valid = disturbed()
    valid[:, :4, 12, 13] = False
    valid[:, 6, 20, 11] = False  # 即使四短支持池完整，中央未知长程也不能受力。
    boundary, interior = fused_ranking_domains(target, valid)
    assert not boundary[:, :4].any() and not interior[:, :4].any()
    assert not interior[0, 8:17, 9:18].any()
    assert not interior[0, 20, 11]
    loss, _ = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    gradient, = torch.autograd.grad(loss, logits)
    assert torch.count_nonzero(gradient[~valid]) == 0
    assert torch.count_nonzero(gradient[:, :4, 12, 13]) == 0


def test_gated_support_gradient_reaches_unselected_trusted_neighbour():
    target = torch.ones((1, 8, 24, 24))
    valid = torch.ones_like(target, dtype=torch.bool)
    target[:, 0, 10, 10] = 0  # 唯一真分界。
    target[:, 6, 10, 11] = 0  # 不是真分界，也不是深内部：不能被两尾直接选择。
    boundary_scores = torch.full_like(target, .01)
    boundary_scores[:, :4, 10, 10] = .05
    boundary_scores[:, 4:, 10, 10] = .8
    boundary_scores[:, :4, 10, 11] = .3  # 仅通过该可信邻居的support gate影响弱分界。
    boundary_scores[:, :, 10, 17] = .9
    logits = torch.logit(1 - boundary_scores).requires_grad_()
    boundary, interior = fused_ranking_domains(target, valid)
    assert not boundary[0, 10, 11] and not interior[0, 10, 11]
    loss, details = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    gradient, = torch.autograd.grad(loss, logits)
    assert details['active_pairs'] > 0
    assert gradient[0, :4, 10, 11].abs().sum() > 1.e-5


def test_per_sample_normalization_does_not_drop_valid_zero_loss_group():
    logits, target, valid = disturbed()
    one, _ = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    ideal = target * 32 - 16
    mixed, details = fused_boundary_ranking_loss(torch.cat([logits, ideal]),
        torch.cat([target, target]), torch.cat([valid, valid]), fusion_kwargs=OPTIONS)
    assert details['valid_groups'] == 2 and details['groups'][1]['loss'] == 0
    assert mixed.item() == pytest.approx(one.item() / 2)


def test_two_tails_use_ceil_cap_and_all_pair_mean_not_only_active_pairs():
    logits, target, valid = disturbed()
    loss, details = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    boundary, interior = fused_ranking_domains(target, valid)
    scores = affinity_boundary_probability(logits, **OPTIONS)[0, 0]
    positive, negative = scores[boundary[0]], scores[interior[0]]
    npick, nnick = min(256, (positive.numel()+9)//10), min(256, (negative.numel()+9)//10)
    p = positive.detach().topk(npick, largest=False).indices
    n = negative.detach().topk(nnick, largest=True).indices
    expected = torch.relu(.1 + negative[n][None] - positive[p][:, None]).mean()
    assert torch.equal(loss, expected)
    assert details['selected_true_boundary'] == npick
    assert details['selected_deep_interior'] == nnick
    assert details['total_pairs'] == npick * nnick > details['active_pairs'] > 0
    big = torch.ones((1, 8, 180, 180))
    big[:, 0, :, ::2] = 0
    _, large = fused_boundary_ranking_loss(torch.zeros_like(big, requires_grad=True), big,
                                         torch.ones_like(big, dtype=torch.bool), fusion_kwargs=OPTIONS)
    assert large['selected_true_boundary'] == large['selected_deep_interior'] == 256
    assert large['total_pairs'] == 65536


@pytest.mark.parametrize('which', ['no_valid', 'no_boundary', 'no_interior', 'tiny_grid'])
def test_missing_supervision_returns_differentiable_exact_zero(which):
    logits, target, valid = disturbed()
    if which == 'no_valid':
        valid.zero_()
    elif which == 'no_boundary':
        target.fill_(1)
    elif which == 'no_interior':
        target.zero_()
    else:
        logits, target, valid = logits[:, :, :5, :5], target[:, :, :5, :5], valid[:, :, :5, :5]
    loss, details = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    gradient, = torch.autograd.grad(loss, logits)
    assert loss.item() == 0 and loss.requires_grad and details['valid_groups'] == 0
    assert gradient.count_nonzero() == 0


def test_loss_preserves_parent_mode_buffers_and_parameter_values():
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 8, 1), torch.nn.GroupNorm(2, 8)).train()
    model.register_buffer('sentinel', torch.tensor([123.]))
    logits = model(torch.randn(1, 2, 48, 48))
    target, valid = geometry()
    before, modes = deepcopy(model.state_dict()), [m.training for m in model.modules()]
    loss, details = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert [m.training for m in model.modules()] == modes
    assert all(p.grad is None for p in model.parameters())
    assert details['valid_groups'] == 1


def test_autocast_and_bfloat16_inputs_are_computed_in_float32():
    logits, target, valid = disturbed()
    logits = logits.detach().to(torch.bfloat16).requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, _ = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    gradient, = torch.autograd.grad(loss, logits)
    assert gradient.dtype == torch.bfloat16 and torch.isfinite(gradient).all()


def test_empty_mask_zero_remains_finite_even_when_finite_logits_sum_would_overflow():
    target, valid = geometry()
    valid.zero_()
    logits = torch.full_like(target, 1.e38, requires_grad=True)
    loss, _ = fused_boundary_ranking_loss(logits, target, valid, fusion_kwargs=OPTIONS)
    gradient, = torch.autograd.grad(loss, logits)
    assert loss.item() == 0 and torch.isfinite(loss)
    assert gradient.count_nonzero() == 0


@pytest.mark.parametrize('mutation', ['logits_nan', 'target_nonbinary', 'valid_not_bool', 'bad_shape',
                                     'bad_fusion', 'margin_nan', 'fraction_bool', 'cap_float'])
def test_malformed_inputs_and_options_fail_before_training(mutation):
    logits, target, valid = disturbed()
    options = dict(fusion_kwargs=OPTIONS)
    if mutation == 'logits_nan':
        with torch.no_grad(): logits[0, 0, 0, 0] = float('nan')
    elif mutation == 'target_nonbinary':
        target[0, 0, 0, 0] = .5
    elif mutation == 'valid_not_bool':
        valid = valid.float()
    elif mutation == 'bad_shape':
        logits = logits[:, :4]
    elif mutation == 'bad_fusion':
        options['fusion_kwargs'] = dict(OPTIONS, short_reduction='mean')
    elif mutation == 'margin_nan':
        options['margin'] = float('nan')
    elif mutation == 'fraction_bool':
        options['tail_fraction'] = True
    else:
        options['max_per_group'] = 256.
    with pytest.raises(ValueError):
        fused_boundary_ranking_loss(logits, target, valid, **options)
