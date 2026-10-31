# -*- coding: utf-8 -*-
"""三点共同法向覆盖的反例、未知域安全、旧目标兼容与优化器分批合同。"""
from copy import deepcopy
import json
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.boundary_naf import BoundaryDiscriminator
from train_boundary_rgb import reconstruction_metrics, smoke_checks, validate_config
from utils.boundary_rgb_training import (
    build_tolerant_targets, coherent3_coverage, directional_coverage_logits,
    rgb_gan_update, tolerant_boundary_loss,
)
from utils.config import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def line(size=64, kind='vertical'):
    target = torch.zeros(1, 1, size, size)
    y = torch.arange(8, size - 8)
    if kind == 'vertical':
        x = torch.full_like(y, size // 2)
        normal = (0, 1)
    elif kind == 'horizontal':
        x, y = y, torch.full_like(y, size // 2)
        normal = (1, 0)
    elif kind == 'diagonal_up':
        x = size - 1 - y
        normal = (1, 1)
    else:
        x = y
        normal = (-1, 1)
    target[0, 0, y, x] = 1
    return target, torch.ones_like(target, dtype=torch.bool), (y, x), normal


def output_line(target, points, normal, displacement=0):
    logits = torch.full_like(target, -8.)
    y, x = points
    logits[0, 0, y + displacement * normal[0], x + displacement * normal[1]] = 8
    return logits


@pytest.mark.parametrize('kind', ['vertical', 'horizontal', 'diagonal_up', 'diagonal_down'])
def test_continuous_line_and_all_allowed_shifts_have_same_low_loss(kind):
    target, valid, points, normal = line(kind=kind)
    values = []
    for displacement in (0, -1, 1, -2, 2):
        logits = output_line(target, points, normal, displacement)
        loss, audit = tolerant_boundary_loss(logits, target, valid, coverage_mode='coherent3')
        assert audit['coherent3_eligible_gt_points'] > 40
        assert audit['coherent3_covered_gt_fraction'] == 1.
        assert float(audit['coverage']) < .001
        values.append(float(loss))
    assert max(values) - min(values) < 1e-7


def test_zigzag_48_logits_do_not_replace_a_continuous_shared_offset_and_receive_fill_gradient():
    target, valid, (y, x), normal = line()
    logits = torch.full_like(target, -8.)
    offsets = torch.where(y.remainder(2) == 0, -2, 2)
    logits[0, 0, y, x + offsets] = 48
    logits.requires_grad_()
    pointwise, old = tolerant_boundary_loss(logits, target, valid, interior_weight=0)
    coherent, audit = tolerant_boundary_loss(logits, target, valid, interior_weight=0, coverage_mode='coherent3')
    assert float(pointwise.detach()) < 1e-12
    assert float(coherent.detach()) > 2.4
    assert old['covered_gt_points'] == audit['positive_points']
    assert audit['coherent3_covered_gt_points'] == 0
    assert audit['pointwise_covered_gt_points'] == audit['positive_points']
    coherent.backward()
    # row32在x30有强点，邻行共用x34；目标对row32,x34的空点产生补线梯度。
    assert logits.grad[0, 0, 32, 34] < 0
    assert torch.isfinite(logits.grad).all()


def test_very_large_center_logit_cannot_pay_for_missing_neighbors():
    target, valid, _, _ = line()
    logits = torch.full_like(target, -8.)
    logits[0, 0, 32, 32] = 48
    logits.requires_grad_()
    result = coherent3_coverage(logits, build_tolerant_targets(target, valid))
    cost = result['losses'][0, 0, 32, 32]
    assert float(cost.detach()) > 5.3
    assert float(F.softplus(-torch.tensor((-8 + 48 - 8) / 3))) < .001
    cost.backward()
    assert logits.grad[0, 0, 31, 32] < 0 and logits.grad[0, 0, 33, 32] < 0


def test_unknown_targets_and_logits_cannot_change_loss_or_known_gradient_including_diagonal_extremes():
    target, valid, _, _ = line(kind='diagonal_up')
    # 组合位移q+t+2n在单一q的法向截面之外，仍必须被三点各自的trusted排除。
    valid[..., 30, 33] = False
    first_target = target.clone()
    second_target = target.clone()
    first_target[~valid] = float('nan')
    second_target[~valid] = -900
    rng = torch.Generator().manual_seed(58)
    first = torch.randn(target.shape, generator=rng).requires_grad_()
    second = first.detach().clone()
    second[~valid] = 800
    second.requires_grad_()
    loss_a, stats_a = tolerant_boundary_loss(first, first_target, valid, coverage_mode='coherent3')
    loss_b, stats_b = tolerant_boundary_loss(second, second_target, valid, coverage_mode='coherent3')
    loss_a.backward()
    loss_b.backward()
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(first.grad, second.grad)
    assert torch.count_nonzero(first.grad[~valid]) == 0
    prepared = build_tolerant_targets(first_target, valid)
    common = coherent3_coverage(first.detach(), prepared)
    eligible = common['eligible']
    assert eligible.any() and stats_a['coherent3_eligible_gt_points'] == stats_b['coherent3_eligible_gt_points']
    # 独立枚举所有实际采样组合，证明无一触及unknown或图外。
    directions = ((0, 1, 1, 0), (1, 1, 1, -1), (1, 0, 0, 1), (-1, 1, 1, 1))
    for b, c, y, x in eligible.nonzero().tolist():
        ny, nx, ty, tx = directions[int(prepared['normal_index'][b, c, y, x])]
        for step in (-1, 0, 1):
            for delta in range(-2, 3):
                yy, xx = y + step * ty + delta * ny, x + step * tx + delta * nx
                assert 0 <= yy < target.shape[-2] and 0 <= xx < target.shape[-1]
                assert valid[b, c, yy, xx]


def test_curve_and_fallback_keep_all_positives_without_inventing_neighbors():
    target = torch.zeros(1, 1, 64, 64)
    target[..., 32, 8:49] = 1
    target[..., 12:33, 48] = 1
    target[..., 48, 16] = 1  # 单点明确fallback，仍须计入positive。
    valid = torch.ones_like(target, dtype=torch.bool)
    prepared = build_tolerant_targets(target, valid)
    logits = torch.zeros_like(target, requires_grad=True)
    result = coherent3_coverage(logits, prepared)
    assert result['eligible'][0, 0, 32, 20]
    assert prepared['fallback'][0, 0, 48, 16]
    assert not result['eligible'][0, 0, 48, 16]
    # 量化方向转折的中心没有同向±t时不纳入三点；真实GT不因此消失。
    support = prepared['positive'] & ~prepared['fallback']
    directions = ((0, 1, 1, 0), (1, 1, 1, -1), (1, 0, 0, 1), (-1, 1, 1, 1))
    for b, c, y, x in prepared['positive'].nonzero().tolist():
        index = int(prepared['normal_index'][b, c, y, x])
        _, _, ty, tx = directions[index]
        expected = bool(support[b, c, y, x])
        for step in (-1, 1):
            yy, xx = y + step * ty, x + step * tx
            expected = expected and bool(support[b, c, yy, xx]) and int(prepared['normal_index'][b, c, yy, xx]) == index
        assert bool(result['eligible'][b, c, y, x]) == expected
    loss, audit = tolerant_boundary_loss(logits, target, valid, interior_weight=0, coverage_mode='coherent3')
    assert audit['positive_points'] == int(prepared['positive'].sum()) == int(target.sum())
    assert float(loss.detach()) == pytest.approx(math.log(2), abs=1e-7)
    assert audit['coherent3_eligible_gt_points'] < audit['positive_points']


def test_diagonal_neighbor_cannot_sample_unknown_beyond_center_trusted_radius():
    target, valid, _, _ = line(kind='diagonal_up')
    valid[0, 0, 33, 34] = False  # q=(30,33), t=(1,-1), d=2 => q+t+2n。
    prepared = build_tolerant_targets(target, valid, tolerance=2, orientation_radius=1)
    assert prepared['positive'][0, 0, 30, 33]  # unknown位于q的radius2之外。
    assert not prepared['positive'][0, 0, 31, 32]  # 但位于邻点的radius2之内。
    first = torch.full_like(target, -3., requires_grad=True)
    second = first.detach().clone()
    second[~valid] = 900
    second.requires_grad_()
    common = coherent3_coverage(first, prepared)
    assert not common['eligible'][0, 0, 30, 33]
    a, _ = tolerant_boundary_loss(first, target, valid, prepared=prepared, coverage_mode='coherent3')
    b, _ = tolerant_boundary_loss(second, target, valid, prepared=prepared, coverage_mode='coherent3')
    a.backward()
    b.backward()
    assert torch.equal(a, b) and torch.equal(first.grad, second.grad)
    assert first.grad[~valid].count_nonzero() == 0


def test_far_negative_and_gt_counts_unchanged_and_fallback_retains_old_cover():
    target, valid, _, _ = line()
    target[0, 0, 48, 16] = 1
    rng = torch.Generator().manual_seed(84)
    logits = torch.randn(target.shape, generator=rng)
    _, old = tolerant_boundary_loss(logits, target, valid)
    _, coherent = tolerant_boundary_loss(logits, target, valid, coverage_mode='coherent3')
    assert torch.equal(old['interior'], coherent['interior'])
    for key in ('positive_points', 'negative_pixels', 'trusted_pixels', 'valid_pixels',
                'invalid_pixels', 'invalid_gt_points', 'fallback_gt_points'):
        assert old[key] == coherent[key]
    isolated = torch.zeros_like(target)
    isolated[0, 0, 32, 32] = 1
    first = logits.clone().requires_grad_()
    second = logits.clone().requires_grad_()
    a, _ = tolerant_boundary_loss(first, isolated, valid)
    b, stats = tolerant_boundary_loss(second, isolated, valid, coverage_mode='coherent3')
    a.backward()
    b.backward()
    assert stats['coherent3_eligible_gt_points'] == 0
    assert torch.equal(a, b) and torch.equal(first.grad, second.grad)


def legacy_reference(logits, target, valid):
    prepared = build_tolerant_targets(target, valid)
    positive, negative = prepared['positive'], prepared['negative']
    peaks = directional_coverage_logits(logits, prepared['normal_index'], prepared['valid'], 2, prepared['fallback'])
    coverage = F.softplus(-peaks[positive]).sum() / int(positive.sum())
    interior = F.softplus(logits[negative]).sum() / int(negative.sum())
    probabilities = peaks[positive].detach().sigmoid()
    stats = dict(coverage=coverage, interior=interior, positive_points=int(positive.sum()),
        negative_pixels=int(negative.sum()), trusted_pixels=int(prepared['trusted'].sum()),
        valid_pixels=int(prepared['valid'].sum()), invalid_pixels=int((~prepared['trusted']).sum()),
        invalid_gt_points=int(((prepared['safe_target'] > .5) & ~prepared['trusted']).sum()),
        fallback_gt_points=int(prepared['fallback'].sum()), covered_gt_points=int((probabilities > .65).sum()),
        coverage_probability_sum=float(probabilities.sum()), exact_pixel_BCE=False, sparse_cut_selection=False)
    return coverage + interior, stats


def test_default_and_explicit_pointwise_are_bitwise_legacy_loss_gradient_and_stats():
    target, valid, _, _ = line()
    valid[..., 20:26, 30:36] = False
    target[~valid] = float('nan')
    rng = torch.Generator().manual_seed(71)
    reference = torch.randn(target.shape, generator=rng).requires_grad_()
    implicit = reference.detach().clone().requires_grad_()
    explicit = reference.detach().clone().requires_grad_()
    expected, original = legacy_reference(reference, target, valid)
    loss_a, a = tolerant_boundary_loss(implicit, target, valid)
    loss_b, b = tolerant_boundary_loss(explicit, target, valid, coverage_mode='pointwise')
    for value in (expected, loss_a, loss_b):
        value.backward()
    assert torch.equal(expected, loss_a) and torch.equal(expected, loss_b)
    assert torch.equal(reference.grad, implicit.grad) and torch.equal(reference.grad, explicit.grad)
    assert set(original) == set(a) == set(b)
    for key in original:
        if torch.is_tensor(original[key]):
            assert torch.equal(original[key], a[key]) and torch.equal(original[key], b[key])
        else:
            assert original[key] == a[key] == b[key]


def test_global_denominator_preserves_full_vs_microbatch_loss_and_gradient():
    target, valid, _, _ = line()
    target, valid = target.repeat(3, 1, 1, 1), valid.repeat(3, 1, 1, 1)
    valid[1, :, 20:30, 30:34] = False
    rng = torch.Generator().manual_seed(22)
    full = torch.randn(target.shape, generator=rng, dtype=torch.float64).requires_grad_()
    micro = full.detach().clone().requires_grad_()
    prepared = build_tolerant_targets(target, valid)
    denominators = int(prepared['positive'].sum()), int(prepared['negative'].sum())
    full_loss, _ = tolerant_boundary_loss(full, target, valid, coverage_mode='coherent3')
    values = []
    for index in range(3):
        sliced = {k: v[index:index + 1] if torch.is_tensor(v) else v for k, v in prepared.items()}
        loss, _ = tolerant_boundary_loss(micro[index:index + 1], target[index:index + 1], valid[index:index + 1],
            prepared=sliced, denominators=denominators, coverage_mode='coherent3')
        values.append(loss)
    micro_loss = sum(values)
    full_loss.backward()
    micro_loss.backward()
    torch.testing.assert_close(full_loss, micro_loss, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(full.grad, micro.grad, rtol=1e-13, atol=1e-13)


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(62)
            self.layer = nn.Conv2d(1, 1, 1)

    def forward_logits(self, inputs):
        return self.layer(inputs)

    def forward(self, inputs):
        return self.forward_logits(inputs).sigmoid()


def test_coherent_gan_microbatch_uses_same_denominators_and_keeps_old_metrics_explicit():
    target, valid, _, _ = line(96)
    target, valid = target.repeat(3, 1, 1, 1), valid.repeat(3, 1, 1, 1)
    valid[1, :, 18:26, 46:50] = False
    inputs = target.clone()
    inputs[..., 30:40, :] = 0
    batch = dict(input=inputs, target=target, valid=valid)
    ga, gb = TinyGenerator(), TinyGenerator()
    da, db = (BoundaryDiscriminator(width=2, seed=17) for _ in range(2))
    results = []
    for g, d, size in ((ga, da, 3), (gb, db, 1)):
        go, do = torch.optim.SGD(g.parameters(), lr=.001), torch.optim.SGD(d.parameters(), lr=.001)
        results.append(rgb_gan_update(g, d, go, do, batch, micro_batch_size=size,
                                      grad_clip=None, coverage_mode='coherent3'))
    a, b = results
    assert a['generator_updated'] and a['discriminator_updated']
    assert a['coherent3_eligible_gt_points'] > 0
    for key in ('loss_g', 'loss_d', 'coverage', 'pointwise_coverage', 'interior', 'adv_g',
                'grad_norm_g', 'grad_norm_d', 'coherent3_eligible_gt_points',
                'coherent3_eligible_gt_fraction', 'coherent3_covered_gt_fraction',
                'coherent3_coverage_probability_mean'):
        assert a[key] == pytest.approx(b[key], rel=5e-5, abs=1e-6)
    for first, second in zip(ga.parameters(), gb.parameters()):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-7)
    for first, second in zip(da.parameters(), db.parameters()):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-7)
    assert a['pointwise_covered_gt_points'] == a['covered_gt_points']
    json.dumps(a, allow_nan=False)


def test_reconstruction_and_smoke_report_pointwise_and_shared_objectives_separately():
    cfg = load_config('config/train/boundary_coherent.yaml')['boundary_naf']
    target, valid, (y, x), _ = line()
    logits = torch.full_like(target, -8.)
    logits[0, 0, y, x + torch.where(y.remainder(2) == 0, -2, 2)] = 48
    # 整段输入缺失，逐点亮点虽然覆盖GT，但不满足共同偏移。
    batch = dict(target=target, valid=valid, input=torch.zeros_like(target))
    audit = reconstruction_metrics(logits, batch, cfg)
    assert audit['missing_probability'] > .99
    assert audit['coherent3_missing_probability'] < .2
    assert audit['missing_above065'] == 1
    # 两个端点没有±t三点，按要求仍保留原逐点覆盖，不能悄悄丢弃。
    assert audit['coherent3_missing_above065'] == pytest.approx(2 / audit['missing_points'])
    # 即便原pointwise字段看似改善，新实验smoke仍由共同覆盖指标判断。
    before = dict(loss=2., missing_points=10, supported_points=10,
        recorded_interior_input_points=0, missing_probability=.7, supported_probability=.9,
        recorded_interior_input_probability=None, coverage_mode='coherent3',
        coherent3_missing_probability=.6, coherent3_supported_probability=.9)
    after = dict(before, loss=.5, missing_probability=1., coherent3_missing_probability=.4)
    assert not smoke_checks(before, after, True, 20)['missing_progress']


def test_recipe_changes_only_optional_coverage_mode_and_output_directory():
    baseline = load_config('config/train/boundary_width.yaml')
    candidate = load_config('config/train/boundary_coherent.yaml')
    assert validate_config(candidate)['rgb_loss']['coverage_mode'] == 'coherent3'
    candidate['boundary_naf']['output_dir'] = baseline['boundary_naf']['output_dir']
    candidate['boundary_naf']['rgb_loss'].pop('coverage_mode')
    assert candidate == baseline
    candidate['boundary_naf']['rgb_loss']['coverage_mode'] = 'pointwise'
    validate_config(candidate)
    candidate['boundary_naf']['rgb_loss']['coverage_mode'] = 'unknown'
    with pytest.raises(ValueError, match='coverage_mode'):
        validate_config(candidate)
