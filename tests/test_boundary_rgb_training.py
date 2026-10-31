# -*- coding: utf-8 -*-
"""RGB补边的容差覆盖、连续性、未知域隔离与真实优化器合同。"""
from copy import deepcopy
import json

import pytest
import torch
from torch import nn

from models.boundary_naf import BoundaryDiscriminator
from utils.boundary_rgb_training import (
    build_tolerant_targets, directional_coverage_logits, rgb_gan_update,
    tolerant_boundary_loss,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def straight_line(size=64):
    target = torch.zeros(1, 1, size, size)
    target[..., 8:size - 8, size // 2] = 1
    return target, torch.ones_like(target, dtype=torch.bool)


def line_logits(target, offset=0):
    logits = torch.full_like(target, -8.)
    size = target.shape[-1]
    logits[..., 8:size - 8, size // 2 + offset] = 8
    return logits


class TinyRGB(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(81)
            self.conv = nn.Conv2d(channels, 1, 1)

    def forward_logits(self, value):
        return self.conv(value)

    def forward(self, value):
        return self.forward_logits(value).sigmoid()


def state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def identical(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def training_batch(count=3, channels=4):
    target, valid = straight_line(96)
    target, valid = target.repeat(count, 1, 1, 1), valid.repeat(count, 1, 1, 1)
    rng = torch.Generator().manual_seed(53)
    inputs = torch.rand(count, channels, 96, 96, generator=rng)
    inputs[:, -1:] = target
    inputs[:, -1:, 40:50, 48] = 0
    if count > 1:
        valid[1, :, 10:20, 10:20] = False
    target[~valid] = float("nan")
    return {"input": inputs, "target": target, "valid": valid, "metadata": "ignored"}


def models(channels=4):
    return TinyRGB(channels), BoundaryDiscriminator(width=2, seed=7)


def test_two_pixel_shift_is_allowed_without_exact_gt_match_but_three_is_not():
    target, valid = straight_line()
    baseline, a = tolerant_boundary_loss(line_logits(target), target, valid)
    shifted, b = tolerant_boundary_loss(line_logits(target, 2), target, valid)
    outside, c = tolerant_boundary_loss(line_logits(target, 3), target, valid)
    assert float(shifted) == pytest.approx(float(baseline), abs=1e-7)
    assert float(outside) > float(shifted) + 7
    assert float(a["coverage"]) < .001 and float(b["coverage"]) < .001
    assert float(c["coverage"]) > 7
    assert not a["exact_pixel_BCE"] and not a["sparse_cut_selection"]


def test_transverse_coverage_rejects_dots_that_two_dimensional_pool_would_accept():
    target, valid = straight_line()
    points = torch.full_like(target, -8)
    points[..., 8:56:4, 32] = 8
    full_loss, _ = tolerant_boundary_loss(line_logits(target), target, valid)
    dots_loss, audit = tolerant_boundary_loss(points, target, valid)
    assert float(dots_loss) > float(full_loss) + 4
    assert audit["covered_gt_points"] < audit["positive_points"] / 2
    ordinary = torch.nn.functional.max_pool2d(points.sigmoid(), 5, stride=1, padding=2)
    prepared = build_tolerant_targets(target, valid)
    assert float((ordinary[prepared["positive"]] > .65).float().mean()) > .95


def test_missing_boundary_receives_positive_gradient_and_false_input_line_can_be_deleted():
    target, valid = straight_line()
    logits = line_logits(target)
    logits[..., 28:36, :] = -8
    logits[..., 20:44, 48] = 8  # 原输入假线也不享受负监督豁免。
    logits.requires_grad_()
    loss, audit = tolerant_boundary_loss(logits, target, valid)
    loss.backward()
    assert logits.grad[..., 29:35, 30:35].sum() < 0
    assert logits.grad[..., 20:44, 48].sum() > 0
    assert audit["positive_points"] == 48


def test_unknown_target_sentinels_do_not_affect_masks_loss_or_gradient():
    target, valid = straight_line()
    valid[..., 26:34, 28:36] = False
    target[~valid] = -1
    other = target.clone()
    other[~valid] = float("nan")
    first = line_logits(target, 2).requires_grad_()
    second = first.detach().clone().requires_grad_()
    a, sa = tolerant_boundary_loss(first, target, valid)
    b, sb = tolerant_boundary_loss(second, other, valid)
    a.backward()
    b.backward()
    assert torch.equal(a, b) and torch.equal(first.grad, second.grad)
    assert torch.count_nonzero(first.grad[~valid]) == 0
    assert sa["invalid_gt_points"] > 0
    for key in ("positive_points", "negative_pixels", "fallback_gt_points"):
        assert sa[key] == sb[key]


def test_direction_fallback_does_not_drop_isolated_or_isotropic_gt():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 16, 16] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    prepared = build_tolerant_targets(target, valid)
    assert int(prepared["positive"].sum()) == int(prepared["fallback"].sum()) == 1
    assert not bool((prepared["positive"] & prepared["negative"]).any())
    logits = torch.full_like(target, -8)
    logits[..., 18, 18] = 8  # L_inf为2，即斜向欧氏距离sqrt(8)。
    _, stats = tolerant_boundary_loss(logits, target, valid)
    assert stats["covered_gt_points"] == 1


def test_unknown_output_cannot_supply_positive_cover_for_known_gt():
    target, valid = straight_line()
    valid[..., 28:36, 33] = False
    logits = torch.full_like(target, -8)
    logits[~valid] = 8
    prepared = build_tolerant_targets(target, valid)
    peaks = directional_coverage_logits(logits, prepared["normal_index"], valid, 2,
                                       prepared["fallback"])
    assert not bool((peaks[prepared["positive"]].sigmoid() > .65).any())


def test_microbatch_global_denominators_and_gan_isolation_match_full_batch():
    g1, d1 = models()
    g2, d2 = models()
    go1, do1 = torch.optim.SGD(g1.parameters(), lr=.001), torch.optim.SGD(d1.parameters(), lr=.001)
    go2, do2 = torch.optim.SGD(g2.parameters(), lr=.001), torch.optim.SGD(d2.parameters(), lr=.001)
    batch = training_batch()
    a = rgb_gan_update(g1, d1, go1, do1, batch, micro_batch_size=3, grad_clip=None)
    b = rgb_gan_update(g2, d2, go2, do2, batch, micro_batch_size=1, grad_clip=None)
    assert a["generator_updated"] and a["discriminator_updated"]
    assert a["spectral_power_iterations"] == b["spectral_power_iterations"] == 1
    assert a["adversarial_patches"] > 0
    for key in ("loss_g", "loss_d", "coverage", "interior", "adv_g", "grad_norm_g", "grad_norm_d"):
        assert a[key] == pytest.approx(b[key], rel=5e-5, abs=1e-6)
    for first, second in zip(g1.parameters(), g2.parameters()):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-7)
    for first, second in zip(d1.parameters(), d2.parameters()):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-7)
    assert a["missing_gt_points"] > 0
    assert 0 <= a["missing_covered_gt_fraction"] <= 1
    json.dumps(a, allow_nan=False)


def test_all_unknown_skips_adam_momentum_and_spectral_buffers():
    generator, discriminator = models()
    go, do = torch.optim.Adam(generator.parameters()), torch.optim.Adam(discriminator.parameters())
    batch = training_batch(1)
    batch["valid"].zero_()
    batch["target"].fill_(float("nan"))
    before_g, before_d = state(generator), state(discriminator)
    stats = rgb_gan_update(generator, discriminator, go, do, batch)
    assert not stats["generator_updated"] and not stats["discriminator_updated"]
    assert stats["adversarial_patches"] == stats["positive_points"] == stats["negative_pixels"] == 0
    assert identical(before_g, state(generator)) and identical(before_d, state(discriminator))
    assert not go.state and not do.state


@pytest.mark.parametrize("channels", [1, 4])
def test_warmup_updates_only_generator_and_restores_modes_and_flags(channels):
    generator, discriminator = models(channels)
    generator.eval()
    discriminator.train()
    go, do = torch.optim.Adam(generator.parameters()), torch.optim.Adam(discriminator.parameters())
    before_g, before_d = state(generator), state(discriminator)
    flags = [value.requires_grad for value in discriminator.parameters()]
    stats = rgb_gan_update(generator, discriminator, go, do,
                           training_batch(1, channels), adversarial_weight=0)
    assert stats["generator_updated"] and not stats["discriminator_updated"]
    assert not identical(before_g, state(generator)) and identical(before_d, state(discriminator))
    assert not generator.training and discriminator.training
    assert [value.requires_grad for value in discriminator.parameters()] == flags


def test_unknown_values_do_not_change_actual_gan_update():
    g1, d1 = models()
    g2, d2 = models()
    go1, do1 = torch.optim.Adam(g1.parameters()), torch.optim.Adam(d1.parameters())
    go2, do2 = torch.optim.Adam(g2.parameters()), torch.optim.Adam(d2.parameters())
    batch = training_batch(2)
    other = deepcopy(batch)
    other["target"][~other["valid"]] = -1
    a = rgb_gan_update(g1, d1, go1, do1, batch)
    b = rgb_gan_update(g2, d2, go2, do2, other)
    assert a == b and identical(state(g1), state(g2)) and identical(state(d1), state(d2))


def test_parameter_and_input_contracts_fail_explicitly():
    target, valid = straight_line()
    with pytest.raises(ValueError, match="nonnegative integer"):
        tolerant_boundary_loss(line_logits(target), target, valid, tolerance=-1)
    with pytest.raises(ValueError, match="binary"):
        tolerant_boundary_loss(line_logits(target), target + .1, valid)
    g, d = models()
    go, do = torch.optim.Adam(g.parameters()), torch.optim.Adam(d.parameters())
    data = training_batch(1)
    data["input"][..., 0, 0] = 2
    with pytest.raises(ValueError, match="Input"):
        rgb_gan_update(g, d, go, do, data)
