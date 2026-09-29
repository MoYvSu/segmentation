# -*- coding: utf-8 -*-
"""CPU验证：affinity更新不改变冻结语义路径，附加梯度上限确实生效。"""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from train_affinity_connectivity import capped_weight, configure_training, frozen_digest


class TinyBackend(nn.Module):
    """保留共享编码器、任务头及状态缓冲，无需真实权重或GPU。"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout(.4))
        self.encoder.trainable_lora = True
        self.semantic_decoder = nn.Sequential(
            nn.Conv2d(4, 1, 1), nn.BatchNorm2d(1), nn.Dropout(.4))
        self.affinity_decoder = nn.Sequential(nn.Conv2d(4, 2, 1), nn.BatchNorm2d(2))


def test_affinity_optimization_preserves_frozen_semantics_and_buffers():
    torch.manual_seed(81)
    model = TinyBackend().train()
    configure_training(model)
    assert not model.training and not model.encoder.training
    assert not model.semantic_decoder.training and model.affinity_decoder.training
    assert not model.encoder.trainable_lora
    assert {name.split('.')[0] for name, p in model.named_parameters() if p.requires_grad} == {
        'affinity_decoder'}
    image = torch.randn(2, 3, 5, 7)
    frozen_before = frozen_digest(model)
    trainable_before = [p.detach().clone() for p in model.affinity_decoder.parameters()]
    with torch.no_grad():
        raw_semantics = model.semantic_decoder(model.encoder(image)).clone()
    optimizer = torch.optim.AdamW(model.affinity_decoder.parameters(), lr=.03)

    for _ in range(3):
        configure_training(model)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            features = model.encoder(image)
        logits = model.affinity_decoder(features)
        base = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        auxiliary = 20 * F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        weight, _ = capped_weight(base, auxiliary, logits, requested=.1, ratio=.25)
        (base + weight * auxiliary).backward()
        assert all(p.grad is None for n, p in model.named_parameters()
                   if not n.startswith('affinity_decoder.'))
        optimizer.step()

    with torch.no_grad():
        raw_after = model.semantic_decoder(model.encoder(image))
    assert torch.equal(raw_after, raw_semantics)
    assert frozen_digest(model) == frozen_before
    assert any(not torch.equal(before, after) for before, after in
               zip(trainable_before, model.affinity_decoder.parameters()))


def test_frozen_digest_includes_frozen_buffers_but_excludes_affinity_state():
    model = TinyBackend()
    before = frozen_digest(model)
    with torch.no_grad():
        model.affinity_decoder[0].weight.add_(1)
        model.affinity_decoder[1].running_mean.add_(1)
    assert frozen_digest(model) == before
    with torch.no_grad():
        model.encoder[1].running_mean.add_(1)
    assert frozen_digest(model) != before


def test_shared_trainable_parameter_cannot_silently_modify_encoder():
    model = TinyBackend()
    model.affinity_decoder.register_parameter('shared_encoder_weight', model.encoder[0].weight)
    with pytest.raises(RuntimeError, match='Only affinity_decoder'):
        configure_training(model)


@pytest.mark.parametrize('auxiliary_scale, expected_weight, expected_total_ratio', [
    (100., .0025, 1.25),
    (.1, .1, 1.01),
])
def test_cap_limits_added_logit_gradient_and_keeps_backward_graph(
        auxiliary_scale, expected_weight, expected_total_ratio):
    logits = torch.tensor([-1., .2, 2.], requires_grad=True)
    base = F.softplus(logits).mean()
    auxiliary = auxiliary_scale * base
    base_gradient = torch.autograd.grad(base, logits, retain_graph=True)[0]
    weight, details = capped_weight(base, auxiliary, logits, requested=.1, ratio=.25)
    assert weight == pytest.approx(expected_weight)
    (base + weight * auxiliary).backward()
    added_gradient = logits.grad - base_gradient
    assert float(added_gradient.norm()) <= float(base_gradient.norm()) * .250001
    torch.testing.assert_close(logits.grad, expected_total_ratio * base_gradient)
    assert details['weighted_logit_grad_ratio'] <= .250001


def test_empty_auxiliary_and_zero_base_are_finite_and_backward_safe():
    for empty_auxiliary in (True, False):
        logits = torch.tensor([-1., 1.], requires_grad=True)
        nonzero = F.softplus(logits).mean()
        zero = logits.sum() * 0
        base, auxiliary = (nonzero, zero) if empty_auxiliary else (zero, nonzero)
        weight, details = capped_weight(base, auxiliary, logits, requested=.1, ratio=.25)
        (base + weight * auxiliary).backward()
        assert torch.isfinite(logits.grad).all()
        assert details['weighted_logit_grad_ratio'] == 0
        if empty_auxiliary:
            assert torch.count_nonzero(logits.grad) > 0
        else:
            assert weight == 0 and torch.count_nonzero(logits.grad) == 0


def test_zero_weight_control_does_not_require_auxiliary_gradient_graph():
    detached = torch.tensor(1.)
    weight, details = capped_weight(detached, detached, detached, requested=0., ratio=.25)
    assert weight == 0
    assert details['base_logit_grad'] is None and details['aux_logit_grad'] is None
