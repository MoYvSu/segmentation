# -*- coding: utf-8 -*-
"""NAF补边器与GAN隔离的CPU契约；合成检查不代表竞赛收益。"""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from models.boundary_naf import BoundaryDiscriminator, BoundaryNAFNet


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_generator(seed=9):
    return BoundaryNAFNet(width=4, encoder_blocks=(1, 1), middle_blocks=1,
                          decoder_blocks=(1, 1), seed=seed)


def _state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _same(first, second):
    return set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)


@pytest.mark.parametrize("shape", [(2, 1, 32, 32), (1, 1, 31, 47), (1, 1, 1, 3)])
def test_shape_probability_logits_and_residual(shape):
    generator = small_generator()
    value = torch.rand(shape)
    logits = generator.forward_logits(value)
    probability = generator(value)
    assert logits.shape == value.shape == probability.shape
    assert torch.equal(probability, logits.sigmoid())
    torch.testing.assert_close(generator.residual(value), probability - value, rtol=0, atol=0)
    assert probability.min() >= 0 and probability.max() <= 1
    assert not torch.equal(probability, value)  # 随机初态不得伪称旧模型恒等。
    F.binary_cross_entropy_with_logits(logits, torch.zeros_like(value)).backward()
    assert generator.intro.weight.grad.abs().sum() > 0
    assert generator.ending.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("value", [torch.zeros(1, 3, 8, 8), torch.ones(1, 1, 8, 8, dtype=torch.long),
                                    torch.full((1, 1, 8, 8), 1.1), torch.full((1, 1, 8, 8), float("nan"))])
def test_input_validation(value):
    with pytest.raises(ValueError):
        small_generator()(value)


@pytest.mark.parametrize("kwargs", [{"width": 1}, {"width": True}, {"encoder_blocks": []},
                                     {"decoder_blocks": [1]}, {"middle_blocks": 0}, {"seed": 1.5}])
def test_invalid_generator_architecture(kwargs):
    with pytest.raises(ValueError):
        BoundaryNAFNet(**kwargs)


def test_seed_is_local_architecture_reload_and_no_hidden_weights():
    before = torch.get_rng_state()
    generator = small_generator(17)
    discriminator = BoundaryDiscriminator(width=4, seed=19)
    assert torch.equal(before, torch.get_rng_state())
    reloaded = BoundaryNAFNet(**generator.architecture)
    assert _same(_state(generator), _state(reloaded))
    assert not _same(_state(generator), _state(small_generator(18)))
    reloaded.load_state_dict(generator.state_dict(), strict=True)
    value = torch.rand(1, 1, 32, 32)
    assert torch.equal(generator(value), reloaded(value))
    cloned_d = BoundaryDiscriminator(**discriminator.architecture)
    assert _same(_state(discriminator), _state(cloned_d))
    damaged = deepcopy(generator.state_dict())
    damaged["unexpected"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        reloaded.load_state_dict(damaged, strict=True)
    assert generator.checkpoint_format == "boundary_naf_v1"


def test_default_generator_parameter_count_matches_reported_rounding():
    # meta只构建架构，不分配约70MB参数载荷。
    with torch.device("meta"):
        generator = BoundaryNAFNet()
    assert sum(value.numel() for value in generator.parameters()) == 17_632_273
    assert generator.parameter_summary()["output_contract"]["initial_identity"] is False


def test_cpu_initialization_does_not_reseed_cuda_streams(monkeypatch):
    # 无需初始化CUDA：拦截CUDA设种子入口即可验证CPU模型构造不触碰GPU随机流。
    calls = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    small_generator(31)
    BoundaryDiscriminator(width=4, seed=31)
    assert calls == []


def test_discriminator_patch_validity_and_unknown_receptive_field():
    discriminator = BoundaryDiscriminator(width=4)
    value = torch.rand(2, 1, 128, 144)
    logits = discriminator(value)
    valid = torch.ones_like(value, dtype=torch.bool)
    mask = discriminator.valid_patch_mask(valid)
    assert mask.shape == logits.shape
    assert mask.any() and not mask.all()  # 即使全图已知，卷积padding也不能监督。
    before = mask.clone()
    valid[0, 0, 64, 72] = False
    mask = discriminator.valid_patch_mask(valid)
    assert torch.count_nonzero(before[0] & ~mask[0]) > 0
    assert torch.equal(before[1], mask[1])
    assert not discriminator.valid_patch_mask(torch.zeros_like(valid)).any()
    assert len(discriminator.convs) == 5
    assert all(hasattr(conv, "weight_orig") for conv in discriminator.convs)
    assert discriminator.receptive_field == 70
    with pytest.raises(ValueError, match="at least"):
        discriminator(torch.rand(1, 1, 16, 32))


def test_detach_discriminator_step_and_frozen_eval_generator_step():
    generator = small_generator()
    discriminator = BoundaryDiscriminator(width=4)
    input_map = torch.rand(1, 1, 96, 96)
    target = (torch.rand_like(input_map) > 0.7).float()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    initial_g = _state(generator)
    fake = generator(input_map)
    mask = discriminator.valid_patch_mask(torch.ones_like(input_map, dtype=torch.bool))
    dis_optimizer.zero_grad(set_to_none=True)
    real_logits, fake_logits = discriminator(target), discriminator(fake.detach())
    d_loss = (F.binary_cross_entropy_with_logits(real_logits[mask], torch.ones_like(real_logits[mask]))
              + F.binary_cross_entropy_with_logits(fake_logits[mask], torch.zeros_like(fake_logits[mask]))) * 0.5
    d_loss.backward()
    dis_optimizer.step()
    assert _same(initial_g, _state(generator))
    assert all(value.grad is None for value in generator.parameters())
    # 参数冻结之外还需eval，保证谱归一化u/v缓冲也不更新。
    discriminator.eval().requires_grad_(False)
    frozen_d = _state(discriminator)
    gen_optimizer.zero_grad(set_to_none=True)
    fake_logits = discriminator(generator(input_map))
    g_loss = F.binary_cross_entropy_with_logits(fake_logits[mask], torch.ones_like(fake_logits[mask]))
    g_loss.backward()
    assert generator.ending.weight.grad.abs().sum() > 0
    gen_optimizer.step()
    assert _same(frozen_d, _state(discriminator))
    assert not _same(initial_g, _state(generator))
