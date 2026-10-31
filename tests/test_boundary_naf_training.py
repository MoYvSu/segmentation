# -*- coding: utf-8 -*-
"""GAN全局分母、可靠感受野与分阶段更新的合成CPU验证。"""
from copy import deepcopy

import pytest
import torch

from models.boundary_naf import BoundaryDiscriminator, BoundaryNAFNet
from utils.boundary_naf_training import gan_update


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def models(seed=17):
    generator = BoundaryNAFNet(width=4, encoder_blocks=(1,), middle_blocks=1,
                              decoder_blocks=(1,), seed=seed)
    discriminator = BoundaryDiscriminator(width=4, seed=seed)
    return generator, discriminator


def optimizers(generator, discriminator):
    return (torch.optim.Adam(generator.parameters(), lr=1e-3),
            torch.optim.Adam(discriminator.parameters(), lr=1e-3))


def batch(count=4):
    rng = torch.Generator().manual_seed(99)
    input_map = torch.rand(count, 1, 96, 112, generator=rng)
    target = (torch.rand(input_map.shape, generator=rng) > 0.8).float()
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, :, :12, :7] = False
    target[~valid] = -1
    return dict(input=input_map, target=target, valid=valid)


def snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def equal_state(first, second):
    return all(torch.equal(first[key], second[key]) for key in first)


def test_microbatch_uses_global_pixel_patch_denominators_and_one_spectral_iteration():
    first_g, first_d = models()
    second_g, second_d = models()
    # 分母契约用SGD比较真实梯度更新。FP32卷积不同batch形状有舍入差，Adam在
    # 近零梯度处的归一化会放大该差；不能把逐位权重不同直接当作分母错误。
    # 其余测试仍用正式类型Adam，分别验收冻结、unknown和有限非零更新。
    first_go = torch.optim.SGD(first_g.parameters(), lr=1e-3)
    first_do = torch.optim.SGD(first_d.parameters(), lr=1e-3)
    second_go = torch.optim.SGD(second_g.parameters(), lr=1e-3)
    second_do = torch.optim.SGD(second_d.parameters(), lr=1e-3)
    data = batch()
    a = gan_update(first_g, first_d, first_go, first_do, data, micro_batch_size=4)
    b = gan_update(second_g, second_d, second_go, second_do, data, micro_batch_size=2)
    assert a["valid_pixels"] == b["valid_pixels"] == int(data["valid"].sum())
    assert a["adversarial_patches"] == b["adversarial_patches"] > 0
    assert a["spectral_power_iterations"] == b["spectral_power_iterations"] == 1
    for key in ("loss_g", "adv_d", "bce", "adv_g", "grad_norm_g", "grad_norm_d"):
        assert a[key] == pytest.approx(b[key], rel=3e-5, abs=1e-6)
    for first, second in zip(first_g.parameters(), second_g.parameters()):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-7)
    for first, second in zip(first_d.parameters(), second_d.parameters()):
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-7)
    assert a["nonzero_grad_g"] and a["nonzero_grad_d"]


def test_all_unknown_skips_both_optimizers_and_buffers():
    generator, discriminator = models()
    go, do = optimizers(generator, discriminator)
    data = batch(1)
    data["valid"].zero_()
    data["target"].fill_(float("nan"))
    old_g, old_d = snapshot(generator), snapshot(discriminator)
    result = gan_update(generator, discriminator, go, do, data)
    assert result["valid_pixels"] == result["adversarial_patches"] == 0
    assert not result["generator_updated"] and not result["discriminator_updated"]
    assert not go.state and not do.state
    assert equal_state(old_g, snapshot(generator)) and equal_state(old_d, snapshot(discriminator))


def test_sparse_known_support_uses_bce_without_claiming_gan_signal():
    generator, discriminator = models()
    go, do = optimizers(generator, discriminator)
    data = batch(1)
    data["valid"].zero_()
    data["valid"][..., 44:52, 44:52] = True
    data["target"][~data["valid"]] = -1
    old_d = snapshot(discriminator)
    result = gan_update(generator, discriminator, go, do, data)
    assert result["valid_pixels"] == 64 and result["adversarial_patches"] == 0
    assert result["generator_updated"] and not result["discriminator_updated"]
    assert not result["adversarial_active"] and result["adv_g"] == result["adv_d"] == 0
    assert equal_state(old_d, snapshot(discriminator))


def test_unknown_target_values_do_not_change_update_and_unknown_output_gradient_zero():
    first_g, first_d = models()
    second_g, second_d = models()
    first_go, first_do = optimizers(first_g, first_d)
    second_go, second_do = optimizers(second_g, second_d)
    data = batch(1)
    changed = deepcopy(data)
    changed["target"][~changed["valid"]] = float("nan")
    observed = []
    original_forward = first_g.forward_logits

    def traced_forward(value):
        logits = original_forward(value)
        if logits.requires_grad:
            logits.retain_grad()
            observed.append(logits)
        return logits

    first_g.forward_logits = traced_forward
    a = gan_update(first_g, first_d, first_go, first_do, data)
    b = gan_update(second_g, second_d, second_go, second_do, changed)
    assert a["loss_g"] == b["loss_g"]
    assert equal_state(snapshot(first_g), snapshot(second_g))
    assert equal_state(snapshot(first_d), snapshot(second_d))
    assert len(observed) == 1
    assert torch.count_nonzero(observed[0].grad[~data["valid"]]) == 0


def test_topology_micro_sample_weights_and_restored_modes_flags():
    generator, discriminator = models()
    go, do = optimizers(generator, discriminator)
    generator.eval()
    discriminator.train()
    old_flags = [parameter.requires_grad for parameter in discriminator.parameters()]
    records = []

    def topology(probability, target, valid):
        per_sample = ((probability - target).square() * valid).flatten(1).mean(1)
        records.append(len(probability))
        return per_sample.mean(), {"critical_points": 2 * len(probability), "enabled": True}

    result = gan_update(generator, discriminator, go, do, batch(3), micro_batch_size=2,
                        topology_fn=topology, topology_weight=0.001)
    assert records == [2, 1]
    assert result["topology_counts"]["critical_points"] == 6
    assert len(result["topology_audit"]) == 2
    assert result["loss_g"] == pytest.approx(result["bce"] + result["adv_g"]
                                             + 0.001 * result["topology_loss"], abs=1e-6)
    assert not generator.training and discriminator.training
    assert [parameter.requires_grad for parameter in discriminator.parameters()] == old_flags


def test_zero_adversarial_weight_does_not_train_discriminator():
    generator, discriminator = models()
    go, do = optimizers(generator, discriminator)
    old_d = snapshot(discriminator)
    result = gan_update(generator, discriminator, go, do, batch(1), adversarial_weight=0)
    assert result["adversarial_patches"] > 0 and not result["adversarial_active"]
    assert result["generator_updated"] and not result["discriminator_updated"]
    assert equal_state(old_d, snapshot(discriminator))


def test_known_invalid_target_rejected_and_topology_failure_restores_flags():
    generator, discriminator = models()
    go, do = optimizers(generator, discriminator)
    data = batch(1)
    data["target"][0, 0, 60, 60] = -1
    with pytest.raises(ValueError, match="Known GT"):
        gan_update(generator, discriminator, go, do, data)
    with pytest.raises(ValueError, match="requires topology_fn"):
        gan_update(generator, discriminator, go, do, batch(1), topology_weight=1)

    def broken_topology(*args):
        raise RuntimeError("topology intentionally broken")

    generator.eval()
    discriminator.eval()
    with pytest.raises(RuntimeError, match="intentionally broken"):
        gan_update(generator, discriminator, go, do, batch(1), topology_weight=1,
                   topology_fn=broken_topology)
    assert not generator.training and not discriminator.training
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
