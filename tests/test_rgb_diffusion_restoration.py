# -*- coding: utf-8 -*-
"""残差桥数学、完整采样和严格版本接口；不把这些测试当作竞赛增益。"""

import copy

import pytest
import torch

from models.rgb_diffusion_restoration import CHECKPOINT_FORMAT_DIFFUSION, PixelDiffusionRGBRestorer
from models.rgb_restoration import build_rgb_restorer, load_rgb_restorer


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model(**kwargs):
    options = dict(width=4, encoder_blocks=[1, 1], middle_blocks=1,
                   decoder_blocks=[1, 1], time_dim=8, num_steps=4)
    options.update(kwargs)
    return PixelDiffusionRGBRestorer(**options)


def seeded(seed=71):
    return torch.Generator().manual_seed(seed)


def test_schedule_and_training_start_match_deployment_prior():
    model = small_model(num_steps=16)
    assert model.eta.shape == (17,)
    assert model.eta[0] == 0 and model.eta[-1] == 1
    assert model.eta[1].item() == pytest.approx(0.001)
    assert (model.eta[1:] > model.eta[:-1]).all()
    # 终点必须与未知 x0 无关，且不是近似 eta_T=1。
    x0, y, noise = torch.rand(2, 3, 7, 11), torch.rand(2, 3, 7, 11), torch.randn(2, 3, 7, 11)
    t = torch.full((2,), 16, dtype=torch.long)
    expected = y + model.kappa * noise
    torch.testing.assert_close(model.q_sample(x0, y, t, noise), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.q_sample(1 - x0, y, t, noise), expected, rtol=0, atol=0)


def test_posterior_matches_independent_gaussian_conditioning_and_oracle_endpoint():
    model = small_model(num_steps=16)
    x0, y = torch.rand(16, 3, 3, 5), torch.rand(16, 3, 3, 5)
    t = torch.arange(1, 17)
    xt = model.q_sample(x0, y, t, generator=seeded())
    actual_mean, actual_variance = model.posterior_mean_variance(xt, x0, t)
    eta = model.eta[t, None, None, None]
    previous = model.eta[t - 1, None, None, None]
    prior_mean = (1 - previous) * x0 + previous * y
    marginal_mean = (1 - eta) * x0 + eta * y
    covariance = model.kappa ** 2 * previous
    marginal_variance = model.kappa ** 2 * eta
    expected_mean = prior_mean + covariance / marginal_variance * (xt - marginal_mean)
    expected_variance = covariance - covariance.square() / marginal_variance
    torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(actual_variance, expected_variance, rtol=1e-5, atol=1e-9)
    assert torch.equal(actual_mean[0], x0[0]) and actual_variance[0] == 0
    assert (actual_variance[1:] > 0).all()


def test_full_oracle_sampler_recovers_clean_image(monkeypatch):
    model = small_model(num_steps=16)
    clean, y = torch.rand(2, 3, 7, 11), torch.zeros(2, 3, 7, 11)
    seen_steps, seen_states = [], []

    def oracle(xt, condition, t):
        seen_steps.append(t[0].item())
        seen_states.append(xt.detach().clone())
        assert xt.dtype == torch.float32
        return clean

    monkeypatch.setattr(model, "denoise", oracle)
    torch.testing.assert_close(model(y, generator=seeded()), clean, rtol=0, atol=0)
    assert seen_steps == list(range(16, 0, -1))
    assert (seen_states[0] < 0).any(), "不能裁切带噪状态"


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (1, 3, 1, 13), (2, 3, 17, 29)])
def test_sampler_initial_identity_and_finite_nonzero_output(shape):
    model = small_model()
    y = 0.2 + 0.6 * torch.rand(shape)
    with torch.no_grad():
        assert torch.equal(model(y, generator=seeded()), y)
        model.correction.weight.normal_(0, 0.002)
        model.correction.bias.fill_(0.01)
        result = model(y, generator=seeded())
    assert result.shape == y.shape and result.dtype == torch.float32
    assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1
    assert not torch.equal(result, y)


def test_seed_replay_and_bypass_consume_no_rng():
    model = small_model()
    y = torch.rand(1, 3, 13, 15)
    with torch.no_grad():
        model.correction.weight.normal_(0, 0.02)
        assert torch.equal(model(y, generator=seeded()), model(y, generator=seeded()))
        assert not torch.equal(model(y, generator=seeded()), model(y, generator=seeded(72)))
    generator = seeded()
    before_global, before_local = torch.get_rng_state(), generator.get_state()
    assert model(y, strength=0, generator=generator) is y
    assert model(y, strength=0) is y
    assert torch.equal(before_global, torch.get_rng_state())
    assert torch.equal(before_local, generator.get_state())


def test_denoise_unbounded_and_autocast_keeps_bridge_fp32():
    model = small_model()
    y = torch.zeros(2, 3, 8, 12)
    xt, t = torch.full_like(y, -2), torch.tensor([1, 4])
    with torch.no_grad():
        model.correction.bias.fill_(2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            estimate = model.denoise(xt, y, t)
            sampled = model.q_sample(y, y, t, generator=seeded())
            mean, variance = model.posterior_mean_variance(xt, estimate, t)
            output = model(y, generator=seeded())
    assert estimate.min() > 1
    assert all(tensor.dtype == torch.float32 for tensor in (estimate, sampled, mean, variance, output))
    assert output.min() >= 0 and output.max() <= 1


def test_training_gradients_reach_time_and_multiscale_features():
    torch.manual_seed(41)
    model = small_model()
    y = 0.2 + 0.6 * torch.rand(2, 3, 17, 21)
    clean = 0.7 * y
    t = torch.tensor([1, 4])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        xt = model.q_sample(clean, y, t, generator=seeded())
        loss = (model.denoise(xt, y, t) - clean).square().mean()
        loss.backward()
        optimizer.step()
    for parameter in (model.correction.weight, model.stem.weight, model.time_mlp[0].weight,
                      model.encoders[0].time_affine[1].weight,
                      model.middle.time_affine[1].weight, model.decoders[-1].time_affine[1].weight,
                      model.middle.blocks[0].ffn_expand.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_strict_checkpoint_roundtrip_and_mismatch_rejection(tmp_path):
    model = small_model()
    with torch.no_grad():
        model.correction.weight.normal_(0, 0.002)
    payload = {"format": model.checkpoint_format, "model_config": model.model_config, "model": model.state_dict()}
    path = tmp_path / "diffusion.pt"
    torch.save(payload, path)
    restored = load_rgb_restorer(str(path))
    assert type(restored) is PixelDiffusionRGBRestorer and not restored.training
    assert restored.checkpoint_format == CHECKPOINT_FORMAT_DIFFUSION
    y = torch.rand(1, 3, 11, 13)
    with torch.no_grad():
        assert torch.equal(model(y, generator=seeded()), restored(y, generator=seeded()))
    for change in ("wrong_format", "wrong_architecture", "missing_architecture"):
        invalid = copy.deepcopy(payload)
        if change == "wrong_format":
            invalid["format"] = "rgb_restoration_nafnet_v1"
        elif change == "wrong_architecture":
            invalid["model_config"]["architecture"] = "nafnet"
        else:
            del invalid["model_config"]["architecture"]
        torch.save(invalid, path)
        with pytest.raises(ValueError, match="format.*architecture"):
            load_rgb_restorer(str(path))
    invalid = copy.deepcopy(payload)
    del invalid["model"]["time_mlp.0.weight"]
    torch.save(invalid, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_rgb_restorer(str(path))


def test_factory_and_invalid_timestep():
    original = small_model().model_config
    config = copy.deepcopy(original)
    assert type(build_rgb_restorer(config)) is PixelDiffusionRGBRestorer
    assert config == original
    model = small_model()
    y = torch.rand(1, 3, 7, 9)
    for t in (torch.tensor([0]), torch.tensor([5]), torch.tensor([1.0])):
        with pytest.raises(ValueError, match="t_index"):
            model.q_sample(y, y, t)
    with pytest.raises(ValueError, match="num_steps"):
        small_model(num_steps=1)
