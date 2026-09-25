# -*- coding: utf-8 -*-
"""NAF 修复器的尺寸/训练/严格版本兼容契约，不用代理分割成绩选择模型。"""

import copy

import pytest
import torch

from models.rgb_restoration import (
    CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_NAFNET, NAFRGBRestorer, RGBRestorer,
    build_rgb_restorer, load_rgb_restorer,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_naf():
    return NAFRGBRestorer(width=4, encoder_blocks=[1, 1, 1], middle_blocks=1, decoder_blocks=[1, 1, 1])


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (1, 3, 1, 13), (2, 3, 17, 29), (1, 3, 32, 48)])
def test_naf_initial_identity_arbitrary_size_and_trained_bypass(shape):
    model = small_naf()
    inputs = 0.2 + 0.6 * torch.rand(shape)
    assert torch.equal(model(inputs), inputs)
    with torch.no_grad():
        model.correction.weight.normal_(0, 0.01)
        model.correction.bias.fill_(0.1)
    output = model(inputs)
    assert output.shape == inputs.shape and torch.isfinite(output).all()
    assert not torch.equal(output, inputs)
    assert output.min() >= 0 and output.max() <= 1
    assert (output - inputs).abs().max() <= model.max_residual
    assert model(inputs, strength=0) is inputs
    halfway = model(inputs, strength=0.5)
    torch.testing.assert_close(halfway, (inputs + output) / 2, rtol=1e-5, atol=1e-6)


def test_naf_backprop_reaches_multiscale_blocks_after_zero_initialization():
    torch.manual_seed(41)
    model = small_naf()
    inputs = 0.2 + 0.6 * torch.rand(2, 3, 17, 29)
    target = 0.7 * inputs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        optimizer.step()
    # 首步只训练零初始化末层；随后残差门与深层卷积均须得到有效梯度。
    for parameter in (
        model.correction.weight, model.stem.weight,
        model.encoders[0][0].depthwise.weight, model.middle[0].ffn_expand.weight,
        model.decoders[-1][0].expand.weight, model.middle[0].beta,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.isfinite(model(inputs)).all()


@pytest.mark.parametrize("architecture", ["legacy", "nafnet"])
def test_both_checkpoint_versions_roundtrip_with_nonzero_correction(tmp_path, architecture):
    model = RGBRestorer(width=4) if architecture == "legacy" else small_naf()
    with torch.no_grad():
        model.correction.weight.normal_(0, 0.01)
        model.correction.bias.fill_(0.025)
    path = tmp_path / "model.pt"
    payload = {"format": model.checkpoint_format, "model_config": model.model_config, "model": model.state_dict()}
    torch.save(payload, path)
    restored = load_rgb_restorer(str(path))
    inputs = torch.rand(1, 3, 15, 19)
    assert not restored.training
    assert type(restored) is type(model)
    assert restored.model_config == model.model_config
    assert torch.equal(restored(inputs), model(inputs))
    assert restored(inputs, strength=0) is inputs
    assert restored.checkpoint_format == (CHECKPOINT_FORMAT if architecture == "legacy" else CHECKPOINT_FORMAT_NAFNET)


def test_checkpoint_rejects_architecture_version_mismatch_and_partial_weights(tmp_path):
    model = small_naf()
    payload = {"format": model.checkpoint_format, "model_config": model.model_config, "model": model.state_dict()}
    path = tmp_path / "model.pt"
    variants = []
    wrong_format = copy.deepcopy(payload)
    wrong_format["format"] = CHECKPOINT_FORMAT
    variants.append(wrong_format)
    missing_architecture = copy.deepcopy(payload)
    del missing_architecture["model_config"]["architecture"]
    variants.append(missing_architecture)
    unknown_architecture = copy.deepcopy(payload)
    unknown_architecture["model_config"]["architecture"] = "nafnet_typo"
    variants.append(unknown_architecture)
    for invalid in variants:
        torch.save(invalid, path)
        with pytest.raises(ValueError, match="format.*architecture"):
            load_rgb_restorer(str(path))
    del payload["model"]["stem.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_rgb_restorer(str(path))


def test_factory_preserves_legacy_default_and_rejects_unknown_architecture():
    assert type(build_rgb_restorer({"width": 4})) is RGBRestorer
    assert type(build_rgb_restorer({"architecture": "rgb_restorer", "width": 4})) is RGBRestorer
    config = {"architecture": "nafnet", "width": 4, "encoder_blocks": [1], "middle_blocks": 1, "decoder_blocks": [1]}
    original = copy.deepcopy(config)
    assert type(build_rgb_restorer(config)) is NAFRGBRestorer
    assert config == original
    with pytest.raises(ValueError, match="unknown.*architecture"):
        build_rgb_restorer({"architecture": "nafnet_typo"})
    with pytest.raises(ValueError, match="same nonzero length"):
        NAFRGBRestorer(encoder_blocks=[1, 2], decoder_blocks=[1])
    with pytest.raises(ValueError, match="positive integers"):
        NAFRGBRestorer(middle_blocks=0)
