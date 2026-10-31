# -*- coding: utf-8 -*-
"""上采样结构消融的CPU合同，不读取训练图像或已训练权重。"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from models.boundary_naf import BoundaryNAFNet
from models.boundary_rgb import BoundaryRGBNAFNet


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small(cls=BoundaryRGBNAFNet, upsampling='pixelshuffle', input_mode='boundary', seed=37):
    kwargs = dict(width=4, encoder_blocks=(1, 1), middle_blocks=1,
                  decoder_blocks=(1, 1), seed=seed, upsampling=upsampling)
    if cls is BoundaryRGBNAFNet:
        kwargs['input_mode'] = input_mode
    return cls(**kwargs)


def assert_same_state(first, second):
    a, b = first.state_dict(), second.state_dict()
    assert set(a) == set(b)
    assert all(torch.equal(value, b[key]) for key, value in a.items())


def test_default_metadata_and_explicit_pixelshuffle_keep_legacy_contract():
    default = BoundaryRGBNAFNet(width=4, encoder_blocks=[1, 1], middle_blocks=1,
                                decoder_blocks=[1, 1], seed=37, input_mode='boundary')
    explicit = small()
    assert default.architecture == dict(width=4, encoder_blocks=[1, 1], middle_blocks=1,
        decoder_blocks=[1, 1], seed=37, input_mode='boundary')
    assert 'upsampling' not in default.output_contract
    assert default.output_contract == explicit.output_contract
    assert default.architecture == explicit.architecture
    assert_same_state(default, explicit)
    # 旧architecture没有新字段，必须仍能严格重建及读取。
    loaded = BoundaryRGBNAFNet(**default.architecture)
    loaded.load_state_dict(default.state_dict(), strict=True)
    value = (torch.arange(31 * 47).reshape(1, 1, 31, 47) % 7 == 0).float()
    assert torch.equal(default(value), loaded(value))
    assert torch.equal(default(value), explicit(value))


@pytest.mark.parametrize('cls', [BoundaryNAFNet, BoundaryRGBNAFNet])
def test_shared_backbone_random_tensors_are_exact_and_new_up_has_own_version(cls):
    before = torch.get_rng_state()
    old = small(cls)
    new = small(cls, 'project_bilinear_v1')
    assert torch.equal(before, torch.get_rng_state())
    a, b = old.state_dict(), new.state_dict()
    assert set(a) == set(b)
    shared = [key for key in a if not key.startswith('ups.')]
    assert shared and all(torch.equal(a[key], b[key]) for key in shared)
    assert new.architecture == dict(old.architecture, upsampling='project_bilinear_v1')
    assert new.output_contract['upsampling'] == dict(
        version='project_bilinear_v1', projection='low_resolution_1x1_bias_false',
        resize='bilinear_x2_align_corners_false',
        initialization='independent_random_after_shared_legacy_construction',
        initialization_seed=37 + 20261029)
    assert not any(isinstance(module, nn.PixelShuffle) for module in new.modules())
    for previous, current in zip(old.ups, new.ups):
        assert current[0].in_channels == previous[0].in_channels
        assert current[0].out_channels * 4 == previous[0].out_channels
        assert current[0].kernel_size == (1, 1) and current[0].bias is None
    clone = small(cls, 'project_bilinear_v1')
    assert_same_state(new, clone)
    different_seed = small(cls, 'project_bilinear_v1', seed=38)
    assert any(not torch.equal(p, different_seed.state_dict()[key]) for key, p in b.items()
               if key.startswith('ups.'))


@pytest.mark.parametrize('cls,mode,channels', [
    (BoundaryNAFNet, 'boundary', 1),
    (BoundaryRGBNAFNet, 'boundary', 1),
    (BoundaryRGBNAFNet, 'rgb_boundary', 4),
])
@pytest.mark.parametrize('hw', [(32, 32), (31, 47), (1, 3)])
def test_resize_keeps_native_shape_and_gradients_through_all_up_levels(cls, mode, channels, hw):
    model = small(cls, 'project_bilinear_v1', input_mode=mode)
    rng = torch.Generator().manual_seed(19)
    value = torch.rand((1, channels, *hw), generator=rng)
    if cls is BoundaryRGBNAFNet:
        value[:, -1:] = (value[:, -1:] > .5).float()
    value.requires_grad_(True)
    prediction = model(value)
    assert prediction.shape == (1, 1, *hw)
    assert bool(torch.isfinite(prediction).all())
    assert bool(((prediction >= 0) & (prediction <= 1)).all())
    prediction.square().mean().backward()
    assert value.grad is not None and bool(torch.isfinite(value.grad).all())
    for up in model.ups:
        gradient = up[0].weight.grad
        assert gradient is not None and bool(torch.isfinite(gradient).all()) and gradient.abs().sum() > 0
    assert model.ending.weight.grad.abs().sum() > 0


def test_projection_then_bilinear_has_expected_half_pixel_geometry():
    # 手算2x2→4x4线性斜坡：检测align_corners或scale因子被误改。
    model = BoundaryNAFNet(width=2, encoder_blocks=[1], middle_blocks=1,
                           decoder_blocks=[1], upsampling='project_bilinear_v1')
    up = model.ups[0]
    with torch.no_grad():
        up[0].weight.zero_()
        up[0].weight[0, 0, 0, 0] = 1
        up[0].weight[1, 1, 0, 0] = 1
    value = torch.zeros(1, 4, 2, 2)
    value[0, 0] = torch.tensor([[0., 2.], [4., 6.]])
    value[0, 1] = 7
    actual = up(value)
    expected = torch.tensor([[0., .5, 1.5, 2.], [1., 1.5, 2.5, 3.],
                             [3., 3.5, 4.5, 5.], [4., 4.5, 5.5, 6.]])
    torch.testing.assert_close(actual[0, 0], expected, rtol=0, atol=0)
    assert torch.equal(actual[0, 1], torch.full((4, 4), 7.))


@pytest.mark.parametrize('cls', [BoundaryNAFNet, BoundaryRGBNAFNet])
def test_resize_architecture_strict_reload_and_incompatible_old_weights_rejected(cls):
    model = small(cls, 'project_bilinear_v1')
    clone = cls(**model.architecture)
    clone.load_state_dict(model.state_dict(), strict=True)
    value = (torch.arange(29 * 35).reshape(1, 1, 29, 35) % 5 < 2).float()
    assert torch.equal(model(value), clone(value))
    with pytest.raises(RuntimeError, match='size mismatch'):
        clone.load_state_dict(small(cls).state_dict(), strict=True)
    with pytest.raises(RuntimeError, match='size mismatch'):
        small(cls).load_state_dict(model.state_dict(), strict=True)
    malformed = deepcopy(model.state_dict())
    malformed['extra'] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        clone.load_state_dict(malformed, strict=True)


def test_parameter_count_and_rgb_channels_do_not_change_shared_resize_initialization():
    with torch.device('meta'):
        old = BoundaryRGBNAFNet(input_mode='boundary')
        new = BoundaryRGBNAFNet(input_mode='boundary', upsampling='project_bilinear_v1')
    assert sum(p.numel() for p in old.parameters()) == 17_632_273
    assert sum(p.numel() for p in new.parameters()) == 16_457_233
    rgb, single = small(upsampling='project_bilinear_v1', input_mode='rgb_boundary'), small(upsampling='project_bilinear_v1')
    assert not rgb.intro.weight[:, :3].any()
    assert torch.equal(rgb.intro.weight[:, 3:], single.intro.weight)
    for key, value in single.state_dict().items():
        if key != 'intro.weight':
            assert torch.equal(value, rgb.state_dict()[key])


@pytest.mark.parametrize('value', [None, True, 0, 'bilinear', 'nearest', 'project_bilinear_v2'])
def test_unknown_upsampling_cannot_silently_change_architecture(value):
    with pytest.raises(ValueError, match='upsampling'):
        small(upsampling=value)


def test_new_up_initialization_does_not_reseed_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda seed: calls.append(seed))
    small(upsampling='project_bilinear_v1')
    assert calls == []
