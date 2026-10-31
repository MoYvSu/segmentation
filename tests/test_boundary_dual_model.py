# -*- coding: utf-8 -*-
"""双输入NAF合成CPU合同；不读取赛题数据或已训练权重。"""
import copy
import io

import pytest
import torch

from models.boundary_dual import BoundaryDualNAFNet, FORMAT, INPUT_MODE
from models.boundary_rgb import BoundaryRGBNAFNet


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _kwargs():
    return dict(width=4, encoder_blocks=(1, 1), middle_blocks=1,
                decoder_blocks=(1, 1), seed=17)


def _input(shape=(2, 2, 31, 47)):
    value = torch.rand(shape)
    value[:, :1] = (value[:, :1] >= .5).float()
    return value


def test_common_initialization_and_rng_are_exact():
    before = torch.get_rng_state().clone()
    old = BoundaryRGBNAFNet(**_kwargs(), input_mode='boundary')
    dual = BoundaryDualNAFNet(**_kwargs())
    assert torch.equal(before, torch.get_rng_state())
    original = old.state_dict()
    current = dual.state_dict()
    assert set(current) - set(original) == {'continuous_intro.weight'}
    assert all(torch.equal(current[key], tensor) for key, tensor in original.items())
    assert not dual.continuous_intro.weight.any()
    assert dual.intro.in_channels == 1 and dual.continuous_intro.bias is None
    assert sum(x.numel() for x in dual.parameters()) - sum(x.numel() for x in old.parameters()) == 36


@pytest.mark.parametrize('shape', [(2, 2, 32, 32), (2, 2, 31, 47), (1, 2, 1, 3)])
def test_zero_branch_and_disabled_path_exact_legacy_cpu(shape):
    old = BoundaryRGBNAFNet(**_kwargs(), input_mode='boundary')
    dual = BoundaryDualNAFNet(**_kwargs())
    value = _input(shape)
    expected = old(value[:, :1].contiguous())
    assert torch.equal(dual(value), expected)
    assert torch.equal(dual(value, use_continuous=False), expected)
    assert dual(value).shape == (shape[0], 1, *shape[-2:])


def test_continuous_branch_learns_and_disabled_path_ignores_its_weights():
    model = BoundaryDualNAFNet(**_kwargs())
    old = BoundaryRGBNAFNet(**_kwargs(), input_mode='boundary')
    value = _input()
    model(value).mean().backward()
    grad = model.continuous_intro.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        model.continuous_intro.weight.fill_(.125)
    assert not torch.equal(model(value), old(value[:, :1].contiguous()))
    assert torch.equal(model(value, use_continuous=False), old(value[:, :1].contiguous()))


def test_global_skip_uses_t8_only():
    model = BoundaryDualNAFNet(**_kwargs())
    with torch.no_grad():
        model.ending.weight.zero_()
        model.ending.bias.zero_()
        model.continuous_intro.weight.fill_(1.)
    value = _input()
    assert torch.equal(model.forward_logits(value), value[:, :1])
    assert torch.equal(model.residual(value), model(value) - value[:, :1])


def test_explicit_contract_strict_save_reload_and_reject_old_state():
    model = BoundaryDualNAFNet(**_kwargs())
    assert model.checkpoint_format == FORMAT
    assert model.architecture['input_mode'] == INPUT_MODE
    assert model.output_contract['input_channels'] == 2
    assert not model.output_contract['continuous_input_binarized']
    buffer = io.BytesIO()
    torch.save(dict(format=FORMAT, architecture=model.architecture, state_dict=model.state_dict()), buffer)
    buffer.seek(0)
    receipt = torch.load(buffer, map_location='cpu', weights_only=True)
    clone = BoundaryDualNAFNet(**receipt['architecture'])
    clone.load_state_dict(receipt['state_dict'], strict=True)
    value = _input()
    assert torch.equal(model(value), clone(value))
    with pytest.raises(RuntimeError):
        clone.load_state_dict(BoundaryRGBNAFNet(**_kwargs(), input_mode='boundary').state_dict(), strict=True)
    damaged = copy.deepcopy(model.state_dict())
    damaged['extra'] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        clone.load_state_dict(damaged, strict=True)


@pytest.mark.parametrize('bad', [
    torch.zeros(1, 1, 8, 8), torch.zeros(1, 3, 8, 8), torch.zeros(1, 2, 0, 8),
    torch.zeros(1, 2, 8, 8, dtype=torch.int64),
    torch.full((1, 2, 8, 8), float('nan')), torch.full((1, 2, 8, 8), float('inf')),
    torch.full((1, 2, 8, 8), -.1), torch.full((1, 2, 8, 8), 1.1),
])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        BoundaryDualNAFNet(**_kwargs())(bad)


def test_binary_t8_required_but_second_channel_can_be_continuous():
    model = BoundaryDualNAFNet(**_kwargs())
    value = _input()
    value[:, 1:2] = .49
    assert torch.isfinite(model(value)).all()
    value[:, :1] = .49
    with pytest.raises(ValueError, match='binary T8'):
        model(value)


def test_oversized_batch_rejected_before_any_tensor_value_read():
    model = BoundaryDualNAFNet(**_kwargs())
    oversized = torch.empty(1, 2, 4096, 4096, device='meta')
    with pytest.raises(ValueError, match='bounded'):
        model(oversized)


def test_mode_and_runtime_flag_are_explicit():
    with pytest.raises(ValueError):
        BoundaryDualNAFNet(**_kwargs(), input_mode='rgb_boundary')
    with pytest.raises(ValueError):
        BoundaryDualNAFNet(**_kwargs())(_input(), use_continuous=0)
