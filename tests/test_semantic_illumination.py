# -*- coding: utf-8 -*-
"""检查光照探针不偷换几何、颜色差及随机流。"""
import math
import random

import numpy as np
import pytest
import torch

from data.semantic_illumination import apply_illumination, fixed_smooth_field, training_illumination


def test_identity_preserves_object_and_rng():
    image = torch.full((1, 3, 8, 12), .5)
    state = torch.random.get_rng_state().clone()
    for kind, value in (("offset", 0), ("gamma", 1), ("local", 0)):
        assert apply_illumination(image, kind, value) is image
    assert torch.equal(state, torch.random.get_rng_state())


def test_channel_differences_and_clipping_are_explicit():
    image = torch.tensor([.2, .4, .9]).view(1, 3, 1, 1).expand(1, 3, 4, 4)
    changed = apply_illumination(image, "gamma", .8, clamp=False)
    torch.testing.assert_close(changed[:, 2] - changed[:, 0], image[:, 2] - image[:, 0])
    unclipped = apply_illumination(image, "offset", .2, clamp=False)
    assert unclipped.max() > 1
    assert apply_illumination(image, "offset", .2).max() == 1


def test_field_reflects_narrow_content_and_is_smooth():
    image = torch.zeros(1, 3, 12, 20)
    field = fixed_smooth_field(image, content_shape=(8, 5))
    torch.testing.assert_close(field[..., 5], field[..., 3])
    torch.testing.assert_close(field[..., 8], field[..., 0])
    assert field.min() >= -1 and field.max() <= 1
    assert torch.all(field[0, 0, 0, :4] > field[0, 0, 0, 1:5])
    assert field[0, 0, :8, :5].mean().abs() < 1e-6


def test_field_uses_moved_rectangle_after_geometry():
    image = torch.zeros(1, 3, 12, 20)
    mask = torch.zeros(1, 1, 12, 20)
    mask[..., 3:11, 7:12] = 1
    field = fixed_smooth_field(image, content_mask=mask)
    assert field[0, 0, 3, 7] == 1
    assert field[0, 0, 3, 11] == -1
    torch.testing.assert_close(field[..., 6], field[..., 8])
    vertical = fixed_smooth_field(image, content_mask=mask, angle=math.pi / 2)
    torch.testing.assert_close(vertical[0, 0, 3, 7], torch.tensor(1.))
    torch.testing.assert_close(vertical[0, 0, 10, 7], torch.tensor(-1.))


def training_config(**overrides):
    return {"enabled": True, "probability": 1.0, "local_probability": .5,
            "max_offset": .1, "max_clipped_fraction": .01, **overrides}


@pytest.mark.parametrize("config", [None, {}, {"enabled": False},
                                   {"enabled": True, "probability": 0},
                                   {"enabled": True, "max_offset": 0}])
def test_training_disabled_is_exact_identity(config):
    image = torch.linspace(0, 1, 60).reshape(1, 3, 4, 5)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    changed, stats = training_illumination(image, config, seed=9)
    assert changed is image
    assert not stats["applied"]
    assert stats["actual_offset"] == stats["clipped_fraction"] == 0
    assert np.random.get_state()[0] == numpy_state[0]
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("local_probability", [0, 1])
def test_training_seed_replays_without_advancing_any_global_rng(local_probability):
    image = torch.full((1, 3, 8, 12), .5)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    config = training_config(local_probability=local_probability)
    changed, stats = training_illumination(image, config, seed=0)
    replay, replay_stats = training_illumination(image, config, seed=0)
    another, _ = training_illumination(image, config, seed=41)
    assert torch.equal(changed, replay) and stats == replay_stats
    assert not torch.equal(changed, another)
    assert stats["applied"] and stats["scale"] == 1
    assert np.random.get_state()[0] == numpy_state[0]
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("local_probability", [0, 1])
def test_training_preserves_chroma_and_needs_no_labels(local_probability):
    image = torch.tensor([.2, .4, .6]).reshape(1, 3, 1, 1).expand(1, 3, 8, 12).clone()
    original = image.clone()
    changed, stats = training_illumination(image, training_config(local_probability=local_probability), 0)
    assert torch.equal(image, original)
    assert stats["clipped_fraction"] == 0
    torch.testing.assert_close(changed[:, 2] - changed[:, 0], image[:, 2] - image[:, 0])
    torch.testing.assert_close(changed[:, 1] - changed[:, 0], image[:, 1] - image[:, 0])
    assert changed.dtype == image.dtype and changed.shape == image.shape


def test_training_local_field_uses_moved_geometric_content():
    image = torch.full((1, 3, 12, 20), .5)
    mask = torch.zeros(1, 1, 12, 20, dtype=torch.bool)
    mask[..., 3:11, 7:12] = True
    changed, stats = training_illumination(image, training_config(local_probability=1), 0, mask)
    expected_field = fixed_smooth_field(image, content_mask=mask, angle=stats["angle"])
    torch.testing.assert_close(changed, image + stats["actual_offset"] * expected_field)
    assert stats["kind"] == "local" and stats["scale"] == 1
    # 即使有效区翻转到右下，边框的饱和值不消耗有效区截断预算。
    saturated_padding = torch.where(mask, image, torch.ones_like(image))
    _, padding_stats = training_illumination(
        saturated_padding, training_config(local_probability=0), 0, mask)
    assert padding_stats["scale"] == 1 and padding_stats["clipped_fraction"] == 0


def test_training_reduces_offset_to_valid_pixel_clipping_budget():
    image = torch.linspace(0, 1, 10000).reshape(1, 1, 100, 100).expand(1, 3, 100, 100).clone()
    changed, stats = training_illumination(image, training_config(local_probability=0), 0)
    assert 0 < stats["scale"] < 1
    assert 0 < stats["actual_offset"] < stats["requested_offset"]
    raw = image + stats["actual_offset"]
    measured = float(((raw < 0) | (raw > 1)).any(dim=1).float().mean())
    assert measured <= .010001
    assert stats["clipped_fraction"] == pytest.approx(measured)
    torch.testing.assert_close(changed, raw.clamp(0, 1))


def test_training_saturated_image_can_become_exact_identity():
    image = torch.ones(1, 3, 8, 12)
    changed, stats = training_illumination(image, training_config(local_probability=0), 0)
    assert stats["requested_offset"] > 0
    assert stats["actual_offset"] == stats["scale"] == stats["clipped_fraction"] == 0
    assert not stats["applied"] and changed is image


@pytest.mark.parametrize("overrides", [
    {"enabled": "true"}, {"probability": -.1}, {"probability": float("nan")},
    {"local_probability": 1.1}, {"max_offset": -.1}, {"max_offset": float("inf")},
    {"max_clipped_fraction": -1}, {"max_clipped_fraction": True},
])
def test_training_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        training_illumination(torch.full((1, 3, 4, 5), .5), training_config(**overrides), 0)


def test_training_rejects_empty_content_mask():
    with pytest.raises(ValueError, match="content mask"):
        training_illumination(torch.full((1, 3, 4, 5), .5), training_config(), 0,
                              torch.zeros(1, 1, 4, 5))
