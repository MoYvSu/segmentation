# -*- coding: utf-8 -*-
"""检查光照探针不偷换几何、颜色差及随机流。"""
import math

import torch

from data.semantic_illumination import apply_illumination, fixed_smooth_field


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
