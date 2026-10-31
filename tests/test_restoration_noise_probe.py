# -*- coding: utf-8 -*-
"""首步噪声探针的裁剪顺序与隔离性。"""
from unittest.mock import patch

import pytest
import torch

from utils.restoration_probe import restore_probe


class Dummy:
    num_steps = 16

    def q_sample(self, x0, condition, step, noise):
        assert torch.equal(x0, condition) and torch.all(step == 16)
        return condition + noise

    def denoise(self, state, condition, step):
        return 2 * state


def test_pair_averages_before_clipping_and_reuses_positive():
    image = torch.full((1, 3, 2, 2), .4)
    noise = torch.ones_like(image)
    with patch('utils.restoration_probe.restoration_noise', return_value=noise):
        single, raw = restore_probe(Dummy(), image, 7)
        pair, _ = restore_probe(Dummy(), image, 7, 'pair', positive_raw=raw)
        independent, _ = restore_probe(Dummy(), image, 7, 'pair')
    assert torch.equal(single, torch.ones_like(image))
    assert torch.allclose(pair, torch.full_like(image, .8))
    assert torch.equal(pair, independent)
    # 先裁剪再平均会成为 .5，不能用它取代真实 x0 平均。
    assert not torch.allclose(pair, torch.full_like(image, .5))


def test_zero_still_calls_the_network_and_leaves_input_unchanged():
    image = torch.full((1, 3, 2, 2), .2)
    original = image.clone()
    with patch('utils.restoration_probe.restoration_noise', side_effect=AssertionError):
        result, _ = restore_probe(Dummy(), image, 7, 'zero')
    assert torch.equal(image, original)
    assert torch.equal(result, image * 2)


def test_invalid_mode_and_cache_rejected():
    image = torch.zeros(1, 3, 2, 2)
    with pytest.raises(ValueError):
        restore_probe(Dummy(), image, 7, 'bypass')
    with pytest.raises(ValueError):
        restore_probe(Dummy(), image, 7, 'zero', positive_raw=image)
