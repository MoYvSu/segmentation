# -*- coding: utf-8 -*-
"""校光配对、恒等部署和前后顺序的必要契约。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from data.illumination_pairs import light_pair, IlluminationPairs
from models.illumination_calibration import IlluminationCalibrator, apply_curve, calibration_loss
from tools.run_illumination_calibration import calibrated, reflect_content, ordered_inputs, VARIANTS
from utils.config import load_config


def test_identity_including_black_and_white_and_nonreflect_padding():
    model = IlluminationCalibrator(width=4, context_size=32)
    image = torch.rand(2, 3, 64, 64)
    image[:, :, :2] = 0
    image[:, :, 2:4] = 1
    assert torch.equal(model(image), image)
    assert torch.equal(calibrated(model, image, 11, 5), image)
    native = torch.rand(64, 64, 3).permute(2, 0, 1).unsqueeze(0)
    result = calibrated(model, native, 11, 5)
    assert result.stride() == native.stride() and torch.equal(result, native)


def test_known_illumination_inverse_preserves_rgb_ratios():
    image = torch.rand(1, 3, 64, 64) * .6 + .1
    gain = torch.tensor(-.12)
    gamma = torch.tensor(1.15)
    changed = apply_curve(image, gain, gamma)
    restored = apply_curve(changed, -gain / gamma, 1 / gamma)
    torch.testing.assert_close(restored, image)
    torch.testing.assert_close(changed[:, 0] / changed[:, 1], image[:, 0] / image[:, 1])


def test_learning_gradient_reaches_zero_initialized_correction():
    model = IlluminationCalibrator(width=4, context_size=32)
    target = torch.rand(2, 3, 64, 64) * .6 + .1
    prediction, fields = model(target * .8, return_fields=True)
    loss, _ = calibration_loss(prediction, target, torch.ones_like(target[:, :1]), fields)
    loss.backward()
    assert model.gain_head.weight.grad.abs().sum() > 0
    assert model.gamma_head.weight.grad.abs().sum() > 0


def test_padding_matches_opencv_reflect():
    image = torch.arange(64 * 64).reshape(1, 1, 64, 64).float()
    expected = cv2.copyMakeBorder(image[0, 0, :49, :61].numpy(), 0, 15, 0, 3, cv2.BORDER_REFLECT)
    np.testing.assert_array_equal(reflect_content(image, 49, 61)[0, 0].numpy(), expected)


def test_light_pair_determinism_identity_and_clipping():
    cfg = load_config('config/train/illumination_calibration.yaml')['illumination_calibration']['augmentation']
    image = np.random.default_rng(1).uniform(.1, 1, (45, 56, 3)).astype(np.float32)
    a, info = light_pair(image, np.random.default_rng(2), cfg)
    b, _ = light_pair(image, np.random.default_rng(2), cfg)
    np.testing.assert_array_equal(a, b)
    assert info['clipped_fraction'] <= cfg['max_clipped_fraction']
    identity, _ = light_pair(image, np.random.default_rng(2), cfg, True)
    np.testing.assert_array_equal(identity, image)


def test_dataset_target_is_same_blur_not_clean(tmp_path):
    config = load_config('config/train/illumination_calibration.yaml')
    cfg = deepcopy(config['illumination_calibration'])
    cfg.update(expected_sources=1, image_size=64, blur_probability=1., identity_probability=1.)
    rng = np.random.default_rng(3)
    cv2.imwrite(str(tmp_path / 'train_1.png'), rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    degradation = load_config('config/train/rgb_diffusion_d5a.yaml')['rgb_restoration']['degradation']
    ds = IlluminationPairs(tmp_path, cfg, degradation)
    item = ds[0]
    assert item['blurred'] and item['identity']
    assert torch.equal(item['input'], item['target'])
    assert item['valid'].sum() == 48 * 64
    with pytest.raises(ValueError, match='all'):
        IlluminationPairs(tmp_path, {**cfg, 'expected_sources': 1000}, degradation)


def test_comparison_contains_both_orders_with_same_noise_seed(monkeypatch):
    calls = []
    def restore(_, image, seed):
        calls.append(seed)
        return image.square()
    monkeypatch.setattr('tools.run_illumination_calibration.restore_first', restore)
    class Calibrator:
        def __call__(self, image, valid):
            return image * .8
    image = torch.ones(1, 3, 64, 64) * .5
    values = ordered_inputs(image, 0, 0, Calibrator(), None, 123)
    assert tuple(values) == VARIANTS
    assert calls == [123, 123]
    assert not torch.equal(values['light_d5a'], values['d5a_light'])
    torch.testing.assert_close(values['light_d5a'], torch.full_like(image, .16))
    torch.testing.assert_close(values['d5a_light'], torch.full_like(image, .2))


def test_config_freezes_original_simple_and_whole_pool():
    cfg = load_config('config/train/illumination_calibration.yaml')['illumination_calibration']
    assert cfg['backend']['checkpoint'] == 'outputs/semantic_d5a/simple/epoch_060.pt'
    assert cfg['backend']['affinity_epoch'] == 115
    assert cfg['expected_sources'] == 1000 and cfg['epochs'] == 60
