# -*- coding: utf-8 -*-
"""D5a实际裁块端点覆盖、D4兼容以及整图退化/几何配对契约。"""

import copy

import cv2
import numpy as np
import pytest
import torch

import data.rgb_restoration_dataset as dataset_module
from data.rgb_restoration_dataset import RGBRestorationDataset
from data.rgb_spatial_blur import (apply_sigma_map_blur, make_endpoint_sigma_map,
                                   validate_spatial_blur_config)


ENDPOINT = {"enabled": True, "probability": 0.5, "weak_sigma": [0.6, 1.5],
            "strong_sigma": [4.0, 5.2], "transition_fraction": [0.2, 0.5],
            "min_region_fraction": 0.05}
SPATIAL = {"enabled": True, "probability": 0.5, "grid_size": [3, 5],
           "strength": 1.0, "levels": 8}
DEGRADATION = {"profile_probabilities": [0.2, 0.8, 0.0, 0.0], "blur_sigma": [0.4, 5.2],
               "resample_probability": 0.5, "resample_scale": [0.65, 1.0],
               "gamma": [0.8, 1.25], "exposure": [0.8, 1.15], "channel_gain": [0.95, 1.05],
               "illumination_strength": 0.15, "spatial_blur": SPATIAL}


@pytest.mark.parametrize("shape,crop", [((1024, 640), (512, 128, 512, 512)),
                                        ((153, 1024), (0, 400, 512, 512)),
                                        ((1024, 153), (400, 0, 512, 512))])
def test_endpoint_coverage_on_actual_valid_crop_and_smooth_field(shape, crop):
    for seed in range(12):
        sigma = make_endpoint_sigma_map(shape, crop, ENDPOINT, np.random.default_rng(seed), [0.4, 5.2])
        y, x, h, w = crop
        region = sigma[y:y + h, x:x + w]
        assert sigma.dtype == np.float32 and sigma.shape == shape
        assert sigma.min() >= 0.6 - 1e-6 and sigma.max() <= 5.2 + 1e-6
        assert (region <= 1.5).mean() >= 0.05 and (region >= 4.0).mean() >= 0.05
        variance = sigma ** 2
        assert max(np.abs(np.diff(variance, axis=0)).max(),
                   np.abs(np.diff(variance, axis=1)).max()) < np.ptp(variance) / 8
        assert np.array_equal(sigma, make_endpoint_sigma_map(
            shape, crop, ENDPOINT, np.random.default_rng(seed), [0.4, 5.2]))


def test_endpoint_interpolation_preserves_constant_and_strong_region_is_more_blurred():
    yy, xx = np.mgrid[:128, :256]
    sigma = make_endpoint_sigma_map((128, 256), (0, 32, 128, 128), ENDPOINT,
                                    np.random.default_rng(7), [0.4, 5.2])
    constant = np.full((128, 256, 3), 0.37, dtype=np.float32)
    np.testing.assert_allclose(apply_sigma_map_blur(constant, sigma, [0.4, 5.2]),
                               constant, atol=2e-7, rtol=0)
    wave = np.repeat((0.5 + 0.4 * np.sin(2 * np.pi * xx / 16))[..., None], 3, axis=2).astype(np.float32)
    blurred = apply_sigma_map_blur(wave, sigma, [0.4, 5.2])
    interior = (xx > 20) & (xx < 236) & (yy > 20) & (yy < 108)
    weak, strong = interior & (sigma <= 1.5), interior & (sigma >= 4)
    assert weak.any() and strong.any()
    assert np.mean((blurred[strong] - 0.5) ** 2) < np.mean((blurred[weak] - 0.5) ** 2) * 0.4


@pytest.fixture
def source_dir(tmp_path):
    yy, xx = np.mgrid[:39, :100]
    image = np.stack(((xx * 3) % 256, (yy * 4) % 256,
                      ((xx // 4 + yy // 7) % 2) * 180 + 30), axis=-1).astype(np.uint8)
    assert cv2.imwrite(str(tmp_path / "train_000.png"), image)
    return tmp_path


def dataset(source_dir, endpoint=None, **overrides):
    cfg = copy.deepcopy(DEGRADATION)
    cfg.update(overrides)
    if endpoint is not None:
        cfg["spatial_blur"]["endpoint_transition"] = endpoint
    return RGBRestorationDataset(str(source_dir), split="train", split_policy="all_train",
                                 image_size=96, crop_size=48, degradation=cfg, seed=42)


@pytest.mark.parametrize("endpoint", [{"enabled": False}, dict(ENDPOINT, probability=0.0)])
def test_missing_disabled_and_zero_probability_paths_are_byte_identical(source_dir, endpoint):
    old, new = dataset(source_dir), dataset(source_dir, endpoint)
    for epoch in range(12):
        old.set_epoch(epoch)
        new.set_epoch(epoch)
        expected, actual = old[0], new[0]
        for key in expected:
            if isinstance(expected[key], torch.Tensor):
                assert torch.equal(expected[key], actual[key])
            else:
                assert expected[key] == actual[key]
        assert not actual["endpoint_blur_applied"]


def test_endpoint_replay_keeps_rng_geometry_and_reports_only_valid_crop(source_dir, monkeypatch):
    original_degrade = dataset_module.degrade_rgb
    original_field = dataset_module.make_endpoint_sigma_map
    calls, fields = [], []

    def capture_degrade(clean, rng, *args, **kwargs):
        result = original_degrade(clean, rng, *args, **kwargs)
        calls.append({"shape": clean.shape, "rng": rng, "state": copy.deepcopy(rng.bit_generator.state),
                      "spatial_rng": kwargs["spatial_rng"]})
        return result

    def capture_field(shape, crop, *args):
        field = original_field(shape, crop, *args)
        fields.append((field, crop))
        return field

    monkeypatch.setattr(dataset_module, "degrade_rgb", capture_degrade)
    monkeypatch.setattr(dataset_module, "make_endpoint_sigma_map", capture_field)
    spatial = dict(SPATIAL, probability=1.0)
    # mixed同时覆盖重采样、照明和裁块之后的旋转/翻转随机流。
    old = dataset(source_dir, spatial_blur=spatial, profile_probabilities=[0, 0, 0, 1])
    new = dataset(source_dir, dict(ENDPOINT, probability=1.0),
                  spatial_blur=copy.deepcopy(spatial), profile_probabilities=[0, 0, 0, 1])
    for epoch in [0, 4, 60]:
        old.set_epoch(epoch)
        new.set_epoch(epoch)
        calls.clear()
        expected, actual = old[0], new[0]
        assert len(calls) == 3
        assert all(call["shape"] == (37, 96, 3) for call in calls)
        assert calls[0]["rng"].bit_generator.state == calls[1]["rng"].bit_generator.state
        assert calls[0]["spatial_rng"].bit_generator.state == calls[1]["spatial_rng"].bit_generator.state
        assert calls[1]["state"] == calls[2]["state"]
        assert torch.equal(expected["target"], actual["target"])
        assert torch.equal(expected["valid"], actual["valid"])
        assert not torch.equal(expected["input"], actual["input"])
        assert actual["endpoint_blur_applied"] and actual["sigma_crop_coexist"]
        assert actual["base_sigma"] == expected["base_sigma"]
        sigma, (y, x, h, w) = fields[-1]
        region = sigma[y:y + h, x:x + w]
        assert actual["sigma_crop_weak_fraction"] == float((region <= 1.5).mean())
        assert actual["sigma_crop_strong_fraction"] == float((region >= 4).mean())
        assert actual["sigma_crop_mean_square"] == pytest.approx(np.mean(region.astype(np.float64) ** 2))
        assert actual["sigma_crop_span"] == pytest.approx(np.quantile(region, 0.9) - np.quantile(region, 0.1))
        assert all(not isinstance(value, (np.ndarray, torch.Tensor)) for key, value in actual.items()
                   if key not in {"input", "target", "valid"})


def test_nonendpoint_statistics_use_crop_and_identity_is_zero(source_dir, monkeypatch):
    observed = []
    original_degrade = dataset_module.degrade_rgb

    def capture(clean, *args, **kwargs):
        image = original_degrade(clean, *args, **kwargs)
        observed.append(copy.deepcopy(kwargs["spatial_info"]))
        return image

    monkeypatch.setattr(dataset_module, "degrade_rgb", capture)
    spatial = dataset(source_dir, spatial_blur=dict(SPATIAL, probability=1.0),
                      profile_probabilities=[0, 1, 0, 0])
    item = spatial[0]
    # 从同一主随机流重建原裁块位置，确认没有偷用整图统计。
    rng = np.random.default_rng(np.random.SeedSequence([42, 0, 0, 0]))
    rng.choice(dataset_module.PROFILES, p=[0, 1, 0, 0])
    original_degrade(np.zeros((37, 96, 3), np.float32), rng, "blur", spatial.degradation,
                     spatial_rng=np.random.default_rng(np.random.SeedSequence([42, 0, 0, 6])))
    y, x = int(rng.integers(1)), int(rng.integers(49))
    region = observed[0]["sigma_map"][y:y + 48, x:x + 48]
    assert item["sigma_crop_mean_square"] == pytest.approx(np.mean(region.astype(np.float64) ** 2))
    uniform = dataset(source_dir, spatial_blur=dict(SPATIAL, probability=0.0),
                      profile_probabilities=[0, 1, 0, 0])[0]
    assert uniform["sigma_crop_mean_square"] == pytest.approx(uniform["base_sigma"] ** 2)
    assert uniform["sigma_crop_span"] == 0 and not uniform["sigma_crop_coexist"]
    identity = dataset(source_dir, ENDPOINT, profile_probabilities=[1, 0, 0, 0])[0]
    assert torch.equal(identity["input"], identity["target"])
    assert not identity["endpoint_blur_applied"]
    assert all(value == 0 for key, value in identity.items() if key.startswith("sigma_crop_"))


@pytest.mark.parametrize("override", [{"weak_sigma": [0.6, 1.6]}, {"strong_sigma": [3.9, 5.2]},
                                     {"transition_fraction": [0.0, 0.5]}, {"min_region_fraction": 0.5},
                                     {"probability": -0.1}])
def test_invalid_endpoint_config_is_rejected(override):
    with pytest.raises(ValueError, match="endpoint_transition"):
        validate_spatial_blur_config(dict(SPATIAL, endpoint_transition=dict(ENDPOINT, **override)))
