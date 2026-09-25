# -*- coding: utf-8 -*-
"""空间模糊的物理量范围、旧随机流兼容及裁块前配对契约。"""

import copy

import cv2
import numpy as np
import pytest
import torch

import data.rgb_restoration_dataset as dataset_module
from data.rgb_restoration_dataset import RGBRestorationDataset, degrade_rgb, prepare_rgb, read_rgb
from data.rgb_spatial_blur import apply_spatial_blur, make_sigma_map, validate_spatial_blur_config


RECIPE = {"enabled": True, "probability": 1.0, "grid_size": [3, 5], "strength": 1.0, "levels": 8}
DEGRADATION = {"profile_probabilities": [0.0, 1.0, 0.0, 0.0], "blur_sigma": [0.4, 5.2],
               "resample_probability": 0.5, "resample_scale": [0.65, 1.0],
               "gamma": [0.8, 1.25], "exposure": [0.8, 1.15], "channel_gain": [0.95, 1.05],
               "illumination_strength": 0.15}


def legacy_degrade(clean, rng, profile, cfg):
    """保留新增模块前的外观公式，验证非命中分支不改变字节或rng游标。"""
    image = clean.copy()
    if profile in {"blur", "mixed"}:
        sigma = float(rng.uniform(*cfg["blur_sigma"]))
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
        if rng.random() < cfg["resample_probability"]:
            scale = float(rng.uniform(*cfg["resample_scale"]))
            h, w = image.shape[:2]
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    if profile in {"illumination", "mixed"}:
        gamma, exposure = float(rng.uniform(*cfg["gamma"])), float(rng.uniform(*cfg["exposure"]))
        gains = rng.uniform(*cfg["channel_gain"], size=(1, 1, 3)).astype(np.float32)
        amount = float(cfg["illumination_strength"])
        field = rng.uniform(1 - amount, 1 + amount, size=(2, 2)).astype(np.float32)
        field = cv2.resize(field, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        image = np.power(np.clip(image, 0, 1), gamma) * exposure * gains * field[..., None]
    return np.ascontiguousarray(np.clip(image, 0, 1), dtype=np.float32)


@pytest.mark.parametrize("profile", ["identity", "blur", "illumination", "mixed"])
def test_disabled_path_and_primary_random_stream_are_unchanged(profile):
    clean = np.random.default_rng(18).random((27, 41, 3), dtype=np.float32)
    for variant in (None, {"enabled": False}, dict(RECIPE, probability=0.0),
                    dict(RECIPE, strength=0.0)):
        cfg = copy.deepcopy(DEGRADATION)
        if variant is not None:
            cfg["spatial_blur"] = variant
        expected_rng, actual_rng = np.random.default_rng(29), np.random.default_rng(29)
        expected = legacy_degrade(clean, expected_rng, profile, cfg)
        info = {}
        actual = degrade_rgb(clean, actual_rng, profile, cfg,
                             spatial_rng=np.random.default_rng(42), spatial_info=info)
        assert np.array_equal(actual, expected)
        assert expected_rng.bit_generator.state == actual_rng.bit_generator.state
        assert not info["selected"] and info["sigma_map"] is None
    cfg["spatial_blur"] = RECIPE
    expected_rng, actual_rng = np.random.default_rng(29), np.random.default_rng(29)
    legacy_degrade(clean, expected_rng, profile, cfg)
    info = {}
    degrade_rgb(clean, actual_rng, profile, cfg,
                spatial_rng=np.random.default_rng(42), spatial_info=info)
    assert expected_rng.bit_generator.state == actual_rng.bit_generator.state
    assert info["selected"] == (profile in {"blur", "mixed"})
    # 不显式提供训练专用随机流时，旧固定monitor和验证继续走原退化。
    assert np.array_equal(degrade_rgb(clean, np.random.default_rng(29), profile, cfg),
                          legacy_degrade(clean, np.random.default_rng(29), profile, cfg))


def test_sigma_field_is_smooth_bounded_reproducible_and_keeps_mean_variance():
    for base in [0.4, 0.401, 1.1, 3.0, 5.199, 5.2]:
        sigma = make_sigma_map((256, 384), base, RECIPE, np.random.default_rng(67), [0.4, 5.2])
        repeat = make_sigma_map(sigma.shape, base, RECIPE, np.random.default_rng(67), [0.4, 5.2])
        assert sigma.dtype == np.float32 and np.array_equal(sigma, repeat)
        assert sigma.min() >= 0.4 - 1e-6 and sigma.max() <= 5.2 + 1e-6
        assert np.mean(sigma.astype(np.float64) ** 2) == pytest.approx(base ** 2, abs=2e-6)
        if 0.4 < base < 5.2:
            assert sigma.min() < base < sigma.max()
            # 分位区域均有实际面积；场相邻变化远小于跨图动态范围。
            assert (sigma <= np.quantile(sigma, 1 / 3)).sum() > sigma.size / 4
            assert (sigma >= np.quantile(sigma, 2 / 3)).sum() > sigma.size / 4
            variance = sigma ** 2
            assert max(np.abs(np.diff(variance, axis=0)).max(),
                       np.abs(np.diff(variance, axis=1)).max()) < np.ptp(variance) / 12
    original = make_sigma_map((256, 384), 3.0, RECIPE, np.random.default_rng(67), [0.4, 5.2])
    alternate = make_sigma_map((256, 384), 3.0, RECIPE, np.random.default_rng(68), [0.4, 5.2])
    assert not np.array_equal(original, alternate)


def test_constant_preservation_and_local_strength_are_effective():
    constant = np.full((128, 192, 3), 0.37, dtype=np.float32)
    result, _ = apply_spatial_blur(constant, 3.0, RECIPE, np.random.default_rng(12), [0.4, 5.2])
    np.testing.assert_allclose(result, constant, atol=2e-7, rtol=0)
    yy, xx = np.mgrid[:256, :384]
    wave = np.repeat((0.5 + 0.4 * np.sin(2 * np.pi * xx / 16))[..., None], 3, axis=2).astype(np.float32)
    untouched = wave.copy()
    blurred, sigma = apply_spatial_blur(wave, 3.0, RECIPE, np.random.default_rng(12), [0.4, 5.2])
    assert np.array_equal(wave, untouched)
    interior = (yy > 24) & (yy < 232) & (xx > 24) & (xx < 360)
    weak = interior & (sigma <= np.quantile(sigma, 0.25))
    strong = interior & (sigma >= np.quantile(sigma, 0.75))
    assert weak.any() and strong.any()
    assert np.mean((blurred[strong] - 0.5) ** 2) < np.mean((blurred[weak] - 0.5) ** 2) * 0.6
    # 零强度或恰在范围端点时，退回原高斯而非插值近似。
    for base, recipe in ((0.4, RECIPE), (5.2, RECIPE), (3.0, dict(RECIPE, strength=0.0))):
        actual, _ = apply_spatial_blur(wave, base, recipe, np.random.default_rng(12), [0.4, 5.2])
        expected = cv2.GaussianBlur(wave, (0, 0), sigmaX=base, borderType=cv2.BORDER_REFLECT)
        assert np.array_equal(actual, expected)


@pytest.fixture
def source_dir(tmp_path):
    yy, xx = np.mgrid[:60, :100]
    image = np.stack(((xx * 3) % 256, (yy * 4) % 256,
                      ((xx // 4 + yy // 7) % 2) * 180 + 30), axis=-1).astype(np.uint8)
    assert cv2.imwrite(str(tmp_path / "train_000.png"), image)
    return tmp_path


def make_dataset(source_dir, degradation, **kwargs):
    return RGBRestorationDataset(str(source_dir), split="train", split_policy="all_train",
                                 image_size=96, crop_size=48, degradation=degradation, seed=42, **kwargs)


def test_dataset_uses_full_valid_content_before_crop_and_keeps_geometry(source_dir, monkeypatch):
    original_apply = dataset_module.apply_spatial_blur
    observed = []

    def capture(clean, *args):
        observed.append(clean.shape)
        return original_apply(clean, *args)

    monkeypatch.setattr(dataset_module, "apply_spatial_blur", capture)
    masking = {"epochs": 1, "start_strength": 0.5, "patch_size": [8, 16],
               "coverage": [0.08, 0.16], "lowpass_sigma": [2.0, 4.0], "blend": [0.65, 0.95],
               "continuation_probability": 0.25}
    baseline = make_dataset(source_dir, copy.deepcopy(DEGRADATION), masked_pretraining=masking)
    recipe = dict(copy.deepcopy(DEGRADATION), spatial_blur=copy.deepcopy(RECIPE))
    candidate = make_dataset(source_dir, recipe, masked_pretraining=masking)
    cfg_before = copy.deepcopy(recipe)
    _, valid = prepare_rgb(read_rgb(str(source_dir / "train_000.png")), 96)
    expected_shape = (int(valid[:, 0].sum()), int(valid[0].sum()), 3)
    first = None
    for epoch in [0, 1, 5]:
        baseline.set_epoch(epoch)
        candidate.set_epoch(epoch)
        old, new = baseline[0], candidate[0]
        assert torch.equal(old["target"], new["target"]) and torch.equal(old["valid"], new["valid"])
        assert old["source"] == new["source"] and old["profile"] == new["profile"]
        assert old["mask_mean"] == new["mask_mean"]
        assert not torch.equal(old["input"], new["input"])
        assert new["spatial_blur_applied"] and 0.4 <= new["base_sigma"] <= 5.2
        assert new["spatial_sigma_min"] < new["spatial_sigma_max"]
        assert all(not isinstance(v, (np.ndarray, torch.Tensor)) for k, v in new.items()
                   if k not in {"input", "target", "valid"})
        assert torch.equal(new["input"], candidate[0]["input"])
        if epoch == 0:
            first = new["input"].clone()
    assert observed and all(shape == expected_shape for shape in observed)
    candidate.set_epoch(0)
    assert torch.equal(first, candidate[0]["input"])
    assert recipe == cfg_before
    assert "spatial_blur_applied" not in baseline[0]


def test_identity_and_holdout_remain_unchanged(source_dir):
    recipe = dict(copy.deepcopy(DEGRADATION), spatial_blur=copy.deepcopy(RECIPE),
                  profile_probabilities=[1.0, 0.0, 0.0, 0.0])
    identity = make_dataset(source_dir, recipe)[0]
    assert torch.equal(identity["input"], identity["target"])
    assert not identity["spatial_blur_applied"] and identity["base_sigma"] == 0
    manifest = source_dir / "holdout.txt"
    manifest.write_text("train_000.png\n", encoding="utf-8")
    options = dict(data_dir=str(source_dir), holdout_manifest=str(manifest), split="holdout",
                   image_size=96, crop_size=48, seed=42)
    old = RGBRestorationDataset(degradation=DEGRADATION, **options)
    new = RGBRestorationDataset(degradation=recipe, **options)
    for index in range(4):
        assert torch.equal(old[index]["input"], new[index]["input"])
        assert torch.equal(old[index]["target"], new[index]["target"])
        assert "spatial_blur_applied" not in new[index]


@pytest.mark.parametrize("override", [{"probability": -0.1}, {"probability": float("nan")},
                                     {"strength": 1.1}, {"grid_size": [1, 3]},
                                     {"grid_size": [5, 3]}, {"levels": 1}])
def test_invalid_recipe_is_rejected(override):
    with pytest.raises(ValueError, match="spatial_blur"):
        validate_spatial_blur_config(dict(RECIPE, **override))


def test_empty_images_and_invalid_sigma_bounds_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        make_sigma_map((0, 20), 3.0, RECIPE, np.random.default_rng(12), [0.4, 5.2])
    with pytest.raises(ValueError, match="bounds"):
        make_sigma_map((20, 20), 6.0, RECIPE, np.random.default_rng(12), [0.4, 5.2])
