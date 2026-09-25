# -*- coding: utf-8 -*-
"""D5a 端点模糊在完整人工图后端适配中的接线与配对契约。"""

from copy import deepcopy
import json
import random

import cv2
import numpy as np
import pytest
import torch

import data.backend_adaptation as backend_data
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.direct_dual_head_dataset import _spatial_transform


@pytest.fixture
def canonical(tmp_path):
    raw, completed = tmp_path / "raw", tmp_path / "completed"
    raw.mkdir()
    completed.mkdir()
    yy, xx = np.mgrid[:48, :64]
    texture = (50 + 100 * ((xx // 3 + yy // 3) % 2) + xx).astype(np.uint8)
    cv2.imwrite(str(raw / "sample.png"), np.repeat(texture[..., None], 3, axis=2))
    (raw / "sample.json").write_text("identity only", encoding="utf-8")
    ids = np.ones((48, 64), dtype=np.uint16)
    ids[:, 32:] = 2
    ids[8:14, 8:14] = 0
    np.savez(completed / "sample_gt.npz", instance_map=ids, residual_unknown=ids == 0)
    (completed / "sample_class.json").write_text(json.dumps({1: 1, 2: 0}), encoding="utf-8")
    return CanonicalBackendDataset(raw, completed, image_size=64, affinity_grid=32)


def degradation():
    return {
        "profile_probabilities": [0, 1, 0, 0], "blur_sigma": [0.4, 5.2],
        "resample_probability": 0.5, "resample_scale": [0.65, 1.0],
        "spatial_blur": {
            "enabled": True, "probability": 1.0,
            "grid_size": [3, 5], "strength": 1.0, "levels": 8,
            "endpoint_transition": {
                "enabled": True, "probability": 1.0,
                "weak_sigma": [0.6, 1.5], "strong_sigma": [4.0, 5.2],
                "transition_fraction": [0.2, 0.5], "min_region_fraction": 0.05,
            },
        },
    }


def paired(base, cfg, seed=41, repeats=4):
    return PairedDegradationDataset(
        base, cfg, {"enabled": True, "probability": 1.0, "sigma": [0.002, 0.006]},
        seed=seed, repeats=repeats,
    )


def assert_samples_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if torch.is_tensor(left[key]):
            assert torch.equal(left[key], right[key]), key
        else:
            assert left[key] == right[key], key


def test_endpoint_field_reaches_degradation_and_contains_both_ends(canonical, monkeypatch):
    """确认不是只读了配置/写了日志：实际第二次模糊必须收到端点场。"""
    calls = []
    original = backend_data.degrade_rgb

    def record(*args, **kwargs):
        field = kwargs.get("sigma_map_override")
        if field is not None:
            calls.append(field.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(backend_data, "degrade_rgb", record)
    cfg = degradation()
    endpoint = paired(canonical, cfg)[0]
    assert endpoint["endpoint_applied"] and endpoint["is_spatial"]
    assert len(calls) == 1
    sigma = calls[0]
    assert sigma.shape == (48, 64)  # 只作用真实内容，端点不锚定反射填充。
    assert 0.6 <= sigma.min() <= 1.5 and 4.0 <= sigma.max() <= 5.2
    assert np.mean(sigma <= 1.5) >= 0.05 and np.mean(sigma >= 4.0) >= 0.05
    legacy_cfg = deepcopy(cfg)
    legacy_cfg["spatial_blur"].pop("endpoint_transition")
    legacy = paired(canonical, legacy_cfg)[0]
    assert not torch.equal(endpoint["image"], legacy["image"])
    for key in ("profile", "base_sigma", "noise_sigma", "horizontal_flip", "vertical_flip", "rotation_k"):
        assert endpoint[key] == legacy[key], key
    assert torch.equal(endpoint["semantic_target"], legacy["semantic_target"])


def test_endpoint_pairs_repeat_without_global_random_state_changes(canonical):
    full, simple = paired(canonical, degradation()), paired(canonical, degradation())
    full.set_epoch(3)
    simple.set_epoch(3)
    np.random.seed(31)
    torch.manual_seed(37)
    random.seed(41)
    numpy_state, torch_state, python_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
    first_image = None
    for index in range(len(full)):
        left, right = full[index], simple[index]
        assert left["endpoint_applied"]
        assert_samples_equal(left, right)
        assert_samples_equal(left, full[index])
        if index == 0:
            first_image = left["image"]
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    full.set_epoch(4)
    simple.set_epoch(4)
    assert_samples_equal(full[0], simple[0])
    assert not torch.equal(first_image, full[0]["image"])


@pytest.mark.parametrize("mode", ["disabled", "zero_probability"])
def test_legacy_inputs_bitwise_unchanged_when_endpoint_is_inactive(canonical, mode):
    cfg = degradation()
    cfg["profile_probabilities"] = [0.2, 0.8, 0, 0]
    legacy_cfg = deepcopy(cfg)
    legacy_cfg["spatial_blur"].pop("endpoint_transition")
    if mode == "disabled":
        cfg["spatial_blur"]["endpoint_transition"]["enabled"] = False
    else:
        cfg["spatial_blur"]["endpoint_transition"]["probability"] = 0.0
    old, new = paired(canonical, legacy_cfg, repeats=8), paired(canonical, cfg, repeats=8)
    for index in range(len(old)):
        assert not new[index]["endpoint_applied"]
        assert_samples_equal(old[index], new[index])


def test_endpoint_keeps_every_gt_transform_and_ignore_aligned(canonical):
    original = canonical[0]
    dataset = paired(canonical, degradation(), repeats=8)
    for sample in dataset:
        assert sample["endpoint_applied"]
        for key in dataset._SPATIAL_KEYS:
            if key == "image":
                continue
            expected = _spatial_transform(
                original[key], sample["horizontal_flip"], sample["vertical_flip"], sample["rotation_k"],
            )
            assert torch.equal(sample[key], expected), key
        assert torch.equal(sample["semantic_target"][0] < 0, sample["semantic_instance_map"] == 0)
        assert torch.equal(sample["semantic_valid_content"][0], sample["semantic_instance_map"] > 0)


@pytest.mark.parametrize("mode", ["identity", "uniform"])
def test_endpoint_not_applied_to_nonspatial_profiles(canonical, mode, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("nonspatial input must not construct an endpoint field")

    monkeypatch.setattr(backend_data, "make_endpoint_sigma_map", unexpected)
    cfg = degradation()
    if mode == "identity":
        cfg["profile_probabilities"] = [1, 0, 0, 0]
    else:
        cfg["spatial_blur"]["probability"] = 0.0
    for sample in paired(canonical, cfg):
        assert not sample["endpoint_applied"]
        assert not sample["is_spatial"]
