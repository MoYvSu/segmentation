# -*- coding: utf-8 -*-
"""原尺寸裁剪的坐标一致性、unknown ignore 和整图兼容性。"""

import json

import cv2
import numpy as np
import pytest
import torch

from data.dataset import letterbox
from data.direct_dual_head_dataset import DirectDualHeadDataset, instance_boundary_target
from utils.affinity_loss import build_affinity_targets_torch


NO_PHOTOMETRIC = {
    "horizontal_flip": False,
    "vertical_flip": False,
    "rotation90": False,
    "brightness": 0,
    "contrast": 0,
    "gamma": 0,
    "noise_probability": 0,
    "blur_probability": 0,
}


@pytest.fixture
def manual_sample(tmp_path):
    raw, target = tmp_path / "raw", tmp_path / "gt"
    raw.mkdir()
    target.mkdir()
    yy, xx = np.indices((12, 18))
    rgb = np.stack((xx * 11, yy * 17, xx * 3 + yy * 5), axis=-1).astype(np.uint8)
    instances = np.full((12, 18), 300, dtype=np.uint16)
    instances[:, :5] = 11
    instances[:, 12:] = 65535
    instances[3:5, 8:10] = 0
    valid = instances > 0
    original = valid & (xx % 2 == 0)
    filled = valid & ~original
    assert cv2.imwrite(str(raw / "sample.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (raw / "sample.json").write_text('{"shapes": []}', encoding="utf-8")
    np.savez_compressed(
        target / "sample_gt.npz", instance_map=instances,
        original_covered=original, filled=filled, residual_unknown=~valid,
    )
    (target / "sample_class.json").write_text(
        json.dumps({"11": 0, "300": 1, "65535": 0}), encoding="utf-8"
    )
    return {"raw": raw, "target": target, "rgb": rgb, "instances": instances,
            "original": original, "filled": filled}


def make_dataset(source, **kwargs):
    options = dict(image_size=8, affinity_grid=8, augment=True,
                   augmentation=NO_PHOTOMETRIC, manual_target_dir=source["target"])
    options.update(kwargs)
    return DirectDualHeadDataset(source["raw"], None, **options)


@pytest.mark.parametrize("spatial", [False, True])
def test_native_crop_keeps_rgb_ids_classes_and_unknown_aligned(manual_sample, monkeypatch, spatial):
    source = manual_sample
    augmentation = dict(NO_PHOTOMETRIC, horizontal_flip=spatial, rotation90=spatial)
    dataset = make_dataset(source, augmentation=augmentation,
                           native_crop={"enabled": True, "probability": 1, "size": 8})
    draws = iter([2, 8, 1] if spatial else [2, 8])
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "randint", lambda *args: next(draws))
    sample = dataset[0]

    def transformed(value):
        value = value[2:10, 8:16]
        return np.rot90(np.fliplr(value), 1).copy() if spatial else value.copy()

    expected_ids = transformed(source["instances"])
    expected_rgb = torch.from_numpy(transformed(source["rgb"])).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(sample["image"], expected_rgb, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(sample["semantic_instance_map"].numpy(), expected_ids)
    np.testing.assert_array_equal(sample["affinity_instance_map"].numpy(), expected_ids)
    assert set(torch.unique(sample["affinity_instance_map"]).tolist()) == {0, 300, 65535}
    expected_semantic = (expected_ids == 300).astype(np.float32)
    np.testing.assert_array_equal(sample["semantic_target"][0].numpy(), expected_semantic)
    for key in ("semantic_valid_content", "affinity_valid_content", "semantic_loss_weight"):
        np.testing.assert_array_equal(sample[key][0].numpy(), expected_ids > 0)
    full_boundary = instance_boundary_target(source["instances"], dilation=2)
    np.testing.assert_array_equal(sample["semantic_boundary"][0].numpy(), transformed(full_boundary))

    assert sample["sample_bbox"].tolist() == [8, 2, 16, 10]
    assert sample["sample_shape"].tolist() == [8, 8]
    assert sample["original_shape"].tolist() == [12, 18]
    assert sample["is_native_crop"].item() is True
    assert sample["uncovered_pixels"] == 4
    assert sample["original_covered_pixels"] == 30
    assert sample["filled_pixels"] == 30
    assert sample["uncovered_boundary_source"].item() is False

    _, edge_valid, uncovered = build_affinity_targets_torch(
        sample["affinity_instance_map"].unsqueeze(0),
        sample["affinity_valid_content"].unsqueeze(0),
        uncovered_as_boundary=sample["uncovered_boundary_source"].reshape(1),
        return_uncovered_mask=True,
    )
    assert not uncovered.any()
    assert not edge_valid[0, :, expected_ids == 0].any()
    # 横向连接只有两个端点都已知才可监督；窗口右缘不得连接到窗外。
    expected_pairs = (expected_ids[:, :-1] > 0) & (expected_ids[:, 1:] > 0)
    np.testing.assert_array_equal(edge_valid[0, 0, :, :-1].numpy(), expected_pairs)
    assert not edge_valid[0, 0, :, -1].any()


def test_semantic_boundary_is_derived_before_crop(manual_sample, monkeypatch):
    dataset = make_dataset(manual_sample, image_size=4, affinity_grid=4,
                           native_crop={"enabled": True, "probability": 1, "size": 4})
    draws = iter([6, 6])
    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "randint", lambda *args: next(draws))
    sample = dataset[0]
    ids_crop = manual_sample["instances"][6:10, 6:10]
    assert np.unique(ids_crop).tolist() == [300]
    assert not instance_boundary_target(ids_crop, dilation=2).any()
    expected = instance_boundary_target(manual_sample["instances"], dilation=2)[6:10, 6:10]
    assert expected[:, 0].all()  # 窗外真实界面膨胀进入首列，完整 GT 上下文必须保留。
    np.testing.assert_array_equal(sample["semantic_boundary"][0].numpy(), expected)


@pytest.mark.parametrize("native_crop", [None, {"enabled": False, "probability": 1, "size": 4},
                                        {"enabled": True, "probability": 0, "size": 4}])
def test_disabled_crop_preserves_whole_image_and_rng(manual_sample, monkeypatch, native_crop):
    dataset = make_dataset(manual_sample, native_crop=native_crop)
    expected_draw = np.random.RandomState(813).rand()
    seen = []

    def inspect_augmentation(sample):
        seen.append(np.random.rand())
        return sample

    monkeypatch.setattr(dataset, "_augment", inspect_augmentation)
    state = np.random.get_state()
    try:
        np.random.seed(813)
        sample = dataset[0]
    finally:
        np.random.set_state(state)
    assert seen == [expected_draw]
    assert sample["sample_bbox"].tolist() == [0, 0, 18, 12]
    assert sample["sample_shape"].tolist() == [12, 18]
    assert not sample["is_native_crop"].item()
    expected_image = torch.from_numpy(letterbox(manual_sample["rgb"], 8)[0]).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(sample["image"], expected_image, rtol=0, atol=0)
    reference = make_dataset(manual_sample, augment=False)[0]
    for key, value in reference.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, sample[key]), key
        else:
            assert value == sample[key], key


def test_validation_stays_whole_image_without_rng(manual_sample, monkeypatch):
    dataset = make_dataset(manual_sample, augment=False,
                           native_crop={"enabled": True, "probability": 1, "size": 4})

    def unexpected_random(*args):
        raise AssertionError("validation must not sample a crop")

    monkeypatch.setattr(np.random, "rand", unexpected_random)
    monkeypatch.setattr(np.random, "randint", unexpected_random)
    sample = dataset[0]
    assert sample["sample_bbox"].tolist() == [0, 0, 18, 12]
    assert sample["sample_shape"].tolist() == [12, 18]
    assert sample["content_shape"].tolist() == [5, 8]
    assert not sample["is_native_crop"].item()
    assert sample["uncovered_pixels"] == int((manual_sample["instances"] == 0).sum())


def test_crop_size_is_limited_by_short_image_side(manual_sample, monkeypatch):
    dataset = make_dataset(manual_sample,
                           native_crop={"enabled": True, "probability": 1, "size": 1024})
    bounds = []

    def last_position(low, high):
        bounds.append((low, high))
        return high - 1

    monkeypatch.setattr(np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(np.random, "randint", last_position)
    sample = dataset[0]
    assert bounds == [(0, 1), (0, 7)]
    assert sample["sample_shape"].tolist() == [12, 12]
    assert sample["sample_bbox"].tolist() == [6, 0, 18, 12]


@pytest.mark.parametrize("key,value", [
    ("probability", -0.1), ("probability", 1.1), ("probability", np.nan),
    ("probability", np.inf), ("probability", -np.inf),
    ("size", 0), ("size", -1), ("size", 2.5), ("size", np.nan), ("size", np.inf),
])
def test_native_crop_configuration_rejects_invalid_values(manual_sample, key, value):
    with pytest.raises(ValueError, match=f"native_crop {key}"):
        make_dataset(manual_sample, native_crop={key: value})


def test_enabled_native_crop_requires_manual_targets(manual_sample):
    with pytest.raises(ValueError, match="native_crop requires manual_target_dir"):
        make_dataset(manual_sample, manual_target_dir=None, native_crop={"enabled": True})
