# -*- coding: utf-8 -*-
"""补缝 GT 的训练隔离、unknown 忽略和增强后的真实跨界监督。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.affinity_geometry_augmentation import crop_affinity_sample
from data.offset_geometry_augmentation import augment_cached_geometry_sample
from data.offset_geometry_dataset import OffsetGeometryDataset, load_completed_instance_map
from train_affinity_geometry_g1 import build_manual_geometry_dataset
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config


@pytest.fixture
def completed_sample(tmp_path):
    raw, gt = tmp_path / "raw", tmp_path / "completed"
    raw.mkdir()
    gt.mkdir()
    assert cv2.imwrite(str(raw / "sample.png"), np.full((48, 64, 3), 120, np.uint8))
    annotation = {"imageHeight": 48, "imageWidth": 64, "shapes": [
        {"label": "ferrite", "shape_type": "polygon",
         "points": [[0, 0], [29, 0], [29, 47], [0, 47]]},
        {"label": "pearlite", "shape_type": "polygon",
         "points": [[34, 0], [63, 0], [63, 47], [34, 47]]},
    ]}
    (raw / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")
    instances = np.ones((48, 64), dtype=np.uint16)
    instances[:, 32:] = 2
    instances[10:15, 5:10] = 0
    path = gt / "sample_gt.npz"
    np.savez_compressed(path, instance_map=instances, residual_unknown=instances == 0)
    return raw, gt, path, instances


def test_training_uses_completed_map_but_validation_matches_legacy(completed_sample):
    raw, gt, _, instances = completed_sample
    config = {"paths": {"project_root": str(raw.parent)}}
    cfg = {"manual_train_completed_gt_dir": gt.name}
    kwargs = {"image_size": 64, "output_grid": 64, "cache_in_memory": True}
    train = build_manual_geometry_dataset(
        config, cfg, raw, ["sample"], training=True, **kwargs,
    )[0]
    val = build_manual_geometry_dataset(
        config, cfg, raw, ["sample"], training=False, **kwargs,
    )[0]
    legacy = OffsetGeometryDataset(raw, **kwargs)[0]
    assert np.array_equal(train["instance_map"][:48].numpy(), instances)
    assert not train["instance_map"][48:].any()
    assert not train["valid_content"][:, 48:].any()
    assert torch.equal(train["foreground"], (train["instance_map"] > 0).unsqueeze(0))
    assert train["uncovered_pixels"] == 25
    assert not train["uncovered_boundary_source"] and val["uncovered_boundary_source"]
    assert torch.equal(train["image"], val["image"])
    assert not torch.equal(train["instance_map"], val["instance_map"])
    for key, value in legacy.items():
        if torch.is_tensor(value):
            assert torch.equal(value, val[key]), key
        else:
            assert value == val[key], key


def test_unknown_and_padding_ignored_after_flip_rotation_and_crop(completed_sample):
    raw, gt, _, _ = completed_sample
    sample = OffsetGeometryDataset(raw, image_size=64, output_grid=64, completed_gt_dir=gt)[0]
    transformed = augment_cached_geometry_sample(sample, hflip=True, vflip=False, rotation_k=1)
    cropped = crop_affinity_sample(transformed, 0.9, 0.5, 0.5)
    for item in (sample, transformed, cropped):
        labels = item["instance_map"]
        target, valid, uncovered = build_affinity_targets_torch(
            labels.unsqueeze(0), item["valid_content"].unsqueeze(0),
            offsets=[(0, 1)],
            # 即便调用方开启历史缝隙负连接模式，新 GT 的 unknown 仍不能参与。
            uncovered_as_boundary=item["uncovered_boundary_source"].reshape(1),
            return_uncovered_mask=True,
        )
        expected = ((labels[:, :-1] > 0) & (labels[:, 1:] > 0)
                    & item["valid_content"][0, :, :-1] & item["valid_content"][0, :, 1:])
        assert torch.equal(valid[0, 0, :, :-1], expected)
        assert not valid[0, 0, :, -1].any() and not uncovered.any()
        assert torch.equal(target[0, 0, :, :-1].bool(), expected & (labels[:, :-1] == labels[:, 1:]))
    target, valid = build_affinity_targets_torch(
        sample["instance_map"].unsqueeze(0), sample["valid_content"].unsqueeze(0), offsets=[(0, 1)],
    )
    assert valid[0, 0, 20, 31] and target[0, 0, 20, 31] == 0
    assert valid[0, 0, 20, 30] and target[0, 0, 20, 30] == 1


def test_missing_completed_gt_never_falls_back_to_old_annotation(completed_sample):
    raw, gt, path, _ = completed_sample
    path.unlink()
    with pytest.raises(FileNotFoundError, match="completed GT missing"):
        OffsetGeometryDataset(raw, completed_gt_dir=gt)


@pytest.mark.parametrize("invalid", ["shape", "fractional_ids", "unknown"])
def test_inconsistent_completed_map_rejected(completed_sample, invalid):
    _, _, path, instances = completed_sample
    unknown = instances == 0
    if invalid == "shape":
        instances = instances[:-1]
    elif invalid == "fractional_ids":
        instances = instances.astype(np.float32)
    else:
        unknown[20, 20] = True
    np.savez_compressed(path, instance_map=instances, residual_unknown=unknown)
    with pytest.raises(ValueError, match="completed GT"):
        load_completed_instance_map(path, (48, 64))


def test_newgt_config_changes_only_training_target_and_output():
    root = Path(__file__).resolve().parents[1]
    old = load_config(str(root / "config/train/affinity_geometry_g2_direct_long_oldgt.yaml"))
    new = load_config(str(root / "config/train/affinity_geometry_g2_direct_long_newgt.yaml"))
    assert new["affinity_geometry_g1"].pop("manual_train_completed_gt_dir").endswith("radius_8/derived_maps")
    assert new["affinity_geometry_g1"].pop("output_dir") == "outputs/20260916_g2_long_newgt/training"
    old["affinity_geometry_g1"].pop("output_dir")
    assert old == new
