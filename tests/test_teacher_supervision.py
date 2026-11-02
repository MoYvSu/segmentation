# -*- coding: utf-8 -*-
"""教师监督的严格来源、未知像素梯度及早期训练接线 CPU 合同。"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from data.teacher_supervision import load_teacher_instance_source, teacher_targets, teacher_boundary_targets
from data.dataset import BoundaryDataset
from data.dataset_semi import LabeledDataset
from data.offset_geometry_dataset import OffsetGeometryDataset
from utils.loss import BoundaryLoss


def cohort(tmp_path, count=4):
    raw, targets = tmp_path / "raw", tmp_path / "teacher"
    raw.mkdir(); targets.mkdir()
    ids = np.ones((12, 16), np.uint16)
    ids[:, 8:] = 2
    ids[4:6, 3:5] = 0
    for index in range(count):
        stem = f"train_{index:03d}"
        cv2.imwrite(str(raw / (stem + ".png")), np.full((12, 16, 3), 127, np.uint8))
        # 教师路径不得解析旧 JSON，甚至没有旧净化NPZ。
        (raw / (stem + ".json")).write_text("old labels deliberately unreadable", encoding="utf-8")
        np.savez_compressed(targets / (stem + "_gt.npz"), instance_map=ids,
                            teacher_known=ids > 0, teacher_unknown=ids == 0)
        (targets / (stem + "_class.json")).write_text(json.dumps({"1": 1, "2": 0}), encoding="utf-8")
    return raw, targets, ids


def test_teacher_loader_strict_schema_and_no_manual_provenance(tmp_path):
    raw, targets, ids = cohort(tmp_path)
    path = targets / "train_000_gt.npz"
    loaded, classes, known, unknown, provenance = load_teacher_instance_source(path, ids.shape)
    assert loaded.dtype == np.int32 and np.array_equal(loaded, ids)
    assert classes == {1: 1, 2: 0}
    assert np.array_equal(known, ~unknown)
    assert provenance["source_kind"] == "model_teacher"
    np.savez(path, instance_map=ids, teacher_known=ids > 0, teacher_unknown=ids == 0,
             original_covered=ids > 0)
    with pytest.raises(ValueError, match="manual completion"):
        load_teacher_instance_source(path)


@pytest.mark.parametrize("defect", ["known", "unknown", "dtype", "classes"])
def test_teacher_loader_rejects_inconsistent_targets(tmp_path, defect):
    _, targets, ids = cohort(tmp_path)
    path = targets / "train_000_gt.npz"
    known, unknown = ids > 0, ids == 0
    if defect == "known":
        known[0, 0] = False
    if defect == "unknown":
        unknown[0, 0] = True
    if defect == "dtype":
        ids = ids.astype(np.int32)
    if defect == "classes":
        (targets / "train_000_class.json").write_text('{"1":1,"2":0,"3":0}', encoding="utf-8")
    np.savez(path, instance_map=ids, teacher_known=known, teacher_unknown=unknown)
    with pytest.raises(ValueError):
        load_teacher_instance_source(path)


@pytest.mark.parametrize("dataset_cls", [BoundaryDataset, LabeledDataset])
def test_both_stage_datasets_ignore_old_gt_and_reflect_labels(tmp_path, dataset_cls):
    raw, targets, _ = cohort(tmp_path)
    dataset = dataset_cls(str(raw), str(tmp_path / "no_old_gt"), teacher_gt_dir=str(targets),
                          image_size=16, split="val", train_ratio=.5)
    sample = dataset[0]
    assert sample["target"].shape == (3, 16, 16)
    assert sample["target"][0, -1].eq(-1).all()  # 图像reflect区域无标签监督。
    assert sample["target"][0, 4:6, 3:5].eq(-1).all()
    assert sample["target"][1, 4:6, 3:5].eq(-1).all()
    assert sample["target"][0, 2, 2] == 1
    assert sample["target"][0, 2, 12] == 0
    if dataset_cls is LabeledDataset:
        assert "instance_map" in sample
    assert dataset.instance_centers == {}


def test_teacher_edges_ignore_unknown_contacts_and_use_instance_separation():
    ids = np.ones((9, 12), np.int32)
    ids[:, 6:] = 2
    ids[4, 2] = 0
    semantic, boundary, _, _ = teacher_targets(ids, {1: 0, 2: 0})
    assert boundary[3:6, 1:4].min() == -1
    assert boundary[2, 5:7].tolist() == [1., 1.]  # 同类别不同实例仍有边界。
    assert semantic[4, 2] == -1
    assert not np.any(boundary[1:-1, 1:-1] > 1)


def test_unknown_has_zero_gradient_for_semantic_boundary_and_center():
    ids = np.ones((12, 16), np.int32)
    ids[:, 8:] = 2
    ids[4:6, 3:5] = 0
    semantic, boundary, _, center = teacher_targets(ids, {1: 1, 2: 0}, center_sigma=1.)
    target = torch.from_numpy(np.stack([semantic, boundary, center]))[None]
    logits = torch.randn_like(target, requires_grad=True)
    criterion = BoundaryLoss(seg_dice_weight=.3, semantic_tversky_weight=.2,
                             peak_weight=.4, hard_negative_weight=.3, center_weight=.2)
    loss, _, _ = criterion(logits, target, torch.ones((1, 1, 12, 16)))
    loss.backward()
    ignored = target < 0
    assert torch.count_nonzero(logits.grad[ignored]) == 0
    assert torch.count_nonzero(logits.grad[~ignored]) > 0
    changed = logits.detach().clone()
    changed[ignored] += 50
    other, _, _ = criterion(changed, target, torch.ones((1, 1, 12, 16)))
    assert torch.equal(loss.detach(), other)


def test_legacy_fully_valid_loss_keeps_original_reduction():
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 8, 9)
    target = torch.randint(0, 2, logits.shape).float()
    weight = torch.rand(2, 1, 8, 9) * 8
    criterion = BoundaryLoss(alpha_boundary=3., gamma=2., alpha_focal=.75)
    actual, _, _ = criterion(logits, target, weight)
    semantic = F.binary_cross_entropy_with_logits(logits[:, 0], target[:, 0])
    bce = F.binary_cross_entropy_with_logits(logits[:, 1], target[:, 1], reduction="none")
    focal = (.75 * target[:, 1] + .25 * (1-target[:, 1])) * (1-torch.exp(-bce))**2
    expected = semantic + 3 * (focal * weight[:, 0].clamp(1, 4) * bce).mean()
    assert torch.equal(actual, expected)


def test_v_geometry_train_and_val_both_read_teacher_without_labelme(tmp_path):
    from train_affinity_geometry_g1 import build_manual_geometry_dataset
    raw, targets, _ = cohort(tmp_path)
    config = {"paths": {"project_root": str(tmp_path)}}
    cfg = {"manual_teacher_gt_dir": str(targets)}
    for training in (True, False):
        dataset = build_manual_geometry_dataset(config, cfg, str(raw), ["train_000"],
            training=training, image_size=16, output_grid=16)
        sample = dataset[0]
        assert sample["instance_map"][4:6, 3:5].eq(0).all()
        assert not sample["uncovered_boundary_source"]
        assert dataset.teacher_gt_dir == targets


def test_teacher_joint_dataloader_supports_trainable_both_heads(tmp_path):
    from train_stage2 import build_dataloaders
    raw, targets, _ = cohort(tmp_path)
    config = dict(paths={"project_root": str(tmp_path), "raw_data_dir": str(raw)},
        data={"image_size": 16, "crop_size": 8, "teacher_gt_dir": str(targets),
              "seed": 42, "train_ratio": .5, "num_workers": 0, "batch_size": 1},
        train={"seed": 20261101}, boundary={"gt_dir": "no_old_gt"},
        semi_supervised={"use_unlabeled": False, "unlabeled_dir": "absent",
                         "freeze": {"seg_branch": False, "boundary_branch": False}})
    train, unlabeled, val = build_dataloaders(config)
    assert unlabeled is None and len(train.dataset) == len(val.dataset) == 2
    batch = next(iter(train))
    assert batch["instance_map"].shape == (1, 8, 8)
    assert next(iter(val))["target"][:, 0, -1].eq(-1).all()


def test_explicit_single_pixel_separator_is_boundary_without_filling_ids():
    ids = np.ones((13, 17), np.int32)
    ids[:, 9:] = 2
    ids[:, 8] = 0
    before = ids.copy()
    old, old_valid, _ = teacher_boundary_targets(ids)
    boundary, valid, meta = teacher_boundary_targets(ids, policy="explicit_separators_v1")
    assert not old.any()
    assert not old_valid[:, 8].any()
    assert boundary[:, 8].all() and valid[1:-1, 8].all()
    assert meta["separator_pixels"] == 13 and meta["valid_boundary_pixels"] == 11
    assert meta["ID0_semantic"] == meta["ID0_affinity_endpoint"] == "ignore"
    assert np.array_equal(ids, before)
    from utils.affinity_loss import build_affinity_targets_torch
    _, pair_valid = build_affinity_targets_torch(torch.from_numpy(ids)[None],
        torch.ones((1, 1, *ids.shape), dtype=torch.bool), offsets=[(0, 1)])
    assert not pair_valid[0, 0, :, 7:9].any()  # 两个端点涉及0的pair依然ignore。


@pytest.mark.parametrize("width,same_id", [(2, False), (3, False), (1, True)])
def test_wide_unknown_or_same_id_gap_never_becomes_separator(width, same_id):
    ids = np.ones((13, 17), np.int32)
    ids[:, 8+width:] = 1 if same_id else 2
    ids[:, 8:8+width] = 0
    boundary, valid, meta = teacher_boundary_targets(ids, policy="explicit_separators_v1")
    assert meta["separator_pixels"] == 0 and not boundary.any()
    assert not valid[:, 8:8+width].any()


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("flip", [False, True])
def test_separator_target_and_domain_are_d4_equivariant(rotation, flip):
    ids = np.ones((17, 21), np.int32)
    ids[:, 11:] = 2
    ids[:, 10] = 0
    ids[7:10, 2:5] = 0
    def transform(array):
        value = np.rot90(array, rotation)
        return np.fliplr(value) if flip else value
    for margin in (1, 4):
        boundary, valid, meta = teacher_boundary_targets(ids, margin, "explicit_separators_v1")
        actual, actual_valid, actual_meta = teacher_boundary_targets(transform(ids), margin, "explicit_separators_v1")
        assert np.array_equal(actual, transform(boundary))
        assert np.array_equal(actual_valid, transform(valid))
        assert actual_meta == meta
        assert not actual_valid[[0, -1], :].any() and not actual_valid[:, [0, -1]].any()


def test_separator_semantic_gradient_zero_boundary_gradient_positive_supervision():
    ids = np.ones((13, 17), np.int32)
    ids[:, 9:] = 2
    ids[:, 8] = 0
    semantic, boundary, _, center = teacher_targets(ids, {1: 0, 2: 1},
        boundary_policy="explicit_separators_v1")
    assert np.all(semantic[:, 8] == -1) and np.all(center[:, 8] == -1)
    assert np.all(boundary[1:-1, 8] == 1)
    target = torch.from_numpy(np.stack([semantic, boundary, center]))[None]
    logits = torch.zeros_like(target, requires_grad=True)
    loss, _, _ = BoundaryLoss(center_weight=.2)(logits, target)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 0, :, 8]) == 0
    assert torch.count_nonzero(logits.grad[0, 2, :, 8]) == 0
    assert (logits.grad[0, 1, 1:-1, 8] < 0).all()
