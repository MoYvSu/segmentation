# -*- coding: utf-8 -*-
"""新语义监督的来源、空间对齐、ignore 梯度及配对实验隔离。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.dataset_semi import LabeledDataset
from data.semantic_targets import load_completed_semantic_source
from utils.config import load_config
from utils.loss import BoundaryLoss
from utils.progressive_aug import ProgressiveAppearanceAug


@pytest.fixture
def semantic_sample(tmp_path):
    raw, old, new = (tmp_path / name for name in ("raw", "old", "new"))
    for path in (raw, old, new):
        path.mkdir()
    image = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    assert cv2.imwrite(str(raw / "sample.png"), image)
    # 故意让旧监督与新实例的边界、类别不一致，捕捉混用旧 JSON 的回归。
    annotation = {"shapes": [{"label": "pearlite", "shape_type": "polygon",
                             "points": [[0, 0], [63, 0], [63, 47], [0, 47]]}]}
    (raw / "sample.json").write_text(json.dumps(annotation), encoding="utf-8")
    np.savez(old / "sample_gt.npz", semantic=np.zeros((48, 64), np.uint8),
             boundary=np.ones((48, 64), np.uint8))
    labels = np.ones((48, 64), np.uint16)
    labels[:, 32:] = 2
    labels[10:15, 5:10] = 0
    np.savez(new / "sample_gt.npz", instance_map=labels, residual_unknown=labels == 0)
    (new / "sample_class.json").write_text('{"1": 1, "2": 0}', encoding="utf-8")
    return raw, old, new


def make_dataset(paths, *, completed=True, **kwargs):
    raw, old, new = paths
    return LabeledDataset(str(raw), str(old), image_size=64, split="train",
                          include_instance_map=True,
                          completed_gt_dir=str(new) if completed else None, **kwargs)


def assert_canonical(item):
    labels = item["instance_map"]
    semantic = item["target"][0]
    assert torch.equal(semantic < 0, labels == 0)
    assert torch.all(semantic[labels == 1] == 1)
    assert torch.all(semantic[labels == 2] == 0)
    assert not item["target"][1][labels == 0].any()


def test_completed_source_replaces_every_supervised_target(semantic_sample):
    item = make_dataset(semantic_sample)[0]
    old = make_dataset(semantic_sample, completed=False)[0]
    assert_canonical(item)
    assert (item["target"][0, 48:] == -1).all()
    assert not item["instance_map"][48:].any()
    assert item["target"][1, 20, 31:33].all()
    assert not item["target"][1, 20, 15]
    assert torch.equal(item["image"], old["image"])
    assert not torch.equal(item["target"], old["target"])


def test_crop_flip_rotation_preserve_unknown_and_instance_class(semantic_sample):
    dataset = make_dataset(semantic_sample, augment=True, crop_size=40,
                           augment_config={"horizontal_flip": True, "vertical_flip": True,
                                           "rotation": True})
    for seed in range(6):
        np.random.seed(seed)
        item = dataset[0]
        assert item["image"].shape == (3, 40, 40)
        assert_canonical(item)


def test_missing_completed_metadata_never_silently_uses_old_gt(semantic_sample):
    raw, old, new = semantic_sample
    (new / "sample_class.json").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete semantic GT"):
        make_dataset(semantic_sample)
    (new / "sample_class.json").write_text('{"1": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="every instance"):
        load_completed_semantic_source(new / "sample_gt.npz", (48, 64))


def combined_criterion():
    return BoundaryLoss(freeze_boundary=True, center_weight=0,
                        seg_dice_weight=0.3, semantic_tversky_weight=0.15,
                        semantic_instance_weight=0.75, semantic_core_radius=0,
                        semantic_core_min_pixels=1, semantic_instance_pool_weight=0.5)


def test_ignored_pixels_change_neither_loss_nor_valid_gradients():
    torch.manual_seed(9)
    prediction = torch.randn(1, 2, 4, 4, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 0, :, :2] = 1
    labels = torch.ones(1, 4, 4, dtype=torch.long)
    labels[:, :, 2:] = 2
    criterion = combined_criterion()
    loss = criterion(prediction, target, instance_map=labels)[0]
    loss.backward()

    extended = torch.cat((prediction.detach(), torch.full((1, 2, 4, 3), 12.0)), -1)
    extended.requires_grad_()
    extended_target = torch.cat((target, torch.zeros(1, 2, 4, 3)), -1)
    extended_target[:, 0, :, 4:] = -1
    # 即便调用方在未知区误带正 ID，也不能让 instance loss 产生梯度。
    extended_labels = torch.cat((labels, torch.ones(1, 4, 3, dtype=torch.long)), -1)
    extended_loss = criterion(extended, extended_target, instance_map=extended_labels)[0]
    extended_loss.backward()
    assert torch.allclose(loss, extended_loss)
    assert torch.allclose(prediction.grad, extended.grad[..., :4])
    assert not extended.grad[..., 4:].any()


def test_all_unknown_is_differentiable_zero():
    prediction = torch.randn(1, 2, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 0] = -1
    loss = combined_criterion()(prediction, target,
                                instance_map=torch.ones(1, 8, 8, dtype=torch.long))[0]
    loss.backward()
    assert loss.item() == 0.0
    assert not prediction.grad.any()


def test_explicit_zero_weight_matches_unknown_for_all_semantic_losses():
    torch.manual_seed(11)
    prediction = torch.randn(1, 2, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[:, 0, :4] = 1
    weight = torch.ones(1, 1, 8, 8)
    weight[:, :, 2:6] = 0
    labels = torch.ones(1, 8, 8, dtype=torch.long)
    criterion = combined_criterion()
    weighted = criterion(prediction, target, instance_map=labels, seg_weight=weight)[0]
    target[:, 0, 2:6] = -1
    ignored = criterion(prediction, target, instance_map=labels)[0]
    assert torch.allclose(weighted, ignored)
    weighted.backward()
    assert not prediction.grad[:, 0, 2:6].any()


def test_large_half_precision_grid_keeps_finite_pixel_normalization():
    prediction = torch.zeros(1, 2, 512, 512, dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, 2, 512, 512)
    target[:, 0, :256] = 1
    criterion = BoundaryLoss(freeze_boundary=True, center_weight=0, seg_dice_weight=0.3)
    loss = criterion(prediction, target)[0]
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0.8
    assert torch.isfinite(prediction.grad).all() and prediction.grad.any()


def test_pair_diff_is_only_training_gt_and_output_paths(semantic_sample):
    root = Path(__file__).resolve().parents[1]
    control = load_config(str(root / "config/train/stage2_semantic_gt_control20.yaml"))
    new = load_config(str(root / "config/train/stage2_semantic_gt_new20.yaml"))
    assert control["data"]["semantic_train_completed_gt_dir"] == ""
    assert new["data"]["semantic_train_completed_gt_dir"]
    assert new["semi_supervised"]["selection_metric"] == "val_loss"
    new["data"]["semantic_train_completed_gt_dir"] = ""
    for key in ("output_dir",):
        new["semi_supervised"][key] = control["semi_supervised"][key]
        new["semi_supervised"]["monitor"][key] = control["semi_supervised"]["monitor"][key]
    assert new == control
    augmentor = ProgressiveAppearanceAug(control["progressive_aug"], torch.device("cpu"))
    augmentor.set_epoch(10)
    old_sample = make_dataset(semantic_sample, completed=False)[0]
    new_sample = make_dataset(semantic_sample)[0]
    outputs = []
    for sample in (old_sample, new_sample):
        torch.manual_seed(5)
        outputs.append(augmentor(sample["image"].unsqueeze(0),
                                 boundary_targets=sample["target"][1].unsqueeze(0)))
    assert torch.equal(*outputs)
