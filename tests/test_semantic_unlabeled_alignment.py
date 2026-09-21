# -*- coding: utf-8 -*-
"""教师/学生共享坐标回归：非对称图、缓存目标、遮挡与实际一致性损失。"""

from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.dataset_semi import UnlabeledDataset
from utils.config import load_config
from utils.loss_semi import compute_unsupervised_loss


@pytest.fixture
def unlabeled_sample(tmp_path):
    raw = tmp_path / "raw"
    cache = tmp_path / "cache"
    raw.mkdir()
    cache.mkdir()
    # 每个位置有不同值，能发现翻转或旋转漏做、反向做、重复做。
    channel = 20 + np.arange(64, dtype=np.uint8).reshape(8, 8) * 3
    image = np.stack((channel, 255 - channel, channel // 2), axis=-1)
    assert cv2.imwrite(str(raw / "sample.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(cache / "boundary_probs.npy", (channel.astype(np.float32) / 255)[None])
    (cache / "names.txt").write_text("sample\n", encoding="utf-8")
    return raw, cache, image


def expected_transform(array, params):
    # 直接声明变换次序，避免使用待测 helper 生成期望值。
    if params["hflip"]:
        array = np.flip(array, axis=1)
    if params["vflip"]:
        array = np.flip(array, axis=0)
    return np.ascontiguousarray(np.rot90(array, params["rot_k"]))


def make_dataset(sample, monkeypatch, params, **kwargs):
    raw, cache, _ = sample
    dataset = UnlabeledDataset(
        str(raw), image_size=8, boundary_cache_dir=str(cache), **kwargs
    )
    monkeypatch.setattr(dataset, "_sample_spatial_params", lambda: params)
    return dataset


TRANSFORMS = list(product((False, True), (False, True), range(4)))


@pytest.mark.parametrize("hflip,vflip,rot_k", TRANSFORMS)
def test_teacher_student_and_cached_boundary_share_one_transform(
    unlabeled_sample, monkeypatch, hflip, vflip, rot_k
):
    params = dict(hflip=hflip, vflip=vflip, rot_k=rot_k)
    dataset = make_dataset(unlabeled_sample, monkeypatch, params,
                           shared_spatial_augmentation=True, enable_appearance_aug=False)
    item = dataset[0]
    expected = torch.from_numpy(expected_transform(unlabeled_sample[2], params))
    expected = expected.permute(2, 0, 1).float() / 255
    assert torch.equal(item["img_weak"], expected)
    assert torch.equal(item["img_strong"], expected)
    assert torch.equal(item["boundary_target"], expected[:1])
    assert not item["patch_mask"].any()


def test_appearance_and_patch_mask_affect_only_student_in_final_coordinates(
    unlabeled_sample, monkeypatch
):
    params = dict(hflip=True, vflip=False, rot_k=1)
    dataset = make_dataset(unlabeled_sample, monkeypatch, params,
                           shared_spatial_augmentation=True, enable_patch_mask=True)
    # 外观增强替身故意原地修改，检验两路不共享可写数组。
    def appearance(image):
        image //= 2
        return image

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 5:7] = 1
    monkeypatch.setattr(dataset, "_apply_appearance_aug", appearance)
    monkeypatch.setattr(dataset, "_generate_patch_mask", lambda: mask.copy())
    item = dataset[0]
    clean = expected_transform(unlabeled_sample[2], params)
    student = clean // 2
    student[mask > 0] = student.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    assert torch.equal(item["img_weak"], torch.from_numpy(clean).permute(2, 0, 1).float() / 255)
    assert torch.equal(item["img_strong"], torch.from_numpy(student).permute(2, 0, 1).float() / 255)
    assert torch.equal(item["patch_mask"][0], torch.from_numpy(mask).float())
    assert torch.equal(item["boundary_target"], item["img_weak"][:1])


class PointwiseEquivariantModel(torch.nn.Module):
    """逐像素恒等概率模型，对任意翻转和90度旋转严格等变。"""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, image, output_size=None):
        logit = torch.logit(image[:, :1].clamp(0.01, 0.99)) + self.bias
        return torch.cat((logit, logit), dim=1)


@pytest.mark.parametrize("params", [
    dict(hflip=True, vflip=False, rot_k=0),
    dict(hflip=False, vflip=True, rot_k=0),
    dict(hflip=False, vflip=False, rot_k=1),
    dict(hflip=True, vflip=False, rot_k=3),
])
def test_real_consistency_loss_stops_penalizing_equivariant_predictions(
    unlabeled_sample, monkeypatch, params
):
    losses = []
    for aligned in (False, True):
        dataset = make_dataset(unlabeled_sample, monkeypatch, params,
                               shared_spatial_augmentation=aligned,
                               enable_appearance_aug=False)
        item = dataset[0]
        model = PointwiseEquivariantModel()
        loss, _, _ = compute_unsupervised_loss(
            model, item["img_weak"][None], item["img_strong"][None],
            item["patch_mask"][None], teacher_model=PointwiseEquivariantModel(),
            freeze_boundary=True, seg_sharpen_temperature=1.0,
            seg_confidence_threshold=0.5,
        )
        loss.backward()
        assert torch.isfinite(model.bias.grad)
        losses.append(loss.item())
    assert losses[0] > 0.001  # 旧实现把坐标错位错误地解释为预测不一致。
    assert losses[1] == pytest.approx(0, abs=1e-12)


def test_legacy_config_and_student_randomness_remain_reproducible(unlabeled_sample):
    raw, cache, image = unlabeled_sample
    legacy = UnlabeledDataset(str(raw), image_size=8, boundary_cache_dir=str(cache),
                               enable_patch_mask=True, patch_mask_size=2)
    aligned = UnlabeledDataset(str(raw), image_size=8, boundary_cache_dir=str(cache),
                                enable_patch_mask=True, patch_mask_size=2,
                                shared_spatial_augmentation=True)
    for seed in range(4):
        np.random.seed(seed)
        previous = legacy[0]
        np.random.seed(seed)
        corrected = aligned[0]
        assert torch.equal(previous["img_weak"], torch.from_numpy(image).permute(2, 0, 1).float() / 255)
        for key in ("img_strong", "patch_mask", "boundary_target"):
            assert torch.equal(previous[key], corrected[key])


def test_aligned_config_changes_only_spatial_mode_and_output_paths():
    root = Path(__file__).resolve().parents[1]
    previous = load_config(str(root / "config/train/stage2_semantic_gt_new20.yaml"))
    aligned = load_config(str(root / "config/train/stage2_semantic_gt_aligned20.yaml"))
    assert aligned["semi_supervised"].pop("shared_spatial_augmentation") is True
    assert not previous["semi_supervised"].get("shared_spatial_augmentation", False)
    aligned["semi_supervised"]["output_dir"] = previous["semi_supervised"]["output_dir"]
    aligned["semi_supervised"]["monitor"]["output_dir"] = previous["semi_supervised"]["monitor"]["output_dir"]
    assert aligned == previous
