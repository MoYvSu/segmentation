# -*- coding: utf-8 -*-
"""后端适配的数据同源、配对退化和 ignore 契约。"""

import json
import random

import cv2
import numpy as np
import pytest
import torch

from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.direct_dual_head_dataset import _spatial_transform
from data.semantic_targets import semantic_targets_from_instances
from utils.affinity_loss import build_affinity_targets_torch


def cohort(tmp_path):
    raw, completed = tmp_path / "raw", tmp_path / "completed"
    raw.mkdir()
    completed.mkdir()
    yy, xx = np.mgrid[:24, :32]
    image = np.repeat((50 + xx * 3 + yy)[..., None], 3, axis=2).astype(np.uint8)
    cv2.imwrite(str(raw / "sample.png"), image)
    # 故意不可解析的旧 LabelMe：只用于源图身份，不能读取旧几何监督。
    (raw / "sample.json").write_text("not a target", encoding="utf-8")
    instances = np.ones((24, 32), dtype=np.uint16)
    instances[:, 16:] = 2
    instances[4:10, 4:10] = 0
    np.savez(completed / "sample_gt.npz", instance_map=instances, residual_unknown=instances == 0)
    (completed / "sample_class.json").write_text(json.dumps({1: 1, 2: 0}), encoding="utf-8")
    return CanonicalBackendDataset(raw, completed, image_size=32, affinity_grid=16)


def degradation(profile=(0.2, 0.8, 0, 0)):
    return {"profile_probabilities": list(profile), "blur_sigma": [0.4, 5.2],
            "resample_probability": 0.0, "resample_scale": [0.5, 0.9],
            "spatial_blur": {"enabled": True, "probability": 0.5,
                             "grid_size": [3, 5], "strength": 1.0, "levels": 8}}


def test_canonical_targets_only_and_unknown_affinity_ignored(tmp_path):
    dataset = cohort(tmp_path)
    sample = dataset[0]
    ids = sample["semantic_instance_map"]
    assert torch.equal(sample["semantic_target"][0] < 0, ids == 0)
    assert torch.all(sample["semantic_target"][0][ids == 1] == 1)
    assert torch.all(sample["semantic_target"][0][ids == 2] == 0)
    expected_sem, expected_boundary = semantic_targets_from_instances(ids.numpy(), np.array([-1, 1, 0]))
    assert torch.equal(sample["semantic_boundary"][0], torch.from_numpy(expected_boundary))
    assert not sample["semantic_valid_content"][:, 24:].any()
    assert not sample["valid_content"][:, 12:].any()
    assert not sample["uncovered_boundary_source"]
    affinity = sample["instance_map"]
    assert torch.equal(affinity, ids[::2, ::2])
    _, valid, unknown = build_affinity_targets_torch(
        affinity[None], sample["valid_content"][None],
        uncovered_as_boundary=sample["uncovered_boundary_source"][None], return_uncovered_mask=True
    )
    assert not unknown.any()
    assert not valid[0, :, affinity == 0].any()


def test_missing_completed_target_cannot_silently_reduce_cohort(tmp_path):
    dataset = cohort(tmp_path)
    (dataset.completed_gt_dir / "sample_class.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete canonical"):
        CanonicalBackendDataset(dataset.data_dir, dataset.completed_gt_dir)


def test_paired_draws_reproduce_without_global_rng_and_change_by_epoch(tmp_path):
    base = cohort(tmp_path)
    noise = {"enabled": True, "probability": 1.0, "sigma": [0.002, 0.006]}
    left = PairedDegradationDataset(base, degradation((0, 1, 0, 0)), noise, 41, repeats=3)
    right = PairedDegradationDataset(base, degradation((0, 1, 0, 0)), noise, 41, repeats=3)
    np.random.seed(23)
    torch.manual_seed(29)
    random.seed(31)
    numpy_state, torch_state, python_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
    a, b = left[1], right[1]
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    for key in a:
        if torch.is_tensor(a[key]):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key
    assert not torch.equal(left[0]["image"], a["image"])
    left.set_epoch(1)
    right.set_epoch(1)
    assert not torch.equal(left[1]["image"], a["image"])
    assert torch.equal(left[1]["image"], right[1]["image"])


def test_geometry_transforms_every_target_and_reflect_padding(tmp_path):
    base = cohort(tmp_path)
    paired = PairedDegradationDataset(base, degradation((1, 0, 0, 0)), {"enabled": False}, 18, repeats=8)
    original = base[0]
    for item in paired:
        for key in paired._SPATIAL_KEYS:
            expected = _spatial_transform(original[key], item["horizontal_flip"],
                                          item["vertical_flip"], item["rotation_k"])
            assert torch.equal(item[key], expected), key
        assert torch.equal(item["semantic_target"][0] < 0, item["semantic_instance_map"] == 0)
    plain = PairedDegradationDataset(base, degradation(), {"enabled": True}, 18, augment=False)[0]
    assert torch.equal(plain["image"], original["image"])
    assert plain["profile"] == "identity" and plain["noise_sigma"] == 0


def test_noise_off_is_exact_and_noise_is_achromatic_and_blur_only(tmp_path):
    base = cohort(tmp_path)
    blur = degradation((0, 1, 0, 0))
    off = PairedDegradationDataset(base, blur, {"enabled": False}, 44)[0]
    zero = PairedDegradationDataset(base, blur, {"enabled": True, "probability": 0}, 44)[0]
    noisy = PairedDegradationDataset(base, blur, {"enabled": True, "probability": 1, "sigma": [0.006, 0.006]}, 44)[0]
    assert torch.equal(off["image"], zero["image"])
    assert off["noise_sigma"] == zero["noise_sigma"] == 0
    assert not torch.equal(off["image"], noisy["image"])
    delta = noisy["image"] - off["image"]
    assert torch.equal(delta[0], delta[1]) and torch.equal(delta[0], delta[2])
    identity = PairedDegradationDataset(base, degradation((1, 0, 0, 0)),
                                      {"enabled": True, "probability": 1}, 44)[0]
    assert identity["noise_sigma"] == 0


def test_geometry_only_base_supports_same_wrapper(tmp_path):
    canonical = cohort(tmp_path)
    class PseudoBase:
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return {k: v for k, v in canonical[index].items()
                    if not k.startswith("semantic") and not k.startswith("affinity")}
    sample = PairedDegradationDataset(PseudoBase(), degradation(), {"enabled": False}, 9)[0]
    assert "semantic_target" not in sample
    assert sample["instance_map"].shape == (16, 16)
    assert not sample["uncovered_boundary_source"]
