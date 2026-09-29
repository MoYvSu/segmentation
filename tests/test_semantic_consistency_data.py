# -*- coding: utf-8 -*-
"""核对无标签全覆盖、师生坐标、增强随机流与可靠区损失。"""
import random

import cv2
import numpy as np
import pytest
import torch

from data.direct_dual_head_dataset import _spatial_transform
from data.semantic_consistency import (
    SemanticConsistencyDataset, apply_acquisition_appearance,
    build_reliable_semantic_targets, masked_semantic_consistency, full_content_mask,
)
from data.backend_adaptation import PairedDegradationDataset
from utils.config import load_config


def source_dataset(tmp_path, *, count=7, draws=4):
    for index in range(count):
        yy, xx = np.mgrid[:16, :24]
        image = np.repeat((30 + index * 5 + xx * 3 + yy)[..., None], 3, axis=2).astype(np.uint8)
        cv2.imwrite(str(tmp_path / f"train_{index:03d}.png"), image)
        # 同名JSON既不加载，也不导致源图被排除。
        (tmp_path / f"train_{index:03d}.json").write_text("invalid labels", encoding="utf-8")
    degradation = {"profile_probabilities": [1, 0, 0, 0], "blur_sigma": [.4, 5.2],
                   "resample_probability": 0, "resample_scale": [.5, .9]}
    return SemanticConsistencyDataset(tmp_path, degradation, {"enabled": False}, 17,
        draws_per_epoch=draws, expected_sources=count, image_size=24,
        weak_config={"enabled": False})


def test_complete_source_cycles_cross_epoch_boundaries(tmp_path):
    dataset = source_dataset(tmp_path)
    sampled = []
    for epoch in range(1, 5):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["global_draw"] == len(sampled)
            sampled.append(sample["source_index"])
    assert sorted(sampled[:7]) == list(range(7))
    assert sorted(sampled[7:14]) == list(range(7))
    assert len(dataset.samples) == 7


def test_all_views_and_valid_mask_share_geometry_without_global_rng(tmp_path):
    dataset = source_dataset(tmp_path)
    numpy_state, torch_state, python_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
    for epoch in (1, 3):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            sample = dataset[index]
            original = dataset.base_dataset[sample["source_index"]]
            geometry = sample["horizontal_flip"], sample["vertical_flip"], sample["rotation_k"]
            expected = _spatial_transform(original["image"], *geometry)
            for key in ("weak_image", "weak_check_image", "strong_image"):
                assert torch.equal(sample[key], expected)
            assert torch.equal(sample["valid_content"], _spatial_transform(original["valid_content"], *geometry))
            assert not sample["valid_content"].all()
            assert not any("target" in key or "instance_map" in key for key in sample)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state


def test_source_count_failure_does_not_silently_change_training_pool(tmp_path):
    with pytest.raises(ValueError, match="expected 1000"):
        SemanticConsistencyDataset(tmp_path, {}, {}, 17, draws_per_epoch=64)


def test_full_content_mask_replays_all_geometry_and_ignores_gt_and_coarse_masks():
    original = torch.zeros(1, 25, 25, dtype=torch.bool)
    original[:, :13, :19] = True
    examples, expected = [], []
    for horizontal in (False, True):
        for vertical in (False, True):
            for rotation in range(4):
                examples.append({"input_content_shape": torch.tensor([19, 13] if rotation % 2 else [13, 19]),
                    "horizontal_flip": horizontal, "vertical_flip": vertical, "rotation_k": rotation,
                    # 两者都不可用作真实完整内容mask。
                    "valid_content": torch.zeros(1, 12, 12, dtype=torch.bool),
                    "semantic_valid_content": torch.zeros(1, 25, 25, dtype=torch.bool)})
                expected.append(_spatial_transform(original, horizontal, vertical, rotation))
    batch = torch.utils.data.default_collate(examples)
    actual = full_content_mask(batch, torch.zeros(16, 3, 25, 25))
    assert torch.equal(actual, torch.stack(expected))
    assert torch.all(actual.sum((1, 2, 3)) == 13 * 19)


def test_full_content_mask_rejects_out_of_canvas_shape():
    batch = {"input_content_shape": [[17, 33]], "horizontal_flip": [False],
             "vertical_flip": [False], "rotation_k": [0]}
    with pytest.raises(ValueError, match="exceed"):
        full_content_mask(batch, torch.zeros(1, 3, 32, 32))


def test_real_d5a_degradation_reuses_existing_recipe_and_default_collates(tmp_path):
    source_dataset(tmp_path)
    degradation = load_config("config/train/rgb_diffusion_d5a.yaml")["rgb_restoration"]["degradation"]
    noise = {"enabled": True, "probability": .5, "sigma": [0, .006]}
    dataset = SemanticConsistencyDataset(tmp_path, degradation, noise, 23, draws_per_epoch=16,
        expected_sources=7, image_size=24,
        weak_config={"max_offset": .02, "max_clipped_fraction": .005})
    reference = PairedDegradationDataset(dataset.base_dataset, degradation, noise, seed=23 + 20260927)
    endpoint_count, weak_changes = 0, 0
    for index in range(16):
        sample = dataset[index]
        reference.set_epoch(index)
        expected = reference[sample["source_index"]]
        assert torch.equal(sample["strong_image"], expected["image"])
        assert sample["endpoint_applied"] == expected["endpoint_applied"]
        endpoint_count += sample["endpoint_applied"]
        weak_changes += not torch.equal(sample["weak_image"], sample["weak_check_image"])
        batch = torch.utils.data.default_collate([sample])
        assert batch["weak_image"].shape == batch["strong_image"].shape == (1, 3, 24, 24)
    assert endpoint_count > 0 and weak_changes > 0


def test_affine_appearance_preserves_spatial_order_and_replays():
    image = torch.linspace(.1, .8, 60).reshape(1, 1, 6, 10).repeat(1, 3, 1, 1)
    valid = torch.ones(1, 1, 6, 10, dtype=torch.bool)
    config = {"enabled": True, "probability": 1, "contrast": [.5, .5],
              "offset": [.1, .1], "rgb_shift": [-.03, .03]}
    numpy_state, torch_state = np.random.get_state(), torch.get_rng_state()
    result, stats = apply_acquisition_appearance(image, valid, config, 73)
    replay, replay_stats = apply_acquisition_appearance(image, valid, config, 73)
    assert torch.equal(result, replay) and stats == replay_stats
    assert stats["scale"] == 1 and abs(sum(stats["actual_rgb_shift"])) < 1e-12
    assert torch.allclose(result[..., 1:] - result[..., :-1], .5 * (image[..., 1:] - image[..., :-1]))
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_launch_config_enables_appearance_and_max_shift_without_enabled_key():
    image = torch.full((1, 3, 8, 8), .6)
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    changed, stats = apply_acquisition_appearance(image, valid, {
        "probability": 1., "contrast": [.5, .5], "offset": [.1, .1],
        "max_rgb_shift": .03, "max_clipped_fraction": .01}, 19)
    assert stats["selected"] and stats["applied"] and not torch.equal(changed, image)
    assert stats["clipped_fraction"] == 0
    identity, stats = apply_acquisition_appearance(image, valid, None, 19)
    assert identity is image and not stats["applied"]


def test_affine_clipping_budget_uses_valid_region_and_can_fall_back_to_identity():
    image = torch.full((1, 3, 8, 8), .95)
    valid = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    valid[:, :, :4] = True
    image[:, :, 4:] = 1.0  # padding饱和不应迫使有效区变换恒等
    config = {"enabled": True, "probability": 1, "contrast": [1, 1],
              "offset": [.1, .1], "rgb_shift": [0, 0], "max_clipped_fraction": 0.01}
    result, stats = apply_acquisition_appearance(image, valid, config, 5)
    assert .49 < stats["scale"] < .51 and stats["clipped_fraction"] == 0
    assert result.min() > .99
    saturated = torch.ones_like(image)
    result, stats = apply_acquisition_appearance(saturated, valid, config, 5)
    assert result is saturated and not stats["applied"]
    assert stats["selected"] and stats["actual_offset"] == 0


@pytest.mark.parametrize("config", [{"enabled": 1}, {"enabled": True, "contrast": [0, 1]},
                                    {"enabled": True, "probability": float("nan")}])
def test_affine_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        apply_acquisition_appearance(torch.zeros(1, 3, 4, 4), torch.ones(1, 1, 4, 4), config, 1)


def test_teacher_reliability_is_classwise_eroded_and_never_supervises_padding():
    probability = torch.full((1, 1, 10, 14), .97)
    probability[:, :, :, :7] = .03
    left = torch.logit(probability)
    right = left.clone()
    right[:, :, 4, 3] = torch.logit(torch.tensor(.98))  # 教师类别不一致
    right[:, :, 4, 10] = torch.logit(torch.tensor(.91))  # 同类但概率差过大
    valid = torch.ones(1, 1, 20, 28, dtype=torch.bool)
    valid[:, :, 16:] = False
    target, mask, stats = build_reliable_semantic_targets(left, right, valid)
    assert not mask[:, :, 3:6, 2:5].any()
    assert not mask[:, :, 3:6, 9:12].any()
    assert not mask[:, :, 7:].any()
    assert not mask[:, :, :, 6:8].any()
    assert stats["ferrite_accepted_pixels"] > 0 and stats["pearlite_accepted_pixels"] > 0
    assert stats["accepted_pixels"] == stats["ferrite_accepted_pixels"] + stats["pearlite_accepted_pixels"]
    assert stats["disagreement_pixels"] == 1
    assert torch.equal(target, (left.sigmoid() + right.sigmoid()) / 2)


def test_consistency_loss_only_updates_accepted_student_pixels_and_detaches_teacher():
    student = torch.zeros(1, 1, 2, 3, requires_grad=True)
    target = torch.full_like(student, .97, requires_grad=True)
    mask = torch.zeros_like(student, dtype=torch.bool)
    mask[:, :, 0, 0] = True
    loss, stats = masked_semantic_consistency(student, target, mask)
    loss.backward()
    assert stats["accepted_pixels"] == 1 and target.grad is None
    assert torch.count_nonzero(student.grad) == 1
    empty_student = torch.ones_like(student, requires_grad=True)
    zero, stats = masked_semantic_consistency(empty_student, target, torch.zeros_like(mask))
    zero.backward()
    assert zero.item() == 0 and torch.count_nonzero(empty_student.grad) == 0
