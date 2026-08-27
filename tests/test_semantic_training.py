import numpy as np
import torch

from data.dataset import random_crop
from utils.progressive_aug import ProgressiveAppearanceAug
from utils.semantic_training import (
    dark_boundary_contamination,
    instance_balanced_core_bce,
)


def test_instance_balanced_core_bce_has_gradient_and_balances_classes():
    logits = torch.zeros(1, 6, 8, requires_grad=True)
    target = torch.zeros(1, 6, 8)
    labels = torch.zeros(1, 6, 8, dtype=torch.long)
    labels[:, :5, :] = 1
    labels[:, 5:, :2] = 2
    target[:, 5:, :2] = 1.0

    loss, stats = instance_balanced_core_bce(
        logits,
        target,
        labels,
        core_radius=0,
        min_core_pixels=1,
        class_balance=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert float(logits.grad[:, 5:, :2].abs().mean()) > float(
        logits.grad[:, :5, :].abs().mean()
    )
    assert stats["instances"] == 2.0


def test_instance_core_ignores_boundary_but_tiny_instance_falls_back():
    logits = torch.full((1, 9, 9), -4.0, requires_grad=True)
    target = torch.ones(1, 9, 9)
    labels = torch.ones(1, 9, 9, dtype=torch.long)
    boundary = torch.zeros(1, 9, 9)
    boundary[:, 0, :] = 1
    boundary[:, -1, :] = 1
    boundary[:, :, 0] = 1
    boundary[:, :, -1] = 1
    with torch.no_grad():
        logits[:, 2:7, 2:7] = 4.0

    loss, stats = instance_balanced_core_bce(
        logits,
        target,
        labels,
        boundary,
        core_radius=1,
        min_core_pixels=4,
    )
    assert float(loss.detach()) < 0.05
    assert stats["core_instances"] == 1.0

    tiny_labels = torch.zeros(1, 5, 5, dtype=torch.long)
    tiny_labels[:, 2, 2] = 1
    tiny_target = torch.zeros(1, 5, 5)
    tiny_target[:, 2, 2] = 1.0
    tiny_logits = torch.zeros(1, 5, 5, requires_grad=True)
    tiny_boundary = torch.zeros(1, 5, 5)
    tiny_boundary[:, 2, 2] = 1.0
    tiny_loss, tiny_stats = instance_balanced_core_bce(
        tiny_logits,
        tiny_target,
        tiny_labels,
        tiny_boundary,
        core_radius=2,
        min_core_pixels=4,
    )
    tiny_loss.backward()
    assert tiny_stats["fallback_instances"] == 1.0
    assert float(tiny_logits.grad[:, 2, 2].abs()) > 0


def test_dark_boundary_contamination_and_progressive_context_gate():
    image = torch.ones(3, 11, 11)
    boundary = torch.zeros(11, 11)
    boundary[5, 5] = 1.0
    darkened = dark_boundary_contamination(
        image, boundary, width=1, opacity=0.5
    )
    assert float(darkened[:, 5, 5].mean()) < 0.7
    assert torch.allclose(darkened[:, 0, 0], image[:, 0, 0])

    augmentor = ProgressiveAppearanceAug(
        {
            "enabled": True,
            "policy": "physical_v1",
            "start_epoch": 0,
            "ramp_epochs": 0,
            "max_prob": 1.0,
            "min_ops": 1,
            "max_ops": 1,
            "op_weights": {"dark_rim": 1.0},
            "dark_rim_width_range": [1, 1],
            "dark_rim_opacity_range": [0.5, 0.5],
        },
        torch.device("cpu"),
    )
    with_target = augmentor(image.unsqueeze(0), boundary.unsqueeze(0))
    without_target = augmentor(image.unsqueeze(0))
    assert float(with_target[0, :, 5, 5].mean()) < 0.7
    assert torch.allclose(without_target, image.unsqueeze(0))


def test_random_crop_preserves_optional_instance_map_alignment():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    semantic = np.zeros((4, 4), dtype=np.uint8)
    boundary = np.zeros((4, 4), dtype=np.uint8)
    weight = np.ones((4, 4), dtype=np.float32)
    center = np.zeros((4, 4), dtype=np.float32)
    instance = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = random_crop(
        image, semantic, boundary, weight, center, 4, instance_map=instance
    )
    assert len(result) == 6
    assert np.array_equal(result[-1], instance)
