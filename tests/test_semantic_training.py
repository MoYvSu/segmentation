import numpy as np
import torch

from data.dataset import random_crop
from models.fpn_decoder import FPNDecoder, SemanticResidualAdapter
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


def test_semantic_residual_starts_as_exact_v6_identity_and_is_isolated():
    kwargs = {
        "in_channels": [8, 16, 32, 64],
        "fpn_channels": 16,
        "dropout": 0.0,
        "use_bn": False,
    }
    base = FPNDecoder(**kwargs, semantic_residual=False)
    residual = FPNDecoder(
        **kwargs,
        semantic_residual=True,
        semantic_residual_hidden=8,
        semantic_residual_color_channels=4,
    )
    missing, unexpected = residual.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(name.startswith("semantic_residual.") for name in missing)
    assert unexpected == []
    features = [
        torch.randn(2, 8, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 64, 2, 2),
    ]
    image = torch.rand(2, 3, 64, 64)
    base.eval()
    residual.eval()
    with torch.no_grad():
        expected = base(features, image=image)
        actual = residual(features, image=image)
    assert torch.equal(actual, expected)

    residual.set_semantic_residual_only()
    assert all(
        parameter.requires_grad
        for parameter in residual.semantic_residual.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in residual.named_parameters()
        if not name.startswith("semantic_residual.")
    )


def test_highres_semantic_residual_preserves_v6_logits_at_full_resolution():
    kwargs = {
        "in_channels": [8, 16, 32, 64],
        "fpn_channels": 16,
        "dropout": 0.0,
        "use_bn": False,
    }
    base = FPNDecoder(**kwargs, semantic_residual=False)
    highres = FPNDecoder(
        **kwargs,
        semantic_residual=True,
        semantic_residual_version="highres_v1",
        semantic_residual_hidden=8,
        semantic_residual_color_channels=4,
        semantic_residual_half_channels=8,
        semantic_residual_full_channels=4,
        semantic_residual_max_logit_delta=0.75,
    )
    missing, unexpected = highres.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(name.startswith("semantic_residual.") for name in missing)
    assert unexpected == []
    features = [
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    image = torch.rand(1, 3, 64, 64)
    base.eval()
    highres.eval()
    with torch.no_grad():
        coarse = base(features, image=image)
        actual = highres(features, image=image)
        expected = torch.nn.functional.interpolate(
            coarse, size=(64, 64), mode="bilinear", align_corners=True
        )
    assert actual.shape[-2:] == (64, 64)
    assert torch.equal(actual, expected)


def test_instance_probability_pool_and_thin_weight_do_not_require_a_center():
    target = torch.zeros(1, 5, 12)
    labels = torch.zeros(1, 5, 12, dtype=torch.long)
    # A one-pixel-wide, non-convex ferrite path has no eroded core.
    labels[:, 2, 1:10] = 1
    labels[:, 1:4, 9] = 1
    target[labels == 1] = 1.0
    boundary = (labels > 0).float()
    logits = torch.full((1, 5, 12), -0.2, requires_grad=True)

    loss, stats = instance_balanced_core_bce(
        logits,
        target,
        labels,
        boundary,
        core_radius=2,
        min_core_pixels=4,
        pooled_probability_weight=0.5,
        thin_instance_weight=1.5,
    )
    loss.backward()
    assert stats["fallback_instances"] == 1.0
    assert stats["pooled_probability_loss"] > 0.0
    assert float(logits.grad[labels == 1].abs().mean()) > 0.0


def test_adaptive_photometric_cues_ignore_global_exposure_affine_change():
    image = torch.rand(2, 3, 32, 32) * 0.5 + 0.2
    shifted = image * 0.6 + 0.1
    first = SemanticResidualAdapter.adaptive_photometric_cues(image, (16, 16))
    second = SemanticResidualAdapter.adaptive_photometric_cues(shifted, (16, 16))
    assert torch.allclose(first, second, atol=2.0e-5, rtol=2.0e-5)


def test_hard_instance_focus_and_mild_ferrite_weight_raise_hard_ferrite_loss():
    target = torch.zeros(1, 4, 8)
    labels = torch.zeros(1, 4, 8, dtype=torch.long)
    labels[:, :2] = 1
    labels[:, 2:] = 2
    target[:, 2:] = 1.0
    logits = torch.full((1, 4, 8), -2.0, requires_grad=True)
    logits.data[:, :2] = -2.0  # easy pearlite, hard ferrite

    symmetric, _ = instance_balanced_core_bce(
        logits, target, labels, core_radius=0, min_core_pixels=1
    )
    focused, stats = instance_balanced_core_bce(
        logits,
        target,
        labels,
        core_radius=0,
        min_core_pixels=1,
        ferrite_class_weight=1.15,
        hard_instance_gamma=1.5,
        hard_instance_floor=0.35,
    )
    assert focused > symmetric
    assert stats["ferrite_instances"] == 1.0
    focused.backward()
    assert logits.grad is not None
