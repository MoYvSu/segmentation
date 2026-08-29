import pytest
import torch

from models.affinity_geometry import (
    AffinityGeometryDecoder,
    HighResolutionShortAffinityResidual,
)
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_loss import (
    balanced_affinity_loss,
    build_affinity_targets_torch,
    negative_affinity_tail_loss,
)


def test_affinity_decoder_outputs_requested_channels_and_grid():
    decoder = AffinityGeometryDecoder(
        in_channels=[8, 16, 32, 64],
        affinity_channels=len(DEFAULT_AFFINITY_OFFSETS),
        fpn_channels=16,
        up_channels=16,
        output_grid=64,
    )
    features = [
        torch.randn(2, 8, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    output = decoder(features)
    assert output["affinity_logits"].shape == (2, 8, 64, 64)
    assert output["affinity_feature"].shape == (2, 16, 64, 64)


def test_highres_short_residual_starts_zero_and_preserves_long_channels():
    refiner = HighResolutionShortAffinityResidual(
        feature_channels=8,
        feature_hidden=8,
        image_hidden=8,
        fusion_hidden=8,
        max_logit_delta=1.0,
    )
    feature = torch.randn(1, 8, 8, 8)
    coarse = torch.randn(1, 8, 8, 8)
    image = torch.rand(1, 3, 16, 16)
    output = refiner(feature, coarse, image)
    expected = torch.nn.functional.interpolate(
        coarse, size=(16, 16), mode="bilinear", align_corners=False
    )
    assert output["affinity_logits"].shape == (1, 8, 16, 16)
    assert torch.count_nonzero(output["short_affinity_delta"]) == 0
    assert torch.equal(output["affinity_logits"], expected)


def test_highres_short_residual_never_changes_long_channels_after_update():
    refiner = HighResolutionShortAffinityResidual(
        feature_channels=8,
        feature_hidden=8,
        image_hidden=8,
        fusion_hidden=8,
        max_logit_delta=0.5,
    )
    with torch.no_grad():
        refiner.out.bias.fill_(0.25)
    feature = torch.randn(1, 8, 8, 8)
    coarse = torch.randn(1, 8, 8, 8)
    image = torch.rand(1, 3, 16, 16)
    output = refiner(feature, coarse, image)["affinity_logits"]
    expected = torch.nn.functional.interpolate(
        coarse, size=(16, 16), mode="bilinear", align_corners=False
    )
    assert not torch.equal(output[:, :4], expected[:, :4])
    assert torch.equal(output[:, 4:], expected[:, 4:])


def test_torch_targets_match_same_instance_pairs_and_loss_backpropagates():
    labels = torch.zeros((1, 8, 8), dtype=torch.int64)
    labels[:, 1:5, 1:5] = 1
    labels[:, 5:7, 5:7] = 2
    valid = torch.ones((1, 1, 8, 8), dtype=torch.bool)
    target, edge_valid = build_affinity_targets_torch(labels, valid)
    logits = torch.zeros_like(target, requires_grad=True)
    loss, metrics = balanced_affinity_loss(logits, target, edge_valid)
    assert torch.isfinite(loss)
    assert metrics["positive_edges"] > 0
    assert metrics["negative_edges"] > 0
    loss.backward()
    assert logits.grad is not None


def test_hard_negative_weight_penalizes_false_merge_edges():
    logits = torch.tensor([[[[0.0, -3.0, 3.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    baseline, _ = balanced_affinity_loss(logits, target, valid)
    weighted, _ = balanced_affinity_loss(
        logits,
        target,
        valid,
        negative_weight=1.5,
        hard_negative_weight=2.0,
        hard_negative_gamma=2.0,
    )
    assert weighted > baseline


def test_manual_uncovered_band_can_be_negative_without_supervising_zero_zero():
    labels = torch.tensor([[[1, 0, 0, 2]], [[1, 0, 0, 2]]], dtype=torch.int64)
    valid = torch.ones((2, 1, 1, 4), dtype=torch.bool)
    flags = torch.tensor([True, False])
    target, edge_valid, uncovered = build_affinity_targets_torch(
        labels,
        valid,
        offsets=[(0, 1)],
        uncovered_as_boundary=flags,
        return_uncovered_mask=True,
    )

    assert edge_valid[0, 0, 0].tolist() == [True, False, True, False]
    assert target[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert uncovered[0, 0, 0].tolist() == [True, False, True, False]
    assert not edge_valid[1].any()
    assert not uncovered[1].any()


def test_uncovered_boundary_flag_requires_one_value_per_sample():
    labels = torch.ones((2, 2, 2), dtype=torch.int64)
    valid = torch.ones((2, 1, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="one flag per batch sample"):
        build_affinity_targets_torch(
            labels,
            valid,
            uncovered_as_boundary=torch.tensor([True]),
        )


def test_edge_weight_can_soften_new_gap_negatives():
    logits = torch.tensor([[[[-3.0, 3.0]]]], requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    full, _ = balanced_affinity_loss(logits, target, valid)
    softened, _ = balanced_affinity_loss(
        logits,
        target,
        valid,
        edge_weight=torch.tensor([[[[1.0, 0.2]]]]),
    )
    assert softened < full


def test_absolute_edge_weight_discount_survives_pseudo_only_batch():
    logits = torch.full((1, 1, 1, 4), 2.0, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    full, _ = balanced_affinity_loss(logits, target, valid)
    discounted, _ = balanced_affinity_loss(
        logits,
        target,
        valid,
        edge_weight=torch.full_like(logits, 0.25),
        normalize_edge_weights=False,
    )
    assert torch.allclose(discounted, 0.25 * full)


def test_negative_tail_loss_focuses_sparse_false_merge_edges():
    logits = torch.tensor([[[[-3.0, -1.0, 0.0, 3.0]]]], requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss, metrics = negative_affinity_tail_loss(
        logits, target, valid, margin=0.45, top_fraction=0.25
    )
    assert metrics["tail_edges"] == 4
    assert metrics["tail_selected"] == 1
    loss.backward()
    assert logits.grad[0, 0, 0, 3] > 0
    assert torch.count_nonzero(logits.grad) == 1


def test_negative_tail_loss_can_select_manual_samples_only():
    logits = torch.full((2, 1, 1, 2), 3.0, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss, metrics = negative_affinity_tail_loss(
        logits,
        target,
        valid,
        sample_mask=torch.tensor([True, False]),
        top_fraction=1.0,
    )
    assert metrics["tail_edges"] == 2
    loss.backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].abs().sum() == 0
