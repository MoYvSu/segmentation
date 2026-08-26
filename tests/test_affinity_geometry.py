import pytest
import torch

from models.affinity_geometry import AffinityGeometryDecoder
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_loss import (
    balanced_affinity_loss,
    build_affinity_targets_torch,
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
    target, edge_valid = build_affinity_targets_torch(
        labels,
        valid,
        offsets=[(0, 1)],
        uncovered_as_boundary=flags,
    )

    assert edge_valid[0, 0, 0].tolist() == [True, False, True, False]
    assert target[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert not edge_valid[1].any()


def test_uncovered_boundary_flag_requires_one_value_per_sample():
    labels = torch.ones((2, 2, 2), dtype=torch.int64)
    valid = torch.ones((2, 1, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="one flag per batch sample"):
        build_affinity_targets_torch(
            labels,
            valid,
            uncovered_as_boundary=torch.tensor([True]),
        )
