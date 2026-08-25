import torch

from utils.affinity_fusion import affinity_boundary_probability
from tools.run_affinity_watershed_ab import (
    affinity_mean_boundary_probability,
    crop_letterbox_output,
    probability_to_logit,
)


def test_affinity_mean_is_inverted_into_boundary_probability():
    probability = torch.tensor([0.2, 0.4, 0.6, 0.8] * 2).view(1, 8, 1, 1)
    logits = probability_to_logit(probability)
    boundary = affinity_mean_boundary_probability(logits)
    assert torch.allclose(boundary, torch.tensor([[[[0.5]]]]), atol=1e-6)


def test_probability_logit_round_trip():
    probability = torch.tensor([0.01, 0.25, 0.75, 0.99])
    assert torch.allclose(torch.sigmoid(probability_to_logit(probability)), probability)


def test_crop_letterbox_output_removes_bottom_and_right_padding():
    output = torch.arange(16.0).view(1, 1, 4, 4)
    cropped = crop_letterbox_output(output, 4, pad_h=1, pad_w=1, original_size=(3, 3))
    assert cropped.shape == (1, 1, 3, 3)
    assert torch.equal(cropped, output[:, :, :3, :3])


def test_gated_fusion_rejects_isolated_long_range_fog():
    probability = torch.full((1, 8, 9, 9), 0.95)
    probability[:, 6:, 4, 4] = 0.05
    boundary = affinity_boundary_probability(
        probability_to_logit(probability), mode="gated"
    )
    assert float(boundary[0, 0, 4, 4]) < 0.10


def test_gated_fusion_uses_long_range_signal_near_short_boundary():
    probability = torch.full((1, 8, 9, 9), 0.95)
    probability[:, :4, 4, 1] = 0.05
    probability[:, 6:, 4, 4] = 0.05
    logits = probability_to_logit(probability)
    short = affinity_boundary_probability(logits, mode="short")
    gated = affinity_boundary_probability(logits, mode="gated")
    assert float(gated[0, 0, 4, 4]) > float(short[0, 0, 4, 4]) + 0.05
