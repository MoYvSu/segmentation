import torch

from models.offset_geometry import CenterOffsetGeometryDecoder
from utils.offset_geometry_loss import center_focal_loss, center_offset_loss


def test_geometry_decoder_outputs_center_and_offsets_at_requested_grid():
    decoder = CenterOffsetGeometryDecoder(
        in_channels=[8, 16, 32, 64],
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
    assert output["center_logits"].shape == (2, 1, 64, 64)
    assert output["offsets"].shape == (2, 2, 64, 64)


def test_center_offset_loss_is_finite_and_backpropagates():
    center_logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
    offsets = torch.zeros(1, 2, 16, 16, requires_grad=True)
    center_target = torch.zeros(1, 1, 16, 16)
    center_target[:, :, 8, 8] = 1.0
    offset_target = torch.full((1, 2, 16, 16), 0.1)
    foreground = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    valid = torch.ones_like(foreground)
    loss, metrics = center_offset_loss(
        {"center_logits": center_logits, "offsets": offsets},
        center_target, offset_target, foreground, valid,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["offset_loss"])
    loss.backward()
    assert center_logits.grad is not None
    assert offsets.grad is not None

def test_center_positive_weight_increases_exact_peak_penalty():
    logits = torch.zeros(1, 1, 2, 2)
    target = torch.zeros_like(logits)
    target[:, :, 0, 0] = 1.0
    valid = torch.ones_like(logits, dtype=torch.bool)
    baseline = center_focal_loss(logits, target, valid, positive_weight=1.0)
    emphasized = center_focal_loss(logits, target, valid, positive_weight=2.0)
    assert emphasized > baseline
