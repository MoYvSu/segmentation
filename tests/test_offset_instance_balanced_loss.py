import torch

from utils.offset_geometry_loss import center_offset_loss


def test_instance_balanced_offset_loss_prevents_large_instance_domination():
    instance_map = torch.zeros((1, 4, 4), dtype=torch.int64)
    instance_map[0, 0, 0] = 1
    instance_map[0, 1:3, :] = 2
    foreground = (instance_map > 0).unsqueeze(1)
    valid = torch.ones_like(foreground)
    offsets = torch.zeros((1, 2, 4, 4), requires_grad=True)
    offsets.data[:, :, 0, 0] = 1.0
    prediction = {
        "center_logits": torch.zeros((1, 1, 4, 4), requires_grad=True),
        "offsets": offsets,
    }
    center_target = torch.zeros((1, 1, 4, 4))
    offset_target = torch.zeros((1, 2, 4, 4))
    _, pixel_metrics = center_offset_loss(
        prediction, center_target, offset_target, foreground, valid,
        instance_map, smooth_l1_beta=1.0, offset_reduction="pixel_mean",
    )
    loss, balanced_metrics = center_offset_loss(
        prediction, center_target, offset_target, foreground, valid,
        instance_map, smooth_l1_beta=1.0, offset_reduction="instance_balanced",
    )
    assert balanced_metrics["offset_loss"] > pixel_metrics["offset_loss"] * 4
    loss.backward()
    assert offsets.grad is not None
