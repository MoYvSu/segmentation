import torch

from models.fpn_decoder import FPNDecoder
from utils.semantic_challenger import SemanticChallenger


def build_decoder():
    return FPNDecoder(
        in_channels=[8, 16, 32, 64],
        fpn_channels=16,
        dropout=0.0,
        semantic_residual=True,
        semantic_residual_version="highres_v1",
        semantic_residual_hidden=16,
        semantic_residual_color_channels=8,
        semantic_residual_half_channels=16,
        semantic_residual_full_channels=8,
        semantic_residual_max_logit_delta=2.0,
    )


def test_cold_reset_preserves_boundary_and_resets_semantic():
    torch.manual_seed(7)
    decoder = build_decoder()
    semantic_before = decoder.seg_fpn.lateral_convs[0].weight.detach().clone()
    boundary_before = decoder.boundary_fpn.lateral_convs[0].weight.detach().clone()

    torch.manual_seed(11)
    decoder.reset_semantic_branch()

    assert not torch.equal(
        semantic_before, decoder.seg_fpn.lateral_convs[0].weight
    )
    assert torch.equal(
        boundary_before, decoder.boundary_fpn.lateral_convs[0].weight
    )
    assert torch.count_nonzero(decoder.semantic_residual.out.weight) == 0
    assert torch.count_nonzero(decoder.semantic_residual.out.bias) == 0


def test_cold_start_only_exposes_semantic_parameters():
    decoder = build_decoder()
    decoder.set_semantic_cold_start_only()

    trainable = {
        name for name, parameter in decoder.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(
        name.startswith(("seg_fpn.", "seg_branch.", "semantic_residual."))
        for name in trainable
    )
    assert not any(
        parameter.requires_grad
        for parameter in decoder.boundary_fpn.parameters()
    )


def test_highres_semantic_challenger_uses_image_path():
    source_decoder = build_decoder()
    reference_decoder = FPNDecoder(
        in_channels=[8, 16, 32, 64],
        fpn_channels=16,
        dropout=0.0,
        semantic_residual=False,
    )
    challenger = SemanticChallenger(
        reference_decoder,
        semantic_residual=source_decoder.semantic_residual,
        semantic_residual_version=source_decoder.semantic_residual_version,
    ).eval()
    features = [
        torch.randn(1, 8, 16, 16),
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    image = torch.rand(1, 3, 64, 64)

    with torch.no_grad():
        output = challenger(features, image)

    assert output.shape == (1, 1, 64, 64)
    assert torch.isfinite(output).all()
    assert reference_decoder.semantic_residual is None
