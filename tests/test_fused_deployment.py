import torch
from torch import nn

from models.fused_deployment import FusedPhaseAffinityModel


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return [image + 1.0]


class _Semantic(nn.Module):
    def forward(self, features, image):
        return features[0][:, :1] + image[:, :1]


class _Affinity(nn.Module):
    def forward(self, features):
        return {"affinity_logits": features[0][:, :2]}


def test_fused_model_runs_shared_encoder_once():
    encoder = _Encoder()
    model = FusedPhaseAffinityModel(encoder, _Semantic(), _Affinity())
    image = torch.zeros((1, 3, 8, 8))
    output = model(image)
    assert encoder.calls == 1
    assert output["semantic_logits"].shape == (1, 1, 8, 8)
    assert output["affinity_logits"].shape == (1, 2, 8, 8)
    assert torch.all(output["semantic_logits"] == 1.0)
    assert torch.all(output["affinity_logits"] == 1.0)
