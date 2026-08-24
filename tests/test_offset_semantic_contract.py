import torch
import torch.nn as nn

from models.offset_geometry import (
    FrozenSemanticGeometrySystem,
    semantic_state_digest,
)


class _DummyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 2))


class _DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _DummyTrunk()

    def forward(self, image):
        return [image]


class _DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_fpn = nn.Conv2d(3, 4, 1)
        self.seg_branch = nn.Conv2d(4, 1, 1)

    def forward(self, features, **kwargs):
        semantic = self.seg_branch(self.seg_fpn(features[0]))
        return torch.cat([semantic, torch.zeros_like(semantic)], dim=1)


class _DummyReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _DummyEncoder()
        self.decoder = _DummyDecoder()

    def forward(self, image):
        return self.decoder(self.encoder(image))


class _TinyGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features):
        value = features[0][:, :1] * self.scale
        return {"center_logits": value, "offsets": value.repeat(1, 2, 1, 1)}


def test_geometry_training_cannot_mutate_reference_semantics():
    reference = _DummyReference()
    system = FrozenSemanticGeometrySystem(reference, _TinyGeometry())
    image = torch.randn(1, 3, 8, 8)
    before_digest = semantic_state_digest(reference)
    before_logits = system.semantic_logits(image).clone()
    system.train()
    assert not system.reference_model.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    output = system.geometry_forward(image)
    output["offsets"].sum().backward()
    assert system.geometry_decoder.scale.grad is not None
    assert all(parameter.grad is None for parameter in reference.parameters())
    assert semantic_state_digest(reference) == before_digest
    assert torch.equal(system.semantic_logits(image), before_logits)
