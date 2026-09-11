# -*- coding: utf-8 -*-

import copy

import pytest
import torch
from torch import nn

from models.cross_head_refinement import (
    CROSS_HEAD_FORMAT, CrossHeadRefiner, RefinedMainline, build_refined_from_checkpoint,
)
from models.fused_deployment import FusedPhaseAffinityModel
from utils.semantic_challenger import SemanticChallenger


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


def _features():
    return {
        "semantic_logits": torch.randn(1, 1, 32, 32),
        "affinity_logits": torch.randn(1, 8, 16, 16),
        "semantic_feature": torch.randn(1, 8, 8, 8),
        "affinity_feature": torch.randn(1, 4, 16, 16),
    }


def _refiner(mode):
    return CrossHeadRefiner(semantic_feature_channels=8, affinity_feature_channels=4,
                           hidden_channels=4, context_mode=mode)


def test_cross_head_zero_and_strict_checkpoint_contract():
    values = _features()
    for mode in ("self", "cross"):
        model = _refiner(mode)
        output = model(values)
        for key in ("semantic_logits", "affinity_logits"):
            assert torch.equal(output[key], values[key])
        restored = _refiner(mode)
        restored.load_state_dict(model.state_dict(), strict=True)
        assert torch.equal(restored(values)["semantic_logits"], output["semantic_logits"])
        state = dict(model.state_dict())
        state.pop("semantic.delta.weight")
        with pytest.raises(RuntimeError, match="Missing key"):
            restored.load_state_dict(state, strict=True)


def test_context_routing_and_both_task_gradients():
    values = _features()
    altered = {key: value.clone() for key, value in values.items()}
    altered["affinity_feature"] = torch.randn_like(values["affinity_feature"])
    altered["affinity_logits"] = -values["affinity_logits"]
    for mode in ("self", "cross"):
        torch.manual_seed(42)
        model = _refiner(mode)
        # 非零输出层使信息路径可观测，而不是只检查配置中的模式名。
        nn.init.normal_(model.semantic.delta.weight, std=0.1)
        nn.init.normal_(model.affinity.delta.weight, std=0.1)
        original = model(values)
        changed = model(altered)
        if mode == "self":
            assert torch.equal(original["semantic_logits"], changed["semantic_logits"])
        else:
            assert not torch.equal(original["semantic_logits"], changed["semantic_logits"])
        model.zero_grad(set_to_none=True)
        original["semantic_logits"].sum().backward()
        gradient = model.affinity_projection[0].weight.grad
        assert (gradient is not None and gradient.abs().sum() > 0) == (mode == "cross")


class _FrozenBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout())
        self.semantic = nn.Conv2d(4, 8, 1)
        self.affinity = nn.Conv2d(4, 8, 1)

    def forward(self, image, *, return_features=False):
        feature = self.encoder(image)
        semantic = self.semantic(feature)
        output = {"semantic_logits": semantic[:, :1], "affinity_logits": self.affinity(feature)}
        if return_features:
            output.update(semantic_feature=semantic, affinity_feature=feature)
        return output


def test_refiner_train_freezes_base_buffers_and_all_parameters_update():
    base = _FrozenBase().eval()
    frozen_state = copy.deepcopy(base.state_dict())
    image = torch.randn(1, 3, 16, 16)
    counts = []
    for mode in ("self", "cross"):
        model = RefinedMainline(base, _refiner(mode)).train()
        assert not model.base.training and not model.base.encoder[1].training
        initial = copy.deepcopy(model.refiner.state_dict())
        counts.append(model.parameter_summary()["trainable"])
        optimizer = torch.optim.SGD(model.refiner.parameters(), lr=0.1)
        seen = set()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            output = model(image)
            (output["semantic_logits"].square().mean() + output["affinity_logits"].square().mean()).backward()
            for key, p in model.refiner.named_parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    seen.add(key)
            optimizer.step()
        assert seen == set(dict(model.refiner.named_parameters()))
        assert all(not torch.equal(value, initial[key]) for key, value in model.refiner.state_dict().items())
        assert all(p.grad is None and not p.requires_grad for p in base.parameters())
        assert all(torch.equal(value, frozen_state[key]) for key, value in base.state_dict().items())
    assert counts[0] == counts[1]


def test_refiner_loader_rejects_wrong_base_and_unknown_version():
    model = _refiner("cross")
    payload = {"format": CROSS_HEAD_FORMAT, "base_sha256": "correct", "context_mode": "cross",
               "architecture": model.architecture, "refiner_state_dict": model.state_dict()}
    base = _FrozenBase()
    with pytest.raises(ValueError, match="exact frozen base"):
        build_refined_from_checkpoint(payload, base, "different", "cpu")
    with pytest.raises(ValueError, match="unsupported"):
        build_refined_from_checkpoint({**payload, "format": "future"}, base, "correct", "cpu")
    restored = build_refined_from_checkpoint(payload, base, "correct", "cpu")
    assert not restored.training


def test_semantic_feature_exposure_preserves_old_forward():
    scaffold = nn.Module()
    scaffold.seg_fpn = nn.Conv2d(4, 8, 1)
    scaffold.seg_branch = nn.Conv2d(8, 1, 1)
    decoder = SemanticChallenger(scaffold)
    feature = torch.randn(1, 4, 8, 8)
    logits = decoder(feature)
    exposed, intermediate = decoder(feature, return_features=True)
    assert torch.equal(logits, exposed)
    assert intermediate.shape == (1, 8, 8, 8)
