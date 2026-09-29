# -*- coding: utf-8 -*-
"""验证共享底座隔离、checkpoint重计算梯度以及新旧权重的严格加载。"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from models.lora import LoRALinear
from models.semantic_lora import SemanticLoRA, semantic_features
from models.fused_deployment import FusedPhaseAffinityModel, build_fused_model_from_bundle, FUSED_DEPLOYMENT_FORMAT
from train_semantic_consistency import lora_comparison_config, shared_frozen_digest, verify_lora_input
from utils.config import load_config


class TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = LoRALinear(nn.Linear(3, 4), rank=2, alpha=3)
        self.second = LoRALinear(nn.Linear(4, 3), rank=2, alpha=3)
        with torch.no_grad():
            self.first.lora_B.fill_(.12)
            self.second.lora_B.fill_(.08)

    def forward(self, image):
        value = image.permute(0, 2, 3, 1)
        return [self.second(torch.tanh(self.first(value))).permute(0, 3, 1, 2)]


class TinyEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.trunk = TinyTrunk()
        self.trainable_lora = False

    def forward(self, image):
        with torch.no_grad():
            return self.trunk(image)

    def get_stage_channels(self):
        return [3]


class TinySemantic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.projection = nn.Conv2d(3, 1, 1)

    def forward(self, features, image, *, return_features=False):
        logits = self.projection(features[0])
        return (logits, features[0]) if return_features else logits


class TinyAffinity(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Conv2d(3, 2, 1)

    def forward(self, features):
        return {'affinity_logits': self.projection(features[0]), 'affinity_feature': features[0]}


def backend(checkpointing=True):
    model = FusedPhaseAffinityModel(TinyEncoder(), TinySemantic(), TinyAffinity()).eval().requires_grad_(False)
    model.semantic_lora = SemanticLoRA(model.encoder.trunk, gradient_checkpointing=checkpointing)
    return model


def test_clone_is_exact_consumes_no_rng_and_shares_no_lora_storage():
    model = backend()
    image = torch.rand(1, 3, 8, 8)
    assert torch.equal(model.encoder(image)[0], semantic_features(model, image)[0])
    rng = torch.get_rng_state().clone()
    adapter = SemanticLoRA(model.encoder.trunk)
    assert torch.equal(torch.get_rng_state(), rng)
    original = dict(model.encoder.trunk.named_parameters())
    for name, value in zip(adapter.parameter_names, adapter.weights):
        assert torch.equal(value, original[name]) and value.data_ptr() != original[name].data_ptr()


@pytest.mark.parametrize('checkpointing', [False, True])
def test_backward_updates_only_private_lora_and_head_without_changing_affinity(checkpointing):
    model = backend(checkpointing)
    model.semantic_decoder.requires_grad_(True)
    image = torch.rand(1, 3, 8, 8)
    initial = {k: v.detach().clone() for k, v in model(image).items()}
    frozen = shared_frozen_digest(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=.2)
    model(image)['semantic_logits'].square().mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.semantic_lora.parameters())
    optimizer.step()
    after = model(image)
    assert torch.equal(after['affinity_logits'], initial['affinity_logits'])
    assert not torch.equal(after['semantic_logits'], initial['semantic_logits'])
    assert shared_frozen_digest(model) == frozen
    assert sum(p.numel() for p in model.parameters()) == model.parameter_summary()['total']


def test_checkpoint_recomputes_with_private_weights_after_functional_call_scope_exits():
    model = backend()
    with torch.no_grad():
        for parameter in model.semantic_lora.parameters():
            parameter.add_(.23)
    reference = deepcopy(model.encoder.trunk)
    for name, value in zip(model.semantic_lora.parameter_names, model.semantic_lora.weights):
        reference.get_parameter(name).data.copy_(value)
        reference.get_parameter(name).requires_grad_(True)
    image = torch.rand(1, 3, 5, 7)
    actual, expected = semantic_features(model, image)[0], reference(image)[0]
    assert torch.equal(actual, expected)
    # 先autograd.grad再backward，覆盖正式梯度预算的两次反传。
    first = torch.autograd.grad(actual.square().mean(), tuple(model.semantic_lora.parameters()), retain_graph=True)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for name, value, grad in zip(model.semantic_lora.parameter_names, model.semantic_lora.weights, first):
        torch.testing.assert_close(value.grad, reference.get_parameter(name).grad, rtol=0, atol=0)
        torch.testing.assert_close(grad, value.grad, rtol=0, atol=0)


def test_frozen_private_lora_has_no_graph_and_is_same_as_original_branch():
    model = backend()
    model.semantic_lora.requires_grad_(False)
    image = torch.rand(1, 3, 8, 8)
    features = semantic_features(model, image)[0]
    assert not features.requires_grad and torch.equal(features, model.encoder(image)[0])


def test_optional_features_follow_separate_lora_branches():
    model = backend()
    with torch.no_grad():
        model.semantic_lora.weights[1].add_(.2)
    image = torch.rand(1, 3, 8, 8)
    actual = model(image, return_features=True)
    assert torch.equal(actual['affinity_feature'], model.encoder(image)[0])
    assert torch.equal(actual['semantic_feature'], semantic_features(model, image)[0])
    assert not torch.equal(actual['semantic_feature'], actual['affinity_feature'])


@pytest.mark.parametrize('private', [False, True])
def test_new_and_old_bundles_strictly_roundtrip(monkeypatch, private):
    import models.fused_deployment as fused
    monkeypatch.setattr(fused, 'SAM2Encoder', TinyEncoder)
    monkeypatch.setattr(fused, 'inject_trunk_lora', lambda *args, **kwargs: None)
    monkeypatch.setattr(fused, '_semantic_scaffold', lambda *args: None)
    monkeypatch.setattr(fused, 'SemanticChallenger', TinySemantic)
    monkeypatch.setattr(fused, 'AffinityGeometryDecoder', TinyAffinity)
    model = backend()
    if not private:
        model.semantic_lora = None
    architecture = {'sam2': {'config_file': 'unused', 'sam2_repo_path': '.'},
                    'lora': {'rank': 2, 'alpha': 3}, 'semantic_decoder': {},
                    'affinity_decoder': {'affinity_channels': 2, 'fpn_channels': 4, 'up_channels': 4, 'output_grid': 8}}
    if private:
        architecture['semantic_lora'] = model.semantic_lora.architecture()
    bundle = {'format': FUSED_DEPLOYMENT_FORMAT, 'architecture': architecture, 'model_state_dict': model.state_dict()}
    actual = build_fused_model_from_bundle(bundle, {'paths': {'project_root': '.'}}, 'cpu')
    image = torch.rand(1, 3, 8, 8)
    for name, value in model(image).items():
        assert torch.equal(value, actual(image)[name])
    if private:
        bundle['model_state_dict'] = dict(bundle['model_state_dict'])
        bundle['model_state_dict'].pop('semantic_lora.weights.0')
        with pytest.raises(RuntimeError, match='Missing key'):
            build_fused_model_from_bundle(bundle, {'paths': {'project_root': '.'}}, 'cpu')


def test_rejects_wrong_adapter_version_or_parameter_order():
    model = backend()
    config = model.semantic_lora.architecture()
    with pytest.raises(ValueError, match='version'):
        SemanticLoRA.from_architecture(model.encoder.trunk, {**config, 'version': 'unknown'})
    with pytest.raises(ValueError, match='parameter names'):
        SemanticLoRA.from_architecture(model.encoder.trunk, {**config, 'parameter_names': config['parameter_names'][::-1]})


def test_experiment_changes_only_private_lora_and_output_from_gray4():
    old, new = load_config('config/train/semantic_gray4.yaml'), load_config('config/train/semantic_lora.yaml')
    assert lora_comparison_config(old) == lora_comparison_config(new)
    assert new['semantic_adaptation']['epochs'] == 20
    changed = deepcopy(new)
    changed['semantic_hard']['contrast'][0] = .1
    assert lora_comparison_config(old) != lora_comparison_config(changed)


def test_replay_detects_a_changed_input_or_selection():
    fields = ('epoch', 'step', 'updates', 'name', 'seed', 'draw_index', 'input_sha256',
              'appearance_input_sha256', 'appearance', 'restoration_seed', 'learning_rate')
    row = {name: 1 for name in fields}
    verify_lora_input(row, row)
    with pytest.raises(RuntimeError, match='seed'):
        verify_lora_input(row, {**row, 'seed': 2})
