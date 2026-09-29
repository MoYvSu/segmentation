# -*- coding: utf-8 -*-
"""接续实验必须保留旧权重、部署合同与A/B共有的优化预算。"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from train_semantic_consistency import (
    continue_simple, consistency_weight, unlabeled_rng, verify_contract,
)
from train_semantic_d5a import frozen_digest, learning_rate_factor, semantic_train_mode
from utils.config import load_config


class SmallSemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_fpn = nn.Sequential(nn.Conv2d(3, 4, 1), nn.GroupNorm(2, 4), nn.ReLU())
        self.seg_branch = nn.Sequential(nn.Dropout2d(.25), nn.Conv2d(4, 1, 1))
        self.semantic_residual = None

    def forward(self, features, image=None):
        return self.seg_branch(self.seg_fpn(features))


class SmallBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        self.encoder.trainable_lora = True
        self.encoder.register_parameter('lora_A', nn.Parameter(torch.ones(1, 3)))
        self.affinity_decoder = nn.Conv2d(3, 4, 1)
        self.semantic_decoder = SmallSemantic()


def saved_simple():
    return {
        'config': load_config('config/train/semantic_d5a.yaml'),
        'backend_adaptation': {'semantic_adaptation': {'arm': 'simple'}},
    }


def test_warmstart_preserves_all_tensors_and_rng_then_updates_only_semantics():
    model = SmallBackend()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    original_frozen = frozen_digest(model)
    continue_simple(model)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert not model.encoder.trainable_lora
    assert all(p.requires_grad == name.startswith('semantic_decoder.')
               for name, p in model.named_parameters())

    teacher = deepcopy(model.semantic_decoder).eval().requires_grad_(False)
    teacher_before = {name: value.clone() for name, value in teacher.state_dict().items()}
    semantic_train_mode(model)
    assert model.semantic_decoder.training
    assert not model.encoder.training and not model.affinity_decoder.training
    assert not teacher.training
    optimizer = torch.optim.SGD(model.semantic_decoder.parameters(), lr=.1)
    with torch.no_grad():
        features = model.encoder(torch.ones(1, 3, 6, 6))
    model.semantic_decoder(features).mean().backward()
    optimizer.step()

    assert frozen_digest(model) == original_frozen
    assert any(not torch.equal(value, before['semantic_decoder.' + name])
               for name, value in model.semantic_decoder.state_dict().items())
    assert all(torch.equal(value, teacher_before[name]) for name, value in teacher.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())


@pytest.mark.parametrize('unexpected', ['rgb_residual', 'batchnorm'])
def test_warmstart_rejects_a_different_semantic_architecture(unexpected):
    model = SmallBackend()
    if unexpected == 'rgb_residual':
        model.semantic_decoder.semantic_residual = nn.Conv2d(3, 1, 1)
    else:
        model.semantic_decoder.seg_fpn[1] = nn.BatchNorm2d(4)
    with pytest.raises(ValueError, match='RGB residual|BatchNorm'):
        continue_simple(model)


def test_contract_allows_training_seed_monitor_and_output_changes():
    config = load_config('config/train/semantic_consistency.yaml')
    config['inference']['output_dir'] = 'outputs/another_run'
    verify_contract(config, saved_simple())
    assert config['semantic_adaptation']['seed'] != saved_simple()['config']['semantic_adaptation']['seed']
    assert config['semantic_adaptation']['monitor']['images'] != saved_simple()['config']['semantic_adaptation']['monitor']['images']


@pytest.mark.parametrize(('path', 'value'), [
    (('affinity_deployment', 'short_reduction'), 'mean'),
    (('affinity_deployment', 'support_threshold'), .3),
    (('inference', 'boundary_threshold'), .6),
    (('inference', 'semantic_vote_mode'), 'hard_majority'),
    (('inference', 'min_instance_area'), 1),
    (('semantic_adaptation', 'restoration', 'inference_seed'), 27),
    (('semantic_adaptation', 'restoration', 'mode'), 'full_chain'),
    (('sam2', 'input_normalization'), 'imagenet'),
])
def test_contract_rejects_changes_that_can_alter_final_output(path, value):
    config = load_config('config/train/semantic_consistency.yaml')
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        verify_contract(config, saved_simple())


def test_contract_requires_original_simple_source():
    saved = saved_simple()
    saved['backend_adaptation']['semantic_adaptation']['arm'] = 'full'
    with pytest.raises(ValueError, match='original simple'):
        verify_contract(load_config('config/train/semantic_consistency.yaml'), saved)


def test_twenty_epoch_budget_and_shared_labeled_recipe():
    baseline = load_config('config/train/semantic_d5a.yaml')
    config = load_config('config/train/semantic_consistency.yaml')
    shared = config['semantic_adaptation']
    assert shared['epochs'] == 20
    assert shared['expected_manual_sources'] * shared['manual_repeats'] * shared['epochs'] == 1280
    for key in ('manual_repeats', 'expected_manual_sources', 'completed_gt_dir', 'noise',
                'restoration', 'weight_decay', 'grad_clip', 'amp_dtype'):
        assert shared[key] == baseline['semantic_adaptation'][key]
    assert config['direct_semantic_affinity']['semantic_loss'] == baseline['direct_semantic_affinity']['semantic_loss']
    assert shared.get('illumination') is None


def test_learning_rate_and_consistency_ramp_match_the_twenty_epoch_trial():
    config = load_config('config/train/semantic_consistency.yaml')
    shared, extra = config['semantic_adaptation'], config['semantic_consistency']
    rates = [shared['learning_rates']['simple'] * learning_rate_factor(
        epoch, shared['epochs'], shared['warmup_epochs'], shared['minimum_lr_ratio'])
        for epoch in range(1, 21)]
    assert rates[:3] == pytest.approx([5e-6, 1e-5, 1e-5])
    assert rates[-1] == pytest.approx(1e-6)
    assert all(left >= right for left, right in zip(rates[2:], rates[3:]))
    weights = [consistency_weight(epoch, extra['maximum_weight'], extra['ramp_epochs'])
               for epoch in range(1, 21)]
    assert weights[:5] == pytest.approx([.02, .04, .06, .08, .1])
    assert weights[5:] == pytest.approx([.1] * 15)


def test_extra_zero_weight_forward_preserves_next_supervised_dropout_and_gradients():
    control = SmallSemantic().train()
    candidate = deepcopy(control)
    features = torch.linspace(-1, 1, 3 * 6 * 6).reshape(1, 3, 6, 6)
    torch.manual_seed(413)
    first_control = control(features)
    first_control.square().mean().backward()
    second_control = control(features)
    second_control.square().mean().backward()
    rng_after_control = torch.get_rng_state().clone()

    torch.manual_seed(413)
    first_candidate = candidate(features)
    first_candidate.square().mean().backward()
    rng_before_extra = torch.get_rng_state().clone()
    # 使用训练器实际调用的隔离入口；即使U分支经过Dropout和反向，也不能影响监督流。
    with unlabeled_rng(20260927, torch.device('cpu')):
        extra = candidate(features * .8)
        (0. * extra.square().mean()).backward()
        torch.rand(127)
    assert torch.equal(torch.get_rng_state(), rng_before_extra)
    second_candidate = candidate(features)
    second_candidate.square().mean().backward()
    assert torch.equal(first_candidate, first_control)
    assert torch.equal(second_candidate, second_control)
    assert torch.equal(torch.get_rng_state(), rng_after_control)
    for reference, actual in zip(control.parameters(), candidate.parameters()):
        assert torch.equal(reference.grad, actual.grad)


def test_unlabeled_rng_restores_state_when_a_forward_raises():
    torch.manual_seed(7001)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='synthetic forward failure'):
        with unlabeled_rng(81, 'cpu'):
            torch.rand(25)
            raise RuntimeError('synthetic forward failure')
    assert torch.equal(torch.get_rng_state(), before)
