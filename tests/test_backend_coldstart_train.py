# -*- coding: utf-8 -*-
"""冷启动simple的实际分辨率损失与两阶段预算。"""
import copy

import pytest
import torch

from train_backend_coldstart import phase_plan, simple_semantic_loss
from train_direct_semantic_affinity import build_semantic_criterion
from utils.config import load_config


def test_simple_stride4_loss_backpropagates_against_full_grid_instances():
    config = load_config('config/train/backend_cold.yaml')
    criterion = build_semantic_criterion(config, torch.device('cpu'))
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros(1, 1, 32, 32)
    target[..., 16:] = 1
    ids = torch.ones(1, 32, 32, dtype=torch.long)
    ids[..., 16:] = 2
    loss = simple_semantic_loss(criterion, logits, {
        'semantic_target': target, 'semantic_boundary': torch.zeros_like(target),
        'semantic_instance_map': ids,
    })
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[..., :3].mean() > 0 and logits.grad[..., -3:].mean() < 0


def test_cold_schedule_includes_frozen_warmup_and_joint_without_holdout():
    config = load_config('config/train/backend_cold.yaml')
    assert phase_plan(config) == [('head_warmup', 60, False), ('joint', 60, True)]
    assert phase_plan(config, True) == [('head_warmup', 1, False), ('joint', 1, True)]
    broken = copy.deepcopy(config)
    broken['backend_adaptation']['epochs'] = 60
    with pytest.raises(ValueError, match='matching epochs'):
        phase_plan(broken)
