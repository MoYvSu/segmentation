# -*- coding: utf-8 -*-
"""新补边入口的放行条件、独立学习率和续训合同。"""
import copy
import json

import pytest
import torch

from train_boundary_rgb import (
    build_models, capture_rng, load_checkpoint, optimizers, rates, save_checkpoint,
    scalar_audit, smoke_checks, validate_config,
)
from utils.config import load_config


def config():
    return load_config('config/train/boundary_rgb.yaml')


def test_paired_recipes_only_change_input_and_output():
    a, b = config()['boundary_naf'], load_config('config/train/boundary_rgb_control.yaml')['boundary_naf']
    validate_config({'boundary_naf': a})
    validate_config({'boundary_naf': b})
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for item in (a, b):
        item.pop('input_mode')
        item.pop('output_dir')
        item['inference'].pop('input_mode')
    assert a == b


def test_both_optimizers_keep_distinct_paper_rates():
    cfg = config()['boundary_naf']
    cfg['model'] = dict(width=2, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    cfg['discriminator'] = dict(width=2)
    g, d = build_models(cfg, 'cpu')
    go, do = optimizers(g, d, cfg)
    assert go.param_groups[0]['lr'] == .001
    assert do.param_groups[0]['lr'] == .0001
    assert rates(999, cfg)[1] == .0001
    assert rates(999, cfg)[0] > .0009


def test_exact_bce_and_binary_only_mode_cannot_silently_enter():
    cfg = config()
    cfg['boundary_naf']['training']['bce_weight'] = 1
    with pytest.raises(ValueError, match='exact BCE'):
        validate_config(cfg)
    cfg = config()
    cfg['boundary_naf']['inference']['input_mode'] = 'binary'
    with pytest.raises(ValueError, match='真实输入固定'):
        validate_config(cfg)


def test_width_representation_must_match_at_train_and_inference():
    cfg = load_config('config/train/boundary_width.yaml')
    assert validate_config(cfg)['input_normalization']['kernel_size'] == 8
    cfg['boundary_naf']['inference']['input_normalization']['kernel_size'] = 7
    with pytest.raises(ValueError, match='标准化必须相同'):
        validate_config(cfg)


def metrics(loss=1., missing=.5, supported=.9, interior=.7):
    return dict(loss=loss, missing_points=100, missing_probability=missing,
                supported_points=100, supported_probability=supported,
                recorded_interior_input_points=50, recorded_interior_input_probability=interior)


def test_background_only_improvement_cannot_pass_smoke():
    checks = smoke_checks(metrics(), metrics(loss=.3, missing=.4, supported=.8, interior=.1), True, 20)
    assert checks['reconstruction_reduced']
    assert not checks['missing_progress']
    assert not checks['supported_progress']


def test_smoke_requires_real_gap_and_complete_line_preservation():
    checks = smoke_checks(metrics(), metrics(loss=.3, missing=.8, supported=.95, interior=.2), True, 20)
    assert all(checks.values())
    before = metrics()
    before['missing_points'] = 0
    assert not all(smoke_checks(before, metrics(loss=.3), True, 20).values())
    assert not all(smoke_checks(metrics(), metrics(loss=.3, missing=.8, interior=.2), False, 20).values())


def test_loss_diagnostic_json_does_not_keep_graph():
    value = torch.tensor(2., requires_grad=True)
    assert json.loads(json.dumps(scalar_audit({'loss': value, 'count': 3}))) == {'loss': 2., 'count': 3}


def test_checkpoint_restores_weights_optimizer_rng_and_rejects_wrong_recipe(tmp_path):
    cfg = config()['boundary_naf']
    cfg['model'] = dict(width=2, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    cfg['discriminator'] = dict(width=2)
    cfg['training']['minimum_free_gib'] = .001
    g, d = build_models(cfg, 'cpu')
    go, do = optimizers(g, d, cfg)
    x = torch.rand(1, 4, 16, 16)
    x[:, -1:] = (x[:, -1:] > .5).float()
    go.zero_grad()
    g(x).mean().backward()
    go.step()
    identity = {'recipe': copy.deepcopy(cfg)}
    save_checkpoint(tmp_path / 'latest.pt', g, d, go, do, cfg, identity, 1, [0, 1])
    saved_rng = capture_rng()['torch']
    expected = g(x).detach()
    fresh_g, fresh_d = build_models(cfg, 'cpu')
    fresh_go, fresh_do = optimizers(fresh_g, fresh_d, cfg)
    assert load_checkpoint(tmp_path / 'latest.pt', fresh_g, fresh_d,
                           fresh_go, fresh_do, cfg, identity) == (1, [0, 1])
    assert torch.equal(expected, fresh_g(x).detach())
    assert torch.equal(saved_rng, capture_rng()['torch'])
    assert len(fresh_go.state) == len(go.state) > 0
    with pytest.raises(ValueError, match='来源或源码'):
        load_checkpoint(tmp_path / 'latest.pt', fresh_g, fresh_d, fresh_go, fresh_do, cfg, {'different': True})
