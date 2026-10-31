# -*- coding: utf-8 -*-
"""代表性尾部训练入口的单变量、私有作用域及真实三更新集成检查。"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import train_affinity_fused as parent
import train_affinity_native as original
import train_affinity_tail as trainer
from train_affinity_balance import comparable
from train_backend_adaptation import fusion_options
from train_direct_semantic_affinity import compute_affinity_loss
from utils.affinity_fused_loss import fused_boundary_ranking_loss
from utils.affinity_loss import build_affinity_targets_torch
from utils.affinity_tail_loss import SAMPLING_FORMAT, tail_boundary_ranking_loss
from utils.config import load_config

# 只复用已封存测试的简单模型、batch与读日志工具；不执行或编辑其测试。
_path = Path(__file__).with_name('test_affinity_fused_training.py')
_spec = importlib.util.spec_from_file_location('_tail_test_fused_fixtures', _path)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
SmallModel, rows = _fixtures.SmallModel, _fixtures.rows


def config():
    return load_config(trainer.CONFIG_PATH)


def batch(pseudo=False):
    data = _fixtures.batch(pseudo)
    # 内部10%超过256点，真实集成必须走新采样而非只测未cap兼容分支。
    for key in ('affinity_instance_map', 'affinity_valid_content', 'instance_map', 'valid_content'):
        data[key] = data[key].repeat_interleave(3, dim=-2).repeat_interleave(3, dim=-1)
    return data


def functions(module):
    return tuple(getattr(module, name) for name in
                 ('compute_affinity_loss', 'configure_training', 'append_json'))


def test_only_sampler_changes_and_old_native_fused_namespaces_remain_independent():
    cfg = config()
    expected = load_config(parent.CONFIG_PATH)
    candidate = deepcopy(cfg)
    candidate['affinity_fused'].pop('sampling')
    candidate['affinity_fused']['format'] = parent.FORMAT
    assert comparable(candidate) == comparable(expected)
    base = trainer.reference_config(cfg)
    mixed = load_config('config/train/affinity_balance_mixed.yaml')
    base['backend_adaptation']['output_dir'] = mixed['backend_adaptation']['output_dir']
    assert base == mixed
    assert base['backend_adaptation']['completed_gt_dir'].endswith('radius_8/derived_maps')
    assert trainer._core is not parent._core and trainer._core is not original
    assert trainer._fused is not parent
    assert trainer._core.train.__globals__ is trainer._core.__dict__
    assert trainer.ranking_objective.__wrapped__.__globals__ is trainer._fused.__dict__
    assert trainer._fused.fused_boundary_ranking_loss is tail_boundary_ranking_loss
    assert parent.fused_boundary_ranking_loss is fused_boundary_ranking_loss


@pytest.mark.parametrize('mutation', ['weight', 'margin', 'cap', 'gt', 'gt_hash', 'pseudo',
                                     'view_seed', 'fusion', 'sampler', 'format', 'init'])
def test_reference_rejects_hidden_second_variables(mutation):
    cfg = config()
    if mutation == 'weight': cfg['affinity_fused']['weight'] = .2
    elif mutation == 'margin': cfg['affinity_fused']['margin'] = .2
    elif mutation == 'cap': cfg['affinity_fused']['max_per_group'] = 128
    elif mutation == 'gt': cfg['backend_adaptation']['completed_gt_dir'] = 'data/old_gt'
    elif mutation == 'gt_hash': cfg['affinity_fused']['gt_npz_cohort_sha256'] = 'b'*64
    elif mutation == 'pseudo': cfg['backend_adaptation']['pseudo_weight'] = 1.
    elif mutation == 'view_seed': cfg['affinity_native']['mixed_view_seed'] += 1
    elif mutation == 'fusion': cfg['affinity_deployment']['fusion_mode'] = 'short'
    elif mutation == 'sampler': cfg['affinity_fused']['sampling'] = 'extreme_top256'
    elif mutation == 'format': cfg['affinity_fused']['format'] = parent.FORMAT
    else: cfg['affinity_deployment']['checkpoint'] = trainer.REFERENCE+'/final.pt'
    with pytest.raises(ValueError):
        trainer.reference_config(cfg)


def test_three_adamw_updates_equal_real_bce_plus_tail_and_pseudo_is_unchanged(tmp_path):
    cfg = config()
    torch.manual_seed(117)
    model = SmallModel()
    expected, hooked = deepcopy(model), deepcopy(model)
    opts = [torch.optim.AdamW(m.affinity_decoder.parameters(), lr=2.e-5, eps=1.e-4)
            for m in (expected, hooked)]
    parent_functions, original_functions = functions(parent._core), functions(original)
    private_functions = functions(trainer._core)
    parent_provider = parent.fused_boundary_ranking_loss
    before_states = deepcopy(hooked.state_dict())
    mode, kwargs = fusion_options(cfg)
    fusion = dict(mode=mode, **kwargs)
    generator = torch.Generator().manual_seed(118)
    images = [(torch.randn(1, 2, 108, 108, generator=generator),
               torch.randn(1, 2, 108, 108, generator=generator)) for _ in range(3)]
    raw_manual, totals, rankings, pseudo_values = [], [], [], []
    with trainer.ranking_objective(cfg, tmp_path, smoke=True) as (receipts, extras):
        trainer._core.configure_training(hooked)
        original.configure_training(expected)
        for update, (manual_image, pseudo_image) in enumerate(images, 1):
            for opt in opts: opt.zero_grad(set_to_none=True)
            for pseudo, image in [(False, manual_image), (True, pseudo_image)]:
                data = batch(pseudo)
                actual_logits, expected_logits = hooked(image), expected(image)
                base, base_info = compute_affinity_loss(expected_logits, data,
                    cfg['direct_semantic_affinity']['affinity_loss'], pseudo=pseudo)
                target, valid = build_affinity_targets_torch(data['affinity_instance_map'],
                                                           data['affinity_valid_content'])
                if pseudo:
                    explicit = base
                    n_before = len(receipts)
                    pseudo_values.append(float(base.detach()))
                else:
                    auxiliary, metrics = tail_boundary_ranking_loss(expected_logits, target, valid,
                                                                    fusion_kwargs=fusion)
                    assert metrics['groups'][0]['tail_deep_interior'] > 256
                    if update == 1:
                        legacy, _ = fused_boundary_ranking_loss(expected_logits, target, valid,
                                                               fusion_kwargs=fusion)
                        assert not torch.equal(auxiliary, legacy)
                    explicit = base+.1*auxiliary
                    raw_manual.append(float(base.detach()))
                    totals.append(float(explicit.detach()))
                    rankings.append(.1*float(auxiliary.detach()))
                actual, actual_info = trainer._core.compute_affinity_loss(actual_logits, data,
                    cfg['direct_semantic_affinity']['affinity_loss'], pseudo=pseudo)
                assert torch.equal(actual, explicit) and actual_info.keys() == base_info.keys()
                # 只读probe不得积累参数.grad或改变输入/输出；实际backward紧随其后。
                if not pseudo:
                    assert all(p.grad is None for p in hooked.parameters())
                gradient, = torch.autograd.grad(actual, actual_logits, retain_graph=True)
                assert gradient[~valid].count_nonzero() == 0
                coefficient = .5 if pseudo else 1.
                (actual*coefficient).backward()
                (explicit*coefficient).backward()
                if pseudo: assert len(receipts) == n_before
            for left, right in zip(hooked.parameters(), expected.parameters()):
                assert torch.equal(left.grad, right.grad)
            grad = float(torch.nn.utils.clip_grad_norm_(hooked.affinity_decoder.parameters(), 1.))
            torch.nn.utils.clip_grad_norm_(expected.affinity_decoder.parameters(), 1.)
            trainer._core.append_json(tmp_path/'steps.jsonl',dict(epoch=1,step=update,updates=update,
                manual_bce=totals[-1],pseudo_bce=pseudo_values[-1],grad_norm=grad))
            opts[0].step(); opts[1].step()
            assert all(torch.equal(a, b) for a, b in zip(hooked.parameters(), expected.parameters()))
        trainer._core.append_json(tmp_path/'epochs.jsonl',dict(epoch=1,
            manual_bce=float(np.mean(totals)),pseudo_bce=float(np.mean(pseudo_values))))
        snapshot = extras(1)
        assert snapshot['fused_experiment']['format'] == trainer.FORMAT
        assert snapshot['fused_experiment']['options']['sampling'] == SAMPLING_FORMAT
        assert snapshot['fused_experiment']['manual_updates'] == 3
        assert snapshot['rng_state'].keys() == {'python','numpy','torch','cuda'}
    assert functions(trainer._core) == private_functions
    assert functions(parent._core) == parent_functions and functions(original) == original_functions
    assert parent.fused_boundary_ranking_loss is parent_provider
    assert torch.equal(hooked.frozen_sentinel, before_states['frozen_sentinel'])
    records = rows(tmp_path/'steps.jsonl')
    assert [r['manual_bce'] for r in records] == raw_manual
    assert [r['manual_total'] for r in records] == totals
    assert [r['weighted_ranking'] for r in records] == rankings
    assert [r['pseudo_bce'] for r in records] == pseudo_values
    sidecar = rows(tmp_path/'fused_loss.jsonl')
    assert len(sidecar) == 3 and all(r['ranking']['sampling_format'] == SAMPLING_FORMAT for r in sidecar)
    assert all(r['gradient']['unknown_logit_gradient_max'] == 0 for r in sidecar)
    assert all(r['gradient']['directional_probe_only'] is True for r in sidecar)
    assert all(1 <= r['gradient']['directional_probe_attempts'] <= 7 for r in sidecar)
    epoch = rows(tmp_path/'epochs.jsonl')[0]
    assert epoch['manual_bce'] == float(np.mean(raw_manual))
    assert epoch['manual_total'] == float(np.mean(totals))
    assert epoch['weighted_ranking'] == float(np.mean(rankings))


def test_exception_restores_private_functions_and_lambda_zero_is_only_conceptual(tmp_path):
    cfg = config()
    cfg['affinity_fused']['weight'] = 0.
    model = SmallModel()
    initial_functions = functions(trainer._core)
    with pytest.raises(RuntimeError, match='intentional interruption'):
        with trainer.ranking_objective(cfg, tmp_path, smoke=True):
            trainer._core.configure_training(model)
            logits = model(torch.zeros(1, 2, 108, 108))
            data = batch()
            actual, _ = trainer._core.compute_affinity_loss(logits, data,
                cfg['direct_semantic_affinity']['affinity_loss'], pseudo=False)
            base, _ = compute_affinity_loss(logits, data,
                cfg['direct_semantic_affinity']['affinity_loss'], pseudo=False)
            assert torch.equal(actual, base)
            a, = torch.autograd.grad(actual, model.affinity_decoder.affinity_head[0].weight,retain_graph=True)
            b, = torch.autograd.grad(base, model.affinity_decoder.affinity_head[0].weight)
            assert torch.equal(a,b)
            raise RuntimeError('intentional interruption')
    assert functions(trainer._core) == initial_functions
    with pytest.raises(ValueError): trainer.reference_config(cfg)


def test_checkpoint_rng_capture_and_context_do_not_consume_any_random_stream(tmp_path):
    before = trainer._rng_state()
    with trainer.ranking_objective(config(),tmp_path,smoke=False) as (_,extras):
        snapshot = extras(1)
    after = trainer._rng_state(); saved = snapshot['rng_state']
    assert before['python'] == after['python'] == saved['python']
    for data in (after,saved):
        assert before['numpy'][0] == data['numpy'][0]
        assert np.array_equal(before['numpy'][1],data['numpy'][1])
        assert before['numpy'][2:] == data['numpy'][2:]
        assert torch.equal(before['torch'],data['torch'])
        assert len(before['cuda']) == len(data['cuda'])
        assert all(torch.equal(a,b) for a,b in zip(before['cuda'],data['cuda']))
    random.Random().setstate(saved['python'])
    np.random.RandomState().set_state(saved['numpy'])
    torch.Generator().set_state(saved['torch'])
    assert snapshot['fused_experiment']['format'] == trainer.FORMAT == 'affinity_tail_ranking_v1'
    assert snapshot['fused_experiment']['options']['sampling'] == SAMPLING_FORMAT
