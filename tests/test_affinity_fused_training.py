# -*- coding: utf-8 -*-
"""新GT融合排序入口：单变量隔离、真实损失、来源日志及续训随机状态。"""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import train_affinity_fused as trainer
import train_affinity_native as original
from train_backend_adaptation import fusion_options
from train_direct_semantic_affinity import compute_affinity_loss
from utils.affinity_fused_loss import fused_boundary_ranking_loss
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config


def config():
    return load_config(trainer.CONFIG_PATH)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.affinity_decoder = torch.nn.Module()
        self.affinity_decoder.affinity_head = torch.nn.Sequential(torch.nn.Conv2d(2, 8, 1))
        self.register_buffer('frozen_sentinel', torch.tensor([9.]))

    def forward(self, image):
        return self.affinity_decoder.affinity_head(image)


def batch(pseudo=False):
    ids = torch.ones((1, 36, 36), dtype=torch.long)
    ids[:, :, 18:] = 2
    ids[:, 8:11, 8:11] = 0  # 新GT中仍未知，不能转成背景负例。
    valid = torch.ones((1, 1, 36, 36), dtype=torch.bool)
    valid[:, :, -2:] = False
    return dict(affinity_instance_map=ids, affinity_valid_content=valid,
                instance_map=ids, valid_content=valid,
                uncovered_boundary_source=torch.tensor([False]),
                name=['pseudo.png' if pseudo else 'manual.png'])


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines()]


def test_reference_keeps_new_gt_original_init_and_every_parent_setting():
    cfg = config()
    expected = load_config('config/train/affinity_balance_mixed.yaml')
    base = trainer.reference_config(cfg)
    base['backend_adaptation']['output_dir'] = expected['backend_adaptation']['output_dir']
    assert base == expected
    assert base['backend_adaptation']['completed_gt_dir'].endswith('radius_8/derived_maps')
    assert base['direct_semantic_affinity']['affinity_loss']['manual_uncovered_as_boundary'] is False
    assert cfg['affinity_fused']['manual_only'] is True
    assert cfg['affinity_deployment']['checkpoint'] != trainer.REFERENCE + '/final.pt'
    assert trainer._core.train.__globals__ is trainer._core.__dict__
    assert original.train.__globals__ is original.__dict__


@pytest.mark.parametrize('mutation', ['old_gt', 'gt_hash', 'control', 'weight', 'extra', 'pseudo_ratio',
                                     'fusion', 'loss_bool'])
def test_reference_rejects_old_gt_or_hidden_experiment_changes(mutation):
    cfg = config()
    if mutation == 'old_gt':
        cfg['backend_adaptation']['completed_gt_dir'] = 'data/old_gt'
    elif mutation == 'gt_hash':
        cfg['affinity_fused']['gt_npz_cohort_sha256'] = 'a' * 64
    elif mutation == 'control':
        cfg['affinity_fused']['reference'] = 'outputs/other'
    elif mutation == 'weight':
        cfg['affinity_fused']['weight'] = .2
    elif mutation == 'extra':
        cfg['affinity_fused']['accidental_variable'] = True
    elif mutation == 'pseudo_ratio':
        cfg['backend_adaptation']['pseudo_weight'] = 1.
    elif mutation == 'fusion':
        cfg['affinity_deployment']['fusion_mode'] = 'short'
    else:
        cfg['direct_semantic_affinity']['affinity_loss']['negative_weight'] = True
    with pytest.raises(ValueError):
        trainer.reference_config(cfg)


def test_real_three_update_objective_gradients_logs_and_pseudo_are_exact(tmp_path):
    cfg = config()
    torch.manual_seed(72)
    initial = SmallModel()
    expected, hooked = deepcopy(initial), deepcopy(initial)
    expected_opt = torch.optim.AdamW(expected.affinity_decoder.parameters(), lr=2.e-5, eps=1.e-4)
    hooked_opt = torch.optim.AdamW(hooked.affinity_decoder.parameters(), lr=2.e-5, eps=1.e-4)
    old_functions = (original.compute_affinity_loss, original.configure_training, original.append_json)
    private_functions = (trainer._core.compute_affinity_loss, trainer._core.configure_training,
                         trainer._core.append_json)
    mode, fusion = fusion_options(cfg)
    fusion = dict(mode=mode, **fusion)
    images = [(torch.randn(1, 2, 36, 36), torch.randn(1, 2, 36, 36)) for _ in range(3)]
    manual_values, total_values, auxiliary_values, pseudo_values = [], [], [], []
    with trainer.ranking_objective(cfg, tmp_path, smoke=True) as (receipts, extras):
        trainer._core.configure_training(hooked)
        original.configure_training(expected)
        for update, (manual_image, pseudo_image) in enumerate(images, 1):
            hooked_opt.zero_grad(set_to_none=True)
            expected_opt.zero_grad(set_to_none=True)
            for pseudo, image in [(False, manual_image), (True, pseudo_image)]:
                b = batch(pseudo)
                actual_logits = hooked(image)
                expected_logits = expected(image)
                base, expected_details = compute_affinity_loss(expected_logits, b,
                    cfg['direct_semantic_affinity']['affinity_loss'], pseudo=pseudo)
                if pseudo:
                    explicit = base
                    before_count = len(receipts)
                else:
                    target, valid = build_affinity_targets_torch(b['affinity_instance_map'], b['affinity_valid_content'])
                    auxiliary, _ = fused_boundary_ranking_loss(expected_logits, target, valid, fusion_kwargs=fusion)
                    explicit = base + .1 * auxiliary
                    manual_values.append(float(base.detach()))
                    total_values.append(float(explicit.detach()))
                    auxiliary_values.append(.1 * float(auxiliary.detach()))
                actual, actual_details = trainer._core.compute_affinity_loss(actual_logits, b,
                    cfg['direct_semantic_affinity']['affinity_loss'], pseudo=pseudo)
                assert torch.equal(actual, explicit)
                assert actual_details.keys() == expected_details.keys()
                coefficient = .5 if pseudo else 1.
                gl, = torch.autograd.grad(actual, actual_logits, retain_graph=True)
                target, valid = build_affinity_targets_torch(b['instance_map'], b['valid_content'])
                assert gl[~valid].count_nonzero() == 0
                (actual * coefficient).backward()
                (explicit * coefficient).backward()
                if pseudo:
                    assert len(receipts) == before_count
                    pseudo_values.append(float(base.detach()))
            for actual_parameter, expected_parameter in zip(hooked.parameters(), expected.parameters()):
                if actual_parameter.grad is not None:
                    assert torch.equal(actual_parameter.grad, expected_parameter.grad)
            gn = float(torch.nn.utils.clip_grad_norm_(hooked.affinity_decoder.parameters(), 1.))
            torch.nn.utils.clip_grad_norm_(expected.affinity_decoder.parameters(), 1.)
            trainer._core.append_json(tmp_path/'steps.jsonl', dict(epoch=1, step=update, updates=update,
                manual_bce=total_values[-1], pseudo_bce=pseudo_values[-1], grad_norm=gn))
            hooked_opt.step()
            expected_opt.step()
            assert all(torch.equal(a, b) for a, b in zip(hooked.parameters(), expected.parameters()))
        trainer._core.append_json(tmp_path/'epochs.jsonl', dict(epoch=1,
            manual_bce=float(np.mean(total_values)), pseudo_bce=float(np.mean(pseudo_values))))
        snapshot = extras(1)
        assert snapshot['fused_experiment']['manual_updates'] == 3
    assert (original.compute_affinity_loss, original.configure_training, original.append_json) == old_functions
    assert (trainer._core.compute_affinity_loss, trainer._core.configure_training, trainer._core.append_json) == private_functions
    recorded = rows(tmp_path/'steps.jsonl')
    assert [r['manual_bce'] for r in recorded] == manual_values
    assert [r['manual_total'] for r in recorded] == total_values
    assert [r['weighted_ranking'] for r in recorded] == auxiliary_values
    assert [r['pseudo_bce'] for r in recorded] == pseudo_values
    assert len(rows(tmp_path/'fused_loss.jsonl')) == 3
    assert all(r['gradient']['unknown_logit_gradient_max'] == 0 for r in receipts)
    epoch = rows(tmp_path/'epochs.jsonl')[0]
    assert epoch['manual_bce'] == float(np.mean(manual_values))
    assert epoch['manual_total'] == float(np.mean(total_values))
    assert epoch['weighted_ranking'] == float(np.mean(auxiliary_values))


def test_zero_weight_compatibility_and_exception_restore_private_functions(tmp_path):
    cfg = config()
    cfg['affinity_fused']['weight'] = 0.  # 仅概念兼容检查，正式recipe仍拒绝lambda0。
    model = SmallModel()
    functions = (trainer._core.compute_affinity_loss, trainer._core.configure_training, trainer._core.append_json)
    with pytest.raises(RuntimeError, match='intentional interruption'):
        with trainer.ranking_objective(cfg, tmp_path, smoke=True):
            trainer._core.configure_training(model)
            image = torch.randn(1, 2, 36, 36)
            logits = model(image)
            b = batch()
            actual, _ = trainer._core.compute_affinity_loss(logits, b,
                cfg['direct_semantic_affinity']['affinity_loss'], pseudo=False)
            base, _ = compute_affinity_loss(logits, b,
                cfg['direct_semantic_affinity']['affinity_loss'], pseudo=False)
            assert torch.equal(actual, base)
            ga, = torch.autograd.grad(actual, model.affinity_decoder.affinity_head[0].weight, retain_graph=True)
            gb, = torch.autograd.grad(base, model.affinity_decoder.affinity_head[0].weight)
            assert torch.equal(ga, gb)
            raise RuntimeError('intentional interruption')
    assert (trainer._core.compute_affinity_loss, trainer._core.configure_training, trainer._core.append_json) == functions


@pytest.mark.parametrize('source_flag', [True, False])
def test_manual_uncovered_as_boundary_is_rejected_not_added_as_fake_supervision(tmp_path, source_flag):
    cfg = config()
    model, b = SmallModel(), batch()
    loss_cfg = deepcopy(cfg['direct_semantic_affinity']['affinity_loss'])
    if source_flag:
        b['uncovered_boundary_source'].fill_(True)
    else:
        loss_cfg['manual_uncovered_as_boundary'] = True
    with trainer.ranking_objective(cfg, tmp_path, smoke=True):
        trainer._core.configure_training(model)
        with pytest.raises(RuntimeError, match='unknown pairs ignored'):
            trainer._core.compute_affinity_loss(model(torch.randn(1, 2, 36, 36)), b, loss_cfg, pseudo=False)


def test_continuation_rng_snapshot_is_complete_and_does_not_consume_randomness(tmp_path):
    cfg = config()
    random.seed(93)
    np.random.seed(94)
    torch.manual_seed(95)
    before = trainer._rng_state()
    with trainer.ranking_objective(cfg, tmp_path, smoke=False) as (_, extras):
        snapshot = extras(1)['rng_state']
    after = trainer._rng_state()
    assert snapshot.keys() == {'python', 'numpy', 'torch', 'cuda'}
    assert before['python'] == snapshot['python'] == after['python']
    assert before['numpy'][0] == snapshot['numpy'][0] == after['numpy'][0]
    assert np.array_equal(before['numpy'][1], snapshot['numpy'][1])
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == snapshot['numpy'][2:] == after['numpy'][2:]
    assert torch.equal(before['torch'], snapshot['torch']) and torch.equal(before['torch'], after['torch'])
    # 临时生成器可重载，且不触碰调用者全局随机状态。
    random.Random().setstate(snapshot['python'])
    np.random.RandomState().set_state(snapshot['numpy'])
    torch.Generator().set_state(snapshot['torch'])
    assert len(before['cuda']) == len(snapshot['cuda']) == len(after['cuda'])
    assert all(torch.equal(a, b) for a, b in zip(before['cuda'], snapshot['cuda']))


@pytest.mark.parametrize('field', ['npz_cohort_sha256', 'class_cohort_sha256'])
def test_changed_actual_new_gt_is_rejected_before_entering_any_training(tmp_path, monkeypatch, field):
    cfg = config()
    actual = dict(npz_cohort_sha256=cfg['affinity_fused']['gt_npz_cohort_sha256'],
                  class_cohort_sha256=cfg['affinity_fused']['gt_class_cohort_sha256'])
    actual[field] = 'b' * 64
    calls = []
    monkeypatch.setattr(trainer, 'audit_new_gt', lambda _: actual)
    monkeypatch.setattr(trainer._core, 'train', lambda *a, **k: calls.append(True))
    with pytest.raises(RuntimeError, match='processed newGT files differ'):
        trainer.train(cfg, tmp_path, smoke=True)
    assert calls == []


def test_gt_or_source_change_during_run_is_rejected_after_private_context_restores(tmp_path, monkeypatch):
    cfg = config()
    actual = dict(npz_cohort_sha256=cfg['affinity_fused']['gt_npz_cohort_sha256'],
                  class_cohort_sha256=cfg['affinity_fused']['gt_class_cohort_sha256'], source_sha='same')
    observations = iter([actual, dict(actual, source_sha='changed')])
    monkeypatch.setattr(trainer, 'audit_new_gt', lambda _: next(observations))
    functions = (trainer._core.compute_affinity_loss, trainer._core.configure_training, trainer._core.append_json)
    calls = []
    def train_stub(config, size, output, **kwargs):
        assert size == 1024 and kwargs['dataset_builder'] is trainer.build_datasets
        assert kwargs['keep_smoke_checkpoint'] is True
        assert kwargs['control_verifier'] is trainer.verify_reference
        assert trainer._core.compute_affinity_loss is not functions[0]
        calls.append(True)
    monkeypatch.setattr(trainer._core, 'train', train_stub)
    with pytest.raises(RuntimeError, match='changed during training'):
        trainer.train(cfg, tmp_path, smoke=True)
    assert calls == [True]
    assert (trainer._core.compute_affinity_loss, trainer._core.configure_training, trainer._core.append_json) == functions
