# -*- coding: utf-8 -*-
"""NAF训练入口的合成集成检查：来源身份、阶段日程与G/D优化器真实接续。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.boundary_naf import BoundaryNAFDataset, collate_boundary_naf
from models.boundary_restore import state_sha256
import train_boundary_naf as training
from utils.boundary_naf_training import gan_update
from utils.config import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fixture(tmp_path):
    # 只构造一张合成实例图，以测试实际dataset/manifest接口；不是正式32图预检。
    shape = (112, 128)
    ids = np.ones(shape, dtype=np.int32)
    ids[:, 64:] = 2
    ids[56:, :] += 2
    image = tmp_path / 'train_001.jpg'
    image.write_bytes(b'synthetic-provenance-only-no-real-image')
    gt = tmp_path / 'train_001_gt.npz'
    known = np.ones(shape, dtype=bool)
    np.savez_compressed(gt, instance_map=ids, original_covered=known,
                        filled=np.zeros(shape, dtype=bool), residual_unknown=~known)
    classes = tmp_path / 'train_001_class.json'
    classes.write_text(json.dumps({'1': 1, '2': 0, '3': 1, '4': 0}), encoding='utf-8')
    files = dict(image=image, gt=gt, classes=classes)
    row = dict(source='train_001', **{key: str(path) for key, path in files.items()},
               hashes={key + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                       for key, path in files.items()})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'rows': [row]}), encoding='utf-8')
    fixed = tmp_path / 'fixed.json'
    fixed.write_text(json.dumps({'synthetic': True}), encoding='utf-8')
    config = load_config('config/train/boundary_naf.yaml')
    cfg = config['boundary_naf']
    cfg['manifest'] = str(manifest)
    cfg['model'] = dict(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    cfg['discriminator'] = dict(width=4)
    cfg['data'].update(expected_sources=1, crop_size=96, samples_per_epoch=8)
    cfg['training'].update(main_updates=2, topology_updates=2, main_schedule_updates=8,
                           topology_schedule_updates=4, segment_updates=4, effective_batch_size=2,
                           micro_batch_size=1, checkpoint_every_updates=1, num_workers=0)
    cfg['monitor']['control_manifests'] = [str(fixed)]
    dataset = BoundaryNAFDataset(manifest, config, epoch=0)
    return config, dataset


def _build(config):
    cfg = config['boundary_naf']
    generator, discriminator = training.build_models(cfg, 'cpu')
    return (generator, discriminator, training.optimizer_for(generator, cfg),
            training.optimizer_for(discriminator, cfg))


def _topology(probability, target, valid):
    # 此处仅验证阶段callback和checkpoint轨迹；实际PH算法由其专门测试覆盖。
    sample = ((probability - target).square() * valid).flatten(1).sum(1)
    count = valid.flatten(1).sum(1).clamp_min(1)
    return (sample / count).mean(), dict(critical_points=int(valid.shape[0]))


def _update(models, data, step, cfg):
    generator, discriminator, go, do = models
    phase, _, lr = training.phase_and_rate(step, cfg)
    for optimizer in (go, do):
        for group in optimizer.param_groups:
            group['lr'] = lr
    return gan_update(generator, discriminator, go, do, data, micro_batch_size=1,
                      topology_fn=_topology if phase == 'topology' else None,
                      topology_weight=0.001 if phase == 'topology' else 0)


def _assert_nested_equal(first, second):
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second):
            _assert_nested_equal(a, b)
    else:
        assert first == second


def test_source_identity_is_epoch_invariant_and_binds_gt_controls_and_ph_versions(tmp_path):
    config, dataset = _fixture(tmp_path)
    g, d, _, _ = _build(config)
    first = training.source_identity(config, dataset, g, d)
    initial_plan = dataset.plan_receipt()
    dataset.set_epoch(3)
    second = training.source_identity(config, dataset, g, d)
    assert first == second
    assert initial_plan['plan_sha256'] != dataset.plan_receipt()['plan_sha256']
    assert first['target'] == 'processed_GT_binary_boundaries_not_N0'
    assert first['frozen_backend_not_loaded_for_training'] and first['no_test_training']
    assert first['no_revert_unconnected']
    assert first['library_versions']['gudhi'] and first['library_versions']['scipy']
    assert first['source_sha256']['utils/boundary_naf_training.py']
    assert first['input_manifest_sha256'] == hashlib.sha256(
        Path(config['boundary_naf']['manifest']).read_bytes()).hexdigest()
    control = Path(config['boundary_naf']['monitor']['control_manifests'][0])
    control.write_text(json.dumps({'synthetic': True, 'changed': True}), encoding='utf-8')
    assert training.source_identity(config, dataset, g, d) != first


def test_phase_lr_starts_new_topology_schedule_instead_of_main_schedule(tmp_path):
    config, _ = _fixture(tmp_path)
    cfg = config['boundary_naf']
    main, local, rate = training.phase_and_rate(0, cfg)
    assert (main, local) == ('main', 0)
    assert rate == cfg['training']['learning_rate']
    assert training.phase_and_rate(1, cfg)[2] < rate
    topology, local, rate = training.phase_and_rate(2, cfg)
    assert (topology, local) == ('topology', 0)
    assert rate == cfg['training']['topology_learning_rate']
    assert training.phase_and_rate(3, cfg)[2] < rate


@pytest.mark.parametrize('adversarial_weight', [0.0, 0.2])
def test_smoke_honors_configured_losses_and_freezes_disabled_discriminator(tmp_path, monkeypatch,
                                                                         adversarial_weight):
    config, dataset = _fixture(tmp_path)
    cfg = config['boundary_naf']
    cfg['training'].update(bce_weight=2.75, adversarial_weight=adversarial_weight)
    monkeypatch.setattr(training, 'dataset_for', lambda *_: dataset)
    monkeypatch.setattr(training, 'topology_object', lambda _: _topology)
    monkeypatch.setattr(training, 'monitor', lambda *args, **kwargs: {'synthetic_monitor': True})
    result = training.smoke(config, tmp_path / 'smoke', device='cpu', updates=3)
    assert result['strict_reload_exact']
    for row in result['steps']:
        assert row['loss_g'] == pytest.approx(
            2.75 * row['bce'] + adversarial_weight * row['adv_g']
            + cfg['training']['topology_weight'] * row['topology_loss'], abs=1e-6)
        assert row['discriminator_updated'] is (adversarial_weight > 0)
    unchanged = result['discriminator_initial_sha256'] == result['discriminator_after_sha256']
    assert unchanged is (adversarial_weight == 0)


def test_smoke_purpose_cannot_resume_as_formal_even_with_identical_recipe(tmp_path):
    config, dataset = _fixture(tmp_path)
    models = _build(config)
    formal_identity = training.source_identity(config, dataset, models[0], models[1])
    assert formal_identity['purpose'] == 'formal_candidate'
    smoke_identity = deepcopy(formal_identity)
    smoke_identity['purpose'] = 'smoke_only_not_formal_resume'
    checkpoint = tmp_path / 'smoke.pt'
    training.save_checkpoint(checkpoint, *models, config['boundary_naf'], smoke_identity, 0, [])
    # 短测自己的严格重载仍允许，用途之外的结构、配置与计步完全相同。
    assert training.load_checkpoint(checkpoint, *models, smoke_identity,
                                    config['boundary_naf']) == (0, [])
    before_g, before_d = state_sha256(models[0]), state_sha256(models[1])
    with pytest.raises(ValueError, match='身份'):
        training.load_checkpoint(checkpoint, *models, formal_identity, config['boundary_naf'])
    assert state_sha256(models[0]) == before_g and state_sha256(models[1]) == before_d


@pytest.mark.parametrize('checkpoint_step', [1, 2])
def test_actual_adam_checkpoint_next_update_matches_both_g_and_d(tmp_path, checkpoint_step):
    config, dataset = _fixture(tmp_path)
    cfg = config['boundary_naf']
    reference = _build(config)
    identity = training.source_identity(config, dataset, reference[0], reference[1])
    data = collate_boundary_naf([dataset[0], dataset[1]])
    for step in range(checkpoint_step):
        _update(reference, data, step, cfg)
    checkpoint = tmp_path / 'state.pt'
    training.save_checkpoint(checkpoint, *reference, cfg, identity, checkpoint_step, [0])
    saved = torch.load(checkpoint, weights_only=False)
    assert saved['completed_phase'] == 'main'
    assert saved['next_phase'] == ('topology' if checkpoint_step == 2 else 'main')
    assert saved['next_scheduler'] == training.phase_and_rate(checkpoint_step, cfg)
    expected_rng = (random.random(), float(np.random.random()), float(torch.rand(())))
    expected_stats = _update(reference, data, checkpoint_step, cfg)

    resumed = _build(config)
    loaded_step, monitors = training.load_checkpoint(checkpoint, *resumed, identity, cfg)
    assert loaded_step == checkpoint_step and monitors == [0]
    assert expected_rng == (random.random(), float(np.random.random()), float(torch.rand(())))
    actual_stats = _update(resumed, data, checkpoint_step, cfg)
    assert expected_stats == actual_stats
    assert state_sha256(reference[0]) == state_sha256(resumed[0])
    assert state_sha256(reference[1]) == state_sha256(resumed[1])
    _assert_nested_equal(reference[2].state_dict(), resumed[2].state_dict())
    _assert_nested_equal(reference[3].state_dict(), resumed[3].state_dict())


@pytest.mark.parametrize('field,value', [
    ('global_step', -1), ('global_step', 5), ('global_step', True),
    ('next_phase', 'incorrect'), ('completed_phase', 'incorrect'),
    ('next_scheduler', ('main', 99, 0.001)), ('input_contract', {'wrong': True}),
    ('architecture', {'width': 99}), ('discriminator_architecture', {'width': 99}),
])
def test_resume_rejects_step_phase_schedule_or_interface_tamper(tmp_path, field, value):
    config, dataset = _fixture(tmp_path)
    models = _build(config)
    identity = training.source_identity(config, dataset, models[0], models[1])
    checkpoint = tmp_path / 'state.pt'
    training.save_checkpoint(checkpoint, *models, config['boundary_naf'], identity, 0, [0])
    payload = torch.load(checkpoint, weights_only=False)
    payload[field] = value
    torch.save(payload, checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        training.load_checkpoint(checkpoint, *models, identity, config['boundary_naf'])


def test_monitor_preserves_rng_mode_and_detects_weight_pollution(tmp_path, monkeypatch):
    config, dataset = _fixture(tmp_path)
    generator, _, _, _ = _build(config)
    generator.train()

    def harmless(model, *args, **kwargs):
        assert not model.training and not torch.is_grad_enabled()
        random.random(), np.random.random(), torch.rand(3)
        return {'synthetic_monitor': True}

    monkeypatch.setitem(sys.modules, 'tools.boundary_naf_monitor', SimpleNamespace(save_monitor=harmless))
    state = training.capture_rng()
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    training.restore_rng(state)
    before = state_sha256(generator)
    assert training.monitor(generator, config, tmp_path, 0, dataset)['synthetic_monitor']
    assert before == state_sha256(generator) and generator.training
    assert expected == (random.random(), float(np.random.random()), float(torch.rand(())))

    def polluted(model, *args, **kwargs):
        model.ending.bias.add_(0.1)
        return {}

    monkeypatch.setitem(sys.modules, 'tools.boundary_naf_monitor', SimpleNamespace(save_monitor=polluted))
    with pytest.raises(RuntimeError, match='monitor'):
        training.monitor(generator, config, tmp_path, 0, dataset)
    assert generator.training


def test_log_tail_is_preserved_before_checkpoint_replay(tmp_path):
    path = tmp_path / 'steps.jsonl'
    rows = [dict(global_step=index) for index in range(1, 6)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    training.reconcile_log(tmp_path, 3)
    assert [json.loads(line)['global_step'] for line in path.read_text().splitlines()] == [1, 2, 3]
    backups = list(tmp_path.glob('steps_after_s3_*.jsonl'))
    assert len(backups) == 1
    assert [json.loads(line)['global_step'] for line in backups[0].read_text().splitlines()] == [4, 5]
