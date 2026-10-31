# -*- coding: utf-8 -*-
"""真实热图模式的工厂、来源绑定和监控契约；只用CPU合成数据。"""
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from models.boundary_restore import BoundaryRestoreNet, load_boundary_restorer, state_sha256
from train_boundary_restore import (
    build_dataset, build_identity, canonical_sha, data_mode, load_resume,
    save_checkpoint, train, validate_recipe,
)
from utils.config import load_config


def config(name='boundary_real'):
    return load_config(f'config/train/{name}.yaml')


def test_explicit_real_mode_and_unchanged_default_budget():
    from data.boundary_real import DEFAULT_REAL_RECIPE
    old, real = config('boundary_restore'), config()
    assert data_mode(old) == 'synthetic_gt'
    assert data_mode(real) == 'real_n0_heatmaps'
    a, b = validate_recipe(old), validate_recipe(real)
    assert b['target_sigma'] is None and b['micro_batch_size'] == 4
    assert b['real_recipe'] == DEFAULT_REAL_RECIPE  # 配置必须写全实际配方，不依赖隐含默认。
    for key in ('seed', 'model', 'loss', 'epochs', 'schedule_epochs', 'updates_per_epoch',
                'effective_batch_size', 'views_per_source', 'expected_sources', 'precision',
                'learning_rate', 'weight_decay', 'monitor_epochs'):
        assert a[key] == b[key]
    assert b['epochs']*b['updates_per_epoch'] == 320
    bad = deepcopy(real)
    bad['boundary_restore']['data_mode'] = 'real_heat'
    with pytest.raises(ValueError, match='Unknown'):
        build_dataset(bad)
    del real['boundary_restore']['real_recipe']
    with pytest.raises(ValueError, match='explicit real_recipe'):
        validate_recipe(real)


def test_factory_uses_same_real_recipe_with_independent_monitor_seed(monkeypatch):
    import data.boundary_restore as synthetic
    import data.boundary_real as real
    calls = []
    def fake(kind):
        def constructor(manifest, **kwargs):
            calls.append((kind, manifest, kwargs))
            return SimpleNamespace(identity={'data_mode': kind})
        return constructor
    monkeypatch.setattr(synthetic, 'BoundaryRestoreDataset', fake('synthetic_gt'))
    monkeypatch.setattr(real, 'BoundaryRealDataset', fake('real_n0_heatmaps'))
    old, new = config('boundary_restore'), config()
    build_dataset(old)
    build_dataset(new)
    build_dataset(new, seed_offset=917)
    assert [r[0] for r in calls] == ['synthetic_gt', 'real_n0_heatmaps', 'real_n0_heatmaps']
    assert calls[0][2]['target_sigma'] == 2.
    assert 'target_sigma' not in calls[1][2]
    training, monitor = calls[1][2], calls[2][2]
    assert monitor['seed'] == training['seed']+917
    assert {k:v for k,v in training.items() if k!='seed'} == {k:v for k,v in monitor.items() if k!='seed'}
    assert training['recipe'] == new['boundary_restore']['real_recipe']
    assert training['recipe'] is not new['boundary_restore']['real_recipe']


def identity_fixture(tmp_path):
    cfg = config()
    manifest, real_manifest = tmp_path/'gt.json', tmp_path/'real.json'
    manifest.write_text('{"synthetic_source_only": true}')
    real_manifest.write_text('{"synthetic_cache_identity": "first"}')
    cfg['boundary_restore'].update(manifest=str(manifest), real_manifest=str(real_manifest))
    dataset = SimpleNamespace(identity={'data_mode':'real_n0_heatmaps',
        'real_map_sha256':'synthetic_cache_A', 'N0_protocol':'fixed_noise_v1'})
    model = BoundaryRestoreNet(width=4, seed=cfg['boundary_restore']['seed'])
    return cfg, dataset, model


def test_real_identity_binds_cache_protocol_recipe_and_source_files(tmp_path):
    cfg, dataset, model = identity_fixture(tmp_path)
    original = build_identity(model, dataset, cfg)
    assert original['identity_version'] == 2 and original['data_mode'] == 'real_n0_heatmaps'
    assert 'data/boundary_real.py' in original['runtime_sources']
    assert 'real_manifest_sha256' in original and 'target_sigma' not in original['recipe']
    dataset.identity['N0_protocol'] = 'different_protocol'
    assert canonical_sha(build_identity(model, dataset, cfg)) != canonical_sha(original)
    dataset.identity = deepcopy(original['data'])
    cfg['boundary_restore']['real_recipe']['gap_depth_range'] = [.5, .8]
    assert canonical_sha(build_identity(model, dataset, cfg)) != canonical_sha(original)
    from pathlib import Path
    Path(cfg['boundary_restore']['real_manifest']).write_text('{"synthetic_cache_identity": "second"}')
    assert build_identity(model, dataset, cfg)['real_manifest_sha256'] != original['real_manifest_sha256']
    dataset.identity['data_mode'] = 'synthetic_gt'
    with pytest.raises(ValueError, match='dataset provenance'):
        build_identity(model, dataset, cfg)


def test_synthetic_identity_keeps_legacy_key_schema(tmp_path):
    cfg = config('boundary_restore')
    path = tmp_path/'gt.json'
    path.write_text('{}')
    cfg['boundary_restore']['manifest'] = str(path)
    model = BoundaryRestoreNet(width=4)
    dataset = SimpleNamespace(identity={'synthetic_only':True})
    identity = build_identity(model, dataset, cfg)
    assert set(identity) == {'format','architecture','data','initial_state_sha256','recipe',
        'runtime_sources','monitor_recipe','monitor_asset_sha256','dataset_manifest_sha256',
        'initialization','test_gt_used','deployment_replaced','scoring_implemented'}
    assert 'target_sigma' in identity['recipe'] and 'real_recipe' not in identity['recipe']
    assert 'data/boundary_real.py' not in identity['runtime_sources']


def test_strict_resume_does_not_cross_supervision_modes_and_old_load_still_works(tmp_path):
    cfg, dataset, model = identity_fixture(tmp_path)
    identity = build_identity(model, dataset, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    path = tmp_path/'real.pt'
    save_checkpoint(path, model, optimizer, identity, 0, cfg['boundary_restore'], [0])
    fresh = BoundaryRestoreNet(width=4, seed=cfg['boundary_restore']['seed'])
    restored_opt = torch.optim.AdamW(fresh.parameters(), lr=3e-4)
    assert load_resume(path, fresh, restored_opt, identity, cfg['boundary_restore']) == (0, [0])
    wrong = deepcopy(identity)
    wrong['data_mode'] = 'synthetic_gt'
    with pytest.raises(ValueError, match='identity differs'):
        load_resume(path, fresh, restored_opt, wrong, cfg['boundary_restore'])
    # 历史格式权重仍严格加载；不为兼容性放宽missing/unexpected或resume身份。
    save_checkpoint(tmp_path/'old.pt', model, optimizer, {'historical_identity':True},
                    0, cfg['boundary_restore'], [0])
    assert state_sha256(load_boundary_restorer(tmp_path/'old.pt')) == state_sha256(model)


def test_fresh_real_training_rejects_previous_model_state(tmp_path):
    cfg, dataset, model = identity_fixture(tmp_path)
    with torch.no_grad():
        model.correction.bias.fill_(.2)
    with pytest.raises(ValueError, match='previous weights are forbidden'):
        train(model, dataset, cfg, tmp_path/'run', lambda *args: None)


def test_real_monitor_uses_factory_and_labels_target_as_teacher(tmp_path, monkeypatch):
    import train_boundary_restore as trainer
    import tools.boundary_restore_monitor as monitoring
    cfg = config()
    cfg['boundary_restore']['monitor'].update(pair_indices=[0], thumbnail_width=48)
    seen = []
    class PairDataset:
        def set_epoch(self, epoch):
            assert epoch == 1
        def __getitem__(self, index):
            assert index == 0
            value = torch.full((1, 32, 32), .13)
            return dict(input=value, target=value.clone(), valid=torch.ones_like(value),
                        gap_mask=torch.zeros_like(value), noise_mask=torch.zeros_like(value),
                        identity=torch.tensor(True), source='train_toy', meta={'teacher':'frozen_N0'})
    def factory(config, seed_offset=0):
        assert config is cfg and seed_offset == 917
        return PairDataset()
    monkeypatch.setattr(trainer, 'build_dataset', factory)
    original = monitoring.panels
    def panels(*args, **kwargs):
        seen.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(monitoring, 'panels', panels)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = BoundaryRestoreNet(width=4, seed=3).train()
        before = state_sha256(model)
        report = monitoring.monitor(model, cfg, tmp_path/'out', 0, 'cpu')
    finally:
        torch.set_num_threads(old_threads)
    assert model.training and state_sha256(model) == before
    assert seen[0][2][0][0] == 'Original N0 teacher / NOT GT'
    assert report['data_mode'] == 'real_n0_heatmaps'
    row = report['rows'][0]
    assert row['target_kind'] == 'frozen_N0_teacher_response'
    assert row['valid_before_mae'] == row['valid_after_mae'] == 0.
    assert (tmp_path/'out/monitor/e000/pair_000.jpg').is_file()
    json.dumps(report, allow_nan=False)
