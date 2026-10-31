# -*- coding: utf-8 -*-
"""平移一致性入口、监控与可续训合同；仅合成CPU数据。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import train_boundary_rgb as trainer
import tools.boundary_rgb_monitor as monitor
from models.boundary_restore import state_sha256
from test_boundary_erasure_training import batch, small_models
from utils.boundary_shift_consistency import shift_for_step
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return load_config('config/train/boundary_shift.yaml')


def test_recipe_only_adds_consistency_and_monitor_without_RGB_or_data_changes():
    current = config()
    trainer.validate_config(current)
    cfg = deepcopy(current['boundary_naf'])
    old = load_config('config/train/boundary_erasure.yaml')['boundary_naf']
    assert cfg.pop('shift_consistency') == dict(enabled=True, version='integer_shift_v1',
        weight=.1, margin=32, band_radius=8, shift_seed=20261029)
    assert cfg['monitor'].pop('shift_pair_indices') == [29, 63]
    cfg['output_dir'] = old['output_dir']
    assert cfg == old
    assert cfg['input_mode'] == 'boundary'
    assert cfg['training']['main_updates'] == 1000
    assert cfg['training']['effective_batch_size'] == cfg['training']['micro_batch_size'] == 16
    assert cfg['monitor']['full_readout_steps'] == [0, 1000]


@pytest.mark.parametrize('change', [
    lambda c: c['boundary_naf']['shift_consistency'].update(enabled='true'),
    lambda c: c['boundary_naf']['shift_consistency'].update(weight=0),
    lambda c: c['boundary_naf']['monitor'].update(shift_pair_indices=[99]),
    lambda c: c['boundary_naf']['monitor'].update(shift_pair_indices=[]),
    lambda c: c['boundary_naf'].update(erasure={'enabled': False}),
])
def test_invalid_or_unpaired_shift_recipe_rejected(change):
    cfg = config()
    change(cfg)
    with pytest.raises(ValueError):
        trainer.validate_config(cfg)


def test_identity_adds_only_explicit_shift_dependency_and_keeps_random_G_D(monkeypatch):
    current = config()
    old = load_config('config/train/boundary_erasure.yaml')
    cfg, g, d, _, _ = small_models(current['boundary_naf'])
    _, old_g, old_d, _, _ = small_models(old['boundary_naf'])
    monkeypatch.setattr(trainer, 'file_sha', lambda p: str(p))
    dataset = SimpleNamespace(identity={'synthetic': True})
    identity = trainer.source_identity(current, dataset, g, d)
    control = trainer.source_identity(old, dataset, old_g, old_d)
    assert set(identity['source_sha256']) - set(control['source_sha256']) == {'utils/boundary_shift_consistency.py'}
    assert identity['initial_model_sha256'] == control['initial_model_sha256']
    assert identity['initial_discriminator_sha256'] == control['initial_discriminator_sha256']
    assert identity['recipe']['shift_consistency']['weight'] == .1
    assert 'shift_consistency' not in control['recipe']


def test_update_uses_global_step_then_exact_optimizer_reload_and_next_shift(tmp_path):
    cfg, g, d, go, do = small_models(config()['boundary_naf'])
    sample = batch()
    before = state_sha256(g)
    stats = trainer.update(g, d, go, do, sample, cfg, 64)
    assert stats['generator_updated'] and stats['discriminator_updated']
    assert state_sha256(g) != before
    assert stats['shift_consistency_weighted'] > 0
    assert stats['shift_second_branch_probe_grad_norm'] > 0
    assert (stats['shift_dy'], stats['shift_dx']) == shift_for_step(cfg['shift_consistency'], 64)
    assert stats['erasure_applied_samples'] == 1
    identity = dict(recipe=deepcopy(cfg), test='shift-single-variable')
    trainer.save_checkpoint(tmp_path/'latest.pt', g, d, go, do, cfg, identity, 65, [0, 1])
    _, fresh, fresh_d, fresh_go, fresh_do = small_models(cfg)
    assert trainer.load_checkpoint(tmp_path/'latest.pt', fresh, fresh_d, fresh_go, fresh_do,
                                   cfg, identity) == (65, [0, 1])
    a = trainer.update(g, d, go, do, sample, cfg, 65)
    b = trainer.update(fresh, fresh_d, fresh_go, fresh_do, sample, cfg, 65)
    assert state_sha256(g) == state_sha256(fresh)
    assert state_sha256(d) == state_sha256(fresh_d)
    for key in ['shift_dy', 'shift_dx', 'shift_consistency_loss', 'loss_g', 'loss_d']:
        assert a[key] == b[key]


def shift_after():
    return dict(supported_probability=.98,
        shift_probe=dict(shifted_reconstruction=[dict(
            original_supported_points=30, original_supported_above065=1.,
            artificial_missing_points=10, artificial_missing_above065=1.,
            trusted_negative_points=500, trusted_negative_probability=.001,
            strong_band_area_ratio_to_clean_T8=1.5, strong_band_area_warning=True,
        ) for _ in range(4)]))


def test_shift_smoke_uses_reconstruction_not_random_constant_stability_or_area_warning():
    after = shift_after()
    checks = trainer.shift_smoke_checks(after, 128, 128, {'shift_recompute_max_error': 0.})
    # 面积比只报警；硬门槛仍是已存在的真线/人工缺口与内部保护。
    assert all(checks.values())
    for row in after['shift_probe']['shifted_reconstruction']:
        row['original_supported_above065'] = row['artificial_missing_above065'] = 0.
    after['supported_probability'] = .001  # 常量低输出尽管MAE可为0，也不能放行。
    checks = trainer.shift_smoke_checks(after, 128, 128, {'shift_recompute_max_error': 0.})
    assert not checks['shift_supported_output_not_collapsed']
    assert not checks['shift_all_directions_keep_original_support']
    assert not checks['shift_all_directions_recover_artificial_gap']
    assert not trainer.shift_smoke_checks(shift_after(), 128, 0,
        {'shift_recompute_max_error': 0.})['shift_consistency_independent_gradient']


class PointwiseBoundary(nn.Module):
    input_mode = 'boundary'
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
    def forward(self, value):
        return value + self.anchor * 0


def test_shift_monitor_records_all_fixed_pairs_but_only_selected_picture(tmp_path, monkeypatch):
    sample = batch()
    sample = {k: v[0] if torch.is_tensor(v) else v for k, v in sample.items()}
    sample['meta'] = sample['meta'][0]
    sample['source'] = 'train_001'
    sample['rgb'] = torch.zeros(3, 96, 96)
    sample['meta']['geometry'] = dict(rotation=0, flip=False)
    cfg = config()
    cfg['boundary_naf']['monitor'].update(pair_indices=[0, 1], shift_pair_indices=[1],
        full_readout_steps=[], thumbnail_width=64)
    monkeypatch.setattr(monitor, 'original_view', lambda *_: np.ones((96, 96), bool))
    monkeypatch.setattr(monitor, '_fixed_controls', lambda *_: pytest.fail('No full readout in pair monitor'))
    result = monitor.save_monitor(PointwiseBoundary(), cfg, tmp_path, 7, {0: sample, 1: sample})
    assert len(result['rows']) == 2
    assert not result['full_readout']
    assert result['shift_probe_input'] == 'clean_T8_all_fixed_pairs'
    for row in result['rows']:
        assert row['shift_consistency']['input'] == 'same_clean_T8_before_erasure'
        probe = row['shift_consistency']['metrics']
        assert len(probe['rows']) == 4
        assert all(item['band_mae'] == 0 and item['interior_mae'] == 0 for item in probe['rows'])
    folder = tmp_path/'monitor/s000007'
    assert not (folder/'pair_000_shift.jpg').exists()
    assert Image.open(folder/'pair_001_shift.jpg').width == 6*64
