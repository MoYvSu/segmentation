# -*- coding: utf-8 -*-
"""零噪声后端交叉实验的模式隔离、固定全图输入和统计语义。"""
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import probe_restore_zero as probe


def case(boundary):
    ids = np.ones((4, 4), dtype=np.uint16)
    ids[:, 2:] = 2
    maps = dict(boundary=np.full((4, 4), boundary, dtype=np.float32),
                geometry_semantic=np.full((4, 4), .2, dtype=np.float32),
                raw_probability=np.tile(np.array([.2, .2, .8, .8], dtype=np.float32), (4, 1)),
                instances=ids)
    return maps, {'1': 0, '2': 1}


def test_globals_gate_catches_any_change():
    maps, _ = case(.5)
    fixed = {key+'_sha256': probe.digest(maps[key]) for key in ('geometry_semantic', 'raw_probability')}
    probe.require_globals(maps, fixed)
    maps['raw_probability'][0, 0] += .01
    with pytest.raises(ValueError, match='raw_probability'):
        probe.require_globals(maps, fixed)


def test_crossed_summary_has_real_noisy_variance_and_no_fake_zero_variance():
    old = [case(.50+i*.001) for i in range(8)]
    a = [case(.51+i*.001) for i in range(8)]
    c = [case(.52+i*.002) for i in range(8)]
    a_zero, c_zero = case(.48), case(.49)
    result, std = probe.summarize_cases(old, a, c, a_zero, c_zero)
    assert set(result['series']) == {'old_noisy', 'a_noisy', 'c_noisy'}
    assert set(std) == set(result['series'])
    assert result['series']['c_noisy']['sample_count'] == 8
    assert std['c_noisy'].mean() > std['a_noisy'].mean() > 0
    assert result['zero']['c_zero']['stochastic_variance_not_evaluated']
    assert 'ferrite_cv' not in result['zero']['c_zero']
    assert result['comparisons']['c_zero_vs_a_zero']['boundary']['mae'] == pytest.approx(.01, abs=1e-7)
    assert len(result['comparisons']['c_noisy_vs_a_noisy']) == 8
    assert result['comparisons']['c_zero_vs_a_zero']['raw_semantic']['probability_mae'] == 0
    assert result['comparisons']['c_zero_vs_a_zero']['instances']['matched_count'] == 2
    macro = probe.macro({'one_source': result})
    assert macro['c_zero_vs_a_zero']['ferrite_mean_area_relative_delta'] == 0
    with pytest.raises(ValueError, match='eight'):
        probe.summarize_cases(old[:-1], a, c, a_zero, c_zero)


def test_local_zero_mode_uses_zero_probe_and_only_one_boundary_case(monkeypatch):
    calls = []
    monkeypatch.setattr(probe, 'tile_boxes', lambda shape, size, overlap: [(0, 0, 4, 4)])
    monkeypatch.setattr(probe, 'tensor_from_rgb', lambda rgb, device: (torch.ones(1, 3, 4, 4), 0, 0))

    def restore(restorer, image, seed, mode):
        calls.append((seed, mode))
        return image.clone(), image.clone()

    monkeypatch.setattr(probe, 'restore_probe', restore)
    monkeypatch.setattr(probe, 'boundary_grid', lambda model, image, cfg: image[:, :1])
    monkeypatch.setattr(probe, 'crop_letterbox_output', lambda grid, *args: grid)
    boundaries, boxes = probe.local_boundaries(None, None, np.zeros((4, 4, 3), dtype=np.uint8),
        {'affinity_native': {'overlap': .25}}, 314159, mode='zero', device='cpu')
    assert len(boundaries) == len(boxes) == 1
    assert len(calls) == 2 and all(mode == 'zero' for _, mode in calls)
    assert calls[0][0] != calls[1][0]
    with pytest.raises(ValueError, match='fabricate'):
        probe.local_boundaries(None, None, np.zeros((4, 4, 3), dtype=np.uint8),
            {'affinity_native': {'overlap': .25}}, 1, mode='zero', offsets=(0, 1), device='cpu')


def test_adapted_cache_checks_declared_control_weights_and_all_six_gates(monkeypatch):
    monkeypatch.setattr(probe.deployment, 'sha', lambda path: str(path))
    config = {'paths': {'project_root': 'new'}, 'fixed': 1}
    record = dict(format='restore_consistency_backend_probe_v1', complete=True,
                  offsets=list(probe.OFFSETS), config={'paths': {'project_root': 'old'}, 'fixed': 1},
                  sources=['test_088'], checkpoints={key: {'sha256': key} for key in ('old', 'backend', 'control')},
                  images={'test_088': {'old_s0_gate': {key: True for key in
                      ('continuous_boundary_exact', 'instances_exact', 'classes_exact')}}})
    paths = {'restorer': Path('old'), 'backend': Path('backend')}
    probe.validate_adapted_cache(record, config, paths, Path('control'), ['test_088'])
    with pytest.raises(ValueError, match='control'):
        probe.validate_adapted_cache(record, config, paths, Path('wrong'), ['test_088'])
    record['images']['test_088']['old_s0_gate']['classes_exact'] = False
    with pytest.raises(ValueError, match='gate'):
        probe.validate_adapted_cache(record, config, paths, Path('control'), ['test_088'])
