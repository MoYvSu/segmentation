# -*- coding: utf-8 -*-
"""面积清理在末端语义之后执行；模型和已评分的默认入口保持原行为。"""
from pathlib import Path

import numpy as np
import pytest

from tools import infer_semifinal as entry
from utils.config import load_config
from utils.instance_area_policy import resolve_instance_area_policy


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('threshold', [100, 200, 500])
def test_candidate_configs_change_only_explicit_final_area(threshold):
    original = load_config(str(ROOT / 'config/inference/semifinal_n1.yaml'))
    config = load_config(str(ROOT / f'config/inference/area_floor_{threshold}.yaml'))
    entry.validate_config(config)
    assert resolve_instance_area_policy(config['inference']) == dict(
        seed_min_area=50, final_min_instance_area=threshold,
        final_small_instance_action='neighbor_merge_v1', pre_vote_final_min_area=0)
    for key in ('seed_min_area', 'final_min_instance_area', 'final_small_instance_action'):
        config['inference'].pop(key)
    assert config == original


def test_default_area_finalization_is_original_object_and_classes():
    config = load_config(str(ROOT / 'config/inference/semifinal_n1.yaml'))
    result = dict(instances=np.ones((12, 12), np.uint16), classes={'1': 1})
    assert entry.finalize_instance_area(result, config) is result


def test_finalizer_inherits_final_raw_vote_and_target_score_without_revote(monkeypatch):
    config = load_config(str(ROOT / 'config/inference/area_floor_100.yaml'))
    source = np.zeros((20, 24), np.uint16)
    source[2:18, 2:14] = 1
    source[8:11, 15:18] = 2
    original = source.copy()
    result = dict(instances=source, classes={'1': 0, '2': 1},
                  instance_scores={'1': .3, '2': .8})
    monkeypatch.setattr(entry, 'revote_instances', lambda *a, **k: pytest.fail('must not revote'))
    cleaned = entry.finalize_instance_area(result, config)
    assert cleaned['classes'] == {'1': 0}
    assert cleaned['instance_scores'] == {'1': .3}
    assert cleaned['instance_score_policy'] == 'original_target_score_inherited_not_recomputed'
    assert np.all(cleaned['instances'][source == 1] == 1)
    assert np.all(cleaned['instances'][source == 2] == 1)
    assert np.all(cleaned['instances'][8:11, 14] == 1)
    assert np.array_equal(source, original) and result['classes'] == {'1': 0, '2': 1}


def test_predict_finalization_runs_after_raw_path(monkeypatch):
    config = load_config(str(ROOT / 'config/inference/area_floor_100.yaml'))
    ids = np.ones((12, 12), np.uint16)
    events = []

    def raw_path(*args, **kwargs):
        events.append('raw_vote_done')
        return np.zeros((12, 12, 3), np.uint8), {'patch1024': dict(instances=ids, classes={'1': 0})}

    actual_finalize = entry.finalize_instance_area

    def cleanup(result, actual_config):
        assert events == ['raw_vote_done']
        events.append('final_cleanup')
        return actual_finalize(result, actual_config)

    monkeypatch.setattr(entry.views, 'predict_views', raw_path)
    monkeypatch.setattr(entry, 'finalize_instance_area', cleanup)
    rgb, result = entry.predict(object(), object(), None, Path('test_169.png'), config, 'cpu')
    assert events == ['raw_vote_done', 'final_cleanup']
    assert result['classes'] == {'1': 0}
    assert result['instances'].shape == rgb.shape[:2]


def test_seed_is_still_frozen_while_final_area_is_independent():
    config = load_config(str(ROOT / 'config/inference/area_floor_200.yaml'))
    config['inference']['seed_min_area'] = 100
    with pytest.raises(ValueError, match='seed_min_area'):
        entry.validate_config(config)
    config['inference'].update(seed_min_area=50, min_instance_area=999)
    entry.validate_config(config)
