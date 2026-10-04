# -*- coding: utf-8 -*-
"""独立复赛入口：编号种子、真实reflect几何及末端语义隔离。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.infer_semifinal as standalone
from tools.affinity_connectivity_views import decode_native
from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from tools.semantic_marker_diagnostic import marker_stages
from utils.config import load_config


def sample():
    config = {'inference': dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_border_seal_width=2, bridge_width=1,
        watershed_dilate_width=1, min_instance_area=50, max_instance_id=65535,
        threshold=.5, semantic_vote_mode='probability_mean')}
    boundary = np.full((128, 144), .03, np.float32)
    boundary[:, 65:79] = .9; boundary[58:62, 65:79] = .53
    semantic = torch.linspace(-2, 2, 144)[None, None, None].expand(1, 1, 128, 144).clone()
    return semantic, torch.from_numpy(boundary)[None, None], config


def test_stable_full_cohort_seed_even_when_only_two_images(tmp_path):
    for name in ('test_101', 'test_116'):
        (tmp_path / (name + '.png')).write_bytes(b'input-not-read-by-order-check')
    selected = standalone.input_files(tmp_path, ['test_116', 'test_101'])
    assert [p.stem for p in selected] == ['test_101', 'test_116']
    assert [standalone.source_seed(p.stem) for p in selected] == [314191, 314206]
    with pytest.raises(ValueError, match='exactly'):
        standalone.input_files(tmp_path)
    for name in ('test_68', 'test_068', 'test_169', 'train_101'):
        with pytest.raises(ValueError):
            standalone.source_index(name)


def test_real_reflect_watershed_equals_prior_replay_without_reference_or_cache():
    semantic, boundary, config = sample()
    state = marker_stages(boundary[0, 0].numpy(), config)
    variant = build_marker_variant(state, 'reflect', 32)
    expected = decode_marker_replay(semantic, boundary, config, variant['marker_labels'])
    original_ws = standalone.cv2.watershed
    ids, classes = standalone.decode_reflect(semantic, boundary, config)
    assert np.array_equal(ids, expected['instances'])
    assert classes == expected['classes']
    assert standalone.cv2.watershed is original_ws


def test_reflect_zero_change_when_original_reflected_markers_are_identical():
    semantic, boundary, config = sample()
    state = marker_stages(boundary[0, 0].numpy(), config)
    variant = build_marker_variant(state, 'reflect', 32)
    assert np.array_equal(state['marker_labels'], variant['marker_labels'])
    expected_ids, expected_classes = decode_native(semantic, boundary, config)
    ids, classes = standalone.decode_reflect(semantic, boundary, config)
    assert np.array_equal(ids, expected_ids) and classes == expected_classes


def test_frozen_configuration_contract_rejects_geometry_or_input_changes():
    config = load_config('config/inference/submission_reflect.yaml')
    standalone.validate_config(config)
    for group, name, value in [('inference', 'boundary_threshold', .66),
                                ('sam2', 'input_normalization', 'imagenet'),
                                ('affinity_native', 'overlap', .5),
                                ('submission', 'marker_reflection_padding', 64)]:
        changed = deepcopy(config); changed[group][name] = value
        with pytest.raises(ValueError):
            standalone.validate_config(changed)


def test_no_existing_outputs_needed_for_terminal_vote_and_original_head_stays_upstream(monkeypatch):
    ids = np.ones((4, 5), np.uint16)
    original_head = object(); terminal_calls = []; view_calls = []
    model = SimpleNamespace(semantic_decoder=original_head, encoder=lambda raw: [raw])

    def old_views(actual_model, restorer, path, config, device, seed, sizes):
        assert actual_model.semantic_decoder is original_head
        view_calls.append((seed, sizes))
        result = dict(instances=ids.copy(), classes={'1': 0}, boundary=np.zeros(ids.shape, np.float32),
                      phases={}, blend=dict(tiles=1))
        return np.zeros((*ids.shape, 3), np.uint8), {'patch1024': result}

    def terminal(features, raw):
        terminal_calls.append((features, raw))
        return torch.ones((1, 1, 4, 5))

    monkeypatch.setattr(standalone.views, 'predict_views', old_views)
    monkeypatch.setattr(standalone, 'prepare_image', lambda *a: (None, torch.zeros((1, 3, 4, 5)), 0, 0))
    monkeypatch.setattr(standalone, 'crop_letterbox_output', lambda value, *a: value)
    _, actual = standalone.predict(model, None, terminal, Path('test_101.png'), {}, 'cpu')
    assert view_calls == [(314191, [1024])] and len(terminal_calls) == 1
    assert model.semantic_decoder is original_head
    assert np.array_equal(actual['instances'], ids) and actual['classes'] == {'1': 1}
    assert actual['instance_scores']['1'] > .5


def test_frozen_model_manifest_rejects_modified_source_and_path_escape(tmp_path):
    source = tmp_path / 'infer.py'; source.write_text('original', encoding='utf-8')
    value = dict(files=[dict(path='infer.py', sha256=standalone.sha(source))])
    (tmp_path / 'manifest.json').write_text(json.dumps(value), encoding='utf-8')
    assert standalone.verify_source_manifest(tmp_path)['checked_files'] == 1
    source.write_text('modified', encoding='utf-8')
    with pytest.raises(RuntimeError):
        standalone.verify_source_manifest(tmp_path)
    value['files'][0]['path'] = '../external.py'
    (tmp_path / 'manifest.json').write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(RuntimeError):
        standalone.verify_source_manifest(tmp_path)


def test_model_sha_required_and_terminal_is_optional(tmp_path):
    backend = tmp_path / 'backend.pt'; backend.write_bytes(b'backend')
    restorer = tmp_path / 'restorer.pt'; restorer.write_bytes(b'restorer')
    config = dict(submission=dict(models={name: dict(file=path.name, sha256=standalone.sha(path))
                                          for name, path in [('backend', backend), ('restorer', restorer)]}))
    assert set(standalone.model_paths(config, tmp_path)) == {'backend', 'restorer'}
    config['submission']['models']['backend']['sha256'] = '0' * 64
    with pytest.raises(RuntimeError):
        standalone.model_paths(config, tmp_path)


def test_uint16_class_contract_includes_strict_integer_classes():
    ids = np.array([[0, 65535], [1, 1]], np.uint16)
    standalone.verify_prediction(ids, {'1': 0, '65535': 1}, ids.shape)
    for value in ({'1': 0}, {'1': False, '65535': 1}, {'1': 0, '65535': 2}):
        with pytest.raises(RuntimeError):
            standalone.verify_prediction(ids, value, ids.shape)
