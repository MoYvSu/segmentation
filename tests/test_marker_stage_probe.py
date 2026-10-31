# -*- coding: utf-8 -*-
"""用虚构整图检查marker阶段入口的实际CPU契约，不加载模型或赛题数据。"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

import tools.probe_marker_stages as probe
from tools.probe_affinity_bottleneck import measure, save_variant
from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, 'ROOT', tmp_path)
    config = {'inference': dict(boundary_threshold=.65,
        marker_boundary_low_threshold=.45, marker_boundary_reconstruction_steps=8,
        marker_border_seal_width=2, bridge_width=1, watershed_dilate_width=1,
        min_instance_area=50, max_instance_id=65535, threshold=.5,
        semantic_vote_mode='probability_mean', semantic_vote_erode_width=0,
        semantic_vote_threshold=.5)}
    h, w = 96, 128
    boundary = np.full((h, w), .03, np.float32)
    boundary[:, 60:68] = .9
    semantic = torch.linspace(-2., 2., w)[None, None, None].expand(1, 1, h, w).clone()
    boundary_tensor = torch.from_numpy(boundary)[None, None]
    raw = semantic.sigmoid()[0, 0].numpy().copy()
    gt = np.ones((h, w), np.uint16)
    gt[:, 64:] = 2
    lookup = np.array([-1, 0, 1], np.int8)
    classes = {1: 0, 2: 1}
    data = tmp_path / 'data'
    data.mkdir()
    source_path = data / 'source.png'
    assert cv2.imwrite(str(source_path), np.full((h, w, 3), 128, np.uint8))
    gt_path = data / 'target.txt'
    gt_path.write_text('synthetic target is supplied by the fixture', encoding='utf-8')
    monkeypatch.setattr(probe, 'load_completed_semantic_source',
                        lambda path, shape: (gt.copy(), lookup.copy()))
    ref = tmp_path / 'cache' / 'source'
    ref.mkdir(parents=True)
    baseline = decode_marker_replay(semantic, boundary_tensor, config)
    metrics, votes = measure(baseline, raw, gt, classes)
    save_variant(ref, 'r1', baseline, votes, boundary)
    np.savez_compressed(ref / 'r1_maps.npz', semantic=semantic.numpy(),
        boundary=boundary, raw_probability=raw,
        markers=baseline['actual_markers'], watershed=baseline['watershed'])
    raw_root = tmp_path / 'raw'
    raw_folder = raw_root / 'source'
    raw_folder.mkdir(parents=True)
    save_variant(raw_folder, 'baseline', baseline, votes, boundary)
    variant = build_marker_variant(baseline['original_stages'], 'reflect', 32)
    reflected = decode_marker_replay(semantic, boundary_tensor, config, variant['marker_labels'])
    _, reflect_votes = measure(reflected, raw, gt, classes, metrics)
    save_variant(ref, 'r1_reflect', reflected, reflect_votes, boundary)
    case = dict(case_id='synthetic_pair', direction='negative', gt_ids=[1, 2],
                points=[[48, 24], [48, 104]], censored=True)
    source = dict(source='source.png', source_path='data/source.png',
        gt_path='data/target.txt', reference_dir='cache/source', reference_prefix='r1',
        cases=[case])
    record = dict(source=source,
        input_hashes={str(path): probe.sha(path) for path in (source_path, gt_path)},
        baseline=json.loads(json.dumps(metrics, allow_nan=False)))
    output = tmp_path / 'outputs' / 'marker'
    output.mkdir(parents=True)
    return record, raw_root, config, output, boundary, ref


def test_actual_cpu_source_replay_keeps_tensor_and_final_boundary_contract(synthetic_source, monkeypatch):
    record, raw_root, config, output, boundary, _ = synthetic_source
    actual_decode = probe.decode_marker_replay
    observed = []

    def check_tensor(semantic, native_boundary, *args, **kwargs):
        # The entrypoint loads a HW cache, but actual decode accepts [1,1,H,W].
        assert torch.is_tensor(native_boundary)
        assert native_boundary.device.type == 'cpu'
        assert tuple(native_boundary.shape) == (1, 1, *boundary.shape)
        assert np.array_equal(native_boundary[0, 0].numpy(), boundary)
        observed.append(tuple(native_boundary.shape))
        return actual_decode(semantic, native_boundary, *args, **kwargs)

    monkeypatch.setattr(probe, 'decode_marker_replay', check_tensor)
    result = probe.analyze_source(record, raw_root, config, output)
    assert len(observed) == 3  # original, reflect, nothin; no model forward.
    assert result['baseline_exact']
    assert result['variants']['reflect']['existing_output_exact']['actual_final_exact']
    for mode in ('reflect', 'nothin'):
        item = result['variants'][mode]
        assert item['real_barrier_elevation_exact']
        assert item['actual_capture']['actual_watershed_calls'] == 1
        assert item['actual_capture']['final_dtype'] == 'uint16'
        assert item['actual_map_changes_vs_r1']['boundary']['exact']
        assert item['actual_map_changes_vs_reflect']['boundary']['exact']
        names = item['cases'][0]['stage_trace']['stage_order']
        assert names[-3:] == ['marker_labels', 'watershed', 'instances']
        if mode == 'nothin':
            assert not any('skeleton' in name for name in names)
            assert 'reflect_relative' in item
        else:
            assert 'reflected_skeleton' in names
    for stem in ('baseline', 'reflect', 'nothin'):
        saved = probe.load_prediction(output / 'source', stem)
        assert np.array_equal(saved['boundary'], boundary)


@pytest.mark.parametrize('stem,expected', [
    ('r1', 'Cached raw-vote final classes differ'),
    ('r1_reflect', 'Existing complete reflect output differs'),
])
def test_existing_reflect_and_baseline_mismatches_fail_closed(synthetic_source, stem, expected):
    record, raw_root, config, output, _, ref = synthetic_source
    # A valid JSON with the wrong class must not pass existing-output provenance.
    path = ref / (stem + '_class.json')
    votes = json.loads(path.read_text(encoding='utf-8'))
    first = next(iter(votes))
    votes[first] = 1 - votes[first]
    path.write_text(json.dumps(votes), encoding='utf-8')
    with pytest.raises(RuntimeError, match=expected):
        probe.analyze_source(record, raw_root, config, output)
