# -*- coding: utf-8 -*-
"""逐阶段诊断必须复现实际融合和拼接，不改变 SAME/CUT 语义。"""
import numpy as np
import pytest
import torch
import json

from tools.probe_affinity_gap import blend_weights, decompose, fixed_roi, load_roi_spec, resolve_roi_spec
from utils.affinity_fusion import affinity_boundary_probability
from utils.patch_diagnostic import BlendMap


OPTIONS = dict(distance2_weight=.5, distance4_weight=.25, support_threshold=.2,
               support_temperature=.05, short_reduction='top2', short_softmax_temperature=.15)


def test_decomposition_is_exact_original_fusion():
    logits = torch.randn((1, 8, 17, 23), generator=torch.Generator().manual_seed(52))
    traced = decompose(logits, OPTIONS)
    actual = affinity_boundary_probability(logits, mode='gated', **OPTIONS)
    assert torch.equal(traced['fused'], actual)
    assert torch.equal(traced['cut'], 1 - torch.sigmoid(logits))


def test_same_is_not_cut():
    assert decompose(torch.full((1, 8, 7, 7), 10.), OPTIONS)['fused'].max() < .001
    assert decompose(torch.full((1, 8, 7, 7), -10.), OPTIONS)['fused'].min() > .999
    with pytest.raises(ValueError):
        decompose(torch.zeros((1, 7, 7, 7)), OPTIONS)


def test_window_weight_replay_exact():
    blend = BlendMap((10, 13))
    total, weight = np.zeros((10, 13), np.float32), np.zeros((10, 13), np.float32)
    rng = np.random.default_rng(52)
    for box in [(0, 0, 10, 9), (0, 4, 10, 13)]:
        y, x, y1, x1 = box
        value = rng.random((y1-y, x1-x), dtype=np.float32)
        w = blend_weights(*value.shape)
        blend.add(box, value)
        total[y:y1, x:x1] += value*w
        weight[y:y1, x:x1] += w
    assert np.array_equal(blend.finish()[0], total/weight)


def test_roi_default_is_generic_center():
    assert fixed_roi((80, 120)) == (20, 0, 100, 80)
    assert resolve_roi_spec({}, 'synthetic', (80, 120)) == (
        (20, 0, 100, 80), {'fixed_center': (60, 40)})
    assert load_roi_spec(None) == {}


def test_json_coordinates_preserve_passed_content(tmp_path):
    row = {'roi': [10, 20, 70, 60], 'points': {'probe': [40, 30]}}
    path = tmp_path / 'roi.json'
    path.write_text(json.dumps({'format': 'affinity_gap_roi_v1', 'images': {'synthetic': row}}))
    loaded = load_roi_spec(path)
    assert loaded['synthetic'] == row
    assert resolve_roi_spec(loaded, 'synthetic', (80, 120)) == (
        tuple(row['roi']), {'probe': tuple(row['points']['probe'])})


@pytest.mark.parametrize('row', [
    {'roi': [-1, 20, 70, 60], 'points': {'probe': [40, 30]}},
    {'roi': [10, 20, 121, 60], 'points': {'probe': [40, 30]}},
    {'roi': [10, 20, 70, 81], 'points': {'probe': [40, 30]}},
    {'roi': [10, 20, 10, 60], 'points': {'probe': [40, 30]}},
    {'roi': [10, 20, 70, 60], 'points': {'probe': [70, 30]}},
    {'roi': [10, 20, 70, 60], 'points': {'probe': [40.0, 30]}},
])
def test_reject_invalid_private_coordinates(row):
    with pytest.raises(ValueError):
        resolve_roi_spec({'synthetic': row}, 'synthetic', (80, 120))


def test_reject_unversioned_roi_file(tmp_path):
    path = tmp_path / 'roi.json'
    path.write_text(json.dumps({'images': {}}))
    with pytest.raises(ValueError):
        load_roi_spec(path)
