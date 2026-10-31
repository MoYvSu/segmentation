# -*- coding: utf-8 -*-
"""高阈值种子消融：真实旧细化复用、弱通路差异与障碍别名保护。"""
from copy import deepcopy

import numpy as np
import pytest

from tools.probe_affinity_marker_replay import build_marker_variant
from tools.semantic_marker_diagnostic import marker_stages
from utils.marker_single_threshold import high_only_reflect_markers


def sample(weak=True):
    config = {'inference': dict(boundary_threshold=.65,
        marker_boundary_low_threshold=.45, marker_boundary_reconstruction_steps=8,
        bridge_width=1, watershed_dilate_width=1,
        marker_border_seal_width=2, min_instance_area=50)}
    boundary = np.full((128, 144), .03, np.float32)
    boundary[:, 65:79] = .9
    if weak:
        boundary[56:68, 65:79] = .53
    return boundary, config, marker_stages(boundary, config)


def test_removing_reconstruction_opens_the_weak_separator_but_keeps_strong_mask():
    _, _, stages = sample()
    control = build_marker_variant(stages, 'reflect', 32)
    candidate = high_only_reflect_markers(stages)
    assert control['marker_count'] == 2
    assert candidate['marker_count'] == 1
    assert np.array_equal(candidate['marker_boundary_mask'], stages['high_mask'])
    assert candidate['single_threshold_audit']['removed_weak_boundary_pixels'] > 0
    assert np.array_equal(candidate['barrier_belt'], control['barrier_belt'])


def test_fully_strong_separator_matches_the_existing_reflect_implementation():
    _, _, stages = sample(weak=False)
    expected = build_marker_variant(stages, 'reflect', 32)
    candidate = high_only_reflect_markers(stages)
    for key in ['marker_belt', 'core_labels', 'marker_labels', 'barrier_belt']:
        assert np.array_equal(candidate[key], expected[key])
    assert candidate['single_threshold_audit']['removed_weak_boundary_pixels'] == 0
    assert candidate['resolved']['marker_boundary_reconstruction_steps'] == 0
    assert candidate['resolved']['reference_marker_boundary_reconstruction_steps'] == 8


def test_preserves_original_barrier_instead_of_the_steps_zero_seal_alias():
    boundary, config, stages = sample()
    direct = deepcopy(config)
    direct['inference']['marker_boundary_reconstruction_steps'] = 0
    aliased = marker_stages(boundary, direct)
    candidate = high_only_reflect_markers(stages)
    assert not np.array_equal(aliased['barrier_belt'], stages['barrier_belt'])
    assert np.array_equal(candidate['barrier_belt'], stages['barrier_belt'])
    assert not np.shares_memory(candidate['barrier_belt'], stages['barrier_belt'])


def test_input_stages_are_not_mutated_and_fifty_core_threshold_is_preserved():
    _, _, stages = sample()
    before = deepcopy(stages)
    candidate = high_only_reflect_markers(stages)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(stages[key], value), key
        else:
            assert stages[key] == value, key
    assert candidate['single_threshold_audit']['core_min_area'] == 50
    assert candidate['resolved']['min_instance_area'] == 50
    labels, counts = np.unique(candidate['marker_labels'], return_counts=True)
    assert np.all(counts[labels > 0] >= 50)
    assert candidate['marker_labels'].dtype == np.int32
    assert not candidate['single_threshold_audit']['final_instance_filter_modified']


@pytest.mark.parametrize('padding', [0, -1, True, 1.5])
def test_invalid_reflection_padding_is_rejected(padding):
    _, _, stages = sample()
    with pytest.raises(ValueError, match='reflection_padding'):
        high_only_reflect_markers(stages, padding)


@pytest.mark.parametrize('kind', ['float_high', 'bad_high_value', 'wrong_barrier_shape',
                                 'negative_area', 'missing_settings', 'missing_reference'])
def test_invalid_or_incomplete_stages_are_rejected(kind):
    _, _, stages = sample()
    if kind == 'float_high':
        stages['high_mask'] = stages['high_mask'].astype(np.float32)
    elif kind == 'bad_high_value':
        stages['high_mask'][0, 0] = 2
    elif kind == 'wrong_barrier_shape':
        stages['barrier_belt'] = stages['barrier_belt'][:-1]
    elif kind == 'negative_area':
        stages['resolved']['min_instance_area'] = -1
    elif kind == 'missing_settings':
        stages.pop('resolved')
    else:
        stages.pop('marker_boundary_mask')
    with pytest.raises(ValueError):
        high_only_reflect_markers(stages)
