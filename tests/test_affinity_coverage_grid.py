# -*- coding: utf-8 -*-
"""真实letterbox/端点目标的纯CPU合成门禁，不读赛题图像。"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import probe_affinity_coverage as coverage
from tools import probe_affinity_signal as original


@pytest.fixture(scope='module')
def functions():
    torch.set_num_threads(2)
    return original.load_functions(Path(__file__).resolve().parents[1])


def draw(hf=False, vf=False, k=0):
    return dict(horizontal_flip=[hf], vertical_flip=[vf], rotation_k=[k])


@pytest.mark.parametrize('hf,vf,k', list(product((False, True), (False, True), range(4))))
def test_crop_then_original_letterbox_and_spatial_exact(functions, hf, vf, k):
    gt = np.arange(1, 24*24+1, dtype=np.int32).reshape(24, 24)
    native, grid, valid, meta, yy, xx = coverage.geometry_grid(
        gt, [2, 4, 18, 20], draw(hf, vf, k), functions, input_size=16, output_grid=8)
    expected = gt[2:18:2, 4:20:2]
    if hf:
        expected = np.fliplr(expected)
    if vf:
        expected = np.flipud(expected)
    expected = np.rot90(expected, k)
    assert np.array_equal(grid, expected)
    assert valid.all() and meta.content_height == meta.content_width == 8
    assert np.array_equal(gt[yy, xx], grid)
    assert np.array_equal(native, gt[2:18, 4:20])


def test_unknown_padding_and_long_endpoint_only_target(functions):
    gt = np.ones((8, 16), np.uint16)
    gt[:, 8:] = 2
    gt[:, 7] = 0
    result = coverage.view_coverage(gt, [0, 0, 8, 16], draw(), {1:1, 2:0}, functions, input_size=16, output_grid=8)
    # 采样不保留原一列unknown，但实际target仍只判断已知两端；padding不能出伪目标。
    assert np.all(result['values'][:, 3]+result['values'][:, 4] == result['values'][:, 1])
    assert np.all(result['values'][:, 5]+result['values'][:, 6] == result['values'][:, 2])
    native, grid, valid, meta, yy, xx = coverage.geometry_grid(gt, [0, 0, 8, 16], draw(True, True, 1), functions, input_size=16, output_grid=8)
    assert valid.sum() == 32 and not valid.all()
    codes, values = coverage.supervision_counts(grid, valid, yy, xx, gt.shape, functions)
    assert len(codes) == 1 and values[0, 2] > 0


def test_unknown_path_is_not_new_ignore_rule(functions):
    gt = np.ones((8, 8), np.int32)
    gt[:, 5:] = 2
    gt[:, 4] = 0
    grid, valid, meta = functions['letterbox_instance_geometry'](gt, 8, 8)
    yy, xx = np.indices(gt.shape)
    codes, values = coverage.supervision_counts(grid, valid, yy, xx, gt.shape, functions)
    assert codes.tolist() == [1*65536+2]
    assert values[0, 0] == 0 and values[0, 2] > values[0, 1]
    assert coverage.cardinal_contacts(gt)[0].size == 0


def test_fixed_even_phase_can_remove_thin_gt_and_create_grid_adjacency(functions):
    gt = np.ones((18, 18), np.int32)
    gt[:, 7] = 2
    gt[:, 8:] = 3
    classes = {1:1, 2:1, 3:0}
    even = coverage.view_coverage(gt, [0, 0, 16, 16], draw(), classes, functions, input_size=16, output_grid=8)
    odd = coverage.view_coverage(gt, [0, 1, 16, 17], draw(), classes, functions, input_size=16, output_grid=8)
    assert even['lost_ids'].tolist() == [2]
    assert odd['lost_ids'].tolist() == []
    raw_pairs = coverage.cardinal_contacts(gt)[0]
    generated = 1*65536+3
    assert generated not in raw_pairs and generated in even['codes']
    idx = int(np.flatnonzero(even['codes'] == generated)[0])
    assert even['values'][idx, 0] > 0


def test_disappearance_is_not_claimed_for_gt_outside_crop(functions):
    gt = np.ones((24, 24), np.int32)
    gt[20:, 20:] = 2
    result = coverage.view_coverage(gt, [0, 0, 16, 16], draw(), {1:1, 2:0}, functions, input_size=16, output_grid=8)
    assert result['lost_ids'].tolist() == [] and result['native_sizes'][2] == 0


def test_original_boundary_band_and_actual_endpoint_bands_are_separate(functions):
    gt = np.ones((600, 600), np.int32)
    gt[:, 308:] = 2
    result = coverage.view_coverage(gt, [300, 300, 316, 316], draw(), {1:1, 2:1}, functions, input_size=16, output_grid=8)
    assert result['values'][0, 5] == 0
    assert result['values'][0, 6] == result['values'][0, 2]
    raw_codes, raw_counts = coverage.cardinal_contacts(gt)
    assert raw_counts[0, 1] > 0 and raw_counts[0, 2] > 0
    assert raw_counts[0, 1]+raw_counts[0, 2] == raw_counts[0, 0]


def test_high_uint16_id_pair_encoding_and_phase(functions):
    gt = np.ones((8, 8), np.uint16)
    gt[:, 4:] = 65535
    result = coverage.view_coverage(gt, [0, 0, 8, 8], draw(), {1:1, '65535':0}, functions, input_size=8, output_grid=8)
    assert result['codes'].tolist() == [1*65536+65535]
    lookup = coverage.validate_labels(gt, {1:1, '65535':0})
    assert coverage.phase_names(result['codes'], lookup).tolist() == ['F/P']


def test_native_crop_rescue_can_still_have_no_grid_supervision(functions):
    old = np.zeros((2, len(coverage.METRICS)), np.int64)
    new = old.copy()
    new[0, 0:2] = [1, 4]
    old[1, 0:2] = [1, 4]
    old[1, 3:5] = [1, 1]
    report = coverage.pair_summary(old, new, np.array([True, True]))
    assert report['raw_crop_rescued']['pairs'] == 1
    assert report['raw_crop_rescued']['candidate_short_zero'] == 1
    assert report['raw_crop_rescued']['candidate_any_zero'] == 1
    assert report['raw_crop_new_zero']['baseline_short_nonzero'] == 1
    assert report['transitions']['any_hits']['new_zero'] == 1


def test_all_unknown_has_no_targets_or_disappearance(functions):
    gt = np.zeros((8, 8), np.int32)
    result = coverage.view_coverage(gt, [0, 0, 8, 8], draw(), {}, functions, input_size=8, output_grid=8)
    assert result['codes'].size == 0 and result['lost_ids'].size == 0


@pytest.mark.parametrize('classes', ({1:True}, {True:1}, {1:.5}, {1:1, '1':0}, {}))
def test_strict_known_phase_lookup(classes):
    with pytest.raises(ValueError):
        coverage.validate_labels(np.ones((8, 8), np.int32), classes)


def test_crop_bounds_fail_before_target_build(functions):
    with pytest.raises(ValueError):
        coverage.geometry_grid(np.ones((8, 8), np.int32), [-1, 0, 7, 8], draw(), functions, input_size=8, output_grid=8)
