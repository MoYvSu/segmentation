# -*- coding: utf-8 -*-
"""虚构CPU入口：路径适配、patch归还、空目标与分组缓存隔离。"""
from argparse import Namespace
from copy import deepcopy

import numpy as np
import pytest
import torch

from tools import probe_affinity_readout_path as probe
from utils.offset_letterbox import letterbox_instance_geometry


def tile_for(gt, boundary):
    grid_gt, valid, geometry = letterbox_instance_geometry(gt, 1024, gt.shape[0])
    return dict(index=0, box=[0, 0, *gt.shape], grid_gt=grid_gt, valid=valid,
                geometry=geometry.to_dict(), grid=boundary[None, None].copy())


def direct_select(tile, gt, decoded, case, probabilities):
    return probe.original.constrained_bottleneck_path(
        probabilities, tile['grid_gt'], tile['valid'], case['points'][0], case['points'][1],
        case['direction'])


@pytest.mark.parametrize('direction', ['positive', 'negative'])
def test_adapter_runs_real_selector_and_preserves_original_path_contract(direction):
    gt = np.full((8, 8), 7, np.int32)
    labels = np.full_like(gt, 19)
    if direction == 'negative':
        gt[:, 4:] = 12
    else:
        labels[:, 4:] = 23
    decoded = dict(instances=labels, actual_markers=np.full_like(gt, 31))
    case = dict(direction=direction, gt_ids=[7, 12] if direction == 'negative' else [7],
                points=[[3, 1], [3, 6]])
    boundary = np.full(gt.shape, .4, np.float32)
    probabilities = np.full((4, *gt.shape), .8, np.float32)
    tile = tile_for(gt, boundary)
    original_function = probe.original.constrained_bottleneck_path
    snapshot = deepcopy(tile)
    result = probe.make_selector(probe.original.select_bottlenecks)(
        tile, gt, decoded, case, probabilities)
    assert probe.original.constrained_bottleneck_path is original_function
    assert result['found'] and result['targets_found']
    assert result['native_path_known_support_exact']
    assert result['edges'] == result['capacity_path']['edges']
    assert result['capacity'] == result['raw_capacity']
    assert result['zero_capacity'] == result['capacity_path']['zero_capacity']
    assert result['bottleneck_indices'] == result['target_indices']
    assert result['graph_bottleneck_indices'] == result['capacity_path']['bottleneck_indices']
    assert result['path_cross_gt_steps'] == sum(edge['cross_gt'] for edge in result['edges'])
    assert np.array_equal(tile['grid'], snapshot['grid'])
    assert np.array_equal(tile['grid_gt'], snapshot['grid_gt'])


def test_legal_path_without_targets_does_not_become_a_correction():
    gt = np.ones((3, 4), np.int32)
    gt[1, 2] = 3
    gt[2, 1] = 0
    boundary = np.full(gt.shape, .1, np.float32)
    boundary[1, 1] = .8
    probabilities = np.full((4, *gt.shape), .9, np.float32)
    probabilities[:, 1, 1] = [ .2, .4, .8, .9 ]
    tile = dict(grid_gt=gt, valid=np.ones_like(gt, bool), grid=boundary[None, None])
    case = dict(direction='positive', points=[[0, 1], [1, 1]])
    result = probe.make_selector(direct_select)(tile, gt, None, case, probabilities)
    assert result['found'] and not result['targets_found']
    assert result['graph_bottleneck_indices'] and not result['bottleneck_indices']
    assert result['empty_target_reason'] == 'no_legal_actual_top2_outgoing_targets'
    # 原轮次只沿bottleneck_indices写mask；图路径不能被误当修正关系。
    mask = np.zeros((1, 8, *gt.shape), bool)
    added = 0
    if result['found']:
        for c, y, x in result['bottleneck_indices']:
            if not mask[0, c, y, x]:
                mask[0, c, y, x] = True
                added += 1
    assert added == 0 and not mask.any()


def test_cached_grid_is_used_only_for_matching_short_probabilities_after_arm_reset():
    gt = np.ones((4, 4), np.int32)
    baseline = np.full((4, *gt.shape), .9, np.float32)
    changed = baseline.copy()
    changed[0, 1, 1] = .2
    baseline_grid = np.full((1, 1, *gt.shape), .1, np.float32)
    changed_grid = np.full_like(baseline_grid, .7)
    tile = dict(grid_gt=gt, valid=np.ones_like(gt, bool), grid=baseline_grid,
                readout_current_grid=changed_grid,
                readout_current_short_sha256=probe.array_sha(changed))
    case = dict(direction='positive', points=[[1, 1], [1, 2]])
    selector = probe.make_selector(direct_select)
    current = selector(tile, gt, None, case, changed)
    reset = selector(tile, gt, None, case, baseline)
    assert current['fused_grid_sha256'] == probe.array_sha(changed_grid[0, 0])
    assert reset['fused_grid_sha256'] == probe.array_sha(baseline_grid[0, 0])
    assert current['current_short_probabilities_sha256'] == probe.array_sha(changed)
    assert reset['current_short_probabilities_sha256'] == probe.array_sha(baseline)
    assert current['raw_capacity'] < reset['raw_capacity']
    assert tile['readout_current_short_sha256'] == probe.array_sha(changed)


def test_selector_patch_is_restored_if_original_selector_raises():
    gt = np.ones((2, 2), np.int32)
    p = np.full((4, 2, 2), .5, np.float32)
    tile = dict(grid=np.full((1, 1, 2, 2), .5, np.float32))
    original_function = probe.original.constrained_bottleneck_path

    def broken(*args):
        assert probe.original.constrained_bottleneck_path is not original_function
        raise RuntimeError('synthetic selector failure')

    with pytest.raises(RuntimeError, match='synthetic selector failure'):
        probe.make_selector(broken)(tile, gt, None, None, p)
    assert probe.original.constrained_bottleneck_path is original_function


def test_recompute_tracks_actual_fusion_and_empty_reset_without_mutating_logits():
    config = dict(affinity_deployment={})
    logits = torch.linspace(-1., 1., 8 * 3 * 4).reshape(1, 8, 3, 4)
    mode, options = probe.fusion_options(config)
    grid = probe.affinity_boundary_probability(logits.float(), mode=mode, **options).numpy()
    tile = dict(logits=logits, grid=grid.copy())
    initial = logits.clone()
    negative = np.zeros(tuple(logits.shape), bool)
    positive = negative.copy()
    negative[0, 0, 1, 1] = True

    def original_recompute(views, tiles, masks, cfg, shape, **kwargs):
        values = [probe.original.corrected_logits(t['logits'], n, p)[0, :4].sigmoid().numpy()
                  for t, (n, p) in zip(tiles, masks)]
        return np.zeros(shape, np.float32), dict(actual='synthetic return'), values

    recompute = probe.make_recompute(original_recompute)
    changed = recompute(None, [tile], [(negative, positive)], config, (3, 4))
    actual_logits = probe.original.corrected_logits(logits, negative, positive)
    actual_grid = probe.affinity_boundary_probability(actual_logits.float(), mode=mode, **options).numpy()
    assert np.array_equal(tile['readout_current_grid'], actual_grid)
    assert tile['readout_current_short_sha256'] == probe.array_sha(changed[2][0])
    assert not np.array_equal(actual_grid, grid)
    empty = np.zeros_like(negative)
    reset = recompute(None, [tile], [(empty, empty)], config, (3, 4), force_empty_cuda=True)
    assert np.array_equal(tile['readout_current_grid'], grid)
    assert tile['readout_current_short_sha256'] == probe.array_sha(reset[2][0])
    assert torch.equal(logits, initial) and np.array_equal(tile['grid'], grid)


def test_empty_recompute_rejects_captured_grid_mismatch():
    config = dict(affinity_deployment={})
    logits = torch.zeros((1, 8, 2, 2))
    tile = dict(logits=logits, grid=np.zeros((1, 1, 2, 2), np.float32))
    empty = np.zeros(tuple(logits.shape), bool)

    def fake(*args, **kwargs):
        return np.zeros((2, 2), np.float32), {}, [logits[0, :4].sigmoid().numpy()]

    with pytest.raises(RuntimeError, match='empty-grid capture differs'):
        probe.make_recompute(fake)(None, [tile], [(empty, empty)], config, (2, 2))


@pytest.mark.parametrize('fail', [False, True])
def test_outer_run_restores_both_original_entrypoints(monkeypatch, fail):
    selected = probe.original.select_bottlenecks
    recompute = probe.original.recompute
    observed = {}

    def fake_run(args):
        assert probe.original.select_bottlenecks is not selected
        assert probe.original.recompute is not recompute
        if fail:
            raise RuntimeError('synthetic run failure')

    monkeypatch.setattr(probe.original, 'run', fake_run)
    monkeypatch.setattr(probe, 'read', lambda path: dict(complete=True, frozen_states_exact=True))
    monkeypatch.setattr(probe, 'write', lambda path, value: observed.update(value))
    args = Namespace(output='outputs/fictional_readout')
    if fail:
        with pytest.raises(RuntimeError, match='synthetic run failure'):
            probe.run(args)
    else:
        probe.run(args)
        assert observed['format'] == 'affinity_readout_path_probe_v1'
        assert not observed['training'] and not observed['independent_accuracy']
    assert probe.original.select_bottlenecks is selected
    assert probe.original.recompute is recompute
