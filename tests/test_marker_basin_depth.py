# -*- coding: utf-8 -*-
"""灰度鞍点与种子撤销合同，不把少种子当作准确率改善。"""
import heapq

import numpy as np
import pytest

from utils.marker_basin_depth import seed_basin_depths, filter_shallow_seeds


def _brute(B, source, targets):
    dist = np.full(B.shape, np.inf)
    y, x = source; dist[y, x] = B[y, x]
    heap = [(float(B[y, x]), y, x)]
    while heap:
        cost, y, x = heapq.heappop(heap)
        if cost != dist[y, x]: continue
        if (y, x) in targets: return cost
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                yy, xx = y+dy, x+dx
                if (not (dy or dx) or not 0 <= yy < B.shape[0] or not 0 <= xx < B.shape[1]): continue
                candidate = max(cost, float(B[yy, xx]))
                if candidate < dist[yy, xx]:
                    dist[yy, xx] = candidate; heapq.heappush(heap, (candidate, yy, xx))
    raise AssertionError('Full finite image should be connected')


def test_tree_matches_brute_minimax_on_many_small_images():
    rng = np.random.default_rng(71027)
    for n in range(35):
        B = (rng.integers(0, 21, (7, 8))/20).astype(np.float32)
        markers = np.zeros(B.shape, np.int32)
        places = rng.choice(B.size, size=5, replace=False)
        for i, index in enumerate(places): markers.flat[index] = [3, 17, 8001, 40000, 1000001][i]
        info = seed_basin_depths(markers, B)
        order = sorted(info['seeds'], key=lambda r: (r['birth'], r['seed_id']))
        for i, row in enumerate(order):
            if i == 0:
                assert row['persistent_root'] and row['depth'] is None
            else:
                x, y = row['anchor_xy']
                targets = {(r['anchor_xy'][1], r['anchor_xy'][0]) for r in order[:i]}
                expected = _brute(B, (y, x), targets) - row['birth']
                assert row['depth'] == pytest.approx(expected, abs=1e-12)


def test_zero_threshold_and_single_seed_are_identity_without_mutation():
    B = np.full((6, 7), .6, np.float32)
    m = np.zeros(B.shape, np.int32); m[2:4, 2:4] = 91
    m0, B0 = m.copy(), B.copy()
    out, audit = filter_shallow_seeds(m, B, 0)
    assert np.array_equal(out, m) and audit['disabled']
    out, audit = filter_shallow_seeds(m, B, .1)
    assert np.array_equal(out, m) and audit['removed_seed_count'] == 0
    assert np.array_equal(m, m0) and np.array_equal(B, B0)


def test_same_basin_shoulder_revoked_deep_seed_keeps_original_id_and_pixels():
    B = np.full((7, 9), .8, np.float32)
    B[2:5, 1:8] = .2
    m = np.zeros(B.shape, np.int32); m[3, 2] = 8; m[3, 6] = 57
    info = seed_basin_depths(m, B)
    out, audit = filter_shallow_seeds(m, B, .05, analysis=info)
    assert audit['removed_ids'] == [57]
    assert out[3, 2] == 8 and out[3, 6] == 0
    assert set(np.unique(out)) == {0, 8}


def test_anchor_is_minimum_then_row_major_and_ties_use_seed_id():
    B = np.full((5, 6), .7, np.float32)
    m = np.zeros(B.shape, np.int32); m[1:3, 1:3] = 100; m[3, 4] = 4
    B[1:3, 1:3] = .2; B[3, 4] = .2
    info = seed_basin_depths(m, B)
    rows = {r['seed_id']: r for r in info['seeds']}
    assert rows[100]['anchor_xy'] == [1, 1]
    assert rows[4]['persistent_root']
    assert rows[100]['depth'] == pytest.approx(.5)


def test_deeper_threshold_removes_superset_and_rejects_stale_analysis():
    B = np.full((9, 13), .8, np.float32)
    B[4, 1:12] = [.1, .15, .15, .4, .3, .3, .3, .7, .65, .65, .65]
    m = np.zeros(B.shape, np.int32); m[4, 1] = 1; m[4, 5] = 2; m[4, 10] = 3
    info = seed_basin_depths(m, B)
    r1 = filter_shallow_seeds(m, B, .05, analysis=info)[1]['removed_ids']
    r2 = filter_shallow_seeds(m, B, .15, analysis=info)[1]['removed_ids']
    assert set(r1) <= set(r2)
    changed = B.copy(); changed[0, 0] = .9
    with pytest.raises(ValueError): filter_shallow_seeds(m, changed, .1, analysis=info)


def test_empty_seed_and_tiny_image_and_validation():
    m = np.zeros((1, 3), np.int32); B = np.array([[.1, .8, .2]], np.float32)
    assert seed_basin_depths(m, B)['seeds'] == []
    m[0, 0] = 1; m[0, 2] = 2
    rows = seed_basin_depths(m, B)['seeds']
    assert rows[1]['depth'] == pytest.approx(float(B[0, 1]-B[0, 2]), abs=1e-7)
    with pytest.raises(ValueError): seed_basin_depths(m.astype(np.uint16), B)
    with pytest.raises(ValueError): filter_shallow_seeds(m, B, -.1)
    with pytest.raises(ValueError): filter_shallow_seeds(m, B, True)
