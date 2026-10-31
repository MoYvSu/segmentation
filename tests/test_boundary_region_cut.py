# -*- coding: utf-8 -*-
"""完整八邻顶点割的独立构造检查，不读取比赛原图或GT。"""
import numpy as np
import pytest

import utils.boundary_region_cut as region_cut
from utils.boundary_region_cut import select_region_cut


def ring(gaps=()):
    p = np.full((21, 25), .1, np.float32)
    p[4, 6:19] = p[16, 6:19] = .9
    p[4:17, 6] = p[4:17, 18] = .9
    for y, x in gaps:
        p[y, x] = .1
    source = np.zeros_like(p, bool); source[8:11, 10:13] = True
    sink = np.zeros_like(p, bool); sink[1:3, 1:3] = True
    allowed = np.zeros_like(p, bool)
    for y, x in gaps:
        allowed[y, x] = True
    return p, allowed, source, sink


def test_complete_cut_closes_two_independent_escape_routes_together():
    data = ring([(4, 10), (16, 14)])
    before = [a.copy() for a in data]
    cut, audit = select_region_cut(*data)
    expected = np.zeros_like(cut); expected[4, 10] = expected[16, 14] = True
    np.testing.assert_array_equal(cut, expected)
    assert audit['feasible'] and audit['certificate_valid']
    assert audit['reason'] == 'selected_complete_separator'
    assert audit['source_sink_connected_before'] and not audit['source_sink_connected_after']
    assert audit['flow_value'] == audit['new_cut_capacity_sum'] < audit['infinite_capacity']
    assert audit['elapsed_seconds'] >= 0
    for old, new in zip(before, data):
        np.testing.assert_array_equal(old, new)


def test_diagonal_leak_is_a_real_eight_neighbor_route():
    p = np.full((5, 5), .9, np.float32)
    for y, x in [(1, 1), (2, 2), (3, 3)]:p[y, x] = .1
    allowed = np.zeros_like(p, bool); allowed[2, 2] = True
    source = np.zeros_like(allowed); source[1, 1] = True
    sink = np.zeros_like(allowed); sink[3, 3] = True
    cut, audit = select_region_cut(p, allowed, source, sink)
    assert audit['source_sink_connected_before'] and audit['certificate_valid']
    assert cut.sum() == 1 and cut[2, 2]


def test_unknown_uneditable_route_cannot_be_deleted_or_partially_supervised():
    p, allowed, source, sink = ring([(4, 10), (16, 14)])
    allowed[16, 14] = False  # 一条真实空间通路未知，必须保留，不能当成墙。
    cut, audit = select_region_cut(p, allowed, source, sink)
    assert not cut.any() and not audit['feasible'] and not audit['certificate_valid']
    assert audit['reason'] == 'no_modifiable_complete_separator'
    assert audit['flow_value'] is None
    assert audit['flow_lower_bound'] >= audit['infinite_capacity']
    assert audit['immutable_route_connected'] and not audit['maxflow_called']
    assert audit['source_sink_connected_after'] and not audit['unknown_nodes_removed']


def test_exterior_node_prevents_open_frame_to_frame_line_from_becoming_a_false_wall():
    p = np.full((9, 11), .1, np.float32); p[:, 5] = .9; p[4, 5] = .1
    allowed = np.zeros_like(p, bool); allowed[4, 5] = True
    source = np.zeros_like(allowed); source[4, 2] = True
    sink = np.zeros_like(allowed); sink[4, 8] = True
    cropped, local = select_region_cut(p, allowed, source, sink, exterior=False)
    assert cropped.sum() == 1 and local['certificate_valid']
    cut, full = select_region_cut(p, allowed, source, sink, exterior=True)
    assert not cut.any() and not full['feasible']
    assert full['reason'] == 'no_modifiable_complete_separator'


def test_allowed_ring_controls_where_separator_can_form_and_never_cuts_seed():
    p = np.full((13, 13), .1, np.float32)
    source = np.zeros_like(p, bool); source[5:8, 5:8] = True
    sink = np.zeros_like(p, bool); sink[1, 1] = True
    allowed = np.zeros_like(p, bool)
    allowed[2, 2:11] = allowed[10, 2:11] = True
    allowed[2:11, 2] = allowed[2:11, 10] = True
    cut, audit = select_region_cut(p, allowed, source, sink)
    assert audit['certificate_valid'] and cut.sum() == allowed.sum()
    assert np.all(~cut | allowed) and not (cut & (source | sink)).any()
    # 故意开放一个未知位置，不能退而选择种子周围的非法小圈。
    allowed[2, 6] = False
    rejected, invalid = select_region_cut(p, allowed, source, sink)
    assert not rejected.any() and not invalid['feasible']


def test_existing_complete_wall_is_satisfied_without_any_new_cut():
    cut, audit = select_region_cut(*ring())
    assert not cut.any() and audit['feasible'] and audit['certificate_valid']
    assert audit['reason'] == 'already_separated'
    assert audit['graph_nodes'] == 0 and audit['flow_value'] is None


@pytest.mark.parametrize('which', ['source', 'sink'])
def test_wall_seed_cannot_disappear_and_be_called_success(which):
    p, allowed, source, sink = ring([(4, 10)])
    p[source if which == 'source' else sink] = .9
    cut, audit = select_region_cut(p, allowed, source, sink)
    assert not cut.any() and not audit['feasible']
    assert audit['reason'] == 'no_open_' + which + '_core'


def test_no_allowed_pixel_is_explicitly_infeasible():
    p, allowed, source, sink = ring([(4, 10)])
    allowed[:] = False
    cut, audit = select_region_cut(p, allowed, source, sink)
    assert not cut.any() and not audit['feasible']
    assert audit['infinite_capacity'] == 1


def test_immutable_route_rejects_before_maxflow_or_graph_construction(monkeypatch):
    p, allowed, source, sink = ring([(4, 10), (16, 14)])
    allowed[16, 14] = False

    def forbidden(*args, **kwargs):
        raise AssertionError('An immutable escape route must not invoke maxflow or graph construction')

    monkeypatch.setattr(region_cut, 'maximum_flow', forbidden)
    monkeypatch.setattr(region_cut, '_graph', forbidden)
    cut, audit = select_region_cut(p, allowed, source, sink)
    assert not cut.any() and not audit['feasible']
    assert audit['reason'] == 'no_modifiable_complete_separator'
    assert audit['immutable_route_precheck_ran'] and audit['immutable_route_connected']
    assert audit['graph_nodes'] == audit['graph_arcs'] == 0
    assert audit['flow_value'] is None and not audit['maxflow_called']
    assert audit['source_sink_connected_before'] and audit['source_sink_connected_after']


def test_immutable_exterior_route_also_skips_maxflow(monkeypatch):
    p = np.full((9, 11), .1, np.float32); p[:, 5] = .9; p[4, 5] = .1
    allowed = np.zeros_like(p, bool); allowed[4, 5] = True
    source = np.zeros_like(allowed); source[4, 2] = True
    sink = np.zeros_like(allowed); sink[4, 8] = True
    monkeypatch.setattr(region_cut, 'maximum_flow', lambda *args, **kwargs: pytest.fail('maxflow called'))
    cut, audit = select_region_cut(p, allowed, source, sink, exterior=True)
    assert not cut.any() and audit['immutable_route_connected']
    assert not audit['maxflow_called'] and audit['reason'] == 'no_modifiable_complete_separator'


def test_cut_touching_frame_is_audited_instead_of_hidden():
    p = np.full((11, 13), .1, np.float32)
    p[0, 2:11] = p[8, 2:11] = .9
    p[:9, 2] = p[:9, 10] = .9
    p[0, 6] = .1
    allowed = np.zeros_like(p, bool); allowed[0, 6] = True
    source = np.zeros_like(p, bool); source[3:6, 5:8] = True
    sink = np.zeros_like(p, bool); sink[9, 1] = True
    cut, audit = select_region_cut(p, allowed, source, sink)
    assert cut[0, 6] and audit['certificate_valid'] and audit['cut_touches_frame']


@pytest.mark.parametrize('parameter,value', [
    ('threshold', 0), ('threshold', 1), ('threshold', float('nan')), ('threshold', True),
    ('capacity_scale', 0), ('capacity_scale', 1.2), ('capacity_scale', True), ('exterior', 'yes'),
])
def test_invalid_settings_rejected(parameter, value):
    with pytest.raises(ValueError):select_region_cut(*ring([(4, 10)]), **{parameter: value})


@pytest.mark.parametrize('which', [0, 1, 2, 3])
def test_shape_or_mask_dtype_errors_rejected(which):
    values = list(ring([(4, 10)]))
    values[which] = values[which].ravel()
    with pytest.raises(ValueError):select_region_cut(*values)
    if which:
        values = list(ring([(4, 10)])); values[which] = values[which].astype(np.uint8)
        with pytest.raises(ValueError):select_region_cut(*values)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -0.01, 1.01])
def test_invalid_probability_rejected(bad):
    p, allowed, source, sink = ring([(4, 10)]); p[0, 0] = bad
    with pytest.raises(ValueError):select_region_cut(p, allowed, source, sink)


def test_overlapping_seeds_rejected_and_capacity_overflow_not_silently_clipped():
    p, allowed, source, sink = ring([(4, 10)])
    sink[source] = True
    with pytest.raises(ValueError, match='overlap'):select_region_cut(p, allowed, source, sink)
    p, allowed, source, sink = ring([(4, 10), (16, 14)])
    with pytest.raises(ValueError, match='int32'):select_region_cut(
        p, allowed, source, sink, capacity_scale=np.iinfo(np.int32).max)


def _python_connected(walls, source, sink, exterior):
    # 独立逐点BFS，不复用选择器的OpenCV/残量图实现。
    height, width = walls.shape
    stack = [tuple(position) for position in np.argwhere(source)]
    visited = set(stack); exterior_expanded = False
    while stack:
        y, x = stack.pop()
        if sink[y, x]:return True
        neighbors = [(y+dy, x+dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                     if dy or dx]
        if exterior and not exterior_expanded and (y in [0, height-1] or x in [0, width-1]):
            exterior_expanded = True
            neighbors.extend((int(a), int(b)) for a, b in np.argwhere(~walls)
                             if a in [0, height-1] or b in [0, width-1])
        for a, b in neighbors:
            if 0 <= a < height and 0 <= b < width and not walls[a, b] and (a, b) not in visited:
                visited.add((a, b));stack.append((a, b))
    return False


@pytest.mark.parametrize('exterior', [False, True])
def test_small_graph_cut_cost_matches_exhaustive_independent_native_search(exterior):
    rng = np.random.default_rng(971)
    for _ in range(16):
        p = np.where(rng.random((5, 6)) < .45, .9, .1).astype(np.float32)
        source = np.zeros_like(p, bool); source[2, 1] = True
        sink = np.zeros_like(p, bool); sink[2, 4] = True
        p[source | sink] = .1
        candidates = np.argwhere((p < .65) & ~source & ~sink)
        chosen = rng.choice(len(candidates), size=min(7, len(candidates)), replace=False)
        coords = candidates[chosen]
        allowed = np.zeros_like(p, bool)
        for y, x in coords:allowed[y, x] = True
        best = None
        for bits in range(1 << len(coords)):
            trial = np.zeros_like(allowed); cost = 0
            for i, (y, x) in enumerate(coords):
                if bits & (1 << i):
                    trial[y, x] = True
                    cost += 1 + round(1000 * (1 - float(p[y, x])))
            if not _python_connected((p >= .65) | trial, source, sink, exterior):
                best = cost if best is None else min(best, cost)
        cut, audit = select_region_cut(p, allowed, source, sink, exterior=exterior)
        assert audit['feasible'] is (best is not None)
        if best is None:
            assert not cut.any()
        else:
            actual = sum(1 + round(1000 * (1 - float(p[y, x]))) for y, x in np.argwhere(cut))
            assert actual == best
            assert not _python_connected((p >= .65) | cut, source, sink, exterior)
