# -*- coding: utf-8 -*-
"""网格诊断的合成几何检查；不需要赛题数据、checkpoint或GPU。"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import probe_affinity_grid_fidelity as probe


@pytest.fixture(scope='module')
def functions():
    return probe.original.load_functions(ROOT)


def draw(h=False, v=False, k=0):
    return dict(horizontal_flip=[h], vertical_flip=[v], rotation_k=[k])


def split_map():
    result = np.ones((8, 8), np.int32)
    result[:, 4:] = 2
    return result


def test_native_contact_counts_edges_not_endpoint_pixels():
    gt = split_map()
    edges = probe.contact_edges(gt, (1, 2))
    assert len(edges) == 8
    assert len(np.unique(edges.reshape(-1, 2), axis=0)) == 16
    gt[3:5, 3:5] = 0
    assert len(probe.contact_edges(gt, (1, 2))) == 6


@pytest.mark.parametrize('h,v,k', [(False, False, 0), (True, False, 1), (False, True, 3),
                                 (True, True, 0), (True, True, 2), (True, True, 3)])
def test_augmentation_maps_back_to_native_interface(functions, h, v, k):
    gt = split_map(); edges = probe.contact_edges(gt, (1, 2))
    row, distances, selected = probe.view_fidelity(gt, gt > 0, (1, 2), edges,
        (0, 0, 8, 8), draw(h, v, k), functions, input_size=8, output_grid=4)
    assert selected.all()
    assert row['groups']['cardinal']['negative_relations'] == 4
    assert np.all(distances['restored_cardinal'] == 0)
    assert row['nearest_restored_cardinal']['contact_edges'] == 8


def test_lost_thin_id_is_missing_not_zero_error(functions):
    gt = np.ones((8, 8), np.int32); gt[:, 3] = 2
    edges = probe.contact_edges(gt, (1, 2))
    row, distances, _ = probe.view_fidelity(gt, gt > 0, (1, 2), edges,
        (0, 0, 8, 8), draw(), functions, input_size=8, output_grid=4)
    assert row['sampled_pair_gt_pixels']['2'] == 0
    assert row['groups']['any']['negative_relations'] == 0
    assert np.isinf(distances['short']).all()
    summary = row['nearest_restored_cardinal']['native_contact_to_restored_contact']
    assert summary['no_reference_count'] == len(edges)
    assert summary['quantiles']['0.5'] is None
    assert summary['within_native_pixels']['4.0']['count'] == 0


def test_unknown_and_padding_are_ignored_filled_is_separate(functions):
    gt = split_map(); gt[0:2, 4:] = 0
    trust = gt > 0; trust[2:4, 4:] = False
    edges = probe.contact_edges(gt, (1, 2))
    row, _, _ = probe.view_fidelity(gt, trust, (1, 2), edges,
        (0, 0, 8, 8), draw(), functions, input_size=8, output_grid=4)
    cardinal = row['groups']['cardinal']
    assert cardinal['negative_relations'] == 3
    assert cardinal['both_original_covered'] == 2
    assert cardinal['includes_filled_endpoint'] == 1
    # 非方形letterbox补区及未知不会引入虚构负关系。
    row, _, _ = probe.view_fidelity(gt, trust, (1, 2), edges,
        (0, 0, 4, 8), draw(), functions, input_size=8, output_grid=4)
    assert row['geometry']['content_height'] == 2
    assert row['groups']['cardinal']['negative_relations'] == 1


def test_crop_censoring_and_unique_coverage_are_distinct(functions):
    gt = split_map(); edges = probe.contact_edges(gt, (1, 2))
    top = probe.view_fidelity(gt, gt > 0, (1, 2), edges,
        (0, 0, 4, 8), draw(), functions, input_size=8, output_grid=4)
    bottom = probe.view_fidelity(gt, gt > 0, (1, 2), edges,
        (4, 0, 8, 8), draw(), functions, input_size=8, output_grid=4)
    summary = probe.aggregate([top, top, bottom], len(edges))
    assert summary['interface_edges_ever_in_view'] == 8
    assert summary['interface_edge_exposures'] == 12
    assert summary['groups']['restored_cardinal']['unique_native_interface_coverage']['within_native_pixels']['0.0']['count'] == 8


def test_long_endpoints_can_span_unknown_without_relabeling(functions):
    gt = np.ones((8, 8), np.int32); gt[:, 4:] = 2; gt[:, 2:4] = 0
    edges = probe.contact_edges(gt, (1, 2))
    assert not len(edges)
    row, _, _ = probe.view_fidelity(gt, gt > 0, (1, 2), edges,
        (0, 0, 8, 8), draw(), functions, input_size=8, output_grid=4)
    assert row['groups']['cardinal']['negative_relations'] == 0
    assert row['groups']['long']['negative_relations'] > 0
    assert sum(c['segment_has_unknown'] for c in row['channels'][4:]) > 0


def test_old_receipt_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match='原生接触曝光'):
        probe.verify_old_totals({'interface_edge_exposures': 1}, {'baseline_crop_contact_edges': '2'}, 'baseline')
