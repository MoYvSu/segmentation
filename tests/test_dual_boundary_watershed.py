import cv2
import numpy as np

from utils.post_process import (
    _instance_semantic_vote,
    _merge_region_map_to_cap,
    boundary_watershed_separation,
    reconstruct_marker_boundary,
)


def _instance_count(instance_map):
    return len([value for value in np.unique(instance_map) if int(value) > 0])


def test_independent_marker_boundary_can_close_a_seed_leak():
    semantic = np.ones((48, 64), dtype=np.uint8)
    primary_boundary = np.zeros_like(semantic)
    marker_boundary = np.zeros_like(semantic)
    marker_boundary[:, 31:33] = 1

    baseline, _ = boundary_watershed_separation(
        semantic,
        primary_boundary,
        min_area=20,
        bridge_width=0,
        dilate_width=0,
    )
    repaired, _ = boundary_watershed_separation(
        semantic,
        primary_boundary,
        min_area=20,
        bridge_width=0,
        dilate_width=0,
        marker_boundary_mask=marker_boundary,
        marker_bridge_width=0,
        marker_dilate_width=0,
    )

    assert _instance_count(baseline) == 1
    assert _instance_count(repaired) == 2


def test_none_marker_boundary_preserves_default_output():
    semantic = np.ones((32, 40), dtype=np.uint8)
    boundary = np.zeros_like(semantic)
    boundary[:, 19:21] = 1

    expected = boundary_watershed_separation(
        semantic, boundary, min_area=10, bridge_width=1, dilate_width=1
    )
    actual = boundary_watershed_separation(
        semantic,
        boundary,
        min_area=10,
        bridge_width=1,
        dilate_width=1,
        marker_boundary_mask=None,
    )

    assert np.array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_marker_border_seal_prevents_frame_bypass():
    semantic = np.ones((48, 64), dtype=np.uint8)
    boundary = np.zeros_like(semantic)
    boundary[2:, 31:33] = 1

    leaking, _ = boundary_watershed_separation(
        semantic,
        boundary,
        min_area=20,
        bridge_width=0,
        dilate_width=0,
    )
    sealed, _ = boundary_watershed_separation(
        semantic,
        boundary,
        min_area=20,
        bridge_width=0,
        dilate_width=0,
        marker_border_seal_width=3,
    )

    assert _instance_count(leaking) == 1
    assert _instance_count(sealed) == 2


def test_marker_reconstruction_closes_local_gap_without_absorbing_distant_fog():
    probability = np.zeros((24, 32), dtype=np.float32)
    probability[12, 4:20] = 0.90
    probability[12, 11] = 0.50
    probability[2:6, 25:29] = 0.50

    reconstructed = reconstruct_marker_boundary(
        probability, low_threshold=0.45, high_threshold=0.72, max_steps=2
    )

    assert reconstructed[12, 11] == 1
    assert not np.any(reconstructed[2:6, 25:29])


def test_probability_vote_can_override_weak_hard_majority():
    instance = np.ones((10, 10), dtype=bool)
    semantic = np.zeros((10, 10), dtype=np.uint8)
    semantic.flat[:51] = 1
    probability = np.full((10, 10), 0.10, dtype=np.float32)
    probability.flat[:51] = 0.51

    hard_cls, hard_score = _instance_semantic_vote(instance, semantic)
    probability_cls, probability_score = _instance_semantic_vote(
        instance,
        semantic,
        semantic_probability=probability,
        mode="probability_mean",
    )

    assert hard_cls == 1 and hard_score > 0.5
    assert probability_cls == 0 and probability_score < 0.5


def test_instance_cap_merges_locally_and_keeps_each_id_connected():
    region_map = np.zeros((12, 23), dtype=np.int32)
    for label in range(1, 7):
        start = (label - 1) * 4
        region_map[:, start : start + 3] = label

    merged, merge_edges = _merge_region_map_to_cap(region_map, max_regions=2)
    labels = [int(value) for value in np.unique(merged) if int(value) > 0]

    assert len(labels) == 2
    assert len(merge_edges) == 4
    for label in labels:
        components, _ = cv2.connectedComponents((merged == label).astype(np.uint8), 8)
        assert components - 1 == 1
