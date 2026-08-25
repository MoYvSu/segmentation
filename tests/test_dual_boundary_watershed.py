import numpy as np

from utils.post_process import boundary_watershed_separation


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