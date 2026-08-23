import numpy as np

from utils.flow_instances import (
    build_center_offset_target,
    build_edt_flow_target,
    build_tile_local_center_offset_target,
    cluster_endpoints,
    endpoints_from_center_offsets,
    integrate_flow,
)


def _two_instances():
    instance_map = np.zeros((64, 96), dtype=np.int32)
    instance_map[8:55, 6:42] = 1
    instance_map[12:58, 52:90] = 2
    return instance_map


def test_exact_center_offsets_reconstruct_two_instances():
    instance_map = _two_instances()
    foreground = instance_map > 0
    offsets = build_center_offset_target(instance_map)
    end_y, end_x = endpoints_from_center_offsets(
        offsets, foreground, 0.0, np.random.default_rng(1)
    )
    predicted, audit = cluster_endpoints(end_y, end_x, foreground, close_radius=1)
    assert audit["kept_instances"] == 2
    assert np.all(predicted[instance_map == 1] == predicted[20, 20])
    assert np.all(predicted[instance_map == 2] == predicted[20, 70])
    assert predicted[20, 20] != predicted[20, 70]


def test_edt_flow_keeps_separated_rectangles_separate():
    instance_map = _two_instances()
    foreground = instance_map > 0
    flow = build_edt_flow_target(instance_map, smooth_sigma=1.0)
    end_y, end_x = integrate_flow(flow, foreground, steps=80)
    predicted, audit = cluster_endpoints(end_y, end_x, foreground, close_radius=1)
    assert audit["kept_instances"] == 2
    left_ids = set(np.unique(predicted[instance_map == 1])) - {0}
    right_ids = set(np.unique(predicted[instance_map == 2])) - {0}
    assert len(left_ids) == 1
    assert len(right_ids) == 1
    assert left_ids != right_ids


def test_tile_local_offsets_preserve_shape_and_finite_values():
    instance_map = _two_instances()
    offsets = build_tile_local_center_offset_target(
        instance_map, tile_size=40, overlap=12
    )
    assert offsets.shape == (2, *instance_map.shape)
    assert np.isfinite(offsets).all()
    assert np.all(offsets[:, instance_map == 0] == 0)