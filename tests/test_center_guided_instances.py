import numpy as np

from utils.center_guided_instances import (
    assign_endpoints_to_centers,
    extract_center_peaks,
)
from utils.flow_instances import build_center_offset_target


def _two_instances():
    instance_map = np.zeros((64, 96), dtype=np.int32)
    instance_map[8:55, 6:42] = 1
    instance_map[12:58, 52:90] = 2
    return instance_map


def test_center_guidance_reconstructs_noisy_endpoints():
    instance_map = _two_instances()
    foreground = instance_map > 0
    offsets = build_center_offset_target(instance_map)
    yy, xx = np.indices(instance_map.shape, dtype=np.float32)
    rng = np.random.default_rng(7)
    endpoint_y = yy + offsets[0]
    endpoint_x = xx + offsets[1]
    endpoint_y[foreground] += rng.normal(0.0, 2.0, foreground.sum())
    endpoint_x[foreground] += rng.normal(0.0, 2.0, foreground.sum())
    heatmap = np.zeros(instance_map.shape, dtype=np.float32)
    for instance_id in (1, 2):
        mask = instance_map == instance_id
        centers = np.stack(
            [yy[mask] + offsets[0, mask], xx[mask] + offsets[1, mask]], axis=1
        )
        center_y, center_x = np.rint(centers[0]).astype(int)
        heatmap[center_y, center_x] = 0.95
    centers, scores, peak_audit = extract_center_peaks(heatmap, foreground)
    predicted, audit = assign_endpoints_to_centers(
        endpoint_y, endpoint_x, foreground, centers,
        max_assignment_distance=8.0,
    )
    assert peak_audit["kept_center_count"] == 2
    assert len(scores) == 2
    assert audit["kept_instances"] == 2
    assert audit["unassigned_foreground_pixels"] < int(foreground.sum() * 0.01)
    left_ids = set(np.unique(predicted[instance_map == 1])) - {0}
    right_ids = set(np.unique(predicted[instance_map == 2])) - {0}
    assert len(left_ids) == 1
    assert len(right_ids) == 1
    assert left_ids != right_ids


def test_center_extraction_respects_valid_mask_and_cap():
    probability = np.zeros((32, 32), dtype=np.float32)
    probability[4, 4] = 0.9
    probability[16, 16] = 0.8
    probability[28, 28] = 0.99
    valid = np.zeros_like(probability, dtype=bool)
    valid[:24] = True
    centers, scores, audit = extract_center_peaks(
        probability, valid, threshold=0.25, max_centers=1
    )
    assert centers.tolist() == [[4, 4]]
    assert np.allclose(scores, [0.9])
    assert audit == {
        "raw_center_count": 2,
        "kept_center_count": 1,
        "dropped_for_cap": 1,
    }
