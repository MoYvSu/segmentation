import numpy as np
import pytest

from tools.build_sam2_geometry_dataset import (
    build_class_agnostic_geometry_targets,
    enforce_source_limit,
    partition_mask_records,
)


def _record(mask, score=0.95):
    return {
        "segmentation": mask,
        "predicted_iou": score,
        "stability_score": score,
    }


def test_source_limit_is_strictly_below_250():
    assert enforce_source_limit([f"h{i}" for i in range(249)]) == 249
    with pytest.raises(ValueError, match="hard limit"):
        enforce_source_limit([f"h{i}" for i in range(250)])


def test_partition_produces_unique_instance_assignment():
    first = np.zeros((8, 8), dtype=bool)
    second = np.zeros((8, 8), dtype=bool)
    first[1:6, 1:5] = True
    second[2:7, 4:7] = True
    labels, scores, stability, rejected = partition_mask_records(
        [_record(first, 0.99), _record(second, 0.98)],
        first.shape,
        min_area=3,
        max_area_fraction=0.9,
        max_overlap_fraction=0.5,
    )
    assert labels.max() == 2
    assert set(np.unique(labels)) == {0, 1, 2}
    assert scores.shape == stability.shape == (3,)
    assert rejected["overlap"] == 0


def test_partition_refuses_silent_instance_truncation():
    records = []
    for x in (0, 2, 4):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, x : x + 2] = True
        records.append(_record(mask))
    with pytest.raises(ValueError, match="refusing silent truncation"):
        partition_mask_records(
            records,
            (6, 6),
            min_area=1,
            max_area_fraction=0.9,
            max_overlap_fraction=0.0,
            max_instances=2,
        )


def test_class_agnostic_targets_keep_interfaces_without_shrink():
    labels = np.zeros((10, 12), dtype=np.uint16)
    labels[2:8, 1:6] = 1
    labels[2:8, 6:11] = 2
    interiors, boundary, soft, valid = build_class_agnostic_geometry_targets(
        labels, shrink_radius=0, boundary_radius=2
    )
    assert np.array_equal(interiors, labels)
    assert boundary[:, 5:7].any()
    assert soft.shape == labels.shape
    assert np.all(valid[interiors > 0])


def test_class_agnostic_targets_turn_eroded_rim_into_boundary():
    labels = np.zeros((11, 11), dtype=np.uint16)
    labels[1:10, 1:10] = 1
    interiors, boundary, _, valid = build_class_agnostic_geometry_targets(
        labels, shrink_radius=2, boundary_radius=2
    )
    assert np.sum(interiors > 0) < np.sum(labels > 0)
    assert boundary[1:10, 1:10].any()
    assert not np.any((interiors > 0) & (boundary > 0))
    assert np.all(valid[(interiors > 0) | (boundary > 0)])