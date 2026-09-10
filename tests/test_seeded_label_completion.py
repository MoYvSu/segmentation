# -*- coding: utf-8 -*-

import numpy as np
import pytest

from utils.seeded_label_completion import complete_narrow_instance_gaps


def _two_seed_example(gap_width: int = 3):
    height, width = 11, 10 + gap_width
    labels = np.zeros((height, width), dtype=np.int32)
    labels[:, :5] = 1
    labels[:, 5 + gap_width:] = 2
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : 5 + gap_width // 2] = 45
    image[:, 5 + gap_width // 2:] = 210
    return image, labels


def test_narrow_gap_is_filled_without_changing_markers():
    image, labels = _two_seed_example(gap_width=3)
    original = labels.copy()
    completed, fill, distance, audit = complete_narrow_instance_gaps(
        image, labels, max_fill_distance=2.0, blur_sigma=0.0
    )

    assert np.array_equal(completed[original > 0], original[original > 0])
    assert np.all(completed[:, 5:8] > 0)
    assert set(np.unique(completed)) == {1, 2}
    assert int(fill.sum()) == image.shape[0] * 3
    assert float(distance[:, 6].min()) == pytest.approx(2.0)
    assert audit["changed_original_pixels"] == 0
    assert audit["completed_instance_count"] == 2


def test_wide_gap_keeps_distant_pixels_unknown():
    image, labels = _two_seed_example(gap_width=7)
    completed, fill, distance, audit = complete_narrow_instance_gaps(
        image, labels, max_fill_distance=2.0, blur_sigma=0.0
    )

    assert np.all(completed[:, 7:10] == 0)
    assert np.all(~fill[:, 7:10])
    assert np.all(distance[:, 8] > 2.0)
    assert audit["residual_unknown_pixels"] > 0
    assert audit["fraction_of_gap_filled"] < 1.0


def test_invalid_radius_is_rejected():
    image, labels = _two_seed_example()
    with pytest.raises(ValueError, match="positive"):
        complete_narrow_instance_gaps(image, labels, max_fill_distance=0)
