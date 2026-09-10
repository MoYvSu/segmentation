# -*- coding: utf-8 -*-

import numpy as np

from utils.musam_instances import (
    build_musam_targets,
    musam_watershed,
    perturb_musam_predictions,
)


def test_nonconvex_center_is_corrected_and_stays_one_instance():
    labels = np.zeros((36, 36), dtype=np.int32)
    labels[4:29, 4:9] = 7
    labels[4:29, 27:32] = 7
    labels[24:29, 4:32] = 7

    targets, valid, audit = build_musam_targets(labels)
    prediction, debug = musam_watershed(targets, return_debug=True)

    assert audit["corrected_centers"] == 1
    assert valid.sum() == (labels > 0).sum()
    assert debug["marker_count"] == 1
    assert set(np.unique(prediction)) == {0, 1}
    gt_foreground = labels > 0
    pred_foreground = prediction > 0
    foreground_iou = (gt_foreground & pred_foreground).sum() / (
        gt_foreground | pred_foreground
    ).sum()
    assert foreground_iou > 0.97


def test_touching_instances_are_split_without_reading_instance_count():
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:28, 4:16] = 3
    labels[4:28, 16:28] = 19

    targets, _, _ = build_musam_targets(labels)
    prediction, debug = musam_watershed(targets, return_debug=True)

    assert debug["marker_count"] == 2
    assert int(prediction.max()) == 2
    gt_foreground = labels > 0
    pred_foreground = prediction > 0
    foreground_iou = (gt_foreground & pred_foreground).sum() / (
        gt_foreground | pred_foreground
    ).sum()
    assert foreground_iou > 0.99
    assert prediction[12, 10] != prediction[12, 22]


def test_disconnected_same_id_is_relabelled_like_official_transform():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[3:12, 3:12] = 7
    labels[18:27, 18:27] = 7

    targets, _, audit = build_musam_targets(labels)
    _, debug = musam_watershed(targets, return_debug=True)

    assert audit["original_instance_count"] == 1
    assert audit["instance_count"] == 2
    assert audit["component_split_increase"] == 1
    assert debug["marker_count"] == 2


def test_diagonal_marker_pixels_use_official_full_connectivity():
    predictions = np.ones((3, 8, 8), dtype=np.float32)
    predictions[0] = 1.0
    predictions[1:, 2, 2] = 0.0
    predictions[1:, 3, 3] = 0.0

    _, debug = musam_watershed(
        predictions,
        foreground_smoothing=0.0,
        distance_smoothing=0.0,
        return_debug=True,
    )

    assert debug["marker_count"] == 1


def test_uncovered_pixels_remain_ignored_and_distance_fill_is_one():
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[3:13, 4:12] = 1
    content = np.ones_like(labels, dtype=bool)
    content[:, 14:] = False

    targets, valid, audit = build_musam_targets(labels, content)

    assert not valid[0, 0]
    assert not valid[5, 14]
    assert targets[0, 0, 0] == 0.0
    assert targets[1, 0, 0] == 1.0
    assert targets[2, 0, 0] == 1.0
    assert audit["ignored_pixels"] == int(content.sum() - valid.sum())
    assert audit["padding_pixels"] == int((~content).sum())


def test_light_noise_is_deterministic_and_clipped():
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:10, 2:10] = 1
    targets, _, _ = build_musam_targets(labels)

    first = perturb_musam_predictions(
        targets, 0.03, np.random.default_rng(123)
    )
    second = perturb_musam_predictions(
        targets, 0.03, np.random.default_rng(123)
    )

    assert np.array_equal(first, second)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
