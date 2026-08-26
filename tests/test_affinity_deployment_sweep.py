import argparse

import pytest

from tools.evaluate_affinity_deployment_sweep import (
    flatten_candidate,
    parse_checkpoint,
    selection_key,
)


def _summary(score_total, instance_miou, area_error):
    return {
        "score_total": score_total,
        "score_miou": 50.0 * instance_miou,
        "score_area": 50.0 * max(0.0, 1.0 - area_error),
        "instance_miou_valid": instance_miou,
        "gt_penalized_miou": 0.4,
        "ferrite_area_relative_error": area_error,
        "ferrite_mean_area_gt": 100.0,
        "ferrite_mean_area_pred": 100.0 * (1.0 + area_error),
        "gt_count": 20,
        "pred_count": 18,
        "valid_matches": 15,
        "classes": {
            "ferrite": {"valid_matches": 10, "merged_pred_count": 2},
            "pearlite": {"valid_matches": 5, "merged_pred_count": 1},
        },
    }


def test_parse_checkpoint_requires_safe_alias_and_path():
    assert parse_checkpoint("g3=outputs/g3.pth") == ("g3", "outputs/g3.pth")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_checkpoint("outputs/g3.pth")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_checkpoint("../g3=outputs/g3.pth")


def test_selection_prefers_official_total_before_diagnostic_miou():
    lower_total = flatten_candidate(
        "g2", "affinity_gated_bt055", _summary(80.0, 0.90, 0.30), "g2.pth"
    )
    higher_total = flatten_candidate(
        "g3", "affinity_gated_bt055", _summary(81.0, 0.80, 0.10), "g3.pth"
    )
    assert max([lower_total, higher_total], key=selection_key) == higher_total


def test_selection_tie_breaks_by_miou_then_area_error():
    a = flatten_candidate(
        "a", "affinity_gated_bt055", _summary(80.0, 0.80, 0.10), "a.pth"
    )
    b = flatten_candidate(
        "b", "affinity_gated_bt055", _summary(80.0, 0.81, 0.20), "b.pth"
    )
    c = flatten_candidate(
        "c", "affinity_gated_bt055", _summary(80.0, 0.81, 0.15), "c.pth"
    )
    assert max([a, b, c], key=selection_key) == c