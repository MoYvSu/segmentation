# -*- coding: utf-8 -*-
"""显著阈值不能掩盖有深部贡献的残留小片。"""
import pytest

from tools.analyze_affinity_structure import residual_quality, summarize


def test_below_significant_threshold_fragment_remains_visible():
    quality = dict(gt_rows=[dict(gt=1, known_pixels=10000, deep_pixels=7000,
        dominant_pred=8, dominant_fraction=.9, dominant_deep_fraction=6990/7000,
        contributions=[dict(pred=8,pixels=9000,deep_pixels=6990),
                       dict(pred=9,pixels=590,deep_pixels=10),
                       dict(pred=0,pixels=410,deep_pixels=0)])])
    result = residual_quality(quality)
    assert not result['all_deep_in_target_main']
    assert result['secondary_deep_pixels'] == 10
    assert result['gt_rows'][0]['largest_secondary_pixels'] == 590
    assert result['gt_rows'][0]['unassigned_pixels'] == 410


def test_unassigned_deep_is_not_complete_repair():
    quality = dict(gt_rows=[dict(gt=1,known_pixels=100,deep_pixels=50,dominant_pred=8,
        dominant_fraction=.99,dominant_deep_fraction=.98,
        contributions=[dict(pred=8,pixels=99,deep_pixels=49),dict(pred=0,pixels=1,deep_pixels=1)])])
    result = residual_quality(quality)
    assert not result['all_deep_in_target_main']
    assert result['secondary_deep_pixels'] == 0
    assert result['unassigned_deep_pixels'] == 1


def test_border_residuals_do_not_hide_exact_deep_union():
    quality = dict(gt_rows=[dict(gt=1,known_pixels=100,deep_pixels=50,dominant_pred=8,
        dominant_fraction=.95,dominant_deep_fraction=1.,
        contributions=[dict(pred=8,pixels=95,deep_pixels=50),dict(pred=0,pixels=5,deep_pixels=0)])])
    result = residual_quality(quality)
    assert result['all_deep_in_target_main']
    assert result['gt_rows'][0]['unassigned_pixels'] == 5


@pytest.mark.parametrize('report', [dict(complete=False),
    dict(complete=True,format='affinity_structure_probe_v1',reference_cohort=dict(matched_F=688,safe_area=135))])
def test_incomplete_or_wrong_fixed_cohort_rejected(report):
    with pytest.raises(ValueError):
        summarize(report)
