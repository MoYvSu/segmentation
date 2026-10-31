# -*- coding: utf-8 -*-
"""修补前后身份隔离及相对拓扑口径，全部使用虚构数据。"""
import numpy as np
import pytest

from tools import probe_teacher_seams as probe


def test_generic_three_head_names_are_not_manual_aliases():
    assert probe.ARMS == ('r1', 'before', 'after')
    assert set(probe.EXPECTED) == set(probe.ARMS)
    assert len(set(probe.EXPECTED.values())) == 3
    assert probe.FOLDERS['before'] == 'outputs/gt_source/teacher'
    assert probe.FOLDERS['after'] == 'outputs/teacher_seams/teacher'


def test_status_validation_separates_diagnostic_name_from_training_arm(monkeypatch):
    from tools import evaluate_affinity_gt_source
    calls=[]
    def validate(status, config, arm, sha):
        calls.append((arm, sha)); return dict(status['contract'])
    monkeypatch.setattr(evaluate_affinity_gt_source, 'validate_training_status', validate)
    before=dict(paired_rows_verified=320, contract={})
    assert probe.validate_candidate_status(before, {}, 'before') == {}
    after=dict(paired_rows_verified=320, paired_actual_rows_verified=True,
               variant='teacher_zero_seam_v1', contract={'variant':'teacher_zero_seam_v1'})
    assert probe.validate_candidate_status(after, {}, 'after') == after['contract']
    assert calls == [('teacher',probe.EXPECTED['before']), ('teacher',probe.EXPECTED['after'])]
    with pytest.raises(ValueError):
        probe.validate_candidate_status(before, {}, 'manual')
    with pytest.raises(RuntimeError):
        probe.validate_candidate_status(after, {}, 'before')
    with pytest.raises(RuntimeError):
        probe.validate_candidate_status(before, {}, 'after')
    after['paired_actual_rows_verified'] = False
    with pytest.raises(RuntimeError):
        probe.validate_candidate_status(after, {}, 'after')


def test_relative_ferrite_split_merge_not_assumed_error_and_ids_invariant():
    from tools.evaluate_affinity_gt_source import compare_predictions
    one=np.ones((20,20),np.uint16)
    two=one.copy();two[:,10:]=2
    split=compare_predictions(one,{'1':1},two,{'1':1,'2':1})
    assert split['partition']['ferrite_significant_splits'] == 1
    assert split['partition']['ferrite_significant_merges'] == 0
    merge=compare_predictions(two,{'1':1,'2':1},one,{'1':1})
    assert merge['partition']['ferrite_significant_merges'] == 1
    renamed=np.where(two==1,23,65535).astype(np.uint16)
    equal=compare_predictions(two,{'1':1,'2':1},renamed,{'23':1,'65535':1})
    assert equal['partition']['complete_partition_identical'] is True
    assert split['official_accuracy'] is False and split['test_GT_used'] is False


def test_seen_gt_probability_masks_keep_original_coverage_requirement():
    gt=np.ones((20,20),np.uint16);gt[:,10:]=2
    known=np.ones((20,20),bool)
    assert probe.native_gt_masks(gt,known)['original_different_ID_band_radius2'].any()
    known[:,9:11]=False
    assert not probe.native_gt_masks(gt,known)['original_different_ID_band_radius2'].any()
