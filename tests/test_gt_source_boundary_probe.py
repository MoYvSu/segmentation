# -*- coding: utf-8 -*-
"""虚构概率和GT验证诊断统计，不读取赛题数据或调用真实推理。"""
import numpy as np
import pytest

from tools.probe_gt_source_boundary import calibrate_same_count, native_gt_masks, probability_summary


def test_same_count_ties_report_actual_recall_and_specificity():
    positive=np.array([.9,.8,.8,.1],np.float32)
    negative=np.array([.79,.8,.81],np.float32)
    value=calibrate_same_count(positive,negative,2)
    assert value['same_count']==3 and value['same_recall']==.75
    assert value['tied_same_at_threshold']==2 and value['recall_excess_due_ties']==.25
    assert value['different_specificity']==pytest.approx(1/3)


def test_zero_full_and_missing_positive_targets_are_explicit():
    pos=np.array([.2,.7],np.float32);neg=np.array([.1,.5],np.float32)
    assert calibrate_same_count(pos,neg,0)['same_count']==0
    assert calibrate_same_count(pos,neg,2)['same_count']==2
    assert calibrate_same_count(pos[:0],neg,0)['same_recall'] is None
    with pytest.raises(ValueError):
        calibrate_same_count(np.array([np.nan],np.float32),neg,1)


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_zero_target_disconnects_all_probabilities_in_every_supported_precision(dtype):
    pos=np.array([0.,.2,.7,1.],dtype);neg=np.array([0.,.5,.9,1.],dtype)
    value=calibrate_same_count(pos,neg,0)
    assert value['threshold']>1.
    assert value['requested_same_count']==value['same_count']==0
    assert value['same_recall']==0.
    assert value['different_specificity']==1.
    assert value['tied_same_at_threshold']==0
    assert value['recall_excess_due_ties']==0.


@pytest.mark.parametrize('dtype',[np.float16,np.float32,np.float64])
def test_missing_positive_targets_keep_none_contract_in_every_precision(dtype):
    value=calibrate_same_count(np.array([],dtype),np.array([0.,1.],dtype),0)
    assert value['threshold'] is None
    assert value['same_recall'] is None
    assert value['different_specificity'] is None
    assert value['recall_excess_due_ties'] is None


def test_native_bands_ignore_unknown_and_require_original_coverage():
    gt=np.ones((20,20),np.uint16);gt[:,10:]=2;original=np.ones(gt.shape,bool)
    masks=native_gt_masks(gt,original)
    assert masks['original_different_ID_band_radius2'][8,9]
    assert masks['original_deep_local_same'][8,4]
    original[:,9:11]=False
    assert not native_gt_masks(gt,original)['original_different_ID_band_radius2'].any()


def test_probability_means_and_thresholds_use_absolute_scale():
    value=probability_summary(np.array([.1,.45,.65,.9],np.float32))
    assert value['mean']==pytest.approx(.525)
    assert value['ge_045']==.75 and value['ge_065']==.5
    with pytest.raises(ValueError):
        probability_summary(np.array([1.01],np.float32))
