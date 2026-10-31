# -*- coding: utf-8 -*-
"""四draw最终划分诊断的CPU合成删失与匹配测试；不访问赛题图像。"""
import numpy as np
from tools.analyze_affinity_fused_gt import common_safe_change, diagnose_partition, self_test


def test_synthetic_self_test():
    self_test()


def test_unknown_ignored_in_same_prediction_matching():
    gt=np.zeros((31,31),np.int32);gt[5:26,5:26]=1
    pred=np.ones(gt.shape,np.uint16)
    result=diagnose_partition(gt,pred,{1:1},{'1':1})
    assert result['pairs'][0]['iou']==1.
    assert result['conservative']['matched']==0
    assert result['conservative']['censored_pred_ids']==[1]


def test_common_safe_set_does_not_compare_unmatched_populations():
    a={'conservative':{'safe_pairs':[{'gt':1,'pred':5,'iou':.8},{'gt':2,'pred':6,'iou':.7}]}}
    b={'conservative':{'safe_pairs':[{'gt':1,'pred':9,'iou':.9},{'gt':3,'pred':10,'iou':.6}]}}
    result=common_safe_change(a,b)
    assert result['common_matches']==1
    assert abs(result['common_iou_delta_mean']-.1)<1.e-9
    assert result['control_only_safe_matched_gt']==[2]
    assert result['candidate_only_safe_matched_gt']==[3]
