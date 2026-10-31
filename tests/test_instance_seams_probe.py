# -*- coding: utf-8 -*-
"""独立细缝诊断：守住GT-free规则、固定父面积名单和类别闭合。"""
import copy

import numpy as np
import pytest

from tools.assign_instance_seams import assign_instance_seams
from tools.analyze_ferrite_native import analyze_ferrite_native
from tools.probe_instance_seams import analyze_instance_seams, summarize, validate_assignment, selftest


def board():
    gt=np.full((40,40),2,np.uint16)
    gt[10:30,10:30]=1
    baseline=gt.copy(); baseline[15:25,20]=0
    classes={1:1,2:0}
    candidate,_=assign_instance_seams(baseline,classes)
    return gt,baseline,candidate,classes


def test_fixed_complete_f_area_and_common_iou():
    gt,a,b,classes=board()
    parent=analyze_ferrite_native(gt,a,classes,classes)
    report=analyze_instance_seams(gt,a,b,classes,classes,parent)
    area=report['fixed_parent_safe_F_rows'][0]
    assert area['gt_pixels']==400
    assert area['before_pred_pixels']==390 and area['after_pred_pixels']==400
    assert area['before_relative_error']==-.025 and area['after_relative_error']==0
    assert report['fixed_parent_safe_F_summary']['improved']==1
    assert report['matched_change']['common_F']['mean_delta_iou']>0
    assert report['known_phase_delta'][1]==[0,10,-10]
    assert report['counts']['instances']==2
    summary=summarize([dict(source='synthetic',diagnostic=report)])
    assert summary['fixed_parent_safe_F']['n']==1
    assert summary['original_fixed_pair_list']==[dict(source='synthetic',gt=1,pred=1)]


def test_same_pred_class_wrong_gt_assignment_is_counted():
    gt,a,b,classes=board()
    # 模型类别P错误；规则自身只看预测类别仍合法，GT仅事后计数。
    pc={1:0,2:0}
    b,_=assign_instance_seams(a,pc)
    report=analyze_instance_seams(gt,a,b,classes,pc)
    assert report['added_pixels']['known_wrong_phase']==10
    assert report['known_phase_delta'][1]==[10,0,-10]
    assert report['fixed_parent_safe_F_summary']['n']==0


def test_parent_cohort_survives_candidate_censor():
    # GT1完整；预测轻微外扩靠近邻区未知孔，填一格后预测变censor。
    gt=np.full((40,40),2,np.uint16); gt[10:30,10:30]=1
    gt[15:25,33]=0
    a=gt.copy(); a[15:25,30:32]=1; a[15:25,32:35]=0
    classes={1:1,2:0}
    b,_=assign_instance_seams(a,classes)
    parent=analyze_ferrite_native(gt,a,classes,classes)
    report=analyze_instance_seams(gt,a,b,classes,classes,parent)
    assert parent['gt_rows'][0]['safe_one_to_one_area_eligible']
    assert report['fixed_parent_safe_F_summary']['n']==1
    assert report['fixed_parent_safe_F_summary']['newly_censored']==1
    assert report['fixed_parent_safe_F_remaining_uncensored_summary']['n']==0
    assert report['fixed_parent_safe_F_rows'][0]['candidate_pred_censored']


def test_unknown_fill_is_separate_from_known_matrix():
    gt,a,b,classes=board()
    gt[16,20]=0
    report=analyze_instance_seams(gt,a,b,classes,classes)
    assert report['added_pixels']['total']==10
    assert report['added_pixels']['unknown_or_invalid']==1
    assert report['added_pixels']['known_correct_phase']==9
    assert sum(sum(r) for r in report['known_phase_after'])==int((gt>0).sum())


@pytest.mark.parametrize('mutation',['positive','frame','missed','farther'])
def test_independent_invalid_assignment_rejected(mutation):
    gt,a,b,classes=board()
    if mutation=='positive':
        b[11,11]=2
    elif mutation=='frame':
        a[0,20]=0; b[0,20]=2
    elif mutation=='missed':
        b[16,20]=0
    else:
        b[16,20]=2
    with pytest.raises(ValueError):
        validate_assignment(a,b,classes)


def test_no_propagation_into_wide_hole():
    a=np.ones((13,13),np.uint16); a[4:9,4:9]=0
    b,_=assign_instance_seams(a,{1:1})
    assert b[6,6]==0
    validate_assignment(a,b,{1:1})
    b[6,6]=1
    with pytest.raises(ValueError,match='single-pass'):
        validate_assignment(a,b,{1:1})


def test_changed_parent_diagnostic_rejected():
    gt,a,b,classes=board()
    parent=copy.deepcopy(analyze_ferrite_native(gt,a,classes,classes))
    parent['gt_rows'][0]['area']+=1
    with pytest.raises(ValueError,match='parent'):
        analyze_instance_seams(gt,a,b,classes,classes,parent)


def test_empty_gt_and_outputs_have_no_fake_area():
    empty=np.zeros((5,7),np.uint16)
    report=analyze_instance_seams(empty,empty,empty,{}, {})
    assert report['fixed_parent_safe_F_summary']['n']==0
    assert report['fixed_parent_safe_F_summary']['before_mean_signed'] is None
    assert report['matched_change']['common_F']['n']==0
    assert report['known_phase_after']==[[0,0,0],[0,0,0]]


def test_selftest():
    assert selftest()['passed']
