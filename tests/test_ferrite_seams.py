# -*- coding: utf-8 -*-
"""ID0距离、真实阶段来源、固定完整F名单和未知保护的合成CPU门禁。"""
from copy import deepcopy

import numpy as np
import pytest

from tools.analyze_ferrite_native import analyze_ferrite_native
from tools.analyze_ferrite_seams import analyze_ferrite_seams, select_ferrite_seam_cases, selftest


def analyze(gt,pred,gc,pc,markers=None,**kwargs):
    parent=analyze_ferrite_native(gt,pred,gc,pc)
    if markers is None:
        markers=pred.astype(np.int32)
    return analyze_ferrite_seams(gt,pred,gc,pc,markers,parent,**kwargs)


def board():
    gt=np.full((40,40),2,np.uint16);gt[10:30,10:30]=1
    return gt


def test_exact_chessboard_distances_include_diagonal_neighbors():
    gt=np.ones((7,7),np.uint16);pred=np.zeros_like(gt);pred[3,3]=1
    result=analyze(gt,pred,{1:1},{1:1})
    assert result['distance_bins_known']=={
        'distance1':{'P':0,'F':8},'distance2':{'P':0,'F':16},'distance_gt2':{'P':0,'F':24}}
    assert result['known_phase_pixels']==[[0,0,0],[0,1,48]]
    assert not result['actual_ws_attribution']['available']


def test_diagonal_final0_components_are_eight_connected_and_frame_touch_is_exact():
    gt=np.ones((7,7),np.uint16);pred=gt.copy();pred[1,1]=0;pred[2,2]=0
    result=analyze(gt,pred,{1:1},{1:1})
    assert len(result['zero_components'])==1
    assert result['zero_components'][0]['pixels']==2
    assert not result['zero_components'][0]['touches_frame']
    pred[0,0]=0
    framed=analyze(gt,pred,{1:1},{1:1})
    assert framed['zero_components'][0]['pixels']==3
    assert framed['zero_components'][0]['touches_frame']


def test_unknown_and_invalid_content_are_not_counted_as_false_phase():
    gt=np.ones((7,7),np.uint16);gt[2,2]=0
    pred=np.ones_like(gt);pred[2,2]=0;pred[3,3]=0
    result=analyze(gt,pred,{1:1},{1:1})
    row=result['zero_components'][0]
    assert row['pixels']==2 and row['known_F_pixels']==1 and row['unknown_or_invalid_pixels']==1
    assert result['known_phase_pixels']==[[0,0,0],[0,47,1]]
    content=np.ones(gt.shape,bool);content[3,3]=False
    parent=analyze_ferrite_native(gt,pred,{1:1},{1:1},valid_content=content)
    ignored=analyze_ferrite_seams(gt,pred,{1:1},{1:1},pred.astype(np.int32),parent,valid_content=content)
    assert ignored['final0_known']['F']==0


def test_no_positive_output_has_no_fake_distance1_or2():
    gt=np.ones((7,7),np.uint16);pred=np.zeros_like(gt)
    result=analyze(gt,pred,{1:1},{})
    assert result['no_positive_output']
    assert result['distance_bins_known']['distance_gt2']['F']==49
    assert result['distance_bins_known']['distance1']['F']==0
    assert result['safe_area_summary']['all_eligible']['n']==0


def test_safe_F_area_balance_uses_fixed_parent_GT_and_three_pixel_flows():
    gt=board();pred=gt.copy();pred[10:30,28:30]=0
    result=analyze(gt,pred,{1:1,2:0},{1:1,2:0})
    row=result['safe_area_rows'][0]
    assert row['GT_pixels']==400 and row['pred_pixels']==360
    assert row['signed_area_deficit_pixels']==40 and row['final0_inside_GT']==40
    assert row['area_balance_exact']
    assert row['signed_area_deficit_pixels']==row['final0_inside_GT']+row['other_positive_pred_inside_GT']-row['matched_pred_outside_GT']
    assert result['safe_area_summary']['shrinking']['n']==1
    assert result['safe_area_summary']['expanding']['n']==0


def test_parent_eligible_contract_cannot_be_silently_reused_after_output_change():
    gt=board();pred=gt.copy();parent=analyze_ferrite_native(gt,pred,{1:1,2:0},{1:1,2:0})
    changed=pred.copy();changed[10:30,28:30]=0
    with pytest.raises(ValueError,match='Parent safe'):
        analyze_ferrite_seams(gt,changed,{1:1,2:0},{1:1,2:0},pred.astype(np.int32),parent)
    corrupt=deepcopy(parent);corrupt['gt_rows'][0]['matched_iou']=.9
    with pytest.raises(ValueError,match='Parent safe'):
        analyze_ferrite_seams(gt,pred,{1:1,2:0},{1:1,2:0},pred.astype(np.int32),corrupt)


def test_signed_actual_ws_distinguishes_boundary_original0_and_removed_positive():
    gt=board();pred=gt.copy();pred[15,15:18]=0
    ws=pred.astype(np.int32);ws[15,15]=-1;ws[15,16]=0;ws[15,17]=1
    markers=np.zeros_like(ws);markers[15,16:18]=7
    result=analyze(gt,pred,{1:1,2:0},{1:1,2:0},markers=markers,watershed=ws,watershed_format='signed')
    sources=result['actual_ws_attribution']['final0_known_sources']
    assert sources=={'ws_boundary':{'P':0,'F':1},'ws_original_zero':{'P':0,'F':1},'positive_ws':{'P':0,'F':1}}
    assert result['actual_ws_attribution']['source_partition_exact']
    # 种子重叠并未代替真实WS来源：原ws0和后删正ID都可能在正种子上。
    assert result['marker_observation']['final0_on_positive_actual_marker_known']['F']==2


def test_normalized_ws_requires_explicit_boundary_mask_and_matches_signed_counts():
    gt=board();pred=gt.copy();pred[15,15:18]=0
    ws=pred.astype(np.int32);ws[15,15]=-1;ws[15,16]=0;ws[15,17]=1
    signed=analyze(gt,pred,{1:1,2:0},{1:1,2:0},watershed=ws,watershed_format='signed')
    normalized=analyze(gt,pred,{1:1,2:0},{1:1,2:0},watershed=np.maximum(ws,0),
                       watershed_format='zero_normalized',watershed_boundary_mask=ws==-1)
    assert normalized['actual_ws_attribution']['final0_known_sources']==signed['actual_ws_attribution']['final0_known_sources']
    with pytest.raises(ValueError,match='explicit'):
        analyze(gt,pred,{1:1,2:0},{1:1,2:0},watershed=np.maximum(ws,0),watershed_format='zero_normalized')


@pytest.mark.parametrize('mode',('signed','zero_normalized'))
def test_inconsistent_ws_mask_is_rejected(mode):
    gt=board();pred=gt.copy();ws=pred.astype(np.int32);mask=np.zeros(gt.shape,bool);mask[15,15]=True
    with pytest.raises(ValueError):
        analyze(gt,pred,{1:1,2:0},{1:1,2:0},watershed=ws,watershed_format=mode,watershed_boundary_mask=mask)


def test_marker_only_observation_never_claims_unique_ws_provenance():
    gt=board();pred=gt.copy();pred[15,15]=0;markers=np.ones(gt.shape,np.int32)
    result=analyze(gt,pred,{1:1,2:0},{1:1,2:0},markers=markers)
    assert result['actual_ws_attribution']['unidentified_from_final_and_markers']
    assert 'final0_known_sources' not in result['actual_ws_attribution']


def test_known_phase_matrix_keeps_p_only_F_predictions():
    gt=board();pred=gt.copy();pred[3:6,3:6]=3
    result=analyze(gt,pred,{1:1,2:0},{1:1,2:0,3:1})
    assert result['known_phase_pixels'][0][1]==9
    assert result['known_phase_pixels'][1]==[0,400,0]


def test_fixed_case_order_does_not_rank_by_error_size_or_mutate_inputs():
    gt=board();pred=gt.copy();pred[15,15:18]=0;before=pred.copy()
    result=analyze(gt,pred,{1:1,2:0},{1:1,2:0})
    cases=select_ferrite_seam_cases(result)
    assert cases and all(r['gt_ids']==[1] for r in cases)
    assert np.array_equal(pred,before)
    assert select_ferrite_seam_cases(result,0)==[]


def test_cpu_selftest():
    assert selftest()['passed']
