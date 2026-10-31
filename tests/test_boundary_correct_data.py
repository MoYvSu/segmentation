# -*- coding: utf-8 -*-
"""局部 GT 正纠错候选的合成 CPU 契约，不读取赛题图片或权重。"""
import json

import cv2
import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from data.boundary_correct import build_boundary_correction, _known_boundary


def straight_case(gap=(68,98), shape=(176,192)):
    ids=np.ones(shape,np.int32);ids[:,96:]=2
    _,xx=np.indices(shape)
    heat=(.04+.86*np.exp(-.5*((xx-95)/3.)**2)).astype(np.float32)
    if gap is not None:
        heat[gap[0]:gap[1]]=.04
    return heat,ids,np.ones(shape,bool)


def test_two_sided_short_gap_gets_local_positive_only_target():
    heat,ids,original=straight_case()
    originals=[a.copy() for a in (heat,ids,original)]
    result=build_boundary_correction(heat,ids,original,source={'namespace':'synthetic','id':'case1'})
    assert result['summary']['accepted_count']==1
    assert result['correction_mask'].any()
    assert np.all(result['target']>=heat)
    assert np.array_equal(result['input'],heat)
    assert np.array_equal(result['target'][~result['correction_mask']],heat[~result['correction_mask']])
    assert np.array_equal(result['target'][heat>=.65],heat[heat>=.65])
    assert np.array_equal(result['target'][~result['valid']],heat[~result['valid']])
    assert result['target'][82,95]>.89 and result['target'][82,60]==heat[82,60]
    for a,b in zip((heat,ids,original),originals):np.testing.assert_array_equal(a,b)
    rec=result['records'][0]
    assert 8<=rec['native_length']<=64
    assert rec['before_anchor']['distance_native']<=16 and rec['after_anchor']['distance_native']<=16
    assert rec['before_anchor']['response_peak']>=.65 and rec['after_anchor']['response_peak']>=.65
    assert rec['radius_match_max']<.45
    assert rec['before_anchor']['response_bandwidth_2edt_at_peak']['0.45']>0
    assert not rec['normal_profile']['rejected_as_potential_offset']
    json.dumps(result['summary'],allow_nan=False)
    json.dumps(result['records'],allow_nan=False)


def test_existing_complete_heat_is_exact_identity_including_unlabelled_lines():
    heat,ids,original=straight_case(gap=None)
    heat[70:110,35:40]=.77  # 原有未标响应不允许删掉。
    result=build_boundary_correction(heat,ids,original)
    assert result['summary']['accepted_count']==0
    np.testing.assert_array_equal(result['target'],heat)


@pytest.mark.parametrize('gap,reason', [((68,77),'length_outside_range'),
                                      ((30,140),'length_outside_range'),
                                      ((0,40),'missing_distinct_two_sided_strong_anchors')])
def test_short_long_and_one_ended_weak_runs_are_not_targets(gap,reason):
    heat,ids,original=straight_case(gap=gap)
    result=build_boundary_correction(heat,ids,original)
    assert result['summary']['accepted_count']==0
    assert result['summary']['rejection_counts'].get(reason,0)>0
    np.testing.assert_array_equal(result['target'],heat)


def test_filled_and_unknown_do_not_supply_positive_support_or_anchors():
    heat,ids,original=straight_case()
    # 实例虽已填补，但 original_covered 不包含该区；不把 filled 等同原标。
    original[55:110,80:110]=False
    result=build_boundary_correction(heat,ids,original)
    assert not result['candidate_mask'].any()
    np.testing.assert_array_equal(result['target'],heat)
    ids[55:110,80:110]=0
    heat[ids==0]=.8765432
    result=build_boundary_correction(heat,ids,original,unknown=ids==0)
    assert not result['candidate_mask'].any()
    np.testing.assert_array_equal(result['target'],heat)


def test_no_known_known_interface_means_no_boundary_from_zero_or_texture():
    heat,ids,original=straight_case()
    ids[:]=1;ids[:,96:]=0;original=ids>0
    result=build_boundary_correction(heat,ids,original)
    assert not result['skeleton'].any() and not result['correction_mask'].any()


def test_thin_fill_policy_restores_supported_interface_without_relaxing_response_gates():
    heat,ids,original=straight_case()
    original[:,94:98]=False  # 交界两侧各两像素，最近原标分别属于自身实例。
    strict=build_boundary_correction(heat,ids,original)
    thin=build_boundary_correction(heat,ids,original,config={'support_policy':'processed_thin_fill'})
    assert strict['summary']['accepted_count']==0
    assert thin['summary']['accepted_count']==1
    assert thin['summary']['support']['supported_filled_pixels']==4*heat.shape[0]
    assert thin['summary']['candidate_filled_centers']>0
    assert thin['summary']['correction_filled_pixels']>0
    assert np.all(thin['candidate_mask']<=thin['own_support'])
    assert np.all(thin['correction_mask']<=thin['target_support'])
    assert thin['summary']['support']['positive_support_role']=='processed GT narrow-fill support, not original human truth'
    assert {k:v for k,v in thin['summary']['config'].items() if k!='support_policy'}=={
        k:v for k,v in strict['summary']['config'].items() if k!='support_policy'}
    np.testing.assert_array_equal(thin['target'][~thin['target_support']],heat[~thin['target_support']])


def test_thin_fill_rejects_far_filled_interface_and_wrong_nearest_instance():
    heat,ids,original=straight_case()
    original[:,89:103]=False  # 交界处距原标7px，不允许纳入。
    far=build_boundary_correction(heat,ids,original,config={'support_policy':'processed_thin_fill'})
    assert not far['own_support'][:,95:97].any()
    assert not far['skeleton'].any()
    np.testing.assert_array_equal(far['target'],heat)
    original[:]=True;original[:,90:97]=False
    # x95属于左实例，但最近原标x97属于右实例；不能只按近邻距离接纳。
    wrong=build_boundary_correction(heat,ids,original,config={'support_policy':'processed_thin_fill'})
    assert not wrong['own_support'][:,95].any()
    assert wrong['own_support'][:,96].all()
    assert wrong['summary']['support']['filled_near_but_nearest_other_instance']>0
    assert not wrong['skeleton'].any()  # 两端支持门禁，不能保留右侧单边界像素。
    np.testing.assert_array_equal(wrong['target'],heat)


def test_thin_fill_guard_still_observes_unsupported_filled_ridge():
    ids=np.ones((192,192),np.int32);ids[:,96:]=2
    heat=np.zeros(ids.shape,np.float32);heat[:,96]=.8;heat[75:113,96]=0
    original=np.ones(ids.shape,bool);original[:,94:98]=False;original[:,99:115]=False
    without=build_boundary_correction(heat,ids,original,config={'support_policy':'processed_thin_fill'})
    assert without['summary']['accepted_count']==1
    heat[80:89,104]=.8
    with_ridge=build_boundary_correction(heat,ids,original,config={'support_policy':'processed_thin_fill'})
    assert not with_ridge['own_support'][80:89,104].any()
    assert with_ridge['full_known_safe'][80:89,104].all()
    assert with_ridge['summary']['offset_guard_rejected']==1
    assert with_ridge['summary']['accepted_count']==0
    np.testing.assert_array_equal(with_ridge['target'],heat)


def test_thin_fill_no_original_support_never_invents_support():
    heat,ids,original=straight_case();original[:]=False
    result=build_boundary_correction(heat,ids,original,config={'support_policy':'processed_thin_fill'})
    assert result['full_known_safe'].any() and not result['own_support'].any()
    assert not result['skeleton'].any() and not result['correction_mask'].any()
    np.testing.assert_array_equal(result['target'],heat)


def test_middle_normal_profile_rejects_existing_shifted_ridge():
    heat,ids,original=straight_case()
    _,xx=np.indices(heat.shape)
    shifted=(.04+.86*np.exp(-.5*((xx-103)/1.1)**2)).astype(np.float32)
    heat[68:98]=shifted[68:98]
    permissive=build_boundary_correction(heat,ids,original,config={'offset_guard':False})
    guarded=build_boundary_correction(heat,ids,original)
    assert permissive['summary']['accepted_count']==1
    assert guarded['summary']['accepted_before_offset_guard']==1
    assert guarded['summary']['offset_guard_rejected']==1
    assert guarded['summary']['accepted_count']==0
    np.testing.assert_array_equal(guarded['target'],heat)
    profile=guarded['rejections'][0]['normal_profile']
    assert profile['inner_peak']<.45 and profile['outer_peak']>=.65
    assert all(4<abs(v)<=12 for v in profile['shifted_strong_offsets_native'])


@pytest.mark.parametrize('ridge_strength',[.5,.8])
def test_offset_guard_checks_entire_gap_not_only_midpoint(ridge_strength):
    ids=np.ones((192,192),np.int32);ids[:,96:]=2
    heat=np.zeros(ids.shape,np.float32);heat[:,96]=.8;heat[75:113,96]=0
    original=np.ones(ids.shape,bool)
    without_parallel=build_boundary_correction(heat,ids,original)
    assert without_parallel['summary']['accepted_count']==1
    heat[80:89,102]=ridge_strength  # 仅端部有偏移线，中点 y93 不会发现；.5 也能进入弱边界读出。
    with_parallel=build_boundary_correction(heat,ids,original)
    assert with_parallel['summary']['accepted_count']==0
    assert with_parallel['summary']['offset_guard_rejected']==1
    assert with_parallel['summary']['offset_guard_additional_vs_midpoint']==1
    np.testing.assert_array_equal(with_parallel['target'],heat)
    record=with_parallel['rejections'][0]
    assert record['offset_guard_scope']=='off_midpoint_only'
    assert not record['normal_profile']['midpoint_would_reject']
    assert record['normal_profile']['checked_point_count']==record['points']


def test_no_correction_near_junction_and_margin():
    heat,ids,original=straight_case(gap=None)
    ids[88:,:96]=3;ids[88:,96:]=4
    boundary=_known_boundary(ids)
    distance=distance_transform_edt(~boundary)
    heat=(.04+.86*np.exp(-.5*(distance/3)**2)).astype(np.float32)
    heat[83:94,87:104]=.04
    result=build_boundary_correction(heat,ids,original)
    assert result['summary']['excluded_junction_pixels']>0
    assert not result['candidate_mask'][86:91,93:99].any()
    assert not result['valid'][:8].any() and not result['valid'][-8:].any()


def test_native_diagonal_length_uses_arc_distance_and_deterministic_outputs():
    yy,xx=np.indices((176,192));ids=(1+(xx>=yy)).astype(np.int32)
    boundary=_known_boundary(ids)
    distance=distance_transform_edt(~boundary)
    heat=(.04+.86*np.exp(-.5*(distance/3)**2)).astype(np.float32);heat[68:98]=.04
    original=np.ones(heat.shape,bool)
    a=build_boundary_correction(heat,ids,original)
    b=build_boundary_correction(heat,ids,original)
    assert a['summary']==b['summary'] and a['records']==b['records']
    np.testing.assert_array_equal(a['target'],b['target'])
    assert a['records']
    assert a['records'][0]['native_length']>a['records'][0]['points']-1


@pytest.mark.parametrize('diagonal',[False,True])
@pytest.mark.parametrize('policy',['original_strict','processed_thin_fill'])
def test_exact_targets_and_records_across_opencv_thread_counts(diagonal,policy):
    heat,ids,original=straight_case()
    if policy=='processed_thin_fill' and not diagonal:
        original[:,94:98]=False
    if diagonal:
        yy,xx=np.indices(heat.shape);ids=(1+(xx>=yy)).astype(np.int32)
        distance=distance_transform_edt(~_known_boundary(ids))
        heat=(.04+.86*np.exp(-.5*(distance/3)**2)).astype(np.float32)
        heat[68:98]=.04
    old_threads=cv2.getNumThreads()
    reference=None
    try:
        for threads in (1,2,4):
            cv2.setNumThreads(threads)
            for _ in range(3):
                result=build_boundary_correction(heat,ids,original,config={'support_policy':policy})
                assert cv2.getNumThreads()==threads  # 构造器不能靠修改进程全局线程规避问题。
                assert result['records']
                if reference is None:
                    reference=result
                else:
                    np.testing.assert_array_equal(result['target'],reference['target'])
                    assert result['records']==reference['records']
                    assert result['summary']==reference['summary']
    finally:
        cv2.setNumThreads(old_threads)


def test_reject_wrong_shapes_masks_precision_and_invalid_recipe():
    heat,ids,original=straight_case()
    with pytest.raises(ValueError,match='float32'):
        build_boundary_correction(heat.astype(np.float64),ids,original)
    with pytest.raises(ValueError,match='match N0'):
        build_boundary_correction(heat,ids[:-1],original)
    with pytest.raises(ValueError,match='Unknown provenance'):
        build_boundary_correction(heat,ids,original,unknown=np.ones_like(original))
    with pytest.raises(ValueError,match='Unknown boundary correction keys'):
        build_boundary_correction(heat,ids,original,config={'target_scale':2})
    with pytest.raises(ValueError,match='ordered native gap'):
        build_boundary_correction(heat,ids,original,config={'min_gap_length':70})
