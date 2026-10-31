# -*- coding: utf-8 -*-
"""只替换marker的真实CPU后处理：exact门禁、变体隔离和输入保护。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from tools.probe_affinity_partition import decode_capture
from tools.semantic_marker_diagnostic import marker_stages
from utils.post_process import _boundary_skeleton_belt


def sample():
    config={'inference':dict(boundary_threshold=.65,marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8,marker_border_seal_width=2,bridge_width=1,
        watershed_dilate_width=1,min_instance_area=50,max_instance_id=65535,
        threshold=.5,semantic_vote_mode='probability_mean')}
    boundary=np.full((128,144),.03,np.float32)
    boundary[:,65:79]=.9
    boundary[58:62,65:79]=.53
    semantic=torch.linspace(-2,2,144)[None,None,None].expand(1,1,128,144).clone()
    return semantic,torch.from_numpy(boundary)[None,None],config


def test_none_and_identical_override_exactly_match_original_actual_deployment():
    semantic,boundary,config=sample()
    expected=decode_capture(semantic,boundary,config)
    baseline=decode_marker_replay(semantic,boundary,config)
    copied=decode_marker_replay(semantic,boundary,config,expected['actual_markers'].copy())
    for value in (baseline,copied):
        for field in ('instances','actual_markers','watershed'):
            assert np.array_equal(value[field],expected[field])
        assert value['classes']==expected['classes']
        assert value['capture_audit']['actual_watershed_calls']==1
        assert value['capture_audit']['same_marker_as_baseline']
        assert value['capture_audit']['original_elevation_exact']
    assert baseline['elevation_sha256']==copied['elevation_sha256']
    assert baseline['hashes']==copied['hashes']


@pytest.mark.parametrize('mode',['preseal','nothin','reflect'])
def test_variant_keeps_real_barrier_semantic_input_and_config_frozen(mode):
    semantic,boundary,config=sample()
    before_sem=semantic.clone(); before_boundary=boundary.clone(); before_config=deepcopy(config)
    base=marker_stages(boundary[0,0].numpy(),config)
    stage_snap={k:v.copy() for k,v in base.items() if isinstance(v,np.ndarray)}
    resolved=deepcopy(base['resolved'])
    candidate=build_marker_variant(base,mode)
    assert candidate['diagnostic_only'] and not candidate['real_barrier_changed']
    assert candidate['barrier_belt'] is base['barrier_belt']
    for k,v in stage_snap.items(): assert np.array_equal(base[k],v)
    assert base['resolved']==resolved
    original=decode_marker_replay(semantic,boundary,config)
    injected=candidate['marker_labels'].copy(); before=injected.copy()
    result=decode_marker_replay(semantic,boundary,config,injected)
    assert result['elevation_sha256']==original['elevation_sha256']
    assert np.array_equal(result['original_markers'],original['actual_markers'])
    assert np.array_equal(result['actual_markers'],candidate['marker_labels'])
    assert result['instances'].dtype==np.uint16 and int(result['instances'].max())<=65535
    assert np.array_equal(injected,before)
    assert torch.equal(semantic,before_sem) and torch.equal(boundary,before_boundary)
    assert config==before_config


def test_nothin_is_exact_bridge_then_dilate_then_seal_and_area_filter():
    semantic,boundary,config=sample(); base=marker_stages(boundary[0,0].numpy(),config)
    candidate=build_marker_variant(base,'nothin')
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bridged=cv2.dilate((base['marker_boundary_mask']>0).astype(np.uint8)*255,kernel)
    belt=cv2.dilate(bridged,kernel)
    assert np.array_equal(candidate['bridged_marker_mask'],bridged)
    assert np.array_equal(candidate['dilated_marker_before_seal'],belt)
    belt[:2]=255; belt[-2:]=255; belt[:,:2]=255; belt[:,-2:]=255
    assert np.array_equal(candidate['marker_belt'],belt)
    count,labels,stats,_=cv2.connectedComponentsWithStats(cv2.bitwise_not(belt),connectivity=8)
    expected=np.zeros(belt.shape,np.int32); marker=0
    for label in range(1,count):
        if stats[label,cv2.CC_STAT_AREA]>=50:
            marker+=1; expected[labels==label]=marker
    assert np.array_equal(candidate['marker_labels'],expected)
    assert candidate['marker_count']==marker


@pytest.mark.parametrize('padding',[32,64])
def test_reflect_uses_actual_thinning_after_native_bridge_and_crops_before_dilate(padding):
    semantic,boundary,config=sample(); base=marker_stages(boundary[0,0].numpy(),config)
    candidate=build_marker_variant(base,'reflect',reflection_padding=padding)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bridged=cv2.dilate((base['marker_boundary_mask']>0).astype(np.uint8)*255,kernel)
    padded=cv2.copyMakeBorder(bridged,padding,padding,padding,padding,cv2.BORDER_REFLECT)
    expanded,_=_boundary_skeleton_belt(padded,0,0)
    h,w=boundary.shape[-2:]; skeleton=expanded[padding:padding+h,padding:padding+w]
    assert np.array_equal(candidate['cropped_skeleton'],skeleton)
    assert candidate['cropped_skeleton'].shape==base['high_mask'].shape
    assert candidate['cropped_skeleton'].dtype==np.uint8
    belt=cv2.dilate(skeleton,kernel)
    assert np.array_equal(candidate['dilated_marker_before_seal'],belt)
    belt[:2]=255; belt[-2:]=255; belt[:,:2]=255; belt[:,-2:]=255
    assert np.array_equal(candidate['marker_belt'],belt)
    assert candidate['reflection_padding']==padding
    assert candidate['barrier_belt'] is base['barrier_belt']
    assert not candidate['real_barrier_changed']


def test_reflect_32_and_64_on_synthetic_wide_touchframe_line_only():
    semantic,boundary,config=sample()
    boundary.fill_(.03); boundary[:,:,:,65:79]=.9
    base=marker_stages(boundary[0,0].numpy(),config)
    first=build_marker_variant(base,'reflect',reflection_padding=32)
    second=build_marker_variant(base,'reflect',reflection_padding=64)
    for candidate in (first,second):
        labels=candidate['marker_labels']
        assert labels[64,30]>0 and labels[64,110]>0 and labels[64,30]!=labels[64,110]
        assert candidate['marker_count']==2
        assert np.all(candidate['marker_belt'][:2]==255)
        assert np.all(candidate['marker_belt'][:,:2]==255)
    # This one synthetic infinite straight-line extension is stable, not all data.
    assert np.array_equal(first['marker_labels'],second['marker_labels'])


@pytest.mark.parametrize('padding',[0,-1,False,True,32.,'32',None])
def test_reflect_padding_validation(padding):
    semantic,boundary,config=sample(); base=marker_stages(boundary[0,0].numpy(),config)
    with pytest.raises(ValueError):
        build_marker_variant(base,'reflect',reflection_padding=padding)


@pytest.mark.parametrize('bad',['float','int64','negative','shape'])
def test_marker_validation_rejects_ambiguous_or_invalid_replacement(bad):
    semantic,boundary,config=sample()
    markers=marker_stages(boundary[0,0].numpy(),config)['marker_labels'].copy()
    if bad=='float': markers=markers.astype(np.float32)
    if bad=='int64': markers=markers.astype(np.int64)
    if bad=='negative': markers[4,4]=-1
    if bad=='shape': markers=markers[:-1]
    with pytest.raises(ValueError): decode_marker_replay(semantic,boundary,config,markers)


def test_zero_original_markers_and_changed_restore_contract_rejected():
    semantic,boundary,config=sample()
    with pytest.raises(ValueError): decode_marker_replay(semantic,torch.ones_like(boundary),config)
    config['inference']['marker_partition_restore']={'enabled':True}
    with pytest.raises(ValueError): decode_marker_replay(semantic,boundary,config)
    with pytest.raises(ValueError): build_marker_variant({},'unknown')
