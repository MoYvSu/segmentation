# -*- coding: utf-8 -*-
"""真实CPU WS桥接：C/F0、面积/上限、无种子、输入不变及仅一次stages。"""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tools import infer_boundary_rgb as legacy, infer_flat_semantic as flat
from tools import infer_semifinal as entry
from utils import compact_instance_readout as compact, post_process


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def config():
    return flat.execution_config(entry.load_config(str(flat.ROOT/'config/inference/area_floor_200.yaml')))


def maps(pattern=0):
    after=np.full((64,80),.1,np.float32)
    after[:,39:42]=.95
    if pattern>0:after[30:33,:]=.9
    if pattern>1:after[8:16,8:16]=.8
    reference=after.copy()
    semantic=torch.linspace(-3,3,after.size).reshape(1,1,*after.shape)
    semantic[0,0,10,:6]=torch.tensor([-1e-7,-1e-8,0,1e-8,1e-7,1e-6])
    probability=np.linspace(.13,.87,after.size,dtype=np.float32).reshape(after.shape)
    return semantic,after,reference,probability


@pytest.mark.parametrize('mode',['C','F0'])
@pytest.mark.parametrize('pattern',[0,1,2])
def test_real_WS_pre200_raw_vote_final200_exact_legacy_and_inputs_immutable(mode,pattern):
    cfg=config();semantic,after,reference,raw=maps(pattern)
    if mode=='F0':semantic.zero_()
    original=(semantic.clone(),after.copy(),reference.copy(),raw.copy())
    (expected,_),old_audit=legacy._supported_readout(semantic,after,reference,cfg)
    (actual,temporary),audit=compact.compact_supported_readout(semantic,after,reference,cfg,geometry_mode=mode)
    assert np.array_equal(actual,expected)
    assert audit['readout_hashes']==old_audit['readout_hashes']
    for key in ('input_marker_sha256','output_marker_sha256','retained_ids','removed_ids','seeds'):
        assert audit[key]==old_audit[key]
    assert set(temporary)==set(range(1,int(actual.max())+1)) and set(temporary.values())<={0}
    classes,scores=entry.views.revote_instances(actual,raw)
    old_classes,old_scores=entry.views.revote_instances(expected,raw)
    assert classes==old_classes and scores==old_scores
    final=entry.finalize_instance_area(dict(instances=actual,classes=classes),cfg)
    old_final=entry.finalize_instance_area(dict(instances=expected,classes=old_classes),cfg)
    assert np.array_equal(final['instances'],old_final['instances']) and final['classes']==old_final['classes']
    entry.verify_prediction(final['instances'],final['classes'],after.shape)
    assert torch.equal(semantic,original[0])
    assert all(np.array_equal(a,b) for a,b in zip((after,reference,raw),original[1:]))
    assert audit['compact_readout']['marker_stage_calls']==1
    assert audit['compact_readout']['temporary_semantic_vote_calls']==0
    assert audit['compact_readout']['final200_performed'] is False


def test_one_stage_one_WS_and_no_temporary_vote():
    semantic,after,reference,_=maps()
    original=compact.marker_stages;real=cv2.watershed
    calls=dict(stages=0,WS=0)
    def stage(*a,**k):calls['stages']+=1;return original(*a,**k)
    def ws(*a,**k):calls['WS']+=1;return real(*a,**k)
    def forbidden(*a,**k):raise AssertionError('Temporary semantic vote must not execute')
    with patch.object(compact,'marker_stages',new=stage),patch.object(cv2,'watershed',new=ws), \
            patch.object(post_process,'_instance_semantic_vote',new=forbidden):
        compact.compact_supported_readout(semantic,after,reference,config())
    assert calls==dict(stages=1,WS=1)


@pytest.mark.parametrize('action',['legacy_drop','neighbor_merge_v1'])
@pytest.mark.parametrize('maximum',[2,65535])
def test_area_filter_cap_stable_ID_sort_matches_original_integer_rules(action,maximum):
    watershed=np.array([[11,11,0,21,21,0,31,31,0,41],
                        [11,11,0,21,21,0,31,31,0,41],
                        [11,11,0,21,21,0,31,31,0,-1],
                        [0,0,0,0,0,0,0,0,0,0]],np.int32)
    infer=dict(seed_min_area=1,final_min_instance_area=3,final_small_instance_action=action,
               max_instance_id=maximum)
    with patch.object(cv2,'watershed',new=lambda image,markers:watershed.copy()):
        expected,_=post_process.boundary_watershed_separation(np.zeros(watershed.shape,np.uint8),
            np.zeros(watershed.shape,np.uint8),seed_min_area=1,final_min_instance_area=3,
            final_small_instance_action=action,max_instance_id=maximum,semantic_vote_mode='probability_mean',
            semantic_probability=np.full(watershed.shape,.5,np.float32))
    actual,classes,proof=compact.finish_watershed(watershed,infer)
    assert np.array_equal(actual,expected)
    assert len(classes)<=maximum and actual.dtype==np.uint16 and int(actual.max())<=65535
    assert proof['pre_vote_final_min_area']==(3 if action=='legacy_drop' else 0)


@pytest.mark.parametrize('failure',['no_original_seed','S0_removes_all'])
def test_same_no_seed_failure_contract_without_WS(failure):
    semantic,after,reference,_=maps();cfg=config()
    if failure=='no_original_seed':cfg['inference']['seed_min_area']=after.size+1
    else:reference[:]=1
    messages=[]
    with patch.object(cv2,'watershed',new=lambda *a:(_ for _ in ()).throw(AssertionError('No WS on rejected candidate'))):
        for fn in (legacy._supported_readout,compact.compact_supported_readout):
            with pytest.raises(ValueError) as error:fn(semantic,after,reference,cfg)
            messages.append(str(error.value))
    assert messages[0]==messages[1]


def test_empty_selected_marker_with_nonempty_original_stage_still_runs_WS_like_legacy():
    semantic,after,reference,_=maps();cfg=config()
    def empty(stages,*args):return dict(marker_labels=np.zeros(after.shape,np.int32))
    with patch.object(legacy,'build_marker_variant',new=empty),patch.object(compact,'build_marker_variant',new=empty):
        (old,_),old_audit=legacy._supported_readout(semantic,after,reference,cfg)
        (new,classes),audit=compact.compact_supported_readout(semantic,after,reference,cfg)
    assert np.array_equal(old,new) and not new.any() and classes=={}
    assert audit['readout_hashes']==old_audit['readout_hashes']


def test_reject_geometry_coupling_and_nonzero_F0():
    semantic,after,reference,_=maps();cfg=config()
    with pytest.raises(ValueError,match='zero geometry'):
        compact.compact_supported_readout(semantic,after,reference,cfg,geometry_mode='F0')
    cfg['inference']['marker_partition_restore']={'enabled':True}
    with pytest.raises(ValueError,match='partition'):
        compact.compact_supported_readout(semantic,after,reference,cfg)
