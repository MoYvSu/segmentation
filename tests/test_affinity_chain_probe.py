# -*- coding: utf-8 -*-
"""定位探针的unknown保护与原始部署一致性。"""
import numpy as np
import torch

from tools.probe_affinity_chain import trusted_targets, inject, decode_capture, relations_at, fixed_relations
from tools.affinity_connectivity_views import decode_native


def test_counterfactual_never_crosses_unknown_or_changes_unselected_channels():
    ids=np.ones((12,16),np.int32);ids[:,8:]=2
    trust=np.ones_like(ids,bool);trust[:,7]=False
    target,valid=trusted_targets(ids,trust,np.ones_like(ids,bool))
    # x=5至9跨过unknown列7，尽管两个端点已知，也不能注入distance4。
    assert not valid[0,6,5,5]
    logits=torch.randn_like(target)
    result,mask=inject(logits,target,valid,4)
    assert torch.equal(result[~mask],logits[~mask])
    assert torch.equal(result[:,4:],logits[:,4:])
    assert torch.all(result[mask & (target>.5)]==16)
    assert torch.all(result[mask & (target<.5)]==-16)


def test_trace_capture_matches_actual_deployed_output():
    config={'inference':{'threshold':.5,'boundary_threshold':.65,'min_instance_area':8,
        'watershed_dilate_width':1,'bridge_width':1,'marker_border_seal_width':2,
        'marker_boundary_low_threshold':.45,'marker_boundary_reconstruction_steps':8,
        'semantic_vote_mode':'probability_mean','semantic_vote_threshold':.5}}
    semantic=torch.zeros(1,1,65,79);semantic[:,:,:32]=2
    boundary=torch.zeros_like(semantic);boundary[:,:,:,39:41]=.9;boundary[:,:,30:33,:]=.8
    expected,classes=decode_native(semantic,boundary,config)
    actual,other,stages,ws=decode_capture(semantic,boundary,config)
    assert np.array_equal(expected,actual) and classes==other
    assert stages['marker_labels'].shape==actual.shape==ws.shape


def test_zero_marker_is_unknown_not_separated():
    labels=np.array([[2,0,3],[2,2,3]])
    pairs=[{'points':[[0,0],[0,1]]},{'points':[[0,0],[1,1]]},{'points':[[0,0],[1,2]]}]
    assert relations_at(labels,pairs)==['blocked','same','different']


def test_merge_enumeration_keeps_pairs_without_trusted_supervision_edges():
    gt=np.zeros((32,64),np.int32);gt[:,:20]=1;gt[:,44:]=2
    trusted=gt>0
    target,valid=trusted_targets(gt,trusted,np.ones_like(trusted))
    relations,_=fixed_relations(gt,trusted,np.ones_like(gt),gt,target,valid)
    assert len(relations)==1 and relations[0]['kind']=='merge'
    assert relations[0]['trusted_contact_edges']==0
