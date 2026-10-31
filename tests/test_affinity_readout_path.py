# -*- coding: utf-8 -*-
"""虚构读取瓶颈：真实top2/ties、跨GTcompanion、合法域与输入不变。"""
import json

import numpy as np
import pytest

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_readout_path import readout_bottleneck_targets


def positive_sample(q=(.8,.6,.2,.1)):
    gt=np.ones((3,4),np.int32);valid=np.ones_like(gt,bool)
    boundary=np.full(gt.shape,.1,np.float32);boundary[1,1]=.8
    p=np.full((4,*gt.shape),.9,np.float32);p[:,1,1]=1.-np.array(q,np.float32)
    return boundary,p,gt,valid


def test_positive_uses_actual_top2_members_and_they_change_synthetic_short_readout():
    b,p,gt,valid=positive_sample()
    result=readout_bottleneck_targets(b,p,gt,valid,(0,1),(1,1),'positive')
    assert result['found'] and result['targets_found'] and result['target_gt_legality']
    assert result['bottleneck_nodes']==[[1,1]] and result['target_indices']==[[0,1,1],[1,1,1]]
    assert result['raw_capacity']==1.-float(b[1,1])
    assert result['bottleneck_boundary']==float(b[1,1])
    before=np.sort(1.-p[:,1,1])[-2:].mean();after=p.copy()
    for c,y,x in result['target_indices']:after[c,y,x]=1.
    assert np.sort(1.-after[:,1,1])[-2:].mean()<before
    assert result['coordinate_space']=='input_grid_yx' and not result['native_projection_performed']
    assert not result['watershed_success_claimed']
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('q',[(.8,.6,.6,.6),(.4,.4,.4,.4)])
def test_second_largest_cutoff_keeps_all_actual_ties(q):
    b,p,gt,valid=positive_sample(q)
    result=readout_bottleneck_targets(b,p,gt,valid,(0,1),(1,1),'positive')
    assert result['target_indices']==[[c,1,1] for c in range(4)]
    assert result['node_details'][0]['top2_members_including_ties']==list(range(4))


def test_legality_filter_does_not_promote_non_top2_channels():
    b,p,gt,valid=positive_sample()
    gt[1,2]=3;gt[2,1]=0
    result=readout_bottleneck_targets(b,p,gt,valid,(0,1),(1,1),'positive')
    assert result['found'] and not result['targets_found'] and result['target_indices']==[]
    assert result['empty_target_reason']=='no_legal_actual_top2_outgoing_targets'
    rejected={entry['channel']:entry['reason'] for entry in result['node_details'][0]['rejected_channels']}
    assert rejected=={0:'third_gt',1:'unknown_gt',2:'not_actual_top2_member',3:'not_actual_top2_member'}


def test_positive_excludes_padding_allowed_and_native_edge_invalid_targets():
    b,p,gt,valid=positive_sample((.4,.4,.4,.4));allowed=np.ones_like(gt,bool)
    edges=np.ones_like(p,bool);valid[1,2]=False;allowed[2,1]=False
    edges[2,1,1]=False
    result=readout_bottleneck_targets(b,p,gt,valid,(0,1),(1,1),'positive',allowed=allowed,edge_valid=edges)
    assert result['target_indices']==[[3,1,1]]
    rejected={entry['channel']:entry['reason'] for entry in result['node_details'][0]['rejected_channels']}
    assert rejected=={0:'invalid_content',1:'outside_allowed',2:'canonical_edge_invalid'}


def test_negative_companions_are_local_outgoing_cross_gt_never_interior():
    gt=np.ones((3,4),np.int32);gt[:,2:]=2;valid=np.ones_like(gt,bool)
    b=np.full(gt.shape,.8,np.float32);b[1,1:3]=.2
    p=np.full((4,*gt.shape),.8,np.float32)
    result=readout_bottleneck_targets(b,p,gt,valid,(1,1),(1,2),'negative')
    assert result['found'] and result['target_gt_legality']
    assert set(map(tuple,result['bottleneck_nodes']))=={(1,1),(1,2)}
    assert set(map(tuple,result['target_indices']))=={(0,1,1),(2,1,1),(3,1,2)}
    for target in result['targets']:
        assert target['gt_a']!=target['gt_b'] and target['gt_target']==0
        assert target['a'] in result['bottleneck_nodes']


@pytest.mark.parametrize('direction',['positive','negative'])
def test_canonical_capacity_is_symmetric_when_path_direction_reverses(direction):
    gt=np.array([[1,1 if direction=='positive' else 2]],np.int32)
    valid=np.ones_like(gt,bool);b=np.array([[.2,.5]],np.float64);p=np.full((4,1,2),.3,np.float64)
    forward=readout_bottleneck_targets(b,p,gt,valid,(0,0),(0,1),direction)
    reverse=readout_bottleneck_targets(b,p,gt,valid,(0,1),(0,0),direction)
    assert forward['raw_capacity']==reverse['raw_capacity']==.5
    assert forward['capacity_path']['edge_indices']==reverse['capacity_path']['edge_indices']==[[0,0,0]]
    assert all(0<=target['b'][1]<2 for target in forward['targets']+reverse['targets'])


def test_domain_no_path_and_zero_length_are_explicit_and_inputs_unchanged():
    b,p,gt,valid=positive_sample();allowed=np.ones_like(gt,bool);allowed[:,2]=False
    originals=[value.copy() for value in (b,p,gt,valid,allowed)]
    no_path=readout_bottleneck_targets(b,p,gt,valid,(1,0),(1,3),'positive',allowed=allowed)
    assert not no_path['found'] and not no_path['targets_found']
    assert no_path['empty_target_reason']=='no_legal_readout_path'
    single=readout_bottleneck_targets(b,p,gt,valid,(1,1),(1,1),'positive')
    assert single['found'] and not single['targets_found'] and single['reason']=='zero_length_path'
    for before,after in zip(originals,(b,p,gt,valid,allowed)):assert np.array_equal(before,after)


@pytest.mark.parametrize('bad',['boundary_shape','boundary_int','boundary_nan','boundary_range',
    'probability_shape','probability_nan','long_channels','valid_type','edge_shape','direction'])
def test_shape_finite_types_and_direction_are_validated(bad):
    b,p,gt,valid=positive_sample();kwargs={};direction='positive'
    if bad=='boundary_shape':b=b[:2]
    elif bad=='boundary_int':b=b.astype(np.uint8)
    elif bad=='boundary_nan':b[0,0]=np.nan
    elif bad=='boundary_range':b[0,0]=1.1
    elif bad=='probability_shape':p=p[:3]
    elif bad=='probability_nan':p[0,0,0]=np.nan
    elif bad=='long_channels':p=np.tile(p,(2,1,1))
    elif bad=='valid_type':valid=valid.astype(np.uint8)
    elif bad=='edge_shape':kwargs['edge_valid']=np.ones((4,2,4),bool)
    else:direction='invalid'
    with pytest.raises(ValueError):readout_bottleneck_targets(b,p,gt,valid,(0,1),(1,1),direction,**kwargs)
