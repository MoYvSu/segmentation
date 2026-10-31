# -*- coding: utf-8 -*-
"""虚构短边图：真正maximin、GT限制、canonical反向索引与确定性。"""
import json

import numpy as np
import pytest

from utils.affinity_bottleneck import constrained_bottleneck_path
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS


def check(result, probabilities, gt, valid, start, end, allowed=None):
    assert result['found'] and result['legal']
    assert result['path'][0]==list(start) and result['path'][-1]==list(end)
    assert len(set(map(tuple,result['path'])))==len(result['path'])
    for point in result['path']:
        assert gt[tuple(point)] in result['allowed_gt'] and valid[tuple(point)]
        assert allowed is None or allowed[tuple(point)]
    for edge in result['edges']:
        channel,y,x=edge['canonical']; dy,dx=result['offsets'][channel]
        endpoints={(y,x),(y+dy,x+dx)}
        assert endpoints=={tuple(edge['a']),tuple(edge['b'])}
        assert edge['probability']==float(probabilities[channel,y,x])
    assert result['bottleneck_indices']==[edge['canonical'] for edge in result['edges']
        if edge['target_legal'] and edge['capacity']==result['capacity']]
    json.dumps(result,allow_nan=False)


def set_edge(probabilities,a,b,value):
    delta=b[0]-a[0],b[1]-a[1]
    for channel,(dy,dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        if delta==(dy,dx): probabilities[channel,*a]=value; return
        if delta==(-dy,-dx): probabilities[channel,*b]=value; return
    raise AssertionError('Invalid synthetic short edge')


def test_widest_route_beats_short_weak_route():
    gt=np.ones((3,4),np.int32); valid=np.ones_like(gt,bool)
    p=np.zeros((4,*gt.shape),np.float32)
    start,end=(1,0),(1,3)
    direct=[start,(1,1),(1,2),end]; detour=[start,(0,0),(0,1),(0,2),(0,3),end]
    for a,b in zip(direct,direct[1:]): set_edge(p,a,b,.4)
    for a,b in zip(detour,detour[1:]): set_edge(p,a,b,.8)
    result=constrained_bottleneck_path(p,gt,valid,start,end,'positive')
    check(result,p,gt,valid,start,end)
    assert result['path']==list(map(list,detour)) and result['capacity']==float(np.float32(.8))
    assert len(result['bottleneck_indices'])==5


def test_negative_selects_widest_crossing_rather_than_direct_crossing():
    gt=np.ones((3,4),np.int32); gt[:,2:]=2; valid=np.ones_like(gt,bool)
    p=np.full((4,*gt.shape),.01,np.float64)
    p[0,0,1]=.9; p[0,1,1]=.4; p[0,2,1]=.8
    result=constrained_bottleneck_path(p,gt,valid,(1,0),(1,3),'negative')
    check(result,p,gt,valid,(1,0),(1,3))
    assert result['capacity']==.9 and result['bottleneck_indices']==[[0,0,1]]


def test_negative_ignores_weak_interior_connections_and_only_cuts_cross_gt():
    gt=np.array([[7,7,7,12,12]],np.int32); valid=np.ones_like(gt,bool)
    p=np.full((4,*gt.shape),.01,np.float32); p[0,0,2]=.9
    result=constrained_bottleneck_path(p,gt,valid,(0,0),(0,4),'negative')
    check(result,p,gt,valid,(0,0),(0,4))
    assert result['capacity']==float(np.float32(.9))
    assert result['bottleneck_indices']==[[0,0,2]]
    assert all(e['capacity']==1 for e in result['edges'] if not e['cross_gt'])


def test_positive_cannot_borrow_a_high_probability_other_gt_shortcut():
    gt=np.ones((3,4),np.int32); gt[0]=2; valid=np.ones_like(gt,bool)
    p=np.full((4,*gt.shape),.2,np.float32)
    shortcut=[(1,0),(0,1),(0,2),(1,3)]
    for a,b in zip(shortcut,shortcut[1:]): set_edge(p,a,b,.99)
    result=constrained_bottleneck_path(p,gt,valid,(1,0),(1,3),'positive')
    check(result,p,gt,valid,(1,0),(1,3))
    assert result['capacity']==float(np.float32(.2)) and all(gt[tuple(point)]==1 for point in result['path'])


@pytest.mark.parametrize('barrier',['unknown','third','valid','allowed'])
def test_unknown_third_gt_padding_and_allowed_barriers_have_no_route(barrier):
    gt=np.ones((5,7),np.int32); gt[:,4:]=2; valid=np.ones_like(gt,bool); allowed=np.ones_like(gt,bool)
    if barrier=='unknown': gt[:,3]=0
    elif barrier=='third': gt[:,3]=3
    elif barrier=='valid': valid[:,3]=False
    else: allowed[:,3]=False
    p=np.ones((4,*gt.shape),np.float64)
    result=constrained_bottleneck_path(p,gt,valid,(2,1),(2,5),'negative',allowed=allowed)
    assert not result['found'] and result['reason']=='no_known_valid_allowed_path'
    assert result['path']==[] and result['bottleneck_indices']==[] and result['capacity'] is None


@pytest.mark.parametrize('channel,offset',list(enumerate(DEFAULT_AFFINITY_OFFSETS[:4])))
def test_reverse_direction_returns_actual_canonical_storage_edge(channel,offset):
    dy,dx=offset; a=(1,2); b=(1+dy,2+dx)
    gt=np.zeros((4,5),np.int32); gt[a]=gt[b]=8
    valid=gt>0; p=np.zeros((4,*gt.shape),np.float64); p[channel,*a]=.73
    result=constrained_bottleneck_path(p,gt,valid,b,a,'positive')
    check(result,p,gt,valid,b,a)
    assert result['edge_indices']==[[channel,*a]] and result['bottleneck_indices']==[[channel,*a]]
    assert result['edges'][0]['canonical_direction']==-1 and result['capacity']==.73


def test_ties_and_input_arrays_are_deterministic_and_not_mutated():
    gt=np.ones((6,7),np.int64); valid=np.ones_like(gt,bool); allowed=valid.copy()
    p=np.full((4,*gt.shape),.6,np.float64); originals=[v.copy() for v in (p,gt,valid,allowed)]
    results=[constrained_bottleneck_path(p,gt,valid,(4,1),(1,5),'positive',allowed=allowed) for _ in range(3)]
    assert results[0]==results[1]==results[2]
    check(results[0],p,gt,valid,(4,1),(1,5),allowed)
    assert len(results[0]['bottleneck_indices'])==results[0]['steps']
    for current,original in zip((p,gt,valid,allowed),originals): assert np.array_equal(current,original)


def test_negative_equal_one_capacity_selects_only_actual_crossing_edges():
    gt=np.array([[1,1,2,2]],np.int32); valid=np.ones_like(gt,bool)
    p=np.ones((4,*gt.shape),np.float32)
    result=constrained_bottleneck_path(p,gt,valid,(0,0),(0,3),'negative')
    assert result['capacity']==1 and result['bottleneck_indices']==[[0,0,1]]


def test_zero_capacity_is_an_explicit_legal_path_not_unknown_no_route():
    gt=np.array([[1,2]],np.int32); valid=np.ones_like(gt,bool); p=np.zeros((4,*gt.shape),np.float32)
    result=constrained_bottleneck_path(p,gt,valid,(0,0),(0,1),'negative')
    assert result['found'] and result['zero_capacity'] and result['capacity']==0
    assert result['bottleneck_indices']==[[0,0,0]]


def test_invalid_endpoints_and_direction_return_structured_failure():
    gt=np.array([[1,0,2]],np.int32); valid=np.ones_like(gt,bool); p=np.ones((4,*gt.shape),np.float32)
    for end,reason in [((0,1),'end_unknown_gt'),((1,0),'end_out_of_bounds')]:
        result=constrained_bottleneck_path(p,gt,valid,(0,0),end,'negative')
        assert not result['found'] and result['reason']==reason
    valid[0,2]=False
    assert constrained_bottleneck_path(p,gt,valid,(0,0),(0,2),'negative')['reason']=='end_invalid_content'
    assert constrained_bottleneck_path(p,gt,valid,(0,0),(0,0),'negative')['reason']=='negative_requires_distinct_endpoint_gt'
    valid[:]=True
    assert constrained_bottleneck_path(p,gt,valid,(0,0),(0,2),'positive')['reason']=='positive_requires_same_endpoint_gt'


def test_single_positive_endpoint_returns_empty_edge_path():
    gt=np.ones((1,1),np.int32); valid=np.ones_like(gt,bool); p=np.ones((4,1,1),np.float32)
    result=constrained_bottleneck_path(p,gt,valid,(0,0),(0,0),'positive')
    assert result['found'] and result['capacity']==1 and result['steps']==0 and result['bottleneck_indices']==[]


@pytest.mark.parametrize('reverse',[False,True])
def test_canonical_edge_obstacle_blocks_forward_and_reverse_without_mutation(reverse):
    gt=np.array([[7,12]],np.int32); valid=np.ones_like(gt,bool)
    p=np.ones((4,*gt.shape),np.float32); edge_valid=np.ones_like(p,bool)
    start,end=((0,1),(0,0)) if reverse else ((0,0),(0,1))
    baseline=constrained_bottleneck_path(p,gt,valid,start,end,'negative')
    assert baseline['found'] and baseline['bottleneck_indices']==[[0,0,0]]
    edge_valid[0,0,0]=False; snapshots=[v.copy() for v in (p,gt,valid,edge_valid)]
    result=constrained_bottleneck_path(p,gt,valid,start,end,'negative',edge_valid=edge_valid)
    assert not result['found'] and result['reason']=='no_known_valid_allowed_path'
    for before,after in zip(snapshots,(p,gt,valid,edge_valid)): assert np.array_equal(before,after)


def test_optional_edge_mask_preserves_default_route_and_reports_grid_coordinate_space():
    gt=np.ones((3,4),np.int32); valid=np.ones_like(gt,bool); p=np.full((4,*gt.shape),.8,np.float32)
    base=constrained_bottleneck_path(p,gt,valid,(0,0),(2,3),'positive')
    same=constrained_bottleneck_path(p,gt,valid,(0,0),(2,3),'positive',edge_valid=None)
    full=constrained_bottleneck_path(p,gt,valid,(0,0),(2,3),'positive',edge_valid=np.ones_like(p,bool))
    assert base==same and base['path']==full['path'] and base['capacity']==full['capacity']
    assert base['coordinate_space']=='input_grid_yx' and not base['native_projection_performed']
    assert not base['edge_valid_supplied'] and full['edge_valid_supplied']


@pytest.mark.parametrize('bad',[np.ones((3,2,3),bool),np.ones((4,2,3),np.uint8),np.ones((4,2,2),bool)])
def test_canonical_edge_mask_shape_and_dtype_are_strict(bad):
    gt=np.ones((2,3),np.int32); valid=np.ones_like(gt,bool); p=np.ones((4,*gt.shape),np.float32)
    with pytest.raises(ValueError,match='edge_valid'):
        constrained_bottleneck_path(p,gt,valid,(0,0),(1,2),'positive',edge_valid=bad)


@pytest.mark.parametrize('bad',['shape','integer_probability','nan','range','float_gt','negative_gt',
    'int_valid','allowed_shape','offset_duplicate','long_offset','float_point','direction'])
def test_illegal_input_shapes_types_and_nan_raise(bad):
    gt=np.ones((2,3),np.int32); valid=np.ones_like(gt,bool); p=np.ones((4,*gt.shape),np.float32)
    kwargs={}; start=(0,0); direction='positive'
    if bad=='shape': p=p[:3]
    elif bad=='integer_probability': p=p.astype(np.int32)
    elif bad=='nan': p[0,0,0]=np.nan
    elif bad=='range': p[0,0,0]=1.01
    elif bad=='float_gt': gt=gt.astype(float)
    elif bad=='negative_gt': gt[0,0]=-1
    elif bad=='int_valid': valid=valid.astype(np.uint8)
    elif bad=='allowed_shape': kwargs['allowed']=np.ones((2,2),bool)
    elif bad=='offset_duplicate': kwargs['offsets']=((0,1),(0,-1),(1,1),(1,-1))
    elif bad=='long_offset': kwargs['offsets']=((0,2),(1,0),(1,1),(1,-1))
    elif bad=='float_point': start=(0.,0)
    else: direction='invalid'
    with pytest.raises(ValueError): constrained_bottleneck_path(p,gt,valid,start,(1,2),direction,**kwargs)
