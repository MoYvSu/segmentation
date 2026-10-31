# -*- coding: utf-8 -*-
"""虚构CPU投影契约：原图障碍、nearest差异、padding与短边干预隔离。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from tools.probe_affinity_bottleneck import (corrected_logits, native_segment_path,
    project_native_support, select_bottlenecks)
from utils.affinity_bottleneck import constrained_bottleneck_path
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.offset_letterbox import letterbox_instance_geometry


def tile_for(gt,box=None,grid=8):
    if box is None: box=[0,0,*gt.shape]
    y,x,y1,x1=box
    grid_gt,valid,geometry=letterbox_instance_geometry(gt[y:y1,x:x1],1024,grid)
    return dict(index=0,box=box,grid_gt=grid_gt,valid=valid,geometry=geometry.to_dict())


def test_letterbox_padding_is_ignored_and_projection_returns_full_native_offsets():
    gt=np.ones((12,20),np.int32); box=[3,4,8,15]; tile=tile_for(gt,box)
    support=np.ones_like(gt,bool); snapshot=deepcopy(tile)
    projected,edges,yy,xx=project_native_support(tile,gt,support)
    valid=tile['valid']; ch=tile['geometry']['content_height']; cw=tile['geometry']['content_width']
    assert projected.dtype==np.bool_ and edges.dtype==np.bool_ and edges.shape==(4,8,8)
    assert np.array_equal(projected,valid) and not projected[ch:,:].any()
    assert (yy[0,0],xx[0,0])==(3,4) and (yy[ch-1,cw-1],xx[ch-1,cw-1])==(7,14)
    for c,(dy,dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        for y,x in zip(*np.nonzero(edges[c])):
            assert valid[y,x] and valid[y+dy,x+dx]
    assert np.array_equal(tile['grid_gt'],snapshot['grid_gt'])
    assert np.array_equal(tile['valid'],snapshot['valid']) and tile['geometry']==snapshot['geometry']


@pytest.mark.parametrize('block',['unknown','third_gt','marker'])
def test_native_obstacle_between_known_grid_centers_cannot_be_jumped(block):
    gt=np.ones((17,17),np.int32); support=np.ones_like(gt,bool)
    if block=='unknown': gt[:,3]=0
    elif block=='third_gt': gt[:,3]=3
    else: support[:,3]=False
    if block!='marker': support&=gt==1
    tile=tile_for(gt)
    allowed,edges,yy,xx=project_native_support(tile,gt,support)
    # nearest和bilinear中心均未采中col3，但两中心间的原生段经过它。
    assert tile['grid_gt'][3,1]==tile['grid_gt'][3,2]==1
    assert allowed[3,1] and allowed[3,2] and (xx[3,1],xx[3,2])==(2,5)
    assert not edges[0,3,1]
    p=np.ones((4,*tile['grid_gt'].shape),np.float32)
    result=constrained_bottleneck_path(p,tile['grid_gt'],tile['valid'],(3,0),(3,7),
                                     'positive',allowed=allowed,edge_valid=edges)
    assert not result['found'] and result['reason']=='no_known_valid_allowed_path'


def test_same_gt_grid_edge_cannot_clamp_over_thin_other_permitted_gt_strip():
    gt=np.ones((17,17),np.int32); gt[:,3]=2
    support=np.isin(gt,[1,2]); tile=tile_for(gt)
    allowed,edges,_,_=project_native_support(tile,gt,support)
    assert tile['grid_gt'][3,1]==tile['grid_gt'][3,2]==1
    assert allowed[3,1] and allowed[3,2]
    assert not edges[0,3,1]  # 两GT全域合法仍不足以保证本条同GT关系合法。


def test_nearest_gt_and_align_corners_center_conflict_is_explicitly_ignored():
    gt=np.ones((17,17),np.int32); gt[:,5:]=2; support=gt>0
    tile=tile_for(gt); allowed,edges,yy,xx=project_native_support(tile,gt,support)
    assert tile['grid_gt'][3,2]==1 and gt[yy[3,2],xx[3,2]]==2
    assert not allowed[:,2].any()
    for c,(dy,dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        for y,x in zip(*np.nonzero(edges[c])):
            assert allowed[y,x] and allowed[y+dy,x+dx]


def test_native_segment_expansion_is_eight_neighbor_and_retains_native_origin():
    yy=np.array([[10,10],[15,15]],np.int32); xx=np.array([[20,26],[20,26]],np.int32)
    path=native_segment_path([[0,0],[1,1]],yy,xx)
    assert path[0]==[10,20] and path[-1]==[15,26]
    assert all(max(abs(a[0]-b[0]),abs(a[1]-b[1]))==1 for a,b in zip(path,path[1:]))
    assert native_segment_path([],yy,xx)==[]
    assert native_segment_path([[0,1]],yy,xx)==[[10,26]]


def test_projection_rejects_support_shape_dtype_and_invalid_geometry():
    gt=np.ones((8,8),np.int32); tile=tile_for(gt)
    for support in (np.ones((7,8),bool),np.ones_like(gt,np.uint8)):
        with pytest.raises(ValueError,match='support'): project_native_support(tile,gt,support)
    bad=deepcopy(tile); bad['geometry']['content_height']=9
    with pytest.raises(ValueError,match='geometry'): project_native_support(bad,gt,np.ones_like(gt,bool))


@pytest.mark.parametrize('direction',['negative','positive'])
def test_selected_bottlenecks_have_actual_support_and_only_correct_gt_relations(direction):
    gt=np.full((8,8),7,np.int32); labels=np.full_like(gt,19)
    if direction=='negative': gt[:,4:]=12
    else: labels[:,4:]=23
    decoded=dict(instances=labels,actual_markers=np.full_like(gt,31))
    case=dict(direction=direction,gt_ids=[7,12] if direction=='negative' else [7],
              points=[[3,1],[3,6]])
    tile=tile_for(gt); p=np.full((4,8,8),.4,np.float32)
    before_gt=gt.copy(); before_p=p.copy(); before_decoded=deepcopy(decoded)
    result=select_bottlenecks(tile,gt,decoded,case,p)
    assert result['found'] and result['native_path_known_support_exact']
    assert result['tile_contains_fixed_points']==[True,True] and result['endpoint_native_distances']==[0.,0.]
    assert result['native_path'][0]==case['points'][0] and result['native_path'][-1]==case['points'][1]
    assert result['bottleneck_indices']
    for c,y,x in result['bottleneck_indices']:
        dy,dx=DEFAULT_AFFINITY_OFFSETS[c]
        same=gt[y,x]==gt[y+dy,x+dx]
        assert same==(direction=='positive')
    for point in result['native_path']:
        assert gt[tuple(point)] in case['gt_ids'] and (direction=='positive' or decoded['actual_markers'][tuple(point)]==31)
    assert np.array_equal(gt,before_gt) and np.array_equal(p,before_p)
    for key in decoded: assert np.array_equal(decoded[key],before_decoded[key])


def test_negative_requires_actual_shared_positive_marker():
    gt=np.ones((8,8),np.int32); gt[:,4:]=2; tile=tile_for(gt)
    markers=np.ones_like(gt); markers[:,4:]=3
    decoded=dict(instances=np.ones_like(gt),actual_markers=markers)
    case=dict(direction='negative',gt_ids=[1,2],points=[[3,1],[3,6]])
    result=select_bottlenecks(tile,gt,decoded,case,np.ones((4,8,8),np.float32))
    assert not result['found'] and result['reason']=='no_shared_positive_actual_marker'


def test_outside_fixed_endpoints_are_reported_as_window_local_witness():
    gt=np.full((12,12),7,np.int32); labels=np.full_like(gt,19); labels[:,6:]=23
    tile=tile_for(gt,box=[3,3,9,9],grid=6)
    decoded=dict(instances=labels,actual_markers=np.full_like(gt,31))
    case=dict(direction='positive',gt_ids=[7],points=[[1,1],[10,10]])
    result=select_bottlenecks(tile,gt,decoded,case,np.full((4,6,6),.4,np.float32))
    assert result['found'] and result['tile_contains_fixed_points']==[False,False]
    assert all(value>0 for value in result['endpoint_native_distances'])
    assert result['native_path'][0]!=case['points'][0] and result['native_path'][-1]!=case['points'][1]
    assert 'window-local' in result['witness_scope']


def test_correction_masks_are_short_only_disjoint_and_do_not_mutate_inputs():
    logits=torch.linspace(-2,2,128).reshape(1,8,4,4)
    negative=np.zeros(tuple(logits.shape),bool); positive=negative.copy()
    negative[0,0,1,1]=True; positive[0,3,2,2]=True
    original=logits.clone(); before_n=negative.copy(); before_p=positive.copy()
    result=corrected_logits(logits,negative,positive)
    assert result[0,0,1,1]==-16 and result[0,3,2,2]==16
    selected=torch.from_numpy(negative|positive)
    assert torch.equal(result[~selected],original[~selected]) and torch.equal(logits,original)
    assert np.array_equal(negative,before_n) and np.array_equal(positive,before_p)
    conflict=positive.copy(); conflict[0,0,1,1]=True
    with pytest.raises(ValueError,match='mutually exclusive'): corrected_logits(logits,negative,conflict)
    long=negative.copy(); long[0,4,0,0]=True
    with pytest.raises(ValueError,match='four original short'): corrected_logits(logits,long,positive)
    with pytest.raises(ValueError,match='shape'): corrected_logits(logits,negative[:,:,:3],positive)
