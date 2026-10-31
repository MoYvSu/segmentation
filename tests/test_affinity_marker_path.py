# -*- coding: utf-8 -*-
"""真实marker已知域路径：未知隔断、第三GT、八邻及全图坐标。"""
from collections import deque

import numpy as np
import pytest

from utils.affinity_marker_path import known_marker_path, NEIGHBORS


def check_path(result,markers,gt,pair,points):
    assert result['found'] and result['all_known_pair_only']
    path=result['path']
    assert path[0]==list(points[0]) and path[-1]==list(points[1])
    assert result['length']==len(path) and result['steps']==len(path)-1
    assert len({tuple(p) for p in path})==len(path)
    for p in path:
        assert gt[tuple(p)] in pair and markers[tuple(p)]==result['marker']>0
    for a,b in zip(path,path[1:]):
        assert max(abs(a[0]-b[0]),abs(a[1]-b[1]))==1
    expected=[dict(a=a,b=b,gt_a=int(gt[tuple(a)]),gt_b=int(gt[tuple(b)]))
              for a,b in zip(path,path[1:]) if gt[tuple(a)]!=gt[tuple(b)]]
    assert result['crossings']==expected


def test_diagonal_shortest_path_native_bbox_offset_and_inputs_unchanged():
    gt=np.zeros((16,18),np.int32); markers=np.zeros_like(gt)
    for i in range(5):
        gt[8+i,10+i]=7 if i<2 else 12; markers[8+i,10+i]=31
    beforegt=gt.copy(); beforemarkers=markers.copy()
    points=[[8,10],[12,14]]
    result=known_marker_path(markers,gt,[7,12],points)
    check_path(result,markers,gt,[7,12],points)
    assert result['steps']==4 and result['search_bbox']==[8,10,13,15]
    assert len(result['crossings'])==1
    assert np.array_equal(gt,beforegt) and np.array_equal(markers,beforemarkers)


def test_shared_id_unknown_wall_is_not_a_valid_path():
    gt=np.ones((5,9),np.uint16); gt[:,5:]=2; gt[:,4]=0
    markers=np.full(gt.shape,13,np.int32)
    result=known_marker_path(markers,gt,[1,2],[[2,1],[2,7]])
    assert not result['found'] and result['path']==[] and result['crossings']==[]
    assert result['marker']==13 and result['reason']=='no_known_pair_path_in_shared_marker'


def test_third_gt_shortcut_excluded_and_legal_detour_selected():
    gt=np.ones((7,11),np.int32); gt[:,6:]=2; gt[2:5,5]=3
    markers=np.full(gt.shape,4,np.int32)
    points=[[3,2],[3,8]]
    result=known_marker_path(markers,gt,[1,2],points)
    check_path(result,markers,gt,[1,2],points)
    assert all(gt[tuple(p)]!=3 for p in result['path'])
    assert any(p[0] in (1,5) for p in result['path'])
    # An independent distance-only BFS checks shortestness within the legal domain.
    allowed=np.isin(gt,[1,2]); start,end=map(tuple,points); distances={start:0}; queue=deque([start])
    while queue:
        y,x=queue.popleft()
        for dy,dx in NEIGHBORS:
            p=y+dy,x+dx
            if 0<=p[0]<7 and 0<=p[1]<11 and allowed[p] and p not in distances:
                distances[p]=distances[(y,x)]+1; queue.append(p)
    assert result['steps']==distances[end]


def test_third_gt_complete_wall_cannot_be_used_even_with_shared_marker():
    gt=np.ones((6,9),np.int64); gt[:,5:]=2; gt[:,4]=3
    result=known_marker_path(np.ones_like(gt),gt,[1,2],[[2,1],[2,7]])
    assert not result['found'] and result['length']==0


@pytest.mark.parametrize('right_marker',[0,8])
def test_different_or_nonpositive_endpoint_markers_return_no_path(right_marker):
    gt=np.array([[1,2]],np.int32); markers=np.array([[7,right_marker]],np.int32)
    result=known_marker_path(markers,gt,[1,2],[[0,0],[0,1]])
    assert not result['found'] and result['path']==[]
    assert result['reason']==('nonpositive_endpoint_marker' if right_marker==0 else 'different_endpoint_markers')


def test_shared_marker_disconnected_by_other_marker_cannot_connect():
    gt=np.ones((5,8),np.int32); gt[:,4:]=2
    markers=np.ones_like(gt); markers[:,3]=2
    result=known_marker_path(markers,gt,[1,2],[[2,1],[2,6]])
    assert not result['found']


@pytest.mark.parametrize('badmap',[np.ones((3,3),bool),np.ones((3,3),float),np.full((3,3),-1,np.int32)])
def test_invalid_maps_rejected(badmap):
    with pytest.raises(ValueError):
        known_marker_path(badmap,np.ones((3,3),np.int32),[1,2],[[0,0],[1,1]])


def test_bad_points_pair_or_shape_rejected():
    gt=np.array([[1,2]],np.int32); markers=np.ones_like(gt)
    for pair,points in [([1,1],[[0,0],[0,1]]),([1,2],[[0,1],[0,0]]),
                        ([1,2],[[0,0],[0,3]]),([1,2],[[False,0],[0,1]])]:
        with pytest.raises(ValueError):
            known_marker_path(markers,gt,pair,points)
    with pytest.raises(ValueError):
        known_marker_path(np.ones((2,2),np.int32),gt,[1,2],[[0,0],[0,1]])
