# -*- coding: utf-8 -*-
"""只读证明已知GT对中的实际marker连通，返回全图八邻最短路径。"""
from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np


NEIGHBORS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))


def _integer_map(value, name):
    value=np.asarray(value)
    if (value.ndim != 2 or not value.size or value.dtype == np.bool_
            or not np.issubdtype(value.dtype,np.integer) or np.any(value<0)):
        raise ValueError(name+' must be a nonempty nonnegative integer HW map')
    return value


def _positive_pair(pair):
    if (not isinstance(pair,(list,tuple)) or len(pair)!=2
            or any(isinstance(g,bool) or not isinstance(g,Integral) or g<=0 for g in pair)
            or pair[0]==pair[1]):
        raise ValueError('pair must contain two distinct positive integer GT IDs')
    return tuple(map(int,pair))


def known_marker_path(marker_labels, gt, pair, points):
    """在同一正marker且GT属于pair的域内找最短八邻通路，不修改输入。

    两个固定端点必须依次属于pair[0]/pair[1]。同ID但被未知/第三GT
    隔断时返回found=False。length是路径像素数，steps是相邻连接数。
    内部bbox只减少搜索开销，path/crossings始终为native全图坐标。
    """
    markers=_integer_map(marker_labels,'marker_labels'); gt=_integer_map(gt,'gt')
    if markers.shape != gt.shape:
        raise ValueError('GT and marker shapes must match')
    pair=_positive_pair(pair)
    if (not isinstance(points,(list,tuple)) or len(points)!=2
            or any(not isinstance(p,(list,tuple)) or len(p)!=2 for p in points)):
        raise ValueError('points must contain two [y,x] coordinates')
    h,w=gt.shape; fixed=[]
    for index,point in enumerate(points):
        if any(isinstance(v,bool) or not isinstance(v,Integral) for v in point):
            raise ValueError('Point coordinates must be integers')
        y,x=map(int,point)
        if not (0<=y<h and 0<=x<w) or int(gt[y,x])!=pair[index]:
            raise ValueError('Each point must lie in its corresponding positive GT ID')
        fixed.append((y,x))
    endpoint_markers=[int(markers[p]) for p in fixed]
    result=dict(found=False,reason=None,path=[],length=0,steps=0,marker=None,
                all_known_pair_only=False,crossings=[],connectivity=8,
                length_unit='path_pixels',endpoint_markers=endpoint_markers)
    if min(endpoint_markers)<=0:
        result['reason']='nonpositive_endpoint_marker'; return result
    if endpoint_markers[0]!=endpoint_markers[1]:
        result['reason']='different_endpoint_markers'; return result
    marker=endpoint_markers[0]; result['marker']=marker
    allowed=(markers==marker)&np.isin(gt,pair)
    yy,xx=np.nonzero(allowed)
    y0,y1=int(yy.min()),int(yy.max())+1; x0,x1=int(xx.min()),int(xx.max())+1
    region=allowed[y0:y1,x0:x1]; rh,rw=region.shape
    start=(fixed[0][0]-y0)*rw+fixed[0][1]-x0
    end=(fixed[1][0]-y0)*rw+fixed[1][1]-x0
    # -2=unvisited, -1=the unique root. Only visited legal pixels receive parents.
    parent=np.full(region.size,-2,np.int64); parent[start]=-1
    queue=deque([start]); explored=0
    while queue and parent[end]==-2:
        current=queue.popleft(); explored+=1
        cy,cx=divmod(current,rw)
        for dy,dx in NEIGHBORS:
            ny,nx=cy+dy,cx+dx
            if not (0<=ny<rh and 0<=nx<rw) or not region[ny,nx]:
                continue
            neighbor=ny*rw+nx
            if parent[neighbor]!=-2:
                continue
            parent[neighbor]=current
            queue.append(neighbor)
    result.update(explored_pixels=explored,search_bbox=[y0,x0,y1,x1],allowed_pixels=int(allowed.sum()))
    if parent[end]==-2:
        result['reason']='no_known_pair_path_in_shared_marker'; return result
    reverse=[]; current=end
    while current!=-1:
        cy,cx=divmod(current,rw)
        reverse.append([int(cy+y0),int(cx+x0)])
        current=int(parent[current])
    path=reverse[::-1]; crossings=[]
    for a,b in zip(path,path[1:]):
        ga,gb=int(gt[tuple(a)]),int(gt[tuple(b)])
        if ga!=gb:
            crossings.append(dict(a=a,b=b,gt_a=ga,gt_b=gb))
    result.update(found=True,reason='known_pair_connected',path=path,length=len(path),
                  steps=len(path)-1,all_known_pair_only=True,crossings=crossings)
    return result
