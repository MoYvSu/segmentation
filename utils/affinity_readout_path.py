# -*- coding: utf-8 -*-
"""冻结读取诊断：按实际融合边界找通路，再选择同像素合法短关系。"""
from __future__ import annotations

import numpy as np

from utils.affinity_bottleneck import constrained_bottleneck_path
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS


def readout_bottleneck_targets(boundary, probabilities, gt, valid, start, end, direction, *,
                               allowed=None, edge_valid=None):
    """返回输入网格通路及读取目标；不执行模型修正或原生/WS推理。

    网格节点连接容量为1-B；canonical边容量取两端节点容量的较小值。
    negative仍将同GT边容量固定1，仅跨GT边参与瓶颈；positive的瓶颈
    节点满足1-B==raw_capacity，即B==1-raw_capacity。原图障碍只通过
    调用方提供的edge_valid约束，本模块不声称完成原生投影。

    positive按原像素四个q=1-p的第二大值选全部ties，再过滤合法性，
    不用过滤后的排序冒充真实top2。negative只选瓶颈跨GT边两端节点
    上实际存储的outgoing跨GT关系，不罚粒内，也不覆盖整条界面。
    """
    boundary=np.asarray(boundary); probabilities=np.asarray(probabilities)
    gt=np.asarray(gt); valid=np.asarray(valid)
    if (boundary.ndim!=2 or not boundary.size or boundary.shape!=gt.shape
            or not np.issubdtype(boundary.dtype,np.floating)
            or not np.isfinite(boundary).all() or np.any((boundary<0)|(boundary>1))):
        raise ValueError('boundary must be finite floating HW in [0,1] matching GT')
    if (probabilities.shape!=(4,*boundary.shape)
            or not np.issubdtype(probabilities.dtype,np.floating)
            or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1))):
        raise ValueError('probabilities must be finite floating short [4,H,W] in [0,1]')
    if allowed is not None:
        allowed=np.asarray(allowed)
    if edge_valid is not None:
        edge_valid=np.asarray(edge_valid)
    h,w=boundary.shape
    node_capacity=1.-boundary.astype(np.float64)
    capacities=np.zeros((4,h,w),np.float64)
    for channel,(dy,dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        y0,y1=max(0,-dy),min(h,h-dy); x0,x1=max(0,-dx),min(w,w-dx)
        if y0<y1 and x0<x1:
            capacities[channel,y0:y1,x0:x1]=np.minimum(node_capacity[y0:y1,x0:x1],
                node_capacity[y0+dy:y1+dy,x0+dx:x1+dx])
    path=constrained_bottleneck_path(capacities,gt,valid,start,end,direction,
                                     allowed=allowed,edge_valid=edge_valid)
    result=dict(found=path['found'],reason=path['reason'],targets_found=False,
        direction=direction,path=path['path'],raw_capacity=path['capacity'],
        bottleneck_boundary=None,bottleneck_nodes=[],node_details=[],target_indices=[],targets=[],
        target_gt_legality=False,coordinate_space='input_grid_yx',
        canonical_index_space='input_grid_channel_yx',native_projection_performed=False,
        watershed_success_claimed=False,capacity_path=path,
        path_capacity_input='canonical min(1-boundary[a],1-boundary[b]); negative sameGT fixed1',
        target_scope='canonical outgoing short relations at selected bottleneck nodes only',
        empty_target_reason=None,
        caveat='Frozen readout mechanism only; no training gain, native barrier or watershed success inferred')
    if not path['found']:
        result['empty_target_reason']='no_legal_readout_path'
        return result
    if not path['edges']:
        result.update(reason='zero_length_path',empty_target_reason='zero_length_path')
        return result
    capacity=path['capacity']; result['bottleneck_boundary']=1.-capacity
    if direction=='positive':
        nodes=[point for point in path['path'] if float(node_capacity[tuple(point)])==capacity]
    else:
        chosen=set(path['bottleneck_edge_indices']); nodes=[]
        for edge in path['edges']:
            if edge['path_index'] in chosen:
                for point in (edge['a'],edge['b']):
                    if point not in nodes:
                        nodes.append(point)
    result['bottleneck_nodes']=nodes
    legal_pixels=np.ones(gt.shape,bool) if allowed is None else allowed
    groups=set(path['allowed_gt']); q=1.-probabilities
    for node in nodes:
        y,x=node; cutoff=float(np.sort(q[:,y,x])[-2])
        active=[channel for channel in range(4) if q[channel,y,x]>=cutoff]
        node_record=dict(point=node,gt=int(gt[y,x]),boundary=float(boundary[y,x]),
            connection_capacity=float(node_capacity[y,x]),
            short_disconnection_probabilities=[float(v) for v in q[:,y,x]],
            top2_cutoff=cutoff,top2_members_including_ties=active,selected_channels=[],
            rejected_channels=[])
        for channel,(dy,dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
            ny,nx=y+dy,x+dx; rejected=None
            if direction=='positive' and channel not in active:
                rejected='not_actual_top2_member'
            elif not (0<=ny<h and 0<=nx<w):
                rejected='canonical_target_out_of_bounds'
            elif not valid[y,x] or not valid[ny,nx]:
                rejected='invalid_content'
            elif not legal_pixels[y,x] or not legal_pixels[ny,nx]:
                rejected='outside_allowed'
            elif int(gt[y,x])<=0 or int(gt[ny,nx])<=0:
                rejected='unknown_gt'
            elif int(gt[y,x]) not in groups or int(gt[ny,nx]) not in groups:
                rejected='third_gt'
            elif (gt[y,x]!=gt[ny,nx]) if direction=='positive' else (gt[y,x]==gt[ny,nx]):
                rejected='wrong_gt_relation'
            elif edge_valid is not None and not edge_valid[channel,y,x]:
                rejected='canonical_edge_invalid'
            if rejected is not None:
                node_record['rejected_channels'].append(dict(channel=channel,reason=rejected))
                continue
            index=[channel,int(y),int(x)]
            node_record['selected_channels'].append(channel)
            result['target_indices'].append(index)
            result['targets'].append(dict(canonical=index,a=[int(y),int(x)],b=[int(ny),int(nx)],
                gt_a=int(gt[y,x]),gt_b=int(gt[ny,nx]),gt_target=1 if direction=='positive' else 0,
                same_instance_probability=float(probabilities[channel,y,x]),
                disconnection_probability=float(q[channel,y,x]),
                actual_top2_member_including_ties=channel in active,
                both_endpoints_known_valid_allowed=True,canonical_edge_valid=True))
        result['node_details'].append(node_record)
    result['targets_found']=bool(result['target_indices'])
    result['target_gt_legality']=bool(result['targets']) and all(
        (target['gt_a']==target['gt_b']) if direction=='positive' else
        (target['gt_a']!=target['gt_b'] and target['gt_a'] in groups and target['gt_b'] in groups)
        for target in result['targets'])
    if not result['targets_found']:
        result.update(reason='readout_path_without_legal_targets',
            empty_target_reason='no_legal_actual_top2_outgoing_targets' if direction=='positive'
                                else 'no_legal_cross_gt_outgoing_companions')
    else:
        result['reason']='legal_readout_bottleneck_targets'
    return result
