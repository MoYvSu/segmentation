# -*- coding: utf-8 -*-
"""新GT父对象的完整裁窗与训练前梯度门禁；不推断物理GT或使用测试答案。"""
from __future__ import annotations

import math
import cv2
import numpy as np
import torch
from scipy.ndimage import find_objects

from utils.offset_letterbox import letterbox_instance_geometry
from utils.semantic_object_fit import build_semantic_batch
from utils.semantic_vote import adaptive_instance_core


def select_grain_anchors(gt,original,classes,minimum_area=10000,minimum_original=.8):
    gt=np.asarray(gt); original=np.asarray(original)
    if (gt.ndim!=2 or not np.issubdtype(gt.dtype,np.integer) or gt.min()<0 or gt.max()>65535
            or original.shape!=gt.shape or original.dtype!=bool or np.any(original&(gt==0))):
        raise ValueError('Matching processed integer GT in uint16 range and original-covered boolean mask required')
    counts=np.bincount(gt.ravel()); boxes=find_objects(gt.astype(np.int32)); selected=[]
    for cls in (0,1):
        eligible=[]
        for gid in np.flatnonzero(counts):
            gid=int(gid)
            if gid==0: continue
            if gid not in classes or type(classes[gid]) is not int or classes[gid] not in (0,1): raise ValueError('GT class0/1 required')
            if classes[gid]!=cls or counts[gid]<minimum_area: continue
            sl=boxes[gid-1]; mask=gt[sl]==gid
            touches=(sl[0].start==0 or sl[0].stop==gt.shape[0] or sl[1].start==0 or sl[1].stop==gt.shape[1])
            known=int((mask&original[sl]).sum()); area=int(counts[gid])
            if touches or known/area<minimum_original: continue
            eligible.append(dict(id=gid,cls=cls,native_area=area,original_pixels=known,original_fraction=known/area,
                bbox=[sl[0].start,sl[0].stop,sl[1].start,sl[1].stop],frame_censored=False))
        if not eligible: raise ValueError('No eligible non-frame original-covered large GT anchor for class '+str(cls))
        eligible.sort(key=lambda r:(r['native_area'],r['id'])); index=(len(eligible)-1)//2
        selected.append({**eligible[index], 'eligible_count':len(eligible),'median_index':index,
            'selection_rule':'Per-class sorted(native_area,id), lower median floor((n-1)/2), chosen without model output'})
    return selected


def grain_window(shape,bbox,context=1.5,minimum_side=512):
    if len(shape)!=2 or len(bbox)!=4 or context<1 or minimum_side<1: raise ValueError('Valid shape/bbox/context required')
    h,w=map(int,shape); y0,y1,x0,x1=map(int,bbox)
    if not (0<=y0<y1<=h and 0<=x0<x1<=w): raise ValueError('Anchor bbox out of bounds')
    side=max(int(minimum_side),int(math.ceil(context*max(y1-y0,x1-x0))))
    vh,vw=min(h,side),min(w,side)
    cy=max(0,min(h-vh,math.floor((y0+y1-vh)/2)))
    cx=max(0,min(w-vw,math.floor((x0+x1-vw)/2)))
    box=(cy,cy+vh,cx,cx+vw)
    if not (box[0]<=y0 and box[1]>=y1 and box[2]<=x0 and box[3]>=x1): raise ValueError('Window lost part of processed GT parent')
    return box


def build_grain_batch(gt,original,classes,box,anchor_id,size=1024):
    y0,y1,x0,x1=box; cropped=np.asarray(gt)[y0:y1,x0:x1]; known=np.asarray(original)[y0:y1,x0:x1]
    if int((cropped==anchor_id).sum())!=int((np.asarray(gt)==anchor_id).sum()): raise ValueError('Crop must retain entire processed GT parent')
    batch=build_semantic_batch(cropped,known,classes,size)
    grid,valid,geometry=letterbox_instance_geometry(cropped,size,size)
    original_grid=np.zeros((size,size),bool)
    original_grid[:geometry.content_height,:geometry.content_width]=cv2.resize(known.astype(np.uint8),
        (geometry.content_width,geometry.content_height),interpolation=cv2.INTER_NEAREST).astype(bool)
    if np.any(original_grid & ((grid==0)|~valid)): raise ValueError('Original support ownership differs after nearest resize')
    present=[int(v) for v in np.unique(cropped) if v]
    truncated=[gid for gid in present if int((cropped==gid).sum())!=int((np.asarray(gt)==gid).sum())]
    anchor_mask=grid==anchor_id; truncated_mask=np.isin(grid,truncated)
    complete_other=(grid>0)&~anchor_mask&~truncated_mask
    partitions=dict(anchor_original=original_grid&anchor_mask,anchor_filled=valid&anchor_mask&~original_grid,
        other_complete_original=original_grid&complete_other,other_complete_filled=valid&complete_other&~original_grid,
        truncated_neighbor_original=original_grid&truncated_mask,truncated_neighbor_filled=valid&truncated_mask&~original_grid,
        unknown=valid&(grid==0),padding=~valid)
    if not np.all(sum(mask.astype(np.uint8) for mask in partitions.values())==1): raise ValueError('Gradient domains must partition network output')
    receipt=dict(box=list(box),anchor_id=int(anchor_id),processed_parent_complete=True,physical_GT_completeness_not_claimed=True,
        native_shape=list(cropped.shape),geometry=geometry.to_dict(),network_support_pixels={k:int(v.sum()) for k,v in partitions.items()},
        neighboring_GT_instances=len(present)-1,neighbor_truncated_ids=truncated,
        loss_support='processed-known including filled; unknown/padding ignored; anchor diagnostics original-covered only')
    return batch,partitions,receipt


def anchor_probability(probability,gt,original,anchor):
    probability=np.asarray(probability)
    if probability.shape!=gt.shape or original.shape!=gt.shape or not np.isfinite(probability).all(): raise ValueError('Native grids differ')
    parent=np.asarray(gt)==anchor['id']; mask=parent&original
    core,_=adaptive_instance_core(parent,fraction=.4,min_pixels=8); core&=mask
    if not mask.any() or not core.any(): raise ValueError('Original-covered anchor/core empty')
    mean=float(probability[mask].mean(dtype=np.float32)); pc=float(probability[core].mean(dtype=np.float32)); cls=anchor['cls']
    return dict(original_pixels=int(mask.sum()),core_original_pixels=int(core.sum()),mean_ferrite_probability=mean,
        mean_true_class_probability=mean if cls else 1-mean,core_ferrite_probability=pc,
        true_class_margin=2*(mean if cls else 1-mean)-1,
        core_true_class_probability=pc if cls else 1-pc,predicted_class=int(mean>.5),
        wrong_original_pixel_fraction=float(((probability[mask]>.5)!=bool(cls)).mean()))


def gradient_partition(gradient,partitions):
    if gradient.shape[0:2]!=(1,1) or gradient.ndim!=4 or gradient.dtype!=torch.float32 or not torch.isfinite(gradient).all(): raise ValueError('Finite B1 FP32 output gradient required')
    values=gradient.detach().cpu()[0,0].abs().numpy()
    if any(mask.shape!=values.shape or mask.dtype!=bool for mask in partitions.values()) or not np.all(sum(v.astype(np.uint8) for v in partitions.values())==1): raise ValueError('Disjoint exhaustive gradient domains required')
    totals={k:float(values[v].sum(dtype=np.float64)) for k,v in partitions.items()}; total=sum(totals.values())
    if totals['unknown']!=0 or totals['padding']!=0: raise RuntimeError('Unknown/padding final-output gradient must be exactly zero')
    return dict(absolute_sum=total,absolute_by_support=totals,fraction_by_support={k:v/total if total else 0 for k,v in totals.items()},
        original_absolute=sum(v for k,v in totals.items() if k.endswith('_original')),
        filled_absolute=sum(v for k,v in totals.items() if k.endswith('_filled')),
        anchor_original_absolute=totals['anchor_original'],anchor_filled_absolute=totals['anchor_filled'],
        ancestor_interpolation_gradient_not_restricted=True)


def preflight_gate(rows):
    if not rows: raise ValueError('Nonempty predeclared views required')
    difficult=[r for r in rows if r['whole']['mean_true_class_probability']-r['crop']['mean_true_class_probability']>=.05
        or r['crop']['wrong_original_pixel_fraction']-r['whole']['wrong_original_pixel_fraction']>=.01]
    legal=all(r['parameter_gradient_norms']['seg_fpn']>0 and r['parameter_gradient_norms']['seg_branch']>0
        and all(np.isfinite(v) and v>=0 for v in r['parameter_gradient_norms'].values())
        and all(np.isfinite(v) and v>=0 for v in r['output_gradient']['absolute_by_support'].values())
        and r['output_gradient']['absolute_by_support']['unknown']==0 and r['output_gradient']['absolute_by_support']['padding']==0 for r in rows)
    actionable=[r for r in difficult if r['anchor_parameter_gradient_norms']['seg_fpn']>0
        and r['anchor_parameter_gradient_norms']['seg_branch']>0
        and all(np.isfinite(v) and v>=0 for v in r['anchor_parameter_gradient_norms'].values())
        and r['output_gradient']['anchor_original_absolute']>r['output_gradient']['anchor_filled_absolute']]
    return dict(passed=bool(actionable and legal),difficult_views=[r['view'] for r in difficult],
        actionable_anchor_views=[r['view'] for r in actionable],legal_coarse_gradient_and_support=bool(legal),test_used=False,
        predeclared_rule='At least one GT-anchor true-probability drop >=.05 or wrong-original-pixel increase >=.01, with anchor-only BCE finite nonzero FPN/classifier gradient and common-loss anchor-original gradient mass > anchor-filled; every view legal finite coarse gradient and exact zero unknown/padding output gradient')
