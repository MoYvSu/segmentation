# -*- coding: utf-8 -*-
"""固定实例末端语义短测：选位从预测，分类答案仅来自新GT原标域。"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np
import torch
import torch.nn.functional as F

from data.semantic_targets import semantic_targets_from_instances
from utils.offset_letterbox import letterbox_instance_geometry


def array_sha(value):
    if torch.is_tensor(value):value=value.detach().cpu().numpy()
    value=np.asarray(value); digest=hashlib.sha256()
    digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode())
    digest.update(np.ascontiguousarray(value).tobytes());return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _gt_contract(gt, original, classes):
    gt,original=np.asarray(gt),np.asarray(original)
    if (gt.ndim!=2 or not gt.size or not np.issubdtype(gt.dtype,np.integer)
            or gt.dtype==bool or gt.min()<0 or original.dtype!=bool or original.shape!=gt.shape
            or np.any(original & (gt==0))):
        raise ValueError('Matching nonnegative integer GT and original-covered bool required')
    present={int(g) for g in np.unique(gt) if g}
    if (not present.issubset(classes) or any(type(g) is not int or g<=0 for g in classes)
            or any(type(v) is not int or v not in (0,1) for v in classes.values())):
        raise ValueError('GT class 0/1 required for every processed instance')
    return gt,original


def select_regions(pred, gt, original, classes):
    """全部全原覆盖、纯相、非直接触图框预测区域；无预测类别输入。"""
    gt,original=_gt_contract(gt,original,classes);pred=np.asarray(pred)
    if pred.shape!=gt.shape or pred.dtype!=np.uint16:
        raise ValueError('Native uint16 fixed geometry must match processed GT')
    count=int(pred.max())+1
    areas=np.bincount(pred.ravel(),minlength=count)
    coverage=np.bincount(pred[original],minlength=count)
    lookup=np.full(int(gt.max())+1,-1,np.int8)
    for gid,cls in classes.items():
        if gid<len(lookup):lookup[gid]=cls
    phase=lookup[gt]
    F_pixels=np.bincount(pred[original & (phase==1)],minlength=count)
    P_pixels=np.bincount(pred[original & (phase==0)],minlength=count)
    frame=set(map(int,np.unique(np.concatenate((pred[0],pred[-1],pred[:,0],pred[:,-1])))))
    regions=[]
    for pid in np.flatnonzero(areas):
        if pid==0 or pid in frame or coverage[pid]!=areas[pid]:continue
        if bool(F_pixels[pid])==bool(P_pixels[pid]):continue
        regions.append(dict(id=int(pid),cls=int(F_pixels[pid]>0),native_area=int(areas[pid])))
    audit=dict(regions=len(regions),ferrite=sum(r['cls']==1 for r in regions),
               pearlite=sum(r['cls']==0 for r in regions),selected_native_pixels=sum(r['native_area'] for r in regions),
               region_sha256=canonical_sha(regions),selection_independent_of_predicted_class=True,
               all_original_pure_nonframe=True,unknown_filled_and_frame_excluded=True)
    return regions,audit


def build_semantic_batch(gt, original, classes, input_size=1024):
    """共同监督沿CanonicalBackendDataset：processed-known含filled，unknown/padding忽略。"""
    gt,original=_gt_contract(gt,original,classes)
    if type(input_size) is not int or input_size<1:raise ValueError('Positive input size required')
    grid,valid,geometry=letterbox_instance_geometry(gt,input_size,input_size)
    lookup=np.full(int(gt.max())+1,-1,np.float32)
    for gid,cls in classes.items():
        if gid<len(lookup):lookup[gid]=cls
    semantic,boundary=semantic_targets_from_instances(grid,lookup)
    semantic[~valid | (grid==0)] = -1.
    return dict(semantic_target=torch.from_numpy(semantic.copy()).float()[None,None],
                semantic_boundary=torch.from_numpy(boundary.copy()).float()[None,None],
                semantic_instance_map=torch.from_numpy(grid.astype(np.int64))[None],
                semantic_valid_content=torch.from_numpy(valid & (grid>0))[None,None],
                geometry=geometry.to_dict(),
                support='processed-known including filled; unknown/padding -1',
                original_pixels=int(original.sum()),filled_pixels=int(((gt>0)&~original).sum()))


def paired_schedule(names,seed=20261004):
    names=list(names)
    if len(names)!=32 or len(set(names))!=32 or type(seed) is not int:
        raise ValueError('Unique 32-source paired schedule required')
    order=np.random.default_rng(seed).permutation(sorted(names)).tolist()
    return order+order


def feature_copy(value,device):
    """保留实际张量stride；不以contiguous换布局或更改精度。"""
    if not torch.is_tensor(value) or value.dtype!=torch.float32 or not torch.isfinite(value).all():
        raise ValueError('Actual finite FP32 tensor required')
    out=torch.empty_strided(value.size(),value.stride(),dtype=value.dtype,device=device)
    out.copy_(value.detach())
    if out.stride()!=value.stride():raise RuntimeError('Actual input/feature tensor layout changed')
    return out.detach()


def region_probability_loss(native_logits,region_ids,regions):
    """先恢复原生logit再sigmoid，在每个固定区域平均概率；不平均logit。"""
    if (native_logits.ndim!=4 or native_logits.shape[:2]!=(1,1) or native_logits.dtype!=torch.float32
            or not torch.isfinite(native_logits).all()):
        raise ValueError('Finite FP32 singleton native logits required')
    ids=torch.as_tensor(region_ids,device=native_logits.device)
    if ids.shape!=native_logits.shape[-2:] or ids.dtype not in (torch.int32,torch.int64,torch.uint16):
        raise ValueError('Region IDs must retain integer native coordinates')
    if len({r['id'] for r in regions})!=len(regions):raise ValueError('Unique fixed region IDs required')
    p=native_logits[0,0].sigmoid();losses={0:[],1:[]};means=[]
    for row in regions:
        if (type(row['id']) is not int or row['id']<=0 or type(row['cls']) is not int or row['cls'] not in (0,1)
                or type(row['native_area']) is not int or row['native_area']<=0):
            raise ValueError('Positive region ID and GT class 0/1 required')
        mask=ids==row['id']
        if not bool(mask.any()) or int(mask.sum())!=row['native_area']:
            raise ValueError('Fixed region area changed')
        mean=p[mask].mean(dtype=torch.float32)
        # 与现有pool损失一致的稳定概率BCE；保存未clamp均值作诊断。
        prob_logit=torch.logit(mean.float(),eps=1e-5)
        loss=F.binary_cross_entropy_with_logits(prob_logit,mean.new_tensor(float(row['cls'])))
        losses[row['cls']].append(loss)
        means.append(dict(id=row['id'],cls=row['cls'],probability=float(mean.detach()),loss=float(loss.detach())))
    grouped=[torch.stack(values).mean() for values in losses.values() if values]
    total=torch.stack(grouped).mean() if grouped else native_logits.sum()*0.
    return total,dict(regions=len(regions),ferrite=len(losses[1]),pearlite=len(losses[0]),
                      active_classes=len(grouped),class_balance=True,missing_class_fallback='mean present class',
                      probability_mean_after_native_logit_restore=True,region_rows=means)


def object_outcomes(regions,baseline_classes,current_classes,ids):
    areas=np.bincount(np.asarray(ids).ravel());present={str(int(i)) for i in np.flatnonzero(areas) if i}
    if set(baseline_classes)!=present or set(current_classes)!=present:
        raise ValueError('Fixed instance ownership/class coverage changed')
    if any(type(v) is not int or v not in (0,1) for mapping in (baseline_classes,current_classes) for v in mapping.values()):
        raise ValueError('Strict 0/1 predicted classes required')
    if (len({r['id'] for r in regions})!=len(regions) or any(
            type(r['id']) is not int or str(r['id']) not in present or type(r['cls']) is not int
            or r['cls'] not in (0,1) or type(r['native_area']) is not int
            or r['native_area']!=int(areas[r['id']]) for r in regions)):
        raise ValueError('Fixed region identity, GT class or area changed')
    selected=[]
    for row in regions:
        old=int(baseline_classes[str(row['id'])]);new=int(current_classes[str(row['id'])]);truth=row['cls']
        selected.append(dict(**row,baseline_class=old,predicted_class=new,
                             baseline_wrong=old!=truth,wrong=new!=truth,
                             corrected=old!=truth and new==truth,regressed=old==truth and new!=truth))
    F=[int(i) for i,v in current_classes.items() if v==1]
    flips=[int(i) for i in present if current_classes[i]!=baseline_classes[i]]
    buckets={name:dict(count=int(sum(lo<=areas[i] and (hi is None or areas[i]<hi) for i in F)),
                      pixels=int(sum(areas[i] for i in F if lo<=areas[i] and (hi is None or areas[i]<hi))))
             for name,lo,hi in (('lt200',0,200),('200to999',200,1000),('1000to9999',1000,10000),('ge10000',10000,None))}
    return dict(fixed_regions=len(regions),initial_wrong=sum(r['baseline_wrong'] for r in selected),
                wrong=sum(r['wrong'] for r in selected),corrected=sum(r['corrected'] for r in selected),
                original_correct_regressed=sum(r['regressed'] for r in selected),
                F_to_P_wrong=sum(r['wrong'] and r['cls']==1 for r in selected),
                P_to_F_wrong=sum(r['wrong'] and r['cls']==0 for r in selected),
                objects=selected,all_fixed_predictions=len(present),ferrite_count=len(F),
                ferrite_pixels=int(sum(areas[i] for i in F)),
                ferrite_mean_area=float(sum(areas[i] for i in F)/len(F)) if F else None,
                ferrite_size_buckets=buckets,changed_instances=len(flips),changed_pixels=int(sum(areas[i] for i in flips)))
