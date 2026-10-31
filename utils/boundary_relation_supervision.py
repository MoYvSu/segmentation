# -*- coding: utf-8 -*-
"""原生相邻预测区域的保守关系监督；GT只参与label/审计，绝不进入模型通道。"""
from __future__ import annotations

from collections import Counter
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects

KEEP, MERGE, IGNORE = 1, 0, 255

def _disk(radius):
    y,x=np.ogrid[-radius:radius+1,-radius:radius+1]
    return (x*x+y*y<=radius*radius).astype(np.uint8)

def _instances(value):
    a=np.asarray(value)
    if a.ndim!=2 or not a.size or a.dtype.kind not in 'iu' or np.any(a<0) or a.max()>65535:
        raise ValueError('Expected a native nonnegative <=65535 integer region map')
    return a

def original_components(gt,original_covered):
    """未侵蚀、逐GT-ID的原标8连通分量；filled/unknown不会被桥接。"""
    gt=_instances(gt);original=np.asarray(original_covered,dtype=bool)
    if original.shape!=gt.shape or np.any(original&(gt==0)):raise ValueError('Invalid original GT coverage')
    output=np.zeros(gt.shape,np.int32);rows=[];next_id=0
    for gid,box in enumerate(find_objects(gt),1):
        if box is None:continue
        local=(gt[box]==gid)&original[box]
        count,labels,stats,_=cv2.connectedComponentsWithStats(local.astype(np.uint8),connectivity=8)
        for cid in range(1,count):
            next_id+=1;output[box][labels==cid]=next_id
            rows.append(dict(component_id=next_id,gt_id=gid,pixels=int(stats[cid,cv2.CC_STAT_AREA])))
    return output,rows

def predicted_interiors(instances,radius=2):
    a=_instances(instances).astype(np.uint16,copy=False);kernel=_disk(radius)
    low=cv2.erode(a,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    high=cv2.dilate(a,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    return (a>0)&(low==a)&(high==a)

def adjacent_region_pairs(instances):
    """全部直接4邻接及经一个零像素的4邻接接触；不跨宽零洞。

    shared_seam为零缝中心（直接相邻时包含两端）；contact还包含跨缝两侧
    实际端点。三路交汇不删掉坏像素：每个合法A-zero-B接触都进入该pair。
    """
    a=_instances(instances);h,w=a.shape;grid=np.arange(h*w,dtype=np.int64).reshape(h,w)
    records=[]
    def add(x,y,mask,seam,contact):
        if not mask.any():return
        av,bv=x[mask].astype(np.int64),y[mask].astype(np.int64)
        key=np.minimum(av,bv)*65536+np.maximum(av,bv)
        for positions,kind in ((seam,0),(contact,1)):
            for where in positions:records.append((key,where[mask],np.full(key.shape,kind,np.uint8)))
    x,y=a[:,:-1],a[:,1:];mask=(x>0)&(y>0)&(x!=y)
    add(x,y,mask,[grid[:,:-1],grid[:,1:]],[grid[:,:-1],grid[:,1:]])
    x,y=a[:-1],a[1:];mask=(x>0)&(y>0)&(x!=y)
    add(x,y,mask,[grid[:-1],grid[1:]],[grid[:-1],grid[1:]])
    p=np.pad(a,1);g=np.pad(grid,1,constant_values=-1)
    neighbor=[p[:-2,1:-1],p[2:,1:-1],p[1:-1,:-2],p[1:-1,2:]]
    locations=[g[:-2,1:-1],g[2:,1:-1],g[1:-1,:-2],g[1:-1,2:]]
    for i in range(4):
        for j in range(i+1,4):
            x,y=neighbor[i],neighbor[j];mask=(a==0)&(x>0)&(y>0)&(x!=y)
            add(x,y,mask,[grid],[locations[i],grid,locations[j]])
    if not records:return [],dict(all_native_adjacent_pairs=0,definition='direct4 or one-zero4 contact')
    keys=np.concatenate([r[0] for r in records]);pixels=np.concatenate([r[1] for r in records]);kinds=np.concatenate([r[2] for r in records])
    order=np.lexsort((pixels,kinds,keys));keys,pixels,kinds=keys[order],pixels[order],kinds[order]
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1,len(keys)];pairs=[]
    for start,end in zip(starts[:-1],starts[1:]):
        key=int(keys[start]);seam=np.unique(pixels[start:end][kinds[start:end]==0]);contact=np.unique(pixels[start:end][kinds[start:end]==1])
        yy,xx=np.divmod(seam,w)
        pairs.append(dict(pair_ids=[key//65536,key%65536],seam_flat=seam,contact_flat=contact,
            seam_pixels=len(seam),seam_bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]))
    return pairs,dict(all_native_adjacent_pairs=len(pairs),shared_seam_pixels=sum(p['seam_pixels'] for p in pairs),
        definition='all direct4 positive-ID transitions and any four-neighbor A-zero-B path with exactly one zero; no wide-zero or out-of-image connections',
        junction_contact_pixels_retained=True)

def region_shape_descriptors(instances):
    a=_instances(instances);areas=np.bincount(a.ravel(),minlength=65536);perimeter=np.zeros(65536,np.int64)
    p=np.pad(a,1)
    for neighbor in (p[:-2,1:-1],p[2:,1:-1],p[1:-1,:-2],p[1:-1,2:]):
        choose=(a>0)&(neighbor!=a);perimeter+=np.bincount(a[choose],minlength=65536)
    return areas,perimeter

def _side(ids,interior,band,core,component,region):
    side=(ids==region)&interior&band;total=int(side.sum());trusted=side&(core>0)
    values,counts=np.unique(core[trusted],return_counts=True)
    support=int(trusted.sum())
    if support:
        order=np.lexsort((values,-counts));gid=int(values[order[0]]);main=int(counts[order[0]])
        comp,cn=np.unique(component[trusted&(core==gid)],return_counts=True)
        components={str(int(k)):int(n) for k,n in zip(comp,cn)}
    else:gid=main=0;components={}
    return dict(region_id=int(region),side_band_pixels=total,trusted_support_pixels=support,
        coverage=support/total if total else 0.,dominant_GT_id=gid,dominant_GT_pixels=main,
        purity=main/support if support else 0.,original_component_support=components)

def label_region_pair(instances,candidate,gt,original,filled,unknown,core_map,component_map,*,interiors=None):
    """keep=1/merge=0/ignore=255；严格冻结32/.8/.95，走廊完整原生分母。"""
    ids=_instances(instances);h,w=ids.shape
    values=[gt,original,filled,unknown,core_map,component_map]
    if any(np.asarray(v).shape!=ids.shape for v in values):raise ValueError('Supervision maps must be native aligned')
    interior=predicted_interiors(ids,2) if interiors is None else interiors
    x0,y0,x1,y1=candidate['seam_bbox_xyxy'];x0=max(0,x0-24);y0=max(0,y0-24);x1=min(w,x1+24);y1=min(h,y1+24)
    sl=np.s_[y0:y1,x0:x1];seam=np.zeros((y1-y0,x1-x0),np.uint8)
    yy,xx=np.divmod(candidate['seam_flat'],w);seam[yy-y0,xx-x0]=1
    band=distance_transform_edt(seam==0)<=24
    sides=[_side(ids[sl],interior[sl],band,core_map[sl],component_map[sl],rid) for rid in candidate['pair_ids']]
    failures=[]
    for name,side in zip(('A','B'),sides):
        if side['side_band_pixels']==0:failures.append(name+'_empty_predicted_interior_band')
        if side['trusted_support_pixels']<32:failures.append(name+'_trusted_support_lt32')
        if side['coverage']<.8:failures.append(name+'_trusted_coverage_lt080')
        if side['purity']<.95:failures.append(name+'_GT_purity_lt095')
    result=dict(label=IGNORE,reason=failures[0] if failures else None,side_A=sides[0],side_B=sides[1],failures=failures,
        coverage_denominator='all predicted erode2 interior pixels within Euclidean24 of full shared seam; includes GT unknown',
        purity_denominator='actual trusted original GT core support only',label_scope='original annotated supervision, not physical correctness proof')
    if failures:return result
    if sides[0]['dominant_GT_id']!=sides[1]['dominant_GT_id']:
        result.update(label=KEEP,reason='different_trusted_GT_keep');return result
    gid=sides[0]['dominant_GT_id'];contact=candidate['contact_flat']
    original=np.asarray(original,dtype=bool);filled=np.asarray(filled,dtype=bool);unknown=np.asarray(unknown,dtype=bool)
    actual_good=original.ravel()[contact]&~filled.ravel()[contact]&~unknown.ravel()[contact]&(gt.ravel()[contact]==gid)&(component_map.ravel()[contact]>0)
    comps=np.unique(component_map.ravel()[contact])
    result['actual_cross_contact']=dict(pixels=len(contact),all_original_same_GT=bool(actual_good.all()),
        components=[int(v) for v in comps],zero_seam_max_width=1,all_contact_pixels_retained_in_check=True)
    if not actual_good.all():result['reason']='merge_actual_contact_unknown_filled_or_other_GT';return result
    if len(comps)!=1:result['reason']='merge_actual_contact_multiple_original_components';return result
    cid=int(comps[0]);corridor=cv2.dilate(seam,_disk(2))>0;denominator=int(corridor.sum())
    in_component=int((component_map[sl][corridor]==cid).sum());ratio=in_component/denominator
    result['merge_corridor']=dict(original_component_id=cid,GT_id=gid,pixels=denominator,same_component_pixels=in_component,
        coverage=ratio,minimum_coverage=.95,denominator='entire native shared-seam disk2 corridor, no GT/A/B masking',
        eroded_component_used=False)
    if ratio<.95:result['reason']='merge_full_corridor_component_coverage_lt095';return result
    result.update(label=MERGE,reason='same_original_GT_component_merge');return result

def representative_anchor(candidate,width):
    yy,xx=np.divmod(candidate['seam_flat'],width);cy,cx=np.median(yy),np.median(xx)
    d=(yy-cy)**2+(xx-cx)**2;order=np.lexsort((xx,yy,d));i=int(order[0])
    return int(xx[i]),int(yy[i])

def native_relation_patch(rgb,N1,NAF,instances,candidate,anchor_xy,*,size=256):
    """只裁原生像素；RGB/maps反射，预测A/B/seam及原图valid补0，不输入GT。"""
    ids=_instances(instances);h,w=ids.shape
    if rgb.shape!=(h,w,3) or rgb.dtype!=np.uint8 or N1.shape!=ids.shape or NAF.shape!=ids.shape:raise ValueError('Input feature alignment differs')
    cx,cy=map(int,anchor_xy);x0,y0=cx-size//2,cy-size//2;x1,y1=x0+size,y0+size
    xa,ya,xb,yb=max(0,x0),max(0,y0),min(w,x1),min(h,y1)
    if xa>=xb or ya>=yb:raise ValueError('Patch has no native pixels')
    top,bottom,left,right=ya-y0,y1-yb,xa-x0,x1-xb;sl=np.s_[ya:yb,xa:xb]
    image=cv2.copyMakeBorder(rgb[sl],top,bottom,left,right,cv2.BORDER_REFLECT)
    maps=np.stack([N1[sl],NAF[sl]],axis=-1)
    maps=cv2.copyMakeBorder(maps,top,bottom,left,right,cv2.BORDER_REFLECT).transpose(2,0,1).astype(np.float16)
    masks=np.zeros((4,size,size),np.uint8);dst=np.s_[top:size-bottom if bottom else size,left:size-right if right else size]
    masks[0][dst]=ids[sl]==candidate['pair_ids'][0];masks[1][dst]=ids[sl]==candidate['pair_ids'][1];masks[3][dst]=1
    yy,xx=np.divmod(candidate['seam_flat'],w);selected=(xx>=xa)&(xx<xb)&(yy>=ya)&(yy<yb)
    masks[2,yy[selected]-y0,xx[selected]-x0]=1
    if image.shape!=(size,size,3) or maps.shape!=(2,size,size):raise RuntimeError('Native patch shape contract failed')
    return dict(rgb=np.ascontiguousarray(image.transpose(2,0,1)),maps=np.ascontiguousarray(maps),masks=masks,
        metadata=dict(anchor_native_xy=[cx,cy],native_crop_xyxy=[x0,y0,x1,y1],native_valid_box_xyxy=[xa,ya,xb,yb],
            pad_tblr=[top,bottom,left,right],resize=False,padding='RGB/maps cv2.BORDER_REFLECT; predicted masks/valid zero',
            valid_source='original image extent only; no GT/known/ignore mask used'))
