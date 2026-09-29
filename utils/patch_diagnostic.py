# -*- coding: utf-8 -*-
"""原生裁窗的坐标/连续图拼接与分类别诊断；不修改部署后处理。"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def tile_boxes(shape, size, overlap=.25):
    if size<1 or not 0<=overlap<1 or min(shape)<1:
        raise ValueError('Invalid tile geometry')
    def starts(length):
        if length<=size:return [0]
        step=max(1,round(size*(1-overlap)))
        values=list(range(0,length-size+1,step))
        if values[-1]!=length-size:values.append(length-size)
        return values
    h,w=map(int,shape)
    return [(y,x,min(h,y+size),min(w,x+size)) for y in starts(h) for x in starts(w)]


class BlendMap:
    """逐块累积连续概率；所有像素权重为正，重叠处降低窗口边沿影响。"""
    def __init__(self,shape):
        self.total=np.zeros(shape,np.float32);self.square=np.zeros(shape,np.float32)
        self.weight=np.zeros(shape,np.float32);self.count=np.zeros(shape,np.uint16)

    def add(self,box,value):
        y,x,y1,x1=box;value=np.asarray(value,dtype=np.float32)
        if value.shape!=(y1-y,x1-x) or not np.isfinite(value).all():raise ValueError('Invalid patch map')
        h,w=value.shape
        wy=np.maximum(.05,1-np.abs(np.linspace(-1,1,h,dtype=np.float32)))
        wx=np.maximum(.05,1-np.abs(np.linspace(-1,1,w,dtype=np.float32)))
        weights=wy[:,None]*wx[None,:];region=np.s_[y:y1,x:x1]
        self.total[region]+=value*weights;self.square[region]+=value*value*weights
        self.weight[region]+=weights;self.count[region]+=1

    def finish(self):
        if not np.all(self.weight>0):raise ValueError('Uncovered pixels in stitched map')
        mean=self.total/self.weight
        variance=np.maximum(0,self.square/self.weight-mean*mean);overlap=self.count>1
        return mean,dict(tiles_max_overlap=int(self.count.max()),covered_fraction=float((self.count>0).mean()),
            overlap_fraction=float(overlap.mean()),overlap_std_mean=float(np.sqrt(variance[overlap]).mean()) if overlap.any() else 0.)


def phase_description(ids,classes):
    areas=np.bincount(ids.ravel());result={}
    for c,name in [(0,'pearlite'),(1,'ferrite')]:
        selected=[int(k) for k,v in classes.items() if int(v)==c and int(k)<len(areas) and areas[int(k)]>0]
        a=areas[selected];result[name]=dict(instances=len(selected),pixels=int(a.sum()))
        if c==1:result[name].update(mean_area=float(a.mean()) if len(a) else None,tiny_under200=int((a<200).sum()))
    return result


def trusted_phase_metrics(gt,classes,trust,pred,pred_classes):
    """仅已知原标域的匹配诊断。未知区域从双方移除，不报告伪官方面积/总分。"""
    if gt.shape!=pred.shape or trust.shape!=gt.shape:raise ValueError('Shape mismatch')
    keep=trust&(gt>0)
    g,gi=np.unique(gt[keep],return_inverse=True);p,pi=np.unique(pred[keep],return_inverse=True)
    matrix=np.bincount(gi*len(p)+pi,minlength=len(g)*len(p)).reshape(len(g),len(p))
    ga=matrix.sum(1);pa=matrix.sum(0);result={}
    for c,name in [(0,'pearlite'),(1,'ferrite')]:
        ig=np.array([j for j,k in enumerate(g) if int(classes[int(k)])==c],int)
        ip=np.array([j for j,k in enumerate(p) if k>0 and int(pred_classes[str(int(k))])==c],int)
        intersections=matrix[np.ix_(ig,ip)];ious=intersections/np.maximum(1,ga[ig,None]+pa[None,ip]-intersections)
        left,right=linear_sum_assignment(-ious);values=ious[left,right];valid=values>=.5
        total=float(values[valid].sum());matches=int(valid.sum())
        split=(intersections>=np.maximum(50,.1*ga[ig,None])).sum(1)>=2
        merge=(intersections>=np.maximum(50,.1*pa[None,ip])).sum(0)>=2
        result[name]=dict(gt_instances=len(ig),pred_instances_in_known_domain=len(ip),matches=matches,iou_sum=total,
            matched_iou=total/matches if matches else 0.,gt_penalized_iou=total/max(1,len(ig)),
            split_gt=int(split.sum()),merged_pred=int(merge.sum()))
    return result


def reconstruction_metrics(reference,prediction):
    """清晰像素参照的恢复诊断；纹理边缘指标不是GT实例准确率。"""
    ref=np.asarray(reference,np.float32);pred=np.asarray(prediction,np.float32)
    if ref.shape!=pred.shape or ref.ndim!=3:raise ValueError('Expected same RGB shape')
    def gradient(a):
        gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
        return cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
    gx,gy=gradient(ref);px,py=gradient(pred)
    gm=np.hypot(gx,gy);pm=np.hypot(px,py)
    # 阈值仅由相同清晰参照确定，所有候选共用；3像素邻域容忍轻微位移。
    threshold=max(.02,float(np.quantile(gm,.85)))
    edge=gm>=threshold;estimate=pm>=threshold;k=np.ones((7,7),np.uint8)
    near=cv2.dilate(edge.astype(np.uint8),k)>0;found=cv2.dilate(estimate.astype(np.uint8),k)>0
    return dict(rgb_mse=float(np.mean((pred-ref)**2)),gradient_l1=float(np.mean(np.abs(gx-px)+np.abs(gy-py))),
        reference_edge_threshold=threshold,edge_support_recall=float(found[edge].mean()) if edge.any() else None,
        novel_edge_fraction=float((estimate&~near).sum()/max(1,estimate.sum())),
        edge_band_area_ratio=float((estimate&near).sum()/max(1,edge.sum())))
