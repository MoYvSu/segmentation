# -*- coding: utf-8 -*-
"""固定训练配对／无标签测试缩略图；首步输出与结构诊断，不作选优。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from data.mim_dataset import list_images
from data.restore_native import NativeRestorationDataset, crop_native
from data.rgb_restoration_dataset import read_rgb
from models.backend_adaptation import restore_first
from train_rgb_restoration import save_checkpoint, write_json
from utils.config import load_config, project_path


def as_rgb(tensor):
    return tensor.detach().float().cpu().permute(1,2,0).numpy()


def weak_roi(reference, size):
    """按清晰训练参照选择弱梯度位置，仅供放大显示，不产生训练标签。"""
    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(np.rint(gray*255).astype(np.uint8), 20, 40)>0
    gx, gy = [cv2.Sobel(gray, cv2.CV_32F, x,y,ksize=3)/8 for x,y in [(1,0),(0,1)]]
    mag = np.hypot(gx,gy)
    radius = min(size, *gray.shape)//2
    eligible=edge.copy()
    eligible[:radius]=False; eligible[-radius:]=False
    eligible[:,:radius]=False; eligible[:,-radius:]=False
    yy,xx=np.where(eligible)
    if len(yy):
        order=np.argsort(mag[yy,xx],kind='stable'); k=order[len(order)//4]
        cy,cx=int(yy[k]),int(xx[k])
    else:
        cy,cx=gray.shape[0]//2,gray.shape[1]//2
    return [cy-radius,cx-radius,2*radius,2*radius]


def structure_metrics(reference, value, valid):
    """固定清晰参照的边缘与闭合区代理；不是晶粒GT、测试精度或官方指标。"""
    ref=np.asarray(reference,np.float32); value=np.asarray(value,np.float32)
    valid=np.asarray(valid)>0.999
    inner=cv2.erode(valid.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    def edges(a):
        gray=cv2.cvtColor(np.clip(a,0,1),cv2.COLOR_RGB2GRAY)
        grad=np.hypot(cv2.Sobel(gray,cv2.CV_32F,1,0)/8,cv2.Sobel(gray,cv2.CV_32F,0,1)/8)
        return (cv2.Canny(np.rint(gray*255).astype(np.uint8),20,40)>0)&inner,grad
    edge,grad=edges(ref); pred,pgrad=edges(value)
    near=cv2.dilate(edge.astype(np.uint8),np.ones((7,7),np.uint8))>0
    found=cv2.dilate(pred.astype(np.uint8),np.ones((7,7),np.uint8))>0
    weak=edge&(grad<=np.quantile(grad[edge],.5)) if edge.any() else edge
    def closed(mask):
        barrier=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
        count,ids,stats,_=cv2.connectedComponentsWithStats((~barrier&inner).astype(np.uint8),connectivity=8)
        border=cv2.dilate((~inner).astype(np.uint8),np.ones((3,3),np.uint8))>0
        bad=set(np.unique(ids[border]))|set(ids[0])|set(ids[-1])|set(ids[:,0])|set(ids[:,-1])|{0}
        selected=[i for i in range(1,count) if i not in bad and 16<=stats[i,cv2.CC_STAT_AREA]<=4096]
        return ids,stats[:,cv2.CC_STAT_AREA],selected
    rid,areas,small=closed(edge); pid,pareas,_=closed(pred)
    retained=0
    for i in small:
        labels,counts=np.unique(pid[rid==i],return_counts=True)
        good=labels>0
        ious=counts[good]/np.maximum(1,areas[i]+pareas[labels[good]]-counts[good])
        retained+=int(len(ious)>0 and ious.max()>=.5)
    diff=(value-ref)[valid]
    return dict(rgb_l1=float(np.abs(diff).mean()),mse=float((diff**2).mean()),
        gradient_magnitude_l1=float(np.abs(pgrad-grad)[inner].mean()),
        reference_edge_pixels=int(edge.sum()),weak_reference_pixels=int(weak.sum()),
        supported_edge_pixels=int((found&edge).sum()),supported_weak_pixels=int((found&weak).sum()),
        unsupported_edge_pixels=int((pred&~near).sum()),predicted_edge_pixels=int(pred.sum()),
        small_closed_reference_regions=len(small),small_closed_retained=retained,
        scope='clear-image edge/closed-region proxy, not grain labels or segmentation accuracy')


def build_fixed(config, output, *, smoke=False):
    dataset=NativeRestorationDataset(config,'native'); opt=config['restore_native']['monitor']
    indices=np.linspace(0,len(dataset)-1,opt['training_sources'],dtype=int)
    pairs=[]
    # 两种视野各覆盖无退化、均匀、旧空间和弱强端点；训练源并未留出。
    for pos,index in enumerate(indices):
        kind=['identity','uniform','spatial','endpoint'][pos%4]
        for size in (1024,512):
            item=dataset.item(int(index),size=size,kind=kind,downsample=1. if kind=='identity' else .5,
                              local_task=False,augment=False)
            item['name']=f'{Path(item["source"]).stem}_{size}_{kind}'
            item['roi']=weak_roi(as_rgb(item['target']),opt['roi_size'])
            pairs.append(item)
    backend_cfg=load_config(project_path(config,config['restore_native']['backend_config']))
    files=[Path(p) for p in list_images(project_path(backend_cfg,backend_cfg['inference']['test_dir']))]
    if len(files)!=100:
        raise ValueError('Expected independent 100-image semifinal cohort')
    excluded=set(opt['exclude_previous']); candidates=[p.stem for p in files if p.stem not in excluded]
    random_names=sorted(np.random.default_rng(opt['seed']).choice(candidates,opt['real_random_count'],replace=False).tolist())
    names=opt['real_typical']+random_names
    reals=[]
    for path in files:
        if path.stem not in names: continue
        rgb=read_rgb(str(path))
        for size in (1024,512):
            clean,valid,box=crop_native(rgb,size,(.5,.5))
            if size!=1024:
                clean=cv2.resize(clean,(1024,1024),interpolation=cv2.INTER_LINEAR)
            reals.append(dict(input=torch.from_numpy(np.ascontiguousarray(clean.transpose(2,0,1))),
                source=path.name,name=f'{path.stem}_{size}',native_size=size,box=list(box)))
    if smoke:
        pairs=[p for p in pairs if json.loads(p['recipe'])['kind'] in ('uniform','endpoint')]
        reals=reals[:2]
    bundle=dict(pairs=pairs,reals=reals,real_names=names,random_names=random_names,scope='fixed process observation; no holdout or epoch selection')
    save_checkpoint(Path(output)/'fixed.pt',bundle)
    write_json(Path(output)/'fixed_manifest.json',dict(pairs=[dict(name=p['name'],roi=p['roi'],recipe=json.loads(p['recipe'])) for p in pairs],
        reals=[{k:v for k,v in p.items() if k!='input'} for p in reals],random_names=random_names,scope=bundle['scope']))
    return bundle


def save_panel(path, images, labels, title, *, size=224, roi=None):
    def strip(values):
        panels=[]
        for image,label in zip(values,labels):
            thumb=cv2.resize(np.rint(np.clip(image,0,1)*255).astype(np.uint8),(size,size),interpolation=cv2.INTER_AREA)
            thumb=cv2.cvtColor(thumb,cv2.COLOR_RGB2BGR)
            panel=cv2.copyMakeBorder(thumb,26,0,0,0,cv2.BORDER_CONSTANT,value=(245,245,245))
            cv2.putText(panel,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.40,(20,20,20),1,cv2.LINE_AA)
            panels.append(panel)
        return np.concatenate(panels,axis=1)
    rows=[strip(images)]
    if roi is not None:
        y,x,h,w=roi; rows.append(strip([im[y:y+h,x:x+w] for im in images]))
    header=np.full((30,size*len(images),3),245,np.uint8)
    cv2.putText(header,title,(5,21),cv2.FONT_HERSHEY_SIMPLEX,.44,(20,20,20),1,cv2.LINE_AA)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    ok,encoded=cv2.imencode('.png',np.concatenate([header,*rows],axis=0))
    if not ok:raise IOError(path)
    encoded.tofile(str(path))


@torch.no_grad()
def old_reference(old, fixed, output, device, seed):
    cached={'pairs':[],'reals':[]}
    for kind in cached:
        for i,item in enumerate(fixed[kind]):
            tensor=restore_first(old,item['input'][None].to(device),seed+i)
            cached[kind].append(tensor[0].cpu())
    save_checkpoint(Path(output)/'old_fixed.pt',cached)
    return cached


@torch.no_grad()
def monitor(model,fixed,cached,output,epoch,device,options):
    previous=model.training;model.eval();rows=[]
    folder=Path(output)/'monitor'/f'e{epoch:02d}';folder.mkdir(parents=True,exist_ok=True)
    try:
        for kind in ('pairs','reals'):
            for i,item in enumerate(fixed[kind]):
                prediction=restore_first(model,item['input'][None].to(device),options['seed']+i)[0].cpu()
                if not torch.isfinite(prediction).all():raise FloatingPointError('Monitor prediction is not finite')
                inp,pred,old=map(as_rgb,(item['input'],prediction,cached[kind][i]))
                if kind=='pairs':
                    target=as_rgb(item['target']); valid=item['valid'][0].numpy()
                    values={name:structure_metrics(target,value,valid) for name,value in [('input',inp),('old',old),('current',pred)]}
                    edge=cv2.Canny(np.rint(cv2.cvtColor(target,cv2.COLOR_RGB2GRAY)*255).astype(np.uint8),20,40)>0
                    def support(a):
                        e=cv2.Canny(np.rint(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)*255).astype(np.uint8),20,40)
                        return cv2.dilate(e,np.ones((7,7),np.uint8))>0
                    missing=edge&~support(inp)&(valid>0.999)
                    values['gap_proxy']=dict(missing_reference_pixels=int(missing.sum()),
                        recovered_old=int((missing&support(old)).sum()),recovered_current=int((missing&support(pred)).sum()))
                    rows.append(dict(name=item['name'],recipe=json.loads(item['recipe']),metrics=values))
                    save_panel(folder/(item['name']+'.png'),[target,inp,old,pred],
                        ['Clean reference','Degraded','Old D5a e60',f'Current e{epoch}'],
                        f'e{epoch} {item["name"]} | native blur, first x0',size=options['thumbnail_size'],roi=item['roi'])
                else:
                    roi=[384,384,256,256]
                    save_panel(folder/(item['name']+'.png'),[inp,old,pred],['Original','Old D5a e60',f'Current e{epoch}'],
                        f'e{epoch} {item["name"]} | unlabeled observation',size=options['thumbnail_size'],roi=roi)
        result=dict(epoch=epoch,scope='seen training pairs and unlabeled observation only; not validation',pairs=rows)
        write_json(folder/'metrics.json',result)
        return result
    finally:
        model.train(previous)
