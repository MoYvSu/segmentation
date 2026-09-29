# -*- coding: utf-8 -*-
"""固定面积档抽取变化区域；只目检及预筛，不过滤、不制造测试标签。"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.mim_dataset import list_images
from models.backend_adaptation import load_backend
from tools.render_backend_ablation import overlay,text_line,write_png
from tools.run_affinity_preserve import decode_pair
from train_affinity_connectivity import get_restorer
from train_backend_adaptation import prediction_maps,sha,tensor_digest,write_json
from utils.affinity_deployment import crop_letterbox_output,prepare_image
from utils.config import load_config,project_path


def read(z,stem):
    ids=cv2.imdecode(np.frombuffer(z.read(stem+'_inst.png'),np.uint8),cv2.IMREAD_UNCHANGED)
    classes=json.loads(z.read(stem+'_class.json'))
    return ids,classes


def shortlist(control,candidate):
    bins=defaultdict(list);previews=[]
    with zipfile.ZipFile(control) as za,zipfile.ZipFile(candidate) as zb:
        names=sorted(n[:-9] for n in za.namelist() if n.endswith('_inst.png'))
        assert len(names)==100
        for stem in names:
            a,ac=read(za,stem);b,bc=read(zb,stem)
            aa=np.bincount(a.ravel());ab=np.bincount(b.ravel())
            af=np.array([int(k) for k,v in ac.items() if v==1]);bf=np.array([int(k) for k,v in bc.items() if v==1])
            median=float(np.median(aa[af]))
            overlap=np.bincount((a.astype(np.int64)*len(ab)+b).ravel(),minlength=len(aa)*len(ab)).reshape(len(aa),len(ab))
            inter=overlap[np.ix_(af,bf)];iou=inter/(aa[af,None]+ab[None,bf]-inter).clip(1)
            ii,jj=linear_sum_assignment(-iou);matched=set(bf[jj[iou[ii,jj]>=.5]])
            q1,q3=np.quantile(np.log(ab[1:][ab[1:]>0]),[.25,.75]);threshold=int(np.floor(np.exp(q1-1.5*(q3-q1))))
            small=[int(i) for i in np.where((ab<threshold)&(ab>0))[0] if i]
            previews.append(dict(source=stem,threshold_final_map=threshold,small_all=len(small),
                small_ferrite=sum(bc[str(i)]==1 for i in small)))
            for iid in bf:
                if iid in matched:continue
                ratio=ab[iid]/median
                band='lt10' if ratio<.1 else '10to25' if ratio<.25 else '25to100' if ratio<1 else 'ge100'
                parent=int(overlap[1:,iid].argmax()+1)
                bins[band].append(dict(source=stem,id=int(iid),parent=parent,area=int(ab[iid]),ratio=float(ratio),
                    parent_area=int(aa[parent]),overlap_fraction=float(overlap[parent,iid]/ab[iid]),band=band))
    rng=np.random.default_rng(20260928);selected=[]
    for band in ['lt10','10to25','25to100','ge100']:
        pool=bins[band]
        for index in rng.choice(len(pool),min(2,len(pool)),replace=False):selected.append(pool[int(index)])
    return selected,previews,{k:len(v) for k,v in bins.items()}


def figure(path,rgb,boundary,a,b,ac,bc,item):
    affected=(a==item['parent'])|(b==item['id'])
    yy,xx=np.nonzero(affected);y0=max(0,int(yy.min())-24);y1=min(a.shape[0],int(yy.max())+25)
    x0=max(0,int(xx.min())-24);x1=min(a.shape[1],int(xx.max())+25);crop=np.s_[y0:y1,x0:x1]
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    bodies=[bgr,cv2.applyColorMap(np.rint(boundary*255).astype(np.uint8),cv2.COLORMAP_INFERNO),
            overlay(bgr,a,{int(k):v for k,v in ac.items()}),overlay(bgr,b,{int(k):v for k,v in bc.items()})]
    labels=['Original, full parent bounds','Affinity response 0..1','Control','Restored']
    def row(region, prefix):
        panels=[]
        for body,label in zip(bodies,labels):
            body=body[region];scale=min(420/body.shape[1],460/body.shape[0]);w,h=round(body.shape[1]*scale),round(body.shape[0]*scale)
            canvas=np.full((h+30,420,3),245,np.uint8);canvas[30:30+h,:w]=cv2.resize(body,(w,h))
            text_line(canvas,prefix+label,(6,21),420,size=.44);panels.append(canvas)
        return np.concatenate(panels,1)
    edge=(b==item['id']);cy,cx=np.nonzero(edge)
    margin=min(int(cy.min()),int(cx.min()),a.shape[0]-1-int(cy.max()),a.shape[1]-1-int(cx.max()))
    dy0=max(0,int(cy.min())-32);dy1=min(a.shape[0],int(cy.max())+33)
    dx0=max(0,int(cx.min())-32);dx1=min(a.shape[1],int(cx.max())+33)
    top=cv2.resize(bgr,(210,round(210*a.shape[0]/a.shape[1])))
    cv2.rectangle(top,(round(x0*210/a.shape[1]),round(y0*210/a.shape[1])),(round(x1*210/a.shape[1]),round(y1*210/a.shape[1])),(0,0,255),2)
    header=np.full((max(90,top.shape[0]),1680,3),255,np.uint8);header[:top.shape[0],:210]=top
    text_line(header,f"{item['source']} {item['band']} area={item['area']} ratio={item['ratio']:.3f} border_distance={margin}",(230,35),1400,size=.7)
    text_line(header,'Upper: whole parent. Lower: changed region detail. No correctness label.',(230,75),1400,size=.55)
    write_png(path,np.concatenate([header,row(crop,'Parent: '),row(np.s_[dy0:dy1,dx0:dx1],'Detail: ')],0))
    return dict(bounds=[y0,y1,x0,x1],detail_bounds=[dy0,dy1,dx0,dx1],distance_to_frame=margin,within_8px_frame=margin<=8)


@torch.no_grad()
def run(args):
    cfg=load_config(args.config);out=Path(project_path(cfg,args.output));out.mkdir(parents=True,exist_ok=False)
    paths=[Path(project_path(cfg,p)) for p in [args.control,args.candidate]]
    selected,previews,pools=shortlist(*paths)
    write_json(out/'selection.json',dict(seed=20260928,items=selected,pool_sizes=pools,selection='two per fixed relative area band, same-class unmatched IoU0.5; not false positives'))
    write_json(out/'filter_preview.json',dict(scope='all100 final maps only, not exact first-watershed filter execution',rows=previews))
    torch.set_num_threads(4);cv2.setNumThreads(2)
    checkpoint=project_path(cfg,cfg['affinity_preserve']['checkpoint']);assert sha(checkpoint)==cfg['affinity_preserve']['checkpoint_sha256']
    model,_=load_backend(checkpoint,cfg,'cuda');model.eval().requires_grad_(False);restorer=get_restorer(cfg,'cuda')
    states=[tensor_digest(v.state_dict().items()) for v in [model,restorer]]
    files=list_images(project_path(cfg,cfg['inference']['test_dir']));groups=defaultdict(list)
    for item in selected:groups[item['source']].append(item)
    audits=[]
    with zipfile.ZipFile(paths[0]) as za,zipfile.ZipFile(paths[1]) as zb:
        for index,file in enumerate(files):
            path=Path(file)
            if path.stem not in groups:continue
            rgb,sem,boundary=prediction_maps(model,path,cfg,'cuda',restorer,cfg['backend_adaptation']['restoration']['inference_seed']+index)
            _,raw,ph,pw=prepare_image(path,1024,'cuda');raw_sem=model.semantic_decoder(model.encoder(raw.float()),raw.float())
            probability=crop_letterbox_output(raw_sem.float(),1024,ph,pw,rgb.shape[:2]).cpu().sigmoid()[0,0].numpy()
            original=cv2.watershed;observed=[]
            def capture(image,markers):
                result=original(image,markers);area=np.bincount(result[result>0]);positive=area[1:][area[1:]>0]
                q1,q3=np.quantile(np.log(positive),[.25,.75]);threshold=int(np.floor(np.exp(q1-1.5*(q3-q1))))
                observed.append(dict(threshold=threshold,regions=int(len(positive)),small=int((positive<threshold).sum())))
                return result
            with patch('utils.post_process.cv2.watershed',side_effect=capture):
                a,b,ac,bc,audit,_,_=decode_pair(sem,boundary,probability,cfg)
            ea,eac=read(za,path.stem);eb,ebc=read(zb,path.stem)
            assert np.array_equal(a,ea) and np.array_equal(b,eb) and ac==eac and bc==ebc
            for item in groups[path.stem]:item.update(figure(out/(path.stem+'_'+str(item['id'])+'.png'),rgb,boundary[0,0].cpu().numpy(),a,b,ac,bc,item))
            audits.append(dict(source=path.stem,exact_outputs=True,watersheds=observed))
            print(path.stem,observed,flush=True)
    assert states==[tensor_digest(v.state_dict().items()) for v in [model,restorer]]
    write_json(out/'report.json',dict(complete=True,scope='unlabelled mechanism and fixed-rule coverage; no filter, no test GT',
        package_sha256=[sha(p) for p in paths],selected=selected,first_watershed_audits=audits,frozen_exact=True,
        final_map_preview=dict(images=100,images_with_small=sum(r['small_all']>0 for r in previews),
            small_all=sum(r['small_all'] for r in previews),small_ferrite=sum(r['small_ferrite'] for r in previews))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/experiments/affinity_preserve.yaml')
    p.add_argument('--output',default='outputs/fragment_probe')
    p.add_argument('--control',default='outputs/affinity_connectivity/affinity_control.zip')
    p.add_argument('--candidate',default='outputs/affinity_preserve/affinity_preserve.zip')
    run(p.parse_args())
