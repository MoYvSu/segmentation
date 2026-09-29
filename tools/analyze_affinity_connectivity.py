# -*- coding: utf-8 -*-
"""读取既有100图输出；不推断测试GT，比较训练及最终划分变化。"""
from __future__ import annotations
import argparse,json,sys,zipfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.mim_dataset import list_images
from tools.render_backend_ablation import overlay,text_line,write_png
from tools.semantic_crossover import _read_prediction
from utils.config import load_config,project_path

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def load_directory(directory,stem):
    ids=cv2.imdecode(np.fromfile(Path(directory)/(stem+'_inst.png'),np.uint8),cv2.IMREAD_UNCHANGED)
    classes=read(Path(directory)/(stem+'_class.json'))
    if ids is None or ids.dtype!=np.uint16 or ids.ndim!=2:raise ValueError('Invalid instance PNG')
    if {str(int(i)) for i in np.unique(ids) if i>0}!=set(classes):raise ValueError('Instance/class mismatch')
    return ids,classes

def description(ids,classes):
    areas=np.bincount(ids.ravel());valid=np.flatnonzero(areas);valid=valid[valid>0]
    ferrite=np.array([i for i in valid if classes[str(i)]==1],np.int64)
    return dict(instances=len(valid),ferrite=len(ferrite),pearlite=len(valid)-len(ferrite),
                covered_pixels=int(areas[valid].sum()),tiny_50_199=int(((areas[valid]>=50)&(areas[valid]<200)).sum()),
                ferrite_mean_area=float(areas[ferrite].mean()) if len(ferrite) else None)

def relation(before,after,bc,ac):
    left,left_index=np.unique(before,return_inverse=True);right,right_index=np.unique(after,return_inverse=True)
    counts=np.bincount(left_index.ravel()*len(right)+right_index.ravel(),minlength=len(left)*len(right)).reshape(len(left),len(right))
    la=counts.sum(1);ra=counts.sum(0);li=np.flatnonzero(left>0);ri=np.flatnonzero(right>0)
    common=counts[np.ix_(li,ri)]
    iou=common/np.maximum(1,la[li,None]+ra[None,ri]-common)
    old,new=linear_sum_assignment(-iou)
    overlap=iou[old,new];accepted=overlap>=.5
    # 关系变化门槛仅作可读诊断，不能解释为正确/错误划分。
    split=((common>=50)&(common>=.1*la[li,None])).sum(1)>=2
    merge=((common>=50)&(common>=.1*ra[None,ri])).sum(0)>=2
    class_left=np.array([-1 if value==0 else bc[str(int(value))] for value in left],np.int8)
    class_right=np.array([-1 if value==0 else ac[str(int(value))] for value in right],np.int8)
    paired=class_left[left_index.ravel()];other=class_right[right_index.ravel()];both=(paired>=0)&(other>=0)
    flips=int(((paired!=other)&both).sum())
    cls_changes=sum(bc[str(int(left[li[i]]))]!=ac[str(int(right[ri[j]]))] for i,j,ok in zip(old,new,accepted) if ok)
    return dict(old_instances=len(li),new_instances=len(ri),split_relations=int(split.sum()),merge_relations=int(merge.sum()),
                matched_iou50=int(accepted.sum()),matched_iou95=int((overlap>=.95).sum()),
                mean_matched_iou=float(overlap.mean()) if len(overlap) else None,
                class_changed_matched_iou50=int(cls_changes),class_changed_pixels=flips,
                common_covered_pixels=int(both.sum()),changed_class_fraction=flips/max(1,int(both.sum())))

def render(path,predictions,names,target,title,crop=None,width=430):
    image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(str(path))
    if crop is None:region=np.s_[:,:]
    else:
        h,w=image.shape[:2];x0,y0,x1,y1=crop;region=np.s_[round(y0*h):round(y1*h),round(x0*w):round(x1*w)]
    bodies=[image[region]]
    for ids,classes in predictions:
        bodies.append(overlay(image[region],ids[region],{int(k):v for k,v in classes.items()}))
    panels=[]
    for label,body in zip(['Raw image',*names],bodies):
        small=cv2.resize(body,(width,round(width*body.shape[0]/body.shape[1])),interpolation=cv2.INTER_AREA)
        small=cv2.copyMakeBorder(small,35,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        text_line(small,label,(8,24),width,size=.52);panels.append(small)
    canvas=np.concatenate(panels,axis=1)
    canvas=cv2.copyMakeBorder(canvas,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    text_line(canvas,title+' | F gold / P blue',(8,21),canvas.shape[1],size=.52)
    write_png(target,canvas)

def analyze(run,baseline_zip,output,config_path):
    run,output=Path(run),Path(output);output.mkdir(parents=True,exist_ok=True)
    config=load_config(config_path)
    pipeline=read(run/'pipeline_status.json')
    if pipeline['status']!='complete' or not pipeline['paired_audit']['passed']:raise ValueError('Incomplete pipeline')
    epochs={arm:rows(run/arm/'epochs.jsonl') for arm in ('control','candidate')}
    steps={arm:rows(run/arm/'steps.jsonl') for arm in epochs}
    for arm in epochs:
        state=read(run/arm/'status.json')
        if state['updates']!=1280 or state['epoch']!=20 or len(epochs[arm])!=20 or len(steps[arm])!=1280:raise ValueError('Incomplete training')
    train={}
    for arm in epochs:
        selected=[r for r in steps[arm] if r['connectivity']]
        train[arm]=dict(first=epochs[arm][0],last=epochs[arm][-1],
            last5_manual=float(np.mean([r['manual_bce'] for r in epochs[arm][-5:]])),
            last5_pseudo=float(np.mean([r['pseudo_bce'] for r in epochs[arm][-5:]])),
            auxiliary_steps=len(selected),effective_weights={k:float(v) for k,v in zip(['min','median','max'],np.quantile([r['auxiliary_weight'] for r in selected],[0,.5,1]))},
            frozen_checks={k:read(run/arm/'status.json')[k] for k in ('frozen_unchanged','restorer_unchanged','strict_reload_equal')})
    fig,axes=plt.subplots(1,3,figsize=(14,3.8))
    for arm in epochs:
        x=[r['epoch'] for r in epochs[arm]]
        for ax,key,title in zip(axes,['manual_bce','pseudo_bce','auxiliary'],['Manual affinity BCE','SAM2 affinity BCE','Selected-relation loss (diagnostic)']):
            ax.plot(x,[r[key] for r in epochs[arm]],label=arm);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25)
    axes[0].legend();fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=150);plt.close(fig)
    files=[Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))]
    if len(files)!=100:raise ValueError('Expected all 100 tests')
    fixed=config['backend_adaptation']['monitor']['images']
    others=[p.stem for p in files if p.stem not in fixed]
    selected=set(fixed)|set(np.random.default_rng(20260928).choice(others,4,replace=False).tolist())
    summaries=[]
    with zipfile.ZipFile(baseline_zip) as archive:
        for index,path in enumerate(files):
            a,ac,_=_read_prediction(archive,path.stem);b,bc=load_directory(run/'control/deployment',path.stem);c,cc=load_directory(run/'candidate/deployment',path.stem)
            if a.shape!=b.shape or a.shape!=c.shape:raise ValueError('Shape mismatch')
            summaries.append(dict(image=path.stem,baseline=description(a,ac),control=description(b,bc),candidate=description(c,cc),
                baseline_control=relation(a,b,ac,bc),baseline_candidate=relation(a,c,ac,cc),control_candidate=relation(b,c,bc,cc)))
            if path.stem in selected:
                render(path,[(a,ac),(b,bc),(c,cc)],['Original deployment','BCE control e20','Connectivity e20'],output/(path.stem+'.png'),path.stem)
            if path.stem in ('test_101','test_116'):
                render(path,[(a,ac),(b,bc),(c,cc)],['Original deployment','BCE control e20','Connectivity e20'],output/(path.stem+'_detail.png'),path.stem+' right detail',crop=(.48,.05,.95,.7),width=480)
            if index%20==0:print('analyzed',index+1,path.stem,flush=True)
    totals={}
    for arm in ('baseline','control','candidate'):
        totals[arm]={k:sum(r[arm][k] for r in summaries) for k in ('instances','ferrite','pearlite','tiny_50_199','covered_pixels')}
    comparisons={}
    for key in ('baseline_control','baseline_candidate','control_candidate'):
        left,right=key.split('_')
        area=[100*(r[right]['ferrite_mean_area']/r[left]['ferrite_mean_area']-1) for r in summaries if r[right]['ferrite_mean_area'] and r[left]['ferrite_mean_area']]
        value={k:sum(r[key][k] for r in summaries) for k in ('split_relations','merge_relations','matched_iou50','matched_iou95','class_changed_matched_iou50','class_changed_pixels','common_covered_pixels')}
        value['changed_class_fraction']=value['class_changed_pixels']/value['common_covered_pixels']
        value['ferrite_mean_area_relative_percentiles']=dict(zip(['p10','median','p90'],map(float,np.quantile(area,[.1,.5,.9]))))
        value['images_instance_count_changed']=sum(r[left]['instances']!=r[right]['instances'] for r in summaries)
        comparisons[key]=value
    result=dict(scope='Unlabeled deployment differences, not accuracy or official score',training=train,totals=totals,
                comparisons=comparisons,images=summaries,visualized=sorted(selected),random_seed=20260928)
    write(output/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ('totals','comparisons','visualized')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='config/train/affinity_connectivity.yaml');a=p.parse_args()
    analyze(a.run,a.baseline,a.output,a.config)
