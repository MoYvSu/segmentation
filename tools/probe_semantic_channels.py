# -*- coding: utf-8 -*-
"""对指定的混合预测实例追溯种子连接阶段；不修改边界或生成候选结果。"""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.mim_dataset import list_images
from models.backend_adaptation import load_backend
from train_backend_adaptation import prediction_maps,write_json
from train_semantic_d5a import get_restorer,sha
from tools.analyze_semantic_split import load_prediction,tile,write_png,prob_rgb
from tools.semantic_marker_diagnostic import marker_stages,connectivity_stages,connectivity_maps
from utils.config import load_config,project_path


@torch.inference_mode()
def run(args):
    config=load_config(args.config)
    out=Path(project_path(config,args.output_dir))
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device=torch.device('cuda')
    entry=config['semantic_split']['models']['lora']
    checkpoint=Path(project_path(config,entry['checkpoint']))
    if sha(checkpoint)!=entry['sha256']:
        raise ValueError('Checkpoint mismatch')
    model,_=load_backend(checkpoint,config,device)
    model.eval().requires_grad_(False)
    restorer=get_restorer(config,device)
    files=[Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))]
    wanted={}
    render_requested=set(args.cases or [])
    if args.records:
        for line in Path(project_path(config,args.records)).read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['models']['lora']['interior_heterogeneous']:
                wanted.setdefault(row['stem'],[]).append(int(row['id']))
    for value in args.cases or []:
        stem,iid=value.rsplit(':',1)
        if int(iid) not in wanted.get(stem,[]):
            wanted.setdefault(stem,[]).append(int(iid))
    if not wanted:
        raise ValueError('Provide diagnostic cases or completed instance records')
    records=[]
    for index,path in enumerate(files):
        if path.stem not in wanted:
            continue
        image,semantic,boundary=prediction_maps(model,path,config,device,restorer,
            config['semantic_adaptation']['restoration']['inference_seed']+index)
        probability=torch.sigmoid(semantic)[0,0].numpy()
        stages=marker_stages(boundary[0,0].numpy(),config)
        maps=connectivity_maps(stages)
        labels,_,_=load_prediction(Path(project_path(config,entry['deployment'])),path.stem)
        bridge=stages['resolved']['bridge_width']
        bridged=(stages['marker_boundary_mask']>0).astype(np.uint8)
        if bridge>0:
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*bridge+1,2*bridge+1))
            bridged=cv2.dilate(bridged,kernel)
        for iid in wanted[path.stem]:
            mask=labels==iid
            if not mask.any():
                raise ValueError('Selected instance missing')
            record=connectivity_stages(stages,mask,probability,maps=maps)
            record.update(stem=path.stem,id=iid,area=int(mask.sum()))
            records.append(record)
            if render_requested and f'{path.stem}:{iid}' not in render_requested:
                continue
            if not render_requested and len(records)>6:
                continue
            yy,xx=np.nonzero(mask)
            margin=24
            sl=(slice(max(0,int(yy.min())-margin),min(mask.shape[0],int(yy.max())+margin+1)),
                slice(max(0,int(xx.min())-margin),min(mask.shape[1],int(xx.max())+margin+1)))
            def panel(rgb,title):
                rgb=rgb[sl].copy()
                for number,key in enumerate(['ferrite','pearlite'],1):
                    point=record['points'][key]
                    if point is None:
                        continue
                    xy=(point['x']-sl[1].start,point['y']-sl[0].start)
                    cv2.circle(rgb,xy,4,(0,0,0),2)
                    cv2.circle(rgb,xy,2,(255,255,255),-1)
                    cv2.putText(rgb,str(number), (xy[0]+5,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1,cv2.LINE_AA)
                return tile(rgb,title,width=400,height=380)
            def binary(array):
                return np.repeat(((array>0)*255).astype(np.uint8)[...,None],3,2)
            pictures=[panel(image,f'{path.stem} ID{iid} | 1:F 2:P predictions'),panel(prob_rgb(probability),'Semantic P(F): orange=F blue=P')]
            for key,array in [('high_mask',stages['high_mask']),('marker_boundary_mask',stages['marker_boundary_mask']),
                              ('bridged_marker_mask',bridged),('marker_belt',stages['marker_belt'])]:
                same=record['stages'][key]['same_component']
                state='same' if same is True else 'separate' if same is False else 'blocked/unknown'
                pictures.append(panel(binary(array),key+' | '+state))
            write_png(out/f'{path.stem}_{iid}.png',np.concatenate([np.concatenate(pictures[:3],1),np.concatenate(pictures[3:],1)],0))
        print(json.dumps({'image':path.stem,'cases_done':len(records)},ensure_ascii=False),flush=True)
    if len(records)!=sum(len(v) for v in wanted.values()):
        raise ValueError('Requested image missing')
    totals={}
    for key in records[0]['stages']:
        totals[key]={name:sum(r['stages'][key]['same_component'] is state for r in records)
                     for name,state in [('connected',True),('separated',False),('blocked_or_missing',None)]}
    keys=list(records[0]['stages'])
    transitions={}
    for a,b in zip(keys,keys[1:]):
        transitions[a+'_to_'+b]=dict(
            reopened=sum(r['stages'][a]['same_component'] is False and r['stages'][b]['same_component'] is True for r in records),
            closed=sum(r['stages'][a]['same_component'] is True and r['stages'][b]['same_component'] is False for r in records))
    report=dict(checkpoint=entry,no_training=True,no_test_labels=True,cases=records,totals=totals,transitions=transitions,
                selected_from=args.records,case_count=len(records),images=len(wanted))
    write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/experiments/semantic_split.yaml')
    parser.add_argument('--output-dir',default='outputs/semantic_channels')
    parser.add_argument('--cases',nargs='+',help='私有诊断对象／渲染对象：图像stem:实例ID')
    parser.add_argument('--records',help='已完成instances.jsonl，仅分析LoRA内部混合对象')
    run(parser.parse_args())
