# -*- coding: utf-8 -*-
"""固定分割后端，只替换局部恢复器；旧输出复用、全100图及小样本512对照。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.mim_dataset import list_images
from models.backend_adaptation import load_backend
from models.rgb_restoration import load_rgb_restorer
from tools.affinity_native_views import predict_views, verify_prediction
from tools.analyze_affinity_connectivity import load_directory, relation, render
from tools.probe_affinity_patch import load_gt
from train_backend_adaptation import tensor_digest, write_json
from train_restore_native import file_sha
from utils.config import load_config, project_path
from utils.patch_diagnostic import phase_description, trusted_phase_metrics


def checked_assets(config):
    opt=config['restore_native']
    for name in ('old_restorer','backend1024','backend512'):
        path=project_path(config,opt[name])
        if file_sha(path)!=opt[name+'_sha256']:raise RuntimeError(f'Unexpected checkpoint: {name}')
    backend=load_config(project_path(config,opt['backend_config']))
    return backend


def save_prediction(folder,stem,view):
    folder.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(folder/(stem+'_inst.png')),view['instances']):raise IOError(stem)
    write_json(folder/(stem+'_class.json'),view['classes'])


@torch.no_grad()
def parity(config,root):
    """GPU前测：新入口以旧恢复器逐像素复现两个后端的已存最终输出。"""
    backend=checked_assets(config);opt=config['restore_native']
    old=load_rgb_restorer(project_path(config,opt['old_restorer']),'cuda').requires_grad_(False)
    files=[Path(p) for p in list_images(project_path(backend,backend['inference']['test_dir']))]
    rows=[]
    for size in (1024,512):
        model,_=load_backend(project_path(config,opt[f'backend{size}']),backend,'cuda')
        before=tensor_digest(model.state_dict().items())
        for index,path in enumerate(files):
            if path.stem not in config['restore_native']['monitor']['real_typical']:continue
            audit={};seed=backend['backend_adaptation']['restoration']['inference_seed']+index
            _,views=predict_views(model,old,path,backend,'cuda',seed,[size],local_restorer=old,fixed_audit=audit)
            expected,classes=load_directory(project_path(config,opt[f'old_predictions{size}']),path.stem)
            view=views[f'patch{size}']
            if not np.array_equal(view['instances'],expected) or view['classes']!=classes:
                raise RuntimeError(f'Deployed old output differs: {path.stem}/{size}')
            rows.append(dict(source=path.stem,size=size,final_output_equal=True,fixed_maps=audit))
        if tensor_digest(model.state_dict().items())!=before:raise RuntimeError('Backend changed')
        del model;gc.collect();torch.cuda.empty_cache()
    if len(rows)!=4:raise RuntimeError('Incomplete deployment parity')
    report=dict(passed=True,rows=rows,old_local_equals_legacy=True)
    write_json(Path(root)/'deployment_parity.json',report)
    return report


def trusted_core(gt,trust,classes):
    g=gt.astype(np.float32);kernel=np.ones((9,9),np.uint8)
    same=cv2.erode(g,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)==cv2.dilate(g,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    known=cv2.erode(trust.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    ferrite=np.zeros(int(gt.max())+1,bool)
    for key,value in classes.items():ferrite[int(key)]=value==1
    return same&known&(gt>0)&ferrite[gt]


@torch.no_grad()
def evaluate(config,root,epoch=20):
    root=Path(root);out=root/('analysis' if epoch==20 else f'analysis_e{epoch}');out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    opt=config['restore_native'];backend=checked_assets(config)
    fixed=json.loads((root/'fixed_manifest.json').read_text(encoding='utf-8'))
    selected={Path(p['source']).stem for p in fixed['reals']}
    old=load_rgb_restorer(project_path(config,opt['old_restorer']),'cuda').requires_grad_(False)
    restorers={'old':old}
    for arm in ('native','mixed'):
        status=json.loads((root/arm/'status.json').read_text())
        if status['status']!='stage_complete' or status['epoch']!=epoch or status['updates']!=epoch*250:
            raise RuntimeError('Both complete, fixed-endpoint formal models are required')
        restorers[arm]=load_rgb_restorer(root/arm/f'epoch_{epoch:03d}.pt','cuda').requires_grad_(False)
    frozen_r={key:tensor_digest(r.state_dict().items()) for key,r in restorers.items()}
    files=[Path(p) for p in list_images(project_path(backend,backend['inference']['test_dir']))]
    if len(files)!=100:raise RuntimeError('Expected all 100 test images')
    result={'scope':'Unlabeled predicted changes; not accuracy or checkpoint selection','test':[],'gt':[]}
    gallery=out/'gallery';gallery.mkdir()
    for size in (1024,512):
        model,_=load_backend(project_path(config,opt[f'backend{size}']),backend,'cuda')
        frozen=tensor_digest(model.state_dict().items())
        for index,path in enumerate(files):
            if size==512 and path.stem not in selected:continue
            old_ids,old_classes=load_directory(project_path(config,opt[f'old_predictions{size}']),path.stem)
            views={'old':dict(instances=old_ids,classes=old_classes)}
            seed=backend['backend_adaptation']['restoration']['inference_seed']+index
            audits={}
            for arm in ('native','mixed'):
                audit={}
                rgb,predicted=predict_views(model,old,path,backend,'cuda',seed,[size],local_restorer=restorers[arm],fixed_audit=audit)
                view=predicted[f'patch{size}'];views[arm]=view;audits[arm]=audit
                save_prediction(root/arm/'deployment'/f'patch{size}',path.stem,view)
                verify_prediction(old_ids,old_classes,rgb.shape[:2])
            if audits['native']!=audits['mixed']:raise RuntimeError('Whole-image semantic maps drifted')
            row=dict(source=path.stem,size=size,fixed_maps=audits['native'],
                phases={arm:phase_description(v['instances'],v['classes']) for arm,v in views.items()},
                relations={arm:relation(old_ids,v['instances'],old_classes,v['classes']) for arm,v in views.items() if arm!='old'})
            result['test'].append(row)
            render(path,[(v['instances'],v['classes']) for v in views.values()],['Old D5a',f'A e{epoch}',f'B e{epoch}'],
                gallery/f'{path.stem}_{size}.png',f'{path.stem} fixed p{size} only local restorer changes',width=320)
            if path.stem in selected:
                render(path,[(v['instances'],v['classes']) for v in views.values()],['Old D5a',f'A e{epoch}',f'B e{epoch}'],
                    gallery/f'{path.stem}_{size}_detail.png',path.stem,crop=[.25,.25,.75,.75],width=320)
            append_path=out/'test.jsonl'
            with append_path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print('DEPLOY',size,path.stem,flush=True)
        if size==1024:
            prior=Path(project_path(backend,backend['affinity_native']['prior_probe']))
            cases=[c for c in json.loads((prior/'cases.json').read_text()) if c['kind']=='train']
            if len(cases)!=4:raise RuntimeError('Existing four trusted training cases required')
            for case in cases:
                gt,trust,classes=load_gt(case,backend);core=trusted_core(gt,trust,classes)
                entries={};predictions=[];audits=[]
                for arm,r in restorers.items():
                    audit={};_,predicted=predict_views(model,old,case['path'],backend,'cuda',case['seed'],[size],local_restorer=r,fixed_audit=audit)
                    view=predicted[f'patch{size}'];audits.append(audit);predictions.append((view['instances'],view['classes']))
                    save_prediction(out/'trusted'/case['name']/arm,case['name'],view)
                    entries[arm]=dict(metrics=trusted_phase_metrics(gt,classes,trust,view['instances'],view['classes']),
                        trusted_ferrite_core_pixels=int(core.sum()),
                        internal_high_boundary_pixels=int(((view['boundary']>backend['inference']['boundary_threshold'])&core).sum()))
                if not all(a==audits[0] for a in audits):raise RuntimeError('GT semantic maps drifted')
                result['gt'].append(dict(source=case['name'],variants=entries,scope='seen, incomplete GT; unknown excluded'))
                render(case['path'],predictions,['Old D5a',f'A e{epoch}',f'B e{epoch}'],gallery/f'{case["name"]}_trusted.png',case['name'],width=320)
        if tensor_digest(model.state_dict().items())!=frozen:raise RuntimeError('Backend parameters changed')
        del model;gc.collect();torch.cuda.empty_cache()
    for arm,r in restorers.items():
        if tensor_digest(r.state_dict().items())!=frozen_r[arm]:raise RuntimeError('Restorer parameters changed')
    expected=100+len(selected)
    if len(result['test'])!=expected:raise RuntimeError('Incomplete final output set')
    # 三组恢复诊断使用同一配对、同一首步噪声；旧恢复器结果已缓存，不重算。
    result['restoration']={arm:json.loads((root/arm/'monitor'/f'e{epoch:02d}'/'metrics.json').read_text()) for arm in ('native','mixed')}
    result['provenance']=dict(epoch=epoch,local_restorer_only=True,backend1024_sha256=opt['backend1024_sha256'],
        backend512_sha256=opt['backend512_sha256'],old_restorer_sha256=opt['old_restorer_sha256'],
        new_restorer_sha256={a:file_sha(root/a/f'epoch_{epoch:03d}.pt') for a in ('native','mixed')},
        old_predictions_reused=True,all_frozen=True,official_scores=None)
    write_json(out/'report.json',result)
    totals={}
    for size in (1024,512):
        rows=[r for r in result['test'] if r['size']==size]
        totals[str(size)]={'images':len(rows),'variants':{arm:{phase:{key:sum(r['phases'][arm][phase][key] for r in rows)
            for key in ('instances','pixels')} for phase in ('ferrite','pearlite')} for arm in restorers}}
    write_json(out/'summary.json',dict(totals=totals,scope=result['scope'],provenance=result['provenance']))
    html=['<!doctype html><meta charset="utf-8"><title>Restore native comparison</title>',
          '<p>Fixed backend, only local restorer changes. Unlabeled predictions are not accuracy.</p>']
    for image in sorted(gallery.glob('*.png')):html.append(f'<p>{image.name}<br><img loading="lazy" width="960" src="gallery/{image.name}"></p>')
    (out/'index.html').write_text('\n'.join(html),encoding='utf-8')
    return dict(complete=True,main_test_images=100,secondary_test_images=len(selected),trusted_train_images=4)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config/train/restore_native.yaml')
    p.add_argument('--root',required=True);p.add_argument('--epoch',type=int,default=20);p.add_argument('--parity',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    print(json.dumps(parity(cfg,a.root) if a.parity else evaluate(cfg,a.root,a.epoch)),flush=True)
