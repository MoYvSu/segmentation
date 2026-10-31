# -*- coding: utf-8 -*-
"""连接项增权后的同采样配对分析；测试无GT、已见GT仅定位机制。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_affinity_balance import CONFIGS
from train_affinity_balance import verify_run
from tools.analyze_affinity_connectivity import read,rows,write,load_directory,relation,render
from tools.analyze_affinity_native import quantiles
from train_backend_adaptation import sha
from utils.config import load_config,project_path
from utils.patch_diagnostic import phase_description,trusted_phase_metrics
from data.mim_dataset import list_images


def training(root,out):
    state=read(root/'pipeline_status.json');assert state['status']=='complete'
    assert all(sha(ROOT/n)==v==sha(root/'source'/n) for n,v in state['sources'].items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');report={}
    for j,(kind,path) in enumerate(CONFIGS.items()):
        cfg=load_config(path);folder=root/kind;audit=verify_run(cfg,folder,1280)
        status=read(folder/'status.json');epochs=rows(folder/'epochs.jsonl');assert len(epochs)==20
        old=rows(Path(project_path(cfg,cfg['affinity_native']['penalty_reference']))/'steps.jsonl');new=rows(folder/'steps.jsonl')
        expected=[folder/'monitor'/f'epoch_{e:03d}'/(s+'_'+suffix+'.png') for e in (0,1,5,10,15,20)
            for s in cfg['backend_adaptation']['monitor']['images'] for suffix in ('full','detail','boundary')]
        assert len(expected)==108 and all(p.exists() for p in expected)
        report[kind]=dict(audit=audit,checkpoint_sha256=status['final_checkpoint_sha256'],epoch=20,process_images=108,
            frozen={k:status[k] for k in ('frozen_unchanged','restorer_unchanged','strict_reload_equal')},metrics={})
        for view in (['native1024'] if kind=='native' else ['whole','native1024']):
            report[kind]['metrics'][view]={}
            for label,start,stop in [('e1_5',1,5),('e11_15',11,15),('e16_20',16,20)]:
                pairs=[(a,b) for a,b in zip(old,new) if start<=a['epoch']<=stop and a.get('training_view','native1024')==view]
                values={}
                for i,source in enumerate(('manual','sam2')):
                    for metric in ('recall','specificity'):
                        values[source+'_'+metric]={name:float(np.mean([p[k]['supervision'][i][metric] for p in pairs]))
                            for k,name in enumerate(('old','balance'))}
                report[kind]['metrics'][view][label]=dict(updates=len(pairs),**values)
        for k,metric in enumerate(('recall','specificity')):
            for label,data in [('old',old),('balance',new)]:
                axes[j,k].plot(range(1,21),[np.mean([r['supervision'][0][metric] for r in data if r['epoch']==e]) for e in range(1,21)],label=label)
            axes[j,k].set_title(kind+' / manual '+('inside connection recall' if k==0 else 'true separation specificity'))
            axes[j,k].set_xlabel('Epoch');axes[j,k].legend();axes[j,k].grid(alpha=.2)
    fig.suptitle('Same sampling and inputs; training relation metrics, not held-out accuracy')
    fig.savefig(out/'training.png',dpi=140);plt.close(fig);write(out/'training.json',report)
    print('TRAIN',json.dumps(report),flush=True)


def test(root,out):
    cfg=load_config(CONFIGS['native']);files=list(map(Path,list_images(project_path(cfg,cfg['inference']['test_dir']))));assert len(files)==100
    dirs={'best_whole':Path(project_path(cfg,cfg['affinity_native']['reused_control'],'deployment'))}
    for kind,path in CONFIGS.items():
        config=load_config(path);ref=Path(project_path(config,config['affinity_native']['penalty_reference'],'deployment'))
        for suffix,view in [('whole','global'),('local','patch1024')]:
            dirs[kind+'_old_'+suffix]=ref/view;dirs[kind+'_new_'+suffix]=root/kind/'deployment'/view
    pairs=[(kind+'_old_'+view,kind+'_new_'+view) for kind in CONFIGS for view in ('whole','local')]
    fixed=['test_101','test_116'];mon=cfg['backend_adaptation']['monitor']['images']
    random=sorted(np.random.default_rng(20261001).choice([p.stem for p in files if p.stem not in mon],6,replace=False).tolist())
    report=dict(scope='Unlabeled predicted changes, not accuracy',selection=dict(fixed=fixed,random=random,seed=20261001),images=[])
    keys=['native_old_local','native_new_local','mixed_old_local','mixed_new_local']
    labels=['p1024 old','p1024 balance','mix old','mix balance']
    for i,path in enumerate(files):
        image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR);pred={k:load_directory(d,path.stem) for k,d in dirs.items()}
        assert all(ids.shape==image.shape[:2] and set(cls.values())<={0,1} for ids,cls in pred.values())
        base,cls=pred['best_whole'];ar=np.bincount(base.ravel());f=[ar[int(k)] for k,v in cls.items() if v==1];median=float(np.median(f)) if f else None
        row=dict(source=path.stem,shape=list(image.shape[:2]),views={},relations={})
        for key,(ids,cls) in pred.items():
            value=phase_description(ids,cls);ar=np.bincount(ids.ravel());f=np.array([ar[int(k)] for k,v in cls.items() if v==1])
            value['ferrite']['under_quarter_best_median']=int((f<median*.25).sum()) if median else 0;row['views'][key]=value
        for a,b in pairs:row['relations'][a+'__'+b]=relation(pred[a][0],pred[b][0],pred[a][1],pred[b][1])
        report['images'].append(row)
        if path.stem in fixed+random:
            for suffix,crop in [('full',None),('detail',[.25,.25,.75,.75])]:
                render(path,[pred[k] for k in keys],labels,out/(path.stem+'_'+suffix+'.png'),path.stem+' native1024 e20',crop=crop,width=420)
        if i%20==0:print('TEST',i+1,path.stem,flush=True)
    report['totals']={key:{phase:{metric:sum(r['views'][key][phase][metric] for r in report['images'])
        for metric in ('instances','pixels',*(['tiny_under200','under_quarter_best_median'] if phase=='ferrite' else []))}
        for phase in ('ferrite','pearlite')} for key in dirs}
    report['comparisons']={}
    for a,b in pairs:
        key=a+'__'+b;groups={}
        for name,items in [('all',report['images']),('large',[r for r in report['images'] if max(r['shape'])>2048]),('other',[r for r in report['images'] if max(r['shape'])<=2048])]:
            values=[100*(r['views'][b]['ferrite']['mean_area']/r['views'][a]['ferrite']['mean_area']-1) for r in items if r['views'][a]['ferrite']['mean_area'] and r['views'][b]['ferrite']['mean_area']]
            groups[name]=dict(images=len(items),area_percentiles=quantiles(values),area_decrease=sum(v<0 for v in values))
        counts={k:sum(r['relations'][key][k] for r in report['images']) for k in ('split_relations','merge_relations','matched_iou50','matched_iou95','class_changed_matched_iou50','class_changed_pixels','common_covered_pixels')}
        report['comparisons'][key]=dict(groups=groups,**counts)
    write(out/'test.json',report);print('TOTALS',json.dumps(report['totals']),flush=True);print('COMPARISONS',json.dumps(report['comparisons']),flush=True)


def gt(root,out):
    import torch
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views
    from tools.probe_affinity_patch import load_gt
    from train_affinity_connectivity import get_restorer
    torch.set_num_threads(4);cv2.setNumThreads(2);cfg=load_config(CONFIGS['native'])
    prior=Path(project_path(cfg,cfg['affinity_native']['prior_probe']));cases=[c for c in read(prior/'cases.json') if c['kind']=='train'];assert len(cases)==4
    saved={kind:read(ROOT/('outputs/native_analysis/gt.json' if kind=='native' else 'outputs/affinity_mixed/analysis/gt.json')) for kind in CONFIGS}
    assert read(ROOT/'outputs/affinity_mixed/analysis/training.json')['checkpoint_sha256']==load_config(CONFIGS['mixed'])['affinity_native']['penalty_sha256']
    records=[]
    for kind,path in CONFIGS.items():
        config=load_config(path);model,_=load_backend(root/kind/'final.pt',config,'cuda');model.eval().requires_grad_(False);restorer=get_restorer(config,'cuda')
        for case in cases:
            target,trust,classes=load_gt(case,config)
            with torch.no_grad():_,views=predict_views(model,restorer,case['path'],config,'cuda',case['seed'],[0,1024])
            predictions={}
            for view,value in views.items():
                suffix='whole' if view=='global' else 'local'
                old_key=('p1024_global' if view=='global' else 'p1024_native') if kind=='native' else ('mixed_whole' if view=='global' else 'mixed_native')
                directory=ROOT/('outputs/native_analysis/train' if kind=='native' else 'outputs/affinity_mixed/analysis/train')/case['name']/old_key
                old=load_directory(directory,case['name']);new=(value['instances'],value['classes'])
                previous=next(r for r in saved[kind]['rows'] if r['source']==case['name'] and r['view']==old_key)
                if kind=='native':assert previous['checkpoint_sha256']==config['affinity_native']['penalty_sha256']
                for label,pred in [('old',old),('new',new)]:
                    metrics=trusted_phase_metrics(target,classes,trust,*pred)
                    if label=='old':assert metrics==previous['metrics']
                    records.append(dict(source=case['name'],view=kind+'_'+label+'_'+suffix,metrics=metrics))
                folder=out/'train'/case['name']/(kind+'_'+suffix);folder.mkdir(parents=True,exist_ok=True)
                assert cv2.imwrite(str(folder/(case['name']+'_inst.png')),new[0]);write(folder/(case['name']+'_class.json'),new[1])
                if suffix=='local':predictions=dict(old=old,new=new)
            render(case['path'],[(target.astype(np.uint16),{str(k):v for k,v in classes.items()}),*predictions.values()],
                ['GT has omissions',kind+' old',kind+' balance'],out/(case['name']+'_'+kind+'_gt.png'),case['name']+' SEEN training image',crop=[.25,.25,.75,.75],width=420)
            print('GT',kind,case['name'],flush=True)
        del model,restorer;torch.cuda.empty_cache()
    totals={}
    for key in sorted({r['view'] for r in records}):
        totals[key]={}
        for phase in ('ferrite','pearlite'):
            vals=[r['metrics'][phase] for r in records if r['view']==key]
            v={k:sum(r[k] for r in vals) for k in ('gt_instances','pred_instances_in_known_domain','matches','iou_sum','split_gt','merged_pred')}
            v['matched_iou']=v['iou_sum']/max(1,v['matches']);totals[key][phase]=v
    write(out/'gt.json',dict(scope='Four SEEN training sources, original covered domain; not official accuracy',rows=records,totals=totals));print('GT_TOTALS',json.dumps(totals),flush=True)


def index(out):
    parts=['<!doctype html><meta charset="utf-8"><title>Affinity balance e20</title>',
        '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style><h1>假边界惩罚：两种采样 e20</h1>',
        '<p>金色铁素体／蓝色珠光体。主比较均原生1024推理。固定2图＋种子20261001随机6图。无测试GT，训练图仅作机制诊断。</p>',
        '<p><a href="training.json">训练审计</a> | <a href="test.json">100图统计</a> | <a href="gt.json">已见GT</a></p>']
    for p in [out/'training.png',*sorted(out.glob('test*_full.png')),*sorted(out.glob('test*_detail.png')),*sorted(out.glob('train*_gt.png'))]:
        if p.exists():parts.append(f'<p>{p.name}<br><a href="{p.name}"><img loading="lazy" src="{p.name}"></a></p>')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf8')
    with zipfile.ZipFile(out/'report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p.suffix in ('.json','.html','.png'):z.write(p,p.name)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['all','test','gt'],default='all');a=p.parse_args()
    root=ROOT/'outputs/affinity_balance';out=root/'analysis';out.mkdir(parents=True,exist_ok=True)
    if a.mode in ('all','test'):training(root,out);test(root,out)
    if a.mode in ('all','gt'):gt(root,out)
    index(out)
