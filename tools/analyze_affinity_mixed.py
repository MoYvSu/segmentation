# -*- coding: utf-8 -*-
"""混合视野完成审计、同输入训练统计、全测试变化与已见GT机制诊断。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from tools.analyze_affinity_connectivity import load_directory, read, relation, render, rows, write
from tools.analyze_affinity_native import quantiles
from tools.run_affinity_mixed import verify_mixed, check_references
from train_backend_adaptation import sha
from utils.config import load_config, project_path
from utils.patch_diagnostic import phase_description, trusted_phase_metrics


def directories(config, root):
    opt=config['affinity_native']
    return dict(control=Path(project_path(config,opt['reused_control'],'deployment')),
        native=Path(project_path(config,opt['reused_native'],'deployment','patch1024')),
        mixed_whole=root/'candidate/deployment/global', mixed_native=root/'candidate/deployment/patch1024')


PAIRS=[('control','mixed_whole'),('native','mixed_native'),('control','mixed_native')]


def training(config, root, output):
    pipeline=read(root/'pipeline_status.json');state=read(root/'candidate/status.json')
    assert pipeline['status']=='complete'
    audit=verify_mixed(config,root/'candidate',1280)
    check_references(config)
    assert all(sha(ROOT/name)==digest==sha(root/'source'/name) for name,digest in pipeline['sources'].items())
    assert sha(root/'candidate/final.pt')==state['final_checkpoint_sha256']
    expected=[root/'candidate/monitor'/f'epoch_{e:03d}'/(name+'_'+suffix+'.png')
        for e in (0,1,5,10,15,20) for name in config['backend_adaptation']['monitor']['images']
        for suffix in ('full','detail','boundary')]
    assert len(expected)==108 and all(p.is_file() for p in expected)
    current=rows(root/'candidate/steps.jsonl');epochs=rows(root/'candidate/epochs.jsonl')
    assert len(epochs)==20 and epochs[-1]['updates']==1280
    control=rows(Path(project_path(config,config['affinity_native']['reused_control']))/'steps.jsonl')
    native=rows(Path(project_path(config,config['affinity_native']['reused_native']))/'steps.jsonl')
    result=dict(audit=audit,checkpoint_sha256=state['final_checkpoint_sha256'],
        initialization_sha256=state['initial_state_sha256'],source_sha256=pipeline['sources'],
        frozen_checks={k:state[k] for k in ('frozen_unchanged','restorer_unchanged','strict_reload_equal')},
        process_images=len(expected),first=epochs[0],last=epochs[-1],paired_views={})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for vi,view in enumerate(('whole','native1024')):
        reference=control if view=='whole' else native
        selected=[(a,b) for a,b in zip(current,reference) if a['training_view']==view]
        if view=='native1024':
            assert all(a[k]==b[k] for a,b in selected for k in ('manual_crop','pseudo_crop'))
        result['paired_views'][view]={}
        for label,lo,hi in [('e1_5',1,5),('e11_15',11,15),('e16_20',16,20)]:
            segment=[(a,b) for a,b in selected if lo<=a['epoch']<=hi]
            stats={}
            for key in ('manual_bce','pseudo_bce'):
                a=float(np.mean([a[key] for a,b in segment]));b=float(np.mean([b[key] for a,b in segment]))
                stats[key]=dict(mixed=a,reference=b,relative=a/b-1)
            for i,source in enumerate(('manual','pseudo')):
                for metric in ('recall','specificity'):
                    stats[source+'_'+metric]={name:(float(np.mean([pair[j]['supervision'][i][metric] for pair in segment]))
                        if all('supervision' in pair[j] for pair in segment) else None)
                        for j,name in enumerate(('mixed','reference'))}
            result['paired_views'][view][label]=dict(updates=len(segment),**stats)
        for ki,key in enumerate(('manual_bce','pseudo_bce')):
            ax=axes[vi,ki]
            for j,name in enumerate(('mixed','same-view control')):
                values=[float(np.mean([pair[j][key] for pair in selected if pair[0]['epoch']==e])) for e in range(1,21)]
                ax.plot(range(1,21),values,label=name)
            ax.set_title(view+' / '+key);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same source, augmentation and view; training fit, not held-out accuracy')
    fig.savefig(output/'training.png',dpi=140);plt.close(fig)
    write(output/'training.json',result)
    print('TRAIN',json.dumps(result['paired_views']),flush=True)


def test(config, root, output):
    files=[Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))]
    assert len(files)==100
    folders=directories(config,root)
    fixed=config['backend_adaptation']['monitor']['images']
    random=sorted(np.random.default_rng(20261001).choice([p.stem for p in files if p.stem not in fixed],6,replace=False).tolist())
    chosen=set(random)|{'test_101','test_116'}
    report=dict(scope='Unlabeled output changes, not accuracy; no test GT',selection=dict(random=random,seed=20261001,fixed=['test_101','test_116']),images=[])
    for index,path in enumerate(files):
        image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
        pred={k:load_directory(p,path.stem) for k,p in folders.items()}
        assert all(ids.shape==image.shape[:2] and set(classes.values())<={0,1} for ids,classes in pred.values())
        base,classes=pred['control'];area=np.bincount(base.ravel());fa=[area[int(k)] for k,c in classes.items() if c==1]
        median=float(np.median(fa)) if fa else None
        record=dict(source=path.stem,shape=list(image.shape[:2]),views={},relations={})
        for k,(ids,cls) in pred.items():
            stat=phase_description(ids,cls);ar=np.bincount(ids.ravel());f=np.array([ar[int(i)] for i,c in cls.items() if c==1])
            stat['ferrite']['under_quarter_control_median']=int((f<median*.25).sum()) if median else 0
            record['views'][k]=stat
        for left,right in PAIRS:
            a,ac=pred[left];b,bc=pred[right]
            record['relations'][left+'__'+right]=relation(a,b,ac,bc)
        report['images'].append(record)
        if path.stem in chosen:
            for suffix,crop in [('full',None),('detail',[.25,.25,.75,.75])]:
                render(path,list(pred.values()),['control whole','p1024 native','mixed whole','mixed native'],
                    output/(path.stem+'_'+suffix+'.png'),path.stem,crop=crop,width=360 if crop is None else 480)
        if index%20==0:print('TEST',index+1,path.stem,flush=True)
    report['totals']={k:{phase:{metric:sum(r['views'][k][phase][metric] for r in report['images'])
        for metric in ('instances','pixels',*(['tiny_under200','under_quarter_control_median'] if phase=='ferrite' else []))}
        for phase in ('ferrite','pearlite')} for k in folders}
    report['comparisons']={}
    for left,right in PAIRS:
        key=left+'__'+right;by_group={}
        for group,items in [('all',report['images']),('large',[r for r in report['images'] if max(r['shape'])>2048]),('other',[r for r in report['images'] if max(r['shape'])<=2048])]:
            values=[100*(r['views'][right]['ferrite']['mean_area']/r['views'][left]['ferrite']['mean_area']-1)
                for r in items if r['views'][right]['ferrite']['mean_area'] and r['views'][left]['ferrite']['mean_area']]
            by_group[group]=dict(images=len(items),area_percentiles=quantiles(values),area_decrease=sum(v<0 for v in values))
        sums={metric:sum(r['relations'][key][metric] for r in report['images']) for metric in
            ('old_instances','new_instances','split_relations','merge_relations','matched_iou50','matched_iou95',
             'class_changed_matched_iou50','class_changed_pixels','common_covered_pixels')}
        report['comparisons'][key]=dict(**sums,groups=by_group)
    extremes={}
    for left,right in PAIRS[:2]:
        ordered=sorted([r for r in report['images'] if r['views'][left]['ferrite']['mean_area'] and r['views'][right]['ferrite']['mean_area']],
            key=lambda r:r['views'][right]['ferrite']['mean_area']/r['views'][left]['ferrite']['mean_area'])
        for side,row in [('min',ordered[0]),('max',ordered[-1])]:extremes.setdefault(row['source'],[]).append(left+'__'+right+'_'+side)
    report['extremes']=extremes
    for path in files:
        if path.stem in extremes:
            pred=[load_directory(p,path.stem) for p in folders.values()]
            render(path,pred,['control whole','p1024 native','mixed whole','mixed native'],
                output/(path.stem+'_extreme.png'),path.stem+' area-change extreme',width=420)
    write(output/'test.json',report)
    print('TOTALS',json.dumps(report['totals']),flush=True)
    print('COMPARISONS',json.dumps(report['comparisons']),flush=True)


def gt(config, root, output):
    import torch
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views
    from tools.probe_affinity_patch import load_gt
    from train_affinity_connectivity import get_restorer
    torch.set_num_threads(4);cv2.setNumThreads(2)
    prior=Path(project_path(config,config['affinity_native']['prior_probe']))
    old=Path(project_path(config,'outputs/native_analysis'))
    cases=[c for c in read(prior/'cases.json') if c['kind']=='train'];assert len(cases)==4
    old_report=read(old/'gt.json')
    expected=config['affinity_native']['native_sha256']
    assert all(r['checkpoint_sha256']==expected for r in old_report['rows'] if r['view']=='p1024_native')
    model,_=load_backend(root/'candidate/final.pt',config,'cuda');model.eval().requires_grad_(False)
    restorer=get_restorer(config,'cuda');records=[]
    for case in cases:
        target,trust,gc=load_gt(case,config)
        pred=dict(control=load_directory(prior/case['name']/'global_d5a',case['name']),
            native=load_directory(old/'train'/case['name']/'p1024_native',case['name']))
        with torch.no_grad():
            _,views=predict_views(model,restorer,case['path'],config,'cuda',case['seed'],[0,1024])
        for view,values in views.items():
            key='mixed_whole' if view=='global' else 'mixed_native'
            ids,cls=values['instances'],values['classes'];pred[key]=(ids,cls)
            folder=output/'train'/case['name']/key;folder.mkdir(parents=True,exist_ok=True)
            assert cv2.imwrite(str(folder/(case['name']+'_inst.png')),ids)
            write(folder/(case['name']+'_class.json'),cls)
        for key,(ids,cls) in pred.items():
            metrics=trusted_phase_metrics(target,gc,trust,ids,cls)
            if key=='native':
                saved=next(r['metrics'] for r in old_report['rows'] if r['source']==case['name'] and r['view']=='p1024_native')
                assert metrics==saved,'Prior GT diagnostic changed'
            records.append(dict(source=case['name'],view=key,metrics=metrics))
        render(case['path'],[(target.astype(np.uint16),{str(k):v for k,v in gc.items()}),*pred.values()],
            ['GT has omissions','control whole','p1024 native','mixed whole','mixed native'],
            output/(case['name']+'_gt.png'),case['name']+' SEEN training image',crop=[.25,.25,.75,.75],width=380)
        print('GT',case['name'],flush=True)
    totals={}
    for key in pred:
        totals[key]={}
        for phase in ('ferrite','pearlite'):
            values=[r['metrics'][phase] for r in records if r['view']==key]
            total={k:sum(v[k] for v in values) for k in ('gt_instances','pred_instances_in_known_domain','matches','iou_sum','split_gt','merged_pred')}
            total['matched_iou']=total['iou_sum']/max(1,total['matches']);totals[key][phase]=total
    write(output/'gt.json',dict(scope='Four seen training sources, original covered domain only; not held-out or official accuracy',rows=records,totals=totals))
    print('GT_TOTALS',json.dumps(totals),flush=True)


def index(output):
    lines=['<!doctype html><meta charset="utf-8"><title>Mixed affinity analysis</title>',
        '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}section{margin:25px 0}</style>',
        '<h1>全图 / 原生1024混合训练 e20</h1><p>金色铁素体，蓝色珠光体。测试无GT；训练图诊断不是泛化精度。固定2图 + 预选随机6图；极端变化单列。</p>',
        '<p><a href="test.json">100图统计</a> | <a href="training.json">训练审计</a> | <a href="gt.json">已见GT诊断</a></p>']
    for p in [output/'training.png',*sorted(output.glob('test*_full.png')),*sorted(output.glob('test*_detail.png')),*sorted(output.glob('*_extreme.png')),*sorted(output.glob('train*_gt.png'))]:
        if p.exists():lines.append(f'<section><p>{p.name}</p><a href="{p.name}"><img loading="lazy" src="{p.name}"></a></section>')
    (output/'index.html').write_text('\n'.join(lines),encoding='utf8')
    with zipfile.ZipFile(output/'report.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in output.iterdir():
            if p.is_file() and p.suffix in ('.png','.json','.html'):archive.write(p,p.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['test','gt','all'],default='all')
    args=parser.parse_args();config=load_config('config/train/affinity_mixed.yaml')
    root=Path(project_path(config,config['backend_adaptation']['output_dir']))
    output=root/'analysis';output.mkdir(parents=True,exist_ok=True)
    if args.mode in ('test','all'):training(config,root,output);test(config,root,output)
    if args.mode in ('gt','all'):gt(config,root,output)
    index(output)
