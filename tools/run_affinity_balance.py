# -*- coding: utf-8 -*-
"""先纯1024再混合，独立同初态训练；复用两套旧控制与最终输出。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_affinity_mixed import SOURCES as MIXED_SOURCES
from train_affinity_balance import verify_reference,verify_run
from train_backend_adaptation import sha,write_json
from utils.config import load_config,project_path

CONFIGS={kind:'config/train/affinity_balance_'+kind+'.yaml' for kind in ('native','mixed')}
SOURCES=tuple(dict.fromkeys((*MIXED_SOURCES,*CONFIGS.values(),'train_affinity_balance.py','tools/run_affinity_balance.py')))


def audit_outputs(output):
    import numpy as np
    import torch
    from data.mim_dataset import list_images
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views
    from tools.analyze_affinity_connectivity import load_directory
    from train_affinity_connectivity import get_restorer
    torch.set_num_threads(4);records=[]
    for kind,path in CONFIGS.items():
        config=load_config(path);verify_reference(config)
        folder=Path(project_path(config,config['affinity_native']['penalty_reference']))
        model,_=load_backend(folder/'final.pt',config,'cuda');restorer=get_restorer(config,'cuda')
        files=[Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))]
        for i,p in enumerate(files):
            if p.stem not in config['affinity_connectivity']['parity_images']:continue
            _,views=predict_views(model,restorer,p,config,'cuda',config['backend_adaptation']['restoration']['inference_seed']+i,[0,1024])
            for view,values in views.items():
                ids,cls=load_directory(folder/'deployment'/view,p.stem)
                if not np.array_equal(ids,values['instances']) or cls!=values['classes']:
                    raise RuntimeError('Control final output changed: '+kind+'/'+p.stem+'/'+view)
                records.append(dict(sampling=kind,source=p.stem,view=view,equal=True))
        del model,restorer;torch.cuda.empty_cache()
    if len(records)!=8:raise RuntimeError('Incomplete control parity')
    write_json(Path(output)/'parity.json',dict(passed=True,rows=records))


def render_final(root):
    from data.mim_dataset import list_images
    from tools.analyze_affinity_connectivity import load_directory,render
    from utils.patch_diagnostic import phase_description
    for kind,path in CONFIGS.items():
        config=load_config(path);folder=root/kind;gallery=folder/'gallery';gallery.mkdir(exist_ok=False)
        control=Path(project_path(config,config['affinity_native']['penalty_reference'],'deployment'))
        folders=[control/'global',control/'patch1024',folder/'deployment/global',folder/'deployment/patch1024']
        records=[];pages=['<!doctype html><meta charset="utf-8"><title>Affinity balance</title>',
            '<p>同采样只改损失，均为e20。金色铁素体／蓝色珠光体；无测试GT。</p>']
        for p in map(Path,list_images(project_path(config,config['inference']['test_dir']))):
            predictions=[load_directory(d,p.stem) for d in folders]
            render(p,predictions,['old whole','old native1024','balance whole','balance native1024'],gallery/(p.stem+'.png'),kind+'/'+p.stem,width=320)
            records.append(dict(source=p.stem,phases=[phase_description(*x) for x in predictions]))
            pages.append(f'<p>{p.stem}<br><img loading="lazy" width="1600" src="gallery/{p.stem}.png"></p>')
        if len(records)!=100:raise RuntimeError('Incomplete gallery')
        write_json(folder/'summary.json',dict(scope='Predictions, not accuracy',images=records))
        (folder/'index.html').write_text('\n'.join(pages),encoding='utf8')


def run(smoke):
    root=ROOT/'outputs'/('affinity_balance_smoke' if smoke else 'affinity_balance')
    configs={k:load_config(v) for k,v in CONFIGS.items()}
    for config in configs.values():verify_reference(config)
    if shutil.disk_usage(ROOT).free<6*1024**3:raise RuntimeError('Insufficient fast-disk space')
    if not smoke:
        gate=ROOT/'outputs/affinity_balance_smoke';prior=json.loads((gate/'pipeline_status.json').read_text())
        if prior['status']!='complete' or prior['configs']!=json.loads(json.dumps(configs)):
            raise RuntimeError('Two-arm smoke must pass with exact configs')
        if not json.loads((gate/'parity.json').read_text())['passed']:raise RuntimeError('Parity gate failed')
        for name in SOURCES:
            if (gate/'source'/name).read_bytes()!=(ROOT/name).read_bytes():raise RuntimeError('Source changed: '+name)
    root.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        dest=root/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    status=dict(status='running',pid=os.getpid(),stage='preflight',configs=configs,smoke=smoke,completed=[],audits={},
        control_retrained=False,automatic_extension=False,competition_submission=False,sources={n:sha(ROOT/n) for n in SOURCES})
    def save():write_json(root/'pipeline_status.json',status)
    def execute(stage,command):
        status['stage']=stage;save()
        with (root/(stage+'.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;save()
            if child.wait():raise RuntimeError(stage+' failed; inspect log')
        status['completed'].append(stage);save()
    save()
    try:
        if smoke:execute('parity',[sys.executable,'-u',__file__,'--audit',str(root)])
        # 两组训练接续完成后统一推理；没有重训控制，也没有继承上一组末态。
        for kind,path in CONFIGS.items():
            execute(kind,[sys.executable,'-u','train_affinity_balance.py','--config',path,'--output',str(root/kind)]+(['--smoke'] if smoke else []))
            status['audits'][kind]=verify_run(configs[kind],root/kind,8 if smoke else 1280);save()
        if len({a['initialization_sha256'] for a in status['audits'].values()})!=1:raise RuntimeError('Different initial weights')
        if not smoke:
            for kind,path in CONFIGS.items():
                execute(kind+'_infer',[sys.executable,'-u','train_affinity_native.py','--config',path,'--mode','infer',
                    '--checkpoint',str(root/kind/'final.pt'),'--output',str(root/kind/'deployment'),'--views','0','1024'])
            status['stage']='render';save();render_final(root)
        status.update(status='complete',stage='done')
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--audit');a=p.parse_args()
    if a.audit:audit_outputs(a.audit)
    else:run(a.smoke)
