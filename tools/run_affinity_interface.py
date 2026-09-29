# -*- coding: utf-8 -*-
"""连续界面试验：只训练候选，复用旧control，锁定检查和smoke源码。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_affinity_connectivity import SOURCES as BASE_SOURCES
from train_backend_adaptation import sha,write_json
from utils.config import load_config,project_path

SOURCES=tuple(dict.fromkeys((*BASE_SOURCES,'utils/affinity_interface.py','config/train/affinity_interface.yaml',
    'tools/run_affinity_interface.py','tools/probe_affinity_interface.py')))


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def run(args):
    config=load_config(args.config)
    root=Path(project_path(config,config['backend_adaptation']['output_dir']+('_smoke' if args.smoke else '')))
    if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('Need 3 GiB free')
    if not args.smoke:
        gate=read(project_path(config,args.gate))
        if gate.get('passed') is not True:raise RuntimeError('Mechanism review has not passed')
        for file,digest in gate['locked_sources'].items():
            if sha(ROOT/file)!=digest:raise RuntimeError('Reviewed source changed: '+file)
        for file,digest in gate['reports'].items():
            if sha(project_path(config,file))!=digest:raise RuntimeError('Reviewed diagnostic report changed')
        smoke=Path(str(root)+'_smoke');prior=read(smoke/'pipeline_status.json')
        if prior['status']!='complete' or prior['config']!=json.loads(json.dumps(config)):
            raise RuntimeError('Same configuration must pass smoke')
        for name in SOURCES:
            if (smoke/'source'/name).read_bytes()!=(ROOT/name).read_bytes():
                raise RuntimeError('Source changed after smoke: '+name)
    root.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    status=dict(status='running',stage='candidate',config=config,smoke=args.smoke,control_retrained=False,
        automatic_extension=False,competition_submission=False)
    def execute(stage,extra):
        status['stage']=stage;write_json(root/'pipeline_status.json',status)
        with (root/(stage+'.log')).open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-u','train_affinity_connectivity.py','--config',args.config,
                '--arm','candidate',*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        execute('candidate',['--output',str(root/'candidate')]+(['--smoke'] if args.smoke else []))
        result=read(root/'candidate/status.json')
        expected=config['affinity_connectivity']['smoke_updates'] if args.smoke else 1280
        for key in ['frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal','reused_control_verified','both_directions_active']:
            if result.get(key) is not True:raise RuntimeError('Candidate audit failed: '+key)
        if result['paired_control_updates']!=expected or result['updates']!=expected:
            raise RuntimeError('Incomplete paired updates')
        status['paired_control_updates']=expected
        if not args.smoke:
            execute('inference',['--infer','--checkpoint',str(root/'candidate/final.pt'),'--output',str(root/'candidate/deployment')])
        status.update(status='complete',stage='done');write_json(root/'pipeline_status.json',status)
    except Exception as error:
        status.update(status='failed',error=repr(error));write_json(root/'pipeline_status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='config/train/affinity_interface.yaml')
    p.add_argument('--gate',default='outputs/interface_gate.json');p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
