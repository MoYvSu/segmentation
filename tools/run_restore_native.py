# -*- coding: utf-8 -*-
"""恢复器尺度实验顺序队列：短测→共享初态→A/B→固定后端全量检查。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.restore_native import NativeRestorationDataset
from models.rgb_restoration import load_rgb_restorer
from tools.restore_native_monitor import build_fixed,old_reference
from tools.evaluate_restore_native import checked_assets,parity
from train_restore_native import file_sha,make_initial,recipe_signature
from train_rgb_restoration import write_json
from utils.config import load_config,project_path

SOURCES=['config/train/restore_native.yaml','data/restore_native.py','train_restore_native.py',
    'tools/restore_native_monitor.py','tools/evaluate_restore_native.py','tools/run_restore_native.py',
    'tools/affinity_native_views.py','data/rgb_restoration_dataset.py','data/rgb_spatial_blur.py',
    'models/rgb_diffusion_restoration.py','models/rgb_restoration.py','models/backend_adaptation.py',
    'train_rgb_diffusion_restoration.py','train_rgb_restoration.py','utils/config.py']


def versions():
    return {name:file_sha(ROOT/name) for name in SOURCES}


COMMON_RECIPE=('source','source_index','epoch','profile','kind','base_sigma_native','crop_fractions',
               'turns','flip','downsample_scale','local_task','local_task_strength')


def audit_inputs(config,root):
    sets=[NativeRestorationDataset(config,arm) for arm in ('native','mixed')]
    if sets[0].samples!=sets[1].samples or len(sets[0])!=1000:raise RuntimeError('Training sources differ')
    if any('autodl-fs' in str(Path(p).resolve()) for p in sets[0].samples):
        raise RuntimeError('Use the existing fast-disk training copy')
    rows=[];same1024=0
    counts={name:int((sets[1].view_sizes==name).sum()) for name in (1024,512)}
    if counts!={1024:500,512:500}:raise RuntimeError('Mixed view distribution must be exactly balanced')
    for index in np.linspace(0,999,32,dtype=int):
        a,b=[ds[int(index)] for ds in sets]
        ra,rb=[json.loads(item['recipe']) for item in (a,b)]
        if any(ra[k]!=rb[k] for k in COMMON_RECIPE):raise RuntimeError('Unpaired degradation/geometry RNG')
        if rb['native_size']==1024:
            if not all(torch.equal(a[k],b[k]) for k in ('input','target','valid')):raise RuntimeError('Same-view pairs differ')
            same1024+=1
        for recipe in (ra,rb):
            if recipe['sigma_network_max']!=recipe['sigma_native_max']*recipe['zoom']:
                raise RuntimeError('Blur scale mapping differs')
            if recipe['kind']=='endpoint' and min(recipe['weak_fraction'],recipe['strong_fraction'])<.05:
                raise RuntimeError('Endpoint blur does not contain both weak and strong regions')
        rows.append(dict(index=int(index),native=ra,mixed=rb))
    if same1024==0:raise RuntimeError('Pair check missed common-view samples')
    forced=sets[0].item(0,size=1024,kind='uniform',downsample=.5,local_task=False)
    if json.loads(forced['recipe'])['resampled_native_hw']!=[512,512]:raise RuntimeError('Half-resolution pair not exercised')
    result=dict(passed=True,sources=1000,heldout=0,mixed_sizes=counts,same1024_exact=same1024,records=rows,
                scope='engineering input audit, not validation')
    write_json(Path(root)/'input_audit.json',result)
    return result


def verify_pair(root,epoch,*,smoke=False):
    root=Path(root);rows=[];statuses=[]
    expected=epoch*(4 if smoke else 250)
    for arm in ('native','mixed'):
        status=json.loads((root/arm/'status.json').read_text())
        if status['status']!='stage_complete' or status['epoch']!=epoch or status['updates']!=expected:
            raise RuntimeError(f'Incomplete arm {arm}')
        statuses.append(status)
        records=[json.loads(line) for line in (root/arm/'metrics.jsonl').read_text().splitlines()]
        if len(records)!=expected:raise RuntimeError('Incomplete update receipts')
        rows.append(records)
    for key in ('initial_sha256','source_pool_sha256','recipe_signature'):
        if statuses[0][key]!=statuses[1][key]:raise RuntimeError(f'A/B mismatch {key}')
    for a,b in zip(*rows):
        for key in ('sources','timesteps','learning_rate','noise_rng_before','epoch','update','samples','micro_batch'):
            if a[key]!=b[key]:raise RuntimeError(f'Unpaired update: {key}')
        for ra,rb in zip(a['recipes'],b['recipes']):
            if any(ra[k]!=rb[k] for k in COMMON_RECIPE):raise RuntimeError('Unpaired recipe receipt')
    if not smoke:
        for records,arm in zip(rows,('native','mixed')):
            for e in range(1,epoch+1):
                recipes=[rp for r in records if r['epoch']==e for rp in r['recipes']]
                if len(recipes)!=1000 or len({rp['source'] for rp in recipes})!=1000:raise RuntimeError('Source coverage differs')
                if sum(rp['native_size']==512 for rp in recipes)!=(500 if arm=='mixed' else 0):raise RuntimeError('View budget differs')
    result=dict(passed=True,epoch=epoch,updates_per_arm=expected,shared_random_initialization=True,
        same_sources_order_profiles_parameters_timesteps_noise_lr=True,control_retrained=False,official_scores=None)
    write_json(root/'pair_check.json',result)
    return result


def run(config_path,*,smoke=False,resume=False,stop=None,root_override=None):
    config=load_config(config_path);opt=config['restore_native']
    stop=(2 if smoke else opt['stage_epochs']) if stop is None else stop
    root=Path(root_override or project_path(config,'outputs/restore_smoke' if smoke else config['rgb_restoration']['output_dir']))
    if root.exists() and not resume:raise FileExistsError(root)
    if shutil.disk_usage(ROOT).free/1024**3<opt['minimum_free_gib']:raise RuntimeError('Insufficient fast-disk space')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if not smoke and not resume:
        proof=ROOT/'outputs/restore_smoke/pipeline_status.json'
        if not proof.exists():raise RuntimeError('Complete the GPU short test first')
        verified=json.loads(proof.read_text())
        if verified['status']!='complete' or verified['source_versions']!=versions() or verified['recipe_signature']!=recipe_signature(config):
            raise RuntimeError('Short-test source/config gate differs')
    root.mkdir(parents=True,exist_ok=resume)
    status=dict(status='running',stage='preflight',completed=[],smoke=smoke,stop_epoch=stop,
                source_versions=versions(),recipe_signature=recipe_signature(config),platform_submission=False)
    if resume:
        prior=json.loads((root/'pipeline_status.json').read_text())
        if prior['source_versions']!=versions() or prior['recipe_signature']!=recipe_signature(config):
            raise RuntimeError('Queue resume requires unchanged source and recipe')
    def save():write_json(root/'pipeline_status.json',status)
    def execute(name,command):
        status['stage']=name;save()
        with (root/(name+'.log')).open('a' if resume else 'w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=process.pid;save()
            if process.wait():raise RuntimeError(f'{name} failed; inspect its log')
        status['completed'].append(name);save()
    save()
    try:
        if not resume:
            checked_assets(config);audit_inputs(config,root)
            make_initial(config,root/'initial.pt')
            # 源码快照只含可公开代码，不复制赛题数据或模型。
            for relative in SOURCES:
                target=root/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,target)
            fixed=build_fixed(config,root,smoke=smoke)
            old=load_rgb_restorer(project_path(config,opt['old_restorer']),'cuda').requires_grad_(False)
            old_reference(old,fixed,root,'cuda',opt['monitor']['seed']);del old,fixed
            torch.cuda.empty_cache()
            if smoke:parity(config,root)
        for arm in ('native','mixed'):
            command=[sys.executable,'-u','train_restore_native.py','--config',str(Path(config_path).resolve()),
                     '--arm',arm,'--root',str(root),'--stop',str(stop)]
            if smoke:command.append('--smoke')
            if resume:command.append('--resume')
            execute(arm,command)
        status['pair_check']=verify_pair(root,stop,smoke=smoke)
        if not smoke:
            execute('analysis',[sys.executable,'-u','tools/evaluate_restore_native.py','--config',str(Path(config_path).resolve()),
                                '--root',str(root),'--epoch',str(stop)])
        status.update(status='complete',stage='complete')
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));raise
    finally:save()
    print(json.dumps(status,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='config/train/restore_native.yaml')
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop',type=int);p.add_argument('--root')
    a=p.parse_args();run(a.config,smoke=a.smoke,resume=a.resume,stop=a.stop,root_override=a.root)
