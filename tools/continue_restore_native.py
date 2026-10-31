# -*- coding: utf-8 -*-
"""保持训练源码／配方不变，利用已有阶段终点保存过程图并完整接续。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from data.restore_native import NativeRestorationDataset
from tools.evaluate_restore_native import checked_assets
from tools.run_restore_native import verify_pair, versions
from train_restore_native import file_sha, recipe_signature, state_sha
from train_rgb_restoration import write_json
from utils.config import load_config, project_path


def checked_copy(source, target):
    """阶段原件保持原位，副本逐文件校验后才允许后续写入。"""
    source,target=Path(source),Path(target)
    if target.exists():raise FileExistsError(target)
    if source.is_dir():shutil.copytree(source,target)
    else:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    files=sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
    receipt={}
    for p in files:
        relative=p.relative_to(source) if source.is_dir() else Path(p.name)
        q=target/relative if source.is_dir() else target
        digest=file_sha(p)
        if file_sha(q)!=digest:raise RuntimeError(f'Stage backup differs: {p}')
        receipt[str(relative)]=digest
    return receipt


def preflight(config,root,stop):
    root=Path(root);prior=json.loads((root/'pipeline_status.json').read_text())
    if prior['status']!='complete' or prior.get('smoke'):
        raise RuntimeError('Only a complete formal stage can be continued')
    if prior['source_versions']!=versions() or prior['recipe_signature']!=recipe_signature(config):
        raise RuntimeError('Original training source/config changed; refusing silent migration')
    start=int(prior['stop_epoch'])
    if not start<stop<=config['rgb_restoration']['train']['epochs'] or (stop-start)%5:
        raise ValueError('Continue by complete five-epoch intervals within the original schedule')
    if (root/f'analysis_e{stop}').exists():raise FileExistsError(root/f'analysis_e{stop}')
    if shutil.disk_usage(ROOT).free/1024**3<config['restore_native']['minimum_free_gib']:
        raise RuntimeError('Insufficient fast-disk space')
    checked_assets(config)
    dataset=NativeRestorationDataset(config,'native')
    if len(dataset)!=1000 or any('autodl-fs' in str(Path(p).resolve()) for p in dataset.samples):
        raise RuntimeError('Expected existing fast-disk 1000-source pool')
    pair=verify_pair(root,start)
    receipts={}
    for arm in ('native','mixed'):
        status=json.loads((root/arm/'status.json').read_text())
        saved=torch.load(root/arm/'last.pt',map_location='cpu',weights_only=True)
        slim=torch.load(root/arm/f'epoch_{start:03d}.pt',map_location='cpu',weights_only=True)
        if saved['epoch']!=start or saved['updates']!=start*250 or saved['arm']!=arm or saved['smoke']:
            raise RuntimeError('Resume checkpoint endpoint differs')
        if saved['recipe_signature']!=recipe_signature(config) or saved['source_pool_sha256']!=status['source_pool_sha256']:
            raise RuntimeError('Resume recipe or source identity differs')
        if saved['initial_sha256']!=status['initial_sha256'] or state_sha(saved['model'])!=state_sha(slim['model']):
            raise RuntimeError('Resume weights differ from the analyzed endpoint')
        for key in ('optimizer','scheduler','scaler','rng'):
            if key not in saved:raise RuntimeError(f'Missing resume state: {key}')
        if saved['scheduler']['last_epoch']!=start or saved['scheduler']['T_max']!=60:
            raise RuntimeError('Learning-rate schedule differs')
        if file_sha(root/arm/f'epoch_{start:03d}.pt')!=status['final_checkpoint_sha256']:
            raise RuntimeError('Analyzed weights were replaced')
        receipts[arm]=dict(epoch=start,updates=saved['updates'],next_learning_rate=saved['scheduler']['_last_lr'],
            resume_sha256=file_sha(root/arm/'last.pt'),epoch_sha256=status['final_checkpoint_sha256'])
        del saved,slim
    return dict(passed=True,start_epoch=start,stop_epoch=stop,milestones=list(range(start+5,stop+1,5)),
        training_source_versions=versions(),recipe_signature=recipe_signature(config),arms=receipts,pair_check=pair,
        continuation_source_sha256=file_sha(__file__),free_gib=shutil.disk_usage(ROOT).free/1024**3)


def run(config_path,stop=40,check_only=False):
    config=load_config(config_path);root=Path(project_path(config,config['rgb_restoration']['output_dir']))
    torch.set_num_threads(4)
    proof=preflight(config,root,stop)
    if check_only:
        print(json.dumps(proof,ensure_ascii=False,indent=2),flush=True);return proof
    start=proof['start_epoch'];archive=root/'stages'/f'e{start:02d}'
    archive.mkdir(parents=True,exist_ok=False)
    backup={}
    for name in ('pipeline_status.json','pair_check.json'):
        backup[name]=checked_copy(root/name,archive/name)
    for arm in ('native','mixed'):
        for name in ('last.pt','status.json','run_config.json','deployment'):
            backup[f'{arm}/{name}']=checked_copy(root/arm/name,archive/arm/name)
    write_json(archive/'backup_receipt.json',backup)
    write_json(root/f'continuation_e{stop}.json',proof)
    status=dict(status='running',stage='preflight',completed=[],smoke=False,stop_epoch=stop,start_epoch=start,
        source_versions=proof['training_source_versions'],recipe_signature=proof['recipe_signature'],
        continuation_source_sha256=proof['continuation_source_sha256'],monitor_milestones=proof['milestones'],
        prior_stage_archive=str(archive.relative_to(root)),platform_submission=False)
    def save():write_json(root/'pipeline_status.json',status)
    def execute(name,command,logname):
        if versions()!=proof['training_source_versions']:raise RuntimeError('Original source changed during continuation')
        status['stage']=name;save()
        with (root/logname).open('a',encoding='utf8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=process.pid;save()
            if process.wait():raise RuntimeError(f'{name} failed; inspect {logname}')
        status['completed'].append(name);save()
    save()
    try:
        for arm in ('native','mixed'):
            for endpoint in proof['milestones']:
                execute(f'{arm}_e{endpoint}',[sys.executable,'-u','train_restore_native.py','--config',str(Path(config_path).resolve()),
                    '--arm',arm,'--root',str(root),'--stop',str(endpoint),'--resume'],f'{arm}.log')
            for epoch in proof['milestones']:
                folder=root/arm/'monitor'/f'e{epoch:02d}'
                if len(list(folder.glob('*.png')))!=16 or not (folder/'metrics.json').exists():
                    raise RuntimeError(f'Missing process monitor {arm} e{epoch}')
        status['pair_check']=verify_pair(root,stop)
        execute(f'analysis_e{stop}',[sys.executable,'-u','tools/evaluate_restore_native.py','--config',str(Path(config_path).resolve()),
            '--root',str(root),'--epoch',str(stop)],f'analysis_e{stop}.log')
        status.update(status='complete',stage='complete')
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));raise
    finally:save()
    print(json.dumps(status,ensure_ascii=False),flush=True)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/train/restore_native.yaml');p.add_argument('--stop',type=int,default=40)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    run(a.config,a.stop,a.check_only)
