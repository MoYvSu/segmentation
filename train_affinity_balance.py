# -*- coding: utf-8 -*-
"""两种既有采样的损失单变量实验：显式核验允许差异，不放宽旧控制检查。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

from data.affinity_native import build_datasets as native_data
from data.affinity_mixed import build_datasets as mixed_data
from train_affinity_native import train, verify_control, compare_draw, read
from train_backend_adaptation import sha
from utils.config import load_config, project_path


def reference_config(config):
    result=deepcopy(config)
    loss=result['direct_semantic_affinity']['affinity_loss']
    if loss['negative_weight']!=1.0:
        raise ValueError('Candidate negative_weight must equal 1.0')
    loss['negative_weight']=1.5
    for key in ('penalty_sampling','penalty_reference','penalty_sha256'):
        result['affinity_native'].pop(key)
    return result


def comparable(config):
    result=deepcopy(config);result['backend_adaptation'].pop('output_dir',None)
    return json.loads(json.dumps(result))


def verify_reference(config, state=None):
    old_config=reference_config(config)
    verify_control(old_config,state)
    opt=config['affinity_native'];folder=Path(project_path(config,opt['penalty_reference']))
    prior=read(folder/'status.json')
    if comparable(old_config)!=comparable(prior['config']):
        raise RuntimeError('More than the authorized loss weight changed')
    if prior['status']!='complete' or prior['updates']!=1280 or sha(folder/'final.pt')!=opt['penalty_sha256']:
        raise RuntimeError('Selected sampling control is incomplete or differs')
    for key in ('frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if prior.get(key) is not True:raise RuntimeError('Control audit failed: '+key)
    if state is not None:
        for key in ('initial_state_sha256','frozen_state_sha256','restorer_state_sha256','data'):
            if prior[key]!=state[key]:raise RuntimeError('Sampling/initialization differs: '+key)
    steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    if len(steps)!=1280:raise RuntimeError('Missing control receipts')
    if opt['penalty_sampling'] not in ('native','mixed'):raise ValueError('Unknown sampling')
    return prior,steps


def verify_run(config, output, updates):
    output=Path(output);state=read(output/'status.json')
    prior,reference=verify_reference(config,state)
    if state['config']!=json.loads(json.dumps(config)) or state['status']!='complete' or state['updates']!=updates:
        raise RuntimeError('Candidate incomplete or actual configuration differs')
    for key in ('frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if state.get(key) is not True:raise RuntimeError('Candidate audit failed: '+key)
    steps=[json.loads(s) for s in (output/'steps.jsonl').read_text().splitlines()]
    if len(steps)!=updates:raise RuntimeError('Missing candidate receipts')
    for a,b in zip(steps,reference):
        compare_draw(a,b)
        for key in ('manual_crop','pseudo_crop','crop_attempts'):
            if a[key]!=b[key]:raise RuntimeError('Input crop differs: '+key)
        if a.get('training_view','native1024')!=b.get('training_view','native1024') or a['auxiliary_weight']!=0:
            raise RuntimeError('View schedule or auxiliary loss differs')
    if updates==1280 and sha(output/'final.pt')!=state['final_checkpoint_sha256']:
        raise RuntimeError('Candidate checkpoint differs')
    return dict(passed=True,updates=updates,control_retrained=False,paired_crops_and_augmentation=True,
        initialization_sha256=state['initial_state_sha256'],negative_weight=1.0,
        reference_negative_weight=1.5,sampling=config['affinity_native']['penalty_sampling'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();config=load_config(a.config)
    builder=mixed_data if config['affinity_native']['penalty_sampling']=='mixed' else native_data
    train(config,1024,a.output,smoke=a.smoke,dataset_builder=builder,control_verifier=verify_reference)
