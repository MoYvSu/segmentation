# -*- coding: utf-8 -*-
"""全图／1024单组短测、正式训练与完整输出；控制只读复用。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.affinity_mixed import view_plan
from tools.run_affinity_native import SOURCES as NATIVE_SOURCES, read
from train_affinity_native import common_config, compare_draw, verify_control
from train_backend_adaptation import sha, write_json
from utils.config import load_config, project_path

SOURCES = tuple(dict.fromkeys((*NATIVE_SOURCES, 'config/train/backend_d4.yaml',
    'config/train/affinity_mixed.yaml', 'data/affinity_mixed.py', 'train_affinity_mixed.py',
    'tools/run_affinity_mixed.py', 'models/affinity_geometry.py', 'utils/affinity_graph.py')))


def check_references(config):
    control, rows = verify_control(config)
    folder = Path(project_path(config, config['affinity_native']['reused_native']))
    native = read(folder/'status.json')
    if native['status'] != 'complete' or native['updates'] != 1280:
        raise RuntimeError('Native1024 reference is incomplete')
    if sha(folder/'final.pt') != config['affinity_native']['native_sha256']:
        raise RuntimeError('Native1024 reference checkpoint changed')
    if common_config(native['config']) != common_config(config):
        raise RuntimeError('Non-view options differ from native1024 control')
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256'):
        if native[key] != control[key]:
            raise RuntimeError('Existing controls have different initialization: '+key)
    for key in ('frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if native.get(key) is not True:
            raise RuntimeError('Existing native control audit failed: '+key)
    other = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    if len(other) != len(rows):
        raise RuntimeError('Native control update count differs')
    for a,b in zip(rows,other):
        compare_draw(a,b)
    return dict(passed=True, control_retrained=False, control_updates=1280,
        initial_sha256=control['initial_state_sha256'],
        control_sha256=config['affinity_native']['control_sha256'],
        native_sha256=config['affinity_native']['native_sha256'])


def verify_mixed(config, folder, expected):
    folder = Path(folder)
    status = read(folder/'status.json')
    if status['status'] != 'complete' or status['updates'] != expected:
        raise RuntimeError('Mixed training is incomplete')
    for key in ('reused_control_verified','frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if status.get(key) is not True:
            raise RuntimeError('Mixed audit failed: '+key)
    _, control = verify_control(config, status)
    rows = [json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    if len(rows) != expected:
        raise RuntimeError('Missing mixed receipts')
    totals = {'whole':0,'native1024':0}
    for row, reference in zip(rows, control):
        compare_draw(row, reference)
        plan = view_plan(config['affinity_native']['mixed_view_seed'],row['epoch'],64)
        view = 'whole' if plan[row['step']-1] == 0 else 'native1024'
        if row['training_view'] != view or row['auxiliary_weight'] != 0:
            raise RuntimeError('View schedule or unchanged loss contract differs')
        totals[view] += 1
    if min(totals.values()) == 0 or (expected == 1280 and totals != {'whole':640,'native1024':640}):
        raise RuntimeError('Both training views must be represented with the agreed budget')
    return dict(passed=True, updates=expected, view_updates=totals, control_retrained=False,
                same_initialization_sources_augmentation=True, loss_unchanged=True)


def audit_reuse(config, root):
    import cv2
    import numpy as np
    import torch
    from data.mim_dataset import list_images
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from tools.affinity_native_views import predict_views
    from tools.analyze_affinity_connectivity import load_directory
    torch.set_num_threads(4);cv2.setNumThreads(2)
    restorer = get_restorer(config,'cuda')
    opt=config['affinity_native']; rows=[]
    files=[Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))]
    for kind, directory in [('control',opt['reused_control']),('native',opt['reused_native'])]:
        model,_=load_backend(project_path(config,directory,'final.pt'),config,'cuda')
        for index,path in enumerate(files):
            if path.stem not in config['affinity_connectivity']['parity_images']:continue
            _,views=predict_views(model,restorer,path,config,'cuda',
                config['backend_adaptation']['restoration']['inference_seed']+index,[0,1024])
            for view,pred in views.items():
                if kind=='control':
                    expected=(project_path(config,directory,'deployment') if view=='global' else
                        project_path(config,'outputs/affinity_native/control_views/patch1024'))
                else:expected=project_path(config,directory,'deployment',view)
                ids,classes=load_directory(expected,path.stem)
                if not np.array_equal(ids,pred['instances']) or classes!=pred['classes']:
                    raise RuntimeError(f'Existing final output differs: {kind}/{view}/{path.stem}')
                rows.append(dict(reference=kind,view=view,source=path.stem,final_output_equal=True))
        del model;torch.cuda.empty_cache()
    if len(rows)!=8:raise RuntimeError('Incomplete existing-output check')
    write_json(Path(root)/'parity.json',dict(passed=True,rows=rows))


def render_final(config, root):
    from data.mim_dataset import list_images
    from tools.analyze_affinity_connectivity import load_directory, render
    from utils.patch_diagnostic import phase_description
    root=Path(root);gallery=root/'gallery';gallery.mkdir(exist_ok=False)
    opt=config['affinity_native']; rows=[]
    directories=[project_path(config,opt['reused_control'],'deployment'),
        project_path(config,opt['reused_native'],'deployment','patch1024'),
        root/'candidate/deployment/global',root/'candidate/deployment/patch1024']
    labels=['control whole','p1024 native','mixed whole','mixed native1024']
    pages=['<!doctype html><meta charset="utf-8"><title>Affinity mixed</title>',
        '<p>Unlabeled predictions; no test GT. Same initialization, fixed D5a and semantic heads.</p>']
    for path in map(Path,list_images(project_path(config,config['inference']['test_dir']))):
        predictions=[load_directory(p,path.stem) for p in directories]
        render(path,predictions,labels,gallery/(path.stem+'.png'),path.stem,width=320)
        rows.append(dict(source=path.stem,variants={label:phase_description(*pred) for label,pred in zip(labels,predictions)}))
        pages.append(f'<p>{path.stem}<br><img loading="lazy" width="1280" src="gallery/{path.stem}.png"></p>')
    if len(rows)!=100:raise RuntimeError('Expected 100 final comparisons')
    write_json(root/'summary.json',dict(scope='Predicted changes, not accuracy',images=rows,official_scores=None))
    (root/'index.html').write_text('\n'.join(pages),encoding='utf8')


def run(args):
    config=load_config(args.config)
    if config['backend_adaptation']['epochs']!=20:
        raise ValueError('Authorized stage is 20 epochs')
    root=Path(project_path(config,config['backend_adaptation']['output_dir']+('_smoke' if args.smoke else '')))
    if shutil.disk_usage(ROOT).free/1024**3<config['affinity_native']['minimum_free_gib']:
        raise RuntimeError('Insufficient fast-disk space')
    references=check_references(config)
    if not args.smoke:
        gate=Path(str(root)+'_smoke'); state=read(gate/'pipeline_status.json')
        if state['status']!='complete' or state['config']!=json.loads(json.dumps(config)) or not read(gate/'parity.json')['passed']:
            raise RuntimeError('Same configuration must pass smoke and final-output parity')
        for name in SOURCES:
            if (gate/'source'/name).read_bytes()!=(ROOT/name).read_bytes():
                raise RuntimeError('Source changed after smoke: '+name)
    root.mkdir(parents=True,exist_ok=False)
    for name in SOURCES:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    status=dict(status='running',pid=os.getpid(),smoke=args.smoke,config=config,stage='preflight',completed=[],
        references=references,sources={name:sha(ROOT/name) for name in SOURCES},
        automatic_extension=False,competition_submission=False,control_retrained=False)
    def save():write_json(root/'pipeline_status.json',status)
    def execute(stage,command):
        status['stage']=stage;save()
        with (root/(stage+'.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;save()
            if child.wait():raise RuntimeError(stage+' failed; inspect its log')
        status['completed'].append(stage);save()
    save()
    try:
        if args.smoke:
            execute('parity',[sys.executable,'-u',__file__,'--config',args.config,'--audit',str(root)])
        execute('train',[sys.executable,'-u','train_affinity_mixed.py','--config',args.config,
            '--output',str(root/'candidate')]+(['--smoke'] if args.smoke else []))
        status['training_audit']=verify_mixed(config,root/'candidate',8 if args.smoke else 1280);save()
        if not args.smoke:
            execute('inference',[sys.executable,'-u','train_affinity_native.py','--config',args.config,
                '--mode','infer','--checkpoint',str(root/'candidate/final.pt'),
                '--output',str(root/'candidate/deployment'),'--views','0','1024'])
            status['stage']='render';save();render_final(config,root)
        status.update(status='complete',stage='done')
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/train/affinity_mixed.yaml')
    p.add_argument('--smoke',action='store_true');p.add_argument('--audit')
    a=p.parse_args()
    if a.audit:audit_reuse(load_config(a.config),a.audit)
    else:run(a)
