# -*- coding: utf-8 -*-
"""覆盖采样门禁与独立20轮：默认只8步，formal必须显式--go20，旧控制只读。"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.affinity_coverage import OPTIONS, REFERENCE_SHA, REFERENCE_STEPS_SHA, METADATA_SHA, TRIAL_SHA, read_plan, sha
from train_affinity_coverage import (CONFIG_PATH, ORIGINAL_NATIVE_SHA, ORIGINAL_PIPELINE_SHA,
    canonical, comparable, load_original_core, read, reference_config, verify_reference, verify_run)
from train_backend_adaptation import tensor_digest, write_json
from utils.config import load_config, project_path

SMOKE=ROOT/'outputs/affinity_coverage_smoke'
FORMAL=ROOT/'outputs/affinity_coverage'
GRID=ROOT/'outputs/affinity_coverage_grid/report.json'
MINIMUM_FREE_GIB=1.7 # 单臂约1.2GiB产物，加0.5GiB余量；不改旧双臂配置里的6GiB。
NEW_FILES=(CONFIG_PATH,'data/affinity_coverage.py','train_affinity_coverage.py','tools/run_affinity_coverage.py',
           'tools/probe_affinity_coverage.py','tests/test_affinity_coverage.py',
           'tests/test_affinity_coverage_training.py','tests/test_affinity_coverage_runner.py')
TECHNICAL_FILES={'status.json':'24ab383095510c0ebd0d8ec13629ec07fb4cf37b0d46774ec1bc9971e3d9a706',
 'steps.jsonl':'0c75a996cda4fb52c0a3808851f8822e6182d5cb39d0a86eb69ae67a2ebd2698',
 'final.pt':'287a76241920369b0a8c4cbac91b71c803c3f4d144bd9c3ba88417d8857dab01'}


def inventory(parent):
    """保留148份旧来源，递归补齐新增入口；快照只含代码/配置。"""
    found,pending=set(parent),list(NEW_FILES)
    while pending:
        name=pending.pop()
        if name in found:continue
        path=ROOT/name
        if not path.is_file():raise RuntimeError('Missing source: '+name)
        found.add(name)
        if path.suffix in ('.yaml','.yml'):
            base=(yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('_base')
            if base:pending.append((path.parent/base).resolve().relative_to(ROOT).as_posix())
            continue
        modules=[]
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            if isinstance(node,ast.Import):modules.extend(a.name.split('.') for a in node.names)
            elif isinstance(node,ast.ImportFrom):
                prefix=list(Path(name).parent.parts[:len(Path(name).parent.parts)-node.level+1]) if node.level else []
                module=prefix+(node.module.split('.') if node.module else [])
                modules.append(module);modules.extend(module+a.name.split('.') for a in node.names if a.name!='*')
        for parts in modules:
            if not parts:continue
            for item in (Path(*parts).with_suffix('.py'),Path(*parts)/'__init__.py'):
                if (ROOT/item).is_file():pending.append(item.as_posix())
    return {name:sha(ROOT/name) for name in sorted(found)}


def verify_snapshot(root,sources,*,current=True):
    for name,digest in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(Path(root)/'source'/name)!=digest:
            raise RuntimeError('Source snapshot changed: '+name)
        if current and sha(ROOT/name)!=digest:raise RuntimeError('Current protected source changed: '+name)


def grid_contract(config):
    receipt=read(GRID)
    if (receipt.get('complete') is not True or receipt.get('format')!='affinity_coverage_grid_v1'
        or receipt.get('checkpoint_sha256')!=REFERENCE_SHA or receipt.get('epochs')!=20
        or receipt.get('original_updates')!=1280 or receipt.get('original49_sources')!=49
        or receipt.get('cached148_sources')!=148 or receipt.get('actual_four_draw_grid_content_exact')!=[24,25,28,37]
        or receipt.get('actual_target_mask_gate') is not True or receipt.get('protected_before_after_exact') is not True
        or receipt.get('full_processed_newgt_including_filled') is not True):
        raise RuntimeError('Actual512grid/full processed-newGT gate must complete before GPU short training')
    parents=receipt['parents']
    if (parents.get('trial_sha256')!=TRIAL_SHA or parents.get('trial_metadata_sha256')!=METADATA_SHA
            or parents.get('original_steps_sha256')!=REFERENCE_STEPS_SHA):
        raise RuntimeError('Actual-grid gate was made from another recipe or original R1')
    for path,digest in receipt['protected_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Grid protected input/source/map changed: '+path)
    return dict(passed=True,report_sha256=sha(GRID),parents=parents,newgt=receipt['newgt'],
        actual_four_draw_grid_content_exact=receipt['actual_four_draw_grid_content_exact'],
        actual_target_mask_gate=True,full_processed_newgt_including_filled=True,
        protected_sha256=receipt['protected_sha256'],original49_sources=49,cached148_sources=148)


def technical_reuse(config):
    """旧8步的完整loss/冻结/重载审计直接复用，不再次训练控制。"""
    from train_affinity_source import verify_run as verify_technical, reference_config as source_base
    folder=ROOT/'outputs/affinity_source_smoke/technical_control'
    for name,digest in TECHNICAL_FILES.items():
        if sha(folder/name)!=digest:raise RuntimeError('Historical technical8 changed: '+name)
    status=read(folder/'status.json')
    if comparable(source_base(status['config']))!=comparable(reference_config(config)):
        raise RuntimeError('Historical technical8 is not the same R1 mathematical recipe')
    audit=verify_technical(status['config'],folder,8)
    pipeline=read(folder.parent/'pipeline_status.json')
    if pipeline.get('status')!='complete' or pipeline.get('runtime') is None:
        raise RuntimeError('Historical technical8 pipeline/runtime missing')
    verify_snapshot(folder.parent,pipeline['sources'])
    return dict(passed=True,control_retrained=False,updates_reused=8,files_sha256=TECHNICAL_FILES,
        pipeline_sha256=sha(folder.parent/'pipeline_status.json'),audit=audit,runtime=pipeline['runtime'])


def contract(config):
    from tools.run_affinity_short import precision_record,verify_package
    from tools.run_affinity_sweep import runtime_contract
    from tools.verify_affinity_newgt import audit_new_gt
    from data.mim_dataset import list_images
    verify_reference(config)
    original=ROOT/'outputs/affinity_balance';old=read(original/'pipeline_status.json')
    if (sha(original/'pipeline_status.json')!=ORIGINAL_PIPELINE_SHA or old.get('status')!='complete'
            or old.get('stage')!='done' or len(old.get('sources',{}))!=49):
        raise RuntimeError('Actual completed original49 R1 pipeline differs')
    verify_snapshot(original,old['sources'],current=False)
    parent=ROOT/'outputs/affinity_tail';sealed=read(parent/'pipeline_status.json')
    if (sealed.get('status')!='complete' or sealed.get('stage')!='done' or len(sealed.get('sources',{}))!=148):
        raise RuntimeError('Protected148 source/runtime parent differs')
    verify_snapshot(parent,sealed['sources'])
    # 二处历史接口扩展不进入原训练/推理，其余依赖完整AST保持原49。
    for name in old['sources']:
        if name in ('train_affinity_native.py','tools/affinity_native_views.py'):continue
        before=(original/'source'/name).read_text(encoding='utf8');after=(ROOT/name).read_text(encoding='utf8')
        equal=(ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(after),include_attributes=False)
               if name.endswith('.py') else yaml.safe_load(before)==yaml.safe_load(after))
        if not equal:raise RuntimeError('Original49 dependency logic changed: '+name)
    runtime=runtime_contract(config)
    if runtime!=sealed['runtime']:raise RuntimeError('Current sam2_env/runtime differs from protected148')
    precision=precision_record(torch);gt=audit_new_gt(config);grid=grid_contract(config)
    for key in ('npz_cohort_sha256','class_cohort_sha256','manifest_sha256'):
        if gt[key]!=grid['newgt'][key]:raise RuntimeError('Actual-grid newGT identity differs: '+key)
    reused=technical_reuse(config)
    if reused['runtime']!=runtime:raise RuntimeError('Historical technical8 runtime differs')
    files=list(map(Path,list_images(project_path(config,config['inference']['test_dir']))))
    ref=Path(project_path(config,OPTIONS['reference']))
    package=verify_package(ref/'deployment/patch1024',ROOT/'outputs/affinity_balance/mixedbalance1024.zip',
        ROOT/'outputs/affinity_balance/mixed_package.json',files,read(ref/'deployment/manifest.json'))
    sources=inventory(sealed['sources'])
    return dict(format='affinity_coverage_contract_v1',original_pipeline_sha256=ORIGINAL_PIPELINE_SHA,
        original_sources=old['sources'],original_runtime='not recorded in historical49; current148 is verified',
        original_native_sha256=ORIGINAL_NATIVE_SHA,parent148_pipeline_sha256=sha(parent/'pipeline_status.json'),
        sources=sources,runtime=runtime,precision=precision,newgt=gt,grid=grid,
        technical_reuse=reused,control_package=package,trial_sha256=TRIAL_SHA,metadata_sha256=METADATA_SHA)


@torch.no_grad()
def parity(config,output):
    """只读2图，保留它们在100图排序中的实际seed索引。"""
    from models.backend_adaptation import load_backend
    from data.mim_dataset import list_images
    from tools.analyze_affinity_connectivity import load_directory
    from train_affinity_connectivity import get_restorer
    core=load_original_core(config);base=reference_config(config)
    ref=Path(project_path(config,OPTIONS['reference']))
    model,_=load_backend(ref/'final.pt',base,'cuda');restorer=get_restorer(base,'cuda')
    initial=tensor_digest(model.state_dict().items());d5a=tensor_digest(restorer.state_dict().items())
    files=list(map(Path,list_images(project_path(base,base['inference']['test_dir']))));records=[]
    for index,path in enumerate(files):
        if path.stem not in base['affinity_connectivity']['parity_images']:continue
        seed=base['backend_adaptation']['restoration']['inference_seed']+index
        _,views=core.predict_views(model,restorer,path,base,'cuda',seed,[1024])
        ids,classes=load_directory(ref/'deployment/patch1024',path.stem)
        if not np.array_equal(ids,views['patch1024']['instances']) or classes!=views['patch1024']['classes']:
            raise RuntimeError('Original49 final deployment differs: '+path.stem)
        records.append(dict(source=path.stem,full_cohort_index=index,seed=seed,final_output_equal=True))
    if ([r['source'] for r in records]!=base['affinity_connectivity']['parity_images']
        or tensor_digest(model.state_dict().items())!=initial or tensor_digest(restorer.state_dict().items())!=d5a):
        raise RuntimeError('Two-source control parity incomplete or state mutated')
    receipt=dict(passed=True,control_retrained=False,checkpoint_sha256=REFERENCE_SHA,
                 predictor='actual original49 snapshot',rows=records)
    write_json(Path(output)/'parity.json',receipt)
    del model,restorer;gc.collect();torch.cuda.empty_cache()
    return receipt


def verify_gate(config,expected):
    gate=read(SMOKE/'pipeline_status.json')
    if (gate.get('status')!='complete' or gate.get('stage')!='done' or gate.get('smoke') is not True
        or gate.get('completed')!=['parity','training'] or gate.get('contract')!=expected
        or gate.get('config')!=canonical(config) or gate.get('control_retrained') is not False):
        raise RuntimeError('Exact isolated completed coverage8 gate is required')
    verify_snapshot(SMOKE,expected['sources'])
    audit=verify_run(config,SMOKE/'candidate',8,expected)
    if audit!=gate['audit'] or read(SMOKE/'parity.json')!=gate['parity'] or not gate['parity']['passed']:
        raise RuntimeError('Coverage8 state or actual frozen output parity changed')
    return gate


def render(config,root):
    import cv2
    from data.mim_dataset import list_images
    from tools.analyze_affinity_connectivity import load_directory,render as render_image
    from utils.patch_diagnostic import phase_description
    core=load_original_core(config);ref=Path(project_path(config,OPTIONS['reference'],'deployment/patch1024'))
    candidate=root/'candidate/deployment/patch1024';gallery=root/'gallery';gallery.mkdir(exist_ok=False)
    records=[]
    for path in map(Path,list_images(project_path(config,config['inference']['test_dir']))):
        shape=cv2.imread(str(path)).shape[:2]
        arms=[load_directory(p,path.stem) for p in (ref,candidate)]
        for ids,classes in arms:core.verify_prediction(ids,classes,shape)
        render_image(path,arms,['fixed R1 mixed balance e20','coverage e20'],gallery/(path.stem+'.png'),path.stem,width=360)
        records.append(dict(source=path.stem,phases=[phase_description(*arm) for arm in arms]))
    manifest=read(root/'candidate/deployment/manifest.json')
    if (len(records)!=100 or manifest.get('complete') is not True or manifest.get('epoch')!=20
        or manifest.get('sizes')!=[1024] or len(manifest.get('images',[]))!=100
        or manifest.get('checkpoint_sha256')!=sha(root/'candidate/final.pt')):
        raise RuntimeError('Complete100 fixed native1024 final output contract differs')
    write_json(root/'summary.json',dict(scope='unlabeled final outputs, not accuracy',images=records))


def run(*,smoke=True,go20=False,resume=None,output=None):
    if not smoke and not go20:raise RuntimeError('Formal20 requires the independent explicit --go20 mode')
    if smoke and (go20 or resume):raise RuntimeError('Smoke does not resume or automatically continue to20')
    config=load_config(CONFIG_PATH);expected=contract(config)
    root=Path(output).resolve() if output else (SMOKE if smoke else FORMAL)
    if root.parent!=ROOT/'outputs' or (not resume and root not in (SMOKE,FORMAL)) or root.exists():
        raise RuntimeError('Fresh private output root required; resume uses a fresh sibling folder')
    if shutil.disk_usage(ROOT).free<MINIMUM_FREE_GIB*1024**3:
        raise RuntimeError('Single-arm artifact budget1.2GiB plus0.5GiB reserve is unavailable')
    if not smoke:verify_gate(config,expected)
    root.mkdir(parents=True)
    for name in expected['sources']:
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write_json(root/'contract.json',expected)
    status=dict(status='running',stage='preflight',pid=os.getpid(),smoke=smoke,config=config,contract=expected,
        sources=expected['sources'],completed=[],control_retrained=False,automatic_extension=False,
        competition_submission=False,automatic_package=False,formal_explicit_go=bool(go20),resume=resume)
    def save():write_json(root/'pipeline_status.json',status)
    def execute(stage,command):
        status['stage']=stage;save()
        with (root/(stage+'.log')).open('x',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;save()
            if child.wait():raise RuntimeError(stage+' failed; inspect the stage log')
        status['completed'].append(stage);status.pop('child_pid',None);save()
    save()
    try:
        if smoke:
            execute('parity',[sys.executable,'-u',__file__,'--audit',str(root)])
            status['parity']=read(root/'parity.json')
        execute('training',[sys.executable,'-u','train_affinity_coverage.py','--config',CONFIG_PATH,
            '--output',str(root/'candidate'),'--contract',str(root/'contract.json')]
            +(['--smoke'] if smoke else [])+(['--resume',str(resume)] if resume else []))
        status['audit']=verify_run(config,root/'candidate',8 if smoke else 1280,expected);save()
        if not smoke:
            execute('inference',[sys.executable,'-u',__file__,'--infer',str(root/'candidate/final.pt'),
                                '--infer-output',str(root/'candidate/deployment')])
            status['stage']='render';save();render(config,root);status['completed'].append('render')
        verify_snapshot(root,expected['sources'])
        if contract(config)!=expected:raise RuntimeError('Protected source/runtime/data/control changed during the run')
        status.update(status='complete',stage='done')
    except BaseException as exc:status.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--go20',action='store_true')
    parser.add_argument('--resume');parser.add_argument('--output');parser.add_argument('--audit')
    parser.add_argument('--infer');parser.add_argument('--infer-output')
    args=parser.parse_args()
    if args.smoke and args.go20:parser.error('--smoke and independent --go20 are mutually exclusive')
    if args.audit:parity(load_config(CONFIG_PATH),args.audit)
    elif args.infer:
        if not args.infer_output:parser.error('--infer-output required')
        cfg=load_config(CONFIG_PATH);core=load_original_core(cfg)
        core.inference(cfg,args.infer,args.infer_output,[1024])
    else:run(smoke=not args.go20,go20=args.go20,resume=args.resume,output=args.output)
