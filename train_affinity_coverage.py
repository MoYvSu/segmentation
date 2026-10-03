# -*- coding: utf-8 -*-
"""精确原R1训练循环的独立采样入口；只增加路由审计和epoch边界续训。"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import types

import numpy as np
import torch

from data.affinity_coverage import (FORMAT, OPTIONS, REFERENCE_SHA, REFERENCE_STEPS_SHA, FIXED_KEYS,
    CoveragePlan, build_datasets, read_plan, sha)
from utils.config import load_config, project_path

ROOT=Path(__file__).resolve().parent
CONFIG_PATH='config/train/affinity_coverage.yaml'
ORIGINAL_NATIVE_SHA='b3015d014df4f328f758a0812bb5545dd8ef7c9d31eecc368415cfe0821e7956'
ORIGINAL_VIEWS_SHA='c49c762537a8aae9d4c67c3fc872f7d3d8539d1a4890c1b06fe8876bd0847ebb'
ORIGINAL_PIPELINE_SHA='dfda0cbe675dfc8d92cf327e484be9af7eda3cf9f1d8fb334e17f0672225c7ef'
HEAD_FORMAT='affinity_coverage_head_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def canonical(value):
    return json.loads(json.dumps(value,allow_nan=False))


def comparable(config):
    result=deepcopy(config);result['backend_adaptation'].pop('output_dir',None)
    return canonical(result)


def reference_config(config):
    if config.get('affinity_coverage')!=OPTIONS:
        raise ValueError('Coverage format/seed/namespaces/16+16/reference must match the approved recipe')
    if any(type(config['affinity_coverage'][k]) is not type(v) for k,v in OPTIONS.items()):
        raise ValueError('Coverage options must retain their registered types')
    result=deepcopy(config);result.pop('affinity_coverage')
    registered=load_config(str(ROOT/'config/train/affinity_balance_mixed.yaml'))
    if comparable(result)!=comparable(registered):
        raise ValueError('A second variable changed outside manual-native crop selection')
    # bool is equal to 1 in Python; mathematical options must have their registered types too.
    def strict_types(a,b):
        if type(a) is not type(b):
            raise ValueError('Configuration type differs from registered R1')
        if isinstance(a,dict):
            for key in a: strict_types(a[key],b[key])
        elif isinstance(a,list):
            for x,y in zip(a,b): strict_types(x,y)
    strict_types(comparable(result),comparable(registered))
    return result


def verify_reference(config,state=None):
    """调用未放宽的旧control核验；coverage元数据只在其外围明确剥离。"""
    from train_affinity_balance import verify_reference as original_verifier
    base=reference_config(config)
    original_state=deepcopy(state)
    if original_state is not None:
        original_state['data'].pop('coverage')
    ancestor,steps=original_verifier(base,original_state)
    plan=read_plan(config)
    if [{k:r[k] for k in FIXED_KEYS} for r in steps] != [{k:r[k] for k in FIXED_KEYS} for r in plan.reference]:
        raise RuntimeError('Original ancestor sampling receipts differ from R1')
    prior=read(project_path(config,OPTIONS['reference'],'status.json'))
    if comparable(prior['config'])!=comparable(base):
        raise RuntimeError('Registered completed R1 configuration differs')
    for key in ('frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if prior.get(key) is not True:raise RuntimeError('R1 audit failed: '+key)
    if original_state is not None:
        for key in ('initial_state_sha256','frozen_state_sha256','restorer_state_sha256','data'):
            if prior[key]!=original_state[key]:raise RuntimeError('R1 initialization/data differs: '+key)
    return prior,plan.reference


# 原源码字节哈希固定；这些hook不更改前向、BCE、backward、clip或AdamW表达式。
PATCHES=(
    ("    save_monitor(model, restorer, config, device, output, 0, size, smoke=smoke)",
     "    coverage_start_epoch, coverage_start_updates = coverage_resume(model, optimizer, scheduler, status, output)\n"
     "    if coverage_start_epoch == 0:\n        save_monitor(model, restorer, config, device, output, 0, size, smoke=smoke)"),
    ("    updates, started = 0, time.time()","    updates, started = coverage_start_updates, time.time()"),
    ("    for epoch in range(1, epochs+1):","    for epoch in range(coverage_start_epoch+1, epochs+1):"),
    ("            compare_draw(row, reference[updates])",
     "            compare_draw(row, reference[updates])\n            coverage_batch_audit(raw, raw_pseudo, epoch, index+1)"),
    ("        torch.save(checkpoint, output/'last_head.pt')",
     "        checkpoint.update(coverage_checkpoint(epoch, updates, model, optimizer, scheduler, status, output))\n"
     "        torch.save(checkpoint, output/'last_head.pt')"),
    ("    if smoke:\n        del reloaded\n        (output/'final.pt').unlink()",
     "    if smoke and not coverage_keep_smoke:\n        del reloaded\n        (output/'final.pt').unlink()"),
)


def patched_source(source):
    """也提供给CPU测试：每个确定hook必须恰好存在一次，否则拒绝运行。"""
    for before,after in PATCHES:
        if source.count(before)!=1:
            raise RuntimeError('Original R1 hook site differs: '+before.splitlines()[0])
        source=source.replace(before,after,1)
    ast.parse(source)
    return source


def load_original_core(config):
    folder=Path(project_path(config,'outputs/affinity_balance/source'))
    path=folder/'train_affinity_native.py'
    if sha(path)!=ORIGINAL_NATIVE_SHA or sha(folder/'tools/affinity_native_views.py')!=ORIGINAL_VIEWS_SHA:
        raise RuntimeError('Actual original49 R1 training/view snapshots are required')
    module=types.ModuleType('_coverage_original_r1_native')
    # ROOT-dependent imports must see the actual project, not an ignored snapshot folder.
    module.__file__=str(ROOT/'train_affinity_native.py')
    source=path.read_text(encoding='utf8')
    exec(compile(patched_source(source),str(path),'exec'),module.__dict__)
    views=types.ModuleType('_coverage_original_r1_views')
    views.__file__=str(ROOT/'tools/affinity_native_views.py')
    exec(compile((folder/'tools/affinity_native_views.py').read_text(encoding='utf8'),
                 str(folder/'tools/affinity_native_views.py'),'exec'),views.__dict__)
    for name in ('predict_views','save_monitor','save_training_draw','verify_prediction'):
        setattr(module,name,getattr(views,name))
    module.coverage_original_views=views
    module.coverage_original_sha256=ORIGINAL_NATIVE_SHA
    return module


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(state):
    if set(state)!= {'python','numpy','torch','cuda'}:
        raise RuntimeError('Incomplete saved RNG states')
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda'])!=torch.cuda.device_count():
            raise RuntimeError('CUDA RNG device count differs')
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_equal(a,b):
    return (a['python']==b['python'] and a['numpy'][0]==b['numpy'][0]
        and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:]
        and torch.equal(a['torch'],b['torch']) and len(a['cuda'])==len(b['cuda'])
        and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])))


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines()]


def prefix_sha(path,count):
    lines=Path(path).read_bytes().splitlines(keepends=True)
    if len(lines)<count:raise RuntimeError('Committed log prefix is missing')
    return hashlib.sha256(b''.join(lines[:count])).hexdigest()


def batch_tensor_audit(actual,expected):
    """原CPU实际输入/增强/目标逐张量exact，不使用loss观感代替配对检查。"""
    for key,value in expected.items():
        if key not in actual:raise RuntimeError('Actual batch lacks original field: '+key)
        if torch.is_tensor(value):
            other=actual[key]
            if not torch.is_tensor(other) or value.dtype!=other.dtype or not torch.equal(value,other):
                raise RuntimeError('Actual candidate input/GT differs from declared route: '+key)
        elif value!=actual[key]:raise RuntimeError('Actual candidate batch metadata differs: '+key)
    unexpected=set(actual)-set(expected)-{'coverage_anchor','coverage_update'}
    if unexpected:raise RuntimeError('Unexpected candidate-only training fields: '+','.join(sorted(unexpected)))
    return True


class DeclaredCropDataset:
    """独立短测参考：直接裁封存坐标，不复用候选selector或读取旧GT。"""
    def __init__(self,original,box,attempt):
        self.original,self.box,self.attempt=original,box,attempt
    def __len__(self):return len(self.original)
    def __getitem__(self,index):
        from data.rgb_restoration_dataset import read_rgb
        from data.semantic_targets import load_completed_semantic_source
        from data.affinity_native import geometry_sample
        base=self.original.base;path=base.samples[index];rgb=read_rgb(str(path))
        ids,_=load_completed_semantic_source(base.completed_gt_dir/(path.stem+'_gt.npz'),rgb.shape[:2])
        y,x,y1,x1=self.box
        sample=geometry_sample(rgb[y:y1,x:x1],ids[y:y1,x:x1],image_size=self.original.image_size,grid=self.original.grid)
        sample.update(name=path.name,image_name=path.name,image_path=str(path),source_kind='manual',
            crop_box=torch.tensor(self.box,dtype=torch.int32),native_source_shape=torch.tensor(ids.shape,dtype=torch.int32),
            crop_attempt=self.attempt,native_crop_size=1024)
        return sample


def checkpoint_state(epoch,updates,status,output,contract,*,smoke=False):
    """完整epoch日志前缀连同RNG保存；smoke永远不允许作为formal续训起点。"""
    output=Path(output)
    steps=rows(output/'steps.jsonl');epochs=rows(output/'epochs.jsonl')
    if len(steps)!=updates or len(epochs)!=epoch or (not smoke and updates!=epoch*64):
        raise RuntimeError('Checkpoint is not a committed epoch boundary')
    count=len(rows(output/'input_audit.jsonl')) if (output/'input_audit.jsonl').is_file() else 0
    prefixes={'steps.jsonl':dict(lines=updates,sha256=prefix_sha(output/'steps.jsonl',updates)),
              'epochs.jsonl':dict(lines=epoch,sha256=prefix_sha(output/'epochs.jsonl',epoch))}
    if count:prefixes['input_audit.jsonl']=dict(lines=count,sha256=prefix_sha(output/'input_audit.jsonl',count))
    return dict(format=HEAD_FORMAT,rng_state=rng_state(),coverage_state=dict(format=FORMAT,epoch=epoch,
        updates=updates,smoke=bool(smoke),epoch_boundary=not smoke,contract=deepcopy(contract),
        committed_logs=prefixes,initial_state_sha256=status['initial_state_sha256'],
        frozen_state_sha256=status['frozen_state_sha256'],restorer_state_sha256=status['restorer_state_sha256'],
        data=deepcopy(status['data']),original_native_sha256=ORIGINAL_NATIVE_SHA))


def resume_checkpoint(path,config,contract):
    path=Path(path)
    bundle=torch.load(path,map_location='cpu',weights_only=False)
    state=bundle.get('coverage_state',{})
    if (bundle.get('format')!=HEAD_FORMAT or state.get('format')!=FORMAT or state.get('smoke') is not False
            or state.get('epoch_boundary') is not True or not 1<=state.get('epoch',0)<20
            or state.get('updates')!=state['epoch']*64 or bundle.get('epoch')!=state['epoch']
            or bundle.get('size')!=1024 or state.get('contract')!=contract
            or comparable(bundle.get('config',{}))!=comparable(config)
            or state.get('original_native_sha256')!=ORIGINAL_NATIVE_SHA
            or not {'optimizer','scheduler','rng_state','geometry_state_dict'}<=set(bundle)):
        raise RuntimeError('Strict same-contract epoch-boundary coverage checkpoint required')
    for name,record in state['committed_logs'].items():
        if name not in {'steps.jsonl','epochs.jsonl','input_audit.jsonl'} or prefix_sha(path.parent/name,record['lines'])!=record['sha256']:
            raise RuntimeError('Resume committed log prefix differs')
    if state['committed_logs']['steps.jsonl']['lines']!=state['updates'] or state['committed_logs']['epochs.jsonl']['lines']!=state['epoch']:
        raise RuntimeError('Resume checkpoint and committed logs differ')
    return bundle


@contextmanager
def coverage_hooks(core,config,output,contract,*,smoke=False,resume=None,reference_head=None):
    """只修改私有模块globals，异常退出恢复；原模块和冻结权重不被写入。"""
    from train_backend_adaptation import tensor_digest
    from utils.affinity_loss import build_affinity_targets_torch
    plan=read_plan(config)
    output=Path(output);names=('configure_training','compute_affinity_loss','compare_draw','append_json')
    originals={name:getattr(core,name) for name in names}
    hooks=[];audit=[];pending={};reference_model=[reference_head];original_datasets=[None]
    completed=resume_checkpoint(resume,config,contract) if resume else None
    def configure(model):
        originals['configure_training'](model)
        if not smoke or hooks:return
        if reference_model[0] is None:
            # 一个额外小head；encoder/restorer使用同一实际candidate输入和同一features。
            reference_model[0]=deepcopy(model.affinity_decoder).eval().requires_grad_(False)
            bundle=torch.load(project_path(config,OPTIONS['reference'],'final.pt'),map_location='cpu',weights_only=False)
            weights={k.removeprefix('affinity_decoder.'):v for k,v in bundle['model_state_dict'].items()
                     if k.startswith('affinity_decoder.')}
            reference_model[0].load_state_dict(weights,strict=True);del bundle;gc.collect()
        reference_model[0].eval().requires_grad_(False)
        def capture(_module,args):
            with torch.no_grad():pending['reference_logits']=reference_model[0](args[0])['affinity_logits']
        hooks.append(model.affinity_decoder.register_forward_pre_hook(capture))
        pending['parameter']=model.affinity_decoder.affinity_head[0].weight
    def compare(row,old):
        originals['compare_draw'](row,old) # 原父核验失败不能被吞掉。
        update=(row['epoch']-1)*64+row['step'];expected=plan.by_update[update]
        for key in ('manual_crop','pseudo_crop','crop_attempts','training_view'):
            if row[key]!=expected[key]:raise RuntimeError('Real sampling differs from sealed plan: '+key)
        pending['draw']=deepcopy(row);pending['update']=update
    def compute(logits,batch,loss_cfg,*,pseudo=False):
        loss,details=originals['compute_affinity_loss'](logits,batch,loss_cfg,pseudo=pseudo)
        if smoke:
            reference,reference_details=originals['compute_affinity_loss'](pending.pop('reference_logits'),batch,loss_cfg,pseudo=pseudo)
            # 严格沿原compute_affinity_loss的真实输入合同：SAM2没有人工affinity aliases。
            instance_key='instance_map' if pseudo else 'affinity_instance_map'
            valid_key='valid_content' if pseudo else 'affinity_valid_content'
            target,valid=build_affinity_targets_torch(batch[instance_key],batch[valid_key])
            grad=torch.autograd.grad(loss,logits,retain_graph=True)[0]
            if grad[~valid].count_nonzero():raise RuntimeError('Unknown/padding affinity gradient must be zero')
            param_grad=torch.autograd.grad(loss,pending['parameter'],retain_graph=True)[0]
            record=dict(epoch=pending['draw']['epoch'],step=pending['draw']['step'],updates=pending['update'],
                kind='pseudo' if pseudo else 'manual',same_candidate_input=True,
                candidate_raw_bce=float(loss.detach()),r1_e20_same_input_raw_bce=float(reference.detach()),
                first_shared_conv_gradient_norm=float(param_grad.float().norm()),unknown_logit_gradient_max=0.,
                input_sha256=tensor_digest([('input',batch['image'])]),
                gt_sha256=tensor_digest([('canonical_gt',batch[instance_key])]),
                content_sha256=tensor_digest([('content',batch[valid_key])]),
                source_draw=deepcopy(pending['draw']),
                raw_route_tensor_exact=pending.get('raw_route_exact',False),
                supervision={k:details[k] for k in ('positive_edges','negative_edges','precision','recall','specificity')},
                reference_supervision={k:reference_details[k] for k in ('positive_edges','negative_edges','precision','recall','specificity')},
                scope='diagnostic8 actual rawBCE, not convergence/generalization',
                parameter_testpoint='affinity_decoder.affinity_head.0.weight')
            audit.append(record);originals['append_json'](output/'input_audit.jsonl',record)
        return loss,details
    def append(path,row):
        row=deepcopy(row)
        if Path(path).name=='steps.jsonl':
            expected=plan.by_update[row['updates']]
            row['coverage']=dict(format=FORMAT,anchor=bool(expected.get('counterfactual_anchor',{}).get('selected',False)),
                actual_update=row['updates'],executed=True)
        originals['append_json'](path,row)
    def checkpoint(epoch,updates,model,optimizer,scheduler,status,path):
        return checkpoint_state(epoch,updates,status,path,contract,smoke=smoke)
    def raw_audit(raw,raw_pseudo,epoch,step):
        if not smoke:return
        before_rng=rng_state()
        from data.affinity_mixed import build_datasets as original_builder
        from torch.utils.data import default_collate
        if original_datasets[0] is None:
            manual,pseudo,_=original_builder(reference_config(config),1024)
            manual.set_epoch(epoch);pseudo.set_epoch(epoch);original_datasets[0]=(manual,pseudo)
        manual,pseudo=original_datasets[0];expected=plan.by_update[(epoch-1)*64+step]
        tag=expected.get('counterfactual_anchor',{})
        if tag.get('selected'):
            prior=manual.native.base_dataset
            manual.native.base_dataset=DeclaredCropDataset(prior,expected['manual_crop'][0],expected['crop_attempts'][0][0])
            try:expected_manual=default_collate([manual[int(expected['manual_draw']['draw_index'][0])]])
            finally:manual.native.base_dataset=prior
        else:expected_manual=default_collate([manual[int(expected['manual_draw']['draw_index'][0])]])
        expected_pseudo=default_collate([pseudo[int(expected['pseudo_draw']['draw_index'][0])]])
        batch_tensor_audit(raw,expected_manual);batch_tensor_audit(raw_pseudo,expected_pseudo)
        if not rng_equal(before_rng,rng_state()):raise RuntimeError('Read-only raw route audit consumed original RNG')
        pending['raw_route_exact']=True
    def resume_into(model,optimizer,scheduler,status,path):
        if completed is None:return 0,0
        state=completed['coverage_state']
        for key in ('initial_state_sha256','frozen_state_sha256','restorer_state_sha256','data'):
            if status[key]!=state[key]:raise RuntimeError('Resume initialization/data mismatch: '+key)
        model.affinity_decoder.load_state_dict(completed['geometry_state_dict'],strict=True)
        optimizer.load_state_dict(completed['optimizer']);scheduler.load_state_dict(completed['scheduler'])
        if scheduler.last_epoch!=state['epoch']:raise RuntimeError('Resume scheduler epoch differs')
        for name,record in state['committed_logs'].items():
            lines=(Path(resume).parent/name).read_bytes().splitlines(keepends=True)
            (Path(path)/name).write_bytes(b''.join(lines[:record['lines']]))
        restore_rng(completed['rng_state'])
        status['resume']=dict(checkpoint_sha256=sha(resume),epoch=state['epoch'],updates=state['updates'],
            committed_logs=state['committed_logs'],strict_head_optimizer_scheduler_rng=True)
        return state['epoch'],state['updates']
    extra_names=('coverage_resume','coverage_checkpoint','coverage_keep_smoke','coverage_batch_audit')
    old_extras={name:core.__dict__.get(name) for name in extra_names}
    core.configure_training,core.compute_affinity_loss,core.compare_draw,core.append_json=configure,compute,compare,append
    core.coverage_resume,core.coverage_checkpoint,core.coverage_keep_smoke=resume_into,checkpoint,True
    core.coverage_batch_audit=raw_audit
    try:yield audit
    finally:
        for hook in hooks:hook.remove()
        for name,value in originals.items():setattr(core,name,value)
        for name,value in old_extras.items():
            if value is None:core.__dict__.pop(name,None)
            else:setattr(core,name,value)


def verify_run(config,folder,updates,contract=None):
    from train_backend_adaptation import tensor_digest
    folder=Path(folder);status=read(folder/'status.json');data=rows(folder/'steps.jsonl')
    plan=read_plan(config)
    if (updates not in (8,1280) or status.get('status')!='complete' or status.get('updates')!=updates
            or status.get('epoch')!=(1 if updates==8 else 20) or status.get('precision')!='FP32'
            or comparable(status['config'])!=comparable(config) or len(data)!=updates
            or sha(folder/'final.pt')!=status['final_checkpoint_sha256']):
        raise RuntimeError('Coverage complete state/budget/checkpoint differs')
    for key in ('frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal','reused_control_verified'):
        if status.get(key) is not True:raise RuntimeError('Coverage audit failed: '+key)
    for row,expected in zip(data,plan.candidate):
        for key in ('epoch','step','seed','manual','pseudo','manual_draw','pseudo_draw','manual_crop','pseudo_crop','crop_attempts','training_view','updates','auxiliary_weight'):
            if row[key]!=expected[key]:raise RuntimeError('Executed receipt changed fixed field: '+key)
        if row.get('counterfactual_not_executed') is not None or row['coverage']['executed'] is not True:
            raise RuntimeError('Training must report actual execution, not copied counterfactual loss')
    head=torch.load(folder/'last_head.pt',map_location='cpu',weights_only=False)
    state=head.get('coverage_state',{})
    if (head.get('format')!=HEAD_FORMAT or state.get('updates')!=updates or head.get('epoch')!=status['epoch']
            or (contract is not None and state.get('contract')!=contract)
            or not {'optimizer','scheduler','rng_state'}<=set(head)
            or set(head.get('rng_state',{}))!={'python','numpy','torch','cuda'}
            or head.get('scheduler',{}).get('last_epoch')!=status['epoch']
            or not head.get('optimizer',{}).get('state')):
        raise RuntimeError('Saved coverage/version/full resume state differs')
    for item in head['optimizer']['state'].values():
        if (float(item['step'])!=updates or not torch.isfinite(item['exp_avg']).all()
            or not torch.isfinite(item['exp_avg_sq']).all()):
            raise RuntimeError('Saved optimizer update budget/moments differ')
    random.Random().setstate(head['rng_state']['python'])
    np.random.RandomState().set_state(head['rng_state']['numpy'])
    torch.Generator().set_state(head['rng_state']['torch'])
    bundle=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    if tensor_digest(bundle['model_state_dict'].items())!=status['final_state_sha256']:
        raise RuntimeError('Final model state digest differs from strict-reloaded state')
    geometry={k.removeprefix('affinity_decoder.'):v for k,v in bundle['model_state_dict'].items() if k.startswith('affinity_decoder.')}
    if tensor_digest(geometry.items())!=tensor_digest(head['geometry_state_dict'].items()):
        raise RuntimeError('Final bundle and saved head differ')
    if updates==8:
        inputs=rows(folder/'input_audit.jsonl')
        if len(inputs)!=16 or any(not r['same_candidate_input'] or not r['raw_route_tensor_exact'] or r['unknown_logit_gradient_max']!=0
            or not np.isfinite(r['first_shared_conv_gradient_norm']) or r['first_shared_conv_gradient_norm']<=0 for r in inputs):
            raise RuntimeError('Actual same-input8 paired BCE/gradient/input audit is incomplete')
    return dict(passed=True,updates=updates,epoch=status['epoch'],control_retrained=False,
        final_checkpoint_sha256=status['final_checkpoint_sha256'],head_sha256=sha(folder/'last_head.pt'),
        original_native_sha256=ORIGINAL_NATIVE_SHA,full_state_saved=True)


def train(config,output,*,smoke=False,resume=None,contract=None):
    reference_config(config)
    if contract is None:raise RuntimeError('Audited parent/data/runtime/grid contract is required')
    core=load_original_core(config)
    with coverage_hooks(core,config,output,contract,smoke=smoke,resume=resume):
        core.train(config,1024,output,smoke,dataset_builder=build_datasets,control_verifier=verify_reference)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=CONFIG_PATH);parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume')
    parser.add_argument('--contract',required=True)
    args=parser.parse_args()
    if args.smoke and args.resume:parser.error('Smoke cannot resume or continue to formal')
    train(load_config(args.config),args.output,smoke=args.smoke,resume=args.resume,contract=read(args.contract))
