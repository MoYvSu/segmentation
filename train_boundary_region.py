# -*- coding: utf-8 -*-
"""随机NAF的可靠内核分隔短训；不使用精确GT像素BCE、GAN或PH。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, Subset

from data.boundary_naf import canonical_sha, file_sha256
from data.boundary_naf_region import BoundaryNAFRegionDataset, collate_boundary_naf_region
from models.boundary_naf import BoundaryNAFNet
from models.boundary_restore import state_sha256
from train_boundary_naf import monitor as old_monitor, optimizer_for, phase_and_rate
from train_boundary_restore import capture_rng, restore_rng, write_json, reconcile_log
from utils.boundary_region_training import region_loss, region_update
from utils.config import load_config, project_path

FORMAT='boundary_region_training_v1'


def validate_config(config):
    cfg=config['boundary_naf'];training=cfg['training']
    if cfg.get('input_source')!='real_n1' or cfg['inference']['input_mode']!='skeleton':
        raise ValueError('区域监督只允许同一真实N1整图细化输入')
    if any(float(training[k])!=0 for k in ['bce_weight','adversarial_weight','topology_weight']):
        raise ValueError('本轮不混入精确GT像素BCE、GAN或PH')
    if training['topology_updates']!=0 or training['precision']!='float32':
        raise ValueError('仅单阶段FP32区域监督')
    if any(type(training[k]) is not int or training[k]<1 for k in
           ['main_updates','segment_updates','effective_batch_size','checkpoint_every_updates']):
        raise ValueError('更新及batch须为正整数')
    if training['segment_updates']*training['effective_batch_size']!=cfg['data']['samples_per_epoch']:
        raise ValueError('必须保留原采样计划/有效batch')
    if cfg['inference']['iterations']!=1 or cfg['inference']['revert_unconnected']:
        raise ValueError('不加入迭代或推理回退')
    if cfg['region']['loss'].get('exterior') is not True:
        raise ValueError('必须保留窗外绕行的保守检查')
    return cfg


def dataset_for(config,epoch):
    cfg=copy.deepcopy(config['boundary_naf'])
    cfg['real_input_manifest']=str(project_path(config,cfg['real_input_manifest']))
    return BoundaryNAFRegionDataset(project_path(config,cfg['manifest']),cfg,epoch=epoch)


def build_model(cfg,device):
    model=BoundaryNAFNet(**cfg['model'],seed=int(cfg['seed'])).to(device)
    if state_sha256(model)!=cfg['region']['initial_generator_sha256']:
        raise ValueError('本轮必须使用与旧控制相同的随机G初态')
    return model


def identity_for(config,dataset,model):
    cfg=config['boundary_naf'];root=Path(__file__).resolve().parent
    names=['train_boundary_region.py','train_boundary_naf.py','train_boundary_restore.py',
           'data/boundary_naf.py','data/boundary_naf_real.py','data/boundary_naf_region.py',
           'data/boundary_restore.py','models/boundary_naf.py','models/boundary_restore.py',
           'utils/boundary_region_cut.py','utils/boundary_region_training.py',
           'utils/boundary_core_supervision.py','utils/config.py','tools/boundary_naf_monitor.py',
           'tools/boundary_restore_monitor.py','tools/boundary_restore_readout.py']
    assets={}
    paths=[cfg['manifest'],cfg['real_input_manifest']]+cfg['monitor'].get('control_manifests',[])
    if cfg['monitor'].get('roi_spec'):paths.append(cfg['monitor']['roi_spec'])
    for name in paths:
        path=Path(project_path(config,name));assets[str(path)]=file_sha256(path)
    return dict(format=FORMAT,purpose='formal_region_candidate',
        initial_generator_sha256=state_sha256(model),architecture=model.architecture,
        dataset=dataset.identity,recipe=copy.deepcopy(cfg),source_sha256={n:file_sha256(root/n) for n in names},
        assets=assets,random_initialization_no_inherited_weights=True,
        objective='full_vertex_separator_between_trusted_different_GT_atoms; no same_GT_global_connectivity',
        no_exact_GT_pixel_BCE=True,no_test_training=True,
        versions=dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__))


def verify_inputs(identity,dataset):
    dataset.verify_sources();root=Path(__file__).resolve().parent
    for name,digest in identity['source_sha256'].items():
        if file_sha256(root/name)!=digest:raise RuntimeError('封存源码改变: '+name)
    for name,digest in identity['assets'].items():
        if file_sha256(name)!=digest:raise RuntimeError('固定资产改变: '+name)


def save_checkpoint(path,model,optimizer,cfg,identity,step,monitors):
    payload=dict(format=FORMAT,architecture=model.architecture,
        model_state_dict={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},
        model_sha256=state_sha256(model),optimizer=optimizer.state_dict(),rng=capture_rng(),
        identity=identity,identity_sha256=canonical_sha(identity),global_step=int(step),
        monitor_steps=list(monitors),config=copy.deepcopy(cfg),input_contract=copy.deepcopy(cfg['inference']),
        next_lr=phase_and_rate(step,cfg)[2] if step<cfg['training']['main_updates'] else None)
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def load_checkpoint(path,model,optimizer,cfg,identity):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format')!=FORMAT or saved.get('identity')!=identity or saved.get('identity_sha256')!=canonical_sha(identity):
        raise ValueError('区域续训来源/用途/配方/源码不一致')
    step=saved['global_step']
    if type(step) is not int or not 0<=step<=cfg['training']['main_updates']:
        raise ValueError('非法更新次数')
    if saved['architecture']!=model.architecture or saved['input_contract']!=cfg['inference']:
        raise ValueError('架构或输入接口改变')
    expected=phase_and_rate(step,cfg)[2] if step<cfg['training']['main_updates'] else None
    if saved['next_lr']!=expected:raise ValueError('学习率位置不一致')
    model.load_state_dict(saved['model_state_dict'],strict=True)
    if state_sha256(model)!=saved['model_sha256']:raise RuntimeError('严格加载SHA不一致')
    optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
    return step,list(saved['monitor_steps'])


def loader(dataset,cfg,first=0):
    t=cfg['training'];workers=int(t['num_workers'])
    args=dict(batch_size=t['effective_batch_size'],shuffle=False,drop_last=True,
        num_workers=workers,pin_memory=torch.cuda.is_available(),collate_fn=collate_boundary_naf_region,
        generator=torch.Generator().manual_seed(int(cfg['seed'])+dataset.epoch+117))
    if workers:args.update(prefetch_factor=t['prefetch_factor'])
    return DataLoader(Subset(dataset,range(first,len(dataset))),**args)


def to_device(batch,device):
    return {key:value.to(device,non_blocking=True) if torch.is_tensor(value) else value
            for key,value in batch.items()}


def fixed_atoms(sample,minimum=64):
    """评估队列固定，不随模型改善重新选择或删失败分母。"""
    ids=sample['region_ids'][0].cpu().numpy();low=sample['region_keep_low'][0].cpu().numpy().astype(bool)
    atoms=[]
    for gt in sorted(set(map(int,np.unique(ids[low])))-{0}):
        count,labels,stats,_=cv2.connectedComponentsWithStats((low&(ids==gt)).astype(np.uint8),connectivity=8)
        for label in range(1,count):
            if stats[label,cv2.CC_STAT_AREA]>=minimum:atoms.append((gt,labels==label))
    pairs=[]
    adjacent=set()
    for a,b in [(ids[1:],ids[:-1]),(ids[:,1:],ids[:,:-1])]:
        valid=(a>0)&(b>0)&(a!=b)
        if valid.any():adjacent.update(map(tuple,np.unique(np.sort(np.stack([a[valid],b[valid]],1),1),axis=0)))
    for first in range(len(atoms)):
        for second in range(first+1,len(atoms)):
            if tuple(sorted((atoms[first][0],atoms[second][0]))) in adjacent:
                pairs.append((atoms[first][0],atoms[second][0],atoms[first][1],atoms[second][1]))
    if pairs:
        rng=np.random.default_rng(int(sample['meta']['draw_seed']))
        pairs=[pairs[int(k)] for k in rng.permutation(len(pairs))[:4]]
    return pairs


def pair_status(probability,source,sink,threshold=.65):
    """原生8邻接加窗外汇流；种子被涂掉不算分隔成功。"""
    _,labels=cv2.connectedComponents((probability<threshold).astype(np.uint8),connectivity=8)
    first=np.unique(labels[source]);second=np.unique(labels[sink])
    coverage=[float((labels[mask]>0).mean()) for mask in [source,sink]]
    first=first[first>0];second=second[second>0]
    frame=np.zeros(labels.shape,bool);frame[0]=frame[-1]=True;frame[:,0]=frame[:,-1]=True
    outside=np.unique(labels[frame]);outside=outside[outside>0]
    connected=bool(np.intersect1d(first,second).size or
                   (np.intersect1d(first,outside).size and np.intersect1d(second,outside).size))
    return dict(separated=not connected and min(coverage)>=.9,connected=connected,core_coverage=coverage)


def quality(model,samples,device):
    batch=to_device(collate_boundary_naf_region(samples),device)
    before=model.training;model.eval()
    with torch.no_grad():prob=model(batch['input']).detach().float().cpu().numpy()[:,0]
    model.train(before);rows=[]
    for sample,p in zip(samples,prob):
        high=sample['region_keep_high'][0].numpy().astype(bool)
        low=sample['region_keep_low'][0].numpy().astype(bool)
        pairs=[dict(first_gt=int(a),second_gt=int(b),**pair_status(p,s,t)) for a,b,s,t in fixed_atoms(sample)]
        rows.append(dict(source=sample['source'],draw=sample['draw'],
            high_pixels=int(high.sum()),high_mean=float(p[high].mean()) if high.any() else None,
            high_above065=float((p[high]>.65).mean()) if high.any() else None,
            low_pixels=int(low.sum()),low_mean=float(p[low].mean()) if low.any() else None,pairs=pairs))
    def average(prefix,key):
        total=sum(row[prefix+'_pixels'] for row in rows)
        return sum(row[key]*row[prefix+'_pixels'] for row in rows if row[prefix+'_pixels'])/total if total else None
    return dict(rows=rows,high_mean=average('high','high_mean'),high_above065=average('high','high_above065'),
        low_mean=average('low','low_mean'),pair_count=sum(len(row['pairs']) for row in rows),
        separated_pairs=sum(p['separated'] for row in rows for p in row['pairs']))


def monitor(model,config,output,step,dataset,**kwargs):
    result=old_monitor(model,config,output,step,dataset,**kwargs)
    # 追加真实监督范围图。GT只作训练源参考，不给测试图推断答案。
    from tools.boundary_naf_monitor import _rgb_view
    from tools.boundary_restore_monitor import panels,_heat
    folder=Path(output)/'monitor'/f's{step:06d}'
    for index in config['boundary_naf']['monitor']['pair_indices']:
        sample=dataset[index];rgb=_rgb_view(dataset,sample)
        cores=sample['region_keep_low'][0].numpy().astype(bool)
        allowed=sample['region_allowed'][0].numpy().astype(bool)
        high=sample['region_keep_high'][0].numpy().astype(bool)
        field=np.zeros_like(rgb);field[allowed]=[200,160,0];field[cores]=[0,100,255];field[high]=[0,255,100]
        panels(folder/f'region_{index:03d}.jpg',f'{sample["source"]} | region supervision | {step}',
            [('RGB / training source',rgb),('N1 input',_heat(sample['input'][0].numpy())),
             ('Blue trusted interior / yellow allowed / green keep line',field)],280,
            'Different-GT atoms must separate. No precise GT-line BCE; no same-GT global merge target.')
    return result


def step_record(stat, *, step, learning_rate, update_seconds, sources, draws, elapsed_seconds):
    """保留损失内部计时，同时用独立字段记录完整更新/作业耗时。"""
    row=dict(stat)
    if 'elapsed_seconds' in row:
        row['region_update_seconds']=row.pop('elapsed_seconds')
    row.update(global_step=step,learning_rate=learning_rate,update_seconds=update_seconds,
               sources=sources,draws=draws,elapsed_seconds=elapsed_seconds)
    return row


def preflight(config,output):
    cfg=validate_config(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dataset=dataset_for(config,0);model=build_model(cfg,'cpu');identity=identity_for(config,dataset,model)
    samples=[dataset[i] for i in range(32)];batch=collate_boundary_naf_region(samples)
    probability=batch['input']*.8+.05;logits=torch.logit(probability).requires_grad_()
    loss,audit=region_loss(logits,batch,cfg['region']['loss']);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
    assert not torch.any(logits.grad[~batch['valid'].bool()])
    verify_inputs(identity,dataset)
    fixed_plan=dataset.plan_receipt();dataset.set_epoch(1);formal_plan=dataset.plan_receipt();dataset.set_epoch(0)
    result=dict(complete=True,identity=identity,plan=fixed_plan,formal_epoch1_plan=formal_plan,sample_count=32,
        loss=float(loss.detach()),loss_audit=audit,gradient_nonzero=True,unknown_output_gradient_zero=True,
        parameters=sum(p.numel() for p in model.parameters()),no_parameter_updates=True)
    write_json(output/'preflight.json',result);return result


def smoke(config,output,updates=128,device='cuda'):
    cfg=validate_config(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dataset=dataset_for(config,0);model=build_model(cfg,device);optimizer=optimizer_for(model,cfg)
    identity=identity_for(config,dataset,model);identity['purpose']='region_smoke_not_formal_resume'
    samples=[dataset[i] for i in range(cfg['training']['effective_batch_size'])]
    batch=to_device(collate_boundary_naf_region(samples),device)
    before=quality(model,samples,device);stats=[];initial=state_sha256(model)
    for step in range(updates):
        stat=region_update(model,optimizer,batch,cfg['region']['loss'],grad_clip=cfg['training']['grad_clip'])
        stats.append(dict(step=step+1,**stat))
        if step==0 or (step+1)%16==0:print(json.dumps(dict(event='region_smoke',step=step+1,**stat)),flush=True)
    after=quality(model,samples,device)
    save_checkpoint(output/'smoke.pt',model,optimizer,cfg,identity,updates,[])
    with torch.no_grad():expected=model(batch['input'][:2]).detach().cpu()
    fresh=build_model(cfg,device);fresh_optimizer=optimizer_for(fresh,cfg)
    load_checkpoint(output/'smoke.pt',fresh,fresh_optimizer,cfg,identity)
    with torch.no_grad():actual=fresh(batch['input'][:2]).detach().cpu()
    if not torch.equal(expected,actual):raise RuntimeError('GPU严格重载不一致')
    monitored=monitor(fresh,config,output,updates,dataset,force_full=True,source_limit=2)
    verify_inputs(identity,dataset)
    result=dict(complete=True,kind='GPU_fixed_batch_region_fit_not_formal_weights',updates=updates,
        initial_generator_sha256=initial,model_sha256=state_sha256(model),strict_reload_exact=True,
        quality_before=before,quality_after=after,steps=stats,monitor=monitored)
    write_json(output/'smoke.json',result);return result


def train(config,output,device='cuda',resume=None):
    cfg=validate_config(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dataset=dataset_for(config,1);fixed=dataset_for(config,0)
    model=build_model(cfg,device);optimizer=optimizer_for(model,cfg)
    identity=identity_for(config,dataset,model)
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('拒绝覆盖其它实验')
    if not resume and (output/'latest.pt').exists():raise ValueError('已有权重须显式续训')
    write_json(output/'identity.json',identity);write_json(output/'config.json',config)
    step,monitors=load_checkpoint(resume,model,optimizer,cfg,identity) if resume else (0,[])
    if (output/'steps.jsonl').exists():reconcile_log(output,step)
    if step==0 and 0 not in monitors:
        monitor(model,config,output,0,fixed);monitors.append(0)
        save_checkpoint(output/'initial.pt',model,optimizer,cfg,identity,0,monitors)
    total=cfg['training']['main_updates'];segment=cfg['training']['segment_updates'];started=time.monotonic()
    print(json.dumps(dict(event='region_formal_started',updates=total,initial_generator_sha256=identity['initial_generator_sha256'])),flush=True)
    while step<total:
        epoch=step//segment+1;dataset.set_epoch(epoch)
        write_json(output/f'plan_{epoch:03d}.json',dataset.plan_receipt())
        first=(step%segment)*cfg['training']['effective_batch_size']
        for batch in loader(dataset,cfg,first):
            if step>=total or step//segment+1!=epoch:break
            rate=phase_and_rate(step,cfg)[2]
            for group in optimizer.param_groups:group['lr']=rate
            then=time.monotonic();batch=to_device(batch,device)
            stat=region_update(model,optimizer,batch,cfg['region']['loss'],grad_clip=cfg['training']['grad_clip'])
            if str(device).startswith('cuda'):torch.cuda.synchronize()
            step+=1;row=step_record(stat,step=step,learning_rate=rate,update_seconds=time.monotonic()-then,
                sources=batch['source'],draws=batch['draw'],elapsed_seconds=time.monotonic()-started)
            with (output/'steps.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if step%cfg['training']['log_every_updates']==0:print(json.dumps(row,ensure_ascii=False),flush=True)
            if step in cfg['region']['monitor_steps'] and step not in monitors:
                monitor(model,config,output,step,fixed);monitors.append(step)
            if step%cfg['training']['checkpoint_every_updates']==0 or step in monitors:
                save_checkpoint(output/'latest.pt',model,optimizer,cfg,identity,step,monitors)
            if step%250==0 or step==total:
                verify_inputs(identity,dataset)
                save_checkpoint(output/f'step_{step:06d}.pt',model,optimizer,cfg,identity,step,monitors)
                write_json(output/'status.json',dict(complete=step==total,global_step=step,total_updates=total,
                    model_sha256=state_sha256(model),no_full_test_inference=True))
    verify_inputs(identity,dataset)
    result=dict(complete=True,global_step=step,model_sha256=state_sha256(model),
        elapsed_seconds=time.monotonic()-started,monitor_steps=monitors,no_test_training=True,
        no_full_test_inference=True,deployment_replaced=False)
    write_json(output/'complete.json',result);print(json.dumps(dict(event='region_complete',**result)),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output');parser.add_argument('--device',default='cuda')
    parser.add_argument('--mode',choices=['preflight','smoke','train'],default='train')
    parser.add_argument('--smoke-updates',type=int,default=128);parser.add_argument('--resume')
    args=parser.parse_args(argv);config=load_config(args.config);cfg=validate_config(config)
    torch.set_num_threads(cfg['training']['torch_threads']);cv2.setNumThreads(cfg['training']['opencv_threads'])
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    output=args.output or project_path(config,cfg['output_dir'])
    if args.mode=='preflight':result=preflight(config,output)
    elif args.mode=='smoke':result=smoke(config,output,args.smoke_updates,args.device)
    else:result=train(config,output,args.device,args.resume)
    print(json.dumps(dict(complete=result.get('complete'),mode=args.mode)),flush=True)


if __name__=='__main__':main()
