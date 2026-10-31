# -*- coding: utf-8 -*-
"""独立NAFNet边界恢复：随机初始化、GT配对、GAN主训后PH微调。

现有分割模型不加载到训练图中；可显式使用封存N1热图作输入，GT始终为目标。默认不做
全测试集推理或提交，也不实现原论文的迭代扩展回退。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.boundary_naf import BoundaryNAFDataset, collate_boundary_naf
from models.boundary_naf import BoundaryNAFNet, BoundaryDiscriminator
from models.boundary_restore import state_sha256
from train_boundary_restore import capture_rng, restore_rng, canonical_sha, file_sha, write_json, reconcile_log
from utils.boundary_naf_training import gan_update
from utils.boundary_topology import BoundaryTopologyLoss
from utils.config import load_config, project_path

FORMAT = 'boundary_naf_training_v1'


def validate_config(config):
    cfg = config['boundary_naf']
    t = cfg['training']
    if t['precision'] != 'float32':
        raise ValueError('本阶段只允许明确的FP32数值协议')
    for key in ('main_updates', 'topology_updates', 'main_schedule_updates',
                'topology_schedule_updates', 'segment_updates', 'effective_batch_size',
                'micro_batch_size', 'checkpoint_every_updates'):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError('Expected positive integer: '+key)
    if t['main_updates'] > t['main_schedule_updates'] or t['topology_updates'] > t['topology_schedule_updates']:
        raise ValueError('阶段更新预算不能超过预先声明日程')
    if t['effective_batch_size'] % t['micro_batch_size']:
        raise ValueError('micro batch必须整除有效batch')
    if t['segment_updates'] * t['effective_batch_size'] != cfg['data']['samples_per_epoch']:
        raise ValueError('每段采样数与更新预算不一致')
    inf = cfg['inference']
    if inf['iterations'] != 1 or inf['revert_unconnected'] is not False:
        raise ValueError('本实验不加入迭代或未连接扩展回退')
    if inf['input_mode'] not in ('binary', 'continuous', 'skeleton') or not 0 < inf['input_threshold'] < 1:
        raise ValueError('边界接口必须显式定义')
    source=cfg.get('input_source','synthetic_gt')
    if source not in ('synthetic_gt','real_n1'):
        raise ValueError('Unknown NAF input source')
    if source=='real_n1' and (inf['input_mode']!='skeleton' or not cfg.get('real_input_manifest')):
        raise ValueError('真实N1配对必须显式使用整图细化输入及封存热图清单')
    if int(cfg['expected_sources']) != 32:
        raise ValueError('本实验复用全部32份处理后GT')
    return cfg


def build_models(cfg, device):
    generator = BoundaryNAFNet(**cfg['model'], seed=int(cfg['seed'])).to(device)
    discriminator = BoundaryDiscriminator(**cfg['discriminator'], seed=int(cfg['seed'])+1).to(device)
    return generator, discriminator


def optimizer_for(module, cfg):
    t = cfg['training']
    return torch.optim.Adam(module.parameters(), lr=float(t['learning_rate']),
                            betas=tuple(t['adam_betas']), eps=float(t['adam_eps']))


def phase_and_rate(step, cfg):
    t = cfg['training']
    main = int(t['main_updates'])
    if step < main:
        phase, local = 'main', step
        initial, horizon = float(t['learning_rate']), int(t['main_schedule_updates'])
    else:
        phase, local = 'topology', step-main
        initial, horizon = float(t['topology_learning_rate']), int(t['topology_schedule_updates'])
    floor = float(t['minimum_lr_ratio'])
    rate = initial*(floor+(1-floor)*.5*(1+math.cos(math.pi*local/max(1,horizon-1))))
    return phase, local, rate


def source_identity(config, dataset, generator, discriminator):
    cfg = config['boundary_naf']
    root = Path(__file__).resolve().parent
    sources = ['train_boundary_naf.py', 'models/boundary_naf.py', 'models/rgb_restoration.py',
               'models/boundary_restore.py', 'data/boundary_naf.py', 'data/boundary_restore.py',
               'utils/boundary_topology.py', 'utils/boundary_naf_training.py',
               'tools/boundary_naf_monitor.py', 'tools/boundary_restore_readout.py',
               'tools/boundary_restore_monitor.py', 'train_boundary_restore.py', 'utils/config.py']
    if cfg.get('input_source')=='real_n1':
        sources.append('data/boundary_naf_real.py')
    monitor = cfg.get('monitor', {})
    assets = {}
    for value in monitor.get('control_manifests', []):
        p = Path(project_path(config, value)); assets[str(p)] = file_sha(p)
    if monitor.get('roi_spec'):
        p = Path(project_path(config, monitor['roi_spec'])); assets[str(p)] = file_sha(p)
    if cfg.get('input_source')=='real_n1':
        p=Path(project_path(config,cfg['real_input_manifest']));assets[str(p)]=file_sha(p)
    return dict(format=FORMAT, purpose='formal_candidate',generator_architecture=copy.deepcopy(generator.architecture),
                discriminator_architecture=copy.deepcopy(discriminator.architecture),
                initial_generator_sha256=state_sha256(generator),
                initial_discriminator_sha256=state_sha256(discriminator),
                dataset=copy.deepcopy(dataset.identity),
                recipe={k:copy.deepcopy(cfg[k]) for k in ('seed','model','discriminator','data','training','topology','inference','monitor',
                    'input_source','real_input_manifest','real_input_models') if k in cfg},
                source_sha256={n:file_sha(root/n) for n in sources},
                input_manifest_sha256=file_sha(project_path(config,cfg['manifest'])),
                monitor_asset_sha256=assets, initialization='random_no_supervised_or_restorer_weights',
                target='processed_GT_binary_boundaries_not_N0', no_test_training=True,
                frozen_backend_not_loaded_for_training=True, no_revert_unconnected=True,
                library_versions=dict(torch=torch.__version__, numpy=np.__version__,opencv=cv2.__version__,
                    gudhi=importlib.metadata.version('gudhi'),scipy=importlib.metadata.version('scipy')))


def save_checkpoint(path, generator, discriminator, g_opt, d_opt, cfg, identity, step, monitors):
    payload = dict(format=FORMAT, architecture=generator.architecture,
        model_state_dict={k:v.detach().cpu().contiguous() for k,v in generator.state_dict().items()},
        discriminator_architecture=discriminator.architecture,
        discriminator_state_dict={k:v.detach().cpu().contiguous() for k,v in discriminator.state_dict().items()},
        model_sha256=state_sha256(generator), discriminator_sha256=state_sha256(discriminator),
        optimizer=g_opt.state_dict(), discriminator_optimizer=d_opt.state_dict(),
        identity=identity, identity_sha256=canonical_sha(identity), global_step=int(step),
        completed_phase=phase_and_rate(step-1,cfg)[0] if step else None,
        next_phase=phase_and_rate(step,cfg)[0],
        next_scheduler=phase_and_rate(step,cfg), rng=capture_rng(), monitor_steps=list(monitors),
        input_contract=copy.deepcopy(cfg['inference']), generator_parameters=sum(p.numel() for p in generator.parameters()),
        discriminator_parameters=sum(p.numel() for p in discriminator.parameters()),
        config=copy.deepcopy(cfg))
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,tmp); tmp.replace(path)


def load_checkpoint(path, generator, discriminator, g_opt, d_opt, identity, cfg):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format')!=FORMAT or saved.get('identity')!=identity or saved.get('identity_sha256')!=canonical_sha(identity):
        raise ValueError('续训来源、配方或代码身份不一致')
    step=saved.get('global_step'); total=cfg['training']['main_updates']+cfg['training']['topology_updates']
    if type(step) is not int or not 0<=step<=total:
        raise ValueError('续训更新次数不在本阶段范围')
    if (saved.get('architecture')!=generator.architecture
            or saved.get('discriminator_architecture')!=discriminator.architecture
            or saved.get('input_contract')!=cfg['inference']
            or saved.get('completed_phase')!=(phase_and_rate(step-1,cfg)[0] if step else None)
            or saved.get('next_phase')!=phase_and_rate(step,cfg)[0]):
        raise ValueError('checkpoint架构、输入接口或训练阶段不一致')
    generator.load_state_dict(saved['model_state_dict'],strict=True)
    discriminator.load_state_dict(saved['discriminator_state_dict'],strict=True)
    if state_sha256(generator)!=saved['model_sha256'] or state_sha256(discriminator)!=saved['discriminator_sha256']:
        raise RuntimeError('严格重载模型SHA不一致')
    g_opt.load_state_dict(saved['optimizer']); d_opt.load_state_dict(saved['discriminator_optimizer'])
    if tuple(saved['next_scheduler'])!=phase_and_rate(int(saved['global_step']),cfg):
        raise ValueError('续训日程与更新位置不一致')
    restore_rng(saved['rng'])
    return int(saved['global_step']),list(saved['monitor_steps'])


def dataset_for(config,epoch):
    cfg=config['boundary_naf']
    if cfg.get('input_source')=='real_n1':
        from data.boundary_naf_real import BoundaryNAFRealDataset
        resolved=copy.deepcopy(cfg)
        resolved['real_input_manifest']=str(project_path(config,cfg['real_input_manifest']))
        return BoundaryNAFRealDataset(project_path(config,cfg['manifest']),resolved,epoch=epoch)
    return BoundaryNAFDataset(project_path(config,cfg['manifest']),config,epoch=epoch)


def _loader(dataset, first, cfg):
    t=cfg['training']; workers=int(t['num_workers'])
    if workers<0:
        raise ValueError('Invalid workers')
    generator=torch.Generator().manual_seed(int(cfg['seed'])+int(dataset.epoch)+117)
    kwargs=dict(batch_size=int(t['effective_batch_size']),shuffle=False,
        collate_fn=collate_boundary_naf,num_workers=workers,pin_memory=torch.cuda.is_available(),
        drop_last=True,generator=generator)
    if workers:
        kwargs.update(prefetch_factor=int(t['prefetch_factor']),persistent_workers=False)
    return DataLoader(Subset(dataset,range(first,len(dataset))),**kwargs)


def verify_fixed_inputs(config, identity, dataset):
    dataset.verify_sources()
    root=Path(__file__).resolve().parent
    for n,h in identity['source_sha256'].items():
        if file_sha(root/n)!=h:
            raise RuntimeError('封存运行源码改变: '+n)
    for p,h in identity['monitor_asset_sha256'].items():
        if file_sha(p)!=h:
            raise RuntimeError('控制资产清单改变: '+p)


def monitor(generator,config,output,step,dataset,**kwargs):
    from tools.boundary_naf_monitor import save_monitor
    before=state_sha256(generator); rng=capture_rng(); mode=generator.training
    try:
        generator.eval()
        with torch.no_grad():
            result=save_monitor(generator,config,output,step,dataset,**kwargs)
        if state_sha256(generator)!=before:
            raise RuntimeError('monitor修改生成器参数或buffer')
        return result
    finally:
        restore_rng(rng); generator.train(mode)


def topology_object(cfg):
    return BoundaryTopologyLoss(**cfg['topology'])


def preflight(config,output,device='cpu'):
    cfg=validate_config(config); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    dataset=dataset_for(config,0); g,d=build_models(cfg,device)
    identity=source_identity(config,dataset,g,d)
    batch=collate_boundary_naf([dataset[i] for i in range(min(32,len(dataset)))])
    valid=batch['valid'].bool(); target=batch['target']
    dmask=d.valid_patch_mask(valid.to(device)).detach().cpu()
    if not valid.any() or not dmask.any() or not batch['gap_mask'].any():
        raise RuntimeError('真实训练预检缺少BCE、GAN或GT缺失线诊断位置')
    topo=topology_object(cfg)
    _, audit=topo((target*.8+.05).clamp(0,1),target,valid)
    # 固定GT配对上的PH暴露普查，不用测试图调整配方。
    surveys=[]
    for size in (64,128,192,224):
        obj=BoundaryTopologyLoss(tile_size=size,max_tiles_per_image=1,border_margin=2,stride=32)
        _, stat=obj((target*.8+.05).clamp(0,1),target,valid)
        surveys.append(dict(tile_size=size,**stat))
    result=dict(complete=True,kind='CPU_real_data_preflight_not_training',
        generator_parameters=sum(p.numel() for p in g.parameters()),
        discriminator_parameters=sum(p.numel() for p in d.parameters()),
        initialization_identity=identity, plan=dataset.plan_receipt(),
        sample_count=int(target.shape[0]), valid_pixels=int(valid.sum()),
        gap_pixels=int(batch['gap_mask'].sum()), adversarial_patches=int(dmask.sum()),
        topology=audit,topology_size_survey=surveys, no_parameter_updates=True)
    verify_fixed_inputs(config,identity,dataset)
    write_json(output/'preflight.json',result)
    return result


def smoke(config,output,device='cuda',updates=3):
    if updates<2: raise ValueError('短测须至少包含一次主训与一次拓扑更新')
    config=copy.deepcopy(config)
    config['boundary_naf']['training']['main_updates']=updates-1
    config['boundary_naf']['training']['topology_updates']=1
    cfg=validate_config(config); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    dataset=dataset_for(config,0); g,d=build_models(cfg,device)
    go,do=optimizer_for(g,cfg),optimizer_for(d,cfg)
    samples=[dataset[i] for i in range(int(cfg['training']['effective_batch_size']))]
    batch=collate_boundary_naf(samples)
    batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    originals=None
    if cfg.get('input_source')=='real_n1':
        from tools.boundary_naf_monitor import original_view
        originals=[original_view(dataset,sample) for sample in samples]
    initial=state_sha256(g); initial_d=state_sha256(d); stats=[]
    topo=topology_object(cfg)
    def quality():
        mode=g.training; g.eval()
        with torch.no_grad():
            logits=g.forward_logits(batch['input']); prob=logits.sigmoid(); valid=batch['valid'].bool()
            gap=batch['gap_mask'].bool()&valid; positive=(batch['target']>.5)&valid; background=(batch['target']<.5)&valid
            bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['target'],reduction='none')
            value=dict(known_bce=float(bce[valid].mean()),gap_pixels=int(gap.sum()),
                gap_probability=float(prob[gap].mean()) if gap.any() else None,
                gap_mae=float((prob-batch['target']).abs()[gap].mean()) if gap.any() else None,
                boundary_probability=float(prob[positive].mean()) if positive.any() else None,
                background_probability=float(prob[background].mean()) if background.any() else None)
            if originals is not None:
                from tools.boundary_naf_monitor import real_pair_metrics
                rows=[];probabilities=prob.detach().float().cpu().numpy()[:,0]
                for sample,prediction,original in zip(samples,probabilities,originals):
                    row,_=real_pair_metrics(sample['input'][0].numpy(),sample['target'][0].numpy(),
                        prediction,sample['valid'][0].numpy()>0,original)
                    rows.append(row)
                checks={}
                for name in ['supported_gt','natural_missing_gt','original_deep_input']:
                    count=sum(row[name+'_pixels'] for row in rows)
                    checks[name+'_pixels']=count
                    for suffix in ['probability','above065']:
                        checks[name+'_'+suffix]=(sum(row[name+'_'+suffix]*row[name+'_pixels']
                            for row in rows if row[name+'_pixels'])/count) if count else None
                value['real_input_checks']=checks
        g.train(mode); return value
    before_quality=quality()
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for step in range(updates):
        started=time.perf_counter()
        stat=gan_update(g,d,go,do,batch,micro_batch_size=cfg['training']['micro_batch_size'],
            bce_weight=cfg['training']['bce_weight'],adversarial_weight=cfg['training']['adversarial_weight'],
            topology_fn=(topo if step==updates-1 else None),
            topology_weight=(cfg['training']['topology_weight'] if step==updates-1 else 0.),
            grad_clip=cfg['training']['grad_clip'])
        if str(device).startswith('cuda'): torch.cuda.synchronize()
        stats.append(dict(step=step+1,phase='smoke_topology' if step==updates-1 else 'smoke_main',seconds=time.perf_counter()-started,**stat))
        if step in (0,updates-1) or (step+1)%10==0:
            print(json.dumps(dict(event='smoke_update',step=step+1,seconds=stats[-1]['seconds'],
                                 bce=stat['bce'],adv_g=stat['adv_g'],topology_loss=stat['topology_loss'])),flush=True)
    after_quality=quality()
    if initial==state_sha256(g):
        raise RuntimeError('短测未实际更新G')
    if cfg['training']['adversarial_weight']>0 and initial_d==state_sha256(d):
        raise RuntimeError('已启用GAN的短测未实际更新D')
    if cfg['training']['adversarial_weight']==0 and initial_d!=state_sha256(d):
        raise RuntimeError('纯重建短测意外改变D')
    with torch.no_grad():
        g.eval(); expected=g(batch['input'][:2].to(device)).detach().cpu()
    identity=source_identity(config,dataset,*build_models(cfg,'cpu'))
    identity['purpose']='smoke_only_not_formal_resume'
    save_checkpoint(output/'smoke.pt',g,d,go,do,cfg,identity,updates,[])
    fresh,fresh_d=build_models(cfg,device)
    fresh_go,fresh_do=optimizer_for(fresh,cfg),optimizer_for(fresh_d,cfg)
    load_checkpoint(output/'smoke.pt',fresh,fresh_d,fresh_go,fresh_do,identity,cfg)
    fresh.eval()
    with torch.no_grad(): actual=fresh(batch['input'][:2].to(device)).detach().cpu()
    if not torch.equal(expected,actual):
        raise RuntimeError('实际GPU生成器保存／严格重载后的概率不一致')
    # 只对固定少图做实际首步/整图读出，不触碰后台模型或150图输出。
    result=monitor(fresh,config,output,updates,dataset,force_full=True,source_limit=2)
    receipt=dict(complete=True,kind='actual_GPU_smoke_not_formal_weights',device=str(device),
        updates=updates,steps=stats,strict_reload_exact=True,
        generator_initial_sha256=initial,generator_after_sha256=state_sha256(g),
        discriminator_initial_sha256=initial_d,discriminator_after_sha256=state_sha256(d),
        max_memory_allocated=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
        max_memory_reserved=torch.cuda.max_memory_reserved() if str(device).startswith('cuda') else 0,
        quality_before=before_quality,quality_after=after_quality,fixed_monitor=result)
    write_json(output/'smoke.json',receipt)
    return receipt


def train(config,output,device='cuda',resume=None):
    cfg=validate_config(config); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    dataset=dataset_for(config,1); fixed=dataset_for(config,0)
    g,d=build_models(cfg,device); go,do=optimizer_for(g,cfg),optimizer_for(d,cfg)
    identity=source_identity(config,dataset,g,d)
    total=cfg['training']['main_updates']+cfg['training']['topology_updates']
    segment=int(cfg['training']['segment_updates']); batch_size=int(cfg['training']['effective_batch_size'])
    state_path=output/'identity.json'
    if state_path.exists() and json.loads(state_path.read_text(encoding='utf-8'))!=identity:
        raise ValueError('输出目录属于不同实验，拒绝覆盖')
    if not resume and (output/'latest.pt').exists():
        raise ValueError('已有checkpoint；请显式续训，不能重新随机训练覆盖')
    write_json(state_path,identity); write_json(output/'config.json',config)
    step,monitors=load_checkpoint(resume,g,d,go,do,identity,cfg) if resume else (0,[])
    log_path=output/'steps.jsonl'
    if log_path.exists():
        reconcile_log(output,step)
    topology=topology_object(cfg); started=time.perf_counter()
    if step==0 and 0 not in monitors:
        monitor(g,config,output,0,fixed); monitors.append(0)
        save_checkpoint(output/'initial.pt',g,d,go,do,cfg,identity,0,monitors)
    print(json.dumps(dict(event='formal_started',step=step,total_updates=total,
        main_updates=cfg['training']['main_updates'],topology_updates=cfg['training']['topology_updates'],
        generator_parameters=sum(p.numel() for p in g.parameters()),
        discriminator_parameters=sum(p.numel() for p in d.parameters())),ensure_ascii=False),flush=True)
    while step<total:
        epoch=step//segment+1; dataset.set_epoch(epoch)
        first=(step%segment)*batch_size
        write_json(output/f'plan_{epoch:03d}.json',dataset.plan_receipt())
        for batch in _loader(dataset,first,cfg):
            if step>=total or step//segment+1!=epoch: break
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            phase,local,lr=phase_and_rate(step,cfg)
            for opt in (go,do):
                for group in opt.param_groups: group['lr']=lr
            then=time.perf_counter()
            kwargs=dict(micro_batch_size=cfg['training']['micro_batch_size'],
                bce_weight=cfg['training']['bce_weight'],adversarial_weight=cfg['training']['adversarial_weight'],
                topology_fn=(lambda p,t,v:topology(p,t,v,step=step)) if phase=='topology' else None,
                topology_weight=cfg['training']['topology_weight'] if phase=='topology' else 0.,
                grad_clip=cfg['training']['grad_clip'])
            stat=gan_update(g,d,go,do,batch,**kwargs)
            step+=1
            # 准确计时：GPU同步成本明确纳入，不只记录异步提交耗时。
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            row=dict(global_step=step,phase=phase,phase_update=local+1,learning_rate=lr,
                update_seconds=time.perf_counter()-then,elapsed_seconds=time.perf_counter()-started,
                sources=list(batch['source']) if 'source' in batch else [r['source'] for r in batch.get('views',[])],**stat)
            with log_path.open('a',encoding='utf-8') as out: out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            if step%cfg['training']['log_every_updates']==0:
                print(json.dumps({k:row[k] for k in ['global_step','phase','learning_rate','update_seconds']}|
                    {k:row[k] for k in ['loss_g','bce','adv_g','adv_d','topology_loss'] if k in row}),flush=True)
            at_monitor=step%segment==0 or step in (1,100, total)
            if at_monitor and step not in monitors:
                monitor(g,config,output,step,fixed); monitors.append(step)
            if step%cfg['training']['checkpoint_every_updates']==0 or at_monitor or step==total:
                save_checkpoint(output/'latest.pt',g,d,go,do,cfg,identity,step,monitors)
            if step%segment==0 or step==total:
                verify_fixed_inputs(config,identity,dataset)
                save_checkpoint(output/f'step_{step:06d}.pt',g,d,go,do,cfg,identity,step,monitors)
                write_json(output/'status.json',dict(complete=step==total,global_step=step,total_updates=total,
                    phase=phase,monitor_steps=monitors,model_sha256=state_sha256(g),
                    elapsed_seconds=time.perf_counter()-started,no_full_test_inference=True))
    verify_fixed_inputs(config,identity,dataset)
    result=dict(complete=True,global_step=step,main_updates=cfg['training']['main_updates'],
        topology_updates=cfg['training']['topology_updates'],monitor_steps=monitors,
        generator_sha256=state_sha256(g),discriminator_sha256=state_sha256(d),
        elapsed_seconds=time.perf_counter()-started,identity_sha256=canonical_sha(identity),
        no_test_training=True,no_revert_unconnected=True,deployment_replaced=False)
    write_json(output/'complete.json',result)
    print(json.dumps(dict(event='formal_complete',**result)),flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--mode',choices=['preflight','smoke','train'],default='train')
    parser.add_argument('--resume')
    parser.add_argument('--smoke-updates',type=int,default=3)
    args=parser.parse_args(argv)
    config=load_config(args.config); cfg=validate_config(config)
    torch.set_num_threads(int(cfg['training']['torch_threads']))
    cv2.setNumThreads(int(cfg['training']['opencv_threads']))
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    output=args.output or project_path(config,cfg['output_dir'])
    if args.mode=='preflight': result=preflight(config,output,device=args.device)
    elif args.mode=='smoke': result=smoke(config,output,device=args.device,updates=args.smoke_updates)
    else: result=train(config,output,device=args.device,resume=args.resume)
    print(json.dumps(dict(mode=args.mode,complete=result.get('complete')),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
