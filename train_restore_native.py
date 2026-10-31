# -*- coding: utf-8 -*-
"""独立恢复器A/B训练：有效batch4、60轮日程分段停止、完整续训状态。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.restore_native import NativeRestorationDataset
from models.rgb_restoration import build_rgb_restorer, load_rgb_restorer
from models.backend_adaptation import restore_first
from train_rgb_diffusion_restoration import sample_timesteps, capture_rng, restore_rng
from train_rgb_restoration import reconstruction_loss, save_checkpoint, seed_worker, write_json
from tools.restore_native_monitor import monitor
from utils.config import load_config, project_path


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def state_sha(state):
    result=hashlib.sha256()
    for key,value in sorted(state.items()):
        result.update(key.encode());result.update(str((value.dtype,tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def append(path,row):
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def make_initial(config,path):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['rgb_restoration']['seed'])
        model=build_rgb_restorer(config['rgb_restoration']['model'])
    payload=dict(format=model.checkpoint_format,model_config=model.model_config,model=model.state_dict(),
                 initialization='scratch_shared',epoch=0,updates=0,model_sha256=state_sha(model.state_dict()))
    save_checkpoint(path,payload)
    return payload['model_sha256']


def recipe_signature(config):
    cfg=copy.deepcopy(config)
    cfg['paths']['project_root']='auto'
    cfg['rgb_restoration'].pop('output_dir',None)
    cfg['rgb_restoration']['train'].pop('num_workers',None)
    cfg['restore_native'].pop('stage_epochs',None)
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def saved_rng(noise,loader,device):
    state=capture_rng(noise,loader,device)
    state['python']=random.getstate()
    name,array,pos,has_gauss,cached=np.random.get_state()
    state['numpy']=[name,torch.from_numpy(array.copy()),pos,has_gauss,cached]
    return state


def load_rng(state,noise,loader,device):
    restore_rng(state,noise,loader,device);random.setstate(state['python'])
    name,array,pos,has_gauss,cached=state['numpy']
    np.random.set_state((name,array.numpy(),pos,has_gauss,cached))


def update(model,batch,optimizer,scaler,noise_rng,device,cfg,micro_batch):
    inputs,target,valid=[batch[k].to(device,non_blocking=True) for k in ('input','target','valid')]
    count=len(inputs)
    if count!=cfg['train']['batch_size'] or count%micro_batch:
        raise ValueError('Effective batch and accumulation must be exact')
    t=sample_timesteps(count,model.num_steps,device=device,generator=noise_rng,
                       policy=cfg['train']['timestep_sampling'])
    # 按完整逻辑batch抽噪声，与微批切分方式无关。
    xt=model.q_sample(target,inputs,t,generator=noise_rng)
    optimizer.zero_grad(set_to_none=True);model.train();total=0.;errors={}
    use_amp=bool(cfg['train']['amp']) and device.type=='cuda'
    dtype=torch.bfloat16 if device.type!='cuda' or torch.cuda.is_bf16_supported() else torch.float16
    for start in range(0,count,micro_batch):
        selection=slice(start,start+micro_batch)
        with torch.autocast(device_type=device.type,enabled=use_amp,dtype=dtype):
            prediction=model.denoise(xt[selection],inputs[selection],t[selection])
            loss,err=reconstruction_loss(prediction,target[selection],valid[selection],cfg['loss'],return_errors=True)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite reconstruction loss')
        scaler.scale(loss*(micro_batch/count)).backward()
        total+=float(loss.detach())*(micro_batch/count)
        for key,value in err.items():errors[key]=errors.get(key,0.)+float(value.detach().sum())/count
        del prediction,loss
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['grad_clip'],error_if_nonfinite=True)
    before=scaler.get_scale();scaler.step(optimizer);scaler.update()
    if scaler.get_scale()<before:raise FloatingPointError('Skipped optimizer update; refusing unmatched budgets')
    return dict(loss=total,errors=errors,grad_norm=float(norm),timesteps=t.tolist(),
                learning_rate=optimizer.param_groups[0]['lr'],samples=count,micro_batch=micro_batch)


def train(config,arm,root,*,device='cuda',stop=20,resume=False,smoke=False):
    root=Path(root);out=root/arm;cfg=config['rgb_restoration'];opt=config['restore_native'];device=torch.device(device)
    if not 1<=stop<=cfg['train']['epochs']:raise ValueError('Invalid stage endpoint')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU sam2_env required')
    if opt['micro_batch']<1 or cfg['train']['batch_size']%opt['micro_batch']:raise ValueError('Invalid micro batch')
    if out.exists() and not resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=build_rgb_restorer(cfg['model']).to(device)
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=True)
    if initial.get('initialization')!='scratch_shared' or initial['epoch'] or initial['updates']:
        raise ValueError('Only a fresh shared random initialization is accepted')
    model.load_state_dict(initial['model'],strict=True)
    if state_sha(model.state_dict())!=initial['model_sha256']:raise RuntimeError('Initial model mismatch')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['train']['learning_rate'],weight_decay=cfg['train']['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['train']['epochs'],eta_min=cfg['train']['min_learning_rate'])
    amp=bool(cfg['train']['amp']) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp and not torch.cuda.is_bf16_supported())
    noise_rng=torch.Generator(device=device).manual_seed(seed+271828)
    loader_rng=torch.Generator().manual_seed(seed)
    dataset=NativeRestorationDataset(config,arm)
    if len(dataset)!=cfg['expected_train_sources']:raise ValueError('Training pool does not contain all expected sources')
    if len(dataset)%cfg['train']['batch_size']:raise ValueError('No partial or dropped training batch permitted')
    source_ids=[Path(p).name for p in dataset.samples]
    pool_sha=hashlib.sha256(json.dumps(source_ids).encode()).hexdigest()
    completed=updates=0
    if resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if (saved.get('training_format')!='restore_native_v1' or saved['recipe_signature']!=recipe_signature(config)
                or saved['arm']!=arm or saved['source_pool_sha256']!=pool_sha or saved['smoke']!=smoke
                or saved['initial_sha256']!=initial['model_sha256']):
            raise ValueError('Resume contract differs')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        completed,updates=saved['epoch'],saved['updates'];load_rng(saved['rng'],noise_rng,loader_rng,device)
        if completed>=stop:raise ValueError('Requested endpoint must exceed completed epoch')
        # 不允许在部分epoch失败后悄悄重放并重复日志；当前只承诺完整epoch边界续训。
        if (out/'metrics.jsonl').exists():
            lines=(out/'metrics.jsonl').read_text(encoding='utf-8').splitlines()
            if len(lines)!=updates:raise ValueError('Partial-epoch log exists; recover explicitly before resuming')
    fixed=torch.load(root/'fixed.pt',map_location='cpu',weights_only=True)
    cached=torch.load(root/'old_fixed.pt',map_location='cpu',weights_only=True)
    status=dict(status='running',arm=arm,epoch=completed,updates=updates,schedule_epochs=cfg['train']['epochs'],
                stage_epochs=stop,source_count=len(dataset),effective_batch=cfg['train']['batch_size'],
                micro_batch=opt['micro_batch'],from_scratch=True,initial_sha256=initial['model_sha256'],
                source_pool_sha256=pool_sha,recipe_signature=recipe_signature(config),smoke=smoke,
                heldout_sources=0,checkpoint_selection='fixed_stage_endpoint',official_score=None)
    write_json(out/'run_config.json',config);write_json(out/'status.json',status)
    started=time.monotonic()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    def checkpoint(epoch,report):
        payload=dict(format=model.checkpoint_format,training_format='restore_native_v1',model_config=model.model_config,
            model=model.state_dict(),epoch=epoch,updates=updates,arm=arm,smoke=smoke,
            initial_sha256=initial['model_sha256'],source_pool_sha256=pool_sha,recipe_signature=recipe_signature(config),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            rng=saved_rng(noise_rng,loader_rng,device),experiment_config=config,metrics=report)
        save_checkpoint(out/'last.pt',payload)
        if epoch in cfg['train']['save_epochs'] or epoch==stop:
            # 历史轮只保存可部署小权重；完整续训状态保存在last.pt。
            slim={k:v for k,v in payload.items() if k not in ('optimizer','scheduler','scaler','rng')}
            save_checkpoint(out/f'epoch_{epoch:03d}.pt',slim)
    try:
        if not resume:
            monitor(model,fixed,cached,out,0,device,opt['monitor'])
            checkpoint(0,{})
        for epoch in range(completed,stop):
            dataset.set_epoch(epoch)
            order=torch.randperm(len(dataset),generator=loader_rng).tolist()
            if smoke:order=order[:16]
            data=DataLoader(Subset(dataset,order),batch_size=cfg['train']['batch_size'],shuffle=False,
                num_workers=cfg['train']['num_workers'],pin_memory=device.type=='cuda',worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed+epoch+12345))
            rows=[]
            for batch in data:
                rng_sha=hashlib.sha256(noise_rng.get_state().cpu().numpy().tobytes()).hexdigest()
                row=update(model,batch,optimizer,scaler,noise_rng,device,cfg,opt['micro_batch'])
                updates+=1;row.update(epoch=epoch+1,update=updates,sources=list(batch['source']),
                    recipes=[json.loads(s) for s in batch['recipe']],noise_rng_before=rng_sha)
                append(out/'metrics.jsonl',row);rows.append(row)
                if updates<=2 or updates%25==0:
                    print(json.dumps({k:row[k] for k in ('epoch','update','loss','grad_norm','learning_rate')}),flush=True)
            scheduler.step();completed=epoch+1
            report=dict(epoch=completed,updates=updates,epoch_updates=len(rows),samples=sum(r['samples'] for r in rows),
                loss=float(np.mean([r['loss'] for r in rows])),next_learning_rate=optimizer.param_groups[0]['lr'],
                native_sizes={str(size):sum(rp['native_size']==size for r in rows for rp in r['recipes']) for size in (1024,512)},
                kinds={kind:sum(rp['kind']==kind for r in rows for rp in r['recipes']) for kind in ('identity','uniform','spatial','endpoint')},
                exact_half_samples=sum(rp['downsample_scale']==.5 for r in rows for rp in r['recipes']))
            if completed in opt['monitor_epochs'] or completed==stop:
                monitor(model,fixed,cached,out,completed,device,opt['monitor'])
            checkpoint(completed,report);append(out/'epochs.jsonl',report)
            status.update(epoch=completed,updates=updates,elapsed_seconds=time.monotonic()-started,
                peak_memory_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None)
            write_json(out/'status.json',status)
        expected=stop*(16 if smoke else len(dataset))//cfg['train']['batch_size']
        if updates!=expected:raise RuntimeError(f'Unmatched effective update budget {updates}/{expected}')
        if state_sha(model.state_dict())==initial['model_sha256']:raise RuntimeError('No learned parameter changes')
        final=out/f'epoch_{stop:03d}.pt';reloaded=load_rgb_restorer(final,device)
        if state_sha(reloaded.state_dict())!=state_sha(model.state_dict()):raise RuntimeError('Strict reload differs')
        probe=fixed['pairs'][0]['input'][None].to(device)
        if not torch.equal(restore_first(model,probe,opt['monitor']['seed']),restore_first(reloaded,probe,opt['monitor']['seed'])):
            raise RuntimeError('Reload changed actual first-x0 inference')
        status.update(status='stage_complete',strict_reload_equal=True,first_inference_reload_equal=True,final_checkpoint_sha256=file_sha(final))
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));raise
    finally:
        write_json(out/'status.json',status)
    print(f'COMPLETE {arm} e{completed} updates={updates}',flush=True)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/restore_native.yaml')
    parser.add_argument('--arm',choices=['native','mixed'],required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--stop',type=int,default=20)
    parser.add_argument('--device',default='cuda');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();train(load_config(args.config),args.arm,args.root,device=args.device,stop=args.stop,resume=args.resume,smoke=args.smoke)
