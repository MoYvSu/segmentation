# -*- coding: utf-8 -*-
"""固定T8/连续N1双输入NAF；复用原损失、更新、存载和1000步循环。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import math
from pathlib import Path
import random
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_boundary_rgb as base
from data.boundary_dual import BoundaryDualDataset, collate_boundary_dual
from models.boundary_dual import BoundaryDualNAFNet
from models.boundary_naf import BoundaryDiscriminator
from models.boundary_restore import state_sha256
from tools.boundary_restore_monitor import panels, _heat
from tools.boundary_naf_monitor import _fixed_controls
from train_boundary_restore import capture_rng, restore_rng
from utils.boundary_input import normalize_boundary_input
from utils.config import load_config, project_path

FORMAT = 'naf_dual_training_v1'
CONTRACT = 't8_continuous_v1'
_TEST_CROPS = {}


def validate_config(config):
    cfg = config['boundary_naf']
    if (cfg.get('input_mode') != 'boundary_continuous'
            or cfg.get('input_contract') != CONTRACT
            or cfg['inference'].get('input_mode') != 'boundary_continuous'
            or cfg['inference'].get('input_contract') != CONTRACT
            or cfg['inference'].get('boundary_representation') != 'binary_t8_and_continuous_n1'):
        raise ValueError('双输入须明确[T8,continuous_N1]新合同')
    legacy = copy.deepcopy(config)
    old = legacy['boundary_naf']
    old['input_mode'] = old['inference']['input_mode'] = 'boundary'
    old['inference']['boundary_representation'] = 'binary'
    base_original_validate(legacy)
    t = cfg['training']
    if (cfg.get('input_normalization') != {'mode': 'skeleton_dilate', 'kernel_size': 8}
            or cfg['model'].get('upsampling', 'pixelshuffle') != 'pixelshuffle'
            or base.erasure_enabled(cfg) or base.shift_consistency_enabled(cfg)
            or base.ridge_supervision_enabled(cfg) or cfg.get('restored_rgb_manifest')
            or cfg.get('input_perturbation') or cfg['monitor'].get('full_readout_steps')
            or cfg['rgb_loss'] != {'tolerance': 2, 'coverage_weight': 1.0, 'interior_weight': 1.0}
            or any(t[k] != v for k, v in dict(main_updates=1000, segment_updates=1000,
                effective_batch_size=16, micro_batch_size=16, main_schedule_updates=53000,
                learning_rate=.001, discriminator_learning_rate=.0001,
                adversarial_weight=.01, adversarial_warmup_updates=64).items())):
        raise ValueError('本轮只加连续N1信息，不改变原1000步配方')
    return cfg


base_original_validate = base.validate_config


def build_models(cfg, device):
    g = BoundaryDualNAFNet(**cfg['model'], seed=int(cfg['seed']),
                           input_mode=cfg['input_mode']).to(device)
    d = BoundaryDiscriminator(**cfg['discriminator'], seed=int(cfg['seed']) + 1).to(device)
    return g, d


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _test_crops(config):
    cfg = config['boundary_naf']; settings = cfg['monitor']
    manifests=tuple((str(project_path(config,p)),base.file_sha(project_path(config,p)))
                    for p in settings.get('control_manifests', []))
    key=(manifests,tuple(settings.get('source_names',[])),
         tuple(sorted(cfg['input_normalization'].items())),cfg['data']['crop_size'])
    if key not in _TEST_CROPS:
        rows=[]
        for row, _ in _fixed_controls(config, settings):
            if base.file_sha(row['path']) != row['sha256']:raise RuntimeError('测试monitor缓存改变')
            with np.load(row['path'], allow_pickle=False) as z:
                boundary=z['boundary']; rgb=z['rgb']
                if (boundary.dtype != np.float32 or boundary.ndim != 2 or boundary.size > 6_000_000
                        or rgb.dtype != np.uint8 or rgb.shape != (*boundary.shape,3)):
                    raise ValueError('测试monitor单图缓存合同错误/尺寸超预算')
                t8, normalization = normalize_boundary_input(boundary, settings=cfg['input_normalization'])
                h,w=boundary.shape;size=cfg['data']['crop_size'];x,y=(w-size)//2,(h-size)//2
                if min(x,y)<0: raise ValueError('固定测试monitor必须能裁原生256')
                rows.append(dict(source=row['source'], path=row['path'], sha256=row['sha256'],
                    box=[x,y,size,size], normalization=normalization,
                    rgb=rgb[y:y+size,x:x+size].copy(),
                    continuous=boundary[y:y+size,x:x+size].copy(),
                    t8=t8[y:y+size,x:x+size].copy()))
        _TEST_CROPS[key]=rows
    return _TEST_CROPS[key]


def monitor(g, config, output, step, dataset, **_kwargs):
    cfg=config['boundary_naf'];settings=cfg['monitor'];folder=Path(output)/'monitor'/f's{step:06d}'
    folder.mkdir(parents=True,exist_ok=True)
    before,rng,mode=state_sha256(g),capture_rng(),g.training
    device=next(g.parameters()).device; rows=[]
    width=int(settings.get('thumbnail_width',280))
    try:
        g.eval()
        with torch.inference_mode():
            for draw in settings['pair_indices']:
                sample=dataset[draw]
                value=sample['input'][None].to(device)
                probability=g(value)[0,0].float().cpu().numpy()
                no_b=g(value,use_continuous=False)[0,0].float().cpu().numpy()
                rgb=np.rint(sample['image'].numpy().transpose(1,2,0)*255).astype(np.uint8)
                target=sample['target'][0].numpy();valid=sample['valid'][0].numpy()>.5
                t8=sample['input_boundary'][0].numpy();continuous=sample['input'][1].numpy()
                file=folder/f'draw_{draw:05d}.jpg'
                panels(file,f'{sample["source"]} | draw {draw} | step {step}',[
                    ('Original RGB',rgb),('Processed GT (known)',_heat(target*valid)),
                    ('Input T8',_heat(t8)),('Continuous N1',_heat(continuous)),
                    ('Dual output',_heat(probability)),('Same G, B branch off',_heat(no_b))],width,
                    footer='Same crop/D4/GT; branch-off is a mechanism probe, not separately trained control. Seen training source; unknown ignored.')
                rows.append(dict(domain='train',draw=draw,source=sample['source'],file=file.name,
                    crop=sample['box'],input_sha256=_sha(sample['input'].numpy()),
                    output_sha256=_sha(probability), branch_effect_mae=float(np.abs(probability-no_b).mean())))
            for row in _test_crops(config):
                if base.file_sha(row['path']) != row['sha256']:raise RuntimeError('测试monitor缓存改变')
                value=torch.from_numpy(np.stack([row['t8'],row['continuous']]).astype(np.float32))[None].to(device)
                probability=g(value)[0,0].float().cpu().numpy()
                no_b=g(value,use_continuous=False)[0,0].float().cpu().numpy()
                file=folder/(row['source']+'.jpg')
                panels(file,f'{row["source"]} | fixed native256 | step {step}',[
                    ('Original RGB',row['rgb']),('Continuous N1',_heat(row['continuous'])),
                    ('Input T8',_heat(row['t8'])),('Dual output',_heat(probability)),
                    ('Same G, B branch off',_heat(no_b))],width,
                    footer='Unlabeled test crop. No GT, WS, backend forward or accuracy claim.')
                rows.append(dict(domain='monitor_test',source=row['source'],file=file.name,
                    crop=row['box'],output_sha256=_sha(probability),
                    branch_effect_mae=float(np.abs(probability-no_b).mean())))
        if state_sha256(g)!=before:raise RuntimeError('过程monitor修改模型')
        result=dict(complete=True,step=step,rows=rows,no_WS=True,no_full_test_inference=True,
                    training_weight_sha_unchanged=True, no_test_GT=True)
        base.write_json(folder/'receipt.json',result)
        links=''.join(f'<p>{r["source"]}<br><img width="100%" src="{r["file"]}"></p>' for r in rows)
        (folder/'index.html').write_text('<meta charset="utf-8">'+links,encoding='utf-8')
        steps=sorted(p.name for p in (Path(output)/'monitor').iterdir() if p.is_dir())
        (Path(output)/'monitor'/'index.html').write_text('<meta charset="utf-8">'+''.join(
            f'<p><a href="{name}/index.html">{name}</a></p>' for name in steps),encoding='utf-8')
        return result
    finally:
        restore_rng(rng);g.train(mode)


@contextmanager
def adapted_trainer():
    original_identity, original_update = base.source_identity, base.update
    original_metrics, original_gan = base.reconstruction_metrics, base.rgb_gan_update
    templates={}
    def dataset_for(config,epoch):
        cfg=copy.deepcopy(config['boundary_naf'])
        cfg['real_input_manifest']=str(project_path(config,cfg['real_input_manifest']))
        key=str(project_path(config,cfg['manifest']))
        if key not in templates:
            templates[key]=BoundaryDualDataset(key,cfg,epoch=epoch)
            return templates[key]
        return templates[key].clone_for_epoch(epoch)
    def identity_for(config,dataset,g,d):
        result=original_identity(config,dataset,g,d)
        for name in ('tools/train_naf_dual.py','models/boundary_dual.py','data/boundary_dual.py'):
            result['source_sha256'][name]=base.file_sha(ROOT/name)
        result.update(experiment_format=FORMAT,input_contract=CONTRACT,
            common_initialization='same old random seed/state, not trained old NAF',
            continuous_branch='separate zero-init bias-free intro; skip uses T8 only',
            loss_unchanged=True, original_control_reused=True,
            dataset_cache_shared_readonly=True)
        return result
    def update(g,d,go,do,batch,cfg,step):
        result=original_update(g,d,go,do,batch,cfg,step)
        gradient=g.continuous_intro.weight.grad
        norm=float(gradient.detach().norm()) if gradient is not None else 0.
        weights=float(g.continuous_intro.weight.detach().norm())
        if not math.isfinite(norm+weights):raise FloatingPointError('连续输入分支非有限')
        return dict(result,continuous_intro_grad_norm=norm,continuous_intro_weights_l2=weights)
    def metrics(logits,batch,cfg,**kwargs):
        return original_metrics(logits,dict(batch,input=batch['input_boundary']),cfg,**kwargs)
    def gan(*args,**kwargs):
        return original_gan(*args,**kwargs,boundary_input_channel=0)
    replacements=dict(FORMAT=FORMAT,validate_config=validate_config,dataset_for=dataset_for,
        build_models=build_models,collate_boundary_rgb=collate_boundary_dual,
        source_identity=identity_for,update=update,reconstruction_metrics=metrics,
        rgb_gan_update=gan,monitor=monitor)
    with ExitStack() as stack:
        for name,value in replacements.items():stack.enter_context(patch.object(base,name,value))
        yield


def run(config,output,*,mode='train',device='cuda',resume=None,smoke_updates=80):
    cfg=validate_config(config)
    torch.set_num_threads(int(cfg['training']['torch_threads']));cv2.setNumThreads(int(cfg['training']['opencv_threads']))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    with adapted_trainer():
        if mode=='preflight':return base.preflight(config,output,device)
        if mode=='smoke':return base.smoke(config,output,device,smoke_updates)
        if mode=='train':return base.train(config,output,device,resume)
    raise ValueError('Unknown mode')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--device',default='cuda')
    p.add_argument('--mode',choices=('preflight','smoke','train'),default='train')
    p.add_argument('--resume');p.add_argument('--smoke-updates',type=int,default=80)
    args=p.parse_args(argv);config=load_config(args.config)
    run(config,args.output or project_path(config,config['boundary_naf']['output_dir']),mode=args.mode,
        device=args.device,resume=args.resume,smoke_updates=args.smoke_updates)


if __name__=='__main__':main()
