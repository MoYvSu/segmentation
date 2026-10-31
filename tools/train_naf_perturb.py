# -*- coding: utf-8 -*-
"""固定前端受扰热图训练NAF；复用旧1000更新损失/优化器，新增来源与过程图。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
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
from data.boundary_perturb import BoundaryPerturbDataset, collate_boundary_perturb, perturb_settings
from models.boundary_restore import state_sha256
from tools.boundary_restore_monitor import panels, _heat
from tools.boundary_naf_monitor import _fixed_controls
from train_boundary_restore import capture_rng, restore_rng
from utils.boundary_input import normalize_boundary_input
from utils.config import load_config, project_path


FORMAT = 'naf_perturbed_input_experiment_v1'
_TEST_MONITOR_CACHE = {}


def dataset_for(config, epoch):
    cfg = copy.deepcopy(config['boundary_naf'])
    cfg['real_input_manifest'] = str(project_path(config, cfg['real_input_manifest']))
    cfg['input_perturbation']['manifest'] = str(project_path(config, cfg['input_perturbation']['manifest']))
    return BoundaryPerturbDataset(project_path(config, cfg['manifest']), cfg, epoch=epoch)


def validate_config(config):
    cfg = base.validate_config(config)
    perturb_settings(cfg.get('input_perturbation'))
    if (cfg['input_mode'] != 'boundary'
            or cfg.get('input_normalization') != {'mode': 'skeleton_dilate', 'kernel_size': 8}
            or base.erasure_enabled(cfg) or base.ridge_supervision_enabled(cfg)
            or base.shift_consistency_enabled(cfg)):
        raise ValueError('本轮只改变冻结热图输入来源，T8、loss、旧NAF结构保持原义')
    if cfg['monitor'].get('full_readout_steps'):
        raise ValueError('本轮过程monitor只看固定缓存热图，不自动运行WS或完整读出')
    return cfg


def identity_for(config, dataset, g, d, *, original):
    identity = original(config, dataset, g, d)
    identity.update(experiment_format=FORMAT,
        input_change='independent clean/perturbed frozen N1 source per original epoch/draw; no extra sample budget',
        loss_unchanged=True, backend_or_restorer_loaded_for_training=False,
        initial_weights_not_from_old_trained_NAF=True)
    for name in ('data/boundary_perturb.py', 'tools/train_naf_perturb.py'):
        identity['source_sha256'][name] = base.file_sha(ROOT/name)
    path = str(dataset.perturb_manifest_path)
    identity['assets_sha256'][path] = dataset.perturb_manifest_sha256
    return identity


def update_audit(batch):
    records = batch.get('meta', [])
    if not records or any('input_variant' not in record for record in records):
        raise ValueError('受扰输入训练须逐样本保留variant身份')
    variants = Counter(record['input_variant'] for record in records)
    target_points = int(((batch['target'] > .5) & (batch['valid'] > .5)).sum())
    return dict(input_variant_counts=dict(sorted(variants.items())),
        new_missing_GT_points=sum(int(record['new_missing_gt_points']) for record in records),
        clean_missing_GT_points=sum(int(record['clean_gap_pixels']) for record in records),
        actual_missing_GT_points=sum(int(record['gap_pixels']) for record in records),
        supervised_GT_positive_points=target_points,
        new_missing_denominator='all GT-positive valid pixels in this actual training batch',
        variant_selection_not_based_on_GT=True)


def save_paired_monitor(g, config, output, step, dataset):
    """同一个原生裁块并排展示clean/受扰输入和输出，不重新运行前端。"""
    cfg = config['boundary_naf']
    settings = cfg['monitor']
    indices = list(settings.get('perturb_pair_indices', settings['pair_indices'][:4]))
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('受扰monitor须有固定不重复draw')
    folder = Path(output)/'monitor'/f's{int(step):06d}'/'perturb'
    folder.mkdir(parents=True, exist_ok=True)
    device = next(g.parameters()).device
    width = int(settings.get('thumbnail_width', 224))
    before, rng, mode = state_sha256(g), capture_rng(), g.training
    rows = []
    try:
        g.eval()
        with torch.inference_mode():
            for draw in indices:
                clean = dataset.sample_variant(draw, 'clean')
                clean_prediction = g(clean['input'][None].to(device))[0,0].detach().float().cpu().numpy()
                for variant in dataset.variant_ids:
                    sample = dataset.sample_variant(draw, variant)
                    prediction = g(sample['input'][None].to(device))[0,0].detach().float().cpu().numpy()
                    rgb = np.rint(sample['image'].numpy().transpose(1,2,0)*255).astype(np.uint8)
                    target = sample['target'][0].numpy()
                    valid = sample['valid'][0].numpy()>.5
                    cells = [('Original RGB',rgb),('Processed GT (valid)',_heat(target*valid)),
                        ('Clean T8',_heat(clean['input_boundary'][0].numpy())),
                        ('Perturbed T8',_heat(sample['input_boundary'][0].numpy())),
                        ('Clean output',_heat(clean_prediction)),('Perturbed output',_heat(prediction)),
                        ('New missing GT',_heat(sample['new_missing_gt_mask'][0].numpy()))]
                    file = folder/f'draw_{int(draw):05d}_{variant}.jpg'
                    panels(file,f'{sample["source"]} | draw {draw} | {variant} | step {step}',cells,width,
                        footer='Same crop/D4/GT. No backend forward, no test GT, no accuracy claim.')
                    rows.append(dict(source=sample['source'],draw=int(draw),variant=variant,
                        crop=sample['box'],meta=sample['meta'],file=file.name,
                        clean_output_sha256=hashlib.sha256(clean_prediction.tobytes()).hexdigest(),
                        perturbed_output_sha256=hashlib.sha256(prediction.tobytes()).hexdigest()))
        if state_sha256(g) != before:
            raise RuntimeError('受扰monitor修改了NAF状态')
        result = dict(complete=True, format=FORMAT, step=int(step), rows=rows,
            fixed_epoch=dataset.epoch,no_backend_forward=True,no_GT_based_variant_selection=True,
            input_and_GT_same_geometry=True,source_plan_unchanged=True)
        base.write_json(folder/'receipt.json',result)
        return result
    finally:
        restore_rng(rng);g.train(mode)


def save_test_monitor(g, config, output, step):
    """固定原测试缓存的原生256窗，只显示输入和NAF输出，不读测试GT或运行WS。"""
    cfg=config['boundary_naf'];settings=cfg['monitor']
    roots=tuple((str(project_path(config,p)),base.file_sha(project_path(config,p)))
                for p in settings.get('control_manifests',[]))
    names=tuple(settings.get('source_names',[]))
    if not roots:
        return dict(complete=True,rows=[],no_test_cache_configured=True)
    key=(roots,names,tuple(sorted(cfg['input_normalization'].items())))
    if key not in _TEST_MONITOR_CACHE:
        cached=[]
        for row,_ in _fixed_controls(config,settings):
            with np.load(row['path'],allow_pickle=False) as z:
                rgb=z['rgb'].copy();boundary=z['boundary'].copy()
            if rgb.dtype!=np.uint8 or rgb.shape!=(*boundary.shape,3):
                raise ValueError('测试monitor缓存的RGB/边界图坐标或dtype错误')
            normalized,normalization=normalize_boundary_input(boundary,settings=cfg['input_normalization'])
            h,w=boundary.shape;size=cfg['data']['crop_size']
            if h<size or w<size:
                raise ValueError('本轮测试monitor只接受可直接原生裁256的缓存')
            x,y=(w-size)//2,(h-size)//2
            cached.append(dict(source=row['source'],crop_xywh=[x,y,size,size],
                rgb=np.ascontiguousarray(rgb[y:y+size,x:x+size]),
                raw=np.ascontiguousarray(boundary[y:y+size,x:x+size]),
                normalized=np.ascontiguousarray(normalized[y:y+size,x:x+size],dtype=np.float32),
                cache_file=row['path'],cache_sha256=row['sha256'],normalization=normalization))
        _TEST_MONITOR_CACHE[key]=cached
    cached=_TEST_MONITOR_CACHE[key]
    folder=Path(output)/'monitor'/f's{int(step):06d}'/'real_fixed'
    folder.mkdir(parents=True,exist_ok=True)
    before,rng,mode=state_sha256(g),capture_rng(),g.training
    rows=[]
    try:
        g.eval()
        device=next(g.parameters()).device
        with torch.inference_mode():
            for row in cached:
                if base.file_sha(row['cache_file'])!=row['cache_sha256']:
                    raise RuntimeError('固定测试monitor缓存改变：'+row['source'])
                value=torch.from_numpy(row['normalized'].copy())[None,None].to(device)
                probability=g(value)[0,0].detach().float().cpu().numpy()
                file=folder/(row['source']+'.jpg')
                panels(file,f'{row["source"]} | fixed original cache center | step {step}',
                    [('Original RGB',row['rgb']),('Frozen N1',_heat(row['raw'])),
                     ('Input T8',_heat(row['normalized'])),('NAF output',_heat(probability))],
                    int(settings.get('thumbnail_width',224)),
                    footer='Unlabeled fixed native crop; no augmentation, test GT or final readout.')
                rows.append({k:v for k,v in row.items() if k not in ('rgb','raw','normalized')})
                rows[-1].update(file=file.name,input_array_sha256=hashlib.sha256(row['normalized'].tobytes()).hexdigest(),
                    output_sha256=hashlib.sha256(probability.tobytes()).hexdigest())
        if state_sha256(g)!=before:raise RuntimeError('测试过程图修改了NAF权重')
        result=dict(complete=True,step=int(step),rows=rows,no_test_GT=True,no_test_augmentation=True,
            no_WS=True,no_backend_forward=True,crop_before_NAF='whole-source T8 then fixed center256',
            no_final_inference_or_accuracy_claim=True)
        base.write_json(folder/'receipt.json',result)
        return result
    finally:
        restore_rng(rng);g.train(mode)


@contextmanager
def adapted_trainer():
    """有界替换数据/审计/monitor钩子；原优化器、loss、更新和严格续训代码不变。"""
    original_identity, original_update, original_monitor = base.source_identity, base.update, base.monitor
    def source_identity(config, dataset, g, d):
        return identity_for(config,dataset,g,d,original=original_identity)
    def update(g,d,go,do,batch,cfg,step):
        return dict(original_update(g,d,go,do,batch,cfg,step), **update_audit(batch))
    def monitor(g,config,output,step,dataset,**kwargs):
        ordinary = original_monitor(g,config,output,step,dataset,**kwargs)
        paired = save_paired_monitor(g,config,output,step,dataset)
        real=save_test_monitor(g,config,output,step)
        return dict(original=ordinary,perturbed_pairs=paired,real_fixed=real)
    with patch.object(base,'dataset_for',dataset_for), \
            patch.object(base,'source_identity',source_identity), \
            patch.object(base,'collate_boundary_rgb',collate_boundary_perturb), \
            patch.object(base,'update',update), \
            patch.object(base,'monitor',monitor):
        yield


def run(config, output, *, mode='train', device='cuda', resume=None, smoke_updates=128):
    cfg=validate_config(config)
    torch.set_num_threads(int(cfg['training']['torch_threads']))
    cv2.setNumThreads(int(cfg['training']['opencv_threads']))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    with adapted_trainer():
        if mode=='preflight':return base.preflight(config,output,device)
        if mode=='smoke':return base.smoke(config,output,device,smoke_updates)
        if mode=='train':return base.train(config,output,device,resume)
        raise ValueError('Unknown training mode')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--mode',choices=('preflight','smoke','train'),default='train')
    parser.add_argument('--resume')
    parser.add_argument('--smoke-updates',type=int,default=128)
    args=parser.parse_args(argv)
    config=load_config(args.config)
    output=args.output or project_path(config,config['boundary_naf']['output_dir'])
    result=run(config,output,mode=args.mode,device=args.device,resume=args.resume,smoke_updates=args.smoke_updates)
    print(json.dumps(dict(mode=args.mode,complete=result.get('complete')),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
