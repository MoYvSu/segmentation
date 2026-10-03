# -*- coding: utf-8 -*-
"""固定四源原生采集模糊压力；只读S-align/raw，不训练或重新划分实例。"""
from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.probe_semantic_raw_stress import (
    INITIAL, build_cohort, file_sha, probability_rows, read, save_preview,
    source_order, summarize_condition, write,
)

FORMAT='semantic_acquisition_v1'
CONDITIONS=('clean','gaussian3','gaussian8','spatial3_8','spatial3_8_global')


def native_blur_variants(rgb, *, seed, target_size=1024, sigmas=(3.,8.), transition=(.2,.8)):
    """原生RGB先模糊，再letterbox；两高斯凸组合不是单一高斯PSF。"""
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or min(rgb.shape[:2])<2:
        raise ValueError('Expected nonempty native uint8 RGB')
    if tuple(sigmas)!=(3.,8.) or tuple(transition)!=(.2,.8) or target_size!=1024:
        raise ValueError('Fixed bounded native blur recipe required')
    height,width=rgb.shape[:2]
    clean=rgb.astype(np.float32)/255.
    low=cv2.GaussianBlur(clean,(0,0),sigmaX=3.,sigmaY=3.,borderType=cv2.BORDER_REFLECT)
    high=cv2.GaussianBlur(clean,(0,0),sigmaX=8.,sigmaY=8.,borderType=cv2.BORDER_REFLECT)
    roundoff=dict(gaussian3=int(((low<0)|(low>1)).sum()),
                  gaussian8=int(((high<0)|(high>1)).sum()))
    # 正核不应改变RGB范围；仅消除float32卷积的边缘舍入越界。
    low=np.clip(low,0,1);high=np.clip(high,0,1)
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),20261004,137]))
    angle=float(rng.uniform(0,2*np.pi))
    dy,dx=np.sin(angle),np.cos(angle)
    # 固定方向的大尺度平滑场，端点不产生硬拼接；不根据任何预测决定方向或幅度。
    projection=np.arange(height,dtype=np.float64)[:,None]*dy+np.arange(width,dtype=np.float64)[None,:]*dx
    span=float(np.ptp(projection))
    if span<=0:
        raise ValueError('Degenerate spatial field')
    normalized=(projection-projection.min())/span
    phase=np.clip((normalized-transition[0])/(transition[1]-transition[0]),0,1)
    blend=np.asarray(phase**3*(phase*(6*phase-15)+10),dtype=np.float32)
    spatial=(1-blend[...,None])*low+blend[...,None]*high
    roundoff['spatial3_8']=int(((spatial<0)|(spatial>1)).sum())
    spatial=np.clip(spatial,0,1)
    scale=target_size/max(height,width)
    effective=np.sqrt((1-blend)*9+blend*64)
    quantiles={str(q):float(np.quantile(effective,q)) for q in (0,.05,.5,.95,1)}
    meta=dict(native_shape=[height,width],network_size=target_size,letterbox_scale=scale,
        sigmas_native=[3.,8.],sigmas_network=[3.*scale,8.*scale],border='BORDER_REFLECT',
        floating_roundoff_clipped_channels=roundoff,
        order='native_RGB_float32_blur_then_letterbox',spatial_field=dict(seed=int(seed),angle=angle,
            transition=list(transition),blend_sha256=hashlib.sha256(blend.tobytes()).hexdigest(),
            blend_min=float(blend.min()),blend_max=float(blend.max())),
        nominal_second_moment_sigma_native=quantiles,
        nominal_second_moment_sigma_network={k:v*scale for k,v in quantiles.items()},
        caveat='Spatial RGB is a convex combination of sigma3 and sigma8 Gaussian outputs. Nominal second-moment width is not a single Gaussian kernel or measured test PSF; final resize adds its own filtering.')
    return {'gaussian3':np.ascontiguousarray(low), 'gaussian8':np.ascontiguousarray(high),
            'spatial3_8':np.ascontiguousarray(spatial,dtype=np.float32)},meta


def run(args):
    import torch
    from data.backend_adaptation import CanonicalBackendDataset
    from data.dataset import letterbox
    from data.semantic_hard import hard_appearance
    from models.backend_adaptation import load_backend
    from tools.probe_semantic_reflect import load_fixed, protected_sources, validate_prediction
    from tools.run_marker_reflect import precision_record
    from tools.semantic_crossover import revote_instances
    from train_backend_adaptation import tensor_digest
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    from utils.config import load_config, project_path

    if not torch.cuda.is_available():
        raise RuntimeError('Use server sam2_env CUDA')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    config=load_config(args.config)
    cfg,acq=config['semantic_raw_stress'],config['semantic_acquisition']
    if (acq['format']!=FORMAT or tuple(acq['sources'])!=INITIAL or acq['native_sigmas']!=[3.,8.]
            or acq['transition_fraction']!=[.2,.8] or acq['global_photometric_view']!=0
            or cfg['contrast']!=[.25,.60] or cfg['offset']!=[.12,.30]
            or cfg['max_clipped_fraction']!=.01 or cfg['minimum_native_area']!=10000):
        raise RuntimeError('Bounded predeclared acquisition recipe differs')
    resolve=lambda value:Path(project_path(config,value)).resolve()
    out=resolve(args.output or acq['output_dir'])
    if out.exists() or not out.is_relative_to((ROOT/'outputs').resolve()):
        raise RuntimeError('Fresh ignored outputs directory required')
    out.mkdir(parents=True);(out/'gallery').mkdir()
    write(out/'status.json',dict(status='running',training=False,no_optimizer_steps=True))
    protected={};hooks=[]
    def protect(path,expected=None):
        path=resolve(path);digest=file_sha(path)
        if expected is not None and digest!=expected:
            raise RuntimeError('Protected input SHA differs: '+str(path))
        protected[str(path)]=digest
        return path
    try:
        protect(args.config)
        # 继承的旧压力配置/工具也是封存依赖，既不改写也不隐藏其版本。
        protect('config/inference/semantic_raw_stress.yaml')
        protect('tools/probe_semantic_raw_stress.py')
        checkpoint=protect(cfg['checkpoint'],cfg['checkpoint_sha256'])
        inputs=read(protect(cfg['reference_inputs']))
        reference=read(protect(cfg['reference_report']))
        prior={r['source']:r for r in read(protect(cfg['reference_rows']))}
        precision=precision_record()
        if (reference.get('complete') is not True or reference.get('format')!='semantic_reflect_probe_v1'
                or reference['precision']!=precision
                or reference['checkpoints']['baseline']['sha256']!=cfg['checkpoint_sha256']):
            raise RuntimeError('Prior actual raw semantic identity/precision differs')
        cases={r['name']:r for r in inputs['cases'] if r['kind']=='train'}
        if set(cases)!=set(INITIAL) or not set(INITIAL).issubset(prior):
            raise RuntimeError('Previously fixed four training cases required')
        dataset=CanonicalBackendDataset(resolve(config['paths']['raw_data_dir']),resolve(config['backend_adaptation']['completed_gt_dir']))
        sources={p.stem:p for p in dataset.samples}
        if len(sources)!=32:
            raise RuntimeError('Expected all32 processed manual sources')
        names=source_order(sources,INITIAL,4)
        indices={name:i for i,name in enumerate(sorted(sources))}
        model,bundle=load_backend(checkpoint,config,'cuda')
        if bundle['epoch']!=cfg['checkpoint_epoch'] or model.semantic_lora is not None:
            raise RuntimeError('Original R1 S-align/raw route required')
        model.eval().requires_grad_(False)
        state=tensor_digest(model.state_dict().items())
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Pressure model not fully frozen')
        def forbidden(*_):
            raise RuntimeError('No affinity forward permitted')
        hooks.append(model.affinity_decoder.register_forward_pre_hook(forbidden))
        code,virtual=protected_sources(out)
        write(out/'config.json',config)
        write(out/'selection.json',dict(names=names,conditions=list(CONDITIONS),
            reason='Unchanged prior four training sources and fixed bounded conditions before outputs'))
        records,calls,started=[],0,time.time()
        first=None;first_source=None
        def forward(image,ph,pw,shape):
            nonlocal calls
            logits=model.semantic_decoder(model.encoder(image.float()),image)
            calls+=1
            if isinstance(logits,dict):logits=logits['semantic_logits']
            return np.ascontiguousarray(crop_letterbox_output(logits.float(),1024,ph,pw,shape).cpu().sigmoid()[0,0].numpy(),dtype=np.float32)
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False), \
                patch('cv2.watershed',side_effect=RuntimeError('No watershed allowed')), \
                patch('models.backend_adaptation.restore_first',side_effect=RuntimeError('No D5a allowed')):
            for name in names:
                case=cases[name]
                source=protect(sources[name],case['source_sha256'])
                gtroot=resolve(config['backend_adaptation']['completed_gt_dir'])
                gtpath=protect(gtroot/(name+'_gt.npz'),case['gt_npz_sha256'])
                classpath=protect(gtroot/(name+'_class.json'),case['gt_classes_sha256'])
                with np.load(gtpath,allow_pickle=False) as blob:
                    ids,original=blob['instance_map'].copy(),blob['original_covered'].astype(bool)
                    if not np.array_equal(blob['filled'].astype(bool),(ids>0)&~original) or not np.array_equal(blob['residual_unknown'].astype(bool),ids==0):
                        raise RuntimeError('Processed original/fill/unknown partition differs')
                classes={int(k):v for k,v in read(classpath).items()}
                rgb,image,ph,pw=prepare_image(source,1024,'cuda')
                if ids.shape!=rgb.shape[:2]:raise RuntimeError('Native GT/source grids differ')
                cohort=build_cohort(ids,original,classes,cfg['minimum_native_area'],cfg['core_fraction'],cfg['core_min_pixels'])
                clean=forward(image,ph,pw,rgb.shape[:2])
                clear_sha=hashlib.sha256(clean.tobytes()).hexdigest()
                if clear_sha!=prior[name]['routes']['salign_raw']['probability_sha256']:
                    raise RuntimeError('Clean raw probability fails prior actual SHA: '+name)
                fixed_ids,fixed_classes,png,labels=load_fixed(resolve(case['fixed_dir']),name)
                protect(png,case['fixed_png_sha256']);protect(labels,case['fixed_classes_sha256'])
                voted,_=revote_instances(fixed_ids,clean)
                validate_prediction(fixed_ids,voted,rgb.shape[:2])
                if voted!=fixed_classes:raise RuntimeError('Clean raw fixed reflect classes differ')
                clear_rows=probability_rows(cohort,clean)
                native,recipe=native_blur_variants(rgb,seed=acq['source_seed']+indices[name]*100003,
                    sigmas=acq['native_sigmas'],transition=acq['transition_fraction'])
                record=dict(source=name,shape=list(ids.shape),input_stride=list(image.stride()),
                    clear_probability_sha256=clear_sha,clear_reference_exact=True,acquisition_recipe=recipe,
                    original_pixels=int(original.sum()),unknown_pixels=int((ids==0).sum()),
                    filled_pixels=int(((ids>0)&~original).sum()),conditions={
                        'clean':dict(summary=summarize_condition(clear_rows,clear_rows),objects=clear_rows)})
                save_preview(out/'gallery'/(name+'_clean.png'),rgb,rgb,clean,'Clean raw',cfg['thumbnail_width'])
                spatial_tensor=None
                for label in CONDITIONS[1:]:
                    appearance=None
                    if label=='spatial3_8_global':
                        valid=torch.zeros((1,1,1024,1024),dtype=torch.bool,device='cuda')
                        valid[:,:,:1024-ph,:1024-pw]=True
                        # 复用上轮固定global_0随机流，不根据本轮结果重抽。
                        seed=cfg['source_seed']+indices[name]*100003
                        tensor,appearance=hard_appearance(spatial_tensor,valid,cfg,seed,'global')
                        if appearance['minimum_slope']<=0 or appearance['clipped_fraction']>cfg['max_clipped_fraction']+1e-7:
                            raise RuntimeError('Positive-slope/clipping gate failed')
                        preview=np.rint(crop_letterbox_output(tensor,1024,ph,pw,rgb.shape[:2]).cpu()[0].permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)
                    else:
                        lb,scale,ah,aw=letterbox(native[label],1024)
                        if (ah,aw)!=(ph,pw) or scale!=recipe['letterbox_scale']:
                            raise RuntimeError('Blur-first letterbox coordinates/scale differ')
                        tensor=torch.from_numpy(lb).permute(2,0,1).float().unsqueeze(0).to('cuda')
                        if label=='spatial3_8':spatial_tensor=tensor
                        preview=np.rint(native[label]*255).clip(0,255).astype(np.uint8)
                    probability=forward(tensor,ph,pw,rgb.shape[:2])
                    values=probability_rows(cohort,probability)
                    record['conditions'][label]=dict(appearance=appearance,
                        probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                        input_tensor_sha256=tensor_digest([('input',tensor)]),
                        summary=summarize_condition(clear_rows,values),objects=values)
                    save_preview(out/'gallery'/(name+'_'+label+'.png'),rgb,preview,probability,label,cfg['thumbnail_width'])
                if first is None:first,first_source=clean.copy(),source
                records.append(record);write(out/'rows.json',records)
                write(out/'status.json',dict(status='running',completed=len(records),total=4,training=False))
                print('SEMANTIC_ACQUISITION',len(records),name,'clear_exact',flush=True)
            rgb,image,ph,pw=prepare_image(first_source,1024,'cuda')
            if not np.array_equal(forward(image,ph,pw,rgb.shape[:2]),first):
                raise RuntimeError('First source clear probability changed after pressure')
        if calls!=21 or tensor_digest(model.state_dict().items())!=state or precision_record()!=precision:
            raise RuntimeError('Forward budget/frozen model/precision changed')
        if any(file_sha(Path(path))!=digest for path,digest in protected.items()):
            raise RuntimeError('Protected input changed')
        if any(file_sha(ROOT/path)!=digest for path,digest in code.items()):
            raise RuntimeError('Loaded code changed')
        totals={condition:{group:{key:sum(row['conditions'][condition]['summary'][group][key] for row in records)
            for key in records[0]['conditions'][condition]['summary'][group]} for group in ('all','large')} for condition in CONDITIONS}
        report=dict(format=FORMAT,complete=True,training=False,no_optimizer_steps=True,full_test_inference=False,
            official_submission=False,best_deployment_changed=False,sources=4,semantic_forward_calls=calls,
            affinity_forward_calls=0,watershed_calls=0,D5a_forward_calls=0,
            checkpoint=dict(path=str(checkpoint),sha256=cfg['checkpoint_sha256'],epoch=cfg['checkpoint_epoch']),
            clear_four_probability_and_fixed_classes_exact=True,first_clear_repeat_exact=True,
            frozen_state_sha256=state,frozen_exact=True,precision=precision,totals=totals,
            protected_inputs=protected,protected_inputs_exact=True,protected_sources=code,virtual_module_paths=virtual,
            elapsed_seconds=time.time()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
            caveat='Seen new-GT original-covered region classification under a fixed bounded stress recipe; not held-out accuracy, physical complete GT, official instance mIoU, or a fitted test PSF. No threshold or strength search after observing results.')
        write(out/'report.json',report)
        page=['<!doctype html><meta charset="utf-8"><title>Semantic acquisition stress</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>Frozen raw semantic: bounded native acquisition blur</h1>',
            '<p>Left: original. Middle: fixed stress input. Right: P(F) fixed 0..1. Labeled training sources only; no training or test truth.</p>',
            '<p><a href="report.json">Summary</a> | <a href="rows.json">Per-object original-covered/core diagnostics</a></p>']
        for path in sorted((out/'gallery').glob('*.png')):
            page.append('<h2>'+html.escape(path.stem)+'</h2><img src="gallery/'+html.escape(path.name)+'">')
        (out/'index.html').write_text('\n'.join(page),encoding='utf-8')
        write(out/'status.json',dict(status='complete',training=False,sources=4,semantic_forward_calls=21))
        print('SEMANTIC_ACQUISITION_COMPLETE',4,21,flush=True)
        return report
    except Exception as error:
        write(out/'status.json',dict(status='failed',training=False,error=repr(error)))
        raise
    finally:
        for hook in hooks:hook.remove()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/inference/semantic_acquisition.yaml')
    parser.add_argument('--output')
    run(parser.parse_args())
