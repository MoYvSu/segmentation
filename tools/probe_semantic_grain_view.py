# -*- coding: utf-8 -*-
"""新GT父对象视野e0预检；默认计划，--preflight执行4源8窗或--all-sources32源64窗。

仅检查裁窗是否暴露新增监督信号。原部署冻结，复制整头只计算梯度而不更新；
每个裁窗重新编码RGB，不能复用全图features。正式64更新尚未实现，不自动启动。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.probe_semantic_fixed import file_sha,write
from tools.probe_semantic_native import TRAIN,verify_frozen_model,verify_whole_baseline
from tools.probe_semantic_object_fit import EXPECTED_LOSS,HEAD_FORMAT,probability
from tools.probe_semantic_components import component_grids
from utils.semantic_grain_view import (anchor_probability,build_grain_batch,gradient_partition,
    grain_window,preflight_gate,select_grain_anchors)
from utils.semantic_native import prepare_native_tile
from utils.semantic_object_fit import array_sha,canonical_sha,feature_copy

FORMAT='semantic_grain_view_preflight_v1'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def plan(config,all_sources=False):
    cfg=config['semantic_grain_view']
    expected=dict(format=FORMAT,sources=list(TRAIN),minimum_native_area=10000,minimum_original_fraction=.8,
        context_factor=1.5,minimum_context_side=512,input_size=1024,training_implemented=False)
    if any(cfg.get(k)!=v for k,v in expected.items()): raise ValueError('Predeclared grain-view contract differs')
    return dict(format=FORMAT,status='plan_only',sources=32 if all_sources else list(TRAIN),views=64 if all_sources else 8,
        selection='Each GT class sorted(native_area,id), lower median; >=10000 pixels, >=.8 original, non-frame',
        viewport='Original RGB bbox context max(512,ceil(1.5*max_bbox_side)), shift inward; complete processed GT parent',
        criterion='Original pixel/core/pool, processed-known including filled; no 182 predicted-region extra loss',
        encoder='Fresh frozen encoder forward for every native crop; old CPU features only for exact whole baseline',
        scope='Combined context/texture scale and GT neighbor pool diagnostic, not pure scale or coarse-only ablation',
        training=False,optimizer_updates=0,train64_implemented=False,test_used=False,official_submission=False,
        gate='One or more GT true-class mean drop >=.05 or original error increase >=.01 with legal anchor-only coarse gradient and anchor-original > anchor-filled common gradient; every view legal coarse gradient and zero unknown/padding')


def preview(path,rgb,probability,gt,original,classes,anchor,width=320):
    color=np.zeros_like(rgb); color[(gt>0)&~original]=(80,180,200)
    for gid,cls in classes.items(): color[(gt==gid)&original]=(74,190,104) if cls else (185,74,80)
    heat=cv2.cvtColor(cv2.applyColorMap(np.rint(probability*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
    contours,_=cv2.findContours((gt==anchor['id']).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color,contours,-1,(255,245,45),2); cv2.drawContours(heat,contours,-1,(255,245,45),2)
    panels=[]
    for image,label in ((rgb,'Native crop RGB'),(color,'GT original Fgreen/Pred; fillcyan'),(heat,'Full P(F); anchor '+str(anchor['id']))):
        thumb=cv2.resize(image,(width,max(1,round(image.shape[0]*width/image.shape[1]))),interpolation=cv2.INTER_AREA)
        thumb=cv2.copyMakeBorder(thumb,25,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255)); cv2.putText(thumb,label,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1,cv2.LINE_AA); panels.append(thumb)
    if not cv2.imwrite(str(path),cv2.cvtColor(np.concatenate(panels,axis=1),cv2.COLOR_RGB2BGR)): raise IOError('Private preview write failed')


def validate_control_reference(control,capture,cfg,old):
    """旧A64只读对照可复用门禁；绝不将此状态加载为本预检的初始化。"""
    from train_backend_adaptation import tensor_digest
    if (control.get('format')!=HEAD_FORMAT or control.get('arm')!='control' or control.get('step')!=64
            or control.get('terminal_only') is not True or control.get('do_not_replace_geometry_upstream') is not True
            or control.get('region_weight')!=0 or control.get('base_checkpoint_sha256')!=old['checkpoint_sha256']
            or control.get('identity')!=capture['identity'] or control.get('schedule')!=capture['source_order']
            or control.get('mode')!='eval' or tensor_digest(control['semantic_state_dict'].items())!=cfg['control_head_digest']):
        raise RuntimeError('Old control64 terminal checkpoint/initialization/GT/schedule/state identity differs')
    groups=control['optimizer']['param_groups']
    if not groups or any(g['lr']!=old['learning_rate'] or g['eps']!=old['adam_eps'] or g['weight_decay']!=old['weight_decay'] for g in groups):
        raise RuntimeError('Old control64 optimizer recipe differs')
    if canonical_sha(capture['embedded_loss_config'])!=canonical_sha(EXPECTED_LOSS): raise RuntimeError('Old control common GT criterion differs')
    return dict(verified=True,used_as_initialization=False,training_repeated=False,step=64,arm='control',
        semantic_head_state_sha256=cfg['control_head_digest'],same_R1_init_and_GT_identity=True,
        schedule_sha256=canonical_sha(control['schedule']),optimizer_groups=[{key:g[key] for key in ('lr','eps','weight_decay')} for g in groups],
        mode='eval',region_weight=0,common_criterion_sha256=canonical_sha(capture['embedded_loss_config']))


def run(args):
    import yaml
    from models.backend_adaptation import load_backend
    from tools.probe_semantic_reflect import protected_sources,validate_prediction
    from tools.run_marker_reflect import precision_record
    from train_backend_adaptation import tensor_digest
    from train_direct_semantic_affinity import build_semantic_criterion,compute_semantic_loss
    from utils.affinity_deployment import crop_letterbox_output,prepare_image
    from utils.config import load_config,project_path
    config=load_config(args.config); selection=plan(config,args.all_sources); cfg=config['semantic_grain_view']; old=config['semantic_object_fit']
    if not torch.cuda.is_available(): raise RuntimeError('Server sam2_env CUDA required')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    resolve=lambda p:Path(project_path(config,p)).resolve()
    out=resolve(args.output or cfg['preflight_all_dir' if args.all_sources else 'preflight_dir'])
    if out.exists() or not out.is_relative_to((ROOT/'outputs').resolve()): raise RuntimeError('Fresh ignored outputs directory required')
    out.mkdir(parents=True); (out/'gallery').mkdir(); (out/'cache').mkdir(); hooks=[]; protected={}
    write(out/'status.json',dict(status='running',training=False,optimizer_updates=0))
    def protect(p,expected=None):
        p=resolve(p); digest=file_sha(p)
        if expected is not None and expected!=digest: raise RuntimeError('Protected input differs: '+str(p))
        protected[str(p)]=digest; return p
    try:
        p=Path(args.config).resolve(); visited=set(); config_sources={}
        while True:
            if p in visited or not p.is_relative_to(ROOT): raise RuntimeError('Invalid config inheritance')
            visited.add(p); protect(p); relative=p.relative_to(ROOT); dest=out/'config_source'/relative; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
            if file_sha(dest)!=file_sha(p): raise RuntimeError('Config snapshot differs')
            config_sources[relative.as_posix()]=file_sha(p); base=(yaml.safe_load(p.read_text(encoding='utf-8')) or {}).get('_base')
            if not base: break
            p=(p.parent/base).resolve()
        manifest=read(protect(old['manifest'],old['manifest_sha256'])); capture=read(protect(cfg['capture']))
        if manifest.get('status')!='complete' or len(manifest['rows'])!=32 or manifest.get('test_training') is not False or capture['identity']['checkpoint_sha256']!=old['checkpoint_sha256']: raise RuntimeError('Completed processed32/capture source contract differs')
        sources={r['source']:r for r in manifest['rows']}
        if len(sources)!=32: raise RuntimeError('Unique32 source contract required')
        names=tuple(sorted(sources)) if args.all_sources else TRAIN
        control_path=protect(cfg['control_checkpoint']); control=torch.load(control_path,map_location='cpu',weights_only=False)
        control_reference=validate_control_reference(control,capture,cfg,old)
        control_reference.update(path=str(control_path),checkpoint_sha256=file_sha(control_path)); del control
        if canonical_sha(capture['config']['semantic_object_fit'])!=canonical_sha(old): raise RuntimeError('Old source/optimizer/common support configuration differs')
        checkpoint=protect(old['checkpoint'],old['checkpoint_sha256']); model,bundle=load_backend(checkpoint,config,'cuda')
        model.eval().requires_grad_(False); state=tensor_digest(model.state_dict().items()); initial=tensor_digest(model.semantic_decoder.state_dict().items())
        if state!=capture['identity']['full_R1_state_sha256'] or initial!=capture['initial_head_sha256'] or model.semantic_lora is not None or bundle['epoch']!=20: raise RuntimeError('Original R1 identity differs')
        loss_cfg=bundle['config']['direct_semantic_affinity']['semantic_loss']
        if canonical_sha(loss_cfg)!=canonical_sha(EXPECTED_LOSS): raise RuntimeError('Original common criterion differs')
        criterion=build_semantic_criterion({'direct_semantic_affinity':{'semantic_loss':loss_cfg}},'cuda')
        precision=precision_record()
        if precision!=capture['identity']['precision']: raise RuntimeError('FP32 precision contract differs')
        head=deepcopy(model.semantic_decoder).eval().requires_grad_(True)
        if tensor_digest(head.state_dict().items())!=initial: raise RuntimeError('Copied head initialization differs')
        parameters=sum(p.numel() for p in head.parameters())
        if parameters!=6302010 or 84951045+parameters>=500000000: raise RuntimeError('Parameter accounting differs')
        def forbid(*_): raise RuntimeError('No affinity forward allowed')
        hooks.append(model.affinity_decoder.register_forward_pre_hook(forbid))
        seen=[]
        def capture_parts(module,positional,kwargs,output):
            if 'coarse_logits' not in kwargs: raise RuntimeError('Expected highres coarse input')
            seen.append((kwargs['coarse_logits'].detach().clone(),output.detach().clone()))
        hooks.append(head.semantic_residual.register_forward_hook(capture_parts,with_kwargs=True))
        code,virtual=protected_sources(out); write(out/'config.json',config); write(out/'selection.json',selection)
        data={}; whole_calls=0; encoder_calls=0; rows=[]
        def get_probabilities(features,image,record):
            seen.clear(); logits=head(features,image)
            if len(seen)!=1: raise RuntimeError('Exactly one actual semantic residual call required')
            coarse,delta=seen[0]; up=component_grids(coarse,delta,logits)
            _,full=probability(logits,record); _,coarse_p=probability(up,record)
            return logits,full,coarse_p
        with torch.autocast(device_type='cuda',enabled=False),patch('cv2.watershed',side_effect=RuntimeError('No watershed permitted')),patch('models.backend_adaptation.restore_first',side_effect=RuntimeError('No D5a permitted')):
            # 所有whole旧缓存及锚点名单先过门禁，之后才运行任何裁窗。
            for name in names:
                row=sources[name]; source=protect(row['image'],row['image_sha256']); gp=protect(row['manual_gt'],row['manual_gt_sha256'])
                key=gp.relative_to(ROOT).as_posix().removesuffix('_gt.npz')+'_class.json'; cp=protect(gp.with_name(name+'_class.json'),manifest['protected_inputs'][key])
                with np.load(gp,allow_pickle=False) as z:
                    gt=z['instance_map'].copy(); original=z['original_covered'].astype(bool)
                    if not np.array_equal(z['filled'].astype(bool),(gt>0)&~original) or not np.array_equal(z['residual_unknown'].astype(bool),gt==0): raise RuntimeError('NewGT partitions differ')
                classes={int(k):v for k,v in read(cp).items()}; anchors=select_grain_anchors(gt,original,classes)
                cachepath=protect(resolve(cfg['cache_dir'])/(name+'.pt'),capture['captures'][name]['cache_sha256']); cache=torch.load(cachepath,map_location='cpu',weights_only=False)
                rgb,image,ph,pw=prepare_image(source,1024,'cuda')
                if rgb.shape[:2]!=gt.shape: raise RuntimeError('Source RGB and processed GT native shapes differ')
                cached_image=feature_copy(cache['image'],'cuda'); features=[feature_copy(v,'cuda') for v in cache['features']]
                if cache['source']!=name or not torch.equal(image,cached_image) or image.stride()!=cached_image.stride() or tensor_digest([(str(i),v) for i,v in enumerate(features)])!=capture['captures'][name]['feature_sha256'] or [list(v.stride()) for v in features]!=capture['captures'][name]['feature_strides']: raise RuntimeError('Whole image/features/cache layout differs')
                record=dict(shape=list(gt.shape),pad_h=ph,pad_w=pw)
                with torch.no_grad(): _,full,coarse=get_probabilities(features,cached_image,record)
                whole_calls+=1
                png=protect(resolve(old['manifest']).parent/'teacher'/(name+'_inst.png'),row['prediction_hashes']['png_sha256']); jc=protect(png.with_name(name+'_class.json'),row['prediction_hashes']['classes_sha256']); ids=cv2.imread(str(png),cv2.IMREAD_UNCHANGED); predicted=read(jc)
                validate_prediction(ids,predicted,gt.shape); verify_whole_baseline(name,full,capture['captures'][name]['baseline_probability_sha256'],ids,predicted)
                if not np.array_equal(full,cache['reference_probability']): raise RuntimeError('Cached reference probability differs')
                data[name]=dict(rgb=rgb,gt=gt,original=original,classes=classes,anchors=anchors,
                    whole={a['id']:anchor_probability(full,gt,original,a) for a in anchors},
                    whole_coarse={a['id']:anchor_probability(coarse,gt,original,a) for a in anchors})
                print('GRAIN_WHOLE_EXACT',name,flush=True)
            write(out/'anchor_selection.json',{name:item['anchors'] for name,item in data.items()})
            for name in names:
                item=data[name]
                for anchor in item['anchors']:
                    box=grain_window(item['gt'].shape,anchor['bbox']); y0,y1,x0,x1=box; view=name+'_'+str(anchor['cls'])
                    image,ph,pw,image_receipt=prepare_native_tile(item['rgb'],box,1024,'cuda')
                    batch,partitions,target_receipt=build_grain_batch(item['gt'],item['original'],item['classes'],box,anchor['id'])
                    crop_gt=item['gt'][y0:y1,x0:x1]; crop_original=item['original'][y0:y1,x0:x1]
                    with torch.no_grad(): features=model.encoder(image.float()); encoder_calls+=1
                    record=dict(shape=list(crop_gt.shape),pad_h=ph,pad_w=pw); logits,full,coarse=get_probabilities(features,image,record)
                    loss=compute_semantic_loss(criterion,logits,{k:v.to('cuda') for k,v in batch.items() if torch.is_tensor(v)})
                    named=list(head.named_parameters())
                    # 独立诊断原标锚点自身的监督能否到达粗分支；不加入候选共同训练目标。
                    native,_=probability(logits,record)
                    anchor_support=torch.from_numpy((crop_gt==anchor['id'])&crop_original)
                    anchor_mean=native.cpu().sigmoid()[0,0][anchor_support].mean(dtype=torch.float32)
                    anchor_loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.logit(anchor_mean,eps=1e-5),anchor_mean.new_tensor(float(anchor['cls'])))
                    anchor_gradients=torch.autograd.grad(anchor_loss,[p for _,p in named],retain_graph=True,allow_unused=True)
                    anchor_norms={prefix:sum(float(g.detach().double().square().sum()) for (key,_),g in zip(named,anchor_gradients) if key.startswith(prefix+'.') and g is not None)**.5 for prefix in ('seg_fpn','seg_branch','semantic_residual')}
                    gradients=torch.autograd.grad(loss,[p for _,p in named]+[logits],allow_unused=True)
                    norms={prefix:sum(float(g.detach().double().square().sum()) for (key,_),g in zip(named,gradients[:-1]) if key.startswith(prefix+'.') and g is not None)**.5 for prefix in ('seg_fpn','seg_branch','semantic_residual')}
                    output=gradient_partition(gradients[-1],partitions)
                    whole=item['whole'][anchor['id']]; current=anchor_probability(full,crop_gt,crop_original,anchor)
                    result=dict(view=view,source=name,anchor=anchor,image_receipt=image_receipt,target_receipt=target_receipt,whole=whole,crop=current,
                        whole_coarse=item['whole_coarse'][anchor['id']],crop_coarse=anchor_probability(coarse,crop_gt,crop_original,anchor),
                        common_loss=float(loss.detach()),parameter_gradient_norms=norms,output_gradient=output,
                        anchor_diagnostic_BCE=float(anchor_loss.detach()),anchor_parameter_gradient_norms=anchor_norms,anchor_diagnostic_not_added_to_training_loss=True,
                        feature_sha256=tensor_digest([(str(i),v) for i,v in enumerate(features)]),input_sha256=array_sha(image),
                        fresh_crop_encoder=True,no_optimizer_step=True)
                    saved=dict(view=view,anchor=anchor,box=box,image=feature_copy(image,'cpu'),features=[feature_copy(v,'cpu') for v in features],
                        batch={k:v for k,v in batch.items() if torch.is_tensor(v)},image_receipt=image_receipt,target_receipt=target_receipt,
                        source_sha256=sources[name]['image_sha256'],processed_gt_sha256=sources[name]['manual_gt_sha256'],
                        image_sha256=result['input_sha256'],feature_sha256=result['feature_sha256'],feature_strides=[list(v.stride()) for v in features],
                        target_sha256={k:array_sha(v) for k,v in batch.items() if torch.is_tensor(v)})
                    cachefile=out/'cache'/(view+'.pt'); torch.save(saved,cachefile); result['crop_cache_sha256']=file_sha(cachefile)
                    rows.append(result); write(out/'rows.json',rows); preview(out/'gallery'/(view+'.jpg'),item['rgb'][y0:y1,x0:x1],full,crop_gt,crop_original,item['classes'],anchor)
                    if tensor_digest(head.state_dict().items())!=initial or any(p.grad is not None for p in head.parameters()) or any(m.training for m in head.modules()): raise RuntimeError('Preflight copied head changed or accumulated gradient')
                    verify_frozen_model(model,state)
                    print('GRAIN_VIEW',view,'trueprob',whole['mean_true_class_probability'],current['mean_true_class_probability'],'grad',norms,flush=True)
        expected=len(names)*2
        if encoder_calls!=expected or len(rows)!=expected or whole_calls!=len(names): raise RuntimeError('Predeclared view/forward count differs')
        verify_frozen_model(model,state)
        if precision_record()!=precision or any(file_sha(Path(p))!=v for p,v in protected.items()) or any(file_sha(ROOT/p)!=v for p,v in code.items()): raise RuntimeError('Precision/input/code changed')
        gate=preflight_gate(rows)
        report=dict(format=FORMAT,complete=True,training=False,optimizer_updates=0,test_used=False,whole_baseline_all_exact=True,
            original_R1_frozen_exact=True,copied_head_unchanged=True,sources=len(names),views=expected,encoder_crop_calls=encoder_calls,
            cached_whole_head_calls=whole_calls,affinity_forward_calls=0,D5a_forward_calls=0,watershed_calls=0,gate=gate,training_started=False,
            checkpoint=dict(path=str(checkpoint),sha256=old['checkpoint_sha256'],epoch=20),head_parameters=parameters,
            control_reference=control_reference,initialization='Original R1, not old control64',
            conservative_terminal_deployment_parameters=84951045+parameters,precision=precision,protected_inputs=protected,
            protected_sources=code,virtual_module_paths=virtual,inherited_config_sources=config_sources,
            caveat='Complete means processed GT parent retained, not complete physical GT. Anchor diagnostics use original labels only; common loss includes filled and altered neighboring GT pool. Joint context/texture-scale/head-input experiment; no test labels or official score conclusion. Formal64 training is not implemented.')
        write(out/'report.json',report); write(out/'status.json',dict(status='complete',training=False,gate_passed=gate['passed']))
        page=['<!doctype html><meta charset="utf-8"><title>GT parent grain views</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>e0 processed-GT parent view preflight</h1><p>Training sources only. RGB / original GT (filled cyan, unknown black) / current full probability. No optimizer updates.</p>',
            '<p><a href="report.json">Gate report</a> | <a href="rows.json">Anchor, source window and gradient receipts</a></p>']
        for path in sorted((out/'gallery').glob('*.jpg')): page.append('<h2>'+html.escape(path.stem)+'</h2><img src="gallery/'+html.escape(path.name)+'">')
        (out/'index.html').write_text('\n'.join(page),encoding='utf-8'); print('GRAIN_PREFLIGHT_COMPLETE',expected,gate,flush=True); return report
    except Exception as error:
        write(out/'status.json',dict(status='failed',error=repr(error),training=False)); raise
    finally:
        for hook in hooks: hook.remove()


def main():
    from utils.config import load_config
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',default='config/train/semantic_grain_view.yaml'); parser.add_argument('--output')
    parser.add_argument('--preflight',action='store_true'); parser.add_argument('--all-sources',action='store_true'); parser.add_argument('--train64',action='store_true')
    args=parser.parse_args()
    if args.train64: raise NotImplementedError('训练64更新尚未实现；先分析preflight，不静默启动训练')
    if args.preflight: run(args)
    else: print(json.dumps(plan(load_config(args.config),args.all_sources),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
