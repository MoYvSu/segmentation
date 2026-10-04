# -*- coding: utf-8 -*-
"""全32源e0门禁通过后，R1末端整语义头64更新；只读缓存、固定几何、不自动训练。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.probe_semantic_fixed import class_independent_pairs,copy_fixed_mask,file_sha,fixed_confusion,write
from tools.probe_semantic_native import TRAIN,TEST,verify_frozen_model,verify_whole_baseline
from tools.probe_semantic_object_fit import EXPECTED_LOSS,MONITORS,preview,probability
from tools.probe_semantic_raw_stress import build_cohort,probability_rows,summarize_condition
from tools.probe_semantic_grain_view import preview as crop_preview,validate_control_reference
from utils.semantic_grain_view import anchor_probability,build_grain_batch,preflight_gate,select_grain_anchors
from utils.semantic_object_fit import array_sha,build_semantic_batch,canonical_sha,feature_copy
from utils.retention_fit import capture_rng,restore_rng

FORMAT='semantic_grain_fit_v1'
HEAD_FORMAT='semantic_grain_terminal_only_v1'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_fit_recipe(config):
    cfg=config['semantic_grain_fit']; old=config['semantic_object_fit']
    expected=dict(format=FORMAT,steps=64,smoke_steps=2,whole_weight=.5,crop_weight=.5,snapshots=[0,16,32,64],
        learning_rate=2e-5,adam_eps=1e-4,weight_decay=1e-4,grad_clip=1.,seed=20261004,monitor_sources=MONITORS)
    if any(cfg.get(k)!=v for k,v in expected.items()) or any(cfg[k]!=old[k] for k in ('learning_rate','adam_eps','weight_decay','grad_clip','seed')):
        raise ValueError('Predeclared paired whole/crop64 recipe differs')
    return cfg


def grain_schedule(source_order):
    names=list(source_order)
    if len(names)!=64 or len(set(names))!=32 or any(v!=2 for v in Counter(names).values()) or names[:32]!=names[32:]:
        raise ValueError('Exact old32-source two-exposure64 schedule required')
    seen=Counter(); result=[]
    for name in names:
        result.append((name,1 if seen[name]==0 else 0)); seen[name]+=1
    return result


def validate_preflight_contract(report,rows,names,checkpoint_sha):
    if (report.get('format')!='semantic_grain_view_preflight_v1' or report.get('complete') is not True
            or report.get('training') is not False or report.get('optimizer_updates')!=0 or report.get('test_used') is not False
            or report.get('sources')!=32 or report.get('views')!=64 or report.get('encoder_crop_calls')!=64
            or report.get('whole_baseline_all_exact') is not True or report.get('original_R1_frozen_exact') is not True
            or report.get('copied_head_unchanged') is not True or report['checkpoint']['sha256']!=checkpoint_sha
            or not report.get('gate',{}).get('passed')):
        raise RuntimeError('Complete32-source64-view GT-only preflight gate required')
    expected={name+'_'+str(cls) for name in names for cls in (0,1)}
    if len(rows)!=64 or {r['view'] for r in rows}!=expected or any(r['source'] not in names or r['view']!=r['source']+'_'+str(r['anchor']['cls']) for r in rows):
        raise RuntimeError('Preflight source/anchor view ownership differs')
    computed=preflight_gate(rows)
    if computed!=report['gate'] or not computed['passed']: raise RuntimeError('Stored preflight gate cannot be reproduced')
    return computed


def validate_resume_state(state,identity,schedule,architecture,total_steps):
    """只接续同一实验的完整末端状态，更新记录必须恰为已完成的日程前缀。"""
    step=state.get('step')
    if (state.get('format')!=HEAD_FORMAT or state.get('identity')!=identity or state.get('complete') is not False
            or state.get('terminal_only') is not True or state.get('do_not_replace_geometry_upstream') is not True
            or state.get('architecture')!=architecture or type(step) is not int or not 0<=step<=total_steps
            or type(state.get('total_steps')) is not int or state['total_steps']!=total_steps or state.get('schedule')!=schedule
            or any(k not in state for k in ('semantic_state_dict','optimizer','rng','steps','snapshots'))
            or not isinstance(state['semantic_state_dict'],dict) or not isinstance(state['optimizer'],dict)
            or not isinstance(state['rng'],dict) or not isinstance(state['snapshots'],dict)
            or not isinstance(state['steps'],list) or len(state['steps'])!=step):
        raise RuntimeError('Incomplete same-identity terminal-only resume state required')
    for index,row in enumerate(state['steps'],1):
        if (not isinstance(row,dict) or type(row.get('step')) is not int or row['step']!=index
                or (row.get('source'),row.get('crop_class'))!=schedule[index-1]
                or type(row.get('crop_class')) is not int or type(row.get('optimizer_updates')) is not int or row['optimizer_updates']!=1
                or type(row.get('head_views')) is not int or row['head_views']!=2 or row.get('region_extra_weight')!=0):
            raise RuntimeError('Resume update/source/crop-class budget differs')
    if (not {'state','param_groups'}<=state['optimizer'].keys() or not state['optimizer']['param_groups']
            or not {'python','numpy','torch','cuda'}<=state['rng'].keys()
            or any(group.get('lr')!=2e-5 or group.get('eps')!=1e-4 or group.get('weight_decay')!=1e-4 for group in state['optimizer']['param_groups'])):
        raise RuntimeError('Resume optimizer/RNG payload incomplete')
    return step


def verify_resume_archives(out,identity):
    """先核旧源码、继承配置快照和当前文件；拒绝恢复时绝不覆盖旧证据。"""
    for key,digest in identity['protected_sources'].items():
        if file_sha(ROOT/key)!=digest or file_sha(out/'source'/key)!=digest:
            raise RuntimeError('Resume sealed source archive/live source differs: '+key)
    for path,digest in identity['protected_inputs'].items():
        live=Path(path)
        if live.suffix in ('.yaml','.yml') and live.is_relative_to(ROOT):
            if file_sha(live)!=digest or file_sha(out/'config_source'/live.relative_to(ROOT))!=digest:
                raise RuntimeError('Resume sealed YAML archive/live source differs: '+str(live))


def validate_crop_cache(cache,row,anchor,source_sha,gt_sha,image,image_receipt,rebuilt,receipt):
    from train_backend_adaptation import tensor_digest
    target_hashes={k:array_sha(v) for k,v in rebuilt.items() if torch.is_tensor(v)}
    actual_features=tensor_digest([(str(i),v) for i,v in enumerate(cache['features'])])
    if (cache['view']!=row['view'] or cache['anchor']!=row['anchor'] or cache['anchor']!=anchor
            or cache['source_sha256']!=source_sha or cache['processed_gt_sha256']!=gt_sha
            or list(cache['box'])!=row['image_receipt']['box'] or image_receipt!=row['image_receipt']
            or cache['image_receipt']!=image_receipt or cache['target_receipt']!=row['target_receipt']
            or cache['target_receipt']!=receipt or cache['image_sha256']!=row['input_sha256']
            or array_sha(cache['image'])!=row['input_sha256'] or array_sha(image)!=row['input_sha256']
            or list(cache['image'].stride())!=image_receipt['input_stride'] or list(image.stride())!=image_receipt['input_stride']
            or cache['feature_sha256']!=row['feature_sha256'] or actual_features!=row['feature_sha256']
            or [list(v.stride()) for v in cache['features']]!=cache['feature_strides']
            or cache['target_sha256']!=target_hashes
            or {k:array_sha(v) for k,v in cache['batch'].items()}!=target_hashes):
        raise RuntimeError('Crop cache/preflight/RGB/GT/feature/target binding differs')


def verify_terminal_output(probability,classes,expected,source_png,output_png,class_path):
    if (hashlib.sha256(probability.tobytes()).hexdigest()!=expected['probability_sha256']
            or classes!=read(class_path) or file_sha(source_png)!=expected['png_sha256']
            or file_sha(output_png)!=expected['png_sha256']):
        raise RuntimeError('Strict final-scope native probability/class/source/output PNG replay differs')


def evaluation_sources(names,test_names,step,total_steps,smoke):
    """短测只读8全图/8裁窗；正式中间6源裁窗，末态全32源裁窗。"""
    if smoke: return list(TRAIN+TEST),list(TRAIN)
    whole=list(names) if step in (0,total_steps) else MONITORS+list(test_names)
    crop=[name for name in names if name not in test_names] if step==total_steps else list(MONITORS)
    return whole,crop


def verify_crop_replay(probability,metrics,expected):
    if hashlib.sha256(probability.tobytes()).hexdigest()!=expected['probability_sha256'] or metrics!=expected['metrics']:
        raise RuntimeError('Strict final crop native probability/anchor metrics replay differs')


def plan(config):
    cfg=validate_fit_recipe(config)
    return dict(format=FORMAT,status='plan_only',preflight=cfg['preflight_dir'],required='complete32-source64-view GT-only gate passed, cache/GT/source/code identity exact',
        initialization='Original R1, smoke and oldA64 weights never inherited',control='Old A64 readonly, no repeat training',
        schedule='Old64 source order; first exposure F1 crop, second P0 crop',loss='.5 whole common+.5 crop common, one averaged loss and optimizer update',
        region_extra_weight=0,trainable='Entire semantic head 6302010 parameters, eval mode',
        frozen='Encoder/LoRA/affinity/original semantic/D5a/upstream; fixed PNG, no new encoder forwards',
        checkpoints='Optimizer/RNG/source schedule and strict resumable state, snapshots0/16/32/64',
        monitoring='Formal fixed6 training+fixed8 test whole views and6xFP crop previews; final40 whole+64crop with readonly oldA64 crop control. Smoke8 whole+8crop. No test selection or100-image inference',
        smoke_steps=2,formal_steps=64,conservative_parameters=91253055,training=False)


def run(args):
    import yaml
    from models.backend_adaptation import load_backend
    from tools.probe_semantic_reflect import load_fixed,protected_sources,validate_prediction
    from utils.affinity_deployment import prepare_image
    from utils.semantic_native import prepare_native_tile
    from tools.run_marker_reflect import precision_record
    from tools.semantic_crossover import revote_instances
    from train_backend_adaptation import tensor_digest
    from train_direct_semantic_affinity import build_semantic_criterion,compute_semantic_loss
    from utils.config import load_config,project_path
    config=load_config(args.config); cfg=validate_fit_recipe(config); old=config['semantic_object_fit']; viewcfg=config['semantic_grain_view']
    if not torch.cuda.is_available(): raise RuntimeError('Server sam2_env CUDA required')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    resolve=lambda p:Path(project_path(config,p)).resolve(); total=cfg['smoke_steps'] if args.smoke else cfg['steps']
    out=resolve(args.resume or args.output or cfg['smoke_dir' if args.smoke else 'output_dir']); resuming=bool(args.resume)
    if not out.is_relative_to((ROOT/'outputs').resolve()) or (out.exists() and not resuming) or (resuming and (not out.exists() or read(out/'status.json').get('status')=='complete')):
        raise RuntimeError('Fresh ignored run or unfinished owned resume required')
    out.mkdir(parents=True,exist_ok=resuming); protected={}; hooks=[]
    def protect(p,expected=None):
        p=resolve(p); digest=file_sha(p)
        if expected is not None and expected!=digest: raise RuntimeError('Protected SHA differs: '+str(p))
        protected[str(p)]=digest; return p
    try:
        resumed_state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False) if resuming else None
        if resuming: verify_resume_archives(out,resumed_state['identity'])
        p=Path(args.config).resolve(); visited=set(); config_sources={}
        while True:
            if p in visited or not p.is_relative_to(ROOT): raise RuntimeError('Invalid YAML inheritance')
            visited.add(p); protect(p); rel=p.relative_to(ROOT); target=out/'config_source'/rel; target.parent.mkdir(parents=True,exist_ok=True)
            if resuming:
                if not target.exists() or file_sha(target)!=file_sha(p): raise RuntimeError('Resume config archive differs')
            else: shutil.copyfile(p,target)
            if file_sha(target)!=file_sha(p): raise RuntimeError('Config archive differs')
            config_sources[rel.as_posix()]=file_sha(p); parent=(yaml.safe_load(p.read_text(encoding='utf-8')) or {}).get('_base')
            if not parent: break
            p=(p.parent/parent).resolve()
        pfroot=resolve(cfg['preflight_dir']); pref=read(protect(pfroot/'report.json')); prefrows=read(protect(pfroot/'rows.json'))
        manifest=read(protect(old['manifest'],old['manifest_sha256'])); capture=read(protect(viewcfg['capture']))
        manual={r['source']:r for r in manifest['rows']}
        if len(manual)!=32 or manifest.get('status')!='complete' or manifest.get('test_training') is not False: raise RuntimeError('Completed32 processedGT source manifest required')
        gate=validate_preflight_contract(pref,prefrows,set(manual),old['checkpoint_sha256'])
        for path,digest in pref['protected_inputs'].items(): protect(path,digest)
        for path,digest in pref['protected_sources'].items(): protect(ROOT/path,digest)
        crop_rows={r['view']:r for r in prefrows}
        control_path=protect(viewcfg['control_checkpoint']); control=torch.load(control_path,map_location='cpu',weights_only=False)
        control_receipt=validate_control_reference(control,capture,viewcfg,old)
        if control_receipt!= {k:v for k,v in pref['control_reference'].items() if k not in ('path','checkpoint_sha256')}: raise RuntimeError('Preflight oldA64 reference differs')
        source_schedule=control['schedule']; schedule=grain_schedule(source_schedule); control_snapshots=control['snapshots']['64']['rows']; control_state=control['semantic_state_dict']; control_architecture=control['architecture']; del control
        checkpoint=protect(old['checkpoint'],old['checkpoint_sha256']); model,bundle=load_backend(checkpoint,config,'cuda')
        model.eval().requires_grad_(False); base_state=tensor_digest(model.state_dict().items()); initial_head=tensor_digest(model.semantic_decoder.state_dict().items())
        if (bundle['epoch']!=20 or model.semantic_lora is not None or base_state!=capture['identity']['full_R1_state_sha256']
                or initial_head!=capture['initial_head_sha256'] or canonical_sha(bundle['config']['direct_semantic_affinity']['semantic_loss'])!=canonical_sha(EXPECTED_LOSS)):
            raise RuntimeError('Original R1/whole-cache/common-loss identity differs')
        criterion=build_semantic_criterion({'direct_semantic_affinity':{'semantic_loss':deepcopy(EXPECTED_LOSS)}},'cuda')
        head=deepcopy(model.semantic_decoder).eval().requires_grad_(True); buffers=tensor_digest(head.named_buffers()); architecture=bundle['architecture']['semantic_decoder']; del bundle
        if control_architecture!=architecture: raise RuntimeError('Readonly control64 semantic architecture differs')
        control_head=deepcopy(head).eval().requires_grad_(False); control_head.load_state_dict(control_state,strict=True); del control_state
        verify_frozen_model(control_head,viewcfg['control_head_digest'])
        if sum(p.numel() for p in head.parameters())!=6302010 or sum(p.numel() for p in model.parameters())+3283651!=84951045: raise RuntimeError('Parameter ledger differs')
        precision=precision_record()
        if precision!=pref['precision'] or precision!=capture['identity']['precision']: raise RuntimeError('Precision identity differs')
        def forbidden(*_): raise RuntimeError('Frozen upstream forward forbidden; only verified feature caches permitted')
        for module in (model.encoder,model.semantic_decoder,model.affinity_decoder): hooks.append(module.register_forward_pre_hook(forbidden))
        restoration=config['backend_adaptation']['restoration']; protect(restoration['checkpoint'],restoration['checkpoint_sha256'])
        inputs=read(protect(old['reference_inputs'])); test_cases=[r for r in inputs['cases'] if r['kind']=='test']
        if len(test_cases)!=8 or len({r['name'] for r in test_cases})!=8: raise RuntimeError('Fixed8 unlabeled test sources required')
        records={}; whole_cache={}; crop_cache={}
        for row in list(manual.values())+test_cases:
            train='source' in row; name=row['source'] if train else row['name']; kind='train' if train else 'test'
            source=protect(row['image'] if train else row['path'],row['image_sha256'] if train else row['source_sha256'])
            rgb,input_tensor,input_ph,input_pw=prepare_image(source,1024,'cuda')
            if train:
                png=protect(resolve(old['manifest']).parent/'teacher'/(name+'_inst.png'),row['prediction_hashes']['png_sha256']); labels=protect(png.with_name(name+'_class.json'),row['prediction_hashes']['classes_sha256']); ids=cv2.imread(str(png),cv2.IMREAD_UNCHANGED); classes=read(labels)
                gp=protect(row['manual_gt'],row['manual_gt_sha256']); key=gp.relative_to(ROOT).as_posix().removesuffix('_gt.npz')+'_class.json'; cp=protect(gp.with_name(name+'_class.json'),manifest['protected_inputs'][key])
                with np.load(gp,allow_pickle=False) as z:
                    gt=z['instance_map'].copy(); original=z['original_covered'].astype(bool)
                    if not np.array_equal(z['filled'].astype(bool),(gt>0)&~original) or not np.array_equal(z['residual_unknown'].astype(bool),gt==0): raise RuntimeError('NewGT provenance differs')
                gtclasses={int(k):v for k,v in read(cp).items()}; batch=build_semantic_batch(gt,original,gtclasses); anchors={a['cls']:a for a in select_grain_anchors(gt,original,gtclasses)}
                pairs,stats=class_independent_pairs(gt,original,ids)
                if canonical_sha(pairs)!=capture['captures'][name]['fixed_matching_sha256']: raise RuntimeError('Fixed geometry matching changed')
                cohort=build_cohort(gt,original,gtclasses); gtdata=dict(gt=gt,original=original,classes=gtclasses,pairs=pairs,stats=stats,cohort=cohort)
                for cls in (0,1):
                    view=name+'_'+str(cls); r=crop_rows[view]; path=protect(pfroot/'cache'/(view+'.pt'),r['crop_cache_sha256']); cache=torch.load(path,map_location='cpu',weights_only=False)
                    rebuilt,_,receipt=build_grain_batch(gt,original,gtclasses,tuple(cache['box']),anchors[cls]['id'])
                    crop_input,_,_,image_receipt=prepare_native_tile(rgb,cache['box'],1024,'cuda')
                    validate_crop_cache(cache,r,anchors[cls],row['image_sha256'],row['manual_gt_sha256'],crop_input,image_receipt,rebuilt,receipt)
                    crop_cache[view]=cache
            else:
                ids,classes,png,labels=load_fixed(resolve(row['fixed_dir']),name); protect(png,row['fixed_png_sha256']); protect(labels,row['fixed_classes_sha256']); batch=None; gtdata=None
            validate_prediction(ids,classes,rgb.shape[:2])
            cachepath=protect(resolve(viewcfg['cache_dir'])/(name+'.pt'),capture['captures'][name]['cache_sha256']); cache=torch.load(cachepath,map_location='cpu',weights_only=False)
            if cache['source']!=name: raise RuntimeError('Whole-cache source binding differs')
            if (array_sha(input_tensor)!=cache['image_sha256'] or array_sha(cache['image'])!=cache['image_sha256']
                    or list(input_tensor.stride())!=cache['image_stride'] or list(cache['image'].stride())!=cache['image_stride']
                    or cache['image_stride']!=capture['captures'][name]['input_stride']):
                raise RuntimeError('Native whole RGB/letterbox/cache values or stride differ')
            cache['batch']=batch; whole_cache[name]=cache
            geometry=(batch['geometry'] if train else None)
            shape=list(rgb.shape[:2]); pad_h=1024-int(round(shape[0]*1024/max(shape))); pad_w=1024-int(round(shape[1]*1024/max(shape)))
            if (input_ph!=pad_h or input_pw!=pad_w or (train and (gt.shape!=rgb.shape[:2] or geometry['resized_height']!=1024-pad_h or geometry['resized_width']!=1024-pad_w))): raise RuntimeError('Native RGB/GT/letterbox coordinates differ')
            if train and {k:array_sha(v) for k,v in batch.items() if torch.is_tensor(v)}!=capture['captures'][name]['common_batch_sha256']: raise RuntimeError('Old whole commonGT target changed')
            records[name]=dict(kind=kind,rgb=rgb,ids=ids,classes=classes,png=png,shape=shape,pad_h=pad_h,pad_w=pad_w,gtdata=gtdata,
                baseline_probability_sha256=capture['captures'][name]['baseline_probability_sha256'])
        if len(records)!=40: raise RuntimeError('Complete32+fixed8 source scope required')
        def cache_inputs(cache):
            if (array_sha(cache['image'])!=cache['image_sha256'] or list(cache['image'].stride())!=cache.get('image_stride',cache.get('image_receipt',{}).get('input_stride'))
                    or tensor_digest([(str(i),v) for i,v in enumerate(cache['features'])])!=cache['feature_sha256'] or [list(v.stride()) for v in cache['features']]!=cache['feature_strides']): raise RuntimeError('Cached values/layout mutated')
            if cache.get('target_sha256') is not None and {k:array_sha(v) for k,v in cache['batch'].items()}!=cache['target_sha256']: raise RuntimeError('Crop target cache mutated')
            image=feature_copy(cache['image'],'cuda'); features=[feature_copy(v,'cuda') for v in cache['features']]
            batch={k:v.to('cuda') for k,v in cache['batch'].items() if torch.is_tensor(v)} if cache.get('batch') else None
            return image,features,batch
        for name,cache in whole_cache.items():
            if array_sha(cache['image'])!=capture['captures'][name].get('image_sha256',cache['image_sha256']) or tensor_digest([(str(i),v) for i,v in enumerate(cache['features'])])!=capture['captures'][name]['feature_sha256']: raise RuntimeError('Whole frozen cache receipt differs')
        if resuming:
            # 只读取原封存源码；当前导入集合也必须相同，不能在失败恢复时覆盖旧归档。
            code=resumed_state['identity']['protected_sources']; virtual=[]; loaded=set()
            for module in list(sys.modules.values()):
                name=getattr(module,'__file__',None)
                if name:
                    path=Path(name).resolve()
                    if path.suffix=='.py' and path.is_relative_to(ROOT) and path.relative_to(ROOT).parts[0] not in ('output','outputs'):
                        if path.exists(): loaded.add(path.relative_to(ROOT).as_posix())
                        else: virtual.append(path.relative_to(ROOT).as_posix())
            if loaded!=set(code): raise RuntimeError('Resume loaded project source set differs')
            virtual=sorted(set(virtual)); verify_resume_archives(out,resumed_state['identity'])
            if canonical_sha(read(out/'config.json'))!=canonical_sha(config): raise RuntimeError('Resume merged config archive differs')
        else:
            code,virtual=protected_sources(out); write(out/'config.json',config)
        identity=dict(format=FORMAT,base_checkpoint_sha256=old['checkpoint_sha256'],initial_head_sha256=initial_head,base_state_sha256=base_state,
            preflight_report_sha256=file_sha(pfroot/'report.json'),preflight_rows_sha256=file_sha(pfroot/'rows.json'),pref_gate=gate,
            manifest_sha256=old['manifest_sha256'],control_head_sha256=viewcfg['control_head_digest'],schedule_sha256=canonical_sha(schedule),
            recipe_sha256=canonical_sha(cfg),precision=precision,protected_inputs=protected,protected_sources=code,total_steps=total,mode='smoke' if args.smoke else 'formal')
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'])
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        snapshots={}; steps=[]; start=0
        if resuming:
            start=validate_resume_state(resumed_state,identity,schedule,architecture,total)
            head.load_state_dict(resumed_state['semantic_state_dict'],strict=True); optimizer.load_state_dict(resumed_state['optimizer']); restore_rng(resumed_state['rng'])
            snapshots=resumed_state['snapshots']; steps=resumed_state['steps']; del resumed_state
        def forward(head,cache):
            image,features,batch=cache_inputs(cache); logits=head(features,image)
            if isinstance(logits,dict): logits=logits['semantic_logits']
            return logits,batch
        def evaluate(step,all_sources=False):
            selected,crop_sources=evaluation_sources(records,[r['name'] for r in test_cases],step,total,bool(args.smoke)); results={}
            for name in selected:
                item=records[name]
                with torch.no_grad(): logits,_=forward(head,whole_cache[name]); _,p=probability(logits,item)
                classes,scores=revote_instances(item['ids'],p)
                if step==0: verify_whole_baseline(name,p,item['baseline_probability_sha256'],item['ids'],item['classes'])
                folder=out/f's{step:02d}'/name; folder.mkdir(parents=True,exist_ok=True); copy_fixed_mask(item['png'],folder/item['png'].name); write(folder/(name+'_class.json'),classes)
                area=np.bincount(item['ids'].ravel()); ferrite=[int(k) for k,v in classes.items() if v]; pixels=sum(int(area[k]) for k in ferrite)
                row=dict(kind=item['kind'],source=name,probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),png_sha256=file_sha(folder/item['png'].name),
                    instances=len(classes),ferrite_count=len(ferrite),ferrite_pixels=pixels,ferrite_mean_area=pixels/len(ferrite) if ferrite else None,
                    changed_ids=[k for k in classes if classes[k]!=item['classes'][k]],instance_scores=scores,
                    old_control64=control_snapshots[name])
                gt=item['gtdata']
                if gt is not None:
                    object_rows=probability_rows(gt['cohort'],p); baseline_rows=probability_rows(gt['cohort'],whole_cache[name]['reference_probability'])
                    row['training']=dict(objects=object_rows,summary=summarize_condition(baseline_rows,object_rows),fixed_pair_confusion=fixed_confusion(gt['pairs'],gt['classes'],classes),matching=gt['stats'])
                if args.smoke or name in MONITORS or item['kind']=='test': preview(folder/'monitor.png',item['rgb'],item['ids'],classes,p,'grain s'+str(step),320)
                write(folder/'report.json',row); results[name]=row
            crop_results={}
            for name in crop_sources:
                item=records[name]; gt=item['gtdata']
                for cls in (1,0):
                    view=name+'_'+str(cls); cache=crop_cache[view]; y0,y1,x0,x1=cache['box']
                    record=dict(shape=[y1-y0,x1-x0],pad_h=cache['image_receipt']['pad_h'],pad_w=cache['image_receipt']['pad_w'])
                    native_gt=gt['gt'][y0:y1,x0:x1]; native_original=gt['original'][y0:y1,x0:x1]
                    with torch.no_grad(): logits,_=forward(head,cache); _,p=probability(logits,record)
                    metrics=anchor_probability(p,native_gt,native_original,cache['anchor'])
                    if step==0 and metrics!=crop_rows[view]['crop']: raise RuntimeError('R1 e0 cached crop anchor replay differs')
                    row=dict(view=view,source=name,anchor=cache['anchor'],box=list(cache['box']),metrics=metrics,
                        probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),R1_e0=crop_rows[view]['crop'],R1_whole_e0=crop_rows[view]['whole'],test_used=False)
                    folder=out/f's{step:02d}'/'crops'/view; folder.mkdir(parents=True,exist_ok=True)
                    crop_preview(folder/'monitor.png',item['rgb'][y0:y1,x0:x1],p,native_gt,native_original,gt['classes'],cache['anchor'])
                    if step==total:
                        with torch.no_grad(): control_logits,_=forward(control_head,cache); _,control_p=probability(control_logits,record)
                        row['readonly_oldA64']=dict(metrics=anchor_probability(control_p,native_gt,native_original,cache['anchor']),
                            probability_sha256=hashlib.sha256(control_p.tobytes()).hexdigest(),used_as_initialization=False,training_repeated=False)
                        crop_preview(folder/'control.png',item['rgb'][y0:y1,x0:x1],control_p,native_gt,native_original,gt['classes'],cache['anchor'])
                    write(folder/'report.json',row); crop_results[view]=row
            verify_frozen_model(control_head,viewcfg['control_head_digest'])
            return dict(step=step,rows=results,crop_rows=crop_results,scope='smoke fixed4 training+fixed4 unlabeled' if args.smoke else ('all32+fixed8' if all_sources else 'fixed6 training+fixed8 unlabeled'),test_used_for_selection=False)
        def save(step,complete=False):
            payload=dict(format=HEAD_FORMAT,terminal_only=True,do_not_replace_geometry_upstream=True,identity=identity,architecture=architecture,
                step=step,total_steps=total,complete=complete,schedule=schedule,semantic_state_dict=head.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(),steps=steps,snapshots=snapshots)
            tmp=out/'state.tmp.pt'; torch.save(payload,tmp); tmp.replace(out/'last_state.pt')
            if step in cfg['snapshots'] or step==total: torch.save(payload,out/f'head_s{step:02d}.pt')
        write(out/'status.json',dict(status='running',training=True,total_steps=total,step=start,smoke=bool(args.smoke)))
        with torch.autocast(device_type='cuda',enabled=False),patch('cv2.watershed',side_effect=RuntimeError('No watershed permitted')),patch('models.backend_adaptation.restore_first',side_effect=RuntimeError('No D5a permitted')):
            if not snapshots: snapshots['0']=evaluate(0,all_sources=True); save(0)
            for index,(name,cls) in enumerate(schedule[:total],1):
                if index<=start: continue
                optimizer.zero_grad(set_to_none=True); whole_logits,whole_batch=forward(head,whole_cache[name]); crop_logits,crop_batch=forward(head,crop_cache[name+'_'+str(cls)])
                whole_loss=compute_semantic_loss(criterion,whole_logits,whole_batch); crop_loss=compute_semantic_loss(criterion,crop_logits,crop_batch); loss=.5*whole_loss+.5*crop_loss
                if not torch.isfinite(loss): raise FloatingPointError('Finite actual averaged common loss required')
                loss.backward(); grouped={prefix:sum(float(p.grad.detach().double().square().sum()) for key,p in head.named_parameters() if key.startswith(prefix+'.') and p.grad is not None)**.5 for prefix in ('seg_fpn','seg_branch','semantic_residual')}
                norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),cfg['grad_clip'],error_if_nonfinite=True))
                if norm<=0 or grouped['seg_fpn']<=0 or grouped['seg_branch']<=0: raise RuntimeError('Actual averaged loss must reach coarse trainable parameters')
                optimizer.step()
                if tensor_digest(head.named_buffers())!=buffers or any(m.training for m in head.modules()): raise RuntimeError('Copied eval head buffers/mode changed')
                verify_frozen_model(model,base_state)
                steps.append(dict(step=index,source=name,crop_class=cls,whole_loss=float(whole_loss.detach()),crop_loss=float(crop_loss.detach()),averaged_loss=float(loss.detach()),
                    optimizer_updates=1,head_views=2,region_extra_weight=0,gradient_norm_before_clip=norm,parameter_gradient_norms=grouped))
                if index in cfg['snapshots'] or index==total: snapshots[str(index)]=evaluate(index,all_sources=index==total)
                save(index); write(out/'steps.json',steps); write(out/'status.json',dict(status='running',training=True,step=index,total_steps=total))
                print('SEMANTIC_GRAIN_STEP',index,name,cls,float(loss.detach()),flush=True); del whole_logits,crop_logits,whole_loss,crop_loss,loss; gc.collect()
            # 原部署头已安装禁止forward的hook；复制候选头，避免把保护hook带入重载模型。
            saved=torch.load(out/f'head_s{total:02d}.pt',map_location='cpu',weights_only=False); reloaded=deepcopy(head).eval().requires_grad_(False); reloaded.load_state_dict(saved['semantic_state_dict'],strict=True)
            final=tensor_digest(head.state_dict().items())
            if tensor_digest(reloaded.state_dict().items())!=final or final==initial_head: raise RuntimeError('Strict terminal reload or actual parameter update gate failed')
            for name in snapshots[str(total)]['rows']:
                item=records[name]
                with torch.no_grad(): logits,_=forward(reloaded,whole_cache[name]); _,p=probability(logits,item)
                classes,replay_scores=revote_instances(item['ids'],p); expected=snapshots[str(total)]['rows'][name]
                folder=out/f's{total:02d}'/name
                verify_terminal_output(p,classes,expected,item['png'],folder/item['png'].name,folder/(name+'_class.json'))
                # 新回执保存实例均值票分；旧回执缺此字段时不伪造置信度记录。
                if 'instance_scores' in expected and replay_scores!=expected['instance_scores']:
                    raise RuntimeError('Strict final native instance score replay differs')
            for view,expected in snapshots[str(total)]['crop_rows'].items():
                cache=crop_cache[view]; item=records[expected['source']]; y0,y1,x0,x1=cache['box']; gt=item['gtdata']
                record=dict(shape=[y1-y0,x1-x0],pad_h=cache['image_receipt']['pad_h'],pad_w=cache['image_receipt']['pad_w'])
                with torch.no_grad(): logits,_=forward(reloaded,cache); _,p=probability(logits,record)
                metrics=anchor_probability(p,gt['gt'][y0:y1,x0:x1],gt['original'][y0:y1,x0:x1],cache['anchor'])
                verify_crop_replay(p,metrics,expected)
            verify_frozen_model(control_head,viewcfg['control_head_digest'])
        verify_frozen_model(model,base_state)
        if precision_record()!=precision or any(file_sha(Path(p))!=v for p,v in protected.items()) or any(file_sha(ROOT/p)!=v for p,v in code.items()): raise RuntimeError('Sealed precision/input/code identity changed')
        if len(steps)!=total: raise RuntimeError('One optimizer update per source exposure budget differs')
        save(total,complete=True)
        report=dict(format=FORMAT,complete=True,training=False,identity=identity,steps=total,optimizer_updates=total,effective_head_views=total*2,
            old_control_readonly=control_receipt,original_initialization=True,final_head_sha256=final,upstream_frozen_exact=True,no_encoder_forward=True,
            zero_affinity_forward=True,zero_D5a_forward=True,zero_watershed=True,strict_reload_final_scope_exact=True,
            final_sources=len(snapshots[str(total)]['rows']),strict_reload_final40_exact=not bool(args.smoke),fixed_png_bytes_exact=True,
            final_crop_views=len(snapshots[str(total)]['crop_rows']),strict_reload_crop_scope_exact=True,
            full_test_inference=False,official_submission=False,best_deployment_changed=False,conservative_parameters=91253055,
            snapshots=snapshots,inherited_config_sources=config_sources,virtual_module_paths=virtual,
            caveat='Combined context/texture-scale/neighbor-pool training. Seen incomplete/coarse processed GT is diagnostic; test observations never select epochs. Terminal-only copied head, original semantic stays upstream for geometry; no official gain established.')
        write(out/'report.json',report); write(out/'status.json',dict(status='complete',training=False,steps=total,smoke=bool(args.smoke))); print('SEMANTIC_GRAIN_COMPLETE',total,flush=True); return report
    except Exception as error:
        write(out/'status.json',dict(status='failed',training=False,error=repr(error))); raise
    finally:
        for hook in hooks: hook.remove()


def main():
    from utils.config import load_config
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',default='config/train/semantic_grain_fit.yaml'); parser.add_argument('--output'); parser.add_argument('--resume')
    mode=parser.add_mutually_exclusive_group(); mode.add_argument('--go64',action='store_true'); mode.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    if args.go64 or args.smoke: run(args)
    elif args.resume: raise ValueError('Resume requires explicit --go64 or --smoke matching saved mode')
    else: print(json.dumps(plan(load_config(args.config)),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
