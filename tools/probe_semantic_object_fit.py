# -*- coding: utf-8 -*-
"""原图末端语义64更新A/B机制拟合；固定reflect实例，不调用恢复/几何/分水岭。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools.probe_semantic_fixed import class_independent_pairs,copy_fixed_mask,file_sha,fixed_confusion,write
from tools.probe_semantic_reflect import load_fixed,protected_sources,validate_prediction
from tools.semantic_crossover import revote_instances
from tools.run_marker_reflect import precision_record
from train_backend_adaptation import tensor_digest
from utils.affinity_deployment import crop_letterbox_output,prepare_image
from utils.config import load_config,project_path
from utils.retention_fit import capture_rng,restore_rng
from utils.semantic_object_fit import (array_sha,build_semantic_batch,canonical_sha,feature_copy,
    object_outcomes,paired_schedule,region_probability_loss,select_regions)


FORMAT='semantic_object_fit_v1'
HEAD_FORMAT='semantic_object_terminal_only_v1'
EXPECTED_LOSS=dict(dice_weight=.3,instance_weight=.75,core_radius=3,core_min_pixels=12,
    core_boundary_threshold=.2,class_balance=True,ferrite_weight=1.,hard_gamma=.5,
    hard_floor=.5,pool_weight=.5,thin_weight=1.5)
MONITORS=['train_001','train_173','train_558','train_889','train_175','train_873']


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def validate_recipe(cfg):
    expected=dict(format=FORMAT,input_size=1024,expected_sources=32,expected_regions=182,
        expected_F_regions=101,expected_P_regions=81,expected_initial_region_errors=64,
        expected_geometric_pairs=4918,steps=64,learning_rate=2e-5,weight_decay=1e-4,
        adam_eps=1e-4,grad_clip=1.,region_weight=.5,snapshots=[0,16,32,64],seed=20261004,
        head_mode='eval_mode_fitting',monitor_sources=MONITORS,cache_features_on_cpu=True,
        test_scope='fixed_eight_unlabeled_only')
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Registered paired terminal semantic mechanism-fit recipe differs')


def probability(logits,record):
    native=crop_letterbox_output(logits.float(),1024,record['pad_h'],record['pad_w'],record['shape'])
    # 原部署顺序：CUDA插值恢复logit→CPU sigmoid→float32实例概率均值。
    return native,np.ascontiguousarray(native.detach().cpu().sigmoid()[0,0].numpy(),dtype=np.float32)


def frozen_check(model,digest):
    if (tensor_digest(model.state_dict().items())!=digest
            or any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(m.training for m in model.modules())):
        raise RuntimeError('Immutable full R1 encoder/LoRA/affinity/base semantic changed')


def preview(path,rgb,ids,classes,probability_map,label,width):
    edge=np.zeros(ids.shape,bool);edge[:,1:]|=ids[:,1:]!=ids[:,:-1];edge[1:]|=ids[1:]!=ids[:-1]
    overlay=rgb.astype(np.float32)*.35
    for pid,cls in classes.items():overlay[ids==int(pid)]+=.65*np.asarray((245,186,57) if cls else (65,133,220))
    overlay=np.clip(overlay,0,255).astype(np.uint8);overlay[edge]=(40,40,40)
    heat=cv2.cvtColor(cv2.applyColorMap(np.rint(probability_map*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
    panels=[]
    for image,text in ((rgb,'raw'),(heat,'P(F): 0..1'),(overlay,label+' fixed geometry')):
        thumb=cv2.resize(image,(width,max(1,round(image.shape[0]*width/image.shape[1]))),interpolation=cv2.INTER_AREA)
        thumb=cv2.copyMakeBorder(thumb,25,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
        cv2.putText(thumb,text,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1,cv2.LINE_AA);panels.append(thumb)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),cv2.cvtColor(np.concatenate(panels,axis=1),cv2.COLOR_RGB2BGR)):
        raise IOError('Private monitor thumbnail write failed')


def add_totals(rows):
    outcomes=[r['objects'] for r in rows.values() if r['kind']=='train']
    keys=('fixed_regions','initial_wrong','wrong','corrected','original_correct_regressed',
          'F_to_P_wrong','P_to_F_wrong','all_fixed_predictions','ferrite_count','ferrite_pixels',
          'changed_instances','changed_pixels')
    result={key:sum(r[key] for r in outcomes) for key in keys}
    if result['fixed_regions']!=182 or result['initial_wrong']!=64:
        raise RuntimeError('Fixed GT-selected region denominator changed')
    confusion=[r['fixed_pair_confusion'] for r in rows.values() if r['kind']=='train']
    result['fixed_pair_confusion']={key:sum(r[key] for r in confusion) for key in confusion[0]}
    if result['fixed_pair_confusion']['correct']+result['fixed_pair_confusion']['wrong']!=4918:
        raise RuntimeError('Class-independent fixed 4918 pair denominator changed')
    result['ferrite_size_buckets']={name:{key:sum(r['ferrite_size_buckets'][name][key] for r in outcomes)
        for key in ('count','pixels')} for name in outcomes[0]['ferrite_size_buckets']}
    tests=[r['objects'] for r in rows.values() if r['kind']=='test']
    result['unlabeled_test']={key:sum(r[key] for r in tests) for key in
        ('all_fixed_predictions','ferrite_count','ferrite_pixels','changed_instances','changed_pixels')} if tests else None
    return result


def run(args):
    from models.backend_adaptation import load_backend
    from train_direct_semantic_affinity import build_semantic_criterion,compute_semantic_loss
    merged=load_config(args.config);cfg=merged['semantic_object_fit'];validate_recipe(cfg)
    config=deepcopy(merged);config.pop('semantic_object_fit')
    if not args.go64:
        print('Validated 32-source/64-step plan; use --go64 for actual sam2_env CUDA training.');return
    if not torch.cuda.is_available():raise RuntimeError('Actual sam2_env CUDA required')
    resolve=lambda p:Path(project_path(config,p)).resolve()
    out=resolve(cfg['output_dir'])
    if not out.is_relative_to(ROOT/'outputs'):raise ValueError('Ignored outputs required')
    resumed=bool(args.resume)
    recovering=bool(args.recover_capture)
    reuse_capture=resumed or recovering
    if resumed and recovering:raise ValueError('Capture recovery is separate from formal optimizer resume')
    if out.exists() and not reuse_capture:raise FileExistsError('Fresh output, explicit resume, or capture recovery required')
    if recovering:
        if (not (out/'capture.json').is_file() or read(out/'status.json').get('status')!='failed'
                or (out/'last_state.pt').exists() or (out/'last_state.tmp.pt').exists()
                or list(out.glob('*/head_s*.pt'))):
            raise ValueError('Capture-only recovery requires failed completed capture and no optimizer/head checkpoints')
    if resumed and (not (out/'last_state.pt').is_file() or read(out/'status.json')['status']=='complete'):
        raise ValueError('Resume requires an unfinished owned run and complete optimizer checkpoint')
    out.mkdir(parents=True,exist_ok=reuse_capture);args.owned_output=out
    torch.set_num_threads(4);cv2.setNumThreads(2);started=time.time()
    protected={}
    def protect(path,expected=None):
        path=resolve(path);actual=file_sha(path)
        if expected is not None and actual!=expected:raise RuntimeError('Protected input SHA differs: '+str(path))
        protected[str(path)]=actual;return path
    protect(args.config)
    checkpoint=protect(cfg['checkpoint'],cfg['checkpoint_sha256'])
    manifest=read(protect(cfg['manifest'],cfg['manifest_sha256']))
    ref_inputs=read(protect(cfg['reference_inputs']))
    ref_rows={r['source']:r for r in read(protect(cfg['reference_rows'],cfg['reference_rows_sha256']))}
    if (manifest.get('status')!='complete' or manifest['teacher_checkpoint_sha256']!=cfg['checkpoint_sha256']
            or manifest['test_training'] is not False or len(manifest['rows'])!=32
            or len({r['source'] for r in manifest['rows']})!=32
            or manifest['teacher_readout']!='native1024 overlap .25, reflect32, original continuous WS elevation/raw semantic vote'):
        raise RuntimeError('Complete fixed current R1 reflect cohort required')
    test_cases=[r for r in ref_inputs['cases'] if r['kind']=='test']
    if len(test_cases)!=8 or len({r['name'] for r in test_cases})!=8:
        raise RuntimeError('Fixed predeclared eight unlabeled test sources required')
    model,bundle=load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False)
    if bundle['epoch']!=20 or model.semantic_lora is not None:
        raise RuntimeError('Original R1 S-align/raw semantic route required')
    loss_cfg=bundle['config']['direct_semantic_affinity']['semantic_loss']
    if canonical_sha(loss_cfg)!=canonical_sha(EXPECTED_LOSS):raise RuntimeError('Embedded actual S-align pixel/core/pool loss config differs')
    criterion_config={'direct_semantic_affinity':{'semantic_loss':deepcopy(loss_cfg)}}
    criterion=build_semantic_criterion(criterion_config,'cuda')
    if criterion.semantic_instance_pool_weight!=.5 or not criterion.freeze_boundary:
        raise RuntimeError('Original pool .5 and semantic-only criterion required')
    full_digest=tensor_digest(model.state_dict().items());head_digest=tensor_digest(model.semantic_decoder.state_dict().items())
    original_buffers=tensor_digest(model.semantic_decoder.named_buffers())
    precision=precision_record()
    if precision!=manifest['precision']:raise RuntimeError('Original actual precision differs')
    base_params=sum(p.numel() for p in model.parameters())
    additional_head_params=sum(p.numel() for p in model.semantic_decoder.parameters())
    if base_params+3283651!=84951045:
        raise RuntimeError('Previously verified R1 plus frozen D5a parameter ledger differs')
    params=84951045+additional_head_params
    if params>=500000000:raise RuntimeError('Parameter budget exceeded')
    prior=read(out/'capture.json') if reuse_capture else None
    if resumed:
        code=prior['protected_code'];virtual=prior['virtual_code_paths']
        for path,digest in code.items():
            if file_sha(ROOT/path)!=digest:raise RuntimeError('Resumed protected code changed: '+path)
    elif recovering:
        expected_json_failure_code={
            'tools/probe_semantic_object_fit.py':'fbd738f80c3050c76142b7bc06452e0464141e7a70e752ee79f11ea29d3b65ae',
            'utils/semantic_object_fit.py':'5e5b1c25fb9de42547a7e692a03cc862472f50c433e15d0bbf6819f0e539eb5a'}
        if (prior.get('format')!=FORMAT or len(prior.get('captures',{}))!=40
                or prior.get('encoder_calls')!=40 or prior.get('fixed_regions')!=182
                or prior.get('geometric_pairs')!=4918
                or canonical_sha(prior.get('config'))!=canonical_sha(merged)):
            raise RuntimeError('Original completed 40-source capture receipt and unchanged configuration required')
        for path,digest in prior['protected_code'].items():
            archived=out/'source'/path
            if file_sha(archived)!=digest:raise RuntimeError('Original failed-run code snapshot changed: '+path)
            if path in expected_json_failure_code:
                if digest!=expected_json_failure_code[path]:raise RuntimeError('Unregistered original JSON-failure source version')
            elif file_sha(ROOT/path)!=digest:raise RuntimeError('Unrelated source changed during capture recovery: '+path)
        if not set(expected_json_failure_code).issubset(prior['protected_code']):
            raise RuntimeError('Explicit original JSON-failure tool/helper source receipt missing')
        for path,digest in prior['protected_inputs'].items():
            if file_sha(path)!=digest:raise RuntimeError('Original frozen input/cache changed before capture recovery')
        old_source=out/'source_json_failure'
        if old_source.exists():raise FileExistsError('Previously archived JSON-failure source exists; do not overwrite it')
        shutil.copytree(out/'source',old_source)
        shutil.copyfile(out/'capture.json',out/'capture_json_failure.json')
        code,virtual=protected_sources(out)
    else:code,virtual=protected_sources(out)
    write(out/'status.json',dict(status='running',stage='raw_feature_capture',training=False))
    calls={'encoder':0,'affinity':0,'D5a':0,'watershed':0}
    def forbid(*_):calls['affinity']+=1;raise RuntimeError('No affinity forward permitted')
    hook=model.affinity_decoder.register_forward_pre_hook(forbid)
    records={};cpu_cache={};selection={};captures={}
    cache_dir=out/'cache';cache_dir.mkdir(exist_ok=reuse_capture)
    all_sources=[dict(kind='train',name=row['source'],row=row) for row in manifest['rows']]
    all_sources.extend(dict(kind='test',name=row['name'],row=row) for row in test_cases)
    try:
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False), \
                patch('cv2.watershed',side_effect=RuntimeError('No watershed permitted')), \
                patch('models.backend_adaptation.restore_first',side_effect=RuntimeError('No D5a permitted')):
            for item in all_sources:
                name,kind,row=item['name'],item['kind'],item['row']
                image_path=protect(row['image'] if kind=='train' else row['path'],row['image_sha256'] if kind=='train' else row['source_sha256'])
                if kind=='train':
                    mask_path=protect(resolve(cfg['manifest']).parent/'teacher'/(name+'_inst.png'),row['prediction_hashes']['png_sha256'])
                    class_path=protect(mask_path.with_name(name+'_class.json'),row['prediction_hashes']['classes_sha256'])
                    ids=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED);classes=read(class_path)
                    gtpath=protect(row['manual_gt'],row['manual_gt_sha256'])
                    gcpath=protect(gtpath.with_name(name+'_class.json'),manifest['protected_inputs'][str(gtpath.relative_to(ROOT)).replace('\\','/') .removesuffix('_gt.npz')+'_class.json'])
                    with np.load(gtpath,allow_pickle=False) as z:
                        gt=z['instance_map'].copy();original=z['original_covered'].astype(bool)
                        if not np.array_equal(z['filled'].astype(bool),(gt>0)&~original) or not np.array_equal(z['residual_unknown'].astype(bool),gt==0):
                            raise RuntimeError('Processed original/filled/unknown provenance differs')
                    gt_classes={int(k):v for k,v in read(gcpath).items()}
                    regions,selection_audit=select_regions(ids,gt,original,gt_classes)
                    pairs,pair_stats=class_independent_pairs(gt,original,ids)
                    batch=build_semantic_batch(gt,original,gt_classes,1024)
                else:
                    ids,classes,mask_path,class_path=load_fixed(resolve(row['fixed_dir']),name)
                    protect(mask_path,row['fixed_png_sha256']);protect(class_path,row['fixed_classes_sha256'])
                    gt,original,gt_classes=None,None,None;regions=[];pairs=[];pair_stats=None;batch=None;selection_audit=None
                rgb,image,ph,pw=prepare_image(image_path,1024,'cuda')
                validate_prediction(ids,classes,rgb.shape[:2])
                record=dict(name=name,kind=kind,shape=list(rgb.shape[:2]),pad_h=ph,pad_w=pw,
                    image_path=str(image_path),mask_path=str(mask_path),classes=classes,regions=regions,
                    gt_classes=gt_classes,pairs=pairs,pair_stats=pair_stats,rgb=rgb,ids=ids)
                path=cache_dir/(name+'.pt')
                if reuse_capture:
                    protect(path,prior['captures'][name]['cache_sha256'])
                    cached=torch.load(path,map_location='cpu',weights_only=False)
                    if cached['source']!=name:raise RuntimeError('Cache source ownership changed')
                    loaded_image=feature_copy(cached['image'],'cuda')
                    if loaded_image.stride()!=image.stride() or not torch.equal(loaded_image,image):
                        raise RuntimeError('Cached actual raw RGB tensor/layout changed')
                else:
                    features=model.encoder(image.float());calls['encoder']+=1
                    logits=model.semantic_decoder(features,image)
                    if isinstance(logits,dict):logits=logits['semantic_logits']
                    _,reference_probability=probability(logits,record)
                    cached=dict(source=name,image=feature_copy(image,'cpu'),features=[feature_copy(v,'cpu') for v in features],
                        logit_shape=list(logits.shape),logit_stride=list(logits.stride()),reference_probability=reference_probability,
                        feature_sha256=tensor_digest([(str(i),v) for i,v in enumerate(features)]),
                        feature_strides=[list(v.stride()) for v in features],
                        image_sha256=array_sha(image),image_stride=list(image.stride()))
                    torch.save(cached,path);protected[str(path)]=file_sha(path)
                    cached=torch.load(path,map_location='cpu',weights_only=False)
                    loaded_image=feature_copy(cached['image'],'cuda')
                features=[feature_copy(v,'cuda') for v in cached['features']]
                replay=model.semantic_decoder(features,loaded_image)
                if isinstance(replay,dict):replay=replay['semantic_logits']
                _,replayed_probability=probability(replay,record)
                if (list(replay.shape)!=cached['logit_shape'] or list(replay.stride())!=cached['logit_stride']
                        or not np.array_equal(replayed_probability,cached['reference_probability'])
                        or [list(v.stride()) for v in features]!=cached['feature_strides']
                        or tensor_digest([(str(i),v) for i,v in enumerate(features)])!=cached['feature_sha256']):
                    raise RuntimeError('CPU cache replay does not reproduce actual raw semantic/FP32/stride')
                voted,_=revote_instances(ids,replayed_probability)
                if voted!=classes:raise RuntimeError('Raw semantic baseline fixed classes differ: '+name)
                raw_sha=hashlib.sha256(replayed_probability.tobytes()).hexdigest()
                if name in ref_rows and raw_sha!=ref_rows[name]['routes']['salign_raw']['probability_sha256']:
                    raise RuntimeError('Existing historical raw native probability differs: '+name)
                record['baseline_probability_sha256']=raw_sha
                records[name]=record;cpu_cache[name]=cached
                if kind=='train':selection[name]=dict(regions=regions,audit=selection_audit)
                captures[name]=dict(kind=kind,shape=record['shape'],cache_sha256=file_sha(path),
                    feature_sha256=cached['feature_sha256'],feature_strides=cached['feature_strides'],
                    input_stride=list(image.stride()),baseline_probability_sha256=raw_sha,
                    current_original_model_cache_exact=True,historical_classes_exact=True,
                    historical_probability_SHA_checked=name in ref_rows,region_audit=selection_audit,
                    fixed_matching_sha256=canonical_sha(pairs),fixed_pairs=len(pairs),
                    common_batch_sha256={k:array_sha(v) for k,v in batch.items() if torch.is_tensor(v)} if batch else None)
                cached['batch']=batch
                print('SEMANTIC_CAPTURE',name,'regions',len(regions),'pairs',len(pairs),flush=True)
                del features,replay,image,loaded_image;gc.collect();torch.cuda.empty_cache()
        frozen_check(model,full_digest)
        selected=[r for s in selection.values() for r in s['regions']]
        initial_errors=sum(r['cls']!=records[name]['classes'][str(r['id'])] for name,s in selection.items() for r in s['regions'])
        if ((len(selected),sum(r['cls']==1 for r in selected),sum(r['cls']==0 for r in selected),initial_errors)
                !=(182,101,81,64) or sum(len(r['pairs']) for r in records.values())!=4918):
            raise RuntimeError('Actual fixed complete 182/101/81/64/4918 predeclared selection differs')
        schedule=paired_schedule(selection,cfg['seed'])
        identity=dict(checkpoint_sha256=cfg['checkpoint_sha256'],initial_head_sha256=head_digest,
            full_R1_state_sha256=full_digest,initial_BN_buffers_sha256=original_buffers,
            config_sha256=file_sha(args.config),code_sha256=canonical_sha(code),
            protected_inputs_sha256=canonical_sha(protected),captures_sha256=canonical_sha(captures),
            selection_sha256=canonical_sha(selection),schedule_sha256=canonical_sha(schedule),precision=precision,
            head_mode='eval_mode_fitting',head_format=HEAD_FORMAT)
        report=dict(format=FORMAT,complete=False,training=True,no_submission=True,terminal_only=True,
            immutable_instance_PNG=True,geometry_upstream_unchanged=True,full_test_inference=False,
            config=merged,embedded_loss_config=loss_cfg,common_support='processed-known including filled; unknown/padding ignored',
            additional_support='entire original-covered pure-phase nonframe predicted regions; labels solely from GT',
            classification_source='raw S-align terminal vote only; no restored image or encoder update',
            mechanism_fit='Full1024 clean raw, eval-mode head fitting, finite 64-source updates; not a reproduction of historical crop/augmentation training',
            parameter_count=params,parameter_count_includes_existing_D5a=True,
            diagnostic_resident_model_plus_copy_parameters=base_params+additional_head_params,
            verified_existing_deployment_parameters=84951045,existing_D5a_parameters=3283651,
            base_model_parameters=base_params,
            additional_terminal_head_parameters=additional_head_params,
            deployment_keeps_original_semantic_for_geometry=True,
            initial_head_sha256=head_digest,identity=identity,
            source_order=schedule,selection=selection,captures=captures,
            fixed_regions=182,ferrite_regions=101,pearlite_regions=81,initial_errors=64,geometric_pairs=4918,
            protected_inputs=protected,protected_code=code,virtual_code_paths=virtual,
            encoder_calls=calls['encoder'],zero_geometry_forwards=True,arms={})
        if recovering:
            if any(value!=prior['identity'].get(key) for key,value in identity.items() if key!='code_sha256'):
                raise RuntimeError('Capture recovery changed immutable source, selected GT, cache, layout, or runtime identity')
            if calls['encoder']!=0:raise RuntimeError('Capture-only recovery repeated an encoder capture')
            report['capture_recovery']=dict(original_encoder_captures=40,current_encoder_calls=0,
                optimizer_updates_before_recovery=0,formal_training_starts_at_zero=True,
                prior_capture_sha256=file_sha(out/'capture_json_failure.json'),
                old_code_sha256=prior['identity']['code_sha256'],new_code_sha256=identity['code_sha256'],
                only_source_repairs=['tools/probe_semantic_object_fit.py','utils/semantic_object_fit.py'],
                all40_cached_raw_probabilities_classes_layout_values_exact=True,
                frozen_inputs_and_selection_exact=True)
            write(out/'capture.json',report);state=None
        elif resumed:
            if prior['identity']!=identity:raise RuntimeError('Resume immutable source/code/capture identity differs')
            state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False)
            if (state.get('format')!=HEAD_FORMAT or state.get('identity')!=identity or state.get('mode')!='eval'
                    or state.get('terminal_only') is not True or state.get('do_not_replace_geometry_upstream') is not True
                    or state.get('base_checkpoint_sha256')!=cfg['checkpoint_sha256']
                    or state.get('architecture')!=bundle['architecture']['semantic_decoder']
                    or state.get('arm') not in ('control','region') or type(state.get('step')) is not int
                    or not 0<=state['step']<=64 or state.get('schedule')!=schedule
                    or len(state.get('steps',[]))!=state['step']
                    or any(type(s.get('step')) is not int for s in state.get('steps',[]))
                    or state.get('region_weight')!=(0. if state.get('arm')=='control' else .5)
                    or any(k not in state for k in ('semantic_state_dict','optimizer','rng','snapshots','completed_arms'))
                    or [s['step'] for s in state['steps']]!=list(range(1,state['step']+1))):
                raise RuntimeError('Incomplete or mismatched explicit terminal-only resume state')
            report['arms']=state['completed_arms']
        else:
            write(out/'capture.json',report);state=None
        def get_source(name):
            cache=cpu_cache[name]
            if (array_sha(cache['image'])!=cache['image_sha256'] or list(cache['image'].stride())!=cache['image_stride']
                    or tensor_digest([(str(i),v) for i,v in enumerate(cache['features'])])!=cache['feature_sha256']
                    or [list(v.stride()) for v in cache['features']]!=cache['feature_strides']
                    or (cache.get('batch') is not None and
                        {k:array_sha(v) for k,v in cache['batch'].items() if torch.is_tensor(v)}!=captures[name]['common_batch_sha256'])):
                raise RuntimeError('Actual in-memory frozen feature/image/target cache mutated')
            image=feature_copy(cache['image'],'cuda');features=[feature_copy(v,'cuda') for v in cache['features']]
            batch={k:v.to('cuda') for k,v in cache['batch'].items() if torch.is_tensor(v)} if cache.get('batch') else None
            return image,features,batch
        def evaluate(head,arm,step,all_sources=False):
            rows={};names=list(records) if all_sources else cfg['monitor_sources']
            for name in names:
                item=records[name];image,features,_=get_source(name)
                with torch.no_grad():
                    logits=head(features,image);logits=logits['semantic_logits'] if isinstance(logits,dict) else logits
                    _,p=probability(logits,item)
                classes,_=revote_instances(item['ids'],p);validate_prediction(item['ids'],classes,item['shape'])
                if step==0 and (classes!=item['classes'] or hashlib.sha256(p.tobytes()).hexdigest()!=item['baseline_probability_sha256']):
                    raise RuntimeError('Trainable-copy s0 full terminal output differs')
                folder=out/arm/f's{step:02d}'/name;folder.mkdir(parents=True,exist_ok=True)
                target=folder/(name+'_inst.png');copy_fixed_mask(item['mask_path'],target)
                write(folder/(name+'_class.json'),classes)
                outcome=object_outcomes(item['regions'],item['classes'],classes,item['ids'])
                row=dict(kind=item['kind'],source=name,objects=outcome,png_sha256=file_sha(target),
                    classes_sha256=file_sha(folder/(name+'_class.json')),
                    probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
                    fixed_pair_confusion=fixed_confusion(item['pairs'],item['gt_classes'],classes) if item['kind']=='train' else None)
                if item['kind']=='train':
                    strict_ids={r['id'] for r in item['regions']}
                    row['fixed_pair_errors']=[dict(pair,GT_class=item['gt_classes'][pair['gt']],
                        baseline_class=item['classes'][str(pair['pred'])],predicted_class=classes[str(pair['pred'])],
                        strict_region_eligible=pair['pred'] in strict_ids)
                        for pair in item['pairs'] if item['gt_classes'][pair['gt']]!=classes[str(pair['pred'])]]
                    row['fixed_pair_original_correct_regressed']=[dict(pair,GT_class=item['gt_classes'][pair['gt']],
                        predicted_class=classes[str(pair['pred'])],strict_region_eligible=pair['pred'] in strict_ids)
                        for pair in item['pairs'] if item['gt_classes'][pair['gt']]==item['classes'][str(pair['pred'])]
                        and item['gt_classes'][pair['gt']]!=classes[str(pair['pred'])]]
                write(folder/'report.json',row);rows[name]=row
                if name in cfg['monitor_sources'] or item['kind']=='test':
                    preview(folder/'monitor.png',item['rgb'],item['ids'],classes,p,arm+' s'+str(step),cfg['thumbnail_width'])
                del features,image,logits;gc.collect()
            return dict(step=step,scope='all32 train and fixed8 unlabeled test' if all_sources else 'fixed6 process thumbnails',
                        rows=rows,totals=add_totals(rows) if all_sources else None)
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        initial_rng=capture_rng()
        for arm,weight in (('control',0.),('region',.5)):
            if resumed and state['arm']=='region' and arm=='control':continue
            head=deepcopy(model.semantic_decoder).eval().requires_grad_(True)
            if tensor_digest(head.state_dict().items())!=head_digest:raise RuntimeError('A/B same initialization differs')
            restore_rng(initial_rng)
            # 固定最高面积错相代表源，仅用于通路梯度门禁；region选位仍是全部182，非按错选训练。
            gate_name='train_873';gate_item=records[gate_name]
            gate_rng=capture_rng();gate_image,gate_features,_=get_source(gate_name)
            gate_logits=head(gate_features,gate_image)
            gate_logits=gate_logits['semantic_logits'] if isinstance(gate_logits,dict) else gate_logits
            gate_native,_=probability(gate_logits,gate_item);gate_native=gate_native.cpu()
            gate_extra,_=region_probability_loss(gate_native,gate_item['ids'],gate_item['regions'])
            gate_gradients=torch.autograd.grad(gate_extra,(*head.parameters(),gate_native),allow_unused=True)
            gate_norm=sum(float(g.detach().double().square().sum()) for g in gate_gradients[:-1] if g is not None)**.5
            gate_native_norm=float(gate_gradients[-1].double().square().sum())**.5 if gate_gradients[-1] is not None else 0.
            if (not gate_norm>0 or not gate_native_norm>0
                    or any(not bool(torch.isfinite(g).all()) for g in gate_gradients if g is not None)
                    or any(p.grad is not None for p in head.parameters())
                    or tensor_digest(head.state_dict().items())!=head_digest):
                raise RuntimeError('Actual differentiable native-region loss did not reach immutable-start decoder parameters')
            report.setdefault('actual_region_gradient_gate',{})[arm]=dict(source=gate_name,passed=True,
                region_count=len(gate_item['regions']),actual_parameter_gradient_norm=gate_norm,
                native_selected_gradient_norm=gate_native_norm,optimizer_steps=0,initial_head_unchanged=True)
            restore_rng(gate_rng)
            del gate_image,gate_features,gate_logits,gate_native,gate_extra,gate_gradients;gc.collect()
            optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['learning_rate'],eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'])
            (out/arm).mkdir(exist_ok=True)
            snapshots={};steps=[];start_step=0
            if resumed and state['arm']==arm:
                head.load_state_dict(state['semantic_state_dict'],strict=True);optimizer.load_state_dict(state['optimizer'])
                restore_rng(state['rng']);snapshots=state['snapshots'];steps=state['steps'];start_step=state['step']
                with (out/arm/'steps.jsonl').open('w',encoding='utf8') as stream:
                    for row in steps:stream.write(json.dumps(row,allow_nan=False)+'\n')
            else:snapshots['0']=evaluate(head,arm,0)
            def commit(step,snapshot=False):
                saved=dict(format=HEAD_FORMAT,terminal_only=True,do_not_replace_geometry_upstream=True,
                    base_checkpoint_sha256=cfg['checkpoint_sha256'],arm=arm,step=step,mode='eval',
                    identity=identity,semantic_state_dict=head.state_dict(),optimizer=optimizer.state_dict(),
                    rng=capture_rng(),schedule=schedule,steps=steps,snapshots=snapshots,
                    completed_arms=report['arms'],region_weight=weight,architecture=bundle['architecture']['semantic_decoder'])
                tmp=out/'last_state.tmp.pt';torch.save(saved,tmp);tmp.replace(out/'last_state.pt')
                if snapshot:torch.save(saved,out/arm/f'head_s{step:02d}.pt')
            if start_step in cfg['snapshots'] and str(start_step) not in snapshots:
                snapshots[str(start_step)]=evaluate(head,arm,start_step,all_sources=start_step==64)
            commit(start_step,snapshot=start_step in cfg['snapshots'])
            for step,name in enumerate(schedule,1):
                if step<=start_step:continue
                image,features,batch=get_source(name);item=records[name]
                optimizer.zero_grad(set_to_none=True)
                logits=head(features,image);logits=logits['semantic_logits'] if isinstance(logits,dict) else logits
                if logits.shape!=batch['semantic_target'].shape:raise RuntimeError('Actual semantic logit/1024 target shape differs')
                common=compute_semantic_loss(criterion,logits,batch)
                native,_=probability(logits,item)
                # CPU sigmoid与部署一致；可微拷贝，原生支持门禁不误约束网络插值邻点。
                native_cpu=native.cpu()
                extra,detail=region_probability_loss(native_cpu,item['ids'],item['regions'])
                native_gradient=torch.autograd.grad(extra,native_cpu,retain_graph=True)[0]
                support=np.isin(item['ids'],[r['id'] for r in item['regions']])
                if bool(torch.count_nonzero(native_gradient[0,0][~torch.from_numpy(support)])):
                    raise RuntimeError('Added region gradient entered unselected native pixels')
                total=common+weight*extra.to(common.device)
                if not torch.isfinite(total):raise FloatingPointError('Nonfinite actual semantic objective')
                common_stats=dict(common_loss=float(common.detach()),
                    core_instance_loss=float(criterion.last_semantic_instance_loss),
                    core_instance_stats=deepcopy(criterion.last_semantic_instance_stats),
                    pixel_BCE_plus_dice=float(common.detach())-.75*float(criterion.last_semantic_instance_loss))
                total.backward();norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True))
                if norm<=0:raise RuntimeError('No trainable actual semantic head gradient')
                optimizer.step()
                if tensor_digest(head.named_buffers())!=original_buffers or any(m.training for m in head.modules()):
                    raise RuntimeError('BN buffers or decoder eval mode changed')
                row=dict(step=step,source=name,common=common_stats,region_weight=weight,
                    region_loss=float(extra.detach()),region_stats=detail,total_loss=float(total.detach()),
                    gradient_norm_before_clip=norm,extra_outside_selected_native_gradient_max=0.,
                    feature_sha256=cpu_cache[name]['feature_sha256'],common_batch_sha256=captures[name]['common_batch_sha256'])
                steps.append(row)
                with (out/arm/'steps.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                write(out/'status.json',dict(status='running',stage='training',arm=arm,step=step,total=64,training=True))
                commit(step)
                print('SEMANTIC_OBJECT_STEP',arm,step,name,'loss',float(total.detach()),flush=True)
                del features,image,logits,native,native_cpu,total,common,extra,native_gradient;gc.collect()
                if step in cfg['snapshots']:
                    snapshots[str(step)]=evaluate(head,arm,step,all_sources=step==64);commit(step,snapshot=True)
            saved=torch.load(out/arm/'head_s64.pt',map_location='cuda',weights_only=False)
            reloaded=deepcopy(model.semantic_decoder).eval().requires_grad_(False)
            reloaded.load_state_dict(saved['semantic_state_dict'],strict=True)
            if tensor_digest(reloaded.state_dict().items())!=tensor_digest(head.state_dict().items()):
                raise RuntimeError('Strict terminal head reload differs')
            for name,item in records.items():
                image,features,_=get_source(name)
                with torch.no_grad():
                    logits=reloaded(features,image);logits=logits['semantic_logits'] if isinstance(logits,dict) else logits
                    _,p=probability(logits,item)
                classes,_=revote_instances(item['ids'],p);expected=snapshots['64']['rows'][name]
                if (hashlib.sha256(p.tobytes()).hexdigest()!=expected['probability_sha256']
                        or classes!=read(out/arm/'s64'/name/(name+'_class.json'))
                        or file_sha(item['mask_path'])!=expected['png_sha256']):
                    raise RuntimeError('Strict reloaded complete final native classes/PNG/probability differ')
                del features,image,logits;gc.collect()
            report['arms'][arm]=dict(steps=64,microbatches=64,region_weight=weight,snapshots=snapshots,
                final_head_sha256=tensor_digest(head.state_dict().items()),strict_reload_exact=True,
                BN_buffers_exact=True,head_mode='eval',upstream_frozen=True)
            write(out/'report.json',report);commit(64,snapshot=True)
            frozen_check(model,full_digest)
            del head,reloaded,saved,optimizer;gc.collect();torch.cuda.empty_cache()
            resumed=False;state=None
        if calls['encoder']!=(0 if reuse_capture else 40):raise RuntimeError('One raw encoder capture or explicit cache reuse count violated')
        if calls['affinity']!=0:raise RuntimeError('Geometry path was used')
        frozen_check(model,full_digest)
        for name,cache in cpu_cache.items():
            if (array_sha(cache['image'])!=cache['image_sha256']
                    or tensor_digest([(str(i),v) for i,v in enumerate(cache['features'])])!=cache['feature_sha256']
                    or [list(v.stride()) for v in cache['features']]!=cache['feature_strides']):
                raise RuntimeError('Final actual frozen CPU feature/input values or layout changed')
        if precision_record()!=precision:raise RuntimeError('Actual precision changed')
        for path,digest in protected.items():
            if file_sha(path)!=digest:raise RuntimeError('Protected input/cache changed')
        for path,digest in code.items():
            if file_sha(ROOT/path)!=digest:raise RuntimeError('Protected code changed')
        report.update(complete=True,training=False,frozen_R1_exact=True,protected_inputs_exact=True,
            protected_code_exact=True,semantic_only_forward_counts=calls,elapsed_seconds=time.time()-started)
        write(out/'report.json',report);write(out/'status.json',dict(status='complete',training=False,steps_per_arm=64))
        print('SEMANTIC_OBJECT_COMPLETE',flush=True)
    finally:hook.remove()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/semantic_object_fit.yaml')
    parser.add_argument('--go64',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--recover-capture',action='store_true',help='仅修复已知JSON失败且零优化更新，严格复用40源捕获')
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        out=getattr(args,'owned_output',None)
        if out is not None and out.exists():write(out/'status.json',dict(status='failed',training=False,error=repr(error)))
        raise
