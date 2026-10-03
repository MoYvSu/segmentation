# -*- coding: utf-8 -*-
"""同R1起点、复用旧connect的processed-known负保留32步独立短测。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.semantic_targets import load_completed_semantic_source
from tools.affinity_feature_capture import capture_frozen_features, replay_head_boundary, validate_frozen_tile_features
from tools.analyze_affinity_bottleneck import source_name, load_prediction
from tools.analyze_affinity_structure import case_summary
from tools.analyze_marker_stages import verify_hashes
from tools.probe_affinity_bottleneck import measure, validate_case
from tools.probe_affinity_interior import R1_SHA, RESTORER_SHA, STAGE_SHA, cpu_runtime
from tools.probe_affinity_structure import decode_reflect, safety_metrics, target_result
from tools.probe_affinity_weak_retain import loss_options, parameter_change, scalar_record, save_thumbnail
from tools.probe_marker_stages import compare_actual
from tools.run_marker_reflect import array_sha, config_gate, precision_record, read, replay_candidates, save_prediction, sha, source_snapshot, write
from train_backend_adaptation import tensor_digest
from utils.affinity_interior_loss import build_interior_supervision, interior_objective
from utils.config import load_config, project_path
from utils.retention_fit import (actual_added_gradient_gate, canonical_sha, capture_rng,
    expand_supervision, restore_rng, validate_peer, validate_resume_state, retention_guard_objective,
    retention_gradient_summary)


from tools.probe_affinity_interior_fit import (training_schedule, validate_plan,
    gradient_summary, aggregate_safety, freeze_contract)


def run(args):
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from tools.run_affinity_sweep import runtime_contract

    merged=load_config(args.config); opts=merged['affinity_interior_fit']; variant=merged['affinity_retention_guard']
    config=deepcopy(merged);config.pop('affinity_interior_fit');config.pop('affinity_retention_guard')
    if (variant['format'] != 'affinity_retention_guard_v1' or variant['retention_objective'] != 'old_plus_added_kl'
            or variant['training_sources'] != 'original_three'
            or variant['actual_gate_requires_nonzero_added_parameter_gradient'] is not True
            or variant['added_retention_weight'] != 1.
            or variant['loss_contract'] != 'retention_guard_loss_v1'):
        raise ValueError('Explicit old-plus-added-KL, original-three, actual-gradient-gated variant required')
    if (opts['format']!='affinity_interior_fit_v1' or opts['steps']!=32 or opts['accumulation']!=3
            or opts['snapshots']!=[0,8,16,32] or opts['learning_rate']!=2e-5
            or (opts['full_weight'],opts['outer_weight'],opts['retention_weight'],opts['local_weight'])!=(1.,.5,1.,.25)):
        raise ValueError('Registered short-test budget and single-variable recipe required')
    output=Path(project_path(config,variant['output_dir'])).resolve()
    if args.preflight_only:
        if args.resume or args.go32:raise ValueError('Preflight-only cannot train or resume')
        output=output.with_name(output.name+'_preflight')
    resumed = bool(args.resume)
    if output.exists() and not resumed:raise FileExistsError('Fresh ignored output or explicit --resume required')
    if resumed and (not output.exists() or read(output/'status.json').get('status') == 'complete'):
        raise ValueError('Explicit resume requires an existing unfinished owned experiment')
    resume_path = output/'guard'/'last_state.pt'
    if resumed and not resume_path.exists():raise ValueError('Atomic committed update resume state missing')
    resume_state = torch.load(resume_path,map_location='cpu',weights_only=False) if resumed else None
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Ignored output required')
    if not args.go32 and not args.preflight_only:
        print('Validated recipe; --go32 explicitly starts this finite 32-update diagnostic.');return
    if not torch.cuda.is_available():raise RuntimeError('Actual sam2_env CUDA required')
    previous_paths = {key:Path(project_path(config,variant[key])) for key in ('previous_fit','previous_head','previous_steps')}
    for key,path in previous_paths.items():
        if sha(path) != variant[key+'_sha256']:raise RuntimeError('Sealed control '+key+' hash differs')
    peer = read(previous_paths['previous_fit']); paired_peer = validate_peer(peer,opts)
    verify_hashes(peer['protected_inputs']);verify_hashes(peer['protected_code'],code=True)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    paths={k:Path(project_path(config,opts[k])) for k in ('plan','structure','representation','receipt','stage')}
    for key in ('plan','structure','representation'):
        expected=opts[key+'_sha256']
        if not expected or sha(paths[key])!=expected:raise RuntimeError('Pinned private '+key+' hash differs')
    if sha(paths['stage'])!=STAGE_SHA:raise RuntimeError('Frozen stage receipt differs')
    structure,representation,receipt,stage,plan=[read(paths[k]) for k in ('structure','representation','receipt','stage','plan')]
    if not all(r.get('complete') is True for r in (structure,representation,stage,receipt)):
        raise RuntimeError('Completed original diagnostic artifacts required')
    if representation['safety_policy_version']!='known_matched_phase_only_v2':raise RuntimeError('Matched class semantics required')
    verify_hashes(representation['protected_inputs']);verify_hashes(representation['protected_code'],code=True)
    training=validate_plan(plan,structure,stage)
    r1=receipt['arms']['r1'];config_gate(config,r1['config'])
    precision,runtime=precision_record(),runtime_contract(config)
    if precision!=receipt['precision'] or runtime!=receipt['current_runtime'] or cpu_runtime()!=representation['CPU_runtime']:
        raise RuntimeError('Pinned actual CUDA/CPU runtime differs')
    checkpoint=Path(r1['checkpoint']);restore=Path(project_path(config,config['backend_adaptation']['restoration']['checkpoint']))
    if sha(checkpoint)!=R1_SHA or sha(restore)!=RESTORER_SHA:raise RuntimeError('R1/D5a identity differs')
    protected={**representation['protected_inputs'],**{str(p):sha(p) for p in paths.values()}}
    protected[str(Path(args.config).resolve())]=sha(args.config)
    protected.update({str(path):sha(path) for path in previous_paths.values()})
    core=load_original_core(config);views=core.coverage_original_views
    model,bundle=load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False)
    restorer=get_restorer(config,'cuda').eval().requires_grad_(False)
    frozen=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items()))
    if frozen!=(r1['model_state_sha256'],receipt['restorer_state_sha256']) or bundle['epoch']!=20:
        raise RuntimeError('Inherited R1 e20/Stage1/joint-v3 states differ')
    freeze_contract(model,restorer)
    parameters=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    if parameters>=500_000_000:raise RuntimeError('Parameter budget exceeded')
    output.mkdir(exist_ok=resumed);args.owned_output=output;started=time.time()
    if resumed:
        old_capture = read(output/'capture.json')
        code,virtual = old_capture['protected_code'],old_capture['virtual_code_paths']
        verify_hashes(code,code=True);verify_hashes(old_capture['protected_inputs'])
    else:code,virtual=source_snapshot(output)
    control_state = torch.load(previous_paths['previous_head'],map_location='cpu',weights_only=False)
    control_head = deepcopy(model.affinity_decoder).cpu().eval().requires_grad_(False)
    control_head.load_state_dict(control_state['geometry_state_dict'],strict=True)
    if (control_state.get('format') != 'affinity_interior_head_v1' or control_state.get('arm') != 'connect'
            or control_state.get('step') != 32 or control_state.get('schedule') != peer['schedule']
            or tensor_digest(control_head.state_dict().items()) != peer['arms']['connect']['final_head_sha256']):
        raise RuntimeError('Sealed old connect strict head/schedule identity differs')
    del control_head,control_state
    for name,row in peer['arms']['connect']['snapshots']['32']['sources'].items():
        folder=previous_paths['previous_fit'].parent/'connect'/'s32'/name
        for file,key in (('prediction_inst.png','png_sha256'),('prediction_class.json','classes_sha256')):
            path=folder/file
            if sha(path) != row['output'][key]:raise RuntimeError('Sealed old connect final outputs differ')
            protected[str(path)] = sha(path)
    report=dict(format='affinity_retention_guard_v1',complete=False,training=True,no_submission=True,GT_enrichment='postponed',
        config=merged,optimizer_steps_per_arm=32,microbatches_per_update=3,snapshots=[0,8,16,32],
        unique_variable='add processed-known reliable negative KL with independent normalization; preserve every old connect term',
        paired_control=paired_peer,control_report_sha256=variant['previous_fit_sha256'],
        control_checkpoint_sha256=variant['previous_head_sha256'],retention_objective='old_plus_added_kl',
        loss_contract='retention_guard_loss_v1',added_retention_weight=1.,
        added_normalization='Same KL expression; negative-only class terms are averaged over active channels separately from the old positive/negative class means. Coefficient is 1 for the added term, not equal per-edge weight to old term.',
        normalization_caveat='Old retention channel/class KL normalization remains exact. New high-confidence negative edges use independent KL term with fixed coefficient 1. Expanded means are diagnostics only.',
        fixed_weights=dict(full=1.,outer=.5,retention=1.),parameter_count=parameters,
        initial_checkpoint_sha256=R1_SHA,restorer_checkpoint_sha256=RESTORER_SHA,
        initial_head_sha256=tensor_digest(model.affinity_decoder.state_dict().items()),
        frozen_model_state_sha256=frozen[0],frozen_restorer_state_sha256=frozen[1],
        CPU_runtime=cpu_runtime(),runtime=runtime,precision=precision,
        protected_inputs=protected,protected_code=code,virtual_code_paths=virtual,
        private_plan_sha256=opts['plan_sha256'],training_case_scope='Three seen sources, fixed deployment windows, no random augmentation or new pseudo labels',
        optimizer=dict(name='AdamW',lr=2e-5,eps=opts['adam_eps'],weight_decay=config['backend_adaptation']['weight_decay'],
            grad_clip=config['backend_adaptation']['grad_clip'],schedule='constant 32-update diagnostic'),
        caveat='GT-relative mechanism learning, not independent validation; censored objects and physical annotation ambiguity remain. No automatic long training or best-deployment replacement.',
        captures={},arms={'old_connect':deepcopy(peer['arms']['connect'])})
    for key in ('initial_checkpoint_sha256','restorer_checkpoint_sha256','initial_head_sha256',
                'frozen_model_state_sha256','frozen_restorer_state_sha256','CPU_runtime','runtime','precision','optimizer'):
        if report[key] != peer[key]:raise RuntimeError('Actual paired upstream/runtime/optimizer differs: '+key)
    write(output/'status.json',dict(status='running',stage='seven_source_capture',training=True))
    write(output/'preflight.json',report)
    caches={}; related={}; loss_kwargs=loss_options(config)
    # 全七源真实捕获及s0复现；头学习后全部窗重算，不把旧PNG作为新头回归。
    for row in structure['sources']:
        source=row['source'];name=source_name(source)
        source_record=next(s for s in stage['sources'] if source_name(s['source'])==name)
        verify_hashes(source_record['input_hashes']);protected.update(source_record['input_hashes'])
        image,gtpath=ROOT/source['source_path'],ROOT/source['gt_path']
        rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
        gt,lookup=load_completed_semantic_source(gtpath,rgb.shape[:2])
        classes={int(g):int(lookup[g]) for g in np.unique(gt) if g>0}
        with np.load(gtpath,allow_pickle=False) as z:
            original=z['original_covered'].astype(bool)
            if not np.array_equal(z['filled'],(gt>0)&~original) or not np.array_equal(z['residual_unknown'],gt==0):
                raise RuntimeError('Completed newGT provenance differs')
        for case in source['cases']:validate_case(case,gt)
        _,capture,tiles,audit=capture_frozen_features(views,model,restorer,image,config,gt,'cuda',source['seed'])
        old=ROOT/source['reference_dir'];prefix=source.get('reference_prefix','r1')
        with np.load(old/(prefix+'_maps.npz'),allow_pickle=False) as z:
            original_maps={k:np.squeeze(z[k]).copy() for k in ('boundary','markers','watershed')}
            if not np.array_equal(np.squeeze(z['semantic']),np.squeeze(capture['semantic'].numpy())) or not np.array_equal(z['raw_probability'],capture['raw_probability']):
                raise RuntimeError('Frozen semantic maps differ')
        baseline,_,actual_replay=replay_candidates(capture,config)
        compare_actual(baseline,capture['raw_probability'],original_maps,old,prefix,capture['baseline']['boundary'])
        reference,_,_=decode_reflect(capture['semantic'],capture['baseline']['boundary'],config)
        cached=load_prediction(ROOT/'outputs/marker_stages'/name,'reflect')
        if any(not np.array_equal(cached[k],value) for k,value in (
                ('instances',reference['instances']),('markers',reference['actual_markers']),
                ('watershed',reference['watershed']),('boundary',capture['baseline']['boundary']))):
            raise RuntimeError('Fresh complete reflect stage/cache differs: '+name)
        metrics,votes=measure(reference,capture['raw_probability'],gt,classes)
        if votes!=cached['classes']:raise RuntimeError('Fresh reflect classes differ')
        if json.loads(json.dumps(metrics,allow_nan=False))!=row['reference_metrics']:
            raise RuntimeError('Fresh reflect reference metrics differ')
        folder=output/'reference'/name;folder.mkdir(parents=True,exist_ok=resumed)
        hashes=save_prediction(folder,'reference',reference['instances'],votes)
        info=dict(source=source,image=image,rgb=rgb,gt=gt,original=original,classes=classes,capture=capture,
            tiles=tiles,metrics=metrics,reference_hashes=hashes,votes=votes,masks={},old_masks={},expanded_masks={})
        if name in training:
            gid=training[name]['gt_id']
            if classes[gid]!=1:raise RuntimeError('Predeclared target is not recorded ferrite')
            selection=[];supervision=[]
            for tile in tiles:
                masks,ma=build_interior_supervision(tile,gt,original,tile['logits'],gid=gid,
                    margin=opts['interior_margin'],low=opts['activity_threshold'],confidence=opts['retention_confidence'])
                if bool(masks['interior'].any()) or bool(masks['outer'].any()):
                    expanded, parts, expansion_audit = expand_supervision(tile,gt,original,tile['logits'],masks,
                        confidence=opts['retention_confidence'])
                    info['old_masks'][tile['index']]=masks;info['masks'][tile['index']]=expanded
                    info['expanded_masks'][tile['index']]=parts;selection.append(tile['index'])
                    old_pin=next(p for p in peer['captures'][name]['supervision'] if p['index']==tile['index'])
                    if old_pin != dict(index=tile['index'],box=tile['box'],seed=tile['seed'],audit=ma):
                        raise RuntimeError('Original paired supervision/audit differs before expansion')
                    supervision.append(dict(index=tile['index'],box=tile['box'],seed=tile['seed'],
                        audit=ma,expansion=expansion_audit))
            if not selection or not any(bool(m['local'].any()) for m in info['masks'].values()):
                raise RuntimeError('Training source has no complete interior/local response support')
            related[name]=selection
            if selection != [p['index'] for p in peer['captures'][name]['supervision']]:
                raise RuntimeError('Original training-window selection differs')
            report['captures'][name]=dict(actual=audit,original_replay=actual_replay,supervision=supervision,recorded_GT=gid,
                original_boundary_sha256=array_sha(capture['baseline']['boundary']))
        else:report['captures'][name]=dict(actual=audit,original_replay=actual_replay,training=False)
        caches[name]=info
        print('FIT_CAPTURE',name,'tiles',len(tiles),'training',name in training,flush=True)
        del reference,baseline,cached
        gc.collect()
    schedule=training_schedule(related);validate_peer(peer,opts,schedule)
    for name,info in caches.items():
        peer_tiles=peer['arms']['connect']['snapshots']['0']['sources'][name]['blend']['tiles']
        if len(peer_tiles)!=len(info['tiles']):raise RuntimeError('Paired actual feature tile count differs')
        for tile,old in zip(info['tiles'],peer_tiles):
            if tile['index']!=old['index'] or list(tile['box'])!=old['box'] or tile['features_sha256']!=old['features_sha256']:
                raise RuntimeError('Paired actual frozen encoder features differ')
    report['schedule']=schedule
    report['exposure']={name:dict(microbatches=32,tiles=dict(Counter(m['tile'] for batch in schedule for m in batch if m['source']==name))) for name in related}
    if resumed and canonical_sha(report['captures']) != canonical_sha(old_capture['captures']):
        raise RuntimeError('Resumed actual complete captures/masks differ')
    if not resumed:write(output/'capture.json',report)
    initial={k:v.detach().cpu().clone() for k,v in model.affinity_decoder.state_dict().items()}

    def evaluate(head,arm,step,all_sources=False):
        outcomes={}; names=list(caches) if all_sources else list(related)
        for name in names:
            info=caches[name];folder=output/arm/f's{step:02d}'/name;folder.mkdir(parents=True,exist_ok=resumed)
            boundary,blend=replay_head_boundary(views,info['tiles'],head,config,info['gt'].shape,empty_exact=step==0)
            decoded,_,_=decode_reflect(info['capture']['semantic'],boundary,config)
            metrics,votes=measure(decoded,info['capture']['raw_probability'],info['gt'],info['classes'],info['metrics'])
            if step==0 and (array_sha(boundary)!=array_sha(info['capture']['baseline']['boundary'])
                    or json.loads(json.dumps(metrics,allow_nan=False))!=info['metrics']):
                raise RuntimeError('Copied head s0 complete output/metrics differ')
            hashes=save_prediction(folder,'prediction',decoded['instances'],votes)
            if step==0 and hashes!=info['reference_hashes']:
                raise RuntimeError('Copied head s0 final PNG/classes bytes differ')
            cases=[target_result(decoded,info['gt'],info['original'],c,metrics,info['classes'],votes) for c in info['source']['cases']]
            summaries=[case_summary(c) for c in cases]
            safety=safety_metrics(info['metrics'],metrics)
            outcomes[name]=dict(all_source_windows_recomputed=len(info['tiles']),blend=blend,
                metrics=metrics,safety=safety,target_outcomes=summaries,output=hashes,
                native_boundary_sha256=array_sha(boundary),stage0_exact=step==0)
            if name in training:
                save_thumbnail(folder/'monitor.png',info['rgb'],boundary,decoded['instances'],info['gt'],
                    [training[name]['gt_id']],arm+' s'+str(step)+' | raw / boundary / instances')
            write(folder/'report.json',outcomes[name]);print('FIT_EVAL',arm,step,name,'safe',safety['no_new_failures'],flush=True)
            del decoded,metrics;gc.collect()
        return dict(step=step,sources=outcomes,global_safety=aggregate_safety(outcomes) if all_sources else None,
            scope='All seven sources full windows' if all_sources else 'Three fixed training source process monitors')

    # 同源同状态、相同eval/dropout/BN模式；无随机增强，不以观感选择步数。
    random.seed(opts['seed']);np.random.seed(opts['seed']);torch.manual_seed(opts['seed']);torch.cuda.manual_seed_all(opts['seed'])
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    for arm,local_weight in (('guard',opts['local_weight']),):
        head=deepcopy(model.affinity_decoder).eval().requires_grad_(True)
        if tensor_digest(head.state_dict().items())!=report['initial_head_sha256']:raise RuntimeError('Common copied initialization differs')
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        optimizer=torch.optim.AdamW(head.parameters(),lr=opts['learning_rate'],eps=opts['adam_eps'],weight_decay=config['backend_adaptation']['weight_decay'])
        optimizer_options=dict(lr=opts['learning_rate'],eps=opts['adam_eps'],weight_decay=config['backend_adaptation']['weight_decay'])
        identity=dict(config_sha256=sha(args.config),code_sha256=canonical_sha(code),
            protected_inputs_sha256=canonical_sha(protected),peer_sha256=variant['previous_fit_sha256'],
            peer_head_sha256=variant['previous_head_sha256'],initial_checkpoint_sha256=R1_SHA,
            initial_head_sha256=report['initial_head_sha256'],restorer_sha256=RESTORER_SHA,
            retention_objective='old_plus_added_kl')
        gate=[]
        if not resumed:
            for name in related:
                index=next((i for i in related[name] if bool(caches[name]['expanded_masks'][i]['added'].any())),None)
                if index is None:raise RuntimeError('Training source has no actual eligible new high-confidence negatives')
                info=caches[name]
                gate.append(dict(source=name,tile=index,details=actual_added_gradient_gate(
                    head,info['tiles'][index],info['old_masks'][index],info['masks'][index],loss_kwargs,optimizer_options)))
            report['actual_gradient_gate']=gate
            if tensor_digest(head.state_dict().items())!=report['initial_head_sha256']:raise RuntimeError('Gate altered formal initialization')
            if args.preflight_only:
                verify_hashes(protected);verify_hashes(code,code=True)
                if frozen!=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items())):
                    raise RuntimeError('Preflight gradient gate altered frozen upstream')
                report.update(complete=True,training=False,actual_gradient_gate_passed=True,
                    formal_optimizer_updates=0,protected_inputs_exact=True,protected_code_exact=True,
                    elapsed_seconds=time.time()-started)
                write(output/'report.json',report)
                write(output/'status.json',dict(status='complete',stage='preflight_only',training=False,formal_steps=0))
                print('RETENTION_GUARD_PREFLIGHT_COMPLETE',flush=True);return
            snapshots={'0':evaluate(head,arm,0,all_sources=True)};steps=[];start_step=0
            if snapshots['0']['global_safety'] != peer['arms']['connect']['snapshots']['0']['global_safety']:
                raise RuntimeError('Actual formal s0 fixed R1 cohort differs')
        else:
            start_step=validate_resume_state(resume_state,identity,report)
            head.load_state_dict(resume_state['geometry_state_dict'],strict=True);optimizer.load_state_dict(resume_state['optimizer'])
            restore_rng(resume_state['rng']);snapshots=resume_state['snapshots'];steps=resume_state['steps']
            report['actual_gradient_gate']=resume_state['actual_gradient_gate']
            report['resumed_from_step']=start_step
        def commit_state(step, snapshot=False):
            state=dict(format='affinity_retention_guard_head_v1',arm=arm,step=step,geometry_state_dict=head.state_dict(),
                optimizer=optimizer.state_dict(),config=merged,identity=identity,
                initial_head_sha256=report['initial_head_sha256'],schedule=schedule,rng=capture_rng(),
                capture_sha256=canonical_sha(report['captures']),steps=steps,snapshots=snapshots,
                runtime=runtime,precision=precision,CPU_runtime=report['CPU_runtime'],local_weight=local_weight,
                actual_gradient_gate=report['actual_gradient_gate'])
            temporary=output/arm/'last_state.tmp.pt';torch.save(state,temporary);temporary.replace(output/arm/'last_state.pt')
            if snapshot:torch.save(state,output/arm/f'head_s{step:02d}.pt')
        if resumed:
            with (output/arm/'steps.jsonl').open('w',encoding='utf8') as stream:
                for row in steps:stream.write(json.dumps(row,allow_nan=False)+'\n')
            if start_step in opts['snapshots'] and str(start_step) not in snapshots:
                snapshots[str(start_step)]=evaluate(head,arm,start_step,all_sources=start_step in (0,32))
            commit_state(start_step,snapshot=start_step in opts['snapshots'])
        else:commit_state(0,snapshot=True)
        for step,batch in enumerate(schedule,1):
            if step<=start_step:continue
            optimizer.zero_grad(set_to_none=True);micro=[]
            for chosen in batch:
                info=caches[chosen['source']];tile=info['tiles'][chosen['tile']];masks=info['masks'][chosen['tile']]
                logits=head(tile['features'])['affinity_logits'].float();logits.retain_grad()
                loss,terms,details=retention_guard_objective(logits,tile['logits'],masks,local_weight=local_weight,loss_options=loss_kwargs)
                gradient=retention_gradient_summary(terms,head,local_weight) if step in (1,16,32) else None
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real head objective')
                logits.grad=None;(loss/3).backward()
                if torch.count_nonzero(logits.grad[~masks['full_legal']]):raise RuntimeError('Gradient entered unknown/padding supervision')
                micro.append(dict(**chosen,loss=float(loss.detach()),component_losses={k:float(v.detach()) for k,v in terms.items()},
                    component_gradients=gradient,details=scalar_record(details),outside_legal_gradient_max=0.,
                    feature_sha256=tile['features_sha256'],
                    added_retention_edges=int(info['expanded_masks'][chosen['tile']]['added'].sum()),
                    old_retention_edges=int(info['expanded_masks'][chosen['tile']]['old'].sum()),
                    expanded_retention_edges=int(masks['retention_expanded'].sum())))
                del logits,loss,terms;gc.collect()
            norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),config['backend_adaptation']['grad_clip'],error_if_nonfinite=True))
            if norm<=0:raise RuntimeError('No real trainable head gradient')
            freeze_contract(model,restorer);optimizer.step()
            row=dict(step=step,microbatches=micro,gradient_norm_before_clip=norm,local_weight=local_weight,
                parameter_change=parameter_change(head,initial))
            if row['parameter_change']['l2']<=0:raise RuntimeError('Head parameters failed to update')
            steps.append(row)
            with (output/arm/'steps.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            write(output/'status.json',dict(status='running',stage='training',arm=arm,step=step,total=32))
            print('FIT_STEP',arm,step,'loss',sum(m['loss'] for m in micro)/3,flush=True)
            commit_state(step)
            if step in opts['snapshots']:
                snapshots[str(step)]=evaluate(head,arm,step,all_sources=step==32)
                commit_state(step,snapshot=True)
        saved=torch.load(output/arm/'head_s32.pt',map_location='cuda',weights_only=False)
        reloaded=deepcopy(model.affinity_decoder).eval().requires_grad_(False)
        reloaded.load_state_dict(saved['geometry_state_dict'],strict=True)
        if tensor_digest(reloaded.state_dict().items())!=tensor_digest(head.state_dict().items()):raise RuntimeError('Strict head reload differs')
        reload_results={}
        for name,info in caches.items():
            boundary,_=replay_head_boundary(views,info['tiles'],reloaded,config,info['gt'].shape)
            old=snapshots['32']['sources'][name]
            if array_sha(boundary)!=old['native_boundary_sha256']:raise RuntimeError('Reloaded actual full-source boundary differs')
            decoded,_,_=decode_reflect(info['capture']['semantic'],boundary,config)
            _,votes=measure(decoded,info['capture']['raw_probability'],info['gt'],info['classes'])
            temp=output/arm/'reload'/name;temp.mkdir(parents=True,exist_ok=resumed)
            hashes=save_prediction(temp,'prediction',decoded['instances'],votes)
            if hashes!=old['output']:raise RuntimeError('Reloaded full native PNG/classes differ')
            reload_results[name]=dict(boundary_exact=True,PNG_and_classes_exact=True,all_windows_recomputed=len(info['tiles']))
            del decoded;gc.collect()
        if frozen!=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items())):raise RuntimeError('Frozen inherited upstream changed')
        freeze_contract(model,restorer)
        for info in caches.values():
            for tile in info['tiles']:
                if validate_frozen_tile_features(tile,model.affinity_decoder)!=tile['features_sha256']:
                    raise RuntimeError('Frozen actual feature cache mutated')
        report['arms'][arm]=dict(steps=32,microbatches=96,local_weight=local_weight,snapshots=snapshots,
            final_head_sha256=tensor_digest(head.state_dict().items()),strict_reload_equal=True,
            actual_complete_reload=reload_results,non_head_frozen_exact=True,feature_cache_exact=True)
        write(output/'report.json',report)
        del head,reloaded,saved,optimizer;gc.collect();torch.cuda.empty_cache()
    verify_hashes(protected);verify_hashes(code,code=True)
    if precision_record()!=precision or runtime_contract(config)!=runtime:raise RuntimeError('Actual runtime changed')
    report.update(complete=True,protected_inputs_exact=True,protected_code_exact=True,elapsed_seconds=time.time()-started)
    write(output/'report.json',report);write(output/'status.json',dict(status='complete',training=False,steps_per_arm=32))
    print('RETENTION_GUARD_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/affinity_retention_guard.yaml')
    parser.add_argument('--go32',action='store_true')
    parser.add_argument('--preflight-only',action='store_true',help='独立真实梯度门禁，不进行32次正式更新')
    parser.add_argument('--resume',action='store_true',help='严格恢复本变体最后一个原子提交更新')
    args=parser.parse_args()
    try:run(args)
    except FileExistsError:raise
    except BaseException as error:
        folder=getattr(args,'owned_output',None)
        if folder is not None and folder.exists() and folder.is_relative_to(ROOT/'outputs'):
            write(folder/'status.json',dict(status='failed',training=False,error=repr(error)))
        raise
