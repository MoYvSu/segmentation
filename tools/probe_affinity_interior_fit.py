# -*- coding: utf-8 -*-
"""继承R1、冻结上游的三源32步内部连通对照；不修改GT或正式部署。"""
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


def training_schedule(source_tiles, steps=32):
    """每个更新三源各一次，均匀循环源内全部相关部署窗。"""
    if len(source_tiles) != 3 or len(set(source_tiles)) != 3 or steps != 32:
        raise ValueError('Exactly three sources and 32 paired updates required')
    if any(not indices or len(indices) != len(set(indices)) for indices in source_tiles.values()):
        raise ValueError('Each source needs unique nonempty relevant tile indices')
    return [[dict(source=name, tile=indices[(step-1) % len(indices)])
             for name, indices in source_tiles.items()] for step in range(1, steps+1)]


def validate_plan(plan, structure, stage):
    if (plan.get('format') != 'affinity_interior_plan_v1'
            or plan.get('GT_enrichment') != 'postponed'
            or plan.get('test_training') is not False):
        raise ValueError('Private three-source plan must postpone GT changes and test training')
    all_sources = {source_name(s['source']):s for s in structure['sources']}
    stage_sources = {source_name(s['source']):s for s in stage['sources']}
    if len(all_sources) != 7 or set(stage_sources) != set(all_sources):
        raise ValueError('Actual seven-source structure/stage identity required')
    choices = plan['training_cases']
    if len(choices) != 3 or len({c['source'] for c in choices}) != 3:
        raise ValueError('Three distinct training sources required')
    excluded = {(c['source'], c['case_id']) for c in plan['excluded_training_cases']}
    for entry in choices:
        if (entry['source'], entry['case_id']) in excluded:
            raise ValueError('A pre-excluded ambiguity/failure entered training')
        row = all_sources[entry['source']]
        cases = row['selected']['interior']['cases']
        case = next((c for c in cases if c['case_id'] == entry['case_id']), None)
        if case is None or case['direction'] != 'positive' or case['gt_ids'] != [entry['gt_id']]:
            raise ValueError('Training case is not the predeclared same-GT interior target')
        if case.get('previous_gt_structure_ambiguity') or row['source'].get('reference_receipt'):
            raise ValueError('Previously registered GT ambiguity cannot be reinforced')
        if row['source'] != stage_sources[entry['source']]['source']:
            raise ValueError('Stage source/seed differs')
    return {c['source']:c for c in choices}


def gradient_summary(terms, head, local_weight):
    """只诊断真实参数梯度；不把初态零KL当成有效保护。"""
    parameters = tuple(head.parameters())
    grads = {}
    for name, loss in terms.items():
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        grads[name] = [torch.zeros_like(p) if g is None else g.detach()
                       for p,g in zip(parameters,values)]
    norms = {name:sum(float(g.double().square().sum()) for g in values)**.5
             for name,values in grads.items()}
    common = [a + .5*b + c for a,b,c in zip(grads['full'],grads['outer'],grads['retention'])]
    common_norm = sum(float(g.double().square().sum()) for g in common)**.5
    dot = sum(float((a.double()*b.double()).sum()) for a,b in zip(common,grads['local']))
    return dict(unweighted_parameter_gradient_norms=norms, common_parameter_gradient_norm=common_norm,
        weighted_local_to_common=local_weight*norms['local']/common_norm if common_norm else None,
        local_common_cosine=dot/(common_norm*norms['local']) if common_norm and norms['local'] else None)


def aggregate_safety(rows):
    """固定七源689/135名单；失配与相别翻转分开，面积资格失败不删行。"""
    matches, areas, delta, before_area, after_area = 0,0,0.,0.,0.
    missing, changed_classes, failures, added_errors = [],[],[],[]
    for name,row in rows.items():
        s=row['safety']; m=s['fixed_matching']; a=s['same_fixed_area_MAE']
        matches += m['fixed_count']; areas += a['denominator']
        delta += sum(r['candidate_iou_missing_as_zero']-r['reference_iou'] for r in m['rows'])
        before_area += sum(abs(r['r1_relative_error']) for r in s['fixed_area_rows']['rows'])
        after_area += sum(abs(r['candidate_relative_error']) for r in s['fixed_area_rows']['rows'] if r['candidate_comparable'])
        missing.extend(name+'/'+str(g) for g in m['missing_gt_ids'])
        changed_classes.extend(name+'/'+str(g) for g in s['newly_wrong_F_class_gt_ids'])
        failures.extend(name+'/'+str(g) for g in s['fixed_area_status']['incomparable_gt_ids'])
        added_errors.extend(dict(source=name,category=k,gt=g)
            for k,v in s['error_changes'].items() for g in v['added'])
    if (matches,areas)!=(689,135):
        raise RuntimeError('Final actual fixed reflect 689/135 cohort differs')
    return dict(fixed_F_denominator=matches,fixed_F_mean_iou_delta=delta/matches,
        fixed_F_missing=missing,newly_matched_wrong_F_class=changed_classes,
        fixed_area_denominator=areas,reference_area_MAE=before_area/areas,
        candidate_area_MAE=after_area/areas if not failures else None,
        fixed_area_failures=failures,added_errors=added_errors,
        no_new_failures=not(missing or changed_classes or failures or added_errors),
        no_average_regression=delta>=-1e-12 and not failures and after_area<=before_area+1e-12,
        training_extension_permission=False,
        caveat='Seen-source diagnostics; categories overlap. Fixed-area subset does not cover new objects.')


def freeze_contract(model, restorer):
    if (any(p.requires_grad or p.grad is not None for p in model.parameters())
            or any(p.requires_grad or p.grad is not None for p in restorer.parameters())
            or any(m.training for m in list(model.modules())+list(restorer.modules()))):
        raise RuntimeError('Original model and restorer must remain fully frozen in eval mode')


def run(args):
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from tools.run_affinity_sweep import runtime_contract

    merged=load_config(args.config); opts=merged['affinity_interior_fit']
    config=deepcopy(merged);config.pop('affinity_interior_fit')
    if (opts['format']!='affinity_interior_fit_v1' or opts['steps']!=32 or opts['accumulation']!=3
            or opts['snapshots']!=[0,8,16,32] or opts['learning_rate']!=2e-5
            or (opts['full_weight'],opts['outer_weight'],opts['retention_weight'],opts['local_weight'])!=(1.,.5,1.,.25)):
        raise ValueError('Registered short-test budget and single-variable recipe required')
    output=Path(project_path(config,opts['output_dir'])).resolve()
    if output.exists():raise FileExistsError('Fresh ignored output required')
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Ignored output required')
    if not args.go32:
        print('Validated recipe; --go32 explicitly starts this finite 32-update diagnostic.');return
    if not torch.cuda.is_available():raise RuntimeError('Actual sam2_env CUDA required')
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
    core=load_original_core(config);views=core.coverage_original_views
    model,bundle=load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False)
    restorer=get_restorer(config,'cuda').eval().requires_grad_(False)
    frozen=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items()))
    if frozen!=(r1['model_state_sha256'],receipt['restorer_state_sha256']) or bundle['epoch']!=20:
        raise RuntimeError('Inherited R1 e20/Stage1/joint-v3 states differ')
    freeze_contract(model,restorer)
    parameters=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    if parameters>=500_000_000:raise RuntimeError('Parameter budget exceeded')
    output.mkdir();args.owned_output=output;started=time.time()
    code,virtual=source_snapshot(output)
    report=dict(format='affinity_interior_fit_v1',complete=False,training=True,no_submission=True,GT_enrichment='postponed',
        config=merged,optimizer_steps_per_arm=32,microbatches_per_update=3,snapshots=[0,8,16,32],
        unique_variable='local positive interior weight control=0 versus connect=.25',
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
        captures={},arms={})
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
        folder=output/'reference'/name;folder.mkdir(parents=True)
        hashes=save_prediction(folder,'reference',reference['instances'],votes)
        info=dict(source=source,image=image,rgb=rgb,gt=gt,original=original,classes=classes,capture=capture,
            tiles=tiles,metrics=metrics,reference_hashes=hashes,votes=votes,masks={})
        if name in training:
            gid=training[name]['gt_id']
            if classes[gid]!=1:raise RuntimeError('Predeclared target is not recorded ferrite')
            selection=[];supervision=[]
            for tile in tiles:
                masks,ma=build_interior_supervision(tile,gt,original,tile['logits'],gid=gid,
                    margin=opts['interior_margin'],low=opts['activity_threshold'],confidence=opts['retention_confidence'])
                if bool(masks['interior'].any()) or bool(masks['outer'].any()):
                    info['masks'][tile['index']]=masks;selection.append(tile['index'])
                    supervision.append(dict(index=tile['index'],box=tile['box'],seed=tile['seed'],audit=ma))
            if not selection or not any(bool(m['local'].any()) for m in info['masks'].values()):
                raise RuntimeError('Training source has no complete interior/local response support')
            related[name]=selection
            report['captures'][name]=dict(actual=audit,original_replay=actual_replay,supervision=supervision,recorded_GT=gid,
                original_boundary_sha256=array_sha(capture['baseline']['boundary']))
        else:report['captures'][name]=dict(actual=audit,original_replay=actual_replay,training=False)
        caches[name]=info
        print('FIT_CAPTURE',name,'tiles',len(tiles),'training',name in training,flush=True)
        del reference,baseline,cached
        gc.collect()
    schedule=training_schedule(related)
    report['schedule']=schedule
    report['exposure']={name:dict(microbatches=32,tiles=dict(Counter(m['tile'] for batch in schedule for m in batch if m['source']==name))) for name in related}
    write(output/'capture.json',report)
    initial={k:v.detach().cpu().clone() for k,v in model.affinity_decoder.state_dict().items()}

    def evaluate(head,arm,step,all_sources=False):
        outcomes={}; names=list(caches) if all_sources else list(related)
        for name in names:
            info=caches[name];folder=output/arm/f's{step:02d}'/name;folder.mkdir(parents=True)
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
    for arm,local_weight in (('control',0.),('connect',opts['local_weight'])):
        head=deepcopy(model.affinity_decoder).eval().requires_grad_(True)
        if tensor_digest(head.state_dict().items())!=report['initial_head_sha256']:raise RuntimeError('Common copied initialization differs')
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        optimizer=torch.optim.AdamW(head.parameters(),lr=opts['learning_rate'],eps=opts['adam_eps'],weight_decay=config['backend_adaptation']['weight_decay'])
        snapshots={'0':evaluate(head,arm,0,all_sources=True)};steps=[]
        for step,batch in enumerate(schedule,1):
            optimizer.zero_grad(set_to_none=True);micro=[]
            for chosen in batch:
                info=caches[chosen['source']];tile=info['tiles'][chosen['tile']];masks=info['masks'][chosen['tile']]
                logits=head(tile['features'])['affinity_logits'].float();logits.retain_grad()
                loss,terms,details=interior_objective(logits,tile['logits'],masks,local_weight=local_weight,loss_options=loss_kwargs)
                gradient=gradient_summary(terms,head,local_weight) if step in (1,16,32) else None
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite real head objective')
                logits.grad=None;(loss/3).backward()
                if torch.count_nonzero(logits.grad[~masks['full_legal']]):raise RuntimeError('Gradient entered unknown/padding supervision')
                micro.append(dict(**chosen,loss=float(loss.detach()),component_losses={k:float(v.detach()) for k,v in terms.items()},
                    component_gradients=gradient,details=scalar_record(details),outside_legal_gradient_max=0.,
                    feature_sha256=tile['features_sha256']))
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
            if step in opts['snapshots']:
                state=dict(format='affinity_interior_head_v1',arm=arm,step=step,geometry_state_dict=head.state_dict(),
                    optimizer=optimizer.state_dict(),config=merged,initial_head_sha256=report['initial_head_sha256'],
                    schedule=schedule,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),local_weight=local_weight)
                torch.save(state,output/arm/f'head_s{step:02d}.pt')
                snapshots[str(step)]=evaluate(head,arm,step,all_sources=step==32)
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
            temp=output/arm/'reload'/name;temp.mkdir(parents=True)
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
    print('INTERIOR_FIT_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/affinity_interior_fit.yaml')
    parser.add_argument('--go32',action='store_true')
    args=parser.parse_args()
    try:run(args)
    except FileExistsError:raise
    except BaseException as error:
        folder=getattr(args,'owned_output',None)
        if folder is not None and folder.exists() and folder.is_relative_to(ROOT/'outputs'):
            write(folder/'status.json',dict(status='failed',training=False,error=repr(error)))
        raise
