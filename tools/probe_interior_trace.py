# -*- coding: utf-8 -*-
"""冻结已有头，定位内部残留分隔和邻粒界面失守；不训练、不改GT。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.semantic_targets import load_completed_semantic_source
from tools.affinity_feature_capture import capture_frozen_features, replay_head_boundary, validate_frozen_tile_features
from tools.analyze_affinity_bottleneck import source_name
from tools.analyze_affinity_coverage import fixed_area_cohort
from tools.analyze_affinity_structure import case_summary
from tools.analyze_marker_stages import verify_hashes
from tools.probe_affinity_bottleneck import measure, validate_case, error_sets
from tools.probe_affinity_interior import R1_SHA, RESTORER_SHA, cpu_runtime, positive_relation_mask, recompute_positive
from tools.probe_affinity_interior_fit import freeze_contract
from tools.probe_affinity_structure import decode_reflect, safety_metrics, target_result
from tools.probe_marker_stages import compare_actual, connectivity_for_variant
from tools.run_marker_reflect import array_sha, config_gate, precision_record, read, replay_candidates, save_prediction, sha, source_snapshot, write
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import fusion_options, tensor_digest
from utils.affinity_channel_trace import selected_channel_stats, trace_fusion
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_interior_loss import build_interior_supervision
from utils.affinity_spatial_trace import native_grid_footprint, edge_roi_mask
from utils.affinity_structure import interior_mask, interface_mask, mask_provenance
from utils.config import load_config, project_path
from utils.marker_stage_trace import trace_case_stages, classify_stage_failure
from utils.patch_diagnostic import BlendMap


def restore_interface(before, reference, mask, gt, original):
    """只恢复预定已知域内的连续响应；拒绝unknown，并核对mask外精确不变。"""
    if (before.dtype != np.float32 or reference.dtype != np.float32
            or before.shape != reference.shape or before.shape != gt.shape
            or mask.shape != gt.shape or mask.dtype != np.bool_
            or not np.isfinite(before).all() or not np.isfinite(reference).all()
            or np.any((before < 0) | (before > 1)) or np.any((reference < 0) | (reference > 1))):
        raise ValueError('Matching finite native FP32 boundaries and bool mask required')
    provenance = mask_provenance(gt, original, mask)
    result = before.copy(); result[mask] = reference[mask]
    if not np.array_equal(result[~mask], before[~mask]):
        raise RuntimeError('Restoration leaked outside predeclared interface mask')
    return result, dict(provenance=provenance, changed_pixels=int(np.count_nonzero(result != before)),
        outside_mask_exact=True, unknown_unchanged=True, scalar_response_restore_only=True)


def scalar_stats(value, mask):
    selected = np.asarray(value)[mask]
    if not len(selected): return dict(count=0, mean=None, quantiles=None)
    return dict(count=int(len(selected)), mean=float(selected.mean()),
        quantiles={str(q):float(np.quantile(selected, q)) for q in [0,.01,.05,.1,.5,.9,.95,.99,1]},
        above_low=int((selected >= .45).sum()), above_high=int((selected >= .65).sum()))


@torch.no_grad()
def rebased_tiles(views, tiles, logits_list, config):
    """为connect反事实建立完整基线，空编辑窗也不能退回原R1响应。"""
    mode,options=fusion_options(config);result=[]
    for tile,logits in zip(tiles,logits_list):
        grid=affinity_boundary_probability(logits,mode=mode,**options)
        y,x,y1,x1=tile['box']
        native=views.crop_letterbox_output(grid,1024,tile['pad_h'],tile['pad_w'],(y1-y,x1-x))
        result.append(dict(tile,logits=logits,logits_stride=list(logits.stride()),
            grid=grid.cpu().numpy(),native=native.cpu().numpy()))
    return result


def stage_record(decoded, stage, variant, gt, original, cases):
    maps = connectivity_for_variant(stage, variant, 'reflect')
    records = []
    for case in cases:
        full = trace_case_stages(maps, decoded['actual_markers'], decoded['instances'], gt, case,
                                watershed=decoded['watershed'])
        row = dict(case_id=case['case_id'], full=full, classification=classify_stage_failure(full))
        if all(original[tuple(point)] for point in case['points']):
            strict = trace_case_stages(maps, decoded['actual_markers'], decoded['instances'],
                np.where(original, gt, 0), case, watershed=decoded['watershed'])
            row.update(original_only=strict, original_classification=classify_stage_failure(strict))
        else: row['original_only'] = None
        records.append(row)
    return records


def with_baseline(metrics, baseline):
    """复用同一次完整划分诊断，仅重建固定名单和误差变化的参照。"""
    result=dict(metrics);old=baseline['full_ferrite'];diag=metrics['full_ferrite']
    rows={r['gt']:r for r in diag['gt_rows']};matched=[r['gt'] for r in old['gt_rows'] if r['matched_pred'] is not None]
    errors=error_sets(diag,metrics['full_partition']);previous=error_sets(old,baseline['full_partition'])
    result.update(original_matched_F_count=len(matched),
        original_matched_F_missing=[g for g in matched if rows[g]['matched_pred'] is None],
        original_matched_F_mean_IoU_missing_as_zero=sum((rows[g]['matched_iou'] or 0) for g in matched)/max(1,len(matched)),
        fixed_area=fixed_area_cohort(old,diag),
        error_changes={k:dict(added=sorted(errors[k]-previous[k]),removed=sorted(previous[k]-errors[k])) for k in errors})
    return result


def roi_trace(views, tiles, arm_logits, masks, gt, original, roi, cases, config):
    """逐窗反查真实双线性插值足迹；原监督合同保持不变。"""
    mode, options = fusion_options(config)
    blends = {key:BlendMap(gt.shape) for key in ('short', 'fusion_minus_short',
        'contribution_short', 'contribution2', 'contribution4')}
    rows = []
    for tile, logits in zip(tiles, arm_logits):
        parts, fusion_audit = trace_fusion(logits, mode, options)
        y,x,y1,x1 = tile['box']
        for key, blend in blends.items():
            native = views.crop_letterbox_output(parts[key], 1024, tile['pad_h'], tile['pad_w'],
                                                (y1-y,x1-x)).cpu()[0,0].numpy()
            blend.add(tile['box'], native)
        footprint, footprint_audit = native_grid_footprint(tile, roi, device=logits.device)
        record = dict(index=tile['index'], box=tile['box'], fusion=fusion_audit,
                      interpolation=footprint_audit)
        if footprint.any():
            candidate, candidate_audit = edge_roi_mask(tile, gt, original, footprint,
                cases[0]['gt_ids'], cases[0]['direction'])
            selections = {'ROI_nodes':dict(kind='node',mask=torch.from_numpy(footprint).to(logits.device)),
                'ROI_same_or_pair_candidates':dict(kind='edge',mask=torch.from_numpy(candidate).to(logits.device))}
            supervision = masks.get(tile['index'])
            coverage = {}
            if supervision is not None:
                candidate_device = torch.from_numpy(candidate).to(logits.device)
                for key in ('full_legal','native_legal','interior','local','outer','retention'):
                    selected = candidate_device & supervision[key]
                    selections['ROI_'+key] = dict(kind='edge',mask=selected)
                    coverage[key] = [int(selected[0,c].sum()) for c in range(8)]
                excluded = {'not_full_legal':candidate_device & ~supervision['full_legal'],
                    'full_but_not_native':candidate_device & supervision['full_legal'] & ~supervision['native_legal'],
                    'native_but_not_interior':candidate_device & supervision['native_legal'] & ~supervision['interior'],
                    'interior_outside_fixed_local':candidate_device & supervision['interior'] & ~supervision['local']}
                record['exclusion_counts'] = {k:[int(v[0,c].sum()) for c in range(8)] for k,v in excluded.items()}
            record.update(candidates=candidate_audit, supervised_tile=supervision is not None,
                supervision_coverage_per_channel=coverage,
                channels=selected_channel_stats(logits, selections),
                components={key:scalar_stats(parts[key].cpu()[0,0].numpy(),footprint)
                    for key in ('short','distance2','distance4','support2','support4',
                        'contribution_short','contribution2','contribution4','fused','fusion_minus_short')},
                top2_selection_ambiguous=int(parts['top2_selection_ambiguous'].cpu()[0,0].numpy()[footprint].sum()),
                top2_channel_counts=[int(((parts['top2_indices'].cpu()[0].numpy()==c)&footprint[None]).sum()) for c in range(4)])
        rows.append(record)
    maps = {key:blend.finish()[0] for key,blend in blends.items()}
    return maps, dict(tiles=rows, native_ROI_pixels=int(roi.sum()), native_ROI_sha256=array_sha(roi),
        native_components={k:scalar_stats(v,roi) for k,v in maps.items()},
        warning='Overlapping windows count relation exposures, not unique native pixels. Node ROI is interpolation influence only, not GT supervision. Long normalized terms may lower short.')


def run(args):
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from tools.run_affinity_sweep import runtime_contract
    from tools.interior_trace_gallery import save_gallery

    merged = load_config(args.config); opts = merged['affinity_interior_trace']
    config = deepcopy(merged); config.pop('affinity_interior_trace')
    if opts['format'] != 'affinity_interior_trace_v1': raise ValueError('Registered diagnostic required')
    output = Path(project_path(config,opts['output_dir'])).resolve()
    if output.exists(): raise FileExistsError('Fresh ignored output required')
    if not output.is_relative_to(ROOT/'outputs'): raise ValueError('Ignored output required')
    if not args.run: print('Validated recipe; --run starts frozen finite diagnosis.'); return
    if not torch.cuda.is_available(): raise RuntimeError('Actual sam2_env CUDA required')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    paths = {k:Path(project_path(config,opts[k])) for k in ('fit','plan','structure','receipt')}
    for k in paths:
        if sha(paths[k]) != opts[k+'_sha256']: raise RuntimeError('Pinned '+k+' differs')
    fit,plan,structure,receipt = [read(paths[k]) for k in ('fit','plan','structure','receipt')]
    if not fit['complete'] or plan['training'] or plan['test_training'] or len(plan['roles']) != 2:
        raise ValueError('Frozen two-role private plan required')
    if (plan['fit_sha256'] != opts['fit_sha256'] or plan['GT_enrichment'] != 'postponed'
            or {r['role'] for r in plan['roles']} != {'remaining_interior','lost_interface'}
            or (plan['interior_margin'],plan['interface_radius'],plan['analysis_radius'])!=(4,2,12)):
        raise ValueError('Plan provenance differs')
    verify_hashes(fit['protected_inputs']); verify_hashes(fit['protected_code'],code=True)
    r1 = receipt['arms']['r1']; config_gate(config,r1['config'])
    precision,runtime = precision_record(),runtime_contract(config)
    if (precision != fit['precision'] or runtime != fit['runtime'] or cpu_runtime() != fit['CPU_runtime']):
        raise RuntimeError('Frozen actual runtime differs')
    checkpoint = Path(r1['checkpoint'])
    restore = Path(project_path(config,config['backend_adaptation']['restoration']['checkpoint']))
    if sha(checkpoint) != R1_SHA or sha(restore) != RESTORER_SHA: raise RuntimeError('R1/D5a changed')
    protected = {**fit['protected_inputs'],**{str(p):sha(p) for p in paths.values()}}
    protected[str(Path(args.config).resolve())]=sha(args.config)
    core = load_original_core(config); views = core.coverage_original_views
    model,bundle = load_backend(checkpoint,config,'cuda');model.eval().requires_grad_(False)
    restorer = get_restorer(config,'cuda').eval().requires_grad_(False)
    frozen = (tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items()))
    if frozen != (fit['frozen_model_state_sha256'],fit['frozen_restorer_state_sha256']) or bundle['epoch'] != 20:
        raise RuntimeError('Original inherited states differ')
    parameters = sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    if parameters >= 500_000_000: raise RuntimeError('Parameter budget exceeded')
    heads={'r1':model.affinity_decoder}; head_receipts={}
    for arm in ('control','connect'):
        pinned=plan['heads'][arm];path=ROOT/pinned['path'];protected[str(path)]=pinned['sha256']
        if sha(path)!=pinned['sha256'] or path.stat().st_size!=pinned['bytes']:raise RuntimeError('Copied head file differs')
        state=torch.load(path,map_location='cuda',weights_only=False)
        if state['format']!='affinity_interior_head_v1' or state['step']!=32:raise RuntimeError('Head format/step differs')
        head=deepcopy(model.affinity_decoder).eval().requires_grad_(False)
        head.load_state_dict(state['geometry_state_dict'],strict=True)
        if tensor_digest(head.state_dict().items())!=pinned['state_sha256'] or pinned['state_sha256']!=fit['arms'][arm]['final_head_sha256']:
            raise RuntimeError('Strict copied-head state differs')
        heads[arm]=head;head_receipts[arm]=dict(**pinned,strict_load=True,step=32)
        del state
    freeze_contract(model,restorer)
    output.mkdir();args.owned_output=output;started=time.time();code,virtual=source_snapshot(output)
    report=dict(format='affinity_interior_trace_v1',complete=False,training=False,optimizer_created=False,
        no_submission=True,GT_enrichment='postponed',parameter_count_original=parameters,
        diagnostic_head_copies_only=True,config=merged,precision=precision,runtime=runtime,
        CPU_runtime=cpu_runtime(),heads=head_receipts,protected_inputs=protected,protected_code=code,
        virtual_code_paths=virtual,sources={},caveat='Seen processed-GT mechanism diagnosis; not independent accuracy, physical GT correction or deployable GT-guided inference.')
    write(output/'preflight.json',report)
    for role in plan['roles']:
        name=role['source'];write(output/'status.json',dict(status='running',source=name,training=False))
        row=next(r for r in structure['sources'] if source_name(r['source'])==name);source=row['source']
        rgb=cv2.cvtColor(cv2.imread(str(ROOT/source['source_path'])),cv2.COLOR_BGR2RGB)
        gtpath=ROOT/source['gt_path'];gt,lookup=load_completed_semantic_source(gtpath,rgb.shape[:2])
        classes={int(g):int(lookup[g]) for g in np.unique(gt) if g>0}
        with np.load(gtpath,allow_pickle=False) as z:
            original=z['original_covered'].astype(bool)
            if not np.array_equal(z['filled'],(gt>0)&~original) or not np.array_equal(z['residual_unknown'],gt==0):
                raise RuntimeError('New GT provenance differs')
        case=role['case'];cases=[case]+([role['original_positive_case']] if 'original_positive_case' in role else [])
        for c in cases:validate_case(c,gt)
        if any(classes[g]!=1 for g in case['gt_ids']):raise ValueError('Predeclared ferrite target required')
        _,capture,tiles,audit=capture_frozen_features(views,model,restorer,ROOT/source['source_path'],config,gt,'cuda',source['seed'])
        oldfolder=ROOT/source['reference_dir'];prefix=source.get('reference_prefix','r1')
        with np.load(oldfolder/(prefix+'_maps.npz'),allow_pickle=False) as z:
            oldmaps={k:np.squeeze(z[k]).copy() for k in ('boundary','markers','watershed')}
            if not np.array_equal(np.squeeze(z['semantic']),np.squeeze(capture['semantic'].numpy())) or not np.array_equal(z['raw_probability'],capture['raw_probability']):raise RuntimeError('Frozen semantic differs')
        olddecoded,_,actual=replay_candidates(capture,config)
        compare_actual(olddecoded,capture['raw_probability'],oldmaps,oldfolder,prefix,capture['baseline']['boundary'])
        del olddecoded,oldmaps
        masks={};mask_receipts=[];old_masks={m['index']:m['audit'] for m in fit['captures'][name]['supervision']}
        for tile in tiles:
            supervision,ma=build_interior_supervision(tile,gt,original,tile['logits'],gid=role['gt_id'],margin=4,low=.45,confidence=.9)
            if tile['index'] in old_masks:
                if ma!=old_masks[tile['index']]:raise RuntimeError('Original supervision masks/audit differ')
                masks[tile['index']]=supervision;mask_receipts.append(dict(index=tile['index'],all_masks_exact=True,audit=ma))
        if len(masks)!=len(old_masks):raise RuntimeError('Missing original training tile')
        folder=output/name;folder.mkdir();sreport=dict(role=role,actual_capture=audit,original_replay=actual,
            supervision_exact=mask_receipts,arms={},interventions={})
        outputs={};logits_by_arm={}
        for arm,head in heads.items():
            collected=[]
            def hook(_module,_inputs,result):collected.append(result['affinity_logits'].detach())
            handle=head.register_forward_hook(hook)
            try:boundary,blend=replay_head_boundary(views,tiles,head,config,gt.shape,empty_exact=arm=='r1')
            finally:handle.remove()
            if len(collected)!=len(tiles):raise RuntimeError('Incomplete actual head collection')
            decoded,stage,variant=decode_reflect(capture['semantic'],boundary,config)
            votes,_=revote_instances(decoded['instances'],capture['raw_probability'])
            outfolder=folder/arm;outfolder.mkdir();hashes=save_prediction(outfolder,'prediction',decoded['instances'],votes)
            oldfit=fit['arms']['control' if arm=='r1' else arm]['snapshots']['0' if arm=='r1' else '32']['sources'][name]
            if array_sha(boundary)!=oldfit['native_boundary_sha256'] or hashes!=oldfit['output']:raise RuntimeError('Complete original s0/s32 output differs '+arm)
            metrics=oldfit['metrics']
            outcome=[case_summary(target_result(decoded,gt,original,c,metrics,classes,votes)) for c in cases]
            stages=stage_record(decoded,stage,variant,gt,original,cases)
            sreport['arms'][arm]=dict(all_windows=len(tiles),boundary_sha256=array_sha(boundary),output=hashes,
                previous_output_exact=True,blend=blend,target_outcomes=outcome,stages=stages)
            outputs[arm]=dict(label=arm,boundary=boundary,decoded=decoded,stage=stage,variant=variant,metrics=metrics,votes=votes)
            logits_by_arm[arm]=collected
            print('TRACE_REPLAY',name,arm,len(tiles),'exact',flush=True)
        base=outputs['connect'];gid=role['gt_id']
        if role['role']=='remaining_interior':
            domain=interior_mask(gt,original,gid,margin=4)
            roi=domain['strict_mask'] & (base['variant']['marker_belt']>0)
            sreport['ROI']=dict(kind='connect_s32_actual_internal_marker_belt',provenance=mask_provenance(gt,original,roi),
                full_internal_belt=int((domain['mask']&(base['variant']['marker_belt']>0)).sum()),
                actual_internal_watershed_zero=int((domain['strict_mask']&(base['decoded']['watershed']==0)).sum()))
        else:
            narrow=interface_mask(gt,original,role['pair'],radius=plan['interface_radius'])
            wide=interface_mask(gt,original,role['pair'],radius=plan['analysis_radius'])
            roi=wide['mask']
            sreport['ROI']={k:v for k,v in wide.items() if not isinstance(v,np.ndarray)}
            sreport['interface_edit_support']={k:v for k,v in narrow.items() if not isinstance(v,np.ndarray)}
            sreport['old_strong_response_outside_edit_band']={arm:int(((value['boundary']>=.45)&roi&~narrow['strict_mask']).sum()) for arm,value in outputs.items()}
        if not roi.any():raise RuntimeError('Actual fixed diagnostic ROI is empty')
        for arm,value in outputs.items():
            component_maps,details=roi_trace(views,tiles,logits_by_arm[arm],masks,gt,original,roi,[case],config)
            value['components']=component_maps;sreport['arms'][arm]['channel_trace']=details
            sreport['arms'][arm]['native_boundary_ROI']=scalar_stats(value['boundary'],roi)
            normalized=sum(component_maps[k] for k in ('contribution_short','contribution2','contribution4'))
            sreport['arms'][arm]['normalized_component_stitch_roundoff_max']=float(np.abs(normalized-value['boundary']).max())
        gallery=[]
        rebase_verified=False
        def append_gallery(label,value):
            gallery.append(dict(label=label,boundary=value['boundary'],instances=value['decoded']['instances'],
                marker_belt=value['variant']['marker_belt'],short=value.get('components',{}).get('short'),
                long_shift=value.get('components',{}).get('fusion_minus_short')))
        for arm,value in outputs.items():append_gallery(arm,value)

        def evaluate(label,boundary,intervention_audit):
            nonlocal rebase_verified
            decoded,stage,variant=decode_reflect(capture['semantic'],boundary,config)
            metrics,votes=measure(decoded,capture['raw_probability'],gt,classes,outputs['r1']['metrics'])
            connect_metrics=with_baseline(metrics,base['metrics'])
            if not rebase_verified:
                direct,_=measure(decoded,capture['raw_probability'],gt,classes,base['metrics'])
                if json.loads(json.dumps(connect_metrics,allow_nan=False))!=json.loads(json.dumps(direct,allow_nan=False)):
                    raise RuntimeError('Fixed-cohort reference reuse differs from direct complete diagnostic')
                rebase_verified=True
            target=[case_summary(target_result(decoded,gt,original,c,metrics,classes,votes)) for c in cases]
            subfolder=folder/label;subfolder.mkdir()
            hashes=save_prediction(subfolder,'prediction',decoded['instances'],votes)
            delta=np.abs(boundary-base['boundary']);changed=delta>0
            summary=dict(audit=intervention_audit,boundary_sha256=array_sha(boundary),output=hashes,
                target_outcomes=target,stages=stage_record(decoded,stage,variant,gt,original,cases),
                changed_native_pixels=int(changed.sum()),filled_changed_pixels=int((changed&(gt>0)&~original).sum()),
                unknown_changed_pixels=int((changed&(gt==0)).sum()),
                outside_target_changed_pixels=int((changed&~np.isin(gt,case['gt_ids'])).sum()),
                safety_vs_r1=safety_metrics(outputs['r1']['metrics'],metrics),
                safety_vs_connect=safety_metrics(base['metrics'],connect_metrics),
                metrics=metrics,native_boundary_ROI=scalar_stats(boundary,roi))
            sreport['interventions'][label]=summary
            append_gallery(label,dict(boundary=boundary,decoded=decoded,variant=variant))
            print('TRACE_INTERVENTION',name,label,'pass',target[0]['thresholded_diagnostic_pass'],flush=True)
            write(folder/'report.json',sreport)
            return target[0]['thresholded_diagnostic_pass']

        if role['role']=='remaining_interior':
            editedtiles=rebased_tiles(views,tiles,logits_by_arm['connect'],config)
            empty_masks=[np.zeros(tuple(t['logits'].shape),bool) for t in editedtiles]
            empty_boundary,empty_blend,empty_rows=recompute_positive(views,editedtiles,empty_masks,config,gt.shape,empty_exact=True)
            if not np.array_equal(empty_boundary,base['boundary']):raise RuntimeError('Counterfactual empty-mask connect baseline differs')
            sreport['counterfactual_connect_zero_edit_exact']=dict(boundary_sha256=array_sha(empty_boundary),blend=empty_blend,tiles=empty_rows)
            for channels in ('short','all'):
                edits=[];editaudits=[]
                for tile,logits in zip(tiles,logits_by_arm['connect']):
                    mask,ma=positive_relation_mask(tile,gt,original,channels=channels,gid=gid,margin=4)
                    edits.append(mask);editaudits.append(ma)
                boundary,blend,edit_rows=recompute_positive(views,editedtiles,edits,config,gt.shape)
                evaluate('trusted_'+channels,boundary,dict(selected=editaudits,replay=edit_rows,blend=blend,
                    GT_guided_counterfactual=True,training=False,complete_trusted_native_path=True))
        else:
            strict_pass=[]
            for arm in ('r1','control'):
                boundary,edit_audit=restore_interface(base['boundary'],outputs[arm]['boundary'],narrow['strict_mask'],gt,original)
                strict_pass.append(evaluate('restore_'+arm+'_strict',boundary,edit_audit))
            # 若原覆盖接触带不足，完成版只给出乐观机制上界，不进入训练或部署。
            if not all(strict_pass):
                for arm in ('r1','control'):
                    boundary,edit_audit=restore_interface(base['boundary'],outputs[arm]['boundary'],narrow['mask'],gt,original)
                    edit_audit.update(optimistic_completed_support=True,includes_filled=True,not_trusted_training=True)
                    evaluate('restore_'+arm+'_completed',boundary,edit_audit)
        save_gallery(folder/'comparison.png',rgb,gt,case['gt_ids'],gallery,name+' frozen diagnostic')
        sreport['fixed_cohort_reference_reuse_directly_verified']=rebase_verified
        # 保存真实局部连续响应及ROI供离线复核；不下载特征或模型。
        ys,xs=np.nonzero(np.isin(gt,case['gt_ids']));y0,x0,y1,x1=max(0,int(ys.min())-50),max(0,int(xs.min())-50),min(gt.shape[0],int(ys.max())+51),min(gt.shape[1],int(xs.max())+51)
        crop=np.s_[y0:y1,x0:x1];arrays={'box':np.array([y0,x0,y1,x1]),'roi':roi[crop],'gt':gt[crop],'original':original[crop]}
        for arm,value in outputs.items():
            arrays[arm+'_boundary']=value['boundary'][crop];arrays[arm+'_marker']=value['variant']['marker_belt'][crop]
            arrays[arm+'_instances']=value['decoded']['instances'][crop]
            for k,v in value['components'].items():arrays[arm+'_'+k]=v[crop]
        np.savez_compressed(folder/'detail.npz',**arrays)
        sreport['gallery']=dict(path=str((folder/'comparison.png').relative_to(ROOT)),sha256=sha(folder/'comparison.png'))
        write(folder/'report.json',sreport);report['sources'][name]=sreport;write(output/'report.json',report)
        for tile in tiles:
            if validate_frozen_tile_features(tile,model.affinity_decoder)!=tile['features_sha256']:raise RuntimeError('Frozen features changed')
        del outputs,logits_by_arm,tiles,capture,gallery;gc.collect();torch.cuda.empty_cache()
    if frozen!=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items())):raise RuntimeError('Original upstream changed')
    freeze_contract(model,restorer)
    for arm,head in heads.items():
        expected=fit['initial_head_sha256'] if arm=='r1' else head_receipts[arm]['state_sha256']
        if tensor_digest(head.state_dict().items())!=expected or any(p.grad is not None or p.requires_grad for p in head.parameters()):raise RuntimeError('Frozen diagnostic head changed')
    verify_hashes(protected);verify_hashes(code,code=True)
    if precision_record()!=precision or runtime_contract(config)!=runtime:raise RuntimeError('Runtime changed')
    report.update(complete=True,all_original_inputs_and_code_exact=True,all_frozen_model_heads_features_exact=True,elapsed_seconds=time.time()-started)
    write(output/'report.json',report);write(output/'status.json',dict(status='complete',training=False))
    print('INTERIOR_TRACE_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/inference/affinity_interior_trace.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    try:run(args)
    except FileExistsError:raise
    except BaseException as error:
        folder=getattr(args,'owned_output',None)
        if folder is not None and folder.exists() and folder.is_relative_to(ROOT/'outputs'):
            write(folder/'status.json',dict(status='failed',training=False,error=repr(error)))
        raise
