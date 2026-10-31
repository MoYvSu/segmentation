# -*- coding: utf-8 -*-
"""对已封存冻结定位补四个最小支持反事实；不训练、不改GT。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data.semantic_targets import load_completed_semantic_source
from tools.affinity_feature_capture import capture_frozen_features, replay_head_boundary, validate_frozen_tile_features
from tools.analyze_affinity_bottleneck import source_name
from tools.analyze_affinity_structure import case_summary
from tools.analyze_marker_stages import verify_hashes
from tools.interior_trace_gallery import save_gallery
from tools.probe_affinity_bottleneck import measure
from tools.probe_affinity_interior import R1_SHA, RESTORER_SHA, cpu_runtime, recompute_positive
from tools.probe_affinity_interior_fit import freeze_contract
from tools.probe_affinity_marker_replay import decode_marker_replay
from tools.probe_affinity_structure import decode_reflect, safety_metrics, target_result
from tools.probe_interior_trace import restore_interface, rebased_tiles, stage_record, with_baseline
from tools.run_marker_reflect import array_sha, config_gate, precision_record, read, save_prediction, sha, source_snapshot, write
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import tensor_digest
from utils.affinity_interior_loss import build_interior_supervision
from utils.affinity_structure import interior_mask
from utils.config import load_config, project_path
from utils.interior_support_masks import interface_support_factor, footprint_local_masks


def markers_from_belt(belt,min_area):
    count,labels,stats,_=cv2.connectedComponentsWithStats((belt==0).astype(np.uint8),connectivity=8)
    markers=np.zeros(belt.shape,np.int32);number=0
    for label in range(1,count):
        if stats[label,cv2.CC_STAT_AREA]>=min_area:
            number+=1;markers[labels==label]=number
    return markers


def run(args):
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from tools.run_affinity_sweep import runtime_contract
    merged=load_config(args.config);opts=merged.pop('affinity_support_trace');oldopts=merged.pop('affinity_interior_trace');config=merged
    output=Path(project_path(config,opts['output_dir'])).resolve()
    if output.exists():raise FileExistsError('Fresh ignored output required')
    if not output.is_relative_to(ROOT/'outputs'):raise ValueError('Ignored output required')
    if not args.run:print('Validated recipe; --run starts finite frozen support diagnosis.');return
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA sam2_env required')
    tracepath=ROOT/opts['trace'];fitpath=ROOT/oldopts['fit'];planpath=ROOT/oldopts['plan'];receiptpath=ROOT/oldopts['receipt']
    if sha(tracepath)!=opts['trace_sha256']:raise RuntimeError('Sealed trace differs')
    trace,fit,plan,receipt=[read(p) for p in (tracepath,fitpath,planpath,receiptpath)]
    if not trace['complete'] or trace['training'] or not fit['complete']:raise ValueError('Completed frozen receipts required')
    verify_hashes(trace['protected_inputs']);verify_hashes(trace['protected_code'],code=True)
    r1=receipt['arms']['r1'];config_gate(config,r1['config'])
    torch.set_num_threads(4);cv2.setNumThreads(2)
    precision,runtime=precision_record(),runtime_contract(config)
    if precision!=trace['precision'] or runtime!=trace['runtime'] or cpu_runtime()!=trace['CPU_runtime']:raise RuntimeError('Runtime differs')
    if sha(Path(r1['checkpoint']))!=R1_SHA:raise RuntimeError('R1 changed')
    restore=Path(project_path(config,config['backend_adaptation']['restoration']['checkpoint']))
    if sha(restore)!=RESTORER_SHA:raise RuntimeError('D5a changed')
    core=load_original_core(config);views=core.coverage_original_views
    model,bundle=load_backend(r1['checkpoint'],config,'cuda');model.eval().requires_grad_(False)
    restorer=get_restorer(config,'cuda').eval().requires_grad_(False)
    frozen=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items()))
    if frozen!=(fit['frozen_model_state_sha256'],fit['frozen_restorer_state_sha256']):raise RuntimeError('Inherited upstream differs')
    head=deepcopy(model.affinity_decoder).eval().requires_grad_(False)
    pin=plan['heads']['connect'];state=torch.load(ROOT/pin['path'],map_location='cuda',weights_only=False)
    head.load_state_dict(state['geometry_state_dict'],strict=True);del state
    if tensor_digest(head.state_dict().items())!=pin['state_sha256']:raise RuntimeError('Strict connect differs')
    freeze_contract(model,restorer);output.mkdir();args.owned_output=output;started=time.time()
    protected={**trace['protected_inputs'],str(tracepath):sha(tracepath),str(Path(args.config).resolve()):sha(args.config)}
    code,virtual=source_snapshot(output)
    report=dict(format='affinity_interior_support_v1',complete=False,training=False,optimizer_created=False,
        no_submission=True,GT_enrichment='postponed',config=load_config(args.config),protected_inputs=protected,
        protected_code=code,virtual_code_paths=virtual,sources={},precision=precision,runtime=runtime,
        caveat='Four fixed seen-GT counterfactuals; no GT rewrite or test answer. Marker deletion and saturated logits are mechanism references, not deployable predictions.')
    write(output/'preflight.json',report)
    structure=read(ROOT/oldopts['structure'])
    for role in plan['roles']:
        name=role['source'];folder=output/name;folder.mkdir();source=next(x['source'] for x in structure['sources'] if source_name(x['source'])==name)
        gtpath=ROOT/source['gt_path'];image=ROOT/source['source_path']
        rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB)
        gt,lookup=load_completed_semantic_source(gtpath,rgb.shape[:2]);classes={int(g):int(lookup[g]) for g in np.unique(gt) if g>0}
        with np.load(gtpath) as z:original=z['original_covered'].astype(bool)
        _,capture,tiles,audit=capture_frozen_features(views,model,restorer,image,config,gt,'cuda',source['seed'])
        collected=[]
        handle=head.register_forward_hook(lambda module,inputs,result:collected.append(result['affinity_logits'].detach()))
        try:boundary,replay=replay_head_boundary(views,tiles,head,config,gt.shape)
        finally:handle.remove()
        decoded,stage,variant=decode_reflect(capture['semantic'],boundary,config);votes,_=revote_instances(decoded['instances'],capture['raw_probability'])
        basehash=save_prediction(folder,'base',decoded['instances'],votes);prior=trace['sources'][name]['arms']['connect']
        if array_sha(boundary)!=prior['boundary_sha256'] or basehash!=prior['output']:raise RuntimeError('Sealed complete connect output differs')
        base_metrics=fit['arms']['connect']['snapshots']['32']['sources'][name]['metrics'];ref_metrics=fit['arms']['control']['snapshots']['0']['sources'][name]['metrics']
        case=role['case'];cases=[case]+([role['original_positive_case']] if 'original_positive_case' in role else [])
        sr=dict(role=role,actual_capture=audit,replay=replay,base_exact=True,base_output=basehash,variants={})
        gallery=[dict(label='connect',boundary=boundary,instances=decoded['instances'],marker_belt=variant['marker_belt'])]
        def evaluate(label,b,newdecoded,newstage,newvariant,edit):
            metrics,v=measure(newdecoded,capture['raw_probability'],gt,classes,base_metrics)
            targets=[case_summary(target_result(newdecoded,gt,original,c,metrics,classes,v)) for c in cases]
            sub=folder/label;sub.mkdir();hashes=save_prediction(sub,'prediction',newdecoded['instances'],v)
            sr['variants'][label]=dict(edit=edit,output=hashes,boundary_sha256=array_sha(b),target_outcomes=targets,
                stages=stage_record(newdecoded,newstage,newvariant,gt,original,cases),safety_vs_connect=safety_metrics(base_metrics,metrics),
                safety_vs_r1=safety_metrics(ref_metrics,with_baseline(metrics,ref_metrics)),metrics=metrics)
            gallery.append(dict(label=label,boundary=b,instances=newdecoded['instances'],marker_belt=newvariant['marker_belt']))
            write(folder/'report.json',sr);print('SUPPORT_RESULT',name,label,'pass',targets[0]['thresholded_diagnostic_pass'],flush=True)
        if role['role']=='lost_interface':
            detail=ROOT/'outputs/interior_trace'/name/'detail.npz';protected[str(detail)]=sha(detail)
            with np.load(detail) as z:
                y,x,y1,x1=map(int,z['box']);local_reference=z['control_boundary'];local_base=z['connect_boundary']
                if not np.array_equal(local_base,boundary[y:y1,x:x1]):raise RuntimeError('Sealed boundary crop differs')
                reference=boundary.copy();reference[y:y1,x:x1]=local_reference
            masks,support_audit=interface_support_factor(gt,original,role['pair'],radius=2)
            sr['support_partition']=support_audit
            for label in ('strict_plus_original','strict_plus_filled'):
                b,edit=restore_interface(boundary,reference,masks[label],gt,original)
                edit['reference']='sealed control continuous response';newdecoded,newstage,newvariant=decode_reflect(capture['semantic'],b,config)
                evaluate(label,b,newdecoded,newstage,newvariant,edit)
        else:
            domain=interior_mask(gt,original,role['gt_id'],margin=4)
            roi=domain['strict_mask']&(variant['marker_belt']>0)
            if int(roi.sum())!=trace['sources'][name]['ROI']['provenance']['masked_pixels']:raise RuntimeError('Fixed marker footprint differs')
            zero_markers=markers_from_belt(variant['marker_belt'],int(variant['resolved']['min_instance_area']))
            if not np.array_equal(zero_markers,decoded['actual_markers']):raise RuntimeError('Zero marker edit differs from actual reflect seeds')
            belt=variant['marker_belt'].copy();belt[roi]=0
            markers=markers_from_belt(belt,int(variant['resolved']['min_instance_area']))
            newdecoded=decode_marker_replay(capture['semantic'],torch.from_numpy(boundary)[None,None],config,markers)
            newvariant=dict(variant,marker_belt=belt,marker_labels=markers)
            evaluate('marker_footprint_clear',boundary,newdecoded,stage,newvariant,
                dict(selected_native_pixels=int(roi.sum()),continuous_boundary_exact=True,real_watershed_elevation_unchanged=True,
                    marker_GT_guided=True,unselected_marker_belt_exact=bool(np.array_equal(belt[~roi],variant['marker_belt'][~roi]))))
            selections=[];selection_audits=[];old_masks={m['index']:m['audit'] for m in fit['captures'][name]['supervision']}
            for tile in tiles:
                supervision,ma=build_interior_supervision(tile,gt,original,tile['logits'],gid=role['gt_id'],margin=4,low=.45,confidence=.9)
                if tile['index'] in old_masks and ma!=old_masks[tile['index']]:raise RuntimeError('Sealed supervision differs')
                mask,ma=footprint_local_masks(tile,gt,original,roi,supervision,role['gt_id'],device=tile['logits'].device)
                if tile['index'] not in old_masks and mask.any():raise RuntimeError('Nontraining tile selected')
                selections.append(mask);selection_audits.append(ma)
            candidate_tiles=rebased_tiles(views,tiles,collected,config)
            empty,_,_=recompute_positive(views,candidate_tiles,[np.zeros(m.shape,bool) for m in selections],config,gt.shape,empty_exact=True)
            if not np.array_equal(empty,boundary):raise RuntimeError('Support empty correction differs')
            b,blend,edits=recompute_positive(views,candidate_tiles,selections,config,gt.shape)
            newdecoded,newstage,newvariant=decode_reflect(capture['semantic'],b,config)
            evaluate('local_footprint_strong',b,newdecoded,newstage,newvariant,dict(selections=selection_audits,
                selected_edges=sum(int(m.sum()) for m in selections),per_channel=[sum(int(m[0,c].sum()) for m in selections) for c in range(8)],
                whole_connect_zero_edit_exact=True,blend=blend,tiles=edits,
                changed_native_pixels=int((b!=boundary).sum()),outside_GT_changed_pixels=int(((b!=boundary)&(gt!=role['gt_id'])).sum()),
                filled_changed_pixels=int(((b!=boundary)&(gt>0)&~original).sum()),unknown_changed_pixels=int(((b!=boundary)&(gt==0)).sum())))
        save_gallery(folder/'comparison.png',rgb,gt,case['gt_ids'],gallery,name+' minimal support')
        np.savez_compressed(folder/'base_maps.npz',boundary=boundary,semantic=capture['semantic'].numpy(),
            raw_probability=capture['raw_probability'],instances=decoded['instances'],markers=decoded['actual_markers'],watershed=decoded['watershed'])
        sr['gallery']=dict(path=str((folder/'comparison.png').relative_to(ROOT)),sha256=sha(folder/'comparison.png'))
        write(folder/'report.json',sr);report['sources'][name]=sr;write(output/'report.json',report)
        for tile in tiles:
            if validate_frozen_tile_features(tile,model.affinity_decoder)!=tile['features_sha256']:raise RuntimeError('Frozen feature changed')
        del collected,tiles,capture,decoded,gallery;gc.collect();torch.cuda.empty_cache()
    if frozen!=(tensor_digest(model.state_dict().items()),tensor_digest(restorer.state_dict().items())) or tensor_digest(head.state_dict().items())!=pin['state_sha256']:raise RuntimeError('Frozen states changed')
    freeze_contract(model,restorer)
    if any(p.requires_grad or p.grad is not None for p in head.parameters()):raise RuntimeError('Frozen copied head received gradient')
    verify_hashes(protected);verify_hashes(code,code=True)
    report.update(complete=True,all_frozen_inputs_and_models_exact=True,elapsed_seconds=time.time()-started)
    write(output/'report.json',report);write(output/'status.json',dict(status='complete',training=False));print('SUPPORT_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='config/inference/affinity_support_trace.yaml');parser.add_argument('--run',action='store_true');args=parser.parse_args()
    try:run(args)
    except FileExistsError:raise
    except BaseException as error:
        folder=getattr(args,'owned_output',None)
        if folder is not None and folder.is_relative_to(ROOT/'outputs'):write(folder/'status.json',dict(status='failed',training=False,error=repr(error)))
        raise
