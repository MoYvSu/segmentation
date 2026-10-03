# -*- coding: utf-8 -*-
"""复用七源真实连续图，检查原／reflect／取消细化的marker完整读出。

只在sam2_env的原CPU后处理环境运行；没有模型前向、训练或测试标签。
reflect为已评分参照，nothin仅作机制诊断，不自动生成提交或更换默认。
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from data.semantic_targets import load_completed_semantic_source
from tools.analyze_affinity_bottleneck import (load_prediction,map_changes,metrics_summary,
                                              quality_summary,source_name)
from tools.analyze_affinity_readout_path import IDENTITY_KEYS,load_run,validate_common
from tools.analyze_affinity_fused_gt import gallery_panel
from tools.probe_affinity_bottleneck import case_quality,measure,save_variant,validate_case
from tools.probe_affinity_marker_replay import build_marker_variant,decode_marker_replay
from tools.run_marker_reflect import array_sha,read,sha,source_snapshot,write
from tools.semantic_marker_diagnostic import connectivity_maps
from utils.marker_stage_trace import trace_case_stages,classify_stage_failure


def connectivity_for_variant(original_stages,variant,mode):
    """候选必须使用实际数组，不能由候选marker_boundary重算原thin流程。"""
    masks=dict(high_mask=original_stages['high_mask'],
        marker_boundary_mask=original_stages['marker_boundary_mask'],
        bridged_marker_mask=variant['bridged_marker_mask'])
    if mode=='reflect': masks['reflected_skeleton']=variant['cropped_skeleton']
    masks['dilated_before_seal']=variant['dilated_marker_before_seal']
    masks['marker_belt']=variant['marker_belt']
    result={}
    for key,value in masks.items():
        count,labels=cv2.connectedComponents((value==0).astype(np.uint8),connectivity=8)
        result[key]=dict(labels=labels,component_count=int(count-1))
    return result


def compare_actual(decoded,raw,reference,folder,stem,boundary):
    """复投票后，逐值核对之前已封存的正式实例及类别。"""
    from tools.semantic_crossover import revote_instances
    votes,_=revote_instances(decoded['instances'],raw)
    ids=cv2.imread(str(folder/(stem+'_inst.png')),cv2.IMREAD_UNCHANGED)
    if ids is None or not np.array_equal(ids,decoded['instances']):
        raise RuntimeError('Cached final native instance differs: '+str(folder))
    if read(folder/(stem+'_class.json'))!={str(k):v for k,v in votes.items()}:
        raise RuntimeError('Cached raw-vote final classes differ: '+str(folder))
    if reference is not None:
        for key,actual in [('markers',decoded['actual_markers']),('watershed',decoded['watershed']),
                           ('boundary',boundary)]:
            if not np.array_equal(reference[key],actual):
                raise RuntimeError('Cached actual native stage differs: '+key)
    return votes


def reflect_reference(source):
    if source.get('reference_receipt'):
        # 已有单源镜像detail.npz只是局部；只复用完整PNG/JSON。
        return ROOT/'outputs/marker_reflect/r1_p32',''
    return ROOT/source['reference_dir'],source.get('reference_prefix','r1')+'_reflect'


def reflection_exact(decoded,votes,folder,stem):
    png=folder/(stem+'_inst.png' if stem else 'inst.png')
    labels=folder/(stem+'_class.json' if stem else 'class.json')
    ids=cv2.imread(str(png),cv2.IMREAD_UNCHANGED)
    if ids is None or not np.array_equal(ids,decoded['instances']) or read(labels)!={str(k):v for k,v in votes.items()}:
        raise RuntimeError('Existing complete reflect output differs: '+str(folder))
    return dict(png=str(png.relative_to(ROOT)),png_sha256=sha(png),
        classes=str(labels.relative_to(ROOT)),classes_sha256=sha(labels),actual_final_exact=True)


def trace_cases(maps,decoded,gt,cases):
    result=[]
    for case in cases:
        trace=trace_case_stages(maps,decoded['actual_markers'],decoded['instances'],gt,case,
                                watershed=decoded['watershed'])
        quality=case_quality(decoded,gt,case)
        result.append(dict(case_id=case['case_id'],direction=case['direction'],
            whole_object_censored=case.get('censored',True),
            previous_gt_structure_ambiguity=case.get('previous_gt_structure_ambiguity'),
            stage_trace=trace,stage_classification=classify_stage_failure(trace),
            full_known_quality=quality_summary(quality,quality['endpoint_relation'],case['direction'])))
    return result


def analyze_source(record,raw_root,config,output):
    source=record['source']; name=source_name(source); folder=output/name; folder.mkdir()
    path=ROOT/source['source_path']; gt_path=ROOT/source['gt_path']
    for p,digest in record['input_hashes'].items():
        if sha(Path(p))!=digest: raise RuntimeError('Source/GT/class changed: '+p)
    rgb_bgr=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if rgb_bgr is None: raise RuntimeError('Source image missing')
    rgb=cv2.cvtColor(rgb_bgr,cv2.COLOR_BGR2RGB)
    gt,lookup=load_completed_semantic_source(gt_path,rgb.shape[:2])
    classes={int(g):int(lookup[g]) for g in np.unique(gt) if g>0}
    for case in source['cases']: validate_case(case,gt)
    refdir=ROOT/source['reference_dir']; prefix=source.get('reference_prefix','r1')
    with np.load(refdir/(prefix+'_maps.npz'),allow_pickle=False) as payload:
        semantic=payload['semantic'].copy(); boundary=np.squeeze(payload['boundary']).copy()
        raw=payload['raw_probability'].copy()
        reference={key:np.squeeze(payload[key]).copy() for key in ('boundary','markers','watershed')}
    boundary_tensor=torch.from_numpy(boundary)[None,None]
    baseline=decode_marker_replay(semantic,boundary_tensor,config)
    votes=compare_actual(baseline,raw,reference,refdir,prefix,boundary)
    base_metrics,base_votes=measure(baseline,raw,gt,classes)
    if votes!=base_votes or json.loads(json.dumps(base_metrics,allow_nan=False))!=record['baseline']:
        raise RuntimeError('Actual original raw-vote or full processedGT baseline differs')
    # 再与本轮已精确封存七源的实际baseline比，保护缓存归属。
    actual=load_prediction(raw_root/name,'baseline')
    if any(not np.array_equal(actual[k],v) for k,v in [('instances',baseline['instances']),
        ('markers',baseline['actual_markers']),('watershed',baseline['watershed']),('boundary',boundary)]):
        raise RuntimeError('Seven-source actual baseline differs')
    save_variant(folder,'baseline',baseline,base_votes,boundary)
    maps=connectivity_maps(baseline['original_stages'])
    row=dict(source=source,input_hashes=record['input_hashes'],baseline_exact=True,
        fixed_maps=dict(boundary_sha256=array_sha(boundary),semantic_sha256=array_sha(semantic),
                        raw_probability_sha256=array_sha(raw)),
        baseline_metrics=base_metrics,baseline_cases=trace_cases(maps,baseline,gt,source['cases']),variants={})
    del maps
    panels=[gallery_panel(rgb,baseline['instances'],gt>0,gt)]
    stage=baseline['original_stages']
    reflected=build_marker_variant(stage,'reflect',reflection_padding=32)
    decoded_reflect=decode_marker_replay(semantic,boundary_tensor,config,reflected['marker_labels'])
    reflect_metrics,reflect_votes=measure(decoded_reflect,raw,gt,classes,base_metrics)
    ref_folder,ref_stem=reflect_reference(source)
    ref_receipt=reflection_exact(decoded_reflect,reflect_votes,ref_folder,ref_stem)
    save_variant(folder,'reflect',decoded_reflect,reflect_votes,boundary)
    old_png=load_prediction(folder,'baseline'); reflect_png=load_prediction(folder,'reflect')
    for mode,variant,decoded,metrics,new_votes in [
        ('reflect',reflected,decoded_reflect,reflect_metrics,reflect_votes),
        ('nothin',None,None,None,None)]:
        if mode=='nothin':
            variant=build_marker_variant(stage,'nothin')
            decoded=decode_marker_replay(semantic,boundary_tensor,config,variant['marker_labels'])
            metrics,new_votes=measure(decoded,raw,gt,classes,base_metrics)
        if (decoded['elevation_sha256']!=baseline['elevation_sha256']
                or not np.array_equal(decoded['original_stages']['barrier_belt'],stage['barrier_belt'])
                or not np.array_equal(variant['marker_labels'],decoded['actual_markers'])):
            raise RuntimeError('Frozen actual WS barrier/elevation or chosen seeds changed')
        if mode=='nothin': save_variant(folder,mode,decoded,new_votes,boundary)
        maps=connectivity_for_variant(stage,variant,mode)
        candidate=load_prediction(folder,mode)
        item=dict(actual_capture=decoded['capture_audit'],stage_resolved=variant['resolved'],
            real_barrier_elevation_exact=True,baseline_relative=metrics_summary(base_metrics,metrics),
            actual_map_changes_vs_r1=map_changes(old_png,candidate),
            actual_map_changes_vs_reflect=map_changes(reflect_png,candidate),
            cases=trace_cases(maps,decoded,gt,source['cases']),
            marker_zero_pixels=int((decoded['actual_markers']==0).sum()),
            ignored_small_core_count=int(len(np.unique(variant['core_labels']))-1-variant['marker_count']),
            existing_output_exact=ref_receipt if mode=='reflect' else None)
        if mode=='nothin':
            new_base_metrics,_=measure(decoded,raw,gt,classes,reflect_metrics)
            item['reflect_relative']=metrics_summary(reflect_metrics,new_base_metrics)
        row['variants'][mode]=item
        panels.append(gallery_panel(rgb,decoded['instances'],gt>0,gt))
        del maps
    small=[cv2.resize(p,(500,round(p.shape[0]*500/p.shape[1])),interpolation=cv2.INTER_AREA) for p in panels]
    cv2.imwrite(str(folder/'compare.png'),cv2.cvtColor(np.concatenate(small,axis=1),cv2.COLOR_RGB2BGR))
    for p,digest in record['input_hashes'].items():
        if sha(Path(p))!=digest: raise RuntimeError('Source/GT/class mutated during analysis')
    write(folder/'report.json',row)
    return row


def run(args):
    output=(ROOT/args.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Fresh ignored outputs directory required')
    runs=[];sources=[];protected={};identities=[]
    for path in args.runs:
        root,report,receipt=load_run(ROOT/path,'affinity_bottleneck_probe_v1')
        if runs: validate_common(report,runs[0][1])
        runs.append((root,report,receipt));identities.append(receipt)
        sources.extend((root,source) for source in report['sources'])
        protected[receipt['path']]=receipt['sha256']
    names=[source_name(s['source']) for _,s in sources]
    if len(names)!=7 or len(set(names))!=7: raise ValueError('Actual seven-source cohort required')
    config=runs[0][1]['config'];capture=read(ROOT/args.capture)
    if capture['arms']['r1']['config']!=config: raise ValueError('Pinned original config differs')
    checkpoint=Path(capture['arms']['r1']['checkpoint'])
    if sha(checkpoint)!=runs[0][1]['checkpoint_sha256']: raise ValueError('R1 checkpoint changed')
    protected[str(checkpoint)]=sha(checkpoint);protected[str(ROOT/args.capture)]=sha(ROOT/args.capture)
    for root,source in sources:
        meta=source['source'];ref=ROOT/meta['reference_dir'];prefix=meta.get('reference_prefix','r1')
        for file in (ref/(prefix+'_maps.npz'),ref/(prefix+'_inst.png'),ref/(prefix+'_class.json')):
            protected[str(file)]=sha(file)
        folder,stem=reflect_reference(meta)
        for file in (folder/(stem+'_inst.png' if stem else 'inst.png'),folder/(stem+'_class.json' if stem else 'class.json')):
            protected[str(file)]=sha(file)
    output.mkdir(parents=True);started=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
    code,_=source_snapshot(output)
    report=dict(format='marker_stage_probe_v1',complete=False,training=False,no_gpu_forward=True,
        no_model_load=True,no_optimizer_steps=True,best_configuration_changed=False,
        official_accuracy=False,reference_score=dict(source='user_corrected',mIoU=.8645,area=.9006,total=88.255),
        identity={k:runs[0][1][k] for k in IDENTITY_KEYS},input_runs=identities,
        protected_inputs=protected,protected_code=code,expected_sources=names,sources=[],
        candidate='nothin diagnosis only',reference='existing scored reflect32, original R1 rollback',
        caveat='Seen processed training GT; global/domain connectivity and seed labels are not final grain accuracy')
    write(output/'preflight.json',report)
    write(output/'status.json',dict(status='running',training=False,completed=0,expected=7))
    try:
        selected=sources[:args.limit] if args.limit else sources
        for root,source in selected:
            report['sources'].append(analyze_source(source,root,config,output))
            write(output/'report.json',report)
            write(output/'status.json',dict(status='running',training=False,completed=len(report['sources']),expected=7))
            print('MARKER_STAGE',len(report['sources']),source['source']['source'],flush=True)
            gc.collect()
        if any(sha(Path(p))!=digest for p,digest in protected.items()):
            raise RuntimeError('Protected input changed')
        if any(sha(ROOT/p)!=digest for p,digest in code.items()): raise RuntimeError('Protected code changed')
        report.update(complete=len(report['sources'])==7,protected_inputs_exact=True,
            protected_code_exact=True,elapsed_seconds=time.time()-started)
        write(output/'report.json',report)
        write(output/'status.json',dict(status='complete',complete=report['complete'],training=False,
            completed=len(report['sources']),expected=7))
        print('MARKER_STAGE_COMPLETE',report['complete'],flush=True)
    except Exception as exc:
        write(output/'status.json',dict(status='failed',error=repr(exc),training=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',default=['outputs/affinity_bottleneck','outputs/bottleneck_extra'])
    p.add_argument('--output',default='outputs/marker_stages')
    p.add_argument('--capture',default='outputs/affinity_weak/capture.json')
    p.add_argument('--limit',type=int,default=0)
    run(p.parse_args())


if __name__=='__main__': main()
