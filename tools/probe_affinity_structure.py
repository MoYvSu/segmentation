# -*- coding: utf-8 -*-
"""七源新GT完整结构反事实：只改native边界，不训练或生成比赛候选。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time
import zipfile
import platform
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.semantic_targets import load_completed_semantic_source
from tools.analyze_affinity_bottleneck import load_prediction, quality_summary, source_name
from tools.analyze_marker_stages import area_status, fixed_matching, verify_hashes
from tools.probe_affinity_bottleneck import case_quality, measure, validate_case
from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from tools.probe_marker_stages import compare_actual, connectivity_for_variant, trace_cases
from tools.run_marker_reflect import array_sha, config_gate, read, save_prediction, sha, source_snapshot, write
from tools.semantic_marker_diagnostic import marker_stages
from utils.affinity_structure import apply_structure, interface_mask, interior_mask, mask_provenance

STAGE_SHA = 'd30220665facf4f8013a4b6afe955e115816e4a1e5304ade88342c543598f152'
R1_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'


def select_cases(record, kind):
    """沿用已封存病例；排除已修复、早期未知归因和既存GT结构歧义。"""
    observed = {row['case_id']:row for row in record['variants']['reflect']['cases']}
    selected, skipped = [], []
    direction = 'negative' if kind == 'interface' else 'positive'
    for case in record['source']['cases']:
        if case['direction'] != direction:
            continue
        row = observed[case['case_id']]
        reason = None
        if row['full_known_quality']['whole_goal']:
            reason = 'already_resolved_by_reflect'
        elif (row.get('previous_gt_structure_ambiguity') or case.get('previous_gt_structure_ambiguity')
              or (direction == 'positive' and record['source'].get('reference_receipt'))):
            reason = 'known_GT_structure_ambiguity_positive_excluded'
        elif direction == 'negative':
            first = row['stage_trace']['stages'][0]['known_target_domain']
            if case.get('phase_kind') != 'FF':
                reason = 'not_remaining_FF_target'
            elif first['connected'] is not True:
                reason = 'early_relation_unknown_or_already_separated'
        if reason:
            skipped.append(dict(case_id=case['case_id'], reason=reason))
        else:
            selected.append(deepcopy(case))
    return selected, skipped


def target_result(decoded, gt, original, case, metrics, gt_classes, pred_classes):
    full = case_quality(decoded, gt, case)
    partial = case_quality(decoded, np.where(original, gt, 0), case)
    summary = quality_summary(full, full['endpoint_relation'], case['direction'])
    by_gt = {row['gt']:row for row in metrics['full_ferrite']['gt_rows']}
    pairs = {p['gt']:p for p in metrics['full_partition']['pairs']}
    geometry = metrics['full_partition']['geometry']
    rows = []
    for g in case['gt_ids']:
        if g in by_gt:
            rows.append(by_gt[g])
        else:
            pair = pairs.get(g); p = pair['pred'] if pair else None
            rows.append(dict(gt=g,area=int((gt==g).sum()),matched_pred=p,
                matched_iou=pair['iou'] if pair else None,predicted_phase=pred_classes.get(str(p)) if p else None,
                censored=g in metrics['full_partition']['conservative']['censored_gt_ids'],
                significant_geometry_split=g in geometry['split_gt_ids'],
                matched_pred_significant_geometry_merge=any(m['pred']==p for m in geometry['merged_predictions'])))
    matches = [row['matched_pred'] for row in rows]
    match_ok = all(p is not None and p > 0 and row['matched_iou'] >= .5
                   and row['predicted_phase'] == gt_classes[row['gt']] for p, row in zip(matches, rows))
    if case['direction'] == 'negative':
        match_ok = match_ok and len(set(matches)) == len(matches)
    topology_ok = all(not row['significant_geometry_split']
                      and not row['matched_pred_significant_geometry_merge'] for row in rows)
    original_deep95 = all(row['deep_pixels'] > 0 and row['dominant_deep_fraction'] >= .95
                          for row in partial['gt_rows'])
    return dict(case_id=case['case_id'], direction=case['direction'], gt_ids=case['gt_ids'],
        whole_known_quality=summary, original_covered_quality=partial,
        matched_gt=[{k:row.get(k) for k in ('gt','area','matched_pred','matched_iou','predicted_phase',
            'censored','topology_censored','significant_geometry_split',
            'matched_pred_significant_geometry_merge','safe_one_to_one_area_eligible','area_relative_error')}
            for row in rows],
        full_target_pass=bool(summary['whole_goal'] and match_ok and topology_ok and original_deep95),
        target_pass_scope='Processed-GT complete/deep plus original-covered deep, IoU>=.5, recorded phase and no significant split/merge; censored objects remain diagnostic only')


def safety_metrics(reference, candidate):
    before = reference['full_ferrite']['gt_rows']; after = candidate['full_ferrite']['gt_rows']
    matching = fixed_matching(before, after)
    old = {r['gt']:r for r in before}; new = {r['gt']:r for r in after}
    cohort = matching['fixed_gt_ids']
    class_changes = [g for g in cohort if old[g]['predicted_phase'] in (0, 1)
                     and new[g]['predicted_phase'] in (0, 1)
                     and old[g]['predicted_phase'] != new[g]['predicted_phase']]
    new_wrong = [g for g in cohort if old[g]['predicted_phase'] == 1 and new[g]['predicted_phase'] == 0]
    unassessed = [g for g in cohort if new[g]['predicted_phase'] is None]
    areas = area_status(candidate['fixed_area'])
    area_rows = candidate['fixed_area']['rows']
    reference_mae = sum(abs(r['r1_relative_error']) for r in area_rows)/len(area_rows) if area_rows else None
    comparable = [r for r in area_rows if r['candidate_comparable']]
    candidate_mae = (sum(abs(r['candidate_relative_error']) for r in comparable)/len(area_rows)
                     if area_rows and len(comparable)==len(area_rows) else None)
    area_deltas = [dict(gt=r['gt'],absolute_error_delta=abs(r['candidate_relative_error'])-abs(r['r1_relative_error']))
                   for r in comparable]
    iou_deltas = [dict(gt=r['gt'],delta=r['candidate_iou_missing_as_zero']-r['reference_iou'])
                  for r in matching['rows']]
    added = {k:v['added'] for k,v in candidate['error_changes'].items() if v['added']}
    safe = not matching['missing_gt_ids'] and not new_wrong and not areas['incomparable_count'] and not added
    return dict(class_change_semantics='known_matched_phase_only_v2',
        fixed_matching=matching, fixed_F_class_changes=class_changes,
        class_unassessed_unmatched_gt_ids=unassessed,
        newly_wrong_F_class_gt_ids=new_wrong, fixed_area_status=areas,
        fixed_area_rows=candidate['fixed_area'], error_changes=candidate['error_changes'],
        no_new_failures=bool(safe),
        same_fixed_area_MAE=dict(reference=reference_mae,candidate=candidate_mae,
            delta=candidate_mae-reference_mae if candidate_mae is not None else None,
            denominator=len(area_rows),failures_not_dropped=True,
            worse_rows=[r for r in area_deltas if r['absolute_error_delta']>0],
            worst_absolute_error_delta=max((r['absolute_error_delta'] for r in area_deltas),default=0)),
        fixed_iou_decreased_gt=[r for r in iou_deltas if r['delta']<0],
        worst_fixed_iou_delta=min((r['delta'] for r in iou_deltas),default=0),
        no_average_regression=bool(matching['fixed_mean_iou_delta']>=-1e-12 and
            (not area_rows or (candidate_mae is not None and candidate_mae<=reference_mae+1e-12))),
        caveat='Errors can overlap; counts must not be added as independent events. Comparable area changes are not official area score.')


def decode_reflect(semantic, boundary, config, *, real_boundary=None):
    stage = marker_stages(boundary, config)
    variant = build_marker_variant(stage, 'reflect', reflection_padding=32)
    real = boundary if real_boundary is None else real_boundary
    decoded = decode_marker_replay(semantic, torch.from_numpy(real)[None,None], config, variant['marker_labels'])
    if not np.array_equal(decoded['actual_markers'], variant['marker_labels']):
        raise RuntimeError('Actual watershed seeds differ from chosen reflect marker')
    return decoded, stage, variant


def analyze_source(record, config, output, protected):
    source = record['source']; name = source_name(source); folder = output/name; folder.mkdir()
    image = ROOT/source['source_path']; gtpath = ROOT/source['gt_path']
    rgb = cv2.cvtColor(cv2.imread(str(image)), cv2.COLOR_BGR2RGB)
    gt, lookup = load_completed_semantic_source(gtpath, rgb.shape[:2])
    classes = {int(g):int(lookup[g]) for g in np.unique(gt) if g > 0}
    with np.load(gtpath, allow_pickle=False) as payload:
        original = payload['original_covered'].astype(bool)
        filled = payload['filled'].astype(bool); unknown = payload['residual_unknown'].astype(bool)
    if (not np.array_equal(filled, (gt > 0) & ~original) or not np.array_equal(unknown, gt == 0)
            or np.any(original & (gt == 0))):
        raise RuntimeError('New processed GT support provenance differs')
    for case in source['cases']: validate_case(case, gt)
    refdir = ROOT/source['reference_dir']; prefix = source.get('reference_prefix','r1')
    mapfile = refdir/(prefix+'_maps.npz'); protected[str(mapfile)] = sha(mapfile)
    with np.load(mapfile, allow_pickle=False) as payload:
        semantic = payload['semantic'].copy(); boundary = np.squeeze(payload['boundary']).copy()
        raw = payload['raw_probability'].copy()
        actual = {k:np.squeeze(payload[k]).copy() for k in ('boundary','markers','watershed')}
    original_decoded = decode_marker_replay(semantic, torch.from_numpy(boundary)[None,None], config)
    compare_actual(original_decoded, raw, actual, refdir, prefix, boundary)
    reference, refstages, refvariant = decode_reflect(semantic, boundary, config)
    cached = load_prediction(ROOT/'outputs/marker_stages'/name, 'reflect')
    if any(not np.array_equal(cached[k],value) for k,value in (
            ('instances',reference['instances']),('markers',reference['actual_markers']),
            ('watershed',reference['watershed']),('boundary',boundary))):
        raise RuntimeError('Actual reflect stage/cache differs: '+name)
    refmetrics, refvotes = measure(reference, raw, gt, classes)
    if refvotes != cached['classes']: raise RuntimeError('Actual reflect raw classes differ')
    save_prediction(folder,'reference',reference['instances'],refvotes)
    row = dict(source=source, new_GT_provenance=dict(original_pixels=int(original.sum()),
        filled_pixels=int(filled.sum()), unknown_pixels=int(unknown.sum())), baseline_R1_exact=True,
        baseline_reflect_maps_and_classes_exact=True,
        frozen_maps=dict(semantic_sha256=array_sha(semantic),boundary_sha256=array_sha(boundary),raw_sha256=array_sha(raw)),
        reference_metrics=refmetrics,
        reference_cases=[target_result(reference,gt,original,c,refmetrics,classes,refvotes) for c in source['cases']],
        selected={}, arms={})
    for kind in ('interface','interior'):
        cases, skipped = select_cases(record,kind); row['selected'][kind] = dict(cases=cases,skipped=skipped)
        if not cases: continue
        fullmask = np.zeros(gt.shape,bool); strictmask = fullmask.copy(); masks = []
        for case in cases:
            proposal = (interface_mask(gt,original,case['gt_ids'],radius=2) if kind=='interface'
                        else interior_mask(gt,original,case['gt_ids'][0],margin=4))
            fullmask |= proposal['mask']; strictmask |= proposal['strict_mask']
            masks.append(dict(case_id=case['case_id'],**{k:v for k,v in proposal.items() if not isinstance(v,np.ndarray)}))
        if kind == 'interior':
            # 只清除会参与弱重建的内部响应；保留低于low的原数值。
            active = boundary >= float(config['inference']['marker_boundary_low_threshold'])
            fullmask &= active; strictmask &= active
        for scope, mask in (('trusted',strictmask),('completed',fullmask)):
            edited, audit = apply_structure(boundary,mask,1. if kind=='interface' else 0.,gt=gt,original_covered=original)
            if not np.array_equal(boundary[~mask],edited[~mask]): raise RuntimeError('Mask-exterior changed')
            modes = ['full']
            # 两主病例另做只换marker的交叉，其余六源用于独立全流程复查。
            if (name=='train_659' and kind=='interface') or (name=='train_347' and kind=='interior'):
                modes.append('marker')
            for mode in modes:
                arm = kind+'_'+scope+'_'+mode
                decoded, stage, variant = decode_reflect(semantic,edited,config,
                    real_boundary=boundary if mode=='marker' else None)
                if mode=='marker' and decoded['elevation_sha256'] != reference['elevation_sha256']:
                    raise RuntimeError('Marker-only branch changed real WS elevation')
                metrics, votes = measure(decoded,raw,gt,classes,refmetrics)
                hashes = save_prediction(folder,arm,decoded['instances'],votes)
                safety = safety_metrics(refmetrics,metrics)
                outcomes = [target_result(decoded,gt,original,c,metrics,classes,votes) for c in source['cases']]
                selected_ids = {c['case_id'] for c in cases}
                benefits = [c['case_id'] for c in outcomes if c['case_id'] in selected_ids and c['full_target_pass']
                    and not next(r for r in row['reference_cases'] if r['case_id']==c['case_id'])['full_target_pass']]
                maps = connectivity_for_variant(stage,variant,'reflect')
                item = dict(mask=audit, mask_construction=masks, scope=scope, readout=mode,
                    actual_native_boundary_sha256=array_sha(edited), marker_source_boundary_sha256=array_sha(edited),
                    real_WS_source_boundary_sha256=array_sha(boundary if mode=='marker' else edited),
                    real_WS_elevation_equal_to_reference=decoded['elevation_sha256']==reference['elevation_sha256'],
                    elevation_sha256=decoded['elevation_sha256'],output=hashes,
                    final_instances_equal_to_reference=bool(np.array_equal(decoded['instances'],reference['instances'])),
                    final_classes_equal_to_reference=votes==refvotes,
                    target_outcomes=outcomes, cases=trace_cases(maps,decoded,gt,source['cases']),
                    new_full_target_pass_cases=benefits, safety=safety, metrics=metrics,
                    training_permission=False, actual_affinity_or_fusion_changed=False,
                    next_representation_check_signal=bool(benefits and safety['no_new_failures'] and safety['no_average_regression']))
                row['arms'][arm] = item
                yy,xx = np.nonzero(mask)
                np.savez_compressed(folder/(arm+'_delta.npz'),y=yy.astype(np.int32),x=xx.astype(np.int32),
                                    before=boundary[mask],after=edited[mask])
                del maps,decoded,metrics,stage,variant
                write(folder/'report.json',row)
                print('STRUCTURE',name,arm,'new_complete',benefits,'safe',safety['no_new_failures'],flush=True)
                gc.collect()
    # 图册使用同一原图，仅显示最终划分／分类；不隐藏图外通路。
    from tools.analyze_affinity_connectivity import render
    panels=[(reference['instances'],refvotes)]; labels=['reflect reference']
    for kind in ('interface','interior'):
        for scope in ('trusted','completed'):
            arm=kind+'_'+scope+'_full'
            if arm in row['arms']:
                ids=cv2.imread(str(folder/(arm+'_inst.png')),cv2.IMREAD_UNCHANGED)
                panels.append((ids,read(folder/(arm+'_class.json')))); labels.append(kind+' '+scope)
    render(image,panels,labels,folder/'compare.png',name+' TRAIN GT counterfactual only',width=380)
    write(folder/'report.json',row)
    return row


def run(args):
    from utils.config import load_config
    out=(ROOT/args.output).resolve()
    if out.exists(): raise FileExistsError('Fresh ignored outputs directory required')
    if not out.is_relative_to(ROOT/'outputs'): raise ValueError('Ignored outputs directory required')
    origin=ROOT/args.input
    if sha(origin)!=STAGE_SHA: raise RuntimeError('Actual seven-source stage identity differs')
    prior=read(origin)
    if not prior.get('complete') or len(prior['sources'])!=7 or prior['identity']['checkpoint_sha256']!=R1_SHA:
        raise RuntimeError('Actual complete mixed-R1 stage receipt required')
    verify_hashes(prior['protected_inputs']); verify_hashes(prior['protected_code'],code=True)
    capture=read(ROOT/'outputs/affinity_weak/capture.json')
    config=capture['arms']['r1']['config']; config_gate(config,load_config('config/train/affinity_balance_mixed.yaml'))
    if sha(capture['arms']['r1']['checkpoint'])!=R1_SHA: raise RuntimeError('Frozen checkpoint changed')
    protected={**prior['protected_inputs'],str(origin):STAGE_SHA}
    for s in prior['sources']:
        for p,h in s['input_hashes'].items():
            if sha(Path(p))!=h: raise RuntimeError('Source/newGT/class differs')
            protected[p]=h
        for suffix in ('_maps.npz','_inst.png','_class.json'):
            p=ROOT/'outputs/marker_stages'/source_name(s['source'])/('reflect'+suffix)
            protected[str(p)]=sha(p)
    torch.set_num_threads(4); cv2.setNumThreads(2); out.mkdir(); started=time.time()
    import skimage
    import skimage.morphology
    cpu_runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
        opencv=cv2.__version__,skimage=skimage.__version__,torch_threads=torch.get_num_threads(),
        opencv_thinning_available=hasattr(cv2,'ximgproc') and hasattr(cv2.ximgproc,'thinning'))
    thin_calls=[]; real_skeletonize=skimage.morphology.skeletonize
    def observed_skeletonize(*a,**kw):
        thin_calls.append(True);return real_skeletonize(*a,**kw)
    code,_=source_snapshot(out)
    report=dict(format='affinity_structure_probe_v1',complete=False,training=False,no_gpu_forward=True,
        no_model_load=True,no_optimizer_steps=True,no_submission=True,best_deployment_changed=False,
        actual_affinity_or_fusion_changed=False,identity=prior['identity'],current_CPU_runtime=cpu_runtime,reference_score=[.8645,.9006],
        policy=dict(interface_radius=2,interface_adjacency=4,interior_margin=4,
                    interior_active_low=.45,reflection_padding=32,scopes=['trusted','completed'],
                    completed_is_processed_GT_optimistic_counterfactual=True),
        protected_inputs=protected,protected_code=code,sources=[],
        caveat='Seen training newGT native-scalar counterfactual only. Completed masks are not trusted supervision; even successful trusted masks are not actual affinity/gradient or generalization evidence.')
    write(out/'preflight.json',report)
    write(out/'status.json',dict(status='running',completed=0,total=7,training=False))
    # 先跑两主病例，再按原预选清单补其余源。
    indexed={source_name(s['source']):s for s in prior['sources']}
    ordered=['train_659','train_347']+[n for n in indexed if n not in {'train_659','train_347'}]
    with patch('skimage.morphology.skeletonize',side_effect=observed_skeletonize):
        for name in ordered:
            report['sources'].append(analyze_source(indexed[name],config,out,protected))
            write(out/'report.json',report);write(out/'status.json',dict(status='running',completed=len(report['sources']),total=7,training=False))
    verify_hashes(protected);verify_hashes(code,code=True)
    refs=[s['reference_metrics'] for s in report['sources']]
    counts=dict(matched_F=sum(sum(r['matched_pred'] is not None for r in m['full_ferrite']['gt_rows']) for m in refs),
                safe_area=sum(m['fixed_area']['original_fixed_count'] for m in refs))
    if counts!={'matched_F':689,'safe_area':135}: raise RuntimeError('Actual reflect fixed cohort differs')
    report.update(complete=True,protected_inputs_exact=True,protected_code_exact=True,
                  reference_cohort=counts,elapsed_seconds=time.time()-started,
                  observed_skimage_skeletonize_calls=len(thin_calls))
    write(out/'report.json',report);write(out/'status.json',dict(status='complete',completed=7,training=False))
    html=['<!doctype html><meta charset="utf-8"><title>完整结构反事实</title><h1>已见训练源的几何机制检查</h1>',
          '<p>原reflect、仅原覆盖修改、processed-GT完整修改分别展示；不是比赛候选或测试真值。</p>']
    for name in ordered: html.append('<h2>'+name+'</h2><img style="max-width:100%" src="'+name+'/compare.png">')
    (out/'index.html').write_text('\n'.join(html),encoding='utf8')
    with zipfile.ZipFile(out/'report.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in [out/'report.json',out/'preflight.json',out/'status.json',out/'index.html',*out.glob('*/compare.png')]:
            archive.write(p,p.relative_to(out).as_posix())
    print('STRUCTURE COMPLETE',counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='outputs/marker_stages/report.json')
    parser.add_argument('--output',default='outputs/geometry_structure')
    args=parser.parse_args()
    try: run(args)
    except Exception as error:
        out=(ROOT/args.output).resolve()
        if isinstance(error,FileExistsError): raise
        if out.exists() and out.is_relative_to(ROOT/'outputs') and not (out/'preflight.json').exists():
            raise
        if out.exists() and out.is_relative_to(ROOT/'outputs'):
            write(out/'status.json',dict(status='failed',training=False,error=repr(error)))
        raise
