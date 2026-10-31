# -*- coding: utf-8 -*-
"""虚构七源CPU证据：跨run类别、哈希、实际图与阶段归因验收。"""
from copy import deepcopy
import json
import shutil

import cv2
import numpy as np
import pytest

from tools import analyze_marker_stages as analysis
from tools.analyze_affinity_bottleneck import endpoint_relation, map_changes, metrics_summary, quality_summary
from utils.marker_stage_trace import classify_stage_failure, trace_case_stages
from utils.patch_diagnostic import phase_description


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def prediction(folder, stem, merged=False):
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.ones((20,20), np.uint16)
    if not merged: ids[:,10:] = 2
    markers = ids.astype(np.int32)
    if not merged: markers[:,10] = 0
    boundary = np.zeros(ids.shape, np.float32); boundary[:,10] = .8
    classes = {str(int(i)):1 for i in np.unique(ids)}
    cv2.imwrite(str(folder/(stem+'_inst.png')), ids)
    write(folder/(stem+'_class.json'), classes)
    np.savez_compressed(folder/(stem+'_maps.npz'), boundary=boundary, markers=markers, watershed=ids.astype(np.int32))
    return analysis.load_prediction(folder, stem)


def metric(prediction):
    return dict(instances=len(prediction['classes']), markers=int(prediction['markers'].max()),
        original_matched_F_count=1, original_matched_F_missing=[], original_matched_F_mean_IoU_missing_as_zero=.6,
        all_F_count=1, all_F_mean_IoU_missing_as_zero=.6, F_predicted_P=0, P_predicted_F=0, F_unassigned=0,
        phases=phase_description(prediction['instances'], prediction['classes']),
        fixed_area=dict(original_fixed_count=0, rows=[], comparable_count=0, failure_counts={}),
        error_changes=dict(conservative_split=dict(added=[], removed=[])),
        full_ferrite=dict(gt_rows=[dict(gt=7, matched_pred=1, matched_iou=.6)]))


def case_observation(prediction, case, mode):
    gt = np.ones((20,20), np.int32)*7
    labels = prediction['markers']
    maps = {'high_mask':dict(labels=labels)}
    if mode == 'reflect': maps['reflected_skeleton'] = dict(labels=labels)
    elif mode == 'baseline': maps['skeleton_only'] = dict(labels=labels)
    maps['marker_belt'] = dict(labels=labels)
    trace = trace_case_stages(maps, labels, prediction['instances'], gt, case, watershed=prediction['watershed'])
    gt_rows = []
    for row in trace['stages'][-1]['gt_rows']:
        g = dict(row); g['dominant_pred'] = g.pop('dominant_label'); g['significant_pred_ids'] = g.pop('significant_label_ids')
        g['contributions'] = [dict(pred=item['label'], pixels=item['pixels'], deep_pixels=item['deep_pixels']) for item in g['contributions']]
        gt_rows.append(g)
    relation = endpoint_relation(prediction, case)
    quality = quality_summary(dict(gt_rows=gt_rows, endpoint_relation=relation, whole_object_censored=False), relation, 'positive')
    return dict(case_id=case['case_id'], direction='positive', stage_trace=trace,
                stage_classification=classify_stage_failure(trace), full_known_quality=quality)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    monkeypatch.setattr(analysis, 'EXPECTED_COHORT', dict(original_matched_F=7, original_safe_area=0))
    run = tmp_path/'outputs/marker'; raw_root = tmp_path/'outputs/raw'
    code = tmp_path/'fictional.py'; code.write_text('# fictional protected code', encoding='utf8')
    protected = {}; raw_sources = []; rows = []
    identity = {key:dict(fictional=True) if key in ('config','precision','runtime') else key for key in analysis.IDENTITY_KEYS}
    for index in range(7):
        name = 'fictional_'+str(index); folder = run/name
        source_file = tmp_path/('data/'+name+'.png'); source_file.parent.mkdir(exist_ok=True); source_file.write_bytes(b'fictional source')
        case = dict(case_id=name+'_positive', direction='positive', gt_ids=[7], points=[[10,4],[10,15]], censored=False, phase_kind='F')
        metadata = dict(source=name+'.png', source_path='data/'+name+'.png', gt_path='gt/'+name+'.npz', seed=77+index, cases=[case])
        values = {mode:prediction(folder, mode, merged=mode=='nothin') for mode in ('baseline','reflect','nothin')}
        prediction(raw_root/name, 'baseline')
        baseline = metric(values['baseline']); hashes = {str(source_file):analysis.sha(source_file)}
        raw_sources.append(dict(source=metadata, baseline_exact=True, input_hashes=hashes, baseline=baseline))
        row = dict(source=metadata, baseline_exact=True, input_hashes=hashes, baseline_metrics=baseline,
                   baseline_cases=[case_observation(values['baseline'],case,'baseline')], variants={})
        for mode in ('reflect','nothin'):
            item = dict(real_barrier_elevation_exact=True, baseline_relative=metrics_summary(baseline, metric(values[mode])),
                actual_map_changes_vs_r1=map_changes(values['baseline'],values[mode]),
                actual_map_changes_vs_reflect=map_changes(values['reflect'],values[mode]),
                cases=[case_observation(values[mode],case,mode)])
            if mode == 'reflect':
                refs = tmp_path/'outputs/refs'/name; refs.mkdir(parents=True)
                receipt = dict(actual_final_exact=True)
                for key,suffix in [('png','_inst.png'),('classes','_class.json')]:
                    path = refs/('reflect'+suffix); shutil.copyfile(folder/('reflect'+suffix),path)
                    receipt[key] = str(path.relative_to(tmp_path)); receipt[key+'_sha256'] = analysis.sha(path)
                    protected[str(path)] = analysis.sha(path)
                item['existing_output_exact'] = receipt
            else: item['reflect_relative'] = metrics_summary(metric(values['reflect']),metric(values[mode]))
            row['variants'][mode] = item
        rows.append(row)
    raw_report = dict(format='affinity_bottleneck_probe_v1',complete=True,frozen_states_exact=True,
        no_optimizer_steps=True,training=False,max_rounds=8,sources=raw_sources,**identity)
    write(raw_root/'report.json',raw_report)
    receipt = dict(path=str(raw_root/'report.json'),sha256=analysis.sha(raw_root/'report.json'))
    protected[receipt['path']] = receipt['sha256']
    report = dict(format='marker_stage_probe_v1',complete=True,protected_inputs_exact=True,protected_code_exact=True,
        training=False,no_gpu_forward=True,no_optimizer_steps=True,identity=identity,input_runs=[receipt],
        protected_inputs=protected,protected_code={str(code):analysis.sha(code)},expected_sources=[analysis.source_name(row['source']) for row in rows],sources=rows)
    write(run/'report.json',report)
    return dict(run=run,raw=raw_root,report=report,output=run/'summary.json')


def test_complete_private_seven_source_summary_and_unique_new_gain(bundle):
    result = analysis.analyze(bundle['run'],bundle['output'])
    assert result['complete'] and not result['train_go'] and not result['package_go']
    counts = result['aggregate']
    assert counts['original_fixed_F_count'] == 7
    assert counts['by_direction']['positive']['nothin_newly_wholeknown_met'] == 7
    assert counts['by_variant']['reflect']['original_fixed_missing_count'] == 0
    assert counts['by_variant']['nothin']['F_pixels_delta_vs_reference'] == 0
    assert counts['by_variant']['nothin']['reference_fixed_F_count'] == 7
    relative = result['source_results'][0]['variants']['nothin']['reflect_relative']
    assert 'original_matched_F_mean_IoU_missing_as_zero' not in relative['scalar_deltas']
    assert 'saved_noncomparable_original_match_fields' in relative
    assert all(row['baseline_raw_four_maps_and_classes_exact'] for row in result['source_results'])


def test_cross_run_class_difference_is_explicitly_rejected(bundle):
    path = bundle['raw']/'fictional_0/baseline_class.json'
    write(path, {'1':0,'2':1})
    with pytest.raises(ValueError,match='four maps or classes'):
        analysis.analyze(bundle['run'],bundle['output'])


@pytest.mark.parametrize('flag',['complete','protected_inputs_exact','protected_code_exact'])
def test_incomplete_or_unprotected_run_rejected(bundle,flag):
    bundle['report'][flag] = False; write(bundle['run']/'report.json',bundle['report'])
    with pytest.raises(ValueError,match='Complete protected'):
        analysis.analyze(bundle['run'],bundle['output'])


def test_reflect_reference_hash_and_duplicate_source_are_rejected(bundle):
    reference = bundle['report']['sources'][0]['variants']['reflect']['existing_output_exact']['classes']
    write(analysis.resolve(reference),{'1':0,'2':1})
    with pytest.raises(ValueError,match='Protected input changed'):
        analysis.analyze(bundle['run'],bundle['output'])


def test_transient_reopen_is_not_a_persistent_final_failure():
    gt = np.ones((20,20),np.int32)*7; gt[:,10:] = 9
    split = np.ones(gt.shape,np.int32); split[:,10:] = 2; split[:,10] = 0
    merged = np.ones(gt.shape,np.int32)
    case = dict(direction='negative',gt_ids=[7,9],points=[[10,4],[10,15]])
    trace = trace_case_stages({'bridged':dict(labels=split),'reflected_skeleton':dict(labels=merged),
                             'dilated':dict(labels=split)},split,np.where(gt==7,1,2).astype(np.uint16),gt,case,
                             watershed=np.where(gt==7,1,2).astype(np.int32))
    outcomes = analysis.stage_outcomes(trace,classify_stage_failure(trace))
    assert outcomes['full']['had_reopening']
    assert outcomes['full']['transient_reopening'] and not outcomes['full']['persistent_reopening']


def test_zero_early_stage_does_not_assign_a_split_cause():
    gt = np.ones((20,20),np.int32)*7
    case = dict(direction='positive',gt_ids=[7],points=[[10,4],[10,15]])
    early = np.ones(gt.shape,np.int32); early[10,4] = 0
    split = np.ones(gt.shape,np.int32); split[:,10:] = 2
    trace = trace_case_stages({'early':dict(labels=early),'later':dict(labels=split)},split,split.astype(np.uint16),gt,case,
                             watershed=split)
    outcomes = analysis.stage_outcomes(trace,classify_stage_failure(trace))
    assert outcomes['full']['attribution_unknown_before_first_positive_relation']
    assert outcomes['full']['first_observable_separated_stage'] == 'later'
    assert outcomes['full']['first_confirmed_split_transition'] is None


def test_actual_reference_matching_fixes_added_match_and_retains_missing_as_zero():
    reference = [dict(gt=10,matched_pred=1,matched_iou=.8),
                 dict(gt=20,matched_pred=2,matched_iou=.6),
                 dict(gt=30,matched_pred=None,matched_iou=None)]
    candidate = [dict(gt=10,matched_pred=1,matched_iou=.7),
                 dict(gt=20,matched_pred=None,matched_iou=None),
                 dict(gt=30,matched_pred=3,matched_iou=.9)]
    result = analysis.fixed_matching(reference,candidate)
    assert result['fixed_gt_ids'] == [10,20] and result['missing_gt_ids'] == [20]
    assert result['reference_mean_iou'] == pytest.approx(.7)
    assert result['candidate_mean_iou_missing_as_zero'] == pytest.approx(.35)
    assert result['fixed_mean_iou_delta'] == pytest.approx(-.35)
    assert result['rows'][1]['candidate_iou_missing_as_zero'] == 0


def test_fixed_area_retained_matches_do_not_hide_censoring_or_area_changes():
    rows = [dict(gt=10,r1_relative_error=.01,candidate_relative_error=.02,
                 candidate_matched_pred=1,candidate_comparable=True,excluded_reasons=[]),
            dict(gt=20,r1_relative_error=.01,candidate_relative_error=.03,
                 candidate_matched_pred=2,candidate_comparable=True,excluded_reasons=[]),
            dict(gt=30,r1_relative_error=.01,candidate_relative_error=.04,
                 candidate_matched_pred=3,candidate_comparable=False,excluded_reasons=['unknown_or_frame_censored'])]
    result = analysis.area_status(dict(rows=rows,original_fixed_count=3,comparable_count=2,
                                      failure_counts={'unknown_or_frame_censored':1}))
    assert result['matched_missing_count'] == 0
    assert result['comparable_count'] == 2 and result['incomparable_count'] == 1
    assert result['comparable_area_changed_count'] == 2


def test_same_parent_and_conservative_split_overlap_is_one_F_error():
    changes = dict(safe_same_parent=dict(added=[71],removed=[]),
                   conservative_split=dict(added=[71,72,73,74,75,76],removed=[]))
    result = analysis.object_error_accounting(changes,{71:1,72:0,73:0,74:0,75:0,76:0})
    assert result['added_unique_single_GT_count'] == 6
    assert result['added_unique_single_GT_counts_by_phase'] == {'F':1,'P':5,'unknown':0}
    assert result['overlapping_same_parent_and_conservative_split_gt_ids'] == [71]
    assert result['added_unique_single_GT_errors'][0]['diagnostic_categories'] == ['safe_same_parent','conservative_split']
