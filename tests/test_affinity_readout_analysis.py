# -*- coding: utf-8 -*-
"""虚构两源CPU汇总：身份、实际输出、case匹配、排除与独立计数。"""
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest

from tools import analyze_affinity_readout_path as analysis
from tools.analyze_affinity_bottleneck import endpoint_relation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def prediction(folder, prefix, split=False):
    folder.mkdir(parents=True, exist_ok=True)
    ids = np.ones((8, 8), np.uint16)
    if split:
        ids[:, 4:] = 2
    markers = ids.astype(np.int32)
    boundary = np.zeros(ids.shape, np.float32)
    if split:
        boundary[:, 4] = .8
    assert cv2.imwrite(str(folder / (prefix + '_inst.png')), ids)
    write_json(folder / (prefix + '_class.json'), {str(int(i)): 1 for i in np.unique(ids)})
    np.savez_compressed(folder / (prefix + '_maps.npz'), boundary=boundary,
                        markers=markers, watershed=markers.copy())
    return dict(instances=ids, markers=markers)


def metric():
    return dict(instances=1, markers=1, original_matched_F_count=2,
        original_matched_F_missing=[], original_matched_F_mean_IoU_missing_as_zero=.6,
        all_F_count=2, all_F_mean_IoU_missing_as_zero=.6, F_predicted_P=0,
        P_predicted_F=0, F_unassigned=0, phases={}, fixed_area={'failure_counts': {}},
        error_changes={'conservative_split': {'added': [], 'removed': []}},
        full_ferrite={'gt_rows': [{'gt': 7, 'matched_pred': 1, 'matched_iou': .6,
                                   'contributions': []}]})


def quality(value, case):
    relation = endpoint_relation(value, case)
    rows = []
    for index, gt in enumerate(case['gt_ids']):
        dominant = int(value['instances'][tuple(case['points'][min(index, 1)])])
        rows.append(dict(gt=gt, known_pixels=32, deep_pixels=10,
            dominant_pred=dominant, dominant_fraction=1., dominant_deep_fraction=1.,
            significant_pred_ids=[dominant], contributions=[dict(pred=dominant, pixels=32, deep_pixels=10)]))
    return dict(endpoint_relation=relation, gt_rows=rows, whole_object_censored=False)


def source_report(root, name, include_positive, split=False):
    negative = dict(case_id=name + '_negative', direction='negative', gt_ids=[7, 12],
                    points=[[3, 1], [3, 6]], censored=False, phase_kind='FF')
    positive = dict(case_id=name + '_positive', direction='positive', gt_ids=[19],
                    points=[[1, 1], [2, 2]], censored=True, phase_kind='F')
    cases = [negative] + ([positive] if include_positive else [])
    folder = root / name
    base = prediction(folder, 'baseline')
    report = dict(source=dict(source_path='data/' + name + '.png', gt_path='gt/' + name + '.npz',
        seed=77, cases=cases), baseline_exact=True,
        input_hashes={'image': name + '_image_hash', 'gt': name + '_gt_hash'}, baseline=metric(), arms={})
    for arm in ['negative'] + (['positive'] if include_positive else []) + ['combined']:
        selected = [c for c in cases if arm == 'combined' or arm == c['direction']]
        final = prediction(folder, arm + '_final', split=split and arm != 'positive')
        report['arms'][arm] = dict(cases=deepcopy(selected), rounds=[{'selections': []}],
            selected_negative=1 if arm != 'positive' and split else 0,
            selected_positive=0, final_metrics=metric(),
            baseline_case_quality=[quality(base, c) for c in selected],
            final_case_quality=[quality(final, c) for c in selected])
    return report


def run_report(fmt, sources, rounds):
    return dict(format=fmt, complete=True, frozen_states_exact=True, no_optimizer_steps=True,
        training=False, max_rounds=rounds, sources=sources, checkpoint_sha256='checkpoint',
        model_state_sha256='model', restorer_state_sha256='restorer',
        restorer_checkpoint_sha256='restore_checkpoint', config={'contract': 'same'},
        precision={'fp32': True}, runtime={'cuda': 'fictional'})


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    focused = tmp_path / 'outputs/focused'
    raw_a = tmp_path / 'outputs/raw_a'
    raw_b = tmp_path / 'outputs/raw_b'
    raw = [source_report(raw_a, 'fictional_a', True), source_report(raw_b, 'fictional_b', True)]
    focus = [source_report(focused, 'fictional_a', True, split=True),
             source_report(focused, 'fictional_b', False, split=True)]
    f_report = run_report('affinity_readout_path_probe_v1', focus, 4)
    raw_reports = [run_report('affinity_bottleneck_probe_v1', [s], 8) for s in raw]
    write_json(focused / 'report.json', f_report)
    for path, report in zip((raw_a, raw_b), raw_reports):
        write_json(path / 'report.json', report)
    return dict(focused=focused, raw=[raw_a, raw_b], f_report=f_report, raw_reports=raw_reports,
                output=tmp_path / 'outputs/summary/report.json')


def test_completed_two_sources_retain_case_exclusion_duplicate_and_budget_limits(bundle):
    result = analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'])
    assert result['complete'] and result['focused_source_count'] == 2
    assert not result['train_go'] and result['manual_review_required']
    assert result['unique_primary_case_count'] == 3
    assert result['combined_cases_not_independent']
    row = result['source_results'][1]
    assert row['omitted_raw_cases'][0]['direction'] == 'positive'
    assert not row['omitted_raw_cases'][0]['included_in_evidence_count']
    repeat = row['negative_combined_repetition']
    assert repeat['exact_duplicate_output'] and not repeat['count_as_independent_sample']
    arm = row['arms']['negative']
    assert arm['actual_final_raw_to_focused_changes']['instances']['different']
    assert arm['budgets']['raw_max_rounds'] == 8 and arm['budgets']['focused_max_rounds'] == 4
    assert not arm['budgets']['direct_ranking_supported']
    assert 'fixed_area' in arm['raw_to_focused_full_GT_metrics']
    assert 'final_full_ferrite_GT_contributions' in arm['raw_to_focused_full_GT_metrics']


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'model_state_sha256',
    'restorer_state_sha256', 'restorer_checkpoint_sha256', 'runtime', 'precision', 'config'])
def test_frozen_identity_difference_is_rejected(bundle, field):
    report = bundle['raw_reports'][0]
    report[field] = 'changed'
    write_json(bundle['raw'][0] / 'report.json', report)
    with pytest.raises(ValueError, match='identity differs'):
        analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'])


@pytest.mark.parametrize('field', ['complete', 'frozen_states_exact', 'no_optimizer_steps'])
def test_incomplete_or_unfrozen_run_is_rejected(bundle, field):
    bundle['f_report'][field] = False
    write_json(bundle['focused'] / 'report.json', bundle['f_report'])
    with pytest.raises(ValueError, match='complete frozen'):
        analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'])


@pytest.mark.parametrize('change', ['seed', 'hash', 'case', 'baseline'])
def test_source_seed_hash_case_and_actual_baseline_are_verified(bundle, change):
    report = bundle['raw_reports'][0]
    source = report['sources'][0]
    if change == 'seed':
        source['source']['seed'] += 1
    elif change == 'hash':
        source['input_hashes']['gt'] = 'other_gt'
    elif change == 'case':
        source['source']['cases'][0]['points'][0] = [4, 1]
    else:
        prediction(bundle['raw'][0] / 'fictional_a', 'baseline', split=True)
    write_json(bundle['raw'][0] / 'report.json', report)
    with pytest.raises(ValueError):
        analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'])


def test_wrong_format_and_single_source_do_not_create_a_training_gate(bundle):
    bundle['f_report']['format'] = 'affinity_bottleneck_probe_v1'
    write_json(bundle['focused'] / 'report.json', bundle['f_report'])
    with pytest.raises(ValueError, match='format'):
        analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'])
    bundle['f_report']['format'] = 'affinity_readout_path_probe_v1'
    bundle['f_report']['sources'] = bundle['f_report']['sources'][:1]
    write_json(bundle['focused'] / 'report.json', bundle['f_report'])
    with pytest.raises(ValueError, match='two focused'):
        analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'])


def test_verified_private_plan_preserves_prior_exclusion_rationale(bundle):
    plan = bundle['focused'].parent / 'plan.json'
    payload = dict(complete=True, excluded='Prior structural ambiguity; not an output-based selection',
                   source_selection='fixed fictional pair',
                   sources=[s['source'] for s in bundle['f_report']['sources']])
    write_json(plan, payload)
    bundle['f_report']['plan_sha256'] = analysis.sha(plan)
    write_json(bundle['focused'] / 'report.json', bundle['f_report'])
    result = analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'], plan=plan)
    assert result['inputs']['private_plan']['verified']
    assert result['inputs']['private_plan']['exclusions'] == payload['excluded']


def test_missing_quality_keeps_summary_incomplete_and_train_go_false(bundle):
    source = bundle['f_report']['sources'][0]
    source['arms']['positive']['final_case_quality'] = []
    write_json(bundle['focused'] / 'report.json', bundle['f_report'])
    result = analysis.analyze(bundle['focused'], bundle['raw'], bundle['output'])
    assert not result['complete'] and not result['quality_evidence_complete']
    assert not result['train_go']
