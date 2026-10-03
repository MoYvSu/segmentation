# -*- coding: utf-8 -*-
"""只读验收七源 marker 阶段诊断；不训练、打包或自动切换部署。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_affinity_bottleneck import (endpoint_relation, load_prediction,
    map_changes, read, sha, source_name)
from tools.analyze_affinity_readout_path import (CASE_KEYS, IDENTITY_KEYS, SOURCE_KEYS,
    all_exact, index_cases, index_sources, load_run)
from utils.marker_stage_trace import classify_stage_failure
from utils.patch_diagnostic import phase_description

EXPECTED_COHORT = dict(original_matched_F=688, original_safe_area=135)
DOMAINS = ('full', 'known_all', 'known_target_domain')


def resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def verify_hashes(files, *, code=False):
    if not isinstance(files, dict) or not files:
        raise ValueError('Protected file hashes are required')
    for path, digest in files.items():
        actual = resolve(path)
        if not isinstance(digest, str) or sha(actual) != digest:
            raise ValueError('Protected ' + ('code' if code else 'input') + ' changed: ' + str(actual))


def raw_sources(report):
    indexed = {}; receipts = []
    for receipt in report.get('input_runs', []):
        if not isinstance(receipt, dict) or sha(resolve(receipt['path'])) != receipt.get('sha256'):
            raise ValueError('Raw input-run receipt hash differs')
        root, raw, actual_receipt = load_run(resolve(receipt['path']), 'affinity_bottleneck_probe_v1')
        if any(raw.get(key) != report['identity'].get(key) for key in IDENTITY_KEYS):
            raise ValueError('Raw frozen identity differs from marker diagnostic')
        receipts.append(actual_receipt)
        for name, source in index_sources(raw).items():
            if name in indexed:
                raise ValueError('Duplicate raw source identity')
            indexed[name] = (root, source)
    if len(indexed) != 7:
        raise ValueError('Exactly seven unique protected raw sources required')
    return indexed, receipts


def actual_contract(prediction):
    boundary = prediction['boundary']
    if not np.issubdtype(boundary.dtype, np.floating) or np.any((boundary < 0) | (boundary > 1)):
        raise ValueError('Native boundary must be floating probabilities in [0,1]')
    for key in ('markers', 'watershed'):
        value = prediction[key]
        if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
            raise ValueError('Actual marker/WS must be nonnegative full-shape integer labels')


def validate_reflection(item, prediction, shape, protected):
    receipt = item.get('existing_output_exact')
    if not isinstance(receipt, dict) or receipt.get('actual_final_exact') is not True:
        raise ValueError('Actual existing reflect output receipt required')
    paths = {}
    for name, hash_key in [('png', 'png_sha256'), ('classes', 'classes_sha256')]:
        path = resolve(receipt[name]); digest = receipt[hash_key]
        if protected.get(str(path)) != digest or sha(path) != digest:
            raise ValueError('Reflect reference hash lacks exact protected identity')
        paths[name] = str(path)
    ids = cv2.imread(paths['png'], cv2.IMREAD_UNCHANGED)
    if ids is None or ids.dtype != np.uint16 or ids.ndim != 2 or ids.shape != shape:
        raise ValueError('Invalid existing reflect uint16/native shape')
    if not np.array_equal(ids, prediction['instances']) or read(paths['classes']) != prediction['classes']:
        raise ValueError('Saved reflect output differs from protected existing output')
    return dict(**receipt, independent_png_classes_exact=True)


def compact_quality(item):
    quality = item['full_known_quality']
    if quality.get('available') is not True or type(quality.get('whole_goal')) is not bool:
        raise ValueError('Complete whole-known quality is required')
    return quality


def stage_outcomes(trace, classification):
    """曾重开不等于终态粘连；过滤种子或最终端点为0时不能作持续证据。"""
    rows = trace['stages']; by_name = {row['name']: row for row in rows}; result = {}
    for domain in DOMAINS:
        def relation(row): return row if domain == 'full' else row[domain]
        marker = relation(by_name['marker_labels']); final = relation(by_name['instances'])
        available = all(marker['endpoints_positive']) and all(final['endpoints_positive'])
        had = classification['reopened_'+domain]
        persistent = had and marker['connected'] is True and final['connected'] is True if available else None
        observable = [(index, row) for index,row in enumerate(rows)
                      if all(relation(row)['endpoints_positive']) and relation(row)['connected'] is not None]
        first = observable[0] if observable else None
        separated = next((row['name'] for _,row in observable if relation(row)['connected'] is False), None)
        transitions = classification[domain+'_transitions']
        first_split = next((item for item in transitions if item['transition'] == 'split'), None)
        result[domain] = dict(had_reopening=had, final_relation_available=available,
            persistent_reopening=persistent,
            transient_reopening=had and not persistent if available else None,
            final_positive_filtered_marker_connected=marker['connected'] if available else None,
            final_positive_instance_connected=final['connected'] if available else None,
            first_observable_positive_relation_stage=first[1]['name'] if first else None,
            attribution_unknown_before_first_positive_relation=first is None or first[0] > 0,
            first_observable_separated_stage=separated,
            first_confirmed_split_transition=first_split,
            caveat='Observed stage relations only; transient crossing or first separation is not proof of a physical GT error')
    return result


def case_trace(item, case, prediction, mode):
    trace = item['stage_trace']; rows = trace.get('stages', [])
    order = trace.get('stage_order')
    if (trace.get('format') != 'marker_stage_trace_v1' or order != [row['name'] for row in rows]
            or len(order) != len(set(order)) or order[-3:] != ['marker_labels', 'watershed', 'instances']
            or trace.get('direction') != case['direction'] or trace.get('gt_ids') != case['gt_ids']
            or trace.get('points') != case['points'] or trace.get('shape') != list(prediction['instances'].shape)):
        raise ValueError('Private native case/stage-order contract differs')
    if ('reflected_skeleton' in order) != (mode == 'reflect'):
        raise ValueError('Reflected skeleton must belong only to actual reflect candidate')
    if mode == 'nothin' and any('skeleton' in name for name in order):
        raise ValueError('nothin may not contain a recomputed thinning stage')
    classification = classify_stage_failure(trace)
    if classification != item.get('stage_classification'):
        raise ValueError('Saved stage failure classification differs from trace')
    by_name = {row['name']: row for row in rows}
    for name, key in [('marker_labels', 'markers'), ('watershed', 'watershed'), ('instances', 'instances')]:
        values = [int(prediction[key][tuple(point)]) for point in case['points']]
        if by_name[name]['endpoint_labels'] != values:
            raise ValueError('Saved stage endpoint labels differ from actual maps')
    quality = compact_quality(item)
    relation = endpoint_relation(prediction, case)
    if any(quality['endpoint'].get(key) != value for key, value in relation.items()):
        raise ValueError('Whole-known quality differs from actual final endpoint relation')
    final_rows = by_name['instances']['gt_rows']
    contribution_rows = quality.get('gt_contributions', [])
    if [row['gt'] for row in final_rows] != case['gt_ids'] or len(final_rows) != len(contribution_rows):
        raise ValueError('Whole-known GT contribution cohort differs')
    for a, b in zip(final_rows, contribution_rows):
        pairs = [('gt', 'gt'), ('known_pixels', 'known_pixels'), ('deep_pixels', 'deep_pixels'),
                 ('dominant_label', 'dominant_pred'), ('dominant_fraction', 'dominant_fraction'),
                 ('dominant_deep_fraction', 'dominant_deep_fraction'),
                 ('significant_label_ids', 'significant_pred_ids')]
        if any(a.get(left) != b.get(right) for left, right in pairs):
            raise ValueError('Final trace and complete GT quality disagree')
        converted = [dict(pred=part['label'], pixels=part['pixels'], deep_pixels=part['deep_pixels'])
                     for part in a['contributions']]
        if converted != b.get('contributions'):
            raise ValueError('Tiny/zero/full GT contributions disagree')
    keep = ('name', 'kind', 'endpoint_labels', 'endpoints_positive', 'connected', 'endpoint_relation_goal',
            'zero_pixels', 'zero_pixels_known_target_domain', 'known_all', 'known_target_domain',
            'dominant_labels_separate', 'all_dominant_deep_fraction_at_least_95')
    compact = []
    for row in rows:
        selected = {key: row[key] for key in keep}
        selected['gt_rows'] = [{key: value for key, value in g.items() if key != 'contributions'}
                               for g in row['gt_rows']]
        compact.append(selected)
    return dict(whole_known_quality=quality, stage_order=order, stage_rows=compact,
                stage_classification=classification,
                stage_outcomes=stage_outcomes(trace, classification),
                all_small_stage_contributions_preserved_in_private_run=True)


def index_observations(items, cases):
    result = {}
    for item in items:
        identity = item.get('case_id')
        if identity in result or identity not in cases or item.get('direction') != cases[identity]['direction']:
            raise ValueError('Case observation identity/direction differs')
        result[identity] = item
    if set(result) != set(cases):
        raise ValueError('Every predefined case must be retained')
    return result


def compact_metrics(value, *, reference_cohort_switch=False):
    result = {key: value[key] for key in ('baseline', 'final', 'scalar_deltas', 'fixed_area', 'error_changes')}
    if reference_cohort_switch:
        fields = ('original_matched_F_count', 'original_matched_F_missing',
                  'original_matched_F_mean_IoU_missing_as_zero')
        result['saved_noncomparable_original_match_fields'] = {
            side:{key:value[side].get(key) for key in fields} for side in ('baseline','final')}
        for side in ('baseline','final'):
            result[side] = {key:item for key,item in value[side].items() if key not in fields}
        result['scalar_deltas'] = {key:item for key,item in value['scalar_deltas'].items() if key not in fields}
        result['cohort_caveat'] = 'Saved reflect metrics carry R1 matched IDs; candidate measurement uses actual reflect matched IDs. Original-match scalar deltas are removed; use independently fixed reference_matching.'
    return result


def fixed_matching(before_rows, after_rows):
    """由参照实际正 matched_pred 固定GT名单，候选失配以0保留分母。"""
    before = {row['gt']:row for row in before_rows}; after = {row['gt']:row for row in after_rows}
    if len(before) != len(before_rows) or len(after) != len(after_rows) or set(before) != set(after):
        raise ValueError('Complete fixed matching GT cohort differs')
    cohort = [row['gt'] for row in before_rows if row.get('matched_pred') is not None and row['matched_pred'] > 0]
    rows = []
    for gid in cohort:
        old, new = before[gid], after[gid]
        missing = new.get('matched_pred') is None or new['matched_pred'] <= 0
        a, b = old['matched_iou'], 0. if missing else new['matched_iou']
        if any(type(value) not in (int,float) or not np.isfinite(value) or not 0 <= value <= 1 for value in (a,b)):
            raise ValueError('Fixed matching IoU must be finite in [0,1]')
        rows.append(dict(gt=gid, reference_matched_pred=old['matched_pred'],
            candidate_matched_pred=new.get('matched_pred'), reference_iou=float(a),
            candidate_iou_missing_as_zero=float(b), candidate_missing=missing))
    old_sum = sum(row['reference_iou'] for row in rows)
    new_sum = sum(row['candidate_iou_missing_as_zero'] for row in rows)
    return dict(fixed_count=len(cohort), fixed_gt_ids=cohort, rows=rows,
        missing_gt_ids=[row['gt'] for row in rows if row['candidate_missing']],
        reference_iou_sum=old_sum, candidate_iou_sum_missing_as_zero=new_sum,
        reference_mean_iou=old_sum/max(1,len(cohort)),
        candidate_mean_iou_missing_as_zero=new_sum/max(1,len(cohort)),
        fixed_mean_iou_delta=(new_sum-old_sum)/max(1,len(cohort)),
        denominator_policy='Actual reference matched GT IDs fixed; candidate missing is zero, never dropped')


def area_status(fixed):
    rows = fixed['rows']; comparable = [row for row in rows if row['candidate_comparable']]
    failures = Counter(reason for row in rows for reason in row['excluded_reasons'])
    if len(rows) != fixed['original_fixed_count'] or len(comparable) != fixed['comparable_count']:
        raise ValueError('Fixed area cohort/comparable count differs')
    if any(failures[key] != count for key,count in fixed['failure_counts'].items()):
        raise ValueError('Fixed area failure counters disagree with retained rows')
    for row in comparable:
        if any(type(row[key]) not in (int,float) or not np.isfinite(row[key])
               for key in ('r1_relative_error','candidate_relative_error')):
            raise ValueError('Comparable area item has no finite original/candidate error')
    return dict(original_fixed_count=len(rows), comparable_count=len(comparable),
        incomparable_count=len(rows)-len(comparable),
        incomparable_gt_ids=[row['gt'] for row in rows if not row['candidate_comparable']],
        matched_missing_count=sum(row['candidate_matched_pred'] is None for row in rows),
        comparable_area_changed_count=sum(row['candidate_relative_error'] != row['r1_relative_error'] for row in comparable),
        comparable_area_exact_count=sum(row['candidate_relative_error'] == row['r1_relative_error'] for row in comparable),
        failure_counts=dict(failures), failure_reasons_can_overlap=True,
        caveat='Matched IDs retained does not imply area safety: censored/split/merged items remain in total cohort. Area changes use only comparable subset and are not score gains.')


def object_error_accounting(changes, gt_classes):
    """单GT诊断先按GT去重；不同错误分类可能描述同一次内部误切。"""
    singles = {}; category_counts = {}
    for category in ('safe_same_parent','conservative_split'):
        ids = changes.get(category, {}).get('added', [])
        counts = Counter()
        for gid in ids:
            if type(gid) is not int or gid <= 0:
                raise ValueError('Single-GT error identity must be a positive integer')
            phase = gt_classes.get(gid)
            label = 'F' if phase == 1 else 'P' if phase == 0 else 'unknown'
            counts[label] += 1
            item = singles.setdefault(gid, dict(gt=gid, phase=label, diagnostic_categories=[]))
            item['diagnostic_categories'].append(category)
        category_counts[category] = {phase:counts[phase] for phase in ('F','P','unknown')}
    values = [singles[gid] for gid in sorted(singles)]
    counts = Counter(item['phase'] for item in values)
    return dict(added_unique_single_GT_errors=values, added_unique_single_GT_count=len(values),
        added_unique_single_GT_counts_by_phase={phase:counts[phase] for phase in ('F','P','unknown')},
        category_added_counts_by_phase=category_counts,
        overlapping_same_parent_and_conservative_split_gt_ids=[item['gt'] for item in values if len(item['diagnostic_categories']) > 1],
        pair_errors_reported_separately=True,
        caveat='Category counts overlap; never add safe_same_parent and conservative_split as independent objects')


def gt_phase_lookup(metadata, record):
    result = {row['gt']:1 for row in record['baseline_metrics']['full_ferrite']['gt_rows']}
    path = resolve(metadata['gt_path'])
    path = path.with_name(path.name.removesuffix('_gt.npz')+'_class.json')
    if path.is_file():
        if record['input_hashes'].get(str(path)) != sha(path):
            raise ValueError('GT classes used for error attribution are not protected inputs')
        classes = read(path)
        if any(type(value) is not int or value not in (0,1) for value in classes.values()):
            raise ValueError('GT class convention differs')
        result.update({int(key):value for key,value in classes.items()})
    return result


def compare_cases(record, predictions):
    cases = index_cases(record['source']['cases'])
    observations = {'baseline': index_observations(record['baseline_cases'], cases)}
    observations.update({mode: index_observations(record['variants'][mode]['cases'], cases)
                         for mode in ('reflect', 'nothin')})
    rows = []
    for identity, case in cases.items():
        variants = {mode: case_trace(observations[mode][identity], case, predictions[mode], mode)
                    for mode in ('baseline', 'reflect', 'nothin')}
        qualities = {mode: item['whole_known_quality'] for mode, item in variants.items()}
        baseline, reflect, nothin = [qualities[mode] for mode in ('baseline', 'reflect', 'nothin')]
        ambiguity = case.get('previous_gt_structure_ambiguity') or observations['baseline'][identity].get('previous_gt_structure_ambiguity')
        rows.append(dict(case_id=identity, direction=case['direction'], phase_kind=case.get('phase_kind'),
            gt_ids=case['gt_ids'], censored=case.get('censored', True), previous_gt_structure_ambiguity=ambiguity,
            variants=variants,
            reflect_already_wholeknown_met=reflect['whole_goal'],
            reflect_resolved_negative_with_transient_reopening=case['direction']=='negative'
                and reflect['whole_goal'] and any(variants['reflect']['stage_outcomes'][domain]['transient_reopening'] is True for domain in DOMAINS),
            reflect_newly_wholeknown_met=not baseline['whole_goal'] and reflect['whole_goal'],
            nothin_newly_wholeknown_met=not reflect['whole_goal'] and nothin['whole_goal'],
            nothin_broke_reflect_wholeknown_goal=reflect['whole_goal'] and not nothin['whole_goal'],
            nothin_newly_endpoint_met=not reflect['endpoint']['endpoint_goal'] and nothin['endpoint']['endpoint_goal'],
            significant_piece_deltas_nothin_vs_reflect=[b-a for a,b in zip(reflect['significant_piece_counts'], nothin['significant_piece_counts'])],
            deep_fraction_deltas_nothin_vs_reflect=[b['dominant_deep_fraction']-a['dominant_deep_fraction']
                for a,b in zip(reflect['gt_contributions'], nothin['gt_contributions'])],
            eligible_unambiguous_diagnostic_evidence=not bool(ambiguity),
            caveat='Seen processed known GT; endpoint/stage changes are not physical correctness or official score'))
    return rows


def analyze_source(root, record, raw_root, raw_record, protected):
    name = source_name(record['source']); metadata = record['source']; raw_metadata = raw_record['source']
    if (any(metadata.get(key) != raw_metadata.get(key) for key in SOURCE_KEYS)
            or metadata['cases'] != raw_metadata['cases'] or record.get('baseline_exact') is not True
            or record.get('input_hashes') != raw_record.get('input_hashes')
            or record['baseline_metrics'] != raw_record['baseline']):
        raise ValueError('Actual source/GT/case/seed/baseline identity differs')
    verify_hashes(record['input_hashes'])
    predictions = {mode: load_prediction(root/name, mode) for mode in ('baseline', 'reflect', 'nothin')}
    raw_baseline = load_prediction(raw_root/name, 'baseline')
    for prediction in [raw_baseline, *predictions.values()]: actual_contract(prediction)
    baseline_changes = map_changes(raw_baseline, predictions['baseline'])
    if not all_exact(baseline_changes):
        raise ValueError('Actual baseline four maps or classes differ from protected raw run')
    metrics = record['baseline_metrics']
    if phase_description(predictions['baseline']['instances'], predictions['baseline']['classes']) != metrics['phases']:
        raise ValueError('Baseline actual phase composition differs')
    original_matched = [row['gt'] for row in metrics['full_ferrite']['gt_rows'] if row['matched_pred'] is not None]
    original_gt_rows = metrics['full_ferrite']['gt_rows']
    gt_classes = gt_phase_lookup(metadata, record)
    safe_rows = metrics['fixed_area']['rows']; safe_ids = [row['gt'] for row in safe_rows]
    if (len(original_matched) != metrics['original_matched_F_count'] or len(safe_ids) != len(set(safe_ids))
            or len(safe_ids) != metrics['fixed_area']['original_fixed_count']):
        raise ValueError('Original fixed F/area cohort identities are incomplete')
    variants = {}
    for mode in ('reflect', 'nothin'):
        item = record['variants'][mode]; prediction = predictions[mode]
        change = map_changes(predictions['baseline'], prediction)
        change_reflect = map_changes(predictions['reflect'], prediction)
        if (item.get('real_barrier_elevation_exact') is not True or not change['boundary']['exact']
                or change != item.get('actual_map_changes_vs_r1')
                or change_reflect != item.get('actual_map_changes_vs_reflect')):
            raise ValueError('Actual frozen native maps differ from captured change audit')
        relative = item['baseline_relative']
        if phase_description(prediction['instances'], prediction['classes']) != relative['final']['phases']:
            raise ValueError('Actual candidate F/phase composition differs')
        fixed = relative['fixed_area']
        if ([row['gt'] for row in fixed['rows']] != safe_ids or fixed['original_fixed_count'] != len(safe_ids)
                or relative['final']['original_matched_F_count'] != len(original_matched)):
            raise ValueError('Candidate substituted original fixed cohort')
        final_gt = {row['gt']: row for row in relative['final_full_ferrite_GT_contributions']}
        missing = [gid for gid in original_matched if final_gt[gid]['matched_pred'] is None]
        if missing != relative['final']['original_matched_F_missing']:
            raise ValueError('Saved original fixed match failures differ from actual GT contributions')
        comparable = all(row['candidate_comparable'] and not row['excluded_reasons'] for row in fixed['rows'])
        unchanged = all(row['candidate_relative_error'] == row['r1_relative_error'] for row in fixed['rows'])
        variants[mode] = dict(actual_changes_vs_baseline=change, actual_changes_vs_reflect=change_reflect,
            baseline_relative=compact_metrics(relative), original_fixed_missing=missing,
            original_safe_area_all_comparable=comparable, original_safe_area_errors_exact=unchanged,
            original_fixed_matching=fixed_matching(original_gt_rows, relative['final_full_ferrite_GT_contributions']),
            original_safe_area_status=area_status(fixed),
            actual_phases=relative['final']['phases'])
        if mode == 'reflect':
            variants[mode]['protected_existing_output'] = validate_reflection(item, prediction,
                predictions['baseline']['instances'].shape, protected)
            variants[mode]['reference_matching'] = variants[mode]['original_fixed_matching']
            variants[mode]['reference_area_status'] = variants[mode]['original_safe_area_status']
            variants[mode]['reference_error_object_accounting'] = object_error_accounting(relative['error_changes'], gt_classes)
        else:
            reference_rows = record['variants']['reflect']['baseline_relative']['final_full_ferrite_GT_contributions']
            matching = fixed_matching(reference_rows, relative['final_full_ferrite_GT_contributions'])
            if (item['reflect_relative']['final']['original_matched_F_count'] != matching['fixed_count']
                    or item['reflect_relative']['final']['original_matched_F_missing'] != matching['missing_gt_ids']
                    or not np.isclose(item['reflect_relative']['final']['original_matched_F_mean_IoU_missing_as_zero'],
                                      matching['candidate_mean_iou_missing_as_zero'], rtol=0, atol=1e-12)):
                raise ValueError('Candidate actual reflect-fixed matching differs from complete GT contributions')
            area = item['reflect_relative']['fixed_area']
            reflect_safe = [row['gt'] for row in reference_rows if row.get('safe_one_to_one_area_eligible')]
            if [row['gt'] for row in area['rows']] != reflect_safe:
                raise ValueError('Actual reflect-safe area cohort was substituted')
            variants[mode]['reflect_relative'] = compact_metrics(item['reflect_relative'], reference_cohort_switch=True)
            variants[mode]['reference_matching'] = matching
            variants[mode]['reference_area_status'] = area_status(area)
            variants[mode]['reference_error_object_accounting'] = object_error_accounting(item['reflect_relative']['error_changes'], gt_classes)
            if item['reflect_relative']['baseline'] != record['variants']['reflect']['baseline_relative']['final']:
                # original_matched_F_count may change only when reflect gains/losses full GT matches;
                # exact original match policy is independently retained above.
                fields = ('instances','markers','all_F_count','all_F_mean_IoU_missing_as_zero',
                          'F_predicted_P','P_predicted_F','F_unassigned','phases')
                if any(item['reflect_relative']['baseline'].get(key) != record['variants']['reflect']['baseline_relative']['final'].get(key) for key in fields):
                    raise ValueError('nothin relative baseline is not the actual reflect reference')
    return dict(source=name, baseline_raw_four_maps_and_classes_exact=True,
                original_fixed_F_count=len(original_matched), original_safe_area_count=len(safe_ids),
                baseline_phases=metrics['phases'], variants=variants,
                cases=compare_cases(record, predictions))


def aggregate(rows):
    cases = [case for row in rows for case in row['cases']]
    result = dict(source_count=len(rows), original_fixed_F_count=sum(row['original_fixed_F_count'] for row in rows),
                  original_safe_area_count=sum(row['original_safe_area_count'] for row in rows),
                  cases=len(cases), by_variant={}, by_direction={})
    for mode in ('reflect','nothin'):
        values = [row['variants'][mode] for row in rows]
        relative = 'reflect_relative' if mode == 'nothin' else 'baseline_relative'
        comparison = [value[relative] for value in values]
        matching = [value['reference_matching'] for value in values]
        areas = [value['reference_area_status'] for value in values]
        total_match = sum(value['fixed_count'] for value in matching)
        old_iou = sum(value['reference_iou_sum'] for value in matching)
        new_iou = sum(value['candidate_iou_sum_missing_as_zero'] for value in matching)
        objects = [dict(source=row['source'], **item) for row in rows
                   for item in row['variants'][mode]['reference_error_object_accounting']['added_unique_single_GT_errors']]
        object_phases = Counter(item['phase'] for item in objects)
        failures = Counter(); changes = {}
        for value in comparison:
            failures.update(value['fixed_area']['failure_counts'])
            for kind, change in value['error_changes'].items():
                count = changes.setdefault(kind, dict(added=0, removed=0))
                count['added'] += len(change['added']); count['removed'] += len(change['removed'])
        result['by_variant'][mode] = dict(
            original_fixed_missing_count=sum(len(value['original_fixed_missing']) for value in values),
            original_safe_area_all_comparable=all(value['original_safe_area_all_comparable'] for value in values),
            original_safe_area_errors_exact=all(value['original_safe_area_errors_exact'] for value in values),
            relative_reference='reflect' if mode == 'nothin' else 'baseline',
            reference_fixed_F_count=total_match,
            reference_fixed_missing_count=sum(len(value['missing_gt_ids']) for value in matching),
            reference_fixed_mean_iou=old_iou/max(1,total_match),
            candidate_reference_fixed_mean_iou_missing_as_zero=new_iou/max(1,total_match),
            reference_fixed_mean_iou_delta=(new_iou-old_iou)/max(1,total_match),
            reference_matching_recomputed_from_complete_GT_contributions=True,
            reference_safe_area_count=sum(value['fixed_area']['original_fixed_count'] for value in comparison),
            reference_safe_area_comparable_count=sum(value['comparable_count'] for value in areas),
            reference_safe_area_incomparable_count=sum(value['incomparable_count'] for value in areas),
            reference_safe_area_matched_missing_count=sum(value['matched_missing_count'] for value in areas),
            reference_safe_area_changed_comparable_count=sum(value['comparable_area_changed_count'] for value in areas),
            reference_safe_area_exact_comparable_count=sum(value['comparable_area_exact_count'] for value in areas),
            reference_safe_area_failure_counts=dict(failures), error_changes=changes,
            error_category_counts_overlap=True,
            added_unique_single_GT_error_count=len(objects),
            added_unique_single_GT_error_counts_by_phase={phase:object_phases[phase] for phase in ('F','P','unknown')},
            added_unique_single_GT_errors=objects,
            overlapping_same_parent_conservative_split_count=sum(len(value['reference_error_object_accounting']['overlapping_same_parent_and_conservative_split_gt_ids']) for value in values),
            F_instances=sum(value['actual_phases']['ferrite']['instances'] for value in values),
            F_pixels=sum(value['actual_phases']['ferrite']['pixels'] for value in values),
            F_tiny_under200=sum(value['actual_phases']['ferrite']['tiny_under200'] for value in values),
            F_pixels_delta_vs_reference=sum(value['final']['phases']['ferrite']['pixels']-value['baseline']['phases']['ferrite']['pixels'] for value in comparison),
            map_changed_sources={key:sum(value['actual_changes_vs_reflect'][key]['different'] for value in values)
                                 for key in ('boundary','markers','watershed','instances')})
    for direction in ('negative','positive'):
        selected = [case for case in cases if case['direction'] == direction]
        result['by_direction'][direction] = dict(cases=len(selected),
            reflect_already_wholeknown_met=sum(case['reflect_already_wholeknown_met'] for case in selected),
            reflect_newly_wholeknown_met=sum(case['reflect_newly_wholeknown_met'] for case in selected),
            nothin_newly_wholeknown_met=sum(case['nothin_newly_wholeknown_met'] for case in selected),
            nothin_newly_unambiguous_wholeknown_met=sum(case['nothin_newly_wholeknown_met'] and case['eligible_unambiguous_diagnostic_evidence'] for case in selected),
            nothin_broke_reflect_wholeknown_goal=sum(case['nothin_broke_reflect_wholeknown_goal'] for case in selected),
            GT_structure_ambiguous=sum(not case['eligible_unambiguous_diagnostic_evidence'] for case in selected),
            stage_reopen_case_counts={mode:{domain:sum(case['variants'][mode]['stage_classification']['reopened_'+domain]
                for case in selected) for domain in DOMAINS} for mode in ('baseline','reflect','nothin')},
            transient_reopening_case_counts={mode:{domain:sum(case['variants'][mode]['stage_outcomes'][domain]['transient_reopening'] is True
                for case in selected) for domain in DOMAINS} for mode in ('baseline','reflect','nothin')},
            persistent_reopening_case_counts={mode:{domain:sum(case['variants'][mode]['stage_outcomes'][domain]['persistent_reopening'] is True
                for case in selected) for domain in DOMAINS} for mode in ('baseline','reflect','nothin')},
            early_unobservable_case_counts={mode:{domain:sum(case['variants'][mode]['stage_outcomes'][domain]['attribution_unknown_before_first_positive_relation']
                for case in selected) for domain in DOMAINS} for mode in ('baseline','reflect','nothin')})
    return result


def analyze(run, output):
    path = resolve(run); root = path if path.is_dir() else path.parent
    report_path = root/'report.json' if path.is_dir() else path; report = read(report_path)
    if (report.get('format') != 'marker_stage_probe_v1' or report.get('complete') is not True
            or report.get('protected_inputs_exact') is not True or report.get('protected_code_exact') is not True
            or report.get('training') is not False or report.get('no_gpu_forward') is not True
            or report.get('no_optimizer_steps') is not True):
        raise ValueError('Complete protected CPU-only marker_stage_probe_v1 required')
    verify_hashes(report.get('protected_inputs')); verify_hashes(report.get('protected_code'), code=True)
    indexed, receipts = raw_sources(report)
    sources = report.get('sources', []); names = [source_name(source['source']) for source in sources]
    if len(names) != 7 or len(set(names)) != 7 or set(names) != set(indexed) or names != report.get('expected_sources'):
        raise ValueError('Actual complete seven-source ordered cohort differs')
    protected = {str(resolve(path)):digest for path,digest in report['protected_inputs'].items()}
    rows = [analyze_source(root, source, *indexed[name], protected) for name,source in zip(names,sources)]
    counts = aggregate(rows)
    if (counts['original_fixed_F_count'] != EXPECTED_COHORT['original_matched_F']
            or counts['original_safe_area_count'] != EXPECTED_COHORT['original_safe_area']):
        raise ValueError('Historical fixed 688F/135safe cohort differs; investigate before reporting')
    result = dict(format='marker_stage_analysis_v1', complete=True, cpu_only=True, training=False,
        metric_policy_version='actual_reference_fixed_matching_v2',
        train_go=False, package_go=False, deployment_promoted=False, manual_review_required=True,
        input=dict(path=str(report_path), sha256=sha(report_path), raw_runs=receipts),
        reference_score=report.get('reference_score'), aggregate=counts, source_results=rows,
        conclusions_scope='Seen training GT diagnostics; unknown/frame censor retained; no test GT, official accuracy or automatic model conclusion',
        full_stage_contributions= 'Private run/report.json retains every label including zero and small fragments')
    target = resolve(output)
    if not any(target.is_relative_to(ROOT/folder) for folder in ('output','outputs')):
        raise ValueError('Derived private summary must stay under ignored output(s)')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='outputs/marker_stages')
    parser.add_argument('--output', default='outputs/marker_stages/summary.json')
    args = parser.parse_args(); result = analyze(args.run, args.output)
    print(json.dumps(dict(complete=result['complete'], aggregate=result['aggregate']), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
