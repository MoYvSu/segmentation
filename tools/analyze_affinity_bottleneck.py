# -*- coding: utf-8 -*-
"""CPU汇总已封存关键通路试验；不训练、不自动晋级，不内嵌逐图GT或坐标。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_name(value):
    if isinstance(value, dict):
        value = value.get('source', value.get('source_path'))
    if not isinstance(value, str) or not value:
        raise ValueError('A source identity is required in the private report')
    return Path(value).stem


def load_prediction(folder, prefix):
    folder = Path(folder); png = folder / (prefix + '_inst.png')
    maps = folder / (prefix + '_maps.npz'); labels = folder / (prefix + '_class.json')
    for path in (png, maps, labels):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if ids is None or ids.dtype != np.uint16 or ids.ndim != 2:
        raise ValueError('Invalid final uint16 prediction: ' + str(png))
    classes = read(labels)
    if (set(classes) != {str(int(i)) for i in np.unique(ids) if i > 0}
            or any(type(v) is not int or v not in (0, 1) for v in classes.values())):
        raise ValueError('Final instance/class contract differs: ' + str(labels))
    with np.load(maps, allow_pickle=False) as payload:
        arrays = {key: payload[key].copy() for key in ('boundary', 'markers', 'watershed')}
    if any(value.shape != ids.shape for value in arrays.values()):
        raise ValueError('Native map and final PNG shapes differ')
    if not np.isfinite(arrays['boundary']).all():
        raise ValueError('Nonfinite native boundary')
    return dict(instances=ids, classes=classes, **arrays,
                file_sha256=dict(png=sha(png), maps=sha(maps), classes=sha(labels)))


def map_changes(before, after):
    changes = {}
    for key in ('boundary', 'markers', 'watershed', 'instances'):
        a, b = before[key], after[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Candidate native shape/dtype differs: ' + key)
        exact = bool(np.array_equal(a, b)); entry = dict(exact=exact, different=not exact,
            changed_pixels=int(np.count_nonzero(a != b)), dtype=str(a.dtype), shape=list(a.shape))
        if key == 'boundary':
            delta = b.astype(np.float64) - a.astype(np.float64)
            entry.update(maximum_absolute_delta=float(np.abs(delta).max()),
                         mean_absolute_delta=float(np.abs(delta).mean()), mean_signed_delta=float(delta.mean()))
        changes[key] = entry
    changes['classes'] = dict(exact=before['classes'] == after['classes'])
    changes['caveat'] = 'Marker/instance integer changes include ID renumbering; case/object contributions determine topology improvement.'
    return changes


def endpoint_relation(prediction, case):
    points = case['points']; ids = prediction['instances']; markers = prediction['markers']
    if (len(points) != 2 or any(len(p) != 2 or not (0 <= p[0] < ids.shape[0]
            and 0 <= p[1] < ids.shape[1]) for p in points)):
        raise ValueError('Private fixed endpoint is outside final native prediction')
    values = [int(ids[tuple(p)]) for p in points]
    same = values[0] > 0 and values[0] == values[1]
    goal = (not same and min(values) > 0) if case['direction'] == 'negative' else same
    return dict(instance_ids=values, marker_ids=[int(markers[tuple(p)]) for p in points],
                same_instance=same, endpoint_goal=bool(goal))


def quality_summary(quality, relation, direction):
    if not isinstance(quality, dict) or not quality.get('gt_rows'):
        return dict(available=False, endpoint=relation, whole_goal=None)
    rows = quality['gt_rows']
    if quality.get('endpoint_relation') is not None:
        saved = quality['endpoint_relation']
        if any(saved.get(key) != relation[key] for key in ('instance_ids', 'marker_ids', 'same_instance', 'endpoint_goal')):
            raise ValueError('Saved case quality endpoint differs from actual native PNG/maps')
    deep_valid = all(r.get('deep_pixels', 0) > 0 for r in rows)
    deep95 = deep_valid and all(r.get('dominant_deep_fraction', 0.) >= .95 for r in rows)
    dominant = [r.get('dominant_pred', 0) for r in rows]
    separate = len(dominant) == 2 and min(dominant) > 0 and dominant[0] != dominant[1]
    significant = [len(r.get('significant_pred_ids', [])) for r in rows]
    whole = bool(relation['endpoint_goal'] and deep95 and (separate if direction == 'negative'
                 else len(rows) == 1 and significant[0] <= 1 and dominant[0] > 0))
    return dict(available=True, endpoint=relation, dominant_predictions=dominant,
        dominant_instances_separate=separate, deep_domain_nonempty=deep_valid,
        all_dominant_deep_fraction_at_least_95=deep95, significant_piece_counts=significant,
        whole_goal=whole, whole_object_censored=quality.get('whole_object_censored', True),
        known_domain_only=True, gt_contributions=rows)


def full_contributions(metrics):
    fields = ('gt', 'phase', 'area', 'deep_gt_pixels', 'censored', 'topology_censored',
              'matched_pred', 'matched_iou', 'predicted_phase', 'significant_geometry_split',
              'matched_pred_significant_geometry_merge', 'F_pixels_predicted_P',
              'safe_one_to_one_area_eligible', 'area_relative_error', 'contributions')
    return [{key: row[key] for key in fields if key in row}
            for row in metrics.get('full_ferrite', {}).get('gt_rows', [])]


def metrics_summary(baseline, candidate):
    fields = ('instances', 'markers', 'original_matched_F_count', 'original_matched_F_missing',
        'original_matched_F_mean_IoU_missing_as_zero', 'all_F_count', 'all_F_mean_IoU_missing_as_zero',
        'F_predicted_P', 'P_predicted_F', 'F_unassigned', 'phases')
    before = {key: baseline.get(key) for key in fields}
    after = {key: candidate.get(key) for key in fields}
    delta = {key: after[key] - before[key] for key in fields
             if type(before[key]) in (int, float) and type(after[key]) in (int, float)}
    return dict(baseline=before, final=after, scalar_deltas=delta,
        fixed_area=candidate.get('fixed_area'), error_changes=candidate.get('error_changes'),
        baseline_full_ferrite_GT_contributions=full_contributions(baseline),
        final_full_ferrite_GT_contributions=full_contributions(candidate),
        caveat='Full processed known GT including filled; original matched/cohort failures retained; not official validation.')


def readout_lookup(path, allow_partial):
    if path is None:
        return {}, set(), None
    report = read(path)
    if report.get('complete') is not True and not allow_partial:
        raise ValueError('Incomplete readout-only report requires --allow-partial')
    lookup = {}
    for source in report.get('rows', []):
        for case in source.get('selected_cases', []):
            key = source_name(source['source']), case['case_id']
            if key in lookup:
                raise ValueError('Duplicate readout-only case identity')
            lookup[key] = case.get('classification')
    return lookup, {source_name(s) for s in report.get('sources', [])}, dict(
        path=str(Path(path).resolve()), sha256=sha(path), complete=report.get('complete') is True)


def summarize_source(run_root, report, readout, sampled):
    source = report['source']; name = source_name(source)
    folder = Path(run_root) / name; before = load_prediction(folder, 'baseline')
    cohort = ('sampled6' if name in sampled else 'known1' if source.get('reference_receipt') else
              'sampled6' if not sampled else 'unclassified')
    arms = {}
    for arm, record in report.get('arms', {}).items():
        final = load_prediction(folder, arm + '_final'); difference = map_changes(before, final)
        cases = record.get('cases', []); bq = record.get('baseline_case_quality', [])
        fq = record.get('final_case_quality', []); qualities = []
        for index, case in enumerate(cases):
            baseline_relation = endpoint_relation(before, case); final_relation = endpoint_relation(final, case)
            old = quality_summary(bq[index] if index < len(bq) else None, baseline_relation, case['direction'])
            new = quality_summary(fq[index] if index < len(fq) else None, final_relation, case['direction'])
            classification = readout.get((name, case['case_id']))
            mirror = classification == 'readout_only_resolved'
            reduced = None
            if old['available'] and new['available'] and case['direction'] == 'positive':
                reduced = all(a >= b for a, b in zip(old['significant_piece_counts'], new['significant_piece_counts'])) \
                    and any(a > b for a, b in zip(old['significant_piece_counts'], new['significant_piece_counts']))
            qualities.append(dict(case_id=case['case_id'], direction=case['direction'],
                phase_kind=case.get('phase_kind'), gt_ids=case['gt_ids'], censored=case.get('censored', True),
                baseline=old, final=new, significant_pieces_reduced=reduced,
                mirror_readout_classification=classification, mirror_already_resolved=mirror,
                eligible_weak_response_evidence=not mirror,
                endpoint_newly_met=not baseline_relation['endpoint_goal'] and final_relation['endpoint_goal'],
                whole_goal_newly_met=old['whole_goal'] is False and new['whole_goal'] is True,
                attribution_caveat='Mirror-resolved cases excluded from bottleneck weak-response benefit counts' if mirror else
                                    'Known-domain seen training case; improvement is diagnostic only'))
        negative = int(record.get('selected_negative', 0)); positive = int(record.get('selected_positive', 0))
        attempts = [s for r in record.get('rounds', []) for s in r.get('selections', [])]
        inactive = positive > 0 and negative == 0 and difference['boundary']['exact']
        arms[arm] = dict(selected_negative=negative, selected_positive=positive,
            selected_total=negative + positive, no_intervention_applied=negative + positive == 0,
            inactive_readout=inactive,
            inactive_readout_explanation='Positive short relationships changed but native blended boundary is exact; not evidence training is ineffective' if inactive else None,
            actual_map_changes=difference, baseline_file_sha256=before['file_sha256'],
            final_file_sha256=final['file_sha256'], cases=qualities,
            selection_attempts=len(attempts), selection_attempt_reasons=dict(Counter(s.get('reason', 'unspecified') for s in attempts)),
            metrics=metrics_summary(report['baseline'], record['final_metrics']),
            quality_complete=all(q['baseline']['available'] and q['final']['available'] for q in qualities))
    return dict(source=name, cohort=cohort, run=str(Path(run_root).resolve()),
        baseline_exact=report.get('baseline_exact') is True, input_hashes=report.get('input_hashes'),
        capture_audit=report.get('capture_audit'), arms=arms)


def aggregate(rows):
    result = {}
    for cohort in ('all7', 'sampled6', 'known1', 'unclassified'):
        selected = rows if cohort == 'all7' else [r for r in rows if r['cohort'] == cohort]
        by_arm = {}
        for arm in sorted({a for r in selected for a in r['arms']}):
            values = [r['arms'][arm] for r in selected if arm in r['arms']]
            cases = [c for value in values for c in value['cases']]
            eligible = [c for c in cases if c['eligible_weak_response_evidence']]
            by_arm[arm] = dict(source_run_rows=len(values), unique_sources=len({r['source'] for r in selected if arm in r['arms']}),
                selected_negative=sum(v['selected_negative'] for v in values),
                selected_positive=sum(v['selected_positive'] for v in values),
                native_boundary_changed=sum(v['actual_map_changes']['boundary']['different'] for v in values),
                marker_changed=sum(v['actual_map_changes']['markers']['different'] for v in values),
                watershed_changed=sum(v['actual_map_changes']['watershed']['different'] for v in values),
                final_instances_changed=sum(v['actual_map_changes']['instances']['different'] for v in values),
                inactive_readout_sources=sum(v['inactive_readout'] for v in values),
                case_observations=len(cases), mirror_resolved_observations=sum(c['mirror_already_resolved'] for c in cases),
                weak_response_eligible_observations=len(eligible),
                weak_response_endpoint_newly_met=sum(c['endpoint_newly_met'] for c in eligible),
                weak_response_whole_goal_newly_met=sum(c['whole_goal_newly_met'] for c in eligible),
                positive_significant_piece_reductions=sum(c['significant_pieces_reduced'] is True for c in eligible),
                original_matched_F_missing=sum(len(v['metrics']['final'].get('original_matched_F_missing') or []) for v in values),
                fixed_area_failure_counts=dict(sum((Counter((v['metrics'].get('fixed_area') or {}).get('failure_counts', {}))
                    for v in values), Counter())))
        result[cohort] = dict(unique_sources=sorted({r['source'] for r in selected}), by_arm=by_arm)
    return result


def analyze(runs, output, *, readout_report=None, allow_partial=False):
    output = Path(output).resolve()
    if output.exists() or not any(output.is_relative_to(ROOT / root) for root in ('output', 'outputs')):
        raise ValueError('A fresh ignored output/ or outputs/ JSON file is required')
    lookup, sampled, readout = readout_lookup(readout_report, allow_partial)
    rows = []; inputs = []; missing = []; identities = []
    for value in runs:
        path = Path(value).resolve(); root = path if path.is_dir() else path.parent
        report_path = root / 'report.json' if path.is_dir() else path
        report = read(report_path)
        complete = report.get('complete') is True
        if not complete and not allow_partial:
            raise ValueError('Incomplete run requires explicit --allow-partial: ' + str(report_path))
        if report.get('format') != 'affinity_bottleneck_probe_v1' or report.get('training') is not False:
            raise ValueError('Expected a frozen bottleneck probe report')
        if complete and (report.get('no_optimizer_steps') is not True or report.get('frozen_states_exact') is not True):
            raise ValueError('Complete run must verify frozen states and no optimizer steps')
        inputs.append(dict(root=str(root), report=str(report_path), sha256=sha(report_path), complete=complete))
        keys = ('checkpoint_sha256', 'model_state_sha256', 'restorer_state_sha256',
                'restorer_checkpoint_sha256', 'config', 'precision', 'runtime')
        identities.append({key: report.get(key) for key in keys})
        sources = report.get('sources', [])
        for source in sources:
            try:
                rows.append(summarize_source(root, source, lookup, sampled))
            except FileNotFoundError as exc:
                if not allow_partial: raise
                missing.append(dict(run=str(root), source=source_name(source['source']), missing=str(exc)))
    observation_ids = [(r['run'], r['source']) for r in rows]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError('Duplicate source/run observation')
    counts = {group: len({r['source'] for r in rows if r['cohort'] == group}) for group in ('sampled6', 'known1', 'unclassified')}
    cohort_complete = len({r['source'] for r in rows}) == 7 and counts == dict(sampled6=6, known1=1, unclassified=0)
    evidence_complete = bool(rows) and all(r['baseline_exact'] and bool(r['arms'])
        and all(a['quality_complete'] for a in r['arms'].values()) for r in rows)
    identity_complete = bool(identities) and all(all(value is not None for value in identity.values())
        and identity == identities[0] for identity in identities)
    complete = bool(inputs and all(r['complete'] for r in inputs) and cohort_complete and evidence_complete
                    and identity_complete and not missing and (readout is None or readout['complete']))
    summary = dict(format='affinity_bottleneck_analysis_v1', complete=complete, cpu_only=True,
        no_inference=True, no_training=True, train_go=False, manual_review_required=True,
        deployment_promoted=False, allow_partial=allow_partial, inputs=inputs, readout_report=readout,
        expected_cohort=dict(all_sources=7, sampled_sources=6, known_sources=1),
        cohort_complete=cohort_complete, actual_unique_sources=len({r['source'] for r in rows}),
        common_frozen_identity=identities[0] if identities else None,
        frozen_run_identities_equal_and_complete=identity_complete,
        quality_evidence_complete=evidence_complete, missing_outputs=missing,
        aggregate=aggregate(rows), sources=rows,
        caveats=['Summary never launches training or promotes deployment.',
            'A first-source/partial result remains complete=False and train_go=False.',
            'All conclusions concern seen processed training GT; censoring and filled GT are retained.',
            'Source-run repetitions are explicit observations and not independent validation.',
            'Mirror readout-only cases are excluded from weak-response benefit counts.',
            'Inactive native readout is separate from model learning capacity.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--readout-report')
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    value = analyze(args.runs, args.output, readout_report=args.readout_report, allow_partial=args.allow_partial)
    print('BOTTLENECK_ANALYSIS', json.dumps(dict(complete=value['complete'],
        unique_sources=value['actual_unique_sources'], train_go=False)), flush=True)


if __name__ == '__main__':
    main()
