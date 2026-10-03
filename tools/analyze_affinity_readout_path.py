# -*- coding: utf-8 -*-
"""只读比较focused读取通路与原短关系通路；不训练、不晋级、不内嵌逐图答案。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_affinity_bottleneck import (load_prediction, map_changes,
    metrics_summary, read, sha, source_name, summarize_source)


IDENTITY_KEYS = ('checkpoint_sha256', 'model_state_sha256', 'restorer_state_sha256',
    'restorer_checkpoint_sha256', 'config', 'precision', 'runtime')
SOURCE_KEYS = ('source_path', 'gt_path', 'seed')
CASE_KEYS = ('case_id', 'direction', 'gt_ids', 'points')


def load_run(path, expected_format):
    path = Path(path).resolve()
    root = path if path.is_dir() else path.parent
    report_path = root / 'report.json' if path.is_dir() else path
    report = read(report_path)
    if report.get('format') != expected_format:
        raise ValueError('Unexpected frozen probe format: ' + str(report_path))
    if (report.get('complete') is not True or report.get('frozen_states_exact') is not True
            or report.get('no_optimizer_steps') is not True or report.get('training') is not False):
        raise ValueError('Only complete frozen runs with no optimizer steps may be compared')
    if any(report.get(key) is None for key in IDENTITY_KEYS):
        raise ValueError('Frozen model/runtime identity is incomplete')
    if type(report.get('max_rounds')) is not int or report['max_rounds'] < 1:
        raise ValueError('Actual intervention round budget is required')
    return root, report, dict(path=str(report_path), sha256=sha(report_path))


def index_sources(report):
    indexed = {}
    for source in report.get('sources', []):
        name = source_name(source['source'])
        if name in indexed:
            raise ValueError('Duplicate source identity in one run')
        if source.get('baseline_exact') is not True or not source.get('input_hashes'):
            raise ValueError('Exact baseline and actual source input hashes are required')
        indexed[name] = source
    return indexed


def index_cases(cases):
    result = {}
    for case in cases:
        identity = case.get('case_id')
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError('Case identities must be nonempty and unique')
        if any(key not in case for key in CASE_KEYS):
            raise ValueError('Private case endpoint/GT contract is incomplete')
        result[identity] = case
    return result


def all_exact(change):
    return all(change[key]['exact'] for key in ('boundary', 'markers', 'watershed', 'instances', 'classes'))


def validate_common(focused, raw):
    if any(focused[key] != raw[key] for key in IDENTITY_KEYS):
        raise ValueError('Checkpoint/model/restorer/config/runtime/precision identity differs')


def validate_source(focused, raw):
    fs, rs = focused['source'], raw['source']
    if any(key not in fs or key not in rs or fs[key] != rs[key] for key in SOURCE_KEYS):
        raise ValueError('Source/GT/seed identity differs')
    if focused['input_hashes'] != raw['input_hashes']:
        raise ValueError('Source input hashes differ')
    if focused['baseline'] != raw['baseline']:
        raise ValueError('Complete baseline GT diagnostics differ')


def compare_source(focus_root, focus, raw_root, raw, focus_budget, raw_budget):
    validate_source(focus, raw)
    name = source_name(focus['source'])
    before = load_prediction(focus_root / name, 'baseline')
    raw_before = load_prediction(raw_root / name, 'baseline')
    baseline_changes = map_changes(raw_before, before)
    if not all_exact(baseline_changes):
        raise ValueError('Actual source baseline maps/PNG/classes differ between runs')
    f_cases = index_cases(focus['source']['cases'])
    r_cases = index_cases(raw['source']['cases'])
    if not f_cases or not set(f_cases).issubset(r_cases):
        raise ValueError('Focused cases must be a nonempty subset of the original fixed cases')
    for identity, case in f_cases.items():
        if any(case[key] != r_cases[identity][key] for key in CASE_KEYS):
            raise ValueError('Matched case GT/endpoints/direction differ')
    # 复用逐值地图和实际端点/GT贡献核查，不将整数ID变化直接当拓扑改善。
    f_summary = summarize_source(focus_root, focus, {}, set())
    r_summary = summarize_source(raw_root, raw, {}, set())
    arms = {}
    for arm, f_value in f_summary['arms'].items():
        if arm not in r_summary['arms']:
            raise ValueError('Original raw run lacks a matching arm')
        r_value = r_summary['arms'][arm]
        f_final = load_prediction(focus_root / name, arm + '_final')
        r_final = load_prediction(raw_root / name, arm + '_final')
        fq = {row['case_id']: row for row in f_value['cases']}
        rq = {row['case_id']: row for row in r_value['cases']}
        if len(fq) != len(f_value['cases']) or len(rq) != len(r_value['cases']):
            raise ValueError('Duplicate arm case quality identity')
        matched = []
        for identity, value in fq.items():
            if identity not in rq or identity not in f_cases:
                raise ValueError('Arm contains an unmatched case identity')
            old = rq[identity]
            matched.append(dict(case_id=identity, direction=value['direction'],
                raw=old, focused=value,
                primary_single_direction_observation=arm == value['direction'],
                repeated_case_observation=arm == 'combined',
                endpoint_goal_changed=old['final']['endpoint']['endpoint_goal'] != value['final']['endpoint']['endpoint_goal'],
                whole_goal_changed=old['final']['whole_goal'] != value['final']['whole_goal'],
                attribution_caveat='Unequal iteration budgets and possibly omitted interventions prohibit direct method ranking'))
        r_record, f_record = raw['arms'][arm], focus['arms'][arm]
        for record, declared in ((f_record, f_cases), (r_record, r_cases)):
            for case in record.get('cases', []):
                if case['case_id'] not in declared or any(case[key] != declared[case['case_id']][key]
                        for key in CASE_KEYS):
                    raise ValueError('Arm case GT/endpoints differ from its source contract')
        arms[arm] = dict(raw=r_value, focused=f_value, matched_cases=matched,
            actual_final_raw_to_focused_changes=map_changes(r_final, f_final),
            raw_to_focused_full_GT_metrics=metrics_summary(r_record['final_metrics'], f_record['final_metrics']),
            raw_case_ids=list(rq), focused_case_ids=list(fq),
            same_intervention_case_set=set(rq) == set(fq),
            budgets=dict(raw_max_rounds=raw_budget, focused_max_rounds=focus_budget,
                raw_actual_rounds=len(r_record.get('rounds', [])),
                focused_actual_rounds=len(f_record.get('rounds', [])),
                equal_max_rounds=raw_budget == focus_budget, direct_ranking_supported=False),
            count_as_independent_source=False)
    duplicate = None
    if 'negative' in arms and 'combined' in arms:
        n, c = arms['negative'], arms['combined']
        same_cases = set(n['focused_case_ids']) == set(c['focused_case_ids'])
        no_positive = all(f_cases[identity]['direction'] == 'negative' for identity in c['focused_case_ids'])
        if same_cases and no_positive:
            change = map_changes(load_prediction(focus_root / name, 'negative_final'),
                                 load_prediction(focus_root / name, 'combined_final'))
            duplicate = dict(same_negative_case_set=True, actual_final_changes=change,
                exact_duplicate_output=all_exact(change), count_as_independent_sample=False,
                caveat='Combined repeats the negative-only intervention; any output mismatch requires investigation')
            arms['combined']['duplicate_of_negative'] = True
    primary = []
    for identity, case in f_cases.items():
        arm = case['direction']
        if arm not in arms:
            raise ValueError('Each focused case needs its single-direction primary arm')
        row = next((r for r in arms[arm]['matched_cases'] if r['case_id'] == identity), None)
        if row is None:
            raise ValueError('Primary case quality is missing')
        primary.append(row)
    omitted = [dict(case_id=identity, direction=case['direction'], gt_ids=case['gt_ids'],
        status='not_in_focused_case_set', included_in_evidence_count=False)
        for identity, case in r_cases.items() if identity not in f_cases]
    return dict(source=name, source_identity={key: focus['source'][key] for key in SOURCE_KEYS},
        input_hashes=focus['input_hashes'], baseline_actual_exact=True,
        baseline_file_sha256=dict(raw=raw_before['file_sha256'], focused=before['file_sha256']),
        raw_run=str(raw_root), focused_case_ids=list(f_cases), omitted_raw_cases=omitted,
        omitted_cases_caveat='Case-set difference is explicit; exclusion rationale comes only from a verified private plan',
        arms=arms, negative_combined_repetition=duplicate, unique_primary_case_results=primary)


def analyze(run, raw_runs, output, *, plan=None):
    output = Path(output).resolve()
    if output.exists() or not any(output.is_relative_to(ROOT / root) for root in ('output', 'outputs')):
        raise ValueError('A fresh ignored output/ or outputs/ JSON file is required')
    f_root, focused, f_receipt = load_run(run, 'affinity_readout_path_probe_v1')
    f_sources = index_sources(focused)
    if len(f_sources) != 2:
        raise ValueError('Completion covers exactly the two focused sources, not a seven-source gate')
    raw_index = {}; raw_receipts = []
    for path in raw_runs:
        root, raw, receipt = load_run(path, 'affinity_bottleneck_probe_v1')
        validate_common(focused, raw)
        raw_receipts.append(receipt)
        for name, source in index_sources(raw).items():
            if name in raw_index:
                raise ValueError('Duplicate source across original raw runs')
            raw_index[name] = root, raw, source
    plan_receipt = dict(available=False, verified=False)
    if plan is not None:
        plan = Path(plan).resolve(); payload = read(plan)
        if sha(plan) != focused.get('plan_sha256'):
            raise ValueError('Private focused plan hash differs')
        p_sources = {source_name(source): source for source in payload.get('sources', [])}
        if payload.get('complete') is not True or set(p_sources) != set(f_sources):
            raise ValueError('Private focused plan cohort differs')
        if any(p_sources[name] != f_sources[name]['source'] for name in f_sources):
            raise ValueError('Private focused plan source/case metadata differs')
        plan_receipt = dict(available=True, verified=True, path=str(plan), sha256=sha(plan),
            exclusions=payload.get('excluded'), source_selection=payload.get('source_selection'),
            interpretation='Exclusions were fixed in the verified plan, not selected from final output')
    rows = []
    for name, source in f_sources.items():
        if name not in raw_index:
            raise ValueError('Focused source missing from original raw runs')
        root, raw, original = raw_index[name]
        rows.append(compare_source(f_root, source, root, original, focused['max_rounds'], raw['max_rounds']))
    primary = [item for row in rows for item in row['unique_primary_case_results']]
    quality_complete = bool(primary) and all(item[method][stage]['available']
        for item in primary for method in ('raw', 'focused') for stage in ('baseline', 'final'))
    quality_complete = quality_complete and all(row['arms'][arm][method]['quality_complete']
        for row in rows for arm in row['arms'] for method in ('raw', 'focused'))
    result = dict(format='affinity_readout_path_analysis_v1', complete=quality_complete,
        complete_scope='two focused sources only; not the seven-source training gate',
        focused_source_count=len(rows), focused_sources=list(f_sources),
        training=False, train_go=False, manual_review_required=True, automatic_promotion=False,
        official_accuracy=False, independent_accuracy=False,
        identity_exact=True, identities={key: focused[key] for key in IDENTITY_KEYS},
        inputs=dict(focused=f_receipt, raw=raw_receipts, private_plan=plan_receipt),
        quality_evidence_complete=quality_complete, source_results=rows,
        unique_primary_case_count=len(primary), combined_cases_not_independent=True,
        summary=dict(raw_endpoint_goals=sum(item['raw']['final']['endpoint']['endpoint_goal'] for item in primary),
            focused_endpoint_goals=sum(item['focused']['final']['endpoint']['endpoint_goal'] for item in primary),
            raw_whole_known_GT_goals=sum(item['raw']['final']['whole_goal'] is True for item in primary),
            focused_whole_known_GT_goals=sum(item['focused']['final']['whole_goal'] is True for item in primary)),
        budget_caveat='Original raw and focused budgets are reported per arm; raw8 versus focused4 cannot directly rank methods',
        evidence_caveat='Seen processed known GT only; retain fixed-cohort failures and omitted cases; no native/WS gain inferred from grid targets')
    output.parent.mkdir(parents=True, exist_ok=True)
    import json
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--raw-runs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--plan', help='Optional private focused plan; its SHA and full cohort are checked')
    args = p.parse_args()
    plan = args.plan
    default_plan = ROOT / 'outputs/bottleneck_plan/readout_plan.json'
    if plan is None and default_plan.is_file():
        plan = default_plan
    result = analyze(args.run, args.raw_runs, args.output, plan=plan)
    print('focused_sources=%d complete=%s train_go=False' % (result['focused_source_count'], result['complete']))


if __name__ == '__main__':
    main()
