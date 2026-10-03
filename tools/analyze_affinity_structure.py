# -*- coding: utf-8 -*-
"""汇总封存的结构反事实，并显式揭示显著阈值以下的残片。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def residual_quality(quality):
    """保留所有深部贡献，不将10%显著门槛以下的小片视作消失。"""
    rows = []
    for row in quality['gt_rows']:
        dominant = row['dominant_pred']
        secondary = [piece for piece in row['contributions']
                     if piece['pred'] > 0 and piece['pred'] != dominant]
        unassigned = sum(piece['pixels'] for piece in row['contributions'] if piece['pred'] == 0)
        unassigned_deep = sum(piece['deep_pixels'] for piece in row['contributions'] if piece['pred'] == 0)
        secondary_deep = sum(piece['deep_pixels'] for piece in secondary)
        rows.append(dict(gt=row['gt'], known_pixels=row['known_pixels'],
            deep_pixels=row['deep_pixels'], dominant_pred=dominant,
            dominant_fraction=row['dominant_fraction'], dominant_deep_fraction=row['dominant_deep_fraction'],
            secondary_pixels=sum(piece['pixels'] for piece in secondary),
            secondary_deep_pixels=secondary_deep, secondary_contributions=secondary,
            largest_secondary_pixels=max((piece['pixels'] for piece in secondary), default=0),
            unassigned_pixels=unassigned, unassigned_deep_pixels=unassigned_deep,
            all_deep_in_one_prediction=bool(row['deep_pixels'] > 0 and dominant > 0
                                           and not secondary_deep and not unassigned_deep)))
    return dict(gt_rows=rows, all_deep_in_target_main=all(row['all_deep_in_one_prediction'] for row in rows),
                secondary_deep_pixels=sum(row['secondary_deep_pixels'] for row in rows),
                unassigned_deep_pixels=sum(row['unassigned_deep_pixels'] for row in rows))


def case_summary(case):
    known = residual_quality(dict(gt_rows=case['whole_known_quality']['gt_contributions']))
    original = residual_quality(case['original_covered_quality'])
    return dict(case_id=case['case_id'], gt_ids=case['gt_ids'], direction=case['direction'],
        thresholded_diagnostic_pass=case['full_target_pass'], processed_known_residuals=known,
        original_covered_residuals=original, matched_gt=case['matched_gt'],
        censored=case['whole_known_quality']['whole_object_censored'],
        caveat='Thresholded pass is not complete repair. Nonmain pixels can belong to a neighbor main instance or an independent fragment; inspect actual PNG.')


def summarize(report):
    if report.get('complete') is not True or report.get('format') != 'affinity_structure_probe_v1':
        raise ValueError('Complete sealed seven-source structure probe required')
    if report['reference_cohort'] != dict(matched_F=689, safe_area=135):
        raise ValueError('Reflect fixed reference cohort differs')
    sources = report['sources']
    names = [Path(source['source']['source_path']).stem for source in sources]
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError('Seven unique sources required')
    result = dict(format='affinity_structure_analysis_v1', training=False, no_submission=True,
        actual_affinity_or_fusion_changed=False, reference_cohort=report['reference_cohort'],
        groups={}, sources=[],
        caveat='Seen training source native-scalar GT counterfactual only; no learning or generalization evidence. Completed GT includes automatic fill. Every small contribution retained.')
    for name, source in zip(names, sources):
        row = dict(name=name, reference_cases=[case_summary(c) for c in source['reference_cases']], arms={})
        for arm, value in source['arms'].items():
            row['arms'][arm] = dict(mask=value['mask'], safety=value['safety'],
                target_cases=[case_summary(c) for c in value['target_outcomes']],
                phases=value['metrics']['phases'], selected_cases=source['selected'][arm.split('_')[0]],
                actual_WS_elevation_equal=value['real_WS_elevation_equal_to_reference'])
        result['sources'].append(row)
    # 不存在相应修改的源按原输出计入，固定689/135分母；marker-only另列，绝不重复累计。
    for kind in ('interface', 'interior'):
        for scope in ('trusted', 'completed'):
            arm = kind+'_'+scope+'_full'
            item = dict(selected_targets=0, active_sources=0, changed_original_pixels=0, changed_filled_pixels=0,
                thresholded_pass_targets=[], exact_known_deep_targets=[], residual_deep_targets=[],
                lost_fixed_matches=[], new_wrong_F_classes=[], new_area_failures=[], new_errors=[],
                fixed_iou_delta_sum=0., reference_area_absolute_error_sum=0.,
                candidate_area_absolute_error_sum=0., area_denominator=0,
                reference_F_count=0, candidate_F_count=0, reference_F_pixels=0, candidate_F_pixels=0,
                reference_tiny_F_under200=0, candidate_tiny_F_under200=0)
            for name, source in zip(names, sources):
                base = source['reference_metrics']; branch = source['arms'].get(arm)
                final = branch['metrics'] if branch else base
                for prefix, metrics in (('reference', base), ('candidate', final)):
                    phases = metrics['phases']['ferrite']
                    item[prefix+'_F_count'] += phases['instances']
                    item[prefix+'_F_pixels'] += phases['pixels']
                    item[prefix+'_tiny_F_under200'] += phases['tiny_under200']
                area_rows = final['fixed_area']['rows']
                item['area_denominator'] += len(area_rows)
                item['reference_area_absolute_error_sum'] += sum(abs(r['r1_relative_error']) for r in area_rows)
                item['candidate_area_absolute_error_sum'] += sum(abs(r['candidate_relative_error'])
                    for r in area_rows if r['candidate_comparable'])
                if not branch:
                    continue
                item['active_sources'] += 1
                audit = branch['mask']['changed_provenance']
                item['changed_original_pixels'] += audit['original_covered_pixels']
                item['changed_filled_pixels'] += audit['filled_pixels']
                selected = {c['case_id'] for c in source['selected'][kind]['cases']}
                item['selected_targets'] += len(selected)
                for case in branch['target_outcomes']:
                    if case['case_id'] not in selected:
                        continue
                    identity = name+'/'+case['case_id']
                    parsed = case_summary(case)
                    if parsed['thresholded_diagnostic_pass']:
                        item['thresholded_pass_targets'].append(identity)
                    if parsed['processed_known_residuals']['all_deep_in_target_main']:
                        item['exact_known_deep_targets'].append(identity)
                    else:
                        item['residual_deep_targets'].append(identity)
                safety = branch['safety']; matching = safety['fixed_matching']
                item['fixed_iou_delta_sum'] += sum(r['candidate_iou_missing_as_zero']-r['reference_iou']
                                                 for r in matching['rows'])
                item['lost_fixed_matches'].extend(name+'/'+str(g) for g in matching['missing_gt_ids'])
                # 旧封存v1将None失配也算入phase!=1；失配另列，不能叫相别翻转。
                old_gt = {r['gt']:r for r in base['full_ferrite']['gt_rows']}
                new_gt = {r['gt']:r for r in final['full_ferrite']['gt_rows']}
                item['new_wrong_F_classes'].extend(name+'/'+str(g) for g in matching['fixed_gt_ids']
                    if old_gt[g]['predicted_phase']==1 and new_gt[g]['predicted_phase']==0)
                item['new_area_failures'].extend(name+'/'+str(r['gt']) for r in area_rows if not r['candidate_comparable'])
                item['new_errors'].extend(dict(source=name, kind=k, gt=g)
                    for k,v in safety['error_changes'].items() for g in v['added'])
            if item['area_denominator'] != 135:
                raise ValueError('Fixed area denominator drift')
            item['fixed_689_mean_IoU_delta'] = item['fixed_iou_delta_sum']/689
            item['fixed_135_reference_area_MAE'] = item['reference_area_absolute_error_sum']/135
            item['fixed_135_candidate_area_MAE'] = (item['candidate_area_absolute_error_sum']/135
                                                  if not item['new_area_failures'] else None)
            for prefix in ('reference', 'candidate'):
                item[prefix+'_pooled_prediction_F_mean_area'] = item[prefix+'_F_pixels']/max(1,item[prefix+'_F_count'])
            item['training_permission'] = False
            item['caveat'] = 'GT error categories overlap; do not add them. Pooled prediction mean and fixed-area diagnostics are not official score.'
            result['groups'][arm] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not any(output.is_relative_to(repo/root) for root in ('output', 'outputs')):
        raise ValueError('Summary contains private per-object details; ignored output directory required')
    if output.exists():
        raise FileExistsError(output)
    payload = args.input.read_bytes()
    result = summarize(json.loads(payload))
    result['input_report_sha256'] = hashlib.sha256(payload).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result['groups'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
