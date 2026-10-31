# -*- coding: utf-8 -*-
"""虚构三源小步训练的预算、排除规则和固定分母回归测试。"""
from copy import deepcopy

import pytest

from tools.probe_affinity_interior_fit import aggregate_safety, training_schedule, validate_plan


def fixture_plan():
    sources=[]
    for index in range(7):
        name='fiction_'+str(index)
        case=dict(case_id='positive_'+str(index),direction='positive',gt_ids=[index+1])
        source=dict(source_path='synthetic/'+name+'.jpg',cases=[case],seed=index)
        sources.append(dict(source=source,selected=dict(interior=dict(cases=[case]))))
    structure=dict(sources=sources)
    stage=dict(sources=[dict(source=deepcopy(row['source'])) for row in sources])
    plan=dict(format='affinity_interior_plan_v1',GT_enrichment='postponed',test_training=False,
        training_cases=[dict(source='fiction_'+str(i),case_id='positive_'+str(i),gt_id=i+1) for i in range(3)],
        excluded_training_cases=[dict(source='fiction_4',case_id='positive_4')])
    return plan,structure,stage


def test_all_three_sources_exposed_once_per_update_and_balanced_windows():
    sources=dict(first=[0,3,7],second=[2,4],third=[0])
    schedule=training_schedule(sources)
    assert len(schedule)==32 and sum(map(len,schedule))==96
    for step,batch in enumerate(schedule):
        assert [row['source'] for row in batch]==list(sources)
        assert [row['tile'] for row in batch]==[indices[step%len(indices)] for indices in sources.values()]
    assert training_schedule(sources)==schedule


@pytest.mark.parametrize('sources,steps',[
    (dict(a=[1],b=[2]),32),(dict(a=[],b=[2],c=[3]),32),
    (dict(a=[1,1],b=[2],c=[3]),32),(dict(a=[1],b=[2],c=[3]),31)])
def test_budget_or_missing_window_rejected(sources,steps):
    with pytest.raises(ValueError):training_schedule(sources,steps)


def test_private_targets_must_have_same_stage_identity_and_ignore_gt_enrichment():
    plan,structure,stage=fixture_plan()
    assert len(validate_plan(plan,structure,stage))==3
    stage['sources'][1]['source']['seed']+=1
    with pytest.raises(ValueError,match='seed'):validate_plan(plan,structure,stage)
    plan['GT_enrichment']='enabled'
    with pytest.raises(ValueError,match='postpone'):validate_plan(plan,structure,stage)


def test_registered_ambiguity_or_excluded_target_cannot_enter_training():
    plan,structure,stage=fixture_plan()
    plan['excluded_training_cases'].append(plan['training_cases'][0])
    with pytest.raises(ValueError,match='excluded'):validate_plan(plan,structure,stage)
    plan,structure,stage=fixture_plan()
    structure['sources'][0]['selected']['interior']['cases'][0]['previous_gt_structure_ambiguity']=True
    with pytest.raises(ValueError,match='ambiguity'):validate_plan(plan,structure,stage)


def safety_rows():
    result={}
    for i in range(7):
        count=99 if i<3 else 98
        areas=20 if i<2 else 19
        result[str(i)]=dict(safety=dict(
            fixed_matching=dict(fixed_count=count,missing_gt_ids=[],
                rows=[dict(reference_iou=.8,candidate_iou_missing_as_zero=.8) for _ in range(count)]),
            same_fixed_area_MAE=dict(denominator=areas),
            fixed_area_rows=dict(rows=[dict(r1_relative_error=.03,candidate_relative_error=.03,
                candidate_comparable=True) for _ in range(areas)]),
            fixed_area_status=dict(incomparable_gt_ids=[]),newly_wrong_F_class_gt_ids=[],
            error_changes=dict(split=dict(added=[]),merge=dict(added=[]))))
    return result


def test_fixed_cohort_keeps_missing_and_area_failures_without_calling_them_class_flips():
    rows=safety_rows();baseline=aggregate_safety(rows)
    assert baseline['fixed_F_denominator']==689 and baseline['fixed_area_denominator']==135
    assert baseline['no_new_failures'] and baseline['no_average_regression']
    failed=rows['0']['safety']
    failed['fixed_matching']['rows'][0]['candidate_iou_missing_as_zero']=0.
    failed['fixed_matching']['missing_gt_ids']=[17]
    failed['fixed_area_rows']['rows'][0]['candidate_comparable']=False
    failed['fixed_area_status']['incomparable_gt_ids']=[17]
    result=aggregate_safety(rows)
    assert result['fixed_F_mean_iou_delta']==pytest.approx(-.8/689)
    assert result['fixed_F_missing']==['0/17'] and result['newly_matched_wrong_F_class']==[]
    assert result['fixed_area_denominator']==135 and result['candidate_area_MAE'] is None
    assert not result['no_new_failures'] and not result['no_average_regression']
    assert result['training_extension_permission'] is False


def test_regression_cannot_silently_shrink_denominators():
    rows=safety_rows();rows.pop('6')
    with pytest.raises(RuntimeError,match='cohort'):aggregate_safety(rows)
