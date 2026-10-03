# -*- coding: utf-8 -*-
"""原图语义压力检查的对象分母、ignore、方向和事前取样契约。"""
from pathlib import Path

import numpy as np
import pytest

from tools.probe_semantic_raw_stress import INITIAL, build_cohort, probability_rows, source_order, summarize_condition
from utils.config import load_config


def test_original_unknown_fill_and_core_are_not_labels():
    ids=np.zeros((12,16),np.uint16)
    ids[2:10,2:10]=1
    ids[3:9,11:15]=2
    original=ids>0
    original[2:10,2:4]=False
    cohort=build_cohort(ids,original,{1:1,2:0},minimum_area=50)
    clear=np.full(ids.shape,.5,np.float32)
    clear[ids==1]=.9;clear[ids==2]=.1
    stressed=clear.copy()
    stressed[~original]=1
    a,b=probability_rows(cohort,clear),probability_rows(cohort,stressed)
    assert [r['probability_original'] for r in a]==[r['probability_original'] for r in b]
    assert a[0]['original_pixels']==48 and a[0]['filled_pixels']==16
    assert a[0]['large'] and not a[1]['large']
    assert summarize_condition(a,b)['large']['object_wrong']==0
    assert all(not np.any(x['core'] & ~x['trusted']) for x in cohort)


def test_fixed_denominator_and_two_directions_with_half_threshold():
    ids=np.zeros((8,14),np.uint16);ids[1:7,1:6]=1;ids[1:7,8:13]=2
    cohort=build_cohort(ids,ids>0,{1:1,2:0},minimum_area=30)
    clean=np.where(ids==1,.9,.1).astype(np.float32)
    changed=np.where(ids==1,.5,.9).astype(np.float32)
    a,b=probability_rows(cohort,clean),probability_rows(cohort,changed)
    row=summarize_condition(a,b)['large']
    assert row['objects']==2 and row['clean_correct_now_wrong']==2
    assert row['clean_correct_F_now_P']==1 and row['clean_correct_P_now_F']==1
    assert row['F_to_P_wrong']==1 and row['P_to_F_wrong']==1
    assert b[0]['predicted_class_original']==0
    with pytest.raises(ValueError,match='fixed GT cohort'):
        summarize_condition(a,b[::-1])


def test_zero_original_object_remains_censored_without_fake_failure():
    ids=np.ones((4,4),np.uint16);original=np.zeros_like(ids,bool)
    cohort=build_cohort(ids,original,{1:1},minimum_area=1)
    rows=probability_rows(cohort,np.zeros(ids.shape,np.float32))
    assert rows[0]['probability_original'] is None and not rows[0]['usable']
    assert rows[0]['frame_censored']
    assert summarize_condition(rows,rows)['large']['objects']==0


def test_prior_fixed_source_selection_and_portable_config():
    names=['train_999',*reversed(INITIAL),'train_000']
    assert source_order(names,limit=4)==list(INITIAL)
    assert source_order(names)[4:]==['train_000','train_999']
    with pytest.raises(ValueError):
        source_order(names[:-2])
    cfg=load_config(str(Path(__file__).parents[1]/'config/inference/semantic_raw_stress.yaml'))
    assert cfg['semantic_raw_stress']['initial_sources']==list(INITIAL)
    assert not Path(cfg['semantic_raw_stress']['checkpoint']).is_absolute()
    assert cfg['semantic_raw_stress']['views_per_family']==2
