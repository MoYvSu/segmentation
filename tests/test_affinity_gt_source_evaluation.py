# -*- coding: utf-8 -*-
"""无测试GT的预测比较合同：编号无关、空覆盖及相别变化分离。"""
from copy import deepcopy
import json

import numpy as np
import pytest

from tools.evaluate_affinity_gt_source import compare_predictions, validate_training_status
from train_affinity_gt_source import R1_SHA, D5A_SHA, canonical_sha
from utils.config import load_config


def test_id_renumbering_is_zero_partition_and_phase_difference():
    ids = np.ones((24,30),np.uint16);ids[:,15:] = 2;ids[0] = 0
    renamed = np.where(ids==1,91,np.where(ids==2,65535,0)).astype(np.uint16)
    stats = compare_predictions(ids,{'1':1,'2':0},renamed,{'91':1,'65535':0})
    assert stats['partition']['complete_partition_identical'] is True
    assert stats['partition']['pair_disagreement_count'] == 0
    assert stats['phase_changed_pixels'] == 0 and stats['ferrite_mean_area_delta'] == 0


def test_split_and_merge_direction_and_exact_pair_count():
    one = np.ones((20,20),np.uint16)
    two = one.copy();two[:,10:] = 2
    split = compare_predictions(one,{'1':1},two,{'1':1,'2':1})
    assert split['partition']['pair_disagreement_count'] == 200*200
    assert split['partition']['significant_split_reference_ids'] == [1]
    assert split['partition']['significant_merge_candidate_ids'] == []
    merge = compare_predictions(two,{'1':1,'2':1},one,{'1':1})
    assert merge['partition']['significant_merge_candidate_ids'] == [1]
    assert merge['partition']['pair_disagreement_count'] == 200*200
    assert not split['official_accuracy'] and not merge['test_GT_used']


def test_class_change_does_not_imply_partition_change_or_zero_coverage():
    ids = np.ones((10,11),np.uint16)
    stats = compare_predictions(ids,{'1':1},ids.copy(),{'1':0})
    assert stats['partition']['complete_partition_identical']
    assert stats['phase_changed_pixels'] == 110 and stats['ferrite_to_pearlite_pixels'] == 110
    assert stats['coverage_changed_pixels'] == 0
    assert stats['candidate_ferrite']['mean_area'] is None


def test_zero_change_is_separate_and_empty_common_has_no_invented_pairs():
    ids = np.ones((9,10),np.uint16);blank = np.zeros_like(ids)
    stats = compare_predictions(ids,{'1':1},blank,{})
    assert stats['coverage_changed_pixels'] == 90
    assert stats['ferrite_to_zero_pixels'] == 90
    assert stats['partition']['complete_partition_identical'] is False
    assert stats['partition']['pair_disagreement_fraction'] is None
    empty = compare_predictions(blank,{},blank.copy(),{})
    assert empty['partition']['complete_partition_identical']
    assert empty['partition']['pair_disagreement_count'] == 0


def test_invalid_uint16_shape_or_class_map_rejected():
    ids = np.ones((10,11),np.uint16)
    for bad,classes in ((ids.astype(np.uint8),{'1':1}),(ids,{}),(ids,{'1':True})):
        with pytest.raises((ValueError,RuntimeError)):
            compare_predictions(ids,{'1':1},bad,classes)
    with pytest.raises(ValueError,match='shape'):
        compare_predictions(ids,{'1':1},np.ones((9,11),np.uint16),{'1':1})


def test_only_complete_e5_320_exact_recipe_can_be_evaluated():
    config = load_config('config/train/affinity_gt_source.yaml')
    contract = dict(checkpoint_sha256=R1_SHA,restorer_checkpoint_sha256=D5A_SHA,
        recipe_sha256=canonical_sha(config),frozen_state_sha256='f'*64,
        restorer_state_sha256='a'*64,manifest_sha256='b'*64)
    status = dict(format='affinity_gt_source_v1',arm='manual',status='complete',epoch=5,updates=320,
        semantic_supervision=False,frozen_unchanged=True,restorer_unchanged=True,strict_reload_equal=True,
        final_checkpoint_sha256='c'*64,contract=contract,config=json.loads(json.dumps(config)))
    assert validate_training_status(status,config,'manual','c'*64) == contract
    for key,value in (('epoch',4),('updates',319),('status','running'),('arm','teacher'),
                      ('frozen_unchanged',False),('strict_reload_equal',False),('semantic_supervision',True)):
        bad = deepcopy(status);bad[key] = value
        with pytest.raises(RuntimeError):validate_training_status(bad,config,'manual','c'*64)
