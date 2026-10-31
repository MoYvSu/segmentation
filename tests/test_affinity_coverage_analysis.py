# -*- coding: utf-8 -*-
"""Coverage分析隔离、真实更新预算与最终文件契约；不以合成图声称竞赛收益。"""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from data.affinity_coverage import FIXED_KEYS, FORMAT, OPTIONS, flags
from tools.analyze_affinity_coverage import (Predictions, audit_steps, normalized_config,
    capture_config_projection, fixed_area_cohort, fixed_contact_trace,
    pin_files, safe_member, sha, validate_manifest)


def make_steps():
    old, actual, declared = [], [], []
    supervision = [dict(recall=.9, specificity=.8)] * 2
    for epoch in range(1, 21):
        selected = iter(flags(epoch))
        for step in range(1, 65):
            view = 'native1024' if step % 2 else 'whole'
            row = dict(epoch=epoch, step=step, seed=epoch * 100 + step, updates=(epoch-1)*64+step,
                       manual=['m.jpg'], pseudo=['p.jpg'], manual_draw=dict(draw_index=[step]),
                       pseudo_draw=dict(draw_index=[step]), training_view=view,
                       manual_crop=[[1, 1, 5, 5]], pseudo_crop=[[1, 1, 5, 5]],
                       crop_attempts=[[0], [0]], auxiliary_weight=0., manual_bce=.2,
                       pseudo_bce=.3, grad_norm=.4, supervision=supervision)
            old.append(deepcopy(row))
            planned = {k: deepcopy(row[k]) for k in FIXED_KEYS}
            planned['counterfactual_not_executed'] = True
            anchor = next(selected) if view == 'native1024' else False
            if view == 'native1024':
                planned['counterfactual_anchor'] = (dict(selected=True, accepted_attempt=0,
                    known_cardinal_pairs=16) if anchor else dict(selected=False))
            if anchor:
                planned['manual_crop'] = [[2, 2, 6, 6]]
            declared.append(planned)
            row['manual_crop'] = deepcopy(planned['manual_crop'])
            row['coverage'] = dict(format=FORMAT, anchor=anchor, actual_update=row['updates'], executed=True)
            actual.append(row)
    return old, actual, declared


def test_exact_budget_and_single_coordinate_variable():
    report = audit_steps(*make_steps())
    assert report['actual_changed_manual_coordinates'] == 320
    assert report['strata']['whole']['updates'] == 640
    assert report['strata']['native_uniform']['same_input_recipe'] is True
    assert report['strata']['native_anchor']['same_input_recipe'] is False


@pytest.mark.parametrize('key,value', [('seed', -1), ('pseudo_crop', [[9,9,10,10]]),
                                     ('manual_draw', {'draw_index': [-1]})])
def test_fixed_source_augmentation_receipts_cannot_change(key, value):
    old, actual, declared = make_steps()
    actual[4][key] = value
    with pytest.raises(RuntimeError):
        audit_steps(old, actual, declared)


def test_no_counterfactual_losses_or_zero_gradients():
    old, actual, declared = make_steps()
    actual[0]['counterfactual_not_executed'] = True
    with pytest.raises(RuntimeError):
        audit_steps(old, actual, declared)
    actual[0].pop('counterfactual_not_executed'); actual[0]['grad_norm'] = 0.
    with pytest.raises(RuntimeError):
        audit_steps(old, actual, declared)


def test_config_projection_removes_only_declared_metadata():
    base = dict(paths=dict(project_root='/old', data='a'), backend_adaptation=dict(output_dir='b'), loss=1)
    new = deepcopy(base); new['paths']['project_root']='/new'; new['affinity_coverage']=OPTIONS
    assert normalized_config(new, candidate=True) == normalized_config(base)
    new['loss']=2
    assert normalized_config(new, candidate=True) != normalized_config(base)


def prediction_files(folder, *, bad_class=False):
    ids = np.array([[0, 1], [2, 2]], np.uint16)
    folder.mkdir(exist_ok=True)
    assert cv2.imwrite(str(folder / 'test_inst.png'), ids)
    (folder / 'test_class.json').write_text(json.dumps({'1': True if bad_class else 1, '2': 0}), encoding='utf8')


def test_directory_and_zip_preserve_uint16_and_class_mapping(tmp_path):
    prediction_files(tmp_path / 'dir')
    zpath = tmp_path / 'pred.zip'
    with zipfile.ZipFile(zpath, 'w') as z:
        for file in (tmp_path / 'dir').iterdir():
            z.write(file, file.name)
    for path in (tmp_path / 'dir', zpath):
        reader = Predictions(path); reader.require(['test'])
        ids, classes = reader.load('test', (2, 2)); reader.close()
        assert ids.dtype == np.uint16 and classes == {'1': 1, '2': 0}


def test_unknown_remains_unknown_and_wrong_shape_rejected(tmp_path):
    prediction_files(tmp_path)
    reader = Predictions(tmp_path)
    assert reader.load('test')[0][0,0] == 0
    with pytest.raises(RuntimeError):
        reader.load('test', (3, 2))
    reader.close()


def test_bool_category_is_not_integer_semantics(tmp_path):
    prediction_files(tmp_path, bad_class=True)
    reader = Predictions(tmp_path)
    with pytest.raises(RuntimeError):
        reader.load('test')
    reader.close()


@pytest.mark.parametrize('member', ['../secret', '/absolute', 'C:/absolute', 'a\\b'])
def test_no_archive_path_escape(member):
    with pytest.raises(RuntimeError):
        safe_member(member)


def test_manifest_requires_exact_fixed_epoch_cohort():
    manifest = dict(complete=True, epoch=20, precision='FP32', checkpoint_sha256='cp', sizes=[1024],
                    images=[dict(source=f'test_{i}', view='patch1024') for i in range(100)])
    assert len(validate_manifest(manifest, 'cp', candidate=True)) == 100
    manifest['images'][1]['source']='test_0'
    with pytest.raises(RuntimeError):
        validate_manifest(manifest, 'cp', candidate=True)


def test_cache_content_tampering_fails(tmp_path):
    (tmp_path / 'cache').write_bytes(b'original')
    manifest = dict(complete=True, files={'cache': sha(tmp_path / 'cache')})
    pin_files(tmp_path, manifest)
    (tmp_path / 'cache').write_bytes(b'changed')
    with pytest.raises(RuntimeError):
        pin_files(tmp_path, manifest)


def test_only_exact_registered_capture_metadata_is_removed():
    config=dict(paths=dict(project_root='a'),backend_adaptation=dict(output_dir='b'),
                affinity_fused=dict(weight=.01,format='registered'),inference=dict(threshold=.65))
    result=capture_config_projection(config,config)
    assert 'affinity_fused' not in result and result['inference']['threshold']==.65
    changed=deepcopy(config);changed['affinity_fused']['weight']=.02
    with pytest.raises(RuntimeError): capture_config_projection(changed,config)


def test_original_fixed_area_cohort_retains_lost_and_wrong_class_objects():
    one=dict(gt=1,area=100,safe_one_to_one_area_eligible=True,area_relative_error=.1,
             matched_pred=1,predicted_phase=1,censored=False,significant_geometry_split=False,
             matched_pred_significant_geometry_merge=False)
    two=deepcopy(one);two['gt']=2
    before=dict(gt_rows=[one,two],censored_pred_ids=[])
    wrong=deepcopy(one);wrong.update(safe_one_to_one_area_eligible=False,area_relative_error=None,predicted_phase=0)
    missed=deepcopy(two);missed.update(safe_one_to_one_area_eligible=False,area_relative_error=None,matched_pred=None)
    candidate=dict(gt_rows=[wrong,missed],censored_pred_ids=[])
    report=fixed_area_cohort(before,candidate)
    assert report['original_fixed_count']==2 and report['comparable_count']==0
    assert report['failure_counts']['wrong_class']==1 and report['failure_counts']['unmatched']==1
    assert len(report['rows'])==2


def test_fixed_interface_trace_does_not_drop_subthreshold_contributions():
    diagnostic=dict(gt_rows=[dict(gt=69,area=100,contributions=[dict(pred=8,pixels=5),dict(pred=9,pixels=95)]),
                             dict(gt=70,area=1000,contributions=[dict(pred=8,pixels=1000)])])
    report=fixed_contact_trace(diagnostic)
    assert report['same_dominant_prediction'] is False
    assert report['shared_prediction_rows'][0]['gt_fractions'][69]==.05
    assert report['shared_prediction_rows'][0]['pred']==8
