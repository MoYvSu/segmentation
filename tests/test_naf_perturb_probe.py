# -*- coding: utf-8 -*-
"""NAF扰动诊断的固定GT集合、缺口分母、来源隔离及CPU梯度合同。"""
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest
import torch

from tools import probe_naf_perturbation as probe
from utils.boundary_rgb_training import build_tolerant_targets


def scene():
    shape = (96, 112)
    target = np.zeros(shape, bool); target[8:88, 55] = True
    original = np.ones(shape, bool); original[55:80] = False
    filled = np.zeros(shape, bool); filled[55:80] = True
    original[:6, :12] = False
    valid = original | filled
    clean = np.zeros(shape, np.float32); clean[8:88, 54:57] = .9
    return clean, target, valid, original, filled


def test_identity_is_exact_and_input_objects_unchanged():
    clean, target, valid, original, filled = scene()
    before = [a.copy() for a in (clean, target, valid, original, filled)]
    result, masks = probe.analyze_pair(clean, clean, target, valid, original, filled, return_masks=True)
    assert result['normalized_input_changed_pixels'] == 0
    assert result['by_provenance']['all']['new_missing'] == 0
    assert result['by_provenance']['all']['recovered_old_missing'] == 0
    assert np.array_equal(masks['clean_T8'], masks['variant_T8'])
    for a, b in zip((clean, target, valid, original, filled), before):
        assert np.array_equal(a, b)
    json.dumps(result, allow_nan=False)


def test_supervision_masks_match_real_loss_and_ignore_unknown():
    clean, target, valid, original, filled = scene()
    result, masks = probe.analyze_pair(clean, np.zeros_like(clean), target, valid, original, filled, return_masks=True)
    exact = build_tolerant_targets(torch.from_numpy(target.astype(np.float32))[None, None],
                                  torch.from_numpy(valid.astype(np.float32))[None, None], tolerance=2)
    for key in ('positive', 'negative', 'trusted'):
        assert np.array_equal(masks[key], exact[key][0, 0].numpy())
    unknown = result['by_provenance']['unknown']
    assert unknown['positive'] == unknown['negative'] == unknown['trusted'] == 0
    for key in ('positive', 'negative', 'new_missing', 'natural_missing'):
        assert result['by_provenance']['all'][key] == sum(result['by_provenance'][p][key] for p in ('original', 'filled'))


def test_added_missing_keeps_original_denominator_and_old_missing_cohort():
    clean, target, valid, original, filled = scene()
    clean[18:35] = 0
    variant = clean.copy(); variant[55:79] = 0
    result, masks = probe.analyze_pair(clean, variant, target, valid, original, filled, return_masks=True)
    all_ = result['by_provenance']['all']
    assert all_['natural_missing'] > 0 and all_['new_missing'] > 0
    assert all_['new_missing'] <= result['by_provenance']['filled']['positive']
    assert all_['actual_variant_missing'] == all_['still_missing'] + all_['new_missing']
    assert all_['positive'] == all_['unchanged_supported'] + all_['new_missing'] + all_['natural_missing']
    assert not np.any(masks['new_missing'] & masks['natural_missing'])


def test_recovered_old_missing_is_not_hidden_in_net_count():
    clean, target, valid, original, filled = scene()
    variant = clean.copy(); clean[30:70] = 0
    result = probe.analyze_pair(clean, variant, target, valid, original, filled)
    all_ = result['by_provenance']['all']
    assert all_['recovered_old_missing'] > 0 and all_['new_missing'] == 0
    assert all_['natural_missing'] == all_['recovered_old_missing'] + all_['still_missing']


def test_unknown_input_is_not_masked_out_to_make_diagnostic_look_better():
    clean, target, valid, original, filled = scene()
    clean[:4, :10] = .8
    result = probe.analyze_pair(clean, clean, target, valid, original, filled)
    assert result['by_provenance']['unknown']['clean_raw_boundary_pixels'] == 40
    assert result['by_provenance']['unknown']['positive'] == 0


def test_logit_gradient_uses_global_positive_denominator_without_loading_model():
    clean, target, valid, original, filled = scene()
    variant = np.zeros_like(clean)
    _, masks = probe.analyze_pair(clean, variant, target, valid, original, filled, return_masks=True)
    logits = torch.full((1, 1, *clean.shape), -4., requires_grad=True)
    before = logits.detach().clone()
    result = probe.coverage_gradient_probe(logits, target, valid, masks)
    row = result['cohorts']['new_missing']
    assert row['points'] > 0 and row['global_positive_denominator'] == int(masks['positive'].sum())
    assert row['coverage_contribution'] > 0 and row['logit_gradient_L1'] > 0
    assert row['nonzero_logit_gradient_pixels'] > 0
    assert torch.equal(logits, before) and logits.grad is None
    assert result['no_optimizer_update'] and result['no_parameter_gradient_claim']


@pytest.mark.parametrize('which', ['overlap', 'unknown_valid', 'unknown_target'])
def test_bad_provenance_cannot_enter_supervision(which):
    clean, target, valid, original, filled = scene()
    if which == 'overlap':
        original[60, 20] = True
    elif which == 'unknown_valid':
        valid[0, 0] = True
    else:
        target[0, 0] = True
    with pytest.raises(ValueError, match='provenance'):
        probe.analyze_pair(clean, clean, target, valid, original, filled)


@pytest.mark.parametrize('bad', [np.full((96,112), np.nan), np.full((96,112), 1.01), np.zeros((4,4)), np.zeros((96,112,1))])
def test_bad_heatmaps_rejected(bad):
    clean, target, valid, original, filled = scene()
    with pytest.raises(ValueError):
        probe.analyze_pair(clean, bad, target, valid, original, filled)


def test_empty_GT_and_full_heatmaps_have_finite_width_summary():
    a = np.ones((12, 15), np.float32); z = np.zeros(a.shape, bool); o = np.ones(a.shape, bool)
    result = probe.analyze_pair(a, a, z, o, o, z)
    assert result['by_provenance']['all']['positive'] == 0
    assert result['by_provenance']['all']['new_missing_fraction_of_positive'] is None
    assert np.isfinite(result['widths']['clean_raw']['median'])
    json.dumps(result, allow_nan=False)


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def manifest_fixture(tmp_path):
    h, w = 80, 96
    rgb = np.full((h, w, 3), 130, np.uint8)
    image = tmp_path/'train_001.jpg'; cv2.imencode('.jpg', rgb)[1].tofile(image)
    ids = np.ones((h, w), np.uint16); ids[:, w//2:] = 2
    original = np.ones((h,w), bool); original[50:65] = False
    gt = tmp_path/'train_001_gt.npz'
    np.savez_compressed(gt, instance_map=ids, original_covered=original,
                        filled=~original, residual_unknown=np.zeros((h,w), bool))
    classes = tmp_path/'train_001_class.json'; write(classes, {'1': 0, '2': 1})
    hashes = {key+'_sha256': probe.file_sha(value) for key, value in dict(image=image, gt=gt, classes=classes).items()}
    processed = tmp_path/'processed.json'
    write(processed, dict(rows=[dict(source='train_001',image=image.name,gt=gt.name,classes=classes.name,hashes=hashes)]))
    clean = np.zeros((h,w), np.float32); clean[:,w//2-1:w//2+1] = .9
    cleanfile = tmp_path/'clean.npz'; np.savez_compressed(cleanfile,boundary=clean)
    base = dict(source='train_001',shape=[h,w],**hashes)
    clean_manifest = tmp_path/'clean.json'
    write(clean_manifest, dict(format='boundary_naf_real_heatmaps_v1',complete=True,no_test_training=True,
                              local_restoration={'mode':'zero','model':'restorer'},
                              rows=[dict(base,heat_path=cleanfile.name,heat_sha256=probe.file_sha(cleanfile))]))
    binary = clean>=.5;binary[23:60] = False
    pertfile = tmp_path/'perturbed.npz';np.savez_compressed(pertfile,boundary=binary)
    row = dict(base,heat_path=pertfile.name,heat_sha256=probe.file_sha(pertfile),representation='thresholded_binary',
               threshold=.5,binary_sha256=probe.array_sha(binary))
    manifest = tmp_path/'perturbation.json'
    doc = dict(format=probe.FORMAT,complete=True,no_test_training=True,
               processed_gt_manifest=dict(path=processed.name,sha256=probe.file_sha(processed)),
               processed_gt_manifest_sha256=probe.file_sha(processed),
               clean_real_input_manifest=dict(path=clean_manifest.name,sha256=probe.file_sha(clean_manifest)),
               clean_real_input_manifest_sha256=probe.file_sha(clean_manifest),
               variants=[dict(id='moderate',rows=[row])])
    write(manifest, doc)
    return manifest,processed,doc


def test_cli_subset_roundtrip_records_gallery_and_preserves_all_sources(tmp_path):
    manifest, processed, doc = manifest_fixture(tmp_path)
    pins = {str(p): probe.file_sha(p) for p in tmp_path.iterdir() if p.is_file()}
    result = probe.run(manifest,processed,tmp_path/'output',sources=['train_001'])
    assert result['complete'] and result['no_training'] and result['no_model_forward']
    assert len(result['rows']) == 1 and len(result['gallery']) == 2
    assert result['rows'][0]['by_provenance']['all']['new_missing'] > 0
    assert result['rows'][0]['continuous_variant_not_available']
    assert (tmp_path/'output/index.html').exists()
    assert all(probe.file_sha(path)==digest for path,digest in pins.items())
    with pytest.raises(FileExistsError):
        probe.run(manifest,processed,tmp_path/'output')


@pytest.mark.parametrize('mutation', ['test_source', 'heat_sha', 'GT_binding', 'binary_sha', 'no_test_flag', 'conflicting_sha', 'N0_mode'])
def test_manifest_rejects_test_and_unsealed_source_changes(tmp_path, mutation):
    manifest, processed, doc = manifest_fixture(tmp_path)
    row = doc['variants'][0]['rows'][0]
    if mutation == 'test_source':row['source'] = 'test_169'
    elif mutation == 'heat_sha':row['heat_sha256'] = 'f'*64
    elif mutation == 'GT_binding':row['gt_sha256'] = 'f'*64
    elif mutation == 'binary_sha':row['binary_sha256'] = 'f'*64
    elif mutation == 'conflicting_sha':doc['processed_gt_manifest_sha256'] = 'f'*64
    elif mutation == 'N0_mode':doc['local_restoration'] = {'mode':'noise','model':'restorer'}
    else:doc['no_test_training'] = False
    write(manifest,doc)
    with pytest.raises(ValueError):
        probe.run(manifest,processed,tmp_path/'output')
