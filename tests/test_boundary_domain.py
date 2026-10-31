# -*- coding: utf-8 -*-
"""边界域差分析的 CPU 合成契约，不读取赛题图像或加载模型。"""
import hashlib
import json

import numpy as np
import pytest

from tools.analyze_boundary_domain import (
    analyze_cache, analyze_manifest, array_statistics, equal_source_summary, response_statistics,
)


def toy_cache(size=96):
    yy, xx = np.indices((size, size))
    edge = xx == size//2
    response = np.exp(-.5*((xx-size//2)/3.)**2).astype(np.float32)
    unknown = np.zeros(edge.shape, bool)
    unknown[:, :8] = True
    original = ~unknown
    original[35:60] = False
    filled = ~unknown & ~original
    valid = ~unknown.copy()
    valid[:8] = valid[-8:] = False
    valid[:, :16] = valid[:, -8:] = False
    return dict(rgb=np.full((size, size, 3), 128, np.uint8), real_boundary=response,
                valid=valid, truth_boundary=edge, known_interior=(~unknown & ~edge),
                original_known=original, filled_known=filled, unknown=unknown,
                gt_nominal=response.copy(), recipe_target=response.copy(), recipe_input=response*.8)


def test_width_tracks_native_response_and_does_not_cut_provenance_bands():
    shape = (96, 128)
    valid = np.ones(shape, bool)
    narrow = np.zeros(shape, np.float32); narrow[:, 60:67] = 1
    broad = np.zeros(shape, np.float32); broad[:, 56:71] = 1
    center = np.zeros(shape, bool); center[40:45] = True
    a = response_statistics(narrow, valid)['thresholds']['0.65']['response_band_width']
    b = response_statistics(broad, valid)['thresholds']['0.65']['response_band_width']
    c = response_statistics(broad, valid, center)['thresholds']['0.65']['response_band_width']
    assert a['count'] and b['median'] > a['median'] * 1.8
    assert c['median'] == b['median']  # 不把仅5像素高的来源区域当作响应截断边。


def test_mask_edge_clipped_stripes_are_excluded_from_width():
    probability = np.zeros((96, 96), np.float32)
    probability[:, 30:50] = 1
    valid = np.ones(probability.shape, bool); valid[:, :40] = False
    stats = response_statistics(probability, valid)
    assert stats['thresholds']['0.65']['response_pixels'] > 0
    assert stats['thresholds']['0.65']['reliable_skeleton_pixels'] == 0


def test_gt_distance_and_matched_peak_separate_shift_from_intensity():
    cache = toy_cache()
    xx = np.indices(cache['valid'].shape)[1]
    cache['recipe_input'] = np.exp(-.5*((xx-48-7)/3.)**2).astype(np.float32)
    result = analyze_cache(cache, dict(source='synthetic_a', split='train_gt'))
    gt = result['gt_matched']
    real = gt['real_boundary']['all_valid']
    shifted = gt['recipe_input']['all_valid']
    assert real['gt_skeleton_center']['mean'] == 1.
    assert shifted['gt_skeleton_center']['mean'] < .1
    assert shifted['neighborhood_peaks']['8']['mean'] == 1.
    assert real['known_deep_interior']['count'] > 0
    assert real['known_deep_interior']['mean'] < 1e-5
    assert result['paired_difference_from_real']['gt_nominal']['gt_matched.all_valid.gt_skeleton_center.mean'] == 0.


def test_unknown_is_not_background_and_original_filled_are_separate():
    cache = toy_cache()
    baseline = analyze_cache(cache, dict(source='a', split='train_gt'))
    cache['real_boundary'][cache['unknown']] = 1.
    changed = analyze_cache(cache, dict(source='a', split='train_gt'))
    assert changed['maps']['real_boundary'] == baseline['maps']['real_boundary']
    assert changed['gt_matched']['real_boundary'] == baseline['gt_matched']['real_boundary']
    assert changed['maps']['real_boundary']['filled_known']['values']['count'] > 0
    assert changed['maps']['real_boundary']['original_known']['values']['count'] > 0
    cache['valid'][cache['unknown']] = True
    with pytest.raises(ValueError, match='Unknown'):
        analyze_cache(cache, dict(source='a', split='train_gt'))


def test_unlabeled_output_has_no_gt_or_accuracy_assumption():
    cache = toy_cache()
    data = {key: cache[key] for key in ('rgb', 'real_boundary', 'valid')}
    report = analyze_cache(data, dict(source='arbitrary_source', split='test', source_shape=[2048, 2448]))
    assert 'gt_matched' not in report and 'paired_difference_from_real' not in report
    assert set(report['maps']['real_boundary']) == {'all_valid'}
    with pytest.raises(ValueError, match='test/unlabeled GT forbidden'):
        analyze_cache(cache, dict(source='arbitrary_source', split='test'))


@pytest.mark.parametrize('split', ['train_unlabeled', 'test'])
def test_optional_local_map_compares_to_stitched_without_gt(split):
    cache = toy_cache()
    data = {key: cache[key] for key in ('rgb', 'real_boundary', 'valid')}
    data['local_boundary'] = data['real_boundary'] * .8
    report = analyze_cache(data, dict(source='same_number_different_domain', split=split))
    assert 'gt_matched' not in report
    paired = report['paired_difference_from_real']['local_boundary']
    assert paired['response.all_valid.values.mean'] < 0
    assert not any('gt_matched' in key for key in paired)
    assert report['maps']['local_boundary']['all_valid']['values']['count'] == int(data['valid'].sum())


def test_source_equal_weighting_not_pixel_or_repeated_view_weighting():
    rows = [dict(source='a', split='test', maps=dict(score=0.), coverage=dict(valid_pixels=10000)) for _ in range(10)]
    rows += [dict(source='b', split='test', maps=dict(score=1.), coverage=dict(valid_pixels=10))]
    result = equal_source_summary(rows)
    assert result['sources'] == 2 and result['views'] == 11
    assert result['metrics']['maps.score'] == dict(sources=2, median=.5, q25=.25, q75=.75)


def test_empty_region_is_missing_not_zero_and_nan_rejected():
    assert array_statistics([])['median'] is None
    with pytest.raises(ValueError, match='finite'):
        array_statistics([np.nan])
    cache = toy_cache(); cache['real_boundary'][20, 20] = np.nan
    with pytest.raises(ValueError, match='finite floating'):
        analyze_cache(cache, dict(source='a', split='train_gt'))


def test_manifest_sha_grouping_reproducibility_and_no_gt_leak(tmp_path):
    cache = toy_cache()
    path = tmp_path/'cache.npz'
    np.savez_compressed(path, **cache)
    document = dict(complete=True, rows=[dict(source='toy', split='train_gt', cache_path=path.name,
        source_shape=[1536, 2048], sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    manifest = tmp_path/'manifest.json'; manifest.write_text(json.dumps(document))
    first = analyze_manifest(manifest)
    second = analyze_manifest(manifest)
    assert first == second
    assert 'train_gt/resolution_1536x2048' in first['groups']
    json.dumps(first, allow_nan=False)
    document['rows'][0]['sha256'] = 'bad'
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='SHA256 differs'):
        analyze_manifest(manifest)
