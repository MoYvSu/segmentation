# -*- coding: utf-8 -*-
"""全图缓存的内容身份、逐元素桥接和完整拼接契约；不加载实际权重。"""
import numpy as np
import pytest

from tools.capture_boundary_real import exact_roi_bridge, matched_sources


def _records():
    gt = dict(all_identities_match=True, rows=[dict(source='train_001', image='/raw/train_001.png',
        gt='/gt/train_001.npz', classes='/gt/train_001.json', shape=[1200, 1400],
        hashes=dict(image_sha256='a'*64, gt_sha256='b'*64, classes_sha256='c'*64))])
    # 同名池图故意内容不同，真正相同图的名称不同。
    identity = dict(pool_files=[dict(source='train_001', path='/pool/train_001.png',
        sha256='d'*64, full_pool_index=0), dict(source='train_999', path='/pool/train_999.png',
        sha256='a'*64, full_pool_index=1)])
    bridge = dict(complete=True, files=[dict(source='train_001', domain='train_gt', input_sha256='a'*64,
        source_index=1, source_shape=[1200, 1400])])
    return gt, identity, bridge


def test_content_match_wins_over_same_filename():
    rows = matched_sources(*_records(), expected_count=1, pool_size=2)
    assert rows[0]['source'] == 'train_001'
    assert rows[0]['source_index'] == 1
    assert rows[0]['pool_match']['source'] == 'train_999'


def test_ambiguous_or_missing_content_is_rejected():
    gt, identity, bridge = _records()
    identity['pool_files'][0]['sha256'] = 'a'*64
    with pytest.raises(ValueError, match='unique exact'):
        matched_sources(gt, identity, bridge, expected_count=1, pool_size=2)
    identity['pool_files'][0]['sha256'] = 'e'*64
    identity['pool_files'][1]['sha256'] = 'f'*64
    with pytest.raises(ValueError, match='unique exact'):
        matched_sources(gt, identity, bridge, expected_count=1, pool_size=2)


@pytest.mark.parametrize('field,value', [('input_sha256', 'd'*64), ('source_index', 0),
                                        ('source_shape', [1200, 1401])])
def test_prior_reference_seed_identity_and_shape_must_match(field, value):
    gt, identity, bridge = _records()
    bridge['files'][0][field] = value
    with pytest.raises(ValueError, match='bridge'):
        matched_sources(gt, identity, bridge, expected_count=1, pool_size=2)


def test_incomplete_reference_cannot_be_used():
    gt, identity, bridge = _records()
    bridge['complete'] = False
    with pytest.raises(ValueError, match='complete'):
        matched_sources(gt, identity, bridge, expected_count=1, pool_size=2)


def test_full_map_bridge_preserves_amplitude_and_requires_bit_exact():
    boundary = np.random.default_rng(2).random((100, 130), dtype=np.float32)*.8+.1
    roi = [11, 19, 71, 79]
    reference = boundary[11:71, 19:79].copy()
    before = boundary.copy()
    result = exact_roi_bridge(boundary, reference, roi)
    assert result['exact'] and result['max_absolute_difference'] == 0
    assert np.array_equal(boundary, before)
    reference[0, 0] = np.nextafter(reference[0, 0], np.float32(1))
    with pytest.raises(RuntimeError, match='maxdiff'):
        exact_roi_bridge(boundary, reference, roi)
    with pytest.raises(ValueError, match='outside'):
        exact_roi_bridge(boundary, reference, [-1, 19, 59, 79])


def test_complete_original_blend_matches_previous_roi_calculation():
    from tools.capture_boundary_domain import RoiBlendMap, roi_window_plan
    from utils.patch_diagnostic import BlendMap, tile_boxes
    shape = (1200, 1400)
    _, _, roi, selected = roi_window_plan(shape, {})
    full, previous = BlendMap(shape), RoiBlendMap(shape, roi)
    rng = np.random.default_rng(5)
    for index, box in enumerate(tile_boxes(shape, 1024, .25)):
        data = rng.random((1024, 1024), dtype=np.float32)
        full.add(box, data)
        if index in {i for i, _ in selected}:
            previous.add(box, data)
    completed, _ = full.finish()
    reference, _ = previous.finish()
    assert exact_roi_bridge(completed, reference, roi)['exact']
