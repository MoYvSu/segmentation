# -*- coding: utf-8 -*-
"""分布诊断的有效域、非标签指标及跨源覆盖契约。"""
import cv2
import numpy as np
import pytest
import torch

from data.backend_adaptation import PairedDegradationDataset
from tools.analyze_semantic_domain import (
    DISTANCE_METRICS, ImagePool, appearance_rows, content_box, nearest_coverage, verify_cohorts,
)
from utils.config import load_config


def texture(size=128):
    rng = np.random.default_rng(73)
    noise = cv2.GaussianBlur(rng.random((size, size), dtype=np.float32), (0, 0), 1.2)
    return np.repeat((.25 + .5 * noise)[..., None], 3, axis=-1)


def test_padding_is_excluded_even_after_geometry_moves_valid_rectangle():
    content = texture(64)
    padded = np.ones((96, 96, 3), np.float32)
    padded[16:80, 24:88] = content
    valid = np.zeros((96, 96), np.uint8)
    valid[16:80, 24:88] = 1
    assert content_box(valid) == (16, 80, 24, 88)
    actual = appearance_rows(padded, valid)
    expected = appearance_rows(content, np.ones((64, 64), np.uint8))
    assert actual == expected
    assert len(actual) == 17
    assert sum(row['pixels'] for row in actual if row['scope'] == 'patch') == 64 * 64
    valid[32, 32] = 0
    with pytest.raises(ValueError, match='rectangular'):
        content_box(valid)


def test_blur_proxy_and_brightness_have_distinct_responses():
    image = texture()
    valid = np.ones(image.shape[:2], np.uint8)
    sharp = appearance_rows(image, valid)[0]
    blurred = appearance_rows(cv2.GaussianBlur(image, (0, 0), 4), valid)[0]
    lighter = appearance_rows(image + .1, valid)[0]
    lower_contrast = appearance_rows(.5 + (image - .5) * .5, valid)[0]
    assert blurred['detail_l1'] < sharp['detail_l1'] * .3
    assert blurred['blur_scale_ratio'] > sharp['blur_scale_ratio']
    assert lighter['brightness_mean'] == pytest.approx(sharp['brightness_mean'] + .1, abs=1e-6)
    assert lighter['detail_l1'] == pytest.approx(sharp['detail_l1'], rel=1e-4)
    assert lower_contrast['blur_scale_ratio'] == pytest.approx(sharp['blur_scale_ratio'], rel=1e-3)
    flat = appearance_rows(np.full_like(image, .4), valid)[0]
    assert not flat['texture_informative']
    assert all(np.isfinite(flat[key]) for key in DISTANCE_METRICS)


def test_nearest_reference_excludes_all_patches_of_same_source():
    def row(name, value):
        return {'name': name, **{key: value for key in DISTANCE_METRICS}}
    reference = [row('a', 0), row('a', 0), row('b', 1), row('b', 1)]
    result = nearest_coverage(reference, [row('test1', .5), row('test2', 2.1)])
    # 同图重复块不会使参照阈值成为零。
    assert result['reference_cross_source_nn_q95'] == pytest.approx(1.)
    assert result['query_within_reference_nn_q95'] == .5
    assert result['query_inside_all_marginal_05_95'] == .5
    with pytest.raises(ValueError, match='two source'):
        nearest_coverage(reference[:2], [row('test', 1)])


def test_shared_scale_keeps_pool_expansion_distance_and_coverage_comparable():
    def row(name, value):
        return {'name': name, **{key: value for key in DISTANCE_METRICS}}
    manual = [row('a', 0), row('b', 1)]
    whole = manual + [row('c', 2), row('d', 3)]
    query = [row('test', 3.1)]
    shared = nearest_coverage(whole, query)
    small = nearest_coverage(manual, query, calibration=shared)
    assert small['center'] == shared['center']
    assert small['scale_iqr_floor_0001'] == shared['scale_iqr_floor_0001']
    assert small['reference_cross_source_nn_q95'] == shared['reference_cross_source_nn_q95']
    assert small['normalization_reference_sources'] == 4
    assert small['query_within_reference_nn_q95'] == 0
    assert shared['query_within_reference_nn_q95'] == 1
    assert small['query_nn_quantiles']['0.5'] > shared['query_nn_quantiles']['0.5']


def test_pool_degradation_replays_and_preserves_transformed_content(tmp_path):
    image = np.rint(texture(64)[:40] * 255).astype(np.uint8)
    path = tmp_path / 'train_001.png'
    cv2.imwrite(str(path), image[..., ::-1])
    base = ImagePool([path], image_size=64)
    cfg = load_config('config/train/rgb_diffusion_d5a.yaml')['rgb_restoration']['degradation']
    ds = PairedDegradationDataset(base, cfg, {'enabled': True, 'probability': 1., 'sigma': [.002, .006]},
                                  seed=20260927, repeats=1, augment=True)
    ds.set_epoch(10003)
    first, replay = ds[0], ds[0]
    assert torch.equal(first['image'], replay['image'])
    assert torch.equal(first['valid_content'], replay['valid_content'])
    assert torch.equal(first['content_shape'], first['input_content_shape'])
    assert int(first['valid_content'].sum()) == 40 * 64
    stats = appearance_rows(first['image'].permute(1, 2, 0).numpy(), first['valid_content'].numpy())
    assert stats[0]['pixels'] == 40 * 64


def test_manual_membership_uses_file_identity_instead_of_renamed_stems(tmp_path):
    pool, manual = tmp_path / 'pool', tmp_path / 'manual'
    pool.mkdir()
    manual.mkdir()
    image = np.rint(texture(64) * 255).astype(np.uint8)
    path, copy = pool / 'train_782.png', manual / 'train_001.png'
    cv2.imwrite(str(path), image)
    cv2.imwrite(str(copy), image)
    (manual / 'train_001.json').write_text('{}', encoding='utf-8')
    mapping = verify_cohorts([path], manual, 1)
    assert len(mapping) == 1
    assert mapping[0]['manual_name'] == 'train_001.png' and mapping[0]['pool_name'] == 'train_782.png'
    duplicate = pool / 'train_099.png'
    cv2.imwrite(str(duplicate), image)
    with pytest.raises(ValueError, match='unique SHA-identical'):
        verify_cohorts([path, duplicate], manual, 1)
    image[0, 0] = 0
    cv2.imwrite(str(copy), image)
    with pytest.raises(ValueError, match='unique SHA-identical'):
        verify_cohorts([path], manual, 1)
