# -*- coding: utf-8 -*-
"""输入短段擦断的CPU合成合同；不读取赛题图或模型预测。"""
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize
import torch

from data.boundary_restore import known_instance_boundary
from data.boundary_rgb import BoundaryRGBDataset, collate_boundary_rgb
from utils.boundary_erasure import (
    DEFAULTS, draw_seed, erasure_settings, make_erasure_mask, prepare_erasure_source,
    transform_erasure_mask,
)
from utils.boundary_input import normalize_boundary_input

from test_boundary_rgb import _fixture, _force, _sha


def _source(shape=(192, 192), *, filled=False, diagonal=False):
    yy, xx = np.indices(shape)
    gt = np.where(xx >= (yy if diagonal else shape[1]//2), 2, 1).astype(np.uint16)
    original = gt > 0
    if filled:
        original[:, shape[1]//2-5:shape[1]//2+5] = False
    target = known_instance_boundary(gt)
    valid = np.ones(shape, bool)
    valid[:4] = valid[-4:] = False
    valid[:, :4] = valid[:, -4:] = False
    inp_skeleton = skeletonize(target)
    boundary, _ = normalize_boundary_input(inp_skeleton,
        settings={'mode': 'skeleton_dilate', 'kernel_size': 8})
    prepared = prepare_erasure_source(gt, original, target, valid, inp_skeleton)
    return prepared, boundary, target, valid


def _mask(prepared, boundary, **kwargs):
    cfg = dict(DEFAULTS, probability=1., count_range=[1, 1], lengths=[16])
    cfg.update(kwargs.pop('settings', {}))
    return make_erasure_mask(prepared, boundary, box=kwargs.pop('box', [0, 0, *boundary.shape]),
                             crop_margin=4, settings=cfg, seed=kwargs.pop('seed', 19), **kwargs)


@pytest.mark.parametrize('bad', [[], {'x': 1}, {'enabled': 1},
    {'enabled': True, 'version': 'bad'}, {'enabled': True, 'probability': float('nan')},
    {'enabled': True, 'probability': True}, {'enabled': True, 'count_range': [2, 1]},
    {'enabled': True, 'lengths': [2]}, {'enabled': True, 'lengths': [16, 16]},
    {'enabled': True, 'normal_half_width': 4}, {'enabled': True, 'max_attempts': 0}])
def test_bad_settings_rejected(bad):
    with pytest.raises(ValueError):
        erasure_settings(bad)


def test_default_disabled_and_seed_independent_reproducibility():
    assert erasure_settings() == erasure_settings({'enabled': False}) == {'enabled': False}
    np.random.seed(11)
    old = deepcopy(np.random.get_state())
    seed = draw_seed(5, 1, 3, 'train_001')
    assert seed == draw_seed(5, 1, 3, 'train_001')
    assert len({seed, draw_seed(5, 2, 3, 'train_001'), draw_seed(5, 1, 4, 'train_001')}) == 3
    prepared, boundary, _, _ = _source()
    a, ma = _mask(prepared, boundary, seed=seed)
    b, mb = _mask(prepared, boundary, seed=seed)
    np.testing.assert_array_equal(a, b)
    assert ma == mb
    current = np.random.get_state()
    assert old[0] == current[0] and old[2:] == current[2:]
    np.testing.assert_array_equal(old[1], current[1])


@pytest.mark.parametrize('filled,diagonal', [(False, False), (True, False), (False, True)])
def test_supported_gap_erases_full_T8_without_modifying_GT_or_cache(filled, diagonal):
    prepared, boundary, target, valid = _source(filled=filled, diagonal=diagonal)
    originals = {k: prepared[k].copy() for k in ('gt', 'original', 'skeleton', 'input_skeleton')}
    before, target_before, valid_before = boundary.copy(), target.copy(), valid.copy()
    mask, meta = _mask(prepared, boundary)
    assert meta['applied'] and meta['applied_count'] == 1
    assert meta['segments'][0]['provenance'] == ('B' if filled else 'A')
    assert meta['segments'][0]['length_native'] == 16
    assert meta['segments'][0]['normal_half_width_native'] == 8
    damaged = boundary & ~mask
    # 在擦除矩形内，原T8的两肩同样必须清除，而不是只删除GT中心。
    assert not damaged[mask].any()
    assert ((mask & before) & ~target).any()
    assert not (mask & ~valid).any()
    near_before = cv2.dilate(before.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    near_after = cv2.dilate(damaged.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert (target & valid & near_before & ~near_after).any()
    # 同一个原输入分量在请求ROI内真正断开（不是仅少几个GT正像素）。
    y, x = meta['segments'][0]['center_native_yx']
    r = 24
    local_before = before[y-r:y+r+1, x-r:x+r+1].astype(np.uint8)
    local_after = damaged[y-r:y+r+1, x-r:x+r+1].astype(np.uint8)
    assert cv2.connectedComponents(local_before, connectivity=8)[0] == 2
    assert cv2.connectedComponents(local_after, connectivity=8)[0] == 3
    for k, value in originals.items():
        np.testing.assert_array_equal(prepared[k], value)
    np.testing.assert_array_equal(boundary, before)
    np.testing.assert_array_equal(target, target_before)
    np.testing.assert_array_equal(valid, valid_before)


def _center_only(prepared):
    prepared = dict(prepared)
    points = prepared['candidates']
    prepared['candidates'] = points[np.argmin(np.abs(points[:, 0]-96))][None]
    return prepared


@pytest.mark.parametrize('reason', ['unknown', 'third_GT', 'no_core', 'missing_support', 'extra_skeleton'])
def test_conservative_rejection_does_not_redraw_source(reason):
    p, b, _, _ = _source(filled=reason == 'no_core')
    p = _center_only(p)
    y, x = p['candidates'][0]
    if reason == 'unknown':
        p['valid'] = p['valid'].copy()
        p['valid'][y, x+7] = False
    elif reason == 'third_GT':
        p['gt'] = p['gt'].copy()
        p['gt'][y, x+7] = 3
    elif reason == 'no_core':
        p['core_ids'] = frozenset([1])
    elif reason == 'missing_support':
        p['input_skeleton'] = p['input_skeleton'].copy()
        p['input_skeleton'][y-4:y+5] = False
    else:
        p['input_skeleton'] = p['input_skeleton'].copy()
        p['input_skeleton'][y-3:y+4, x+6] = True
    mask, meta = _mask(p, b, settings={'max_attempts': 3})
    assert not mask.any() and not meta['applied']
    assert meta['reason'] == 'attempts_exhausted' and meta['funnel']['attempts'] == 3


def test_crop_rim_probability_skip_empty_and_bounded_attempts():
    p, b, _, _ = _source()
    p = _center_only(p)
    _, meta = _mask(p, b, settings={'probability': 0.})
    assert not meta['requested'] and meta['reason'] == 'probability_skip'
    _, meta = _mask(p, b, box=[90, 64, 64, 64], settings={'max_attempts': 2})
    assert not meta['applied'] and meta['funnel']['unknown_frame_crop_rim_or_padding'] == 2
    p['candidates'] = np.empty((0, 2), np.int32)
    mask, meta = _mask(p, b)
    assert not mask.any() and meta['reason'] == 'no_crop_candidates'
    assert not meta['funnel'].get('attempts', 0)


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in range(4) for f in (False, True)])
def test_mask_D4_and_zero_padding(rotation, flip):
    source = np.zeros((23, 31), bool)
    source[7:11, 9:13] = True
    actual = transform_erasure_mask(source, 40, rotation, flip)
    expected = np.rot90(source, rotation)
    if flip:
        expected = expected[:, ::-1]
    np.testing.assert_array_equal(actual[:expected.shape[0], :expected.shape[1]], expected)
    assert not actual[expected.shape[0]:].any() and not actual[:, expected.shape[1]:].any()


def _dataset_config(tmp_path):
    manifest, heat_manifest, cfg, *_ = _fixture(tmp_path, shape=(192, 192))
    heat_doc = json.loads(heat_manifest.read_text())
    gt = np.load(tmp_path / 'train_001_gt.npz')['instance_map']
    boundary = known_instance_boundary(gt).astype(np.float32)
    np.savez_compressed(tmp_path / 'train_001_heat.npz', boundary=boundary)
    heat_doc['rows'][0]['heat_sha256'] = _sha(tmp_path / 'train_001_heat.npz')
    heat_manifest.write_text(json.dumps(heat_doc))
    cfg['boundary_naf'].update(input_mode='boundary',
        input_normalization={'mode': 'skeleton_dilate', 'kernel_size': 8})
    cfg['boundary_naf']['data'].update(crop_size=128, stride=64)
    return manifest, cfg


def test_dataset_disabled_is_exact_and_enabled_changes_input_only(tmp_path):
    manifest, cfg = _dataset_config(tmp_path)
    old = BoundaryRGBDataset(manifest, cfg)
    off_cfg = deepcopy(cfg)
    off_cfg['boundary_naf']['erasure'] = {'enabled': False}
    off = BoundaryRGBDataset(manifest, off_cfg)
    assert old.identity == off.identity and old.plan_receipt() == off.plan_receipt()
    plan_before_forcing = old.plan_receipt()
    a, b = _force(old, box=[32, 32, 128, 128]), _force(off, box=[32, 32, 128, 128])
    assert set(a) == set(b) and a['meta'] == b['meta']
    for key in a:
        if torch.is_tensor(a[key]):
            assert torch.equal(a[key], b[key])
    on_cfg = deepcopy(cfg)
    on_cfg['boundary_naf']['erasure'] = dict(DEFAULTS, probability=1., lengths=[16], count_range=[1, 1])
    on = BoundaryRGBDataset(manifest, on_cfg)
    assert on.plan_receipt() == plan_before_forcing
    sample = _force(on, box=[32, 32, 128, 128])
    assert sample['meta']['erasure']['applied']
    assert on.identity['format'] == 'boundary_rgb_data_v3'
    assert on.identity['erasure_sources'][0]['supported_candidate_count'] > 0
    assert on.identity['erasure_sources_sha256']
    for key in ('target', 'valid', 'rgb', 'image', 'raw_input_boundary', 'raw_gap_mask', 'gap_mask'):
        assert torch.equal(sample[key], a[key]), key
    assert torch.equal(sample['clean_input_boundary'], a['input_boundary'])
    assert torch.equal(sample['input'], sample['input_boundary'])
    assert not torch.equal(sample['input_boundary'], a['input_boundary'])
    assert sample['artificial_gap_mask'].any()
    assert not (sample['artificial_gap_mask'].bool() & ~sample['valid'].bool()).any()
    on.verify_sources()
    batch = collate_boundary_rgb([sample, sample])
    assert batch['clean_input_boundary'].shape == (2, 1, 128, 128)
    with pytest.raises(ValueError, match='mixed'):
        collate_boundary_rgb([a, sample])


@pytest.mark.parametrize('mode,norm', [('rgb_boundary', {'mode': 'skeleton_dilate', 'kernel_size': 8}),
    ('boundary', {'mode': 'none'}), ('boundary', {'mode': 'skeleton_dilate', 'kernel_size': 6})])
def test_enabled_only_allows_frozen_T8_contract(tmp_path, mode, norm):
    cfg = {'boundary_naf': {'input_mode': mode, 'input_normalization': norm, 'erasure': {'enabled': True}}}
    with pytest.raises(ValueError, match='whole-source'):
        BoundaryRGBDataset('unused.json', cfg)


def test_enabled_skip_always_returns_clean_and_zero_additional_fields(tmp_path):
    manifest, cfg = _dataset_config(tmp_path)
    cfg['boundary_naf']['erasure'] = dict(DEFAULTS, probability=0.)
    ds = BoundaryRGBDataset(manifest, cfg)
    sample = ds[0]
    assert torch.equal(sample['input_boundary'], sample['clean_input_boundary'])
    assert not sample['erasure_mask'].any() and not sample['artificial_gap_mask'].any()
    assert not sample['meta']['artificial_corruption']
    assert not sample['meta']['erasure']['applied']


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in range(4) for f in (False, True)])
def test_dataset_damage_is_whole_native_mask_then_synchronized_D4(tmp_path, rotation, flip):
    manifest, cfg = _dataset_config(tmp_path)
    cfg['boundary_naf']['erasure'] = dict(DEFAULTS, probability=1., lengths=[16], count_range=[1, 1])
    ds = BoundaryRGBDataset(manifest, cfg)
    base = _force(ds, box=[32, 32, 128, 128])
    other = _force(ds, box=[32, 32, 128, 128], rotation=rotation, flip=flip)
    assert base['meta']['erasure'] == other['meta']['erasure']
    for key in ('input_boundary', 'clean_input_boundary', 'erasure_mask', 'artificial_gap_mask',
                'target', 'valid', 'raw_input_boundary', 'gap_mask', 'raw_gap_mask'):
        expected = torch.rot90(base[key], rotation, dims=(-2, -1))
        if flip:
            expected = torch.flip(expected, [-1])
        assert torch.equal(expected, other[key]), key


def test_two_segments_are_separate_and_leave_original_supported_endpoints():
    p, b, _, _ = _source(shape=(384, 192))
    mask, meta = _mask(p, b, settings={'count_range': [2, 2]})
    assert meta['applied_count'] == 2 and meta['funnel']['accepted'] == 2
    for segment in meta['segments']:
        chain = np.asarray(segment['chain_native_yx'])
        for endpoint in (chain[0], chain[-1]):
            y, x = endpoint
            assert not mask[y-2:y+3, x-2:x+3].any()
            assert b[y-2:y+3, x-2:x+3].any()


def test_dataset_qualification_in_memory_mutation_is_detected(tmp_path):
    manifest, cfg = _dataset_config(tmp_path)
    cfg['boundary_naf']['erasure'] = {'enabled': True, 'probability': 0.}
    ds = BoundaryRGBDataset(manifest, cfg)
    ds.erasure_sources['train_001']['gt'][60, 60] = 42
    with pytest.raises(RuntimeError, match='qualification changed'):
        ds.verify_sources()
