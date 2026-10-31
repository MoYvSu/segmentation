# -*- coding: utf-8 -*-
"""单连通额外监督的实例ID载荷；旧输入/GT/随机流保持，ID不作模型输入。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from data.boundary_rgb import BoundaryRGBDataset, collate_boundary_rgb
from utils.boundary_erasure import DEFAULTS
from test_boundary_erasure import _dataset_config
from test_boundary_rgb import _force


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(tmp_path):
    manifest, cfg = _dataset_config(tmp_path)
    cfg['boundary_naf']['erasure'] = dict(DEFAULTS, probability=0.)
    cfg['boundary_naf']['model'] = {'upsampling': 'project_bilinear_v1'}
    return manifest, cfg


def _same_sample_except_ids(actual, expected):
    assert set(actual) - {'ridge_instances'} == set(expected)
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.equal(actual[key], value), key
        else:
            assert actual[key] == value, key


def test_default_and_explicit_disabled_are_exact_and_enabling_only_adds_ids(tmp_path):
    manifest, cfg = config(tmp_path)
    torch_before = torch.get_rng_state().clone()
    numpy_before = deepcopy(np.random.get_state())
    base = BoundaryRGBDataset(manifest, cfg)
    disabled_cfg = deepcopy(cfg)
    disabled_cfg['boundary_naf']['ridge_supervision'] = {'enabled': False}
    disabled = BoundaryRGBDataset(manifest, disabled_cfg)
    enabled_cfg = deepcopy(cfg)
    enabled_cfg['boundary_naf']['ridge_supervision'] = {'enabled': True}
    enabled = BoundaryRGBDataset(manifest, enabled_cfg)
    assert base.identity == disabled.identity == enabled.identity
    assert base.plan_receipt() == disabled.plan_receipt() == enabled.plan_receipt()
    expected = _force(base, box=[32, 32, 128, 128], rotation=1, flip=True)
    off = _force(disabled, box=[32, 32, 128, 128], rotation=1, flip=True)
    on = _force(enabled, box=[32, 32, 128, 128], rotation=1, flip=True)
    assert 'ridge_instances' not in expected and set(expected) == set(off)
    _same_sample_except_ids(off, expected)
    _same_sample_except_ids(on, expected)
    assert on['ridge_instances'].dtype == torch.int64
    assert on['ridge_instances'].shape == on['target'].shape
    assert torch.equal(torch_before, torch.get_rng_state())
    after = np.random.get_state()
    assert after[0] == numpy_before[0] and after[2:] == numpy_before[2:]
    np.testing.assert_array_equal(after[1], numpy_before[1])


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in range(4) for f in (False, True)])
def test_instance_ids_follow_same_crop_D4_and_have_zero_padding(tmp_path, rotation, flip):
    manifest, cfg = config(tmp_path)
    cfg['boundary_naf']['ridge_supervision'] = {'enabled': True}
    ds = BoundaryRGBDataset(manifest, cfg)
    source_ids = ds.erasure_sources['train_001']['gt'].copy()
    box = [8, 8, 113, 99]
    sample = _force(ds, box=box, rotation=rotation, flip=flip)
    expected = np.rot90(source_ids[8:121, 8:107], rotation)
    if flip:
        expected = expected[:, ::-1]
    ids = sample['ridge_instances'][0].numpy()
    h, w = expected.shape
    np.testing.assert_array_equal(ids[:h, :w], expected)
    assert not ids[h:, :].any() and not ids[:, w:].any()
    assert not sample['valid'][0, h:, :].any() and not sample['valid'][0, :, w:].any()
    # 原图中的unknown仍为0；不是把valid外一切标注都涂掉。
    assert (expected == 0).any()
    assert torch.all(sample['valid'][sample['ridge_instances'] == 0] == 0)
    np.testing.assert_array_equal(source_ids, ds.erasure_sources['train_001']['gt'])


def test_collate_stacks_only_all_enabled_and_rejects_mixed(tmp_path):
    manifest, cfg = config(tmp_path)
    baseline = BoundaryRGBDataset(manifest, cfg)
    off = _force(baseline, box=[32, 32, 128, 128])
    cfg['boundary_naf']['ridge_supervision'] = {'enabled': True}
    on = _force(BoundaryRGBDataset(manifest, cfg), box=[32, 32, 128, 128])
    batch = collate_boundary_rgb([on, on])
    assert batch['ridge_instances'].dtype == torch.int64
    assert batch['ridge_instances'].shape == (2, 1, 128, 128)
    assert torch.equal(batch['ridge_instances'][0], on['ridge_instances'])
    assert 'ridge_instances' not in collate_boundary_rgb([off, off])
    with pytest.raises(ValueError, match='mixed'):
        collate_boundary_rgb([off, on])


@pytest.mark.parametrize('mutate', [
    lambda cfg: cfg.update(input_mode='rgb_boundary'),
    lambda cfg: cfg.update(erasure={'enabled': False}),
    lambda cfg: cfg.update(model={'upsampling': 'pixelshuffle'}),
    lambda cfg: cfg.update(model=None),
    lambda cfg: cfg.pop('model'),
])
def test_enabled_payload_requires_frozen_resize_erasure_recipe(tmp_path, mutate):
    manifest, cfg = config(tmp_path)
    cfg['boundary_naf']['ridge_supervision'] = {'enabled': True}
    mutate(cfg['boundary_naf'])
    with pytest.raises(ValueError, match='ridge supervision'):
        BoundaryRGBDataset(manifest, cfg)


@pytest.mark.parametrize('bad', [[], True, {'enabled': 1}, {'enabled': None}])
def test_bad_ridge_enabled_is_rejected(tmp_path, bad):
    manifest, cfg = config(tmp_path)
    cfg['boundary_naf']['ridge_supervision'] = bad
    with pytest.raises(ValueError, match='ridge_supervision'):
        BoundaryRGBDataset(manifest, cfg)


def test_missing_retained_instance_gt_is_an_error_not_silent_fallback(tmp_path):
    manifest, cfg = config(tmp_path)
    cfg['boundary_naf']['ridge_supervision'] = {'enabled': True}
    cfg['boundary_naf']['erasure']['probability'] = 1.
    ds = BoundaryRGBDataset(manifest, cfg)
    ds.erasure_sources['train_001'].pop('gt')
    with pytest.raises(ValueError, match='requires retained processed instance GT'):
        _force(ds, box=[32, 32, 128, 128])
