# -*- coding: utf-8 -*-
"""T8/连续N1配对合同；使用合成数据，不读取赛题图或权重。"""
import copy
import random

import numpy as np
import pytest
import torch

from data.boundary_dual import (
    BoundaryDualDataset, FORMAT, INPUT_CONTRACT, INPUT_MODE,
    collate_boundary_dual, continuous_heat_header, crop_continuous_boundary,
)
from data.boundary_rgb import BoundaryRGBDataset
from test_boundary_rgb import _fixture, _force


def _config(config):
    result = copy.deepcopy(config)
    result['boundary_naf'].update(input_mode=INPUT_MODE, input_contract=INPUT_CONTRACT,
        input_normalization={'mode': 'skeleton_dilate', 'kernel_size': 8})
    return result


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in range(4) for f in (False, True)])
def test_native_continuous_and_T8_share_crop_and_all_D4(tmp_path, rotation, flip):
    manifest, _, config, _, heat, _ = _fixture(tmp_path)
    config = _config(config)
    ds = BoundaryDualDataset(manifest, config)
    sample = _force(ds, rotation=rotation, flip=flip)
    expected = np.rot90(heat[8:72, 8:72], rotation)
    if flip:
        expected = expected[:, ::-1]
    np.testing.assert_array_equal(sample['continuous_boundary'][0].numpy(), expected)
    assert torch.equal(sample['input'][0:1], sample['input_boundary'])
    assert torch.equal(sample['input'][1:2], sample['continuous_boundary'])
    assert sample['input'].shape == (2, 64, 64)
    assert (sample['input'][1] == .49).any()
    assert set(sample['input'][0].unique().tolist()) <= {0., 1.}
    assert sample['meta']['input_contract'] == INPUT_CONTRACT
    assert sample['meta']['rgb_role'] == 'display_only'


@pytest.mark.parametrize('shape', [(40, 48), (1, 48)])
def test_continuous_padding_follows_T8_reflect_or_edge(shape):
    heat = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    crop = crop_continuous_boundary(heat, (0, 0, *shape), 64, 1, True)
    expected = np.rot90(heat, 1)[:, ::-1]
    expected = np.pad(expected, ((0, 64 - expected.shape[0]), (0, 64 - expected.shape[1])),
                      mode='reflect' if min(shape) > 1 else 'edge')
    np.testing.assert_array_equal(crop, expected)


def test_legacy_source_plan_targets_and_all_diagnostics_are_exact(tmp_path):
    manifest, _, config, _, _, _ = _fixture(tmp_path)
    config = _config(config)
    legacy_cfg = copy.deepcopy(config)
    legacy_cfg['boundary_naf']['input_mode'] = 'boundary'
    legacy = BoundaryRGBDataset(manifest, legacy_cfg)
    dual = BoundaryDualDataset(manifest, config)
    assert legacy.plan == dual.plan
    assert legacy.identity == dual.base_identity
    for index in range(len(dual)):
        old, new = legacy[index], dual[index]
        for key in ('target', 'valid', 'input_boundary', 'gap_mask', 'raw_input_boundary',
                    'raw_gap_mask', 'fully_known_mask', 'identity', 'image', 'rgb'):
            assert torch.equal(old[key], new[key]), key
        assert old['source'] == new['source']
        assert old['box'] == new['box']
        for key in ('geometry', 'draw_seed', 'source_draw', 'augmentation', 'gap_pixels',
                    'raw_input_boundary_pixels', 'input_boundary_pixels'):
            assert old['meta'][key] == new['meta'][key]
    assert dual.identity['format'] == FORMAT


def test_unknown_context_retained_and_padding_not_supervised(tmp_path):
    manifest, _, config, _, heat, ids = _fixture(tmp_path, shape=(40, 48))
    dual = BoundaryDualDataset(manifest, _config(config))
    sample = _force(dual, box=(0, 0, 40, 48))
    unknown = ids == 0
    np.testing.assert_array_equal(sample['continuous_boundary'][0, :40, :48].numpy()[unknown], heat[unknown])
    assert not sample['valid'][0, :40, :48].numpy()[unknown].any()
    assert not sample['valid'][0, 40:].any()
    assert not sample['valid'][0, :, 48:].any()


def test_shared_epoch_clone_avoids_source_cache_duplicate_and_separates_plan(tmp_path):
    manifest, _, config, _, _, _ = _fixture(tmp_path)
    ds = BoundaryDualDataset(manifest, _config(config), epoch=1)
    clone = ds.clone_for_epoch(0)
    for name in ('cache', 'input_boundaries', 'input_skeletons', 'rgb_images', 'continuous_boundaries'):
        assert getattr(clone, name) is getattr(ds, name)
    assert clone.epoch == 0 and ds.epoch == 1
    assert clone.plan is not ds.plan
    original = copy.deepcopy(ds.plan)
    clone.plan[0]['rotation'] = 99
    assert ds.plan == original
    clone.identity['sentinel'] = True
    assert 'sentinel' not in ds.identity
    for group in ds._source_array_groups():
        for values in group.values():
            for array in (values if isinstance(values, tuple) else (values,)):
                assert not array.flags.writeable
    assert ds.cache_receipt['continuous_bytes'] == ds.continuous_boundaries['train_001'].nbytes
    assert ds.cache_receipt['numpy_source_cache_bytes'] > ds.cache_receipt['continuous_bytes']


def test_cache_budget_checked_before_adding_continuous_source(tmp_path):
    manifest, _, config, _, heat, _ = _fixture(tmp_path)
    config = _config(config)
    config['boundary_naf']['continuous_cache_max_bytes'] = heat.nbytes - 1
    with pytest.raises(ValueError, match='exceeds'):
        BoundaryDualDataset(manifest, config)


@pytest.mark.parametrize('change', [dict(input_mode='boundary'), dict(input_contract=None),
    dict(input_normalization={'mode': 'none'}), dict(restored_rgb_manifest='some.json'),
    dict(erasure={'enabled': True}), dict(continuous_cache_max_bytes=True)])
def test_rejects_other_experiments_or_ambiguous_input_contract(tmp_path, change):
    manifest, _, config, _, _, _ = _fixture(tmp_path)
    config = _config(config)
    config['boundary_naf'].update(change)
    with pytest.raises(ValueError):
        BoundaryDualDataset(manifest, config)


def test_collate_two_inputs_without_changing_existing_keys(tmp_path):
    manifest, _, config, _, _, _ = _fixture(tmp_path)
    ds = BoundaryDualDataset(manifest, _config(config))
    batch = collate_boundary_dual([ds[0], ds[1]])
    assert batch['input'].shape == (2, 2, 64, 64)
    assert torch.equal(batch['input'][:, 0:1], batch['input_boundary'])
    assert torch.equal(batch['input'][:, 1:2], batch['continuous_boundary'])
    assert batch['target'].shape == (2, 1, 64, 64)
    with pytest.raises(ValueError):
        collate_boundary_dual([])


def test_sources_verify_continuous_integrity_and_rng_unchanged(tmp_path):
    manifest, _, config, _, _, _ = _fixture(tmp_path)
    py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    ds = BoundaryDualDataset(manifest, _config(config))
    assert py_before == random.getstate()
    after = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(np_before, after))
    assert torch.equal(torch_before, torch.get_rng_state())
    identity = ds.verify_sources()
    assert isinstance(identity, str) and len(identity) == 64
    heat = ds.continuous_boundaries['train_001']
    heat.setflags(write=True)
    heat[0, 0] = .3
    heat.setflags(write=False)
    with pytest.raises(RuntimeError, match='continuous N1 changed'):
        ds.verify_sources()


@pytest.mark.parametrize('dtype,shape', [('float64', (2, 3)), ('float32', (3, 2))])
def test_header_rejects_wrong_dtype_or_shape_before_decompression(tmp_path, dtype, shape):
    path = tmp_path / 'header.npz'
    np.savez_compressed(path, boundary=np.zeros(shape, dtype=dtype))
    with pytest.raises(ValueError):
        continuous_heat_header(path, (2, 3))
