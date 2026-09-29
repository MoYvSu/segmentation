# -*- coding: utf-8 -*-
"""可信域独立于原BCE，几何同步且不改变原数据流。"""
import json
import random

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

import data.affinity_connectivity as module
from data.affinity_connectivity import TrustedManualDataset, resize_trusted, undo_spatial
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.direct_dual_head_dataset import _spatial_transform
from utils.affinity_loss import build_affinity_targets_torch


def paired_cohort(tmp_path, *, repeats=8):
    raw, completed = tmp_path / 'raw', tmp_path / 'completed'
    raw.mkdir()
    completed.mkdir()
    yy, xx = np.mgrid[:24, :32]
    image = np.repeat((50 + xx * 3 + yy)[..., None], 3, axis=2).astype(np.uint8)
    cv2.imwrite(str(raw / 'sample.png'), image)
    (raw / 'sample.json').write_text('not a target', encoding='utf-8')
    instances = np.ones((24, 32), dtype=np.uint16)
    instances[:, 16:] = 2
    instances[4:8, 4:8] = 0
    covered = np.ones_like(instances, dtype=np.uint8)
    covered[instances == 0] = 0
    covered[:, 14:18] = 0  # 自动补缝带有ID，仍接受原BCE，但不得参加拓扑项。
    covered[10, 10] = 0  # 单个原图非原标像素也会排除混合格点。
    np.savez(completed / 'sample_gt.npz', instance_map=instances,
             original_covered=covered, residual_unknown=instances == 0)
    (completed / 'sample_class.json').write_text(json.dumps({1: 1, 2: 0}), encoding='utf-8')
    base = CanonicalBackendDataset(raw, completed, image_size=32, affinity_grid=16)
    degradation = {'profile_probabilities': [0., 1., 0., 0.], 'blur_sigma': [.4, 1.0],
                   'resample_probability': 0., 'resample_scale': [.5, .9],
                   'spatial_blur': {'enabled': True, 'probability': .5,
                                    'grid_size': [3, 5], 'strength': 1., 'levels': 8}}
    paired = PairedDegradationDataset(base, degradation,
        {'enabled': True, 'probability': 1., 'sigma': [.002, .006]}, seed=18, repeats=repeats)
    return paired, covered


def test_area_trust_excludes_mixed_cells_and_letterbox_padding():
    covered = np.ones((8, 12), dtype=np.uint8)
    covered[0, 0] = 0
    trusted = resize_trusted(covered, input_size=12, output_grid=6)
    assert trusted.shape == (1, 6, 6) and trusted.dtype == torch.bool
    assert not trusted[0, 0, 0]
    assert trusted[0, :4].sum() == 23
    assert not trusted[0, 4:].any()


def test_area_is_conservative_when_nearest_would_miss_a_single_gap():
    covered = np.ones((8, 8), dtype=np.uint8)
    covered[3, 3] = 0
    nearest = cv2.resize(covered, (2, 2), interpolation=cv2.INTER_NEAREST)
    assert nearest.all()
    trusted = resize_trusted(covered, input_size=8, output_grid=2)
    assert trusted[0].tolist() == [[False, True], [True, True]]


@pytest.mark.parametrize('hflip', [False, True])
@pytest.mark.parametrize('vflip', [False, True])
@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_inverse_spatial_all_sixteen_combinations(hflip, vflip, k):
    tensor = torch.arange(2 * 3 * 5).reshape(2, 3, 5)
    transformed = _spatial_transform(tensor, hflip, vflip, k)
    assert torch.equal(undo_spatial(transformed, hflip, vflip, k), tensor)
    mask = tensor > 10
    assert torch.equal(undo_spatial(_spatial_transform(mask, hflip, vflip, k), hflip, vflip, k), mask)


def test_wrapper_keeps_all_original_fields_and_random_streams(tmp_path):
    paired, covered = paired_cohort(tmp_path)
    wrapped = TrustedManualDataset(paired)
    wrapped.set_epoch(3)
    assert wrapped.epoch == paired.epoch == 3 and len(wrapped) == len(paired)
    canonical = resize_trusted(covered, input_size=32, output_grid=16)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.get_rng_state().clone()
    for index in range(len(paired)):
        before, after = paired[index], wrapped[index]
        for key, value in before.items():
            assert torch.equal(value, after[key]) if torch.is_tensor(value) else value == after[key], key
        expected = _spatial_transform(canonical, before['horizontal_flip'],
                                      before['vertical_flip'], before['rotation_k'])
        expected &= before['affinity_valid_content'] & (before['affinity_instance_map'] > 0)[None]
        assert torch.equal(after['trusted_pixels'], expected)
        assert after['native_shape'].tolist() == [24, 32]
        assert after['source_kind'] == 'manual'
        assert not after['trusted_pixels'][~before['affinity_valid_content']].any()
    assert len(wrapped._trusted_cache) == 1
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0] and np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]
    batch = default_collate([wrapped[0]])
    assert batch['trusted_pixels'].shape == (1, 1, 16, 16)
    assert batch['native_shape'].shape == (1, 2)


def test_excluding_filled_trust_does_not_remove_original_bce_edges(tmp_path):
    paired, _ = paired_cohort(tmp_path)
    wrapped = TrustedManualDataset(paired)
    original, sample = paired[0], wrapped[0]
    target, valid = build_affinity_targets_torch(original['affinity_instance_map'][None],
                                                original['affinity_valid_content'][None])
    actual, actual_valid = build_affinity_targets_torch(sample['affinity_instance_map'][None],
                                                       sample['affinity_valid_content'][None])
    assert torch.equal(target, actual) and torch.equal(valid, actual_valid)
    rejected = (~sample['trusted_pixels'][0]) & sample['affinity_valid_content'][0] & (sample['affinity_instance_map'] > 0)
    assert rejected.any()
    assert valid[0, :, rejected].any()
    assert not sample['trusted_pixels'][0][sample['affinity_instance_map'] == 0].any()


def test_builder_delegates_original_source_checks_and_leaves_pseudo_unchanged(monkeypatch, tmp_path):
    paired, _ = paired_cohort(tmp_path)
    pseudo, config = object(), {'backend_adaptation': {'expected_manual_sources': 32}}
    original_metadata = {'manual_sources': 32, 'sam2_sources': 64, 'split_policy': 'all_manual_no_holdout'}
    called = []
    def original_builder(received):
        called.append(received)
        return paired, pseudo, original_metadata
    monkeypatch.setattr(module, '_build_backend_datasets', original_builder)
    manual, returned_pseudo, metadata = module.build_datasets(config)
    assert called == [config] and isinstance(manual, TrustedManualDataset)
    assert returned_pseudo is pseudo
    assert all(metadata[key] == value for key, value in original_metadata.items())
    assert 'topology_supervision' not in original_metadata
    assert metadata['topology_supervision']['sam2'] == 'original_bce_only'
    def invalid_manifest(received):
        raise ValueError('SAM2 source manifest no longer matches')
    monkeypatch.setattr(module, '_build_backend_datasets', invalid_manifest)
    with pytest.raises(ValueError, match='SAM2 source manifest'):
        module.build_datasets(config)


def test_missing_original_coverage_fails_instead_of_trusting_filled(tmp_path):
    paired, _ = paired_cohort(tmp_path)
    path = paired.base_dataset.completed_gt_dir / 'sample_gt.npz'
    with np.load(path) as payload:
        values = {key: payload[key] for key in payload.files if key != 'original_covered'}
    np.savez(path, **values)
    with pytest.raises(ValueError, match='Missing original_covered'):
        TrustedManualDataset(paired)[0]


@pytest.mark.parametrize('invalid', [np.ones((2, 3, 4)), np.zeros((0, 4)),
                                     np.array([[.5]]), np.array([[np.nan]])])
def test_invalid_trust_masks_fail(invalid):
    with pytest.raises(ValueError):
        resize_trusted(invalid)
