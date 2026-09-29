# -*- coding: utf-8 -*-
"""整图 D5a 后的尺度视图：有效内容、同步监督及可复现性。"""
import json
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from data.semantic_scale import (
    apply_scale_batch, apply_scale_image, apply_scale_target, choose_scale_view,
)


def scene():
    image = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1, 1, 1024, 1024)
    image = image.expand(1, 3, 1024, 1024) / (1024 * 1024)
    valid = torch.zeros(1, 1, 1024, 1024, dtype=torch.bool)
    valid[..., 52:972, 116:908] = True
    return image, valid


def local_spec():
    return {'mode': 'local', 'image_size': [1024, 1024], 'crop': [128, 256, 512, 512]}


def test_full_branch_is_bitwise_identity_including_batch_fields():
    image, valid = scene()
    batch = {'image': image, 'semantic_target': torch.rand(1, 1, 256, 256),
             'semantic_instance_map': torch.arange(512 * 512).reshape(1, 512, 512),
             'semantic_valid_content': valid, 'valid_content': valid,
             'affinity_instance_map': torch.ones(1, 512, 512, dtype=torch.int64), 'name': 'sample'}
    spec = choose_scale_view(image, valid, {}, 17, force_mode='full')
    result = apply_scale_batch(batch, spec)
    assert result is not batch
    assert all(result[key] is value for key, value in batch.items())
    assert torch.equal(apply_scale_image(image, spec), image)


def test_image_matches_explicit_bilinear_crop_and_backpropagates():
    image, _ = scene()
    image = image.clone().requires_grad_()
    scaled = apply_scale_image(image, local_spec())
    expected = F.interpolate(image[..., 128:640, 256:768], (1024, 1024),
                             mode='bilinear', align_corners=False)
    assert torch.equal(scaled, expected)
    scaled.mean().backward()
    assert image.grad[..., 128:640, 256:768].sum() > 0
    assert image.grad[..., :128, :].sum() == 0


@pytest.mark.parametrize('grid', [256, 512, 1024])
@pytest.mark.parametrize('rank', [2, 3, 4])
def test_target_crop_preserves_ids_ignore_and_grid_alignment(grid, rank):
    target = torch.zeros(grid, grid, dtype=torch.int64)
    target[:grid // 2, :grid // 2] = -1
    target[:grid // 2, grid // 2:] = 65535
    target[grid // 2:, :grid // 2] = 2**40 + 1
    target[grid // 2:, grid // 2:] = 17
    target = target.reshape(*([1] * (rank - 2)), grid, grid)
    scaled = apply_scale_target(target, local_spec())
    y, x, size = grid // 8, grid // 4, grid // 2
    expected = target[..., y:y + size, x:x + size].repeat_interleave(2, -2).repeat_interleave(2, -1)
    assert scaled.shape == target.shape and scaled.dtype == target.dtype
    assert torch.equal(scaled, expected)
    assert set(scaled.unique().tolist()) == {-1, 65535, 2**40 + 1, 17}


def test_semantic_targets_are_consistent_across_all_three_grids():
    small = torch.arange(256 * 256).reshape(1, 256, 256)
    result = apply_scale_target(small, local_spec())
    for factor in (2, 4):
        large = small.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
        actual = apply_scale_target(large, local_spec())
        assert torch.equal(actual, result.repeat_interleave(factor, -2).repeat_interleave(factor, -1))


@pytest.mark.parametrize('rotation', [0, 1, 2, 3])
def test_crop_avoids_padding_after_flips_rotations_and_lower_grid(rotation):
    image, valid = scene()
    valid = torch.rot90(valid.flip(-1), rotation, (-2, -1))
    for mask in (valid, valid[..., ::4, ::4]):
        for seed in range(8):
            spec = choose_scale_view(image, mask, {}, seed, force_mode='local')
            top, left, height, width = spec['crop']
            assert spec['mode'] == 'local' and height == width == 512
            assert top % 4 == left % 4 == 0
            assert valid[..., top:top + height, left:left + width].all()
            assert apply_scale_target(valid, spec).all()


def test_nearest_prior_target_and_mask_remain_paired_without_new_labels():
    target = torch.zeros(1, 1, 256, 256)
    target[..., 50:130, 70:170] = .93
    mask = target > .5
    target[..., 40:50, 70:170] = .5
    scaled_target = apply_scale_target(target, local_spec())
    scaled_mask = apply_scale_target(mask, local_spec())
    assert scaled_mask.dtype == torch.bool
    assert torch.equal(scaled_mask, scaled_target == torch.tensor(.93))
    assert set(scaled_target.unique().tolist()) == set(target.unique().tolist())


def test_local_batch_does_not_mutate_source_or_affinity():
    image, valid = scene()
    target = torch.arange(256 * 256).reshape(1, 1, 256, 256)
    batch = {'image': image, 'semantic_target': target, 'semantic_boundary': target > 32,
             'semantic_instance_map': target[:, 0], 'semantic_valid_content': valid,
             'valid_content': valid, 'affinity_instance_map': target, 'metadata': {'name': 'sample'}}
    original = {k: v.clone() for k, v in batch.items() if torch.is_tensor(v)}
    result = apply_scale_batch(batch, local_spec())
    assert result['affinity_instance_map'] is batch['affinity_instance_map']
    assert result['metadata'] is batch['metadata']
    assert result['semantic_target'] is not batch['semantic_target']
    assert all(torch.equal(batch[k], v) for k, v in original.items())


def test_independent_rng_and_serializable_reproducible_spec():
    image, valid = scene()
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    specs = [choose_scale_view(image, valid, {}, seed) for seed in range(16)]
    assert specs == [choose_scale_view(image, valid, {}, seed) for seed in range(16)]
    assert json.loads(json.dumps(specs)) == specs
    assert {s['mode'] for s in specs} == {'full', 'local'}
    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == random.getstate()


def test_too_narrow_content_keeps_full_scale_instead_of_including_padding():
    image, valid = scene()
    valid.zero_()
    valid[..., 100:924, 340:640] = True
    spec = choose_scale_view(image, valid, {}, 10, force_mode='local')
    assert spec['mode'] == 'full' and spec['reason'] == 'content_smaller_than_crop'
    assert apply_scale_image(image, spec) is image


def test_disabled_setting_keeps_identity_even_with_force_local():
    image, valid = scene()
    spec = choose_scale_view(image, valid, {'enabled': False}, 10, force_mode='local')
    assert spec['mode'] == 'full' and spec['reason'] == 'disabled'


def test_gt_coverage_with_holes_is_not_accepted_as_content_mask():
    image, valid = scene()
    valid[..., 200, 200] = False
    with pytest.raises(ValueError, match='not GT coverage'):
        choose_scale_view(image, valid, {}, 10)


def test_unaligned_target_coordinates_are_rejected_instead_of_rounding():
    spec = dict(local_spec(), crop=[129, 256, 512, 512])
    with pytest.raises(ValueError, match='align exactly'):
        apply_scale_target(torch.zeros(256, 256), spec)
