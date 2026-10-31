# -*- coding: utf-8 -*-
"""可移动界面保持可信内核/unknown/随机流隔离，完整路径不能跨像素跳跃。"""
import json

import numpy as np
import pytest
import torch

from data.boundary_context import BoundaryContextDataset
from data.boundary_restore import reliable_region
from data.boundary_structure import (BoundaryStructureDataset, build_boundary_structure,
    collate_boundary_structure, feasible_candidate_mask, validated_structure_geometry)
from models.boundary_structure_loss import geometry_feasible
from test_boundary_context_data import fixture_files


def vertical_case(shape=(128, 160)):
    ids = np.ones(shape, np.int32)
    ids[:, shape[1]//2:] = 2
    original = ids > 0
    heat = np.full(shape, .1, np.float32)
    valid = reliable_region(ids, 8)
    return ids, original, heat, valid


def test_complete_open_interface_without_old_strong_anchors_and_exact_protection():
    ids, original, heat, valid = vertical_case()
    before = [value.copy() for value in (ids, original, heat, valid)]
    built = build_boundary_structure(ids, original, heat, valid)
    assert built['paths']
    assert built['meta']['counts']['open_paths'] >= 1
    assert not (built['corridor'] & built['cores']).any()
    assert not (built['corridor'] & ~valid).any()
    assert not (built['free_mask'] & ~valid).any()
    for path in built['paths']:
        assert len(path['coords_yx']) >= 8
        assert geometry_feasible(path['coords_yx'], path['allowed'], path['closed'], path['anchor_index'])
        assert not path['closed'] and path['anchor_index'] is None
        assert path['meta']['points'] > 64  # 保留完整界面，不能人为切成短缺段。
    for a, b in zip(before, (ids, original, heat, valid)):
        np.testing.assert_array_equal(a, b)
    json.dumps(built['meta'], allow_nan=False)


def test_unknown_and_missing_original_core_cannot_create_supervision():
    ids, original, heat, valid = vertical_case()
    original[:, ids.shape[1]//2:] = False
    built = build_boundary_structure(ids, original, heat, valid)
    assert not built['paths'] and not built['free_mask'].any()
    assert built['meta']['fallback_original_pair']
    assert built['meta']['rejected']['pair_without_two_original_deep_cores'] == 1
    ids, original, heat, valid = vertical_case()
    ids[50:70] = 0; original[50:70] = False
    valid = reliable_region(ids, 8)
    built = build_boundary_structure(ids, original, heat, valid)
    assert built['paths']
    assert not built['corridor'][ids == 0].any()
    assert not built['free_mask'][ids == 0].any()
    # unknown 将界面自然截断为两条完整 local chain，不能跨越空白连起来。
    for path in built['paths']:
        coords = path['coords_yx'][path['allowed']]
        assert np.all((coords[:, 0] < 50) | (coords[:, 0] >= 70))


def test_closed_loop_needs_actual_strong_clean_anchor_and_remains_closed():
    ids = np.ones((160, 160), np.int32)
    ids[42:118, 42:118] = 2
    original = ids > 0; valid = reliable_region(ids, 8)
    heat = np.full(ids.shape, .1, np.float32)
    weak = build_boundary_structure(ids, original, heat, valid)
    assert not weak['paths']
    assert weak['meta']['rejected']['closed_without_strong_clean_n0_anchor'] >= 1
    heat[42, 72] = .9
    built = build_boundary_structure(ids, original, heat, valid)
    assert built['paths'] and all(p['closed'] for p in built['paths'])
    for path in built['paths']:
        anchor = path['anchor_index']
        assert path['allowed'][0].sum() == 1 and path['allowed'][0, anchor]
        y, x = path['coords_yx'][0, anchor]
        assert heat[y, x] >= .65
        assert geometry_feasible(path['coords_yx'], path['allowed'], True, anchor)
    assert not built['free_mask'][42, 72]


def test_candidate_reachability_uses_actual_eight_neighbor_pixels_not_column_offsets():
    coords = np.array([[[0,0],[0,5]], [[1,0],[1,5]], [[2,0],[2,8]]], np.int32)
    allowed = np.ones((3,2), bool)
    kept = feasible_candidate_mask(coords, allowed)
    assert kept[:, 0].all() and not kept[:, 1].any()
    coords[1, 0] = [10, 0]
    assert not feasible_candidate_mask(coords, allowed).any()
    assert not geometry_feasible(coords, allowed)


def test_closed_path_cannot_switch_to_another_first_anchor_on_return():
    coords = np.array([[[0,0],[0,5]], [[1,0],[1,5]], [[2,0],[2,5]]], np.int32)
    allowed = np.ones((3,2), bool)
    assert not feasible_candidate_mask(coords, allowed, True, 0).any()
    assert not geometry_feasible(coords, allowed, True, 0)
    coords[-1, 0] = [1, 1]
    kept = feasible_candidate_mask(coords, allowed, True, 0)
    assert kept[:, 0].all() and not kept[:, 1].any()
    assert geometry_feasible(coords, kept, True, 0)


@pytest.mark.parametrize('rotation', range(4))
@pytest.mark.parametrize('flip', [False, True])
def test_geometry_for_all_rotations_and_flips_is_valid_and_matches_support(rotation, flip):
    ids, original, heat, _ = vertical_case()
    original[20:24, 25:35] = False
    arrays = []
    for value in (ids, original, heat):
        moved = np.rot90(value, rotation)
        arrays.append(np.ascontiguousarray(moved[:, ::-1] if flip else moved))
    ids, original, heat = arrays
    valid = reliable_region(ids, 8)
    built = build_boundary_structure(ids, original, heat, valid)
    assert built['paths']
    assert not (built['corridor'] & (~valid | built['cores'])).any()
    for path in built['paths']:
        assert geometry_feasible(path['coords_yx'], path['allowed'], path['closed'], path['anchor_index'])


def test_dataset_keeps_old_pairs_plans_geometry_rng_and_padding_exact(tmp_path):
    manifest, real, _ = fixture_files(tmp_path, (88, 112))
    common = dict(real_manifest_path=real, expected_sources=1, crop_size=128,
                  views_per_source=2, seed=37)
    old = BoundaryContextDataset(manifest, **common)
    new = BoundaryStructureDataset(manifest, **common)
    before = np.random.get_state()
    for epoch in (1, 3, 5):
        old.set_epoch(epoch); new.set_epoch(epoch)
        assert old.plan == new.plan and old.plan_receipt() == new.plan_receipt()
        for index in range(len(new)):
            a, b = old[index], new[index]
            for key in ('input','target','valid','gap_mask','noise_mask','identity','context_rgb'):
                assert torch.equal(a[key], b[key])
            assert a['meta'] == {k:v for k,v in b['meta'].items() if k != 'structure'}
            h,w = b['meta']['geometry']['content_shape']
            for key in ('structure_corridor','structure_free_mask','structure_cores'):
                assert b[key].dtype == torch.bool
                assert not b[key][:,h:,:].any() and not b[key][:,:,w:].any()
            assert not (b['structure_free_mask'] & ~b['valid'].bool()).any()
            json.dumps(b['meta'], allow_nan=False)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    new.verify_sources()
    batch = collate_boundary_structure([new[0],new[1]])
    assert batch['structure_corridor'].shape == (2,1,128,128)
    assert len(batch['structure_paths']) == 2
    assert batch['structure_cores'].dtype == torch.bool


def test_no_candidates_falls_back_with_empty_masks_and_deterministic_metadata():
    ids, original, heat, valid = vertical_case((48,48))
    a = build_boundary_structure(ids, original, heat, valid)
    b = build_boundary_structure(ids, original, heat, valid)
    assert not a['paths'] and not a['corridor'].any() and not a['free_mask'].any()
    assert a['meta'] == b['meta']


def test_recipe_unknown_invalid_and_loss_keys_do_not_change_geometry():
    baseline = validated_structure_geometry()
    assert validated_structure_geometry(dict(weight=.5,response_margin=.8)) == baseline
    for bad in ({'coridor_radius':3}, {'core_radius':True}, {'min_path_points':0},
                {'closed_anchor_threshold':float('nan')}):
        with pytest.raises(ValueError):
            validated_structure_geometry(bad)
