# -*- coding: utf-8 -*-
"""真实cubical PH的坐标、缺口梯度、假孔梯度与unknown隔离检查。"""

import numpy as np
import pytest
import torch

pytest.importorskip('gudhi')

from utils.boundary_topology import BoundaryTopologyLoss, _diagram_matching, _finite_h1, eligible_origins


def ring(size=32):
    value = torch.zeros(1, 1, size, size)
    value[:, :, 8:24, 8] = 1
    value[:, :, 8:24, 23] = 1
    value[:, :, 8, 8:24] = 1
    value[:, :, 23, 8:24] = 1
    return value


def criterion(size=32, **kwargs):
    return BoundaryTopologyLoss(tile_size=size, border_margin=2, **kwargs)


def test_partial_gap_below_half_gets_upward_gradient():
    target = ring()
    probability = target * 0.85 + 0.05
    probability[0, 0, 8, 15] = 0.2
    probability.requires_grad_()
    loss, audit = criterion()(probability, target, torch.ones_like(target))
    loss.backward()
    assert audit['gt_h1'] == audit['matched_h1'] == 1
    assert audit['critical_points'] == 2
    # 梯度下降在真正弱gap处抬高概率，而非只降低其它假环。
    assert probability.grad[0, 0, 8, 15].item() < 0
    assert probability.grad[0, 0, 8, 15].abs().item() > 0.1
    assert probability.grad.abs().count_nonzero().item() <= 2
    updated = (probability.detach() - 0.1 * probability.grad).clamp(0, 1)
    new_loss, _ = criterion()(updated, target, torch.ones_like(target))
    assert new_loss.item() < loss.item()


def test_false_ring_persistence_is_pushed_toward_diagonal():
    target = torch.zeros_like(ring())
    probability = (ring() * 0.85 + 0.05).requires_grad_()
    loss, audit = criterion()(probability, target, torch.ones_like(target))
    loss.backward()
    assert audit['gt_h1'] == audit['matched_h1'] == 0
    assert audit['spurious_h1'] == 1
    assert audit['critical_points'] == 2
    # 对角线惩罚同时减小出生/死亡差；BCE负责决定具体删线还是填孔。
    assert (probability.grad[ring().bool()] > 0).any()
    assert (probability.grad[~ring().bool()] < 0).any()
    updated = (probability.detach() - 0.1 * probability.grad).clamp(0, 1)
    new_loss, _ = criterion()(updated, target, torch.ones_like(target))
    assert new_loss.item() < loss.item()


def test_exact_target_has_zero_ph_loss():
    target = ring()
    probability = target.clone().requires_grad_()
    loss, audit = criterion()(probability, target, torch.ones_like(target))
    loss.backward()
    assert audit['matched_h1'] == 1
    assert loss.item() == 0
    assert probability.grad.abs().sum().item() == 0


def test_no_ph_pair_for_completely_flat_prediction_is_audited():
    target = ring()
    probability = torch.full_like(target, 0.05, requires_grad=True)
    loss, audit = criterion()(probability, target, torch.ones_like(target))
    loss.backward()
    assert audit['gt_h1'] == 1
    assert audit['unmatched_target_h1'] == 1
    assert audit['critical_points'] == 0
    assert audit['missing_target_constant_loss'] > 0
    assert not audit['has_differentiable_signal']
    assert probability.grad.abs().sum().item() == 0


def test_unknown_pixel_rejects_whole_tile_not_filled_with_background():
    target = ring()
    probability = torch.full_like(target, 0.5, requires_grad=True)
    valid = torch.ones_like(target)
    valid[0, 0, 15, 15] = 0
    loss, audit = criterion()(probability, target, valid)
    loss.backward()
    assert audit['eligible_tiles'] == 0
    assert audit['selected_tiles'] == 0
    assert audit['skipped_unknown_tiles'] == 1
    assert loss.item() == 0
    assert probability.grad.abs().sum().item() == 0


def test_half_stride_finds_interior_tile_despite_valid_outer_margin():
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 24:40, 24] = 1
    target[:, :, 24:40, 39] = 1
    target[:, :, 24, 24:40] = 1
    target[:, :, 39, 24:40] = 1
    valid = torch.zeros_like(target)
    valid[:, :, 4:-4, 4:-4] = 1
    probability = (target * 0.85 + 0.05).requires_grad_()
    _, audit = criterion()(probability, target, valid, step=0)
    assert audit['candidate_tiles'] == 33 ** 2
    assert audit['all_legal_origins'] == 25 ** 2
    assert audit['eligible_tiles'] == 9 and audit['selected_tiles'] == 1
    assert audit['selected_origins'] == [[0, 16, 16]]
    assert audit['matched_h1'] == 1


def test_large_window_can_align_with_valid_bbox_instead_of_padding():
    valid = np.zeros((256, 256), dtype=bool)
    valid[4:252, 4:252] = True
    origins = eligible_origins(valid, 224, 32)
    assert origins[0] == (16, 16)
    assert (4, 4) in origins and (28, 28) in origins
    assert len(origins) == 9
    assert all(valid[y:y+224, x:x+224].all() for y, x in origins)
    assert all(4 <= y <= 28 and 4 <= x <= 28 for y, x in origins)


def test_disconnected_valid_origin_regions_are_both_represented():
    valid = np.zeros((64, 96), dtype=bool)
    valid[3:30, 5:32] = True
    valid[34:61, 63:90] = True
    origins = eligible_origins(valid, 24, 24)
    assert any(3 <= y <= 6 and 5 <= x <= 8 for y, x in origins)
    assert any(34 <= y <= 37 and 63 <= x <= 66 for y, x in origins)
    assert all(valid[y:y+24, x:x+24].all() for y, x in origins)


def test_eligible_origins_are_content_based_cached_and_deterministic():
    valid = torch.ones(50, 70)
    valid[:3] = 0
    first = eligible_origins(valid, 32, 16)
    second = eligible_origins(valid.clone().bool(), 32, 16)
    assert first == second
    first.clear()
    assert eligible_origins(valid, 32, 16) == second


def test_integral_origin_legality_matches_brute_force_with_irregular_unknown():
    from scipy import ndimage
    rng = np.random.default_rng(13)
    valid = rng.random((31, 43)) > 0.015
    valid[:2] = False
    valid[:, -3:] = False
    tile = 9
    legal = np.array([[valid[y:y+tile, x:x+tile].all() for x in range(43-tile+1)]
                      for y in range(31-tile+1)])
    origins = eligible_origins(valid, tile, 7)
    assert origins and all(legal[y, x] for y, x in origins)
    labels, component_count = ndimage.label(legal)
    assert {labels[y, x] for y, x in origins} == set(range(1, component_count+1))


def test_origin_rotation_reuses_only_real_legal_windows():
    target = torch.zeros(1, 1, 64, 64)
    valid = torch.zeros_like(target)
    valid[:, :, 4:-4, 4:-4] = 1
    fn = criterion()
    origins = []
    for step in range(9):
        _, audit = fn(torch.full_like(target, 0.2, requires_grad=True), target, valid, step=step)
        image, y, x = audit['selected_origins'][0]
        assert valid[image, 0, y:y+32, x:x+32].all()
        origins.append((y, x))
    assert len(set(origins)) == 9


def test_fortran_mapping_on_non_square_tile_matches_actual_values():
    p = np.zeros((25, 37), dtype=np.float64) + 0.03
    p[5:20, 7] = p[5:20, 27] = 0.9
    p[5, 7:28] = p[19, 7:28] = 0.9
    p[5, 18] = 0.2
    result = _finite_h1(p, min_persistence=0.01, border_margin=1, max_features=16)
    assert len(result['diagram']) == 1
    row, col = result['rows'][0], result['cols'][0]
    assert (int(row[0]), int(col[0])) == (5, 18)
    np.testing.assert_allclose(result['diagram'][0], [1-p[row[0], col[0]], 1-p[row[1], col[1]]])


def test_matching_keeps_required_weak_loop_and_audits_missing_points():
    pred = np.array([[0.45, 0.55]])
    target = np.array([[0.0, 1.0]])
    matches, spurious, missing = _diagram_matching(pred, target)
    assert matches == [(0, 0)] and spurious == missing == []
    matches, spurious, missing = _diagram_matching(np.empty((0, 2)), target)
    assert matches == [] and spurious == [] and missing == [0]
    # 这类弱环在自由Wasserstein下会被删掉；GT数量约束必须保留并强化它。
    matches, spurious, missing = _diagram_matching(np.array([[0.8, 0.95]]), target)
    assert matches == [(0, 0)] and spurious == missing == []


def test_float_valid_boolean_valid_and_cache_have_same_loss():
    target = ring()
    probability = (target * 0.85 + 0.05).requires_grad_()
    fn = criterion()
    a, aa = fn(probability, target, torch.ones_like(target))
    b, bb = fn(probability, target, torch.ones_like(target).bool())
    assert a.item() == b.item()
    assert aa['target_cache_hits'] == 0 and bb['target_cache_hits'] == 1


def test_soft_target_and_nonbinary_valid_are_rejected():
    target = ring()
    with pytest.raises(ValueError, match='二值'):
        criterion()(torch.ones_like(target) * 0.5, target * 0.9, torch.ones_like(target))
    with pytest.raises(ValueError, match='valid必须'):
        criterion()(torch.ones_like(target) * 0.5, target, torch.ones_like(target) * 0.5)
