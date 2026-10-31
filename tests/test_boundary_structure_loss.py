# -*- coding: utf-8 -*-
"""结构路径损失的反投机、几何及有效batch梯度合同。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from models.boundary_context import BoundaryContextNet
from models.boundary_structure_loss import (geometry_feasible, select_path,
    structural_boundary_loss, structure_normalizers, validated_structure_loss_recipe)


def straight_path(height=32, width=40, offset=0):
    yy = np.arange(4, height-4, dtype=np.int32)
    xx = np.arange(4, width-4, dtype=np.int32)
    coords = np.stack(np.broadcast_arrays(yy[:, None], xx[None, :]), axis=-1).astype(np.int32)
    coords[..., 1] += offset
    allowed = ((coords[..., 1] >= 0) & (coords[..., 1] < width))
    return dict(coords_yx=coords, allowed=allowed, closed=False, anchor_index=None, meta={})


def batch_for(paths=None, n=1, size=(32, 40), target=.1):
    if paths is None:
        paths = [[straight_path(*size)] for _ in range(n)]
    teacher = torch.full((n, 1, *size), float(target))
    corridor = torch.zeros_like(teacher, dtype=torch.bool)
    for i, items in enumerate(paths):
        for path in items:
            coords = path['coords_yx'][path['allowed']]
            corridor[i, 0, coords[:, 0], coords[:, 1]] = True
    return dict(target=teacher, valid=torch.ones_like(teacher), structure_paths=paths,
        structure_corridor=corridor, structure_free_mask=corridor & (teacher < .65))


def test_missing_gap_true_model_has_nonzero_parameter_gradient():
    torch.set_num_threads(1)
    batch = batch_for()
    teacher = batch['target']
    teacher[:, :, :, 20] = .8
    teacher[:, :, 12:20, 20] = .1
    batch['structure_free_mask'] &= teacher < .65
    model = BoundaryContextNet(width=4)
    model.train()
    output = model(teacher, torch.zeros(1, 3, 32, 40))
    loss, stats = structural_boundary_loss(output, batch)
    loss.backward()
    assert stats['path_loss'] > 0
    assert model.correction.weight.grad is not None
    assert model.correction.weight.grad.abs().sum() > 0
    assert model.correction.bias.grad.abs().sum() > 0


def test_offset_complete_line_is_reused_without_parallel_supervision():
    batch = batch_for()
    batch['target'][:, :, :, 30] = .8  # 对照中心20偏移10原生像素
    batch['structure_free_mask'] &= batch['target'] < .65
    pred = batch['target'].clone().requires_grad_()
    loss, stats = structural_boundary_loss(pred, batch)
    assert loss.item() == 0 and stats['path_loss'] == 0
    loss.backward()
    assert torch.count_nonzero(pred.grad) == 0
    selected = select_path(pred.detach().numpy()[0, 0], batch['target'].numpy()[0, 0],
                           batch['structure_paths'][0][0])
    assert (selected['coords_yx'][:, 1] == 30).all()


def test_alternating_isolated_high_points_do_not_fake_a_continuous_line():
    batch = batch_for()
    pred = batch['target'].clone()
    for y in range(4, 28):
        pred[0, 0, y, 10 if y % 2 else 30] = .8
    _, stats = structural_boundary_loss(pred, batch)
    assert stats['path_loss'] > .2


def test_full_corridor_brightening_costs_more_than_a_thin_complete_line():
    batch = batch_for()
    thin = batch['target'].clone()
    thin[:, :, :, 20] = .7
    bright = batch['target'].clone()
    bright[batch['structure_corridor']] = .7
    thin_loss, thin_stats = structural_boundary_loss(thin, batch)
    bright_loss, bright_stats = structural_boundary_loss(bright, batch)
    assert thin_stats['path_loss'] == 0 and bright_stats['path_loss'] == 0
    assert bright_loss > thin_loss


def test_unknown_candidates_are_rejected_and_no_direct_unknown_gradient():
    batch = batch_for()
    batch['valid'][:, :, 0] = 0
    pred = batch['target'].clone().requires_grad_()
    loss, _ = structural_boundary_loss(pred, batch)
    loss.backward()
    assert torch.count_nonzero(pred.grad[:, :, 0]) == 0
    batch['valid'][:, :, 5, 20] = 0
    batch['structure_corridor'][:, :, 5, 20] = False
    batch['structure_free_mask'][:, :, 5, 20] = False
    with pytest.raises(ValueError, match='Unknown/invalid'):
        structural_boundary_loss(pred, batch)


def test_disconnected_geometry_is_not_silently_removed_from_denominator():
    path = straight_path()
    path['allowed'][10] = False
    assert not geometry_feasible(path['coords_yx'], path['allowed'])
    assert select_path(np.full((32, 40), .1), np.full((32, 40), .1), path) is None
    batch = batch_for([[path]])
    with pytest.raises(ValueError, match='without a legal complete path'):
        structure_normalizers([batch])


def closed_path():
    coords = np.array([[4, 4], [4, 5], [5, 5], [5, 4]], np.int32)[:, None]
    return dict(coords_yx=coords, allowed=np.ones((4, 1), bool), closed=True,
                anchor_index=0, meta={})


def test_closed_path_requires_real_closure_and_strong_fixed_anchor():
    path = closed_path()
    assert geometry_feasible(path['coords_yx'], path['allowed'], True, 0)
    batch = batch_for([[path]])
    with pytest.raises(ValueError, match='existing strong N0'):
        structure_normalizers([batch])
    batch['target'][0, 0, 4, 4] = .8
    batch['structure_free_mask'] &= batch['target'] < .65
    assert structure_normalizers([batch])['points'] == 4
    broken = deepcopy(path)
    broken['coords_yx'] = np.array([[4, 4], [4, 5], [4, 6], [4, 7]], np.int32)[:, None]
    assert not geometry_feasible(broken['coords_yx'], broken['allowed'], True, 0)


def test_closed_anchor_cannot_switch_to_a_different_first_candidate():
    path = closed_path()
    path['coords_yx'] = np.concatenate([path['coords_yx'], path['coords_yx'] + 12], axis=1)
    path['allowed'] = np.ones((4, 2), bool)
    pred = np.full((32, 40), .1)
    target = pred.copy()
    for y, x in path['coords_yx'][:, 1]:
        pred[y, x] = .9
    chosen = select_path(pred, target, path)
    assert (chosen['candidate_indices'] == 0).all()


def test_empty_and_weight_zero_are_exact_differentiable_zero_without_selection():
    pred = torch.full((1, 1, 32, 40), .1, requires_grad=True)
    batch = batch_for([[]])
    empty, stats = structural_boundary_loss(pred, batch)
    assert empty.item() == 0 and stats['selection_calls'] == 0
    # weight0必须连不存在/无效的path字典也不访问，兼容原控制batch。
    off, stats = structural_boundary_loss(pred, {}, dict(weight=0))
    off.backward()
    assert off.item() == 0 and stats['selection_calls'] == 0 and not stats['active']
    assert torch.count_nonzero(pred.grad) == 0
    assert structure_normalizers([{}], dict(weight=0)) == dict(paths=0, points=0, free_pixels=0)


def slice_batch(batch, start, stop):
    return {k: (v[start:stop] if torch.is_tensor(v) or isinstance(v, list) else v)
            for k, v in batch.items()}


def test_batch_four_equals_two_micro_batches_in_loss_and_gradient():
    batch = batch_for(n=4)
    pred = (batch['target'] + torch.linspace(0, .2, 4).reshape(4, 1, 1, 1)).requires_grad_()
    normalizers = structure_normalizers([batch])
    whole, _ = structural_boundary_loss(pred, batch, normalizers=normalizers)
    whole.backward()
    expected_grad = pred.grad.clone()
    micro = pred.detach().clone().requires_grad_()
    losses = []
    batches = [slice_batch(batch, 0, 2), slice_batch(batch, 2, 4)]
    assert structure_normalizers(batches) == normalizers
    for start, piece in zip((0, 2), batches):
        value, _ = structural_boundary_loss(micro[start:start+2], piece, normalizers=normalizers)
        losses.append(value.detach())
        value.backward()
    assert torch.allclose(torch.stack(losses).sum(), whole.detach(), atol=1e-7, rtol=1e-6)
    assert torch.equal(micro.grad, expected_grad)


def test_selector_is_repeatable_and_disabled_out_of_bounds_is_safe():
    path = straight_path()
    path['coords_yx'][:, 0] = -1000
    path['allowed'][:, 0] = False
    pred = np.random.default_rng(1).uniform(.1, .9, (32, 40)).astype(np.float32)
    target = np.full_like(pred, .1)
    one, two = select_path(pred, target, path), select_path(pred, target, path)
    assert np.array_equal(one['candidate_indices'], two['candidate_indices'])
    assert one['cost'] == two['cost']
    path['allowed'][:, 0] = True
    with pytest.raises(ValueError, match='inside the native'):
        select_path(pred, target, path)


def test_equal_cost_tie_prefers_center_instead_of_negative_offset_edge():
    path = straight_path()
    pred = np.full((32, 40), .1, np.float32)
    chosen = select_path(pred, pred, path)
    # 本例偶数候选的两个中心等距，最后才用稳定index决胜。
    assert (chosen['candidate_indices'] == (path['coords_yx'].shape[1]-1)//2).all()


def test_equal_cost_tie_prefers_existing_teacher_peak_before_center():
    path = straight_path()
    pred = np.full((32, 40), .1, np.float32)
    target = pred.copy()
    target[:, 30] = .8
    chosen = select_path(pred, target, path)
    assert (chosen['coords_yx'][:, 1] == 30).all()


def test_effective_denominator_and_strong_release_errors_are_explicit():
    batch = batch_for()
    with pytest.raises(ValueError, match='denominator'):
        structural_boundary_loss(batch['target'], batch,
                                 normalizers=dict(paths=0, points=0, free_pixels=0))
    batch['target'][batch['structure_free_mask']] = .8
    with pytest.raises(ValueError, match='release existing strong'):
        structural_boundary_loss(batch['target'], batch)


@pytest.mark.parametrize('recipe', [dict(weight=-1), dict(mass_weight=-.1),
    dict(response_margin=0), dict(selection_change_cost=float('nan')),
    dict(teacher_release_threshold=2)])
def test_bad_recipes_are_rejected(recipe):
    with pytest.raises(ValueError):
        validated_structure_loss_recipe(recipe)
