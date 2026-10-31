# -*- coding: utf-8 -*-
"""合法窄走廊中的连续分隔路径监督；选路停止梯度，逐点损失正常反传。

路径是局部结构替代目标，不等价于真实骨架／分水岭已经完成实例分隔。
本模块不读取 GT，也不在 unknown 或候选路径之外制造新的监督点。
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import torch


DEFAULT_STRUCTURE_LOSS_RECIPE = dict(weight=.25, response_margin=.70, mass_weight=.05,
    selection_change_cost=.01, teacher_release_threshold=.65)


def validated_structure_loss_recipe(recipe=None):
    """只校验 loss 负责的键；共用配方中的几何键由 data 层负责校验。"""
    supplied = {} if recipe is None else dict(recipe)
    result = {key: supplied.get(key, value) for key, value in DEFAULT_STRUCTURE_LOSS_RECIPE.items()}
    for key, value in result.items():
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError('Structure loss recipe requires finite numeric ' + key)
        result[key] = float(value)
    for key in ('weight', 'mass_weight', 'selection_change_cost'):
        if result[key] < 0:
            raise ValueError('Structure loss coefficient must be nonnegative: ' + key)
    for key in ('response_margin', 'teacher_release_threshold'):
        if not 0 < result[key] <= 1:
            raise ValueError('Structure response threshold must be in (0,1]: ' + key)
    return result


def _path_arrays(coords_yx, allowed, closed=False, anchor_index=None):
    coords = np.asarray(coords_yx)
    permitted = np.asarray(allowed)
    if (coords.ndim != 3 or coords.shape[-1] != 2 or min(coords.shape[:2]) < 1
            or coords.dtype != np.int32):
        raise ValueError('Structure coordinates must be int32 [stations,candidates,2]')
    if permitted.shape != coords.shape[:2] or permitted.dtype != np.bool_:
        raise ValueError('Structure allowed must be boolean [stations,candidates]')
    if type(closed) is not bool:
        raise ValueError('Structure closed flag must be bool')
    if anchor_index is not None:
        if (isinstance(anchor_index, bool) or not isinstance(anchor_index, Integral)
                or not 0 <= int(anchor_index) < coords.shape[1]):
            raise ValueError('Structure anchor_index must index a first-station candidate')
        anchor_index = int(anchor_index)
    if closed and (anchor_index is None or len(coords) < 3):
        raise ValueError('Closed structure paths require an anchor and at least three stations')
    permitted = permitted.copy()
    if anchor_index is not None:
        first_allowed = bool(permitted[0, anchor_index])
        permitted[0] = False
        permitted[0, anchor_index] = first_allowed
    return coords, permitted, closed, anchor_index


def _neighbors(previous, current):
    # int64 避免非法且禁用的 int32 极端坐标在差分时溢出。
    return np.abs(previous.astype(np.int64)[:, None, :] -
                  current.astype(np.int64)[None, :, :]).max(axis=-1) <= 1


def _lexicographic_choices(primary, secondary, tertiary):
    """每列严格主代价相同时，依次按教师强度／到候选中心距离决胜。"""
    primary_min = primary.min(axis=0)
    secondary = np.where(primary == primary_min[None], secondary, np.inf)
    secondary_min = secondary.min(axis=0)
    tertiary = np.where((primary == primary_min[None]) &
                        (secondary == secondary_min[None]), tertiary, np.inf)
    return tertiary.argmin(axis=0)


def geometry_feasible(coords_yx, allowed, closed=False, anchor_index=None):
    """只读几何的完整可达性；闭环末站必须邻接固定首站锚点。"""
    coords, permitted, closed, anchor_index = _path_arrays(coords_yx, allowed, closed, anchor_index)
    reachable = permitted[0].copy()
    for row in range(1, len(coords)):
        reachable = permitted[row] & (_neighbors(coords[row-1], coords[row]) & reachable[:, None]).any(axis=0)
        if not reachable.any():
            return False
    if closed:
        reachable &= _neighbors(coords[-1], coords[0, anchor_index:anchor_index+1])[:, 0]
    return bool(reachable.any())


def _validated_map(value, name):
    result = np.asarray(value)
    if (result.ndim != 2 or not result.size or result.dtype.kind != 'f'
            or not np.isfinite(result).all() or np.any((result < 0) | (result > 1))):
        raise ValueError(name + ' must be a finite floating HW probability map')
    return result


def select_path(pred_np, target_np, path, margin=.70, change_cost=.01):
    """确定性 CPU 动态规划；返回整链坐标／候选索引，无合法整链时返回 None。"""
    pred = _validated_map(pred_np, 'prediction')
    target = _validated_map(target_np, 'N0 target')
    if pred.shape != target.shape:
        raise ValueError('Structure prediction and N0 target shapes differ')
    params = validated_structure_loss_recipe(dict(response_margin=margin, selection_change_cost=change_cost))
    coords, allowed, closed, anchor = _path_arrays(path['coords_yx'], path['allowed'],
                                                 path['closed'], path.get('anchor_index'))
    height, width = pred.shape
    inside = ((coords[..., 0] >= 0) & (coords[..., 0] < height)
              & (coords[..., 1] >= 0) & (coords[..., 1] < width))
    if np.any(allowed & ~inside):
        raise ValueError('Allowed structure candidates must stay inside the native map')
    yy, xx = coords[..., 0].clip(0, height-1), coords[..., 1].clip(0, width-1)
    values, teachers = pred[yy, xx].astype(np.float64), target[yy, xx].astype(np.float64)
    costs = np.maximum(params['response_margin'] - values, 0.)
    costs += params['selection_change_cost'] * np.maximum(values - teachers, 0.)
    costs[~allowed] = np.inf
    cumulative = costs[0].copy()
    # 这两级仅在主代价严格相等时使用，不向主目标加入位置先验或额外损失。
    teacher_tie = np.where(allowed[0], 1. - teachers[0], np.inf)
    center_distance = np.abs(np.arange(coords.shape[1]) - (coords.shape[1]-1)/2.)
    center_tie = np.where(allowed[0], center_distance, np.inf)
    parents = np.full(costs.shape, -1, np.int32)
    for row in range(1, len(coords)):
        previous = np.where(_neighbors(coords[row-1], coords[row]), cumulative[:, None], np.inf)
        choices = _lexicographic_choices(previous,
            np.broadcast_to(teacher_tie[:, None], previous.shape),
            np.broadcast_to(center_tie[:, None], previous.shape))
        old_teacher_tie, old_center_tie = teacher_tie, center_tie
        cumulative = previous[choices, np.arange(coords.shape[1])] + costs[row]
        teacher_tie = old_teacher_tie[choices] + (1. - teachers[row])
        center_tie = old_center_tie[choices] + center_distance
        teacher_tie[~np.isfinite(cumulative)] = np.inf
        center_tie[~np.isfinite(cumulative)] = np.inf
        parents[row] = choices
    if closed:
        cumulative[~_neighbors(coords[-1], coords[0, anchor:anchor+1])[:, 0]] = np.inf
    endpoint = int(_lexicographic_choices(cumulative[:, None], teacher_tie[:, None],
                                         center_tie[:, None])[0])
    if not np.isfinite(cumulative[endpoint]):
        return None
    indices = np.empty(len(coords), np.int32)
    indices[-1] = endpoint
    for row in range(len(coords)-1, 0, -1):
        indices[row-1] = parents[row, indices[row]]
    selected = coords[np.arange(len(coords)), indices].copy()
    return dict(coords_yx=selected, candidate_indices=indices,
                cost=float(cumulative[endpoint]), stations=len(coords), closed=closed,
                tie_policy='equal_primary_then_max_N0_sum_then_min_abs_candidate_center_distance_then_index')


def _support(batch, teacher_release_threshold=.65, target_cpu=None):
    target, valid = batch['target'], batch['valid']
    if (not torch.is_tensor(target) or target.ndim != 4 or target.shape[1] != 1
            or not target.is_floating_point() or not bool(torch.isfinite(target).all())
            or bool(((target < 0) | (target > 1)).any())):
        raise ValueError('Structure target must be finite B1HW probabilities')
    if (not torch.is_tensor(valid) or valid.shape != target.shape
            or not bool(((valid == 0) | (valid == 1)).all())):
        raise ValueError('Structure valid support must be binary B1HW')
    free = batch['structure_free_mask']
    corridor = batch['structure_corridor']
    for name, value in (('structure_free_mask', free), ('structure_corridor', corridor)):
        if (not torch.is_tensor(value) or value.shape != target.shape or value.dtype != torch.bool):
            raise ValueError(name + ' must be boolean B1HW')
    if bool((corridor & ~valid.bool()).any()) or bool((free & ~corridor).any()):
        raise ValueError('Structure free/corridor support must remain inside valid known pixels')
    paths = batch['structure_paths']
    if (not isinstance(paths, list) or len(paths) != target.shape[0]
            or any(not isinstance(items, list) for items in paths)):
        raise ValueError('structure_paths must be a per-sample list of path lists')
    counts = dict(paths=0, points=0, free_pixels=int(free.sum()))
    valid_cpu = valid.detach().bool().cpu().numpy()[:, 0]
    if target_cpu is None:
        target_cpu = target.detach().float().cpu().numpy()[:, 0]
    corridor_cpu = corridor.detach().cpu().numpy()[:, 0]
    height, width = target.shape[-2:]
    for sample, items in enumerate(paths):
        for path in items:
            coords, allowed, closed, anchor = _path_arrays(path['coords_yx'], path['allowed'],
                                                         path['closed'], path.get('anchor_index'))
            values = coords[allowed]
            if (np.any(values[:, 0] < 0) or np.any(values[:, 0] >= height)
                    or np.any(values[:, 1] < 0) or np.any(values[:, 1] >= width)):
                raise ValueError('Allowed structure candidates are out of bounds')
            if len(values) and not valid_cpu[sample, values[:, 0], values[:, 1]].all():
                raise ValueError('Unknown/invalid pixels cannot be structure candidates')
            if len(values) and not corridor_cpu[sample, values[:, 0], values[:, 1]].all():
                raise ValueError('Structure candidates must remain inside the registered corridor')
            if not geometry_feasible(coords, allowed, closed, anchor):
                raise ValueError('Data supplied a structure chain without a legal complete path')
            if closed:
                ay, ax = coords[0, anchor]
                if target_cpu[sample, ay, ax] < teacher_release_threshold:
                    raise ValueError('Closed structure anchor requires an existing strong N0 response')
            counts['paths'] += 1
            counts['points'] += len(coords)
    return counts


def structure_normalizers(batches, recipe=None):
    """结构支持独立于预测；可用于 micro batch 的精确有效 batch 累积。"""
    params = validated_structure_loss_recipe(recipe)
    result = dict(paths=0, points=0, free_pixels=0)
    if params['weight'] == 0:
        return result
    if isinstance(batches, dict):
        batches = [batches]
    for batch in batches:
        counts = _support(batch, params['teacher_release_threshold'])
        for key in result:
            result[key] += counts[key]
    return result


def structural_boundary_loss(prediction, batch, recipe=None, normalizers=None):
    """逐station hinge + 走廊新增质量惩罚；未知像素的直接梯度严格为零。

    selection_change_cost 只是 stop-grad 的潜路径选路依据，不在反传目标中
    再加一项；位置并非新标签。所有路径坐标拼接后一次 gather 回传。
    """
    params = validated_structure_loss_recipe(recipe)
    if (not torch.is_tensor(prediction) or prediction.ndim != 4 or prediction.shape[1] != 1
            or not prediction.is_floating_point() or not bool(torch.isfinite(prediction).all())
            or bool(((prediction < 0) | (prediction > 1)).any())):
        raise ValueError('Structure prediction must be finite B1HW probabilities')
    zero = prediction.sum() * 0.
    if params['weight'] == 0:
        return zero, dict(loss=0., path_loss=0., mass_loss=0., weighted_path_loss=0.,
            weighted_mass_loss=0., path_count=0, point_count=0, free_pixels=0,
            normalizers=dict(paths=0, points=0, free_pixels=0), selection_calls=0, active=False)
    if prediction.shape != batch['target'].shape:
        raise ValueError('Structure prediction differs from target shape')
    pred_cpu = prediction.detach().float().cpu().numpy()[:, 0]
    target_cpu = batch['target'].detach().float().cpu().numpy()[:, 0]
    counts = _support(batch, params['teacher_release_threshold'], target_cpu)
    denominators = counts if normalizers is None else dict(normalizers)
    for key, count in counts.items():
        denominator = denominators.get(key)
        if type(denominator) is not int or denominator < count:
            raise ValueError('Effective-batch structure denominator is below local support: ' + key)
    target = batch['target'].to(prediction.device)
    free = batch['structure_free_mask'].to(prediction.device)
    if bool((free & (target >= params['teacher_release_threshold'])).any()):
        raise ValueError('Structure free mask cannot release existing strong teacher response')
    calls = 0
    selections = []
    all_indices = []
    for sample, items in enumerate(batch['structure_paths']):
        for path in items:
            chosen = select_path(pred_cpu[sample], target_cpu[sample], path,
                params['response_margin'], params['selection_change_cost'])
            calls += 1
            if chosen is None:
                raise ValueError('Selector found no full path despite fixed feasible data support')
            coords = chosen['coords_yx'].astype(np.int64)
            height, width = prediction.shape[-2:]
            all_indices.append((sample * height + coords[:, 0]) * width + coords[:, 1])
            selections.append(dict(sample=sample, stations=chosen['stations'], closed=chosen['closed'],
                                   selection_cost=chosen['cost']))
    if all_indices:
        indices = torch.as_tensor(np.concatenate(all_indices), device=prediction.device)
        values = prediction.reshape(-1)[indices]
        point_sum = (params['response_margin'] - values).relu().sum()
    else:
        point_sum = zero
    path_loss = point_sum / denominators['points'] if denominators['points'] else zero
    mass_sum = (prediction - target).relu()[free].sum()
    mass_loss = mass_sum / denominators['free_pixels'] if denominators['free_pixels'] else zero
    weighted_path = params['weight'] * path_loss
    weighted_mass = params['weight'] * params['mass_weight'] * mass_loss
    loss = weighted_path + weighted_mass
    return loss, dict(loss=float(loss.detach()), path_loss=float(path_loss.detach()),
        mass_loss=float(mass_loss.detach()), weighted_path_loss=float(weighted_path.detach()),
        weighted_mass_loss=float(weighted_mass.detach()), path_count=counts['paths'],
        point_count=counts['points'], free_pixels=counts['free_pixels'],
        normalizers=dict(denominators), selection_calls=calls, active=True, selections=selections)
