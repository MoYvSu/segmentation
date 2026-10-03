# -*- coding: utf-8 -*-
"""CPU已知GT域内的短边最宽路径；返回可直接索引真实affinity的瓶颈。"""
from __future__ import annotations

import heapq
from numbers import Integral

import numpy as np

from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS


def _point(value, name):
    if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in value)):
        raise ValueError(name + ' must contain two integer coordinates [y,x]')
    return tuple(map(int, value))


def _offsets(value):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        raise ValueError('Exactly four canonical short offsets are required')
    result = tuple(_point(offset, 'offset') for offset in value)
    if any(max(abs(dy), abs(dx)) != 1 for dy, dx in result):
        raise ValueError('Offsets must be nonzero unit axial/diagonal links')
    undirected = {min((dy, dx), (-dy, -dx)) for dy, dx in result}
    if len(undirected) != 4:
        raise ValueError('Offsets must represent four distinct undirected short links')
    return result


def constrained_bottleneck_path(probabilities, gt, valid, start, end, direction, *,
                                allowed=None, edge_valid=None,
                                offsets=DEFAULT_AFFINITY_OFFSETS[:4]):
    """用maximin而非最短路径搜索：最大化整条路径的最小容量。

    probabilities为同粒连接概率[4,H,W]。negative只允许端点对应的两个
    已知GT；同GT内部容量固定1，跨GT才取真实概率。positive仅允许同一
    GT内部，所有容量取真实概率。四个canonical短边视为无向；反向行走
    时canonical源坐标仍是原offset的起点，不能直接索引当前路径点。

    path、start/end和canonical索引均为输入网格坐标，不是原图坐标；本
    函数不执行原生尺度投影。optional edge_valid为bool[4,H,W] canonical
    边掩码，正反方向均必须通过，便于调用方阻止网格边跳过原图未知区、
    marker缝隙等障碍。仅valid/allowed的网格端点合法不足以证明原图
    像素段合法；原图线段检查由调用方完成并写入edge_valid。

    heap以(-容量,全图扁平索引)确定性排序；同容量不替换已记录parent。
    无合法域路径返回found=False。合法但容量为0的路径返回found=True及
    zero_capacity=True，区别于未知/填充/第三GT造成的几何断路。
    """
    probabilities = np.asarray(probabilities); gt = np.asarray(gt); valid = np.asarray(valid)
    if (gt.ndim != 2 or not gt.size or gt.dtype == np.bool_
            or not np.issubdtype(gt.dtype, np.integer) or np.any(gt < 0)):
        raise ValueError('gt must be a nonempty nonnegative integer HW map')
    if (probabilities.shape != (4, *gt.shape)
            or not np.issubdtype(probabilities.dtype, np.floating)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('probabilities must be finite floating [4,H,W] in [0,1]')
    if valid.shape != gt.shape or valid.dtype != np.bool_:
        raise ValueError('valid must be a bool HW map matching GT')
    if allowed is None:
        allowed = np.ones(gt.shape, bool)
    else:
        allowed = np.asarray(allowed)
        if allowed.shape != gt.shape or allowed.dtype != np.bool_:
            raise ValueError('allowed must be a bool HW map matching GT')
    if edge_valid is not None:
        edge_valid = np.asarray(edge_valid)
        if edge_valid.shape != probabilities.shape or edge_valid.dtype != np.bool_:
            raise ValueError('edge_valid must be a bool canonical [4,H,W] map')
    if direction not in ('negative', 'positive'):
        raise ValueError('direction must be negative or positive')
    offsets = _offsets(offsets); start = _point(start, 'start'); end = _point(end, 'end')
    h, w = gt.shape
    result = dict(found=False, reason=None, direction=direction, start=list(start), end=list(end),
        path=[], length=0, steps=0, edges=[], edge_indices=[], bottleneck_indices=[],
        bottleneck_edge_indices=[], capacity=None, path_capacity=None, zero_capacity=False,
        legal=False, all_known_valid_allowed=False, all_allowed_gt_only=False,
        allowed_gt=[], offsets=[list(v) for v in offsets], connectivity=8,
        objective='widest/maximin', tie_rule='heap(-capacity,input_grid_flat_index); equal capacity keeps first parent',
        coordinate_space='input_grid_yx', canonical_index_space='input_grid_channel_yx',
        native_projection_performed=False, edge_valid_supplied=edge_valid is not None,
        capacity_equality='exact converted floating values; no tolerance', explored_pixels=0)

    def fail(reason):
        result['reason'] = reason
        return result

    for name, point in (('start', start), ('end', end)):
        y, x = point
        if not (0 <= y < h and 0 <= x < w):
            return fail(name + '_out_of_bounds')
        if int(gt[point]) == 0:
            return fail(name + '_unknown_gt')
        if not valid[point]:
            return fail(name + '_invalid_content')
        if not allowed[point]:
            return fail(name + '_outside_allowed')
    start_gt, end_gt = int(gt[start]), int(gt[end])
    if direction == 'negative' and start_gt == end_gt:
        return fail('negative_requires_distinct_endpoint_gt')
    if direction == 'positive' and start_gt != end_gt:
        return fail('positive_requires_same_endpoint_gt')
    permitted = sorted({start_gt, end_gt})
    result['allowed_gt'] = permitted
    domain = valid & allowed & np.isin(gt, permitted)
    result['allowed_pixels'] = int(domain.sum())
    root, target = start[0] * w + start[1], end[0] * w + end[1]
    best = np.full(h * w, -1., np.float64); parent = np.full(h * w, -2, np.int64)
    settled = np.zeros(h * w, bool)
    best[root] = 1.; parent[root] = -1; queue = [(-1., root)]
    neighbors = tuple((channel, sign, sign * dy, sign * dx)
                      for channel, (dy, dx) in enumerate(offsets) for sign in (1, -1))
    explored = 0
    while queue:
        negative_capacity, current = heapq.heappop(queue)
        capacity = -negative_capacity
        if settled[current] or capacity != best[current]:
            continue
        settled[current] = True; explored += 1
        if current == target:
            break
        y, x = divmod(current, w)
        for channel, sign, dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or not domain[ny, nx]:
                continue
            neighbor = ny * w + nx
            if settled[neighbor]:
                continue
            sy, sx = (y, x) if sign == 1 else (ny, nx)
            if edge_valid is not None and not edge_valid[channel, sy, sx]:
                continue
            edge_capacity = (1. if direction == 'negative' and gt[y, x] == gt[ny, nx]
                             else float(probabilities[channel, sy, sx]))
            candidate = min(capacity, edge_capacity)
            if candidate > best[neighbor]:
                best[neighbor] = candidate; parent[neighbor] = current
                heapq.heappush(queue, (-candidate, neighbor))
    result['explored_pixels'] = explored
    if not settled[target]:
        return fail('no_known_valid_allowed_path')
    reverse = []; current = target
    while current != -1:
        y, x = divmod(current, w); reverse.append([int(y), int(x)])
        current = int(parent[current])
    path = reverse[::-1]; edges = []
    for index, (a, b) in enumerate(zip(path, path[1:])):
        dy, dx = b[0] - a[0], b[1] - a[1]
        for channel, offset in enumerate(offsets):
            if (dy, dx) == offset:
                sign = 1; canonical_source = a; break
            if (-dy, -dx) == offset:
                sign = -1; canonical_source = b; break
        else:
            raise AssertionError('Recovered parent step is not a canonical short edge')
        ga, gb = int(gt[tuple(a)]), int(gt[tuple(b)])
        probability = float(probabilities[channel, *canonical_source])
        cross = ga != gb
        edge_capacity = 1. if direction == 'negative' and not cross else probability
        target_legal = cross if direction == 'negative' else not cross
        edges.append(dict(path_index=index, a=a, b=b, canonical=[channel, *canonical_source],
            actual_delta=[dy, dx], canonical_offset=list(offsets[channel]),
            canonical_direction=sign, probability=probability, capacity=edge_capacity,
            gt_a=ga, gt_b=gb, cross_gt=cross, target_legal=target_legal))
    capacity = float(best[target])
    bottlenecks = [edge for edge in edges if edge['target_legal'] and edge['capacity'] == capacity]
    # parent只指向已settled合法节点；仍显式审计，避免训练索引泄漏未知区。
    legal = (all(domain[tuple(point)] for point in path)
             and len({tuple(point) for point in path}) == len(path)
             and all(edge['gt_a'] in permitted and edge['gt_b'] in permitted for edge in edges)
             and (edge_valid is None or all(edge_valid[tuple(edge['canonical'])] for edge in edges))
             and (not edges or min(edge['capacity'] for edge in edges) == capacity))
    if not legal:
        raise AssertionError('Recovered widest path violated GT/domain/capacity invariants')
    result.update(found=True, reason='widest_known_valid_path', path=path, length=len(path),
        steps=len(edges), edges=edges, edge_indices=[edge['canonical'] for edge in edges],
        bottleneck_indices=[edge['canonical'] for edge in bottlenecks],
        bottleneck_edge_indices=[edge['path_index'] for edge in bottlenecks],
        capacity=capacity, path_capacity=capacity, zero_capacity=capacity == 0.,
        legal=True, all_known_valid_allowed=True, all_allowed_gt_only=True)
    return result
