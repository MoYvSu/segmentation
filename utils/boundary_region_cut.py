# -*- coding: utf-8 -*-
"""原生八邻像素图的完整顶点割；只选监督位置，不修改预测或制造GT。

每个像素拆为in/out节点：截断in->out即将该像素变为屏障。走廊外和
种子像素不可新增屏障；已有高响应屏障仍参与整图连通判定。裁框的所有
开放边缘通过同一外部节点连通，不能靠删掉窗外通路来虚报分隔。
"""
from __future__ import annotations

from numbers import Integral, Real
from time import perf_counter

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


def _inputs(probability, allowed, source_core, sink_core, threshold, capacity_scale, exterior):
    probability = np.asarray(probability)
    if (probability.ndim != 2 or probability.size == 0
            or probability.dtype.kind not in 'fiub'
            or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('probability must be a finite nonempty HW map within [0,1]')
    masks = []
    for name, value in [('allowed', allowed), ('source_core', source_core), ('sink_core', sink_core)]:
        value = np.asarray(value)
        if value.shape != probability.shape or value.dtype != np.bool_:
            raise ValueError(name + ' must be a boolean HW mask matching probability')
        masks.append(value)
    if np.any(masks[1] & masks[2]):
        raise ValueError('source_core and sink_core must not overlap')
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
            or not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError('threshold must be finite and strictly between zero and one')
    if (isinstance(capacity_scale, (bool, np.bool_)) or not isinstance(capacity_scale, Integral)
            or capacity_scale < 1):
        raise ValueError('capacity_scale must be a positive integer')
    if capacity_scale > np.iinfo(np.int32).max:
        raise ValueError('capacity_scale exceeds SciPy int32 flow safety')
    if not isinstance(exterior, (bool, np.bool_)):
        raise ValueError('exterior must be boolean')
    return probability.astype(np.float64, copy=False), *masks


def _frame(shape):
    result = np.zeros(shape, bool)
    result[0, :] = result[-1, :] = True
    result[:, 0] = result[:, -1] = True
    return result


def _connected(walls, source, sink, exterior, frame):
    """八邻开放像素连通；所有开放框边分量经外部节点归并。"""
    _, labels = cv2.connectedComponents(np.ascontiguousarray(~walls, dtype=np.uint8), connectivity=8)
    first = np.unique(labels[source]); first = first[first != 0]
    second = np.unique(labels[sink]); second = second[second != 0]
    if np.intersect1d(first, second).size:
        return True
    if exterior:
        outside = np.unique(labels[frame]); outside = outside[outside != 0]
        if np.intersect1d(first, outside).size and np.intersect1d(second, outside).size:
            return True
    return False


def _graph(node_capacity, source, sink, exterior, frame, infinity):
    height, width = node_capacity.shape
    count = node_capacity.size
    source_node, sink_node = 2 * count, 2 * count + 1
    exterior_node = 2 * count + 2 if exterior else None
    total = 2 * count + 2 + int(exterior)
    indices = np.arange(count, dtype=np.int64).reshape(height, width)
    rows = [2 * indices.ravel()]
    columns = [2 * indices.ravel() + 1]
    values = [node_capacity.ravel()]

    def arcs(first, second):
        rows.append(np.asarray(first, dtype=np.int64).ravel())
        columns.append(np.asarray(second, dtype=np.int64).ravel())
        values.append(np.full(np.asarray(first).size, infinity, dtype=np.int64))

    # 四种无向方向，各加入两条有向弧，恰为八邻接。
    for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        first = indices[y0:y1, x0:x1].ravel()
        second = indices[y0+dy:y1+dy, x0+dx:x1+dx].ravel()
        arcs(2 * first + 1, 2 * second)
        arcs(2 * second + 1, 2 * first)

    first = np.flatnonzero(source)
    second = np.flatnonzero(sink)
    arcs(np.full(first.size, source_node, np.int64), 2 * first)
    arcs(2 * second + 1, np.full(second.size, sink_node, np.int64))
    if exterior:
        border = np.flatnonzero(frame)
        arcs(2 * border + 1, np.full(border.size, exterior_node, np.int64))
        arcs(np.full(border.size, exterior_node, np.int64), 2 * border)

    capacity = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                          shape=(total, total), dtype=np.int64)
    capacity.eliminate_zeros()
    return capacity, source_node, sink_node


def select_region_cut(probability, allowed, source_core, sink_core, *, threshold=.65,
                      capacity_scale=1000, exterior=True):
    """返回完整新增屏障mask及审计，不存在合法完整割时返回空mask。

    ``probability``是当前G的停止梯度概率，不得混合N1外部构造oracle。
    当前>=threshold像素视为已有墙；只允许在allowed且非种子的位置新增墙。
    Unknown应以allowed=False传入，但其节点不会从空间图中删掉。
    ``feasible=True``包括已经分隔；必须结合reason/cut_pixels解读。
    证书只对应当前原生硬屏障的八邻连通，不等于细化/marker/分水岭最终分隔。
    """
    started = perf_counter()
    probability, allowed, source_core, sink_core = _inputs(
        probability, allowed, source_core, sink_core, threshold, capacity_scale, exterior)
    exterior = bool(exterior)
    walls = probability >= float(threshold)
    source = source_core & ~walls
    sink = sink_core & ~walls
    frame = _frame(probability.shape)
    empty = np.zeros(probability.shape, bool)
    audit = dict(format='boundary_region_vertex_cut_v1', feasible=False, reason=None,
        threshold=float(threshold), capacity_scale=int(capacity_scale), connectivity=8,
        exterior=exterior, graph_nodes=0, graph_arcs=0,
        source_core_pixels=int(source_core.sum()), sink_core_pixels=int(sink_core.sum()),
        source_open_pixels=int(source.sum()), sink_open_pixels=int(sink.sum()),
        source_wall_pixels=int((source_core & walls).sum()), sink_wall_pixels=int((sink_core & walls).sum()),
        allowed_pixels=int(allowed.sum()), existing_wall_pixels=int(walls.sum()),
        cut_pixels=0, cut_touches_frame=False, editable_pixels=0,
        existing_wall_cut_pixels=0, new_cut_capacity_sum=0,
        flow_value=None, infinite_capacity=None,
        finite_capacity_sum=None, source_sink_connected_before=None,
        source_sink_connected_after=None, certificate_valid=False,
        immutable_route_precheck_ran=False, immutable_route_connected=None,
        immutable_route_precheck_seconds=0.0, maxflow_called=False,
        selection_only=True, unknown_nodes_removed=False)

    def finish(mask, reason, *, feasible=False, certificate=False, connected_after=None):
        audit.update(reason=reason, feasible=bool(feasible), certificate_valid=bool(certificate),
            cut_pixels=int(mask.sum()), cut_touches_frame=bool((mask & frame).any()),
            source_sink_connected_after=connected_after, elapsed_seconds=perf_counter()-started)
        return mask, audit

    if not source.any():
        return finish(empty, 'no_open_source_core')
    if not sink.any():
        return finish(empty, 'no_open_sink_core')
    connected = _connected(walls, source, sink, exterior, frame)
    audit['source_sink_connected_before'] = connected
    if not connected:
        return finish(empty, 'already_separated', feasible=True, certificate=True, connected_after=False)

    # 种子不能以涂成墙的方式从问题中消失。
    editable = allowed & ~walls & ~source & ~sink
    costs = 1 + np.rint(int(capacity_scale) * (1 - probability[editable])).astype(np.int64)
    finite_sum = int(costs.sum(dtype=np.int64))
    infinity = finite_sum + 1
    # SciPy的flow边值使用int32；不能截断M而失去“合法割必小于M”的保证。
    if infinity > np.iinfo(np.int32).max:
        raise ValueError('capacity sum exceeds SciPy int32 flow safety; reduce crop or capacity_scale')
    audit.update(infinite_capacity=infinity, finite_capacity_sum=finite_sum,
                 editable_pixels=int(editable.sum()))
    # 将所有允许修改的像素一次性当墙，若种子仍经不可改像素/窗外相通，
    # 任意合法子集都不可能分开它们。此拒绝与后面的完整割问题等价，
    # 不必为注定无解的样本建立大图再跑maxflow。
    checked = perf_counter()
    immutable_connected = _connected(walls | editable, source, sink, exterior, frame)
    audit.update(immutable_route_precheck_ran=True,
                 immutable_route_connected=immutable_connected,
                 immutable_route_precheck_seconds=perf_counter()-checked)
    if immutable_connected:
        audit.update(infeasible_flow_at_least_M=True, flow_lower_bound=infinity)
        # flow_value保持None：这里只证明下界，没有实际计算maxflow数值。
        return finish(empty, 'no_modifiable_complete_separator', connected_after=True)
    node_capacity = np.full(probability.shape, infinity, np.int64)
    node_capacity[walls] = 0
    node_capacity[editable] = costs
    capacity, source_node, sink_node = _graph(node_capacity, source, sink, exterior, frame, infinity)
    audit.update(graph_nodes=int(capacity.shape[0]), graph_arcs=int(capacity.nnz))
    audit['maxflow_called'] = True
    flow = maximum_flow(capacity, source_node, sink_node, method='dinic')
    value = int(flow.flow_value)
    audit['flow_value'] = value
    if value < 0:
        raise RuntimeError('SciPy returned negative maxflow')
    if value >= infinity:
        # 合法新增割的总成本至多finite_sum；达到M说明不可改通路无法切断。
        audit['infeasible_flow_at_least_M'] = True
        return finish(empty, 'no_modifiable_complete_separator', connected_after=True)

    residual = capacity.astype(np.int64) - flow.flow.astype(np.int64)
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    order = breadth_first_order(residual, source_node, directed=True, return_predecessors=False)
    reachable = np.zeros(capacity.shape[0], bool); reachable[order] = True
    crossing = (reachable[:2*probability.size:2] & ~reachable[1:2*probability.size:2]).reshape(probability.shape)
    audit['existing_wall_cut_pixels'] = int((crossing & walls).sum())
    illegal = crossing & ~walls & ~editable
    if illegal.any() or (crossing & (source | sink)).any():
        audit['illegal_cut_pixels'] = int(illegal.sum())
        return finish(empty, 'illegal_or_seed_vertex_in_cut', connected_after=True)
    cut = crossing & editable
    audit['new_cut_capacity_sum'] = int(node_capacity[cut].sum(dtype=np.int64))
    if not cut.any():
        return finish(empty, 'empty_cut_without_separation', connected_after=True)
    after = _connected(walls | cut, source, sink, exterior, frame)
    if after:
        # 不返回仅一部分合法像素冒称已经产生完整屏障。
        audit['uncertified_candidate_pixels'] = int(cut.sum())
        return finish(empty, 'cut_failed_native_connectivity_certificate', connected_after=True)
    if audit['new_cut_capacity_sum'] != value:
        raise RuntimeError('Maxflow value differs from complete vertex-cut cost')
    return finish(np.ascontiguousarray(cut), 'selected_complete_separator', feasible=True,
                  certificate=True, connected_after=False)
