# -*- coding: utf-8 -*-
"""既有种子的连续灰度盆地深度；只用于独立读出候选，不生成新种子。"""
from __future__ import annotations

import hashlib
from numbers import Real

import numpy as np
from skimage.morphology import max_tree


FORMAT = 'marker_basin_depth_v1'


def _sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'))
    h.update(str(a.shape).encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def _inputs(markers, boundary):
    if (not isinstance(markers, np.ndarray) or markers.dtype != np.int32
            or markers.ndim != 2 or not markers.size or np.any(markers < 0)):
        raise ValueError('markers must be nonempty nonnegative int32 HW')
    if (not isinstance(boundary, np.ndarray) or boundary.shape != markers.shape
            or not np.issubdtype(boundary.dtype, np.floating)
            or not np.isfinite(boundary).all() or np.any((boundary < 0) | (boundary > 1))):
        raise ValueError('boundary must be finite floating HW probability matching markers')


def seed_basin_depths(markers: np.ndarray, boundary: np.ndarray) -> dict:
    """按原连续B的8邻接min-tree计算既有种子的elder-rule深度。

    每个种子的anchor固定为其B最小像素，同值取原生行优先第一个。birth为
    该点B值；death为从anchor到birth更低（同值按seed ID）的任一种子所需
    最小逃逸鞍点。depth=death-birth。最低elder没有更低种子，明确保留。

    这不是语义正确概率。种子本身可以覆盖多个原B盆地，固定anchor是本候选
    明示的读出规则，不把h-minima与旧seed简单相交，也不重新布置种子。
    图边外没有自由连通；padding只用于max_tree的小图边界兼容，值2高于
    概率上界1，不会降低任意两个原图anchor之间的鞍点。
    """
    _inputs(markers, boundary)
    ids, inverse, counts = np.unique(markers.ravel(), return_inverse=True, return_counts=True)
    minimum = np.full(len(ids), np.inf, np.float64)
    np.minimum.at(minimum, inverse, boundary.ravel())
    equal = (markers.ravel() > 0) & (boundary.ravel() == minimum[inverse])
    anchor = np.full(len(ids), markers.size, np.int64)
    np.minimum.at(anchor, inverse[equal], np.flatnonzero(equal))
    positive = ids > 0
    seed_ids = ids[positive]; births = minimum[positive]; anchors = anchor[positive]
    sizes = counts[positive]
    del inverse, minimum, equal
    order = np.lexsort((seed_ids, births))
    rows = []
    visited_nodes = 0
    if len(seed_ids) > 1:
        # -B preserves float32's exact order after float64 promotion; 1-B in
        # float32 would merge distinct small values and alter saddle heights.
        padded = np.pad(boundary.astype(np.float64), 1, constant_values=2.)
        parent, traversal = max_tree(-padded, connectivity=2)
        parent = parent.ravel(); del traversal
        height = padded.ravel()
        owner = np.full(parent.size, -1, np.int32)
        width = markers.shape[1]; stride = width + 2
        for rank, index in enumerate(order):
            original_anchor = int(anchors[index]); y, x = divmod(original_anchor, width)
            node = (y + 1) * stride + x + 1
            while owner[node] < 0:
                owner[node] = rank; visited_nodes += 1
                p = int(parent[node])
                if p == node:
                    break
                node = p
            root = rank == 0
            if not root and owner[node] >= rank:
                raise RuntimeError('Min-tree seed did not reach an earlier elder')
            saddle = None if root else float(height[node])
            depth = None if root else max(0., saddle - float(births[index]))
            earlier = None if root else int(seed_ids[order[int(owner[node])]])
            if not root and not (float(births[index]) <= saddle <= 1.):
                raise RuntimeError('Saddle outside original-image probability range')
            rows.append(dict(seed_id=int(seed_ids[index]), marker_pixels=int(sizes[index]),
                anchor_xy=[x, y], birth=float(births[index]), saddle=saddle, depth=depth,
                elder_seed_id=earlier, persistent_root=root))
    elif len(seed_ids) == 1:
        y, x = divmod(int(anchors[0]), markers.shape[1])
        rows.append(dict(seed_id=int(seed_ids[0]), marker_pixels=int(sizes[0]),
            anchor_xy=[x, y], birth=float(births[0]), saddle=None, depth=None,
            elder_seed_id=None, persistent_root=True))
    rows.sort(key=lambda row: row['seed_id'])
    return dict(format=FORMAT, input_seed_count=len(rows), connectivity=8,
        anchor_rule='min B over existing marker pixels; row-major first tie',
        elder_rule='ascending birth, then seed ID; one global elder retained',
        outside_rule='no exterior shortcut; max-tree padding has height 2 > max(B)',
        tree_visited_nodes=visited_nodes, seeds=rows,
        marker_sha256=_sha(markers), reference_boundary_sha256=_sha(boundary),
        interpretation='Existing-seed gray-level persistence only, not GT or correctness probability.')


def filter_shallow_seeds(markers: np.ndarray, boundary: np.ndarray, depth: float,
                         *, analysis: dict | None = None) -> tuple[np.ndarray, dict]:
    """只撤销depth<h的既有种子；h=0关闭，剩余ID与像素位置不变。"""
    _inputs(markers, boundary)
    if (isinstance(depth, (bool, np.bool_)) or not isinstance(depth, Real)
            or not np.isfinite(depth) or not 0 <= depth <= 1):
        raise ValueError('depth must be finite in [0,1]')
    h = float(depth)
    if h == 0:
        result = markers.copy()
        return result, dict(format=FORMAT, threshold=h, disabled=True,
            removed_ids=[], removed_seed_count=0, markers_unchanged=True,
            no_seed_relabeling=True, no_boundary_change=True)
    info = seed_basin_depths(markers, boundary) if analysis is None else analysis
    if (info.get('format') != FORMAT or info.get('marker_sha256') != _sha(markers)
            or info.get('reference_boundary_sha256') != _sha(boundary)):
        raise ValueError('Depth analysis does not bind these markers/boundary')
    removed = [r['seed_id'] for r in info['seeds']
               if not r['persistent_root'] and r['depth'] < h]
    result = markers.copy()
    if removed:
        result[np.isin(markers, removed)] = 0
    if np.any(markers > 0) and not np.any(result > 0):
        raise RuntimeError('All seeds removed; candidate is invalid')
    return result, dict(format=FORMAT, threshold=h, disabled=False,
        input_seed_count=info['input_seed_count'], removed_ids=removed,
        removed_seed_count=len(removed), retained_seed_count=info['input_seed_count']-len(removed),
        marker_sha256=info['marker_sha256'], output_marker_sha256=_sha(result),
        reference_boundary_sha256=info['reference_boundary_sha256'],
        markers_unchanged=bool(np.array_equal(markers, result)), no_seed_relabeling=True,
        no_boundary_change=True, analysis=info)
