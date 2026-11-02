# -*- coding: utf-8 -*-
"""在分水岭前撤销无连续热图低响应支撑的种子，不修改边界或最终实例。"""
from __future__ import annotations

import hashlib
from numbers import Real

import numpy as np


FORMAT = 'marker_seed_support_v1'


def _array_sha(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.tobytes())
    return digest.hexdigest()


def filter_unsupported_seeds(markers: np.ndarray, reference_boundary: np.ndarray,
                             low_threshold: float = .45) -> tuple[np.ndarray, dict]:
    """只将零低响应支撑的正种子ID置0，保留其余ID和全部原位置。

    参考图为完整原生尺寸的连续边界响应；高值表示边界。种子中至少一个
    ``reference_boundary <= low_threshold`` 像素就保留，不加面积、比例或
    类别门槛。参考图不应由调用方悄悄替换为修复器自生成的低谷。

    本函数不运行分水岭、不修改边界、不对最终实例做面积删除，返回int32
    种子副本。所有正种子都被撤销时明确拒绝候选；不静默保留最后一个，
    不制造空实例输出。原来没有正种子则按恒等处理，由调用方保留原回退。
    支撑只是预测一致性证据，不等于物理上正确；最终uint16契约由读出验证。
    """
    if (not isinstance(markers, np.ndarray) or markers.dtype != np.int32
            or markers.ndim != 2 or not markers.size or np.any(markers < 0)):
        raise ValueError('markers must be a nonempty nonnegative int32 HW array')
    if (not isinstance(reference_boundary, np.ndarray)
            or reference_boundary.shape != markers.shape
            or not np.issubdtype(reference_boundary.dtype, np.floating)
            or not np.all(np.isfinite(reference_boundary))
            or np.any((reference_boundary < 0) | (reference_boundary > 1))):
        raise ValueError('reference_boundary must be a finite floating HW probability matching markers')
    if (isinstance(low_threshold, (bool, np.bool_)) or not isinstance(low_threshold, Real)
            or not np.isfinite(low_threshold) or not 0 <= low_threshold <= 1):
        raise ValueError('low_threshold must be finite and within [0, 1]')
    threshold = float(low_threshold)

    # 将稀疏int32 ID映到紧凑索引；绝不分配max(marker_id)+1的巨型表。
    ids, inverse, counts = np.unique(markers.ravel(), return_inverse=True, return_counts=True)
    supported = reference_boundary.ravel() <= threshold
    low_counts = np.bincount(inverse[supported], minlength=len(ids))
    positive = ids > 0
    keep = (ids == 0) | (low_counts > 0)
    if positive.any() and not np.any(positive & keep):
        raise ValueError('All positive seeds would be removed; candidate watershed is rejected')

    filtered = np.where(keep[inverse], markers.ravel(), np.int32(0)).reshape(markers.shape)
    filtered = np.array(filtered, dtype=np.int32, copy=True, order='C')
    rows = [dict(seed_id=int(seed), area=int(area), low_support_pixels=int(low),
                 retained=bool(retain))
            for seed, area, low, retain in zip(ids, counts, low_counts, keep) if seed > 0]
    removed_ids = [row['seed_id'] for row in rows if not row['retained']]
    retained_ids = [row['seed_id'] for row in rows if row['retained']]
    audit = dict(format=FORMAT, low_threshold=threshold, comparison='reference_boundary <= low_threshold',
        input_seed_count=len(rows), retained_seed_count=len(retained_ids), removed_seed_count=len(removed_ids),
        input_marker_pixels=sum(row['area'] for row in rows),
        retained_marker_pixels=sum(row['area'] for row in rows if row['retained']),
        removed_marker_pixels=sum(row['area'] for row in rows if not row['retained']),
        removed_ids=removed_ids, retained_ids=retained_ids, seeds=rows,
        input_marker_sha256=_array_sha(markers), output_marker_sha256=_array_sha(filtered),
        reference_boundary_sha256=_array_sha(reference_boundary),
        markers_unchanged=bool(np.array_equal(filtered, markers)),
        no_seed_relabeling=True, no_probability_change=True, pre_watershed_seed_only=True,
        interpretation='Predicted support conflict only; no GT, class, final-area or correctness decision.')
    return filtered, audit
