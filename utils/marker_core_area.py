# -*- coding: utf-8 -*-
"""从完整原生核心连通域重建候选种子，独立于最终实例面积过滤。"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np


FORMAT = 'marker_core_area_v1'


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or int(value) < 1):
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(str(value.shape).encode('ascii'))
    digest.update(value.tobytes())
    return digest.hexdigest()


def rebuild_core_markers(core_labels: np.ndarray, min_core_area: int = 50,
                         reference_min_core_area: int = 50) -> tuple[np.ndarray, dict]:
    """依完整核心面积生成连续正编号int32种子，0始终表示非核心背景。

    ``core_labels`` 必须来自整幅原生尺寸的 reflect 阶段连通域，不能先裁
    ROI 再据局部面积决定种子。函数无法从数组内容证明其覆盖范围，因此
    由调用方核验缓存身份和完整尺寸；本函数不重新求连通域、不制造新位置。

    编号按正源label升序、只对保留核心依次分配1..N，与旧 reflect 过滤
    循环一致。新增较小核心可能使后续旧核心ID变化，不能按数值ID直接
    比较候选与控制；audit同时记录源label及参考种子ID以供空间配对。

    本函数只处理分水岭之前的核心门槛，不应用或修改最终实例面积过滤，
    不访问GT、类别、边界概率或图像。25/50候选须由调用方继续使用原有
    最终50像素过滤。空HW数组、全0图均返回独立的全0副本。
    """
    if (not isinstance(core_labels, np.ndarray) or core_labels.ndim != 2
            or not np.issubdtype(core_labels.dtype, np.integer)
            or np.issubdtype(core_labels.dtype, np.bool_)
            or np.any(core_labels < 0)):
        raise ValueError('core_labels must be a nonnegative integer HW array')
    minimum = _positive_integer(min_core_area, 'min_core_area')
    reference_minimum = _positive_integer(reference_min_core_area, 'reference_min_core_area')

    # 按实际存在的源label压缩索引，避免稀疏大ID造成max(label)+1级分配。
    labels, inverse, counts = np.unique(core_labels.ravel(), return_inverse=True,
                                        return_counts=True)
    positive = labels > 0
    retained = positive & (counts >= minimum)
    reference_retained = positive & (counts >= reference_minimum)
    marker_count = int(np.count_nonzero(retained))
    if marker_count > np.iinfo(np.int32).max:
        raise ValueError('Retained core count exceeds the int32 marker range')
    lookup = np.zeros(len(labels), dtype=np.int32)
    lookup[retained] = np.arange(1, marker_count + 1, dtype=np.int32)
    reference_lookup = np.zeros(len(labels), dtype=np.int32)
    reference_count = int(np.count_nonzero(reference_retained))
    if reference_count > np.iinfo(np.int32).max:
        raise ValueError('Reference core count exceeds the int32 marker range')
    reference_lookup[reference_retained] = np.arange(1, reference_count + 1, dtype=np.int32)
    markers = np.array(lookup[inverse].reshape(core_labels.shape), dtype=np.int32,
                       copy=True, order='C')
    rows = [dict(source_label=int(label), area=int(area), marker_id=int(marker),
                 reference_marker_id=int(reference_marker), retained=bool(keep),
                 reference_retained=bool(reference_keep),
                 added=bool(keep and not reference_keep),
                 removed=bool(reference_keep and not keep))
            for label, area, marker, reference_marker, keep, reference_keep
            in zip(labels, counts, lookup, reference_lookup, retained, reference_retained)
            if label > 0]
    added = [dict(row) for row in rows if row['added']]
    removed = [dict(row) for row in rows if row['removed']]
    audit = dict(format=FORMAT, min_core_area=minimum,
        reference_min_core_area=reference_minimum, comparison='core_area >= min_core_area',
        source_core_count=len(rows), retained_core_count=marker_count,
        reference_core_count=reference_count, added_core_count=len(added),
        removed_core_count=len(removed),
        retained_core_pixels=sum(row['area'] for row in rows if row['retained']),
        added_core_pixels=sum(row['area'] for row in added),
        removed_core_pixels=sum(row['area'] for row in removed),
        cores=rows, added_cores=added, removed_cores=removed,
        input_core_labels_sha256=_array_sha(core_labels),
        output_markers_sha256=_array_sha(markers),
        reference_selection_unchanged=bool(np.array_equal(retained, reference_retained)),
        zero_background_preserved=True, requires_complete_native_core_labels=True,
        pre_watershed_seed_only=True, no_gt_or_class_access=True,
        final_instance_filter_applied=False, final_instance_filter_modified=False,
        interpretation=('Core size is a candidate-generation condition, not evidence of '
                        'physical correctness. Preserve the original final-instance area '
                        'filter in the caller.'))
    return markers, audit
