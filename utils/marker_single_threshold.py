# -*- coding: utf-8 -*-
"""只取消种子弱边重建，保留原真实分水岭障碍的独立候选。"""
from __future__ import annotations

import hashlib
from numbers import Integral

import numpy as np


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _mask(stages, key, shape=None, binary=False):
    value = stages.get(key)
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint8
            or value.ndim != 2 or not value.size
            or (shape is not None and value.shape != shape)):
        raise ValueError(key + ' must be a nonempty uint8 HW mask on the native grid')
    allowed = (0, 1) if binary else (0, 255)
    if not np.isin(value, allowed).all():
        raise ValueError(key + ' contains values outside its mask convention')
    return value


def high_only_reflect_markers(stages: dict, reflection_padding: int = 32) -> dict:
    """从完整原尺寸stages生成高阈值reflect种子，不重建真实障碍。

    调用方继续使用原N1配置，将返回的marker_labels注入原watershed。
    不能把重建步数直接设0后重算全部阶段：旧代码此时会让种子带与真实
    障碍共用数组，封边操作会同时修改真实障碍，超出本候选的唯一变量。

    本函数沿用原bridge、实际细化算法、reflect、dilate、seal和核心面积
    门槛；不运行watershed、不应用最终实例过滤、不读取GT或类别。
    """
    if not isinstance(stages, dict) or not isinstance(stages.get('resolved'), dict):
        raise ValueError('Expected complete native marker stages with resolved settings')
    if (isinstance(reflection_padding, (bool, np.bool_))
            or not isinstance(reflection_padding, Integral) or reflection_padding <= 0):
        raise ValueError('reflection_padding must be a positive integer')
    high = _mask(stages, 'high_mask', binary=True)
    barrier = _mask(stages, 'barrier_belt', high.shape)
    reference = _mask(stages, 'marker_boundary_mask', high.shape, binary=True)
    resolved = stages['resolved']
    for key in ['bridge_width', 'watershed_dilate_width', 'marker_border_seal_width',
                'min_instance_area']:
        value = resolved.get(key)
        minimum = 1 if key == 'min_instance_area' else 0
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                or value < minimum):
            raise ValueError('Invalid resolved setting: ' + key)
    threshold = resolved.get('boundary_threshold')
    if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (float, int,
            np.floating, np.integer)) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Invalid resolved boundary_threshold')
    if np.any((high > 0) & (reference == 0)):
        raise ValueError('Reference marker boundary must contain the strong boundary')

    # 延迟导入，以便调用方先装入与控制完全一致的封存解码环境。
    from tools.probe_affinity_marker_replay import build_marker_variant

    source = dict(stages, high_mask=high.copy(), marker_boundary_mask=high.copy(),
                  barrier_belt=barrier.copy(), resolved=dict(resolved))
    result = build_marker_variant(source, 'reflect', int(reflection_padding))
    if not np.array_equal(result['barrier_belt'], barrier):
        raise AssertionError('High-only marker candidate changed the real watershed barrier')
    result['resolved'] = dict(result['resolved'],
        diagnostic_variant='marker_reflect_high_only_v1',
        reference_marker_boundary_low_threshold=resolved.get('marker_boundary_low_threshold'),
        reference_marker_boundary_reconstruction_steps=resolved.get('marker_boundary_reconstruction_steps', 0),
        marker_boundary_low_threshold=None, marker_boundary_reconstruction_steps=0)
    result['change'] = ('use high-threshold mask for reflect seeds only; preserve original '
                        'watershed barrier, bridge, dilate, seal and area threshold')
    result['single_threshold_audit'] = dict(
        version='marker_reflect_high_only_v1',
        marker_source='original_high_mask', boundary_threshold=float(threshold),
        weak_reconstruction_applied=False,
        removed_weak_boundary_pixels=int(((reference > 0) & (high == 0)).sum()),
        strong_boundary_pixels=int((high > 0).sum()),
        real_barrier_unchanged=True, real_barrier_sha256=_array_sha(barrier),
        marker_source_sha256=_array_sha(high),
        core_min_area=int(resolved['min_instance_area']),
        final_instance_filter_modified=False,
        reference_config_must_remain_unchanged=True,
        requires_complete_native_stages=True,
        no_gt_or_class_access=True)
    return result
