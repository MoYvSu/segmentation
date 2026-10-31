# -*- coding: utf-8 -*-
"""一次性为同类别实例旁的细零缝指定归属，不重算类别或传播填充。

本规则只读取最终实例图，不能识别测试图中的零像素是否确为分水岭界线。
GT、实例面积和ID大小均不参与选位；未知区与真实边界的判断留给事后诊断。
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from numbers import Integral

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt


FORMAT = 'instance_same_phase_seams_v1'
TIE_RULE = ('scipy.ndimage.distance_transform_edt return_indices: Euclidean nearest '
            'original positive pixel; equal-distance ties use the library spatial '
            'traversal order, never instance ID, area, phase preference or GT')


def _validate(ids, classes, enabled):
    if (not isinstance(ids, np.ndarray) or ids.ndim != 2 or ids.dtype != np.uint16
            or min(ids.shape) < 1):
        raise ValueError('ids must be a nonempty two-dimensional uint16 array')
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')
    if not isinstance(classes, Mapping):
        raise ValueError('classes must be a complete instance-to-phase mapping')
    normalized = {}
    for key, phase in classes.items():
        if isinstance(key, (bool, np.bool_)):
            raise ValueError('Instance class keys must be integer IDs or decimal strings')
        if isinstance(key, Integral):
            value = int(key)
        elif isinstance(key, str) and key.isascii() and key.isdecimal():
            value = int(key)
        else:
            raise ValueError('Instance class keys must be integer IDs or decimal strings')
        if not 1 <= value <= 65535 or value in normalized:
            raise ValueError('Instance class keys must be unique positive uint16 IDs')
        if isinstance(phase, (bool, np.bool_)) or not isinstance(phase, Integral) or int(phase) not in (0, 1):
            raise ValueError('Instance classes must be integer 0=pearlite or 1=ferrite')
        normalized[value] = int(phase)
    present = {int(value) for value in np.unique(ids) if value > 0}
    if present != set(normalized):
        raise ValueError('Class table must contain exactly the nonzero instance IDs')
    return normalized


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _class_sha(classes):
    value = '\n'.join(f'{i}:{classes[i]}' for i in sorted(classes))
    return hashlib.sha256(value.encode('ascii')).hexdigest()


def assign_instance_seams(ids, classes, enabled=True):
    """返回新uint16实例图及回执；输入数组和原类别表完全不修改。

    只观察一次原图：非外图框的ID0像素，若其8邻域存在正ID且这些ID的
    类别全部一致，则归最近的原正像素。欧氏距离使四邻优先于斜邻；平局
    由SciPy的空间次序解决。新填入像素不会成为下一批填充的来源。
    enabled=False仍核验输入合同，但不调用距离变换、不修改任何像素。
    """
    class_table = _validate(ids, classes, enabled)
    input_sha = _sha(ids)
    classes_sha = _class_sha(class_table)
    output = ids.copy()
    zero, positive = ids == 0, ids > 0
    height, width = ids.shape
    phase_lut = np.full(65536, -1, np.int8)
    for iid, phase in class_table.items():
        phase_lut[iid] = phase
    phases = phase_lut[ids]

    # padding只用于8邻观察，不产生新的正像素或实例。
    padded_phase = np.pad(phases, 1, constant_values=-1)
    has_P = np.zeros(ids.shape, bool)
    has_F = np.zeros(ids.shape, bool)
    cardinal_count = np.zeros(ids.shape, np.uint8)
    diagonal_count = np.zeros(ids.shape, np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            neighbor = padded_phase[1+dy:1+dy+height, 1+dx:1+dx+width]
            has_P |= neighbor == 0
            has_F |= neighbor == 1
            if abs(dy)+abs(dx) == 1:
                cardinal_count += neighbor >= 0
            else:
                diagonal_count += neighbor >= 0
    frame = np.zeros(ids.shape, bool)
    frame[0] = frame[-1] = True
    frame[:, 0] = frame[:, -1] = True
    interior_zero = zero & ~frame
    adjacent = has_P | has_F
    mixed = has_P & has_F
    eligible = interior_zero & adjacent & ~mixed
    nearest_cardinal = eligible & (cardinal_count > 0)
    nearest_diagonal = eligible & (cardinal_count == 0)
    tied = eligible & ((cardinal_count > 1) | ((cardinal_count == 0) & (diagonal_count > 1)))
    assigned = eligible if bool(enabled) else np.zeros(ids.shape, bool)

    if assigned.any():
        # EDT只以原图positive为源；eligible限定最近源必然在8邻域内。
        nearest = distance_transform_edt(zero, return_distances=False, return_indices=True)
        chosen_ids = ids[tuple(nearest[:, assigned])]
        chosen_phases = phase_lut[chosen_ids]
        expected_phases = has_F[assigned].astype(np.int8)
        if np.any(chosen_ids == 0) or not np.array_equal(chosen_phases, expected_phases):
            raise RuntimeError('Nearest original positive ownership violates same-phase eligibility')
        output[assigned] = chosen_ids

    positive_unchanged = bool(np.array_equal(output[positive], ids[positive]))
    # 最后一个零像素可能被填完，故只比较非零ID集合。
    ids_unchanged = bool(np.array_equal(np.unique(output[output > 0]), np.unique(ids[positive])))
    if not positive_unchanged or not ids_unchanged or _validate(output, classes, True) != class_table:
        raise RuntimeError('Seam assignment changed original positive pixels or instance/class identities')
    partition = dict(frame=int(np.count_nonzero(zero & frame)),
                     interior_without_positive_neighbor=int(np.count_nonzero(interior_zero & ~adjacent)),
                     interior_mixed_phase=int(np.count_nonzero(interior_zero & mixed)),
                     eligible=int(np.count_nonzero(eligible)))
    zero_before, zero_after = int(zero.sum()), int(np.count_nonzero(output == 0))
    if sum(partition.values()) != zero_before or zero_before-zero_after != int(assigned.sum()):
        raise RuntimeError('Original-zero eligibility and actual assigned pixels do not close')
    by_phase = lambda mask: dict(P=int(np.count_nonzero(mask & has_P)), F=int(np.count_nonzero(mask & has_F)))
    receipt = dict(format=FORMAT, enabled=bool(enabled), shape=list(ids.shape), dtype='uint16',
                   input_sha256=input_sha, output_sha256=_sha(output),
                   classes_sha256=classes_sha, classes_unchanged=True,
                   positive_pixels_unchanged=positive_unchanged, instance_ids_unchanged=ids_unchanged,
                   original_positive_pixels=int(positive.sum()), original_instance_count=len(class_table),
                   zero_pixels_before=zero_before, zero_pixels_after=zero_after,
                   zero_partition=partition, zero_partition_exact=True,
                   eligible_pixels=int(eligible.sum()), assigned_pixels=int(assigned.sum()),
                   eligible_by_phase=by_phase(eligible), assigned_by_phase=by_phase(assigned),
                   eligible_nearest_cardinal_pixels=int(nearest_cardinal.sum()),
                   eligible_nearest_diagonal_pixels=int(nearest_diagonal.sum()),
                   eligible_tied_nearest_pixel_pixels=int(tied.sum()),
                   assigned_tied_nearest_pixel_pixels=int(np.count_nonzero(tied & assigned)),
                   tie_rule=TIE_RULE, scipy_version=scipy.__version__,
                   single_pass=True, original_positive_sources_only=True, outer_frame_unchanged=True,
                   GT_or_image_used=False, semantic_revote=False,
                   scope='Final-mask rule only; does not identify all zeros as watershed boundaries')
    return output, receipt
