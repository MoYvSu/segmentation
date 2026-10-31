# -*- coding: utf-8 -*-
"""NAF边界通道的显式线宽表示：阈值化后整图细化和受控膨胀。"""
from __future__ import annotations

import hashlib
from numbers import Real

import cv2
import numpy as np
from skimage.morphology import skeletonize


def input_normalization_settings(settings=None):
    """返回规范配方；缺省保持旧二值输入，不隐式启用标准化。"""
    if settings is None:
        return {'mode': 'none'}
    if not isinstance(settings, dict) or set(settings) - {'mode', 'kernel_size'}:
        raise ValueError('input_normalization must contain only mode/kernel_size')
    mode = settings.get('mode', 'none')
    if mode not in ('none', 'skeleton_dilate'):
        raise ValueError('input_normalization mode must be none or skeleton_dilate')
    size = settings.get('kernel_size', 8)
    if type(size) is not int or size < 1:
        raise ValueError('input_normalization kernel_size must be a positive integer')
    if mode == 'none':
        return {'mode': 'none'}
    return {'mode': mode, 'kernel_size': size}


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def normalize_boundary_input(boundary, *, threshold=.5, settings=None):
    """返回(binary_bool_HW, receipt)，高值始终表示边界。

    必须在原生整图调用，再作裁块/同步D4和reflect填充；不读取GT、valid或语义。
    ``skeleton_dilate``按作者实现使用skimage细化与OpenCV椭圆核、默认anchor。
    偶数核8的默认anchor为(4,4)，保留这个约定及其半像素非对称，不偷偷换成7/9。
    这一步可以改变数字连通或局部接触，不能称为严格保持拓扑的纯线宽变换。
    """
    if (not isinstance(boundary, np.ndarray) or boundary.ndim != 2 or not boundary.size
            or boundary.dtype.kind not in 'bufi' or not np.isfinite(boundary).all()
            or np.any((boundary < 0) | (boundary > 1))):
        raise ValueError('Boundary must be a nonempty finite HW array within [0,1]')
    if (not isinstance(threshold, Real) or isinstance(threshold, bool)
            or float(threshold) != .5):
        raise ValueError('Boundary input threshold must remain 0.5')
    cfg = input_normalization_settings(settings)
    binary = np.ascontiguousarray(boundary >= .5, dtype=bool)
    raw_sha = _array_sha(binary)
    if cfg['mode'] == 'none':
        return binary, dict(mode='none', threshold=.5, polarity='high_is_boundary',
                            output_binary_sha256=raw_sha, raw_binary_sha256=raw_sha)
    skeleton = np.ascontiguousarray(skeletonize(binary), dtype=np.uint8)
    size = cfg['kernel_size']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    normalized = cv2.dilate(skeleton, kernel, iterations=1, anchor=(-1, -1),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    normalized = np.ascontiguousarray(normalized, dtype=bool)
    return normalized, dict(**cfg, threshold=.5, polarity='high_is_boundary',
        algorithm='whole_native_threshold_skimage_skeletonize_cv2_ellipse_dilate',
        opencv_anchor_argument_xy=[-1, -1], resolved_anchor_xy=[size // 2, size // 2],
        even_kernel_anchor_offset_from_center_xy=[.5, .5] if size % 2 == 0 else [0., 0.],
        dilation_border='constant_zero', raw_binary_pixels=int(binary.sum()),
        skeleton_pixels=int(skeleton.sum()), output_binary_pixels=int(normalized.sum()),
        raw_binary_sha256=raw_sha, output_binary_sha256=_array_sha(normalized),
        resize_scale=1., GT_used=False, whole_image_before_crop=True,
        topology_preserved=False)
