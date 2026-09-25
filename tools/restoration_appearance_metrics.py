# -*- coding: utf-8 -*-
"""修复前后的同网格 RGB 外观度量；不把低频亮度解释为物理光照。

输入应为模型输入网格中裁去 padding 的有效内容，两个数组逐像素对应。
不得把原生分辨率的原图与经过缩放的修复图直接传入作比较。
"""

from __future__ import annotations

import cv2
import numpy as np


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma, sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )


def _features(rgb: np.ndarray) -> dict[str, np.ndarray]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    lightness, a, b = (lab[..., i] for i in range(3))
    fine = _gaussian(lightness, 1.5)
    medium = _gaussian(lightness, 8.0)
    grad_x = cv2.Sobel(
        lightness, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0,
        borderType=cv2.BORDER_REFLECT,
    )
    grad_y = cv2.Sobel(
        lightness, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0,
        borderType=cv2.BORDER_REFLECT,
    )
    # 先减掉常数中心，减少 E[L²]-E[L]² 的浮点相消；再钳制微小负方差。
    centered = lightness - np.float32(np.mean(lightness, dtype=np.float64))
    local_mean = _gaussian(centered, 4.0)
    local_variance = _gaussian(centered * centered, 4.0) - local_mean * local_mean
    return {
        "R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2],
        "L": lightness, "a": a, "b": b,
        "chroma": np.hypot(a, b),
        "low_L": _gaussian(lightness, 16.0),
        "high_L": np.abs(lightness - fine),
        "mid_L": np.abs(fine - medium),
        "grad_L": np.hypot(grad_x, grad_y),
        "std_L": np.sqrt(np.maximum(local_variance, np.float32(0.0))),
    }


def appearance_maps(
    before: np.ndarray, after: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """返回逐像素度量和可直接写 JSON 的全图统计。

    ``maps`` 的键使用 ``raw_``、``restored_``、``delta_`` 前缀，delta
    总是 after-before。RGB 单位为 [0,1]；Lab 使用 L* [0,100]、带符号
    a*/b*；delta_ab 是色度平面位移，deltaE76 是 Lab 欧氏距离。
    high_L/mid_L/grad_L/std_L 是纹理强度指标，增加不等同于补边正确。
    Gaussian sigma 的单位是当前模型网格的像素。低频亮度包含材质和
    结构，不能独立识别曝光、光照或真实反射率。
    """
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    if before.shape != after.shape:
        raise ValueError("before 和 after 必须形状相同，且对应同一模型网格")
    if before.ndim != 3 or before.shape[2] != 3 or min(before.shape[:2]) == 0:
        raise ValueError("输入必须是非空的 HWC 三通道 RGB")
    for name, value in (("before", before), ("after", after)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} 含有非有限值")
        if np.min(value) < 0.0 or np.max(value) > 1.0:
            raise ValueError(f"{name} 必须位于 [0,1]，不能隐式截断后比较")
    raw = _features(np.ascontiguousarray(before))
    restored = _features(np.ascontiguousarray(after))
    maps: dict[str, np.ndarray] = {}
    for key in raw:
        maps[f"raw_{key}"] = raw[key]
        maps[f"restored_{key}"] = restored[key]
        maps[f"delta_{key}"] = restored[key] - raw[key]
    for channel in "RGB":
        maps[f"abs_delta_{channel}"] = np.abs(maps[f"delta_{channel}"])
    maps["delta_ab"] = np.hypot(maps["delta_a"], maps["delta_b"])
    maps["deltaE76"] = np.sqrt(
        maps["delta_L"] ** 2 + maps["delta_a"] ** 2 + maps["delta_b"] ** 2
    )

    statistics = {}
    for key, values in maps.items():
        p05, p50, p95 = np.percentile(values, [5, 50, 95])
        statistics[key] = {
            "mean": float(np.mean(values, dtype=np.float64)),
            "std": float(np.std(values, dtype=np.float64)),
            "mean_abs": float(np.mean(np.abs(values), dtype=np.float64)),
            "p05": float(p05), "p50": float(p50), "p95": float(p95),
        }
    summary = {
        "height": int(before.shape[0]), "width": int(before.shape[1]),
        "pixels": int(before.shape[0] * before.shape[1]),
        "statistics": statistics,
        "method": {
            "delta_direction": "restored_minus_raw",
            "rgb_units": "0_to_1",
            "lab_conversion": "opencv_float_RGB2Lab_D65_sRGB",
            "lab_units": "L_star_0_to_100_a_b_signed",
            "low_L": "Gaussian_L_sigma16",
            "high_L": "abs_L_minus_Gaussian_L_sigma1.5",
            "mid_L": "abs_Gaussian_L_sigma1.5_minus_sigma8",
            "grad_L": "Sobel3_magnitude_scale_1_over_8",
            "std_L": "Gaussian_local_std_sigma4_centered_variance_clamped_at_zero",
            "gaussian_sigma_units": "input_model_grid_pixels",
            "border": "BORDER_REFLECT",
            "interpretation": "Appearance_diagnostic_not_physical_illumination_or_quality_ground_truth",
        },
    }
    return maps, summary
