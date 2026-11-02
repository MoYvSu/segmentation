# -*- coding: utf-8 -*-
"""低复杂度、无参数更新的单图质量感知推理辅助。

所有判断只依赖当前输入图像及预先写入配置的阈值；不聚合测试集统计，
不学习参数。第一版仅区分 ``standard`` / ``weak`` 两档，并只允许小幅调整
边界阈值与融合一张确定性增强视图，便于定位问题和逐项消融。
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np


def assess_image_quality(image_rgb: np.ndarray, size: int = 256) -> Dict[str, float]:
    """计算尺度固定的亮度、对比度、清晰度和通道偏色统计。"""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got {image_rgb.shape}")
    sample = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    channel_mean = sample.reshape(-1, 3).mean(axis=0) / 255.0
    return {
        "brightness": float(np.median(gray_f)),
        "contrast": float(np.percentile(gray_f, 90) - np.percentile(gray_f, 10)),
        "sharpness": float(cv2.Laplacian(gray_f, cv2.CV_32F).var()),
        "color_cast": float(channel_mean.max() - channel_mean.min()),
    }


def classify_quality(
    metrics: Dict[str, float], config: Dict,
) -> Tuple[str, List[str]]:
    """按少量可解释规则分为 standard/weak，不预测实例数或面积。"""
    thresholds = config.get("thresholds", {})
    reasons = []
    if metrics["brightness"] < float(thresholds.get("brightness_low", 0.58)):
        reasons.append("low_brightness")
    if metrics["contrast"] < float(thresholds.get("contrast_low", 0.14)):
        reasons.append("low_contrast")
    if metrics["sharpness"] < float(thresholds.get("sharpness_low", 0.02)):
        reasons.append("low_sharpness")
    if metrics["color_cast"] > float(thresholds.get("color_cast_high", 0.04)):
        reasons.append("color_cast")

    min_flags = max(1, int(config.get("weak_min_flags", 2)))
    critical_sharpness = float(thresholds.get("sharpness_critical", 0.008))
    weak = len(reasons) >= min_flags or metrics["sharpness"] < critical_sharpness
    return ("weak" if weak else "standard"), reasons


def enhance_weak_image(
    image_rgb: np.ndarray, metrics: Dict[str, float], config: Dict,
) -> Tuple[np.ndarray, List[str]]:
    """生成一张温和确定性增强视图；原图始终保留并参与 logits 融合。"""
    result = image_rgb.astype(np.float32) / 255.0
    applied: List[str] = []
    thresholds = config.get("thresholds", {})
    enhance = config.get("enhance", {})

    if (
        bool(enhance.get("white_balance", True))
        and metrics["color_cast"] > float(thresholds.get("color_cast_high", 0.04))
    ):
        means = result.reshape(-1, 3).mean(axis=0)
        target = float(means.mean())
        gain_limit = float(enhance.get("white_balance_gain_limit", 0.12))
        gains = target / np.maximum(means, 1e-6)
        gains = np.clip(gains, 1.0 - gain_limit, 1.0 + gain_limit)
        result *= gains.reshape(1, 1, 3)
        applied.append("white_balance")

    if (
        bool(enhance.get("gamma", True))
        and metrics["brightness"] < float(thresholds.get("brightness_low", 0.58))
    ):
        target = float(enhance.get("target_brightness", 0.65))
        current = float(np.clip(metrics["brightness"], 1e-3, 0.999))
        gamma = float(np.log(np.clip(target, 1e-3, 0.999)) / np.log(current))
        gamma = float(np.clip(
            gamma,
            float(enhance.get("gamma_min", 0.75)),
            float(enhance.get("gamma_max", 1.25)),
        ))
        result = np.power(np.clip(result, 0.0, 1.0), gamma)
        applied.append("gamma")

    if (
        bool(enhance.get("clahe", True))
        and metrics["contrast"] < float(thresholds.get("contrast_low", 0.14))
    ):
        uint8 = (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)
        lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=float(enhance.get("clahe_clip", 1.5)),
            tileGridSize=(8, 8),
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        applied.append("clahe")

    if (
        bool(enhance.get("unsharp", True))
        and metrics["sharpness"] < float(thresholds.get("sharpness_low", 0.02))
    ):
        sigma = float(enhance.get("unsharp_sigma", 1.0))
        amount = float(enhance.get("unsharp_amount", 0.35))
        blurred = cv2.GaussianBlur(result, (0, 0), sigma)
        result = result + amount * (result - blurred)
        applied.append("unsharp")

    return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8), applied


def effective_boundary_threshold(
    base_threshold: float, profile: str, config: Dict,
) -> float:
    """只允许配置内的小偏移，不根据实例数或平均面积闭环调节。"""
    offsets = config.get("boundary_threshold_offsets", {})
    offset = float(offsets.get(profile, 0.0))
    low, high = config.get("boundary_threshold_limits", [0.05, 0.95])
    return float(np.clip(base_threshold + offset, float(low), float(high)))
