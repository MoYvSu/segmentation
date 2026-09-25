# -*- coding: utf-8 -*-
"""裁块前的平滑空间模糊；仅改变允许的训练原图外观，不估计测试图焦面。"""

from __future__ import annotations

import cv2
import numpy as np


def validate_spatial_blur_config(cfg: dict | None) -> None:
    """关闭时保留旧路径；开启时检查实际使用的配置。"""
    if cfg is None:
        return
    if not isinstance(cfg, dict) or not isinstance(cfg.get("enabled", False), bool):
        raise ValueError("spatial_blur requires a mapping and boolean enabled")
    if not cfg.get("enabled", False):
        return
    for key in ("probability", "strength"):
        value = cfg.get(key, 0.5 if key == "probability" else 1.0)
        if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"spatial_blur.{key} must be finite and in [0,1]")
    grid = cfg.get("grid_size", [3, 5])
    if (not isinstance(grid, (tuple, list)) or len(grid) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in grid)
            or not 2 <= grid[0] <= grid[1]):
        raise ValueError("spatial_blur.grid_size requires integer 2 <= low <= high")
    levels = cfg.get("levels", 8)
    if isinstance(levels, bool) or not isinstance(levels, int) or levels < 2:
        raise ValueError("spatial_blur.levels must be an integer >= 2")
    endpoint = cfg.get("endpoint_transition")
    if endpoint is not None:
        validate_endpoint_transition_config(endpoint)


def validate_endpoint_transition_config(cfg: dict) -> None:
    """端点场只在显式启用后生效；弱强端采用固定诊断阈值1.5和4.0。"""
    prefix = "spatial_blur.endpoint_transition"
    if not isinstance(cfg, dict) or not isinstance(cfg.get("enabled", False), bool):
        raise ValueError(f"{prefix} requires a mapping and boolean enabled")
    if not cfg.get("enabled", False):
        return
    probability = cfg.get("probability", 0.5)
    if not np.isscalar(probability) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{prefix}.probability must be finite and in [0,1]")
    for name, default, lower, upper in (
            ("weak_sigma", [0.6, 1.5], 0.0, 1.5),
            ("strong_sigma", [4.0, 5.2], 4.0, float("inf")),
            ("transition_fraction", [0.2, 0.5], 0.0, 1.0)):
        values = np.asarray(cfg.get(name, default), dtype=np.float64)
        if (values.shape != (2,) or not np.isfinite(values).all()
                or not lower <= values[0] <= values[1] <= upper or values[0] <= 0):
            raise ValueError(f"{prefix}.{name} has invalid bounds")
    fraction = cfg.get("min_region_fraction", 0.05)
    if not np.isscalar(fraction) or not np.isfinite(fraction) or not 0 < fraction < 0.5:
        raise ValueError(f"{prefix}.min_region_fraction must be finite and in (0,0.5)")


def make_endpoint_sigma_map(shape: tuple[int, int], crop_box: tuple[int, int, int, int],
                            cfg: dict, rng: np.random.Generator, sigma_bounds) -> np.ndarray:
    """在有效整图生成随机方向平滑场，弱强两端锚定实际裁块。

    crop_box为(y,x,height,width)，只使用与有效内容相交的像素。平滑过渡带
    两侧至少保留min_region_fraction的裁块像素；不再匹配原base_sigma²均值。
    cfg是endpoint_transition子配置。σ仍是高斯插值的名义参数。
    """
    validate_endpoint_transition_config(cfg)
    if (len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                             or v <= 0 for v in shape)):
        raise ValueError("endpoint spatial blur requires a nonempty (height, width)")
    if (len(crop_box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                                for v in crop_box)):
        raise ValueError("endpoint spatial blur requires an integer crop_box (y,x,h,w)")
    h, w = map(int, shape)
    y, x, ch, cw = map(int, crop_box)
    if y < 0 or x < 0 or ch <= 0 or cw <= 0 or y >= h or x >= w:
        raise ValueError("endpoint spatial blur requires a crop intersecting valid content")
    ch, cw = min(ch, h - y), min(cw, w - x)
    if ch * cw < 2:
        raise ValueError("endpoint spatial blur requires at least two valid crop pixels")
    bounds = np.asarray(sigma_bounds, dtype=np.float64)
    weak_bounds, strong_bounds = cfg.get("weak_sigma", [0.6, 1.5]), cfg.get("strong_sigma", [4.0, 5.2])
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] <= 0
            or not bounds[0] <= weak_bounds[0] <= weak_bounds[1] < strong_bounds[0]
            <= strong_bounds[1] <= bounds[1]):
        raise ValueError("endpoint sigma bounds must contain configured weak and strong endpoints")
    weak, strong = float(rng.uniform(*weak_bounds)), float(rng.uniform(*strong_bounds))
    angle = float(rng.uniform(0, 2 * np.pi))
    dy, dx = np.sin(angle), np.cos(angle)
    # 从实际有效裁块的离散像素确定分位数，避免长宽比或padding破坏端点覆盖。
    crop_projection = np.arange(ch)[:, None] * dy + np.arange(cw)[None, :] * dx
    fraction = float(cfg.get("min_region_fraction", 0.05))
    low, high = np.quantile(crop_projection, [fraction, 1 - fraction])
    span = float(np.ptp(crop_projection))
    width = min(float(rng.uniform(*cfg.get("transition_fraction", [0.2, 0.5]))) * span,
                float(high - low))
    if width <= 0:
        raise ValueError("endpoint spatial blur requires a nondegenerate projected crop")
    start = float(rng.uniform(low, high - width))
    projection = ((np.arange(h)[:, None] - y) * dy + (np.arange(w)[None, :] - x) * dx)
    phase = np.clip((projection - start) / width, 0.0, 1.0)
    # 五次平滑阶跃在两端的一二阶导数均为零，不引入硬切换接缝。
    blend = phase ** 3 * (phase * (6 * phase - 15) + 10)
    variance = weak ** 2 + (strong ** 2 - weak ** 2) * blend
    return np.ascontiguousarray(np.sqrt(np.clip(variance, weak ** 2, strong ** 2)), dtype=np.float32)


def make_sigma_map(shape: tuple[int, int], base_sigma: float, cfg: dict,
                   rng: np.random.Generator, sigma_bounds) -> np.ndarray:
    """生成缓慢变化的σ场，整幅有效内容上的平均σ²保持为原σ₀²。

    低分辨率随机网格独立抽取高宽，双三次放大后去均值。按距上下界的
    余量整体缩放场，避免逐像素截断改变均值；范围仍在原高斯σ范围内。
    """
    validate_spatial_blur_config(cfg)
    if (len(shape) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                             or v <= 0 for v in shape)):
        raise ValueError("spatial blur requires a nonempty (height, width)")
    bounds = np.asarray(sigma_bounds, dtype=np.float64)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= bounds[1]
            or not np.isfinite(base_sigma) or not bounds[0] <= base_sigma <= bounds[1]):
        raise ValueError("spatial blur requires positive sigma bounds containing base_sigma")
    h, w = (int(v) for v in shape)
    strength = float(cfg.get("strength", 1.0)) if cfg.get("enabled", False) else 0.0
    if strength == 0 or base_sigma in bounds or h * w == 1:
        return np.full((h, w), base_sigma, dtype=np.float32)
    grid_low, grid_high = cfg.get("grid_size", [3, 5])
    gh, gw = rng.integers(grid_low, grid_high + 1, size=2)
    coarse = rng.uniform(-1.0, 1.0, size=(gh, gw))
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    field -= field.mean()
    negative, positive = float(-field.min()), float(field.max())
    if min(negative, positive) <= np.finfo(np.float64).eps:
        return np.full((h, w), base_sigma, dtype=np.float32)
    base_variance = float(base_sigma) ** 2
    amplitude = strength * min((base_variance - bounds[0] ** 2) / negative,
                               (bounds[1] ** 2 - base_variance) / positive)
    variance = base_variance + amplitude * field
    # 这里只消除浮点边界误差；构造本身已经保证上下界。
    variance = np.clip(variance, bounds[0] ** 2, bounds[1] ** 2)
    return np.ascontiguousarray(np.sqrt(variance), dtype=np.float32)


def apply_spatial_blur(clean: np.ndarray, base_sigma: float, cfg: dict,
                       rng: np.random.Generator, sigma_bounds) -> tuple[np.ndarray, np.ndarray]:
    """按σ²在线插值少量高斯结果；流式累加，避免保存整组RGB图。

    返回的σ场是按名义σ²加权的插值参数，不声称混合核严格等于高斯核，
    更不包含调用方随后可能施加的重采样、局部任务或照明变化。
    """
    if clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError("spatial blur requires an H x W x 3 RGB image")
    sigma_map = make_sigma_map(clean.shape[:2], base_sigma, cfg, rng, sigma_bounds)
    if np.ptp(sigma_map) == 0:
        image = cv2.GaussianBlur(clean, (0, 0), sigmaX=float(base_sigma),
                                 borderType=cv2.BORDER_REFLECT)
        return np.ascontiguousarray(image, dtype=np.float32), sigma_map
    return apply_sigma_map_blur(clean, sigma_map, sigma_bounds, cfg.get("levels", 8)), sigma_map


def apply_sigma_map_blur(clean: np.ndarray, sigma_map: np.ndarray, sigma_bounds,
                         levels: int = 8) -> np.ndarray:
    """沿用D4的σ²分段插值；端点场与旧空间场共享相同高斯级别。"""
    if clean.ndim != 3 or clean.shape[2] != 3 or sigma_map.shape != clean.shape[:2]:
        raise ValueError("sigma map must match the H x W extent of the RGB image")
    sigma_levels = np.linspace(*sigma_bounds, levels, dtype=np.float64)
    variance_levels = sigma_levels ** 2
    variance = sigma_map * sigma_map
    image = np.zeros_like(clean, dtype=np.float32)
    for index, sigma in enumerate(sigma_levels):
        if index == 0:
            weight = (variance_levels[1] - variance) / (variance_levels[1] - variance_levels[0])
        elif index == len(sigma_levels) - 1:
            weight = (variance - variance_levels[-2]) / (variance_levels[-1] - variance_levels[-2])
        else:
            left = (variance - variance_levels[index - 1]) / (variance_levels[index] - variance_levels[index - 1])
            right = (variance_levels[index + 1] - variance) / (variance_levels[index + 1] - variance_levels[index])
            weight = np.minimum(left, right)
        weight = np.clip(weight, 0, 1).astype(np.float32, copy=False)
        if not np.any(weight):
            continue
        blurred = cv2.GaussianBlur(clean, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT)
        np.multiply(blurred, weight[..., None], out=blurred)
        np.add(image, blurred, out=image)
    return np.ascontiguousarray(image, dtype=np.float32)
