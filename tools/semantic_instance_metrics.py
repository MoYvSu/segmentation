# -*- coding: utf-8 -*-
"""固定实例内的语义诊断；不生成标签、不改变部署投票。

输入为同尺寸二维 bool 实例掩码和铁素体概率，可为紧包围框裁剪。
距离变换前补一圈零，因而裁剪框或整幅图边缘不会被当成无限内部。
analyze_instance 输出全部使用 Python 标量；全区连通块比例以实例面积为分母，
interior 连通块比例以内区面积为分母。
"""
from __future__ import annotations

import cv2
import numpy as np


SCHEMA_VERSION = "semantic_instance_v1"


def _checked_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional bool array (instance_ids == id)")
    if not mask.any():
        raise ValueError("instance mask must not be empty")
    return mask


def instance_bands(mask: np.ndarray) -> dict:
    """返回距离图、5层编号、外围及内部掩码，供统计和固定色标绘图。

    layers: 外部为0，实例内1..5对应归一化距离的(0,.2]..(.8,1]。
    outer: 距离<=30%分位数；core: 距离>=60%分位数。
    interior: 距离>max(2像素,.15*最大距离)，用于剔除外缘后再检查两类强预测。
    分位数处同距离像素全部保留，不随机打破平局，故面积不严格等于30/40%。
    细小实例仍保留原始统计，但 meaningful_core=False 时不可据此解释核心翻转。
    """
    mask = _checked_mask(mask)
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    values = distance[mask]
    maximum = float(values.max())
    low, high = np.quantile(values, [.30, .60])
    outer = mask & (distance <= low)
    core = mask & (distance >= high)
    interior_cutoff = max(2.0, .15 * maximum)
    interior = mask & (distance > interior_cutoff)
    layers = np.zeros(mask.shape, dtype=np.uint8)
    layers[mask] = np.clip(np.ceil(values / maximum * 5), 1, 5).astype(np.uint8)
    reasons = []
    if int(mask.sum()) < 64:
        reasons.append("area_below_64")
    if maximum < 3:
        reasons.append("maximum_distance_below_3")
    if int(core.sum()) < 32:
        reasons.append("core_below_32")
    if int(outer.sum()) < 32:
        reasons.append("outer_below_32")
    if np.any(core & outer):
        reasons.append("distance_quantiles_overlap")
    return {
        "distance": distance,
        "layers": layers,
        "outer": outer,
        "core": core,
        "interior": interior,
        "interior_distance_cutoff": interior_cutoff,
        "distance_max": maximum,
        "outer_distance_max": float(low),
        "core_distance_min": float(high),
        "meaningful_core": not reasons,
        "core_limitations": reasons,
    }


def _summary(probability: np.ndarray, mask: np.ndarray) -> dict:
    values = probability[mask]
    count = int(values.size)
    if not count:
        return {"area": 0, "mean_probability": None, "hard_ratio": None}
    return {
        "area": count,
        "mean_probability": float(np.mean(values, dtype=np.float64)),
        "hard_ratio": float(np.mean(values > .5)),
    }


def _strong_components(selected: np.ndarray, area: int) -> dict:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), connectivity=8
    )
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest = int(sizes.max()) if sizes.size else 0
    pixels = int(sizes.sum())
    return {
        "area": pixels,
        "fraction": pixels / area if area else 0.0,
        "component_count": int(count - 1),
        "largest_component": largest,
        "largest_fraction": largest / area if area else 0.0,
    }


def analyze_instance(mask: np.ndarray, probability: np.ndarray, *, bands: dict | None = None) -> dict:
    """计算一个固定实例的可序列化诊断记录。

    mean_class/hard_class: 分数严格大于.5为铁素体1，否则珠光体0。
    uncertain_fraction: .4<=P(F)<=.6；强预测F>=.8、P<=.2。
    heterogeneous: 两类各自的最大8连通块同时占实例>=10%且>=32像素。
    interior: distance>max(2像素,.15*最大距离)；内部连通块比例以内区面积为分母，
    interior_heterogeneous 使用同样10%/32像素门槛，帮助区分外围暗环与内部混合。
    whole_vs_core_flip: 仅有足够独立内部时比较全实例均值与核心均值定类。
    多模型比较应传入同一次 instance_bands(mask) 的 bands；避免精确距离变换
    多线程实现的末位舍入差异使等距分层边界像素在模型间发生变化。
    这些固定阈值只组织诊断，不代表GT，也不是候选部署规则。
    """
    mask = _checked_mask(mask)
    probability = np.asarray(probability)
    if probability.ndim != 2 or probability.shape != mask.shape:
        raise ValueError("probability shape must match the two-dimensional mask")
    values = probability[mask]
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("probabilities inside the instance must be finite and in [0, 1]")
    if bands is None:
        bands = instance_bands(mask)
    else:
        for key in ("distance", "layers", "outer", "core", "interior"):
            if key not in bands or np.asarray(bands[key]).shape != mask.shape:
                raise ValueError("precomputed bands shape must match mask: " + key)
        if not np.array_equal(bands["layers"] > 0, mask) or not np.array_equal(bands["distance"] > 0, mask):
            raise ValueError("precomputed bands must describe the same instance mask")
        if any(np.any(bands[key] & ~mask) for key in ("outer", "core", "interior")):
            raise ValueError("precomputed bands must be subsets of the instance mask")
    overall = _summary(probability, mask)
    area = overall["area"]
    outer = _summary(probability, bands["outer"])
    core = _summary(probability, bands["core"])
    strong_f = _strong_components(mask & (probability >= .8), area)
    strong_p = _strong_components(mask & (probability <= .2), area)
    interior = _summary(probability, bands["interior"])
    interior_strong_f = _strong_components(bands["interior"] & (probability >= .8), interior["area"])
    interior_strong_p = _strong_components(bands["interior"] & (probability <= .2), interior["area"])
    whole_class = int(overall["mean_probability"] > .5)
    meaningful = bands["meaningful_core"]
    return {
        "schema_version": SCHEMA_VERSION,
        **overall,
        "mean_class": whole_class,
        "hard_class": int(overall["hard_ratio"] > .5),
        "mean_margin": abs(overall["mean_probability"] - .5),
        "uncertain_fraction": float(np.mean((values >= .4) & (values <= .6))),
        "distance_max": bands["distance_max"],
        "outer_distance_max": bands["outer_distance_max"],
        "core_distance_min": bands["core_distance_min"],
        "layers": [
            {"index": i, "distance_low_fraction": (i - 1) / 5,
             "distance_high_fraction": i / 5,
             **_summary(probability, bands["layers"] == i)}
            for i in range(1, 6)
        ],
        "outer": outer,
        "core": core,
        "meaningful_core": meaningful,
        "core_limitations": bands["core_limitations"],
        "core_minus_outer": core["mean_probability"] - outer["mean_probability"],
        "whole_vs_core_flip": (
            bool(whole_class != int(core["mean_probability"] > .5)) if meaningful else None
        ),
        "strong_f": strong_f,
        "strong_p": strong_p,
        "heterogeneous": all(
            component["largest_component"] >= 32 and component["largest_fraction"] >= .10
            for component in (strong_f, strong_p)
        ),
        "interior": interior,
        "interior_distance_cutoff": bands["interior_distance_cutoff"],
        "interior_strong_f": interior_strong_f,
        "interior_strong_p": interior_strong_p,
        "interior_heterogeneous": all(
            component["largest_component"] >= 32 and component["largest_fraction"] >= .10
            for component in (interior_strong_f, interior_strong_p)
        ),
    }
