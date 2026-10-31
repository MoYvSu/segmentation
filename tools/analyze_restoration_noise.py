# -*- coding: utf-8 -*-
"""固定恢复器噪声实验的 CPU 分层诊断；比较预测变化，不把它当作准确率。

输入约定：maps/{source}/{case}.npz 包含 boundary、geometry_semantic（logits）、
raw_probability、instances；results/{case}/ 包含对应 uint16 PNG 与类别 JSON。
所有派生产物只写入实验根目录，不能进入公开仓库或正式提交包。
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.special import expit


DEFAULT_CASES = (
    "control", *(f"local_s{i}" for i in range(8)),
    *(f"global_s{i}" for i in range(8)),
    "local_zero", "global_zero", "both_zero",
    "local_pair", "global_pair", "both_pair",
)
MAP_KEYS = ("boundary", "geometry_semantic", "raw_probability", "instances")
QUALITY_METRICS = ("rgb_mse", "gradient_l1", "weak_gradient_ratio",
                   "weak_directional_retention", "strong_directional_retention",
                   "novel_edge_fraction", "flat_gradient_mean")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def safe_name(value: str) -> str:
    if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in "/\\\0"):
        raise ValueError(f"Invalid experiment item name: {value!r}")
    return value


def class_lookup(instances: np.ndarray, classes: dict[str, int]) -> np.ndarray:
    """核对逐实例类别；背景用 255，避免把背景误算为珠光体。"""
    if instances.ndim != 2 or instances.dtype != np.uint16 or 0 in instances.shape:
        raise ValueError("instances must be a single-channel uint16 array")
    ids = np.unique(instances)
    positive = {int(i) for i in ids if i > 0}
    converted: dict[int, int] = {}
    if not isinstance(classes, dict):
        raise ValueError("class JSON must be an object")
    for key, value in classes.items():
        if (not isinstance(key, str) or not key.isdecimal() or int(key) <= 0
                or str(int(key)) != key or type(value) is not int or value not in (0, 1)):
            raise ValueError("class IDs must be canonical positive strings and values must be 0/1")
        converted[int(key)] = value
    if positive != set(converted):
        raise ValueError("PNG IDs and class JSON IDs must match exactly")
    lookup = np.full(65536, 255, dtype=np.uint8)
    for key, value in converted.items():
        lookup[key] = value
    return lookup


def ferrite_stats(instances: np.ndarray, classes: dict[str, int]) -> dict[str, Any]:
    lookup = class_lookup(instances, classes)
    areas = np.bincount(instances.ravel(), minlength=65536)
    ferrite_ids = np.flatnonzero((lookup == 1) & (areas > 0))
    count = int(ferrite_ids.size)
    total = int(areas[ferrite_ids].sum())
    return dict(count=count, total_pixels=total,
                mean_area=float(total / count) if count else None,
                instance_count=int(np.count_nonzero(areas[1:])),
                covered_pixels=int(np.count_nonzero(instances)))


def relative_delta(value: float | int | None, reference: float | int | None) -> float | None:
    return float((value - reference) / reference) if value is not None and reference else None


def match_instances(reference: np.ndarray, candidate: np.ndarray,
                    reference_classes: dict[str, int], candidate_classes: dict[str, int],
                    minimum_iou: float = .5) -> dict[str, Any]:
    """先以重叠面积配对实例，再比较类别；实例编号重排不产生几何误差。

    Hungarian 配对最大化所有配对 IoU 总和，随后仅保留 IoU>=0.5 的配对。
    这只是两次预测的一致性，不是相对 GT 的实例准确率。
    """
    if reference.shape != candidate.shape:
        raise ValueError("Instance shapes differ")
    ref_lookup = class_lookup(reference, reference_classes)
    cand_lookup = class_lookup(candidate, candidate_classes)
    ref_ids, ref_inverse, ref_areas = np.unique(reference, return_inverse=True, return_counts=True)
    cand_ids, cand_inverse, cand_areas = np.unique(candidate, return_inverse=True, return_counts=True)
    positive_ref = ref_ids > 0
    positive_cand = cand_ids > 0
    # 实验样本通常仅数百实例；异常碎分割明确报错，避免静默耗尽 CPU 内存。
    if int(ref_ids.size) * int(cand_ids.size) > 16_000_000:
        raise ValueError("Too many instances for bounded dense Hungarian diagnostic")
    overlap = np.bincount(ref_inverse.ravel() * cand_ids.size + cand_inverse.ravel(),
                          minlength=ref_ids.size * cand_ids.size).reshape(ref_ids.size, cand_ids.size)
    union = ref_areas[:, None] + cand_areas[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=np.float64), where=union > 0)
    iou = iou[np.ix_(positive_ref, positive_cand)]
    ref_ids = ref_ids[positive_ref]
    cand_ids = cand_ids[positive_cand]
    row, col = linear_sum_assignment(iou, maximize=True)
    retained = iou[row, col] >= minimum_iou
    row, col = row[retained], col[retained]
    values = iou[row, col]
    pairs = []
    for ri, ci, value in zip(row, col, values):
        rid, cid = int(ref_ids[ri]), int(cand_ids[ci])
        pairs.append(dict(reference_id=rid, candidate_id=cid, iou=float(value),
                          reference_class=int(ref_lookup[rid]), candidate_class=int(cand_lookup[cid])))
    return dict(minimum_iou=minimum_iou, matched_count=len(pairs),
                unmatched_reference=int(ref_ids.size - len(pairs)),
                unmatched_candidate=int(cand_ids.size - len(pairs)),
                mean_matched_iou=float(values.mean()) if values.size else None,
                minimum_matched_iou=float(values.min()) if values.size else None,
                matched_class_changes=sum(p["reference_class"] != p["candidate_class"] for p in pairs),
                ferrite_to_pearlite=sum(p["reference_class"] == 1 and p["candidate_class"] == 0 for p in pairs),
                pearlite_to_ferrite=sum(p["reference_class"] == 0 and p["candidate_class"] == 1 for p in pairs),
                pairs=pairs)


def compare_maps(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
                 reference_classes: dict[str, int], candidate_classes: dict[str, int]) -> dict[str, Any]:
    for key in MAP_KEYS:
        if reference[key].shape != candidate[key].shape:
            raise ValueError(f"Different shapes for {key}")
    reference_boundary = reference["boundary"]
    boundary_delta = np.abs(candidate["boundary"] - reference_boundary)
    reference_phase = class_lookup(reference["instances"], reference_classes)[reference["instances"]]
    candidate_phase = class_lookup(candidate["instances"], candidate_classes)[candidate["instances"]]
    common = (reference_phase != 255) & (candidate_phase != 255)
    changed = (reference_phase != candidate_phase) & common
    stats = ferrite_stats(candidate["instances"], candidate_classes)
    ref_stats = ferrite_stats(reference["instances"], reference_classes)
    return dict(
        boundary=dict(mae=float(boundary_delta.mean()), maximum_delta=float(boundary_delta.max()),
                      threshold_crossing={str(t): float(np.mean((reference_boundary > t)
                                                              != (candidate["boundary"] > t)))
                                          for t in (.65, .45)}),
        geometry_semantic=dict(probability_mae=float(np.mean(np.abs(expit(reference["geometry_semantic"])
                                                     - expit(candidate["geometry_semantic"])))),
                               threshold_crossing=float(np.mean((reference["geometry_semantic"] > 0)
                                                               != (candidate["geometry_semantic"] > 0)))),
        raw_semantic=dict(probability_mae=float(np.mean(np.abs(reference["raw_probability"]
                                                             - candidate["raw_probability"]))),
                          threshold_crossing=float(np.mean((reference["raw_probability"] > .5)
                                                          != (candidate["raw_probability"] > .5)))),
        phase=dict(common_covered_pixels=int(common.sum()), changed_pixels=int(changed.sum()),
                   changed_fraction=float(changed.sum() / common.sum()) if common.any() else None,
                   coverage_changed_fraction=float(np.mean((reference_phase != 255) != (candidate_phase != 255)))),
        ferrite=stats,
        ferrite_relative_delta={key: relative_delta(stats[key], ref_stats[key])
                                for key in ("count", "total_pixels", "mean_area")},
        instances=match_instances(reference["instances"], candidate["instances"],
                                  reference_classes, candidate_classes),
    )


def coefficient_of_variation(values: list[float | int | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    return float(array.std(ddof=0) / abs(array.mean())) if array.mean() else None


def series_summary(cases: dict[str, tuple[dict[str, np.ndarray], dict[str, int]]],
                   expected: list[str], reference: tuple[dict[str, np.ndarray], dict[str, int]]) -> tuple[dict, np.ndarray | None]:
    """噪声种子用总体标准差；缺失时仍明确标记，不能把它当完整八种子实验。"""
    available = [case for case in expected if case in cases]
    missing = [case for case in expected if case not in cases]
    result: dict[str, Any] = dict(expected_cases=expected, available_cases=available,
                                  missing_cases=missing, complete=not missing, sample_count=len(available))
    if len(available) < 2:
        result["insufficient_repeats"] = True
        return result, None
    maps = [cases[case][0] for case in available]
    boundary = np.stack([item["boundary"] for item in maps])
    std = boundary.std(axis=0, dtype=np.float64).astype(np.float32)
    result["boundary_std"] = dict(mean=float(std.mean()), p95=float(np.quantile(std, .95)),
                                   p99=float(np.quantile(std, .99)), maximum=float(std.max()))
    result["geometry_semantic_probability_std_mean"] = float(
        np.std(np.stack([expit(item["geometry_semantic"]) for item in maps]), axis=0, dtype=np.float64).mean())
    result["raw_probability_std_mean"] = float(
        np.std(np.stack([item["raw_probability"] for item in maps]), axis=0, dtype=np.float64).mean())
    phases = np.stack([class_lookup(cases[case][0]["instances"], cases[case][1])
                       [cases[case][0]["instances"]] for case in available])
    # 背景变化单列；珠光体/铁素体翻转仅在所有种子都有覆盖的位置统计。
    all_covered = np.all(phases != 255, axis=0)
    any_change = (phases.min(axis=0) != phases.max(axis=0)) & all_covered
    result["phase_any_seed_change_fraction"] = float(any_change.sum() / all_covered.sum()) if all_covered.any() else None
    result["phase_all_seed_covered_pixels"] = int(all_covered.sum())
    result["coverage_any_seed_change_fraction"] = float(np.mean(np.any(phases == 255, axis=0)
                                                               & np.any(phases != 255, axis=0)))
    ref_map, ref_classes = reference
    ref_phase = class_lookup(ref_map["instances"], ref_classes)[ref_map["instances"]]
    common = (phases != 255) & (ref_phase[None] != 255)
    result["phase_mean_change_fraction_from_control"] = float(np.count_nonzero((phases != ref_phase) & common)
                                                            / np.count_nonzero(common)) if common.any() else None
    statistics = [ferrite_stats(cases[case][0]["instances"], cases[case][1]) for case in available]
    result["ferrite_cv"] = {key: coefficient_of_variation([item[key] for item in statistics])
                             for key in ("count", "total_pixels", "mean_area")}
    return result, std


def quality_summary(payload: dict) -> dict:
    """先对每张训练源图的 offset 求均值，再对源图宏平均，避免多种子图权重更大。

    rgb_mse 的 offset 标准差是重建误差标量的波动，不是预测图像的像素方差。
    训练原图参加过恢复器训练，这里只能诊断机制，不能代替泛化或实例 GT 评估。
    """
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError("Quality reconstruction run is not complete")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Quality reconstruction rows are missing")
    buckets: dict[tuple[str, str, str], list[dict]] = {}
    identities = set()

    def check_metrics(row):
        if not isinstance(row, dict):
            raise ValueError("Quality metrics must be an object")
        for key in QUALITY_METRICS:
            if key not in row:
                raise ValueError("Missing quality metric: " + key)
            value = row[key]
            if value is not None and (type(value) not in (int, float) or not np.isfinite(value)):
                raise ValueError("Nonfinite or invalid quality metric: " + key)

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid quality row")
        source, condition, mode = row.get("source"), row.get("condition"), row.get("mode")
        safe_name(source); safe_name(condition)
        if mode not in ("single", "pair", "zero") or type(row.get("offset")) is not int:
            raise ValueError("Invalid quality mode or offset")
        identity = source, condition, mode, row["offset"]
        if identity in identities:
            raise ValueError("Duplicate quality source/condition/mode/offset")
        identities.add(identity)
        check_metrics(row)
        buckets.setdefault((source, condition, mode), []).append(row)
    sources = sorted({key[0] for key in buckets})
    conditions = sorted({key[1] for key in buckets})
    if payload.get("plan") is not None:
        plan = payload["plan"]
        if (not isinstance(plan, list) or not all(isinstance(item, dict) for item in plan)
                or len(plan) != len(sources) or {item.get("source") for item in plan} != set(sources)):
            raise ValueError("Quality sources differ from predeclared plan")
    source_rows = []
    inputs = {}
    for source in sources:
        for condition in conditions:
            current = {}
            for mode in ("single", "pair", "zero"):
                group = buckets.get((source, condition, mode))
                if not group:
                    raise ValueError(f"Missing quality group: {source}/{condition}/{mode}")
                current[mode] = group
            if {row["offset"] for row in current["single"]} != {row["offset"] for row in current["pair"]}:
                raise ValueError("Single/pair quality offsets differ")
            if len(current["zero"]) != 1:
                raise ValueError("Zero-noise quality group must have exactly one row")
            all_rows = [row for group in current.values() for row in group]
            if any("input_metrics" in row for row in all_rows):
                reference = all_rows[0].get("input_metrics")
                check_metrics(reference)
                if any(row.get("input_metrics") != reference for row in all_rows):
                    raise ValueError("Degraded input metrics changed between quality modes")
                inputs[(source, condition)] = reference
            for mode, group in current.items():
                means, counts = {}, {}
                for key in QUALITY_METRICS:
                    values = [row[key] for row in group if row[key] is not None]
                    means[key] = float(np.mean(values)) if values else None
                    counts[key] = len(values)
                mse_values = [row["rgb_mse"] for row in group]
                scalar_std = (float(np.std(mse_values, ddof=0))
                              if mode in ("single", "pair") and len(mse_values) >= 2
                              and all(value is not None for value in mse_values) else None)
                source_rows.append(dict(source=source, condition=condition, mode=mode,
                    offsets=sorted(row["offset"] for row in group), offset_count=len(group),
                    mean_metrics=means, valid_offset_counts=counts, rgb_mse_across_offset_std=scalar_std))
    groups = {}
    for condition in conditions:
        groups[condition] = {}
        for mode in ("single", "pair", "zero"):
            group = [row for row in source_rows if row["condition"] == condition and row["mode"] == mode]
            metrics, counts = {}, {}
            for key in QUALITY_METRICS:
                values = [row["mean_metrics"][key] for row in group if row["mean_metrics"][key] is not None]
                metrics[key] = float(np.mean(values)) if values else None
                counts[key] = len(values)
            deviations = [row["rgb_mse_across_offset_std"] for row in group
                          if row["rgb_mse_across_offset_std"] is not None]
            groups[condition][mode] = dict(source_count=len(group), macro_mean_metrics=metrics,
                valid_source_counts=counts,
                rgb_mse_across_offset_std_macro_mean=float(np.mean(deviations)) if deviations else None,
                rgb_mse_across_offset_std_valid_sources=len(deviations))
        if inputs:
            metrics, counts = {}, {}
            for key in QUALITY_METRICS:
                values = [inputs[(source, condition)][key] for source in sources
                          if (source, condition) in inputs and inputs[(source, condition)][key] is not None]
                metrics[key] = float(np.mean(values)) if values else None
                counts[key] = len(values)
            groups[condition]["degraded_input"] = dict(source_count=sum((source, condition) in inputs for source in sources),
                                                       macro_mean_metrics=metrics, valid_source_counts=counts)
    return dict(complete=True, scope="Training-source reconstruction mechanism diagnostic; no held-out or instance GT accuracy",
                aggregation="Per-source mean over offsets, then equal-weight macro mean over sources; ddof=0 for scalar MSE std",
                scalar_std_warning="RGB MSE across-offset std measures scalar error variation, not prediction pixel variance",
                sources=sources, conditions=conditions, groups=groups, per_source=source_rows)


def load_case(root: Path, source: str, case: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    map_path = root / "maps" / source / f"{case}.npz"
    with np.load(map_path, allow_pickle=False) as archive:
        missing = set(MAP_KEYS) - set(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ keys: {sorted(missing)}")
        maps = {key: archive[key] for key in MAP_KEYS}
    shape = maps["instances"].shape
    for key in MAP_KEYS:
        value = maps[key]
        if value.ndim != 2 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid map shape or finite values: {key}")
        if key != "instances" and value.dtype != np.float32:
            raise ValueError(f"Expected float32 map: {key}")
    for key in ("boundary", "raw_probability"):
        if np.any((maps[key] < 0) | (maps[key] > 1)):
            raise ValueError(f"Expected probability range [0,1]: {key}")
    classes = read_json(root / "results" / case / f"{source}_class.json")
    class_lookup(maps["instances"], classes)
    result_path = root / "results" / case / f"{source}_inst.png"
    png = cv2.imread(str(result_path), cv2.IMREAD_UNCHANGED)
    if png is None or png.dtype != np.uint16 or png.ndim != 2 or not np.array_equal(png, maps["instances"]):
        raise ValueError(f"PNG differs from archived instance map: {result_path}")
    return maps, classes


def phase_picture(maps: dict[str, np.ndarray], classes: dict[str, int]) -> Image.Image:
    phase = class_lookup(maps["instances"], classes)[maps["instances"]]
    rgb = np.full((*phase.shape, 3), 28, dtype=np.uint8)
    rgb[phase == 0] = (222, 171, 66)
    rgb[phase == 1] = (71, 154, 226)
    labels = maps["instances"]
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    rgb[edge] = (245, 245, 245)
    return Image.fromarray(rgb)


def make_panel(pictures: list[tuple[str, Image.Image]], output: Path, width: int = 384) -> None:
    tiles = []
    for label, picture in pictures:
        picture = picture.copy()
        picture.thumbnail((width, width), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (width, width + 32), "#20242b")
        tile.paste(picture, ((width - picture.width) // 2, 32 + (width - picture.height) // 2))
        ImageDraw.Draw(tile).text((8, 8), label, fill="white")
        tiles.append(tile)
    canvas = Image.new("RGB", (width * len(tiles), width + 32))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, (index * width, 0))
    canvas.save(output)


def instability_roi(std: np.ndarray, size: int = 256) -> tuple[int, int, int, int]:
    """预定规则选最大噪声波动窗口；这不是最差预测/GT 错误区域。"""
    height, width = std.shape
    crop_height, crop_width = min(size, height), min(size, width)
    integral = cv2.integral(std, sdepth=cv2.CV_64F)
    scores = (integral[crop_height:, crop_width:] - integral[:-crop_height, crop_width:]
              - integral[crop_height:, :-crop_width] + integral[:-crop_height, :-crop_width])
    y, x = np.unravel_index(int(scores.argmax()), scores.shape)
    return int(x), int(y), int(x + crop_width), int(y + crop_height)


def original_picture(input_dir: Path, source: str, shape: tuple[int, int]) -> Image.Image:
    paths = [path for path in input_dir.glob(source + ".*")
             if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")]
    if len(paths) != 1:
        raise ValueError(f"Expected one input image for {source}; found {len(paths)}")
    with Image.open(paths[0]) as image:
        image = image.convert("RGB")
        if image.size != (shape[1], shape[0]):
            raise ValueError(f"Original size differs from maps: {source}")
        return image.copy()


def save_gallery(root: Path, input_dir: Path, source: str,
                 cases: dict[str, tuple[dict[str, np.ndarray], dict[str, int]]],
                 comparisons: dict[str, dict], std_maps: dict[str, np.ndarray]) -> list[str]:
    gallery = root / "gallery"
    gallery.mkdir(exist_ok=True)
    control = cases["control"]
    original = original_picture(input_dir, source, control[0]["instances"].shape)
    selected = []
    for series in ("local", "global", "both"):
        seed_cases = [case for case in comparisons if case.startswith(series + "_s")]
        if seed_cases:
            selected.append(max(seed_cases, key=lambda c: (comparisons[c]["phase"]["changed_fraction"] or 0,
                                                          comparisons[c]["boundary"]["mae"], c)))
        selected.extend(case for case in (series + "_zero", series + "_pair") if case in comparisons)
    selected = list(dict.fromkeys(selected))
    if "local_mean" in comparisons:
        selected.append("local_mean")
    paths = []
    for case in selected:
        candidate = cases[case]
        name = f"{source}_{case}.jpg"
        make_panel([("Input", original), ("control: F blue / P gold", phase_picture(*control)),
                    (case, phase_picture(*candidate))], gallery / name)
        paths.append(name)
    for series, std in std_maps.items():
        # global 实验固定边界，零方差图没有“最大波动区域”；不可画左上角冒充诊断。
        if float(std.max()) <= 0:
            continue
        box = instability_roi(std)
        scale = float(np.quantile(std, .995))
        heat = np.clip(std / max(scale, 1e-8) * 255, 0, 255).astype(np.uint8)
        heat = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB))
        marked = original.copy()
        ImageDraw.Draw(marked).rectangle(box, outline="red", width=3)
        seed_cases = [case for case in selected if case.startswith(series + "_s")]
        if not seed_cases:
            continue
        case = seed_cases[0]
        name = f"{source}_{series}_unstable.jpg"
        make_panel([("max noise-variation ROI", marked), (f"boundary std p99.5={scale:.6f}", heat),
                    ("control crop", phase_picture(*control).crop(box)),
                    (case + " crop", phase_picture(*cases[case]).crop(box))], gallery / name, 320)
        paths.append(name)
    return paths


def markdown_report(summary: dict) -> str:
    lines = ["# 恢复器噪声分层实验", "", "这些数值比较同一模型的不同预测，不代表 GT 准确率或官方得分。",
             "类别图只比较铁素体/珠光体，几何变化采用实例 IoU 配对，不把实例编号变化当误差。",
             "种子系列的标准差与 CV 采用总体定义（ddof=0）。CV 是标准差除以均值。", "",
             f"完整性：{'完整' if summary['complete'] else '不完整，见下方失败清单'}。", "",
             "| 图像 | 分组 | 边界MAE | 类别像素变化 | 配对IoU | F数量变化 | F均面积变化 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(value, percent=False):
        if value is None:
            return "N/A"
        return f"{value * 100:+.4f}%" if percent else f"{value:.6f}"
    for source, record in summary["sources"].items():
        for case, row in record.get("comparisons", {}).items():
            lines.append(f"| {source} | {case} | {fmt(row['boundary']['mae'])} | "
                         f"{fmt(row['phase']['changed_fraction'], True)} | {fmt(row['instances']['mean_matched_iou'])} | "
                         f"{fmt(row['ferrite_relative_delta']['count'], True)} | "
                         f"{fmt(row['ferrite_relative_delta']['mean_area'], True)} |")
    lines.extend(["", "## 多种子波动", "", "| 图像 | 路径 | 有效/预期种子 | 边界std均值 | F数量CV | F均面积CV | 完整 |",
                  "|---|---|---:|---:|---:|---:|---|"])
    for source, record in summary["sources"].items():
        for name, row in record.get("series", {}).items():
            cv = row.get("ferrite_cv", {})
            lines.append(f"| {source} | {name} | {row['sample_count']}/{len(row['expected_cases'])} | "
                         f"{fmt(row.get('boundary_std', {}).get('mean'))} | {fmt(cv.get('count'), True)} | "
                         f"{fmt(cv.get('mean_area'), True)} | {row['complete']} |")
    quality = summary.get("quality")
    if quality and quality.get("complete"):
        lines.extend(["", "## 清晰配对恢复机制", "",
                      "使用参加过恢复器训练的原图生成清晰/退化配对；属于训练源机制诊断，不是实例GT准确率或泛化成绩。",
                      "每源图先对多个offset求均值，再对源图等权宏平均。MSE的offset标准差是误差标量波动，不是预测像素方差。",
                      "弱/强方向保留率将预测梯度投影到清晰参照的梯度方向；大于1表示沿原边缘方向响应更强，不自动等于恢复更好。", "",
                      "| 退化 | 模式 | RGB MSE | 梯度L1 | 弱梯度比 | 弱方向保留 | 强方向保留 | 新边缘比例 | 平坦区梯度 | MSE offset std |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for condition, modes in quality["groups"].items():
            for mode, row in modes.items():
                values = [fmt(row["macro_mean_metrics"][key]) for key in QUALITY_METRICS]
                values.append(fmt(row.get("rgb_mse_across_offset_std_macro_mean")))
                lines.append(f"| {condition} | {mode} | " + " | ".join(values) + " |")
    if summary["failures"]:
        lines.extend(["", "## 缺失或失败", ""])
        lines.extend(f"- {row['source']} / {row['case']}: {row['error']}" for row in summary["failures"])
    lines.extend(["", "过程缩略图及最大噪声波动区域见 gallery/index.html；窗口仅用于定位波动，不是测试GT。", ""])
    return "\n".join(lines)


def analyze(root: Path, input_dir: Path, sources: list[str] | None = None,
            expected_cases: list[str] | None = None) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Experiment root does not exist: {root}")
    metadata = read_json(root / "record.json") if (root / "record.json").exists() else {}
    expected_cases = expected_cases or metadata.get("expected_cases") or metadata.get("case_order") or list(DEFAULT_CASES)
    if (not isinstance(expected_cases, list) or not all(isinstance(case, str) for case in expected_cases)
            or len(expected_cases) != len(set(expected_cases)) or "control" not in expected_cases):
        raise ValueError("Expected cases must be a unique list containing control")
    expected_cases = [safe_name(case) for case in expected_cases]
    if sources is None:
        sources = metadata.get("sources")
        if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
            sources = sorted(path.name for path in (root / "maps").iterdir() if path.is_dir())
    sources = [safe_name(source) for source in sources]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Sources must be nonempty and unique")
    summary: dict[str, Any] = dict(schema_version=1, complete=True, diagnostic_only=True,
                                    expected_cases=expected_cases, sources={}, failures=[])
    if metadata.get("format") == "restoration_noise_probe_v1" and metadata.get("complete") is not True:
        summary["failures"].append(dict(source="experiment", case="record",
                                         error="Runner has not completed all predeclared predictions and quality checks"))
    gallery_rows = []
    for source in sources:
        cases = {}
        for case in expected_cases:
            try:
                cases[case] = load_case(root, source, case)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                summary["failures"].append(dict(source=source, case=case, error=f"{type(error).__name__}: {error}"))
        record: dict[str, Any] = dict(available_cases=list(cases), comparisons={}, series={})
        summary["sources"][source] = record
        if "control" not in cases:
            continue
        record["control_ferrite"] = ferrite_stats(cases["control"][0]["instances"], cases["control"][1])
        for case, (maps, classes) in cases.items():
            if case == "control":
                continue
            try:
                record["comparisons"][case] = compare_maps(cases["control"][0], maps,
                                                           cases["control"][1], classes)
            except (ValueError, KeyError) as error:
                summary["failures"].append(dict(source=source, case=case, error=f"Comparison: {error}"))
        std_maps = {}
        for name in ("local", "global", "both"):
            expected_seeds = [case for case in expected_cases if case.startswith(name + "_s") and case[len(name) + 2:].isdigit()]
            if expected_seeds:
                valid_cases = {case: item for case, item in cases.items()
                               if case == "control" or case in record["comparisons"]}
                row, std = series_summary(valid_cases, expected_seeds, cases["control"])
                record["series"][name] = row
                if std is not None:
                    std_maps[name] = std
        try:
            gallery_rows.extend(save_gallery(root, input_dir, source, cases, record["comparisons"], std_maps))
        except (OSError, ValueError) as error:
            summary["failures"].append(dict(source=source, case="gallery", error=f"{type(error).__name__}: {error}"))
    quality_path = root / "quality" / "metrics.json"
    if quality_path.exists():
        try:
            summary["quality"] = quality_summary(read_json(quality_path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            summary["quality"] = dict(complete=False)
            summary["failures"].append(dict(source="training originals", case="quality",
                                             error=f"{type(error).__name__}: {error}"))
    summary["complete"] = not summary["failures"]
    summary["record_metadata"] = metadata
    write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(markdown_report(summary), encoding="utf-8")
    gallery = root / "gallery"
    gallery.mkdir(exist_ok=True)
    (gallery / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Noise diagnostics</title>"
        "<style>body{background:#181c22;color:#eee;font:16px sans-serif}img{max-width:100%;display:block;margin:12px 0}</style>"
        "<h1>Prediction stability; no test GT</h1><p>Ferrite: blue. Pearlite: gold. White: instance borders. "
        "Unstable ROI maximizes mean boundary standard deviation, not prediction error.</p>" +
        "".join(f"<h3>{html.escape(name)}</h3><img src='{html.escape(name, quote=True)}'>" for name in gallery_rows), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--cases", nargs="+", help="预注册分组；优先于 record.json expected_cases")
    args = parser.parse_args()
    summary = analyze(args.root, args.input_dir, args.sources, args.cases)
    print(json.dumps(dict(complete=summary["complete"], sources=list(summary["sources"]),
                          failures=len(summary["failures"]), report=str(args.root / "summary.md")), ensure_ascii=False))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
