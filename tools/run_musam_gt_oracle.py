# -*- coding: utf-8 -*-
"""在固定验证集上运行 μSAM AIS 三通道 GT Oracle。

该实验只验证表示和后处理上限，不加载模型、不启动训练。实例重建只能读取
foreground、center-distance、boundary-distance 三张图；GT instance id/count
仅用于生成三通道真值及事后评分。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np
from skimage.measure import label as measure_label, regionprops

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import load_config, project_path
from utils.instance_metrics import (
    evaluate_instance_pair,
    load_labelme_instances,
    summarize_instance_results,
)
from utils.musam_instances import (
    build_musam_targets,
    musam_watershed,
    perturb_musam_predictions,
)
from utils.offset_letterbox import (
    inverse_letterbox_instances,
    letterbox_instance_geometry,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _stable_rng(seed: int, condition: str, image_name: str):
    token = f"{seed}:{condition}:{image_name}".encode("utf-8")
    return np.random.default_rng(zlib.crc32(token) & 0xFFFFFFFF)


def _deterministic_direct_split(
    names: Iterable[str],
    val_fraction: float,
    seed: int,
    forced_train_names: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """复刻当前 direct 双头训练使用的六图固定划分。"""
    names = sorted({str(name) for name in names})
    forced = {Path(name).stem for name in forced_train_names}
    candidates = [name for name in names if name not in forced]
    generator = np.random.default_rng(int(seed))
    generator.shuffle(candidates)
    val_count = max(1, int(round(len(names) * float(val_fraction))))
    val_names = sorted(candidates[:val_count])
    train_names = sorted(name for name in names if name not in set(val_names))
    if not train_names or not val_names:
        raise ValueError("fixed split requires non-empty train and val")
    return train_names, val_names


def _resolve_samples(
    data_dir: Path,
    config_path: str,
    split_json: str,
    subset: str,
):
    by_stem = {
        path.stem: path
        for path in sorted(data_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and path.with_suffix(".json").is_file()
    }
    if not by_stem:
        raise FileNotFoundError(f"no labeled images found in {data_dir}")

    if split_json:
        split_path = Path(split_json).resolve()
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split_source = str(split_path)
    else:
        config = load_config(config_path)
        direct = config["direct_semantic_affinity"]
        train_names, val_names = _deterministic_direct_split(
            by_stem,
            float(direct.get("val_fraction", 0.20)),
            int(direct.get("seed", 42)),
            direct.get("forced_train_names", ()),
        )
        split = {
            "train": train_names,
            "val": val_names,
            "seed": int(direct.get("seed", 42)),
        }
        split_source = f"derived from {Path(config_path).resolve()}"

    if subset == "all":
        selected_names = sorted(by_stem)
    else:
        selected_names = [Path(name).stem for name in split[subset]]
    missing = sorted(set(selected_names) - set(by_stem))
    if missing:
        raise FileNotFoundError(f"split references missing samples: {missing}")
    return [by_stem[name] for name in selected_names], split, split_source


def _majority_class_map(
    predicted_instances: np.ndarray,
    gt_instances: np.ndarray,
    gt_class_map: Mapping[int, int],
) -> Dict[int, int]:
    """为几何 Oracle 使用完美 GT 语义投票；不参与实例重建。"""
    semantic = np.full(gt_instances.shape, -1, dtype=np.int8)
    for instance_id, class_id in gt_class_map.items():
        semantic[gt_instances == int(instance_id)] = int(class_id)
    result: Dict[int, int] = {}
    for predicted_id in np.unique(predicted_instances):
        predicted_id = int(predicted_id)
        if predicted_id == 0:
            continue
        values = semantic[predicted_instances == predicted_id]
        values = values[values >= 0]
        if values.size:
            result[predicted_id] = int(np.argmax(np.bincount(values, minlength=2)))
    return result


def _compact_metrics(metrics: Mapping) -> Dict:
    keys = (
        "gt_count",
        "pred_count",
        "valid_matches",
        "instance_miou_valid",
        "gt_penalized_miou",
        "symmetric_penalized_miou",
        "ferrite_gt_count",
        "ferrite_pred_count",
        "ferrite_mean_area_gt",
        "ferrite_mean_area_pred",
        "ferrite_area_relative_error",
        "score_total",
        "classes",
    )
    return {key: metrics[key] for key in keys}


def _condition_specs(
    center: float,
    boundary: float,
    delta: float,
    noise: float,
    base_only: bool,
):
    conditions = [
        {
            "name": "exact_default",
            "center_threshold": center,
            "boundary_threshold": boundary,
            "noise_std": 0.0,
        },
        {
            "name": "exact_center_low",
            "center_threshold": center - delta,
            "boundary_threshold": boundary,
            "noise_std": 0.0,
        },
        {
            "name": "exact_center_high",
            "center_threshold": center + delta,
            "boundary_threshold": boundary,
            "noise_std": 0.0,
        },
        {
            "name": "exact_boundary_low",
            "center_threshold": center,
            "boundary_threshold": boundary - delta,
            "noise_std": 0.0,
        },
        {
            "name": "exact_boundary_high",
            "center_threshold": center,
            "boundary_threshold": boundary + delta,
            "noise_std": 0.0,
        },
        {
            "name": "noise_default",
            "center_threshold": center,
            "boundary_threshold": boundary,
            "noise_std": noise,
        },
    ]
    return conditions[:1] if base_only else conditions


def _marker_topology_audit(markers: np.ndarray, gt_instances: np.ndarray):
    """仅在重建完成后用 GT id 归因 marker 失败，不参与 watershed。"""
    zero_marker_gt = 0
    multi_marker_gt = 0
    extra_markers_in_gt = 0
    disconnected_gt = 0
    present_ids = [int(value) for value in np.unique(gt_instances) if int(value) > 0]
    for instance_id in present_ids:
        mask = gt_instances == instance_id
        disconnected_gt += int(
            int(measure_label(mask, connectivity=None).max()) > 1
        )
        marker_ids = np.unique(markers[mask])
        marker_count = int(np.sum(marker_ids > 0))
        zero_marker_gt += int(marker_count == 0)
        multi_marker_gt += int(marker_count > 1)
        extra_markers_in_gt += max(0, marker_count - 1)

    markers_spanning_gt = 0
    for marker_id in range(1, int(markers.max()) + 1):
        gt_ids = np.unique(gt_instances[markers == marker_id])
        markers_spanning_gt += int(np.sum(gt_ids > 0) > 1)
    return {
        "present_gt_instances": len(present_ids),
        "zero_marker_gt": zero_marker_gt,
        "multi_marker_gt": multi_marker_gt,
        "extra_markers_in_gt": extra_markers_in_gt,
        "disconnected_gt": disconnected_gt,
        "markers_spanning_multiple_gt": markers_spanning_gt,
    }


def _colorize(instance_map: np.ndarray) -> np.ndarray:
    output = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(instance_map):
        value = int(instance_id)
        if value == 0:
            continue
        output[instance_map == value] = (
            (37 * value + 53) % 255,
            (97 * value + 29) % 255,
            (17 * value + 193) % 255,
        )
    return output


def _grid_image(image: np.ndarray, content_height: int, content_width: int, grid: int):
    output = np.zeros((grid, grid, 3), dtype=np.uint8)
    output[:content_height, :content_width] = cv2.resize(
        image, (content_width, content_height), interpolation=cv2.INTER_AREA
    )
    return output


def _adjacent_pair(labels: np.ndarray):
    counts: Counter[Tuple[int, int]] = Counter()
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        mask = (first > 0) & (second > 0) & (first != second)
        left = first[mask].astype(np.int64)
        right = second[mask].astype(np.int64)
        for a, b in zip(np.minimum(left, right), np.maximum(left, right)):
            counts[(int(a), int(b))] += 1
    return counts.most_common(1)[0] if counts else None


def _bbox_for_ids(labels: np.ndarray, ids: Iterable[int]):
    mask = np.isin(labels, np.asarray(list(ids), dtype=np.int32))
    yy, xx = np.nonzero(mask)
    if not yy.size:
        return None
    return int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1


def _update_visual_candidates(candidates: Dict, stem: str, labels: np.ndarray):
    for prop in regionprops(labels):
        if int(prop.area) < 16:
            continue
        nonconvex = 1.0 - float(prop.solidity)
        slender = float(prop.eccentricity)
        for kind, score in (("nonconvex", nonconvex), ("slender", slender)):
            if kind not in candidates or score > candidates[kind]["score"]:
                candidates[kind] = {
                    "kind": kind,
                    "stem": stem,
                    "ids": [int(prop.label)],
                    "bbox": list(map(int, prop.bbox)),
                    "score": score,
                }
    pair = _adjacent_pair(labels)
    if pair is not None:
        (first, second), contact = pair
        if "touching" not in candidates or contact > candidates["touching"]["score"]:
            candidates["touching"] = {
                "kind": "touching",
                "stem": stem,
                "ids": [first, second],
                "bbox": list(_bbox_for_ids(labels, (first, second))),
                "score": int(contact),
            }


def _expanded_crop(box, shape, margin_fraction=0.20):
    y0, x0, y1, x1 = map(int, box)
    margin = max(12, int(round(max(y1 - y0, x1 - x0) * margin_fraction)))
    return (
        max(0, y0 - margin),
        max(0, x0 - margin),
        min(shape[0], y1 + margin),
        min(shape[1], x1 + margin),
    )


def _square_preview(image: np.ndarray, box, size=224):
    y0, x0, y1, x1 = box
    crop = image[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    side = max(height, width, 1)
    if crop.ndim == 2:
        canvas = np.zeros((side, side), dtype=crop.dtype)
    else:
        canvas = np.zeros((side, side, crop.shape[2]), dtype=crop.dtype)
    off_y, off_x = (side - height) // 2, (side - width) // 2
    canvas[off_y:off_y + height, off_x:off_x + width] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def _distance_color(distance: np.ndarray, foreground: np.ndarray):
    values = np.where(foreground, distance, 0.0)
    return cv2.applyColorMap(
        np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )


def _render_visual_cases(candidates: Dict, visual_data: Dict, output_path: Path):
    ordered = [candidates[kind] for kind in ("nonconvex", "slender", "touching") if kind in candidates]
    if not ordered:
        return []
    columns = ["image", "GT", "center", "boundary", "markers", "AIS"]
    cell, header, caption = 224, 36, 34
    canvas = np.full(
        (header + len(ordered) * (cell + caption), len(columns) * cell, 3),
        245,
        dtype=np.uint8,
    )
    for column, title in enumerate(columns):
        cv2.putText(
            canvas, title, (column * cell + 8, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (20, 20, 20), 1, cv2.LINE_AA,
        )

    records = []
    for row, case in enumerate(ordered):
        data = visual_data[case["stem"]]
        box = _expanded_crop(case["bbox"], data["gt"].shape)
        foreground = data["targets"][0] > 0.5
        panels = [
            data["image"],
            _colorize(data["gt"]),
            _distance_color(data["debug"]["center_distances"], foreground),
            _distance_color(data["debug"]["boundary_distances"], foreground),
            _colorize(data["debug"]["markers"]),
            _colorize(data["pred"]),
        ]
        y = header + row * (cell + caption)
        for column, panel in enumerate(panels):
            preview = _square_preview(panel, box, size=cell)
            canvas[y:y + cell, column * cell:(column + 1) * cell] = preview
        label_text = (
            f"{case['kind']} | {case['stem']} | ids={case['ids']} | "
            f"score={case['score']:.3f}"
        )
        cv2.putText(
            canvas, label_text, (8, y + cell + 23), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (20, 20, 20), 1, cv2.LINE_AA,
        )
        records.append({**case, "crop": list(box)})
    cv2.imwrite(str(output_path), canvas)
    return records


def _augment_summary(summary: Dict):
    gt_count = int(summary["gt_count"])
    pred_count = int(summary["pred_count"])
    matches = int(summary["valid_matches"])
    summary["valid_match_recall"] = matches / gt_count if gt_count else 0.0
    summary["valid_match_precision"] = matches / pred_count if pred_count else 0.0
    summary["count_relative_error"] = (
        abs(pred_count - gt_count) / gt_count if gt_count else 0.0
    )
    summary["ferrite_area_term"] = max(
        0.0, 1.0 - float(summary["ferrite_area_relative_error"])
    )
    return summary


def _go_no_go(summaries: Mapping[str, Mapping]):
    exact = summaries["exact_default"]
    if "noise_default" not in summaries:
        return {
            "decision": "RESOLUTION_CONTROL_ONLY",
            "exact_near_lossless": (
                float(exact["symmetric_penalized_miou"]) >= 0.95
                and float(exact["valid_match_recall"]) >= 0.98
                and float(exact["count_relative_error"]) <= 0.02
                and float(exact["ferrite_area_term"]) >= 0.95
            ),
            "criteria": {
                "exact": "symmetric_mIoU>=0.95, recall>=0.98, count_error<=0.02, area_term>=0.95"
            },
        }
    noisy = summaries["noise_default"]
    threshold_rows = [
        row for name, row in summaries.items() if name.startswith("exact_")
    ]
    exact_ok = (
        float(exact["symmetric_penalized_miou"]) >= 0.95
        and float(exact["valid_match_recall"]) >= 0.98
        and float(exact["count_relative_error"]) <= 0.02
        and float(exact["ferrite_area_term"]) >= 0.95
    )
    noise_ok = (
        float(noisy["symmetric_penalized_miou"]) >= 0.90
        and float(noisy["valid_match_recall"]) >= 0.90
        and float(noisy["count_relative_error"]) <= 0.10
        and float(noisy["ferrite_area_term"]) >= 0.90
    )
    sensitivity_ok = min(
        float(row["symmetric_penalized_miou"]) for row in threshold_rows
    ) >= 0.90
    return {
        "decision": "GO" if exact_ok and noise_ok and sensitivity_ok else "NO_GO",
        "exact_near_lossless": bool(exact_ok),
        "light_noise_robust": bool(noise_ok),
        "small_threshold_change_robust": bool(sensitivity_ok),
        "criteria": {
            "exact": "symmetric_mIoU>=0.95, recall>=0.98, count_error<=0.02, area_term>=0.95",
            "noise": "symmetric_mIoU>=0.90, recall>=0.90, count_error<=0.10, area_term>=0.90",
            "threshold": "all one-factor +/-delta exact arms symmetric_mIoU>=0.90",
        },
    }


def _write_markdown_report(path: Path, report: Mapping):
    lines = [
        "# μSAM AIS GT Oracle",
        "",
        f"结论：**{report['go_no_go']['decision']}**（只决定是否值得训练独立 geometry decoder）。",
        "",
        "| condition | pred / GT | matches | valid mIoU | symmetric mIoU | ferrite area term | proxy score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["summaries"].items():
        lines.append(
            f"| {name} | {row['pred_count']} / {row['gt_count']} | "
            f"{row['valid_matches']} | {row['instance_miou_valid']:.5f} | "
            f"{row['symmetric_penalized_miou']:.5f} | "
            f"{row['ferrite_area_term']:.5f} | {row['score_total']:.3f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 实例重建没有读取 GT instance id 或 GT 实例数；只读取三张连续真值图。",
        "- 类别评分使用 GT semantic majority vote，因此这是几何上界，不是完整部署成绩。",
        "- 未覆盖像素在训练有效性 mask 中保持 ignore；因 watershed 不支持 ignore，重建使用偏有利的完美已标支持域 mask。",
        "- 该实验不能证明模型能从模糊或低照度图像学出距离图，只验证表示与后处理是否值得训练。",
        "",
        f"固定验证图：{', '.join(report['images'])}",
        "",
        f"形态目检图：`{Path(report['visualization']).name}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/direct_ssl_semantic_affinity.yaml")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--split-json", default="")
    parser.add_argument("--subset", choices=("train", "val", "all"), default="val")
    parser.add_argument(
        "--output-dir", default="outputs/experiments/musam_gt_oracle_v1"
    )
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--output-grid", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--threshold-delta", type=float, default=0.1)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--foreground-smoothing", type=float, default=1.0)
    parser.add_argument("--distance-smoothing", type=float, default=1.6)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument(
        "--base-only", action="store_true",
        help="只运行官方默认参数，用于单次分辨率归因，不作鲁棒性判定",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = Path(project_path(config, config["paths"]["raw_data_dir"]))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, split, split_source = _resolve_samples(
        data_dir, args.config, args.split_json, args.subset
    )
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("no samples selected")

    conditions = _condition_specs(
        float(args.center_threshold),
        float(args.boundary_threshold),
        float(args.threshold_delta),
        float(args.noise_std),
        bool(args.base_only),
    )
    results = defaultdict(list)
    details = defaultdict(list)
    visual_data = {}
    visual_candidates: Dict = {}
    prediction_dir = output_dir / "exact_default_gt_semantic_oracle"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for sample_index, image_path in enumerate(samples, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        gt_map, gt_class_map, gt_audit = load_labelme_instances(
            image_path.with_suffix(".json"), image.shape[:2]
        )
        geometry_map, valid_content, metadata = letterbox_instance_geometry(
            gt_map, input_size=args.input_size, output_grid=args.output_grid
        )
        targets, supervision_valid, target_audit = build_musam_targets(
            geometry_map, valid_content
        )
        print(
            f"[{sample_index}/{len(samples)}] {image_path.name}: "
            f"gt={len(gt_class_map)} corrected_centers={target_audit['corrected_centers']} "
            f"covered={target_audit['covered_pixels']} "
            f"uncovered_native={gt_audit['uncovered_pixels']}"
        )

        for condition in conditions:
            name = condition["name"]
            rng = _stable_rng(args.seed, name, image_path.name)
            predictions = perturb_musam_predictions(
                targets, float(condition["noise_std"]), rng
            )
            pred_grid, debug = musam_watershed(
                predictions,
                center_threshold=float(condition["center_threshold"]),
                boundary_threshold=float(condition["boundary_threshold"]),
                foreground_threshold=float(args.foreground_threshold),
                foreground_smoothing=float(args.foreground_smoothing),
                distance_smoothing=float(args.distance_smoothing),
                min_size=int(args.min_size),
                return_debug=True,
            )
            pred_full = inverse_letterbox_instances(pred_grid, metadata)
            pred_class_map = _majority_class_map(pred_full, gt_map, gt_class_map)
            present_ids = {
                int(value) for value in np.unique(pred_full) if int(value) != 0
            }
            missing_classes = sorted(present_ids - set(pred_class_map))
            if missing_classes:
                raise RuntimeError(
                    f"{name}/{image_path.name} has basins without GT semantic "
                    f"support: {missing_classes}"
                )
            if int(pred_full.max()) > 65535:
                raise RuntimeError(
                    f"{name}/{image_path.name} exceeds uint16 instance id limit: "
                    f"{int(pred_full.max())}"
                )
            metrics = evaluate_instance_pair(
                gt_map, gt_class_map, pred_full, pred_class_map
            )
            marker_audit = (
                _marker_topology_audit(debug["markers"], geometry_map)
                if name == "exact_default"
                else None
            )
            results[name].append(metrics)
            details[name].append({
                "image": image_path.name,
                "condition": condition,
                "metadata": metadata.to_dict(),
                "gt_audit": gt_audit,
                "target_audit": target_audit,
                "supervision_valid_pixels": int(supervision_valid.sum()),
                "marker_count": int(debug["marker_count"]),
                "marker_topology_audit": marker_audit,
                "metrics": _compact_metrics(metrics),
            })
            print(
                f"  {name}: markers={debug['marker_count']} "
                f"pred={metrics['pred_count']} match={metrics['valid_matches']} "
                f"miou={metrics['instance_miou_valid']:.4f} "
                f"sym={metrics['symmetric_penalized_miou']:.4f} "
                f"area_err={metrics['ferrite_area_relative_error']:.4f}"
            )

            if name == "exact_default":
                cv2.imwrite(
                    str(prediction_dir / f"{image_path.stem}_inst.png"),
                    pred_full.astype(np.uint16),
                )
                (prediction_dir / f"{image_path.stem}_class.json").write_text(
                    json.dumps(
                        {str(key): int(value) for key, value in pred_class_map.items()},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                visual_data[image_path.stem] = {
                    "image": _grid_image(
                        image,
                        metadata.content_height,
                        metadata.content_width,
                        args.output_grid,
                    ),
                    "gt": geometry_map,
                    "targets": targets,
                    "pred": pred_grid,
                    "debug": debug,
                }
                _update_visual_candidates(
                    visual_candidates, image_path.stem, geometry_map
                )

    summaries = {
        name: _augment_summary(summarize_instance_results(rows))
        for name, rows in results.items()
    }
    visualization_path = output_dir / "morphology_cases.png"
    visual_records = _render_visual_cases(
        visual_candidates, visual_data, visualization_path
    )
    report = {
        "experiment": "μSAM AIS three-channel GT Oracle",
        "scope": "geometry representation and postprocess only; no model training",
        "reference": {
            "method": "Segment Anything for Microscopy, Nature Methods 2025",
            "implementation": "torch-em watershed_from_center_and_boundary_distances",
        },
        "data_dir": str(data_dir),
        "split_source": split_source,
        "split": split,
        "subset": args.subset,
        "images": [path.stem for path in samples],
        "settings": vars(args),
        "conditions": conditions,
        "summaries": summaries,
        "details": details,
        "visualization": str(visualization_path),
        "visual_cases": visual_records,
        "class_evaluation": "GT semantic majority oracle; geometry reconstruction is class-agnostic",
        "uncovered_policy": (
            "instance_map==0 is excluded from supervision; watershed has no ignore "
            "state, so reconstruction uses a favorable perfect annotated-support mask"
        ),
    }
    report["go_no_go"] = _go_no_go(summaries)
    summary_path = output_dir / "musam_gt_oracle_summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path = output_dir / "REPORT.md"
    _write_markdown_report(markdown_path, report)

    print("\n=== dataset summaries ===")
    for name, summary in summaries.items():
        print(
            f"{name}: pred={summary['pred_count']}/{summary['gt_count']} "
            f"matches={summary['valid_matches']} "
            f"miou={summary['instance_miou_valid']:.5f} "
            f"sym={summary['symmetric_penalized_miou']:.5f} "
            f"area_term={summary['ferrite_area_term']:.5f} "
            f"score={summary['score_total']:.3f}"
        )
    print(f"decision: {report['go_no_go']['decision']}")
    print(f"report: {summary_path}")
    print(f"visualization: {visualization_path}")


if __name__ == "__main__":
    main()
