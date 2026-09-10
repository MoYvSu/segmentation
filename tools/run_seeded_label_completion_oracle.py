# -*- coding: utf-8 -*-
"""评估 LabelMe 窄接缝的 seeded-watershed 派生 GT。

该工具不改写 ``data/raw`` 或 ``data/purified_gt``。原多边形作为不可修改
marker，只在距人工标注有限距离的未覆盖带中生长；随后用同一固定验证集
做一次 μSAM 三通道一致性 Oracle。后者只能验证表示能否复现派生分区，
不能证明 watershed 边界就是物理真值。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

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
from utils.seeded_label_completion import (
    complete_narrow_instance_gaps,
    lab_gradient_elevation,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_radii(text: str) -> List[float]:
    radii = [float(value.strip()) for value in str(text).split(",") if value.strip()]
    if not radii or any(value <= 0 for value in radii):
        raise ValueError("radii must contain positive comma-separated values")
    return list(dict.fromkeys(radii))


def _direct_split(
    names: Iterable[str],
    val_fraction: float,
    seed: int,
    forced_train_names: Iterable[str],
) -> Tuple[List[str], List[str]]:
    names = sorted({str(name) for name in names})
    forced = {Path(name).stem for name in forced_train_names}
    candidates = [name for name in names if name not in forced]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(candidates)
    val_count = max(1, int(round(len(names) * float(val_fraction))))
    val_names = sorted(candidates[:val_count])
    train_names = sorted(name for name in names if name not in set(val_names))
    return train_names, val_names


def _stable_rng(seed: int, token: str):
    value = zlib.crc32(f"{seed}:{token}".encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(value)


def _majority_class_map(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    ground_truth_classes: Mapping[int, int],
) -> Dict[int, int]:
    semantic = np.full(ground_truth.shape, -1, dtype=np.int8)
    for instance_id, class_id in ground_truth_classes.items():
        semantic[ground_truth == int(instance_id)] = int(class_id)
    result: Dict[int, int] = {}
    for predicted_id in np.unique(predicted):
        predicted_id = int(predicted_id)
        if predicted_id == 0:
            continue
        values = semantic[predicted == predicted_id]
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
        "ferrite_mean_area_gt",
        "ferrite_mean_area_pred",
        "ferrite_area_relative_error",
        "score_total",
    )
    return {key: metrics[key] for key in keys}


def _colorize(labels: np.ndarray) -> np.ndarray:
    values = labels.astype(np.int64, copy=False)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    positive = values > 0
    output[..., 0][positive] = ((37 * values[positive] + 53) % 255).astype(np.uint8)
    output[..., 1][positive] = ((97 * values[positive] + 29) % 255).astype(np.uint8)
    output[..., 2][positive] = ((17 * values[positive] + 193) % 255).astype(np.uint8)
    return output


def _resize_panel(image: np.ndarray, width: int = 340) -> np.ndarray:
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _preview_panels(
    image: np.ndarray,
    original: np.ndarray,
    completed: np.ndarray,
    fill_mask: np.ndarray,
    musam_prediction: np.ndarray,
) -> List[np.ndarray]:
    gt_color = _colorize(original)
    original_overlay = cv2.addWeighted(image, 0.48, gt_color, 0.52, 0.0)
    original_overlay[original == 0] = (
        0.35 * image[original == 0] + 0.65 * np.array([30, 30, 230])
    ).astype(np.uint8)

    fill_overlay = image.copy()
    fill_overlay[original == 0] = (
        0.55 * image[original == 0] + 0.45 * np.array([20, 20, 220])
    ).astype(np.uint8)
    fill_overlay[fill_mask] = (
        0.30 * image[fill_mask] + 0.70 * np.array([230, 220, 20])
    ).astype(np.uint8)

    boundary_overlay = image.copy()
    boundaries = find_boundaries(completed, mode="inner", connectivity=1)
    touching_fill = boundaries & cv2.dilate(
        fill_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
    ).astype(bool)
    boundary_overlay[boundaries] = (50, 210, 50)
    boundary_overlay[touching_fill] = (230, 20, 230)

    musam_color = _colorize(musam_prediction)
    musam_overlay = cv2.addWeighted(image, 0.48, musam_color, 0.52, 0.0)
    musam_overlay[musam_prediction == 0] = (
        0.45 * image[musam_prediction == 0]
    ).astype(np.uint8)
    return [
        _resize_panel(value)
        for value in (image, original_overlay, fill_overlay, boundary_overlay, musam_overlay)
    ]


def _render_preview(records: List[Dict], output_path: Path) -> None:
    if not records:
        return
    columns = ["image", "original GT / unknown red", "filled cyan", "new seams magenta", "muSAM"]
    panel_width = records[0]["panels"][0].shape[1]
    panel_height = records[0]["panels"][0].shape[0]
    header_height, caption_height = 42, 34
    canvas = np.full(
        (
            header_height + len(records) * (panel_height + caption_height),
            panel_width * len(columns),
            3,
        ),
        245,
        dtype=np.uint8,
    )
    for index, title in enumerate(columns):
        cv2.putText(
            canvas,
            title,
            (index * panel_width + 8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    for row, record in enumerate(records):
        y0 = header_height + row * (panel_height + caption_height)
        for column, panel in enumerate(record["panels"]):
            x0 = column * panel_width
            canvas[y0:y0 + panel_height, x0:x0 + panel_width] = panel
        caption = (
            f"{record['stem']}  gap={100.0 * record['gap_fraction']:.1f}%  "
            f"filled={100.0 * record['filled_fraction']:.1f}% of gap"
        )
        cv2.putText(
            canvas,
            caption,
            (8, y0 + panel_height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), canvas)


def _summarize_completion(rows: List[Mapping]) -> Dict:
    total = sum(int(row["total_pixels"]) for row in rows)
    original_covered = sum(int(row["original_covered_pixels"]) for row in rows)
    original_gap = sum(int(row["original_uncovered_pixels"]) for row in rows)
    filled = sum(int(row["filled_pixels"]) for row in rows)
    residual = sum(int(row["residual_unknown_pixels"]) for row in rows)
    boundary_pixels = sum(int(row["new_boundary_pixels_touching_fill"]) for row in rows)
    weighted_boundary_gradient = sum(
        float(row["new_boundary_mean_gradient"])
        * int(row["new_boundary_pixels_touching_fill"])
        for row in rows
    ) / max(1, boundary_pixels)
    return {
        "num_images": len(rows),
        "total_pixels": total,
        "original_coverage": original_covered / total,
        "completed_coverage": (original_covered + filled) / total,
        "original_uncovered_pixels": original_gap,
        "filled_pixels": filled,
        "residual_unknown_pixels": residual,
        "fraction_of_gap_filled": filled / max(1, original_gap),
        "original_component_excess": sum(
            int(row["original_component_excess"]) for row in rows
        ),
        "completed_component_excess": sum(
            int(row["completed_component_excess"]) for row in rows
        ),
        "changed_original_pixels": sum(
            int(row["changed_original_pixels"]) for row in rows
        ),
        "new_boundary_mean_gradient": weighted_boundary_gradient,
    }


def _load_original_musam_summary(path: Path):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summaries", {}).get("exact_default")


def _write_report(path: Path, report: Mapping) -> None:
    lines = [
        "# LabelMe 窄接缝补全 Oracle",
        "",
        "原 LabelMe 像素未被修改；是否作为正式训练 GT 由人工目检和实验配置决定。",
        "μSAM 分数只衡量三通道表示对派生分区的自一致性，不能证明物理边界正确。",
        "",
        "## 补全统计",
        "",
        "| 半径 | 原覆盖率 | 补后覆盖率 | 填入原缺口 | 拓扑碎片 excess 前→后 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for radius, values in report["completion_summaries"].items():
        lines.append(
            f"| {radius}px | {100 * values['original_coverage']:.2f}% | "
            f"{100 * values['completed_coverage']:.2f}% | "
            f"{100 * values['fraction_of_gap_filled']:.2f}% | "
            f"{values['original_component_excess']}→{values['completed_component_excess']} |"
        )
    lines.extend([
        "",
        "## μSAM 派生 GT 一致性（固定 6 图，512 grid）",
        "",
        "| 半径/条件 | pred/GT | matches | valid mIoU | symmetric mIoU | 面积项 | proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for radius, conditions in report["musam_summaries"].items():
        for name, values in conditions.items():
            area_term = max(0.0, 1.0 - float(values["ferrite_area_relative_error"]))
            lines.append(
                f"| {radius}px / {name} | {values['pred_count']}/{values['gt_count']} | "
                f"{values['valid_matches']} | {values['instance_miou_valid']:.5f} | "
                f"{values['symmetric_penalized_miou']:.5f} | {area_term:.5f} | "
                f"{values['score_total']:.3f} |"
            )
    lines.extend([
        "",
        "## 判定",
        "",
        f"- μSAM：`{report['decision']['musam']}`。",
        f"- 派生 GT：`{report['decision']['derived_gt']}`；必须以拼图目检和后续完整部署 A/B 为准。",
        "- 本轮人工目检通过的 manual_target_v2：原覆盖区与补区等权，剩余=ignore；provenance 仅用于审计。",
        "",
        f"目检拼图：`{report['visualization']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/direct_ssl_semantic_affinity.yaml")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="outputs/experiments/seeded_label_completion_o1")
    parser.add_argument("--radii", default="8,16")
    parser.add_argument("--blur-sigma", type=float, default=1.2)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--output-grid", type=int, default=512)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument(
        "--original-musam-summary",
        default="outputs/experiments/musam_gt_oracle_v1/musam_gt_oracle_summary.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else Path(
        project_path(config, config["paths"]["raw_data_dir"])
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    radii = _parse_radii(args.radii)

    images = {
        path.stem: path
        for path in sorted(data_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and path.with_suffix(".json").is_file()
    }
    if not images:
        raise FileNotFoundError(f"no LabelMe image/JSON pairs in {data_dir}")
    direct = config["direct_semantic_affinity"]
    train_names, val_names = _direct_split(
        images,
        float(direct.get("val_fraction", 0.20)),
        int(direct.get("seed", 42)),
        direct.get("forced_train_names", ()),
    )

    completion_rows = defaultdict(list)
    musam_rows = {str(radius): defaultdict(list) for radius in radii}
    details = {str(radius): [] for radius in radii}
    preview_by_stem: Dict[str, Dict] = {}
    main_radius = radii[0]

    for index, (stem, image_path) in enumerate(images.items(), start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        original, class_map, load_audit = load_labelme_instances(
            image_path.with_suffix(".json"), image.shape[:2]
        )
        elevation = lab_gradient_elevation(image, blur_sigma=float(args.blur_sigma))
        gap_fraction = float(np.mean(original == 0))
        for radius in radii:
            key = str(radius)
            completed, fill_mask, _, audit = complete_narrow_instance_gaps(
                image,
                original,
                max_fill_distance=radius,
                blur_sigma=float(args.blur_sigma),
                elevation=elevation,
            )
            completion_rows[key].append(audit)
            record = {
                "image": image_path.name,
                "load_audit": load_audit,
                "completion": audit,
            }
            if args.save_maps:
                target_dir = output_dir / f"radius_{radius:g}" / "derived_maps"
                target_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target_dir / f"{stem}_gt.npz",
                    instance_map=completed.astype(np.uint16),
                    original_covered=(original > 0).astype(np.uint8),
                    filled=fill_mask.astype(np.uint8),
                    residual_unknown=(completed == 0).astype(np.uint8),
                )
                (target_dir / f"{stem}_class.json").write_text(
                    json.dumps(
                        {str(instance_id): int(value) for instance_id, value in class_map.items()},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            if stem in val_names:
                grid, valid, metadata = letterbox_instance_geometry(
                    completed,
                    input_size=int(args.input_size),
                    output_grid=int(args.output_grid),
                )
                targets, _, target_audit = build_musam_targets(grid, valid)
                exact_prediction = None
                for condition, noise_std in (("exact", 0.0), ("noise", args.noise_std)):
                    predictions = perturb_musam_predictions(
                        targets,
                        float(noise_std),
                        _stable_rng(args.seed, f"{radius}:{condition}:{stem}"),
                    )
                    pred_grid = musam_watershed(predictions)
                    pred_full = inverse_letterbox_instances(pred_grid, metadata)
                    pred_classes = _majority_class_map(pred_full, completed, class_map)
                    present = {int(value) for value in np.unique(pred_full) if int(value) > 0}
                    missing_classes = sorted(present - set(pred_classes))
                    if missing_classes:
                        raise RuntimeError(
                            f"{stem}/{radius}/{condition} basins lack GT class: {missing_classes}"
                        )
                    metrics = evaluate_instance_pair(
                        completed, class_map, pred_full, pred_classes
                    )
                    musam_rows[key][condition].append(metrics)
                    record[f"musam_{condition}"] = _compact_metrics(metrics)
                    if condition == "exact":
                        exact_prediction = pred_full
                record["musam_target_audit"] = target_audit
                if radius == main_radius and exact_prediction is not None:
                    preview_by_stem[stem] = {
                        "stem": stem,
                        "gap_fraction": gap_fraction,
                        "filled_fraction": float(audit["fraction_of_gap_filled"]),
                        "panels": _preview_panels(
                            image, original, completed, fill_mask, exact_prediction
                        ),
                    }
            details[key].append(record)
        print(
            f"[{index:02d}/{len(images)}] {stem}: gap={100 * gap_fraction:.2f}% "
            + " ".join(
                f"r{radius:g}={100 * completion_rows[str(radius)][-1]['fraction_of_gap_filled']:.1f}%"
                for radius in radii
            )
        )

    completion_summaries = {
        key: _summarize_completion(rows) for key, rows in completion_rows.items()
    }
    musam_summaries = {
        key: {
            condition: summarize_instance_results(rows)
            for condition, rows in conditions.items()
        }
        for key, conditions in musam_rows.items()
    }
    ordered_preview = sorted(
        preview_by_stem.values(), key=lambda row: row["gap_fraction"]
    )
    if len(ordered_preview) > 3:
        ordered_preview = [
            ordered_preview[0],
            ordered_preview[len(ordered_preview) // 2],
            ordered_preview[-1],
        ]
    visualization = output_dir / "seeded_completion_visual_audit.png"
    _render_preview(ordered_preview, visualization)

    if args.save_maps:
        for radius in radii:
            target_dir = output_dir / f"radius_{radius:g}" / "derived_maps"
            manifest = {
                "format": "manual_target_v2",
                "version": 1,
                "source": "LabelMe instance seeds + narrow Lab-gradient watershed",
                "data_dir": (
                    Path(args.data_dir)
                    if args.data_dir
                    else Path(config["paths"]["raw_data_dir"])
                ).as_posix(),
                "max_fill_distance_native": float(radius),
                "blur_sigma": float(args.blur_sigma),
                "sample_count": len(images),
                "samples": sorted(images),
                "provenance": {
                    "original_covered": "immutable LabelMe rasterization",
                    "filled": "watershed-assigned gap within max_fill_distance_native",
                    "residual_unknown": "must remain ignored",
                },
                "summary": completion_summaries[str(radius)],
            }
            (target_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    first_exact = musam_summaries[str(main_radius)]["exact"]
    count_error = abs(int(first_exact["pred_count"]) - int(first_exact["gt_count"])) / max(
        1, int(first_exact["gt_count"])
    )
    representation_go = (
        count_error <= 0.02
        and float(first_exact["symmetric_penalized_miou"]) >= 0.95
    )
    report = {
        "experiment": "LabelMe seeded-watershed narrow-gap completion Oracle",
        "scope": "derived targets only; original LabelMe and purified GT unchanged",
        "data_dir": str(data_dir),
        "config": str(Path(args.config).resolve()),
        "split": {"train": train_names, "val": val_names, "seed": int(direct.get("seed", 42))},
        "settings": vars(args),
        "completion_summaries": completion_summaries,
        "musam_summaries": musam_summaries,
        "original_musam_exact": _load_original_musam_summary(
            Path(args.original_musam_summary).resolve()
        ),
        "details": details,
        "visualization": str(visualization),
        "decision": {
            "musam": (
                "conditional_go_pending_physical_review"
                if representation_go
                else "no_go_at_512_even_after_narrow_gap_completion"
            ),
            "derived_gt": "visual_review_and_training_ab_required",
            "criteria": {"count_relative_error_max": 0.02, "symmetric_miou_min": 0.95},
        },
        "limitations": [
            "The completion is constrained by manual seeds but its new interfaces are not human GT.",
            "Weak-gradient regions can reduce watershed to distance/order-driven partitions.",
            "muSAM evaluation against this derived partition is partly constructive and not an accuracy proof.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = output_dir / "REPORT.md"
    _write_report(report_path, report)

    print("\n=== completion ===")
    for radius, values in completion_summaries.items():
        print(
            f"r={radius}: coverage {100 * values['original_coverage']:.2f}% -> "
            f"{100 * values['completed_coverage']:.2f}% "
            f"filled={100 * values['fraction_of_gap_filled']:.2f}% of gap "
            f"components={values['original_component_excess']}->{values['completed_component_excess']}"
        )
    print("=== muSAM derived-GT consistency ===")
    for radius, conditions in musam_summaries.items():
        for name, values in conditions.items():
            print(
                f"r={radius}/{name}: pred={values['pred_count']}/{values['gt_count']} "
                f"match={values['valid_matches']} miou={values['instance_miou_valid']:.5f} "
                f"sym={values['symmetric_penalized_miou']:.5f} score={values['score_total']:.3f}"
            )
    print(f"decision={report['decision']['musam']}")
    print(f"report={report_path}")
    print(f"visualization={visualization}")


if __name__ == "__main__":
    main()
