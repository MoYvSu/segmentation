# -*- coding: utf-8 -*-
"""Render fixed-scale affinity diagnostics from completed whole-image caches.

The script never loads a model or recomputes a crop. Grid maps and native-size
postprocessing maps are cropped from the corresponding arrays in cache/*.npz.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ARMS = ("g4b_mean", "e110_mean", "g4b_top2", "e110_top2")
OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (0, 4), (4, 0))
DEFAULT_STEMS = ("train_172", "train_873", "test_009", "test_026", "test_045")


def boundaries(labels: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """One-pixel source-side partition edges; unknown pairs are omitted."""
    labels = np.asarray(labels)
    valid = np.ones(labels.shape, dtype=bool) if valid is None else valid.astype(bool)
    result = np.zeros(labels.shape, dtype=bool)
    result[:-1] |= (labels[:-1] != labels[1:]) & valid[:-1] & valid[1:]
    result[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    return result


def instance_rgb(labels: np.ndarray, rgb: np.ndarray | None = None) -> np.ndarray:
    """Deterministic ID colors, without implying cross-model ID matching."""
    values = labels.astype(np.uint64)
    mixed = values * np.uint64(2654435761)
    colors = np.stack([(mixed >> shift) & 255 for shift in (0, 8, 16)], axis=-1)
    colors = 0.25 + 0.70 * colors.astype(np.float32) / 255.0
    colors[labels == 0] = 0.12
    if rgb is not None:
        colors = 0.64 * colors + 0.36 * rgb.astype(np.float32) / 255.0
    colors[boundaries(labels)] = 0.02
    return colors


def put_image(ax, values, title: str = "", *, gt_edge=None, **kwargs):
    handle = ax.imshow(values, interpolation="nearest", **kwargs)
    if gt_edge is not None and np.any(gt_edge):
        overlay = np.zeros((*gt_edge.shape, 4), dtype=np.float32)
        overlay[gt_edge] = (0.0, 1.0, 1.0, 0.85)
        ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    return handle


def crop(values, roi):
    return values[int(roi["y0"]):int(roi["y1"]), int(roi["x0"]):int(roi["x1"])]


def choose_rois(summary: dict, row: dict) -> list[dict]:
    height, width = map(int, row["grid_content_shape"])
    supplied = row.get("rois")
    if not supplied:
        supplied = [item for item in summary.get("rois", []) if item.get("stem") == row["stem"]]
    if not supplied:
        size_y, size_x = min(128, height), min(128, width)
        y0, x0 = (height - size_y) // 2, (width - size_x) // 2
        supplied = [{"kind": "fixed_center", "x0": x0, "y0": y0,
                     "x1": x0 + size_x, "y1": y0 + size_y}]
    rois = []
    for index, original in enumerate(supplied):
        roi = dict(original)
        roi.update(x0=max(0, int(roi["x0"])), y0=max(0, int(roi["y0"])),
                   x1=min(width, int(roi["x1"])), y1=min(height, int(roi["y1"])))
        if roi["x1"] <= roi["x0"] or roi["y1"] <= roi["y0"]:
            raise ValueError(f"{row['stem']}: empty ROI after clipping: {original}")
        roi.setdefault("kind", "diagnostic_selection")
        roi["index"] = index + 1
        roi["selection"] = ("fixed center, independent of predictions" if roi["kind"] == "fixed_center"
                            else "diagnostic selection from summary.json; not a random sample")
        rois.append(roi)
    return rois


def native_roi(roi: dict, row: dict) -> dict:
    native_height, native_width = map(int, row["native_shape"][:2])
    grid_height, grid_width = map(int, row["grid_content_shape"])
    return {
        "x0": max(0, math.floor(roi["x0"] * native_width / grid_width)),
        "y0": max(0, math.floor(roi["y0"] * native_height / grid_height)),
        "x1": min(native_width, math.ceil(roi["x1"] * native_width / grid_width)),
        "y1": min(native_height, math.ceil(roi["y1"] * native_height / grid_height)),
    }


def roi_label(roi: dict) -> str:
    return (f"ROI {roi['index']}: {roi['kind']} | grid x=[{roi['x0']},{roi['x1']}), "
            f"y=[{roi['y0']},{roi['y1']})")


def finish(fig, output: Path, *, dpi=155):
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_overview(data, row: dict, rois: list[dict], output: Path):
    rgb = data["rgb"]
    has_gt = bool(row["has_gt"])
    gt_edge = boundaries(data["gt"], data["valid"]) if has_gt else None
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), layout="constrained")
    put_image(axes[0, 0], rgb, "Original image")
    if has_gt:
        gt_display = instance_rgb(data["gt"], rgb)
        gt_display[~data["valid"].astype(bool)] *= 0.3
        put_image(axes[0, 1], gt_display, "GT partitions (unknown darkened)")
    else:
        put_image(axes[0, 1], rgb, "No GT: visual diagnosis only")
    for arm, ax in zip(ARMS, (axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2])):
        instances = data[f"{arm}_instances"]
        count = len(np.unique(instances[instances > 0]))
        put_image(ax, instance_rgb(instances, rgb), f"{arm} | {count} instance IDs", gt_edge=gt_edge)
    for roi in rois:
        nr = native_roi(roi, row)
        for ax in axes.flat:
            ax.add_patch(Rectangle((nr["x0"] - 0.5, nr["y0"] - 0.5),
                                   nr["x1"] - nr["x0"], nr["y1"] - nr["y0"],
                                   fill=False, edgecolor="#ff4040", linewidth=1.2))
            ax.text(nr["x0"], nr["y0"], str(roi["index"]), color="white", fontsize=9,
                    bbox={"facecolor": "#aa0000", "edgecolor": "none", "pad": 1})
    fig.suptitle(f"{row['stem']} | Whole-image outputs; crop boxes on native grid\n"
                 "ID colors are not matched between models. Black: predicted partitions. "
                 "Cyan: valid GT partitions.", fontsize=12)
    finish(fig, output)


def render_directions(data, row: dict, roi: dict, output: Path):
    fig, axes = plt.subplots(2, 8, figsize=(21, 6.2), layout="constrained")
    edge_valid = data["edge_valid"]
    edge_target = data["edge_target"]
    for index, model in enumerate(("g4b", "e110")):
        for channel, offset in enumerate(OFFSETS):
            probability = data[f"{model}_prob"][channel].astype(np.float32)
            q = crop(1.0 - probability, roi)
            handle = put_image(axes[index, channel], q, f"(dy, dx) = {offset}",
                               cmap="magma", vmin=0.0, vmax=1.0)
            if row["has_gt"]:
                # Cross-grain starts are shown as sparse contours, not as new labels.
                crossing = crop(edge_valid[channel] & ~edge_target[channel].astype(bool), roi)
                if np.any(crossing) and not np.all(crossing):
                    axes[index, channel].contour(crossing.astype(float), levels=[0.5],
                                                 colors=["cyan"], linewidths=0.35, alpha=0.8)
        axes[index, 0].set_ylabel(model.upper(), fontsize=12)
    fig.colorbar(handle, ax=list(axes.flat), shrink=0.8, pad=0.012,
                 label="q = 1 - affinity (probability of different grain)")
    fig.suptitle(f"{row['stem']} | {roi_label(roi)}\n"
                 "All channels share [0, 1]. Cyan contours: valid GT cross-grain starts. "
                 "Crops from whole-image forward.", fontsize=12)
    finish(fig, output, dpi=165)


def render_pipeline(data, row: dict, roi: dict, output: Path):
    nr = native_roi(roi, row)
    rgb = crop(data["rgb"], nr)
    gt_edge = crop(boundaries(data["gt"], data["valid"]), nr) if row["has_gt"] else None
    fig, axes = plt.subplots(4, 10, figsize=(25, 12), layout="constrained")
    titles = ("Image + valid GT", "Short S", "Distance 2", "Distance 4", "Fused B",
              "B - S", "High boundary", "Marker mask", "Marker IDs", "Final instances")
    for index, arm in enumerate(ARMS):
        put_image(axes[index, 0], rgb, titles[0] if index == 0 else "", gt_edge=gt_edge)
        axes[index, 0].set_ylabel(arm, fontsize=11)
        for column, key in enumerate(("short", "d2", "d4", "boundary"), start=1):
            value = crop(data[f"{arm}_{key}"].astype(np.float32), roi)
            positive_handle = put_image(axes[index, column], value,
                                        titles[column] if index == 0 else "",
                                        cmap="magma", vmin=0.0, vmax=1.0)
        delta = data[f"{arm}_boundary"].astype(np.float32) - data[f"{arm}_short"].astype(np.float32)
        signed_handle = put_image(axes[index, 5], crop(delta, roi),
                                  titles[5] if index == 0 else "",
                                  cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        for column, key in ((6, "high"), (7, "marker_mask")):
            put_image(axes[index, column], crop(data[f"{arm}_{key}"], nr),
                      titles[column] if index == 0 else "", cmap="gray", vmin=0, vmax=1)
        for column, key in ((8, "markers"), (9, "instances")):
            colored = instance_rgb(crop(data[f"{arm}_{key}"], nr), rgb if column == 9 else None)
            put_image(axes[index, column], colored, titles[column] if index == 0 else "",
                      gt_edge=gt_edge if column == 9 else None)
    fig.colorbar(positive_handle, ax=list(axes[:, 1:5].flat), shrink=0.62, pad=0.01,
                 location="bottom", label="Boundary strength [0, 1]")
    fig.colorbar(signed_handle, ax=list(axes[:, 5].flat), shrink=0.85, pad=0.01,
                 location="bottom", label="B - S [-0.5, 0.5]")
    fig.suptitle(f"{row['stem']} | {roi_label(roi)}\n"
                 "S / D2 / D4 / B: decoder grid; image / high / markers / instances: native grid. "
                 "All crops follow whole-image processing.\n"
                 "Blue B-S: long-range fusion lowers short response; red: raises it. "
                 "White binary pixels: True. Cyan: valid GT partitions.", fontsize=11)
    finish(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", help="Exact cache stems; default is two fixed validation and three test images.")
    parser.add_argument("--all-rois", action="store_true", help="Render directions/pipeline for every recorded ROI, not only the first.")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows_by_stem = {row["stem"]: row for row in summary["images"]}
    requested = args.stems or DEFAULT_STEMS
    if args.stems:
        missing = [stem for stem in requested if stem not in rows_by_stem]
        if missing:
            parser.error(f"stems not present in summary.json: {missing}")
    selected = [rows_by_stem[stem] for stem in requested if stem in rows_by_stem]
    if not selected:
        selected = list(rows_by_stem.values())[:2]
    if not selected:
        raise ValueError("summary.json has no images")
    output_dir = root / "visual"
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_images = {}
    if (output_dir / "index.json").is_file():
        previous = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
        previous_images = {item["stem"]: item for item in previous.get("images", [])}
    index = {
        "source_summary": "../summary.json",
        "method": "Crops of cached whole-image inference and postprocessing; no local crop inference.",
        "color_scales": {"q_1_minus_affinity": [0, 1], "S_D2_D4_B": [0, 1], "B_minus_S": [-0.5, 0.5]},
        "offsets_dy_dx": OFFSETS,
        "id_colors": "Deterministic per numeric ID; no cross-model instance matching is implied.",
        "gt_overlay": "Cyan valid GT partition lines; directions use GT cross-grain edge-start contours.",
        "coordinate_convention": "Exclusive x1/y1. Decoder letterbox content starts at (0,0); native crops use floor starts and ceil ends.",
        "images": [],
    }
    for row in selected:
        stem = row["stem"]
        rois = choose_rois(summary, row)
        record = {"stem": stem, "has_gt": bool(row["has_gt"]), "rois": [], "files": []}
        with np.load(root / "cache" / f"{stem}.npz", allow_pickle=False) as data:
            overview = output_dir / f"{stem}_overview.png"
            render_overview(data, row, rois, overview)
            record["files"].append(overview.name)
            for roi in rois if args.all_rois else rois[:1]:
                directions = output_dir / f"{stem}_roi{roi['index']}_directions.png"
                pipeline = output_dir / f"{stem}_roi{roi['index']}_pipeline.png"
                render_directions(data, row, roi, directions)
                render_pipeline(data, row, roi, pipeline)
                record["files"].extend((directions.name, pipeline.name))
                record["rois"].append({"grid": roi, "native": native_roi(roi, row),
                                       "directions": directions.name, "pipeline": pipeline.name})
        previous_images[stem] = record
        print(f"Rendered {stem}: {len(record['files'])} figures", flush=True)
    index["images"] = list(previous_images.values())
    (output_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {output_dir / 'index.json'}", flush=True)


if __name__ == "__main__":
    main()
