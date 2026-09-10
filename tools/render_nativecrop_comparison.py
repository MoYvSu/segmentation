# -*- coding: utf-8 -*-
"""Render five-arm native-crop comparisons from whole-image inference caches.

Instance colors follow maximum pixel overlap with one shared reference: valid
GT on labeled images, or G4b partitions on unlabeled images. This is a visual
correspondence, not one-to-one instance matching or an evaluation metric.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
import numpy as np


ARMS = ("g4b", "e110_hybrid", "native_hybrid", "e110_direct", "native_direct")
TITLES = {
    "g4b": "E10a semantic + G4b affinity",
    "e110_hybrid": "E10a semantic + e110 affinity",
    "native_hybrid": "E10a semantic + native-crop affinity",
    "e110_direct": "e110 semantic + affinity",
    "native_direct": "Native-crop semantic + affinity",
}


def partition_edges(labels, valid=None):
    valid = np.ones(labels.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:-1] |= (labels[:-1] != labels[1:]) & valid[:-1] & valid[1:]
    edges[:, :-1] |= (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    return edges


def palette(ids):
    hsv = np.zeros((*ids.shape, 3), dtype=np.float32)
    hsv[..., 0] = (ids.astype(np.float64) * 0.618033988749895) % 1.0
    hsv[..., 1], hsv[..., 2] = 0.60, 0.95
    colors = hsv_to_rgb(hsv)
    colors[ids == 0] = 0.16
    return colors


def match_reference(prediction, reference, valid):
    """Whole-image max-overlap assignment; tied overlaps use the lowest GT ID."""
    lut = np.zeros(int(prediction.max()) + 1, dtype=np.uint16)
    selected = (prediction > 0) & (reference > 0) & valid
    if selected.any():
        base = int(reference.max()) + 1
        pairs = prediction[selected].astype(np.uint64) * base + reference[selected].astype(np.uint64)
        unique, counts = np.unique(pairs, return_counts=True)
        pred_ids, ref_ids = (unique // base).astype(np.int64), (unique % base).astype(np.int64)
        order = np.lexsort((ref_ids, -counts, pred_ids))
        pred_ids, ref_ids = pred_ids[order], ref_ids[order]
        keep = np.r_[True, pred_ids[1:] != pred_ids[:-1]]
        lut[pred_ids[keep]] = ref_ids[keep]
    assignments = {str(int(label)): int(lut[label]) for label in np.unique(prediction) if label > 0}
    return lut, assignments


def colored_partitions(labels, lut=None, rgb=None):
    colors = palette(labels if lut is None else lut[labels])
    colors[labels == 0] = 0.12
    if rgb is not None:
        colors = colors * 0.80 + rgb.astype(np.float32) / 255.0 * 0.20
    colors[partition_edges(labels)] = 0.01
    return colors


def show(ax, value, title, *, reference_edges=None, **kwargs):
    handle = ax.imshow(value, interpolation="nearest", **kwargs)
    if reference_edges is not None and reference_edges.any():
        overlay = np.zeros((*reference_edges.shape, 4), dtype=np.float32)
        overlay[reference_edges] = (0, 1, 1, 0.90)
        ax.imshow(overlay, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return handle


def crop(array, box):
    return array[box[1]:box[3], box[0]:box[2]]


def native_box(roi, row):
    height, width = row["native_shape"][:2]
    ch, cw = row["grid_content_shape"]
    return [max(0, math.floor(roi["x0"] * width / cw)),
            max(0, math.floor(roi["y0"] * height / ch)),
            min(width, math.ceil(roi["x1"] * width / cw)),
            min(height, math.ceil(roi["y1"] * height / ch))]


def selected_rois(row, maximum, extra_rois):
    ch, cw = map(int, row["grid_content_shape"])
    rois = row.get("rois", [])
    if not rois:
        size = min(96, ch, cw)
        x0, y0 = (cw - size) // 2, (ch - size) // 2
        rois = [{"kind": "fixed_center", "x0": x0, "y0": y0,
                 "x1": x0 + size, "y1": y0 + size,
                 "selection": "fixed center; independent of model outputs"}]
    rois = list(rois[:maximum])
    rois.extend(roi for roi in extra_rois if roi["stem"] == row["stem"])
    result = []
    for index, raw in enumerate(rois, start=1):
        roi = dict(raw, index=index)
        roi.update({key: int(roi[key]) for key in ("x0", "y0", "x1", "y1")})
        if not (0 <= roi["x0"] < roi["x1"] <= cw and 0 <= roi["y0"] < roi["y1"] <= ch):
            raise ValueError(f"{row['stem']}: ROI outside letterbox content: {roi}")
        roi.setdefault("selection", "fixed ROI from summary.json; diagnostic selection")
        result.append(roi)
    return result


def load_extra_rois(path):
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("regions_fixed_before_run", payload.get("rois"))
    if rows is None:
        raise ValueError("--rois must contain a ROI list, 'rois', or 'regions_fixed_before_run'")
    result = []
    for raw in rows:
        roi = dict(raw)
        if "box" in roi:
            roi.update(dict(zip(("x0", "y0", "x1", "y1"), roi["box"])))
        roi.setdefault("kind", roi.get("role", "prior_fixed_region"))
        roi.setdefault("selection", f"pre-existing fixed decoder-grid ROI imported from {path.name}")
        if "stem" not in roi or not all(key in roi for key in ("x0", "y0", "x1", "y1")):
            raise ValueError(f"incomplete ROI in {path}: {raw}")
        result.append(roi)
    return result


def save(fig, path):
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)


def overview(data, row, rois, colored, reference_rgb, edge, has_gt, path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 11), layout="constrained")
    flat = list(axes.flat)
    show(flat[0], data["rgb"], "Original image", reference_edges=edge if has_gt else None)
    show(flat[1], reference_rgb, "Valid GT reference" if has_gt else "No GT: G4b color reference")
    for index, arm in enumerate(ARMS, start=2):
        show(flat[index], colored[arm], TITLES[arm])
    ch, cw = row["grid_content_shape"]
    delta = data["native_boundary"].astype(np.float32) - data["e110_boundary"].astype(np.float32)
    handle = show(flat[7], delta[:ch, :cw], "Native - e110 boundary", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    fig.colorbar(handle, ax=flat[7], shrink=0.7, location="bottom", label="Boundary difference")
    for roi in rois:
        for index, ax in enumerate(flat):
            box = [roi[key] for key in ("x0", "y0", "x1", "y1")] if index == 7 else native_box(roi, row)
            ax.add_patch(Rectangle((box[0] - 0.5, box[1] - 0.5), box[2] - box[0], box[3] - box[1],
                                   fill=False, edgecolor="#ff3434", linewidth=1.2))
            ax.text(box[0], box[1], str(roi["index"]), color="white", fontsize=9,
                    bbox={"facecolor": "#aa0000", "edgecolor": "none", "pad": 1})
    domain = "GT-aligned colors" if has_gt else "No GT: visual inspection only; G4b-aligned colors"
    fig.suptitle(f"{row['stem']} | {domain}\n"
                 "Colors follow greatest full-image reference overlap. Black: partition edges. "
                 "Split fragments can share a color; inspect their edges.", fontsize=12)
    save(fig, path)


def detail(data, row, roi, colored, reference_rgb, edge, has_gt, assignments, path):
    box = native_box(roi, row)
    grid_box = [roi[key] for key in ("x0", "y0", "x1", "y1")]
    fig, axes = plt.subplots(3, 4, figsize=(17, 13), layout="constrained")
    show(axes[0, 0], crop(data["rgb"], box), "Image + GT edges" if has_gt else "Image (No GT)",
         reference_edges=crop(edge, box) if has_gt else None)
    for column, model in enumerate(("g4b", "e110", "native"), start=1):
        handle = show(axes[0, column], crop(data[f"{model}_boundary"].astype(np.float32), grid_box),
                      f"{model}: fused boundary", cmap="magma", vmin=0, vmax=1)
    fig.colorbar(handle, ax=list(axes[0, 1:]), shrink=0.7, location="bottom", label="Boundary strength [0, 1]")
    show(axes[1, 0], crop(reference_rgb, box), "Valid GT partitions" if has_gt else "G4b color reference (No GT)")
    locations = ((1, 1), (1, 2), (1, 3), (2, 0), (2, 1))
    for arm, location in zip(ARMS, locations):
        show(axes[location], crop(colored[arm], box), TITLES[arm])
    for column, arm in ((2, "e110_hybrid"), (3, "native_hybrid")):
        markers = data[f"{arm}_markers"]
        # Marker IDs need not equal final instance IDs: match marker regions separately.
        marker_lut, _ = match_reference(markers, assignments["reference"], assignments["valid"])
        marker_colors = colored_partitions(crop(markers, box), marker_lut)
        show(axes[2, column], marker_colors, f"{arm}: watershed markers")
    domain = "valid GT" if has_gt else "G4b reference; No GT, visual inspection only"
    fig.suptitle(f"{row['stem']} | ROI {roi['index']}: {roi.get('kind', 'fixed')} | "
                 f"grid x=[{roi['x0']},{roi['x1']}), y=[{roi['y0']},{roi['y1']})\n"
                 f"Color assignment uses whole-image overlap with {domain}. "
                 "Black: predicted partitions; unmatched regions are gray.\n"
                 "Boundaries use the decoder grid; final outputs use the native grid. "
                 "Fixed diagnostic crops; no crop inference or representative-accuracy claim.", fontsize=11)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", default=["train_172", "train_873", "test_026"])
    parser.add_argument("--max-rois", type=int, default=1,
                        help="Maximum original summary ROIs per image, before optional --rois additions.")
    parser.add_argument("--rois", type=Path,
                        help="Append fixed grid ROIs from an earlier JSON (ROI list, rois, or regions_fixed_before_run).")
    args = parser.parse_args()
    if args.max_rois < 1:
        parser.error("--max-rois must be positive")
    root = args.input_dir.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = {row["stem"]: row for row in summary["images"]}
    extra_rois = load_extra_rois(args.rois)
    missing = [stem for stem in args.stems if stem not in rows]
    if missing:
        parser.error(f"stems missing from summary.json: {missing}")
    out = root / "visual"
    out.mkdir(parents=True, exist_ok=True)
    index = {
        "source": "../summary.json", "arms": TITLES,
        "color_assignment": "Each positive prediction ID inherits the color of the reference ID with maximum valid whole-image overlap; tied overlap uses the smallest reference ID. No overlap is gray. Multiple fragments may share a color. This is not one-to-one metric matching.",
        "unlabeled_reference": "G4b instance map; comparison is visual only, not evaluation against GT.",
        "scales": {"boundary": [0, 1], "native_minus_e110_boundary": [-0.5, 0.5]},
        "coordinates": "Exclusive x1/y1 on top-left letterbox content; native crops use floor starts and ceil ends.",
        "additional_roi_source": str(args.rois) if args.rois else None,
        "images": [],
    }
    for stem in args.stems:
        row = rows[stem]
        rois = selected_rois(row, args.max_rois, extra_rois)
        with np.load(root / "cache" / f"{stem}.npz", allow_pickle=False) as data:
            has_gt = bool(row.get("has_gt", np.any((data["gt"] > 0) & data["valid"].astype(bool))))
            reference = data["gt"] if has_gt else data["g4b_instances"]
            valid = data["valid"].astype(bool) if has_gt else np.ones(reference.shape, dtype=bool)
            edge = partition_edges(reference, valid)
            reference_rgb = colored_partitions(reference)
            reference_rgb[~valid] = 0.12
            colored, arm_assignments = {}, {}
            for arm in ARMS:
                instances = data[f"{arm}_instances"]
                lut, arm_assignments[arm] = match_reference(instances, reference, valid)
                colored[arm] = colored_partitions(instances, lut, data["rgb"])
            record = {"stem": stem, "has_gt": has_gt, "reference": "valid GT" if has_gt else "G4b (no GT)",
                      "files": [], "rois": [], "reference_id_assignments": arm_assignments}
            name = f"{stem}_five_arm_overview.png"
            overview(data, row, rois, colored, reference_rgb, edge, has_gt, out / name)
            record["files"].append(name)
            for roi in rois:
                name = f"{stem}_roi{roi['index']}_five_arm.png"
                detail(data, row, roi, colored, reference_rgb, edge, has_gt,
                       {"reference": reference, "valid": valid}, out / name)
                record["files"].append(name)
                record["rois"].append({"grid": roi, "native_bbox": native_box(roi, row), "file": name})
            index["images"].append(record)
        print(f"Rendered {stem}: {len(record['files'])} figures", flush=True)
    (out / "nativecrop_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out / "nativecrop_index.json", flush=True)


if __name__ == "__main__":
    main()
