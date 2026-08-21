# -*- coding: utf-8 -*-
"""Temporary GT-guided color separability analysis.

This script is intentionally not part of the training pipeline.  It samples
interior pixels from LabelMe ferrite/pearlite polygons and reports whether a
simple color rule can separate the two phases on each image and in aggregate.

Example:
    conda run -n sam2_env python tools/tmp_color_separability.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

# When launched as ``python tools/<script>.py``, Python places ``tools/``
# first on sys.path; add the repository root so the existing data parser is
# imported exactly as the training code imports it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import parse_labelme_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--output-dir", default="tmp_analysis_images/color_separability"
    )
    parser.add_argument("--sample-per-class", type=int, default=12000)
    parser.add_argument("--erode-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def sample_pixels(
    image: np.ndarray,
    ferrite_mask: np.ndarray,
    pearlite_mask: np.ndarray,
    sample_per_class: int,
    erode_radius: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Return RGB pixels sampled from eroded, mutually exclusive GT interiors."""
    ferrite = ferrite_mask.astype(np.uint8)
    pearlite = pearlite_mask.astype(np.uint8)
    raw_overlap = int(np.count_nonzero((ferrite > 0) & (pearlite > 0)))

    if erode_radius > 0:
        kernel_size = 2 * erode_radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        ferrite = cv2.erode(ferrite, kernel, iterations=1)
        pearlite = cv2.erode(pearlite, kernel, iterations=1)

    # Remove overlap instead of assigning ambiguous pixels to either class.
    overlap = (ferrite > 0) & (pearlite > 0)
    ferrite = (ferrite > 0) & ~overlap
    pearlite = (pearlite > 0) & ~overlap

    ferrite_indices = np.flatnonzero(ferrite)
    pearlite_indices = np.flatnonzero(pearlite)

    def choose(indices: np.ndarray) -> np.ndarray:
        if len(indices) <= sample_per_class:
            return indices
        return rng.choice(indices, size=sample_per_class, replace=False)

    f_indices = choose(ferrite_indices)
    p_indices = choose(pearlite_indices)
    flat_image = image.reshape(-1, 3)
    stats = {
        "raw_ferrite_pixels": int(np.count_nonzero(ferrite_mask)),
        "raw_pearlite_pixels": int(np.count_nonzero(pearlite_mask)),
        "raw_overlap_pixels": raw_overlap,
        "interior_ferrite_pixels": int(len(ferrite_indices)),
        "interior_pearlite_pixels": int(len(pearlite_indices)),
        "sampled_ferrite_pixels": int(len(f_indices)),
        "sampled_pearlite_pixels": int(len(p_indices)),
    }
    return flat_image[f_indices], flat_image[p_indices], stats


def rgb_to_spaces(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    lab8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    # Convert OpenCV's 8-bit Lab encoding to approximately CIE Lab units.
    lab = np.empty_like(lab8)
    lab[:, 0] = lab8[:, 0] * (100.0 / 255.0)
    lab[:, 1] = lab8[:, 1] - 128.0
    lab[:, 2] = lab8[:, 2] - 128.0
    return {
        "rgb": rgb.astype(np.float32),
        "hsv": hsv,
        "lab": lab,
    }


def histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int = 48) -> float:
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 1.0
    hist_a, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hist_b, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    hist_a = hist_a.astype(np.float64) / max(hist_a.sum(), 1)
    hist_b = hist_b.astype(np.float64) / max(hist_b.sum(), 1)
    return float(np.minimum(hist_a, hist_b).sum())


def best_threshold(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    """Return best balanced accuracy, threshold, and polarity for one channel."""
    values = np.concatenate([a, b]).astype(np.float64)
    labels = np.concatenate([np.ones(len(a), dtype=np.uint8), np.zeros(len(b), dtype=np.uint8)])
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_labels = labels[order]
    positives_right = np.cumsum(sorted_labels[::-1])[::-1]
    negatives_right = np.cumsum((1 - sorted_labels)[::-1])[::-1]
    positives_total = max(int(labels.sum()), 1)
    negatives_total = max(int((1 - labels).sum()), 1)

    # Threshold between consecutive sorted values; predict class 1 on the right.
    change = np.flatnonzero(np.diff(sorted_values) > 0)
    candidate_indices = np.concatenate((np.array([-1]), change))
    best = (-1.0, float(sorted_values[0]), 1)
    for index in candidate_indices:
        right_start = index + 1
        tp = int(positives_right[right_start]) if right_start < len(values) else 0
        fp = int(negatives_right[right_start]) if right_start < len(values) else 0
        tpr = tp / positives_total
        fpr = fp / negatives_total
        score_right = 0.5 * (tpr + (1.0 - fpr))
        if right_start == 0:
            threshold = sorted_values[0] - 1e-6
        elif right_start == len(values):
            threshold = sorted_values[-1] + 1e-6
        else:
            threshold = 0.5 * (sorted_values[index] + sorted_values[right_start])
        if score_right > best[0]:
            best = (score_right, float(threshold), 1)
        score_left = 1.0 - score_right
        if score_left > best[0]:
            best = (score_left, float(threshold), -1)
    return best


def linear_probe(a: np.ndarray, b: np.ndarray, seed: int) -> Tuple[float, float]:
    """Return held-out balanced accuracy and ROC-AUC for a small linear probe."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return float("nan"), float("nan")

    n = min(len(a), len(b), 6000)
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), size=n, replace=False)]
    b = b[rng.choice(len(b), size=n, replace=False)]
    x = np.concatenate([a, b]).astype(np.float32)
    y = np.concatenate([np.ones(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    prediction = (prob >= 0.5).astype(np.uint8)
    return float(balanced_accuracy_score(y_test, prediction)), float(roc_auc_score(y_test, prob))


def summarize_pair(
    f_spaces: Dict[str, np.ndarray],
    p_spaces: Dict[str, np.ndarray],
    seed: int,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for space_name, names in (
        ("rgb", ("r", "g", "b")),
        ("hsv", ("h", "s", "v")),
        ("lab", ("l", "a", "b_lab")),
    ):
        f = f_spaces[space_name]
        p = p_spaces[space_name]
        for channel, name in enumerate(names):
            f_channel = f[:, channel]
            p_channel = p[:, channel]
            pooled_std = np.sqrt(0.5 * (f_channel.var() + p_channel.var()))
            cohen_d = (f_channel.mean() - p_channel.mean()) / max(float(pooled_std), 1e-6)
            threshold_score, threshold, polarity = best_threshold(f_channel, p_channel)
            prefix = f"{space_name}_{name}"
            result[f"{prefix}_f_mean"] = float(f_channel.mean())
            result[f"{prefix}_p_mean"] = float(p_channel.mean())
            result[f"{prefix}_mean_delta_f_minus_p"] = float(f_channel.mean() - p_channel.mean())
            result[f"{prefix}_cohen_d"] = float(cohen_d)
            result[f"{prefix}_hist_overlap"] = histogram_overlap(f_channel, p_channel)
            result[f"{prefix}_best_threshold_bal_acc"] = float(threshold_score)
            result[f"{prefix}_best_threshold"] = float(threshold)
            result[f"{prefix}_threshold_polarity"] = float(polarity)
            for q in (10, 50, 90):
                result[f"{prefix}_f_p{q}"] = float(np.percentile(f_channel, q))
                result[f"{prefix}_p_p{q}"] = float(np.percentile(p_channel, q))

        probe_bal_acc, probe_auc = linear_probe(f, p, seed)
        result[f"{space_name}_linear_probe_bal_acc"] = probe_bal_acc
        result[f"{space_name}_linear_probe_auc"] = probe_auc

    f_lab = f_spaces["lab"].mean(axis=0)
    p_lab = p_spaces["lab"].mean(axis=0)
    result["lab_mean_delta_e"] = float(np.linalg.norm(f_lab - p_lab))
    return result


def make_plot(
    image_name: str,
    f_spaces: Dict[str, np.ndarray],
    p_spaces: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rng = np.random.default_rng(17)
    f_lab = f_spaces["lab"]
    p_lab = p_spaces["lab"]
    f_rgb = f_spaces["rgb"]
    p_rgb = p_spaces["rgb"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{image_name}: ferrite vs pearlite color distribution")
    for col, channel in enumerate((0, 1, 2)):
        ax = axes[0, col]
        ax.hist(f_rgb[:, channel], bins=40, density=True, alpha=0.55, label="ferrite")
        ax.hist(p_rgb[:, channel], bins=40, density=True, alpha=0.55, label="pearlite")
        ax.set_title(("RGB"[channel]).upper())
        ax.set_xlabel("8-bit value")
        ax.set_ylabel("density")
        if col == 0:
            ax.legend()

    f_choice = rng.choice(len(f_lab), size=min(len(f_lab), 3500), replace=False)
    p_choice = rng.choice(len(p_lab), size=min(len(p_lab), 3500), replace=False)
    axes[1, 0].scatter(f_lab[f_choice, 1], f_lab[f_choice, 2], s=2, alpha=0.18, label="ferrite")
    axes[1, 0].scatter(p_lab[p_choice, 1], p_lab[p_choice, 2], s=2, alpha=0.18, label="pearlite")
    axes[1, 0].set_xlabel("Lab a*")
    axes[1, 0].set_ylabel("Lab b*")
    axes[1, 0].set_title("Lab chromaticity")
    axes[1, 0].legend(markerscale=4)

    axes[1, 1].hist(f_lab[:, 0], bins=40, density=True, alpha=0.55, label="ferrite")
    axes[1, 1].hist(p_lab[:, 0], bins=40, density=True, alpha=0.55, label="pearlite")
    axes[1, 1].set_title("Lab L* brightness")
    axes[1, 1].set_xlabel("L*")
    axes[1, 1].legend()

    axes[1, 2].hist(f_lab[:, 1], bins=40, density=True, alpha=0.55, label="ferrite a*")
    axes[1, 2].hist(p_lab[:, 1], bins=40, density=True, alpha=0.55, label="pearlite a*")
    axes[1, 2].hist(f_lab[:, 2], bins=40, density=True, alpha=0.35, label="ferrite b*")
    axes[1, 2].hist(p_lab[:, 2], bins=40, density=True, alpha=0.35, label="pearlite b*")
    axes[1, 2].set_title("Lab chromatic channels")
    axes[1, 2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def load_samples(args: argparse.Namespace) -> Iterable[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    data_dir = Path(args.data_dir)
    for image_path in sorted(data_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            continue
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[skip] unreadable image: {image_path}")
            continue
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = parse_labelme_json(str(json_path), image.shape[0], image.shape[1])
        yield image_path.stem, image, masks["ferrite"], masks["pearlite"]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "per_image_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    pooled_f = {key: [] for key in ("rgb", "hsv", "lab")}
    pooled_p = {key: [] for key in ("rgb", "hsv", "lab")}
    image_records = []

    for image_name, image, ferrite_mask, pearlite_mask in load_samples(args):
        ferrite_rgb, pearlite_rgb, mask_stats = sample_pixels(
            image,
            ferrite_mask,
            pearlite_mask,
            args.sample_per_class,
            args.erode_radius,
            rng,
        )
        if len(ferrite_rgb) < 20 or len(pearlite_rgb) < 20:
            print(f"[skip] {image_name}: insufficient interior pixels")
            continue
        f_spaces = rgb_to_spaces(ferrite_rgb)
        p_spaces = rgb_to_spaces(pearlite_rgb)
        row = {"image": image_name, **mask_stats}
        row.update(summarize_pair(f_spaces, p_spaces, args.seed))
        rows.append(row)
        image_records.append((f_spaces, p_spaces))
        for key in pooled_f:
            pooled_f[key].append(f_spaces[key])
            pooled_p[key].append(p_spaces[key])
        make_plot(image_name, f_spaces, p_spaces, plot_dir / f"{image_name}_color.png")

    if not rows:
        raise RuntimeError("No usable labeled images found")

    # Leave-one-image-out evaluation answers a more useful question than a
    # threshold fitted on all pixels: does a single global L* cutoff transfer
    # to an unseen sample image?
    loo_scores = []
    for index, (f_spaces, p_spaces) in enumerate(image_records):
        train_f = np.concatenate(
            [record[0]["lab"][: min(len(record[0]["lab"]), 4000), 0]
             for j, record in enumerate(image_records) if j != index]
        )
        train_p = np.concatenate(
            [record[1]["lab"][: min(len(record[1]["lab"]), 4000), 0]
             for j, record in enumerate(image_records) if j != index]
        )
        _, threshold, polarity = best_threshold(train_f, train_p)
        test_f = f_spaces["lab"][:, 0]
        test_p = p_spaces["lab"][:, 0]
        f_pred = polarity * (test_f - threshold) >= 0
        p_pred = polarity * (test_p - threshold) >= 0
        loo_balanced_accuracy = 0.5 * (float(f_pred.mean()) + float((~p_pred).mean()))
        rows[index]["lab_l_loo_threshold"] = float(threshold)
        rows[index]["lab_l_loo_bal_acc"] = float(loo_balanced_accuracy)
        rows[index]["lab_l_loo_polarity"] = float(polarity)
        loo_scores.append(loo_balanced_accuracy)

    csv_path = output_dir / "per_image_color_separability.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    pooled_f_spaces = {key: np.concatenate(value) for key, value in pooled_f.items()}
    pooled_p_spaces = {key: np.concatenate(value) for key, value in pooled_p.items()}
    pooled_summary = summarize_pair(pooled_f_spaces, pooled_p_spaces, args.seed)
    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "images_analyzed": len(rows),
        "seed": args.seed,
        "sample_per_class": args.sample_per_class,
        "erode_radius": args.erode_radius,
        "pooled_sampled_ferrite": int(len(pooled_f_spaces["rgb"])),
        "pooled_sampled_pearlite": int(len(pooled_p_spaces["rgb"])),
        "pooled": pooled_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Analyzed {len(rows)} images")
    print(f"Per-image CSV: {csv_path}")
    print(f"Per-image plots: {plot_dir}")
    print("Pooled separability:")
    for space in ("rgb", "hsv", "lab"):
        print(
            f"  {space}: linear_bal_acc={pooled_summary[f'{space}_linear_probe_bal_acc']:.4f}, "
            f"ROC-AUC={pooled_summary[f'{space}_linear_probe_auc']:.4f}"
        )
    print(
        f"  Lab mean DeltaE={pooled_summary['lab_mean_delta_e']:.3f}; "
        f"Lab L* threshold_bal_acc={pooled_summary['lab_l_best_threshold_bal_acc']:.4f}; "
        f"Lab a* threshold_bal_acc={pooled_summary['lab_a_best_threshold_bal_acc']:.4f}; "
        f"Lab b* threshold_bal_acc={pooled_summary['lab_b_lab_best_threshold_bal_acc']:.4f}"
    )
    print(
        f"  Leave-one-image-out Lab L* threshold: mean={np.mean(loo_scores):.4f}, "
        f"min={np.min(loo_scores):.4f}, max={np.max(loo_scores):.4f}"
    )


if __name__ == "__main__":
    main()
