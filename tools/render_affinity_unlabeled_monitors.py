# -*- coding: utf-8 -*-
"""Render deterministic, label-free affinity checkpoint comparisons.

The selected images exclude manually labeled samples and SAM2 pseudo-training
sources.  Selection is based only on simple image statistics so the monitor is
never used as supervision or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mim_dataset import list_images
from tools.run_affinity_watershed_ab import postprocess, predict_maps
from train_affinity_geometry_g1 import build_system, write_unlabeled_monitor
from utils.config import load_config, project_path


def parse_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be ALIAS=PATH")
    alias, path = value.split("=", 1)
    alias = alias.strip()
    path = path.strip()
    if not alias or not path:
        raise argparse.ArgumentTypeError("checkpoint must be ALIAS=PATH")
    return alias, path


def excluded_stems(split_path: Path | None, manifest_path: Path | None) -> set[str]:
    excluded: set[str] = set()
    if split_path is not None and split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        for key in ("train", "val"):
            excluded.update(Path(name).stem for name in split.get(key, []))
    if manifest_path is not None and manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_relpath", row.get("source_file", ""))
            excluded.add(Path(source).stem)
    return excluded


def image_descriptor(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    values = small.astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(values, cv2.CV_32F)
    return np.asarray(
        [values.mean(), values.std(), math.log1p(float(laplacian.var()))],
        dtype=np.float32,
    )


def select_diverse_images(paths: list[Path], count: int):
    if count <= 0:
        return [], {}
    if len(paths) <= count:
        selected = list(paths)
        return selected, {path.stem: image_descriptor(path).tolist() for path in selected}
    features = np.stack([image_descriptor(path) for path in paths])
    normalized = (features - features.mean(axis=0)) / np.maximum(
        features.std(axis=0), 1.0e-6
    )
    # Start with the most unusual image, then maximize distance to the already
    # selected set.  This is deterministic and does not use labels.
    selected_indices = [int(np.argmax(np.linalg.norm(normalized, axis=1)))]
    minimum_distance = np.full(len(paths), np.inf, dtype=np.float32)
    while len(selected_indices) < count:
        latest = normalized[selected_indices[-1]]
        distance = np.linalg.norm(normalized - latest, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected_indices] = -1.0
        selected_indices.append(int(np.argmax(minimum_distance)))
    selected = [paths[index] for index in selected_indices]
    descriptors = {
        paths[index].stem: {
            "mean_luma": float(features[index, 0]),
            "luma_std": float(features[index, 1]),
            "log_laplacian_variance": float(features[index, 2]),
        }
        for index in selected_indices
    }
    return selected, descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/affinity_geometry_g2_sam2.yaml")
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True,
        help="Repeatable ALIAS=PATH checkpoint specification.",
    )
    parser.add_argument("--image-dir", default="data/unlabeled")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--exclude-split")
    parser.add_argument("--exclude-manifest")
    parser.add_argument("--thresholds", default="0.45,0.55")
    parser.add_argument(
        "--fusion-mode", choices=("mean", "short", "gated"), default="mean"
    )
    parser.add_argument("--distance2-weight", type=float, default=0.50)
    parser.add_argument("--distance4-weight", type=float, default=0.25)
    parser.add_argument("--support-threshold", type=float, default=0.20)
    parser.add_argument("--support-temperature", type=float, default=0.05)
    parser.add_argument(
        "--short-reduction", choices=("mean", "top2", "softmax"), default="mean"
    )
    parser.add_argument("--short-softmax-temperature", type=float, default=0.15)
    parser.add_argument(
        "--output-dir", default="outputs/experiments/affinity_unlabeled_monitor12"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["affinity_geometry_g1"]
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and config["sam2"].get("device") == "cuda"
        else "cpu"
    )
    split_path = (
        Path(project_path(config, args.exclude_split)) if args.exclude_split else None
    )
    manifest_path = (
        Path(project_path(config, args.exclude_manifest))
        if args.exclude_manifest else None
    )
    excluded = excluded_stems(split_path, manifest_path)
    image_dir = Path(project_path(config, args.image_dir))
    candidates = [
        Path(path) for path in list_images(str(image_dir))
        if Path(path).stem not in excluded
    ]
    selected, descriptors = select_diverse_images(candidates, int(args.count))
    if not selected:
        raise RuntimeError("no eligible unlabeled monitor images")

    output_root = Path(project_path(config, args.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    thresholds = [float(value) for value in args.thresholds.split(",")]
    fusion_kwargs = {
        "distance2_weight": args.distance2_weight,
        "distance4_weight": args.distance4_weight,
        "support_threshold": args.support_threshold,
        "support_temperature": args.support_temperature,
        "short_reduction": args.short_reduction,
        "short_softmax_temperature": args.short_softmax_temperature,
    }
    system, _, _, digest = build_system(config, cfg, device)
    rows = []
    for alias, checkpoint_value in args.checkpoint:
        checkpoint_path = Path(project_path(config, checkpoint_value))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_digest = checkpoint.get("semantic_state_digest")
        if expected_digest and expected_digest != digest:
            raise RuntimeError(f"{alias} references a different semantic checkpoint")
        system.geometry_decoder.load_state_dict(
            checkpoint["geometry_state_dict"], strict=True
        )
        system.eval()
        epoch = int(checkpoint.get("epoch", checkpoint.get("step", -1)))
        alias_root = output_root / alias
        for image_path in selected:
            write_unlabeled_monitor(
                system, image_path, alias_root, epoch,
                int(cfg.get("input_size", 1024)), int(cfg.get("output_grid", 512)),
            )
            image, _, affinity_output, boundary_probability = predict_maps(
                system, image_path, int(cfg.get("input_size", 1024)), device,
                args.fusion_mode, fusion_kwargs,
            )
            cv2.imwrite(
                str(alias_root / "unlabeled_monitor" / f"epoch_{epoch:03d}"
                    / image_path.stem / "source.jpg"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            preview = cv2.resize(
                boundary_probability[0, 0].numpy(),
                (int(cfg.get("output_grid", 512)), int(cfg.get("output_grid", 512))),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imwrite(
                str(alias_root / "unlabeled_monitor" / f"epoch_{epoch:03d}"
                    / image_path.stem / f"boundary_{args.fusion_mode}.png"),
                (preview * 255).astype(np.uint8),
            )
            row = {
                "checkpoint": alias,
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "epoch": epoch,
                "image": image_path.name,
                **descriptors[image_path.stem],
                "mean_boundary_probability": float(boundary_probability.mean()),
            }
            for threshold in thresholds:
                arm = f"bt{int(round(threshold * 100)):03d}"
                arm_dir = alias_root / "watershed" / arm
                _, instance_map, _ = postprocess(
                    affinity_output, image.shape[:2], arm_dir, image_path.stem,
                    config["inference"], threshold, True,
                    image_rgb=image,
                )
                row[f"instances_{arm}"] = int(np.unique(instance_map).size - 1)
            rows.append(row)

    report = {
        "definition": "label-free visual audit; never selects checkpoints",
        "selection": "farthest-point sampling of luma/std/Laplacian descriptors",
        "excluded_stem_count": len(excluded),
        "candidate_count": len(candidates),
        "selected_images": [path.name for path in selected],
        "thresholds": thresholds,
        "fusion": {"mode": args.fusion_mode, **fusion_kwargs},
        "rows": rows,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
