# -*- coding: utf-8 -*-
"""Render original | left checkpoint | right checkpoint instance comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fit_panel(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (
        max(1, round(image.shape[1] * scale)),
        max(1, round(image.shape[0] * scale)),
    )
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 245, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def read_classes(path):
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    ferrite = sum(int(value) == 1 for value in values.values())
    return len(values), ferrite, len(values) - ferrite


def annotate(panel, title):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (255, 255, 255), -1)
    cv2.putText(
        panel, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
        0.62, (25, 25, 25), 2, cv2.LINE_AA,
    )
    return panel


def triplet(image_path, left_dir, right_dir, left_name, right_name, size):
    stem = image_path.stem
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    left = cv2.imread(str(left_dir / f"{stem}_inst_color.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_dir / f"{stem}_inst_color.png"), cv2.IMREAD_COLOR)
    if original is None or left is None or right is None:
        raise FileNotFoundError(stem)
    left_count = read_classes(left_dir / f"{stem}_class.json")
    right_count = read_classes(right_dir / f"{stem}_class.json")
    width, height = size
    panels = [
        annotate(fit_panel(original, width, height), f"{stem} original"),
        annotate(
            fit_panel(left, width, height),
            f"{left_name}: {left_count[0]} (F{left_count[1]}/P{left_count[2]})",
        ),
        annotate(
            fit_panel(right, width, height),
            f"{right_name}: {right_count[0]} (F{right_count[1]}/P{right_count[2]})",
        ),
    ]
    return np.concatenate(panels, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    image_dir = Path(args.image_dir)
    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = [
        path.name.removesuffix("_inst_color.png")
        for path in sorted(left_dir.glob("*_inst_color.png"))
        if (right_dir / path.name).exists()
    ]
    image_paths = []
    for stem in stems:
        image_path = next(
            (image_dir / f"{stem}{suffix}" for suffix in (".jpg", ".png", ".jpeg")
             if (image_dir / f"{stem}{suffix}").exists()),
            None,
        )
        if image_path is not None:
            image_paths.append(image_path)
    overview_items = []
    for image_path in image_paths:
        large = triplet(
            image_path, left_dir, right_dir,
            args.left_name, args.right_name, (480, 360),
        )
        cv2.imwrite(str(output_dir / f"{image_path.stem}_comparison.png"), large)
        overview_items.append(triplet(
            image_path, left_dir, right_dir,
            args.left_name, args.right_name, (300, 225),
        ))
    rows = []
    for index in range(0, len(overview_items), 2):
        row = overview_items[index:index + 2]
        if len(row) == 1:
            row.append(np.full_like(row[0], 245))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(output_dir / "overview.png"), np.concatenate(rows, axis=0))
    print(f"Rendered {len(image_paths)} comparisons to {output_dir}")


if __name__ == "__main__":
    main()
