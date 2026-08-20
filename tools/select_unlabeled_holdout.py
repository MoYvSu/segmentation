# -*- coding: utf-8 -*-
"""Select a reproducible appearance-diverse unlabeled monitor holdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mim_dataset import list_images


def image_features(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    quantiles = np.quantile(gray, [0.10, 0.50, 0.90])
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    channel_means = rgb.mean(axis=(0, 1))
    return np.array(
        [
            gray.mean(),
            gray.std(),
            *quantiles,
            np.log1p(laplacian.var() * 1000.0),
            channel_means[0] - channel_means[1],
            channel_means[2] - channel_means[1],
        ],
        dtype=np.float64,
    )


def farthest_point_subset(features: np.ndarray, count: int) -> list[int]:
    median = np.median(features, axis=0)
    scale = np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)
    normalized = (features - median) / np.maximum(scale, 1e-6)
    center = int(np.argmin(np.square(normalized).sum(axis=1)))
    selected = [center]
    min_distance = np.square(normalized - normalized[center]).sum(axis=1)
    while len(selected) < count:
        min_distance[selected] = -1.0
        index = int(np.argmax(min_distance))
        selected.append(index)
        distance = np.square(normalized - normalized[index]).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()

    images = list_images(args.input_dir)
    if len(images) < args.count:
        raise ValueError(f"requested {args.count} images from only {len(images)}")
    features = np.stack([image_features(path) for path in images])
    indices = farthest_point_subset(features, args.count)
    names = sorted(Path(images[index]).name for index in indices)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Appearance-diverse unlabeled holdout; excluded from GDA/semi-supervised training.\n"
        + "\n".join(names)
        + "\n",
        encoding="utf-8",
    )
    print(f"selected {len(names)} / {len(images)} images -> {output}")
    for name in names:
        print(name)


if __name__ == "__main__":
    main()
