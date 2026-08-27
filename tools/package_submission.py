# -*- coding: utf-8 -*-
"""Validate and package flat competition submission files."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_pair(image_path: Path, prediction_dir: Path) -> tuple[Path, Path, int]:
    stem = image_path.stem
    instance_path = prediction_dir / f"{stem}_inst.png"
    class_path = prediction_dir / f"{stem}_class.json"
    if not instance_path.is_file() or not class_path.is_file():
        raise FileNotFoundError(f"missing prediction pair for {stem}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    instance_map = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED)
    if image is None or instance_map is None:
        raise ValueError(f"failed to read image or instance map for {stem}")
    if instance_map.ndim != 2 or instance_map.dtype != np.uint8:
        raise ValueError(
            f"{instance_path.name} must be single-channel uint8, got "
            f"shape={instance_map.shape}, dtype={instance_map.dtype}"
        )
    if instance_map.shape != image.shape[:2]:
        raise ValueError(
            f"{stem} size mismatch: prediction={instance_map.shape}, "
            f"image={image.shape[:2]}"
        )

    raw_classes = json.loads(class_path.read_text(encoding="utf-8"))
    class_map = {int(key): int(value) for key, value in raw_classes.items()}
    if any(value not in (0, 1) for value in class_map.values()):
        raise ValueError(f"{class_path.name} contains a class outside 0/1")
    present_ids = {int(value) for value in np.unique(instance_map) if int(value)}
    declared_ids = set(class_map)
    if present_ids != declared_ids:
        raise ValueError(
            f"{stem} instance/class ids differ: "
            f"mask_only={sorted(present_ids - declared_ids)}, "
            f"json_only={sorted(declared_ids - present_ids)}"
        )
    if any(not 1 <= value <= 255 for value in present_ids):
        raise ValueError(f"{stem} has an instance id outside 1..255")
    return instance_path, class_path, len(present_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    output = Path(args.output).resolve()
    images = sorted(
        path for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no test images under {test_dir}")

    pairs = [validate_pair(path, prediction_dir) for path in images]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for instance_path, class_path, _ in pairs:
            archive.write(instance_path, instance_path.name)
            archive.write(class_path, class_path.name)

    expected_names = {
        path.name for instance_path, class_path, _ in pairs
        for path in (instance_path, class_path)
    }
    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if set(names) != expected_names or len(names) != len(expected_names):
            raise RuntimeError("ZIP entries differ from validated submission files")
        if any("/" in name or "\\" in name for name in names):
            raise RuntimeError("ZIP contains a nested directory")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC failure: {corrupt}")

    print(json.dumps({
        "archive": str(output),
        "images": len(images),
        "files": len(expected_names),
        "instances": sum(count for _, _, count in pairs),
        "flat": True,
        "crc_ok": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
