# -*- coding: utf-8 -*-
"""Audit disconnected instance IDs and the uint8 overflow bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be ALIAS=DIRECTORY")
    alias, directory = value.split("=", 1)
    return alias.strip(), Path(directory.strip())


def audit_map(path: Path, only_id255: bool = False) -> dict:
    instance_map = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if instance_map is None:
        raise FileNotFoundError(path)
    all_ids = [int(value) for value in np.unique(instance_map) if int(value) > 0]
    ids = [255] if only_id255 and 255 in all_ids else ([] if only_id255 else all_ids)
    disconnected = {}
    total_components = 0
    for instance_id in ids:
        ys, xs = np.nonzero(instance_map == instance_id)
        crop = instance_map[
            int(ys.min()) : int(ys.max()) + 1,
            int(xs.min()) : int(xs.max()) + 1,
        ]
        component_count = int(
            cv2.connectedComponents((crop == instance_id).astype(np.uint8), 8)[0] - 1
        )
        total_components += component_count
        if component_count > 1:
            disconnected[str(instance_id)] = component_count
    id255_components = int(disconnected.get("255", 1 if 255 in ids else 0))
    id255_area = int(np.count_nonzero(instance_map == 255))
    return {
        "image": path.stem.replace("_inst", ""),
        "instance_ids": len(all_ids),
        "connected_components": total_components,
        "disconnected_id_count": len(disconnected),
        "extra_components": total_components - len(ids),
        "id255_present": 255 in ids,
        "id255_components": id255_components,
        "id255_area": id255_area,
        "zero_fraction": float(np.mean(instance_map == 0)),
        "disconnected_ids": disconnected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only-id255", action="store_true")
    args = parser.parse_args()
    report = {"inputs": {}}
    for alias, directory in args.input:
        rows = [
            audit_map(path, only_id255=args.only_id255)
            for path in sorted(directory.glob("*_inst.png"))
        ]
        if not rows:
            raise FileNotFoundError(f"no *_inst.png under {directory}")
        report["inputs"][alias] = {
            "directory": str(directory.resolve()),
            "num_images": len(rows),
            "images_at_255": sum(int(row["instance_ids"] == 255) for row in rows),
            "images_with_disconnected_ids": sum(
                int(row["disconnected_id_count"] > 0) for row in rows
            ),
            "total_instance_ids": sum(row["instance_ids"] for row in rows),
            "total_connected_components": sum(row["connected_components"] for row in rows),
            "total_extra_components": sum(row["extra_components"] for row in rows),
            "mean_zero_fraction": float(np.mean([row["zero_fraction"] for row in rows])),
            "images": rows,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
