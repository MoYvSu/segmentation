# -*- coding: utf-8 -*-
"""Print compact checkpoint provenance and architecture without building SAM2."""

import argparse
import hashlib
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.checkpoint import checkpoint_architecture


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        info = {
            "path": os.path.abspath(path),
            "size_mb": round(os.path.getsize(path) / 1024 ** 2, 2),
            "sha256": sha256_file(path),
            "format_version": checkpoint.get("format_version", 1),
            "epoch": checkpoint.get("epoch"),
            "best_composite_score": checkpoint.get(
                "best_composite_score", checkpoint.get("best_val_iou")
            ),
            "architecture": checkpoint_architecture(checkpoint),
            "provenance": checkpoint.get("provenance"),
            "has_optimizer": "optimizer_state_dict" in checkpoint,
            "has_scheduler": "scheduler_state_dict" in checkpoint,
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
