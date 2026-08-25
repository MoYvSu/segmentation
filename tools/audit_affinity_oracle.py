# -*- coding: utf-8 -*-
"""Verify that perfect local affinities reconstruct labeled instances."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.offset_geometry_dataset import OffsetGeometryDataset
from utils.affinity_graph import (
    DEFAULT_AFFINITY_OFFSETS,
    audit_instance_recovery,
    build_affinity_targets,
    reconstruct_affinity_components,
)
from utils.config import load_config, project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/affinity_geometry_g0.yaml")
    parser.add_argument(
        "--output", default="outputs/affinity_geometry_oracle/summary.json"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["affinity_geometry"]
    dataset = OffsetGeometryDataset(
        project_path(config, config["paths"]["raw_data_dir"]),
        image_size=int(cfg.get("input_size", 1024)),
        output_grid=int(cfg.get("output_grid", 512)),
        cache_in_memory=False,
    )
    rows = []
    positive_edges = np.zeros(len(DEFAULT_AFFINITY_OFFSETS), dtype=np.int64)
    negative_edges = np.zeros_like(positive_edges)
    for sample in dataset:
        labels = sample["instance_map"].numpy().astype(np.int32)
        valid_content = sample["valid_content"][0].numpy().astype(bool)
        affinity, edge_valid = build_affinity_targets(labels, valid_content)
        positive_edges += np.sum((affinity > 0.5) & edge_valid, axis=(1, 2))
        negative_edges += np.sum((affinity <= 0.5) & edge_valid, axis=(1, 2))
        prediction, graph_audit = reconstruct_affinity_components(
            labels > 0, affinity, max_instances=255
        )
        recovery = audit_instance_recovery(labels, prediction)
        rows.append({
            "image": sample["image_name"],
            **graph_audit,
            **recovery,
        })
    summary = {
        "image_count": len(rows),
        "offsets": [list(value) for value in DEFAULT_AFFINITY_OFFSETS],
        "all_images_exact": bool(all(row["exact_partition"] for row in rows)),
        "exact_image_count": int(sum(row["exact_partition"] for row in rows)),
        "gt_instance_count": int(sum(row["gt_instance_count"] for row in rows)),
        "pred_instance_count": int(sum(row["pred_instance_count"] for row in rows)),
        "split_gt_instance_count": int(sum(row["split_gt_instance_count"] for row in rows)),
        "merged_pred_instance_count": int(sum(row["merged_pred_instance_count"] for row in rows)),
        "exact_gt_instance_count": int(sum(row["exact_gt_instance_count"] for row in rows)),
        "positive_edges_by_offset": positive_edges.tolist(),
        "negative_edges_by_offset": negative_edges.tolist(),
        "per_image": rows,
    }
    output = Path(project_path(config, args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
