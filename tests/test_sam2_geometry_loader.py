import hashlib
import json

import cv2
import numpy as np
import torch

from data.sam2_geometry_dataset import (
    SAM2GeometryDataset,
    find_high_coverage_crop_candidates,
)


def test_high_coverage_crop_candidates_keep_dense_instance_interfaces():
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[:16, :16] = 1
    labels[:16, 16:32] = 2
    labels[16:32, :16] = 3
    labels[16:32, 16:32] = 4

    candidates = find_high_coverage_crop_candidates(
        labels,
        crop_size=32,
        min_coverage=0.95,
        min_instances=4,
        min_negative_edge_pixels=32,
        stride=16,
        max_candidates=4,
    )

    assert len(candidates) == 1
    assert candidates[0]["top"] == 0
    assert candidates[0]["left"] == 0
    assert candidates[0]["coverage"] == 1.0
    assert candidates[0]["instance_count"] == 4

def test_sam2_geometry_dataset_loads_portable_manifest(tmp_path):
    source_dir = tmp_path / "unlabeled"
    dataset_dir = tmp_path / "sam2_geometry"
    masks_dir = dataset_dir / "masks"
    source_dir.mkdir()
    masks_dir.mkdir(parents=True)
    image = np.full((20, 30, 3), 170, dtype=np.uint8)
    image_path = source_dir / "sample.png"
    cv2.imwrite(str(image_path), image)
    labels = np.zeros((20, 30), dtype=np.uint16)
    labels[2:18, 2:14] = 1
    labels[2:18, 16:28] = 2
    interiors = labels.copy()
    interiors[:, 2:4] = 0
    np.savez_compressed(
        masks_dir / "sample.npz",
        instance_map=labels,
        interior_instance_map=interiors,
    )
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    row = {
        "source_file": "/nonportable/old/location/sample.png",
        "source_relpath": "sample.png",
        "source_sha256": digest,
        "mask_file": "masks/sample.npz",
    }
    (dataset_dir / "manifest.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )

    dataset = SAM2GeometryDataset(
        dataset_dir,
        source_dir,
        image_size=32,
        output_grid=16,
        cache_in_memory=True,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 32)
    assert sample["instance_map"].shape == (16, 16)
    assert sample["instance_count"] == 2
    assert sample["valid_content"].dtype == torch.bool
