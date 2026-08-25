# -*- coding: utf-8 -*-
"""Class-agnostic SAM2 pseudo-instance dataset for geometry-only training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox
from utils.offset_letterbox import letterbox_instance_geometry


class SAM2GeometryDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        source_dir: str | Path,
        *,
        image_size: int = 1024,
        output_grid: int = 512,
        sample_names: Sequence[str] | None = None,
        use_eroded_interiors: bool = False,
        cache_in_memory: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.source_dir = Path(source_dir)
        self.image_size = int(image_size)
        self.output_grid = int(output_grid)
        self.use_eroded_interiors = bool(use_eroded_interiors)
        self.cache_in_memory = bool(cache_in_memory)
        manifest = self.dataset_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        requested = {Path(name).stem for name in (sample_names or [])}
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rows = [
            row
            for row in rows
            if not requested
            or Path(row.get("source_relpath", row["source_file"])).stem in requested
        ]
        if requested:
            found = {
                Path(row.get("source_relpath", row["source_file"])).stem
                for row in self.rows
            }
            missing = sorted(requested - found)
            if missing:
                raise FileNotFoundError(f"requested SAM2 samples missing: {missing}")
        if not self.rows:
            raise ValueError(f"no SAM2 geometry samples in {manifest}")
        if len({row["source_sha256"] for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate SAM2 source hashes in manifest")
        self._cached_samples = None
        if self.cache_in_memory:
            self._cached_samples = [self._load_sample(i) for i in range(len(self.rows))]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self._cached_samples is not None:
            return self._cached_samples[int(index)]
        return self._load_sample(int(index))

    def _source_path(self, row):
        relative = row.get("source_relpath")
        if relative:
            candidate = self.source_dir / relative
            if candidate.is_file():
                return candidate
        original = Path(row["source_file"])
        if original.is_file():
            return original
        fallback = self.source_dir / original.name
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"SAM2 source image missing: {relative or original}")

    def _load_sample(self, index):
        row = self.rows[int(index)]
        image_path = self._source_path(row)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lb, _, _, _ = letterbox(image_rgb, self.image_size)
        mask_path = self.dataset_dir / row["mask_file"]
        with np.load(mask_path) as payload:
            key = (
                "interior_instance_map"
                if self.use_eroded_interiors and "interior_instance_map" in payload
                else "instance_map"
            )
            labels = payload[key].astype(np.int32)
        if labels.shape != image.shape[:2]:
            raise ValueError(
                f"SAM2 labels {labels.shape} != image {image.shape[:2]} for {image_path}"
            )
        geometry_map, valid_content, metadata = letterbox_instance_geometry(
            labels, input_size=self.image_size, output_grid=self.output_grid
        )
        foreground = geometry_map > 0
        return {
            "image": torch.from_numpy(image_lb).permute(2, 0, 1).float() / 255.0,
            "center_target": torch.zeros((1, self.output_grid, self.output_grid)),
            "offset_target": torch.zeros((2, self.output_grid, self.output_grid)),
            "foreground": torch.from_numpy(foreground).unsqueeze(0),
            "valid_content": torch.from_numpy(valid_content).unsqueeze(0),
            "instance_map": torch.from_numpy(geometry_map.astype(np.int64)),
            "image_name": image_path.name,
            "instance_count": int(len(np.unique(geometry_map)) - 1),
            "uncovered_pixels": int(
                np.sum((geometry_map == 0) & valid_content.astype(bool))
            ),
            "content_shape": torch.tensor(
                [metadata.content_height, metadata.content_width], dtype=torch.int32
            ),
            "input_content_shape": torch.tensor(
                [metadata.resized_height, metadata.resized_width], dtype=torch.int32
            ),
        }
