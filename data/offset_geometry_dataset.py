# -*- coding: utf-8 -*-
"""Global letterbox dataset for center-offset geometry supervision."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox
from utils.flow_instances import build_center_offset_target
from utils.instance_metrics import load_labelme_instances
from utils.offset_letterbox import letterbox_instance_geometry


def load_completed_instance_map(path: str | Path, image_shape) -> np.ndarray:
    """读取补缝后的唯一实例目标；零 ID 与剩余 unknown 必须完全一致。"""
    with np.load(path, allow_pickle=False) as payload:
        instances = payload["instance_map"]
        unknown = payload["residual_unknown"]
    if instances.shape != tuple(image_shape) or unknown.shape != instances.shape:
        raise ValueError(f"completed GT shape mismatch: {path}")
    if instances.dtype.kind not in "iu" or np.any(instances < 0):
        raise ValueError(f"completed GT requires nonnegative integer instance IDs: {path}")
    if int(instances.max()) > np.iinfo(np.int32).max:
        raise ValueError(f"completed GT instance IDs exceed int32: {path}")
    if not np.all((unknown == 0) | (unknown == 1)):
        raise ValueError(f"completed GT residual_unknown must be binary: {path}")
    if not np.array_equal(unknown.astype(bool), instances == 0):
        raise ValueError(f"completed GT unknown mask disagrees with instance IDs: {path}")
    return instances.astype(np.int32, copy=False)


def adaptive_center_heatmap(
    instance_map: np.ndarray,
    sigma_scale: float = 0.12,
    min_sigma: float = 2.0,
    max_sigma: float = 8.0,
) -> np.ndarray:
    """Render one adaptive Gaussian peak at the EDT maximum of each instance."""
    height, width = instance_map.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    for instance_id in np.unique(instance_map):
        if int(instance_id) == 0:
            continue
        mask = instance_map == int(instance_id)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        local_mask = mask[y0:y1, x0:x1].astype(np.uint8)
        distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
        local_y, local_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        center_y, center_x = y0 + int(local_y), x0 + int(local_x)
        radius = math.sqrt(float(mask.sum()) / math.pi)
        sigma = float(np.clip(sigma_scale * radius, min_sigma, max_sigma))
        support = max(1, int(math.ceil(3.0 * sigma)))
        yy0, yy1 = max(0, center_y - support), min(height, center_y + support + 1)
        xx0, xx1 = max(0, center_x - support), min(width, center_x + support + 1)
        yy, xx = np.mgrid[yy0:yy1, xx0:xx1]
        gaussian = np.exp(
            -((yy - center_y) ** 2 + (xx - center_x) ** 2) / (2.0 * sigma ** 2)
        ).astype(np.float32)
        heatmap[yy0:yy1, xx0:xx1] = np.maximum(
            heatmap[yy0:yy1, xx0:xx1], gaussian
        )
        heatmap[center_y, center_x] = 1.0
    return heatmap


class OffsetGeometryDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        sample_names: Sequence[str] | None = None,
        image_size: int = 1024,
        output_grid: int = 512,
        center_sigma_scale: float = 0.12,
        center_min_sigma: float = 2.0,
        center_max_sigma: float = 8.0,
        cache_in_memory: bool = False,
        completed_gt_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.completed_gt_dir = (
            Path(completed_gt_dir) if completed_gt_dir is not None else None
        )
        self.image_size = int(image_size)
        self.output_grid = int(output_grid)
        self.center_sigma_scale = float(center_sigma_scale)
        self.center_min_sigma = float(center_min_sigma)
        self.center_max_sigma = float(center_max_sigma)
        self.cache_in_memory = bool(cache_in_memory)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        requested = {
            Path(name).stem for name in (sample_names or [])
        }
        self.samples = [
            path for path in sorted(self.data_dir.iterdir())
            if path.suffix.lower() in extensions
            and path.with_suffix(".json").is_file()
            and (not requested or path.stem in requested)
        ]
        if requested:
            found = {path.stem for path in self.samples}
            missing = sorted(requested - found)
            if missing:
                raise FileNotFoundError(f"requested labeled samples missing: {missing}")
        if not self.samples:
            raise ValueError(f"no labeled samples in {self.data_dir}")
        if self.completed_gt_dir is not None:
            missing = [
                path.stem for path in self.samples
                if not (self.completed_gt_dir / f"{path.stem}_gt.npz").is_file()
            ]
            if missing:
                raise FileNotFoundError(f"requested completed GT missing: {missing}")
        self._cached_samples = None
        if self.cache_in_memory:
            self._cached_samples = [
                self._load_sample(index) for index in range(len(self.samples))
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self._cached_samples is not None:
            return self._cached_samples[int(index)]
        return self._load_sample(index)

    def _load_sample(self, index):
        image_path = self.samples[int(index)]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lb, _, _, _ = letterbox(image_rgb, self.image_size)
        if self.completed_gt_dir is None:
            gt_instances, _, audit = load_labelme_instances(
                image_path.with_suffix(".json"), image.shape[:2]
            )
            uncovered_pixels = int(audit["uncovered_pixels"])
        else:
            # 不再拼接旧 LabelMe 的几何目标；所有几何派生量来自同一补缝图。
            gt_instances = load_completed_instance_map(
                self.completed_gt_dir / f"{image_path.stem}_gt.npz", image.shape[:2]
            )
            uncovered_pixels = int(np.count_nonzero(gt_instances == 0))
        geometry_map, valid_content, metadata = letterbox_instance_geometry(
            gt_instances, input_size=self.image_size, output_grid=self.output_grid
        )
        foreground = geometry_map > 0
        center = adaptive_center_heatmap(
            geometry_map,
            sigma_scale=self.center_sigma_scale,
            min_sigma=self.center_min_sigma,
            max_sigma=self.center_max_sigma,
        )
        offsets = build_center_offset_target(geometry_map) / float(self.output_grid)
        return {
            "image": torch.from_numpy(image_lb).permute(2, 0, 1).float() / 255.0,
            "center_target": torch.from_numpy(center).unsqueeze(0),
            "offset_target": torch.from_numpy(offsets),
            "foreground": torch.from_numpy(foreground).unsqueeze(0),
            "valid_content": torch.from_numpy(valid_content).unsqueeze(0),
            "instance_map": torch.from_numpy(geometry_map.astype(np.int64)),
            # 旧人工缝隙可供历史实验使用；补缝后的 unknown 永不作为负连接。
            "uncovered_boundary_source": torch.tensor(self.completed_gt_dir is None),
            "image_name": image_path.name,
            "instance_count": int(len(np.unique(geometry_map)) - 1),
            "uncovered_pixels": uncovered_pixels,
            "content_shape": torch.tensor(
                [metadata.content_height, metadata.content_width], dtype=torch.int32
            ),
            "input_content_shape": torch.tensor(
                [metadata.resized_height, metadata.resized_width], dtype=torch.int32
            ),
        }
