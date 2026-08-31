# -*- coding: utf-8 -*-
"""Aligned manual supervision for direct semantic + affinity training."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox, letterbox_mask
from utils.instance_metrics import load_labelme_instances
from utils.offset_letterbox import letterbox_instance_geometry


def _spatial_transform(tensor: torch.Tensor, hflip: bool, vflip: bool, k: int):
    result = tensor
    if hflip:
        result = torch.flip(result, dims=(-1,))
    if vflip:
        result = torch.flip(result, dims=(-2,))
    if int(k) % 4:
        result = torch.rot90(result, int(k) % 4, dims=(-2, -1))
    return result.contiguous()


class DirectDualHeadDataset(Dataset):
    """Return one image with aligned class and instance-pair supervision.

    Semantic targets keep the established purified GT. Instance identifiers
    come directly from the same LabelMe file and are rasterized at both the
    semantic image grid and the affinity grid. Spatial augmentation is sampled
    once and applied to every target, so the two heads never see mismatched
    geometry.
    """

    def __init__(
        self,
        data_dir: str | Path,
        gt_dir: str | Path,
        *,
        sample_names: Sequence[str] | None = None,
        image_size: int = 1024,
        affinity_grid: int = 512,
        augment: bool = False,
        augmentation: dict | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.gt_dir = Path(gt_dir)
        self.image_size = int(image_size)
        self.affinity_grid = int(affinity_grid)
        self.augment = bool(augment)
        self.augmentation = augmentation or {}
        requested = {Path(name).stem for name in (sample_names or [])}
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.samples = []
        for image_path in sorted(self.data_dir.iterdir()):
            if image_path.suffix.lower() not in extensions:
                continue
            gt_path = self.gt_dir / f"{image_path.stem}_gt.npz"
            if not image_path.with_suffix(".json").is_file() or not gt_path.is_file():
                continue
            if requested and image_path.stem not in requested:
                continue
            self.samples.append((image_path, gt_path))
        if requested:
            found = {path.stem for path, _ in self.samples}
            missing = sorted(requested - found)
            if missing:
                raise FileNotFoundError(
                    f"requested direct dual-head samples missing: {missing}"
                )
        if not self.samples:
            raise ValueError(f"no aligned labeled samples in {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, gt_path = self.samples[int(index)]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_shape = image_rgb.shape[:2]

        with np.load(gt_path) as payload:
            semantic = np.asarray(payload["semantic"], dtype=np.uint8)
            boundary_key = (
                "boundary_soft" if "boundary_soft" in payload.files else "boundary"
            )
            semantic_boundary = np.asarray(payload[boundary_key], dtype=np.float32)
        if semantic.shape != original_shape or semantic_boundary.shape != original_shape:
            raise ValueError(
                f"purified GT shape mismatch for {image_path.name}: "
                f"semantic={semantic.shape} boundary={semantic_boundary.shape} "
                f"image={original_shape}"
            )

        instance_map, _, audit = load_labelme_instances(
            image_path.with_suffix(".json"), original_shape
        )
        image_lb, _, _, _ = letterbox(image_rgb, self.image_size)
        semantic_lb, _, _, _ = letterbox_mask(semantic, self.image_size)
        boundary_lb, _, _, _ = letterbox_mask(
            semantic_boundary, self.image_size
        )
        semantic_instances, semantic_valid, _ = letterbox_instance_geometry(
            instance_map,
            input_size=self.image_size,
            output_grid=self.image_size,
        )
        affinity_instances, affinity_valid, affinity_meta = (
            letterbox_instance_geometry(
                instance_map,
                input_size=self.image_size,
                output_grid=self.affinity_grid,
            )
        )

        sample = {
            "image": torch.from_numpy(image_lb).permute(2, 0, 1).float() / 255.0,
            "semantic_target": torch.from_numpy(semantic_lb).float().unsqueeze(0),
            "semantic_boundary": torch.from_numpy(boundary_lb).float().unsqueeze(0),
            "semantic_instance_map": torch.from_numpy(
                semantic_instances.astype(np.int64)
            ),
            "semantic_valid_content": torch.from_numpy(semantic_valid).unsqueeze(0),
            "affinity_instance_map": torch.from_numpy(
                affinity_instances.astype(np.int64)
            ),
            "affinity_valid_content": torch.from_numpy(affinity_valid).unsqueeze(0),
            "uncovered_boundary_source": torch.tensor(True),
            "image_name": image_path.name,
            "image_path": str(image_path),
            "uncovered_pixels": int(audit["uncovered_pixels"]),
            "content_shape": torch.tensor(
                [affinity_meta.content_height, affinity_meta.content_width],
                dtype=torch.int32,
            ),
        }
        if self.augment:
            sample = self._augment(sample)
        return sample

    def _augment(self, sample: dict) -> dict:
        cfg = self.augmentation
        hflip = bool(cfg.get("horizontal_flip", True) and np.random.rand() < 0.5)
        vflip = bool(cfg.get("vertical_flip", True) and np.random.rand() < 0.5)
        rotation_k = (
            int(np.random.randint(0, 4)) if cfg.get("rotation90", True) else 0
        )
        spatial_keys = (
            "image",
            "semantic_target",
            "semantic_boundary",
            "semantic_instance_map",
            "semantic_valid_content",
            "affinity_instance_map",
            "affinity_valid_content",
        )
        result = dict(sample)
        for key in spatial_keys:
            result[key] = _spatial_transform(
                result[key], hflip, vflip, rotation_k
            )

        image = result["image"]
        brightness_delta = float(cfg.get("brightness", 0.08))
        contrast_delta = float(cfg.get("contrast", 0.08))
        gamma_delta = float(cfg.get("gamma", 0.06))
        brightness = float(
            np.random.uniform(1.0 - brightness_delta, 1.0 + brightness_delta)
        )
        contrast = float(
            np.random.uniform(1.0 - contrast_delta, 1.0 + contrast_delta)
        )
        gamma = float(np.random.uniform(1.0 - gamma_delta, 1.0 + gamma_delta))
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = ((image - mean) * contrast + mean) * brightness
        image = image.clamp(0.0, 1.0).pow(gamma)
        if np.random.rand() < float(cfg.get("noise_probability", 0.0)):
            image = image + torch.randn_like(image) * float(cfg.get("noise_std", 0.01))
        if np.random.rand() < float(cfg.get("blur_probability", 0.0)):
            image = torch.nn.functional.avg_pool2d(
                image.unsqueeze(0), 3, stride=1, padding=1
            )[0]
        result["image"] = image.clamp(0.0, 1.0).contiguous()
        return result
