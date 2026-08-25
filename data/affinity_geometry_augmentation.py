# -*- coding: utf-8 -*-
"""Geometry-safe augmentation and random patches for local affinities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.offset_geometry_augmentation import augment_cached_geometry_sample


def crop_affinity_sample(
    sample: Dict,
    crop_scale: float,
    top_fraction: float,
    left_fraction: float,
):
    """Crop valid content and resize it back to the model's fixed grids."""
    grid_size = int(sample["instance_map"].shape[-1])
    image_size = int(sample["image"].shape[-1])
    content_height, content_width = map(int, sample["content_shape"].tolist())
    input_height, input_width = map(int, sample["input_content_shape"].tolist())
    crop_size = max(
        32,
        min(
            min(content_height, content_width),
            int(round(min(content_height, content_width) * float(crop_scale))),
        ),
    )
    max_top = max(0, content_height - crop_size)
    max_left = max(0, content_width - crop_size)
    top = int(round(np.clip(top_fraction, 0.0, 1.0) * max_top))
    left = int(round(np.clip(left_fraction, 0.0, 1.0) * max_left))
    bottom, right = top + crop_size, left + crop_size

    image_top = int(round(top * input_height / max(1, content_height)))
    image_left = int(round(left * input_width / max(1, content_width)))
    image_bottom = int(round(bottom * input_height / max(1, content_height)))
    image_right = int(round(right * input_width / max(1, content_width)))
    image_bottom = max(image_top + 1, min(input_height, image_bottom))
    image_right = max(image_left + 1, min(input_width, image_right))
    image_crop = sample["image"][
        :, image_top:image_bottom, image_left:image_right
    ].unsqueeze(0)
    image = F.interpolate(
        image_crop, size=(image_size, image_size), mode="bilinear",
        align_corners=False,
    )[0]
    instance_crop = sample["instance_map"][top:bottom, left:right]
    instances = F.interpolate(
        instance_crop[None, None].float(), size=(grid_size, grid_size),
        mode="nearest",
    )[0, 0].to(sample["instance_map"].dtype)
    foreground = (instances > 0).unsqueeze(0)
    valid = torch.ones((1, grid_size, grid_size), dtype=torch.bool)
    return {
        **sample,
        "image": image.contiguous(),
        "instance_map": instances.contiguous(),
        "foreground": foreground.contiguous(),
        "valid_content": valid,
        "center_target": torch.zeros((1, grid_size, grid_size), dtype=torch.float32),
        "offset_target": torch.zeros((2, grid_size, grid_size), dtype=torch.float32),
        "content_shape": torch.tensor([grid_size, grid_size], dtype=torch.int32),
        "input_content_shape": torch.tensor([image_size, image_size], dtype=torch.int32),
    }


class AffinityGeometryAugmentedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, config: Dict | None = None):
        self.base_dataset = base_dataset
        self.config = config or {}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        cfg = self.config
        hflip = bool(cfg.get("horizontal_flip", True) and np.random.rand() < 0.5)
        vflip = bool(cfg.get("vertical_flip", True) and np.random.rand() < 0.5)
        rotation_k = int(np.random.randint(0, 4)) if cfg.get("rotation90", True) else 0
        brightness_delta = float(cfg.get("brightness", 0.12))
        contrast_delta = float(cfg.get("contrast", 0.12))
        gamma_delta = float(cfg.get("gamma", 0.12))
        sample = augment_cached_geometry_sample(
            self.base_dataset[int(index)],
            hflip=hflip,
            vflip=vflip,
            rotation_k=rotation_k,
            brightness=np.random.uniform(1.0 - brightness_delta, 1.0 + brightness_delta),
            contrast=np.random.uniform(1.0 - contrast_delta, 1.0 + contrast_delta),
            gamma=np.random.uniform(1.0 - gamma_delta, 1.0 + gamma_delta),
            noise_std=(
                float(cfg.get("noise_std", 0.01))
                if np.random.rand() < float(cfg.get("noise_probability", 0.25))
                else 0.0
            ),
            blur=np.random.rand() < float(cfg.get("blur_probability", 0.10)),
        )
        if np.random.rand() < float(cfg.get("crop_probability", 0.5)):
            scale = np.random.uniform(
                float(cfg.get("crop_scale_min", 0.65)),
                float(cfg.get("crop_scale_max", 0.90)),
            )
            sample = crop_affinity_sample(
                sample, scale, np.random.rand(), np.random.rand()
            )
        return sample
