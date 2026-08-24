# -*- coding: utf-8 -*-
"""Geometry-consistent augmentation for cached global center-offset samples."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _spatial_transform(tensor: torch.Tensor, hflip: bool, vflip: bool, k: int):
    result = tensor
    if hflip:
        result = torch.flip(result, dims=(-1,))
    if vflip:
        result = torch.flip(result, dims=(-2,))
    if k:
        result = torch.rot90(result, int(k), dims=(-2, -1))
    return result.contiguous()


def transform_offset_vectors(
    offsets: torch.Tensor, hflip: bool, vflip: bool, rotation_k: int
) -> torch.Tensor:
    """Apply spatial transforms and rotate ``(dy, dx)`` vector components."""
    result = _spatial_transform(offsets, hflip, vflip, rotation_k).clone()
    if hflip:
        result[1].neg_()
    if vflip:
        result[0].neg_()
    k = int(rotation_k) % 4
    dy, dx = result[0].clone(), result[1].clone()
    if k == 1:
        result[0], result[1] = -dx, dy
    elif k == 2:
        result[0], result[1] = -dy, -dx
    elif k == 3:
        result[0], result[1] = dx, -dy
    return result.contiguous()


def augment_cached_geometry_sample(
    sample: Dict,
    *,
    hflip: bool,
    vflip: bool,
    rotation_k: int,
    brightness: float = 1.0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    noise_std: float = 0.0,
    blur: bool = False,
) -> Dict:
    """Transform valid content, then rebuild reflect/zero letterbox padding."""
    grid_size = int(sample["instance_map"].shape[-1])
    image_size = int(sample["image"].shape[-1])
    content_height, content_width = map(int, sample["content_shape"].tolist())
    if "input_content_shape" in sample:
        image_content_height, image_content_width = map(
            int, sample["input_content_shape"].tolist()
        )
    else:
        image_content_height = int(round(content_height * image_size / grid_size))
        image_content_width = int(round(content_width * image_size / grid_size))
    image = sample["image"][:, :image_content_height, :image_content_width].clone()
    center = sample["center_target"][:, :content_height, :content_width].clone()
    offsets = sample["offset_target"][:, :content_height, :content_width].clone()
    foreground = sample["foreground"][:, :content_height, :content_width].clone()
    instances = sample["instance_map"][:content_height, :content_width].clone()

    k = int(rotation_k) % 4
    image = _spatial_transform(image, hflip, vflip, k)
    center = _spatial_transform(center, hflip, vflip, k)
    foreground = _spatial_transform(foreground, hflip, vflip, k)
    instances = _spatial_transform(instances, hflip, vflip, k)
    offsets = transform_offset_vectors(offsets, hflip, vflip, k)
    new_height, new_width = map(int, instances.shape)
    new_image_height, new_image_width = map(int, image.shape[-2:])
    image_pad_h = image_size - image.shape[-2]
    image_pad_w = image_size - image.shape[-1]
    if image_pad_h < 0 or image_pad_w < 0:
        raise ValueError("augmented image content exceeds letterbox size")
    image = F.pad(image, (0, image_pad_w, 0, image_pad_h), mode="reflect")

    pad_h, pad_w = grid_size - new_height, grid_size - new_width
    center = F.pad(center, (0, pad_w, 0, pad_h), value=0.0)
    offsets = F.pad(offsets, (0, pad_w, 0, pad_h), value=0.0)
    foreground = F.pad(foreground, (0, pad_w, 0, pad_h), value=False)
    instances = F.pad(instances, (0, pad_w, 0, pad_h), value=0)
    valid_content = torch.zeros((1, grid_size, grid_size), dtype=torch.bool)
    valid_content[:, :new_height, :new_width] = True

    mean = image.mean(dim=(-2, -1), keepdim=True)
    image = (image - mean) * float(contrast) + mean
    image = image * float(brightness)
    image = image.clamp(0.0, 1.0).pow(float(gamma))
    if noise_std > 0:
        image = image + torch.randn_like(image) * float(noise_std)
    if blur:
        image = F.avg_pool2d(image.unsqueeze(0), 3, stride=1, padding=1)[0]
    image = image.clamp(0.0, 1.0).contiguous()
    return {
        **sample,
        "image": image,
        "center_target": center.contiguous(),
        "offset_target": offsets.contiguous(),
        "foreground": foreground.contiguous(),
        "valid_content": valid_content,
        "instance_map": instances.contiguous(),
        "content_shape": torch.tensor([new_height, new_width], dtype=torch.int32),
        "input_content_shape": torch.tensor(
            [new_image_height, new_image_width], dtype=torch.int32
        ),
    }


class OffsetGeometryAugmentedDataset(Dataset):
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
        return augment_cached_geometry_sample(
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
