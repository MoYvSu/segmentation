# -*- coding: utf-8 -*-
"""Datasets for GDA masked cross-view reconstruction."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_manifest(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"holdout manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return {
            os.path.splitext(os.path.basename(line.strip()))[0]
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }


def list_images(data_dir: str) -> List[str]:
    images: List[str] = []
    for extension in VALID_EXTENSIONS:
        images.extend(glob.glob(os.path.join(data_dir, f"*{extension}")))
        images.extend(glob.glob(os.path.join(data_dir, f"*{extension.upper()}")))
    return sorted(set(images))


class MaskedMetallographyDataset(Dataset):
    """Aligned clean/masked views from the allowed unlabeled image set."""

    def __init__(
        self,
        data_dir: str,
        holdout_manifest: str,
        *,
        split: str = "train",
        crop_size: int = 512,
        mask_ratio_range: Sequence[float] = (0.50, 0.65),
        mask_patch_range: Sequence[int] = (16, 64),
        physical_aug_probability: float = 0.80,
    ):
        super().__init__()
        if split not in {"train", "holdout"}:
            raise ValueError("split must be 'train' or 'holdout'")
        self.split = split
        self.crop_size = int(crop_size)
        self.mask_ratio_range = tuple(float(v) for v in mask_ratio_range)
        self.mask_patch_range = tuple(int(v) for v in mask_patch_range)
        self.physical_aug_probability = float(physical_aug_probability)
        holdout = read_manifest(holdout_manifest)
        if not holdout:
            raise ValueError("holdout manifest must contain at least one image")

        samples = list_images(data_dir)
        if split == "train":
            self.samples = [
                path
                for path in samples
                if Path(path).stem not in holdout
            ]
        else:
            self.samples = [
                path
                for path in samples
                if Path(path).stem in holdout
            ]
        if not self.samples:
            raise ValueError(f"no {split} images selected from {data_dir}")
        if split == "holdout" and len(self.samples) != len(holdout):
            missing = sorted(holdout - {Path(path).stem for path in self.samples})
            raise FileNotFoundError(f"holdout images missing from {data_dir}: {missing}")

    def __len__(self):
        return len(self.samples)

    def _crop(self, image: np.ndarray) -> np.ndarray:
        size = self.crop_size
        height, width = image.shape[:2]
        if min(height, width) < size:
            scale = size / min(height, width)
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
            height, width = image.shape[:2]
        if self.split == "train":
            y0 = np.random.randint(0, height - size + 1)
            x0 = np.random.randint(0, width - size + 1)
        else:
            y0 = (height - size) // 2
            x0 = (width - size) // 2
        return np.ascontiguousarray(image[y0:y0 + size, x0:x0 + size])

    def _geometry(self, image: np.ndarray) -> np.ndarray:
        if self.split != "train":
            return image
        if np.random.rand() < 0.5:
            image = image[:, ::-1]
        if np.random.rand() < 0.5:
            image = image[::-1, :]
        image = np.rot90(image, int(np.random.randint(0, 4)))
        return np.ascontiguousarray(image)

    def _physical_view(self, clean: np.ndarray) -> np.ndarray:
        if self.split != "train" or np.random.rand() >= self.physical_aug_probability:
            return clean.copy()
        image = clean.astype(np.float32) / 255.0
        # Exposure/gamma and restrained channel gains model microscope variation.
        gamma = float(np.random.uniform(0.78, 1.28))
        image = np.power(np.clip(image, 0.0, 1.0), gamma)
        gains = np.random.uniform(0.94, 1.06, size=(1, 1, 3)).astype(np.float32)
        image = np.clip(image * gains, 0.0, 1.0)
        if np.random.rand() < 0.5:
            sigma = float(np.random.uniform(0.3, 1.6))
            image = cv2.GaussianBlur(image, (7, 7), sigma)
        if np.random.rand() < 0.35:
            small = cv2.resize(image, (3, 3), interpolation=cv2.INTER_AREA)
            field = cv2.resize(
                small,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            field = field - field.mean(axis=(0, 1), keepdims=True)
            image = np.clip(image + np.random.uniform(-0.12, 0.12) * field, 0.0, 1.0)
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _mask(self) -> np.ndarray:
        size = self.crop_size
        target_ratio = float(np.random.uniform(*self.mask_ratio_range))
        target_pixels = int(size * size * target_ratio)
        mask = np.zeros((size, size), dtype=np.uint8)
        attempts = 0
        while int(mask.sum()) < target_pixels and attempts < 4096:
            patch = int(np.random.randint(self.mask_patch_range[0], self.mask_patch_range[1] + 1))
            height = int(np.random.randint(max(8, patch // 2), patch + 1))
            width = int(np.random.randint(max(8, patch // 2), patch + 1))
            y0 = int(np.random.randint(0, max(1, size - height + 1)))
            x0 = int(np.random.randint(0, max(1, size - width + 1)))
            mask[y0:y0 + height, x0:x0 + width] = 1
            attempts += 1
        return mask

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        clean = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        clean = self._geometry(self._crop(clean))
        input_view = self._physical_view(clean)
        mask = self._mask()
        mean_color = input_view.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        masked = input_view.copy()
        masked[mask > 0] = mean_color

        return {
            "input": torch.from_numpy(masked).permute(2, 0, 1).float() / 255.0,
            "target": torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0,
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "image_path": path,
        }
