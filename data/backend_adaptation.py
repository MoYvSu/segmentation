# -*- coding: utf-8 -*-
"""后端配对适配：统一人工实例目标与逐轮、逐次可复现的在线退化。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox
from data.direct_dual_head_dataset import _spatial_transform
from data.rgb_restoration_dataset import PROFILES, _add_achromatic_noise, degrade_rgb, read_rgb
from data.rgb_spatial_blur import validate_spatial_blur_config
from data.semantic_targets import load_completed_semantic_source, semantic_targets_from_instances
from utils.offset_letterbox import letterbox_instance_geometry


class CanonicalBackendDataset(Dataset):
    """全人工图无划分；语义、实例、有效域及辅助边界只来自补缝实例图。

    LabelMe 文件仅确认人工源图身份，不读取旧多边形或 purified GT 作为目标。
    semantic_boundary 供语义 core 损失排除边缘，不训练额外的边界分支。
    """

    def __init__(self, data_dir, completed_gt_dir, image_size=1024, affinity_grid=512):
        self.data_dir = Path(data_dir)
        self.completed_gt_dir = Path(completed_gt_dir)
        self.image_size = int(image_size)
        self.affinity_grid = int(affinity_grid)
        if min(self.image_size, self.affinity_grid) < 1:
            raise ValueError("image_size and affinity_grid must be positive")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.samples = [path for path in sorted(self.data_dir.iterdir())
                        if path.suffix.lower() in extensions and path.with_suffix(".json").is_file()]
        if not self.samples:
            raise ValueError(f"no manual image/JSON pairs in {self.data_dir}")
        if len({path.stem for path in self.samples}) != len(self.samples):
            raise ValueError("manual image stems must be unique")
        for path in self.samples:
            for suffix in ("_gt.npz", "_class.json"):
                target_path = self.completed_gt_dir / (path.stem + suffix)
                if not target_path.is_file():
                    raise FileNotFoundError(f"incomplete canonical manual cohort: {target_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[int(index)]
        image = read_rgb(str(path))
        instances, classes = load_completed_semantic_source(
            self.completed_gt_dir / (path.stem + "_gt.npz"), image.shape[:2]
        )
        image_lb, _, _, _ = letterbox(image, self.image_size)
        semantic_ids, semantic_content, _ = letterbox_instance_geometry(
            instances, input_size=self.image_size, output_grid=self.image_size
        )
        affinity_ids, affinity_content, metadata = letterbox_instance_geometry(
            instances, input_size=self.image_size, output_grid=self.affinity_grid
        )
        semantic, boundary = semantic_targets_from_instances(semantic_ids, classes)
        affinity_tensor = torch.from_numpy(affinity_ids.astype(np.int64))
        affinity_valid = torch.from_numpy(affinity_content).unsqueeze(0)
        return {
            "image": torch.from_numpy(image_lb.copy()).permute(2, 0, 1).float() / 255.0,
            "instance_map": affinity_tensor,
            "foreground": (affinity_tensor > 0).unsqueeze(0),
            "valid_content": affinity_valid,
            "affinity_instance_map": affinity_tensor,
            "affinity_valid_content": affinity_valid,
            "semantic_target": torch.from_numpy(semantic).float().unsqueeze(0),
            "semantic_boundary": torch.from_numpy(boundary).float().unsqueeze(0),
            "semantic_instance_map": torch.from_numpy(semantic_ids.astype(np.int64)),
            "semantic_valid_content": torch.from_numpy(
                semantic_content & (semantic_ids > 0)
            ).unsqueeze(0),
            "uncovered_boundary_source": torch.tensor(False),
            "input_content_shape": torch.tensor(
                [metadata.resized_height, metadata.resized_width], dtype=torch.int32
            ),
            "content_shape": torch.tensor(
                [metadata.content_height, metadata.content_width], dtype=torch.int32
            ),
            "name": path.name,
            "image_name": path.name,
            "image_path": str(path),
        }


class PairedDegradationDataset(Dataset):
    """两组共享 seed/epoch/draw，D4 自身随机数不影响输入和标签。

    base_dataset 必须是未增强的完整 letterbox 样本，支持人工或 SAM2 几何流。
    每个源图的 repeats 次抽取有独立退化，不裁块、不改变标签外观。
    多进程 loader 应关闭 persistent_workers，使 set_epoch 在下一轮传入 worker。
    """

    _SPATIAL_KEYS = (
        "image", "instance_map", "foreground", "valid_content",
        "affinity_instance_map", "affinity_valid_content", "semantic_target",
        "semantic_boundary", "semantic_instance_map", "semantic_valid_content",
    )

    def __init__(self, base_dataset, degradation: dict, noise: dict, seed: int,
                 repeats: int = 1, augment=True):
        if not len(base_dataset):
            raise ValueError("base_dataset must not be empty")
        if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        self.base_dataset = base_dataset
        self.degradation = deepcopy(degradation)
        self.noise = deepcopy(noise or {"enabled": False})
        self.seed, self.epoch = int(seed), 0
        self.repeats, self.augment = int(repeats), bool(augment)
        self.probabilities = np.asarray(degradation["profile_probabilities"], dtype=np.float64)
        if (self.probabilities.shape != (4,) or not np.isfinite(self.probabilities).all()
                or np.any(self.probabilities < 0) or not np.isclose(self.probabilities.sum(), 1)
                or np.any(self.probabilities[2:] != 0)):
            raise ValueError("backend adaptation requires identity/blur probabilities summing to one")
        validate_spatial_blur_config(self.degradation.get("spatial_blur"))
        probability = self.noise.get("probability", 0.5)
        sigmas = np.asarray(self.noise.get("sigma", [0.0, 0.006]), dtype=np.float64)
        if not np.isscalar(probability) or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("noise.probability must be within [0,1]")
        if sigmas.shape != (2,) or not np.isfinite(sigmas).all() or not 0 <= sigmas[0] <= sigmas[1] <= 1:
            raise ValueError("noise.sigma requires 0 <= low <= high <= 1")

    def __len__(self):
        return len(self.base_dataset) * self.repeats

    def set_epoch(self, epoch: int):
        if int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = int(epoch)

    def _rng(self, source_index, draw_index, stream):
        return np.random.default_rng(np.random.SeedSequence(
            [self.seed, self.epoch, source_index, draw_index, stream]
        ))

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index, repeat_index = index % len(self.base_dataset), index // len(self.base_dataset)
        sample = dict(self.base_dataset[source_index])
        image = sample["image"].permute(1, 2, 0).numpy().astype(np.float32, copy=False)
        height, width = map(int, sample["input_content_shape"].tolist())
        if not 0 < height <= image.shape[0] or not 0 < width <= image.shape[1]:
            raise ValueError("invalid base input_content_shape")
        rng = self._rng(source_index, repeat_index, 0)
        profile = str(rng.choice(PROFILES, p=self.probabilities)) if self.augment else "identity"
        info = {}
        degraded = degrade_rgb(
            image[:height, :width], rng, profile, self.degradation,
            spatial_rng=self._rng(source_index, repeat_index, 6), spatial_info=info,
        )
        noise_sigma = 0.0
        if self.augment and profile == "blur" and self.noise.get("enabled", True):
            noise_rng = self._rng(source_index, repeat_index, 3)
            if noise_rng.random() < self.noise.get("probability", 0.5):
                noise_sigma = float(noise_rng.uniform(*self.noise.get("sigma", [0.0, 0.006])))
                degraded = _add_achromatic_noise(
                    degraded, np.ones((height, width), dtype=np.float32), noise_rng, noise_sigma
                )
        degraded = cv2.copyMakeBorder(
            degraded, 0, image.shape[0] - height, 0, image.shape[1] - width, cv2.BORDER_REFLECT
        )
        sample["image"] = torch.from_numpy(degraded).permute(2, 0, 1).contiguous()
        geometry_rng = self._rng(source_index, repeat_index, 1)
        hflip, vflip, turns = (False, False, 0)
        if self.augment:
            hflip, vflip, turns = bool(geometry_rng.integers(2)), bool(geometry_rng.integers(2)), int(geometry_rng.integers(4))
        for key in self._SPATIAL_KEYS:
            if key in sample:
                sample[key] = _spatial_transform(sample[key], hflip, vflip, turns)
        # 旋转后尺寸随 mask 交换；翻转后有效区可能不在左上，loss 必须读取 valid mask。
        if turns % 2:
            for key in ("input_content_shape", "content_shape"):
                sample[key] = sample[key].flip(0)
        sample.update(
            profile=profile, is_spatial=bool(info["selected"]), noise_sigma=noise_sigma,
            base_sigma=float(info["base_sigma"]), draw_index=index, source_index=source_index,
            horizontal_flip=hflip, vertical_flip=vflip, rotation_k=turns,
        )
        sample.setdefault("name", sample.get("image_name", str(source_index)))
        return sample
