# -*- coding: utf-8 -*-
"""Aligned manual supervision for direct semantic + affinity training."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset

from data.dataset import letterbox, letterbox_mask
from utils.instance_metrics import load_class_map, load_labelme_instances
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


def semantic_from_instance_classes(
    instance_map: np.ndarray,
    class_map: Mapping[int, int],
):
    """从同一份实例图和类别 LUT 派生 semantic/valid，避免双重栅格化。"""
    labels = np.asarray(instance_map)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("instance_map must be a 2-D integer array")
    if int(labels.min()) < 0 or int(labels.max()) > 65535:
        raise ValueError("instance ids must stay within [0, 65535]")
    max_id = int(labels.max())
    lut = np.full(max_id + 1, -1, dtype=np.int8)
    if max_id >= 0:
        lut[0] = 0
    for instance_id, class_id in class_map.items():
        instance_id = int(instance_id)
        class_id = int(class_id)
        if not 1 <= instance_id <= max_id:
            raise ValueError(f"invalid or stale instance id in class map: {instance_id}")
        if class_id not in (0, 1):
            raise ValueError(f"invalid semantic class for {instance_id}: {class_id}")
        lut[instance_id] = class_id
    present = np.unique(labels[labels > 0]).astype(np.int64, copy=False)
    declared = {int(value) for value in class_map}
    if declared != {int(value) for value in present}:
        raise ValueError("class map ids must exactly match present instance ids")
    missing = [int(value) for value in present if lut[int(value)] < 0]
    if missing:
        raise ValueError(f"instance ids missing from class map: {missing}")
    valid = labels > 0
    semantic = np.zeros(labels.shape, dtype=np.uint8)
    semantic[valid] = lut[labels[valid]].astype(np.uint8, copy=False)
    return semantic, valid


def instance_boundary_target(instance_map: np.ndarray, dilation: int = 2):
    """从 canonical 实例相邻关系派生仅供 semantic core 使用的边界。"""
    if int(dilation) < 0:
        raise ValueError("boundary dilation must be non-negative")
    boundary = find_boundaries(
        np.asarray(instance_map), mode="inner", connectivity=1
    ).astype(np.uint8)
    if int(dilation) > 0:
        radius = int(dilation)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        boundary = cv2.dilate(boundary, kernel)
    return boundary.astype(np.float32, copy=False)


def load_manual_target_archive(path: str | Path):
    """读取并严格验证一个 ``manual_target_v2`` 派生 GT。"""
    target_path = Path(path)
    with np.load(target_path) as payload:
        required = {
            "instance_map", "original_covered", "filled", "residual_unknown"
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"manual target {target_path} missing keys: {missing}")
        raw_instances = np.asarray(payload["instance_map"])
        if not np.issubdtype(raw_instances.dtype, np.integer):
            raise ValueError(f"manual target ids must be integer: {target_path}")
        if int(raw_instances.min()) < 0 or int(raw_instances.max()) > 65535:
            raise ValueError(f"manual target ids outside uint16 range: {target_path}")
        instance_map = raw_instances.astype(np.int32, copy=False)
        provenance = {}
        for key in ("original_covered", "filled", "residual_unknown"):
            raw_mask = np.asarray(payload[key])
            if not (
                np.issubdtype(raw_mask.dtype, np.bool_)
                or np.issubdtype(raw_mask.dtype, np.integer)
            ):
                raise ValueError(
                    f"manual target {key} must be binary: {target_path}"
                )
            if not np.all((raw_mask == 0) | (raw_mask == 1)):
                raise ValueError(
                    f"manual target {key} is not binary: {target_path}"
                )
            provenance[key] = raw_mask.astype(bool, copy=False)
    class_path = target_path.with_name(f"{target_path.stem.removesuffix('_gt')}_class.json")
    if not class_path.is_file():
        raise FileNotFoundError(class_path)
    class_map = load_class_map(class_path)
    semantic, valid = semantic_from_instance_classes(instance_map, class_map)
    original_covered = provenance["original_covered"]
    filled = provenance["filled"]
    residual_unknown = provenance["residual_unknown"]
    if np.any(original_covered & filled):
        raise ValueError(f"manual/filled provenance overlaps in {target_path}")
    if not np.array_equal(valid, original_covered | filled):
        raise ValueError(f"manual target valid mask mismatch in {target_path}")
    if not np.array_equal(residual_unknown, ~valid):
        raise ValueError(f"manual target unknown mask mismatch in {target_path}")
    return instance_map, class_map, semantic, valid, provenance


def load_direct_evaluation_target(
    image_path: str | Path,
    image_shape,
    manual_target_dir: str | Path | None = None,
):
    """加载 direct 代理评估 GT，并显式返回可评分区域。"""
    image_path = Path(image_path)
    expected_shape = tuple(map(int, image_shape))
    if manual_target_dir is None:
        instance_map, class_map, audit = load_labelme_instances(
            image_path.with_suffix(".json"), expected_shape
        )
        return (
            instance_map,
            class_map,
            np.ones(expected_shape, dtype=bool),
            {"source": "labelme", **audit},
        )

    target_path = Path(manual_target_dir) / f"{image_path.stem}_gt.npz"
    instance_map, class_map, _, valid, provenance = load_manual_target_archive(
        target_path
    )
    if instance_map.shape != expected_shape:
        raise ValueError(
            f"manual evaluation target shape mismatch for {image_path.name}: "
            f"{instance_map.shape} vs {expected_shape}"
        )
    return (
        instance_map,
        class_map,
        valid,
        {
            "source": "manual_target_v2",
            "original_covered_pixels": int(provenance["original_covered"].sum()),
            "filled_pixels": int(provenance["filled"].sum()),
            "ignored_unknown_pixels": int(provenance["residual_unknown"].sum()),
        },
    )


def mask_prediction_to_evaluation_domain(
    prediction: np.ndarray,
    valid: np.ndarray,
):
    """让 unknown 对实例 IoU、计数和面积统计都保持 ignore。"""
    prediction = np.asarray(prediction)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != valid.shape:
        raise ValueError(
            f"prediction/valid shape mismatch: {prediction.shape} vs {valid.shape}"
        )
    if bool(valid.all()):
        return prediction
    masked = prediction.copy()
    masked[~valid] = 0
    return masked


class DirectDualHeadDataset(Dataset):
    """Return one image with aligned class and instance-pair supervision.

    Legacy configs keep the established purified semantic GT. ``manual_target``
    configs instead derive both semantic and affinity supervision from one
    canonical completed instance map, with remaining id-0 pixels ignored.
    Spatial augmentation is sampled once and applied to every target.
    """

    def __init__(
        self,
        data_dir: str | Path,
        gt_dir: str | Path | None,
        *,
        sample_names: Sequence[str] | None = None,
        image_size: int = 1024,
        affinity_grid: int = 512,
        augment: bool = False,
        augmentation: dict | None = None,
        manual_target_dir: str | Path | None = None,
        manual_target_boundary_dilation: int = 2,
        native_crop: dict | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.gt_dir = Path(gt_dir) if gt_dir is not None else None
        self.manual_target_dir = (
            Path(manual_target_dir) if manual_target_dir is not None else None
        )
        self.manual_target_boundary_dilation = int(manual_target_boundary_dilation)
        self.image_size = int(image_size)
        self.affinity_grid = int(affinity_grid)
        self.augment = bool(augment)
        self.augmentation = augmentation or {}
        crop_cfg = native_crop or {}
        self.native_crop_enabled = bool(crop_cfg.get("enabled", False))
        self.native_crop_probability = float(crop_cfg.get("probability", 0.5))
        crop_size = float(crop_cfg.get("size", 1024))
        if not np.isfinite(self.native_crop_probability) or not (
            0.0 <= self.native_crop_probability <= 1.0
        ):
            raise ValueError("native_crop probability must be finite and in [0, 1]")
        if not np.isfinite(crop_size) or crop_size < 1 or not crop_size.is_integer():
            raise ValueError("native_crop size must be a finite positive integer")
        self.native_crop_size = int(crop_size)
        if self.native_crop_enabled and self.manual_target_dir is None:
            raise ValueError("native_crop requires manual_target_dir")
        requested = {Path(name).stem for name in (sample_names or [])}
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.samples = []
        for image_path in sorted(self.data_dir.iterdir()):
            if image_path.suffix.lower() not in extensions:
                continue
            gt_path = (
                self.manual_target_dir / f"{image_path.stem}_gt.npz"
                if self.manual_target_dir is not None
                else self.gt_dir / f"{image_path.stem}_gt.npz"
            )
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

        if self.manual_target_dir is not None:
            (
                instance_map,
                _,
                semantic,
                semantic_annotation_valid,
                provenance,
            ) = load_manual_target_archive(gt_path)
            original_covered = provenance["original_covered"]
            filled = provenance["filled"]
            residual_unknown = provenance["residual_unknown"]
            if any(
                value.shape != original_shape
                for value in (
                    instance_map, original_covered, filled, residual_unknown
                )
            ):
                raise ValueError(
                    f"manual target shape mismatch for {image_path.name}: "
                    f"instance={instance_map.shape} image={original_shape}"
                )
            semantic_boundary = instance_boundary_target(
                instance_map, dilation=self.manual_target_boundary_dilation
            )
            audit = {
                "uncovered_pixels": int(residual_unknown.sum()),
                "filled_pixels": int(filled.sum()),
            }
        else:
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
            semantic_annotation_valid = np.ones(original_shape, dtype=bool)

        height, width = original_shape
        sample_bbox = (0, 0, width, height)
        is_native_crop = False
        # 关闭或验证时不额外取随机数，保持既有整图数据与增强序列。
        if (
            self.native_crop_enabled
            and self.augment
            and self.native_crop_probability > 0
            and np.random.rand() < self.native_crop_probability
        ):
            crop_size = min(self.native_crop_size, height, width)
            y0 = int(np.random.randint(0, height - crop_size + 1))
            x0 = int(np.random.randint(0, width - crop_size + 1))
            window = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
            image_rgb = image_rgb[window]
            instance_map = instance_map[window]
            semantic = semantic[window]
            semantic_annotation_valid = semantic_annotation_valid[window]
            # 边界已从完整实例图生成；裁剪后不重新造边，保留窗口外的真实上下文。
            semantic_boundary = semantic_boundary[window]
            provenance = {key: value[window] for key, value in provenance.items()}
            audit["uncovered_pixels"] = int(provenance["residual_unknown"].sum())
            audit["filled_pixels"] = int(provenance["filled"].sum())
            sample_bbox = (x0, y0, x0 + crop_size, y0 + crop_size)
            is_native_crop = True

        image_lb, _, _, _ = letterbox(image_rgb, self.image_size)
        semantic_lb, _, _, _ = letterbox_mask(semantic, self.image_size)
        boundary_lb, _, _, _ = letterbox_mask(
            semantic_boundary, self.image_size
        )
        semantic_instances, semantic_content_valid, _ = letterbox_instance_geometry(
            instance_map,
            input_size=self.image_size,
            output_grid=self.image_size,
        )
        if self.manual_target_dir is not None:
            semantic_valid_grid, _, _ = letterbox_instance_geometry(
                semantic_annotation_valid.astype(np.int32),
                input_size=self.image_size,
                output_grid=self.image_size,
            )
            semantic_supervision_valid = semantic_valid_grid > 0
        else:
            semantic_supervision_valid = semantic_content_valid
        affinity_instances, affinity_valid, affinity_meta = (
            letterbox_instance_geometry(
                instance_map,
                input_size=self.image_size,
                output_grid=self.affinity_grid,
            )
        )
        if self.manual_target_dir is not None:
            affinity_valid = affinity_valid & (affinity_instances > 0)

        sample = {
            "image": torch.from_numpy(image_lb).permute(2, 0, 1).float() / 255.0,
            "semantic_target": torch.from_numpy(semantic_lb).float().unsqueeze(0),
            "semantic_boundary": torch.from_numpy(boundary_lb).float().unsqueeze(0),
            "semantic_instance_map": torch.from_numpy(
                semantic_instances.astype(np.int64)
            ),
            "semantic_valid_content": torch.from_numpy(
                semantic_supervision_valid
            ).unsqueeze(0),
            "affinity_instance_map": torch.from_numpy(
                affinity_instances.astype(np.int64)
            ),
            "affinity_valid_content": torch.from_numpy(affinity_valid).unsqueeze(0),
            "uncovered_boundary_source": torch.tensor(
                self.manual_target_dir is None
            ),
            "image_name": image_path.name,
            "image_path": str(image_path),
            "uncovered_pixels": int(audit["uncovered_pixels"]),
            "filled_pixels": int(audit.get("filled_pixels", 0)),
            # 坐标为原图的 [x0,y0,x1,y1)，shape 均为 letterbox/翻转前的 H,W。
            "sample_bbox": torch.tensor(sample_bbox, dtype=torch.int32),
            "sample_shape": torch.tensor(image_rgb.shape[:2], dtype=torch.int32),
            "original_shape": torch.tensor(original_shape, dtype=torch.int32),
            "is_native_crop": torch.tensor(is_native_crop),
            "content_shape": torch.tensor(
                [affinity_meta.content_height, affinity_meta.content_width],
                dtype=torch.int32,
            ),
        }
        if self.manual_target_dir is not None:
            sample["original_covered_pixels"] = int(provenance["original_covered"].sum())
            sample["semantic_loss_weight"] = torch.from_numpy(
                semantic_supervision_valid.astype(np.float32)
            ).unsqueeze(0)
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
            "semantic_loss_weight",
            "semantic_instance_map",
            "semantic_valid_content",
            "affinity_instance_map",
            "affinity_valid_content",
        )
        result = dict(sample)
        for key in spatial_keys:
            if key not in result:
                continue
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
