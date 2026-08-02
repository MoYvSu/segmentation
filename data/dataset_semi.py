# -*- coding: utf-8 -*-
"""
半监督数据集模块（边界预测版本）
================================
为第二阶段半监督 Mean Teacher 提供双流数据加载：
- 有标签流：复用 BoundaryDataset（图像 + 净化 GT + EDT 权重）
- 无标签流：弱/强两路分叉增强
  - img_weak: 仅 letterbox，无增强（教师预测源）
  - img_strong: 弱图 + 强空间增强（旋转/翻转）+ 外观增强（高斯模糊/亮度对比度）
  - patch_mask: 随机 Patch Masking（30% 像素遮挡），强制网络插值缺失晶界

非侵入式设计：不修改 data/dataset.py，仅复用其工具函数。
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox, random_crop, split_train_val_indices


# =============================================================================
# 有标签数据集（复用第一阶段 BoundaryDataset 逻辑）
# =============================================================================
class LabeledDataset(Dataset):
    """
    有标注数据集：输出原图 + 净化 GT + EDT 权重。
    复用 BoundaryDataset 的 .npz 加载逻辑，并使用相同的 train/val 划分
    （同 seed、同 train_ratio），确保有标签训练流与验证流不重叠，
    避免验证集泄露。
    """

    def __init__(
        self,
        data_dir: str,
        gt_dir: str,
        image_size: int = 1024,
        crop_size: int = 0,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        boundary_scale_factor: float = 10.0,
        boundary_weight_floor: float = 1.0,
        boundary_weight_ceil: float = 4.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.augment = augment
        self.augment_config = augment_config or {}
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.boundary_scale_factor = boundary_scale_factor
        self.boundary_weight_floor = boundary_weight_floor
        self.boundary_weight_ceil = boundary_weight_ceil

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.samples: List[Tuple[str, str]] = []
        for ext in valid_exts:
            for img_path in glob.glob(os.path.join(data_dir, f"*{ext}")):
                json_path = os.path.splitext(img_path)[0] + ".json"
                basename = os.path.splitext(os.path.basename(img_path))[0]
                gt_path = os.path.join(gt_dir, f"{basename}_gt.npz")
                if os.path.exists(json_path) and os.path.exists(gt_path):
                    self.samples.append((img_path, gt_path))
        self.samples.sort()
        # 与 BoundaryDataset 使用同一划分函数（同 seed / train_ratio）
        selected = split_train_val_indices(len(self.samples), train_ratio, seed, split)
        self.samples = [self.samples[i] for i in selected]
        print(f"[LabeledDataset] {len(self.samples)} samples ({split}) from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, gt_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        gt_data = np.load(gt_path)
        semantic = gt_data["semantic"]
        boundary = gt_data["boundary"]

        from data.dataset import letterbox_mask, compute_boundary_weight
        image_lb, scale, pad_h, pad_w = letterbox(image, self.image_size)
        semantic_lb, _, _, _ = letterbox_mask(semantic, self.image_size)
        boundary_lb, _, _, _ = letterbox_mask(boundary, self.image_size)
        weight_lb = compute_boundary_weight(
            boundary_lb,
            scale_factor=self.boundary_scale_factor,
            weight_floor=self.boundary_weight_floor,
            weight_ceil=self.boundary_weight_ceil,
        )

        if self.augment and self.crop_size > 0:
            image_lb, semantic_lb, boundary_lb, weight_lb = random_crop(
                image_lb, semantic_lb, boundary_lb, weight_lb, self.crop_size
            )

        if self.augment:
            image_lb, semantic_lb, boundary_lb, weight_lb = self._augment(
                image_lb, semantic_lb, boundary_lb, weight_lb
            )

        image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1) / 255.0
        semantic_tensor = torch.from_numpy(semantic_lb).float().unsqueeze(0)
        boundary_tensor = torch.from_numpy(boundary_lb).float().unsqueeze(0)
        target_tensor = torch.cat([semantic_tensor, boundary_tensor], dim=0)
        weight_tensor = torch.from_numpy(weight_lb).float().unsqueeze(0)

        return {
            "image": image_tensor,
            "target": target_tensor,
            "weight": weight_tensor,
            "original_size": (h_orig, w_orig),
            "image_path": img_path,
        }

    def _augment(self, image, semantic, boundary, weight):
        cfg = self.augment_config
        if cfg.get("horizontal_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            semantic = np.ascontiguousarray(semantic[:, ::-1])
            boundary = np.ascontiguousarray(boundary[:, ::-1])
            weight = np.ascontiguousarray(weight[:, ::-1])
        if cfg.get("vertical_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[::-1, :])
            semantic = np.ascontiguousarray(semantic[::-1, :])
            boundary = np.ascontiguousarray(boundary[::-1, :])
            weight = np.ascontiguousarray(weight[::-1, :])
        if cfg.get("rotation", False) and np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            image = np.ascontiguousarray(np.rot90(image, k))
            semantic = np.ascontiguousarray(np.rot90(semantic, k))
            boundary = np.ascontiguousarray(np.rot90(boundary, k))
            weight = np.ascontiguousarray(np.rot90(weight, k))
        return image, semantic, boundary, weight


# =============================================================================
# 无标签数据集（弱/强两路 + Patch Masking）
# =============================================================================
class UnlabeledDataset(Dataset):
    """
    无标注数据集：对每张图像生成弱/强两路增强。

    输出：
        img_weak: 仅 letterbox，无增强（教师预测源）
        img_strong: 弱图 + 强空间增强 + 外观增强
        patch_mask: [1, H, W] 随机 Patch Masking（1=遮挡, 0=保留）
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        appearance_aug_config: Optional[dict] = None,
        patch_mask_ratio: float = 0.3,
        patch_mask_size: int = 64,
        num_patches: int = 8,
        enable_appearance_aug: bool = True,
        boundary_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.patch_mask_ratio = patch_mask_ratio
        self.patch_mask_size = patch_mask_size
        self.num_patches = num_patches
        self.enable_appearance_aug = enable_appearance_aug
        self.boundary_cache_dir = boundary_cache_dir
        self._cache = None
        self._cache_names: List[str] = []
        self._cache_index: Dict[str, int] = {}
        self.appearance_config = appearance_aug_config or {
            "gaussian_blur_kernel": 5,
            "gaussian_blur_sigma_range": (0.5, 2.0),
            "brightness_range": (0.7, 1.3),
            "contrast_range": (0.7, 1.3),
        }

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.samples: List[str] = []
        for ext in valid_exts:
            for img_path in glob.glob(os.path.join(data_dir, f"*{ext}")):
                self.samples.append(img_path)
        self.samples.sort()

        # 离线 Stage-1 边界伪标签缓存（tools/precompute_pseudo_labels.py 生成）
        # 仅保留缓存中存在且未被剔除的无标签图
        if boundary_cache_dir is not None:
            names_path = os.path.join(boundary_cache_dir, "names.txt")
            probs_path = os.path.join(boundary_cache_dir, "boundary_probs.npy")
            if not (os.path.exists(names_path) and os.path.exists(probs_path)):
                raise FileNotFoundError(
                    f"Stage-1 边界伪标签缓存缺失: {boundary_cache_dir}\n"
                    f"请先运行: python tools/precompute_pseudo_labels.py "
                    f"--config config/default_config.yaml"
                )
            with open(names_path, "r", encoding="utf-8") as f:
                self._cache_names = [ln.strip() for ln in f if ln.strip()]
            self._cache_index = {
                n: i for i, n in enumerate(self._cache_names)
            }
            exclude = set()
            exclude_path = os.path.join(boundary_cache_dir, "exclude.txt")
            if os.path.exists(exclude_path):
                with open(exclude_path, "r", encoding="utf-8") as f:
                    exclude = {ln.strip() for ln in f if ln.strip()}
            keep = []
            for p in self.samples:
                bn = os.path.splitext(os.path.basename(p))[0]
                if bn in self._cache_index and bn not in exclude:
                    keep.append(p)
            self.samples = keep
            print(
                f"[UnlabeledDataset] boundary cache: {len(self._cache_names)} "
                f"cached, {len(exclude)} excluded -> {len(self.samples)} samples"
            )

        print(f"[UnlabeledDataset] {len(self.samples)} samples from {data_dir}")

    def _load_cache(self):
        """惰性打开边界伪标签 memmap（文件映射，不占用进程内存）。"""
        if self._cache is None and self.boundary_cache_dir is not None:
            probs_path = os.path.join(self.boundary_cache_dir, "boundary_probs.npy")
            self._cache = np.load(probs_path, mmap_mode="r")
        return self._cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        # img_weak: 仅 letterbox
        img_weak, _, _, _ = letterbox(image, self.image_size)

        # img_strong: 空间增强（+ 外观增强，可被渐进式增强替代时禁用）
        img_strong, spatial_params = self._apply_spatial_aug(img_weak)
        if self.enable_appearance_aug:
            img_strong = self._apply_appearance_aug(img_strong)

        # patch_mask: 随机遮挡
        patch_mask = self._generate_patch_mask()

        # 应用 patch mask 到 img_strong
        img_strong_masked = img_strong.copy()
        img_strong_masked[patch_mask > 0] = 0

        # 转张量
        img_weak_tensor = torch.from_numpy(img_weak).float().permute(2, 0, 1) / 255.0
        img_strong_tensor = torch.from_numpy(img_strong_masked).float().permute(2, 0, 1) / 255.0
        patch_mask_tensor = torch.from_numpy(patch_mask).float().unsqueeze(0)

        item = {
            "img_weak": img_weak_tensor,
            "img_strong": img_strong_tensor,
            "patch_mask": patch_mask_tensor,
            "original_size": (h_orig, w_orig),
            "image_path": img_path,
        }

        # 离线 Stage-1 边界目标（[1, H, W] 概率图，1024 letterbox 空间）
        if self.boundary_cache_dir is not None:
            cache = self._load_cache()
            basename = os.path.splitext(os.path.basename(img_path))[0]
            idx_cache = self._cache_index[basename]
            row = np.asarray(cache[idx_cache], dtype=np.float32)  # [H, W]
            # 与学生输入使用同一空间变换，保证边界目标几何对齐
            row = self._apply_spatial_transform(row, spatial_params)
            item["boundary_target"] = torch.from_numpy(row).unsqueeze(0)

        return item

    def _sample_spatial_params(self) -> Dict[str, object]:
        """采样空间增强参数（翻转/旋转），图像与伪标签目标共享同一参数。"""
        return {
            "hflip": bool(np.random.rand() < 0.5),
            "vflip": bool(np.random.rand() < 0.5),
            "rot_k": int(np.random.choice([0, 1, 2, 3])),
        }

    @staticmethod
    def _apply_spatial_transform(
        img: np.ndarray, params: Dict[str, object]
    ) -> np.ndarray:
        """按参数对图像/目标施加相同的翻转/旋转（对目标为最近邻等价变换）。"""
        result = img
        if params["hflip"]:
            result = result[:, ::-1]
        if params["vflip"]:
            result = result[::-1, :]
        if params["rot_k"]:
            result = np.rot90(result, params["rot_k"])
        return np.ascontiguousarray(result)

    def _apply_spatial_aug(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """空间增强：随机旋转/翻转；返回 (变换后图像, 参数)。

        同一参数必须施加到缓存伪标签目标，否则学生输入与目标几何错位，
        一致性损失与 margin 损失的梯度会互相抵消（雾状输出推不起来）。
        """
        params = self._sample_spatial_params()
        return self._apply_spatial_transform(img, params), params

    def _apply_appearance_aug(self, img: np.ndarray) -> np.ndarray:
        """外观增强：高斯模糊 + 亮度/对比度拉伸。"""
        cfg = self.appearance_config
        result = img.copy()

        kernel = cfg.get("gaussian_blur_kernel", 5)
        if kernel > 0:
            sigma = np.random.uniform(*cfg.get("gaussian_blur_sigma_range", (0.5, 2.0)))
            result = cv2.GaussianBlur(result, (kernel, kernel), sigma)

        brightness = np.random.uniform(*cfg.get("brightness_range", (0.7, 1.3)))
        result = np.clip(result * brightness, 0, 255).astype(np.uint8)

        contrast = np.random.uniform(*cfg.get("contrast_range", (0.7, 1.3)))
        mean = result.mean()
        result = np.clip((result - mean) * contrast + mean, 0, 255).astype(np.uint8)

        return result

    def _generate_patch_mask(self) -> np.ndarray:
        """生成随机 Patch Masking 掩码。

        Returns:
            mask: [H, W] uint8 (1=遮挡, 0=保留)
        """
        h, w = self.image_size, self.image_size
        mask = np.zeros((h, w), dtype=np.uint8)

        total_pixels = h * w
        target_masked = int(total_pixels * self.patch_mask_ratio)

        masked_pixels = 0
        attempts = 0
        max_attempts = self.num_patches * 3

        while masked_pixels < target_masked and attempts < max_attempts:
            cy = np.random.randint(0, h)
            cx = np.random.randint(0, w)

            ps = int(self.patch_mask_size * np.random.uniform(0.5, 1.5))
            y1 = max(0, cy - ps // 2)
            y2 = min(h, cy + ps // 2)
            x1 = max(0, cx - ps // 2)
            x2 = min(w, cx + ps // 2)

            new_pixels = int((mask[y1:y2, x1:x2] == 0).sum())
            mask[y1:y2, x1:x2] = 1
            masked_pixels += new_pixels
            attempts += 1

        return mask


# =============================================================================
# Collate 函数
# =============================================================================
def labeled_collate_fn(batch: List[Dict]) -> Dict:
    """有标签批次 collate。"""
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    weights = torch.stack([item["weight"] for item in batch])
    return {
        "image": images,
        "target": targets,
        "weight": weights,
        "original_size": [item["original_size"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def unlabeled_collate_fn(batch: List[Dict]) -> Dict:
    """无标签批次 collate。"""
    img_weak = torch.stack([item["img_weak"] for item in batch])
    img_strong = torch.stack([item["img_strong"] for item in batch])
    patch_masks = torch.stack([item["patch_mask"] for item in batch])
    out = {
        "img_weak": img_weak,
        "img_strong": img_strong,
        "patch_mask": patch_masks,
        "original_size": [item["original_size"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }
    if batch[0].get("boundary_target") is not None:
        out["boundary_target"] = torch.stack(
            [item["boundary_target"] for item in batch]
        )
    return out
