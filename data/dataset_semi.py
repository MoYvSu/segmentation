# -*- coding: utf-8 -*-
"""
半监督数据集模块
================
为第二阶段半监督微调提供双流数据加载：
- 有标签流：输出原图 + 真值掩码 + 距离场（复用第一阶段逻辑）
- 无标签流：三路分叉增强
  - img_weak: 仅 letterbox，无增强（教师预测源）
  - img_strong_appearance: 外观增强（高斯模糊 + 亮度/对比度拉伸）
  - img_strong_geometric: 几何增强（随机90度旋转/翻转）+ 变换元数据 T

非侵入式设计：不修改第一阶段 data/dataset.py，仅复用其工具函数。
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import (
    letterbox,
    letterbox_mask,
    letterbox_vector_field,
    parse_labelme_json,
    create_binary_mask,
    create_offset_vector_field,
)


# =============================================================================
# 有标签数据集（复用第一阶段逻辑）
# =============================================================================
class LabeledDataset(Dataset):
    """
    有标注数据集：输出原图 + 真值掩码 + 距离场。
    与 MetallographicDataset 逻辑一致，但作为独立类避免侵入式修改。
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        dist_scale_factor: float = 10.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.augment = augment
        self.augment_config = augment_config or {}
        self.dist_scale_factor = dist_scale_factor

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.samples: List[Tuple[str, str]] = []
        for ext in valid_exts:
            for img_path in glob.glob(os.path.join(data_dir, f"*{ext}")):
                json_path = os.path.splitext(img_path)[0] + ".json"
                if os.path.exists(json_path):
                    self.samples.append((img_path, json_path))
        self.samples.sort()
        print(f"[LabeledDataset] {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, json_path = self.samples[idx]

        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        # 解析标注
        masks = parse_labelme_json(json_path, h_orig, w_orig)
        ferrite_mask = masks["ferrite"]
        pearlite_mask = masks["pearlite"]
        ferrite_polys = masks["ferrite_polys"]

        # 生成目标
        binary_mask = create_binary_mask(ferrite_mask, pearlite_mask)
        vec_field = create_offset_vector_field(
            ferrite_polys, h_orig, w_orig, normalize_to=float(self.image_size)
        )

        # Letterbox
        image_lb, scale, pad_h, pad_w = letterbox(image, self.image_size)
        mask_lb, _, _, _ = letterbox_mask(binary_mask, self.image_size)
        vec_lb, _, _, _ = letterbox_vector_field(vec_field, self.image_size)

        # 数据增强
        if self.augment:
            image_lb, mask_lb, vec_lb = self._augment(image_lb, mask_lb, vec_lb)

        # 转张量
        image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_lb).float().unsqueeze(0)
        vec_tensor = torch.from_numpy(vec_lb).float().permute(2, 0, 1)
        target_tensor = torch.cat([mask_tensor, vec_tensor], dim=0)

        return {
            "image": image_tensor,
            "target": target_tensor,
            "original_size": (h_orig, w_orig),
            "image_path": img_path,
        }

    def _augment(self, image, mask, vec_field):
        cfg = self.augment_config
        vx = vec_field[..., 0].copy()
        vy = vec_field[..., 1].copy()

        if cfg.get("horizontal_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            vx = np.ascontiguousarray(vx[:, ::-1])
            vy = np.ascontiguousarray(vy[:, ::-1])
            vx = -vx

        if cfg.get("vertical_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
            vx = np.ascontiguousarray(vx[::-1, :])
            vy = np.ascontiguousarray(vy[::-1, :])
            vy = -vy

        if cfg.get("rotation", False) and np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            image = np.ascontiguousarray(np.rot90(image, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
            vx = np.ascontiguousarray(np.rot90(vx, k))
            vy = np.ascontiguousarray(np.rot90(vy, k))
            for _ in range(k):
                vx, vy = vy.copy(), -vx.copy()

        augmented_vec = np.stack([vx, vy], axis=-1)
        return image, mask, augmented_vec


# =============================================================================
# 无标签数据集（三路分叉增强）
# =============================================================================
class UnlabeledDataset(Dataset):
    """
    无标注数据集：对每张图像生成三路分叉增强。

    输出：
        img_weak: 仅 letterbox，无增强（教师预测源）
        img_strong_appearance: 在 img_weak 基础上叠加高斯模糊 + 亮度/对比度拉伸
        img_strong_geometric: 施加随机90度旋转或翻转（与 img_weak 空间不对齐）
        T: 几何变换元数据，用于将 img_weak 的预测对齐到 img_strong_geometric 坐标系
    """

    # 几何变换类型
    GEOM_IDENTITY = "identity"
    GEOM_ROT90 = "rot90"
    GEOM_FLIP_H = "flip_h"
    GEOM_FLIP_V = "flip_v"

    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        appearance_aug_config: Optional[dict] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
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
        print(f"[UnlabeledDataset] {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.samples[idx]

        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        # img_weak: 仅 letterbox
        img_weak, _, _, _ = letterbox(image, self.image_size)

        # img_strong_appearance: 外观增强（空间位置不变，仅改变像素值）
        img_strong_appearance = self._apply_appearance_aug(img_weak)

        # img_strong_geometric: 几何增强（改变空间位置）
        img_strong_geometric, T = self._apply_geometric_aug(img_weak)

        # 转张量
        img_weak_tensor = torch.from_numpy(img_weak).float().permute(2, 0, 1) / 255.0
        img_strong_app_tensor = (
            torch.from_numpy(img_strong_appearance).float().permute(2, 0, 1) / 255.0
        )
        img_strong_geo_tensor = (
            torch.from_numpy(img_strong_geometric).float().permute(2, 0, 1) / 255.0
        )

        return {
            "img_weak": img_weak_tensor,
            "img_strong_appearance": img_strong_app_tensor,
            "img_strong_geometric": img_strong_geo_tensor,
            "T": T,
            "original_size": (h_orig, w_orig),
            "image_path": img_path,
        }

    def _apply_appearance_aug(self, img: np.ndarray) -> np.ndarray:
        """
        外观增强：高斯模糊 + 亮度/对比度随机拉伸。

        空间位置不变，仅改变像素值，因此分类伪标签空间对齐。
        """
        cfg = self.appearance_config
        result = img.copy()

        # 高斯模糊
        kernel = cfg.get("gaussian_blur_kernel", 5)
        if kernel > 0:
            sigma = np.random.uniform(*cfg.get("gaussian_blur_sigma_range", (0.5, 2.0)))
            result = cv2.GaussianBlur(result, (kernel, kernel), sigma)

        # 亮度拉伸
        brightness = np.random.uniform(*cfg.get("brightness_range", (0.7, 1.3)))
        result = np.clip(result * brightness, 0, 255).astype(np.uint8)

        # 对比度拉伸
        contrast = np.random.uniform(*cfg.get("contrast_range", (0.7, 1.3)))
        mean = result.mean()
        result = np.clip((result - mean) * contrast + mean, 0, 255).astype(np.uint8)

        return result

    def _apply_geometric_aug(self, img: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        几何增强：随机选择 90度旋转 / 水平翻转 / 垂直翻转 / 恒等。

        返回变换后图像和元数据 T。T 描述从 img_weak 到 img_strong_geometric 的变换。
        后续在 loss 计算中，需要对 img_weak 的预测施加相同的 T 以对齐。

        Args:
            img: [H, W, C] numpy array (img_weak)

        Returns:
            transformed: [H, W, C] 变换后图像
            T: {"type": str, "k": int} 变换元数据
        """
        choices = [self.GEOM_IDENTITY, self.GEOM_ROT90, self.GEOM_FLIP_H, self.GEOM_FLIP_V]
        choice = np.random.choice(choices)

        if choice == self.GEOM_ROT90:
            k = int(np.random.choice([1, 2, 3]))
            transformed = np.ascontiguousarray(np.rot90(img, k))
            T = {"type": self.GEOM_ROT90, "k": k}
        elif choice == self.GEOM_FLIP_H:
            transformed = np.ascontiguousarray(img[:, ::-1])
            T = {"type": self.GEOM_FLIP_H, "k": 0}
        elif choice == self.GEOM_FLIP_V:
            transformed = np.ascontiguousarray(img[::-1, :])
            T = {"type": self.GEOM_FLIP_V, "k": 0}
        else:
            transformed = np.ascontiguousarray(img)
            T = {"type": self.GEOM_IDENTITY, "k": 0}

        return transformed, T


# =============================================================================
# Collate 函数
# =============================================================================
def labeled_collate_fn(batch: List[Dict]) -> Dict:
    """有标签批次 collate。"""
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    return {
        "image": images,
        "target": targets,
        "original_size": [item["original_size"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def unlabeled_collate_fn(batch: List[Dict]) -> Dict:
    """无标签批次 collate：堆叠三路图像张量，T 以列表形式保留。"""
    img_weak = torch.stack([item["img_weak"] for item in batch])
    img_strong_app = torch.stack([item["img_strong_appearance"] for item in batch])
    img_strong_geo = torch.stack([item["img_strong_geometric"] for item in batch])
    return {
        "img_weak": img_weak,
        "img_strong_appearance": img_strong_app,
        "img_strong_geometric": img_strong_geo,
        "T_list": [item["T"] for item in batch],
        "original_size": [item["original_size"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


# =============================================================================
# 几何变换工具（用于 loss 计算中对齐预测）
# =============================================================================
def apply_transform_to_tensor(tensor: torch.Tensor, T: Dict, is_vector: bool = False) -> torch.Tensor:
    """
    对 [B, C, H, W] 或 [C, H, W] 张量施加几何变换 T。

    用于在 loss 计算中将 img_weak 的预测对齐到 img_strong_geometric 的坐标系。

    当 is_vector=True 时，张量的通道维包含向量场分量 (Vx, Vy)，
    需要随空间变换同步旋转/翻转分量：
    - flip_h: Vx -> -Vx, Vy -> Vy
    - flip_v: Vx -> Vx, Vy -> -Vy
    - rot90 k: 逐次旋转 (Vx, Vy) -> (Vy, -Vx)

    Args:
        tensor: [B, C, H, W] 或 [C, H, W] 张量
        T: {"type": "rot90"/"flip_h"/"flip_v"/"identity", "k": int}
        is_vector: 是否为向量场张量（需分量变换）

    Returns:
        变换后的张量（与输入形状相同，因为所有变换都是 90 度倍数）
    """
    t_type = T["type"]
    k = T.get("k", 0)

    if t_type == "rot90":
        if tensor.dim() == 4:
            result = torch.rot90(tensor, k, dims=(2, 3))
        elif tensor.dim() == 3:
            result = torch.rot90(tensor, k, dims=(1, 2))
        else:
            raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")

        if is_vector:
            # 向量分量旋转: k 次 (Vx, Vy) -> (Vy, -Vx)
            vx = result[..., 0, :, :].clone()
            vy = result[..., 1, :, :].clone()
            for _ in range(k):
                vx, vy = vy.clone(), -vx.clone()
            result = torch.stack([vx, vy], dim=-2 if tensor.dim() == 4 else -2)
            # 确保维度顺序正确
            if tensor.dim() == 4:
                result = torch.stack([vx, vy], dim=1)
            else:
                result = torch.stack([vx, vy], dim=0)
        return result

    elif t_type == "flip_h":
        # 水平翻转（沿 W 轴翻转）
        if tensor.dim() == 4:
            result = torch.flip(tensor, dims=(3,))
        elif tensor.dim() == 3:
            result = torch.flip(tensor, dims=(2,))
        else:
            raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")

        if is_vector:
            # Vx -> -Vx
            result[..., 0, :, :] = -result[..., 0, :, :]
        return result

    elif t_type == "flip_v":
        # 垂直翻转（沿 H 轴翻转）
        if tensor.dim() == 4:
            result = torch.flip(tensor, dims=(2,))
        elif tensor.dim() == 3:
            result = torch.flip(tensor, dims=(1,))
        else:
            raise ValueError(f"Unexpected tensor dim: {tensor.dim()}")

        if is_vector:
            # Vy -> -Vy
            result[..., 1, :, :] = -result[..., 1, :, :]
        return result

    else:
        # identity
        return tensor


def apply_transform_batch(
    tensor: torch.Tensor, T_list: List[Dict], is_vector: bool = False
) -> torch.Tensor:
    """
    对 [B, C, H, W] 张量的每个样本施加不同的几何变换。

    Args:
        tensor: [B, C, H, W] 张量
        T_list: 长度为 B 的变换元数据列表
        is_vector: 是否为向量场张量（需分量变换）

    Returns:
        变换后的 [B, C, H, W] 张量
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")

    transformed = []
    for i, T in enumerate(T_list):
        transformed.append(apply_transform_to_tensor(tensor[i], T, is_vector=is_vector))
    return torch.stack(transformed)
