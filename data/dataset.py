# -*- coding: utf-8 -*-
"""
在线数据管道
============
通过 PyTorch Dataset 在内存中在线处理图像，禁止离线改图。

核心功能：
1. 在线 Letterbox 变换：将任意非标准分辨率图像的长边等比例缩放至 1024，
   短边按相同比例缩放后，在右侧/下方利用 0 像素补齐到 1024*1024。
   保证长宽比保真，禁止强行挤压变形缩放。
2. 双任务目标生成：
   - 通道 0（二分类掩码）：铁素体（核+晶界）并集 = 1，珠光体 = 0
   - 通道 1（连续距离场）：铁素体像素到最近珠光体边界的欧氏距离（归一化至 [0,1]）
3. 距离场使用 scipy.ndimage.distance_transform_edt 计算，非线性缩放 dist / (dist + 10)
"""

import json
import os
import glob
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


# 类别常量（仅用于后处理兼容性，网络不再预测类别 2）
CLASS_PEARLITE = 0       # 珠光体
CLASS_FERRITE = 1        # 铁素体（核+晶界统一为前景）

NUM_OUTPUT_CHANNELS = 2  # 双任务输出：掩码 + 距离场


def letterbox(
    image: np.ndarray,
    target_size: int = 1024,
    pad_value: int = 0,
) -> Tuple[np.ndarray, float, int, int]:
    """
    在线 Letterbox 变换：长边等比例缩放至 target_size，短边补齐。

    禁止强行挤压变形缩放，保证长宽比保真。

    Args:
        image: 原始图像 [H, W, C] (BGR or RGB)
        target_size: 目标长边尺寸（默认 1024）
        pad_value: 填充像素值（默认 0）

    Returns:
        letterboxed: [target_size, target_size, C] 图像
        scale: 缩放比例
        pad_h: 下方填充像素数
        pad_w: 右侧填充像素数
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # 等比例缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充量（右侧和下方）
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    # 创建目标画布并粘贴
    letterboxed = np.full(
        (target_size, target_size, image.shape[2]) if image.ndim == 3 else (target_size, target_size),
        pad_value,
        dtype=image.dtype,
    )
    letterboxed[:new_h, :new_w] = resized

    return letterboxed, scale, pad_h, pad_w


def letterbox_mask(
    mask: np.ndarray,
    target_size: int = 1024,
    pad_value: int = 0,
) -> Tuple[np.ndarray, float, int, int]:
    """
    对掩码进行 Letterbox 变换（使用最近邻插值保持标签不混叠）。

    Args:
        mask: 原始掩码 [H, W]
        target_size: 目标长边尺寸
        pad_value: 填充值

    Returns:
        letterboxed: [target_size, target_size] 掩码
        scale: 缩放比例
        pad_h: 下方填充像素数
        pad_w: 右侧填充像素数
    """
    h, w = mask.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # 最近邻插值保持标签
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    letterboxed = np.full((target_size, target_size), pad_value, dtype=mask.dtype)
    letterboxed[:new_h, :new_w] = resized

    return letterboxed, scale, pad_h, pad_w


def polygons_to_mask(
    polygons: List[List[List[float]]],
    height: int,
    width: int,
) -> np.ndarray:
    """
    将 Labelme 多边形标注转换为二值掩码。

    Args:
        polygons: 多边形点列表，每个多边形是 [[x1,y1], [x2,y2], ...]
        height: 图像高度
        width: 图像宽度

    Returns:
        mask: [H, W] 二值掩码 (0/1)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    return mask


def parse_labelme_json(
    json_path: str,
    height: int,
    width: int,
) -> Dict[str, np.ndarray]:
    """
    解析 Labelme JSON 标注文件，提取 ferrite 和 pearlite 多边形。

    Labelme JSON 结构：
        {
            "imageHeight": H,
            "imageWidth": W,
            "shapes": [
                {
                    "label": "ferrite" / "pearlite",
                    "points": [[x1,y1], [x2,y2], ...],
                    ...
                },
                ...
            ]
        }

    Args:
        json_path: JSON 文件路径
        height: 图像高度
        width: 图像宽度

    Returns:
        dict: {
            "ferrite": [H, W] 二值掩码,
            "pearlite": [H, W] 二值掩码,
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ferrite_polys = []
    pearlite_polys = []

    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower().strip()
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        if label in ("ferrite", "ferrite_core", "铁素体"):
            ferrite_polys.append(points)
        elif label in ("pearlite", "珠光体"):
            pearlite_polys.append(points)

    ferrite_mask = polygons_to_mask(ferrite_polys, height, width)
    pearlite_mask = polygons_to_mask(pearlite_polys, height, width)

    return {"ferrite": ferrite_mask, "pearlite": pearlite_mask}


def create_binary_mask(
    ferrite_mask: np.ndarray,
    pearlite_mask: np.ndarray,
) -> np.ndarray:
    """
    生成二分类掩码：铁素体（核+晶界）= 1，珠光体 = 0。

    Args:
        ferrite_mask: [H, W] ferrite 二值掩码 (0/1)
        pearlite_mask: [H, W] pearlite 二值掩码 (0/1)

    Returns:
        binary_mask: [H, W] 二值掩码 (0=珠光体, 1=铁素体)
    """
    # 铁素体并集为前景
    binary_mask = np.where(ferrite_mask > 0, 1, 0).astype(np.uint8)
    return binary_mask


def create_distance_field(
    ferrite_mask: np.ndarray,
    pearlite_mask: np.ndarray,
    scale_factor: float = 10.0,
) -> np.ndarray:
    """
    计算铁素体像素到最近珠光体边界的欧氏距离场。

    使用 scipy.ndimage.distance_transform_edt 计算铁素体区域内每个像素
    到最近非铁素体像素的欧氏距离，然后进行非线性归一化。

    珠光体区域的距离值完全设为 0。
    非线性缩放公式: dist_norm = dist / (dist + scale_factor)

    Args:
        ferrite_mask: [H, W] ferrite 二值掩码 (0/1)
        pearlite_mask: [H, W] pearlite 二值掩码 (0/1)
        scale_factor: 非线性缩放因子（默认 10.0）

    Returns:
        dist_field: [H, W] 归一化距离场，范围 [0, 1]，珠光体区域为 0
    """
    # distance_transform_edt 计算前景(>0)像素到最近背景(==0)像素的距离
    # ferrite_mask > 0 的区域为铁素体，计算其到非铁素体区域的距离
    if ferrite_mask.sum() == 0:
        return np.zeros_like(ferrite_mask, dtype=np.float32)

    dist = np.asarray(distance_transform_edt(ferrite_mask > 0), dtype=np.float32)

    # 非线性归一化: dist / (dist + scale_factor)
    # 使数值平滑落在 [0, 1] 区间
    dist_norm = dist / (dist + scale_factor)

    # 珠光体区域距离值设为 0
    dist_norm = np.where(ferrite_mask > 0, dist_norm, 0.0).astype(np.float32)

    return dist_norm


def letterbox_distance_field(
    dist_field: np.ndarray,
    target_size: int = 1024,
) -> Tuple[np.ndarray, float, int, int]:
    """
    对距离场进行 Letterbox 变换（使用双线性插值保持连续性）。

    Args:
        dist_field: [H, W] float32 距离场
        target_size: 目标长边尺寸

    Returns:
        letterboxed: [target_size, target_size] 距离场
        scale: 缩放比例
        pad_h: 下方填充像素数
        pad_w: 右侧填充像素数
    """
    h, w = dist_field.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # 双线性插值保持距离场连续性
    resized = cv2.resize(dist_field, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    letterboxed = np.zeros((target_size, target_size), dtype=np.float32)
    letterboxed[:new_h, :new_w] = resized

    return letterboxed, scale, pad_h, pad_w


class MetallographicDataset(Dataset):
    """
    低碳钢金相图像在线数据集（双任务：二分类掩码 + 距离场回归）。

    在 __getitem__ 中实时执行：
    1. 读取原始图像与同名的 labelme .json 文件
    2. 在线 Letterbox 变换（长边缩放至 1024，短边补 0）
    3. 生成二分类掩码（铁素体=1，珠光体=0）
    4. 计算连续距离场（铁素体到珠光体边界的欧氏距离，归一化至 [0,1]）
    5. 转换为 PyTorch 张量

    约束：
    - 禁止离线改图，所有变换在内存中在线完成
    - 长宽比保真，禁止挤压变形
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 1024,
        erode_pixels: int = 2,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        dist_scale_factor: float = 10.0,
    ):
        """
        Args:
            data_dir: 原始数据目录（包含图像与同名 .json）
            image_size: Letterbox 目标长边尺寸
            erode_pixels: （已废弃，保留参数兼容性）
            augment: 是否启用数据增强
            augment_config: 增强配置 dict
            split: "train" / "val"
            train_ratio: 训练集占比
            seed: 随机种子
            dist_scale_factor: 距离场非线性缩放因子
        """
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.erode_pixels = erode_pixels
        self.augment = augment and (split == "train")
        self.augment_config = augment_config or {}
        self.split = split
        self.dist_scale_factor = dist_scale_factor

        # 支持的图像扩展名
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        # 收集所有有对应 .json 的图像
        all_samples = []
        for ext in valid_exts:
            for img_path in glob.glob(os.path.join(data_dir, f"*{ext}")):
                json_path = os.path.splitext(img_path)[0] + ".json"
                if os.path.exists(json_path):
                    all_samples.append((img_path, json_path))

        all_samples.sort()

        # 划分训练/验证集
        np.random.seed(seed)
        n_total = len(all_samples)

        # 确保至少有 1 张训练集和 1 张验证集（当 n_total >= 2 时）
        n_train = int(n_total * train_ratio)
        if n_total >= 2:
            n_train = max(1, min(n_train, n_total - 1))
        else:
            n_train = max(1, n_train)

        indices = np.random.permutation(n_total)

        if split == "train":
            selected = indices[:n_train]
        else:
            val_indices = indices[n_train:]
            if len(val_indices) == 0:
                val_indices = indices
            selected = val_indices

        self.samples = [all_samples[i] for i in selected]

        print(f"[{split}] 数据集: {len(self.samples)} 张图像 (共 {n_total} 张)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        在线处理流程：
        1. 读取图像与 JSON
        2. Letterbox 变换
        3. 生成二分类掩码 + 距离场
        4. 数据增强（可选）
        5. 转张量

        Returns:
            dict: {
                "image": [3, H, W] float32 张量 (0-1 归一化),
                "target": [2, H, W] float32 张量
                       - target[0] 为二值掩码 (0/1)
                       - target[1] 为归一化距离场 [0,1]
                "original_size": (H_orig, W_orig),
                "scale": 缩放比例,
                "pad_h": 下方填充,
                "pad_w": 右侧填充,
                "image_path": 图像路径,
            }
        """
        img_path, json_path = self.samples[idx]

        # 1. 读取原始图像（BGR -> RGB）
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        # 2. 解析 Labelme JSON，获取 ferrite/pearlite 掩码
        masks = parse_labelme_json(json_path, h_orig, w_orig)
        ferrite_mask = masks["ferrite"]
        pearlite_mask = masks["pearlite"]

        # 3. 生成二分类掩码（原始尺寸）
        binary_mask = create_binary_mask(ferrite_mask, pearlite_mask)

        # 4. 计算连续距离场（原始尺寸）
        dist_field = create_distance_field(
            ferrite_mask, pearlite_mask, scale_factor=self.dist_scale_factor
        )

        # 5. Letterbox 变换（图像、掩码、距离场使用相同缩放参数）
        image_lb, scale, pad_h, pad_w = letterbox(image, self.image_size)
        mask_lb, _, _, _ = letterbox_mask(binary_mask, self.image_size)
        dist_lb, _, _, _ = letterbox_distance_field(dist_field, self.image_size)

        # 6. 数据增强（仅训练时，同时对图像、掩码、距离场做相同变换）
        if self.augment:
            image_lb, mask_lb, dist_lb = self._augment(image_lb, mask_lb, dist_lb)

        # 7. 转换为 PyTorch 张量
        # 图像归一化到 [0, 1]，通道优先 (C, H, W)
        image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1) / 255.0

        # 目标：2 通道 [mask, dist_field]
        mask_tensor = torch.from_numpy(mask_lb).float().unsqueeze(0)  # [1, H, W]
        dist_tensor = torch.from_numpy(dist_lb).float().unsqueeze(0)  # [1, H, W]
        target_tensor = torch.cat([mask_tensor, dist_tensor], dim=0)  # [2, H, W]

        # --- 刚性断言：确保 target 通道对齐 ---
        assert target_tensor.shape[0] == 2, \
            f"Target 通道数错误，预期为 2，实际为 {target_tensor.shape[0]}"

        mask_target = target_tensor[0]
        dist_target = target_tensor[1]

        # 检查分类标签的二值性
        unique_vals = torch.unique(mask_target)
        for val in unique_vals:
            assert val.item() in (0.0, 1.0), \
                f"分类标签包含非法值: {val.item()}。" \
                f"Channel 0 必须严格为二值掩码（0=背景/珠光体, 1=铁素体）"

        # 检查回归标签的归一化范围
        assert dist_target.max().item() <= 1.0 and dist_target.min().item() >= 0.0, \
            f"距离场未严格归一化: max={dist_target.max().item()}, " \
            f"min={dist_target.min().item()}"

        return {
            "image": image_tensor,
            "target": target_tensor,
            "original_size": (h_orig, w_orig),
            "scale": scale,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "image_path": img_path,
        }

    def _augment(
        self, image: np.ndarray, mask: np.ndarray, dist_field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        在线数据增强（同时对图像、掩码、距离场施加相同空间变换）。

        Args:
            image: [H, W, C] RGB 图像
            mask: [H, W] 二值掩码
            dist_field: [H, W] 距离场

        Returns:
            augmented_image, augmented_mask, augmented_dist_field
        """
        cfg = self.augment_config

        # 水平翻转
        if cfg.get("horizontal_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            dist_field = np.ascontiguousarray(dist_field[:, ::-1])

        # 垂直翻转
        if cfg.get("vertical_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
            dist_field = np.ascontiguousarray(dist_field[::-1, :])

        # 随机旋转 90/180/270 度
        if cfg.get("rotation", False) and np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            image = np.ascontiguousarray(np.rot90(image, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
            dist_field = np.ascontiguousarray(np.rot90(dist_field, k))

        return image, mask, dist_field


def collate_fn(batch: List[Dict]) -> Dict:
    """
    自定义 collate 函数，处理变长元数据。

    Args:
        batch: List of dict from __getitem__

    Returns:
        batched dict
    """
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    return {
        "image": images,
        "target": targets,
        "original_size": [item["original_size"] for item in batch],
        "scale": [item["scale"] for item in batch],
        "pad_h": [item["pad_h"] for item in batch],
        "pad_w": [item["pad_w"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }