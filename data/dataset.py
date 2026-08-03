# -*- coding: utf-8 -*-
"""
在线数据管道（边界预测版本）
===========================
通过 PyTorch Dataset 在内存中在线处理图像，加载离线净化的边界 GT。

核心功能：
1. 在线 Letterbox 变换：将任意非标准分辨率图像的长边等比例缩放至 1024，
   短边按相同比例缩放后，在右侧/下方利用镜像反射（BORDER_REFLECT）补齐到 1024*1024。
   保证长宽比保真，禁止强行挤压变形缩放，且镜像填充避免零值污染。
2. 双通道目标加载：
   - 通道 0（语义掩码）：铁素体 = 1，珠光体 = 0
   - 通道 1（边界掩码）：净化后的晶界带 = 1，非边界 = 0
3. 在线 EDT 权重图：基于边界通道计算距离权重，边界附近权重高
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


# 类别常量
CLASS_PEARLITE = 0       # 珠光体
CLASS_FERRITE = 1        # 铁素体

NUM_OUTPUT_CHANNELS = 2  # 双通道输出：语义掩码 + 边界掩码


def letterbox(
    image: np.ndarray,
    target_size: int = 1024,
    pad_value: int = 0,  # deprecated: 现使用 BORDER_REFLECT 镜像填充
) -> Tuple[np.ndarray, float, int, int]:
    """在线 Letterbox 变换：长边等比例缩放至 target_size，短边镜像填充。"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    letterboxed = cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )
    return letterboxed, scale, pad_h, pad_w


def letterbox_mask(
    mask: np.ndarray,
    target_size: int = 1024,
    pad_value: int = 0,  # deprecated: 现使用 BORDER_REFLECT 镜像填充
) -> Tuple[np.ndarray, float, int, int]:
    """对掩码进行 Letterbox 变换（最近邻插值 + 镜像填充，保持标签不混叠）。"""
    h, w = mask.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    letterboxed = cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
    )
    return letterboxed, scale, pad_h, pad_w


def polygons_to_mask(
    polygons: List[List[List[float]]],
    height: int,
    width: int,
) -> np.ndarray:
    """将 Labelme 多边形标注转换为二值掩码。"""
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
    """解析 Labelme JSON 标注文件，提取 ferrite 和 pearlite 多边形。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ferrite_polys = []
    pearlite_polys = []

    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower().strip()
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        if label in ("ferrite", "ferrite_core", "铁素体", "1"):
            ferrite_polys.append(points)
        elif label in ("pearlite", "珠光体", "0"):
            pearlite_polys.append(points)

    ferrite_mask = polygons_to_mask(ferrite_polys, height, width)
    pearlite_mask = polygons_to_mask(pearlite_polys, height, width)

    return {
        "ferrite": ferrite_mask,
        "pearlite": pearlite_mask,
        "ferrite_polys": ferrite_polys,
        "pearlite_polys": pearlite_polys,
    } # type: ignore


def create_binary_mask(
    ferrite_mask: np.ndarray,
    pearlite_mask: np.ndarray,
) -> np.ndarray:
    """生成二分类掩码：铁素体 = 1，珠光体 = 0。"""
    binary_mask = np.where(ferrite_mask > 0, 1, 0).astype(np.uint8)
    return binary_mask


def compute_boundary_weight(boundary: np.ndarray, scale_factor: float = 10.0,
                            weight_floor: float = 1.0,
                            weight_ceil: float = 4.0) -> np.ndarray:
    """
    基于边界掩码计算 EDT 边界权重图（梯度过载保护版本）。

    权重在边界处最高（weight_ceil），随着距离边界越远逐渐降低至 weight_floor。
    用于 Focal Loss 调制，使模型重点学习边界附近的高锐度像素。

    改造说明：原先权重范围 [0.3, 1.0]，边界处仅 1.0 倍梯度，
    稀疏边界区域可能释放毁灭性宏观梯度。现改为 [1.0, 4.0] 范围，
    边界处最高 4 倍梯度，远处基线 1.0，配合 clamp 防止梯度过载。

    Args:
        boundary: [H, W] 二值边界掩码（1=边界, 0=非边界）
        scale_factor: EDT 非线性缩放因子
        weight_floor: 权重下限（远处基线）
        weight_ceil: 权重上限（边界处峰值）

    Returns:
        weight: [H, W] float32 权重图，范围 [weight_floor, weight_ceil]
    """
    if boundary.sum() == 0:
        return np.full(boundary.shape, weight_floor, dtype=np.float32)

    # 计算每个非边界像素到最近边界像素的距离
    dist = np.asarray(distance_transform_edt(boundary == 0), dtype=np.float32)

    # 归一化：距离越近边界，权重越高（从 floor 升至 ceil）
    dist_norm = 1.0 - dist / (dist + scale_factor)  # 边界处=1.0, 远处→0.0
    # 映射到 [weight_floor, weight_ceil] 区间
    weight = weight_floor + (weight_ceil - weight_floor) * dist_norm

    # 边界像素本身权重设为 weight_ceil
    weight[boundary > 0] = weight_ceil

    return weight.astype(np.float32)


def random_crop(
    image: np.ndarray,
    semantic: np.ndarray,
    boundary: np.ndarray,
    weight: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    局部随机裁剪：对图像、语义、边界、权重使用相同的裁剪坐标。

    在 Letterbox 变换后的全图上随机选取 crop_size×crop_size 区域。
    若全图小于 crop_size，则 pad 到 crop_size。

    Args:
        image: [H, W, C] 图像
        semantic: [H, W] 语义掩码
        boundary: [H, W] 边界掩码
        weight: [H, W] 权重图
        crop_size: 裁剪边长

    Returns:
        裁剪后的 (image, semantic, boundary, weight)
    """
    h, w = image.shape[:2]

    # 若图像小于 crop_size，先 pad
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='symmetric')
        semantic = np.pad(semantic, ((0, pad_h), (0, pad_w)), mode='symmetric')
        boundary = np.pad(boundary, ((0, pad_h), (0, pad_w)), mode='symmetric')
        weight = np.pad(weight, ((0, pad_h), (0, pad_w)), mode='symmetric')
        h, w = image.shape[:2]

    # 随机选取裁剪起点
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    bottom = top + crop_size
    right = left + crop_size

    image_c = image[top:bottom, left:right]
    semantic_c = semantic[top:bottom, left:right]
    boundary_c = boundary[top:bottom, left:right]
    weight_c = weight[top:bottom, left:right]

    return image_c, semantic_c, boundary_c, weight_c


def split_train_val_indices(
    n_total: int,
    train_ratio: float = 0.8,
    seed: int = 42,
    split: str = "train",
) -> np.ndarray:
    """按固定 seed 生成与 BoundaryDataset 完全一致的 train/val 索引划分。

    先用 seed 重置全局随机状态，再对 [0, n_total) 做同一置换：
    前 train_ratio 部分作为训练索引、其余作为验证索引。
    任何数据集只要样本列表排序方式一致、传入相同参数，即可得到相同划分，
    避免 Stage 2 有标签流与验证流使用不同划分造成的验证集泄露。
    """
    np.random.seed(seed)
    n_train = int(n_total * train_ratio)
    if n_total >= 2:
        n_train = max(1, min(n_train, n_total - 1))
    else:
        n_train = max(1, n_train)

    indices = np.random.permutation(n_total)

    if split == "train":
        return indices[:n_train]

    val_indices = indices[n_train:]
    if len(val_indices) == 0:
        val_indices = indices
    return val_indices


class BoundaryDataset(Dataset):
    """
    低碳钢金相图像边界预测数据集。

    加载离线净化的边界 GT（.npz），在线执行：
    1. 读取原始图像与对应的 _gt.npz 文件
    2. 在线 Letterbox 变换（长边缩放至 1024，短边镜像填充 BORDER_REFLECT）
    3. 在线计算 EDT 边界权重图
    4. 数据增强（可选）
    5. 转换为 PyTorch 张量

    约束：
    - 禁止离线改图，所有变换在内存中在线完成
    - 长宽比保真，禁止挤压变形
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
        """
        Args:
            data_dir: 原始数据目录（包含图像与同名 .json）
            gt_dir: 净化 GT 目录（包含 _gt.npz 文件）
            image_size: Letterbox 目标长边尺寸
            crop_size: 局部随机裁剪尺寸（0=不裁剪，使用全图）
            augment: 是否启用数据增强
            augment_config: 增强配置 dict
            split: "train" / "val"
            train_ratio: 训练集占比
            seed: 随机种子
            boundary_scale_factor: EDT 权重缩放因子
            boundary_weight_floor: EDT 权重下限（远处基线）
            boundary_weight_ceil: EDT 权重上限（边界处峰值）
        """
        super().__init__()
        self.data_dir = data_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.augment = augment and (split == "train")
        self.augment_config = augment_config or {}
        self.split = split
        self.boundary_scale_factor = boundary_scale_factor
        self.boundary_weight_floor = boundary_weight_floor
        self.boundary_weight_ceil = boundary_weight_ceil

        # 支持的图像扩展名
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

        # 收集所有有对应 .json 和 _gt.npz 的图像
        all_samples = []
        for ext in valid_exts:
            for img_path in glob.glob(os.path.join(data_dir, f"*{ext}")):
                json_path = os.path.splitext(img_path)[0] + ".json"
                basename = os.path.splitext(os.path.basename(img_path))[0]
                gt_path = os.path.join(gt_dir, f"{basename}_gt.npz")
                if os.path.exists(json_path) and os.path.exists(gt_path):
                    all_samples.append((img_path, gt_path))

        all_samples.sort()

        # 划分训练/验证集（与 LabeledDataset 共用同一函数，保证划分一致）
        selected = split_train_val_indices(len(all_samples), train_ratio, seed, split)
        self.samples = [all_samples[i] for i in selected]

        print(
            f"[{split}] BoundaryDataset: {len(self.samples)} images "
            f"(total {len(all_samples)})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        在线处理流程：
        1. 读取图像与 GT .npz
        2. Letterbox 变换
        3. 在线计算 EDT 权重图
        4. 数据增强（可选）
        5. 转张量

        Returns:
            dict: {
                "image": [3, H, W] float32 (0-1),
                "target": [2, H, W] float32
                       - target[0] = 语义掩码 (0/1)
                       - target[1] = 边界掩码 (0/1)
                "weight": [1, H, W] float32 边界权重图,
                "original_size": (H_orig, W_orig),
                "image_path": 图像路径,
            }
        """
        img_path, gt_path = self.samples[idx]

        # 1. 读取原始图像（BGR -> RGB）
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        # 2. 加载净化 GT
        gt_data = np.load(gt_path)
        semantic = gt_data["semantic"]      # [H, W] uint8 (0/1)
        boundary = gt_data["boundary"]      # [H, W] uint8 (0/1)

        # 3. Letterbox 变换（图像、语义、边界使用相同缩放参数）
        image_lb, scale, pad_h, pad_w = letterbox(image, self.image_size)
        semantic_lb, _, _, _ = letterbox_mask(semantic, self.image_size)
        boundary_lb, _, _, _ = letterbox_mask(boundary, self.image_size)

        # 4. 在线计算 EDT 边界权重图
        weight_lb = compute_boundary_weight(
            boundary_lb,
            scale_factor=self.boundary_scale_factor,
            weight_floor=self.boundary_weight_floor,
            weight_ceil=self.boundary_weight_ceil,
        )

        # 5. 局部随机裁剪（训练时，斩断空间记忆）
        if self.augment and self.crop_size > 0:
            image_lb, semantic_lb, boundary_lb, weight_lb = random_crop(
                image_lb, semantic_lb, boundary_lb, weight_lb, self.crop_size
            )

        # 6. 数据增强（仅训练时）
        if self.augment:
            image_lb, semantic_lb, boundary_lb, weight_lb = self._augment(
                image_lb, semantic_lb, boundary_lb, weight_lb
            )

        # 7. 转换为 PyTorch 张量
        image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1) / 255.0

        # 目标：2 通道 [semantic, boundary]
        semantic_tensor = torch.from_numpy(semantic_lb).float().unsqueeze(0)  # [1, H, W]
        boundary_tensor = torch.from_numpy(boundary_lb).float().unsqueeze(0)  # [1, H, W]
        target_tensor = torch.cat([semantic_tensor, boundary_tensor], dim=0)  # [2, H, W]

        # 权重图
        weight_tensor = torch.from_numpy(weight_lb).float().unsqueeze(0)  # [1, H, W]

        return {
            "image": image_tensor,
            "target": target_tensor,
            "weight": weight_tensor,
            "original_size": (h_orig, w_orig),
            "scale": scale,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "image_path": img_path,
        }

    def _augment(
        self, image: np.ndarray, semantic: np.ndarray,
        boundary: np.ndarray, weight: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """在线数据增强（同时对图像、语义、边界、权重做相同空间变换）。"""
        cfg = self.augment_config

        # 水平翻转
        if cfg.get("horizontal_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            semantic = np.ascontiguousarray(semantic[:, ::-1])
            boundary = np.ascontiguousarray(boundary[:, ::-1])
            weight = np.ascontiguousarray(weight[:, ::-1])

        # 垂直翻转
        if cfg.get("vertical_flip", False) and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[::-1, :])
            semantic = np.ascontiguousarray(semantic[::-1, :])
            boundary = np.ascontiguousarray(boundary[::-1, :])
            weight = np.ascontiguousarray(weight[::-1, :])

        # 随机旋转 90/180/270 度
        if cfg.get("rotation", False) and np.random.rand() < 0.5:
            k = np.random.choice([1, 2, 3])
            image = np.ascontiguousarray(np.rot90(image, k))
            semantic = np.ascontiguousarray(np.rot90(semantic, k))
            boundary = np.ascontiguousarray(np.rot90(boundary, k))
            weight = np.ascontiguousarray(np.rot90(weight, k))

        return image, semantic, boundary, weight


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义 collate 函数，处理变长元数据。"""
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    weights = torch.stack([item["weight"] for item in batch])
    return {
        "image": images,
        "target": targets,
        "weight": weights,
        "original_size": [item["original_size"] for item in batch],
        "scale": [item["scale"] for item in batch],
        "pad_h": [item["pad_h"] for item in batch],
        "pad_w": [item["pad_w"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }
