# -*- coding: utf-8 -*-
"""
主动学习与反向网关
==================
1. 不确定性采样：基于信息熵或边界响应方差的采样逻辑，记录筛选记录。
2. Mask to JSON 矢量反向网关：将模型推理出的原图尺寸三分类掩码转换为
   有序坐标点，遵循 Labelme 官方 JSON Schema 写出为矢量文件，用于人工微调。
"""

import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class UncertaintySampler:
    """
    不确定性采样器。

    支持两种策略：
    1. entropy: 基于像素级预测的信息熵
    2. boundary_variance: 基于边界区域响应方差
    """

    def __init__(
        self,
        strategy: str = "entropy",
        sample_size: int = 10,
        entropy_threshold: float = 0.5,
        boundary_variance_threshold: float = 0.1,
        log_path: Optional[str] = None,
    ):
        """
        Args:
            strategy: 采样策略 ("entropy" / "boundary_variance")
            sample_size: 每轮采样数量
            entropy_threshold: 信息熵阈值
            boundary_variance_threshold: 边界方差阈值
            log_path: 筛选记录 CSV 路径
        """
        self.strategy = strategy
        self.sample_size = sample_size
        self.entropy_threshold = entropy_threshold
        self.boundary_variance_threshold = boundary_variance_threshold
        self.log_path = log_path

        # 初始化日志
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            if not os.path.exists(log_path):
                with open(log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "image_path", "strategy",
                        "uncertainty_score", "selected"
                    ])

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算像素级预测的信息熵。

        H(p) = -sum(p * log(p))

        Args:
            logits: [B, C, H, W] 模型输出 logits

        Returns:
            entropy_map: [B, H, W] 信息熵图
        """
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy

    def compute_boundary_variance(
        self, logits: torch.Tensor, boundary_class: int = 2
    ) -> torch.Tensor:
        """
        计算边界区域响应方差。

        对晶界类别（类别2）的预测概率图计算局部方差，
        方差越大表示边界越不确定。

        Args:
            logits: [B, C, H, W] 模型输出 logits
            boundary_class: 边界类别索引（默认 2=晶界）

        Returns:
            variance_map: [B, H, W] 边界方差图
        """
        probs = F.softmax(logits, dim=1)
        boundary_prob = probs[:, boundary_class:boundary_class + 1]  # [B, 1, H, W]

        # 使用 5x5 滑动窗口计算局部方差
        kernel_size = 5
        padding = kernel_size // 2

        # 计算局部均值
        mean = F.avg_pool2d(
            boundary_prob, kernel_size=kernel_size, stride=1, padding=padding
        )
        # 计算局部方差 = E[x^2] - E[x]^2
        mean_sq = F.avg_pool2d(
            boundary_prob ** 2, kernel_size=kernel_size, stride=1, padding=padding
        )
        variance = mean_sq - mean ** 2
        variance = variance.squeeze(1)  # [B, H, W]

        return variance

    def compute_uncertainty(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        根据策略计算不确定性图。

        Args:
            logits: [B, C, H, W] 模型输出 logits

        Returns:
            uncertainty: [B, H, W] 不确定性图
        """
        if self.strategy == "entropy":
            return self.compute_entropy(logits)
        elif self.strategy == "boundary_variance":
            return self.compute_boundary_variance(logits)
        else:
            raise ValueError(f"未知采样策略: {self.strategy}")

    def sample(
        self,
        image_paths: List[str],
        logits_list: List[torch.Tensor],
    ) -> List[Dict]:
        """
        对一批推理结果进行不确定性采样，选出最需要标注的样本。

        Args:
            image_paths: 图像路径列表
            logits_list: 每张图像对应的 logits 张量列表 [C, H, W]

        Returns:
            selected: 选中的样本列表，每个元素为
                      {"image_path": str, "uncertainty_score": float}
        """
        scores = []
        for img_path, logits in zip(image_paths, logits_list):
            if logits.ndim == 3:
                logits = logits.unsqueeze(0)  # [1, C, H, W]

            uncertainty = self.compute_uncertainty(logits)

            # 计算图像级不确定性分数（均值）
            score = uncertainty.mean().item()
            scores.append({"image_path": img_path, "uncertainty_score": score})

        # 按不确定性降序排序
        scores.sort(key=lambda x: x["uncertainty_score"], reverse=True)

        # 根据阈值筛选
        threshold = (
            self.entropy_threshold
            if self.strategy == "entropy"
            else self.boundary_variance_threshold
        )
        qualified = [
            s for s in scores if s["uncertainty_score"] >= threshold
        ]

        # 取前 sample_size 个
        selected = qualified[: self.sample_size]

        # 记录日志
        if self.log_path:
            self._log_samples(scores, selected)

        return selected

    def _log_samples(
        self, all_samples: List[Dict], selected: List[Dict]
    ):
        """记录采样日志到 CSV。"""
        selected_paths = {s["image_path"] for s in selected}
        timestamp = datetime.now().isoformat()

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for s in all_samples:
                writer.writerow([
                    timestamp,
                    s["image_path"],
                    self.strategy,
                    f"{s['uncertainty_score']:.6f}",
                    s["image_path"] in selected_paths,
                ])


def mask_to_labelme_json(
    mask: np.ndarray,
    image_path: str,
    output_json_path: str,
    min_contour_area: int = 50,
    approx_epsilon: float = 2.0,
) -> str:
    """
    Mask to JSON 矢量反向网关。

    利用 cv2.findContours 将模型推理出的原图尺寸三分类掩码转换为
    有序坐标点，遵循 Labelme 官方 JSON Schema 写出为矢量文件。

    Args:
        mask: [H, W] 三分类掩码 (0=珠光体, 1=铁素体核, 2=晶界)
        image_path: 原始图像路径（用于获取图像尺寸信息）
        output_json_path: 输出 JSON 文件路径
        min_contour_area: 最小轮廓面积过滤（过滤噪点）
        approx_epsilon: 轮廓近似精度（Douglas-Peucker 算法）

    Returns:
        output_json_path: 输出的 JSON 文件路径
    """
    # 读取图像尺寸
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        h, w = image.shape[:2]
    else:
        h, w = mask.shape[:2]

    # 类别标签映射
    label_map = {0: "pearlite", 1: "ferrite", 2: "grain_boundary"}

    shapes = []

    for class_id, label in label_map.items():
        # 提取当前类别的二值掩码
        binary = (mask == class_id).astype(np.uint8)

        if binary.sum() == 0:
            continue

        # 查找轮廓
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # 面积过滤
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue

            # 轮廓近似（减少点数）
            if approx_epsilon > 0:
                peri = cv2.arcLength(contour, True)
                contour = cv2.approxPolyDP(contour, approx_epsilon * peri * 0.01, True)

            # 转换为 [x, y] 坐标点列表
            points = contour.reshape(-1, 2).astype(float).tolist()

            if len(points) < 3:
                continue

            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            }
            shapes.append(shape)

    # 构建 Labelme JSON
    labelme_json = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }

    # 写出 JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_json, f, ensure_ascii=False, indent=2)

    return output_json_path


def generate_pseudo_labels(
    model,
    image_paths: List[str],
    output_dir: str,
    device: str = "cuda",
    image_size: int = 1024,
) -> List[str]:
    """
    伪标签生成：对未标注图像进行推理，生成 Labelme JSON 伪标签。

    Args:
        model: 训练好的分割模型
        image_paths: 待标注图像路径列表
        output_dir: 伪标签输出目录
        device: 推理设备
        image_size: Letterbox 目标尺寸

    Returns:
        generated_jsons: 生成的 JSON 文件路径列表
    """
    from data.dataset import letterbox

    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    generated_jsons = []

    for img_path in image_paths:
        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"警告: 无法读取图像 {img_path}，跳过")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # Letterbox
        image_lb, scale, pad_h, pad_w = letterbox(image_rgb, image_size)

        # 转张量
        image_tensor = (
            torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        image_tensor = image_tensor.to(device)

        # 推理（动态上采样到原图尺寸）
        with torch.no_grad():
            logits = model(image_tensor, output_size=(h_orig, w_orig))
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # 转换为 Labelme JSON
        basename = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(output_dir, f"{basename}.json")
        mask_to_labelme_json(pred_mask, img_path, json_path)
        generated_jsons.append(json_path)

        print(f"生成伪标签: {json_path}")

    return generated_jsons