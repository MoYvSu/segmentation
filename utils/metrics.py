# -*- coding: utf-8 -*-
"""
评估指标监控
=============
提供三分类分割任务的评估指标：
- 像素级准确率 (Pixel Accuracy)
- 平均准确率 (Mean Accuracy)
- 平均 IoU (Mean IoU)
- 频加权 IoU (Frequency Weighted IoU)
- Dice 系数
- 混淆矩阵

类别定义：0=珠光体, 1=铁素体核, 2=晶界
"""

from typing import Dict, Optional

import numpy as np
import torch


CLASS_NAMES = ["pearlite", "ferrite_core", "grain_boundary"]
NUM_CLASSES = 3


class SegMetrics:
    """
    分割评估指标累积器。

    在验证循环中累积混淆矩阵，最后计算各项指标。
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        """
        Args:
            num_classes: 类别数
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        更新混淆矩阵。

        Args:
            pred: [H, W] 预测掩码 (int)
            target: [H, W] 真实掩码 (int)
        """
        pred = pred.flatten().astype(np.int64)
        target = target.flatten().astype(np.int64)

        # 过滤无效像素
        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        # 计算混淆矩阵
        idx = target * self.num_classes + pred
        bincount = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape(self.num_classes, self.num_classes)

    def update_tensor(self, pred: torch.Tensor, target: torch.Tensor):
        """
        从 PyTorch 张量更新指标。

        Args:
            pred: [B, H, W] 或 [H, W] 预测张量 (long)
            target: [B, H, W] 或 [H, W] 真实张量 (long)
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        self.update(pred, target)

    def pixel_accuracy(self) -> float:
        """像素级准确率 = 对角线 / 总数"""
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def mean_accuracy(self) -> float:
        """平均准确率 = 各类别准确率的均值"""
        per_class_acc = self._per_class_accuracy()
        return np.mean(per_class_acc[~np.isnan(per_class_acc)])

    def _per_class_accuracy(self) -> np.ndarray:
        """每个类别的准确率"""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)

    def per_class_iou(self) -> np.ndarray:
        """每个类别的 IoU"""
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection = np.diag(self.confusion_matrix)
            union = (
                self.confusion_matrix.sum(axis=1)
                + self.confusion_matrix.sum(axis=0)
                - intersection
            )
            iou = intersection / union
        return iou

    def mean_iou(self) -> float:
        """平均 IoU"""
        iou = self.per_class_iou()
        return np.mean(iou[~np.isnan(iou)])

    def frequency_weighted_iou(self) -> float:
        """频加权 IoU"""
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iou = self.per_class_iou()
        valid = ~np.isnan(iou)
        return np.sum(freq[valid] * iou[valid])

    def per_class_dice(self) -> np.ndarray:
        """每个类别的 Dice 系数"""
        with np.errstate(divide="ignore", invalid="ignore"):
            intersection = np.diag(self.confusion_matrix)
            denom = (
                self.confusion_matrix.sum(axis=1)
                + self.confusion_matrix.sum(axis=0)
            )
            dice = 2 * intersection / denom
        return dice

    def mean_dice(self) -> float:
        """平均 Dice 系数"""
        dice = self.per_class_dice()
        return np.mean(dice[~np.isnan(dice)])

    def get_metrics(self) -> Dict:
        """获取所有指标的字典。"""
        iou = self.per_class_iou()
        dice = self.per_class_dice()
        acc = self._per_class_accuracy()

        metrics = {
            "pixel_accuracy": self.pixel_accuracy(),
            "mean_accuracy": self.mean_accuracy(),
            "mean_iou": self.mean_iou(),
            "fw_iou": self.frequency_weighted_iou(),
            "mean_dice": self.mean_dice(),
        }

        # 每个类别的指标
        for i, name in enumerate(CLASS_NAMES[: self.num_classes]):
            metrics[f"{name}_iou"] = float(iou[i]) if not np.isnan(iou[i]) else 0.0
            metrics[f"{name}_dice"] = float(dice[i]) if not np.isnan(dice[i]) else 0.0
            metrics[f"{name}_acc"] = float(acc[i]) if not np.isnan(acc[i]) else 0.0

        return metrics

    def reset(self):
        """重置混淆矩阵。"""
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def __str__(self) -> str:
        metrics = self.get_metrics()
        lines = ["=" * 60, "分割评估指标", "=" * 60]
        lines.append(f"Pixel Accuracy:  {metrics['pixel_accuracy']:.4f}")
        lines.append(f"Mean Accuracy:   {metrics['mean_accuracy']:.4f}")
        lines.append(f"Mean IoU:        {metrics['mean_iou']:.4f}")
        lines.append(f"FW IoU:          {metrics['fw_iou']:.4f}")
        lines.append(f"Mean Dice:       {metrics['mean_dice']:.4f}")
        lines.append("-" * 60)
        for name in CLASS_NAMES[: self.num_classes]:
            lines.append(
                f"  {name:15s}: IoU={metrics[f'{name}_iou']:.4f}  "
                f"Dice={metrics[f'{name}_dice']:.4f}  "
                f"Acc={metrics[f'{name}_acc']:.4f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict:
    """
    快速计算单张图像的评估指标。

    Args:
        pred: [H, W] 预测掩码
        target: [H, W] 真实掩码
        num_classes: 类别数

    Returns:
        metrics dict
    """
    meter = SegMetrics(num_classes)
    meter.update(pred, target)
    return meter.get_metrics()