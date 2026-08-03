# -*- coding: utf-8 -*-
"""
渐进式外观增强（Progressive Appearance Augmentation）
=====================================================
为第二阶段 Mean Teacher 训练提供循序渐进的外观增强，解决训练集与测试集
在清晰度、亮度等方面的域差异。

核心特性：
1. **学生独享**：仅对学生模型输入施加，教师模型始终收到干净数据
2. **渐进式概率**：增强概率随 epoch 线性增长，最终维持高位
3. **GPU 张量操作**：直接在 GPU 上运算，不增加 CPU→GPU 传输开销
4. **逐样本独立**：同一 batch 内每张图像独立采样增强参数

增强类型：
- 亮度抖动（Brightness jitter）
- 对比度抖动（Contrast jitter）
- 锐度抖动（Sharpness jitter，基于 unsharp mask）
- 高斯噪声（Gaussian noise）

使用方式：
    augmentor = ProgressiveAppearanceAug(config, device)
    for epoch in range(epochs):
        augmentor.set_epoch(epoch)
        for batch in loader:
            images = augmentor(images)  # 仅增强学生输入
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class ProgressiveAppearanceAug:
    """渐进式外观增强器。

    增强概率随 epoch 从 0 线性增长到 max_prob，之后维持 max_prob。
    每张图像独立决定是否增强，以及增强参数。
    """

    def __init__(self, config: Dict, device: torch.device):
        """初始化增强器。

        Args:
            config: progressive_aug 配置字典
            device: 计算设备
        """
        self.enabled = config.get("enabled", True)
        self.start_epoch = config.get("start_epoch", 0)
        self.ramp_epochs = config.get("ramp_epochs", 10)
        self.max_prob = config.get("max_prob", 0.8)

        self.brightness_range = config.get("brightness_range", [0.6, 1.4])
        self.contrast_range = config.get("contrast_range", [0.6, 1.4])
        self.sharpness_range = config.get("sharpness_range", [-0.5, 1.0])
        self.gaussian_noise_std = config.get("gaussian_noise_std", 0.01)

        self.device = device
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """设置当前 epoch，更新增强概率。"""
        self.current_epoch = epoch

    @property
    def current_prob(self) -> float:
        """计算当前增强概率。

        epoch < start_epoch:           prob = 0
        start_epoch <= epoch < ramp_end: prob = 线性增长 0 -> max_prob
        epoch >= ramp_end:              prob = max_prob
        """
        if not self.enabled:
            return 0.0
        if self.current_epoch < self.start_epoch:
            return 0.0
        ramp_end = self.start_epoch + self.ramp_epochs
        if self.current_epoch >= ramp_end:
            return self.max_prob
        progress = (self.current_epoch - self.start_epoch) / max(1, self.ramp_epochs)
        return self.max_prob * progress

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """对学生输入图像施加渐进式外观增强。

        Args:
            images: [B, 3, H, W] 张量，范围 [0, 1]

        Returns:
            增强后的张量，相同形状和范围
        """
        if not self.enabled or self.current_prob <= 0.0:
            return images

        prob = self.current_prob
        B = images.shape[0]

        # 为 batch 中每张图独立采样增强掩码
        aug_mask = torch.rand(B, device=self.device) < prob  # [B], True=增强

        if not aug_mask.any():
            return images

        result = images.clone()

        # 对需要增强的样本施加变换
        for i in range(B):
            if not aug_mask[i]:
                continue
            result[i] = self._augment_single(images[i])

        return result

    def _augment_single(self, img: torch.Tensor) -> torch.Tensor:
        """对单张图像 [3, H, W] 施加外观增强。

        每种增强独立采样是否应用，参数独立采样。
        """
        # img: [3, H, W], range [0, 1]

        # 1. 亮度抖动
        b_factor = torch.empty(1, device=self.device).uniform_(*self.brightness_range).item()
        img = img * b_factor

        # 2. 对比度抖动
        c_factor = torch.empty(1, device=self.device).uniform_(*self.contrast_range).item()
        mean = img.mean(dim=(1, 2), keepdim=True)  # [3, 1, 1]
        img = (img - mean) * c_factor + mean

        # 3. 锐度抖动（unsharp mask: sharpened = img + alpha * (img - blurred)）
        s_alpha = torch.empty(1, device=self.device).uniform_(*self.sharpness_range).item()
        if abs(s_alpha) > 1e-6:
            blurred = self._gaussian_blur(img, kernel_size=5, sigma=1.0)
            img = img + s_alpha * (img - blurred)

        # 4. 高斯噪声
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(img) * self.gaussian_noise_std
            img = img + noise

        # Clamp 到 [0, 1]
        return img.clamp(0.0, 1.0)

    @staticmethod
    def _gaussian_blur(
        img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
    ) -> torch.Tensor:
        """高斯模糊（用于锐度的 unsharp mask）。

        Args:
            img: [3, H, W]
            kernel_size: 卷积核大小（奇数）
            sigma: 高斯标准差

        Returns:
            模糊后的图像 [3, H, W]
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 构建高斯卷积核
        x = torch.arange(
            kernel_size, dtype=img.dtype, device=img.device
        ) - kernel_size // 2
        gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)  # [K, K]
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        gauss_2d = gauss_2d.expand(3, 1, -1, -1)  # [3, 1, K, K] - grouped conv

        # 卷积（depthwise）
        img_4d = img.unsqueeze(0)  # [1, 3, H, W]
        padding = kernel_size // 2
        blurred = F.conv2d(img_4d, gauss_2d, groups=3, padding=padding)
        return blurred.squeeze(0)  # [3, H, W]