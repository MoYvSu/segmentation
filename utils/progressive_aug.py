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

增强策略：
- ``legacy_all``：保留历史亮度/对比度/锐度/噪声串联行为。
- ``physical_v1``：每次只抽取 1~2 种显微成像退化，保留干净样本，支持
  失焦、降采样、曝光/白平衡、低频照明与低对比度抛光划痕 hard negative。

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

        self.policy = config.get("policy", "legacy_all")
        if self.policy not in {"legacy_all", "physical_v1"}:
            raise ValueError(
                f"Unsupported progressive augmentation policy: {self.policy!r}"
            )
        self.min_ops = max(1, int(config.get("min_ops", 1)))
        self.max_ops = max(self.min_ops, int(config.get("max_ops", 2)))
        self.op_weights = dict(config.get("op_weights", {}))
        self.gamma_range = config.get("gamma_range", [0.75, 1.35])
        self.white_balance_range = config.get("white_balance_range", [0.92, 1.08])
        self.blur_sigma_range = config.get("blur_sigma_range", [0.3, 2.0])
        self.blur_kernel_size = int(config.get("blur_kernel_size", 9))
        self.downsample_scale_range = config.get(
            "downsample_scale_range", [0.5, 0.95]
        )
        self.illumination_strength = float(
            config.get("illumination_strength", 0.15)
        )
        self.illumination_grid = max(2, int(config.get("illumination_grid", 3)))
        self.scratch_count_range = config.get("scratch_count_range", [1, 3])
        self.scratch_width_range = config.get("scratch_width_range", [0.6, 1.8])
        self.scratch_opacity_range = config.get(
            "scratch_opacity_range", [0.02, 0.10]
        )

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
        if self.policy == "physical_v1":
            return self._augment_physical(img)

        return self._augment_legacy(img)

    def _augment_legacy(self, img: torch.Tensor) -> torch.Tensor:
        """保留历史串联增强，保证旧配置与运行可复现。"""

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

    def _augment_physical(self, img: torch.Tensor) -> torch.Tensor:
        """抽取少量物理退化；不制造硬遮挡或规则黑斑。"""
        operations = {
            "gamma": self._op_gamma,
            "white_balance": self._op_white_balance,
            "gaussian_blur": self._op_gaussian_blur,
            "downsample": self._op_downsample,
            "illumination": self._op_illumination,
            "scratch": self._op_scratch,
            "noise": self._op_noise,
        }
        names = [
            name for name in operations
            if float(self.op_weights.get(name, 0.0)) > 0.0
        ]
        if not names:
            return img

        min_ops = min(self.min_ops, len(names))
        max_ops = min(self.max_ops, len(names))
        num_ops = int(torch.randint(
            min_ops,
            max_ops + 1,
            (1,), device=self.device,
        ).item())
        weights = torch.tensor(
            [float(self.op_weights[name]) for name in names],
            dtype=torch.float32,
            device=self.device,
        )
        selected = torch.multinomial(weights, num_ops, replacement=False).tolist()
        result = img
        for index in selected:
            result = operations[names[index]](result)
        return result.clamp(0.0, 1.0)

    def _sample_uniform(self, value_range) -> float:
        return torch.empty(1, device=self.device).uniform_(*value_range).item()

    def _op_gamma(self, img: torch.Tensor) -> torch.Tensor:
        gamma = self._sample_uniform(self.gamma_range)
        return img.clamp(1e-6, 1.0).pow(gamma)

    def _op_white_balance(self, img: torch.Tensor) -> torch.Tensor:
        low, high = self.white_balance_range
        gains = torch.empty(3, 1, 1, device=img.device, dtype=img.dtype).uniform_(
            low, high
        )
        # 保持平均曝光近似不变，只模拟显微光源/相机白平衡偏移。
        gains = gains / gains.mean().clamp_min(1e-6)
        return img * gains

    def _op_gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        sigma = self._sample_uniform(self.blur_sigma_range)
        return self._gaussian_blur(
            img, kernel_size=self.blur_kernel_size, sigma=sigma,
            padding_mode="reflect",
        )

    def _op_downsample(self, img: torch.Tensor) -> torch.Tensor:
        scale = self._sample_uniform(self.downsample_scale_range)
        h, w = img.shape[-2:]
        small_h = max(8, int(round(h * scale)))
        small_w = max(8, int(round(w * scale)))
        image_4d = img.unsqueeze(0)
        small = F.interpolate(
            image_4d, size=(small_h, small_w), mode="bilinear", align_corners=False
        )
        return F.interpolate(
            small, size=(h, w), mode="bicubic", align_corners=False
        ).squeeze(0)

    def _op_illumination(self, img: torch.Tensor) -> torch.Tensor:
        grid = torch.empty(
            1, 1, self.illumination_grid, self.illumination_grid,
            device=img.device, dtype=img.dtype,
        ).uniform_(-self.illumination_strength, self.illumination_strength)
        field = F.interpolate(
            grid, size=img.shape[-2:], mode="bicubic", align_corners=False
        ).squeeze(0)
        return img * (1.0 + field)

    def _op_scratch(self, img: torch.Tensor) -> torch.Tensor:
        """生成平滑低对比度划痕；GT 不变，使其成为边界 hard negative。"""
        h, w = img.shape[-2:]
        y = torch.arange(h, device=img.device, dtype=img.dtype).view(h, 1)
        x = torch.arange(w, device=img.device, dtype=img.dtype).view(1, w)
        low_count, high_count = [int(v) for v in self.scratch_count_range]
        count = int(torch.randint(
            low_count, high_count + 1, (1,), device=self.device
        ).item())
        result = img
        for _ in range(max(1, count)):
            angle = self._sample_uniform([0.0, 3.141592653589793])
            x0 = self._sample_uniform([0.0, float(max(1, w - 1))])
            y0 = self._sample_uniform([0.0, float(max(1, h - 1))])
            direction_x = torch.cos(torch.tensor(angle, device=img.device))
            direction_y = torch.sin(torch.tensor(angle, device=img.device))
            distance = torch.abs(
                (x - x0) * direction_y - (y - y0) * direction_x
            )
            width = self._sample_uniform(self.scratch_width_range)
            opacity = self._sample_uniform(self.scratch_opacity_range)
            line = torch.exp(-0.5 * (distance / max(width, 1e-3)) ** 2)
            # 明暗划痕各半，避免固定“黑线=非边界”的新捷径。
            sign = (
                -1.0
                if bool((torch.rand(1, device=self.device) < 0.5).item())
                else 1.0
            )
            result = result + sign * opacity * line.unsqueeze(0)
        return result

    def _op_noise(self, img: torch.Tensor) -> torch.Tensor:
        if self.gaussian_noise_std <= 0:
            return img
        return img + torch.randn_like(img) * self.gaussian_noise_std

    @staticmethod
    def _gaussian_blur(
        img: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0,
        padding_mode: str = "zeros",
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
        if padding_mode == "reflect":
            img_4d = F.pad(
                img_4d, (padding, padding, padding, padding), mode="reflect"
            )
            padding = 0
        blurred = F.conv2d(img_4d, gauss_2d, groups=3, padding=padding)
        return blurred.squeeze(0)  # [3, H, W]
