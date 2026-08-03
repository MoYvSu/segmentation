# -*- coding: utf-8 -*-
"""
SAM 2 Image Encoder 提取器
========================
显式冻结 SAM 2 Image Encoder，重写前向流，返回 Stage 1 至 Stage 4 的多尺度特征图。

约束：
1. 禁止依赖 `~/.cache` 等全局隐式路径，权重必须从项目 `weights/` 目录加载。
2. 显式设置 `requires_grad=False`，冻结所有参数。
3. 不加载 SAM 2 原生 Mask Decoder 权重（本项目使用自制 FPN 解码头）。
"""

import logging
import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SAM2Encoder(nn.Module):
    """
    冻结的 SAM 2 Image Encoder 提取器。

    通过 Hydra 配置加载 SAM 2 底座（仅 image_encoder 的 trunk 部分），显式冻结参数，
    并重写前向流以返回 Stage 1 至 Stage 4 的多尺度原始特征图列表。

    注意：本项目仅使用 SAM 2 的 Hiera trunk 提取多尺度特征，不使用
    SAM 2 原生的 FpnNeck / Memory Attention / Memory Encoder / SAM Mask Decoder 等。
    通道对齐与特征融合由自制的 FPN 解码头完成。
    """

    # Hiera base+ 的四阶段输出通道数（从 Stage 1 到 Stage 4）
    # trunk.stage_ends 输出通道: [112, 224, 448, 896]
    # 对应分辨率（1024 输入，PatchEmbed stride=4，3 次 q_pool stride=2）：
    #   Stage 1: 256x256, 112ch
    #   Stage 2: 128x128, 224ch
    #   Stage 3:  64x64,  448ch
    #   Stage 4:  32x32,  896ch
    STAGE_CHANNELS = [112, 224, 448, 896]

    def __init__(
        self,
        config_file: str = "configs/sam2/sam2_hiera_b+.yaml",
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        freeze: bool = True,
        sam2_repo_path: Optional[str] = None,
    ):
        """
        Args:
            config_file: SAM 2 Hydra 配置文件名（相对 sam2 包 configs 目录）。
            ckpt_path: SAM 2 权重文件路径（位于项目 weights/ 目录下）。
            device: 加载设备。
            freeze: 是否冻结参数（默认 True）。
            sam2_repo_path: 本地 segment-anything-2 仓库路径，用于 import sam2。
        """
        super().__init__()
        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.device = device
        self.freeze = freeze

        # 将本地 segment-anything-2 仓库加入 sys.path，以 import sam2
        if sam2_repo_path is not None:
            sam2_abs_path = os.path.abspath(sam2_repo_path)
            if sam2_abs_path not in sys.path:
                sys.path.insert(0, sam2_abs_path)

        # 导入 sam2 构建工具
        try:
            from sam2.build_sam import build_sam2
        except ImportError as e:
            raise ImportError(
                f"无法导入 sam2 包。请确保 sam2_repo_path 正确: {sam2_repo_path}. "
                f"错误: {e}"
            )

        # 构建 SAM 2 完整模型（仅用于提取 image_encoder.trunk）
        # apply_postprocessing=False 避免添加不必要的后处理覆盖
        logger.info(
            "正在加载 SAM 2 底座 (config=%s, ckpt=%s)...", config_file, ckpt_path
        )
        sam2_model = build_sam2(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device=device,
            mode="eval",
            apply_postprocessing=False,
        )

        # 仅保留 Hiera trunk，丢弃 neck / memory_attention / memory_encoder /
        # sam_mask_decoder 等所有原生解码结构，彻底隔离 SAM 2 原生解码权重
        self.trunk = sam2_model.image_encoder.trunk

        # 显式冻结
        if self.freeze:
            self._freeze_encoder()

        logger.info("SAM 2 Hiera trunk 加载并冻结完成。")

    def _freeze_encoder(self):
        """显式冻结 trunk 的所有参数。"""
        for param in self.trunk.parameters():
            param.requires_grad = False
        self.trunk.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播，返回 Stage 1 至 Stage 4 的多尺度原始特征图。

        Hiera trunk 的 forward 返回 List[Tensor]，按 stage_ends 顺序：
            - outputs[0]: Stage 1 结束特征 (最高分辨率, 112ch)
            - outputs[1]: Stage 2 结束特征 (224ch)
            - outputs[2]: Stage 3 结束特征 (448ch)
            - outputs[3]: Stage 4 结束特征 (最低分辨率, 896ch)

        Args:
            x: 输入图像张量, shape [B, 3, 1024, 1024]

        Returns:
            List of 4 feature tensors [feat_s1, feat_s2, feat_s3, feat_s4]，
            从高分辨率到低分辨率，通道数 [112, 224, 448, 896]。
        """
        # 确保 trunk 处于 eval 模式且不计算梯度
        self.trunk.eval()

        with torch.no_grad():
            # trunk.forward 返回 List[Tensor]，顺序为 Stage1 -> Stage4
            features = self.trunk(x)

        return features

    def get_stage_channels(self) -> List[int]:
        """返回各 Stage 输出通道数（从高分辨率 Stage 1 到低分辨率 Stage 4）。"""
        return list(self.STAGE_CHANNELS)

    def param_count(self) -> int:
        """返回 encoder 的参数总量。"""
        return sum(p.numel() for p in self.trunk.parameters())