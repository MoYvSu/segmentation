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

SAM2_INPUT_NORMALIZATION_NONE = "legacy_none"
SAM2_INPUT_NORMALIZATION_IMAGENET = "imagenet_v1"
_SAM2_IMAGE_MEAN = (0.485, 0.456, 0.406)
_SAM2_IMAGE_STD = (0.229, 0.224, 0.225)


def canonical_sam2_input_normalization(mode: str | None) -> str:
    """把配置别名收敛为 checkpoint 中使用的稳定合同名。"""
    normalized_mode = str(mode or SAM2_INPUT_NORMALIZATION_NONE).strip().lower()
    aliases = {
        "none": SAM2_INPUT_NORMALIZATION_NONE,
        "legacy": SAM2_INPUT_NORMALIZATION_NONE,
        SAM2_INPUT_NORMALIZATION_NONE: SAM2_INPUT_NORMALIZATION_NONE,
        "imagenet": SAM2_INPUT_NORMALIZATION_IMAGENET,
        SAM2_INPUT_NORMALIZATION_IMAGENET: SAM2_INPUT_NORMALIZATION_IMAGENET,
    }
    if normalized_mode not in aliases:
        raise ValueError(
            "unsupported SAM2 input normalization: "
            f"{mode!r}; expected legacy_none or imagenet_v1"
        )
    return aliases[normalized_mode]


def normalize_sam2_trunk_input(
    image: torch.Tensor, mode: str = SAM2_INPUT_NORMALIZATION_NONE
) -> torch.Tensor:
    """按显式输入合同准备 Hiera trunk 输入，旧路径默认保持逐值不变。"""
    normalized_mode = canonical_sam2_input_normalization(mode)
    if normalized_mode == SAM2_INPUT_NORMALIZATION_NONE:
        return image
    mean = image.new_tensor(_SAM2_IMAGE_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(_SAM2_IMAGE_STD).view(1, 3, 1, 1)
    return (image - mean) / std


class SAM2Encoder(nn.Module):
    """
    冻结的 SAM 2 Image Encoder 提取器。

    通过 Hydra 配置加载 SAM 2 底座（仅 image_encoder 的 trunk 部分），显式冻结参数，
    并重写前向流以返回 Stage 1 至 Stage 4 的多尺度原始特征图列表。

    注意：本项目仅使用 SAM 2 的 Hiera trunk 提取多尺度特征，不使用
    SAM 2 原生的 FpnNeck / Memory Attention / Memory Encoder / SAM Mask Decoder 等。
    通道对齐与特征融合由自制的 FPN 解码头完成。
    """

    def __init__(
        self,
        config_file: str = "configs/sam2/sam2_hiera_b+.yaml",
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        freeze: bool = True,
        sam2_repo_path: Optional[str] = None,
        input_normalization: str = SAM2_INPUT_NORMALIZATION_NONE,
    ):
        """
        Args:
            config_file: SAM 2 Hydra 配置文件名（相对 sam2 包 configs 目录）。
            ckpt_path: SAM 2 权重文件路径（位于项目 weights/ 目录下）。
            device: 加载设备。
            freeze: 是否冻结参数（默认 True）。
            sam2_repo_path: 本地 segment-anything-2 仓库路径，用于 import sam2。
            input_normalization: Hiera 输入合同。旧 checkpoint 使用 ``legacy_none``；
                新实验可显式选择 SAM2 官方的 ``imagenet_v1``。
        """
        super().__init__()
        self.config_file = config_file
        self.ckpt_path = ckpt_path
        self.device = device
        self.freeze = freeze
        self.input_normalization = canonical_sam2_input_normalization(
            input_normalization
        )
        # LoRA 注入后置为 True（models/lora.py 注入时设置）：
        # 允许 trunk 前向保留梯度，使 LoRA 参数可训练
        self.trainable_lora = False

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
        # Hiera 的 channel_list 按低分辨率到高分辨率保存，而 trunk.forward
        # 按 Stage 1 -> Stage 4（高分辨率到低分辨率）返回特征。
        # 从实际底座推导通道，避免把 Base+ 的 [112,224,448,896]
        # 错用于 Large 的 [144,288,576,1152]。
        self.stage_channels = list(reversed(self.trunk.channel_list))

        # 显式冻结
        if self.freeze:
            self._freeze_encoder()

        logger.info(
            "SAM 2 Hiera trunk 加载并冻结完成，input_normalization=%s。",
            self.input_normalization,
        )

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
            从高分辨率到低分辨率；通道数由实际 Hiera 配置决定。
        """
        # 确保 trunk 处于 eval 模式
        self.trunk.eval()

        trunk_input = normalize_sam2_trunk_input(x, self.input_normalization)
        if self.trainable_lora:
            # LoRA 注入后必须保留梯度（仅 LoRA 参数 requires_grad=True）
            features = self.trunk(trunk_input)
        else:
            with torch.no_grad():
                # trunk.forward 返回 List[Tensor]，顺序为 Stage1 -> Stage4
                features = self.trunk(trunk_input)

        return features

    def get_stage_channels(self) -> List[int]:
        """返回各 Stage 输出通道数（从高分辨率 Stage 1 到低分辨率 Stage 4）。"""
        return list(self.stage_channels)

    def param_count(self) -> int:
        """返回 encoder 的参数总量。"""
        return sum(p.numel() for p in self.trunk.parameters())
