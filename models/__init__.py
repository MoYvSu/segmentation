# -*- coding: utf-8 -*-
"""模型模块：冻结 SAM 2 Image Encoder + 自制轻量 FPN 解码头"""

from .sam2_encoder import SAM2Encoder
from .fpn_decoder import FPNDecoder, FPNBackbone, SegmentationModel
from .gda_mim import GDAMaskedAutoencoder, GenerativeDomainAdapterPyramid
from .edge_prior import GenerativeEdgePrior, FrozenEncoderEdgePrior

__all__ = [
    "SAM2Encoder",
    "FPNDecoder",
    "FPNBackbone",
    "SegmentationModel",
    "GDAMaskedAutoencoder",
    "GenerativeDomainAdapterPyramid",
    "GenerativeEdgePrior",
    "FrozenEncoderEdgePrior",
]
