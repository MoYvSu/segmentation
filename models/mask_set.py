# -*- coding: utf-8 -*-
"""Mask2Former 式实例集合适配：随机 FPN、掩码引导查询与共享类别/掩码表示。

本模块不是官方 Mask2Former 的完整复刻：用现有 PyTorch FPN 代替可变形注意力
pixel decoder。仅复用合规 SAM2 Hiera trunk 和 SSL LoRA，不载入任何旧任务 decoder。
cross-attention 轮流读取 stride 32/16/8 特征；512 网格只用于最终 mask features。
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.direct_semantic_affinity import (
    _load_complete_lora_state,
    direct_input_normalization,
    load_direct_ssl_lora_state,
)
from models.lora import extract_lora_state_dict, inject_trunk_lora
from models.sam2_encoder import SAM2Encoder
from utils.config import project_path


MASK_SET_FORMAT = "mask_set_v1"
_ARCHITECTURE_DEFAULTS = {
    "input_size": 1024,
    "mask_grid": 512,
    "num_queries": 256,
    "hidden_dim": 256,
    "mask_dim": 128,
    "num_heads": 8,
    "decoder_layers": 6,
    "ffn_dim": 1024,
}


def _architecture_options(config: dict) -> dict:
    cfg = config.get("mask_set", {})
    options = {key: int(cfg.get(key, default)) for key, default in _ARCHITECTURE_DEFAULTS.items()}
    if any(value <= 0 for value in options.values()):
        raise ValueError("mask_set architecture dimensions must be positive")
    if options["hidden_dim"] % options["num_heads"]:
        raise ValueError("mask_set hidden_dim must be divisible by num_heads")
    if options["hidden_dim"] % 2:
        raise ValueError("mask_set hidden_dim must be even for learned 2D positions")
    return options


def _group_norm(channels: int) -> nn.GroupNorm:
    # 保证小尺寸 CPU smoke 的每组仍有多个通道；默认 256/128 通道均使用 32 组。
    groups = min(32, max(1, channels // 4))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class MaskSetPixelDecoder(nn.Module):
    """四尺度 top-down FPN；返回高分辨率掩码特征和三个低分辨率 memory。"""

    def __init__(self, in_channels, hidden_dim: int, mask_dim: int, mask_grid: int):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("mask_set requires four Hiera feature scales")
        self.mask_grid = int(mask_grid)
        self.lateral = nn.ModuleList([
            nn.Conv2d(int(channels), hidden_dim, 1) for channels in in_channels
        ])
        self.output = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                          _group_norm(hidden_dim), nn.ReLU(inplace=True))
            for _ in in_channels
        ])
        self.mask_projection = nn.Conv2d(hidden_dim, mask_dim, 1)
        self.mask_refinement = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, 3, padding=1, bias=False),
            _group_norm(mask_dim), nn.ReLU(inplace=True),
        )

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("mask_set encoder returned an unexpected number of scales")
        pyramid = [None] * 4
        merged = None
        for index in range(3, -1, -1):
            lateral = self.lateral[index](features[index])
            merged = lateral if merged is None else lateral + F.interpolate(
                merged, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid[index] = self.output[index](merged)
        mask_features = self.mask_projection(pyramid[0])
        mask_features = F.interpolate(
            mask_features, size=(self.mask_grid, self.mask_grid),
            mode="bilinear", align_corners=False,
        )
        mask_features = self.mask_refinement(mask_features)
        return mask_features, [pyramid[3], pyramid[2], pyramid[1]]


class MaskedQueryLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.ReLU(),
                                 nn.Linear(ffn_dim, hidden_dim))

    def forward(self, query, query_position, memory, memory_position, attention_mask):
        update = self.cross_attention(
            query + query_position, memory + memory_position, memory,
            attn_mask=attention_mask, need_weights=False,
        )[0]
        query = self.cross_norm(query + update)
        positioned = query + query_position
        update = self.self_attention(positioned, positioned, query, need_weights=False)[0]
        query = self.self_norm(query + update)
        return self.ffn_norm(query + self.ffn(query))


class MaskSetDecoder(nn.Module):
    """每个 query 同时生成两相类别/no-object 与一个完整实例掩码。"""

    def __init__(self, in_channels, config: dict):
        super().__init__()
        cfg = _architecture_options(config)
        self.options = cfg
        hidden = cfg["hidden_dim"]
        self.pixel_decoder = MaskSetPixelDecoder(
            in_channels, hidden, cfg["mask_dim"], cfg["mask_grid"]
        )
        self.query_features = nn.Embedding(cfg["num_queries"], hidden)
        self.query_positions = nn.Embedding(cfg["num_queries"], hidden)
        self.level_positions = nn.Embedding(3, hidden)
        spatial_size = max(1, math.ceil(cfg["input_size"] / 8))
        self.row_positions = nn.Embedding(spatial_size, hidden // 2)
        self.column_positions = nn.Embedding(spatial_size, hidden // 2)
        self.layers = nn.ModuleList([
            MaskedQueryLayer(hidden, cfg["num_heads"], cfg["ffn_dim"])
            for _ in range(cfg["decoder_layers"])
        ])
        self.decoder_norm = nn.LayerNorm(hidden)
        self.class_head = nn.Linear(hidden, 3)
        self.mask_embedding = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, cfg["mask_dim"]),
        )
        self.reset_parameters()

    def reset_parameters(self):
        # 所有任务模块随机初始化；不依赖任何分割任务权重。
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 对齐官方 M2F 的 query_feat/query_embed 默认 N(0,1)：过小的
                # query 残差会被第一层共享 cross-attention 更新淹没。这里只改
                # 两张查询表的冷启动尺度；level/空间位置与 checkpoint 结构不变。
                query_embedding = module is self.query_features or module is self.query_positions
                nn.init.normal_(module.weight, std=1.0 if query_embedding else 0.02)

    def _memory_position(self, height: int, width: int, dtype):
        if height > self.row_positions.num_embeddings or width > self.column_positions.num_embeddings:
            raise ValueError("mask_set memory exceeds the configured input_size")
        rows = self.row_positions.weight[:height]
        columns = self.column_positions.weight[:width]
        position = torch.cat([
            columns.unsqueeze(0).expand(height, -1, -1),
            rows.unsqueeze(1).expand(-1, width, -1),
        ], dim=-1)
        return position.reshape(1, height * width, -1).to(dtype=dtype)

    @staticmethod
    def make_attention_mask(mask_embeddings, lowres_mask_features, num_heads: int):
        """True 表示屏蔽；全被屏蔽的 query 放开 memory，避免 attention 产生 NaN。"""
        with torch.no_grad():
            coarse_logits = torch.einsum(
                "bqc,bchw->bqhw", mask_embeddings.detach(), lowres_mask_features.detach()
            )
            blocked = coarse_logits.flatten(2) < 0.0
            blocked = blocked & ~blocked.all(dim=-1, keepdim=True)
            return blocked.repeat_interleave(int(num_heads), dim=0)

    def forward(self, features):
        mask_features, levels = self.pixel_decoder(features)
        batch = mask_features.shape[0]
        query = self.query_features.weight.unsqueeze(0).expand(batch, -1, -1)
        query_position = self.query_positions.weight.unsqueeze(0)
        memories, positions, coarse_mask_features = [], [], []
        for index, level in enumerate(levels):
            height, width = level.shape[-2:]
            memory = level.flatten(2).transpose(1, 2)
            memory = memory + self.level_positions.weight[index].view(1, 1, -1)
            memories.append(memory)
            positions.append(self._memory_position(height, width, memory.dtype))
            # mask-guided attention 不反传阈值；先降采样特征，避免生成额外的512初始mask。
            with torch.no_grad():
                coarse_mask_features.append(F.interpolate(
                    mask_features.detach(), size=(height, width),
                    mode="bilinear", align_corners=False,
                ))
        # 初始投影也必须在正常梯度模式调用：外层 AMP 会缓存 Linear 的低精度
        # 权重；若首次调用放在 no_grad 中，后续预测可能复用无梯度缓存，使整个
        # mask MLP 意外冻结。只 detach 初始结果，仍不对 attention 阈值反传。
        mask_embeddings = self.mask_embedding(self.decoder_norm(query)).detach()
        predictions = []
        for index, layer in enumerate(self.layers):
            level = index % 3
            attention_mask = self.make_attention_mask(
                mask_embeddings, coarse_mask_features[level], self.options["num_heads"]
            )
            query = layer(query, query_position, memories[level], positions[level], attention_mask)
            normalized = self.decoder_norm(query)
            mask_embeddings = self.mask_embedding(normalized)
            predictions.append({
                "pred_logits": self.class_head(normalized),
                "pred_masks": torch.einsum("bqc,bchw->bqhw", mask_embeddings, mask_features),
            })
        return {**predictions[-1], "aux_outputs": predictions[:-1]}


class MaskSetModel(nn.Module):
    def __init__(self, encoder, decoder: MaskSetDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image):
        expected = self.decoder.options["input_size"]
        if image.ndim != 4 or image.shape[1] != 3 or tuple(image.shape[-2:]) != (expected, expected):
            raise ValueError(f"mask_set expects [B,3,{expected},{expected}] letterbox input")
        return self.decoder(self.encoder(image))

    def parameter_summary(self):
        summary = {
            "encoder": sum(value.numel() for value in self.encoder.parameters()),
            "decoder": sum(value.numel() for value in self.decoder.parameters()),
            "trainable": sum(value.numel() for value in self.parameters() if value.requires_grad),
        }
        summary["total"] = summary["encoder"] + summary["decoder"]
        summary["total_M"] = summary["total"] / 1.0e6
        summary["constraint_passed"] = summary["total"] < 500_000_000
        return summary


def _build_mask_set_architecture(config: dict, device):
    _architecture_options(config)
    sam2_cfg, lora_cfg, paths = config["sam2"], config["lora"], config["paths"]
    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=project_path(config, paths["weights_dir"], paths["sam2_ckpt"]),
        device=str(device), freeze=True,
        sam2_repo_path=project_path(config, sam2_cfg["sam2_repo_path"]),
        input_normalization=direct_input_normalization(config),
    )
    injected = inject_trunk_lora(
        encoder, rank=int(lora_cfg.get("rank", 16)), alpha=float(lora_cfg.get("alpha", 32.0)),
        target_layers=lora_cfg.get("target_layers"),
        use_grad_checkpoint=bool(lora_cfg.get("gradient_checkpointing", True)),
    )
    if injected <= 0:
        raise RuntimeError("mask_set injected no SSL LoRA layers")
    model = MaskSetModel(encoder, MaskSetDecoder(encoder.get_stage_channels(), config)).to(device)
    if not model.parameter_summary()["constraint_passed"]:
        raise RuntimeError("mask_set model must contain fewer than 500M parameters")
    configure_mask_set_phase(model, train_lora=False)
    return model


def build_mask_set_model(config: dict, device):
    model = _build_mask_set_architecture(config, device)
    ssl_path = Path(project_path(config, config["lora"]["init_from"]))
    state, normalization = load_direct_ssl_lora_state(ssl_path, config)
    loaded = _load_complete_lora_state(model, state)
    return model, {"ssl_lora_path": str(ssl_path.resolve()), "loaded_tensors": loaded,
                   "input_normalization": normalization, "decoder_initialization": "random"}


def configure_mask_set_phase(model, train_lora: bool):
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.encoder.trunk.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            parameter.requires_grad_(bool(train_lora))
    model.encoder.trainable_lora = bool(train_lora)
    model.encoder.eval()
    for parameter in model.decoder.parameters():
        parameter.requires_grad_(True)


def mask_set_parameter_groups(model, config: dict, train_lora: bool):
    cfg = config["mask_set"].get("learning_rates", {})
    decoder_key = "joint_decoder" if train_lora else "warmup_decoder"
    decoder_lr = float(cfg.get(decoder_key, 2.0e-5 if train_lora else 1.0e-4))
    groups = [{"name": "decoder", "params": [p for p in model.decoder.parameters() if p.requires_grad],
               "lr": decoder_lr}]
    if train_lora:
        lora = [p for name, p in model.encoder.trunk.named_parameters()
                if ("lora_A" in name or "lora_B" in name) and p.requires_grad]
        groups.append({"name": "lora", "params": lora, "lr": float(cfg.get("joint_lora", 2.0e-6))})
    if any(not group["params"] or not math.isfinite(group["lr"]) or group["lr"] <= 0 for group in groups):
        raise ValueError("mask_set optimizer groups require trainable parameters and positive finite LRs")
    return groups


def make_mask_set_checkpoint(model, config: dict, **metadata):
    reserved = {"format", "config", "decoder_state_dict", "lora_state_dict", "input_normalization"}
    if reserved & set(metadata):
        raise ValueError(f"mask_set checkpoint metadata overrides reserved keys: {sorted(reserved & set(metadata))}")
    if direct_input_normalization(config) != model.encoder.input_normalization:
        raise RuntimeError("mask_set checkpoint input_normalization differs from the model")
    return {
        "format": MASK_SET_FORMAT,
        "config": copy.deepcopy(config),
        "input_normalization": direct_input_normalization(config),
        "decoder_state_dict": {key: value.detach().cpu().clone() for key, value in model.decoder.state_dict().items()},
        "lora_state_dict": extract_lora_state_dict(model.encoder),
        **metadata,
    }


def load_mask_set_model(path, config: dict, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != MASK_SET_FORMAT:
        raise RuntimeError("unsupported mask_set checkpoint format")
    stored = payload.get("config")
    if not isinstance(stored, dict) or "mask_set" not in stored:
        raise RuntimeError("mask_set checkpoint is missing its architecture config")
    normalization = direct_input_normalization(stored)
    if payload.get("input_normalization") != normalization or direct_input_normalization(config) != normalization:
        raise RuntimeError("mask_set checkpoint input_normalization mismatch")
    stored_options = _architecture_options(stored)
    for key, value in _architecture_options(config).items():
        if key in config.get("mask_set", {}) and value != stored_options[key]:
            raise RuntimeError(f"mask_set runtime architecture differs from checkpoint: {key}")
    architecture = copy.deepcopy(stored)
    for key in ("project_root", "weights_dir", "sam2_ckpt"):
        architecture["paths"][key] = config["paths"][key]
    architecture["sam2"]["sam2_repo_path"] = config["sam2"]["sam2_repo_path"]
    architecture["lora"]["gradient_checkpointing"] = False
    model = _build_mask_set_architecture(architecture, device)
    model.decoder.load_state_dict(payload["decoder_state_dict"], strict=True)
    _load_complete_lora_state(model, payload["lora_state_dict"])
    model.eval()
    model.encoder.trainable_lora = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload
