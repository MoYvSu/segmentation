# -*- coding: utf-8 -*-
"""
LoRA 注入模块（冻结 SAM2 Hiera trunk 的参数高效适配）
=====================================================
参考范式：LoRA (Hu et al. 2021) / Conv-LoRA (2024) / TopoLoRA-SAM (2026)，
针对薄结构 + 跨域（暗域金相）任务，在冻结 trunk 的注意力投影上注入低秩适配器，
让语义/边界两个分支共享域适配后的特征，突破"冻结 encoder 特征上限"。

关键点：
1. 注入后 trunk 仅 LoRA 参数可训练（requires_grad=True），其余保持冻结。
2. SAM2Encoder 需将 forward 中的 torch.no_grad() 放开（见 sam2_encoder.py 的
   trainable_lora 标志），否则 LoRA 无梯度。
3. LoRA 参数独立成组，随 checkpoint 的 lora_state_dict 保存/加载。

用法：
    from models.lora import inject_trunk_lora
    n = inject_trunk_lora(encoder, rank=16, alpha=32,
                          target_layers=["attn.qkv", "attn.proj"])
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """对 nn.Linear 的 LoRA 包装：冻结原权重，新增 A/B 低秩分支。

    forward(x) = linear(x) + (alpha / rank) * (x @ A^T @ B^T)
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.linear = linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)

        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.rank))
        # LoRA 论文初始化：A 用 Kaiming uniform，B 置零（初始输出=原网络）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # 冻结原层参数（LoRA 之外不允许微调 trunk）
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        # x 末尾两维为 [..., in_f]；@ A^T @ B^T 等价 LoRA 低秩增量
        out = out + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())
        return out


def enable_trunk_gradient_checkpointing(trunk: nn.Module) -> None:
    """用梯度检查点替换 Hiera trunk 的块循环 forward。

    训练（LoRA 反向传播）时逐 block 重计算激活，把 trunk 激活内存从
    O(num_blocks × activation) 降到 O(1 × activation)，从而可以开大 batch。
    仅对 forward 做替换，不影响 state_dict。推理（无梯度）不启用。
    """
    import torch.utils.checkpoint as ckpt

    def forward_with_ckpt(x: torch.Tensor):
        x = trunk.patch_embed(x)
        x = x + trunk._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(trunk.blocks):
            x = ckpt.checkpoint(blk, x, use_reentrant=False)
            if (i == trunk.stage_ends[-1]) or (
                i in trunk.stage_ends and trunk.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs

    trunk.forward = forward_with_ckpt
    logger.info("LoRA: trunk 梯度检查点已启用（可开大 batch）")


def inject_trunk_lora(
    encoder: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_layers=None,
    use_grad_checkpoint: bool = True,
) -> int:
    """向 encoder.trunk 注入 LoRA，返回注入层数。

    Args:
        encoder: SAM2Encoder 实例（含 .trunk）
        rank: LoRA 秩
        alpha: LoRA 缩放系数（scaling = alpha / rank）
        target_layers: 匹配模块名字符串列表，命中即注入；
            默认 ["attn.qkv", "attn.proj"]（注意力 qkv 融合投影 + 输出投影）
    """
    if target_layers is None:
        target_layers = ["attn.qkv", "attn.proj"]

    trunk = encoder.trunk
    targets = []
    for name, module in trunk.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_layers):
            targets.append((name, module))
    if not targets:
        logger.warning(f"LoRA: trunk 中未找到目标层 {target_layers}，注入 0 层")
        return 0

    for name, module in targets:
        # 替换父模块属性
        parent_name, _, attr = name.rpartition(".")
        parent = trunk if not parent_name else trunk.get_submodule(parent_name)
        lora_linear = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, attr, lora_linear)
        logger.info(f"LoRA injected: {name} (in={module.in_features}, out={module.out_features})")

    encoder.trainable_lora = True
    if use_grad_checkpoint:
        enable_trunk_gradient_checkpointing(trunk)
    logger.info(f"LoRA injected {len(targets)} layers (rank={rank}, alpha={alpha}, "
                f"scaling={alpha / max(rank, 1):.3f})")
    return len(targets)


def _get_trunk(model: nn.Module) -> nn.Module:
    """兼容 SegmentationModel（.encoder.trunk）与 SAM2Encoder（.trunk）。"""
    if hasattr(model, "encoder") and hasattr(model.encoder, "trunk"):
        return model.encoder.trunk
    if hasattr(model, "trunk"):
        return model.trunk
    raise AttributeError("模型既没有 .encoder.trunk 也没有 .trunk，无法定位 trunk")


def count_lora_params(encoder: nn.Module) -> int:
    """统计 trunk 中 LoRA 参数数量（lora_A + lora_B）。"""
    n = 0
    for name, p in _get_trunk(encoder).named_parameters():
        if "lora_A" in name or "lora_B" in name:
            n += p.numel()
    return n


def extract_lora_state_dict(model: nn.Module) -> dict:
    """提取 trunk 的 LoRA 参数字典（供 checkpoint 保存）。"""
    return {
        k: v.detach().clone()
        for k, v in _get_trunk(model).state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def load_lora_state_dict(model: nn.Module, state: dict) -> int:
    """把 lora_state_dict 载入 trunk（严格模式，返回载入的参数张量个数）。"""
    if not state:
        return 0
    cur = _get_trunk(model).state_dict()
    keys = [k for k in state if k in cur]
    if not keys:
        raise ValueError("lora_state_dict 与 trunk 参数名不匹配，请检查 LoRA 配置")
    missing = [k for k in state if k not in cur]
    if missing:
        raise ValueError(f"lora_state_dict 含未知参数: {missing[:5]}...")
    for k in keys:
        cur[k].copy_(state[k])
    return len(keys)


def load_lora_from_checkpoint(model: nn.Module, checkpoint: dict) -> bool:
    """推理用：若 checkpoint 含 lora_state_dict，按状态推断 rank 注入并加载。

    返回是否加载了 LoRA。用于 inference.py 加载含 LoRA 的检查点
    （rank/alpha 从 lora_state_dict 与 checkpoint.config 恢复）。
    """
    state = checkpoint.get("lora_state_dict")
    if not state:
        return False
    rank = None
    for k, v in state.items():
        if "lora_A" in k:
            rank = int(v.shape[0])
            break
    if rank is None:
        raise ValueError("lora_state_dict 中找不到 lora_A，无法推断 rank")
    cfg = checkpoint.get("config", {}).get("lora", {})
    alpha = float(cfg.get("alpha", 32.0)) if cfg else 32.0
    target_layers = cfg.get("target_layers") if cfg else None
    inject_trunk_lora(model.encoder, rank=rank, alpha=alpha,
                      target_layers=target_layers, use_grad_checkpoint=False)
    n = load_lora_state_dict(model, state)
    logger.info(f"LoRA loaded from checkpoint: rank={rank}, alpha={alpha}, tensors={n}")
    return True
