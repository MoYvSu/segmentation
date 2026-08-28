# -*- coding: utf-8 -*-
"""
第二阶段半监督微调主入口（边界预测版本 - 多模式伪标签源）
=========================================================
使用有标签数据（BoundaryLoss）与无标签数据（一致性损失）联合训练。

技术路线：
1. 加载第一阶段最优权重到学生模型
2. 可选创建教师模型（学生权重的 EMA 副本，仅 ema 模式需要）
3. 仅冻结 encoder，全量训练 decoder
4. 双流混合 Batch：有标签 DataLoader 耗尽后重新创建 iterator，不缓存旧 batch
5. 有标签流：BoundaryLoss（语义 BCE + 边界 Focal x EDT 权重）
6. 无标签流：一致性损失（MSE + Sobel + TV + 背景抑制）
   - boundary_teacher_mode 决定边界伪标签源：
     "ema": EMA 教师 + Stage-1 锚点混合（默认）
     "stage1_direct": Stage-1 冻结模型直接提供（无 EMA 滞后）
     "self_consistency": 学生弱增强预测 stop-gradient（无 EMA 依赖）
7. 每个 step 结束后可选更新教师模型 EMA 权重（仅 ema 模式）

使用方法：
    conda activate sam2_env
    python train_stage2.py --config config/default_config.yaml
"""

import argparse
import copy
import glob
import json
import logging
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import BoundaryDataset, collate_fn, letterbox
from data.dataset_semi import (
    LabeledDataset,
    UnlabeledDataset,
    labeled_collate_fn,
    unlabeled_collate_fn,
)
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.edge_prior import load_pretrained_edge_prior
from models.gda_mim import load_pretrained_gda
from models.lora import (
    count_lora_params,
    extract_lora_state_dict,
    inject_trunk_lora,
    load_lora_state_dict,
)
from models.sam2_encoder import SAM2Encoder
from utils.loss import BoundaryLoss
from utils.loss_semi import compute_unsupervised_loss, update_ema
from utils.metrics import SegMetrics
from utils.progressive_aug import ProgressiveAppearanceAug
from utils.run_recorder import RunRecorder
from utils.checkpoint import build_checkpoint, validate_checkpoint_architecture
from utils.config import load_config, project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False):
    """固定模型初始化与数据采样随机源，支持成对消融复现。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(
        f"Random seed: {seed} "
        f"(cudnn_deterministic={'on' if deterministic else 'off'})"
    )


def seed_dataloader_worker(worker_id: int):
    """使每个 DataLoader worker 的 NumPy/Python 随机流可复现。"""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_epoch_steps(
    labeled_loader_length: int,
    unlabeled_steps: int = 0,
    labeled_steps_per_epoch: int = 0,
) -> int:
    """独立解析监督训练步数，避免关闭无标签流时隐式缩短训练。"""
    labeled_steps = (
        int(labeled_steps_per_epoch)
        if int(labeled_steps_per_epoch) > 0
        else int(labeled_loader_length)
    )
    return max(labeled_steps, int(unlabeled_steps))


def next_restarting_batch(loader, iterator):
    """取下一 batch；耗尽时重建 iterator，而不是缓存并重放旧 batch。"""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def set_student_train_modes(student_model):
    """训练可学习模块，同时让所有完全冻结的子模块保持 eval 模式。"""
    student_model.train()
    student_model.encoder.eval()
    decoder = student_model.decoder
    modules = [
        decoder.seg_fpn,
        decoder.seg_branch,
        decoder.boundary_fpn,
        decoder.boundary_branch,
    ]
    if getattr(decoder, "semantic_residual", None) is not None:
        modules.append(decoder.semantic_residual)
    if getattr(decoder, "boundary_refine", False):
        modules.append(decoder.boundary_refine_head)
    if getattr(decoder, "center_branch", None) is not None:
        modules.append(decoder.center_branch)
    if getattr(decoder, "edge_prior_fusion", None) is not None:
        modules.append(decoder.edge_prior_fusion)
    for module in modules:
        params = list(module.parameters())
        if params and not any(param.requires_grad for param in params):
            module.eval()
    boundary_adapter = getattr(student_model, "boundary_adapter", None)
    if boundary_adapter is not None:
        # G1 freezes the pretrained convolutional adapters.  eval() does not
        # block gradients to the four gates, and prevents accidental stateful
        # layer updates if the adapter implementation changes later.
        if not any(p.requires_grad for p in boundary_adapter.adapters.parameters()):
            boundary_adapter.adapters.eval()
    edge_prior = getattr(student_model, "edge_prior", None)
    if edge_prior is not None:
        edge_prior.eval()


def snapshot_parameters(named_parameters):
    """把少量实验关注参数复制到 CPU，用于逐 epoch 变化量审计。"""
    return {
        name: param.detach().cpu().clone()
        for name, param in named_parameters
    }


def parameter_delta_stats(named_parameters, initial_state):
    """返回相对实验初始化的 L2 与最大绝对参数变化。"""
    squared_sum = 0.0
    max_abs = 0.0
    for name, param in named_parameters:
        if name not in initial_state:
            continue
        delta = param.detach().cpu().float() - initial_state[name].float()
        squared_sum += float(delta.square().sum())
        if delta.numel() > 0:
            max_abs = max(max_abs, float(delta.abs().max()))
    return math.sqrt(squared_sum), max_abs


def gradient_l2_norm(parameters) -> float:
    squared_sum = 0.0
    for param in parameters:
        if param.grad is not None:
            squared_sum += float(param.grad.detach().float().square().sum())
    return math.sqrt(squared_sum)


def sigmoid_rampup(epoch, rampup_epochs):
    """Sigmoid ramp-up 函数：从 0 平滑渐进到 1。

    用于无监督损失权重的渐进上坡，避免训练初期不可靠的伪标签产生过大影响。

    Args:
        epoch: 当前 epoch（从 0 开始）
        rampup_epochs: ramp-up 长度（达到此 epoch 时输出 ≈ 1.0）

    Returns:
        [0, 1] 之间的权重比例
    """
    if rampup_epochs <= 0:
        return 1.0
    if epoch >= rampup_epochs:
        return 1.0
    # sigmoid 居中缩放：在 rampup_epochs/2 处拐点，5 倍缩放使过渡平滑
    return float(math.exp(-5.0 * (1.0 - epoch / rampup_epochs) ** 2))


def compute_adaptive_ema_decay(base_decay, current_lr_ratio):
    """根据当前学习率比例计算自适应 EMA 衰减系数。

    LR 高时（学生变化快）：降低 decay → 教师更快跟随
    LR 低时（学生变化慢）：提高 decay → 教师更稳定

    策略：decay_eff = 1.0 - (1.0 - base_decay) * lr_ratio
    例: base_decay=0.999, lr_ratio=1.0 → 0.999（教师有 ~1000 步滞后）
        base_decay=0.999, lr_ratio=0.2 → 0.9998（教师有 ~5000 步滞后）

    Args:
        base_decay: 基础 EMA 衰减系数（如 0.999）
        current_lr_ratio: 当前 LR / 峰值 LR（[0, 1]）

    Returns:
        自适应 EMA 衰减系数
    """
    lr_ratio = max(0.0, min(1.0, current_lr_ratio))
    return 1.0 - (1.0 - base_decay) * lr_ratio


def get_current_lr_ratio(scheduler, group_peaks):
    """获取当前学习率相对于该参数组自身峰值学习率的比例。

    历史遗留问题修复：此前用全局 base_lr 作分母，语义阶段（冻结边界后
    唯一训练组是 seg 组，lr = seg_lr_ratio × base_lr）会被错误压成 0.1，
    使自适应 EMA decay 变成 0.9999，教师几乎冻结在初始化权重上。
    现在以每个参数组自身的峰值 lr 为分母，ratio 即调度器倍率 lambda，
    与分支冻结与否无关，flat 阶段稳定在 1.0 → decay = base_decay。

    Args:
        scheduler: LambdaLR 调度器
        group_peaks: 各参数组创建时的峰值 LR 列表（必须与 param_groups 对齐，
            且在调度器首次 step 前记录，因为 LambdaLR 会原地改写 group['lr']）

    Returns:
        [0, 1] 之间的比例
    """
    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    if group_peaks:
        peak_lr = max(float(group_peaks[0]), 1e-12)
    else:
        peak_lr = max(float(current_lr), 1e-12)
    return max(0.0, min(1.0, current_lr / peak_lr))


def build_model(config, device):
    """构建学生模型。"""
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]

    ckpt_path = project_path(config, paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    if not os.path.exists(ckpt_path):
        logger.warning(f"SAM 2 checkpoint not found: {ckpt_path}")

    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
        device=device,
        freeze=sam2_cfg["freeze"],
        sam2_repo_path=os.path.join(paths_cfg["project_root"], sam2_cfg["sam2_repo_path"]),
    )

    # LoRA：冻结 trunk 的参数高效适配（语义/边界共享域适配特征）
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        n_layers = inject_trunk_lora(
            encoder,
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 32.0),
            target_layers=lora_cfg.get("target_layers"),
        )
        n_params = count_lora_params(encoder)
        logger.info(
            f"LoRA: ENABLED ({n_layers} 层, 可训练参数 {n_params / 1e6:.2f}M)"
        )
        # 可选：加载自监督预训练 LoRA 状态（lora.init_from）
        init_from = lora_cfg.get("init_from", "")
        if init_from:
            init_from = project_path(config, init_from)
        if init_from and os.path.exists(init_from):
            st = torch.load(init_from, map_location=device, weights_only=False)
            n_load = load_lora_state_dict(encoder, st)
            logger.info(f"LoRA: 预训练状态已加载 {init_from} ({n_load} 个参数张量)")
        elif init_from:
            logger.warning(f"LoRA init_from 不存在: {init_from}")
        # 可选：冻结 LoRA（B1 校准实验——只训边界头，特征与 LoRA 均固定）
        if lora_cfg.get("freeze", False):
            n_frozen = sum(
                1 for n, p in encoder.trunk.named_parameters()
                if ("lora_A" in n or "lora_B" in n) and p.requires_grad
            )
            for n, p in encoder.trunk.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad_(False)
            logger.info(f"LoRA: FREEZED ({n_frozen} 个参数张量，共 {count_lora_params(encoder) / 1e6:.2f}M 元素，特征固定)")
    else:
        logger.info("LoRA: DISABLED (trunk 全冻结)")

    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=decoder_cfg["fpn_channels"],
        num_classes=decoder_cfg["num_classes"],
        dropout=decoder_cfg["dropout"],
        use_bn=decoder_cfg["use_bn"],
        boundary_refine=decoder_cfg.get("boundary_refine", False),
        boundary_refine_version=decoder_cfg.get(
            "boundary_refine_version", "legacy_lowres"
        ),
        center_head=decoder_cfg.get("center_head", False),
        edge_prior_fusion=decoder_cfg.get("edge_prior_fusion", False),
        edge_prior_fusion_hidden=decoder_cfg.get(
            "edge_prior_fusion_hidden", 32
        ),
        edge_prior_max_logit_delta=decoder_cfg.get(
            "edge_prior_max_logit_delta", 1.0
        ),
        semantic_residual=decoder_cfg.get("semantic_residual", False),
        semantic_residual_hidden=decoder_cfg.get("semantic_residual_hidden", 64),
        semantic_residual_color_channels=decoder_cfg.get(
            "semantic_residual_color_channels", 16
        ),
        semantic_residual_use_photometric=decoder_cfg.get(
            "semantic_residual_use_photometric", True
        ),
        semantic_residual_max_logit_delta=decoder_cfg.get(
            "semantic_residual_max_logit_delta", 2.0
        ),
    )

    boundary_adapter = None
    gda_cfg = config.get("gda", {})
    if gda_cfg.get("enabled", False):
        gda_checkpoint = project_path(config, gda_cfg.get("checkpoint", ""))
        if not gda_checkpoint or not os.path.exists(gda_checkpoint):
            raise FileNotFoundError(
                f"GDA checkpoint not found: {gda_checkpoint or '<empty>'}"
            )
        boundary_adapter = load_pretrained_gda(
            gda_checkpoint,
            channels=encoder.get_stage_channels(),
            bottleneck_ratio=int(gda_cfg.get("bottleneck_ratio", 8)),
            gate_mode=gda_cfg.get("gate_mode", "scalar"),
            active_scales=gda_cfg.get("active_scales"),
            map_location=device,
            freeze_adapters=bool(gda_cfg.get("freeze_adapters", True)),
            train_gates=bool(gda_cfg.get("train_gates", True)),
        )
        logger.info(
            "Boundary GDA: ENABLED (%s, gate_mode=%s, active_scales=%s, "
            "adapter_trainable=%s, gates_trainable=%s)",
            gda_checkpoint,
            boundary_adapter.gate_mode,
            boundary_adapter.active_scales,
            not bool(gda_cfg.get("freeze_adapters", True)),
            bool(gda_cfg.get("train_gates", True)),
        )

    edge_prior = None
    edge_prior_cfg = config.get("edge_prior", {})
    if edge_prior_cfg.get("enabled", False):
        prior_checkpoint = project_path(
            config, edge_prior_cfg.get("checkpoint", "")
        )
        if not prior_checkpoint or not os.path.exists(prior_checkpoint):
            raise FileNotFoundError(
                f"Edge-prior checkpoint not found: {prior_checkpoint or '<empty>'}"
            )
        edge_prior = load_pretrained_edge_prior(
            prior_checkpoint,
            in_channels=encoder.get_stage_channels()[:2],
            hidden_channels=int(edge_prior_cfg.get("hidden_channels", 64)),
            map_location=device,
        )
        logger.info("Retained G0b edge prior: ENABLED (%s)", prior_checkpoint)
    if decoder_cfg.get("edge_prior_fusion", False) and edge_prior is None:
        raise ValueError(
            "decoder.edge_prior_fusion=True requires edge_prior.enabled=True"
        )

    model = SegmentationModel(
        encoder, decoder, boundary_adapter=boundary_adapter,
        edge_prior=edge_prior,
    )
    model = model.to(device)
    return model


def build_teacher_model(student_model):
    """创建教师模型（学生权重的深拷贝）。"""
    teacher_model = copy.deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    return teacher_model


def load_base_checkpoint(model, checkpoint_path, device):
    """加载 Stage 2 初始化锚点（可来自 Stage 1 或已验证的 Stage 2）。"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    from models.fpn_decoder import load_decoder_state
    load_decoder_state(model.decoder, checkpoint["decoder_state_dict"],
                       tag=f"Stage-1({os.path.basename(checkpoint_path)})")
    if checkpoint.get("lora_state_dict"):
        from models.lora import load_lora_state_dict
        n_lora = load_lora_state_dict(model, checkpoint["lora_state_dict"])
        logger.info(f"Stage-1 LoRA state loaded: {n_lora} tensors")
    logger.info(f"Base checkpoint loaded: {checkpoint_path}")
    # 兼容新旧 checkpoint key
    best_score = checkpoint.get(
        "best_composite_score", checkpoint.get("best_val_iou", "?")
    )
    logger.info(
        f"  Epoch: {checkpoint.get('epoch', '?')}, "
        f"Best Score: {best_score}"
    )


def load_gda_checkpoint_state(model, checkpoint, *, required=False, tag="checkpoint"):
    """Restore downstream GDA gates/adapters when the active model has GDA."""
    adapter = getattr(model, "boundary_adapter", None)
    state_dict = checkpoint.get("gda_state_dict")
    if adapter is None:
        if state_dict:
            logger.warning("%s contains GDA state but active config disables GDA", tag)
        return False
    if not state_dict:
        if required:
            raise KeyError(f"{tag} has no gda_state_dict")
        return False
    adapter.load_state_dict(state_dict, strict=True)
    logger.info("%s GDA state loaded; gate_stats=%s", tag, adapter.gate_statistics())
    return True


def load_edge_prior_checkpoint_state(
    model, checkpoint, *, required=False, tag="checkpoint"
):
    """Restore the retained G0b state embedded in a downstream checkpoint."""
    prior = getattr(model, "edge_prior", None)
    state_dict = checkpoint.get("edge_prior_state_dict")
    if prior is None:
        if state_dict:
            logger.warning("%s contains edge-prior state but config disables it", tag)
        return False
    if not state_dict:
        if required:
            raise KeyError(f"{tag} has no edge_prior_state_dict")
        return False
    prior.load_state_dict(state_dict, strict=True)
    prior.eval()
    logger.info("%s retained edge-prior state loaded", tag)
    return True


def build_dataloaders(config, disable_unlabeled_appearance_aug=False,
                      boundary_cache_dir=None):
    """构建有标签和无标签 DataLoader。

    Args:
        config: 全局配置
        disable_unlabeled_appearance_aug: 如果为 True，禁用 UnlabeledDataset
            内置外观增强（用于渐进式外观增强接管时避免双重增强）
    """
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    boundary_cfg = config.get("boundary", {})
    semi_cfg = config.get("semi_supervised", {})

    project_root = paths_cfg["project_root"]
    data_dir = os.path.join(project_root, paths_cfg["raw_data_dir"])
    gt_dir = os.path.join(project_root, boundary_cfg.get("gt_dir", "data/purified_gt"))
    unlabeled_dir = os.path.join(project_root, semi_cfg.get("unlabeled_dir", "data/unlabeled"))

    image_size = data_cfg["image_size"]
    augment_config = data_cfg.get("augmentation", {})
    num_workers = data_cfg.get("num_workers", 4)
    boundary_scale_factor = boundary_cfg.get("edt_scale_factor", 10.0)
    boundary_weight_floor = boundary_cfg.get("edt_weight_floor", 1.0)
    boundary_weight_ceil = boundary_cfg.get("edt_weight_ceil", 4.0)
    boundary_target_key = boundary_cfg.get("target_key", "boundary_soft")
    crop_size = data_cfg.get("crop_size", 0)
    include_instance_map = float(
        train_cfg.get("semantic_instance_weight", 0.0)
    ) > 0

    labeled_dataset = LabeledDataset(
        data_dir=data_dir,
        gt_dir=gt_dir,
        image_size=image_size,
        crop_size=crop_size,
        augment=True,
        augment_config=augment_config,
        split="train",
        train_ratio=data_cfg.get("train_ratio", 0.8),
        seed=data_cfg.get("seed", 42),
        boundary_scale_factor=boundary_scale_factor,
        boundary_weight_floor=boundary_weight_floor,
        boundary_weight_ceil=boundary_weight_ceil,
        boundary_target_key=boundary_target_key,
        center_sigma=data_cfg.get("center_sigma", 4.0),
        native_multiscale_config=data_cfg.get("native_multiscale", {}),
        include_instance_map=include_instance_map,
    )

    val_dataset_cls = LabeledDataset if include_instance_map else BoundaryDataset
    val_kwargs = dict(
        data_dir=data_dir,
        gt_dir=gt_dir,
        image_size=image_size,
        crop_size=0,
        augment=False,
        split="val",
        train_ratio=data_cfg.get("train_ratio", 0.8),
        seed=data_cfg.get("seed", 42),
        boundary_scale_factor=boundary_scale_factor,
        boundary_weight_floor=boundary_weight_floor,
        boundary_weight_ceil=boundary_weight_ceil,
        boundary_target_key=boundary_target_key,
        center_sigma=data_cfg.get("center_sigma", 4.0),
    )
    if include_instance_map:
        val_kwargs["include_instance_map"] = True
    val_dataset = val_dataset_cls(**val_kwargs)

    if len(labeled_dataset) == 0:
        logger.error(f"Labeled dataset is empty: {data_dir}")
        sys.exit(1)

    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty, using labeled dataset for validation.")
        val_dataset = labeled_dataset

    use_unlabeled = bool(semi_cfg.get("use_unlabeled", True))
    unlabeled_dataset = None
    if not use_unlabeled:
        logger.info("Unlabeled training: DISABLED by semi_supervised.use_unlabeled")
    elif os.path.exists(unlabeled_dir) and len(os.listdir(unlabeled_dir)) > 0:
        unlabeled_dataset = UnlabeledDataset(
            data_dir=unlabeled_dir,
            image_size=image_size,
            patch_mask_ratio=semi_cfg.get("patch_mask_ratio", 0.3),
            patch_mask_size=semi_cfg.get("patch_mask_size", 64),
            num_patches=semi_cfg.get("num_patches", 8),
            enable_appearance_aug=not disable_unlabeled_appearance_aug,
            enable_patch_mask=semi_cfg.get("enable_patch_mask", False),
            boundary_cache_dir=boundary_cache_dir,
        )
    else:
        logger.warning(f"Unlabeled dataset is empty: {unlabeled_dir}")
        logger.warning("Training will proceed with supervised loss only.")

    bs_labeled = semi_cfg.get("batch_size_labeled", data_cfg.get("batch_size", 4))
    bs_unlabeled = semi_cfg.get("batch_size_unlabeled", data_cfg.get("batch_size", 4))

    bs_labeled = min(bs_labeled, len(labeled_dataset))
    if unlabeled_dataset is not None:
        bs_unlabeled = min(bs_unlabeled, len(unlabeled_dataset))

    train_seed = int(train_cfg.get("seed", data_cfg.get("seed", 42)))
    labeled_generator = torch.Generator()
    labeled_generator.manual_seed(train_seed)
    unlabeled_generator = torch.Generator()
    unlabeled_generator.manual_seed(train_seed + 1)

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=bs_labeled, shuffle=True,
        num_workers=num_workers, collate_fn=labeled_collate_fn,
        pin_memory=True, drop_last=False,
        worker_init_fn=seed_dataloader_worker, generator=labeled_generator,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs_labeled, shuffle=False,
        num_workers=num_workers,
        collate_fn=labeled_collate_fn if include_instance_map else collate_fn,
        pin_memory=True, worker_init_fn=seed_dataloader_worker,
    )

    unlabeled_loader = None
    if unlabeled_dataset is not None:
        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_size=bs_unlabeled, shuffle=True,
            num_workers=num_workers, collate_fn=unlabeled_collate_fn,
            pin_memory=True, drop_last=True,
            worker_init_fn=seed_dataloader_worker, generator=unlabeled_generator,
        )

    logger.info(f"Labeled batch size: {bs_labeled}")
    logger.info(f"Unlabeled batch size: {bs_unlabeled if unlabeled_loader else 'N/A'}")

    return labeled_loader, unlabeled_loader, val_loader


def train_one_epoch(
    student_model, labeled_loader, unlabeled_iter,
    num_steps, num_unlabeled_steps, criterion, unsup_weight, unsup_seg_weight,
    ema_decay,
    optimizer, scaler, device, grad_clip=1.0, use_amp=False,
    teacher_model=None, boundary_teacher_mode="ema",
    augmentor=None, boundary_anchor_cfg=None, ref_model=None, anchor_alpha=1.0,
    skeleton_filter_cfg=None, freeze_seg=False, freeze_boundary=False,
    seg_mask_region_weight=2.0, boundary_mask_region_weight=0.3,
    seg_sharpen_temperature=1.0,
    seg_confidence_threshold=0.5,
    sobel_weight=1.0, tv_weight=0.1, tv_dilate_radius=3,
    tv_bg_weight=1.0, tv_boundary_weight=0.1,
    bg_suppress_weight=0.5, bg_suppress_threshold=0.1,
    pos_weight=5.0, margin_loss_weight=0.0, margin=0.4,
    peak_hinge_weight=0.0, peak_threshold=0.8,
    rate_regularizer_weight=0.0, rate_slack=0.05,
    sem_boundary_align_weight=0.0,
    gda_gate_l1_weight=0.0,
    diagnostics_enabled=False,
):
    """训练一个 epoch（双流混合 Batch + 可选 EMA 更新）。

    Args:
        teacher_model: EMA 教师模型，可为 None（当 boundary_teacher_mode != "ema"
            且 freeze_seg=True 时）。
        boundary_teacher_mode: 边界伪标签源模式（"ema"/"stage1_direct"/"self_consistency"）。
        augmentor: 可选的渐进式外观增强器，仅施加于学生模型输入
                   （有标签图像 + 无标签强增强图像），教师输入不触碰。
        boundary_anchor_cfg: 边界锚点配置，传入 compute_unsupervised_loss。
        ref_model: Stage-1 冻结参考模型，提供稳定边界伪标签。
        anchor_alpha: Stage-1 锚点权重（1.0=纯 Stage-1, 0.0=纯 EMA 教师）。
        skeleton_filter_cfg: 骨架过滤配置，传入 compute_unsupervised_loss。
        freeze_seg: 冻结语义分支，跳过语义损失和 EMA 更新。
        freeze_boundary: 冻结边界分支，跳过边界损失和 EMA 更新。
        unsup_weight: 边界一致性损失权重（已含 ramp-up）。
        unsup_seg_weight: 语义一致性损失权重（已含 ramp-up）。语义伪标签
            可靠度低于边界，独立权重便于语义阶段（如 0.05~0.1）与边界
            阶段（可沿用 0.3）分别调节。
        seg_sharpen_temperature: 语义一致性目标温度锐化系数（1.0=关闭），
            传给 compute_unsupervised_loss。
        sobel_weight: Sobel 梯度一致性损失权重。
        tv_weight: 各向异性 TV 正则化权重。
        tv_dilate_radius: TV 中边界区域膨胀半径（px）。
        tv_bg_weight: 非边界区域 TV 权重。
        tv_boundary_weight: 边界区域 TV 权重。
        bg_suppress_weight: 背景抑制损失权重。
        bg_suppress_threshold: 背景抑制阈值，低于此值视为背景区域。
    """
    set_student_train_modes(student_model)
    if teacher_model is not None:
        teacher_model.eval()

    total_loss_sum = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    total_seg = 0.0
    total_semantic_instance = 0.0
    total_boundary = 0.0
    total_seg_consist = 0.0
    total_seg_conf_coverage = 0.0
    total_boundary_consist = 0.0
    total_bnd_max = 0.0
    total_bnd_pos = 0.0
    total_bnd_gap = 0.0
    total_bnd_rate = 0.0
    total_refine_grad_norm = 0.0
    total_semantic_residual_grad_norm = 0.0
    total_gda_gate_reg = 0.0
    refine_residual_sum_abs = 0.0
    refine_residual_sum_std = 0.0
    refine_residual_max_abs = 0.0
    refine_residual_calls = 0
    fusion_residual_sum_abs = 0.0
    fusion_residual_sum_std = 0.0
    fusion_residual_max_abs = 0.0
    fusion_residual_calls = 0
    semantic_residual_sum_abs = 0.0
    semantic_residual_sum_std = 0.0
    semantic_residual_max_abs = 0.0
    semantic_residual_calls = 0
    n_steps = 0

    clip_params = list(student_model.decoder.parameters())
    if getattr(student_model, "boundary_adapter", None) is not None:
        clip_params.extend(student_model.boundary_adapter.parameters())
    refine_params = (
        list(student_model.decoder.boundary_refine_head.parameters())
        if getattr(student_model.decoder, "boundary_refine", False)
        else []
    )
    fusion_params = (
        list(student_model.decoder.edge_prior_fusion.parameters())
        if getattr(student_model.decoder, "edge_prior_fusion", None) is not None
        else []
    )
    semantic_residual_params = (
        list(student_model.decoder.semantic_residual.parameters())
        if getattr(student_model.decoder, "semantic_residual", None) is not None
        else []
    )

    refine_hook = None
    if diagnostics_enabled and refine_params:
        def _capture_refine_output(module, inputs, output):
            del module, inputs
            nonlocal refine_residual_sum_abs
            nonlocal refine_residual_sum_std
            nonlocal refine_residual_max_abs
            nonlocal refine_residual_calls
            value = output.detach().float()
            refine_residual_sum_abs += float(value.abs().mean())
            refine_residual_sum_std += float(value.std(unbiased=False))
            refine_residual_max_abs = max(
                refine_residual_max_abs, float(value.abs().max())
            )
            refine_residual_calls += 1

        refine_hook = student_model.decoder.boundary_refine_head.register_forward_hook(
            _capture_refine_output
        )

    fusion_hook = None
    if diagnostics_enabled and fusion_params:
        def _capture_fusion_output(module, inputs, output):
            del module, inputs
            nonlocal fusion_residual_sum_abs
            nonlocal fusion_residual_sum_std
            nonlocal fusion_residual_max_abs
            nonlocal fusion_residual_calls
            value = output.detach().float()
            fusion_residual_sum_abs += float(value.abs().mean())
            fusion_residual_sum_std += float(value.std(unbiased=False))
            fusion_residual_max_abs = max(
                fusion_residual_max_abs, float(value.abs().max())
            )
            fusion_residual_calls += 1

        fusion_hook = student_model.decoder.edge_prior_fusion.register_forward_hook(
            _capture_fusion_output
        )

    semantic_residual_hook = None
    if diagnostics_enabled and semantic_residual_params:
        def _capture_semantic_residual_output(module, inputs, output):
            del module, inputs
            nonlocal semantic_residual_sum_abs
            nonlocal semantic_residual_sum_std
            nonlocal semantic_residual_max_abs
            nonlocal semantic_residual_calls
            value = output.detach().float()
            semantic_residual_sum_abs += float(value.abs().mean())
            semantic_residual_sum_std += float(value.std(unbiased=False))
            semantic_residual_max_abs = max(
                semantic_residual_max_abs, float(value.abs().max())
            )
            semantic_residual_calls += 1

        semantic_residual_hook = (
            student_model.decoder.semantic_residual.register_forward_hook(
                _capture_semantic_residual_output
            )
        )

    # 不使用 itertools.cycle：它会缓存第一轮已增强 batch，之后只重复旧张量。
    # DataLoader 耗尽后重新创建 iterator，重新洗牌并重新采样数据集增强。
    labeled_iter = iter(labeled_loader)

    for step_idx in range(num_steps):
        labeled_batch, labeled_iter = next_restarting_batch(
            labeled_loader, labeled_iter
        )
        images_labeled = labeled_batch["image"].to(device)
        targets_labeled = labeled_batch["target"].to(device)
        weights_labeled = labeled_batch["weight"].to(device)
        instance_maps_labeled = labeled_batch.get("instance_map")
        if instance_maps_labeled is not None:
            instance_maps_labeled = instance_maps_labeled.to(device)

        # 渐进式外观增强：仅施加于学生输入（有标签路径）
        if augmentor is not None:
            images_labeled = augmentor(
                images_labeled, boundary_targets=targets_labeled[:, 1]
            )

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                out_labeled = student_model(images_labeled, output_size=targets_labeled.shape[-2:])
                sup_loss, seg_val, boundary_val = criterion(
                    out_labeled,
                    targets_labeled,
                    weights_labeled,
                    instance_map=instance_maps_labeled,
                )
        else:
            out_labeled = student_model(images_labeled, output_size=targets_labeled.shape[-2:])
            sup_loss, seg_val, boundary_val = criterion(
                out_labeled,
                targets_labeled,
                weights_labeled,
                instance_map=instance_maps_labeled,
            )
        semantic_instance_val = criterion.last_semantic_instance_loss

        unsup_loss = torch.tensor(0.0, device=device)
        seg_consist_val = 0.0
        boundary_consist_val = 0.0
        bnd_stats = {"bnd_max": 0.0, "bnd_pos_frac": 0.0, "bnd_gap": 0.0}

        if unlabeled_iter is not None and step_idx < num_unlabeled_steps:
            try:
                unlabeled_batch = next(unlabeled_iter)
                img_weak = unlabeled_batch["img_weak"]
                img_strong = unlabeled_batch["img_strong"]
                patch_mask = unlabeled_batch["patch_mask"]

                # 渐进式外观增强：仅施加于学生输入（无标签强增强路径）
                # 教师的 img_weak 保持干净
                if augmentor is not None:
                    img_strong = augmentor(img_strong)

                if use_amp:
                    with autocast('cuda'):
                        unsup_seg_loss, unsup_bnd_loss, bnd_stats = (
                            compute_unsupervised_loss(
                                student_model,
                                img_weak, img_strong, patch_mask,
                                output_size=targets_labeled.shape[-2:],
                                teacher_model=teacher_model,
                                boundary_teacher_mode=boundary_teacher_mode,
                                boundary_anchor_cfg=boundary_anchor_cfg,
                                ref_model=ref_model,
                                anchor_alpha=anchor_alpha,
                                skeleton_filter_cfg=skeleton_filter_cfg,
                                freeze_seg=freeze_seg,
                                freeze_boundary=freeze_boundary,
                                seg_mask_region_weight=seg_mask_region_weight,
                                boundary_mask_region_weight=boundary_mask_region_weight,
                                seg_sharpen_temperature=seg_sharpen_temperature,
                                seg_confidence_threshold=seg_confidence_threshold,
                                sobel_weight=sobel_weight,
                                tv_weight=tv_weight,
                                tv_dilate_radius=tv_dilate_radius,
                                tv_bg_weight=tv_bg_weight,
                                tv_boundary_weight=tv_boundary_weight,
                                bg_suppress_weight=bg_suppress_weight,
                                bg_suppress_threshold=bg_suppress_threshold,
                                cached_boundary_target=unlabeled_batch.get("boundary_target"),
                                pos_weight=pos_weight,
                                margin_loss_weight=margin_loss_weight,
                                margin=margin,
                                peak_hinge_weight=peak_hinge_weight,
                                peak_threshold=peak_threshold,
                                rate_regularizer_weight=rate_regularizer_weight,
                                rate_slack=rate_slack,
                                sem_boundary_align_weight=sem_boundary_align_weight,
                            )
                        )
                else:
                    unsup_seg_loss, unsup_bnd_loss, bnd_stats = (
                        compute_unsupervised_loss(
                            student_model,
                            img_weak, img_strong, patch_mask,
                            output_size=targets_labeled.shape[-2:],
                            teacher_model=teacher_model,
                            boundary_teacher_mode=boundary_teacher_mode,
                            boundary_anchor_cfg=boundary_anchor_cfg,
                            ref_model=ref_model,
                            anchor_alpha=anchor_alpha,
                            skeleton_filter_cfg=skeleton_filter_cfg,
                            freeze_seg=freeze_seg,
                            freeze_boundary=freeze_boundary,
                            seg_mask_region_weight=seg_mask_region_weight,
                            boundary_mask_region_weight=boundary_mask_region_weight,
                            seg_sharpen_temperature=seg_sharpen_temperature,
                            seg_confidence_threshold=seg_confidence_threshold,
                            sobel_weight=sobel_weight,
                            tv_weight=tv_weight,
                            tv_dilate_radius=tv_dilate_radius,
                            tv_bg_weight=tv_bg_weight,
                            tv_boundary_weight=tv_boundary_weight,
                            bg_suppress_weight=bg_suppress_weight,
                            bg_suppress_threshold=bg_suppress_threshold,
                            cached_boundary_target=unlabeled_batch.get("boundary_target"),
                            pos_weight=pos_weight,
                            margin_loss_weight=margin_loss_weight,
                            margin=margin,
                            peak_hinge_weight=peak_hinge_weight,
                            peak_threshold=peak_threshold,
                            rate_regularizer_weight=rate_regularizer_weight,
                            rate_slack=rate_slack,
                            sem_boundary_align_weight=sem_boundary_align_weight,
                        )
                    )
            except StopIteration:
                pass
            else:
                # 语义/边界一致性损失独立加权：
                # 语义伪标签可靠度低，权重单独控制（unsup_seg_weight），
                # 边界一致性沿用 unsup_weight
                unsup_loss = (
                    unsup_seg_weight * unsup_seg_loss
                    + unsup_weight * unsup_bnd_loss
                )
                seg_consist_val = unsup_seg_loss.item()
                boundary_consist_val = unsup_bnd_loss.item()

        gda_gate_reg = torch.tensor(0.0, device=device)
        boundary_adapter = getattr(student_model, "boundary_adapter", None)
        if boundary_adapter is not None and gda_gate_l1_weight > 0:
            gda_gate_reg = boundary_adapter.active_gate_l1()
        total_loss = (
            sup_loss + unsup_loss + gda_gate_l1_weight * gda_gate_reg
        )

        if use_amp:
            scaler.scale(total_loss).backward()
            if grad_clip > 0 or diagnostics_enabled:
                scaler.unscale_(optimizer)
            if diagnostics_enabled:
                total_refine_grad_norm += gradient_l2_norm(
                    refine_params + fusion_params
                )
                total_semantic_residual_grad_norm += gradient_l2_norm(
                    semantic_residual_params
                )
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if diagnostics_enabled:
                total_refine_grad_norm += gradient_l2_norm(
                    refine_params + fusion_params
                )
                total_semantic_residual_grad_norm += gradient_l2_norm(
                    semantic_residual_params
                )
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
            optimizer.step()

        # EMA 更新（仅当教师模型存在时）
        if teacher_model is not None:
            update_ema(teacher_model, student_model, ema_decay)

        total_loss_sum += total_loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        total_seg += seg_val
        total_semantic_instance += semantic_instance_val
        total_boundary += boundary_val
        total_seg_consist += seg_consist_val
        total_seg_conf_coverage += bnd_stats.get("seg_conf_coverage", 0.0)
        total_boundary_consist += boundary_consist_val
        total_gda_gate_reg += float(gda_gate_reg.detach())
        total_bnd_max += bnd_stats.get("bnd_max", 0.0)
        total_bnd_pos += bnd_stats.get("bnd_pos_frac", 0.0)
        total_bnd_gap += bnd_stats.get("bnd_gap", 0.0)
        total_bnd_rate += bnd_stats.get("bnd_pred_rate", 0.0)
        n_steps += 1

        if (step_idx + 1) % 5 == 0:
            logger.info(
                f"  Step {step_idx + 1}/{num_steps}: "
                f"total={total_loss.item():.4f} "
                f"sup={sup_loss.item():.4f} "
                f"(seg={seg_val:.4f} inst={semantic_instance_val:.4f} "
                f"bnd={boundary_val:.4f}) "
                f"unsup={unsup_loss.item():.4f} (s_c={seg_consist_val:.4f} b_c={boundary_consist_val:.4f}) "
                f"gda_l1={float(gda_gate_reg.detach()):.5f}"
            )

    if refine_hook is not None:
        refine_hook.remove()
    if fusion_hook is not None:
        fusion_hook.remove()
    if semantic_residual_hook is not None:
        semantic_residual_hook.remove()

    n = max(n_steps, 1)
    refine_n = max(refine_residual_calls, 1)
    fusion_n = max(fusion_residual_calls, 1)
    semantic_residual_n = max(semantic_residual_calls, 1)
    return {
        "loss": total_loss_sum / n,
        "sup_loss": total_sup_loss / n,
        "unsup_loss": total_unsup_loss / n,
        "seg": total_seg / n,
        "semantic_instance": total_semantic_instance / n,
        "boundary": total_boundary / n,
        "seg_consist": total_seg_consist / n,
        "seg_conf_coverage": total_seg_conf_coverage / n,
        "boundary_consist": total_boundary_consist / n,
        "gda_gate_l1": total_gda_gate_reg / n,
        "bnd_max": total_bnd_max / n,
        "bnd_pos_frac": total_bnd_pos / n,
        "bnd_gap": total_bnd_gap / n,
        "bnd_pred_rate": total_bnd_rate / n,
        "optimizer_steps": n_steps,
        "refine_grad_norm": total_refine_grad_norm / n,
        "semantic_residual_grad_norm": total_semantic_residual_grad_norm / n,
        "refine_residual_mean_abs": refine_residual_sum_abs / refine_n,
        "refine_residual_std": refine_residual_sum_std / refine_n,
        "refine_residual_max_abs": refine_residual_max_abs,
        "fusion_residual_mean_abs": fusion_residual_sum_abs / fusion_n,
        "fusion_residual_std": fusion_residual_sum_std / fusion_n,
        "fusion_residual_max_abs": fusion_residual_max_abs,
        "semantic_residual_mean_abs": (
            semantic_residual_sum_abs / semantic_residual_n
        ),
        "semantic_residual_std": (
            semantic_residual_sum_std / semantic_residual_n
        ),
        "semantic_residual_max_abs": semantic_residual_max_abs,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证。"""
    model.eval()
    metrics = SegMetrics(num_classes=2)
    total_loss = 0.0
    n_batches = 0

    # 边界指标累积
    bnd_tp = 0
    bnd_fp = 0
    bnd_fn = 0
    bnd_pos_prob_sum = 0.0
    bnd_bg_prob_sum = 0.0
    bnd_pos_count = 0
    bnd_bg_count = 0
    bnd_pos_recalled_035 = 0
    bnd_bg_false_positive_035 = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        weights = batch["weight"].to(device)
        instance_map = batch.get("instance_map")
        if instance_map is not None:
            instance_map = instance_map.to(device)
        output = model(images, output_size=targets.shape[-2:])
        total_loss_t, _, _ = criterion(
            output, targets, weights, instance_map=instance_map
        )
        total_loss += total_loss_t.item()
        n_batches += 1

        seg_logits = output[:, 0]
        pred = (torch.sigmoid(seg_logits) > 0.5).long()
        mask = targets[:, 0].long()
        metrics.update_tensor(pred, mask)

        # 边界通道 IoU
        bnd_logits = output[:, 1]
        bnd_prob = torch.sigmoid(bnd_logits)
        bnd_pred = (bnd_prob > 0.5).long()
        bnd_gt = (targets[:, 1] > 0.5).long()
        bnd_tp += ((bnd_pred == 1) & (bnd_gt == 1)).sum().item()
        bnd_fp += ((bnd_pred == 1) & (bnd_gt == 0)).sum().item()
        bnd_fn += ((bnd_pred == 0) & (bnd_gt == 1)).sum().item()
        positive = bnd_gt == 1
        background = ~positive
        bnd_pos_prob_sum += float(bnd_prob[positive].sum())
        bnd_bg_prob_sum += float(bnd_prob[background].sum())
        bnd_pos_count += int(positive.sum())
        bnd_bg_count += int(background.sum())
        bnd_pos_recalled_035 += int(((bnd_prob > 0.35) & positive).sum())
        bnd_bg_false_positive_035 += int(
            ((bnd_prob > 0.35) & background).sum()
        )

    val_metrics = metrics.get_metrics()
    val_metrics["loss"] = total_loss / max(n_batches, 1)

    # Boundary IoU = TP / (TP + FP + FN + eps)
    eps = 1e-7
    val_metrics["boundary_iou"] = bnd_tp / (bnd_tp + bnd_fp + bnd_fn + eps)
    val_metrics["boundary_pos_mean"] = bnd_pos_prob_sum / max(bnd_pos_count, 1)
    val_metrics["boundary_bg_mean"] = bnd_bg_prob_sum / max(bnd_bg_count, 1)
    val_metrics["boundary_prob_gap"] = (
        val_metrics["boundary_pos_mean"] - val_metrics["boundary_bg_mean"]
    )
    val_metrics["boundary_recall_035"] = (
        bnd_pos_recalled_035 / max(bnd_pos_count, 1)
    )
    val_metrics["boundary_bg_fp_rate_035"] = (
        bnd_bg_false_positive_035 / max(bnd_bg_count, 1)
    )

    return val_metrics


def resolve_monitor_image_paths(config):
    """Resolve a deterministic monitor subset, optionally from a manifest."""
    paths_cfg = config["paths"]
    monitor_cfg = config.get("semi_supervised", {}).get("monitor", {})
    image_dir = os.path.join(
        paths_cfg["project_root"],
        monitor_cfg.get("image_dir", "data/test"),
    )
    num_images = int(monitor_cfg.get("num_images", 3))
    manifest = monitor_cfg.get("manifest", "")
    if manifest:
        manifest_path = project_path(config, manifest)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Monitor manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            names = [
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
        image_paths = [
            os.path.join(image_dir, name)
            for name in names
            if os.path.isfile(os.path.join(image_dir, name))
        ]
        missing = len(names) - len(image_paths)
        if missing:
            logger.warning(
                "Monitor manifest: %d/%d entries missing under %s",
                missing, len(names), image_dir,
            )
    else:
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.sort()
    return image_paths[:num_images], image_dir


@torch.no_grad()
def monitor_inference(model, config, epoch, device):
    """对 data/test 前 N 张图像做轻量推理，保存语义+边界概率图。

    保存结构：output_dir/epoch_XX/{basename}_seg.png / {basename}_boundary.png
    """
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    monitor_cfg = config.get("semi_supervised", {}).get("monitor", {})

    output_base = os.path.join(
        paths_cfg["project_root"],
        monitor_cfg.get("output_dir", "outputs/stage2/monitor"),
    )
    image_size = data_cfg.get("image_size", 1024)
    raw_probability_size = int(monitor_cfg.get("raw_probability_size", 0))

    epoch_dir = os.path.join(output_base, f"epoch_{epoch + 1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    image_paths, image_dir = resolve_monitor_image_paths(config)

    if len(image_paths) == 0:
        logger.warning(f"Monitor: no images found in {image_dir}")
        return

    model.eval()
    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Letterbox
        image_lb, _, _, _ = letterbox(image_rgb, image_size)
        image_tensor = (
            torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        image_tensor = image_tensor.to(device)

        output = model(image_tensor)

        seg_prob = torch.sigmoid(output[0, 0]).cpu().numpy()
        bnd_prob = torch.sigmoid(output[0, 1]).cpu().numpy()
        center_prob = (
            torch.sigmoid(output[0, 2]).cpu().numpy()
            if output.shape[1] >= 3
            else None
        )

        # 语义概率图：伪彩色（JET），珠光体=蓝，铁素体=红
        seg_vis = (seg_prob * 255).astype(np.uint8)
        seg_color = cv2.applyColorMap(seg_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_seg.png"), seg_color)

        # 边界概率图：热力图（灰度→伪彩色）
        bnd_vis = (bnd_prob * 255).astype(np.uint8)
        bnd_color = cv2.applyColorMap(bnd_vis, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_boundary.png"), bnd_color)

        if raw_probability_size > 0:
            # Compact 16-bit raw probabilities expose small changes hidden by
            # the 8-bit colour map without storing full float tensors.
            bnd_raw = cv2.resize(
                bnd_prob,
                (raw_probability_size, raw_probability_size),
                interpolation=cv2.INTER_AREA,
            )
            bnd_raw16 = np.clip(bnd_raw * 65535.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(
                os.path.join(epoch_dir, f"{basename}_boundary_raw16.png"),
                bnd_raw16,
            )

        if center_prob is not None:
            center_vis = (center_prob * 255).astype(np.uint8)
            center_color = cv2.applyColorMap(center_vis, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(epoch_dir, f"{basename}_center.png"), center_color)

    logger.info(f"  Monitor inference saved: {epoch_dir} ({len(image_paths)} images)")


@torch.no_grad()
def compute_semantic_confidence_stats(model, config, device):
    """统计 data/test 前 N 张图的语义置信度分布（无需 GT）。

    返回 (高置信 >0.8 占比, 模糊带 0.4-0.6 占比)；test 目录为空时返回 (None, None)。
    用于训练中监控"语义变糊/涌动"：若高置信占比持续下降、模糊带持续扩张，
    说明无监督语义一致性在把输出拉向 0.5（见 20260804 运行的 test 置信度
    从 Stage-1 63.5% 掉到 ep60 52%、模糊带 7.5% 扩到 9.3% 的现象）。
    """
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    monitor_cfg = config.get("semi_supervised", {}).get("monitor", {})

    image_size = data_cfg.get("image_size", 1024)
    image_paths, image_dir = resolve_monitor_image_paths(config)

    if len(image_paths) == 0:
        logger.warning(f"Sem conf: no images found in {image_dir}")
        return None, None

    model.eval()
    high_sum = 0.0
    mid_sum = 0.0
    n = 0
    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lb, _, _, _ = letterbox(image_rgb, image_size)
        image_tensor = (
            torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        ).to(device)

        output = model(image_tensor)
        seg_prob = torch.sigmoid(output[0, 0])
        high_sum += float((seg_prob > 0.8).float().mean())
        mid_sum += float(((seg_prob > 0.4) & (seg_prob < 0.6)).float().mean())
        n += 1

    if n == 0:
        return None, None
    return high_sum / n, mid_sum / n


def main():
    parser = argparse.ArgumentParser(
        description="Stage-2 Semi-Supervised Fine-tuning (Mean Teacher + Boundary Prediction)"
    )
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
        help="从 checkpoint 恢复训练（加载 decoder + optimizer + scheduler，继续旧 epoch）")
    parser.add_argument("--init_from_checkpoint", type=str, default=None,
        help="从指定 checkpoint 加载 decoder 权重，但重置 optimizer/scheduler/epoch "
             "（用于分支切换：训练完一个分支后，从此 checkpoint 开始训练另一个分支）")
    parser.add_argument("--tag", type=str, default="",
                        help="运行标签（用于 outputs/runs/<时间戳>_<phase>_<tag> 命名）")
    parser.add_argument("--phase", type=str, default="",
                        help="阶段标签（如 semantic / boundary），仅用于运行目录命名与记录")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="覆盖运行记录目录（默认 outputs/runs/<时间戳>_<phase>_<tag>）")
    args = parser.parse_args()

    config = load_config(args.config)
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]
    train_cfg = config["train"]
    semi_cfg = config.get("semi_supervised", {})
    lora_cfg = config.get("lora", {})
    gda_cfg = config.get("gda", {})
    edge_prior_cfg = config.get("edge_prior", {})

    train_seed = int(train_cfg.get("seed", config.get("data", {}).get("seed", 42)))
    deterministic = bool(train_cfg.get("deterministic", False))
    seed_everything(train_seed, deterministic=deterministic)

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    logger.info(f"Device: {device}")

    # 运行记录器：配置快照 / git / 环境 / 逐 epoch 指标
    recorder = RunRecorder(
        project_root=paths_cfg["project_root"],
        phase=args.phase,
        tag=args.tag,
    )
    if args.run_dir:
        recorder.run_dir = args.run_dir
        os.makedirs(recorder.run_dir, exist_ok=True)
        recorder._metrics_path = os.path.join(recorder.run_dir, "metrics.csv")
        recorder._metrics_written = os.path.exists(recorder._metrics_path)
    recorder.save_config(config)
    recorder.save_manifest()
    logger.info(f"Run dir: {recorder.run_dir}")

    output_dir = os.path.join(paths_cfg["project_root"], semi_cfg.get("output_dir", "outputs/stage2"))
    os.makedirs(output_dir, exist_ok=True)

    student_model = build_model(config, device)

    base_checkpoint = semi_cfg.get(
        "base_checkpoint",
        config["inference"].get("stage1_checkpoint", "outputs/stage1/best_model.pth"),
    )
    stage1_ckpt_path = project_path(config, base_checkpoint)
    load_base_checkpoint(student_model, stage1_ckpt_path, device)

    if semi_cfg.get("reset_boundary_branch", False):
        student_model.decoder.reset_boundary_branch()
        logger.info("Reset boundary branch after loading Stage-1 checkpoint")

    # 渐进式外观增强配置
    prog_aug_cfg = config.get("progressive_aug", {})
    prog_aug_enabled = prog_aug_cfg.get("enabled", False)

    # 如果启用了渐进式外观增强，禁用 UnlabeledDataset 内置外观增强（避免双重增强）
    use_cached_pseudo_labels = semi_cfg.get("use_cached_pseudo_labels", False)
    boundary_cache_dir = None
    if use_cached_pseudo_labels:
        boundary_cache_dir = os.path.join(
            paths_cfg["project_root"],
            semi_cfg.get(
                "pseudo_label_cache_dir", "outputs/pseudo_labels/stage1_boundary"
            ),
        )
        logger.info(f"Stage-1 边界伪标签缓存: ENABLED ({boundary_cache_dir})")
    else:
        logger.info("Stage-1 边界伪标签缓存: DISABLED (每 step 实时前向 ref_model)")

    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(
        config,
        disable_unlabeled_appearance_aug=prog_aug_enabled,
        boundary_cache_dir=boundary_cache_dir,
    )

    # 实例化渐进式外观增强器
    augmentor = None
    if prog_aug_enabled:
        augmentor = ProgressiveAppearanceAug(prog_aug_cfg, device)
        logger.info(
            f"Progressive appearance augmentation: ENABLED "
            f"(start_epoch={augmentor.start_epoch}, ramp={augmentor.ramp_epochs}, "
            f"max_prob={augmentor.max_prob})"
        )
    else:
        logger.info("Progressive appearance augmentation: DISABLED")

    # 冻结开关配置
    freeze_cfg = semi_cfg.get("freeze", {})
    freeze_seg = freeze_cfg.get("seg_branch", False)
    freeze_boundary = freeze_cfg.get("boundary_branch", False)

    if freeze_seg and freeze_boundary:
        logger.warning("Both freeze.seg_branch and freeze.boundary_branch are True! "
                       "No parameters will be trained. Setting both to False.")
        freeze_seg = False
        freeze_boundary = False

    if freeze_seg:
        student_model.decoder.freeze_seg_branch()
        logger.info("Freeze: SEMANTIC branch frozen (seg_fpn + seg_branch)")
    if freeze_boundary:
        student_model.decoder.freeze_boundary_branch()
        logger.info("Freeze: BOUNDARY branch frozen (boundary_fpn + boundary_branch)")
    if not freeze_seg and not freeze_boundary:
        logger.info("Freeze: DISABLED (joint training, both branches active)")

    semantic_residual_training_cfg = semi_cfg.get(
        "semantic_residual_training", {}
    )
    semantic_residual_training_enabled = bool(
        semantic_residual_training_cfg.get("enabled", False)
    )
    if semantic_residual_training_enabled:
        if not student_model.decoder.semantic_residual_enabled:
            raise ValueError(
                "semantic_residual_training.enabled=True requires "
                "decoder.semantic_residual=True"
            )
        if freeze_seg:
            raise ValueError(
                "semantic residual-only training requires freeze.seg_branch=False "
                "so semantic losses remain active"
            )
        if not freeze_boundary:
            raise ValueError(
                "semantic residual-only training requires "
                "freeze.boundary_branch=True"
            )
        if lora_cfg.get("enabled", False) and not lora_cfg.get("freeze", False):
            raise ValueError(
                "semantic residual-only training requires frozen LoRA to preserve "
                "the G4b affinity feature contract"
            )
        student_model.decoder.set_semantic_residual_only()
        logger.info(
            "E8 semantic residual-only training: V6 seg FPN/head, boundary, "
            "and LoRA are frozen; only semantic_residual is trainable"
        )

    # B2 staged refine：先只训练零初始化的高分辨率 residual head，随后再以
    # 较低学习率解冻 V6 coarse boundary 基座。语义分支与 LoRA 由各自的
    # freeze 配置保护，center head 必须关闭，确保这是单变量架构实验。
    refine_training_cfg = semi_cfg.get("refine_training", {})
    refine_training_enabled = bool(refine_training_cfg.get("enabled", False))
    refine_only_epochs = int(refine_training_cfg.get("refine_only_epochs", 0))
    if refine_training_enabled:
        if not student_model.decoder.boundary_refine:
            raise ValueError(
                "refine_training.enabled=True requires decoder.boundary_refine=True"
            )
        if student_model.decoder.center_head_enabled:
            raise ValueError("B2 refine training requires decoder.center_head=False")
        if freeze_boundary:
            raise ValueError(
                "B2 refine training requires freeze.boundary_branch=False; "
                "the coarse base is staged by refine_training.refine_only_epochs"
            )
        student_model.decoder.set_boundary_base_trainable(refine_only_epochs <= 0)
        logger.info(
            "B2 refine training: ENABLED "
            f"(refine_only_epochs={refine_only_epochs})"
        )

    edge_prior_fusion_only = bool(edge_prior_cfg.get("only_fusion", False))
    if edge_prior_fusion_only:
        if refine_training_enabled:
            raise ValueError(
                "edge_prior.only_fusion=True requires refine_training.enabled=False"
            )
        student_model.decoder.set_edge_prior_fusion_only()
        logger.info(
            "G2a edge-prior calibration: ONLY fusion head is trainable; "
            "E1 decoder, refine, semantic, LoRA, and G0b prior are frozen"
        )

    if gda_cfg.get("only_gates", False):
        if getattr(student_model, "boundary_adapter", None) is None:
            raise ValueError("gda.only_gates=True requires gda.enabled=True")
        for parameter in student_model.decoder.parameters():
            parameter.requires_grad_(False)
        if not student_model.boundary_adapter.gates.requires_grad:
            raise ValueError("gda.only_gates=True requires gda.train_gates=True")
        logger.info(
            "GDA gate-only training: decoder/refine fully frozen; "
            "supervised boundary loss remains active"
        )

    # 边界伪标签源模式配置
    boundary_teacher_mode = semi_cfg.get("boundary_teacher_mode", "ema")
    logger.info(f"Boundary teacher mode: {boundary_teacher_mode}")

    # 判断是否需要 EMA 教师模型
    # - 语义分支未冻结（freeze_seg=False）→ 需要 EMA 教师提供语义伪标签
    # - 边界模式为 "ema" → 需要 EMA 教师提供边界伪标签
    need_teacher = (not freeze_seg) or (boundary_teacher_mode == "ema")

    if not need_teacher:
        logger.info(
            f"  EMA teacher model: NOT REQUIRED "
            f"(freeze_seg={freeze_seg}, boundary_teacher_mode='{boundary_teacher_mode}')"
        )

    criterion = BoundaryLoss(
        gamma=train_cfg.get("focal_gamma", 2.0),
        alpha_boundary=train_cfg.get("boundary_alpha", 1.0),
        alpha_focal=train_cfg.get("focal_alpha", 0.75),
        seg_dice_weight=train_cfg.get("seg_dice_weight", 0.0),
        semantic_instance_weight=train_cfg.get(
            "semantic_instance_weight", 0.0
        ),
        semantic_core_radius=train_cfg.get("semantic_core_radius", 3),
        semantic_core_min_pixels=train_cfg.get(
            "semantic_core_min_pixels", 12
        ),
        semantic_core_boundary_threshold=train_cfg.get(
            "semantic_core_boundary_threshold", 0.20
        ),
        semantic_instance_class_balance=train_cfg.get(
            "semantic_instance_class_balance", True
        ),
        semantic_instance_ferrite_weight=train_cfg.get(
            "semantic_instance_ferrite_weight", 1.0
        ),
        semantic_instance_hard_gamma=train_cfg.get(
            "semantic_instance_hard_gamma", 0.0
        ),
        semantic_instance_hard_floor=train_cfg.get(
            "semantic_instance_hard_floor", 0.25
        ),
        semantic_tversky_weight=train_cfg.get(
            "semantic_tversky_weight", 0.0
        ),
        semantic_tversky_alpha=train_cfg.get(
            "semantic_tversky_alpha", 0.40
        ),
        semantic_tversky_beta=train_cfg.get(
            "semantic_tversky_beta", 0.60
        ),
        freeze_seg=freeze_seg,
        freeze_boundary=freeze_boundary,
        peak_weight=train_cfg.get("boundary_peak_weight", 0.0),
        peak_logit=train_cfg.get("boundary_peak_logit", 2.0),
        hard_negative_weight=train_cfg.get("boundary_hard_negative_weight", 0.0),
        hard_negative_logit=train_cfg.get("boundary_hard_negative_logit", -1.5),
        ridge_weight=train_cfg.get("boundary_ridge_weight", 0.0),
        ridge_positive_logit=train_cfg.get("boundary_ridge_positive_logit", 1.0),
        ridge_negative_logit=train_cfg.get("boundary_ridge_negative_logit", -1.5),
        ridge_core_threshold=train_cfg.get("boundary_ridge_core_threshold", 0.5),
        ridge_background_threshold=train_cfg.get(
            "boundary_ridge_background_threshold", 0.05
        ),
        ridge_tolerance=train_cfg.get("boundary_ridge_tolerance", 1),
        ridge_ring_radius=train_cfg.get("boundary_ridge_ring_radius", 5),
        ridge_ring_weight=train_cfg.get("boundary_ridge_ring_weight", 1.0),
        ridge_mode=train_cfg.get("boundary_ridge_mode", "absolute"),
        ridge_margin=train_cfg.get("boundary_ridge_margin", 1.5),
        center_weight=train_cfg.get("center_weight", 0.0),
        center_gamma=train_cfg.get("center_gamma", 2.0),
    ).to(device)
    logger.info(
        "Supervised loss: BoundaryLoss "
        "(semantic BCE/Dice/instance-core + boundary Focal x EDT + "
        "peak/hard-negative/ridge)"
    )
    if criterion.semantic_instance_weight > 0:
        logger.info(
            "  Semantic instance core: weight=%.3f radius=%dpx "
            "min_pixels=%d class_balance=%s ferrite_weight=%.2f "
            "hard_gamma=%.2f",
            criterion.semantic_instance_weight,
            criterion.semantic_core_radius,
            criterion.semantic_core_min_pixels,
            criterion.semantic_instance_class_balance,
            criterion.semantic_instance_ferrite_weight,
            criterion.semantic_instance_hard_gamma,
        )
    if criterion.semantic_tversky_weight > 0:
        logger.info(
            "  Semantic Tversky: weight=%.3f alpha=%.2f beta=%.2f",
            criterion.semantic_tversky_weight,
            criterion.semantic_tversky_alpha,
            criterion.semantic_tversky_beta,
        )
    if criterion.ridge_weight > 0:
        logger.info(
            "  Boundary ridge: mode=%s, weight=%.4f, peak_logit=%.2f, "
            "ring_logit=%.2f, tolerance=%dpx, ring_radius=%dpx, "
            "ring_weight=%.2f, margin=%.2f",
            criterion.ridge_mode,
            criterion.ridge_weight,
            criterion.ridge_positive_logit,
            criterion.ridge_negative_logit,
            criterion.ridge_tolerance,
            criterion.ridge_ring_radius,
            criterion.ridge_ring_weight,
            criterion.ridge_margin,
        )

    # 分层参数优化器：语义分支和边界分支各自独立学习率（方案 B - 三分组）
    base_lr = semi_cfg.get("learning_rate", train_cfg["learning_rate"])
    seg_lr_ratio = semi_cfg.get("seg_lr_ratio", 0.1)
    boundary_lr_ratio = semi_cfg.get("boundary_lr_ratio", 0.1)
    boundary_base_lr_ratio = refine_training_cfg.get(
        "boundary_base_lr_ratio", boundary_lr_ratio
    )
    refine_lr_ratio = refine_training_cfg.get("refine_lr_ratio", boundary_lr_ratio)
    seg_lr = base_lr * seg_lr_ratio
    boundary_lr = base_lr * boundary_base_lr_ratio
    refine_lr = base_lr * refine_lr_ratio

    seg_params = []
    boundary_params = []
    refine_params = []
    edge_prior_fusion_params = []
    for name, param in student_model.decoder.named_parameters():
        if "seg_fpn" in name or "seg_branch" in name:
            if param.requires_grad:
                seg_params.append(param)
        elif "boundary_refine_head" in name:
            if param.requires_grad:
                refine_params.append(param)
        elif "edge_prior_fusion" in name:
            if param.requires_grad:
                edge_prior_fusion_params.append(param)
        elif (
            "boundary_fpn" in name
            or "boundary_branch" in name
            or "center_branch" in name
        ):
            # B2 refine-only 阶段的 base 参数暂时 requires_grad=False，但必须
            # 预先进入 optimizer，后续 epoch 解冻时才能直接开始更新。
            if param.requires_grad or refine_training_enabled:
                boundary_params.append(param)
        else:
            if param.requires_grad:
                seg_params.append(param)

    # LoRA 参数组（trunk 中仅 lora_A/lora_B 可训练）
    lora_params = []
    if lora_cfg.get("enabled", False):
        lora_params = [
            p for n, p in student_model.encoder.trunk.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad
        ]
        lora_lr_ratio = lora_cfg.get("lr_ratio", 2.0)
        lora_lr = base_lr * lora_lr_ratio
    else:
        lora_lr_ratio = 0.0
        lora_lr = 0.0

    gda_gate_params = []
    gda_gate_lr = 0.0
    if getattr(student_model, "boundary_adapter", None) is not None:
        gda_gate_params = [
            student_model.boundary_adapter.gates
        ] if student_model.boundary_adapter.gates.requires_grad else []
        # Four scalar gates need a much larger step than convolution weights.
        # They remain bounded by tanh in the forward pass.
        gda_gate_lr = float(
            gda_cfg.get("gate_learning_rate", base_lr * 100.0)
        )

    # 构建优化器参数组列表（跳过空组，冻结分支的参数已被排除）
    param_groups = []
    if len(seg_params) > 0:
        param_groups.append({"params": seg_params, "lr": seg_lr, "name": "semantic"})
    if len(boundary_params) > 0:
        param_groups.append({
            "params": boundary_params, "lr": boundary_lr, "name": "boundary_base"
        })
    if len(refine_params) > 0:
        param_groups.append({
            "params": refine_params, "lr": refine_lr, "name": "boundary_refine"
        })
    if len(edge_prior_fusion_params) > 0:
        fusion_lr_ratio = float(edge_prior_cfg.get("learning_rate_ratio", 1.0))
        param_groups.append({
            "params": edge_prior_fusion_params,
            "lr": base_lr * fusion_lr_ratio,
            "name": "edge_prior_fusion",
        })
    if len(lora_params) > 0:
        param_groups.append({"params": lora_params, "lr": lora_lr, "name": "lora"})
    if len(gda_gate_params) > 0:
        param_groups.append({
            "params": gda_gate_params,
            "lr": gda_gate_lr,
            "weight_decay": 0.0,
            "name": "gda_gates",
        })

    if not param_groups:
        raise RuntimeError("No trainable parameter groups were configured")

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
        eps=1e-4,
    )
    logger.info(
        "Layered optimizer (dual FPN): "
        f"Seg lr={seg_lr:.2e} ({len(seg_params)} tensors), "
        f"Boundary base lr={boundary_lr:.2e} ({len(boundary_params)} tensors), "
        f"Refine lr={refine_lr:.2e} ({len(refine_params)} tensors), "
        f"LoRA lr={lora_lr:.2e} ({len(lora_params)} tensors, ratio={lora_lr_ratio}), "
        f"GDA gates lr={gda_gate_lr:.2e} ({len(gda_gate_params)} tensors)"
    )

    # 记录每个参数组创建时的峰值 LR（LambdaLR 会原地改写 group['lr']，
    # 必须在调度器第一次 step 前保存，供自适应 EMA ratio 计算使用）
    group_peaks = [g["lr"] for g in optimizer.param_groups]

    warmup_epochs = semi_cfg.get("warmup_epochs", 5)
    total_epochs = semi_cfg.get("epochs", 50)
    warmup_start_factor = semi_cfg.get("warmup_start_factor", 0.01)
    cosine_end_factor = semi_cfg.get("cosine_end_factor", 0.0)
    lr_schedule_mode = semi_cfg.get("lr_schedule", "cosine")

    if lr_schedule_mode == "flat_decay":
        # 三阶段调度：Warmup → Flat（恒定）→ 温和线性衰减
        flat_epochs = semi_cfg.get("flat_epochs", 30)
        decay_end_factor = semi_cfg.get("decay_end_factor", 0.2)
        flat_end_epoch = warmup_epochs + flat_epochs

        def flat_decay_lambda(epoch):
            """Warmup (linear) → Flat (恒定) → 温和线性衰减。"""
            if epoch < warmup_epochs:
                # Phase 1: Warmup
                return warmup_start_factor + (1.0 - warmup_start_factor) * epoch / max(1, warmup_epochs)
            elif epoch < flat_end_epoch:
                # Phase 2: 恒定 LR（主训练阶段，师生关系稳定）
                return 1.0
            else:
                # Phase 3: 温和线性衰减（不降到 0，避免死区）
                if total_epochs > flat_end_epoch:
                    progress = (epoch - flat_end_epoch) / (total_epochs - flat_end_epoch)
                else:
                    progress = 1.0
                return decay_end_factor + (1.0 - decay_end_factor) * (1.0 - progress)

        lr_lambda_fn = flat_decay_lambda
        logger.info(
            f"LR schedule: flat_decay "
            f"(warmup={warmup_epochs}, flat={flat_epochs}, "
            f"decay_end={decay_end_factor:.1f}, total={total_epochs})"
        )
    else:
        # 旧调度：Warmup + Cosine decay
        cosine_end_factor = semi_cfg.get("cosine_end_factor", 0.0)

        def warmup_cosine_lambda(epoch):
            """Warmup (linear start_factor→1.0) + Cosine decay (1.0→end_factor)。"""
            if epoch < warmup_epochs:
                return warmup_start_factor + (1.0 - warmup_start_factor) * epoch / max(1, warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                return cosine_end_factor + 0.5 * (1.0 - cosine_end_factor) * (1.0 + math.cos(math.pi * progress))

        lr_lambda_fn = warmup_cosine_lambda
        logger.info(
            f"LR schedule: cosine "
            f"(warmup={warmup_epochs}, end_factor={cosine_end_factor:.1f}, total={total_epochs})"
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda_fn,
    )

    use_amp = train_cfg.get("amp", False)
    scaler = GradScaler('cuda', enabled=use_amp)

    ema_decay_base = semi_cfg.get("ema_decay", 0.999)
    adaptive_ema = semi_cfg.get("adaptive_ema", False)
    unsup_weight = semi_cfg.get("unsup_weight", 1.0)
    # 语义一致性损失独立权重（历史问题：语义伪标签可靠度低于边界，
    # 与边界共用 unsup_weight 会随 ramp-up 放大无标签对语义的拖拽）
    unsup_seg_weight = semi_cfg.get("unsup_seg_weight", 0.1)
    # 语义一致性目标温度锐化（方案 A）：T<1 推向 0/1，对抗语义输出塌向 0.5
    unsup_seg_sharpen_temperature = semi_cfg.get("unsup_seg_sharpen_temperature", 1.0)
    unsup_seg_confidence_threshold = semi_cfg.get(
        "unsup_seg_confidence_threshold", 0.5
    )
    unsup_rampup_epochs = semi_cfg.get("unsup_rampup_epochs", 10)

    if adaptive_ema and need_teacher:
        logger.info(
            f"Adaptive EMA: ENABLED (base_decay={ema_decay_base}, "
            f"decay scales with LR ratio)"
        )
    elif need_teacher:
        logger.info(f"Adaptive EMA: DISABLED (fixed decay={ema_decay_base})")
    else:
        logger.info("Adaptive EMA: N/A (EMA teacher not required)")

    if unsup_rampup_epochs > 0:
        logger.info(
            f"Unsup weight ramp-up: {unsup_rampup_epochs} epochs "
            f"(sigmoid ramp-up 0→seg {unsup_seg_weight} / bnd {unsup_weight})"
        )
    else:
        logger.info(
            f"Unsup weight: fixed at seg {unsup_seg_weight} / bnd {unsup_weight} (no ramp-up)"
        )

    # 掩码区域损失权重配置
    seg_mask_region_weight = semi_cfg.get("seg_mask_region_weight", 2.0)
    boundary_mask_region_weight = semi_cfg.get("boundary_mask_region_weight", 0.3)
    logger.info(
        f"Mask region weights: seg={seg_mask_region_weight}, "
        f"boundary={boundary_mask_region_weight}"
    )

    # 边界锚点配置（Stage-1 冻结参考模型）
    boundary_anchor_cfg = semi_cfg.get("boundary_anchor", {})
    anchor_enabled = boundary_anchor_cfg.get("enabled", False)
    ref_model = None

    # stage1_direct / anchor_self 模式必须加载 ref_model（或使用缓存）
    if boundary_teacher_mode in ("stage1_direct", "anchor_self") and not anchor_enabled:
        logger.info(
            f"{boundary_teacher_mode} mode: force-enabling Stage-1 ref model "
            "(required as boundary pseudo-label source)"
        )
        anchor_enabled = True

    if anchor_enabled:
        if boundary_teacher_mode in ("stage1_direct", "anchor_self") and use_cached_pseudo_labels:
            # 使用离线缓存时无需在训练中保留 ref_model（目标直接来自缓存）
            logger.info(
                "Boundary anchor / ref model: SKIPPED "
                f"({boundary_teacher_mode} + 离线伪标签缓存，训练目标来自缓存)"
            )
            ref_model = None
        else:
            # 构建 Stage-1 冻结参考模型（共享 student 的 encoder，仅加载 Stage-1 decoder）
            ref_decoder = FPNDecoder(
                in_channels=student_model.encoder.get_stage_channels(),
                fpn_channels=config["decoder"]["fpn_channels"],
                num_classes=config["decoder"]["num_classes"],
                dropout=config["decoder"]["dropout"],
                use_bn=config["decoder"]["use_bn"],
                center_head=config["decoder"].get("center_head", False),
            )
            ref_checkpoint = torch.load(
                stage1_ckpt_path, map_location=device, weights_only=False
            )
            load_info = ref_decoder.load_state_dict(
                ref_checkpoint["decoder_state_dict"], strict=False
            )
            if load_info.missing_keys or load_info.unexpected_keys:
                logger.info(
                    "Reference checkpoint compatibility: missing=%s unexpected=%s",
                    load_info.missing_keys,
                    load_info.unexpected_keys,
                )
            ref_decoder = ref_decoder.to(device)
            for param in ref_decoder.parameters():
                param.requires_grad = False
            ref_decoder.eval()
            if lora_cfg.get("enabled", False):
                # LoRA 启用时 ref_model 使用独立的无 LoRA 编码器，
                # 保证 Stage-1 锚点仍是纯 Stage-1 特征，不被域适配带偏
                sam2_weight_path = project_path(
                    config, paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"]
                )
                ref_encoder = SAM2Encoder(
                    config_file=sam2_cfg["config_file"],
                    ckpt_path=sam2_weight_path if os.path.exists(sam2_weight_path) else None,
                    device=device,
                    freeze=sam2_cfg["freeze"],
                    sam2_repo_path=os.path.join(paths_cfg["project_root"], sam2_cfg["sam2_repo_path"]),
                )
                ref_model = SegmentationModel(ref_encoder, ref_decoder)
                logger.info(
                    "Boundary anchor / ref model: ENABLED "
                    "(独立无 LoRA 编码器，锚点保持 Stage-1 特征)"
                )
            else:
                ref_model = SegmentationModel(student_model.encoder, ref_decoder)
                logger.info(
                    "Boundary anchor / ref model: ENABLED "
                    "(Stage-1 ref model loaded, shared encoder)"
                )

            anchor_floor = boundary_anchor_cfg.get("anchor_floor", 0.3)
            anchor_ramp_epochs = boundary_anchor_cfg.get("anchor_ramp_epochs", 20)
            logger.info(
                f"  anchor_floor={anchor_floor}, "
                f"ramp_epochs={anchor_ramp_epochs}"
            )
    else:
        logger.info("Boundary anchor / ref model: DISABLED")

    # 边界一致性损失配置（梯度感知：MSE + Sobel + TV）
    bnd_consist_cfg = semi_cfg.get("boundary_consistency", {})
    sobel_weight = bnd_consist_cfg.get("sobel_weight", 1.0)
    tv_weight = bnd_consist_cfg.get("tv_weight", 0.1)
    tv_dilate_radius = bnd_consist_cfg.get("tv_dilate_radius", 3)
    tv_bg_weight = bnd_consist_cfg.get("tv_bg_weight", 1.0)
    tv_boundary_weight = bnd_consist_cfg.get("tv_boundary_weight", 0.1)
    bg_suppress_weight = bnd_consist_cfg.get("bg_suppress_weight", 0.5)
    bg_suppress_threshold = bnd_consist_cfg.get("bg_suppress_threshold", 0.1)
    pos_weight = bnd_consist_cfg.get("pos_weight", 5.0)
    margin_loss_weight = bnd_consist_cfg.get("margin_loss_weight", 0.0)
    margin = bnd_consist_cfg.get("margin", 0.4)
    peak_hinge_weight = bnd_consist_cfg.get("peak_hinge_weight", 0.0)
    peak_threshold = bnd_consist_cfg.get("peak_threshold", 0.8)
    rate_w_start = float(bnd_consist_cfg.get("rate_regularizer_weight", 0.0))
    rate_regularizer_weight = rate_w_start
    rate_w_end = float(
        bnd_consist_cfg.get("rate_regularizer_weight_end", rate_w_start)
    )
    rate_w_ramp = int(bnd_consist_cfg.get("rate_regularizer_ramp_epochs", 0))
    rate_slack = bnd_consist_cfg.get("rate_slack", 0.05)
    sem_boundary_align_weight = semi_cfg.get("sem_boundary_align_weight", 0.0)
    logger.info(
        f"Boundary consistency (gradient): "
        f"sobel_w={sobel_weight}, tv_w={tv_weight}, "
        f"tv_dilate={tv_dilate_radius}, "
        f"tv_bg={tv_bg_weight}, tv_bnd={tv_boundary_weight}, "
        f"bg_suppress_w={bg_suppress_weight}, bg_suppress_th={bg_suppress_threshold}, "
        f"pos_w={pos_weight}, margin_w={margin_loss_weight}, margin={margin}, "
        f"peak_w={peak_hinge_weight}, peak_th={peak_threshold}, "
        f"rate_w={rate_regularizer_weight}, rate_slack={rate_slack}"
    )
    if sem_boundary_align_weight > 0:
        logger.info(
            f"  Semantic-boundary alignment: ENABLED "
            f"(weight={sem_boundary_align_weight})"
        )
    else:
        logger.info("Semantic-boundary alignment: DISABLED")
    if rate_w_ramp > 0 and rate_w_end != rate_w_start:
        logger.info(
            f"  Rate regularizer annealing: {rate_w_start:.2f} -> "
            f"{rate_w_end:.2f} over {rate_w_ramp} epochs "
            "(早期松让边界学起来，晚期紧压雾复现)"
        )

    # 骨架过滤配置（边界伪标签形态学精炼）
    skeleton_filter_cfg = semi_cfg.get("skeleton_filter", {})
    if skeleton_filter_cfg.get("enabled", False):
        logger.info(
            f"Skeleton filter: ENABLED "
            f"(threshold={skeleton_filter_cfg.get('threshold', 0.5)}, "
            f"dilate_width={skeleton_filter_cfg.get('dilate_width', 1)}, "
            f"blur_sigma={skeleton_filter_cfg.get('blur_sigma', 1.0)})"
        )
        sk_th_start = float(skeleton_filter_cfg.get("threshold", 0.5))
        sk_th_end = float(skeleton_filter_cfg.get("threshold_end", 0.7))
        sk_th_ramp = int(skeleton_filter_cfg.get("threshold_ramp_epochs", 0))
        if sk_th_ramp > 0:
            logger.info(
                f"  Skeleton threshold annealing: {sk_th_start:.2f} -> "
                f"{sk_th_end:.2f} over {sk_th_ramp} epochs"
            )
    else:
        logger.info("Skeleton filter: DISABLED")
        sk_th_start = 0.5
        sk_th_end = 0.5
        sk_th_ramp = 0

    # 复合评分权重
    sem_w = train_cfg.get("composite_sem_weight", 0.4)
    bnd_w = train_cfg.get("composite_boundary_weight", 0.6)
    logger.info(f"Best model 保存依据: composite_score = {sem_w:.1f}*mIoU + {bnd_w:.1f}*BndIoU")

    start_epoch = 0
    best_composite_score = 0.0
    best_val_miou = 0.0
    if args.resume:
        resume_path = project_path(config, args.resume)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        validate_checkpoint_architecture(checkpoint, config)
        from models.fpn_decoder import load_decoder_state
        load_decoder_state(student_model.decoder, checkpoint["decoder_state_dict"])
        if "lora_state_dict" in checkpoint and checkpoint["lora_state_dict"]:
            n_lora = load_lora_state_dict(student_model, checkpoint["lora_state_dict"])
            logger.info(f"  LoRA state loaded: {n_lora} 个参数张量")
        load_gda_checkpoint_state(
            student_model, checkpoint,
            required=bool(gda_cfg.get("enabled", False)),
            tag="Resume checkpoint",
        )
        load_edge_prior_checkpoint_state(
            student_model, checkpoint,
            required=bool(edge_prior_cfg.get("enabled", False)),
            tag="Resume checkpoint",
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"调度器状态加载失败（可能因 SequentialLR→LambdaLR 迁移），从 epoch 0 重新调度: {e}")
            start_epoch = checkpoint["epoch"] + 1
            for _ in range(start_epoch):
                scheduler.step()
        else:
            start_epoch = checkpoint["epoch"] + 1
        # 兼容旧 checkpoint 的 best_val_iou key
        best_composite_score = checkpoint.get(
            "best_composite_score", checkpoint.get("best_val_iou", 0.0)
        )
        logger.info(
            f"Resumed from {resume_path} at epoch {start_epoch}, "
            f"best Composite Score: {best_composite_score:.4f}"
        )

    # --init_from_checkpoint: 仅加载 decoder 权重，重置训练状态（用于分支切换）
    # 优先级高于 --resume：如果同时指定，init_from_checkpoint 覆盖 resume
    init_ckpt_path = args.init_from_checkpoint or semi_cfg.get("init_from_checkpoint", "")
    if init_ckpt_path:
        init_ckpt_path = project_path(config, init_ckpt_path)
        if not os.path.exists(init_ckpt_path):
            raise FileNotFoundError(f"Init checkpoint not found: {init_ckpt_path}")
        init_checkpoint = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        from models.fpn_decoder import load_decoder_state
        load_decoder_state(student_model.decoder, init_checkpoint["decoder_state_dict"])
        if "lora_state_dict" in init_checkpoint and init_checkpoint["lora_state_dict"]:
            n_lora = load_lora_state_dict(student_model, init_checkpoint["lora_state_dict"])
            logger.info(f"  LoRA state loaded: {n_lora} 个参数张量")
        load_gda_checkpoint_state(
            student_model, init_checkpoint, required=False,
            tag="Init checkpoint",
        )
        load_edge_prior_checkpoint_state(
            student_model, init_checkpoint, required=False,
            tag="Init checkpoint",
        )
        start_epoch = 0
        best_composite_score = 0.0
        init_epoch = init_checkpoint.get("epoch", "?")
        init_score = init_checkpoint.get(
            "best_composite_score", init_checkpoint.get("best_val_iou", "?")
        )
        logger.info(
            f"Init from checkpoint: {init_ckpt_path}\n"
            f"  Source epoch: {init_epoch}, source best score: {init_score}\n"
            f"  Decoder weights loaded, optimizer/scheduler/epoch RESET for branch switching"
        )

    # 创建教师模型（在所有 checkpoint 加载之后，确保教师同步最新学生权重）
    # 仅当需要 EMA 教师时创建（语义未冻结 或 boundary_teacher_mode="ema"）
    teacher_model = None
    if need_teacher:
        teacher_model = build_teacher_model(student_model)
        logger.info("Teacher model created (EMA of student)")
    else:
        logger.info(
            "Teacher model: SKIPPED "
            "(freeze_seg=True and boundary_teacher_mode != 'ema')"
        )

    diagnostics_cfg = semi_cfg.get("diagnostics", {})
    diagnostics_enabled = bool(diagnostics_cfg.get("enabled", False))
    refine_named_parameters = [
        (name, param)
        for name, param in student_model.decoder.named_parameters()
        if name.startswith("boundary_refine_head.")
    ]
    boundary_base_named_parameters = [
        (name, param)
        for name, param in student_model.decoder.named_parameters()
        if name.startswith("boundary_fpn.") or name.startswith("boundary_branch.")
    ]
    frozen_decoder_named_parameters = [
        (name, param)
        for name, param in student_model.decoder.named_parameters()
        if not param.requires_grad
    ]
    frozen_lora_named_parameters = [
        (name, param)
        for name, param in student_model.encoder.trunk.named_parameters()
        if ("lora_A" in name or "lora_B" in name) and not param.requires_grad
    ]
    gda_gate_named_parameters = []
    if getattr(student_model, "boundary_adapter", None) is not None:
        gda_gate_named_parameters = [
            ("gates", student_model.boundary_adapter.gates)
        ]
    if diagnostics_enabled:
        initial_refine_state = snapshot_parameters(refine_named_parameters)
        initial_boundary_base_state = snapshot_parameters(
            boundary_base_named_parameters
        )
        initial_frozen_decoder_state = snapshot_parameters(
            frozen_decoder_named_parameters
        )
        initial_frozen_lora_state = snapshot_parameters(
            frozen_lora_named_parameters
        )
        initial_gda_gate_state = snapshot_parameters(gda_gate_named_parameters)
        logger.info(
            "Stage-0 diagnostics: ENABLED "
            f"(refine={len(refine_named_parameters)} tensors, "
            f"boundary_base={len(boundary_base_named_parameters)}, "
            f"frozen_decoder={len(frozen_decoder_named_parameters)}, "
            f"frozen_lora={len(frozen_lora_named_parameters)}, "
            f"gda_gates={len(gda_gate_named_parameters)})"
        )
    else:
        initial_refine_state = {}
        initial_boundary_base_state = {}
        initial_frozen_decoder_state = {}
        initial_frozen_lora_state = {}
        initial_gda_gate_state = {}

    if diagnostics_enabled:
        initial_val_metrics = validate(student_model, val_loader, criterion, device)
        initial_composite = (
            sem_w * initial_val_metrics["mean_iou"]
            + bnd_w * initial_val_metrics["boundary_iou"]
        )
        refine_head = getattr(
            student_model.decoder, "boundary_refine_head", None
        )
        refine_out_weight_norm = (
            float(refine_head.out.weight.detach().float().norm())
            if refine_head is not None
            else 0.0
        )
        initial_diagnostics = {
            "optimizer_steps": 0,
            "gda_gate_stats": (
                student_model.boundary_adapter.gate_statistics()
                if getattr(student_model, "boundary_adapter", None) is not None
                else []
            ),
            "refine_out_weight_norm": refine_out_weight_norm,
            "val_loss": initial_val_metrics["loss"],
            "mIoU": initial_val_metrics["mean_iou"],
            "boundary_iou": initial_val_metrics["boundary_iou"],
            "boundary_pos_mean": initial_val_metrics["boundary_pos_mean"],
            "boundary_bg_mean": initial_val_metrics["boundary_bg_mean"],
            "boundary_prob_gap": initial_val_metrics["boundary_prob_gap"],
            "boundary_recall_035": initial_val_metrics["boundary_recall_035"],
            "boundary_bg_fp_rate_035": initial_val_metrics[
                "boundary_bg_fp_rate_035"
            ],
            "mean_dice": initial_val_metrics["mean_dice"],
            "composite": initial_composite,
        }
        initial_path = os.path.join(recorder.run_dir, "initial_diagnostics.json")
        with open(initial_path, "w", encoding="utf-8") as f:
            json.dump(initial_diagnostics, f, ensure_ascii=False, indent=2)
        logger.info(
            "Stage-0 initial validation: "
            f"mIoU={initial_val_metrics['mean_iou']:.4f} "
            f"Bnd IoU={initial_val_metrics['boundary_iou']:.4f} "
            f"Composite={initial_composite:.4f}"
        )
        if bool(diagnostics_cfg.get("save_initial_monitor", True)):
            monitor_inference(student_model, config, -1, device)

    logger.info("=" * 60)
    logger.info("Stage-2 Semi-Supervised Fine-tuning")
    logger.info(f"  Epochs: {total_epochs}")
    logger.info(f"  Boundary teacher mode: {boundary_teacher_mode}")
    if need_teacher:
        logger.info(f"  EMA decay: {ema_decay_base}" + (" (adaptive)" if adaptive_ema else " (fixed)"))
    logger.info(
        f"  Unsupervised weight: seg={unsup_seg_weight} bnd={unsup_weight}"
        + (f" (ramp-up {unsup_rampup_epochs}ep)" if unsup_rampup_epochs > 0 else "")
        + (f" | seg sharpen T={unsup_seg_sharpen_temperature}"
           if unsup_seg_sharpen_temperature < 1.0 else " | seg sharpen OFF")
        + f" | seg conf>={unsup_seg_confidence_threshold:.2f}"
    )
    logger.info(f"  LR: {semi_cfg.get('learning_rate', train_cfg['learning_rate'])}")
    logger.info("=" * 60)

    unlabeled_samples_per_epoch = semi_cfg.get("unlabeled_samples_per_epoch", 0)
    labeled_steps_per_epoch = int(semi_cfg.get("labeled_steps_per_epoch", 0))
    bs_unlabeled = semi_cfg.get("batch_size_unlabeled", 4)
    checkpoint_interval = semi_cfg.get("checkpoint_interval", 5)

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{total_epochs}")

        refine_stage = "disabled"
        if refine_training_enabled:
            boundary_base_trainable = epoch >= refine_only_epochs
            refine_stage = (
                "refine_plus_boundary_base"
                if boundary_base_trainable
                else "refine_only"
            )
            if epoch == start_epoch or epoch == refine_only_epochs:
                student_model.decoder.set_boundary_base_trainable(
                    boundary_base_trainable
                )
                logger.info(f"  B2 stage: {refine_stage}")

        # 骨架过滤阈值逐步提高：早段低阈值保 recall，晚段高阈值剔除
        # 伪标签噪声分支（治"雾状背景先被抑制后又复现"）
        if skeleton_filter_cfg.get("enabled", False) and sk_th_ramp > 0:
            thr = sk_th_start + (sk_th_end - sk_th_start) * min(
                1.0, max(0.0, epoch / max(1, sk_th_ramp))
            )
            skeleton_filter_cfg["threshold"] = float(thr)
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                logger.info(f"  Skeleton threshold: {thr:.3f}")

        # 更新渐进式外观增强的当前 epoch
        if augmentor is not None:
            augmentor.set_epoch(epoch)
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                logger.info(f"  Progressive aug prob: {augmentor.current_prob:.3f}")

        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)
            full_unlabeled_steps = len(unlabeled_loader)
            if unlabeled_samples_per_epoch > 0:
                max_unlabeled_steps = max(1, unlabeled_samples_per_epoch // bs_unlabeled)
                num_unlabeled_steps = min(full_unlabeled_steps, max_unlabeled_steps)
            else:
                num_unlabeled_steps = full_unlabeled_steps
            logger.info(f"  Unlabeled steps: {num_unlabeled_steps}/{full_unlabeled_steps}")
        else:
            unlabeled_iter = None
            num_unlabeled_steps = 0

        num_steps = resolve_epoch_steps(
            labeled_loader_length=len(labeled_loader),
            unlabeled_steps=num_unlabeled_steps,
            labeled_steps_per_epoch=labeled_steps_per_epoch,
        )
        logger.info(
            f"  Supervised steps: {num_steps} "
            f"(loader={len(labeled_loader)}, configured={labeled_steps_per_epoch or 'auto'})"
        )

        # 计算 Stage-1 锚点权重（从 1.0 渐进衰减到 anchor_floor）
        anchor_alpha = 1.0
        if anchor_enabled and boundary_teacher_mode in ("ema", "anchor_self"):
            anchor_floor = boundary_anchor_cfg.get("anchor_floor", 0.3)
            anchor_ramp_epochs = boundary_anchor_cfg.get("anchor_ramp_epochs", 20)
            if anchor_ramp_epochs > 0 and epoch < anchor_ramp_epochs:
                anchor_alpha = 1.0 - (1.0 - anchor_floor) * epoch / anchor_ramp_epochs
            else:
                anchor_alpha = anchor_floor
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                logger.info(f"  Boundary anchor alpha: {anchor_alpha:.3f}")

        # 预测占比上限正则权重退火：早期松（让边界学起来），晚期紧（压雾复现）
        if rate_w_ramp > 0 and rate_w_end != rate_w_start:
            rate_regularizer_weight = rate_w_start + (
                rate_w_end - rate_w_start
            ) * min(1.0, max(0.0, epoch / max(1, rate_w_ramp)))
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                logger.info(f"  Rate reg weight: {rate_regularizer_weight:.3f}")

        # 计算无监督损失权重（sigmoid ramp-up，语义/边界独立）
        unsup_weight_eff = unsup_weight * sigmoid_rampup(epoch, unsup_rampup_epochs)
        unsup_seg_weight_eff = unsup_seg_weight * sigmoid_rampup(epoch, unsup_rampup_epochs)
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            logger.info(
                f"  Unsup weight: seg={unsup_seg_weight_eff:.4f} "
                f"bnd={unsup_weight_eff:.4f} "
                f"(base seg={unsup_seg_weight}, bnd={unsup_weight})"
            )

        # 计算自适应 EMA 衰减系数
        if adaptive_ema and need_teacher:
            lr_ratio = get_current_lr_ratio(scheduler, group_peaks)
            ema_decay_eff = compute_adaptive_ema_decay(ema_decay_base, lr_ratio)
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                logger.info(
                    f"  EMA decay: {ema_decay_eff:.6f} "
                    f"(base={ema_decay_base}, lr_ratio={lr_ratio:.3f})"
                )
        else:
            ema_decay_eff = ema_decay_base

        train_metrics = train_one_epoch(
            student_model, labeled_loader, unlabeled_iter,
            num_steps, num_unlabeled_steps if unlabeled_loader is not None else 0,
            criterion, unsup_weight=unsup_weight_eff,
            unsup_seg_weight=unsup_seg_weight_eff, ema_decay=ema_decay_eff,
            optimizer=optimizer, scaler=scaler, device=device,
            grad_clip=train_cfg.get("grad_clip", 1.0), use_amp=use_amp,
            teacher_model=teacher_model,
            boundary_teacher_mode=boundary_teacher_mode,
            augmentor=augmentor,
            boundary_anchor_cfg=boundary_anchor_cfg,
            ref_model=ref_model,
            anchor_alpha=anchor_alpha,
            skeleton_filter_cfg=skeleton_filter_cfg,
            freeze_seg=freeze_seg,
            freeze_boundary=freeze_boundary,
            seg_mask_region_weight=seg_mask_region_weight,
            boundary_mask_region_weight=boundary_mask_region_weight,
            seg_sharpen_temperature=unsup_seg_sharpen_temperature,
            seg_confidence_threshold=unsup_seg_confidence_threshold,
            sobel_weight=sobel_weight,
            tv_weight=tv_weight,
            tv_dilate_radius=tv_dilate_radius,
            tv_bg_weight=tv_bg_weight,
            tv_boundary_weight=tv_boundary_weight,
            bg_suppress_weight=bg_suppress_weight,
            bg_suppress_threshold=bg_suppress_threshold,
            pos_weight=pos_weight,
            margin_loss_weight=margin_loss_weight,
            margin=margin,
            peak_hinge_weight=peak_hinge_weight,
            peak_threshold=peak_threshold,
            rate_regularizer_weight=rate_regularizer_weight,
            rate_slack=rate_slack,
            sem_boundary_align_weight=sem_boundary_align_weight,
            gda_gate_l1_weight=float(gda_cfg.get("gate_l1_weight", 0.0)),
            diagnostics_enabled=diagnostics_enabled,
        )
        val_metrics = validate(student_model, val_loader, criterion, device)
        scheduler.step()

        refine_delta_l2 = 0.0
        boundary_base_delta_l2 = 0.0
        boundary_base_delta_max = 0.0
        frozen_decoder_max_delta = 0.0
        frozen_lora_max_delta = 0.0
        gda_gate_delta_l2 = 0.0
        gda_gate_values = []
        gda_gate_abs_values = []
        gda_gate_max_values = []
        refine_out_weight_norm = 0.0
        refine_out_weight_max = 0.0
        if diagnostics_enabled:
            refine_delta_l2, _ = parameter_delta_stats(
                refine_named_parameters, initial_refine_state
            )
            boundary_base_delta_l2, boundary_base_delta_max = (
                parameter_delta_stats(
                    boundary_base_named_parameters, initial_boundary_base_state
                )
            )
            _, frozen_decoder_max_delta = parameter_delta_stats(
                frozen_decoder_named_parameters, initial_frozen_decoder_state
            )
            _, frozen_lora_max_delta = parameter_delta_stats(
                frozen_lora_named_parameters, initial_frozen_lora_state
            )
            gda_gate_delta_l2, _ = parameter_delta_stats(
                gda_gate_named_parameters, initial_gda_gate_state
            )
            refine_head = getattr(
                student_model.decoder, "boundary_refine_head", None
            )
            if refine_head is not None:
                refine_out = refine_head.out.weight
                refine_out_weight_norm = float(
                    refine_out.detach().float().norm()
                )
                refine_out_weight_max = float(refine_out.detach().abs().max())
            else:
                refine_out_weight_norm = 0.0
                refine_out_weight_max = 0.0
            logger.info(
                "  Refine diagnostics: "
                f"steps={train_metrics['optimizer_steps']} "
                f"grad_l2={train_metrics['refine_grad_norm']:.3e} "
                f"residual_abs={train_metrics['refine_residual_mean_abs']:.3e} "
                f"residual_std={train_metrics['refine_residual_std']:.3e} "
                f"out_norm={refine_out_weight_norm:.3e} "
                f"delta_l2={refine_delta_l2:.3e} "
                f"base_delta_l2={boundary_base_delta_l2:.3e} "
                f"base_delta_max={boundary_base_delta_max:.3e} "
                f"frozen_max_delta={frozen_decoder_max_delta:.3e} "
                f"lora_max_delta={frozen_lora_max_delta:.3e} "
                f"gda_gate_delta={gda_gate_delta_l2:.3e}"
            )
            if getattr(student_model.decoder, "edge_prior_fusion", None) is not None:
                logger.info(
                    "  Edge-prior fusion: "
                    f"residual_abs={train_metrics['fusion_residual_mean_abs']:.3e} "
                    f"residual_std={train_metrics['fusion_residual_std']:.3e} "
                    f"residual_max={train_metrics['fusion_residual_max_abs']:.3e}"
                )
            if getattr(student_model.decoder, "semantic_residual", None) is not None:
                logger.info(
                    "  Semantic residual: "
                    f"grad_l2={train_metrics['semantic_residual_grad_norm']:.3e} "
                    f"mean_abs={train_metrics['semantic_residual_mean_abs']:.3e} "
                    f"std={train_metrics['semantic_residual_std']:.3e} "
                    f"max_abs={train_metrics['semantic_residual_max_abs']:.3e}"
                )
        if getattr(student_model, "boundary_adapter", None) is not None:
            gate_stats = student_model.boundary_adapter.gate_statistics()
            gda_gate_values = [item["mean"] for item in gate_stats]
            gda_gate_abs_values = [item["mean_abs"] for item in gate_stats]
            gda_gate_max_values = [item["max_abs"] for item in gate_stats]
            logger.info("  GDA gate stats: %s", gate_stats)

        # test 语义置信度监控（新日志参数）：无需 GT，统计概率分布
        # >0.8 高置信占比 / 0.4-0.6 模糊带占比
        seg_high_conf, seg_mid_conf = compute_semantic_confidence_stats(
            student_model, config, device
        )
        if seg_high_conf is not None:
            logger.info(
                f"  Sem conf (test): >0.8={seg_high_conf * 100:.1f}% "
                f"0.4-0.6={seg_mid_conf * 100:.1f}%"
            )
        else:
            logger.info("  Sem conf (test): N/A (no test images)")

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} done ({epoch_time:.1f}s)")
        logger.info(
            f"  Train: total={train_metrics['loss']:.4f} "
            f"sup={train_metrics['sup_loss']:.4f} unsup={train_metrics['unsup_loss']:.4f}"
        )
        logger.info(
            f"    sup_detail: seg={train_metrics['seg']:.4f} "
            f"instance_core={train_metrics['semantic_instance']:.4f} "
            f"bnd={train_metrics['boundary']:.4f}"
        )
        logger.info(
            f"    unsup_detail: s_c={train_metrics['seg_consist']:.4f} "
            f"coverage={train_metrics['seg_conf_coverage']:.3f} "
            f"b_c={train_metrics['boundary_consist']:.4f}"
        )
        logger.info(
            f"    bnd_output: max={train_metrics['bnd_max']:.3f} "
            f">0.5={train_metrics['bnd_pos_frac'] * 100:.1f}% "
            f"gap={train_metrics['bnd_gap']:.3f} "
            f"rate={train_metrics['bnd_pred_rate']:.3f}"
        )
        # 复合评分
        composite_score = (
            sem_w * val_metrics["mean_iou"] + bnd_w * val_metrics["boundary_iou"]
        )

        logger.info(
            f"  Val: loss={val_metrics['loss']:.4f} "
            f"mIoU={val_metrics['mean_iou']:.4f} "
            f"Bnd IoU={val_metrics['boundary_iou']:.4f} "
            f"Composite={composite_score:.4f} "
            f"mDice={val_metrics['mean_dice']:.4f}"
        )
        logger.info(
            f"  pearlite_iou={val_metrics['pearlite_iou']:.4f} "
            f"ferrite_iou={val_metrics['ferrite_iou']:.4f}"
        )
        logger.info(
            "  Val boundary confidence: "
            f"pos={val_metrics['boundary_pos_mean']:.4f} "
            f"bg={val_metrics['boundary_bg_mean']:.4f} "
            f"gap={val_metrics['boundary_prob_gap']:.4f} "
            f"recall@0.35={val_metrics['boundary_recall_035']:.4f} "
            f"bg_fp@0.35={val_metrics['boundary_bg_fp_rate_035']:.4f}"
        )

        # 语义退化预警：验证 mIoU 明显低于历史最优时提示（语义崩塌的早期信号）
        sem_warn_factor = semi_cfg.get("sem_degrade_warn_factor", 0.9)
        if sem_warn_factor > 0:
            if val_metrics["mean_iou"] > best_val_miou:
                best_val_miou = val_metrics["mean_iou"]
            elif (
                epoch >= 10
                and best_val_miou > 0.1
                and val_metrics["mean_iou"] < best_val_miou * sem_warn_factor
            ):
                logger.warning(
                    f"Semantic degradation detected: val mIoU={val_metrics['mean_iou']:.4f} "
                    f"< {sem_warn_factor} x best={best_val_miou:.4f}. "
                    f"建议检查语义头是否被对齐项/过高 lr 破坏"
                )

        # 逐 epoch 指标落盘（与配置快照同目录，便于复现/对比）
        lr_by_group = {
            group.get("name", f"group_{idx}"): group["lr"]
            for idx, group in enumerate(optimizer.param_groups)
        }
        recorder.append_metrics({
            "epoch": epoch + 1,
            "refine_stage": refine_stage,
            "lr_boundary_base": lr_by_group.get("boundary_base", 0.0),
            "lr_boundary_refine": lr_by_group.get("boundary_refine", 0.0),
            "lr_edge_prior_fusion": lr_by_group.get("edge_prior_fusion", 0.0),
            "lr_gda_gates": lr_by_group.get("gda_gates", 0.0),
            "train_loss": train_metrics["loss"],
            "sup_loss": train_metrics["sup_loss"],
            "unsup_loss": train_metrics["unsup_loss"],
            "seg": train_metrics["seg"],
            "semantic_instance": train_metrics["semantic_instance"],
            "boundary": train_metrics["boundary"],
            "seg_consist": train_metrics["seg_consist"],
            "seg_conf_coverage": train_metrics["seg_conf_coverage"],
            "boundary_consist": train_metrics["boundary_consist"],
            "gda_gate_l1": train_metrics["gda_gate_l1"],
            "seg_high_conf": seg_high_conf if seg_high_conf is not None else -1.0,
            "seg_mid_conf": seg_mid_conf if seg_mid_conf is not None else -1.0,
            "bnd_max": train_metrics["bnd_max"],
            "bnd_pos_frac": train_metrics["bnd_pos_frac"],
            "bnd_gap": train_metrics["bnd_gap"],
            "bnd_pred_rate": train_metrics["bnd_pred_rate"],
            "optimizer_steps": train_metrics["optimizer_steps"],
            "refine_grad_norm": train_metrics["refine_grad_norm"],
            "semantic_residual_grad_norm": train_metrics[
                "semantic_residual_grad_norm"
            ],
            "refine_residual_mean_abs": train_metrics["refine_residual_mean_abs"],
            "refine_residual_std": train_metrics["refine_residual_std"],
            "refine_residual_max_abs": train_metrics["refine_residual_max_abs"],
            "fusion_residual_mean_abs": train_metrics["fusion_residual_mean_abs"],
            "fusion_residual_std": train_metrics["fusion_residual_std"],
            "fusion_residual_max_abs": train_metrics["fusion_residual_max_abs"],
            "semantic_residual_mean_abs": train_metrics[
                "semantic_residual_mean_abs"
            ],
            "semantic_residual_std": train_metrics["semantic_residual_std"],
            "semantic_residual_max_abs": train_metrics[
                "semantic_residual_max_abs"
            ],
            "refine_out_weight_norm": refine_out_weight_norm,
            "refine_out_weight_max": refine_out_weight_max,
            "refine_delta_l2": refine_delta_l2,
            "boundary_base_delta_l2": boundary_base_delta_l2,
            "boundary_base_delta_max": boundary_base_delta_max,
            "frozen_decoder_max_delta": frozen_decoder_max_delta,
            "frozen_lora_max_delta": frozen_lora_max_delta,
            "gda_gate_delta_l2": gda_gate_delta_l2,
            "gda_gate_0": gda_gate_values[0] if gda_gate_values else 0.0,
            "gda_gate_1": gda_gate_values[1] if gda_gate_values else 0.0,
            "gda_gate_2": gda_gate_values[2] if gda_gate_values else 0.0,
            "gda_gate_3": gda_gate_values[3] if gda_gate_values else 0.0,
            "gda_gate_abs_0": gda_gate_abs_values[0] if gda_gate_abs_values else 0.0,
            "gda_gate_abs_1": gda_gate_abs_values[1] if gda_gate_abs_values else 0.0,
            "gda_gate_abs_2": gda_gate_abs_values[2] if gda_gate_abs_values else 0.0,
            "gda_gate_abs_3": gda_gate_abs_values[3] if gda_gate_abs_values else 0.0,
            "gda_gate_max_0": gda_gate_max_values[0] if gda_gate_max_values else 0.0,
            "gda_gate_max_1": gda_gate_max_values[1] if gda_gate_max_values else 0.0,
            "gda_gate_max_2": gda_gate_max_values[2] if gda_gate_max_values else 0.0,
            "gda_gate_max_3": gda_gate_max_values[3] if gda_gate_max_values else 0.0,
            "val_loss": val_metrics["loss"],
            "mIoU": val_metrics["mean_iou"],
            "boundary_iou": val_metrics["boundary_iou"],
            "boundary_pos_mean": val_metrics["boundary_pos_mean"],
            "boundary_bg_mean": val_metrics["boundary_bg_mean"],
            "boundary_prob_gap": val_metrics["boundary_prob_gap"],
            "boundary_recall_035": val_metrics["boundary_recall_035"],
            "boundary_bg_fp_rate_035": val_metrics["boundary_bg_fp_rate_035"],
            "mean_dice": val_metrics["mean_dice"],
            "composite": composite_score,
        })

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_path = os.path.join(output_dir, "best_model_stage2.pth")
            torch.save(build_checkpoint(
                model=student_model, config=config, epoch=epoch,
                lora_state_dict=extract_lora_state_dict(student_model),
                best_composite_score=best_composite_score,
                optimizer=optimizer, scheduler=scheduler,
            ), best_path)
            logger.info(
                f"  New best model saved: {best_path} "
                f"(composite={best_composite_score:.4f}, "
                f"mIoU={val_metrics['mean_iou']:.4f}, "
                f"bndIoU={val_metrics['boundary_iou']:.4f})"
            )
            recorder.copy_checkpoint(best_path)

        if (epoch + 1) % checkpoint_interval == 0 and semi_cfg.get("save_checkpoints", True):
            ckpt_path = os.path.join(output_dir, f"stage2_epoch{epoch + 1}.pth")
            torch.save(build_checkpoint(
                model=student_model, config=config, epoch=epoch,
                lora_state_dict=extract_lora_state_dict(student_model),
                best_composite_score=best_composite_score,
                optimizer=optimizer, scheduler=scheduler,
            ), ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

        # 监控推理：每 checkpoint_interval 个 epoch 保存概率图
        if (epoch + 1) % checkpoint_interval == 0:
            monitor_inference(student_model, config, epoch, device)

    final_path = os.path.join(output_dir, "final_model_stage2.pth")
    torch.save(build_checkpoint(
        model=student_model, config=config, epoch=total_epochs - 1,
        lora_state_dict=extract_lora_state_dict(student_model),
        best_composite_score=best_composite_score,
    ), final_path)
    recorder.copy_checkpoint(final_path, "final_model_stage2.pth")
    logger.info(f"Stage-2 training complete! Final model: {final_path}")
    logger.info(f"Best Composite Score: {best_composite_score:.4f}")
    logger.info(f"Run artifacts: {recorder.run_dir}")


if __name__ == "__main__":
    main()
