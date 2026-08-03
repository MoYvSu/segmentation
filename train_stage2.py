# -*- coding: utf-8 -*-
"""
第二阶段半监督微调主入口（边界预测版本 - 多模式伪标签源）
=========================================================
使用有标签数据（BoundaryLoss）与无标签数据（一致性损失）联合训练。

技术路线：
1. 加载第一阶段最优权重到学生模型
2. 可选创建教师模型（学生权重的 EMA 副本，仅 ema 模式需要）
3. 仅冻结 encoder，全量训练 decoder
4. 双流混合 Batch：itertools.cycle(labeled_loader) + unlabeled_loader
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
import itertools
import logging
import math
import os
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

import yaml
from data.dataset import BoundaryDataset, collate_fn, letterbox
from data.dataset_semi import (
    LabeledDataset,
    UnlabeledDataset,
    labeled_collate_fn,
    unlabeled_collate_fn,
)
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.sam2_encoder import SAM2Encoder
from utils.loss import BoundaryLoss
from utils.loss_semi import compute_unsupervised_loss, update_ema
from utils.metrics import SegMetrics
from utils.progressive_aug import ProgressiveAppearanceAug
from utils.run_recorder import RunRecorder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def get_current_lr_ratio(scheduler, base_lr):
    """获取当前学习率相对于峰值学习率的比例。

    Args:
        scheduler: LambdaLR 调度器
        base_lr: 峰值学习率

    Returns:
        [0, 1] 之间的比例
    """
    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    return max(0.0, min(1.0, current_lr / max(base_lr, 1e-12)))


def build_model(config, device):
    """构建学生模型。"""
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]

    ckpt_path = os.path.join(paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    if not os.path.exists(ckpt_path):
        logger.warning(f"SAM 2 checkpoint not found: {ckpt_path}")

    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
        device=device,
        freeze=sam2_cfg["freeze"],
        sam2_repo_path=os.path.join(paths_cfg["project_root"], sam2_cfg["sam2_repo_path"]),
    )

    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=decoder_cfg["fpn_channels"],
        num_classes=decoder_cfg["num_classes"],
        dropout=decoder_cfg["dropout"],
        use_bn=decoder_cfg["use_bn"],
    )

    model = SegmentationModel(encoder, decoder)
    model = model.to(device)
    return model


def build_teacher_model(student_model):
    """创建教师模型（学生权重的深拷贝）。"""
    teacher_model = copy.deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    return teacher_model


def load_stage1_checkpoint(model, checkpoint_path, device):
    """加载第一阶段最优权重到学生模型 decoder。"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    logger.info(f"Stage-1 checkpoint loaded: {checkpoint_path}")
    # 兼容新旧 checkpoint key
    best_score = checkpoint.get(
        "best_composite_score", checkpoint.get("best_val_iou", "?")
    )
    logger.info(
        f"  Epoch: {checkpoint.get('epoch', '?')}, "
        f"Best Score: {best_score}"
    )


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
    crop_size = data_cfg.get("crop_size", 0)

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
    )

    val_dataset = BoundaryDataset(
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
    )

    if len(labeled_dataset) == 0:
        logger.error(f"Labeled dataset is empty: {data_dir}")
        sys.exit(1)

    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty, using labeled dataset for validation.")
        val_dataset = labeled_dataset

    unlabeled_dataset = None
    if os.path.exists(unlabeled_dir) and len(os.listdir(unlabeled_dir)) > 0:
        unlabeled_dataset = UnlabeledDataset(
            data_dir=unlabeled_dir,
            image_size=image_size,
            patch_mask_ratio=semi_cfg.get("patch_mask_ratio", 0.3),
            patch_mask_size=semi_cfg.get("patch_mask_size", 64),
            num_patches=semi_cfg.get("num_patches", 8),
            enable_appearance_aug=not disable_unlabeled_appearance_aug,
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

    labeled_loader = DataLoader(
        labeled_dataset, batch_size=bs_labeled, shuffle=True,
        num_workers=num_workers, collate_fn=labeled_collate_fn,
        pin_memory=True, drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs_labeled, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )

    unlabeled_loader = None
    if unlabeled_dataset is not None:
        unlabeled_loader = DataLoader(
            unlabeled_dataset, batch_size=bs_unlabeled, shuffle=True,
            num_workers=num_workers, collate_fn=unlabeled_collate_fn,
            pin_memory=True, drop_last=True,
        )

    logger.info(f"Labeled batch size: {bs_labeled}")
    logger.info(f"Unlabeled batch size: {bs_unlabeled if unlabeled_loader else 'N/A'}")

    return labeled_loader, unlabeled_loader, val_loader


def train_one_epoch(
    student_model, labeled_loader, unlabeled_iter,
    num_steps, num_unlabeled_steps, criterion, unsup_weight, ema_decay,
    optimizer, scaler, device, grad_clip=1.0, use_amp=False,
    teacher_model=None, boundary_teacher_mode="ema",
    augmentor=None, boundary_anchor_cfg=None, ref_model=None, anchor_alpha=1.0,
    skeleton_filter_cfg=None, freeze_seg=False, freeze_boundary=False,
    seg_mask_region_weight=2.0, boundary_mask_region_weight=0.3,
    sobel_weight=1.0, tv_weight=0.1, tv_dilate_radius=3,
    tv_bg_weight=1.0, tv_boundary_weight=0.1,
    bg_suppress_weight=0.5, bg_suppress_threshold=0.1,
    pos_weight=5.0, margin_loss_weight=0.0, margin=0.4,
    rate_regularizer_weight=0.0, rate_slack=0.05,
    sem_boundary_align_weight=0.0,
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
        sobel_weight: Sobel 梯度一致性损失权重。
        tv_weight: 各向异性 TV 正则化权重。
        tv_dilate_radius: TV 中边界区域膨胀半径（px）。
        tv_bg_weight: 非边界区域 TV 权重。
        tv_boundary_weight: 边界区域 TV 权重。
        bg_suppress_weight: 背景抑制损失权重。
        bg_suppress_threshold: 背景抑制阈值，低于此值视为背景区域。
    """
    student_model.train()
    student_model.encoder.eval()
    if teacher_model is not None:
        teacher_model.eval()

    total_loss_sum = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    total_seg = 0.0
    total_boundary = 0.0
    total_seg_consist = 0.0
    total_boundary_consist = 0.0
    total_bnd_max = 0.0
    total_bnd_pos = 0.0
    total_bnd_gap = 0.0
    total_bnd_rate = 0.0
    n_steps = 0

    clip_params = list(student_model.decoder.parameters())
    labeled_iter_cycle = itertools.cycle(labeled_loader)

    for step_idx in range(num_steps):
        labeled_batch = next(labeled_iter_cycle)
        images_labeled = labeled_batch["image"].to(device)
        targets_labeled = labeled_batch["target"].to(device)
        weights_labeled = labeled_batch["weight"].to(device)

        # 渐进式外观增强：仅施加于学生输入（有标签路径）
        if augmentor is not None:
            images_labeled = augmentor(images_labeled)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                out_labeled = student_model(images_labeled, output_size=targets_labeled.shape[-2:])
                sup_loss, seg_val, boundary_val = criterion(out_labeled, targets_labeled, weights_labeled)
        else:
            out_labeled = student_model(images_labeled, output_size=targets_labeled.shape[-2:])
            sup_loss, seg_val, boundary_val = criterion(out_labeled, targets_labeled, weights_labeled)

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
                        unsup_loss, seg_consist_val, boundary_consist_val, bnd_stats = (
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
                                rate_regularizer_weight=rate_regularizer_weight,
                                rate_slack=rate_slack,
                                sem_boundary_align_weight=sem_boundary_align_weight,
                            )
                        )
                else:
                    unsup_loss, seg_consist_val, boundary_consist_val, bnd_stats = (
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
                            rate_regularizer_weight=rate_regularizer_weight,
                            rate_slack=rate_slack,
                            sem_boundary_align_weight=sem_boundary_align_weight,
                        )
                    )
            except StopIteration:
                pass

        total_loss = sup_loss + unsup_weight * unsup_loss

        if use_amp:
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
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
        total_boundary += boundary_val
        total_seg_consist += seg_consist_val
        total_boundary_consist += boundary_consist_val
        total_bnd_max += bnd_stats.get("bnd_max", 0.0)
        total_bnd_pos += bnd_stats.get("bnd_pos_frac", 0.0)
        total_bnd_gap += bnd_stats.get("bnd_gap", 0.0)
        total_bnd_rate += bnd_stats.get("bnd_pred_rate", 0.0)
        n_steps += 1

        if (step_idx + 1) % 5 == 0:
            logger.info(
                f"  Step {step_idx + 1}/{num_steps}: "
                f"total={total_loss.item():.4f} "
                f"sup={sup_loss.item():.4f} (seg={seg_val:.4f} bnd={boundary_val:.4f}) "
                f"unsup={unsup_loss.item():.4f} (s_c={seg_consist_val:.4f} b_c={boundary_consist_val:.4f})"
            )

    n = max(n_steps, 1)
    return {
        "loss": total_loss_sum / n,
        "sup_loss": total_sup_loss / n,
        "unsup_loss": total_unsup_loss / n,
        "seg": total_seg / n,
        "boundary": total_boundary / n,
        "seg_consist": total_seg_consist / n,
        "boundary_consist": total_boundary_consist / n,
        "bnd_max": total_bnd_max / n,
        "bnd_pos_frac": total_bnd_pos / n,
        "bnd_gap": total_bnd_gap / n,
        "bnd_pred_rate": total_bnd_rate / n,
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

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        weights = batch["weight"].to(device)
        output = model(images, output_size=targets.shape[-2:])
        total_loss_t, _, _ = criterion(output, targets, weights)
        total_loss += total_loss_t.item()
        n_batches += 1

        seg_logits = output[:, 0]
        pred = (torch.sigmoid(seg_logits) > 0.5).long()
        mask = targets[:, 0].long()
        metrics.update_tensor(pred, mask)

        # 边界通道 IoU
        bnd_logits = output[:, 1]
        bnd_pred = (torch.sigmoid(bnd_logits) > 0.5).long()
        bnd_gt = targets[:, 1].long()
        bnd_tp += ((bnd_pred == 1) & (bnd_gt == 1)).sum().item()
        bnd_fp += ((bnd_pred == 1) & (bnd_gt == 0)).sum().item()
        bnd_fn += ((bnd_pred == 0) & (bnd_gt == 1)).sum().item()

    val_metrics = metrics.get_metrics()
    val_metrics["loss"] = total_loss / max(n_batches, 1)

    # Boundary IoU = TP / (TP + FP + FN + eps)
    eps = 1e-7
    val_metrics["boundary_iou"] = bnd_tp / (bnd_tp + bnd_fp + bnd_fn + eps)

    return val_metrics


@torch.no_grad()
def monitor_inference(model, config, epoch, device):
    """对 data/test 前 N 张图像做轻量推理，保存语义+边界概率图。

    保存结构：output_dir/epoch_XX/{basename}_seg.png / {basename}_boundary.png
    """
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    monitor_cfg = config.get("semi_supervised", {}).get("monitor", {})

    image_dir = os.path.join(
        paths_cfg["project_root"],
        monitor_cfg.get("image_dir", "data/test"),
    )
    num_images = monitor_cfg.get("num_images", 3)
    output_base = os.path.join(
        paths_cfg["project_root"],
        monitor_cfg.get("output_dir", "outputs/stage2/monitor"),
    )
    image_size = data_cfg.get("image_size", 1024)

    epoch_dir = os.path.join(output_base, f"epoch_{epoch + 1:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        logger.warning(f"Monitor: no images found in {image_dir}")
        return

    image_paths = image_paths[:num_images]

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

        # 语义概率图：伪彩色（JET），珠光体=蓝，铁素体=红
        seg_vis = (seg_prob * 255).astype(np.uint8)
        seg_color = cv2.applyColorMap(seg_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_seg.png"), seg_color)

        # 边界概率图：热力图（灰度→伪彩色）
        bnd_vis = (bnd_prob * 255).astype(np.uint8)
        bnd_color = cv2.applyColorMap(bnd_vis, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_boundary.png"), bnd_color)

    logger.info(f"  Monitor inference saved: {epoch_dir} ({len(image_paths)} images)")


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

    stage1_ckpt_path = os.path.join(
        paths_cfg["project_root"],
        config["inference"].get("stage1_checkpoint", "outputs/stage1/best_model.pth"),
    )
    load_stage1_checkpoint(student_model, stage1_ckpt_path, device)

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
        freeze_seg=freeze_seg,
        freeze_boundary=freeze_boundary,
    ).to(device)
    logger.info("Supervised loss: BoundaryLoss (semantic BCE + boundary Focal x EDT)")

    # 分层参数优化器：语义分支和边界分支各自独立学习率（方案 B - 三分组）
    base_lr = semi_cfg.get("learning_rate", train_cfg["learning_rate"])
    seg_lr_ratio = semi_cfg.get("seg_lr_ratio", 0.1)
    boundary_lr_ratio = semi_cfg.get("boundary_lr_ratio", 0.1)
    seg_lr = base_lr * seg_lr_ratio
    boundary_lr = base_lr * boundary_lr_ratio

    seg_params = []
    boundary_params = []
    for name, param in student_model.decoder.named_parameters():
        if not param.requires_grad:
            continue
        if "seg_fpn" in name or "seg_branch" in name:
            seg_params.append(param)
        elif "boundary_fpn" in name or "boundary_branch" in name:
            boundary_params.append(param)
        else:
            seg_params.append(param)

    # 构建优化器参数组列表（跳过空组，冻结分支的参数已被排除）
    param_groups = []
    if len(seg_params) > 0:
        param_groups.append({"params": seg_params, "lr": seg_lr})
    if len(boundary_params) > 0:
        param_groups.append({"params": boundary_params, "lr": boundary_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
        eps=1e-4,
    )
    logger.info(f"Layered optimizer (dual FPN): "
                f"Seg lr={seg_lr:.2e} ({len(seg_params)} params), "
                f"Boundary lr={boundary_lr:.2e} ({len(boundary_params)} params)")

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
            f"(sigmoid ramp-up 0→{unsup_weight})"
        )
    else:
        logger.info(f"Unsup weight: fixed at {unsup_weight} (no ramp-up)")

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
            )
            ref_checkpoint = torch.load(
                stage1_ckpt_path, map_location=device, weights_only=False
            )
            ref_decoder.load_state_dict(ref_checkpoint["decoder_state_dict"])
            ref_decoder = ref_decoder.to(device)
            for param in ref_decoder.parameters():
                param.requires_grad = False
            ref_decoder.eval()
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
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        student_model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
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
        logger.info(f"Resumed from epoch {start_epoch}, best Composite Score: {best_composite_score:.4f}")

    # --init_from_checkpoint: 仅加载 decoder 权重，重置训练状态（用于分支切换）
    # 优先级高于 --resume：如果同时指定，init_from_checkpoint 覆盖 resume
    init_ckpt_path = args.init_from_checkpoint or semi_cfg.get("init_from_checkpoint", "")
    if init_ckpt_path:
        if not os.path.exists(init_ckpt_path):
            raise FileNotFoundError(f"Init checkpoint not found: {init_ckpt_path}")
        init_checkpoint = torch.load(init_ckpt_path, map_location=device, weights_only=False)
        student_model.decoder.load_state_dict(init_checkpoint["decoder_state_dict"])
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

    logger.info("=" * 60)
    logger.info("Stage-2 Semi-Supervised Fine-tuning")
    logger.info(f"  Epochs: {total_epochs}")
    logger.info(f"  Boundary teacher mode: {boundary_teacher_mode}")
    if need_teacher:
        logger.info(f"  EMA decay: {ema_decay_base}" + (" (adaptive)" if adaptive_ema else " (fixed)"))
    logger.info(f"  Unsupervised weight: {unsup_weight}" + (f" (ramp-up {unsup_rampup_epochs}ep)" if unsup_rampup_epochs > 0 else ""))
    logger.info(f"  LR: {semi_cfg.get('learning_rate', train_cfg['learning_rate'])}")
    logger.info("=" * 60)

    unlabeled_samples_per_epoch = semi_cfg.get("unlabeled_samples_per_epoch", 0)
    bs_unlabeled = semi_cfg.get("batch_size_unlabeled", 4)
    checkpoint_interval = semi_cfg.get("checkpoint_interval", 5)

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{total_epochs}")

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
            num_steps = max(num_unlabeled_steps, len(labeled_loader))
            logger.info(f"  Unlabeled steps: {num_unlabeled_steps}/{full_unlabeled_steps}")
        else:
            unlabeled_iter = None
            num_steps = len(labeled_loader)

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

        # 计算无监督损失权重（sigmoid ramp-up）
        unsup_weight_eff = unsup_weight * sigmoid_rampup(epoch, unsup_rampup_epochs)
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            logger.info(f"  Unsup weight: {unsup_weight_eff:.4f} (base={unsup_weight})")

        # 计算自适应 EMA 衰减系数
        if adaptive_ema and need_teacher:
            lr_ratio = get_current_lr_ratio(scheduler, base_lr)
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
            criterion, unsup_weight=unsup_weight_eff, ema_decay=ema_decay_eff,
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
            rate_regularizer_weight=rate_regularizer_weight,
            rate_slack=rate_slack,
            sem_boundary_align_weight=sem_boundary_align_weight,
        )
        val_metrics = validate(student_model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} done ({epoch_time:.1f}s)")
        logger.info(
            f"  Train: total={train_metrics['loss']:.4f} "
            f"sup={train_metrics['sup_loss']:.4f} unsup={train_metrics['unsup_loss']:.4f}"
        )
        logger.info(
            f"    sup_detail: seg={train_metrics['seg']:.4f} bnd={train_metrics['boundary']:.4f}"
        )
        logger.info(
            f"    unsup_detail: s_c={train_metrics['seg_consist']:.4f} "
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
        recorder.append_metrics({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "sup_loss": train_metrics["sup_loss"],
            "unsup_loss": train_metrics["unsup_loss"],
            "seg": train_metrics["seg"],
            "boundary": train_metrics["boundary"],
            "seg_consist": train_metrics["seg_consist"],
            "boundary_consist": train_metrics["boundary_consist"],
            "bnd_max": train_metrics["bnd_max"],
            "bnd_pos_frac": train_metrics["bnd_pos_frac"],
            "bnd_gap": train_metrics["bnd_gap"],
            "bnd_pred_rate": train_metrics["bnd_pred_rate"],
            "val_loss": val_metrics["loss"],
            "mIoU": val_metrics["mean_iou"],
            "boundary_iou": val_metrics["boundary_iou"],
            "mean_dice": val_metrics["mean_dice"],
            "composite": composite_score,
        })

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_path = os.path.join(output_dir, "best_model_stage2.pth")
            torch.save({
                "epoch": epoch,
                "decoder_state_dict": student_model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_composite_score": best_composite_score,
                "config": config,
            }, best_path)
            logger.info(
                f"  New best model saved: {best_path} "
                f"(composite={best_composite_score:.4f}, "
                f"mIoU={val_metrics['mean_iou']:.4f}, "
                f"bndIoU={val_metrics['boundary_iou']:.4f})"
            )
            recorder.copy_checkpoint(best_path)

        if (epoch + 1) % checkpoint_interval == 0 and semi_cfg.get("save_checkpoints", True):
            ckpt_path = os.path.join(output_dir, f"stage2_epoch{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "decoder_state_dict": student_model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_composite_score": best_composite_score,
                "config": config,
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

        # 监控推理：每 checkpoint_interval 个 epoch 保存概率图
        if (epoch + 1) % checkpoint_interval == 0:
            monitor_inference(student_model, config, epoch, device)

    final_path = os.path.join(output_dir, "final_model_stage2.pth")
    torch.save({
        "epoch": total_epochs - 1,
        "decoder_state_dict": student_model.decoder.state_dict(),
        "best_composite_score": best_composite_score,
        "config": config,
    }, final_path)
    recorder.copy_checkpoint(final_path, "final_model_stage2.pth")
    logger.info(f"Stage-2 training complete! Final model: {final_path}")
    logger.info(f"Best Composite Score: {best_composite_score:.4f}")
    logger.info(f"Run artifacts: {recorder.run_dir}")


if __name__ == "__main__":
    main()
