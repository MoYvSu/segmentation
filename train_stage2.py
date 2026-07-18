# -*- coding: utf-8 -*-
"""
第二阶段半监督微调主入口（边界预测版本 - Mean Teacher）
=========================================================
使用有标签数据（BoundaryLoss）与无标签数据（Mean Teacher 一致性）联合训练。

技术路线：
1. 加载第一阶段最优权重到学生模型
2. 创建教师模型（学生权重的 EMA 副本）
3. 仅冻结 encoder，全量训练 decoder
4. 双流混合 Batch：itertools.cycle(labeled_loader) + unlabeled_loader
5. 有标签流：BoundaryLoss（语义 BCE + 边界 Focal x EDT 权重）
6. 无标签流：Mean Teacher 一致性损失（MSE，遮挡区域加权）
7. 每个 step 结束后更新教师模型 EMA 权重

使用方法：
    conda activate sam2_env
    python train_stage2.py --config config/default_config.yaml
"""

import argparse
import copy
import itertools
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from data.dataset import BoundaryDataset, collate_fn
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_dataloaders(config):
    """构建有标签和无标签 DataLoader。"""
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
    student_model, teacher_model, labeled_loader, unlabeled_iter,
    num_steps, num_unlabeled_steps, criterion, unsup_weight, ema_decay,
    optimizer, scaler, device, grad_clip=1.0, use_amp=False,
):
    """训练一个 epoch（双流混合 Batch + EMA 更新）。"""
    student_model.train()
    student_model.encoder.eval()
    teacher_model.eval()

    total_loss_sum = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    total_seg = 0.0
    total_boundary = 0.0
    total_seg_consist = 0.0
    total_boundary_consist = 0.0
    n_steps = 0

    clip_params = list(student_model.decoder.parameters())
    labeled_iter_cycle = itertools.cycle(labeled_loader)

    for step_idx in range(num_steps):
        labeled_batch = next(labeled_iter_cycle)
        images_labeled = labeled_batch["image"].to(device)
        targets_labeled = labeled_batch["target"].to(device)
        weights_labeled = labeled_batch["weight"].to(device)

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

        if unlabeled_iter is not None and step_idx < num_unlabeled_steps:
            try:
                unlabeled_batch = next(unlabeled_iter)
                img_weak = unlabeled_batch["img_weak"]
                img_strong = unlabeled_batch["img_strong"]
                patch_mask = unlabeled_batch["patch_mask"]

                if use_amp:
                    with autocast('cuda'):
                        unsup_loss, seg_consist_val, boundary_consist_val = (
                            compute_unsupervised_loss(
                                student_model, teacher_model,
                                img_weak, img_strong, patch_mask,
                                output_size=targets_labeled.shape[-2:],
                            )
                        )
                else:
                    unsup_loss, seg_consist_val, boundary_consist_val = (
                        compute_unsupervised_loss(
                            student_model, teacher_model,
                            img_weak, img_strong, patch_mask,
                            output_size=targets_labeled.shape[-2:],
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

        update_ema(teacher_model, student_model, ema_decay)

        total_loss_sum += total_loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        total_seg += seg_val
        total_boundary += boundary_val
        total_seg_consist += seg_consist_val
        total_boundary_consist += boundary_consist_val
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


def main():
    parser = argparse.ArgumentParser(
        description="Stage-2 Semi-Supervised Fine-tuning (Mean Teacher + Boundary Prediction)"
    )
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
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

    output_dir = os.path.join(paths_cfg["project_root"], semi_cfg.get("output_dir", "outputs/stage2"))
    os.makedirs(output_dir, exist_ok=True)

    student_model = build_model(config, device)

    stage1_ckpt_path = os.path.join(
        paths_cfg["project_root"],
        config["inference"].get("stage1_checkpoint", "outputs/stage1/best_model.pth"),
    )
    load_stage1_checkpoint(student_model, stage1_ckpt_path, device)

    teacher_model = build_teacher_model(student_model)
    logger.info("Teacher model created (EMA of student)")

    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(config)

    criterion = BoundaryLoss(
        gamma=train_cfg.get("focal_gamma", 2.0),
        alpha_boundary=train_cfg.get("boundary_alpha", 1.0),
        alpha_focal=train_cfg.get("focal_alpha", 0.75),
    ).to(device)
    logger.info("Supervised loss: BoundaryLoss (semantic BCE + boundary Focal x EDT)")

    # 分层参数优化器：FPN 组件低学习率，解耦头标准学习率
    base_lr = semi_cfg.get("learning_rate", train_cfg["learning_rate"])
    fpn_lr_ratio = semi_cfg.get("fpn_lr_ratio", 0.1)  # FPN 学习率 = 0.1 * base_lr
    fpn_lr = base_lr * fpn_lr_ratio
    head_lr = base_lr

    fpn_params = []
    head_params = []
    for name, param in student_model.decoder.named_parameters():
        if not param.requires_grad:
            continue
        if "seg_branch" in name or "boundary_branch" in name:
            head_params.append(param)
        else:
            fpn_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": fpn_params, "lr": fpn_lr},
            {"params": head_params, "lr": head_lr},
        ],
        lr=base_lr,
        weight_decay=train_cfg["weight_decay"],
        eps=1e-4,
    )
    logger.info(f"Layered optimizer: FPN lr={fpn_lr:.2e} ({len(fpn_params)} params), "
                f"Head lr={head_lr:.2e} ({len(head_params)} params)")

    warmup_epochs = semi_cfg.get("warmup_epochs", 5)
    total_epochs = semi_cfg.get("epochs", 50)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            cosine_scheduler,
        ],
        milestones=[warmup_epochs],
    )

    use_amp = train_cfg.get("amp", False)
    scaler = GradScaler('cuda', enabled=use_amp)

    ema_decay = semi_cfg.get("ema_decay", 0.999)
    unsup_weight = semi_cfg.get("unsup_weight", 1.0)

    # 复合评分权重
    sem_w = train_cfg.get("composite_sem_weight", 0.4)
    bnd_w = train_cfg.get("composite_boundary_weight", 0.6)
    logger.info(f"Best model 保存依据: composite_score = {sem_w:.1f}*mIoU + {bnd_w:.1f}*BndIoU")

    start_epoch = 0
    best_composite_score = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        student_model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        # 兼容旧 checkpoint 的 best_val_iou key
        best_composite_score = checkpoint.get(
            "best_composite_score", checkpoint.get("best_val_iou", 0.0)
        )
        logger.info(f"Resumed from epoch {start_epoch}, best Composite Score: {best_composite_score:.4f}")

    logger.info("=" * 60)
    logger.info("Stage-2 Semi-Supervised Fine-tuning (Mean Teacher)")
    logger.info(f"  Epochs: {total_epochs}")
    logger.info(f"  EMA decay: {ema_decay}")
    logger.info(f"  Unsupervised weight: {unsup_weight}")
    logger.info(f"  LR: {semi_cfg.get('learning_rate', train_cfg['learning_rate'])}")
    logger.info("=" * 60)

    unlabeled_samples_per_epoch = semi_cfg.get("unlabeled_samples_per_epoch", 0)
    bs_unlabeled = semi_cfg.get("batch_size_unlabeled", 4)

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{total_epochs}")

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

        train_metrics = train_one_epoch(
            student_model, teacher_model, labeled_loader, unlabeled_iter,
            num_steps, num_unlabeled_steps if unlabeled_loader is not None else 0,
            criterion, unsup_weight=unsup_weight, ema_decay=ema_decay,
            optimizer=optimizer, scaler=scaler, device=device,
            grad_clip=train_cfg.get("grad_clip", 1.0), use_amp=use_amp,
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

        if (epoch + 1) % 10 == 0 and semi_cfg.get("save_checkpoints", True):
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

    final_path = os.path.join(output_dir, "final_model_stage2.pth")
    torch.save({
        "epoch": total_epochs - 1,
        "decoder_state_dict": student_model.decoder.state_dict(),
        "best_composite_score": best_composite_score,
        "config": config,
    }, final_path)
    logger.info(f"Stage-2 training complete! Final model: {final_path}")
    logger.info(f"Best Composite Score: {best_composite_score:.4f}")


if __name__ == "__main__":
    main()