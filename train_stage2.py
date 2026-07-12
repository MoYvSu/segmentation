# -*- coding: utf-8 -*-
"""
第二阶段半监督微调主入口
========================
冻结 SAM 2 Encoder + 冻结 FPN Decoder 主体，仅微调 cls_branch + reg_branch。
使用有标签数据（Focal+MSE）与无标签数据（双路一致性）联合训练。

技术路线：
1. 加载第一阶段最优权重
2. 硬冻结：encoder + decoder 主体 requires_grad=False
3. 仅解冻：decoder.cls_branch + decoder.reg_branch
4. 双流混合 Batch：itertools.cycle(labeled_loader) + unlabeled_loader
5. 有标签流：FocalDistanceFieldLoss（与第一阶段一致）
6. 无标签流：compute_stage2_unsupervised_loss（分类一致性 + 回归几何一致性）

使用方法：
    conda activate sam2_env
    python train_stage2.py --config config/stage2_config.yaml

约束：
- 非侵入式：不修改第一阶段 train.py
- 仅有 cls_branch + reg_branch 参数产生梯度
- 保持 encoder.eval() 和 decoder 非头模块 eval()
"""

import argparse
import itertools
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from data.dataset import MetallographicDataset, collate_fn
from data.dataset_semi import (
    LabeledDataset,
    UnlabeledDataset,
    labeled_collate_fn,
    unlabeled_collate_fn,
)
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.sam2_encoder import SAM2Encoder
from utils.loss import FocalDistanceFieldLoss
from utils.loss_semi import compute_stage2_unsupervised_loss
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
    """构建模型（与第一阶段一致）。"""
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

    param_info = model.total_param_count()
    logger.info("=" * 60)
    logger.info("Model parameter count:")
    logger.info(f"  Encoder:  {param_info['encoder'] / 1e6:.2f}M")
    logger.info(f"  Decoder:  {param_info['decoder'] / 1e6:.2f}M")
    logger.info(f"  Total:    {param_info['total_M']:.2f}M")
    logger.info("=" * 60)

    return model


def load_stage1_checkpoint(model, checkpoint_path, device):
    """加载第一阶段最优权重到 decoder。"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    logger.info(f"Stage-1 checkpoint loaded: {checkpoint_path}")
    logger.info(
        f"  Epoch: {checkpoint.get('epoch', '?')}, "
        f"Best Val IoU: {checkpoint.get('best_val_iou', '?')}"
    )


def freeze_model_for_stage2(model):
    """
    硬冻结：encoder + decoder 主体 requires_grad=False
    仅解冻：decoder.cls_branch + decoder.reg_branch
    """
    # Step 1: 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: 仅解冻 cls_branch + reg_branch
    trainable_count = 0
    for name, param in model.decoder.cls_branch.named_parameters():
        param.requires_grad = True
        trainable_count += param.numel()
    for name, param in model.decoder.reg_branch.named_parameters():
        param.requires_grad = True
        trainable_count += param.numel()

    logger.info("=" * 60)
    logger.info("Stage-2 Freeze Strategy:")
    logger.info(f"  Frozen: encoder + decoder (except cls_branch + reg_branch)")
    logger.info(f"  Trainable: decoder.cls_branch + decoder.reg_branch")
    logger.info(f"  Trainable params: {trainable_count} ({trainable_count / 1e6:.4f}M)")
    logger.info("=" * 60)

    # 打印详细参数列表（验证基准 3）
    logger.info("Parameter requires_grad status:")
    for name, param in model.named_parameters():
        logger.info(
            f"  {'[TRAIN]' if param.requires_grad else '[FROZEN]'} {name} "
            f"({param.numel()})"
        )

    return trainable_count


def get_trainable_params(model):
    """获取可训练参数（仅 cls_branch + reg_branch）。"""
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(param)
    return params


def build_dataloaders(config):
    """构建有标签和无标签 DataLoader。"""
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    stage2_cfg = config["stage2"]

    project_root = paths_cfg["project_root"]
    labeled_dir = os.path.join(project_root, stage2_cfg["LABELED_DATA_DIR"])
    unlabeled_dir = os.path.join(project_root, stage2_cfg["UNLABELED_DATA_DIR"])

    image_size = data_cfg["image_size"]
    dist_scale_factor = data_cfg.get("dist_scale_factor", 10.0)
    augment_config = data_cfg.get("augmentation", {})
    num_workers = data_cfg.get("num_workers", 4)

    # 有标签数据集（带增强）
    labeled_dataset = LabeledDataset(
        data_dir=labeled_dir,
        image_size=image_size,
        augment=True,
        augment_config=augment_config,
        dist_scale_factor=dist_scale_factor,
    )

    # 验证集（有标签，不增强）—— 复用 MetallographicDataset 的 split 逻辑
    val_dataset = MetallographicDataset(
        data_dir=labeled_dir,
        image_size=image_size,
        augment=False,
        split="val",
        train_ratio=data_cfg.get("train_ratio", 0.8),
        seed=data_cfg.get("seed", 42),
        dist_scale_factor=dist_scale_factor,
    )

    if len(labeled_dataset) == 0:
        logger.error(f"Labeled dataset is empty: {labeled_dir}")
        sys.exit(1)

    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty, using labeled dataset for validation.")
        val_dataset = labeled_dataset

    # 无标签数据集
    if not os.path.exists(unlabeled_dir) or len(os.listdir(unlabeled_dir)) == 0:
        logger.warning(f"Unlabeled dataset is empty: {unlabeled_dir}")
        logger.warning("Training will proceed with supervised loss only (no unsupervised loss).")
        unlabeled_dataset = None
    else:
        unlabeled_dataset = UnlabeledDataset(
            data_dir=unlabeled_dir,
            image_size=image_size,
        )

    bs_labeled = stage2_cfg["BATCH_SIZE_LABELED"]
    bs_unlabeled = stage2_cfg["BATCH_SIZE_UNLABELED"]

    # 自适应：batch_size 不超过数据集长度
    bs_labeled = min(bs_labeled, len(labeled_dataset))
    if unlabeled_dataset is not None:
        bs_unlabeled = min(bs_unlabeled, len(unlabeled_dataset))

    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=bs_labeled,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=labeled_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=bs_labeled,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    unlabeled_loader = None
    if unlabeled_dataset is not None:
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=bs_unlabeled,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=unlabeled_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    logger.info(f"Labeled batch size: {bs_labeled}")
    logger.info(f"Unlabeled batch size: {bs_unlabeled if unlabeled_loader else 'N/A'}")

    return labeled_loader, unlabeled_loader, val_loader


def train_one_epoch(
    model,
    labeled_loader,
    unlabeled_iter,
    num_steps,
    num_unlabeled_steps,
    criterion,
    unsup_weight,
    confidence_threshold,
    dist_weight,
    optimizer,
    scaler,
    device,
    grad_clip=1.0,
    use_amp=False,
):
    """训练一个 epoch（双流混合 Batch）。

    Args:
        num_unlabeled_steps: 无标签数据迭代步数限制（方案 D）。
            超过此步数后仅进行有标签训练，不再抽取无标签数据。
    """
    model.train()
    # 保持 encoder 和 decoder 非头模块在 eval 模式
    model.encoder.eval()
    # decoder 的 BN/Dropout 在非头模块也应保持 eval
    model.decoder.lateral_convs.eval()
    model.decoder.residual_blocks.eval()
    model.decoder.semantic_head.eval()
    # 但 cls_branch 和 reg_branch 保持 train 模式
    model.decoder.cls_branch.train()
    model.decoder.reg_branch.train()

    total_loss_sum = 0.0
    total_sup_loss = 0.0
    total_unsup_loss = 0.0
    total_seg = 0.0
    total_dist = 0.0
    total_cls_consist = 0.0
    total_reg_consist = 0.0
    n_steps = 0

    # 梯度裁剪参数：仅可训练参数
    clip_params = get_trainable_params(model)

    # 使用 itertools.cycle 包裹 labeled_loader
    labeled_iter_cycle = itertools.cycle(labeled_loader)

    for step_idx in range(num_steps):
        # ---- 有标签流 ----
        labeled_batch = next(labeled_iter_cycle)
        images_labeled = labeled_batch["image"].to(device)
        targets_labeled = labeled_batch["target"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                out_labeled = model(
                    images_labeled, output_size=targets_labeled.shape[-2:]
                )
                sup_loss, seg_val, dist_val = criterion(out_labeled, targets_labeled)
        else:
            out_labeled = model(images_labeled, output_size=targets_labeled.shape[-2:])
            sup_loss, seg_val, dist_val = criterion(out_labeled, targets_labeled)

        # ---- 无标签流（方案 D：超过 num_unlabeled_steps 后不再抽取无标签数据）----
        unsup_loss = torch.tensor(0.0, device=device)
        cls_consist_val = 0.0
        reg_consist_val = 0.0

        if unlabeled_iter is not None and step_idx < num_unlabeled_steps:
            try:
                unlabeled_batch = next(unlabeled_iter)
                img_weak = unlabeled_batch["img_weak"]
                img_strong_app = unlabeled_batch["img_strong_appearance"]
                img_strong_geo = unlabeled_batch["img_strong_geometric"]
                T_list = unlabeled_batch["T_list"]

                if use_amp:
                    with autocast():
                        unsup_loss, cls_consist_val, reg_consist_val = (
                            compute_stage2_unsupervised_loss(
                                model,
                                img_weak,
                                img_strong_app,
                                img_strong_geo,
                                T_list,
                                confidence_threshold=confidence_threshold,
                                dist_weight=dist_weight,
                            )
                        )
                else:
                    unsup_loss, cls_consist_val, reg_consist_val = (
                        compute_stage2_unsupervised_loss(
                            model,
                            img_weak,
                            img_strong_app,
                            img_strong_geo,
                            T_list,
                            confidence_threshold=confidence_threshold,
                            dist_weight=dist_weight,
                        )
                    )
            except StopIteration:
                pass

        # ---- 总损失 ----
        total_loss = sup_loss + unsup_weight * unsup_loss

        # ---- 反向传播 ----
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

        total_loss_sum += total_loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        total_seg += seg_val
        total_dist += dist_val
        total_cls_consist += cls_consist_val
        total_reg_consist += reg_consist_val
        n_steps += 1

        if (step_idx + 1) % 5 == 0:
            logger.info(
                f"  Step {step_idx + 1}/{num_steps}: "
                f"total={total_loss.item():.4f} "
                f"sup={sup_loss.item():.4f} (seg={seg_val:.4f} dist={dist_val:.4f}) "
                f"unsup={unsup_loss.item():.4f} (cls_c={cls_consist_val:.4f} reg_c={reg_consist_val:.4f})"
            )

    n = max(n_steps, 1)
    return {
        "loss": total_loss_sum / n,
        "sup_loss": total_sup_loss / n,
        "unsup_loss": total_unsup_loss / n,
        "seg": total_seg / n,
        "dist": total_dist / n,
        "cls_consist": total_cls_consist / n,
        "reg_consist": total_reg_consist / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证（复用第一阶段逻辑）。"""
    model.eval()
    metrics = SegMetrics(num_classes=2)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        output = model(images, output_size=targets.shape[-2:])
        total_loss_t, _, _ = criterion(output, targets)
        total_loss += total_loss_t.item()
        n_batches += 1

        seg_logits = output[:, 0]
        pred = (torch.sigmoid(seg_logits) > 0.5).long()
        mask = targets[:, 0].long()
        metrics.update_tensor(pred, mask)

    val_metrics = metrics.get_metrics()
    val_metrics["loss"] = total_loss / max(n_batches, 1)
    return val_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Stage-2 Semi-Supervised Fine-tuning (Dual-Task Distance Field)"
    )
    parser.add_argument("--config", type=str, default="config/stage2_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]
    stage2_cfg = config["stage2"]

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    logger.info(f"Device: {device}")

    output_dir = os.path.join(paths_cfg["project_root"], paths_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # ---- 构建模型 ----
    model = build_model(config, device)

    # ---- 加载第一阶段权重 ----
    stage1_ckpt_path = os.path.join(
        paths_cfg["project_root"], stage2_cfg["STAGE1_CHECKPOINT"]
    )
    load_stage1_checkpoint(model, stage1_ckpt_path, device)

    # ---- 硬冻结 ----
    freeze_model_for_stage2(model)

    # ---- 数据加载 ----
    labeled_loader, unlabeled_loader, val_loader = build_dataloaders(config)

    # ---- 损失函数 ----
    criterion = FocalDistanceFieldLoss(gamma=2.0, alpha=0.95).to(device)
    logger.info("Supervised loss: FocalDistanceFieldLoss (加权Focal + 0.05*TV + 10*MSE, EDT空间权重)")
    logger.info(
        f"Unsupervised loss: consistency loss (weight={stage2_cfg['UNSUPERVISED_WEIGHT']})"
    )

    # ---- 优化器（仅 cls_branch + reg_branch） ----
    trainable_params = get_trainable_params(model)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=stage2_cfg["LEARNING_RATE"],
        weight_decay=stage2_cfg["WEIGHT_DECAY"],
        eps=1e-4,
    )

    # ---- 调度器 ----
    warmup_epochs = stage2_cfg.get("WARMUP_EPOCHS", 5)
    total_epochs = stage2_cfg["EPOCHS"]
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            cosine_scheduler,
        ],
        milestones=[warmup_epochs],
    )

    use_amp = stage2_cfg.get("AMP", False)
    scaler = GradScaler(enabled=use_amp)

    # ---- 恢复训练 ----
    start_epoch = 0
    best_val_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        # 仅加载可训练参数
        model.decoder.cls_branch.load_state_dict(
            checkpoint["cls_branch_state_dict"]
        )
        model.decoder.reg_branch.load_state_dict(
            checkpoint["reg_branch_state_dict"]
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_iou = checkpoint.get("best_val_iou", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best Val IoU: {best_val_iou:.4f}")

    # ---- 训练循环 ----
    logger.info("=" * 60)
    logger.info("Stage-2 Semi-Supervised Fine-tuning")
    logger.info(f"  Epochs: {total_epochs}")
    logger.info(f"  Batch labeled: {stage2_cfg['BATCH_SIZE_LABELED']}")
    logger.info(f"  Batch unlabeled: {stage2_cfg['BATCH_SIZE_UNLABELED']}")
    logger.info(f"  LR: {stage2_cfg['LEARNING_RATE']}")
    logger.info(f"  Unsupervised weight: {stage2_cfg['UNSUPERVISED_WEIGHT']}")
    logger.info(f"  Confidence threshold: {stage2_cfg['PSEUDO_LABEL_CONFIDENCE']}")
    logger.info("=" * 60)

    # 方案 D：限制每 epoch 无标签采样步数
    unlabeled_samples_per_epoch = stage2_cfg.get("UNLABELED_SAMPLES_PER_EPOCH", 0)
    bs_unlabeled = stage2_cfg["BATCH_SIZE_UNLABELED"]

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{total_epochs}")

        # 每个 epoch 开始前重置 unlabeled_loader 的迭代器
        if unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)
            # 方案 D：限制无标签步数
            full_unlabeled_steps = len(unlabeled_loader)
            if unlabeled_samples_per_epoch > 0:
                max_unlabeled_steps = max(1, unlabeled_samples_per_epoch // bs_unlabeled)
                num_unlabeled_steps = min(full_unlabeled_steps, max_unlabeled_steps)
            else:
                num_unlabeled_steps = full_unlabeled_steps
            # 总步数取 max(无标签步数, 有标签步数) 保证有标签训练充分
            num_steps = max(num_unlabeled_steps, len(labeled_loader))
            logger.info(
                f"  Unlabeled steps: {num_unlabeled_steps}/{full_unlabeled_steps} "
                f"(samples limit: {unlabeled_samples_per_epoch if unlabeled_samples_per_epoch > 0 else 'unlimited'})"
            )
        else:
            unlabeled_iter = None
            num_steps = len(labeled_loader)

        train_metrics = train_one_epoch(
            model,
            labeled_loader,
            unlabeled_iter,
            num_steps,
            num_unlabeled_steps if unlabeled_loader is not None else 0,
            criterion,
            unsup_weight=stage2_cfg["UNSUPERVISED_WEIGHT"],
            confidence_threshold=stage2_cfg["PSEUDO_LABEL_CONFIDENCE"],
            dist_weight=stage2_cfg["DIST_WEIGHT"],
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=stage2_cfg["GRAD_CLIP"],
            use_amp=use_amp,
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} done ({epoch_time:.1f}s)")
        logger.info(
            f"  Train: total={train_metrics['loss']:.4f} "
            f"sup={train_metrics['sup_loss']:.4f} "
            f"unsup={train_metrics['unsup_loss']:.4f}"
        )
        logger.info(
            f"    sup_detail: seg={train_metrics['seg']:.4f} dist={train_metrics['dist']:.4f}"
        )
        logger.info(
            f"    unsup_detail: cls_c={train_metrics['cls_consist']:.4f} "
            f"reg_c={train_metrics['reg_consist']:.4f}"
        )
        logger.info(
            f"  Val: loss={val_metrics['loss']:.4f} "
            f"mIoU={val_metrics['mean_iou']:.4f} "
            f"mDice={val_metrics['mean_dice']:.4f}"
        )
        logger.info(
            f"  pearlite_iou={val_metrics['pearlite_iou']:.4f} "
            f"ferrite_iou={val_metrics['ferrite_iou']:.4f}"
        )

        # 保存最优模型
        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            best_path = os.path.join(output_dir, "best_model_stage2.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "cls_branch_state_dict": model.decoder.cls_branch.state_dict(),
                    "reg_branch_state_dict": model.decoder.reg_branch.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_iou": best_val_iou,
                    "config": config,
                },
                best_path,
            )
            logger.info(f"  New best model saved: {best_path} (mIoU={best_val_iou:.4f})")

        # 定期 checkpoint
        if (epoch + 1) % 10 == 0 and stage2_cfg.get("SAVE_CHECKPOINTS", True):
            ckpt_path = os.path.join(output_dir, f"stage2_epoch{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "cls_branch_state_dict": model.decoder.cls_branch.state_dict(),
                    "reg_branch_state_dict": model.decoder.reg_branch.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_iou": best_val_iou,
                    "config": config,
                },
                ckpt_path,
            )
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(output_dir, "final_model_stage2.pth")
    torch.save(
        {
            "epoch": total_epochs - 1,
            "cls_branch_state_dict": model.decoder.cls_branch.state_dict(),
            "reg_branch_state_dict": model.decoder.reg_branch.state_dict(),
            "best_val_iou": best_val_iou,
            "config": config,
        },
        final_path,
    )
    logger.info(f"Stage-2 training complete! Final model: {final_path}")
    logger.info(f"Best Val mIoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()