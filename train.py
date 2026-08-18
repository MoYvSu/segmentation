# -*- coding: utf-8 -*-
"""
冻结 SAM 2 Image Encoder + 自制轻量 FPN 解码头的训练流程（边界预测版本）。

技术路线：
1. SAM 2 Hiera trunk 冻结，仅作为特征提取器
2. FPN 解码头随机初始化，仅训练解码头参数
3. 在线 Letterbox 数据管道 + 离线净化的边界 GT
4. 双任务输出：语义 BCE Loss + 边界 Focal Loss × EDT 权重

使用方法：
    conda activate sam2_env
    python train.py --config config/default_config.yaml

约束：
- 总参数量 < 500M
- 禁止加载 SAM 2 原生 Mask Decoder 权重
- 权重文件必须位于项目 weights/ 目录
"""

import argparse
import logging
import math
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

from data.dataset import BoundaryDataset, collate_fn
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.lora import (
    count_lora_params,
    extract_lora_state_dict,
    inject_trunk_lora,
    load_lora_state_dict,
)
from models.sam2_encoder import SAM2Encoder
from utils.loss import BoundaryLoss
from utils.metrics import SegMetrics
from utils.checkpoint import build_checkpoint, validate_checkpoint_architecture
from utils.config import load_config as load_yaml_config, project_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    return load_yaml_config(config_path)


def build_model(config, device):
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]

    ckpt_path = project_path(config, paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    if not os.path.exists(ckpt_path):
        logger.warning(f"SAM 2 权重文件不存在: {ckpt_path}")

    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
        device=device,
        freeze=sam2_cfg["freeze"],
        sam2_repo_path=os.path.join(paths_cfg["project_root"], sam2_cfg["sam2_repo_path"]),
    )

    # LoRA（协议 C：从 Stage-1 开始域适配；起点可用 pretrain_lora_ssl 预训练状态）
    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled", False):
        n_layers = inject_trunk_lora(
            encoder,
            rank=lora_cfg.get("rank", 16),
            alpha=lora_cfg.get("alpha", 32.0),
            target_layers=lora_cfg.get("target_layers"),
        )
        n_params = count_lora_params(encoder)
        logger.info(f"LoRA: ENABLED ({n_layers} 层, 可训练参数 {n_params / 1e6:.2f}M)")
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
    else:
        logger.info("LoRA: DISABLED (trunk 全冻结)")

    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=decoder_cfg["fpn_channels"],
        num_classes=decoder_cfg["num_classes"],
        dropout=decoder_cfg["dropout"],
        use_bn=decoder_cfg["use_bn"],
        boundary_refine=decoder_cfg.get("boundary_refine", False),
        center_head=decoder_cfg.get("center_head", False),
    )

    model = SegmentationModel(encoder, decoder)
    model = model.to(device)

    param_info = model.total_param_count()
    logger.info("=" * 60)
    logger.info("模型参数量统计:")
    logger.info(f"  Encoder:  {param_info['encoder'] / 1e6:.2f}M")
    logger.info(f"  Decoder:  {param_info['decoder'] / 1e6:.2f}M")
    logger.info(f"  总计:     {param_info['total_M']:.2f}M")
    logger.info(f"  约束 <500M: {'通过' if param_info['constraint_passed'] else '未通过'}")
    logger.info("=" * 60)

    if not param_info["constraint_passed"]:
        raise RuntimeError(f"总参数量 {param_info['total_M']:.2f}M 超过 500M 限制！")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M (仅 Decoder)")

    return model


def build_dataloaders(config):
    data_cfg = config["data"]
    paths_cfg = config["paths"]
    boundary_cfg = config.get("boundary", {})

    data_dir = os.path.join(paths_cfg["project_root"], paths_cfg["raw_data_dir"])
    gt_dir = os.path.join(paths_cfg["project_root"], boundary_cfg.get("gt_dir", "data/purified_gt"))

    boundary_scale_factor = boundary_cfg.get("edt_scale_factor", 10.0)
    boundary_weight_floor = boundary_cfg.get("edt_weight_floor", 1.0)
    boundary_weight_ceil = boundary_cfg.get("edt_weight_ceil", 4.0)
    crop_size = data_cfg.get("crop_size", 0)

    train_dataset = BoundaryDataset(
        data_dir=data_dir,
        gt_dir=gt_dir,
        image_size=data_cfg["image_size"],
        crop_size=crop_size,
        augment=True,
        augment_config=data_cfg.get("augmentation", {}),
        split="train",
        train_ratio=data_cfg["train_ratio"],
        seed=data_cfg["seed"],
        boundary_scale_factor=boundary_scale_factor,
        boundary_weight_floor=boundary_weight_floor,
        boundary_weight_ceil=boundary_weight_ceil,
        center_sigma=data_cfg.get("center_sigma", 4.0),
    )

    val_dataset = BoundaryDataset(
        data_dir=data_dir,
        gt_dir=gt_dir,
        image_size=data_cfg["image_size"],
        crop_size=0,
        augment=False,
        split="val",
        train_ratio=data_cfg["train_ratio"],
        seed=data_cfg["seed"],
        boundary_scale_factor=boundary_scale_factor,
        boundary_weight_floor=boundary_weight_floor,
        boundary_weight_ceil=boundary_weight_ceil,
        center_sigma=data_cfg.get("center_sigma", 4.0),
    )

    if len(train_dataset) == 0:
        logger.error("训练数据集为空！请先运行 tools/preprocess_labels.py 生成净化 GT。")
        sys.exit(1)

    if len(val_dataset) == 0:
        logger.warning("验证数据集为空，将使用训练集进行验证。")
        val_dataset = train_dataset

    train_bs = min(data_cfg["batch_size"], len(train_dataset))
    if train_bs < data_cfg["batch_size"]:
        logger.warning(f"训练 batch_size ({data_cfg['batch_size']}) > 训练集长度 ({len(train_dataset)})，已调整为 {train_bs}")
    val_bs = min(data_cfg["batch_size"], len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    grad_clip=1.0, use_amp=True):
    model.train()
    model.encoder.eval()

    total_loss = 0.0
    total_seg = 0.0
    total_boundary = 0.0
    n_batches = 0

    clip_params = list(model.decoder.parameters())

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        weights = batch["weight"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                output = model(images, output_size=targets.shape[-2:])
                total_loss_t, seg_val, boundary_val = criterion(output, targets, weights)
            scaler.scale(total_loss_t).backward()

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(images, output_size=targets.shape[-2:])
            total_loss_t, seg_val, boundary_val = criterion(output, targets, weights)
            total_loss_t.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
            optimizer.step()

        total_loss += total_loss_t.item()
        total_seg += seg_val
        total_boundary += boundary_val
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  Batch {batch_idx + 1}/{len(loader)}: "
                f"loss={total_loss_t.item():.4f} seg={seg_val:.4f} boundary={boundary_val:.4f}"
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "seg": total_seg / max(n_batches, 1),
        "boundary": total_boundary / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
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

        # 语义通道 IoU
        seg_logits = output[:, 0]
        pred = (torch.sigmoid(seg_logits) > 0.5).long()
        mask = targets[:, 0].long()
        metrics.update_tensor(pred, mask)

        # 边界通道 IoU
        bnd_logits = output[:, 1]
        bnd_pred = (torch.sigmoid(bnd_logits) > 0.5).long()
        # boundary_soft 是连续目标，验证二值 IoU 时必须显式阈值，不能直接 long 截断。
        bnd_gt = (targets[:, 1] > 0.5).long()
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
    parser = argparse.ArgumentParser(description="低碳钢金相分割训练 (边界预测版本)")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["train"]
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]
    lora_cfg = config.get("lora", {})

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA 不可用，切换到 CPU")
        device = "cpu"

    logger.info(f"使用设备: {device}")

    output_dir = os.path.join(paths_cfg["project_root"], paths_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    model = build_model(config, device)

    train_loader, val_loader = build_dataloaders(config)

    criterion = BoundaryLoss(
        gamma=train_cfg.get("focal_gamma", 2.0),
        alpha_boundary=train_cfg.get("boundary_alpha", 1.0),
        alpha_focal=train_cfg.get("focal_alpha", 0.75),
        center_weight=train_cfg.get("center_weight", 0.0),
        center_gamma=train_cfg.get("center_gamma", 2.0),
    ).to(device)
    logger.info("损失函数: BoundaryLoss (语义 BCE + 边界 Focal × EDT 权重)")

    # 优化器：decoder 组 + 可选 LoRA 组（lr = base × lora.lr_ratio）
    lora_params = []
    if lora_cfg.get("enabled", False):
        lora_params = [
            p for n, p in model.encoder.trunk.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad
        ]
    param_groups = [{"params": model.decoder.parameters(),
                     "lr": train_cfg["learning_rate"]}]
    if lora_params:
        lora_lr = train_cfg["learning_rate"] * lora_cfg.get("lr_ratio", 1.0)
        param_groups.append({"params": lora_params, "lr": lora_lr})
        logger.info(f"LoRA 优化器组: lr={lora_lr:.2e} ({len(lora_params)} params)")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        eps=1e-4,
    )

    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    total_epochs = train_cfg["epochs"]

    def warmup_cosine_lambda(epoch):
        """Warmup (linear 0.01→1.0) + Cosine decay (1.0→0.0)。"""
        start_factor = 0.01
        if epoch < warmup_epochs:
            return start_factor + (1.0 - start_factor) * epoch / max(1, warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=warmup_cosine_lambda,
    )
    scaler = GradScaler('cuda', enabled=train_cfg["amp"])

    # 复合评分权重
    sem_w = train_cfg.get("composite_sem_weight", 0.4)
    bnd_w = train_cfg.get("composite_boundary_weight", 0.6)
    logger.info(f"Best model 保存依据: composite_score = {sem_w:.1f}*mIoU + {bnd_w:.1f}*BndIoU")

    start_epoch = 0
    best_composite_score = 0.0
    if args.resume:
        resume_path = project_path(config, args.resume)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        validate_checkpoint_architecture(checkpoint, config)
        from models.fpn_decoder import load_decoder_state
        load_decoder_state(model.decoder, checkpoint["decoder_state_dict"])
        if "lora_state_dict" in checkpoint and checkpoint["lora_state_dict"]:
            n_lora = load_lora_state_dict(model, checkpoint["lora_state_dict"])
            logger.info(f"  LoRA 状态已加载: {n_lora} 个参数张量")
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
            f"从 {resume_path} 的 epoch {start_epoch} 恢复训练，"
            f"最佳 Composite Score: {best_composite_score:.4f}"
        )

    logger.info("=" * 60)
    logger.info("开始训练 (边界预测版本)")
    logger.info(f"  Epochs: {train_cfg['epochs']}")
    logger.info(f"  Batch size: {config['data']['batch_size']}")
    logger.info(f"  Learning rate: {train_cfg['learning_rate']}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_clip=train_cfg["grad_clip"], use_amp=train_cfg["amp"]
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} 完成 ({epoch_time:.1f}s)")
        logger.info(
            f"  Train Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg']:.4f}, boundary={train_metrics['boundary']:.4f})"
        )
        # 复合评分
        composite_score = (
            sem_w * val_metrics["mean_iou"] + bnd_w * val_metrics["boundary_iou"]
        )

        logger.info(
            f"  Val Loss: {val_metrics['loss']:.4f} | "
            f"Val mIoU: {val_metrics['mean_iou']:.4f} | "
            f"Bnd IoU: {val_metrics['boundary_iou']:.4f} | "
            f"Composite: {composite_score:.4f} | "
            f"Val mDice: {val_metrics['mean_dice']:.4f}"
        )
        logger.info(
            f"  pearlite_iou={val_metrics['pearlite_iou']:.4f} "
            f"ferrite_iou={val_metrics['ferrite_iou']:.4f}"
        )

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save(build_checkpoint(
                model=model, config=config, epoch=epoch,
                lora_state_dict=extract_lora_state_dict(model),
                best_composite_score=best_composite_score,
                optimizer=optimizer, scheduler=scheduler,
            ), best_path)
            logger.info(
                f"  新最佳模型已保存: {best_path} "
                f"(composite={best_composite_score:.4f}, "
                f"mIoU={val_metrics['mean_iou']:.4f}, "
                f"bndIoU={val_metrics['boundary_iou']:.4f})"
            )

        if (epoch + 1) % 10 == 0 and train_cfg.get("save_checkpoints", True):
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save(build_checkpoint(
                model=model, config=config, epoch=epoch,
                lora_state_dict=extract_lora_state_dict(model),
                best_composite_score=best_composite_score,
                optimizer=optimizer, scheduler=scheduler,
            ), ckpt_path)
            logger.info(f"  Checkpoint 已保存: {ckpt_path}")

    final_path = os.path.join(output_dir, "final_model.pth")
    torch.save(build_checkpoint(
        model=model, config=config, epoch=train_cfg["epochs"] - 1,
        lora_state_dict=extract_lora_state_dict(model),
        best_composite_score=best_composite_score,
    ), final_path)
    logger.info(f"训练完成！最终模型: {final_path}")
    logger.info(f"最佳 Composite Score: {best_composite_score:.4f}")


if __name__ == "__main__":
    main()
