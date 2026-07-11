# -*- coding: utf-8 -*-
"""
微调训练主入口
==============
冻结 SAM 2 Image Encoder + 自制轻量 FPN 解码头 的训练流程。

技术路线：
1. SAM 2 Hiera trunk 冻结，仅作为特征提取器
2. FPN 解码头随机初始化，仅训练解码头参数
3. 在线 Letterbox 数据管道
4. 混合精度训练 + 梯度裁剪
5. 交叉熵 + Dice + 边界感知损失

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
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from data.dataset import MetallographicDataset, collate_fn
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.sam2_encoder import SAM2Encoder
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


class CombinedLoss(nn.Module):
    """组合损失：交叉熵 + Dice + 边界感知损失。"""

    def __init__(self, ce_weight=1.0, dice_weight=0.5, boundary_weight=0.3, num_classes=3):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.num_classes = num_classes
        class_weights = torch.tensor([1.0, 1.0, 3.0])
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def dice_loss(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return 1.0 - dice.mean()

    def boundary_loss(self, logits, target):
        pred = torch.argmax(logits, dim=1).float()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(logits.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(logits.device)
        pred_edges_x = F.conv2d(pred.unsqueeze(1), sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred.unsqueeze(1), sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-6)
        target_edges_x = F.conv2d(target.float().unsqueeze(1), sobel_x, padding=1)
        target_edges_y = F.conv2d(target.float().unsqueeze(1), sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-6)
        return F.mse_loss(pred_edges, target_edges)

    def forward(self, logits, target):
        ce = self.ce_loss(logits, target)
        dice = self.dice_loss(logits, target)
        boundary = self.boundary_loss(logits, target)
        total = self.ce_weight * ce + self.dice_weight * dice + self.boundary_weight * boundary
        return {"total": total, "ce": ce.item(), "dice": dice.item(), "boundary": boundary.item()}


def build_model(config, device):
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]

    ckpt_path = os.path.join(paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    if not os.path.exists(ckpt_path):
        logger.warning(f"SAM 2 权重文件不存在: {ckpt_path}\n请将 {paths_cfg['sam2_ckpt']} 下载到 weights/ 目录。")

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
    data_dir = os.path.join(paths_cfg["project_root"], paths_cfg["raw_data_dir"])

    train_dataset = MetallographicDataset(
        data_dir=data_dir,
        image_size=data_cfg["image_size"],
        erode_pixels=data_cfg["erode_pixels"],
        augment=True,
        augment_config=data_cfg.get("augmentation", {}),
        split="train",
        train_ratio=data_cfg["train_ratio"],
        seed=data_cfg["seed"],
    )

    val_dataset = MetallographicDataset(
        data_dir=data_dir,
        image_size=data_cfg["image_size"],
        erode_pixels=data_cfg["erode_pixels"],
        augment=False,
        split="val",
        train_ratio=data_cfg["train_ratio"],
        seed=data_cfg["seed"],
    )

    # 防御性检查：在构建 DataLoader 之前检查数据集长度
    if len(train_dataset) == 0:
        logger.error("训练数据集为空！请将原始图像与同名 .json 标注放入 data/raw/ 目录。")
        sys.exit(1)

    if len(val_dataset) == 0:
        logger.warning("验证数据集为空，将使用训练集进行验证。")
        val_dataset = train_dataset

    # 当 batch_size 大于数据集长度时，自适应调整 batch_size
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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=1.0, use_amp=True):
    model.train()
    model.encoder.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_boundary = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                logits = model(images, output_size=masks.shape[-2:])
                loss_dict = criterion(logits, masks)
            scaler.scale(loss_dict["total"]).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, output_size=masks.shape[-2:])
            loss_dict = criterion(logits, masks)
            loss_dict["total"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss_dict["total"].item()
        total_ce += loss_dict["ce"]
        total_dice += loss_dict["dice"]
        total_boundary += loss_dict["boundary"]
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(loader)}: loss={loss_dict['total'].item():.4f} ce={loss_dict['ce']:.4f} dice={loss_dict['dice']:.4f} boundary={loss_dict['boundary']:.4f}")

    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "dice": total_dice / n_batches,
        "boundary": total_boundary / n_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    metrics = SegMetrics(num_classes=3)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images, output_size=masks.shape[-2:])
        loss_dict = criterion(logits, masks)
        total_loss += loss_dict["total"].item()
        n_batches += 1
        pred = torch.argmax(logits, dim=1)
        metrics.update_tensor(pred, masks)

    val_metrics = metrics.get_metrics()
    val_metrics["loss"] = total_loss / n_batches
    return val_metrics


def main():
    parser = argparse.ArgumentParser(description="低碳钢金相分割训练")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["train"]
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA 不可用，切换到 CPU")
        device = "cpu"

    logger.info(f"使用设备: {device}")

    output_dir = os.path.join(paths_cfg["project_root"], paths_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    model = build_model(config, device)

    train_loader, val_loader = build_dataloaders(config)

    loss_weights = train_cfg["loss_weights"]
    criterion = CombinedLoss(
        ce_weight=loss_weights["ce"],
        dice_weight=loss_weights["dice"],
        boundary_weight=loss_weights["boundary"],
        num_classes=config["decoder"]["num_classes"],
    ).to(device)

    decoder_params = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(decoder_params, lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    scaler = GradScaler(enabled=train_cfg["amp"])

    start_epoch = 0
    best_val_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_iou = checkpoint.get("best_val_iou", 0.0)
        logger.info(f"从 epoch {start_epoch} 恢复训练，最佳 Val IoU: {best_val_iou:.4f}")

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"  Epochs: {train_cfg['epochs']}")
    logger.info(f"  Batch size: {config['data']['batch_size']}")
    logger.info(f"  Learning rate: {train_cfg['learning_rate']}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()
        logger.info(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_clip=train_cfg["grad_clip"], use_amp=train_cfg["amp"])
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch + 1} 完成 ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f} (ce={train_metrics['ce']:.4f}, dice={train_metrics['dice']:.4f}, boundary={train_metrics['boundary']:.4f})")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f} | Val mIoU: {val_metrics['mean_iou']:.4f} | Val mDice: {val_metrics['mean_dice']:.4f}")

        if val_metrics["mean_iou"] > best_val_iou:
            best_val_iou = val_metrics["mean_iou"]
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save({"epoch": epoch, "decoder_state_dict": model.decoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_val_iou": best_val_iou, "config": config}, best_path)
            logger.info(f"  新最佳模型已保存: {best_path} (mIoU={best_val_iou:.4f})")

        if (epoch + 1) % 10 == 0 and train_cfg["save_checkpoints"]:
            ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save({"epoch": epoch, "decoder_state_dict": model.decoder.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_val_iou": best_val_iou, "config": config}, ckpt_path)
            logger.info(f"  Checkpoint 已保存: {ckpt_path}")

    final_path = os.path.join(output_dir, "final_model.pth")
    torch.save({"epoch": train_cfg["epochs"] - 1, "decoder_state_dict": model.decoder.state_dict(), "best_val_iou": best_val_iou, "config": config}, final_path)
    logger.info(f"训练完成！最终模型: {final_path}")
    logger.info(f"最佳 Val mIoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()