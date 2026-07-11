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
import itertools
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


class DynamicCombinedLoss(nn.Module):
    """动态组合损失：基于 Uncertainty Weights 的交叉熵 + Dice + 边界感知损失。

    引入三个可学习的对数方差参数 (log σ²)，通过 Uncertainty Weights 公式
    自动平衡多任务损失权重，打破静态权重的局限。

    公式: Loss = Σ [exp(-log_var_i) * raw_loss_i + 0.5 * log_var_i]

    当某项损失完全不下降时，网络会自动调整 log_var 以改变梯度权重，
    从而放大未收敛通道的梯度惩罚，将模型拉出死锁区。
    """

    def __init__(self, num_classes=3, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

        # 三个可学习的对数方差参数，初始值 0.0（对应权重 exp(0)=1.0）
        self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def dice_loss(self, logits, target):
        """
        通道级非对称 Dice 损失：仅计算类别 1（铁素体核）和类别 2（晶界）。

        显式剔除类别 0（珠光体/背景）对 Dice 损失的贡献，
        防止全盲预测时背景类仍能获得分值奖励。
        """
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        eps = 1e-7

        # 仅计算类别 1 和类别 2 的 Dice 损失
        total_loss = torch.tensor(0.0, device=logits.device)
        for cls in [1, 2]:
            intersection = (probs[:, cls] * target_onehot[:, cls]).sum()
            denom = probs[:, cls].sum() + target_onehot[:, cls].sum()
            dice = (2.0 * intersection + eps) / (denom + eps)
            total_loss = total_loss + (1.0 - dice)

        return total_loss / 2.0

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

        # Uncertainty Weights 动态加权
        # 当某项损失不下降时，log_var 增大 → exp(-log_var) 减小 → 该任务梯度被压制
        # 同时 0.5*log_var 作为正则项，防止 log_var 无限增大
        w_ce = torch.exp(-self.log_vars[0])
        w_dice = torch.exp(-self.log_vars[1])
        w_boundary = torch.exp(-self.log_vars[2])

        total = (w_ce * ce + 0.5 * self.log_vars[0]) + \
                (w_dice * dice + 0.5 * self.log_vars[1]) + \
                (w_boundary * boundary + 0.5 * self.log_vars[2])

        return {
            "total": total,
            "ce": ce.item(),
            "dice": dice.item(),
            "boundary": boundary.item(),
            "w_ce": w_ce.item(),
            "w_dice": w_dice.item(),
            "w_boundary": w_boundary.item(),
        }


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
    total_w_ce = 0.0
    total_w_dice = 0.0
    total_w_boundary = 0.0
    n_batches = 0

    # 动态梯度裁剪：追踪历史 loss，检测异常突变
    loss_history = []

    # 梯度裁剪参数：解码器 + 损失函数可学习参数
    clip_params = list(itertools.chain(model.decoder.parameters(), criterion.parameters()))

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                logits = model(images, output_size=masks.shape[-2:])
                loss_dict = criterion(logits, masks)
            scaler.scale(loss_dict["total"]).backward()

            # 动态梯度裁剪：检测 loss 异常
            current_loss = loss_dict["total"].item()
            effective_clip = grad_clip
            if len(loss_history) > 0:
                avg_loss = sum(loss_history) / len(loss_history)
                if current_loss > 5.0 or current_loss > 3.0 * avg_loss:
                    effective_clip = 0.1  # 强力压低梯度
                    logger.warning(f" 梯度突变检测: loss={current_loss:.4f} avg={avg_loss:.4f} → grad_clip {grad_clip}→{effective_clip}")

            if effective_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, effective_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, output_size=masks.shape[-2:])
            loss_dict = criterion(logits, masks)
            loss_dict["total"].backward()

            # 动态梯度裁剪
            current_loss = loss_dict["total"].item()
            effective_clip = grad_clip
            if len(loss_history) > 0:
                avg_loss = sum(loss_history) / len(loss_history)
                if current_loss > 5.0 or current_loss > 3.0 * avg_loss:
                    effective_clip = 0.1
                    logger.warning(f"  ⚠ 梯度突变检测: loss={current_loss:.4f} avg={avg_loss:.4f} → grad_clip {grad_clip}→{effective_clip}")

            if effective_clip > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, effective_clip)
            optimizer.step()

        loss_history.append(loss_dict["total"].item())
        total_loss += loss_dict["total"].item()
        total_ce += loss_dict["ce"]
        total_dice += loss_dict["dice"]
        total_boundary += loss_dict["boundary"]
        total_w_ce += loss_dict["w_ce"]
        total_w_dice += loss_dict["w_dice"]
        total_w_boundary += loss_dict["w_boundary"]
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(loader)}: loss={loss_dict['total'].item():.4f} ce={loss_dict['ce']:.4f} dice={loss_dict['dice']:.4f} boundary={loss_dict['boundary']:.4f} | w_ce={loss_dict['w_ce']:.4f} w_dice={loss_dict['w_dice']:.4f} w_boundary={loss_dict['w_boundary']:.4f}")

    return {
        "loss": total_loss / n_batches,
        "ce": total_ce / n_batches,
        "dice": total_dice / n_batches,
        "boundary": total_boundary / n_batches,
        "w_ce": total_w_ce / n_batches,
        "w_dice": total_w_dice / n_batches,
        "w_boundary": total_w_boundary / n_batches,
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

    # 显式提取配置中的类别平衡权重，若配置中不存在，默认采用 [1.0, 1.0, 10.0]
    raw_class_weights = train_cfg.get("class_weights", [1.0, 1.0, 10.0])
    class_weights_tensor = torch.tensor(raw_class_weights, dtype=torch.float32).to(device)

    # 使用 Uncertainty Weights 动态多任务损失
    criterion = DynamicCombinedLoss(
        num_classes=config["decoder"]["num_classes"],
        class_weights=class_weights_tensor,  # CE 内部仍使用类别权重
    ).to(device)

    # 将解码器参数和损失函数的可学习参数联合送入优化器
    # 动态权重使用稍大的学习率（1e-3）以快速纠偏
    trainable_params = [
        {"params": model.decoder.parameters()},
        {"params": criterion.parameters(), "lr": 1.0e-3},
    ]
    # AdamW eps 放大至 1e-4，防止混合精度下分母过小导致数值爆炸
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        eps=1e-4,
    )

    # Warmup + CosineAnnealing 调度器
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - warmup_epochs
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
        logger.info(f"  动态权重: w_ce={train_metrics['w_ce']:.4f} w_dice={train_metrics['w_dice']:.4f} w_boundary={train_metrics['w_boundary']:.4f}")
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