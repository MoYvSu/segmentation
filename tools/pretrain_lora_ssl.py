# -*- coding: utf-8 -*-
"""
自监督 LoRA 预训练（MAE 风格，协议 C 第一步）
=============================================
用 1000 张无标签金相图对冻结 SAM2 trunk 做掩码重建，只训练 LoRA + 轻量重建头：

  1. letterbox 1024 -> 随机块掩码（覆盖 mask_ratio 面积，块 32~64px）
  2. 带掩码图 -> SAM2 trunk（仅 LoRA 可训练）-> stage-1 特征 [B,112,256,256]
  3. 轻量重建头（3 层卷积）-> 重建原图 RGB；L1 loss 仅统计掩码区域
  4. 训练结束丢弃重建头，保存 trunk LoRA 状态
     outputs/lora_pretrain/lora_state_dict.pth 作为 Stage-1 域适配起点

动机：26 张有标签图不足以直接训 trunk LoRA（过拟合风险），先在 1000 张无标签上
让 LoRA 学会金相纹理/结构表征，再进 Stage-1 监督（train.py 已集成 LoRA）。

用法：
    python tools/pretrain_lora_ssl.py --config config/default_config.yaml
    python tools/pretrain_lora_ssl.py --epochs 40 --batch_size 2 --lr 1e-4
"""

import argparse
import glob
import logging
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from data.dataset import letterbox
from models.lora import (
    count_lora_params,
    extract_lora_state_dict,
    inject_trunk_lora,
    load_lora_state_dict,
)
from models.sam2_encoder import SAM2Encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class MaskedUnlabeledDataset(Dataset):
    """无标签金相图 + 随机块掩码（自监督重建用）。"""

    def __init__(self, data_dir, image_size=1024, mask_ratio=0.4,
                 patch_min=32, patch_max=64, max_patches=32):
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.samples = []
        for ext in valid_exts:
            self.samples.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        self.samples.sort()
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.patch_min = patch_min
        self.patch_max = patch_max
        self.max_patches = max_patches
        print(f"[MaskedUnlabeledDataset] {len(self.samples)} images from {data_dir}")

    def _gen_mask(self):
        h = w = self.image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        target = int(h * w * self.mask_ratio)
        covered = 0
        attempts = 0
        while covered < target and attempts < self.max_patches * 4:
            cy = np.random.randint(0, h)
            cx = np.random.randint(0, w)
            ps = np.random.randint(self.patch_min, self.patch_max + 1)
            y1, y2 = max(0, cy - ps // 2), min(h, cy + ps // 2)
            x1, x2 = max(0, cx - ps // 2), min(w, cx + ps // 2)
            new = int((mask[y1:y2, x1:x2] == 0).sum())
            mask[y1:y2, x1:x2] = 1
            covered += new
            attempts += 1
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.samples[idx]), cv2.COLOR_BGR2RGB)
        img_lb, _, _, _ = letterbox(img, self.image_size)
        t = torch.from_numpy(img_lb).float().permute(2, 0, 1) / 255.0      # [3,H,W]
        mask = self._gen_mask()
        m = torch.from_numpy(mask).float().unsqueeze(0)                     # [1,H,W]
        return t * (1.0 - m), t, m


class ReconstructionHead(nn.Module):
    """stage-1 特征 -> 原图 RGB 的轻量重建头（训练后丢弃）。"""

    def __init__(self, in_ch=112, hidden=128, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(16, hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)


def masked_l1_loss(pred, target, mask):
    """掩码区域 L1 损失（按掩码像素数归一化）。

    重建头输出为 stage-1 特征分辨率（如 256），先上采样回目标分辨率再算。
    """
    pred = F.interpolate(pred, size=target.shape[-2:],
                         mode="bilinear", align_corners=True)
    err = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return err.sum() / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--outdir", default="outputs/lora_pretrain")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--checkpoint_interval", type=int, default=5)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = config["paths"]["project_root"]
    os.makedirs(args.outdir, exist_ok=True)

    encoder = SAM2Encoder(
        config_file=config["sam2"]["config_file"],
        ckpt_path=os.path.join(root, "weights", config["paths"]["sam2_ckpt"]),
        device=device, freeze=True,
        sam2_repo_path=os.path.join(root, config["sam2"]["sam2_repo_path"]),
    )
    n_layer = inject_trunk_lora(encoder, rank=args.rank, alpha=args.alpha,
                                target_layers=["attn.qkv", "attn.proj"])
    encoder = encoder.to(device)
    head = ReconstructionHead().to(device)
    n_lora = count_lora_params(encoder)
    logger.info(f"LoRA: {n_layer} 层, {n_lora / 1e6:.2f}M 参数; 重建头: "
                f"{sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")

    # 优化器：LoRA 与重建头两组（同 lr，便于分别控制）
    params = [{"params": [p for n, p in encoder.trunk.named_parameters()
                          if ("lora_A" in n or "lora_B" in n) and p.requires_grad],
               "lr": args.lr}]
    params.append({"params": head.parameters(), "lr": args.lr})
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4, eps=1e-4)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        # encoder 本身即 trunk 容器，直接按 trunk 状态键加载 LoRA
        cur = encoder.trunk.state_dict()
        for k, v in ck.get("lora_state_dict", {}).items():
            if k in cur:
                cur[k].copy_(v)
        head.load_state_dict(ck["head_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    dataset = MaskedUnlabeledDataset(
        os.path.join(root, config["semi_supervised"]["unlabeled_dir"]),
        image_size=config["data"]["image_size"],
        mask_ratio=args.mask_ratio,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)
    logger.info(f"dataset: {len(dataset)} images, {len(loader)} steps/epoch, "
                f"batch={args.batch_size}, lr={args.lr}")

    total_epochs = args.epochs
    best_loss = float("inf")
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        encoder.trainable_lora = True
        encoder.trunk.eval()          # 保持 trunk 推理统计（LayerNorm 等）
        head.train()
        epoch_loss = 0.0
        n_steps = 0
        for i, (masked, target, mask) in enumerate(loader):
            masked, target, mask = masked.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad()
            with torch.autocast("cuda", enabled=False):
                feats = encoder(masked)
                pred = head(feats[0])
                loss = masked_l1_loss(pred, target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1
            if (i + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch + 1} Step {i + 1}/{len(loader)}: "
                            f"loss={loss.item():.4f}")
        avg = epoch_loss / max(n_steps, 1)
        logger.info(f"Epoch {epoch + 1}/{total_epochs} done ({time.time() - t0:.0f}s): "
                    f"loss={avg:.4f}")

        # 保存
        save_lora = {
            "epoch": epoch,
            "lora_state_dict": extract_lora_state_dict(encoder),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        if avg < best_loss:
            best_loss = avg
            torch.save(save_lora, os.path.join(args.outdir, "best_lora.pth"))
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == total_epochs - 1:
            torch.save(save_lora, os.path.join(args.outdir, f"lora_epoch{epoch + 1}.pth"))
            logger.info(f"  Checkpoint saved: {os.path.join(args.outdir, f'lora_epoch{epoch + 1}.pth')}")

    # 最终产物：仅 LoRA 状态（协议 C 的 Stage-1 起点）
    final = extract_lora_state_dict(encoder)
    torch.save(final, os.path.join(args.outdir, "lora_state_dict.pth"))
    meta = {
        "rank": args.rank, "alpha": args.alpha,
        "epochs": total_epochs, "best_loss": best_loss,
        "injected_layers": n_layer, "lora_params": n_lora,
    }
    import json
    json.dump(meta, open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    logger.info(f"Done. LoRA state saved: {os.path.join(args.outdir, 'lora_state_dict.pth')}")


if __name__ == "__main__":
    main()
