# -*- coding: utf-8 -*-
"""G0 generative domain-adapter pretraining on the allowed unlabeled images."""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from data.mim_dataset import MaskedMetallographyDataset
from models.gda_mim import GDAMaskedAutoencoder, gda_reconstruction_loss
from train_stage2 import build_model, load_base_checkpoint, seed_dataloader_worker
from utils.config import load_config, project_path


logger = logging.getLogger("train_gda_mim")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(config):
    cfg = config["gda_mim"]
    data_dir = project_path(config, cfg["unlabeled_dir"])
    manifest = project_path(config, cfg["holdout_manifest"])
    common = dict(
        data_dir=data_dir,
        holdout_manifest=manifest,
        crop_size=cfg.get("crop_size", 512),
        mask_ratio_range=cfg.get("mask_ratio_range", [0.50, 0.65]),
        mask_patch_range=cfg.get("mask_patch_range", [16, 64]),
        physical_aug_probability=cfg.get("physical_aug_probability", 0.80),
    )
    return (
        MaskedMetallographyDataset(split="train", **common),
        MaskedMetallographyDataset(split="holdout", **common),
    )


def build_gda_model(config, device):
    cfg = config["gda_mim"]
    base_model = build_model(config, device)
    checkpoint_path = project_path(config, cfg["base_checkpoint"])
    load_base_checkpoint(base_model, checkpoint_path, device)
    encoder = base_model.encoder
    base_model.encoder = None
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = GDAMaskedAutoencoder(
        encoder=encoder,
        channels=encoder.get_stage_channels(),
        bottleneck_ratio=cfg.get("bottleneck_ratio", 8),
        decoder_channels=cfg.get("decoder_channels", 96),
    ).to(device)
    trainable = sum(p.numel() for p in model.trainable_parameters())
    retained = sum(p.numel() for p in model.gda.parameters())
    logger.info(
        "GDA-MIM parameters: trainable=%.2fM, retained_after_pretrain=%.2fM",
        trainable / 1e6,
        retained / 1e6,
    )
    return model


def run_epoch(model, loader, device, config, optimizer=None, scaler=None):
    cfg = config["gda_mim"]
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "rgb": 0.0, "low_frequency": 0.0, "gradient": 0.0}
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast("cuda", enabled=use_amp):
                prediction = model(inputs)
                loss, details = gda_reconstruction_loss(
                    prediction,
                    targets,
                    masks,
                    rgb_weight=cfg.get("rgb_weight", 0.50),
                    low_frequency_weight=cfg.get("low_frequency_weight", 0.25),
                    gradient_weight=cfg.get("gradient_weight", 0.25),
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()), cfg.get("grad_clip", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
        totals["loss"] += float(loss.detach())
        for key in details:
            totals[key] += details[key]
    count = max(1, len(loader))
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def save_monitor(model, loader, device, output_dir, epoch, limit=6):
    model.eval()
    epoch_dir = Path(output_dir) / "monitor" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in loader:
        inputs = batch["input"].to(device)
        prediction = model(inputs).cpu().clamp(0.0, 1.0)
        targets = batch["target"]
        for index in range(inputs.shape[0]):
            name = Path(batch["image_path"][index]).stem
            panels = []
            for tensor in (targets[index], inputs[index].cpu(), prediction[index]):
                rgb = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
                panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            canvas = np.concatenate(panels, axis=1)
            cv2.imwrite(str(epoch_dir / f"{name}_target_masked_recon.jpg"), canvas)
            saved += 1
            if saved >= limit:
                return


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, best_loss):
    torch.save(
        {
            "epoch": epoch,
            "best_holdout_loss": best_loss,
            "gda_state_dict": model.gda.state_dict(),
            "reconstruction_decoder_state_dict": model.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "base_checkpoint": config["gda_mim"]["base_checkpoint"],
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="GDA masked-image pretraining")
    parser.add_argument("--config", default="config/train/gda_mim_g0a.yaml")
    parser.add_argument("--resume", default="")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = load_config(args.config)
    cfg = config["gda_mim"]
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = project_path(config, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    train_set, holdout_set = build_datasets(config)
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = dict(
        batch_size=cfg.get("batch_size", 2),
        num_workers=cfg.get("num_workers", 8),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    holdout_loader = DataLoader(holdout_set, shuffle=False, drop_last=False, **loader_kwargs)
    logger.info("GDA-MIM data: train=%d, holdout=%d", len(train_set), len(holdout_set))

    model = build_gda_model(config, device)
    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=cfg.get("learning_rate", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.05),
    )
    epochs = int(cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.get("min_learning_rate", 1e-6)
    )
    scaler = GradScaler("cuda", enabled=bool(cfg.get("amp", True)) and device.type == "cuda")
    start_epoch = 0
    best_loss = float("inf")
    resume = args.resume.strip()
    if resume:
        checkpoint = torch.load(project_path(config, resume), map_location=device, weights_only=False)
        model.gda.load_state_dict(checkpoint["gda_state_dict"])
        model.decoder.load_state_dict(checkpoint["reconstruction_decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_holdout_loss", best_loss))

    save_monitor(model, holdout_loader, device, output_dir, 0, cfg.get("monitor_images", 6))
    for epoch in range(start_epoch, epochs):
        train_stats = run_epoch(model, train_loader, device, config, optimizer, scaler)
        holdout_stats = run_epoch(model, holdout_loader, device, config)
        scheduler.step()
        logger.info(
            "Epoch %d/%d train=%.5f holdout=%.5f "
            "(rgb=%.5f low=%.5f grad=%.5f) lr=%.3e",
            epoch + 1,
            epochs,
            train_stats["loss"],
            holdout_stats["loss"],
            holdout_stats["rgb"],
            holdout_stats["low_frequency"],
            holdout_stats["gradient"],
            optimizer.param_groups[0]["lr"],
        )
        if holdout_stats["loss"] < best_loss:
            best_loss = holdout_stats["loss"]
            save_checkpoint(
                os.path.join(output_dir, "best_gda_mim.pth"),
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_loss,
            )
        interval = int(cfg.get("checkpoint_interval", 5))
        if (epoch + 1) % interval == 0:
            save_checkpoint(
                os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1:04d}.pth"),
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_loss,
            )
            save_monitor(
                model,
                holdout_loader,
                device,
                output_dir,
                epoch + 1,
                cfg.get("monitor_images", 6),
            )
    save_checkpoint(
        os.path.join(output_dir, "final_gda_mim.pth"),
        model,
        optimizer,
        scheduler,
        config,
        epochs - 1,
        best_loss,
    )
    logger.info("GDA-MIM training complete: %s", output_dir)


if __name__ == "__main__":
    main()
