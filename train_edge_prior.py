# -*- coding: utf-8 -*-
"""G0b pretraining for the fully retained generative edge prior."""

from __future__ import annotations

import argparse
import json
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
from models.edge_prior import (
    FrozenEncoderEdgePrior,
    GenerativeEdgePrior,
    build_structural_edge_target,
    edge_prior_loss,
)
from train_stage2 import build_model, load_base_checkpoint, seed_dataloader_worker
from utils.config import load_config, project_path


logger = logging.getLogger("train_edge_prior")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(config):
    cfg = config["edge_prior_pretrain"]
    common = dict(
        data_dir=project_path(config, cfg["unlabeled_dir"]),
        holdout_manifest=project_path(config, cfg["holdout_manifest"]),
        crop_size=cfg.get("crop_size", 512),
        mask_ratio_range=cfg.get("mask_ratio_range", [0.20, 0.35]),
        mask_patch_range=cfg.get("mask_patch_range", [16, 48]),
        physical_aug_probability=cfg.get("physical_aug_probability", 0.85),
    )
    return (
        MaskedMetallographyDataset(split="train", **common),
        MaskedMetallographyDataset(split="holdout", **common),
    )


def build_edge_model(config, device):
    cfg = config["edge_prior_pretrain"]
    base_model = build_model(config, device)
    load_base_checkpoint(
        base_model, project_path(config, cfg["base_checkpoint"]), device
    )
    encoder = base_model.encoder
    base_model.encoder = None
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    channels = encoder.get_stage_channels()[:2]
    prior = GenerativeEdgePrior(
        channels, hidden_channels=int(cfg.get("hidden_channels", 64))
    )
    model = FrozenEncoderEdgePrior(encoder, prior).to(device)
    retained = sum(parameter.numel() for parameter in model.prior.parameters())
    logger.info("Retained edge-prior parameters: %.3fM", retained / 1e6)
    return model


def run_epoch(model, loader, device, config, optimizer=None, scaler=None):
    cfg = config["edge_prior_pretrain"]
    training = optimizer is not None
    model.train(training)
    keys = (
        "loss", "edge_l1", "dice", "orientation", "multiscale",
        "background", "predicted_edge_mean", "target_edge_mean",
    )
    totals = {key: 0.0 for key in keys}
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        clean = batch["target"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        target = build_structural_edge_target(clean)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast("cuda", enabled=use_amp):
                raw = model(inputs)
                loss, details = edge_prior_loss(
                    raw,
                    target,
                    mask,
                    positive_weight=cfg.get("positive_weight", 4.0),
                    masked_region_weight=cfg.get("masked_region_weight", 1.5),
                    dice_weight=cfg.get("dice_weight", 0.5),
                    orientation_weight=cfg.get("orientation_weight", 0.25),
                    multiscale_weight=cfg.get("multiscale_weight", 0.25),
                    background_weight=cfg.get("background_weight", 0.10),
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
        for key, value in details.items():
            totals[key] += float(value)
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
        clean = batch["target"].to(device)
        target = build_structural_edge_target(clean)
        raw = model(inputs)
        prediction, _ = GenerativeEdgePrior.decode(raw)
        for index in range(inputs.shape[0]):
            name = Path(batch["image_path"][index]).stem
            clean_rgb = (
                clean[index].permute(1, 2, 0).cpu().numpy() * 255.0
            ).round().clip(0, 255).astype(np.uint8)
            input_rgb = (
                inputs[index].permute(1, 2, 0).cpu().numpy() * 255.0
            ).round().clip(0, 255).astype(np.uint8)
            target_edge = target[index, 0].cpu().numpy()
            predicted_edge = prediction[index, 0].float().cpu().numpy()
            error = np.abs(predicted_edge - target_edge)
            edge_panels = []
            for value in (target_edge, predicted_edge, error):
                gray = np.clip(value * 255.0, 0, 255).astype(np.uint8)
                edge_panels.append(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO))
            panels = [
                cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR),
                cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR),
                *edge_panels,
            ]
            cv2.imwrite(
                str(epoch_dir / f"{name}_clean_input_target_pred_error.jpg"),
                np.concatenate(panels, axis=1),
            )
            raw16 = np.clip(predicted_edge * 65535.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(epoch_dir / f"{name}_edge_raw16.png"), raw16)
            saved += 1
            if saved >= int(limit):
                return


def checkpoint_payload(model, optimizer, scheduler, config, epoch, best_loss):
    return {
        "format_version": 1,
        "architecture": "GenerativeEdgePrior",
        "epoch": int(epoch),
        "best_holdout_loss": float(best_loss),
        "edge_prior_state_dict": model.prior.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "base_checkpoint": config["edge_prior_pretrain"]["base_checkpoint"],
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description="G0b retained edge-prior pretraining")
    parser.add_argument(
        "--config", default="config/train/edge_prior_g0b.yaml"
    )
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = load_config(args.config)
    cfg = config["edge_prior_pretrain"]
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = project_path(config, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    train_set, holdout_set = build_datasets(config)
    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=int(cfg.get("batch_size", 4)),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg.get("num_workers", 8)),
        **common,
    )
    # A single-process holdout loader avoids worker teardown issues when the
    # monitor intentionally stops after a few deterministic samples.
    holdout_loader = DataLoader(
        holdout_set,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        **common,
    )
    logger.info("G0b data: train=%d holdout=%d", len(train_set), len(holdout_set))

    model = build_edge_model(config, device)
    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=cfg.get("learning_rate", 2e-4),
        weight_decay=cfg.get("weight_decay", 0.02),
    )
    epochs = int(cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.get("min_learning_rate", 2e-6)
    )
    scaler = GradScaler(
        "cuda", enabled=bool(cfg.get("amp", True)) and device.type == "cuda"
    )
    start_epoch = 0
    best_loss = float("inf")
    history = []
    if args.resume.strip():
        checkpoint = torch.load(
            project_path(config, args.resume.strip()),
            map_location=device,
            weights_only=False,
        )
        model.prior.load_state_dict(checkpoint["edge_prior_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_holdout_loss", best_loss))

    save_monitor(model, holdout_loader, device, output_dir, 0, cfg.get("monitor_images", 6))
    for epoch in range(start_epoch, epochs):
        train_stats = run_epoch(model, train_loader, device, config, optimizer, scaler)
        holdout_stats = run_epoch(model, holdout_loader, device, config)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_stats,
            "holdout": holdout_stats,
        }
        history.append(record)
        with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        logger.info(
            "Epoch %d/%d train=%.5f holdout=%.5f "
            "edge=%.5f dice=%.5f ori=%.5f pred_mean=%.4f target_mean=%.4f lr=%.3e",
            epoch + 1,
            epochs,
            train_stats["loss"],
            holdout_stats["loss"],
            holdout_stats["edge_l1"],
            holdout_stats["dice"],
            holdout_stats["orientation"],
            holdout_stats["predicted_edge_mean"],
            holdout_stats["target_edge_mean"],
            optimizer.param_groups[0]["lr"],
        )
        if holdout_stats["loss"] < best_loss:
            best_loss = holdout_stats["loss"]
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, config, epoch, best_loss
                ),
                os.path.join(output_dir, "best_edge_prior.pth"),
            )
        interval = int(cfg.get("checkpoint_interval", 5))
        if (epoch + 1) % interval == 0:
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, config, epoch, best_loss
                ),
                os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1:04d}.pth"),
            )
            save_monitor(
                model,
                holdout_loader,
                device,
                output_dir,
                epoch + 1,
                cfg.get("monitor_images", 6),
            )
    torch.save(
        checkpoint_payload(
            model, optimizer, scheduler, config, epochs - 1, best_loss
        ),
        os.path.join(output_dir, "final_edge_prior.pth"),
    )
    logger.info("G0b training complete: %s (best=%.5f)", output_dir, best_loss)


if __name__ == "__main__":
    main()
