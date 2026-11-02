# -*- coding: utf-8 -*-
"""独立 RGB 修复预训练。不加载 SAM2 或分割网络。"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.rgb_restoration_dataset import (
    PROFILES, LocalDamageDiagnosticDataset, NoiseDiagnosticDataset, RGBRestorationDataset,
)
from models.rgb_restoration import build_rgb_restorer
from utils.config import load_config, project_path


logger = logging.getLogger("rgb_restoration")


def reconstruction_errors(prediction, target, valid):
    """逐图 RGB/梯度误差；梯度的两个端点都有效才计入，不监督 padding。"""
    prediction, target, valid = prediction.float(), target.float(), valid.float()
    dims = (1, 2, 3)
    channels = prediction.shape[1]
    denominator = (valid.sum(dims) * channels).clamp_min(1)
    diff = prediction - target
    rgb = (diff.abs() * valid).sum(dims) / denominator
    mse = (diff.square() * valid).sum(dims) / denominator
    vx = valid[..., :, 1:] * valid[..., :, :-1]
    vy = valid[..., 1:, :] * valid[..., :-1, :]
    gx = (diff[..., :, 1:] - diff[..., :, :-1]).abs()
    gy = (diff[..., 1:, :] - diff[..., :-1, :]).abs()
    gradient = ((gx * vx).sum(dims) + (gy * vy).sum(dims)) / (
        (vx.sum(dims) + vy.sum(dims)) * channels
    ).clamp_min(1)
    return {"rgb_l1": rgb, "gradient_l1": gradient, "mse": mse}


def reconstruction_loss(prediction, target, valid, cfg, *, return_errors=False):
    errors = reconstruction_errors(prediction, target, valid)
    loss = (cfg["rgb_weight"] * errors["rgb_l1"]
            + cfg["gradient_weight"] * errors["gradient_l1"]).mean()
    return (loss, errors) if return_errors else loss


def seed_worker(_worker_id):
    cv2.setNumThreads(1)


def build_datasets(config):
    cfg = config["rgb_restoration"]
    split_policy = cfg.get("split_policy", "holdout")
    if split_policy not in {"holdout", "all_train"}:
        raise ValueError("split_policy must be holdout or all_train")
    pretraining = cfg.get("masked_pretraining")
    if pretraining and not 0 < pretraining["epochs"] < cfg["train"]["epochs"]:
        raise ValueError("masked pretraining must leave at least one ordinary deblur epoch")
    kwargs = dict(
        data_dir=project_path(config, cfg["data_dir"]),
        holdout_manifest=(project_path(config, cfg["holdout_manifest"])
                          if split_policy == "holdout" else None),
        split_policy=split_policy,
        image_size=cfg["image_size"], crop_size=cfg["crop_size"],
        degradation=cfg["degradation"], seed=cfg["seed"],
    )
    training = RGBRestorationDataset(split="train", masked_pretraining=pretraining,
                                      training_noise=cfg.get("training_noise"), **kwargs)
    validation = RGBRestorationDataset(split="holdout", **kwargs) if split_policy == "holdout" else None
    return training, validation


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_validation_preview(path, inputs, prediction, target, valid):
    h = int(valid[0, 0, :, 0].sum().item())
    w = int(valid[0, 0, 0, :].sum().item())
    panels = [x[0, :, :h, :w].detach().float().cpu() for x in (inputs, prediction, target)]
    rgb = torch.cat(panels, dim=2).permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise OSError(f"cannot encode preview: {path}")
    encoded.tofile(str(path))


def checkpoint_selection(validation, cfg):
    """选优任务显式化；清晰漂移上限与输入基准在训练前固定。"""
    selection = cfg.get("selection", {})
    profile = selection.get("profile", "all")
    if profile not in {"all", *PROFILES}:
        raise ValueError(f"unknown selection profile: {profile}")
    profiles = validation["profiles"]
    chosen = list(profiles.values()) if profile == "all" else [profiles[profile]]
    losses = {
        prefix: sum(cfg["loss"]["rgb_weight"] * row[f"{prefix}_rgb_l1"]
                    + cfg["loss"]["gradient_weight"] * row[f"{prefix}_gradient_l1"]
                    for row in chosen) / len(chosen)
        for prefix in ("input", "restored")
    }
    identity = profiles["identity"]["restored_rgb_l1"]
    limit = selection.get("max_identity_l1")
    eligible = limit is None or identity <= float(limit)
    if selection.get("require_input_improvement", False):
        eligible = eligible and losses["restored"] < losses["input"]
    return {
        "profile": profile, "loss": losses["restored"], "baseline_loss": losses["input"],
        "identity_l1": identity, "max_identity_l1": limit, "eligible": bool(eligible),
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg, use_amp, amp_dtype, preview_path):
    model.eval()
    grouped = {profile: [] for profile in PROFILES}
    preview_saved = False
    preview_profile = cfg.get("selection", {}).get("profile", "mixed")
    if preview_profile == "all":
        preview_profile = "mixed"
    for batch in loader:
        inputs, target, valid = [batch[key].to(device, non_blocking=True) for key in ("input", "target", "valid")]
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            prediction = model(inputs)
        restored = reconstruction_errors(prediction, target, valid)
        original = reconstruction_errors(inputs, target, valid)
        for index, profile in enumerate(batch["profile"]):
            row = {f"{prefix}_{key}": float(value[index].item())
                   for prefix, errors in (("input", original), ("restored", restored))
                   for key, value in errors.items()}
            if not all(math.isfinite(value) for value in row.values()):
                raise FloatingPointError("non-finite validation result")
            grouped[profile].append(row)
        if not preview_saved and batch["profile"][0] == preview_profile:
            save_validation_preview(preview_path, inputs, prediction, target, valid)
            preview_saved = True
    result = {}
    for profile, rows in grouped.items():
        if not rows:
            raise ValueError(f"validation is missing profile {profile}")
        mean = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        mean["images"] = len(rows)
        # identity 的输入就是参考图，PSNR 无穷大无比较意义，只报告绝对漂移。
        if profile != "identity":
            for prefix in ("input", "restored"):
                mean[f"{prefix}_psnr_db"] = -10 * math.log10(max(mean[f"{prefix}_mse"], 1e-12))
        result[profile] = mean
    loss = sum(cfg["loss"]["rgb_weight"] * row["restored_rgb_l1"]
               + cfg["loss"]["gradient_weight"] * row["restored_gradient_l1"]
               for row in result.values()) / len(PROFILES)
    return {"loss": loss, "profiles": result}


@torch.no_grad()
def evaluate_local_damage(model, loader, device, use_amp, amp_dtype):
    """单列全裁块与实际受损像素误差；区域不是晶界标注，不改变 best.pt 选择。"""
    model.eval()
    rows = []
    for batch in loader:
        inputs, target, valid, affected = [batch[key].to(device, non_blocking=True)
                                         for key in ("input", "target", "valid", "affected")]
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            prediction = model(inputs)
        errors = {
            region: {f"{prefix}_{key}": values.detach().cpu().tolist()
                     for prefix, values_by_key in (
                         ("input", reconstruction_errors(inputs, target, region_valid)),
                         ("restored", reconstruction_errors(prediction, target, region_valid)))
                     for key, values in values_by_key.items()}
            for region, region_valid in (("full", valid), ("affected", affected))
        }
        for index, source in enumerate(batch["source"]):
            row = {"source": source,
                   "affected_fraction": float((affected[index].sum() / valid[index].sum()).item())}
            for region, values in errors.items():
                row[region] = {key: value[index] for key, value in values.items()}
                if not all(math.isfinite(value) for value in row[region].values()):
                    raise FloatingPointError("non-finite local damage diagnostic")
            rows.append(row)
    summary = {}
    for region in ("full", "affected"):
        mean = {key: sum(row[region][key] for row in rows) / len(rows) for key in rows[0][region]}
        for prefix in ("input", "restored"):
            mean[f"{prefix}_psnr_db"] = -10 * math.log10(max(mean[f"{prefix}_mse"], 1e-12))
        summary[region] = mean
    return {"images": len(rows), "regions": summary, "per_source": rows}


@torch.no_grad()
def evaluate_noise(model, loader, device, use_amp, amp_dtype):
    """固定保留图的多强度噪声诊断；不参与训练损失或checkpoint选优。"""
    model.eval()
    rows = []
    for batch in loader:
        inputs, target, valid = [batch[key].to(device, non_blocking=True)
                                 for key in ("input", "target", "valid")]
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            prediction = model(inputs)
        errors = {f"{prefix}_{key}": values.detach().cpu().tolist()
                  for prefix, group in (("input", reconstruction_errors(inputs, target, valid)),
                                        ("restored", reconstruction_errors(prediction, target, valid)))
                  for key, values in group.items()}
        for index, source in enumerate(batch["source"]):
            metrics = {key: value[index] for key, value in errors.items()}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("non-finite noise diagnostic")
            rows.append({"source": source, "sigma": float(batch["noise_sigma"][index]), **metrics})
    summary = {}
    for sigma in sorted({row["sigma"] for row in rows}):
        group = [row for row in rows if row["sigma"] == sigma]
        mean = {key: sum(row[key] for row in group) / len(group) for key in errors}
        for prefix in ("input", "restored"):
            mean[f"{prefix}_psnr_db"] = -10 * math.log10(max(mean[f"{prefix}_mse"], 1e-12))
        summary[f"{sigma:g}"] = {"images": len(group), **mean}
    return {"cases": len(rows), "sigmas": summary, "per_source": rows}


def train(config, *, device, output_dir=None, smoke=False, resume=None):
    cfg = copy.deepcopy(config["rgb_restoration"])
    if output_dir is None:
        output_dir = project_path(config, cfg["output_dir"] + ("_smoke" if smoke else ""))
    output_dir = Path(output_dir).resolve()
    if (output_dir / "run_config.json").exists() and not resume:
        raise FileExistsError(f"run already exists: {output_dir}; use --resume or a new --output-dir")
    if smoke and resume:
        raise ValueError("smoke checkpoints are diagnostics, not resumable full runs")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    cv2.setNumThreads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_rgb_restorer(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                 weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["min_learning_rate"],
    )
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16, init_scale=1024)
    start_epoch, updates, best_loss, best_epoch = 0, 0, None, 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != model.checkpoint_format or checkpoint.get("smoke"):
            raise ValueError(f"resume requires a full {model.checkpoint_format} training checkpoint")
        previous = copy.deepcopy(checkpoint["experiment_config"])
        current = copy.deepcopy(cfg)
        for item in (previous, current):
            item.pop("output_dir", None)
            item["train"].pop("num_workers", None)
        if previous != current:
            raise ValueError("resume config differs from checkpoint; keep data/model/loss/training recipe fixed")
        if Path(resume).resolve() != output_dir / "last.pt":
            raise ValueError("resume from last.pt in the same output directory to preserve the run history")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, updates = checkpoint["epoch"], checkpoint["updates"]
        best_loss, best_epoch = checkpoint["best_loss"], checkpoint["best_epoch"]
    epochs = 1 if smoke else int(cfg["train"]["epochs"])
    if start_epoch >= epochs:
        raise ValueError(f"run already completed {start_epoch}/{epochs} epochs")
    train_set, val_set = build_datasets(config)
    validation = (Subset(val_set, range(min(2, len(val_set.samples)) * len(PROFILES)))
                  if smoke and val_set is not None else val_set)
    selection_mode = "holdout" if val_set is not None else "final_epoch"
    workers = int(cfg["train"]["num_workers"])
    generator = torch.Generator()
    loader_kwargs = dict(num_workers=workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              generator=generator, **loader_kwargs)
    val_loader = (DataLoader(validation, batch_size=1, shuffle=False, **loader_kwargs)
                  if validation is not None else None)
    local_loader = None
    local_cfg = cfg.get("diagnostics", {}).get("local_damage")
    if local_cfg and val_set is not None:
        local_set = LocalDamageDiagnosticDataset(val_set, local_cfg)
        if smoke:
            local_set = Subset(local_set, range(min(2, len(local_set))))
        # 独立 generator 避免额外评估消耗训练随机流。
        local_loader = DataLoader(local_set, batch_size=1, shuffle=False,
                                  generator=torch.Generator().manual_seed(cfg["seed"]), **loader_kwargs)
    noise_loader = None
    noise_cfg = cfg.get("diagnostics", {}).get("noise")
    if noise_cfg and val_set is not None:
        noise_set = NoiseDiagnosticDataset(val_set, noise_cfg)
        if smoke:
            noise_set = Subset(noise_set, range(min(2, len(val_set.samples)) * len(noise_cfg["sigmas"])))
        noise_loader = DataLoader(noise_set, batch_size=1, shuffle=False,
                                  generator=torch.Generator().manual_seed(cfg["seed"]), **loader_kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = sum(p.numel() for p in model.parameters())
    metadata = {
        "format": model.checkpoint_format, "experiment_config": cfg, "smoke": smoke,
        "model_class": type(model).__name__, "parameters": parameters, "trainable_parameters": parameters,
        "device": str(device), "torch": str(torch.__version__), "amp_dtype": str(amp_dtype) if use_amp else None,
        "train_sources": [Path(path).name for path in train_set.samples],
        "holdout_sources": [Path(path).name for path in val_set.samples] if val_set is not None else [],
        "validation_cases": len(validation) if validation is not None else 0, "planned_epochs": epochs,
        "split_policy": train_set.split_policy, "selection_mode": selection_mode,
        "intended_checkpoint": "best.pt" if val_set is not None else "last.pt",
        "max_train_steps_per_epoch": 2 if smoke else None,
        "local_diagnostic_cases": len(local_loader.dataset) if local_loader is not None else 0,
        "noise_diagnostic_cases": len(noise_loader.dataset) if noise_loader is not None else 0,
    }
    write_json(output_dir / "run_config.json", metadata)
    logger.info("%s: %s parameters; train=%s sources, holdout=%s sources; selection=%s; smoke=%s",
                type(model).__name__, parameters, len(train_set.samples), len(metadata["holdout_sources"]),
                selection_mode, smoke)
    started = time.perf_counter()
    summary = {"status": "running", "parameters": parameters, "smoke": smoke,
               "selection_mode": selection_mode, "intended_checkpoint": metadata["intended_checkpoint"]}
    write_json(output_dir / "summary.json", summary)
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        train_set.set_epoch(epoch)
        generator.manual_seed(cfg["seed"] + epoch)
        model.train()
        loss_sum, samples, successful, skipped = 0.0, 0, 0, 0
        error_sums = {key: 0.0 for key in ("rgb_l1", "gradient_l1", "mse")}
        masked_samples, mask_weight_sum = 0, 0.0
        noise_samples, noise_sigma_sum = 0, 0.0
        mask_strength = train_set.masking_strength()
        phase = "masked_pretraining" if mask_strength else "ordinary_reconstruction"
        if mask_strength and epoch >= cfg["masked_pretraining"]["epochs"]:
            phase = "mixed_local_reconstruction"
        logger.info("epoch=%s phase=%s masking_strength=%.3f", epoch + 1, phase, mask_strength)
        learning_rate = optimizer.param_groups[0]["lr"]
        for step, batch in enumerate(itertools.islice(train_loader, 2 if smoke else None), start=1):
            inputs, target, valid = [batch[key].to(device, non_blocking=True) for key in ("input", "target", "valid")]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                prediction = model(inputs)
                loss, errors = reconstruction_loss(prediction, target, valid, cfg["loss"], return_errors=True)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1} step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"],
                                           error_if_nonfinite=not scaler.is_enabled())
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1
            else:
                successful += 1
            loss_sum += loss.item() * len(inputs)
            for key, value in errors.items():
                error_sums[key] += float(value.detach().sum().item())
            samples += len(inputs)
            if "mask_mean" in batch:
                masked_samples += int((batch["mask_mean"] > 0).sum())
                mask_weight_sum += float(batch["mask_mean"].sum())
            if "noise_sigma" in batch:
                noise_samples += int((batch["noise_sigma"] > 0).sum())
                noise_sigma_sum += float(batch["noise_sigma"].sum())
            if step == 1 or step % 25 == 0 or smoke:
                logger.info("epoch=%s step=%s train_loss=%.6f", epoch + 1, step, loss_sum / samples)
        if successful == 0:
            raise RuntimeError("no successful optimizer updates")
        updates += successful
        validation_result = (evaluate(model, val_loader, device, cfg, use_amp, amp_dtype,
                                      output_dir / f"validation_epoch_{epoch + 1:03d}.png")
                             if val_loader is not None else None)
        local_result = (evaluate_local_damage(model, local_loader, device, use_amp, amp_dtype)
                        if local_loader is not None else None)
        noise_result = (evaluate_noise(model, noise_loader, device, use_amp, amp_dtype)
                        if noise_loader is not None else None)
        scheduler.step()
        selection = (checkpoint_selection(validation_result, cfg) if validation_result is not None else {
            "enabled": False, "eligible": False, "profile": None, "loss": None,
            "baseline_loss": None, "identity_l1": None, "max_identity_l1": None,
        })
        improved = selection["eligible"] and (best_loss is None or selection["loss"] < best_loss)
        if improved:
            best_loss, best_epoch = selection["loss"], epoch + 1
        row = {
            "epoch": epoch + 1, "train_loss": loss_sum / samples, "train_samples": samples,
            "train_errors": {key: value / samples for key, value in error_sums.items()},
            "updates": successful, "skipped_updates": skipped, "total_updates": updates,
            "learning_rate": learning_rate, "validation": validation_result,
            "phase": phase, "masking_strength": mask_strength,
            "masked_samples": masked_samples, "mean_mask_weight": mask_weight_sum / samples,
            "masking_probability_given_blur": train_set.masking_probability(),
            "noise_samples": noise_samples, "mean_noise_sigma": noise_sigma_sum / samples,
            "selection": selection,
            "seconds": time.perf_counter() - epoch_started,
        }
        if local_result is not None or val_set is None:
            row["local_damage"] = local_result
        if noise_result is not None or val_set is None:
            row["noise_diagnostic"] = noise_result
        payload = {
            "format": model.checkpoint_format, "model_config": model.model_config, "model": model.state_dict(),
            "experiment_config": cfg, "epoch": epoch + 1, "updates": updates, "smoke": smoke,
            "best_loss": best_loss, "best_epoch": best_epoch, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "metrics": row,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if improved:
            save_checkpoint(output_dir / "best.pt", payload)
        if epoch + 1 in cfg["train"].get("save_epochs", []):
            save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pt", payload)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if validation_result is None:
            logger.info("epoch=%s train_loss=%.6f updates=%s skipped=%s selection=final_epoch; saved last.pt",
                        epoch + 1, loss_sum / samples, successful, skipped)
        else:
            logger.info("epoch=%s validation_loss=%.6f identity_L1=%.6f updates=%s skipped=%s "
                        "selection=%s:%.6f eligible=%s",
                        epoch + 1, validation_result["loss"],
                        validation_result["profiles"]["identity"]["restored_rgb_l1"], successful, skipped,
                        selection["profile"], selection["loss"], selection["eligible"])
        summary.update(epoch=epoch + 1, total_updates=updates, best_epoch=best_epoch, best_loss=best_loss,
                       best_checkpoint_available=best_epoch > 0, selection_profile=selection["profile"],
                       latest_metrics=row, elapsed_seconds=time.perf_counter() - started)
        if device.type == "cuda":
            summary.update(cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                           cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                           gpu=torch.cuda.get_device_name(device))
        write_json(output_dir / "summary.json", summary)
    summary["status"] = "completed"
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/rgb_restoration_v1.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", help="private run directory; defaults to config output_dir")
    parser.add_argument("--smoke", action="store_true", help="2 training batches; validate 2 held-out sources if enabled")
    parser.add_argument("--resume", help="same run's last.pt; resumes at the next complete epoch")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use sam2_env on the GPU server, or --device cpu for a smoke test")
    summary = train(load_config(args.config), device=device, output_dir=args.output_dir,
                    smoke=args.smoke, resume=args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
