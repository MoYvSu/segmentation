# -*- coding: utf-8 -*-
"""独立像素域扩散修复训练；固定样本过拟合与全量训练均不建立留出验证集。"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.rgb_restoration import build_rgb_restorer
from train_rgb_restoration import (
    build_datasets, reconstruction_errors, reconstruction_loss, save_checkpoint,
    seed_worker, write_json,
)
from utils.config import load_config, project_path


logger = logging.getLogger("rgb_diffusion_restoration")
ERROR_KEYS = ("rgb_l1", "gradient_l1", "mse")
DEFAULT_GATE = {"rgb_l1": 0.8, "gradient_l1": 0.95, "mse": 0.8}


def fixed_training_samples(config, count):
    """独立数据集固定 epoch=0、纯模糊配对；样本仍属于正式训练池。"""
    fixed_config = copy.deepcopy(config)
    cfg = fixed_config["rgb_restoration"]
    cfg["split_policy"] = "all_train"
    cfg["masked_pretraining"] = None
    cfg["training_noise"] = None
    cfg["degradation"]["profile_probabilities"] = [0.0, 1.0, 0.0, 0.0]
    dataset, holdout = build_datasets(fixed_config)
    if holdout is not None:
        raise ValueError("fixed training monitor must not create holdout data")
    if not 1 <= count <= len(dataset):
        raise ValueError("fixed sample count must be between one and the training pool size")
    dataset.set_epoch(0)
    ordered = sorted(range(len(dataset)), key=lambda i: str(dataset.samples[i]))
    # 排序后均匀取源图；不依赖内容、标签或模型误差挑图。
    positions = np.linspace(0, len(ordered) - 1, count, dtype=int).tolist()
    indices = [ordered[position] for position in positions]
    items = [dataset[index] for index in indices]
    fixed = {key: torch.stack([item[key] for item in items]) for key in ("input", "target", "valid")}
    fixed["source"] = [item["source"] for item in items]
    manifest = {
        "purpose": "training_samples_only_not_holdout_or_model_selection",
        "selection": "uniform_positions_in_sorted_training_source_paths",
        "source_indices": indices, "sources": fixed["source"], "epoch": 0,
        "profile": "blur", "masked_pretraining": None, "training_noise": None,
        "image_size": cfg["image_size"], "crop_size": cfg["crop_size"],
        "seed": cfg["seed"], "degradation": cfg["degradation"],
    }
    return fixed, manifest


def error_report(inputs, predictions, targets, valid, sources):
    baseline = reconstruction_errors(inputs, targets, valid)
    restored = reconstruction_errors(predictions, targets, valid)
    rows = []
    for index, source in enumerate(sources):
        row = {"source": source}
        for prefix, group in (("input", baseline), ("restored", restored)):
            row[prefix] = {key: float(group[key][index]) for key in ERROR_KEYS}
        if not all(math.isfinite(v) for prefix in ("input", "restored") for v in row[prefix].values()):
            raise FloatingPointError("non-finite full-sampling monitor output")
        row["ratios"] = {key: row["restored"][key] / max(row["input"][key], 1e-12)
                         for key in ERROR_KEYS}
        rows.append(row)
    means = {prefix: {key: sum(row[prefix][key] for row in rows) / len(rows) for key in ERROR_KEYS}
             for prefix in ("input", "restored")}
    means["ratios"] = {key: means["restored"][key] / max(means["input"][key], 1e-12)
                       for key in ERROR_KEYS}
    return {"samples": len(rows), "mean": means, "per_source": rows}


def overfit_gate(report, thresholds):
    """门槛只核对完整采样后的训练配对，绝不代表泛化或分割收益。"""
    if set(thresholds) != set(ERROR_KEYS):
        raise ValueError("overfit gate must specify rgb_l1, gradient_l1 and mse ratios")
    if not all(math.isfinite(float(v)) and 0 < float(v) < 1 for v in thresholds.values()):
        raise ValueError("overfit gate ratios must be finite and strictly between zero and one")
    checks = {key: (report["mean"]["input"][key] > 1e-12
                    and math.isfinite(report["mean"]["ratios"][key])
                    and report["mean"]["ratios"][key] <= float(thresholds[key])) for key in ERROR_KEYS}
    per_image_rgb = all(row["input"]["rgb_l1"] > 1e-12 and row["ratios"]["rgb_l1"] < 1.0
                        for row in report.get("per_source", []))
    checks["every_image_rgb_improves"] = per_image_rgb
    return {"passed": bool(all(checks.values())), "thresholds": dict(thresholds), "checks": checks,
            "per_image_rgb_ratio_strict_upper_bound": 1.0,
            "scope": "fixed_training_pair_engineering_overfit_only_not_generalization"}


def _thumbnail_row(tensors, labels, size):
    panels = []
    for tensor, label in zip(tensors, labels):
        rgb = tensor.detach().float().cpu().permute(1, 2, 0).numpy()
        rgb = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        panel = np.full((size + 24, size, 3), 245, dtype=np.uint8)
        panel[24:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1,
                    cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def save_monitor_image(path, inputs, restored, target, *, title, size=224, roi_size=256):
    """固定 [0,1] 色阶；上排完整裁块，下排固定中心 ROI，便于连续观察。"""
    tensors = (inputs, restored, target)
    h, w = inputs.shape[-2:]
    roi = min(roi_size, h, w)
    top, left = (h - roi) // 2, (w - roi) // 2
    full = _thumbnail_row(tensors, ("Input", "Restored", "Clean training target"), size)
    detail = _thumbnail_row([x[:, top:top + roi, left:left + roi] for x in tensors],
                            ("Input center ROI", "Restored center ROI", "Target center ROI"), size)
    header = np.full((36, size * 3, 3), 245, dtype=np.uint8)
    cv2.putText(header, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", np.concatenate((header, full, detail), axis=0))
    if not ok:
        raise OSError(f"cannot encode monitor: {path}")
    encoded.tofile(str(path))


@torch.no_grad()
def monitor(model, fixed, output_dir, *, device, epoch, updates, monitor_cfg, use_amp, amp_dtype):
    """逐图完整采样；专用随机流不改变训练噪声、DataLoader 或全局随机状态。"""
    previous_training = model.training
    model.eval()
    predictions, first_predictions = [], []
    destination = Path(output_dir) / "monitor" / f"epoch_{epoch:03d}_update_{updates:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for index, source in enumerate(fixed["source"]):
            inputs = fixed["input"][index:index + 1].to(device)
            generator = torch.Generator(device=device).manual_seed(int(monitor_cfg.get("sampling_seed", 314159)) + index)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                prediction = model(inputs, generator=generator)
                if monitor_cfg.get("include_first_step", False):
                    first_rng = torch.Generator(device=device).manual_seed(int(monitor_cfg.get("sampling_seed", 314159)) + index)
                    last_t = torch.full((len(inputs),), model.num_steps, device=device, dtype=torch.long)
                    # eta_T=1，起点只依赖模糊输入与噪声，绝不读取清晰参考。
                    initial_state = model.q_sample(inputs, inputs, last_t, generator=first_rng)
                    first_predictions.append(model.denoise(initial_state, inputs, last_t).clamp(0, 1).float().cpu())
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"non-finite full sampling for {source}")
            predictions.append(prediction.detach().float().cpu())
            save_monitor_image(destination / f"{index:02d}_{Path(source).stem}.png",
                               fixed["input"][index], predictions[-1][0], fixed["target"][index],
                               title=f"Epoch {epoch} | update {updates} | {model.num_steps} sampling steps",
                               size=int(monitor_cfg.get("thumbnail_size", 224)),
                               roi_size=int(monitor_cfg.get("roi_size", 256)))
            if first_predictions:
                save_monitor_image(destination / f"{index:02d}_{Path(source).stem}_first.png",
                                   fixed["input"][index], first_predictions[-1][0], fixed["target"][index],
                                   title=f"Epoch {epoch} | update {updates} | first x0 estimate",
                                   size=int(monitor_cfg.get("thumbnail_size", 224)),
                                   roi_size=int(monitor_cfg.get("roi_size", 256)))
        report = error_report(fixed["input"], torch.cat(predictions), fixed["target"],
                              fixed["valid"], fixed["source"])
        if first_predictions:
            report["first_step"] = error_report(fixed["input"], torch.cat(first_predictions), fixed["target"],
                                                fixed["valid"], fixed["source"])
        report.update(epoch=epoch, updates=updates, sampling_steps=model.num_steps,
                      sampling_seed=int(monitor_cfg.get("sampling_seed", 314159)),
                      amp_dtype=str(amp_dtype) if use_amp else None,
                      scope="training_samples_only_not_holdout_or_model_selection")
        write_json(destination / "errors.json", report)
        return report
    finally:
        model.train(previous_training)


def capture_rng(noise_generator, loader_generator, device):
    return {"noise_generator": noise_generator.get_state(), "loader_generator": loader_generator.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def restore_rng(state, noise_generator, loader_generator, device):
    noise_generator.set_state(state["noise_generator"].cpu())
    loader_generator.set_state(state["loader_generator"].cpu())
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _same_recipe(previous, current):
    previous, current = copy.deepcopy(previous), copy.deepcopy(current)
    for item in (previous, current):
        item.pop("output_dir", None)
        item["train"].pop("num_workers", None)
    return previous == current


def _same_overfit_extension_recipe(previous, current):
    previous, current = copy.deepcopy(previous), copy.deepcopy(current)
    if int(current["overfit"]["steps"]) <= int(previous["overfit"]["steps"]):
        return False
    for item in (previous, current):
        item["overfit"].pop("steps", None)
        item["overfit"].pop("save_steps", None)
    return _same_recipe(previous, current)


def _append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def train(config, *, device, output_dir=None, resume=None, overfit=False, extend_overfit_from=None):
    cfg = copy.deepcopy(config["rgb_restoration"])
    if cfg.get("split_policy") != "all_train":
        raise ValueError("diffusion experiment requires split_policy=all_train; no held-out validation")
    if cfg["model"].get("architecture") != "pixel_diffusion_nafnet":
        raise ValueError("this trainer requires architecture=pixel_diffusion_nafnet")
    mode = "overfit_engineering" if overfit else "all_train"
    output_dir = Path(output_dir or project_path(config, cfg["output_dir"] + ("_overfit" if overfit else ""))).resolve()
    if extend_overfit_from:
        if not overfit or resume:
            raise ValueError("overfit extension requires --overfit and cannot use --resume")
        if Path(extend_overfit_from).resolve().parent == output_dir or (output_dir.exists() and any(output_dir.iterdir())):
            raise ValueError("overfit extension requires a fresh output directory distinct from its parent")
    if (output_dir / "run_config.json").exists() and not resume:
        raise FileExistsError(f"run already exists: {output_dir}; use --resume or a new --output-dir")
    if resume and Path(resume).resolve() != output_dir / "last.pt":
        raise ValueError("resume requires last.pt in the same output directory")
    fit_cfg, monitor_cfg = cfg.get("overfit", {}), cfg.get("monitor", {})
    thresholds = fit_cfg.get("gate", DEFAULT_GATE)
    if overfit:
        # 提前校验门槛结构；实际判定只在最终完整采样后执行。
        overfit_gate({"mean": {"input": dict.fromkeys(ERROR_KEYS, 1), "ratios": dict.fromkeys(ERROR_KEYS, 1)}}, thresholds)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cv2.setNumThreads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_rgb_restorer(cfg["model"]).to(device)
    learning_rate = float(fit_cfg.get("learning_rate", 2e-4) if overfit else cfg["train"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg["train"]["weight_decay"])
    epochs = int(cfg["train"]["epochs"])
    scheduler = None if overfit else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(cfg["train"]["min_learning_rate"]))
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16, init_scale=1024)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 271828)
    loader_generator = torch.Generator().manual_seed(seed)
    checkpoint = None
    checkpoint_path = resume or extend_overfit_from
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != model.checkpoint_format or checkpoint.get("mode") != mode:
            raise ValueError("resume requires matching diffusion format and training mode")
        same_recipe = _same_overfit_extension_recipe if extend_overfit_from else _same_recipe
        if not same_recipe(checkpoint["experiment_config"], cfg):
            raise ValueError("resume config differs from checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
    train_set, val_set = build_datasets(config)
    if val_set is not None:
        raise ValueError("diffusion training must not construct held-out validation")
    expected_sources = cfg.get("expected_train_sources")
    if expected_sources is not None and len(train_set) != int(expected_sources):
        raise ValueError(f"expected {expected_sources} training sources, found {len(train_set)}")
    count = int(fit_cfg.get("samples", 4) if overfit else monitor_cfg.get("samples", 4))
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_path:
        fixed_dir = Path(checkpoint_path).resolve().parent
        fixed = torch.load(fixed_dir / "fixed_samples.pt", map_location="cpu", weights_only=True)
        fixed_manifest = json.loads((fixed_dir / "fixed_samples_manifest.json").read_text(encoding="utf-8"))
        if len(fixed["source"]) != count or fixed["source"] != fixed_manifest["sources"]:
            raise ValueError("saved fixed sample set differs from run configuration")
    else:
        fixed, fixed_manifest = fixed_training_samples(config, count)
    if not resume:
        save_checkpoint(output_dir / "fixed_samples.pt", fixed)
        write_json(output_dir / "fixed_samples_manifest.json", fixed_manifest)
    loader = None if overfit else DataLoader(
        train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]), pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=loader_generator)
    planned_updates = int(fit_cfg.get("steps", 800)) if overfit else epochs * len(loader)
    monitor_every = int(fit_cfg.get("monitor_every", 100) if overfit else monitor_cfg.get("every_epochs", 5))
    if planned_updates < 1 or monitor_every < 1:
        raise ValueError("positive training budget and monitor interval required")
    completed_epoch = int(checkpoint["epoch"]) if checkpoint else 0
    updates = int(checkpoint["updates"]) if checkpoint else 0
    failed_updates = int(checkpoint.get("failed_updates", 0)) if checkpoint else 0
    if updates >= planned_updates or (not overfit and completed_epoch >= epochs):
        raise ValueError("run already completed its configured update budget")
    if checkpoint:
        restore_rng(checkpoint["rng"], noise_generator, loader_generator, device)
    continuation = checkpoint.get("continuation") if checkpoint else None
    if extend_overfit_from:
        continuation = {"parent_checkpoint": str(Path(extend_overfit_from).resolve()),
                        "parent_updates": updates,
                        "parent_planned_updates": checkpoint["experiment_config"]["overfit"]["steps"],
                        "preserved": ["model", "optimizer", "scaler", "rng", "fixed_samples"]}
    parameters = sum(p.numel() for p in model.parameters())
    metadata = {
        "format": model.checkpoint_format, "experiment_config": cfg, "mode": mode,
        "model_class": type(model).__name__, "parameters": parameters,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_sources": ([*fixed["source"]] if overfit else [Path(p).name for p in train_set.samples]),
        "training_pool_sources": len(train_set), "holdout_sources": [], "validation_cases": 0,
        "split_policy": "all_train", "selection_mode": "engineering_gate" if overfit else "final_epoch",
        "intended_checkpoint": "last.pt", "planned_epochs": None if overfit else epochs,
        "planned_updates": planned_updates, "overfit_is_holdout": False, "automatic_formal_training": False,
        "fixed_samples": fixed_manifest, "device": str(device), "torch": str(torch.__version__),
        "amp_dtype": str(amp_dtype) if use_amp else None,
        "continuation": continuation,
    }
    if not resume:
        write_json(output_dir / "run_config.json", metadata)
    logger.info("mode=%s parameters=%s pool=%s train_sources=%s planned_updates=%s amp=%s",
                mode, parameters, len(train_set), len(metadata["train_sources"]), planned_updates, metadata["amp_dtype"])
    prior_elapsed = float(checkpoint.get("elapsed_seconds", 0)) if checkpoint else 0.0
    started = time.perf_counter()
    report = checkpoint.get("output_metrics") if checkpoint else None
    summary = {"status": "running", "mode": mode, "gate_passed": None, "parameters": parameters,
               "planned_updates": planned_updates, "updates": updates, "total_updates": updates,
               "failed_updates": failed_updates, "selection_mode": metadata["selection_mode"],
               "intended_checkpoint": "last.pt", "validation": None, "best_checkpoint_available": False,
               "automatic_formal_training": False}

    def update_summary():
        summary.update(epoch=completed_epoch, updates=updates, total_updates=updates,
                       failed_updates=failed_updates, elapsed_seconds=prior_elapsed + time.perf_counter() - started,
                       output_metrics=report)
        if device.type == "cuda":
            summary.update(cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                           cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                           gpu=torch.cuda.get_device_name(device))
        write_json(output_dir / "summary.json", summary)

    def save_training_checkpoint(metrics):
        payload = {
            "format": model.checkpoint_format, "model_config": model.model_config, "model": model.state_dict(),
            "experiment_config": cfg, "mode": mode, "epoch": completed_epoch, "updates": updates,
            "failed_updates": failed_updates, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict(),
            "rng": capture_rng(noise_generator, loader_generator, device), "metrics": metrics,
            "output_metrics": report, "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
            "selection_mode": metadata["selection_mode"], "best_loss": None, "best_epoch": 0,
            "continuation": continuation,
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if overfit and updates in fit_cfg.get("save_steps", []):
            save_checkpoint(output_dir / f"update_{updates:06d}.pt", payload)
        if not overfit and completed_epoch in cfg["train"].get("save_epochs", [1, 10, 20, 40, 60]):
            save_checkpoint(output_dir / f"epoch_{completed_epoch:03d}.pt", payload)

    def take_update(batch, epoch):
        nonlocal updates, failed_updates
        inputs, target, valid = [batch[key].to(device, non_blocking=True) for key in ("input", "target", "valid")]
        model.train()
        for attempt in range(4):
            optimizer.zero_grad(set_to_none=True)
            t_index = torch.randint(1, model.num_steps + 1, (len(inputs),), device=device, generator=noise_generator)
            # 扩散状态不得限制为 RGB 范围；有界化只属于最终显示/部署输出。
            xt = model.q_sample(target, inputs, t_index, generator=noise_generator)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                prediction = model.denoise(xt, inputs, t_index)
                loss, errors = reconstruction_loss(prediction, target, valid, cfg["loss"], return_errors=True)
            if not torch.isfinite(loss):
                failed_updates += 1
                raise FloatingPointError(f"non-finite loss at epoch={epoch} update={updates + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]),
                                                       error_if_nonfinite=False)
            if not torch.isfinite(grad_norm):
                failed_updates += 1
                if not scaler.is_enabled():
                    raise FloatingPointError(f"non-finite gradients at epoch={epoch} update={updates + 1}")
                scaler.step(optimizer)
                scaler.update()
                logger.warning("non-finite gradients; retrying batch attempt=%s failed_updates=%s", attempt + 1, failed_updates)
                continue
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                failed_updates += 1
                continue
            updates += 1
            row = {"epoch": epoch, "update": updates, "mode": mode, "loss": float(loss.detach()),
                   "train_errors": {key: float(value.detach().mean()) for key, value in errors.items()},
                   "learning_rate": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                   "t_min": int(t_index.min()), "t_max": int(t_index.max()), "samples": len(inputs),
                   "failed_updates": failed_updates, "seconds": time.perf_counter() - started}
            _append_json(output_dir / "metrics.jsonl", row)
            if updates == 1 or updates % 25 == 0:
                logger.info("epoch=%s update=%s/%s loss=%.6f RGB=%.6f grad=%.6f failed=%s",
                            epoch, updates, planned_updates, row["loss"], row["train_errors"]["rgb_l1"],
                            row["train_errors"]["gradient_l1"], failed_updates)
            return row
        raise FloatingPointError("four consecutive failed optimizer attempts for the same batch")

    try:
        update_summary()
        if not resume:
            report = monitor(model, fixed, output_dir, device=device, epoch=completed_epoch, updates=updates,
                             monitor_cfg=monitor_cfg, use_amp=use_amp, amp_dtype=amp_dtype)
            update_summary()
        if overfit:
            while updates < planned_updates:
                row = take_update(fixed, 0)
                if updates % monitor_every == 0 or updates == planned_updates or updates in fit_cfg.get("save_steps", []):
                    report = monitor(model, fixed, output_dir, device=device, epoch=0, updates=updates,
                                     monitor_cfg=monitor_cfg, use_amp=use_amp, amp_dtype=amp_dtype)
                    save_training_checkpoint(row)
                    update_summary()
                    logger.info("overfit full-sampling update=%s ratios=%s", updates, report["mean"]["ratios"])
            gate = overfit_gate(report, thresholds)
            summary.update(gate=gate, gate_passed=gate["passed"],
                           status="completed" if gate["passed"] else "failed_gate")
        else:
            for epoch in range(completed_epoch, epochs):
                train_set.set_epoch(epoch)
                loader_generator.manual_seed(seed + epoch)
                epoch_rows = [take_update(batch, epoch + 1) for batch in loader]
                if not epoch_rows:
                    raise RuntimeError("empty training epoch")
                completed_epoch = epoch + 1
                scheduler.step()
                if completed_epoch == 1 or completed_epoch % monitor_every == 0 or completed_epoch == epochs:
                    report = monitor(model, fixed, output_dir, device=device, epoch=completed_epoch, updates=updates,
                                     monitor_cfg=monitor_cfg, use_amp=use_amp, amp_dtype=amp_dtype)
                count_samples = sum(row["samples"] for row in epoch_rows)
                epoch_metrics = {
                    "epoch": completed_epoch, "updates": len(epoch_rows), "total_updates": updates,
                    "train_samples": count_samples, "failed_updates": failed_updates,
                    "train_loss": sum(row["loss"] * row["samples"] for row in epoch_rows) / count_samples,
                    "train_errors": {key: sum(row["train_errors"][key] * row["samples"] for row in epoch_rows) / count_samples
                                     for key in ERROR_KEYS}, "validation": None, "selection": None,
                }
                save_training_checkpoint(epoch_metrics)
                _append_json(output_dir / "epochs.jsonl", epoch_metrics)
                update_summary()
                logger.info("epoch=%s/%s completed updates=%s loss=%.6f; saved last.pt",
                            completed_epoch, epochs, updates, epoch_metrics["train_loss"])
            if updates != planned_updates:
                raise RuntimeError(f"effective update budget incomplete: {updates}/{planned_updates}")
            summary["status"] = "completed"
        update_summary()
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        update_summary()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", help="private output directory; defaults to config output_dir")
    parser.add_argument("--resume", help="same run's last.pt; formal runs resume at a completed epoch")
    parser.add_argument("--extend-overfit-from", help="preserve a parent overfit run and extend its budget in a fresh output directory")
    parser.add_argument("--overfit", action="store_true", help="fixed training pairs only; never starts formal training")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use sam2_env on the GPU server")
    summary = train(load_config(args.config), device=device, output_dir=args.output_dir,
                    resume=args.resume, overfit=args.overfit, extend_overfit_from=args.extend_overfit_from)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if summary["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
