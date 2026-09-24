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
    # 原 monitor 始终保留既有全局模糊配对；空间模糊由独立 monitor 展示。
    cfg["degradation"].pop("spatial_blur", None)
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


def capture_rng(noise_generator, loader_generator, device, short_chain_generator=None):
    state = {"noise_generator": noise_generator.get_state(), "loader_generator": loader_generator.get_state(),
             "torch_cpu": torch.get_rng_state(),
             "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
    if short_chain_generator is not None:
        state["short_chain_generator"] = short_chain_generator.get_state()
    return state


def restore_rng(state, noise_generator, loader_generator, device, short_chain_generator=None):
    if short_chain_generator is not None:
        if "short_chain_generator" not in state:
            raise ValueError("enabled short_chain requires its saved short_chain_generator RNG state")
        short_chain_generator.set_state(state["short_chain_generator"].cpu())
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


def _spatial_fork_changes(previous, current):
    """正式配方分叉仅允许新增空间模糊及其独立过程图，不放宽普通 resume。"""
    previous, current = copy.deepcopy(previous), copy.deepcopy(current)
    changes = {}
    previous_output, current_output = previous.pop("output_dir", None), current.pop("output_dir", None)
    if previous_output != current_output:
        changes["output_dir"] = {"before": previous_output, "after": current_output}
    if "spatial_blur" in previous.get("degradation", {}):
        raise ValueError("spatial fork requires a parent without spatial_blur")
    spatial = current.get("degradation", {}).pop("spatial_blur", None)
    if not isinstance(spatial, dict) or spatial.get("enabled") is not True:
        raise ValueError("spatial fork requires newly enabled degradation.spatial_blur")
    changes["degradation.spatial_blur"] = {"before": None, "after": spatial}
    if "spatial_blur" in previous.get("monitor", {}):
        raise ValueError("spatial fork requires a parent without monitor.spatial_blur")
    spatial_monitor = current.get("monitor", {}).pop("spatial_blur", None)
    if spatial_monitor is not None:
        if not isinstance(spatial_monitor, dict) or spatial_monitor.get("enabled") is not True:
            raise ValueError("spatial fork monitor must be newly enabled")
        changes["monitor.spatial_blur"] = {"before": None, "after": spatial_monitor}
    # 不使用 _same_recipe：此入口连 num_workers 也不能同时改变。
    if previous != current:
        raise ValueError("spatial fork config differs outside the spatial_blur allowlist")
    return changes


CROP_STAT_KEYS = ("sigma_crop_mean_square", "sigma_crop_weak_fraction",
                  "sigma_crop_strong_fraction", "sigma_crop_span", "sigma_crop_coexist")


def crop_blur_statistics(batch):
    """统计真实裁块，不以整图 sigma 或理论概率代替模型看到的覆盖。"""
    if not all(key in batch for key in ("endpoint_blur_applied", *CROP_STAT_KEYS)):
        return None
    groups = {}
    for index, profile in enumerate(batch["profile"]):
        group = ("identity" if profile == "identity" else
                 "endpoint" if bool(batch["endpoint_blur_applied"][index]) else
                 "oldspatial" if bool(batch["spatial_blur_applied"][index]) else
                 "uniform" if profile in ("blur", "mixed") else "illumination")
        row = groups.setdefault(group, {"samples": 0, **{key + "_sum": 0.0 for key in CROP_STAT_KEYS}})
        row["samples"] += 1
        for key in CROP_STAT_KEYS:
            row[key + "_sum"] += float(batch[key][index])
    return merge_crop_blur_statistics([groups])


def merge_crop_blur_statistics(groups_list):
    merged = {}
    for groups in groups_list:
        for name, source in groups.items():
            target = merged.setdefault(name, {"samples": 0, **{key + "_sum": 0.0 for key in CROP_STAT_KEYS}})
            target["samples"] += source["samples"]
            for key in CROP_STAT_KEYS:
                target[key + "_sum"] += source[key + "_sum"]
    for target in merged.values():
        for key in CROP_STAT_KEYS:
            target[key + "_mean"] = target[key + "_sum"] / target["samples"]
    return merged


def _append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sample_timesteps(count, num_steps, *, device, generator, policy="uniform"):
    """每张图独立抽时间步；保留旧抽样路径，起点强化只改变整数映射。"""
    if policy == "uniform":
        return torch.randint(1, num_steps + 1, (count,), device=device, generator=generator)
    if policy == "terminal_half":
        # 一半整数映射到T，另一半一一对应1..T-1；不是在均匀抽样上再叠加50%。
        draw = torch.randint(0, 2 * (num_steps - 1), (count,), device=device, generator=generator)
        return torch.where(draw < num_steps - 1, draw + 1, num_steps)
    raise ValueError(f"unknown timestep_sampling policy: {policy}")


def parse_short_chain_config(train_cfg):
    """短链是独立训练选项；缺省关闭，不改变旧随机流与 checkpoint 格式。"""
    value = train_cfg.get("short_chain")
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {"enabled", "probability", "extra_steps", "loss_weight"}:
        raise ValueError("short_chain requires only enabled, probability, extra_steps and loss_weight")
    cfg = {"enabled": False, "probability": 0.25, "extra_steps": 2, "loss_weight": 0.25, **value}
    if type(cfg["enabled"]) is not bool:
        raise ValueError("short_chain.enabled must be boolean")
    if type(cfg["extra_steps"]) is not int or cfg["extra_steps"] not in (1, 2):
        raise ValueError("short_chain.extra_steps must be 1 or 2")
    for key in ("probability", "loss_weight"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f"short_chain.{key} must be a finite number")
    if not 0 <= cfg["probability"] <= 1 or cfg["loss_weight"] < 0:
        raise ValueError("short_chain requires probability in [0,1] and nonnegative loss_weight")
    return cfg


def short_chain_loss(model, state, prediction, condition, target, valid, t_index, loss_cfg, *,
                     generator, extra_steps, loss_weight, backward, use_amp=False,
                     amp_dtype=torch.bfloat16):
    """沿模型自己的后验走短链；逐步反传，跨步断梯度，按有效 step-sample 平均。

    调用方先对主损失反传。backward 接收已加权的当前辅助标量；返回的损失仅供日志。
    condition 始终是同一退化输入，桥状态和未截断的清晰估计均不 clamp。
    """
    eligible = t_index > 1
    step_samples = int((t_index - 1).clamp(min=0, max=extra_steps).sum())
    result = {"aux_loss": 0.0, "weighted_aux_loss": 0.0, "samples": int(eligible.sum()),
              "step_samples": step_samples, "network_calls": 0}
    if step_samples == 0:
        return result
    state, prediction = state.detach().float(), prediction.detach().float()
    for _ in range(extra_steps):
        active = t_index > 1
        if not bool(active.any()):
            break
        state, prediction, condition, target, valid, t_index = [
            value[active] for value in (state, prediction, condition, target, valid, t_index)]
        # 精确复用部署后验；无梯度且关闭 autocast，避免低精度桥累积误差。
        with torch.no_grad(), torch.autocast(device_type=state.device.type, enabled=False):
            mean, variance = model.posterior_mean_variance(state.float(), prediction.float(), t_index)
            noise = torch.randn(mean.shape, device=mean.device, dtype=torch.float32, generator=generator)
            state = (mean.float() + variance.float().sqrt() * noise).detach()
        t_index = t_index - 1
        with torch.autocast(device_type=state.device.type, enabled=use_amp, dtype=amp_dtype):
            prediction = model.denoise(state, condition, t_index)
            step_loss = reconstruction_loss(prediction, target, valid, loss_cfg)
        if not bool(torch.isfinite(step_loss)):
            raise FloatingPointError("non-finite short_chain auxiliary loss")
        contribution = step_loss * (len(t_index) / step_samples)
        backward(contribution * loss_weight)
        result["aux_loss"] += float(contribution.detach())
        result["network_calls"] += 1
        prediction = prediction.detach()
    result["weighted_aux_loss"] = result["aux_loss"] * loss_weight
    return result


def train(config, *, device, output_dir=None, resume=None, overfit=False, extend_overfit_from=None,
          stop_after_epoch=None, fork_spatial_from=None):
    cfg = copy.deepcopy(config["rgb_restoration"])
    if cfg.get("split_policy") != "all_train":
        raise ValueError("diffusion experiment requires split_policy=all_train; no held-out validation")
    if cfg["model"].get("architecture") != "pixel_diffusion_nafnet":
        raise ValueError("this trainer requires architecture=pixel_diffusion_nafnet")
    timestep_policy = cfg["train"].get("timestep_sampling", "uniform")
    if timestep_policy not in ("uniform", "terminal_half"):
        raise ValueError(f"unknown timestep_sampling policy: {timestep_policy}")
    short_cfg = parse_short_chain_config(cfg["train"])
    epochs = int(cfg["train"]["epochs"])
    if stop_after_epoch is not None:
        if overfit:
            raise ValueError("stop_after_epoch is only supported for formal all_train runs")
        if type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= epochs:
            raise ValueError(f"stop_after_epoch must be an integer in [1, {epochs}]")
    # 分段停止只控制本次循环终点；不缩短学习率日程，不改变后续续训配方。
    run_until_epoch = epochs if stop_after_epoch is None else stop_after_epoch
    mode = "overfit_engineering" if overfit else "all_train"
    output_dir = Path(output_dir or project_path(config, cfg["output_dir"] + ("_overfit" if overfit else ""))).resolve()
    if fork_spatial_from:
        if resume or overfit or extend_overfit_from:
            raise ValueError("spatial fork requires formal training and cannot use resume, overfit or overfit extension")
        if (Path(fork_spatial_from).resolve().parent == output_dir
                or (output_dir.exists() and any(output_dir.iterdir()))):
            raise ValueError("spatial fork requires a fresh output directory distinct from its parent")
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
    timestep_sampling = {"policy": timestep_policy, "num_steps": model.num_steps,
                         "terminal_probability": 0.5 if timestep_policy == "terminal_half" else 1 / model.num_steps}
    learning_rate = float(fit_cfg.get("learning_rate", 2e-4) if overfit else cfg["train"]["learning_rate"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg["train"]["weight_decay"])
    scheduler = None if overfit else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(cfg["train"]["min_learning_rate"]))
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16, init_scale=1024)
    noise_generator = torch.Generator(device=device).manual_seed(seed + 271828)
    loader_generator = torch.Generator().manual_seed(seed)
    short_chain_generator = (torch.Generator(device=device).manual_seed(seed + 1618033)
                             if short_cfg["enabled"] else None)
    checkpoint = None
    fork_changes = None
    checkpoint_path = resume or extend_overfit_from or fork_spatial_from
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != model.checkpoint_format or checkpoint.get("mode") != mode:
            raise ValueError("resume requires matching diffusion format and training mode")
        if fork_spatial_from:
            fork_changes = _spatial_fork_changes(checkpoint["experiment_config"], cfg)
        else:
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
    if checkpoint_path:
        fixed_dir = Path(checkpoint_path).resolve().parent
        fixed = torch.load(fixed_dir / "fixed_samples.pt", map_location="cpu", weights_only=True)
        fixed_manifest = json.loads((fixed_dir / "fixed_samples_manifest.json").read_text(encoding="utf-8"))
        if len(fixed["source"]) != count or fixed["source"] != fixed_manifest["sources"]:
            raise ValueError("saved fixed sample set differs from run configuration")
    else:
        fixed, fixed_manifest = fixed_training_samples(config, count)
    loader = None if overfit else DataLoader(
        train_set, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]), pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker, generator=loader_generator)
    planned_updates = int(fit_cfg.get("steps", 800)) if overfit else epochs * len(loader)
    segment_target_updates = planned_updates if overfit else run_until_epoch * len(loader)
    monitor_every = int(fit_cfg.get("monitor_every", 100) if overfit else monitor_cfg.get("every_epochs", 5))
    if planned_updates < 1 or monitor_every < 1:
        raise ValueError("positive training budget and monitor interval required")
    completed_epoch = int(checkpoint["epoch"]) if checkpoint else 0
    updates = int(checkpoint["updates"]) if checkpoint else 0
    failed_updates = int(checkpoint.get("failed_updates", 0)) if checkpoint else 0
    if fork_spatial_from:
        parent_metrics, parent_schedule = checkpoint.get("metrics", {}), checkpoint.get("scheduler", {})
        if (completed_epoch < 1 or updates != completed_epoch * len(loader)
                or parent_metrics.get("epoch") != completed_epoch
                or parent_metrics.get("updates") != len(loader)
                or parent_metrics.get("train_samples") != len(train_set)
                or parent_schedule.get("last_epoch") != completed_epoch
                or parent_schedule.get("T_max") != epochs):
            raise ValueError("spatial fork requires a completed formal epoch with its original LR schedule")
    if updates >= planned_updates or (not overfit and completed_epoch >= epochs):
        raise ValueError("run already completed its configured update budget")
    if not overfit and completed_epoch >= run_until_epoch:
        raise ValueError("stop_after_epoch must exceed the checkpoint's completed epoch")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        save_checkpoint(output_dir / "fixed_samples.pt", fixed)
        write_json(output_dir / "fixed_samples_manifest.json", fixed_manifest)
    spatial_fixed, spatial_manifest, monitor_spatial = None, None, None
    if monitor_cfg.get("spatial_blur", {}).get("enabled", False):
        from tools.rgb_spatial_monitor import build_spatial_fixed_samples, monitor_spatial
        if resume:
            spatial_fixed = torch.load(output_dir / "spatial_fixed_samples.pt", map_location="cpu", weights_only=True)
            spatial_manifest = json.loads((output_dir / "spatial_fixed_samples_manifest.json").read_text(encoding="utf-8"))
        else:
            spatial_fixed, spatial_manifest = build_spatial_fixed_samples(config, train_set, fixed_manifest)
        spatial_count = int(monitor_cfg["spatial_blur"].get("samples", count))
        if (len(spatial_fixed["source"]) != spatial_count
                or spatial_fixed["source"] != spatial_manifest["sources"]):
            raise ValueError("saved spatial fixed sample set differs from run configuration")
        if not resume:
            save_checkpoint(output_dir / "spatial_fixed_samples.pt", spatial_fixed)
            write_json(output_dir / "spatial_fixed_samples_manifest.json", spatial_manifest)
    endpoint_fixed, endpoint_manifest, monitor_endpoint = None, None, None
    if monitor_cfg.get("endpoint_blur", {}).get("enabled", False):
        from tools.rgb_endpoint_monitor import build_endpoint_fixed_samples, monitor_endpoint
        if resume:
            endpoint_fixed = torch.load(output_dir / "endpoint_fixed_samples.pt", map_location="cpu", weights_only=True)
            endpoint_manifest = json.loads((output_dir / "endpoint_fixed_samples_manifest.json").read_text(encoding="utf-8"))
        else:
            endpoint_fixed, endpoint_manifest = build_endpoint_fixed_samples(config, train_set, fixed_manifest)
        if (len(endpoint_fixed["source"]) != int(monitor_cfg["endpoint_blur"].get("samples", count))
                or endpoint_fixed["source"] != endpoint_manifest["sources"]):
            raise ValueError("saved endpoint fixed sample set differs from run configuration")
        if not resume:
            save_checkpoint(output_dir / "endpoint_fixed_samples.pt", endpoint_fixed)
            write_json(output_dir / "endpoint_fixed_samples_manifest.json", endpoint_manifest)
    real_fixed, real_manifest, monitor_real = None, None, None
    if monitor_cfg.get("real", {}).get("enabled", False):
        from tools.rgb_endpoint_monitor import build_real_fixed_samples, monitor_real
        if resume:
            real_fixed = torch.load(output_dir / "real_fixed_samples.pt", map_location="cpu", weights_only=True)
            real_manifest = json.loads((output_dir / "real_fixed_samples_manifest.json").read_text(encoding="utf-8"))
        else:
            real_fixed, real_manifest = build_real_fixed_samples(config)
        if (len(real_fixed["source"]) != len(monitor_cfg["real"]["images"])
                or real_fixed["source"] != real_manifest["sources"]):
            raise ValueError("saved real fixed sample set differs from run configuration")
        if not resume:
            save_checkpoint(output_dir / "real_fixed_samples.pt", real_fixed)
            write_json(output_dir / "real_fixed_samples_manifest.json", real_manifest)
    if checkpoint:
        restore_rng(checkpoint["rng"], noise_generator, loader_generator, device, short_chain_generator)
    short_counter_keys = ("selected_batches", "selected_samples", "eligible_samples", "step_samples", "network_calls")
    short_totals = {key: 0 for key in short_counter_keys}
    short_totals.update(samples=0, base_loss_sum=0.0, aux_loss_sum=0.0, weighted_aux_loss_sum=0.0)
    if checkpoint and short_cfg["enabled"]:
        if "short_chain_totals" not in checkpoint:
            raise ValueError("enabled short_chain requires saved short_chain_totals")
        short_totals = copy.deepcopy(checkpoint["short_chain_totals"])
    continuation = checkpoint.get("continuation") if checkpoint else None
    if extend_overfit_from:
        continuation = {"parent_checkpoint": str(Path(extend_overfit_from).resolve()),
                        "parent_updates": updates,
                        "parent_planned_updates": checkpoint["experiment_config"]["overfit"]["steps"],
                        "preserved": ["model", "optimizer", "scaler", "rng", "fixed_samples"]}
    if fork_spatial_from:
        continuation = {"kind": "formal_spatial_blur_fork",
                        "parent_checkpoint": str(Path(fork_spatial_from).resolve()),
                        "parent_epoch": completed_epoch, "parent_updates": updates,
                        "parent_planned_updates": planned_updates,
                        "parent_elapsed_seconds": float(checkpoint.get("elapsed_seconds", 0)),
                        "recipe_changes": fork_changes,
                        "preserved": ["model", "optimizer", "scheduler", "scaler", "rng", "epoch",
                                      "updates", "elapsed_seconds", "fixed_samples"],
                        "new_logs_begin_after_epoch": completed_epoch}
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
        "initial_stop_after_epoch": stop_after_epoch,
        "initial_segment_target_updates": segment_target_updates,
        "fixed_samples": fixed_manifest, "device": str(device), "torch": str(torch.__version__),
        "amp_dtype": str(amp_dtype) if use_amp else None,
        "continuation": continuation,
        "timestep_sampling": timestep_sampling,
    }
    if spatial_manifest is not None:
        metadata["spatial_fixed_samples"] = spatial_manifest
    if endpoint_manifest is not None:
        metadata["endpoint_fixed_samples"] = endpoint_manifest
    if real_manifest is not None:
        metadata["real_fixed_samples"] = real_manifest
    if short_cfg["enabled"]:
        metadata["short_chain_training"] = {
            **short_cfg, "rng_seed": seed + 1618033,
            "aux_normalization": "mean_over_all_valid_step_samples_within_selected_batch",
            "log_loss_mean": "primary_sample_weighted_mean_including_zero_aux_for_unselected_batches",
            "bridge": "original_posterior_fp32_detached_unclamped",
        }
    if not resume:
        write_json(output_dir / "run_config.json", metadata)
    logger.info("mode=%s parameters=%s pool=%s train_sources=%s planned_updates=%s segment_target_updates=%s amp=%s",
                mode, parameters, len(train_set), len(metadata["train_sources"]), planned_updates,
                segment_target_updates, metadata["amp_dtype"])
    logger.info("timestep_sampling=%s", timestep_sampling)
    prior_elapsed = float(checkpoint.get("elapsed_seconds", 0)) if checkpoint else 0.0
    started = time.perf_counter()
    report = checkpoint.get("output_metrics") if checkpoint else None
    summary = {"status": "running", "mode": mode, "gate_passed": None, "parameters": parameters,
               "planned_updates": planned_updates, "updates": updates, "total_updates": updates,
               "planned_epochs": None if overfit else epochs,
               "run_until_epoch": None if overfit else run_until_epoch,
               "segment_target_updates": segment_target_updates, "training_complete": False,
               "failed_updates": failed_updates, "selection_mode": metadata["selection_mode"],
               "intended_checkpoint": "last.pt", "validation": None, "best_checkpoint_available": False,
               "automatic_formal_training": False}
    summary["timestep_sampling"] = timestep_sampling
    if continuation is not None:
        summary["continuation"] = continuation

    def run_monitors(epoch, updates):
        original_report = monitor(model, fixed, output_dir, device=device, epoch=epoch, updates=updates,
                                  monitor_cfg=monitor_cfg, use_amp=use_amp, amp_dtype=amp_dtype)
        if spatial_fixed is not None:
            monitor_spatial(model, spatial_fixed, output_dir, device=device, epoch=epoch, updates=updates,
                            monitor_cfg=monitor_cfg, use_amp=use_amp, amp_dtype=amp_dtype)
        if endpoint_fixed is not None:
            monitor_endpoint(model, endpoint_fixed, output_dir, device=device, epoch=epoch, updates=updates,
                             monitor_cfg=monitor_cfg, use_amp=use_amp, amp_dtype=amp_dtype)
        if real_fixed is not None:
            monitor_real(model, real_fixed, output_dir, device=device, epoch=epoch, updates=updates,
                         monitor_cfg=monitor_cfg, use_amp=use_amp, amp_dtype=amp_dtype)
        return original_report

    def update_summary():
        summary.update(epoch=completed_epoch, updates=updates, total_updates=updates,
                       failed_updates=failed_updates, elapsed_seconds=prior_elapsed + time.perf_counter() - started,
                       output_metrics=report)
        if short_cfg["enabled"]:
            summary["short_chain"] = {key: short_totals[key] for key in short_counter_keys}
            for key in ("base_loss", "aux_loss", "weighted_aux_loss"):
                summary[key] = short_totals[key + "_sum"] / max(short_totals["samples"], 1)
            summary["train_loss"] = summary["base_loss"] + summary["weighted_aux_loss"]
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
            "rng": capture_rng(noise_generator, loader_generator, device, short_chain_generator), "metrics": metrics,
            "output_metrics": report, "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
            "selection_mode": metadata["selection_mode"], "best_loss": None, "best_epoch": 0,
            "continuation": continuation,
        }
        if short_cfg["enabled"]:
            payload["short_chain_totals"] = copy.deepcopy(short_totals)
        save_checkpoint(output_dir / "last.pt", payload)
        if overfit and updates in fit_cfg.get("save_steps", []):
            save_checkpoint(output_dir / f"update_{updates:06d}.pt", payload)
        if not overfit and (completed_epoch in cfg["train"].get("save_epochs", [1, 10, 20, 40, 60])
                            or (completed_epoch == 0 and cfg["train"].get("save_initial_checkpoint", False))):
            save_checkpoint(output_dir / f"epoch_{completed_epoch:03d}.pt", payload)

    def take_update(batch, epoch):
        nonlocal updates, failed_updates
        inputs, target, valid = [batch[key].to(device, non_blocking=True) for key in ("input", "target", "valid")]
        model.train()
        retry_rng = (capture_rng(noise_generator, loader_generator, device, short_chain_generator)
                     if short_cfg["enabled"] else None)
        retry_python = random.getstate() if short_cfg["enabled"] else None
        retry_numpy = np.random.get_state() if short_cfg["enabled"] else None
        for attempt in range(4):
            if attempt and retry_rng is not None:
                # 仅 scaler 降尺度；同一失败批次重放原时间、主噪声及整条辅助桥随机数。
                restore_rng(retry_rng, noise_generator, loader_generator, device, short_chain_generator)
                random.setstate(retry_python)
                np.random.set_state(retry_numpy)
            optimizer.zero_grad(set_to_none=True)
            t_index = sample_timesteps(len(inputs), model.num_steps, device=device,
                                       generator=noise_generator, policy=timestep_policy)
            # 扩散状态不得限制为 RGB 范围；有界化只属于最终显示/部署输出。
            xt = model.q_sample(target, inputs, t_index, generator=noise_generator)
            with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                prediction = model.denoise(xt, inputs, t_index)
                loss, errors = reconstruction_loss(prediction, target, valid, cfg["loss"], return_errors=True)
            if not torch.isfinite(loss):
                failed_updates += 1
                raise FloatingPointError(f"non-finite loss at epoch={epoch} update={updates + 1}")
            scaler.scale(loss).backward()
            if short_cfg["enabled"]:
                selected = bool(torch.rand((), device=device, generator=short_chain_generator)
                                < short_cfg["probability"])
                short_result = {"aux_loss": 0.0, "weighted_aux_loss": 0.0, "samples": 0,
                                "step_samples": 0, "network_calls": 0}
                if selected:
                    try:
                        short_result = short_chain_loss(
                            model, xt, prediction, inputs, target, valid, t_index, cfg["loss"],
                            generator=short_chain_generator, extra_steps=short_cfg["extra_steps"],
                            loss_weight=short_cfg["loss_weight"], backward=lambda value: scaler.scale(value).backward(),
                            use_amp=use_amp, amp_dtype=amp_dtype)
                    except FloatingPointError:
                        failed_updates += 1
                        raise
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
                   "t_histogram": torch.bincount(t_index, minlength=model.num_steps + 1)[1:].tolist(),
                   "failed_updates": failed_updates, "seconds": time.perf_counter() - started}
            if short_cfg["enabled"]:
                row.update(base_loss=float(loss.detach()), aux_loss=short_result["aux_loss"],
                           weighted_aux_loss=short_result["weighted_aux_loss"])
                row["loss"] = row["base_loss"] + row["weighted_aux_loss"]
                row["short_chain"] = {"selected_batches": int(selected),
                                      "selected_samples": len(inputs) if selected else 0,
                                      "eligible_samples": short_result["samples"],
                                      "step_samples": short_result["step_samples"],
                                      "network_calls": short_result["network_calls"]}
                short_totals["samples"] += len(inputs)
                for key in short_counter_keys:
                    short_totals[key] += row["short_chain"][key]
                for key in ("base_loss", "aux_loss", "weighted_aux_loss"):
                    short_totals[key + "_sum"] += row[key] * len(inputs)
            if "source" in batch:
                row["sources"] = list(batch["source"])
            if "profile" in batch:
                row["profiles"] = list(batch["profile"])
            crop_statistics = crop_blur_statistics(batch)
            if crop_statistics is not None:
                row["crop_blur"] = crop_statistics
            if "spatial_blur_applied" in batch:
                blur_mask = torch.tensor([profile in ("blur", "mixed") for profile in batch["profile"]])
                blur_count = int(blur_mask.sum())
                sigma_sum = float(batch["base_sigma"][blur_mask].double().sum())
                row["spatial_blur"] = {
                    "applied_samples": int(batch["spatial_blur_applied"].sum()),
                    "blur_samples": blur_count, "base_sigma_sum": sigma_sum,
                    "base_sigma_mean": sigma_sum / blur_count if blur_count else None,
                }
            _append_json(output_dir / "metrics.jsonl", row)
            if updates == 1 or updates % 25 == 0:
                logger.info("epoch=%s update=%s/%s loss=%.6f RGB=%.6f grad=%.6f failed=%s",
                            epoch, updates, planned_updates, row["loss"], row["train_errors"]["rgb_l1"],
                            row["train_errors"]["gradient_l1"], failed_updates)
                if "spatial_blur" in row:
                    logger.info("spatial_blur batch=%s; blur_samples includes blur and mixed profiles",
                                row["spatial_blur"])
                if short_cfg["enabled"]:
                    logger.info("short_chain base=%.6f aux=%.6f weighted_aux=%.6f counts=%s",
                                row["base_loss"], row["aux_loss"], row["weighted_aux_loss"], row["short_chain"])
            return row
        raise FloatingPointError("four consecutive failed optimizer attempts for the same batch")

    try:
        update_summary()
        if fork_spatial_from:
            # 新输出先保存可续训的完整父状态；只新增监控，不复制父运行的训练日志。
            save_training_checkpoint(checkpoint["metrics"])
        if not resume:
            report = run_monitors(completed_epoch, updates)
            update_summary()
        if not checkpoint and not overfit and cfg["train"].get("save_initial_checkpoint", False):
            save_training_checkpoint({"epoch": 0, "updates": 0, "total_updates": 0,
                                      "train_samples": 0, "initial_scratch_checkpoint": True})
        if overfit:
            while updates < planned_updates:
                row = take_update(fixed, 0)
                if updates % monitor_every == 0 or updates == planned_updates or updates in fit_cfg.get("save_steps", []):
                    report = run_monitors(0, updates)
                    save_training_checkpoint(row)
                    update_summary()
                    logger.info("overfit full-sampling update=%s ratios=%s", updates, report["mean"]["ratios"])
            gate = overfit_gate(report, thresholds)
            summary.update(gate=gate, gate_passed=gate["passed"],
                           status="completed" if gate["passed"] else "failed_gate", training_complete=True)
        else:
            for epoch in range(completed_epoch, run_until_epoch):
                train_set.set_epoch(epoch)
                loader_generator.manual_seed(seed + epoch)
                epoch_rows = [take_update(batch, epoch + 1) for batch in loader]
                if not epoch_rows:
                    raise RuntimeError("empty training epoch")
                completed_epoch = epoch + 1
                scheduler.step()
                if completed_epoch == 1 or completed_epoch % monitor_every == 0 or completed_epoch == run_until_epoch:
                    report = run_monitors(completed_epoch, updates)
                count_samples = sum(row["samples"] for row in epoch_rows)
                epoch_metrics = {
                    "epoch": completed_epoch, "updates": len(epoch_rows), "total_updates": updates,
                    "train_samples": count_samples, "failed_updates": failed_updates,
                    "t_histogram": [sum(row["t_histogram"][step] for row in epoch_rows)
                                    for step in range(model.num_steps)],
                    "train_loss": sum(row["loss"] * row["samples"] for row in epoch_rows) / count_samples,
                    "train_errors": {key: sum(row["train_errors"][key] * row["samples"] for row in epoch_rows) / count_samples
                                     for key in ERROR_KEYS}, "validation": None, "selection": None,
                }
                if short_cfg["enabled"]:
                    for key in ("base_loss", "aux_loss", "weighted_aux_loss"):
                        epoch_metrics[key] = sum(row[key] * row["samples"] for row in epoch_rows) / count_samples
                    epoch_metrics["short_chain"] = {
                        key: sum(row["short_chain"][key] for row in epoch_rows) for key in short_counter_keys}
                spatial_rows = [row["spatial_blur"] for row in epoch_rows if "spatial_blur" in row]
                crop_rows = [row["crop_blur"] for row in epoch_rows if "crop_blur" in row]
                if crop_rows:
                    epoch_metrics["crop_blur"] = merge_crop_blur_statistics(crop_rows)
                if spatial_rows:
                    blur_count = sum(row["blur_samples"] for row in spatial_rows)
                    sigma_sum = sum(row["base_sigma_sum"] for row in spatial_rows)
                    epoch_metrics["spatial_blur"] = {
                        "applied_samples": sum(row["applied_samples"] for row in spatial_rows),
                        "blur_samples": blur_count, "base_sigma_sum": sigma_sum,
                        "base_sigma_mean": sigma_sum / blur_count if blur_count else None,
                    }
                save_training_checkpoint(epoch_metrics)
                _append_json(output_dir / "epochs.jsonl", epoch_metrics)
                update_summary()
                logger.info("epoch=%s/%s completed updates=%s loss=%.6f; saved last.pt",
                            completed_epoch, epochs, updates, epoch_metrics["train_loss"])
            if updates != segment_target_updates:
                raise RuntimeError(f"effective segment update budget incomplete: {updates}/{segment_target_updates}")
            summary.update(status="completed" if completed_epoch == epochs else "segment_completed",
                           training_complete=completed_epoch == epochs)
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
    parser.add_argument("--fork-spatial-from", help="fork a completed formal epoch with only spatial blur newly enabled")
    parser.add_argument("--overfit", action="store_true", help="fixed training pairs only; never starts formal training")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="stop a formal run after this cumulative epoch; preserve the full config LR schedule")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use sam2_env on the GPU server")
    summary = train(load_config(args.config), device=device, output_dir=args.output_dir,
                    resume=args.resume, overfit=args.overfit, extend_overfit_from=args.extend_overfit_from,
                    stop_after_epoch=args.stop_after_epoch, fork_spatial_from=args.fork_spatial_from)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if summary["status"] not in ("completed", "segment_completed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
