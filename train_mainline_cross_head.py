# -*- coding: utf-8 -*-
"""顺序训练主线 self/cross 等预算对照；仅保留 best/last 小型修正权重。"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from data.direct_dual_head_dataset import DirectDualHeadDataset
from models.cross_head_refinement import (
    CROSS_HEAD_FORMAT, CrossHeadRefiner, RefinedMainline,
    build_refined_from_checkpoint, configure_mainline_precision, file_digest,
)
from models.fused_deployment import load_fused_deployment_model
from tools.evaluate_mainline_cross_head import deployment_contract, predict_instances
from train_direct_semantic_affinity import (
    build_semantic_criterion, compute_affinity_loss, compute_semantic_loss,
    move_batch, set_seed,
)
from utils.config import load_config, project_path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_datasets(config):
    cfg = config["cross_head"]
    split = json.loads(Path(project_path(config, cfg["split_file"])).read_text(encoding="utf-8"))
    train_names, all_val = split["train"], split["val"]
    excluded = cfg.get("excluded_val_names", [])
    if set(train_names) & set(all_val) or not set(excluded).issubset(all_val):
        raise ValueError("invalid train/validation exclusions")
    if len(set(train_names)) != len(train_names) or len(set(all_val)) != len(all_val):
        raise ValueError("duplicate split names")
    val_names = [name for name in all_val if name not in excluded]
    if not train_names or not val_names:
        raise ValueError("empty train/validation split")
    common = dict(
        data_dir=project_path(config, cfg["data_dir"]), gt_dir=None,
        manual_target_dir=project_path(config, cfg["manual_target_dir"]),
        image_size=cfg["input_size"], affinity_grid=cfg["affinity_grid"],
        manual_target_boundary_dilation=2,
    )
    train = DirectDualHeadDataset(
        **common, sample_names=train_names, augment=True,
        augmentation=cfg["augmentation"], native_crop=cfg["native_crop"],
    )
    calibration = DirectDualHeadDataset(**common, sample_names=train_names)
    val = DirectDualHeadDataset(**common, sample_names=val_names)
    return train, calibration, val, {"train": train_names, "val": val_names, "excluded_val": excluded}


def epoch_loader(dataset, cfg, epoch):
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + epoch)
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=int(cfg["samples_per_epoch"]), generator=generator,
    )
    return DataLoader(
        dataset, sampler=sampler, batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]), pin_memory=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"]) + epoch),
        worker_init_fn=seed_worker,
    )


def task_losses(output, batch, criterion, cfg):
    semantic = compute_semantic_loss(criterion, output["semantic_logits"], batch)
    affinity, _ = compute_affinity_loss(output["affinity_logits"], batch, cfg["affinity_loss"], pseudo=False)
    return {"semantic": semantic, "affinity": affinity}


@torch.no_grad()
def evaluate_losses(model, dataset, criterion, cfg, device):
    model.eval()
    totals = {"semantic": 0.0, "affinity": 0.0}
    for raw in DataLoader(dataset, batch_size=1, num_workers=0):
        batch = move_batch(raw, device)
        losses = task_losses(model(batch["image"]), batch, criterion, cfg)
        for key, value in losses.items():
            totals[key] += float(value)
    return {key: value / len(dataset) for key, value in totals.items()}


def scaled_loss(losses, scales):
    return sum(losses[key] / scales[key] for key in ("semantic", "affinity"))


def checkpoint_payload(model, config, split, scales, epoch, val_losses, best_score):
    cfg = config["cross_head"]
    portable = copy.deepcopy(config)
    portable["paths"]["project_root"] = "auto"
    return {
        "format": CROSS_HEAD_FORMAT, "context_mode": model.refiner.context_mode,
        "architecture": model.refiner.architecture,
        "refiner_state_dict": {k: v.detach().cpu().clone() for k, v in model.refiner.state_dict().items()},
        "base_checkpoint": cfg["base_checkpoint"], "base_sha256": cfg["base_sha256"],
        "epoch": epoch, "validation_losses": val_losses, "best_validation_loss": best_score,
        "loss_scales": scales, "selection": "five-image scaled semantic + affinity validation loss",
        "split": split, "manual_target_dir": cfg["manual_target_dir"],
        "config": portable, "deployment_contract": deployment_contract(config),
        "parameter_summary": model.parameter_summary(),
    }


def zero_compatibility(base, models, config, output_dir, device):
    results = []
    contract = deployment_contract(config)
    for relative in config["cross_head"]["zero_check_images"]:
        path = project_path(config, relative)
        reference, classes, maps = predict_instances(
            base, path, contract, output_dir / "mainline", device, return_maps=True,
        )
        for mode, model in models.items():
            prediction, predicted_classes, candidate = predict_instances(
                model, path, contract, output_dir / mode, device, return_maps=True,
            )
            errors = {key: float((maps[key] - candidate[key]).abs().max()) for key in maps}
            equal_instances = np.array_equal(reference, prediction)
            equal_classes = classes == predicted_classes
            record = {"image": relative, "mode": mode, "max_errors": errors,
                      "instances_equal": equal_instances, "classes_equal": equal_classes}
            results.append(record)
            if any(errors.values()) or not equal_instances or not equal_classes:
                raise RuntimeError(f"zero residual changes deployed output: {record}")
    write_json(output_dir / "zero_compatibility.json", results)
    return results


def preflight(base, config, split, scales, val_losses, train, criterion, output_dir, device):
    cfg = config["cross_head"]
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    for mode in ("self", "cross"):
        set_seed(cfg["seed"])
        model = RefinedMainline(base, CrossHeadRefiner(**cfg["architecture"], context_mode=mode).to(device))
        models[mode] = model
    for key, value in models["self"].refiner.state_dict().items():
        if not torch.equal(value, models["cross"].refiner.state_dict()[key]):
            raise RuntimeError("A/B initial refiner tensors differ")
    if not models["cross"].parameter_summary()["constraint_passed"]:
        raise ValueError("model exceeds the parameter budget")
    initial = checkpoint_payload(models["cross"], config, split, scales, 0, val_losses, scaled_loss(val_losses, scales))
    torch.save(initial, output_dir / "zero_refiner.pth")
    restored = torch.load(output_dir / "zero_refiner.pth", map_location="cpu", weights_only=False)
    models["cross_reloaded"] = build_refined_from_checkpoint(restored, base, cfg["base_sha256"], device)
    compatibility = zero_compatibility(base, models, config, output_dir, device)
    # 连续三步越过零输出层的首步梯度阻断，检查每个可训练张量确实更新。
    base_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    batch = move_batch(next(iter(epoch_loader(train, {**cfg, "num_workers": 0}, 1))), device)
    updates = {}
    for mode in ("self", "cross"):
        model = models[mode].train()
        initial_state = {key: value.detach().clone() for key, value in model.refiner.named_parameters()}
        optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=cfg["learning_rate"], weight_decay=0)
        gradient_seen = set()
        started = time.perf_counter()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = scaled_loss(task_losses(model(batch["image"]), batch, criterion, cfg), scales)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite preflight loss")
            loss.backward()
            for key, parameter in model.refiner.named_parameters():
                if parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0:
                    gradient_seen.add(key)
            torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        unchanged = [key for key, parameter in model.refiner.named_parameters() if torch.equal(parameter, initial_state[key])]
        missing_gradient = sorted(set(initial_state) - gradient_seen)
        if unchanged or missing_gradient:
            raise RuntimeError(f"inactive {mode} refiner tensors: {unchanged}; no gradient: {missing_gradient}")
        updates[mode] = {"updated_tensors": len(initial_state), "nonzero_gradient_tensors": len(gradient_seen),
                         "seconds_per_step": (time.perf_counter() - started) / 3}
    for key, value in base.state_dict().items():
        if not torch.equal(value.cpu(), base_state[key]):
            raise RuntimeError(f"frozen base changed: {key}")
    if any(p.grad is not None or p.requires_grad for p in base.parameters()) or base.training:
        raise RuntimeError("base must remain frozen and in eval mode")
    audit = {"status": "passed", "zero_compatibility": compatibility, "training_updates": updates,
             "base_state_unchanged": True, "parameter_summary": models["cross"].parameter_summary()}
    write_json(output_dir / "preflight.json", audit)
    print(json.dumps({"preflight": audit}, ensure_ascii=False), flush=True)


def train_arm(base, mode, config, split, scales, initial_val, train, val, criterion, output_dir, device):
    cfg = config["cross_head"]
    set_seed(cfg["seed"])
    model = RefinedMainline(base, CrossHeadRefiner(**cfg["architecture"], context_mode=mode).to(device))
    arm_dir = output_dir / mode
    arm_dir.mkdir()
    best = float(scaled_loss(initial_val, scales))
    best_epoch = 0
    torch.save(checkpoint_payload(model, config, split, scales, 0, initial_val, best), arm_dir / "best_refiner.pth")
    optimizer = torch.optim.AdamW(model.refiner.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["minimum_learning_rate"])
    first_started = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        set_seed(cfg["seed"] + epoch)
        model.train()
        started = time.perf_counter()
        totals = {"semantic": 0.0, "affinity": 0.0}
        steps, samples, crops = 0, 0, 0
        lr = optimizer.param_groups[0]["lr"]
        for raw in epoch_loader(train, cfg, epoch):
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            losses = task_losses(model(batch["image"]), batch, criterion, cfg)
            loss = scaled_loss(losses, scales)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {mode} loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.refiner.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value)
            steps += 1
            samples += batch["image"].shape[0]
            crops += int(batch["is_native_crop"].sum())
        train_seconds = time.perf_counter() - started
        val_losses = evaluate_losses(model, val, criterion, cfg, device)
        score = float(scaled_loss(val_losses, scales))
        if not math.isfinite(score):
            raise RuntimeError("non-finite validation loss")
        improved = score < best
        if improved:
            best, best_epoch = score, epoch
        scheduler.step()
        payload = checkpoint_payload(model, config, split, scales, epoch, val_losses, best)
        if improved:
            torch.save(payload, arm_dir / "best_refiner.pth")
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
        payload["best_epoch"] = best_epoch
        torch.save(payload, arm_dir / "last_refiner.pth")
        record = {"mode": mode, "epoch": epoch, "steps": steps, "samples": samples, "native_crops": crops,
                  "learning_rate": lr, "train_losses": {k: v / steps for k, v in totals.items()},
                  "validation_losses": val_losses, "validation_total": score,
                  "best_epoch": best_epoch, "best_validation_loss": best, "train_seconds": train_seconds,
                  "epoch_seconds": time.perf_counter() - started}
        with (arm_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        write_json(output_dir / "status.json", {"status": "training", **record})
        print(json.dumps(record), flush=True)
    return {"mode": mode, "best_epoch": best_epoch, "best_validation_loss": best,
            "seconds": time.perf_counter() - first_started, "epochs": cfg["epochs"],
            "best_checkpoint": str(arm_dir / "best_refiner.pth"),
            "best_sha256": file_digest(arm_dir / "best_refiner.pth")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/mainline_cross_head_ab.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["cross_head"]
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if not torch.cuda.is_available():
        raise RuntimeError("real mainline preflight/training requires sam2_env CUDA")
    device = torch.device("cuda")
    # 冻结模型使用正式 FP32 推理，修正模块也使用 FP32，避免混合精度掩盖零兼容问题。
    configure_mainline_precision()
    torch.set_num_threads(4)
    set_seed(cfg["seed"])
    if cfg["input_size"] != 1024 or cfg["affinity_grid"] != 512:
        raise ValueError("cross-head v1 fixes the mainline 1024 input / 512 affinity grid")
    if cfg["affinity_loss"].get("manual_uncovered_as_boundary", True):
        raise ValueError("remaining unknown must be ignored for completed GT")
    output_dir = Path(project_path(config, cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run.json").exists():
        raise FileExistsError(f"existing run must be preserved: {output_dir}")
    base_path = project_path(config, cfg["base_checkpoint"])
    if file_digest(base_path) != cfg["base_sha256"]:
        raise ValueError("configured base SHA256 does not match the mainline")
    base, _ = load_fused_deployment_model(base_path, config, device)
    train, calibration, val, split = make_datasets(config)
    criterion = build_semantic_criterion({"direct_semantic_affinity": {"semantic_loss": cfg["semantic_loss"]}}, device)
    scales = evaluate_losses(base, calibration, criterion, cfg, device)
    if any(not math.isfinite(v) or v <= 0 for v in scales.values()):
        raise RuntimeError("invalid baseline loss calibration")
    initial_val = evaluate_losses(base, val, criterion, cfg, device)
    write_json(output_dir / "run.json", {
        "format": CROSS_HEAD_FORMAT, "config": config, "split": split, "loss_scales": scales,
        "initial_validation": initial_val, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(), "precision": "float32; matmul_tf32=false; cudnn_tf32=true",
        "sampling": "same seed + epoch, replacement sampler, seeded NumPy/Torch worker augmentation",
        "preflight_only": args.preflight_only,
    })
    write_json(output_dir / "status.json", {"status": "preflight"})
    try:
        preflight(base, config, split, scales, initial_val, train, criterion, output_dir / "preflight", device)
        if args.preflight_only:
            write_json(output_dir / "status.json", {"status": "preflight_complete"})
            return
        results = []
        for mode in ("self", "cross"):
            results.append(train_arm(base, mode, config, split, scales, initial_val, train, val, criterion, output_dir, device))
        write_json(output_dir / "status.json", {"status": "complete", "arms": results})
    except Exception as error:
        write_json(output_dir / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
