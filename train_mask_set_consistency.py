# -*- coding: utf-8 -*-
"""从损失最优学生启动独立短阶段；固定监督预算对照是否增加在线 EMA 一致性。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import torch
import yaml
from torch.utils.data import DataLoader

from data.mask_set_unlabeled import MaskSetUnlabeledDataset, build_unlabeled_pool, disjoint_pseudo_subset, image_digest
from models.lora import enable_trunk_gradient_checkpointing
from models.mask_set import configure_mask_set_phase, load_mask_set_model, make_mask_set_checkpoint, mask_set_parameter_groups
from train_mask_set import (amp_settings, atomic_save, make_loaders, run_epoch, seed_everything,
                            seed_worker, source_info, write_json)
from utils.config import load_config, project_path
from utils.mask_set_consistency import MaskSetConsistency, make_ema_teacher
from utils.mask_set_loss import MaskSetCriterion


def check_continuation(config, payload, split):
    """续训必须保留亲本划分、损失、推理与固定伪标签源；新阶段显式重设优化器。"""
    if payload.get("split") != split:
        raise ValueError("continuation split differs from parent checkpoint")
    for key in ("loss", "inference", "augmentation", "manual_target_dir", "split_file"):
        if payload["config"]["mask_set"].get(key) != config["mask_set"].get(key):
            raise ValueError(f"continuation changes protected mask_set.{key}")
    old_pseudo, new_pseudo = (dict(c["mask_set"]["pseudo"]) for c in (payload["config"], config))
    old_pseudo.pop("samples_per_epoch", None)
    new_pseudo.pop("samples_per_epoch", None)
    # 新阶段两组共同使用按内容排除别名的原标签子集；来源关系在 train() 核验。
    old_pseudo.pop("dataset_dir", None)
    new_pseudo.pop("dataset_dir", None)
    if old_pseudo != new_pseudo:
        raise ValueError("continuation changes protected pseudo source or loss weight")
    if payload.get("phase") != "joint_lora":
        raise ValueError("this experiment must start from the completed joint_lora stage")


def train(config, output, device, split, classes, provenance):
    cfg, continuation = config["mask_set"], config["mask_set"]["continuation"]
    parent = Path(project_path(config, continuation["checkpoint"]))
    model, payload = load_mask_set_model(parent, config, device)
    check_continuation(config, payload, split)
    subset = json.loads(Path(project_path(config, cfg["pseudo"]["dataset_dir"]), "manifest.json").read_text(encoding="utf-8"))
    audit = subset["content_isolation"]
    parent_samples = set(payload["provenance"]["pseudo_target"]["samples"])
    removed = {row["stem"] for row in audit["excluded_images"]}
    if set(provenance["pseudo_target"]["samples"]) != parent_samples - removed:
        raise ValueError("pseudo subset is not exactly the parent's content-isolated samples")
    if provenance["pseudo_target"]["teacher_sha256"] != payload["provenance"]["pseudo_target"]["teacher_sha256"]:
        raise ValueError("pseudo teacher differs from the parent")
    excluded_validation = sorted(continuation["evaluation_excluded_names"])
    if excluded_validation != audit["exposed_validation_names"]:
        raise ValueError("validation exclusions do not match verified parent pseudo aliases")
    effective_split = {**split, "val": [n for n in split["val"] if n not in excluded_validation]}
    if not effective_split["val"]:
        raise ValueError("no independent validation images remain")
    provenance["content_isolation"] = audit
    origin_epoch = int(payload["epoch"])
    provenance["continuation"] = {"parent_checkpoint": str(parent), "parent_sha256": image_digest(parent),
        "parent_epoch": origin_epoch, "parent_val_objective": payload["val_objective"],
        "optimizer": "new AdamW for the new phase in both arms", "stage_format": "mask_set_continuation_v1"}
    del payload
    options = cfg["consistency"]
    enabled_consistency = float(options["weight"]) > 0
    # load_mask_set_model 的推理构造没有闭包式 checkpoint forward；先独立复制 EMA。
    teacher = make_ema_teacher(model) if enabled_consistency else None
    configure_mask_set_phase(model, train_lora=True)
    if config["lora"].get("gradient_checkpointing", True):
        enable_trunk_gradient_checkpointing(model.encoder.trunk)
    train_loader, val_loader = make_loaders(config, effective_split, device)
    consistency = None
    if enabled_consistency:
        holdout = Path(project_path(config, options["holdout_file"])).read_text(encoding="utf-8").splitlines()
        excluded = split["train"] + split["val"] + [s.strip() for s in holdout if s.strip() and not s.lstrip().startswith("#")]
        image_dir = project_path(config, options["image_dir"])
        pool = build_unlabeled_pool(image_dir, excluded, excluded_image_dir=project_path(config, config["paths"]["raw_data_dir"]))
        write_json(output / "unlabeled_pool.json", pool)
        dataset = MaskSetUnlabeledDataset(image_dir, pool, input_size=cfg["input_size"], mask_grid=cfg["mask_grid"])
        workers = int(cfg["num_workers"])
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=workers,
            pin_memory=device.type == "cuda", persistent_workers=workers > 0, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(int(options["seed"])))
        consistency = MaskSetConsistency(teacher, loader, config, device, len(train_loader))
        provenance["consistency_pool"] = {"sample_count": len(dataset), "label_source": pool["label_source"],
            "manifest": "unlabeled_pool.json", "sampling": "shuffled full-pool cycles without replacement within a cycle"}
    write_json(output / "sources.json", provenance)
    write_json(output / "model.json", {"parameters": model.parameter_summary(), "ema_enabled": enabled_consistency,
        "ema_parameters": teacher.parameter_summary()["total"] if teacher is not None else 0,
        "gradient_checkpointing": bool(config["lora"].get("gradient_checkpointing", True))})
    criterion = MaskSetCriterion(cfg["loss"])
    optimizer = torch.optim.AdamW(mask_set_parameter_groups(model, config, train_lora=True), weight_decay=float(cfg["weight_decay"]))
    epochs = int(continuation["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg["minimum_learning_rate"]))
    amp_enabled, dtype = amp_settings(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and dtype == torch.float16)
    initial = run_epoch(model, val_loader, criterion, config, device, classes)
    write_json(output / "initial_validation.json", initial)
    best = initial["loss_supervised"]
    best_epoch, best_step = origin_epoch, 0

    def checkpoint(epoch, phase_epoch, objective, metrics):
        metadata = dict(epoch=epoch, phase="consistency_continuation", phase_epoch=phase_epoch,
            selection="minimum_deterministic_supervised_validation_loss_including_parent",
            val_objective=objective, split=split, provenance=provenance, metrics=metrics,
            evaluation_excluded_names=excluded_validation, selection_validation_names=effective_split["val"],
            optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict(), continuation_format="mask_set_continuation_v1")
        if consistency is not None:
            metadata["ema"] = make_mask_set_checkpoint(teacher, config, updates=consistency.step)
        return make_mask_set_checkpoint(model, config, **metadata)

    atomic_save(checkpoint(origin_epoch, 0, best, {"initial_validation": initial}), output / "best_joint.pth")
    print(f"Initial e{origin_epoch} supervised validation loss={best:.6f}; validation={effective_split['val']}; EMA={enabled_consistency}", flush=True)
    started = time.time()
    for phase_epoch in range(1, epochs + 1):
        if shutil.disk_usage(output).free < 2 * 1024**3:
            raise RuntimeError("less than 2 GiB free; preserving current checkpoints")
        tick = time.time()
        values = run_epoch(model, train_loader, criterion, config, device, classes, optimizer, scaler, consistency)
        val = run_epoch(model, val_loader, criterion, config, device, classes)
        objective = val["loss_supervised"]
        improved = objective < best
        epoch = origin_epoch + phase_epoch
        if improved:
            best, best_epoch, best_step = objective, epoch, phase_epoch
        row = {"epoch": epoch, "phase_epoch": phase_epoch, "phase": "consistency_continuation",
            **{f"train_{k}": v for k, v in values.items()}, **{f"val_{k}": v for k, v in val.items()},
            "lr_decoder": optimizer.param_groups[0]["lr"], "lr_lora": optimizer.param_groups[1]["lr"],
            "seconds": time.time() - tick, "best": int(improved)}
        with (output / "metrics.csv").open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if phase_epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        scheduler.step()
        payload = checkpoint(epoch, phase_epoch, objective, row)
        atomic_save(payload, output / "last_mask_set.pth")
        if improved:
            atomic_save(payload, output / "best_joint.pth")
        del payload
        if consistency is not None:
            write_json(output / "unlabeled_usage.json", {"processed_samples": consistency.processed_samples,
                "unique_samples": len(consistency.samples_seen), "image_names": sorted(consistency.samples_seen),
                "ema_updates": consistency.step})
        print(f"epoch={epoch} phase_epoch={phase_epoch} val={objective:.6f} best={best:.6f} "
              f"unsup_samples={values.get('consistency_samples',0):.0f} "
              f"valid_fraction={values.get('consistency_valid_fraction',0):.4f} seconds={row['seconds']:.1f}", flush=True)
    write_json(output / "COMPLETED.json", {"continued_epochs": epochs, "parent_epoch": origin_epoch,
        "best_joint": {"epoch": best_epoch, "phase_epoch": best_step, "val_objective": best,
                       "checkpoint": str(output / "best_joint.pth")},
        "elapsed_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device), "final_evaluation": "pending"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-steps", type=int, help="短验证专用，须为4的倍数；正式作业不传")
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["mask_set"]
    cfg["output_dir"] = config["paths"]["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["continuation"]["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.smoke_steps is not None:
        if args.smoke_steps < 4 or args.smoke_steps % 4:
            raise ValueError("smoke steps must be a positive multiple of 4")
        cfg["manual_samples_per_epoch"] = args.smoke_steps // 4
        cfg["pseudo"]["samples_per_epoch"] = args.smoke_steps * 3 // 4
    options = cfg["consistency"]
    if (int(cfg["continuation"]["epochs"]) < 1 or int(options["every_n_steps"]) < 1
        or not 0 <= float(options["weight"]) <= 1 or not 0 <= float(options["ema_decay"]) < 1
        or int(options["ramp_epochs"]) < 1 or not .5 <= float(options["iou_threshold"]) <= 1
        or not .5 <= float(options["mask_confidence"]) <= 1):
        raise ValueError("invalid continuation or consistency configuration")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real continuation training requires sam2_env with CUDA")
    output = Path(project_path(config, args.output_dir))
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(int(cfg["seed"]))
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        split = json.loads(Path(project_path(config, cfg["split_file"])).read_text(encoding="utf-8"))
        holdout = Path(project_path(config, options["holdout_file"])).read_text(encoding="utf-8").splitlines()
        holdout = [n.strip() for n in holdout if n.strip() and not n.lstrip().startswith("#")]
        audit = disjoint_pseudo_subset(project_path(config, cfg["continuation"]["pseudo_source"]),
            project_path(config, cfg["pseudo"]["image_dir"]), project_path(config, config["paths"]["raw_data_dir"]),
            split, holdout, project_path(config, cfg["pseudo"]["dataset_dir"]))
        write_json(output / "content_isolation.json", audit)
        split, classes, provenance = source_info(config)
        write_json(output / "split.json", split)
        train(config, output, device, split, classes, provenance)
    except Exception as error:
        write_json(output / "FAILED.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
