# -*- coding: utf-8 -*-
"""Stage1 → 最终语义/affinity任务联合适配，按验证损失选择共享LoRA。"""

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.affinity_geometry_augmentation import AffinityGeometryAugmentedDataset
from data.dataset import split_train_val_indices
from data.sam2_geometry_dataset import SAM2GeometryDataset
from data.stage1_final_tasks import Stage1FinalTaskDataset
from models.direct_semantic_affinity import (
    configure_direct_training_phase, direct_parameter_groups, _load_complete_lora_state,
)
from models.stage1_final_tasks import build_stage1_final_tasks, task_checkpoint
from train_direct_semantic_affinity import (
    build_semantic_criterion, compute_semantic_loss, compute_affinity_loss,
    move_batch, next_restarting, set_seed, validate_sam2_geometry_approval,
)
from utils.config import load_config, project_path


logger = logging.getLogger(__name__)


def restore_tasks(model, payload):
    model.semantic_decoder.load_state_dict(payload["semantic_state_dict"], strict=True)
    model.affinity_decoder.load_state_dict(payload["affinity_state_dict"], strict=True)
    _load_complete_lora_state(model, payload["lora_state_dict"])


def build_loaders(config, device):
    cfg = config["direct_semantic_affinity"]
    task = config["stage1_final_tasks"]
    raw = Path(project_path(config, config["paths"]["raw_data_dir"]))
    gt = Path(project_path(config, config["boundary"]["gt_dir"]))
    names = sorted(p.stem for p in raw.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                   and p.with_suffix(".json").is_file() and (gt / (p.stem + "_gt.npz")).is_file())
    seed = int(cfg["seed"])
    split = {kind: [names[i] for i in split_train_val_indices(len(names), 0.8, seed, kind)]
             for kind in ("train", "val")}
    if set(split["train"]) & set(split["val"]):
        raise ValueError("联合适配训练/验证样本重叠")
    common = dict(data_dir=raw, gt_dir=gt, image_size=int(cfg["input_size"]),
                  affinity_grid=int(cfg["affinity_grid"]),
                  completed_gt_dir=project_path(config, task["completed_gt_dir"]),
                  augmentation=cfg.get("augmentation", {}))
    train = Stage1FinalTaskDataset(sample_names=split["train"], augment=True, **common)
    val = Stage1FinalTaskDataset(sample_names=split["val"], augment=False, **common)
    workers = int(cfg.get("num_workers", 0))
    options = dict(num_workers=workers, pin_memory=device.type == "cuda",
                   persistent_workers=workers > 0)
    sampler = WeightedRandomSampler(torch.ones(len(train), dtype=torch.double),
                                    int(cfg["manual_samples_per_epoch"]), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    manual = DataLoader(train, batch_size=int(cfg["batch_size"]), sampler=sampler, **options)
    validation = DataLoader(val, batch_size=1, shuffle=False, **options)
    sam = cfg["sam2_geometry"]
    approved = validate_sam2_geometry_approval(project_path(config, sam["dataset_dir"]))
    pseudo_base = SAM2GeometryDataset(
        project_path(config, sam["dataset_dir"]), project_path(config, sam["source_dir"]),
        image_size=int(cfg["input_size"]), output_grid=int(cfg["affinity_grid"]),
        use_eroded_interiors=False, cache_in_memory=False,
    )
    pseudo = DataLoader(AffinityGeometryAugmentedDataset(pseudo_base, cfg["augmentation"]),
                        batch_size=1, shuffle=True, **options)
    split["sam2_source_count"] = approved["source_count"]
    split["policy"] = "mainline_semantic_25_7; geometry_final_retains_26_6"
    return manual, validation, pseudo, split


def manual_losses(model, criterion, batch, loss_cfg):
    out = model(batch["image"])
    # 面积累加和逐实例loss固定float32，避免AMP半精度累加溢出。
    semantic = compute_semantic_loss(criterion, out["semantic_logits"].float(), batch)
    affinity, _ = compute_affinity_loss(out["affinity_logits"].float(), batch, loss_cfg, pseudo=False)
    return semantic, affinity


@torch.no_grad()
def validate(model, criterion, loader, loss_cfg, device):
    model.eval()
    sums = np.zeros(2, dtype=np.float64)
    count = 0
    for raw in loader:
        semantic, affinity = manual_losses(model, criterion, move_batch(raw, device), loss_cfg)
        sums += [float(semantic), float(affinity)]
        count += 1
    if count == 0 or not np.isfinite(sums).all():
        raise RuntimeError("验证损失为空或非有限")
    return dict(zip(("semantic", "affinity"), (sums / count).tolist()))


def train_epoch(model, criterion, manual, pseudo, loss_cfg, optimizer, scaler, device,
                scales, pseudo_weight, amp, max_steps=None):
    model.train()
    model.encoder.trunk.eval()
    iterator = iter(pseudo)
    totals = {"semantic": 0.0, "affinity": 0.0, "pseudo_affinity": 0.0,
              "steps": 0, "applied_steps": 0, "lora_grad_norm": 0.0}
    lora_parameters = [p for n, p in model.encoder.trunk.named_parameters()
                       if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    for index, raw in enumerate(manual):
        if max_steps is not None and index >= max_steps:
            break
        batch = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            sem, aff = manual_losses(model, criterion, batch, loss_cfg)
            supervised = sem / scales["semantic"] + aff / scales["affinity"]
        if not torch.isfinite(supervised):
            raise FloatingPointError("联合人工loss非有限")
        scaler.scale(supervised).backward()
        # 分两次反传，避免同时保留两张1024图的encoder激活。
        raw_pseudo, iterator = next_restarting(pseudo, iterator)
        pb = move_batch(raw_pseudo, device)
        with torch.amp.autocast("cuda", enabled=amp):
            features = model.encoder(pb["image"])
            logits = model.affinity_decoder(features)["affinity_logits"].float()
            ploss, _ = compute_affinity_loss(logits, pb, loss_cfg, pseudo=True)
            weighted_pseudo = float(pseudo_weight) * ploss / scales["affinity"]
        if not torch.isfinite(weighted_pseudo):
            raise FloatingPointError("SAM2 affinity loss非有限")
        scaler.scale(weighted_pseudo).backward()
        scaler.unscale_(optimizer)
        if lora_parameters:
            grads = [p.grad.norm().float() for p in lora_parameters if p.grad is not None]
            totals["lora_grad_norm"] = float(torch.stack(grads).norm()) if grads else 0.0
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        applied = bool(scaler.get_scale() >= old_scale)
        if not amp and not torch.isfinite(norm):
            raise FloatingPointError("非AMP梯度非有限")
        totals["applied_steps"] += int(applied)
        totals["steps"] += 1
        totals["semantic"] += float(sem.detach())
        totals["affinity"] += float(aff.detach())
        totals["pseudo_affinity"] += float(ploss.detach())
    if not totals["applied_steps"]:
        raise RuntimeError("本轮没有实际参数更新")
    for key in ("semantic", "affinity", "pseudo_affinity"):
        totals[key] /= totals["steps"]
    totals["amp_skipped_steps"] = totals["steps"] - totals["applied_steps"]
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/stage1_final_tasks_20260919.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    cfg, task = config["direct_semantic_affinity"], config["stage1_final_tasks"]
    set_seed(int(cfg["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("正式任务适配要求sam2_env中的CUDA")
    device = torch.device("cuda")
    model, initialization = build_stage1_final_tasks(config, device)
    initial_lora = {n: p.detach().cpu().clone() for n, p in model.encoder.trunk.named_parameters()
                    if "lora_A" in n or "lora_B" in n}
    manual, validation, pseudo, split = build_loaders(config, device)
    criterion = build_semantic_criterion(config, device)
    loss_cfg = cfg["affinity_loss"]
    output = Path(project_path(config, task["output_dir"])) / ("smoke" if args.smoke else "adaptation")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pth").exists() and not args.resume:
        raise FileExistsError("已有运行，请显式--resume或选新目录")
    payload = torch.load(project_path(config, args.resume), map_location="cpu", weights_only=False) if args.resume else None
    if payload is not None:
        if payload["split"] != split:
            raise ValueError("resume split不一致")
        restore_tasks(model, payload)
        if "rng" in payload:
            rng = payload["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            manual.sampler.generator.set_state(rng["sampler"])
    # 固定尺度只用于合并两个不同量纲的任务；不按部署代理选轮。
    scales = payload["loss_scales"] if payload else {"semantic": 1.0, "affinity": 1.0}
    warmup = 1 if args.smoke else int(cfg["head_warmup_epochs"])
    joint = 1 if args.smoke else int(cfg["joint_epochs"])
    start = int(payload["epoch"]) if payload else 0
    best = dict(payload.get("phase_best_loss", {})) if payload else {}
    manifest = {"initialization": initialization, "split": split, "config": config,
                "loss_scales": scales, "selection": "val_semantic_loss + val_manual_affinity_loss",
                "status": "running", "total_epochs": warmup + joint}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    active = None
    for epoch in range(start, warmup + joint):
        phase = "head_warmup" if epoch < warmup else "joint_lora"
        if phase != active:
            if phase == "joint_lora" and not (payload and payload["phase"] == phase):
                restore_tasks(model, torch.load(output / "best_head_warmup.pth", map_location="cpu", weights_only=False))
            configure_direct_training_phase(model, train_lora=phase == "joint_lora")
            rates = cfg["learning_rates"]
            groups = direct_parameter_groups(model, {
                "semantic": rates["warmup_semantic" if phase == "head_warmup" else "joint_semantic"],
                "affinity": rates["warmup_affinity" if phase == "head_warmup" else "joint_affinity"],
                "lora": rates["joint_lora"],
            })
            optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg["weight_decay"]), eps=1e-4)
            epochs = warmup if phase == "head_warmup" else joint
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]), init_scale=256.0)
            if payload and payload["phase"] == phase and epoch == start:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                scheduler.load_state_dict(payload["scheduler_state_dict"])
                scaler.load_state_dict(payload["scaler_state_dict"])
            active = phase
            logger.info("阶段=%s 学习率=%s", phase, [(g["name"], g["lr"]) for g in groups])
        train = train_epoch(model, criterion, manual, pseudo, loss_cfg, optimizer, scaler,
                            device, scales, cfg["pseudo_affinity_weight"], bool(cfg["amp"]),
                            max_steps=1 if args.smoke else None)
        val = validate(model, criterion, validation, loss_cfg, device)
        score = val["semantic"] / scales["semantic"] + val["affinity"] / scales["affinity"]
        scheduler.step()
        selection = {"metric": manifest["selection"], "direction": "min", "value": score}
        row = {"epoch": epoch + 1, "phase": phase, **train,
               "val_semantic_loss": val["semantic"], "val_affinity_loss": val["affinity"], "val_loss": score}
        row["lora_max_delta_from_stage1"] = max(float((p.detach().cpu() - initial_lora[n]).abs().max())
            for n, p in model.encoder.trunk.named_parameters() if n in initial_lora)
        if phase == "head_warmup" and not args.resume and row["lora_max_delta_from_stage1"] != 0:
            raise RuntimeError("warmup期间LoRA发生变化")
        if phase == "joint_lora" and (row["lora_grad_norm"] <= 0 or row["lora_max_delta_from_stage1"] == 0):
            raise RuntimeError("joint阶段未更新LoRA")
        record = task_checkpoint(model, config, epoch, phase, selection, split)
        if score < best.get(phase, math.inf):
            best[phase] = score
            torch.save(record, output / ("best_" + phase + ".pth"))
        record.update(phase_best_loss=best, loss_scales=scales, optimizer_state_dict=optimizer.state_dict(),
                      scheduler_state_dict=scheduler.state_dict(), scaler_state_dict=scaler.state_dict())
        record["rng"] = {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                         "sampler": manual.sampler.generator.get_state()}
        torch.save(record, output / "latest.pth")
        with (output / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        logger.info("epoch=%s phase=%s train=%.5f/%.5f val=%.5f steps=%d/%d LoRA_delta=%.3g",
                    epoch + 1, phase, train["semantic"], train["affinity"], score,
                    train["applied_steps"], train["steps"], row["lora_max_delta_from_stage1"])
    manifest.update(status="completed", phase_best_loss=best)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
