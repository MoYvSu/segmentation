# -*- coding: utf-8 -*-
"""复用SSL的实例集合训练：冻结预热 -> 恢复损失最优 -> LoRA联训。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler, default_collate

from data.direct_dual_head_dataset import DirectDualHeadDataset
from data.mask_set_pseudo_dataset import (FixedSourceRatioSampler, MainlinePseudoDataset,
                                         MaskSetSampleView, read_pseudo_manifest)
from models.direct_semantic_affinity import _load_complete_lora_state
from models.mask_set import (
    build_mask_set_model, configure_mask_set_phase, load_mask_set_model,
    make_mask_set_checkpoint, mask_set_parameter_groups,
)
from train_direct_semantic_affinity import validate_manual_target_dataset
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, load_class_map
from utils.mask_set_inference import postprocess_mask_set
from utils.mask_set_loss import MaskSetCriterion, targets_from_direct_batch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def source_info(config):
    cfg = config["mask_set"]
    raw = Path(project_path(config, config["paths"]["raw_data_dir"]))
    target_dir = Path(project_path(config, cfg["manual_target_dir"]))
    split = json.loads(Path(project_path(config, cfg["split_file"])).read_text(encoding="utf-8"))
    train, val = list(split["train"]), list(split["val"])
    if not train or not val or len(set(train + val)) != len(train + val):
        raise ValueError("train/val must be nonempty, unique and disjoint")
    audit = validate_manual_target_dataset(target_dir, raw)
    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    if set(train + val) != set(manifest["samples"]):
        raise ValueError("fixed split must cover exactly the reviewed manual target dataset")
    classes = {stem: load_class_map(target_dir / f"{stem}_class.json") for stem in train + val}
    max_instances = max(map(len, classes.values()))
    if max_instances > int(cfg["num_queries"]):
        raise ValueError(f"query capacity {cfg['num_queries']} is below {max_instances} GT instances")
    ssl_path = Path(project_path(config, config["lora"]["init_from"]))
    # 记录实际复用资产，便于与之前的direct对照核对；不重新运行SSL。
    with ssl_path.open("rb") as handle:
        ssl_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    provenance = {"manual_target": audit, "max_instances": max_instances,
                  "ssl_path": str(ssl_path), "ssl_sha256": ssl_sha}
    pseudo = cfg.get("pseudo", {})
    if pseudo.get("enabled", False):
        if int(cfg.get("batch_size", 1)) != 1:
            raise ValueError("mixed-source mask-set training currently requires batch_size=1")
        weight = float(pseudo.get("loss_weight", .5))
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("pseudo loss_weight must be finite in (0,1]")
        if int(pseudo.get("samples_per_epoch", 0)) < 1:
            raise ValueError("pseudo samples_per_epoch must be positive")
        held = Path(project_path(config, pseudo["holdout_file"])).read_text(encoding="utf-8").splitlines()
        excluded = train + val + [n.strip() for n in held if n.strip() and not n.lstrip().startswith("#")]
        manifest, pseudo_classes = read_pseudo_manifest(
            project_path(config, pseudo["dataset_dir"]), project_path(config, pseudo["image_dir"]),
            excluded_names=excluded, query_capacity=cfg["num_queries"])
        classes.update(pseudo_classes)
        provenance["pseudo_target"] = {"format": manifest["format"], "label_source": manifest["label_source"],
            "dataset_dir": pseudo["dataset_dir"], "sample_count": manifest["sample_count"],
            "samples": manifest["samples"], "teacher_sha256": manifest["teacher_sha256"],
            "samples_per_epoch": int(pseudo["samples_per_epoch"]), "loss_weight": weight}
        provenance["max_instances"] = max(map(len, classes.values()))
    return split, classes, provenance


def make_dataset(config, names, *, augment):
    cfg = config["mask_set"]
    return DirectDualHeadDataset(
        data_dir=project_path(config, config["paths"]["raw_data_dir"]), gt_dir=None,
        sample_names=names, image_size=int(cfg["input_size"]),
        affinity_grid=int(cfg["mask_grid"]), augment=augment,
        augmentation=cfg.get("augmentation", {}),
        manual_target_dir=project_path(config, cfg["manual_target_dir"]),
    )


def attach_classes(batch, class_maps):
    batch["instance_classes"] = [class_maps[Path(name).stem] for name in batch["image_name"]]
    return batch


def make_loaders(config, split, device):
    cfg = config["mask_set"]
    train = make_dataset(config, split["train"], augment=True)
    val = make_dataset(config, split["val"], augment=False)
    sampler = WeightedRandomSampler(
        torch.ones(len(train), dtype=torch.double),
        num_samples=int(cfg["manual_samples_per_epoch"]), replacement=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )
    pseudo = cfg.get("pseudo", {})
    if pseudo.get("enabled", False):
        directory = project_path(config, pseudo["dataset_dir"])
        manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
        pseudo_train = MainlinePseudoDataset(
            project_path(config, pseudo["image_dir"]), directory, manifest,
            image_size=cfg["input_size"], mask_grid=cfg["mask_grid"], augment=True,
            augmentation=cfg.get("augmentation", {}))
        sampler = FixedSourceRatioSampler(len(train), len(pseudo_train), cfg["manual_samples_per_epoch"],
                                          pseudo["samples_per_epoch"], cfg["seed"])
        train = ConcatDataset([MaskSetSampleView(train, "manual", 1.),
                               MaskSetSampleView(pseudo_train, "mainline_pseudo", pseudo["loss_weight"])])
    workers = int(cfg.get("num_workers", 2))
    kwargs = dict(batch_size=int(cfg.get("batch_size", 1)), num_workers=workers,
                  pin_memory=device.type == "cuda", persistent_workers=workers > 0,
                  worker_init_fn=seed_worker)
    if workers:
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    validation_generator = torch.Generator().manual_seed(int(cfg.get("validation_seed", 20260910)))
    return (DataLoader(train, sampler=sampler, **kwargs),
            DataLoader(val, shuffle=False, generator=validation_generator, **kwargs))


def amp_settings(config, device):
    cfg = config["mask_set"]
    enabled = bool(cfg.get("amp", True)) and device.type == "cuda"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.get("amp_dtype", "bfloat16")]
    return enabled, dtype


def scalar_items(losses):
    return {key: float(value.detach()) if torch.is_tensor(value) else float(value)
            for key, value in losses.items() if isinstance(value, (int, float))
            or (torch.is_tensor(value) and value.numel() == 1)}


def run_epoch(model, loader, criterion, config, device, class_maps, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    enabled, dtype = amp_settings(config, device)
    # 每次验证用同一随机序列，使匹配点/损失点可比且不扰动训练RNG。
    generator = None if training else torch.Generator(device=device).manual_seed(
        int(config["mask_set"].get("validation_seed", 20260910)))
    totals, count = {}, 0
    source_totals = {}
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            attach_classes(batch, class_maps)
            targets = targets_from_direct_batch(batch, device)
            image = batch["image"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                outputs = model(image)
            losses = criterion(outputs, targets, generator=generator)
            factor = float(batch["sample_loss_weight"][0]) if "sample_loss_weight" in batch else 1.
            # 直接缩放整个批次的梯度，不按权重和重新归一化，避免batch1抵消伪标签降权。
            loss = losses["loss_total"] * factor
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss for {batch['image_name']}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(config["mask_set"].get("grad_clip", 1.0)), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            batch_size = image.shape[0]
            source = batch["label_source"][0] if "label_source" in batch else "manual"
            stats = source_totals.setdefault(source, {"samples": 0, "loss_sum": 0., "optimized_loss_sum": 0.})
            stats["samples"] += batch_size
            stats["loss_sum"] += float(losses["loss_total"].detach()) * batch_size
            stats["optimized_loss_sum"] += float(loss.detach()) * batch_size
            for key, value in scalar_items(losses).items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            count += batch_size
    values = {key: value / max(1, count) for key, value in totals.items()}
    for source, stats in source_totals.items():
        values[f"{source}_samples"] = stats["samples"]
        values[f"{source}_loss"] = stats["loss_sum"] / max(1, stats["samples"])
        values[f"{source}_optimized_loss"] = stats["optimized_loss_sum"] / max(1, stats["samples"])
    return values


def training(config, output, device, split, class_maps, provenance):
    cfg = config["mask_set"]
    model, ssl_meta = build_mask_set_model(config, device)
    write_json(output / "model.json", {"parameters": model.parameter_summary(), "ssl": ssl_meta})
    train_loader, val_loader = make_loaders(config, split, device)
    criterion = MaskSetCriterion(cfg.get("loss", {}))
    enabled, dtype = amp_settings(config, device)
    csv_path = output / "metrics.csv"
    global_epoch = 0
    best_joint = None
    started = time.time()
    for phase, epochs, train_lora in (
        ("head_warmup", int(cfg["head_warmup_epochs"]), False),
        ("joint_lora", int(cfg["joint_epochs"]), True),
    ):
        if train_lora:
            warmup = torch.load(output / "best_warmup.pth", map_location="cpu", weights_only=False)
            model.decoder.load_state_dict(warmup["decoder_state_dict"], strict=True)
            _load_complete_lora_state(model, warmup["lora_state_dict"])
            print(f"Restored warmup validation-loss best epoch {warmup['epoch']}", flush=True)
            del warmup
        configure_mask_set_phase(model, train_lora=train_lora)
        optimizer = torch.optim.AdamW(mask_set_parameter_groups(model, config, train_lora=train_lora),
                                      weight_decay=float(cfg.get("weight_decay", 1e-4)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(cfg.get("minimum_learning_rate", 1e-6)))
        scaler = torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16)
        best = float("inf")
        best_path = output / ("best_joint.pth" if train_lora else "best_warmup.pth")
        for phase_epoch in range(1, epochs + 1):
            global_epoch += 1
            tick = time.time()
            train_values = run_epoch(model, train_loader, criterion, config, device, class_maps, optimizer, scaler)
            val_values = run_epoch(model, val_loader, criterion, config, device, class_maps)
            # 两组对照使用同一人工验证BCE/Dice/类别损失选优，不让新增归属权重改变选优口径。
            objective = val_values["loss_supervised"]
            improved = objective < best
            if improved:
                best = objective
            row = {"epoch": global_epoch, "phase": phase, "phase_epoch": phase_epoch,
                   **{f"train_{k}": v for k, v in train_values.items()},
                   **{f"val_{k}": v for k, v in val_values.items()},
                   "lr_decoder": optimizer.param_groups[0]["lr"],
                   "lr_lora": optimizer.param_groups[-1]["lr"] if train_lora else 0.0,
                   "seconds": time.time() - tick, "best": int(improved)}
            with csv_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if global_epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
            scheduler.step()
            payload = make_mask_set_checkpoint(
                model, config, epoch=global_epoch, phase=phase, phase_epoch=phase_epoch,
                selection="minimum_deterministic_supervised_validation_loss", val_objective=objective,
                split=split, provenance=provenance, metrics=row,
                optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                scaler_state_dict=scaler.state_dict(),
            )
            atomic_save(payload, output / "last_mask_set.pth")
            if improved:
                atomic_save(payload, best_path)
                if train_lora:
                    best_joint = {"epoch": global_epoch, "val_objective": objective,
                                  "checkpoint": str(best_path)}
            if global_epoch % int(cfg.get("checkpoint_interval", 10)) == 0:
                atomic_save(payload, output / f"epoch_{global_epoch:03d}.pth")
            del payload
            print(f"epoch={global_epoch:03d} phase={phase} train={train_values['loss_total']:.6f} "
                  f"val={objective:.6f} best={best:.6f} seconds={row['seconds']:.1f}", flush=True)
    write_json(output / "COMPLETED.json", {"epochs": global_epoch, "best_joint": best_joint,
               "elapsed_seconds": time.time() - started, "final_evaluation": "pending"})


def grid_diagnostics(model, cached_batches, config, device, output, step):
    model.eval()
    cfg = config["mask_set"]
    enabled, dtype = amp_settings(config, device)
    records = []
    with torch.inference_mode():
        for batch in cached_batches:
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                result = model(batch["image"].to(device))
            masks = result["pred_masks"][0].float().cpu()
            logits = result["pred_logits"][0].float().cpu()
            grid = int(cfg["mask_grid"])
            pred, pred_classes, info = postprocess_mask_set(
                masks, logits, (grid, grid), (0, 0), input_size=grid, **cfg["inference"])
            gt = batch["affinity_instance_map"][0].numpy().astype(np.uint16)
            valid = batch["affinity_valid_content"][0, 0].numpy().astype(bool)
            masked = pred.copy()
            masked[~valid] = 0
            classes = batch["instance_classes"][0]
            metrics = evaluate_instance_pair(gt, classes, masked, pred_classes)
            agnostic = evaluate_instance_pair(gt, {k: 1 for k in classes}, masked, {k: 1 for k in pred_classes})
            stem = Path(batch["image_name"][0]).stem
            cv2.imwrite(str(output / f"{stem}_step{step:04d}_instances.png"), pred)
            np.savez_compressed(output / f"{stem}_step{step:04d}_preview.npz", gt=gt, pred=pred,
                                image=batch["image"][0].numpy(), valid=valid)
            records.append({"name": stem, "metrics": metrics, "class_agnostic": agnostic, "inference": info})
    write_json(output / f"overfit_step_{step:04d}.json", records)
    return records


def overfit(config, output, device, split, class_maps, provenance, names, steps, interval):
    if not names or not set(names) <= set(split["train"]):
        raise ValueError("overfit probe must use only explicit training names")
    dataset = make_dataset(config, names, augment=False)
    batches = [attach_classes(default_collate([dataset[i]]), class_maps) for i in range(len(dataset))]
    model, ssl_meta = build_mask_set_model(config, device)
    configure_mask_set_phase(model, train_lora=False)
    optimizer = torch.optim.AdamW(mask_set_parameter_groups(model, config, train_lora=False), weight_decay=1e-4)
    criterion = MaskSetCriterion(config["mask_set"].get("loss", {}))
    enabled, dtype = amp_settings(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=enabled and dtype == torch.float16)
    baseline = run_epoch(model, batches, criterion, config, device, class_maps)
    grid_diagnostics(model, batches, config, device, output, 0)
    write_json(output / "model.json", {"parameters": model.parameter_summary(), "ssl": ssl_meta})
    history = []
    started = time.time()
    for step in range(1, steps + 1):
        values = run_epoch(model, [batches[(step - 1) % len(batches)]], criterion, config, device,
                           class_maps, optimizer, scaler)
        if step % interval == 0 or step == steps:
            measured = run_epoch(model, batches, criterion, config, device, class_maps)
            row = {"step": step, "train": values, "fixed_objective": measured,
                   "elapsed_seconds": time.time() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
            grid_diagnostics(model, batches, config, device, output, step)
            atomic_save(make_mask_set_checkpoint(model, config, epoch=0, phase="overfit_probe",
                        step=step, split=split, provenance=provenance, selection="diagnostic_only"),
                        output / "overfit_mask_set.pth")
    # 严格重新加载，检查保存后最终输出与原模型一致；正式训练重新冷启动decoder。
    reloaded, _ = load_mask_set_model(output / "overfit_mask_set.pth", config, device)
    reloaded.eval()
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
        image = batches[0]["image"].to(device)
        old_output, new_output = model(image), reloaded(image)
        differences = {key: float((old_output[key] - new_output[key]).abs().max())
                       for key in ("pred_logits", "pred_masks")}
    if any(value != 0.0 for value in differences.values()):
        raise RuntimeError(f"checkpoint reload changed final logits: {differences}")
    write_json(output / "OVERFIT_COMPLETED.json", {"names": names, "steps": steps,
               "initial": baseline, "history": history, "reload_max_abs": differences,
               "elapsed_seconds": time.time() - started, "formal_training_initialization": "fresh_decoder_same_ssl"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/mask_set_clean60.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--joint-epochs", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--overfit-names", nargs="+")
    parser.add_argument("--overfit-steps", type=int, default=300)
    parser.add_argument("--probe-interval", type=int, default=50)
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["mask_set"]
    for argument, key in ((args.warmup_epochs, "head_warmup_epochs"), (args.joint_epochs, "joint_epochs"),
                          (args.samples_per_epoch, "manual_samples_per_epoch"), (args.num_workers, "num_workers")):
        if argument is not None:
            cfg[key] = argument
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    config["paths"]["output_dir"] = cfg["output_dir"]
    if min(int(cfg["head_warmup_epochs"]), int(cfg["joint_epochs"]), int(cfg["manual_samples_per_epoch"])) < 1:
        raise ValueError("both phases and samples per epoch must be positive")
    split, class_maps, provenance = source_info(config)
    if args.check:
        print(json.dumps({"split": split, "sources": provenance}, indent=2))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU training requires sam2_env with CUDA")
    seed_everything(int(cfg["seed"]))
    torch.set_float32_matmul_precision("high")
    output = Path(project_path(config, cfg["output_dir"]))
    if output.exists() and any(output.glob("*.pth")):
        raise FileExistsError(f"preserving existing run checkpoints: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output / "split.json", split)
    write_json(output / "sources.json", provenance)
    try:
        if args.overfit_names:
            overfit(config, output, device, split, class_maps, provenance, args.overfit_names,
                    args.overfit_steps, args.probe_interval)
        else:
            training(config, output, device, split, class_maps, provenance)
    except Exception as error:
        write_json(output / "FAILED.json", {"type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
