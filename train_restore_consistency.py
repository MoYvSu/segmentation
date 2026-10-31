# -*- coding: utf-8 -*-
"""固定 D5a 首步的双噪声适配：匹配控制、有效 batch4、可续训短阶段。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.rgb_restoration import load_rgb_restorer
from train_rgb_restoration import (
    build_datasets, reconstruction_loss, save_checkpoint, seed_worker, write_json,
)
from utils.config import load_config, project_path


TRAINING_FORMAT = "restore_consistency_v1"


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(str((value.dtype, tuple(value.shape))).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_sha(*values):
    return state_sha({str(index): value for index, value in enumerate(values)})


def append_json(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def save_rng(noise_generator, loader_generator, device):
    name, values, pos, has_gauss, cached = np.random.get_state()
    return {
        "noise_generator": noise_generator.get_state(),
        "loader_generator": loader_generator.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "python": random.getstate(),
        "numpy": [name, torch.from_numpy(values.copy()), pos, has_gauss, cached],
    }


def load_rng(state, noise_generator, loader_generator, device):
    noise_generator.set_state(state["noise_generator"].cpu())
    loader_generator.set_state(state["loader_generator"].cpu())
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])
    random.setstate(state["python"])
    name, values, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, values.numpy(), pos, has_gauss, cached))


def isolated_callback(callback, model, noise_generator, loader_generator, device, *args, **kwargs):
    """过程图可以自行抽随机数，但不能推进训练的任何随机流。"""
    snapshot = save_rng(noise_generator, loader_generator, device)
    was_training = model.training
    try:
        with torch.no_grad():
            return callback(*args, **kwargs)
    finally:
        load_rng(snapshot, noise_generator, loader_generator, device)
        model.train(was_training)


def validate_options(config, *, arm=None):
    cfg, options = config["rgb_restoration"], config["restore_consistency"]
    if cfg.get("split_policy") != "all_train":
        raise ValueError("This experiment requires all_train without a new holdout")
    if cfg["model"].get("architecture") != "pixel_diffusion_nafnet":
        raise ValueError("Only the deployed pixel diffusion D5a architecture is supported")
    if cfg["train"]["batch_size"] != 4:
        raise ValueError("Effective batch must remain four")
    micro = options.get("micro_batch", 1)
    if type(micro) is not int or micro < 1 or 4 % micro:
        raise ValueError("micro_batch must divide effective batch four")
    for key, default in (("consistency_weight", .1), ("gradient_weight", .2)):
        value = options.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    noise_mode = options.get("noise_mode", "stochastic")
    if noise_mode not in {"stochastic", "zero"}:
        raise ValueError("noise_mode must be stochastic or zero")
    if arm is not None and arm not in {"control", "candidate", "zero"}:
        raise ValueError("Unknown arm")
    if arm is not None and (arm == "zero") != (noise_mode == "zero"):
        raise ValueError("zero arm requires zero noise_mode; original arms require stochastic")
    if noise_mode == "zero" and options.get("consistency_weight", .1) != 0:
        raise ValueError("Zero-noise arm must declare consistency_weight=0")
    if noise_mode == "stochastic" and options.get("consistency_weight", .1) == 0:
        raise ValueError("Candidate consistency_weight must be positive")
    for key, default in (("data_epoch_offset", 60), ("expected_initial_epoch", 60)):
        value = options.get(key, default)
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg["model"].get("num_steps") != 16:
        raise ValueError("The deployed terminal first-step contract requires T=16")
    expected = options.get("initial_sha256", "")
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("A full lowercase initial_sha256 is required")
    return cfg, options


def recipe_signature(config):
    cfg = copy.deepcopy(config["rgb_restoration"])
    for key in ("output_dir", "monitor", "selection", "overfit", "holdout_manifest"):
        cfg.pop(key, None)
    cfg["train"].pop("num_workers", None)
    cfg["train"].pop("save_epochs", None)
    options = copy.deepcopy(config["restore_consistency"])
    for key in ("stage_epochs", "monitor_epochs", "initial_checkpoint", "output_dir", "monitor"):
        options.pop(key, None)
    payload = {"rgb_restoration": cfg, "restore_consistency": options}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def source_identity(dataset):
    """核对真实训练池内容；同名但被替换的图不能悄悄续训。"""
    rows = [{"name": Path(path).name, "size": Path(path).stat().st_size, "sha256": file_sha(path)}
            for path in dataset.samples]
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()
    return rows, signature


def source_code_identity(*, arm="control", monitor_module=None):
    root = Path(__file__).resolve().parent
    names = ("train_restore_consistency.py", "train_rgb_restoration.py",
             "data/rgb_restoration_dataset.py", "data/rgb_spatial_blur.py",
             "data/dataset.py", "data/mim_dataset.py", "models/rgb_restoration.py",
             "models/rgb_diffusion_restoration.py", "utils/config.py",
             "utils/restoration_noise.py", "utils/cuda/restore_noise.ptx")
    if arm == "zero":
        # 零噪声实验的过程图和排队逻辑同样属于可续训配方，不能漏记。
        names += ("tools/restore_zero_monitor.py", "tools/restore_consistency_monitor.py",
                  "tools/run_restore_zero.py", "utils/image_io.py", "utils/restoration_probe.py")
    rows = {name: file_sha(root / name) for name in names}
    if isinstance(monitor_module, str):
        module_path = root.joinpath(*monitor_module.split(".")).with_suffix(".py")
        if not module_path.is_file():
            raise FileNotFoundError(f"Monitor source is unavailable: {module_path}")
        rows[module_path.relative_to(root).as_posix()] = file_sha(module_path)
    signature = hashlib.sha256(json.dumps(rows, sort_keys=True).encode("ascii")).hexdigest()
    return rows, signature


def first_prediction(model, inputs, *, seed=314159, noise_mode="stochastic"):
    """严格重载的小规模首步检查；独立 generator 不消耗训练随机数。"""
    condition = inputs.float()
    t = torch.full((len(inputs),), model.num_steps, dtype=torch.long, device=inputs.device)
    if noise_mode == "zero":
        state = model.q_sample(condition, condition, t, noise=torch.zeros_like(condition))
    elif noise_mode != "stochastic":
        raise ValueError("noise_mode must be stochastic or zero")
    elif inputs.device.type == "cuda":
        from utils.restoration_noise import restoration_noise
        noise = restoration_noise(condition.shape, seed, inputs.device)
        state = model.q_sample(condition, condition, t, noise=noise)
    else:
        generator = torch.Generator(device=inputs.device).manual_seed(seed)
        state = model.q_sample(condition, condition, t, generator=generator)
    raw = model.denoise(state, condition, t).float()
    # 与正式 restore_first 的 strength=1 算术顺序相同。
    return (condition + 1.0 * (raw - condition)).clamp(0, 1)


def update(model, batch, optimizer, scaler, noise_generator, device, cfg, options, *, arm):
    """一次有效更新，两臂噪声/输入相同；B 仅多一个对称一致性项。"""
    if arm not in {"control", "candidate", "zero"}:
        raise ValueError("Unknown arm")
    noise_mode = options.get("noise_mode", "stochastic")
    if noise_mode not in {"stochastic", "zero"} or (arm == "zero") != (noise_mode == "zero"):
        raise ValueError("Training arm and noise_mode do not match")
    if arm == "zero" and options.get("consistency_weight", .1) != 0:
        raise ValueError("Zero-noise arm must declare consistency_weight=0")
    inputs, target, valid = [batch[key].to(device, non_blocking=True)
                             for key in ("input", "target", "valid")]
    count, micro = len(inputs), options.get("micro_batch", 1)
    if count != 4 or count % micro:
        raise ValueError("Each optimizer update must contain exactly four source images")
    if not bool((valid.sum((1, 2, 3)) > 0).all()):
        raise ValueError("Empty valid supervision region")
    t = torch.full((count,), model.num_steps, device=device, dtype=torch.long)
    noise_before = hashlib.sha256(noise_generator.get_state().cpu().numpy().tobytes()).hexdigest()
    # 先抽整个有效 batch；调整显存用的 micro-batch 不改变随机噪声图案。
    eps_a = torch.randn(inputs.shape, device=device, dtype=torch.float32, generator=noise_generator)
    eps_b = torch.randn(inputs.shape, device=device, dtype=torch.float32, generator=noise_generator)
    # T 时 eta=1，只从退化输入构造状态，清晰目标不进入恢复网络。
    # C 仍消费与 A/B 相同的两份抽样，唯一干预是实际注入状态的噪声置零。
    applied_a = torch.zeros_like(eps_a) if noise_mode == "zero" else eps_a
    applied_b = torch.zeros_like(eps_b) if noise_mode == "zero" else eps_b
    state_a = model.q_sample(inputs, inputs, t, noise=applied_a)
    state_b = model.q_sample(inputs, inputs, t, noise=applied_b)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    totals = {key: 0.0 for key in ("loss", "supervised_loss", "consistency_loss",
                                  "consistency_rgb_l1", "consistency_gradient_l1",
                                  "supervised_rgb_l1", "supervised_gradient_l1", "supervised_mse")}
    amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    dtype = torch.bfloat16 if device.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    consistency_cfg = {"rgb_weight": 1.0, "gradient_weight": options.get("gradient_weight", .2)}
    weight = options.get("consistency_weight", .1) if arm == "candidate" else 0.0
    for start in range(0, count, micro):
        selection = slice(start, start + micro)
        with torch.autocast(device_type=device.type, enabled=amp, dtype=dtype):
            left = model.denoise(state_a[selection], inputs[selection], t[selection])
            right = model.denoise(state_b[selection], inputs[selection], t[selection])
            loss_a, errors_a = reconstruction_loss(left, target[selection], valid[selection], cfg["loss"], return_errors=True)
            loss_b, errors_b = reconstruction_loss(right, target[selection], valid[selection], cfg["loss"], return_errors=True)
            supervised = .5 * (loss_a + loss_b)
            consistency, consistency_errors = reconstruction_loss(left, right, valid[selection], consistency_cfg, return_errors=True)
            loss = supervised + weight * consistency
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite dual-noise loss")
        fraction = micro / count
        scaler.scale(loss * fraction).backward()
        for key, value in (("loss", loss), ("supervised_loss", supervised), ("consistency_loss", consistency)):
            totals[key] += float(value.detach()) * fraction
        for key in ("rgb_l1", "gradient_l1"):
            totals["consistency_" + key] += float(consistency_errors[key].detach().sum()) / count
        for key in ("rgb_l1", "gradient_l1", "mse"):
            totals["supervised_" + key] += float((.5 * (errors_a[key] + errors_b[key])).detach().sum()) / count
        del left, right, loss, supervised, consistency
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < previous_scale:
        raise FloatingPointError("Skipped optimizer step; matched update budget no longer holds")
    totals.update(grad_norm=float(norm), learning_rate=optimizer.param_groups[0]["lr"],
                  samples=count, micro_batch=micro, terminal_step=model.num_steps,
                  noise_rng_before=noise_before, noise_sha256=tensor_sha(eps_a, eps_b),
                  applied_noise_sha256=tensor_sha(applied_a, applied_b), noise_mode=noise_mode,
                  training_noise_mode=noise_mode,
                  applied_noise_max_abs=float(torch.maximum(applied_a.abs().max(), applied_b.abs().max())),
                  pair_sha256=tensor_sha(inputs, target, valid), consistency_weight=weight,
                  sources=list(batch["source"]))
    totals["weighted_consistency_fraction"] = weight * totals["consistency_loss"] / max(
        totals["supervised_loss"], 1e-12)
    return totals


def train(config, arm, root, *, device="cuda", stop=None, resume=False, smoke=False,
          monitor_module=None):
    cfg, options = validate_options(config, arm=arm)
    noise_mode = options.get("noise_mode", "stochastic")
    if arm == "zero" and monitor_module is None:
        monitor_module = "tools.restore_zero_monitor"
    stop = options.get("stage_epochs", 5) if stop is None else stop
    if type(stop) is not int or not 1 <= stop <= cfg["train"]["epochs"]:
        raise ValueError("Stop must be an epoch endpoint inside the predeclared schedule")
    root, device = Path(root), torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU sam2_env required for formal training")
    out = root / arm
    if out.exists() and not resume:
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    initial_path = Path(project_path(config, options["initial_checkpoint"]))
    initial_sha = file_sha(initial_path)
    if initial_sha != options["initial_sha256"]:
        raise ValueError("Initial deployed D5a file identity differs")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if "epoch" in initial and initial["epoch"] != options.get("expected_initial_epoch", 60):
        raise ValueError("Initial checkpoint is not the declared deployed D5a epoch")
    model = load_rgb_restorer(initial_path, device)
    if model.model_config != cfg["model"]:
        raise ValueError("Initial model architecture differs from the recipe")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    initial_state_sha = state_sha(model.state_dict())
    dataset, holdout = build_datasets(config)
    if holdout is not None or len(dataset) != cfg["expected_train_sources"]:
        raise ValueError("Training must use all declared permitted originals without holdout")
    if len(dataset) % 4:
        raise ValueError("Partial or dropped source batches are not allowed")
    sources, pool_sha = source_identity(dataset)
    source_file = root / "source_pool.json"
    if source_file.exists():
        if json.loads(source_file.read_text(encoding="utf-8")) != sources:
            raise ValueError("A/B source pools differ")
    else:
        write_json(source_file, sources)
    code_sources, code_sha = source_code_identity(arm=arm, monitor_module=monitor_module)
    signature = recipe_signature(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                 weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["min_learning_rate"])
    amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not torch.cuda.is_bf16_supported())
    noise_generator = torch.Generator(device=device).manual_seed(seed + 271828)
    loader_generator = torch.Generator().manual_seed(seed)
    completed = updates = 0
    if resume:
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        checks = {"training_format": TRAINING_FORMAT, "arm": arm, "initial_sha256": initial_sha,
                  "recipe_signature": signature, "source_pool_sha256": pool_sha,
                  "source_code_sha256": code_sha, "smoke": smoke}
        if any(saved.get(key) != value for key, value in checks.items()):
            raise ValueError("Resume recipe, inputs, source code or initial weights differ")
        if saved.get("training_noise_mode", "stochastic") != noise_mode:
            raise ValueError("Resume training noise mode differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        load_rng(saved["rng"], noise_generator, loader_generator, device)
        completed, updates = saved["epoch"], saved["updates"]
        if completed >= stop:
            raise ValueError("Resume endpoint must exceed the completed epoch")
        lines = (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if len(lines) != updates:
            raise ValueError("Partial epoch log exists; explicit recovery is required")
    status = dict(status="running", arm=arm, epoch=completed, updates=updates,
                  stage_epochs=stop, schedule_epochs=cfg["train"]["epochs"],
                  data_epoch_offset=options.get("data_epoch_offset", 60), source_count=len(dataset),
                  source_pool_sha256=pool_sha, source_code_sha256=code_sha,
                  initial_sha256=initial_sha, initial_state_sha256=initial_state_sha,
                  recipe_signature=signature, effective_batch=4, micro_batch=options.get("micro_batch", 1),
                  noise_views=2, terminal_step=16, no_backend_loaded=True, holdout_sources=0,
                  training_noise_mode=noise_mode,
                  checkpoint_selection="fixed_stage_endpoint", smoke=smoke, official_score=None)
    status["runtime"] = dict(torch_version=str(torch.__version__),
                             training_dtype=("bfloat16" if torch.cuda.is_bf16_supported()
                                             else "float16") if amp else "float32",
                             first_reload_dtype="float32", first_reload_noise_protocol=(
                                 "zero" if noise_mode == "zero" else (
                                     "cuda_philox4090_v1" if device.type == "cuda" else "cpu_test_generator")),
                             matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                             cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
    write_json(out / "run_config.json", config)
    write_json(out / "source_code.json", code_sources)
    write_json(out / "status.json", status)
    started = time.monotonic()
    module = importlib.import_module(monitor_module) if isinstance(monitor_module, str) else monitor_module
    context = None
    probe = None

    def monitor(epoch):
        if module is None:
            return {}
        return isolated_callback(module.run_monitor, model, noise_generator, loader_generator, device,
                                 model, context, out, epoch, updates, device)

    def checkpoint(epoch, report):
        payload = dict(format=model.checkpoint_format, model_config=model.model_config,
                       training_format=TRAINING_FORMAT, model=model.state_dict(), epoch=epoch,
                       updates=updates, arm=arm, initial_sha256=initial_sha,
                       training_noise_mode=noise_mode,
                       initial_state_sha256=initial_state_sha, recipe_signature=signature,
                       source_pool_sha256=pool_sha, source_code_sha256=code_sha, smoke=smoke,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), rng=save_rng(noise_generator, loader_generator, device),
                       experiment_config=config, metrics=report)
        save_checkpoint(out / "last.pt", payload)
        slim = {key: value for key, value in payload.items()
                if key not in {"optimizer", "scheduler", "scaler", "rng"}}
        save_checkpoint(out / f"epoch_{epoch:03d}.pt", slim)

    try:
        if module is not None:
            context = isolated_callback(module.prepare_monitor, model, noise_generator, loader_generator,
                                        device, copy.deepcopy(config), dataset, root)
        if not resume:
            monitor(0)
            checkpoint(0, {})
        for epoch in range(completed, stop):
            dataset.set_epoch(options.get("data_epoch_offset", 60) + epoch)
            order = torch.randperm(len(dataset), generator=loader_generator).tolist()
            if smoke:
                order = order[:min(16, len(order))]
            data = DataLoader(Subset(dataset, order), batch_size=4, shuffle=False,
                              num_workers=cfg["train"]["num_workers"],
                              pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed + epoch + 12345))
            rows = []
            for batch in data:
                if probe is None:
                    probe = batch["input"][:1].clone().to(device)
                row = update(model, batch, optimizer, scaler, noise_generator, device, cfg, options, arm=arm)
                updates += 1
                row.update(epoch=epoch + 1, update=updates, data_epoch=dataset.epoch,
                           profiles=list(batch["profile"]),
                           mask_mean=[float(value) for value in batch.get("mask_mean", [])])
                append_json(out / "metrics.jsonl", row)
                rows.append(row)
                if updates <= 2 or updates % 25 == 0:
                    print(json.dumps({key: row[key] for key in
                                      ("epoch", "update", "loss", "supervised_loss", "consistency_loss",
                                       "grad_norm", "learning_rate")}), flush=True)
            scheduler.step()
            completed = epoch + 1
            report = dict(epoch=completed, updates=updates, epoch_updates=len(rows),
                          samples=sum(row["samples"] for row in rows),
                          next_learning_rate=optimizer.param_groups[0]["lr"],
                          **{key: float(np.mean([row[key] for row in rows]))
                             for key in ("loss", "supervised_loss", "consistency_loss",
                                         "consistency_rgb_l1", "consistency_gradient_l1",
                                         "supervised_rgb_l1", "supervised_gradient_l1", "supervised_mse",
                                         "weighted_consistency_fraction")})
            if completed in options.get("monitor_epochs", [0, 1, 3, 5]) or completed == stop:
                report["monitor"] = monitor(completed)
            checkpoint(completed, report)
            append_json(out / "epochs.jsonl", report)
            status.update(epoch=completed, updates=updates, elapsed_seconds=time.monotonic() - started)
            write_json(out / "status.json", status)
        expected_updates = stop * (min(16, len(dataset)) if smoke else len(dataset)) // 4
        if updates != expected_updates:
            raise RuntimeError(f"Mismatched update budget: {updates}/{expected_updates}")
        if state_sha(model.state_dict()) == initial_state_sha:
            raise RuntimeError("No learned parameter changes")
        final = out / f"epoch_{stop:03d}.pt"
        reloaded = load_rgb_restorer(final, device)
        model.eval()
        with torch.no_grad():
            equal = torch.equal(first_prediction(model, probe, noise_mode=noise_mode),
                                first_prediction(reloaded, probe, noise_mode=noise_mode))
            standard_equal = torch.equal(first_prediction(model, probe), first_prediction(reloaded, probe))
        if state_sha(reloaded.state_dict()) != state_sha(model.state_dict()) or not equal or not standard_equal:
            raise RuntimeError("Strict reload changed model state or actual first-step output")
        if file_sha(initial_path) != initial_sha:
            raise RuntimeError("Initial deployment checkpoint was unexpectedly modified")
        status.update(status="stage_complete", strict_reload_equal=True,
                      first_inference_reload_equal=True, standard_first_inference_reload_equal=True,
                      final_checkpoint_sha256=file_sha(final))
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(out / "status.json", status)
    print(f"COMPLETE {arm} e{completed} updates={updates}", flush=True)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=("control", "candidate", "zero"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--monitor-module")
    args = parser.parse_args()
    train(load_config(args.config), args.arm, args.root, device=args.device, stop=args.stop,
          resume=args.resume, smoke=args.smoke, monitor_module=args.monitor_module or (
              "tools.restore_zero_monitor" if args.arm == "zero" else "tools.restore_consistency_monitor"))
