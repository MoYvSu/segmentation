# -*- coding: utf-8 -*-
"""只运行零噪声C组，并逐更新核对已经完成的A组；不重跑控制。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

# 直接作为 tools/ 脚本运行时使用项目内模块。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config


PAIR_FIELDS = ("update", "epoch", "data_epoch", "sources", "pair_sha256",
               "noise_rng_before", "noise_sha256", "profiles", "mask_mean",
               "learning_rate", "samples", "micro_batch", "terminal_step")
REFERENCE_FILES = ("run_config.json", "status.json", "source_code.json", "metrics.jsonl",
                   "epochs.jsonl", "epoch_000.pt", "epoch_005.pt", "last.pt")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparable_recipe(config):
    result = copy.deepcopy(config)
    result.pop("paths", None)  # 独立源码快照的项目根目录允许不同。
    result["rgb_restoration"].pop("output_dir", None)
    result["restore_consistency"].pop("noise_mode", None)
    # 原A配置继承候选系数0.1，但arm=control实际系数为0；C显式写0。
    result["restore_consistency"]["consistency_weight"] = 0.
    return result


def check_reference(config, reference):
    reference = Path(reference)
    status, old_config = read(reference / "status.json"), read(reference / "run_config.json")
    required = dict(status="stage_complete", arm="control", epoch=5, updates=1250,
                    effective_batch=4, micro_batch=1, terminal_step=16, source_count=1000,
                    holdout_sources=0, no_backend_loaded=True, smoke=False,
                    strict_reload_equal=True, first_inference_reload_equal=True)
    if any(status.get(key) != value for key, value in required.items()):
        raise ValueError("Reference must be the completed matched A5 control")
    if config["restore_consistency"].get("noise_mode") != "zero":
        raise ValueError("Only the explicit zero-noise recipe may use this launcher")
    if config["restore_consistency"].get("consistency_weight") != 0:
        raise ValueError("Zero/control comparison must have no extra consistency objective")
    if comparable_recipe(config) != comparable_recipe(old_config):
        raise ValueError("Zero/control recipes differ outside noise mode and output location")
    files = {name: sha(reference / name) for name in REFERENCE_FILES}
    if files["epoch_005.pt"] != status["final_checkpoint_sha256"]:
        raise ValueError("Reference final checkpoint does not match its receipt")
    # 只允许训练入口增加零模式；数据、模型、loss和随机实现必须复用同一版本。
    original_code = read(reference / "source_code.json")
    changed = []
    for name, expected in original_code.items():
        if sha(ROOT / name) != expected:
            changed.append(name)
    if set(changed) - {"train_restore_consistency.py"}:
        raise ValueError(f"Shared data/model code changed: {changed}")
    return dict(reference=str(reference.resolve()), files=files, status=status,
                intentional_changed_sources=changed)


def check_updates(reference, candidate, *, expected):
    originals = [json.loads(line) for line in (Path(reference) / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    rows = [json.loads(line) for line in (Path(candidate) / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if len(rows) != expected or len(originals) < expected:
        raise ValueError("Unexpected update count")
    for old, new in zip(originals[:expected], rows):
        for key in PAIR_FIELDS:
            if old[key] != new[key]:
                raise ValueError(f"Control/C mismatch at update {new['update']}: {key}")
        if (new.get("training_noise_mode") != "zero"
                or new.get("consistency_weight") != 0
                or old.get("consistency_weight") != 0
                or new.get("applied_noise_max_abs") != 0):
            raise ValueError(f"Actual zero-noise contract failed at update {new['update']}")
        if new["loss"] != new["supervised_loss"]:
            raise ValueError("Zero arm has an unexpected extra objective")
    return dict(matched_updates=len(rows), pair_fields=list(PAIR_FIELDS),
                every_update_inputs_order_draws_lr_equal=True,
                every_update_applied_noise_zero=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/restore_zero.yaml")
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference", required=True, help="只读A组control目录")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["restore_consistency"]["stage_epochs"] != 5:
        raise ValueError("This protocol predeclares five epochs")
    reference = check_reference(config, args.reference)
    root = Path(args.root).resolve()
    if root.exists():
        raise FileExistsError("Use a fresh experiment root; do not overwrite earlier work")
    root.mkdir(parents=True)
    command = [sys.executable, "-u", "-B", str(ROOT / "train_restore_consistency.py"),
               "--config", str(Path(args.config).resolve()), "--root", str(root),
               "--arm", "zero", "--stop", "1" if args.smoke else "5",
               "--monitor-module", "tools.restore_zero_monitor"]
    if args.smoke:
        command.append("--smoke")
    state = dict(status="running", smoke=args.smoke, command=command,
                 reference=reference, control_rerun=False, backend_loaded=False)
    write(root / "queue.json", state)
    try:
        with (root / "zero.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Zero arm failed ({result.returncode}); see zero.log")
        status = read(root / "zero/status.json")
        if status["status"] != "stage_complete":
            raise RuntimeError("Training did not reach its fixed endpoint")
        for key in ("initial_sha256", "initial_state_sha256", "source_pool_sha256",
                    "effective_batch", "micro_batch", "terminal_step"):
            if status[key] != reference["status"][key]:
                raise RuntimeError(f"Control/C identity differs: {key}")
        # 同一真实设备/精度才能复用历史随机张量和优化路径作为严格对照。
        for key in ("torch_version", "training_dtype", "matmul_allow_tf32", "cudnn_allow_tf32"):
            if status["runtime"][key] != reference["status"]["runtime"][key]:
                raise RuntimeError(f"Runtime precision differs from control: {key}")
        audit = check_updates(args.reference, root / "zero", expected=4 if args.smoke else 1250)
        after = {name: sha(Path(args.reference) / name) for name in REFERENCE_FILES}
        if after != reference["files"]:
            raise RuntimeError("Existing control artifacts were modified")
        state.update(status="stage_complete", **audit, reference_unchanged=True,
                     final_checkpoint_sha256=status["final_checkpoint_sha256"])
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write(root / "queue.json", state)


if __name__ == "__main__":
    main()
