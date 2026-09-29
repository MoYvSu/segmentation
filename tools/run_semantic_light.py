# -*- coding: utf-8 -*-
"""固定历史simple对照，只增加光照增强；结束后自动核验、推理和渲染。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, project_path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def comparison_config(config):
    """只排除输出位置和本轮光照变量，其余训练／部署配置必须相同。"""
    result = deepcopy(config)
    result.pop("semantic_light", None)
    result["semantic_adaptation"].pop("illumination", None)
    result["semantic_adaptation"].pop("output_dir", None)
    result["paths"].pop("project_root", None)
    # 历史status经JSON保存，YAML中的整数类别键会变为字符串；统一序列化口径。
    return json.loads(json.dumps(result, allow_nan=False))


def verify_control(config):
    control = Path(project_path(config, config["semantic_light"]["control_dir"]))
    status = read(control / "status.json")
    cfg = config["semantic_adaptation"]
    expected = int(cfg["epochs"]) * int(cfg["expected_manual_sources"]) * int(cfg["manual_repeats"])
    if (status["status"] != "completed" or status["arm"] != "simple"
            or status["updates"] != expected or status["failed_updates"] != 0):
        raise RuntimeError("Historical simple control is incomplete or has a different budget")
    if comparison_config(config) != comparison_config(status["config"]):
        raise RuntimeError("Training/deployment configuration differs beyond illumination")
    if status["config"]["semantic_adaptation"].get("illumination", {}).get("enabled", False):
        raise RuntimeError("Historical control already used illumination augmentation")
    for key in ("frozen_weights_unchanged", "d5a_unchanged", "frozen_affinity_output_equal",
                "strict_reload_passed", "semantic_weights_changed"):
        if not status[key]:
            raise RuntimeError(f"Historical control failed contract: {key}")
    return control


def verify_run(run, control, config, smoke=False, replay=False):
    status, reference = read(run / "status.json"), read(control / "status.json")
    cfg = config["semantic_adaptation"]
    expected = 2 if smoke else int(cfg["epochs"]) * int(cfg["expected_manual_sources"]) * int(cfg["manual_repeats"])
    if status["status"] != "completed" or status["updates"] != expected or status["failed_updates"]:
        raise RuntimeError("Light experiment did not complete its prescribed budget")
    for key in ("frozen_weights_unchanged", "frozen_affinity_output_equal", "d5a_unchanged",
                "strict_reload_passed", "semantic_weights_changed", "historical_simple_initialization_equal"):
        if not status[key]:
            raise RuntimeError(f"Light experiment failed contract: {key}")
    for key in ("init", "sources", "data", "frozen_state_sha256", "d5a_sha256", "d5a_state_sha256"):
        if status[key] != reference[key]:
            raise RuntimeError(f"Light experiment differs from historical simple: {key}")
    with np.load(run / "initial_probe/maps.npz") as left, np.load(control / "initial_probe/maps.npz") as right:
        if any(not np.array_equal(left[key], right[key]) for key in ("semantic", "boundary")):
            raise RuntimeError("Initial outputs differ from historical simple")
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    old = [json.loads(line) for line in (control / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    control_steps = int(cfg["epochs"]) * int(cfg["expected_manual_sources"]) * int(cfg["manual_repeats"])
    if len(rows) != expected or len(old) != control_steps:
        raise RuntimeError("Incomplete training step log")
    maximum_loss_delta, maximum_relative_loss_delta = 0.0, 0.0
    for left, right in zip(rows, old):
        for key in ("epoch", "step", "updates", "name", "draw_index", "input_sha256",
                    "restoration_seed", "seed", "profile", "is_spatial", "endpoint_applied", "noise_sigma"):
            if left[key] != right[key]:
                raise RuntimeError(f"Historical input pairing differs: {key}")
        if replay:
            delta = abs(left["semantic_loss"] - right["semantic_loss"])
            maximum_loss_delta = max(maximum_loss_delta, delta)
            maximum_relative_loss_delta = max(maximum_relative_loss_delta, delta / max(abs(right["semantic_loss"]), 1e-12))
            # BF16训练未启用确定性kernel；旧训练器自身重放也有微差，不能要求loss逐位相同。
            if left["illumination"]["applied"] or not np.isclose(
                    left["semantic_loss"], right["semantic_loss"], rtol=1e-3, atol=1e-6):
                raise RuntimeError("Disabled-light replay does not reproduce historical training")
    epochs = [json.loads(line) for line in (run / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    old_epochs = [json.loads(line) for line in (control / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(epochs) != (1 if smoke else int(cfg["epochs"])) or len(old_epochs) != int(cfg["epochs"]):
        raise RuntimeError("Incomplete training epoch log")
    if any(left["learning_rate"] != right["learning_rate"] for left, right in zip(epochs, old_epochs)):
        raise RuntimeError("Learning-rate schedule differs from historical simple")
    report = {"same_initialization_and_initial_outputs": True, "same_frozen_models_and_gt": True,
              "same_input_sequence_before_light": True, "same_restoration_seeds_and_learning_rates": True,
              "updates": expected, "changed_factor": "post_D5a_training_illumination",
              "disabled_replay": replay, "replay_maximum_loss_delta": maximum_loss_delta if replay else None,
              "replay_maximum_relative_loss_delta": maximum_relative_loss_delta if replay else None,
              "illumination_draws": sum(int(row["illumination"]["applied"]) for row in rows),
              "official_score": None}
    (run / "pair_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/semantic_light.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["semantic_adaptation"]
    control = verify_control(config)
    output = Path(project_path(config, cfg["output_dir"] + ("_smoke" if args.smoke else "")))
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline_status.json"
    if status_path.exists() and not args.postprocess_only:
        raise FileExistsError("Existing run; choose a new output directory")
    snapshot = output / "config.yaml"
    if snapshot.exists():
        config = load_config(str(snapshot))
        cfg = config["semantic_adaptation"]
        control = verify_control(config)
    else:
        if args.smoke:
            # 原seed的头两次抽样都不启用光照，短测单独强制覆盖；正式配置仍为50%。
            cfg["illumination"]["probability"] = 1.0
        snapshot.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    status = {"status": "running", "completed": [], "smoke": args.smoke,
              "control_dir": str(control), "competition_submission": False}

    def write():
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def execute(name, command):
        status["stage"] = name
        write()
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", *command], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status["completed"].append(name)
        write()

    flags = ["--smoke"] if args.smoke else []
    run = output / "simple"
    try:
        if args.smoke and not args.postprocess_only:
            off = deepcopy(config)
            off["semantic_adaptation"]["illumination"]["enabled"] = False
            off_path = output / "replay_config.yaml"
            off_path.write_text(yaml.safe_dump(off, allow_unicode=True), encoding="utf-8")
            execute("replay", ["train_semantic_d5a.py", "--arm", "simple", "--smoke",
                    "--config", str(off_path), "--output-dir", str(output / "replay")])
            status["replay_check"] = verify_run(output / "replay", control, config, smoke=True, replay=True)
        if not args.postprocess_only:
            execute("train", ["train_semantic_d5a.py", "--arm", "simple", "--config", str(snapshot),
                    "--output-dir", str(run), *flags])
        status["pair_check"] = verify_run(run, control, config, smoke=args.smoke)
        epoch = 1 if args.smoke else int(cfg["epochs"])
        prediction = run / "deployment"
        execute("infer", ["train_semantic_d5a.py", "--mode", "infer", "--arm", "simple",
                "--config", str(snapshot), "--output-dir", str(run), "--prediction-dir", str(prediction),
                "--checkpoint", str(run / f"epoch_{epoch:03d}.pt"), *flags])
        if not args.smoke:
            execute("package", ["tools/package_submission.py", "--prediction-dir", str(prediction),
                    "--test-dir", project_path(config, config["inference"]["test_dir"]),
                    "--output", str(output / "submission_simple.zip")])
        execute("render", ["tools/render_backend_ablation.py", "--image-dir",
                project_path(config, config["inference"]["test_dir"]),
                "--prediction", "Simple control=" + str(control / "deployment"),
                "--prediction", "Simple light=" + str(prediction),
                "--images", *cfg["monitor"]["images"], "--output-dir", str(output / "comparison")])
        status.update(status="completed", stage="completed")
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        write()


if __name__ == "__main__":
    main()
