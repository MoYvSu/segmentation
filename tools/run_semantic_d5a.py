# -*- coding: utf-8 -*-
"""依次训练完整/简化语义头，完成后自动推理、打包并渲染对照。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, project_path

ARMS = ("full", "simple")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_pair(output, cfg, smoke=False):
    statuses = [read(output / arm / "status.json") for arm in ARMS]
    expected = (2 if smoke else int(cfg["epochs"]) *
                int(cfg["expected_manual_sources"]) * int(cfg["manual_repeats"]))
    for arm, status in zip(ARMS, statuses):
        if (status["status"] != "completed" or status["updates"] != expected
                or status["failed_updates"] != 0):
            raise RuntimeError(f"{arm}: training is incomplete")
        for key in ("frozen_weights_unchanged", "frozen_affinity_output_equal", "d5a_unchanged",
                    "strict_reload_passed", "semantic_weights_changed"):
            if not status[key]:
                raise RuntimeError(f"{arm}: failed contract {key}")
    for key in ("frozen_state_sha256", "d5a_sha256", "data"):
        if statuses[0][key] != statuses[1][key]:
            raise RuntimeError(f"Paired setup differs: {key}")
    if not statuses[0]["initial_final_output_equal"]:
        raise RuntimeError("Full-head initialization does not reproduce original D5a deployment")
    with np.load(output / "full" / "initial_probe" / "maps.npz") as a, \
            np.load(output / "simple" / "initial_probe" / "maps.npz") as b:
        if not np.array_equal(a["boundary"], b["boundary"]):
            raise RuntimeError("Initial affinity maps differ between the two arms")
    logs = [[json.loads(line) for line in (output / arm / "steps.jsonl")
             .read_text(encoding="utf-8").splitlines()] for arm in ARMS]
    if any(len(log) != expected for log in logs):
        raise RuntimeError("Incomplete paired step logs")
    for left, right in zip(*logs):
        for key in ("epoch", "step", "updates", "name", "draw_index",
                    "input_sha256", "restoration_seed", "endpoint_applied"):
            if left[key] != right[key]:
                raise RuntimeError(f"Training inputs are not paired: {key}")
    report = {
        "same_frozen_models": True, "same_new_gt": True,
        "same_sample_sequence_and_degraded_inputs": True,
        "same_restoration_seeds": True, "updates_per_arm": expected,
        "same_initial_affinity_output": True,
        "full_initial_final_output_equal": True,
        "comparison": "pretrained full head adaptation versus random simplified head",
        "changed_factors": ["semantic structure", "semantic initialization", "learning rate"],
        "learning_rates": cfg["learning_rates"], "smoke": smoke,
        "official_scores": None,
    }
    (output / "pair_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/semantic_d5a.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["semantic_adaptation"]
    output = Path(project_path(config, cfg["output_dir"] + ("_smoke" if args.smoke else "")))
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline_status.json"
    if status_path.exists() and not args.postprocess_only:
        raise FileExistsError("Existing pipeline; choose a new output directory")
    snapshot = output / "config.yaml"
    if snapshot.exists():
        config = load_config(str(snapshot))
        cfg = config["semantic_adaptation"]
    else:
        snapshot.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    status = {"status": "running", "completed": [], "smoke": args.smoke,
              "config": args.config, "automatic_final_inference_and_render": True,
              "competition_submission": False}

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

    run_config = str(snapshot)
    smoke_flags = ["--smoke"] if args.smoke else []
    try:
        if not args.postprocess_only:
            for arm in ARMS:
                execute("train_" + arm, ["train_semantic_d5a.py", "--mode", "train",
                        "--config", run_config, "--arm", arm,
                        "--output-dir", str(output / arm), *smoke_flags])
        status["pair_check"] = verify_pair(output, cfg, args.smoke)
        epoch = 1 if args.smoke else int(cfg["epochs"])
        for arm in ARMS:
            prediction_dir = output / arm / "deployment"
            execute("infer_" + arm, ["train_semantic_d5a.py", "--mode", "infer",
                    "--config", run_config, "--arm", arm,
                    "--checkpoint", str(output / arm / f"epoch_{epoch:03d}.pt"),
                    "--output-dir", str(output / arm),
                    "--prediction-dir", str(prediction_dir), *smoke_flags])
            if not args.smoke:
                execute("package_" + arm, ["tools/package_submission.py",
                        "--prediction-dir", str(prediction_dir),
                        "--test-dir", project_path(config, config["inference"]["test_dir"]),
                        "--output", str(output / f"submission_{arm}.zip")])
        execute("render", ["tools/render_backend_ablation.py",
                "--image-dir", project_path(config, config["inference"]["test_dir"]),
                "--prediction", "Baseline=" + project_path(config, cfg["baseline_prediction_dir"]),
                "--prediction", "D5a original=" + project_path(config, cfg["d5a_prediction_dir"]),
                "--prediction", "A adapted full=" + str(output / "full" / "deployment"),
                "--prediction", "B new simple=" + str(output / "simple" / "deployment"),
                "--images", *cfg["monitor"]["images"],
                "--output-dir", str(output / "comparison")])
        status.update(status="completed", stage="completed", comparison_dir=str(output / "comparison"))
    except BaseException as error:
        status.update(status="failed", error=repr(error))
        raise
    finally:
        write()


if __name__ == "__main__":
    main()
