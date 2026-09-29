# -*- coding: utf-8 -*-
"""冷启动双头分阶段训练，完成后自动核验、推理、打包及渲染。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, project_path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def schedule(config, smoke=False):
    cfg = config["backend_adaptation"]
    warmup, joint = (1, 1) if smoke else (int(cfg["head_warmup_epochs"]), int(cfg["joint_epochs"]))
    if min(warmup, joint) <= 0 or (not smoke and warmup + joint != int(cfg["epochs"])):
        raise ValueError("Cold-start schedule must contain both complete training phases")
    updates = 2 if smoke else int(cfg["expected_manual_sources"]) * int(cfg["manual_repeats"])
    if updates <= 0:
        raise ValueError("Training must contain at least one update per epoch")
    return warmup, joint, updates


def private_output(config, smoke=False):
    """所有权重、逐图预测及提交包仅落在项目的私有产物目录。"""
    root = Path(config["paths"]["project_root"]).resolve()
    output = Path(project_path(config, config["backend_adaptation"]["output_dir"]
                               + ("_smoke" if smoke else ""))).resolve()
    allowed = [root / "output", root / "outputs"]
    if not any(output != base.resolve() and output.is_relative_to(base.resolve()) for base in allowed):
        raise ValueError("Cold-start artifacts must stay inside project output/ or outputs/")
    return output


def verify_run(run, config, smoke=False):
    run = Path(run)
    status = read(run / "status.json")
    warmup, joint, per_epoch = schedule(config, smoke)
    total_epochs, expected = warmup + joint, (warmup + joint) * per_epoch
    if (status.get("status") != "completed" or status.get("epoch") != total_epochs
            or status.get("updates") != expected or status.get("failed_updates") != 0):
        raise RuntimeError("Cold-start training did not complete its prescribed budget")
    for key in ("frozen_encoder_unchanged", "d5a_unchanged", "warmup_lora_unchanged", "strict_reload_passed"):
        if status.get(key) is not True:
            raise RuntimeError(f"Cold-start contract failed: {key}")
    changes = status.get("trainable_max_changes", {})
    for group in ("semantic", "affinity", "lora"):
        value = changes.get(group, 0)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Trainable group did not change correctly: {group}")
    expected_phases = {"head_warmup": {"epochs": warmup, "updates": warmup * per_epoch},
                       "joint": {"epochs": joint, "updates": joint * per_epoch}}
    if status.get("phases") != expected_phases:
        raise RuntimeError("Cold-start phase budgets differ from the prescribed schedule")
    rows, epochs = read_rows(run / "steps.jsonl"), read_rows(run / "epochs.jsonl")
    if len(rows) != expected or len(epochs) != total_epochs:
        raise RuntimeError("Incomplete training step or epoch logs")
    for index, row in enumerate(rows, 1):
        epoch = (index - 1) // per_epoch + 1
        phase = "head_warmup" if epoch <= warmup else "joint"
        if row.get("updates") != index or row.get("epoch") != epoch or row.get("phase") != phase:
            raise RuntimeError("Training step sequence or phase differs from the prescribed schedule")
    for epoch, row in enumerate(epochs, 1):
        phase = "head_warmup" if epoch <= warmup else "joint"
        if row.get("epoch") != epoch or row.get("phase") != phase:
            raise RuntimeError("Training epoch sequence or phase differs from the prescribed schedule")
    monitor = config["backend_adaptation"]["monitor"]
    frequency = int(monitor["every_epochs"])
    if frequency <= 0:
        raise ValueError("Monitor frequency must be positive")
    rounds = sorted({0, 1, total_epochs, *range(frequency, total_epochs + 1, frequency)})
    thumbnails = [run / "monitor" / f"epoch_{epoch:03d}" / f"{stem}.png"
                  for epoch in rounds for stem in monitor["images"]]
    if not thumbnails or any(not path.is_file() or path.stat().st_size == 0 for path in thumbnails):
        raise RuntimeError("Missing prescribed process thumbnails")
    checkpoint = run / f"epoch_{total_epochs:03d}.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError("Missing final checkpoint")
    report = {"status": "passed", "smoke": smoke, "epochs": total_epochs, "updates": expected,
              "phases": expected_phases, "frozen_encoder_and_d5a_unchanged": True,
              "warmup_lora_unchanged": True, "strict_reload_passed": True,
              "trainable_max_changes": changes, "monitor_images": len(thumbnails),
              "checkpoint": str(checkpoint), "official_scores": None}
    (run / "run_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def verify_smoke_predictions(config, prediction):
    # 现有打包器按完整测试目录枚举；短测只校验固定四图，不生成可误交的残缺包。
    from tools.package_submission import IMAGE_SUFFIXES, validate_pair

    directory = Path(project_path(config, config["inference"]["test_dir"]))
    names = config["backend_adaptation"]["monitor"]["images"]
    images = [path for path in directory.iterdir() if path.is_file()
              and path.suffix.lower() in IMAGE_SUFFIXES and path.stem in names]
    if len(images) != len(names) or {path.stem for path in images} != set(names):
        raise RuntimeError("Smoke inference is missing prescribed test images")
    pairs = [validate_pair(path, prediction) for path in images]
    return {"images": len(pairs), "instances": sum(count for _, _, count in pairs),
            "format_validated": True, "package_created": False,
            "reason": "Smoke covers monitor images only; a formal submission requires the full test set"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/backend_cold.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = private_output(config, args.smoke)
    snapshot = output / "config.yaml"
    if args.postprocess_only:
        if not snapshot.is_file():
            raise FileNotFoundError("Postprocessing requires the existing resolved config.yaml")
        config = load_config(str(snapshot))
        if private_output(config, args.smoke) != output:
            raise RuntimeError("Archived configuration belongs to another output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Existing run; choose a distinct output directory")
        output.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    cfg = config["backend_adaptation"]
    warmup, joint, _ = schedule(config, args.smoke)
    flags = ["--smoke"] if args.smoke else []
    run = output / "model"
    prediction = run / "deployment"
    status_path = output / "pipeline_status.json"
    status = {"status": "running", "completed": [], "smoke": args.smoke,
              "postprocess_only": args.postprocess_only, "config": str(snapshot),
              "competition_submission": False, "automatic_final_inference_and_render": True}

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

    try:
        if not args.postprocess_only:
            execute("train", ["train_backend_coldstart.py", "--config", str(snapshot),
                              "--output-dir", str(run), *flags])
        status["stage"] = "verify"
        write()
        status["run_check"] = verify_run(run, config, args.smoke)
        status["completed"].append("verify")
        execute("infer", ["train_backend_coldstart.py", "--config", str(snapshot),
                          "--checkpoint", str(run / f"epoch_{warmup + joint:03d}.pt"),
                          "--output-dir", str(prediction), *flags])
        if args.smoke:
            status["stage"] = "validate_smoke_predictions"
            write()
            status["prediction_check"] = verify_smoke_predictions(config, prediction)
            status["completed"].append("validate_smoke_predictions")
        else:
            execute("package", ["tools/package_submission.py", "--prediction-dir", str(prediction),
                                "--test-dir", project_path(config, config["inference"]["test_dir"]),
                                "--output", str(output / "submission_cold.zip")])
        execute("render", ["tools/render_backend_ablation.py", "--image-dir",
                           project_path(config, config["inference"]["test_dir"]),
                           "--prediction", "D5a original=" + project_path(config, cfg["d5a_prediction_dir"]),
                           "--prediction", "Simple original=" + project_path(config, cfg["simple_prediction_dir"]),
                           "--prediction", "Cold start=" + str(prediction),
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
