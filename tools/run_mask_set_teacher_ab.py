# -*- coding: utf-8 -*-
"""一次性顺序作业：完成伪标签 -> 仅伪标签对照 -> 加归属约束；不轮询训练进展。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, project_path


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "state.json").exists():
        raise FileExistsError("preserving existing sequential job")
    controls = [load_config(args.control_config), load_config(args.ownership_config)]
    normalized = []
    for cfg in controls:
        cfg = copy.deepcopy(cfg)
        cfg["mask_set"].pop("output_dir", None)
        cfg["paths"].pop("output_dir", None)
        cfg["mask_set"]["loss"]["ownership_weight"] = 0.
        normalized.append(cfg)
    if normalized[0] != normalized[1]:
        raise ValueError("A/B configurations differ beyond output directory and ownership weight")
    if controls[0]["mask_set"]["loss"]["ownership_weight"] != 0 or controls[1]["mask_set"]["loss"]["ownership_weight"] <= 0:
        raise ValueError("expected disabled/enabled ownership A/B")
    state = {"started_at": datetime.now().astimezone().isoformat(), "status": "running", "steps": [],
             "preview_review": {"status": "user_accepted", "source": "conversation", "label_source": "mainline_pseudo",
                                "names": ["train_636", "train_597", "train_746", "train_499"]}}

    def save():
        temp = output / "state.tmp.json"
        temp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(output / "state.json")

    generation = load_config(args.generation_config)
    directory = Path(project_path(generation, generation["pseudo_generation"]["output_dir"]))
    commands = [
        ("generation", [sys.executable, "-u", "tools/generate_mainline_pseudo.py", "--config", args.generation_config]
         + (["--resume"] if (directory / "manifest.json").exists() else [])),
        ("teacher_only", [sys.executable, "-u", "train_mask_set.py", "--config", args.control_config,
                          "--output-dir", str(output / "teacher_only")]),
        ("teacher_ownership", [sys.executable, "-u", "train_mask_set.py", "--config", args.ownership_config,
                               "--output-dir", str(output / "teacher_ownership")]),
    ]
    try:
        for name, command in commands:
            free = shutil.disk_usage(output).free
            if free < 2 * 1024**3:
                raise RuntimeError(f"less than 2GiB free before {name}; preserving existing outputs")
            record = {"name": name, "command": command, "started_at": datetime.now().astimezone().isoformat(),
                      "status": "running", "free_bytes_before": free}
            state["steps"].append(record)
            state["current_step"] = name
            save()
            print(f"START {name}", flush=True)
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, finished_at=datetime.now().astimezone().isoformat())
            if result.returncode:
                record["status"] = "failed"
                raise RuntimeError(f"{name} failed, see {output / (name+'.log')}")
            record["status"] = "complete"
            save()
            print(f"COMPLETE {name}", flush=True)
        state.update(status="complete", finished_at=datetime.now().astimezone().isoformat(), final_evaluation="pending")
        save()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-config", default="config/train/mainline_pseudo_generation249.yaml")
    parser.add_argument("--control-config", default="config/train/mask_set_teacher249.yaml")
    parser.add_argument("--ownership-config", default="config/train/mask_set_teacher249_ownership.yaml")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
