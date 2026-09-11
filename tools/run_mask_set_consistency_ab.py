# -*- coding: utf-8 -*-
"""一次性顺序执行同起点短续训对照，保存完成/失败状态，不设置进度轮询。"""
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
from utils.config import load_config


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "state.json").exists():
        raise FileExistsError("preserving an existing continuation job")
    configs = [load_config(args.control_config), load_config(args.consistency_config)]
    normalized = []
    for config in configs:
        config = copy.deepcopy(config)
        config["paths"].pop("output_dir", None)
        config["mask_set"].pop("output_dir", None)
        config["mask_set"]["consistency"]["weight"] = 0
        normalized.append(config)
    if normalized[0] != normalized[1] or configs[0]["mask_set"]["consistency"]["weight"] != 0:
        raise ValueError("A/B must differ only by consistency weight and output paths")
    if configs[1]["mask_set"]["consistency"]["weight"] <= 0:
        raise ValueError("consistency arm requires positive weight")
    state = {"status": "running", "started_at": datetime.now().astimezone().isoformat(), "steps": []}

    def save():
        temporary = output / "state.tmp.json"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "state.json")

    save()
    try:
        for name, config in (("control", args.control_config), ("consistency", args.consistency_config)):
            if shutil.disk_usage(output).free < 2 * 1024**3:
                raise RuntimeError("less than 2 GiB free before the next arm")
            command = [sys.executable, "-u", "train_mask_set_consistency.py", "--config", config,
                       "--output-dir", str(output / name)]
            record = {"name": name, "command": command, "status": "running", "started_at": datetime.now().astimezone().isoformat()}
            state["steps"].append(record)
            state["current_step"] = name
            save()
            print(f"START {name}", flush=True)
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, finished_at=datetime.now().astimezone().isoformat(),
                          status="complete" if result.returncode == 0 else "failed")
            save()
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {name}.log")
            print(f"COMPLETE {name}", flush=True)
        state.update(status="complete", finished_at=datetime.now().astimezone().isoformat(), final_evaluation="pending")
        save()
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", default="config/train/mask_set_continue30.yaml")
    parser.add_argument("--consistency-config", default="config/train/mask_set_consistency30.yaml")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
