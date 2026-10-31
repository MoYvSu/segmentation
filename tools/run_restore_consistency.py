# -*- coding: utf-8 -*-
"""顺序运行恢复器短适配；两组完成后核对每次更新的数据与噪声。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stop", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / "queue.json"
    if receipt.exists():
        raise FileExistsError("Queue already exists; do not overwrite training")
    state = {"status": "running", "arms": {}, "smoke": args.smoke}
    write(receipt, state)
    try:
        for arm in ("control", "candidate"):
            command = [sys.executable, "-u", "-B", "train_restore_consistency.py",
                       "--config", args.config, "--root", str(root), "--arm", arm,
                       "--stop", str(args.stop)]
            if args.smoke:
                command.append("--smoke")
            state["current_arm"] = arm
            state["arms"][arm] = {"command": command, "status": "running"}
            write(receipt, state)
            with (root / f"{arm}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, check=False)
            if result.returncode:
                raise RuntimeError(f"{arm} failed ({result.returncode}); see {arm}.log")
            status = json.loads((root / arm / "status.json").read_text(encoding="utf-8"))
            if status["status"] != "stage_complete":
                raise RuntimeError(f"{arm} did not finish its declared stage")
            state["arms"][arm].update(status="stage_complete", updates=status["updates"])
            write(receipt, state)
        statuses = [json.loads((root / arm / "status.json").read_text(encoding="utf-8"))
                    for arm in ("control", "candidate")]
        keys = ("initial_sha256", "initial_state_sha256", "source_pool_sha256",
                "recipe_signature", "source_code_sha256", "updates", "effective_batch")
        if any(statuses[0][key] != statuses[1][key] for key in keys):
            raise RuntimeError("A/B initialization, recipe, sources or budget differ")
        rows = [[json.loads(line) for line in (root / arm / "metrics.jsonl").read_text(
                    encoding="utf-8").splitlines()] for arm in ("control", "candidate")]
        if len(rows[0]) != len(rows[1]):
            raise RuntimeError("A/B update logs differ in length")
        fields = ("update", "epoch", "data_epoch", "sources", "pair_sha256", "noise_sha256")
        for left, right in zip(*rows):
            if any(left[key] != right[key] for key in fields):
                raise RuntimeError(f"A/B data/noise mismatch at update {left['update']}")
        state.update(status="stage_complete", matched_updates=len(rows[0]),
                     every_update_inputs_and_noise_equal=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        write(receipt, state)


if __name__ == "__main__":
    main()
