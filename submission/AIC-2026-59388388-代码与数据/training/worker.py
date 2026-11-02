# -*- coding: utf-8 -*-
"""单阶段进程；只调用training/source中的训练算法。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from bootstrap import activate_source


def finish(context):
    import torch
    from runner import write_json
    models, elements = {}, 0
    for name, artifact, key in (("backend", "backend", "model_state_dict"),
                                ("restorer", "restorer", "model"), ("naf", "naf_export", "model_state_dict")):
        path = Path(context["artifacts"][artifact])
        saved = torch.load(path, map_location="cpu", weights_only=False)
        state = saved[key]
        if not state or not all(torch.is_tensor(t) and torch.isfinite(t).all() for t in state.values()):
            raise ValueError(f"导出状态格式/有限数错误: {name}")
        count = sum(t.numel() for t in state.values()); elements += count
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        models[name] = dict(path=str(path), file=path.name, sha256=digest,
            format=saved["format"], state_elements=count,
            state_sha256=saved.get("model_sha256", saved.get("state_sha256")))
    if elements >= 500_000_000:
        raise ValueError("三模型状态元素上界达到500M")
    write_json(Path(context["run_dir"]) / "training_export.json", dict(
        format="package_training_export_v1", complete=True, models=models,
        state_elements_parameter_upper_bound=elements,
        inference=dict(algorithm="C_zero_native1024_v1", prediction_started=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text(encoding="utf-8"))
    activate_source()
    if args.stage == "finish":
        finish(context)
        return
    import torch
    if args.stage not in {"export_backend", "naf_export"} and not torch.cuda.is_available():
        raise RuntimeError("训练计算阶段需要CUDA版PyTorch，请使用sam2_env")
    if args.stage in {"lora_ssl", "stage1", "stage1_boundary", "joint", "semantic", "affinity"}:
        from repro.current_early import run_stage
    elif args.stage in {"d5a", "r1", "export_backend"}:
        from repro.current_backend import run_stage
    else:
        from repro.current_naf import run_stage
    result = run_stage(args.stage, context)
    from runner import write_json
    write_json(Path(context["run_dir"]) / "stages" / (args.stage + ".json"), result)


if __name__ == "__main__":
    main()
