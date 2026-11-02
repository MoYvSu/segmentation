# -*- coding: utf-8 -*-
"""13阶段fresh训练编排；输入检查、阶段日志及本轮三模型导出。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
TRAINING = PACKAGE / "training"
STAGES = ("lora_ssl", "stage1", "stage1_boundary", "joint", "semantic", "affinity",
          "d5a", "r1", "export_backend", "n1_cache", "naf_train", "naf_export", "finish")
CUDA_STAGES = set(STAGES) - {"export_backend", "naf_export", "finish"}
CHECKS = {
    "运行": ["GPU计算阶段使用CUDA", "输出目录新建，已有目录保留"],
    "训练数据": ["32份原图/LabelMe及purified/r8目标", "1000训练原图", "64份已授权无类别SAM2几何",
               "原图/目标尺寸一致", "P=0/F=1，未知及padding保持ignore"],
    "模型": ["官方SAM2底座文件", "任务头随机初始化", "checkpoint结构严格加载",
           "joint/S-align/V共享encoder与LoRA兼容", "三模型参数总量小于500M"],
    "更新": ["loss/梯度有限", "R1仅affinity获得梯度，冻结模块保持", "人工/无类别流采用同一whole/native视野"],
    "NAF": ["输入由本轮backend/restorer局部零噪N1生成", "原尺寸FP32边界概率及二值T输入",
           "GT/缓存逐源对齐", "tile/overlap/reflect与概率拼接一致", "G/D按配方更新并严格导出G"],
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def context(config_path, output):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("assets"), dict):
        raise ValueError("训练配置需要assets映射")
    root = TRAINING / "source"
    resolve = lambda value: str((PACKAGE / value).resolve()) if not Path(value).is_absolute() else str(Path(value).resolve())
    assets = {key: resolve(value) for key, value in config["assets"].items()}
    assets["monitor_dir"] = assets["raw_dir"]
    settings = dict(config, semantic_teacher_source="joint")
    stages = settings.get("stages", {})
    if any("import_checkpoint" in value or "completed_joint_continuation" in value for value in stages.values()):
        raise ValueError("本入口执行fresh训练，请使用本轮新输出目录")
    d5a_spec = yaml.safe_load((root / stages["d5a"]["config"]).read_text(encoding="utf-8"))
    epochs = int(d5a_spec["rgb_restoration"]["train"]["epochs"])
    relative = dict(ssl="lora_ssl/lora_state_dict.pth", stage1="stage1/best_model.pth",
        stage1_boundary="stage1_boundary/boundary_probs.npy", joint="joint/best_model_stage2.pth",
        semantic="semantic/best_model_stage2.pth", affinity="affinity/best_affinity.pth",
        d5a=f"d5a/epoch_{epochs:03d}.pt", r1="r1/final.pt", backend="models/backend.pt",
        restorer="models/restorer.pt", n1_cache="n1_cache", naf="naf_train", naf_export="models/naf.pt")
    return dict(format="current_training_context_v1", project_root=str(root.resolve()),
        run_dir=str(output), assets=assets, settings=settings,
        artifacts={key: str(output / value) for key, value in relative.items()})


def check_inputs(ctx, *, cuda=True):
    from PIL import Image
    import numpy as np
    errors = []
    assets = {key: Path(value) for key, value in ctx["assets"].items()}
    for key, path in assets.items():
        if not path.exists():
            errors.append(f"缺少{key}: {path}")
    for name in ("repro/current_early.py", "repro/current_backend.py", "repro/current_naf.py",
                 "train.py", "train_stage2.py", "tools/pretrain_lora_ssl.py"):
        if not (Path(ctx["project_root"]) / name).is_file():
            errors.append(f"缺少训练源码: {name}")
    if errors:
        return errors
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = lambda folder: sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    raw, unlabeled = images(assets["raw_dir"]), images(assets["unlabeled_dir"])
    if len(raw) != 32 or len({p.stem for p in raw}) != 32:
        errors.append("原人工源应为32张唯一图像")
    if len(unlabeled) != 1000:
        errors.append("无标签训练源应为1000张")
    for image in raw:
        files = [image.with_suffix(".json"), assets["purified_gt_dir"] / (image.stem + "_gt.npz"),
                 assets["completed_gt_dir"] / (image.stem + "_gt.npz"),
                 assets["completed_gt_dir"] / (image.stem + "_class.json")]
        if any(not p.is_file() for p in files):
            errors.append(f"缺少源图的标注/目标: {image.stem}")
            continue
        with Image.open(image) as handle:
            shape = (handle.height, handle.width)
        with np.load(files[2], allow_pickle=False) as arrays:
            ids = arrays["instance_map"]
            if ids.ndim != 2 or ids.shape != shape or ids.dtype.kind not in "ui":
                errors.append(f"实例目标尺寸或dtype错误: {image.stem}")
            classes = json.loads(files[3].read_text(encoding="utf-8"))
            positive = {str(int(i)) for i in np.unique(ids) if i > 0}
            if not positive <= set(classes) or any(type(v) is not int or v not in (0, 1) for v in classes.values()):
                errors.append(f"P0/F1类别表与正实例不匹配: {image.stem}")
            for key in ("original_covered", "filled", "residual_unknown"):
                if key not in arrays or arrays[key].shape != shape or not np.isin(arrays[key], (0, 1)).all():
                    errors.append(f"r8已知/未知域格式错误: {image.stem}/{key}")
    manifest = assets["sam2_geometry_dir"] / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 64 or any(row.get("class_label") is not None for row in rows):
        errors.append("SAM2几何应为64源且不带类别")
    for row in rows:
        source = assets["sam2_source_dir"] / row["source_relpath"]
        mask = assets["sam2_geometry_dir"] / row["mask_file"]
        if not source.is_file() or not mask.is_file():
            errors.append(f"SAM2源图或mask缺失: {row.get('source_relpath')}")
    if not (assets["sam2_repo"] / "sam2").is_dir():
        errors.append("SAM2源码目录需要sam2包")
    if cuda:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA不可用，请使用sam2_env及CUDA版PyTorch")
    return errors


def run(ctx):
    output = Path(ctx["run_dir"])
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请选择新目录: {output}")
    errors = check_inputs(ctx)
    if errors:
        raise ValueError("\n".join(errors))
    output.mkdir(parents=True)
    write_json(output / "training_context.json", ctx)
    started = time.monotonic()
    completed = []
    for stage in STAGES:
        print(f"[{stage}]", flush=True)
        command = [sys.executable, str(TRAINING / "worker.py"), "--stage", stage,
                   "--context", str(output / "training_context.json")]
        log = output / "logs" / (stage + ".log")
        log.parent.mkdir(exist_ok=True)
        then = time.monotonic()
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=PACKAGE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                     "SEGMENTATION_PROJECT_ROOT": ctx["project_root"]})
            for line in process.stdout:
                stream.write(line); stream.flush(); print(line, end="", flush=True)
            code = process.wait()
        if code:
            write_json(output / "training_status.json", dict(complete=False, stage=stage, exit_code=code,
                completed_stages=completed, elapsed_seconds=time.monotonic()-started))
            raise RuntimeError(f"训练阶段退出: {stage}，详见{log}")
        completed.append(stage)
        write_json(output / "training_status.json", dict(complete=stage == STAGES[-1], stage=stage,
            completed_stages=completed, stage_seconds=time.monotonic()-then,
            elapsed_seconds=time.monotonic()-started, updated_at=datetime.now(timezone.utc).isoformat()))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(TRAINING / "config.yaml"))
    parser.add_argument("--output-dir", default="outputs/training")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = (PACKAGE / args.output_dir).resolve()
    ctx = context(Path(args.config).resolve(), output)
    if args.plan or args.check:
        errors = check_inputs(ctx) if args.check else []
        print(json.dumps(dict(stages=list(STAGES), output_dir=str(output), assets=ctx["assets"],
                              checks=CHECKS, errors=errors), ensure_ascii=False, indent=2))
        return int(bool(errors))
    return run(ctx)
