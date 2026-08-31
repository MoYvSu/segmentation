# -*- coding: utf-8 -*-
"""Cold-start training reproduction entrypoint.

This module only orchestrates existing repository training programs.  Model,
loss, data, and inference implementations remain in their canonical modules.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: tuple[str, ...] | None
    requires: tuple[str, ...]
    produces: tuple[str, ...]


def python_command(*args: str) -> tuple[str, ...]:
    return (sys.executable, *args)


STAGES: tuple[Stage, ...] = (
    Stage(
        "prepare_labels",
        "从赛方 LabelMe 标注重建语义与净化边界 GT",
        python_command("tools/preprocess_labels.py"),
        ("data/raw",),
        ("data/purified_gt",),
    ),
    Stage(
        "lora_ssl",
        "在 1000 张无标签图上自监督预训练 SAM2 LoRA",
        python_command(
            "tools/pretrain_lora_ssl.py",
            "--config",
            "repro/configs/stage1_lora.yaml",
            "--epochs",
            "30",
            "--batch_size",
            "8",
            "--outdir",
            "outputs/lora_pretrain",
        ),
        (
            "weights/sam2_hiera_base_plus.pt",
            "segment-anything-2/sam2",
            "data/unlabeled",
        ),
        ("outputs/lora_pretrain/lora_state_dict.pth",),
    ),
    Stage(
        "stage1_lora",
        "监督训练 Stage-1 语义/边界双 FPN 与 LoRA",
        python_command("train.py", "--config", "repro/configs/stage1_lora.yaml"),
        (
            "outputs/lora_pretrain/lora_state_dict.pth",
            "data/raw",
            "data/purified_gt",
        ),
        ("outputs/stage1_lora/best_model.pth",),
    ),
    Stage(
        "stage1_pseudo",
        "由 Stage-1 生成无标签边界概率缓存",
        python_command(
            "tools/precompute_pseudo_labels.py",
            "--config",
            "repro/configs/stage1_lora.yaml",
            "--checkpoint",
            "outputs/stage1_lora/best_model.pth",
            "--output_dir",
            "outputs/pseudo_labels/stage1_boundary",
        ),
        ("outputs/stage1_lora/best_model.pth", "data/unlabeled"),
        ("outputs/pseudo_labels/stage1_boundary/boundary_probs.npy",),
    ),
    Stage(
        "joint_v3",
        "以 Stage-1 为起点进行 joint-v3 半监督联合微调",
        python_command(
            "train_stage2.py",
            "--config",
            "repro/configs/stage2_joint_v3.yaml",
            "--init_from_checkpoint",
            "outputs/stage1_lora/best_model.pth",
            "--phase",
            "joint",
            "--tag",
            "lora_v3_repro",
        ),
        (
            "outputs/stage1_lora/best_model.pth",
            "outputs/pseudo_labels/stage1_boundary/boundary_probs.npy",
        ),
        ("outputs/stage2_joint_v3/best_model_stage2.pth",),
    ),
    Stage(
        "semantic_pseudo",
        "由 joint-v3 语义梯度生成 V6 单线边界缓存",
        python_command(
            "tools/precompute_semantic_boundary.py",
            "--config",
            "repro/configs/stage2_joint_v3.yaml",
            "--checkpoint",
            "outputs/stage2_joint_v3/best_model_stage2.pth",
            "--outdir",
            "outputs/pseudo_labels/semantic_boundary",
        ),
        ("outputs/stage2_joint_v3/best_model_stage2.pth", "data/unlabeled"),
        ("outputs/pseudo_labels/semantic_boundary/boundary_probs.npy",),
    ),
    Stage(
        "v6",
        "冻结语义/LoRA，以语义单线缓存训练 V6 边界分支",
        python_command(
            "train_stage2.py",
            "--config",
            "repro/configs/stage2_v6.yaml",
            "--init_from_checkpoint",
            "outputs/stage2_joint_v3/best_model_stage2.pth",
            "--phase",
            "v6",
            "--tag",
            "semantic_label_repro",
        ),
        (
            "outputs/stage2_joint_v3/best_model_stage2.pth",
            "outputs/pseudo_labels/semantic_boundary/boundary_probs.npy",
        ),
        (
            "outputs/stage2_v6/best_model_stage2.pth",
        ),
    ),
    Stage(
        "e10a",
        "从 V6 冷启动完整高分辨率语义解码器",
        python_command(
            "train_stage2.py",
            "--config",
            "config/train/stage2_semantic_e10a_cold20.yaml",
            "--phase",
            "semantic",
            "--tag",
            "e10a_repro",
        ),
        (
            "outputs/stage2_v6/best_model_stage2.pth",
            "config/monitor/unlabeled_holdout_v1.txt",
        ),
        ("outputs/stage2_semantic_e10a_cold20/best_model_stage2.pth",),
    ),
    Stage(
        "affinity_g0",
        "以 V6 边界 FPN 初始化两图 affinity 审计基座",
        python_command(
            "train_affinity_geometry_g0.py",
            "--config",
            "config/train/affinity_geometry_g0.yaml",
        ),
        ("outputs/stage2_v6/best_model_stage2.pth", "data/raw"),
        ("outputs/affinity_geometry_g0/latest_affinity.pth",),
    ),
    Stage(
        "affinity_g0_long",
        "延长 G0 affinity 基座训练",
        python_command(
            "train_affinity_geometry_g0.py",
            "--config",
            "config/train/affinity_geometry_g0_long.yaml",
        ),
        ("outputs/affinity_geometry_g0/latest_affinity.pth",),
        ("outputs/affinity_geometry_g0_long/latest_affinity.pth",),
    ),
    Stage(
        "affinity_g1",
        "在人工实例数据上训练 G1 affinity",
        python_command(
            "train_affinity_geometry_g1.py",
            "--config",
            "config/train/affinity_geometry_g1.yaml",
        ),
        ("outputs/affinity_geometry_g0_long/latest_affinity.pth", "data/raw"),
        ("outputs/affinity_geometry_g1/best_affinity.pth",),
    ),
    Stage(
        "sam2_geometry_review",
        "确认 G2 使用的无类别 SAM2 几何数据已经人工审核",
        None,
        (
            "data/sam2_geometry_g2/manifest.jsonl",
            "data/sam2_geometry_g2/masks",
            "data/sam2_geometry_g2/approval.json",
        ),
        (),
    ),
    Stage(
        "affinity_g2",
        "混合人工实例与已审核 SAM2 几何数据训练 G2",
        python_command(
            "train_affinity_geometry_g1.py",
            "--config",
            "config/train/affinity_geometry_g2_sam2.yaml",
        ),
        (
            "outputs/affinity_geometry_g1/best_affinity.pth",
            "data/sam2_geometry_g2/manifest.jsonl",
            "data/sam2_geometry_g2/approval.json",
        ),
        ("outputs/affinity_geometry_g2_sam2/best_affinity.pth",),
    ),
    Stage(
        "affinity_g4b",
        "从 G2 重跑人工缺口降权 0.20 的 G4b 部署几何",
        python_command(
            "train_affinity_geometry_g1.py",
            "--config",
            "config/train/affinity_geometry_g4b_gap_weight020.yaml",
        ),
        (
            "outputs/affinity_geometry_g2_sam2/best_affinity.pth",
            "data/sam2_geometry_g2/manifest.jsonl",
            "data/sam2_geometry_g2/approval.json",
        ),
        ("outputs/affinity_geometry_g4b_gap_weight020/latest_affinity.pth",),
    ),
)


STAGE_BY_NAME = {stage.name: stage for stage in STAGES}
CONFIGS = (
    "repro/configs/stage1_lora.yaml",
    "repro/configs/stage2_joint_v3.yaml",
    "repro/configs/stage2_v6.yaml",
    "config/train/stage2_semantic_e10a_cold20.yaml",
    "config/train/affinity_geometry_g0.yaml",
    "config/train/affinity_geometry_g0_long.yaml",
    "config/train/affinity_geometry_g1.yaml",
    "config/train/affinity_geometry_g2_sam2.yaml",
    "config/train/affinity_geometry_g4b_gap_weight020.yaml",
)


def root_path(relative: str) -> Path:
    return ROOT / relative


def render_command(command: Sequence[str] | None) -> str:
    if command is None:
        return "[manual approval gate]"
    rendered = ["python" if item == sys.executable else item for item in command]
    return " ".join(rendered)


def selected_stages(first: str | None, last: str | None) -> tuple[Stage, ...]:
    names = [stage.name for stage in STAGES]
    start = names.index(first) if first else 0
    end = names.index(last) if last else len(names) - 1
    if start > end:
        raise ValueError(f"from-stage {names[start]} is after to-stage {names[end]}")
    return STAGES[start : end + 1]


def existing_outputs(stage: Stage) -> list[str]:
    return [path for path in stage.produces if root_path(path).exists()]


def missing_paths(paths: Iterable[str]) -> list[str]:
    return [path for path in paths if not root_path(path).exists()]


def validate_approval() -> None:
    dataset_dir = root_path("data/sam2_geometry_g2")
    path = dataset_dir / "approval.json"
    manifest_path = dataset_dir / "manifest.jsonl"
    try:
        approval = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid SAM2 geometry approval: {path}: {exc}") from exc
    if approval.get("approved") is not True:
        raise RuntimeError("SAM2 geometry approval.json must contain approved=true")
    if not str(approval.get("reviewed_by", "")).strip():
        raise RuntimeError("SAM2 geometry approval.json requires reviewed_by")
    if not str(approval.get("reviewed_at", "")).strip():
        raise RuntimeError("SAM2 geometry approval.json requires reviewed_at")
    try:
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid SAM2 geometry manifest: {manifest_path}: {exc}") from exc
    if not 1 <= len(rows) <= 249:
        raise RuntimeError(
            f"SAM2 geometry manifest source count must be within 1..249, got {len(rows)}"
        )
    hashes = [str(row.get("source_sha256", "")) for row in rows]
    if any(not digest for digest in hashes) or len(hashes) != len(set(hashes)):
        raise RuntimeError("SAM2 geometry manifest source hashes are missing or duplicated")
    if any(row.get("class_label") is not None for row in rows):
        raise RuntimeError("SAM2 geometry manifest must keep class_label=null")
    if int(approval.get("source_count", -1)) != len(rows):
        raise RuntimeError(
            "SAM2 geometry approval source_count does not match manifest rows"
        )
    masks = list((dataset_dir / "masks").glob("*.npz"))
    if len(masks) != len(rows):
        raise RuntimeError(
            f"SAM2 geometry masks={len(masks)} do not match manifest rows={len(rows)}"
        )


def check_sources() -> int:
    failures: list[str] = []
    for path in (
        "weights/sam2_hiera_base_plus.pt",
        "segment-anything-2/sam2",
        "data/raw",
        "data/unlabeled",
        "requirements.txt",
    ):
        if not root_path(path).exists():
            failures.append(path)

    for config_path in CONFIGS:
        if not root_path(config_path).is_file():
            failures.append(config_path)

    for stage in STAGES:
        if stage.command is None:
            continue
        entrypoint = stage.command[1]
        if not root_path(entrypoint).is_file():
            failures.append(f"{stage.name}: missing entrypoint {entrypoint}")

    if not failures:
        sys.path.insert(0, str(ROOT))
        from utils.config import load_config

        for config_path in CONFIGS:
            config = load_config(str(root_path(config_path)))
            if Path(config["paths"]["project_root"]).resolve() != ROOT.resolve():
                failures.append(f"{config_path}: paths.project_root does not resolve to repo")

    raw_json = list(root_path("data/raw").glob("*.json"))
    raw_images = list(root_path("data/raw").glob("*.jpg"))
    unlabeled_images = list(root_path("data/unlabeled").glob("*.jpg"))
    if len(raw_json) != 32 or len(raw_images) != 32:
        failures.append(
            f"data/raw expected 32 JSON + 32 JPG, got {len(raw_json)} + {len(raw_images)}"
        )
    if len(unlabeled_images) != 1000:
        failures.append(
            f"data/unlabeled expected 1000 JPG, got {len(unlabeled_images)}"
        )

    if failures:
        print("PRECHECK FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("PRECHECK OK")
    print("  source data: 32 labeled images + 1000 unlabeled images")
    print("  configs: portable and loadable")
    approval = root_path("data/sam2_geometry_g2/approval.json")
    if approval.exists():
        try:
            validate_approval()
        except RuntimeError as exc:
            print(f"  SAM2 geometry: INVALID ({exc})")
        else:
            print("  SAM2 geometry: approved")
    else:
        print("  SAM2 geometry: pending generation and human approval before affinity_g2")
    return 0


def run_stages(
    stages: Sequence[Stage], *, dry_run: bool, skip_existing: bool
) -> int:
    if check_sources() != 0:
        return 1
    for stage in stages:
        print(f"\n[{stage.name}] {stage.description}")
        print(f"  {render_command(stage.command)}")
        if dry_run:
            continue

        missing = missing_paths(stage.requires)
        if missing:
            raise RuntimeError(
                f"stage {stage.name} missing prerequisites: {', '.join(missing)}"
            )

        if stage.name == "sam2_geometry_review":
            validate_approval()
            print("  approval accepted")
            continue

        existing = existing_outputs(stage)
        if existing:
            if skip_existing and len(existing) == len(stage.produces):
                print(f"  skipped existing output: {', '.join(existing)}")
                continue
            raise RuntimeError(
                f"stage {stage.name} output already exists: {', '.join(existing)}; "
                "use --skip-existing to reuse completed stages"
            )

        assert stage.command is not None
        subprocess.run(stage.command, cwd=ROOT, check=True)
        missing_outputs = missing_paths(stage.produces)
        if missing_outputs:
            raise RuntimeError(
                f"stage {stage.name} completed without expected output: "
                f"{', '.join(missing_outputs)}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list", help="列出训练阶段和命令")
    subparsers.add_parser("check", help="只做冷启动输入与配置预检")
    run_parser = subparsers.add_parser("run", help="按顺序执行训练阶段")
    choices = tuple(stage.name for stage in STAGES)
    run_parser.add_argument("--from-stage", choices=choices)
    run_parser.add_argument("--to-stage", choices=choices)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="复用已经完整生成的阶段产物；默认拒绝覆盖",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "list":
        for index, stage in enumerate(STAGES, start=1):
            print(f"{index:02d} {stage.name:24s} {stage.description}")
            print(f"   {render_command(stage.command)}")
        return 0
    if args.action == "check":
        return check_sources()
    try:
        stages = selected_stages(args.from_stage, args.to_stage)
        return run_stages(
            stages, dry_run=args.dry_run, skip_existing=args.skip_existing
        )
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
