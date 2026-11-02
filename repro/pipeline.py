# -*- coding: utf-8 -*-
"""当前三模型训练链的统一入口；plan/check/dry-run不加载模型、不训练、不改GT。

旧repro/train.py继续描述旧E10a/G4b链。本入口使用隔离输出、本轮实际依赖和完成回执，
不把历史单变量消融的control日志或固定旧checkpoint SHA当作新训练的先决条件。
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config

FORMAT = "current_training_pipeline_v1"
CONTEXT_FORMAT = "current_training_context_v1"
RECEIPT_FORMAT = "current_training_stage_receipt_v1"
LEGACY_SAM_SHA = "4ed3e2e1fcfd2fa369dd647fc5ce7c869c5bcb7f472e29398ee889b368e40749"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    dependencies: tuple[str, ...]
    description: str


LEGACY_STAGES = (
    Stage("lora_ssl", "repro.current_early", (), "原浅层RGB重建SSL；保留历史计算路径"),
    Stage("stage1", "repro.current_early", ("lora_ssl",), "人工监督Stage1双FPN及LoRA"),
    Stage("stage1_boundary", "repro.current_early", ("stage1",), "Stage1无标签边界缓存"),
    Stage("joint", "repro.current_early", ("stage1", "stage1_boundary"), "joint-v3共享特征训练"),
    Stage("semantic_boundary", "repro.current_early", ("joint",), "joint语义单线边界缓存"),
    Stage("v6", "repro.current_early", ("joint", "semantic_boundary"), "V6语义教师载体，保留固定e9"),
    Stage("semantic", "repro.current_early", ("joint", "v6", "semantic_boundary"), "S-align完整语义头冷启动"),
    Stage("affinity", "repro.current_early", ("joint",), "V affinity，从joint边界FPN开始"),
    Stage("d5a", "repro.current_backend", (), "独立D5a RGB恢复60轮"),
    Stage("r1", "repro.current_backend", ("joint", "semantic", "affinity", "d5a"), "R1 mixed balance，只更新affinity"),
    Stage("export_backend", "repro.current_backend", ("r1", "d5a"), "严格导出本轮backend/restorer"),
    Stage("n1_cache", "repro.current_naf", ("export_backend",), "本轮训练源真实局部零噪N1缓存及GT封装"),
    Stage("naf_train", "repro.current_naf", ("n1_cache",), "NAF width随机初始化1000步"),
    Stage("naf_export", "repro.current_naf", ("naf_train",), "导出本轮NAF G，移除训练状态"),
    Stage("verify_export", "repro.pipeline", ("export_backend", "naf_export"), "核对三份导出、参数预算及部署身份"),
)
STAGES = tuple(Stage(s.name, s.module, ("joint",) if s.name == "semantic" else s.dependencies,
                     "S-align：joint固定语义教师＋完整语义头冷启动" if s.name == "semantic" else s.description)
               for s in LEGACY_STAGES if s.name not in {"semantic_boundary", "v6"})
BY_NAME = {stage.name: stage for stage in STAGES}
LEGACY_BY_NAME = {stage.name: stage for stage in LEGACY_STAGES}


def stages_for(context: dict):
    mode = context.get("settings", {}).get("semantic_teacher_source", "joint")
    if mode not in {"joint", "legacy_v6_e9"}:
        raise ValueError("semantic_teacher_source必须为joint或显式legacy_v6_e9")
    return LEGACY_STAGES if mode == "legacy_v6_e9" else STAGES


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_sha(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
        return digest.hexdigest()


def write_json(path: Path, value, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def artifact_paths(run: Path, *, legacy_v6=False) -> dict[str, str]:
    relative = {
        "ssl": "lora_ssl/lora_state_dict.pth", "stage1": "stage1/best_model.pth",
        "stage1_boundary": "stage1_boundary/boundary_probs.npy", "joint": "joint/best_model_stage2.pth",
        "semantic_boundary": "semantic_boundary/boundary_probs.npy", "v6": "v6/stage2_epoch9.pth",
        "semantic": "semantic/best_model_stage2.pth", "affinity": "affinity/best_affinity.pth",
        "d5a": "d5a/epoch_060.pt", "r1": "r1/final.pt",
        "backend": "export_backend/model/backend.pt", "restorer": "export_backend/model/restorer.pt",
        "n1_cache": "n1_cache", "naf": "naf_train", "naf_export": "naf_export/model/naf.pt",
    }
    return {name: str(run / value) for name, value in relative.items()
            if legacy_v6 or name not in {"semantic_boundary", "v6"}}


def build_context(config: str | Path, run_dir: str | Path) -> dict:
    settings = load_config(str(config))
    if settings.get("format") != FORMAT:
        raise ValueError(f"配置format必须为{FORMAT}")
    root = Path(settings["paths"]["project_root"]).resolve()
    run = resolve(root, run_dir)
    # 本入口只向明确隔离的私有产物目录写入，不覆盖代码、原始数据或FS归档。
    if not any(run.is_relative_to(root / base) and run != root / base for base in ("output", "outputs")):
        raise ValueError("run-dir须是本项目output/或outputs/下的独立子目录")
    assets = {name: (str(resolve(root, value)) if value is not None else None)
              for name, value in settings["assets"].items()}
    required = {"raw_dir", "unlabeled_dir", "purified_gt_dir", "completed_gt_dir", "completed_gt_manifest",
                "sam2_geometry_dir", "sam2_source_dir", "sam2_authorization_doc", "backbone", "sam2_repo", "monitor_dir"}
    if required - assets.keys() or any(not assets[name] for name in required):
        raise ValueError("资产映射缺少必需字段: " + ",".join(sorted(required - assets.keys())))
    for value in assets.values():
        if value and (Path(value) == run or Path(value).is_relative_to(run) or run.is_relative_to(Path(value))):
            raise ValueError(f"输入资产与本轮输出目录重叠: {value}")
    selected = stages_for({"settings": settings})
    allowed = {s.name for s in selected}
    for name in settings.get("stages", {}):
        if name not in allowed:
            raise ValueError(f"未知stage配置: {name}")
        if settings["stages"][name].get("import_checkpoint") and name not in ("lora_ssl", "d5a"):
            raise ValueError("只有不依赖分割GT的SSL/D5a可以显式导入；监督阶段必须本轮重训")
        continuation = settings["stages"][name].get("completed_joint_continuation")
        if continuation and (name != "joint" or not isinstance(continuation, dict)):
            raise ValueError("completed_joint_continuation只允许用于已完成的joint")
    return {"format": CONTEXT_FORMAT, "project_root": str(root), "run_dir": str(run),
            "settings": settings, "assets": assets,
            "artifacts": artifact_paths(run, legacy_v6=settings.get("semantic_teacher_source") == "legacy_v6_e9")}


def source_snapshot(root: Path) -> dict:
    """包括本轮未提交源码及SAM2源码，不把数据/权重/私有产物列入源码。"""
    files = set(root.glob("*.py"))
    for folder in ("models", "utils", "tools", "losses", "repro", "config", "segment-anything-2/sam2"):
        directory = root / folder
        if directory.exists():
            files.update(path for path in directory.rglob("*") if path.suffix in (".py", ".yaml", ".yml") and path.is_file())
    files.update((root / "data").glob("*.py"))
    for name in ("requirements.txt", "AGENTS.md"):
        if (root / name).is_file():
            files.add(root / name)
    result = {path.relative_to(root).as_posix(): file_sha(path) for path in sorted(files)}
    return {"sha256": canonical_sha(result), "files": result}


def snapshot(path: Path) -> dict:
    if path.is_file():
        return {"kind": "file", "bytes": path.stat().st_size, "sha256": file_sha(path)}
    if path.is_dir():
        files = {child.relative_to(path).as_posix(): {"bytes": child.stat().st_size, "sha256": file_sha(child)}
                 for child in sorted(path.rglob("*")) if child.is_file()}
        if not files:
            raise ValueError(f"产物目录为空: {path}")
        return {"kind": "directory", "sha256": canonical_sha(files), "files": files}
    raise FileNotFoundError(path)


def images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)


def check_assets(context: dict, *, _verify_continuation=True) -> dict:
    assets, expected = context["assets"], context["settings"]["expected"]
    errors, files = [], {}
    for name, value in assets.items():
        if value is not None and not Path(value).exists():
            errors.append(f"缺少资产{name}: {value}")
    if not Path(assets["sam2_repo"], "sam2").is_dir():
        errors.append("sam2_repo下缺少sam2源码")
    if errors:
        return {"passed": False, "errors": errors}

    def record(path):
        path = Path(path)
        value = {"bytes": path.stat().st_size, "sha256": file_sha(path)}
        files[str(path)] = value
        return value["sha256"]

    # 显式映射到项目外的SAM2实现也必须封存，不能仅检查目录存在。
    for path in sorted(Path(assets["sam2_repo"], "sam2").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
            record(path)

    raw, unlabeled = images(Path(assets["raw_dir"])), images(Path(assets["unlabeled_dir"]))
    labels = sorted(Path(assets["raw_dir"]).glob("*.json"))
    if len(raw) != expected["manual_sources"] or len(labels) != len(raw):
        errors.append(f"人工源应为{expected['manual_sources']}对，当前图像{len(raw)}/JSON{len(labels)}")
    if len(unlabeled) != expected["unlabeled_sources"]:
        errors.append(f"无标签源应为{expected['unlabeled_sources']}张，当前{len(unlabeled)}")
    if len({path.stem for path in raw}) != len(raw) or {path.stem for path in labels} != {path.stem for path in raw}:
        errors.append("人工图像/JSON没有唯一、一一对应的stem")
    raw_hashes = {record(path) for path in raw}
    unlabeled_hashes = {record(path) for path in unlabeled}
    if len(unlabeled_hashes) != len(unlabeled):
        errors.append("无标签源存在重复内容")
    if not raw_hashes <= unlabeled_hashes:
        errors.append("有人工源图无法按内容SHA对应到允许的1000训练源")
    for path in labels:
        record(path)
        read_json(path)

    if record(assets["backbone"]) != expected["backbone_sha256"]:
        errors.append("官方backbone SHA与当前配方不符")
    completed = read_json(Path(assets["completed_gt_manifest"]))
    record(assets["completed_gt_manifest"])
    stems = {path.stem for path in raw}
    if set(completed.get("samples", [])) != stems or completed.get("sample_count") != len(stems):
        errors.append("completed GT manifest与人工训练源不一致")
    for stem in sorted(stems):
        for directory, suffix in ((assets["completed_gt_dir"], "_gt.npz"),
                                  (assets["completed_gt_dir"], "_class.json"),
                                  (assets["purified_gt_dir"], "_gt.npz")):
            path = Path(directory) / (stem + suffix)
            if not path.is_file():
                errors.append(f"缺少逐源GT: {path}")
                continue
            record(path)
            if suffix == "_class.json":
                classes = read_json(path)
                if not classes or any(int(key) <= 0 or value not in (0, 1) for key, value in classes.items()):
                    errors.append(f"实例类别契约错误: {path}")

    sam_manifest = Path(assets["sam2_geometry_dir"]) / "manifest.jsonl"
    if not sam_manifest.is_file():
        errors.append(f"缺少SAM2 manifest: {sam_manifest}")
    else:
        sam_sha = record(sam_manifest)
        rows = [json.loads(line) for line in sam_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        source_hashes = [row.get("source_sha256") for row in rows]
        if (len(rows) != expected["sam2_sources"] or len(set(source_hashes)) != len(rows)
                or any(not value for value in source_hashes) or any(row.get("class_label") is not None for row in rows)):
            errors.append("SAM2几何必须是64个唯一、class_label=null的训练源")
        if sam_sha != expected["sam2_manifest_sha256"]:
            errors.append("SAM2 manifest不符；不能冒用既有64源授权")
        if sam_sha != LEGACY_SAM_SHA:
            errors.append("当前配方只复用已授权原64源；新候选须另立版本及审核合同")
        if set(source_hashes) & raw_hashes:
            errors.append("SAM2源与人工源内容重叠")
        if not set(source_hashes) <= unlabeled_hashes:
            errors.append("SAM2源不全部来自允许的1000源")
        record(assets["sam2_authorization_doc"])
        for row in rows:
            relative = Path(str(row.get("source_relpath", "")))
            source = resolve(Path(assets["sam2_source_dir"]), relative)
            mask = resolve(Path(assets["sam2_geometry_dir"]), row["mask_file"])
            if not source.is_relative_to(Path(assets["sam2_source_dir"])) or not mask.is_relative_to(Path(assets["sam2_geometry_dir"])):
                errors.append("SAM2 manifest路径越出显式资产目录")
                continue
            if not source.is_file() or record(source) != row["source_sha256"]:
                errors.append(f"SAM2原图缺失或SHA错误: {source}")
            if not mask.is_file():
                errors.append(f"SAM2 mask缺失: {mask}")
            else:
                record(mask)
    if assets.get("naf_target_manifest"):
        record(assets["naf_target_manifest"])
    teacher_dir, teacher_manifest = assets.get("teacher_gt_dir"), assets.get("teacher_gt_manifest")
    if bool(teacher_dir) != bool(teacher_manifest):
        errors.append("teacher_gt_dir与teacher_gt_manifest必须同时指定")
    if teacher_dir:
        if (Path(teacher_dir).resolve() != Path(assets["completed_gt_dir"]).resolve()
                or Path(teacher_manifest).resolve() != Path(assets["completed_gt_manifest"]).resolve()):
            errors.append("本轮完全替换要求teacher和completed资产指向同一版本")
        if completed.get("format") != "teacher_supervision_v1":
            errors.append("teacher模式必须使用独立teacher_supervision_v1，不能冒用人工/r8元数据")
        boundary_policy = context["settings"].get("teacher_boundary_policy", "known_neighbors_v1")
        if completed.get("boundary_policy", "known_neighbors_v1") != boundary_policy:
            errors.append("教师训练合同的boundary_policy与本轮派生规则不符")
        record(teacher_manifest)
    for name, options in context["settings"].get("stages", {}).items():
        if not options.get("import_checkpoint"):
            continue
        if name not in ("lora_ssl", "d5a"):
            errors.append("禁止导入依赖旧分割GT的监督成品")
            continue
        imported = resolve(Path(context["project_root"]), options["import_checkpoint"])
        if not imported.is_file() or record(imported) != options.get("import_sha256"):
            errors.append(f"{name}导入权重缺失或SHA不符: {imported}")
        origin = options.get("import_provenance")
        if not isinstance(origin, dict) or origin.get("segmentation_GT_used") is not False:
            errors.append(f"{name}导入必须显式声明原训练来源及不依赖分割GT")
    if _verify_continuation and context["settings"].get("stages", {}).get("joint", {}).get("completed_joint_continuation"):
        try:
            validate_joint_continuation(context, current_assets_sha=canonical_sha(files))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            errors.append("joint接续来源无效: " + str(exc))
    return {"passed": not errors, "errors": errors, "files": files, "sha256": canonical_sha(files),
            "manual_sources": len(raw), "unlabeled_sources": len(unlabeled),
            "sam2_authorization": "same_manifest_existing_authorization_no_new_approval_invented",
            "GT_modified": False}


def check_code(context: dict) -> dict:
    root = Path(context["project_root"])
    errors, configurations = [], {}
    for module in sorted({stage.module for stage in stages_for(context)}):
        path = root.joinpath(*module.split(".")).with_suffix(".py")
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(str(exc))
    for stage in stages_for(context):
        path = context["settings"].get("stages", {}).get(stage.name, {}).get("config")
        if path:
            try:
                parsed = load_config(str(resolve(root, path)))
                if Path(parsed["paths"]["project_root"]).resolve() != root:
                    errors.append(f"{stage.name}配置解析到了别的项目根目录")
                configurations[stage.name] = canonical_sha(parsed)
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{stage.name}: {exc}")
    plans = {}
    try:
        from repro.current_early import stages_for as early_stages, build_stage
        for stage in early_stages(context):
            plan = build_stage(stage, context)
            plans[stage] = {"command": plan["command"], "expanded_config_sha256": canonical_sha(plan["config"]),
                            "seed": plan["seed"], "notes": plan["notes"]}
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"早期配置衔接: {exc}")
    return {"passed": not errors, "errors": errors, "configs": configurations, "early_plans": plans,
            "model_loaded": False, "training_started": False}


def stage_outputs(context: dict, name: str) -> tuple[Path, ...]:
    artifacts, run = context["artifacts"], Path(context["run_dir"])
    keys = {"lora_ssl": ("ssl",), "stage1": ("stage1",), "stage1_boundary": ("stage1_boundary",),
            "joint": ("joint",), "semantic_boundary": ("semantic_boundary",), "v6": ("v6",),
            "semantic": ("semantic",), "affinity": ("affinity",), "d5a": ("d5a",), "r1": ("r1",),
            "export_backend": ("backend", "restorer"), "n1_cache": ("n1_cache",),
            "naf_export": ("naf_export",)}
    if name == "naf_train":
        return (Path(artifacts["naf"]) / "latest.pt", Path(artifacts["naf"]) / "complete.json")
    if name == "d5a":
        # R1实际读取此退化配置，必须与恢复checkpoint共同封存。
        return (Path(artifacts["d5a"]), run / "configs/d5a.yaml")
    if name in ("stage1_boundary", "semantic_boundary"):
        # 数据加载还读取names.txt/exclude.txt等索引，必须和概率数组一起封存。
        return (Path(artifacts[name]).parent,)
    if name == "verify_export":
        return (run / "verify_export" / "manifest.json",)
    return tuple(Path(artifacts[key]) for key in keys[name])


def selection(first=None, last=None, *, context=None) -> tuple[Stage, ...]:
    stages = stages_for(context or {})
    names = [s.name for s in stages]
    start, end = names.index(first) if first else 0, names.index(last) if last else len(names) - 1
    if start > end:
        raise ValueError("from-stage不能晚于to-stage")
    return stages[start:end + 1]


def execution_identity(context: dict, source: dict, assets: dict) -> dict:
    packages = {}
    for package in ("torch", "torchvision", "numpy", "opencv-python", "scipy", "scikit-image",
                    "PyYAML", "hydra-core", "omegaconf", "timm"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"context_sha256": canonical_sha(context), "source_sha256": source["sha256"],
            "assets_sha256": assets["sha256"], "python": str(Path(sys.executable).resolve()),
            "python_version": sys.version, "platform": sys.platform, "packages": packages,
            "stage_configs_sha256": canonical_sha(check_code(context).get("configs", {}))}


def stage_inputs(context: dict, stage: Stage, identity: dict | None = None) -> dict:
    if stage.name == "joint" and context["settings"].get("stages", {}).get("joint", {}).get("completed_joint_continuation"):
        return {"completed_joint_continuation": validate_joint_continuation(context)}
    dependencies = {}
    for name in stage.dependencies:
        receipt_path = Path(context["run_dir"]) / "receipts" / (name + ".json")
        receipt = read_json(receipt_path)
        if receipt.get("status") != "complete" or receipt.get("format") != RECEIPT_FORMAT:
            raise RuntimeError(f"前置阶段未完整完成: {name}")
        if identity is not None and receipt.get("identity") != identity:
            raise RuntimeError(f"前置阶段不是本轮同源配置/源码/资产: {name}")
        outputs = {str(path): snapshot(path) for path in stage_outputs(context, name)}
        if receipt.get("outputs") != outputs:
            raise RuntimeError(f"前置阶段产物变更: {name}")
        dependencies[name] = {"receipt_sha256": file_sha(receipt_path), "outputs": outputs}
    return dependencies


def verify_export(context: dict):
    """最终身份验收只用CPU；部署效果和GPU完整输出仍须后续独立验收。"""
    import torch
    artifacts = context["artifacts"]
    records, total = {}, 0
    for name, field in (("backend", "model_state_dict"), ("restorer", "model"), ("naf_export", "model_state_dict")):
        path = Path(artifacts[name])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get(field)
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError(f"{name}缺少完整模型状态")
        # 状态元素数包含buffer，是参数数目的保守上界，不冒称精确可训练参数数目。
        elements = sum(value.numel() for value in state.values())
        total += elements
        records[name] = {"path": str(path), "file_sha256": file_sha(path), "state_tensors": len(state),
                         "state_elements_parameter_upper_bound": elements, "format": payload.get("format")}
    if total >= 500_000_000:
        raise ValueError("三模型状态元素上界达到500M，必须核实参数预算")
    result = {"format": "current_training_export_manifest_v1", "models": records,
              "state_elements_parameter_upper_bound": total, "parameter_limit_passed": True,
              "inference_noise": "explicit_zero", "GT_modified": False,
              "full_inference_verified": False, "official_score_verified": False,
              "identity_only_not_full_cold_start_or_score_evidence": True}
    write_json(Path(context["run_dir"]) / "verify_export" / "manifest.json", result, exclusive=True)
    return result


def stage_requires_cuda(stage_name: str, context: dict) -> bool:
    """只约束真实GPU计算；CPU导出和有来源校验的显式导入不受限。"""
    if stage_name not in LEGACY_BY_NAME:
        raise ValueError("未知训练阶段: " + stage_name)
    options = context.get("settings", {}).get("stages", {}).get(stage_name, {})
    if stage_name == "joint" and options.get("completed_joint_continuation"):
        return False
    if stage_name in ("lora_ssl", "d5a") and options.get("import_checkpoint"):
        return False
    return stage_name not in {"export_backend", "naf_export", "verify_export"}


def check_stage_cuda(stage_name: str, context: dict):
    if not stage_requires_cuda(stage_name, context):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"{stage_name}需要CUDA版PyTorch；请使用项目sam2_env，当前Python: {sys.executable}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(f"{stage_name}需要CUDA GPU；当前Python的CUDA不可用: {sys.executable}。"
                           "请使用项目sam2_env及CUDA版PyTorch；阶段未启动，不自动退回CPU。")


def dispatch(stage_name: str, context: dict):
    if context.get("format") != CONTEXT_FORMAT:
        raise ValueError("训练context格式错误")
    stage = {s.name: s for s in stages_for(context)}[stage_name]
    # 内部_stage也可直接调用；不能绕过父进程的同一GPU环境检查。
    check_stage_cuda(stage_name, context)
    if stage_name == "joint" and context["settings"].get("stages", {}).get("joint", {}).get("completed_joint_continuation"):
        return import_completed_joint(context)
    if context["settings"].get("stages", {}).get(stage_name, {}).get("import_checkpoint"):
        return import_independent_stage(stage_name, context)
    if stage_name == "verify_export":
        return verify_export(context)
    return importlib.import_module(stage.module).run_stage(stage_name, context)


def validate_joint_continuation(context: dict, *, current_assets_sha=None) -> dict:
    """核已完成joint及原训练谱系；允许换编排源码，不允许换GT/资产或复制旧身份。"""
    options = context["settings"].get("stages", {}).get("joint", {}).get("completed_joint_continuation")
    if not isinstance(options, dict):
        raise ValueError("缺少completed_joint_continuation来源映射")
    root, target_run = Path(context["project_root"]), Path(context["run_dir"]).resolve()
    paths = {}
    for key in ("source_context", "source_receipt", "checkpoint"):
        path = resolve(root, options[key])
        expected = options[key + "_sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or file_sha(path) != expected:
            raise ValueError("joint接续封存SHA不符: " + key)
        if path.is_relative_to(target_run):
            raise ValueError("joint接续必须来自独立已完成run")
        paths[key] = path
    original = read_json(paths["source_context"])
    if original.get("format") != CONTEXT_FORMAT:
        raise ValueError("joint接续源context格式错误")
    source_run = Path(original["run_dir"]).resolve()
    if (source_run == target_run or paths["source_context"] != source_run / "context.json"
            or paths["source_receipt"] != source_run / "receipts/joint.json"
            or paths["checkpoint"] != Path(original["artifacts"]["joint"]).resolve()):
        raise ValueError("joint接续来源位置与源context不一致")
    identity = read_json(source_run / "identity.json")
    if identity.get("context_sha256") != canonical_sha(original):
        raise ValueError("joint源context与源identity不一致")
    if original["assets"] != context["assets"]:
        raise ValueError("joint接续必须保留完整相同资产映射与GT版本")
    if original["settings"].get("teacher_boundary_policy", "known_neighbors_v1") != context["settings"].get("teacher_boundary_policy", "known_neighbors_v1"):
        raise ValueError("joint接续不得改变教师边界监督策略")
    original_assets = check_assets(original, _verify_continuation=False)
    if current_assets_sha is None:
        current_assets = check_assets(context, _verify_continuation=False)
        if not current_assets["passed"]:
            raise ValueError("joint接续当前资产校验失败")
        current_assets_sha = current_assets["sha256"]
    if (not original_assets["passed"] or original_assets["sha256"] != identity.get("assets_sha256")
            or current_assets_sha != identity.get("assets_sha256")):
        raise ValueError("joint源/当前完整训练资产或GT SHA不一致")
    if source_snapshot(Path(original["project_root"]))["sha256"] != identity.get("source_sha256"):
        raise ValueError("joint原封存训练源码发生变化")
    visited = set()
    def verify_receipt(name):
        if name in visited:
            raise ValueError("joint源回执依赖存在环")
        visited.add(name)
        path = source_run / "receipts" / (name + ".json")
        receipt = read_json(path)
        if (receipt.get("format") != RECEIPT_FORMAT or receipt.get("stage") != name
                or receipt.get("status") != "complete" or receipt.get("identity") != identity):
            raise ValueError("joint源训练完成回执/身份不符: " + name)
        outputs = receipt.get("outputs", {})
        if not outputs or any(snapshot(Path(file)) != seal for file, seal in outputs.items()):
            raise ValueError("joint源已封产物发生变化: " + name)
        adapter = source_run / "adapter_results" / (name + ".json")
        if file_sha(adapter) != receipt.get("adapter_result_sha256") or read_json(adapter).get("complete") is not True:
            raise ValueError("joint源adapter完成结果不符: " + name)
        for dependency, recorded in receipt.get("inputs", {}).items():
            if dependency not in LEGACY_BY_NAME:
                raise ValueError("joint源必须是实际训练谱系，不能重复接续导入")
            child_path = source_run / "receipts" / (dependency + ".json")
            if file_sha(child_path) != recorded.get("receipt_sha256"):
                raise ValueError("joint源前置回执SHA不符")
            child = verify_receipt(dependency)
            if child["outputs"] != recorded.get("outputs"):
                raise ValueError("joint源前置产物封存不符")
        visited.remove(name)
        return receipt
    receipt = verify_receipt("joint")
    if str(paths["checkpoint"]) not in receipt["outputs"]:
        raise ValueError("joint接续checkpoint不在源完成回执中")
    adapter = read_json(source_run / "adapter_results/joint.json")
    if adapter.get("result", {}).get("imported"):
        raise ValueError("joint接续仅接受本次实际训练完成的joint")
    return {"format": "completed_joint_continuation_v1", "source_run": str(source_run),
            **{key: str(value) for key, value in paths.items()},
            **{key + "_sha256": options[key + "_sha256"] for key in paths},
            "source_identity_sha256": canonical_sha(identity), "shared_assets_sha256": current_assets_sha,
            "source_code_sha256": identity["source_sha256"], "source_training_complete": True,
            "old_identity_reused": False}


def import_completed_joint(context: dict) -> dict:
    """把同GT的已完成joint作为新run显式前置输入，不声称在新run重训。"""
    proof = validate_joint_continuation(context)
    import torch
    from repro.current_early import verify_shared_encoder_contract
    payload = torch.load(proof["checkpoint"], map_location="cpu", weights_only=False)
    if (not payload.get("decoder_state_dict") or not payload.get("lora_state_dict")
            or not isinstance(payload.get("config"), dict)):
        raise ValueError("joint接续要求完整decoder/LoRA/源配置checkpoint")
    del payload
    target = Path(context["artifacts"]["joint"])
    if target.resolve() != Path(context["run_dir"]).resolve() / "joint/best_model_stage2.pth":
        raise ValueError("joint接续输出必须位于新run的固定stage目录")
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(proof["checkpoint"], target)
    if file_sha(target) != proof["checkpoint_sha256"]:
        raise RuntimeError("joint接续checkpoint复制不一致")
    contract = verify_shared_encoder_contract(context, include_semantic=False)
    result = dict(format="completed_joint_continuation_v1", complete=True, stage="joint",
        imported=True, continuation=True, trained_this_run=False, source=proof,
        output=str(target), output_sha256=file_sha(target), shared_encoder_contract=contract,
        original_receipt_reused=False, historical_output_equivalence_verified=False)
    write_json(target.parent / "continuation_receipt.json", result, exclusive=True)
    return result


def import_independent_stage(stage_name: str, context: dict) -> dict:
    """复用不依赖分割GT的原SSL/D5a；不复制旧回执，不冒称本轮训练。"""
    import torch
    if stage_name not in ("lora_ssl", "d5a"):
        raise ValueError("监督阶段不能用导入跳过本轮重训")
    options = context["settings"]["stages"][stage_name]
    origin = options.get("import_provenance")
    if not isinstance(origin, dict) or origin.get("segmentation_GT_used") is not False:
        raise ValueError("缺少独立自监督训练来源声明")
    source = resolve(Path(context["project_root"]), options["import_checkpoint"])
    if file_sha(source) != options["import_sha256"]:
        raise ValueError("导入权重SHA与封存来源不一致")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    d5a_config = None
    if stage_name == "lora_ssl":
        state = payload.get("lora_state_dict", payload)
        if (not isinstance(state, dict) or len(state) != 96 or
                any(not torch.is_tensor(value) or not (name.endswith("lora_A") or name.endswith("lora_B"))
                    for name, value in state.items())):
            raise ValueError("SSL导入要求完整原96张量LoRA状态，不能导入监督backend")
        if sum(value.numel() for value in state.values()) != 1_019_648:
            raise ValueError("SSL LoRA结构与当前rank16/B+配方不一致")
    else:
        if payload.get("format") != "rgb_restoration_diffusion_v1" or not isinstance(payload.get("model"), dict):
            raise ValueError("D5a导入要求原标准扩散恢复checkpoint")
        model = payload.get("model_config", {})
        if model.get("num_steps") != 16 or model.get("kappa") != .03 or int(payload.get("epoch", -1)) != 60:
            raise ValueError("D5a导入必须为原16步/kappa.03/e60恢复器")
        from repro.current_backend import prepare_d5a_config
        d5a_config = prepare_d5a_config(context)
        generated = d5a_config["rgb_restoration"]
        experiment = payload.get("experiment_config")
        if not isinstance(experiment, dict):
            raise ValueError("D5a导入缺少实际experiment_config来源，不能伪造退化配方")
        if (canonical_sha(experiment.get("model")) != canonical_sha(generated["model"])
                or canonical_sha(model) != canonical_sha(generated["model"])
                or canonical_sha(experiment.get("degradation")) != canonical_sha(generated["degradation"])):
            raise ValueError("D5a导入model/degradation配方与checkpoint实际训练来源不符")
    target = stage_outputs(context, stage_name)[0]
    target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, target)
    if file_sha(target) != options["import_sha256"]:
        raise RuntimeError("导入复制不一致")
    result = {"complete": True, "stage": stage_name, "imported": True, "trained_this_run": False,
              "source": str(source), "source_sha256": options["import_sha256"], "output": str(target),
              "origin": origin, "original_receipt_reused": False}
    if d5a_config is not None:
        config_path = Path(context["run_dir"]) / "configs/d5a.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        with config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(d5a_config, handle, allow_unicode=True, sort_keys=False)
        result.update(config=str(config_path), config_sha256=file_sha(config_path),
                      source_experiment_config_sha256=canonical_sha(payload["experiment_config"]),
                      matched_model_sha256=canonical_sha(payload["model_config"]),
                      matched_degradation_sha256=canonical_sha(payload["experiment_config"]["degradation"]))
    write_json(target.parent / "import_receipt.json", result, exclusive=True)
    return result


def execute_child(context: dict, stage: Stage):
    # 在创建阶段日志或启动旧训练器之前拒绝意外CPU回退；模拟executor不走此处。
    check_stage_cuda(stage.name, context)
    run = Path(context["run_dir"])
    log_path = run / "logs" / (stage.name + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "repro" / "pipeline.py"), "_stage", "--stage", stage.name,
               "--context", str(run / "context.json")]
    env = dict(os.environ, SEGMENTATION_PROJECT_ROOT=context["project_root"], PYTHONUNBUFFERED="1")
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, cwd=context["project_root"], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for line in iter(process.stdout.readline, b""):
            log.write(line)
            log.flush()
            sys.stdout.write(line.decode("utf-8", errors="replace"))
            sys.stdout.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def run(context: dict, selected: tuple[Stage, ...], *, reuse_completed=False,
        executor: Callable | None = None) -> dict:
    code, assets = check_code(context), check_assets(context)
    if not code["passed"] or not assets["passed"]:
        raise RuntimeError("预检失败: " + "; ".join(code["errors"] + assets["errors"]))
    root, run_dir = Path(context["project_root"]), Path(context["run_dir"])
    source = source_snapshot(root)
    identity = execution_identity(context, source, assets)
    context_path = run_dir / "context.json"
    if run_dir.exists():
        if not reuse_completed or not context_path.is_file():
            raise FileExistsError("run-dir已存在；仅完整同源回执可通过--reuse-completed复用，半途产物保留且不覆盖")
        if read_json(context_path) != context or read_json(run_dir / "identity.json") != identity:
            raise RuntimeError("context/源码/资产/运行环境已改变，须使用新run-dir")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(context_path, context, exclusive=True)
        write_json(run_dir / "identity.json", identity, exclusive=True)
        write_json(run_dir / "source_snapshot.json", source, exclusive=True)
        write_json(run_dir / "asset_snapshot.json", assets, exclusive=True)
    completed = []
    for stage in selected:
        current_assets = check_assets(context)
        if (not current_assets["passed"] or execution_identity(context, source_snapshot(root), current_assets) != identity):
            raise RuntimeError("阶段开始前源码或训练资产已变化，保留已有产物并使用新run-dir")
        inputs = stage_inputs(context, stage, identity)
        receipt_path = run_dir / "receipts" / (stage.name + ".json")
        if receipt_path.exists():
            old = read_json(receipt_path)
            outputs = {str(path): snapshot(path) for path in stage_outputs(context, stage.name)}
            adapter_path = run_dir / "adapter_results" / (stage.name + ".json")
            if (not reuse_completed or old.get("status") != "complete" or old.get("identity") != identity
                    or old.get("inputs") != inputs or old.get("outputs") != outputs
                    or not adapter_path.is_file() or old.get("adapter_result_sha256") != file_sha(adapter_path)):
                raise RuntimeError(f"{stage.name}回执/输入/产物不一致，不能跳过或覆盖")
            completed.append({"stage": stage.name, "reused": True,
                              "imported": bool(read_json(adapter_path).get("result", {}).get("imported"))})
            continue
        destination = run_dir / stage.name
        if destination.exists() or any(path.exists() for path in stage_outputs(context, stage.name)):
            raise FileExistsError(f"{stage.name}已有未认证产物，保留原状；请使用新run-dir")
        receipt = {"format": RECEIPT_FORMAT, "stage": stage.name, "status": "running", "identity": identity,
                   "inputs": inputs, "started_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(receipt_path, receipt, exclusive=True)
        started = time.monotonic()
        print(f"[{stage.name}] {stage.description}", flush=True)
        try:
            (executor or execute_child)(context, stage)
            outputs = {str(path): snapshot(path) for path in stage_outputs(context, stage.name)}
            adapter_path = run_dir / "adapter_results" / (stage.name + ".json")
            if not adapter_path.is_file():
                raise RuntimeError("缺少本轮adapter完成结果")
            if read_json(adapter_path).get("complete") is not True:
                raise RuntimeError("本轮adapter没有确认完成")
            current_assets = check_assets(context)
            if (not current_assets["passed"] or execution_identity(context, source_snapshot(root), current_assets) != identity):
                raise RuntimeError("阶段执行期间源码或训练资产发生改变，不能认证完成")
            receipt.update(status="complete", outputs=outputs, adapter_result_sha256=file_sha(adapter_path))
        except BaseException as exc:
            receipt.update(status="failed", error=repr(exc), seconds=time.monotonic() - started)
            write_json(receipt_path, receipt)
            raise
        receipt.update(seconds=time.monotonic() - started, completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        completed.append({"stage": stage.name, "reused": False,
                          "imported": bool(read_json(adapter_path).get("result", {}).get("imported"))})
    return {"passed": True, "stages": completed, "training_executed": any(not row["reused"] and not row["imported"] for row in completed),
            "full_from_scratch_training_executed": len(selected) == len(stages_for(context)) and all(not row["reused"] and not row["imported"] for row in completed),
            "full_inference_verified": False, "official_score_verified": False}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("list", "plan", "check", "run", "_stage"))
    result.add_argument("--config", default="repro/current_pipeline.yaml")
    result.add_argument("--run-dir", default="outputs/repro_current/rehearsal_v1")
    result.add_argument("--from-stage", choices=tuple(LEGACY_BY_NAME))
    result.add_argument("--to-stage", choices=tuple(LEGACY_BY_NAME))
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--code-only", action="store_true")
    result.add_argument("--reuse-completed", action="store_true")
    result.add_argument("--report", help="可选私有JSON报告路径")
    result.add_argument("--stage", choices=tuple(LEGACY_BY_NAME), help=argparse.SUPPRESS)
    result.add_argument("--context", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "_stage":
            if not args.context or not args.stage:
                raise ValueError("内部stage执行缺少context/stage")
            context = read_json(Path(args.context))
            value = dispatch(args.stage, context)
            write_json(Path(context["run_dir"]) / "adapter_results" / (args.stage + ".json"),
                       {"stage": args.stage, "complete": True, "result": value}, exclusive=True)
            return 0
        context = build_context(args.config, args.run_dir)
        selected = selection(args.from_stage, args.to_stage, context=context)
        if args.action in ("list", "plan"):
            result = {"format": FORMAT, "run_dir": context["run_dir"], "training_started": False,
                      "GT_modified": False, "stages": [{"stage": stage.name, "dependencies": stage.dependencies,
                      "adapter": stage.module, "description": stage.description,
                      "outputs": [str(path) for path in stage_outputs(context, stage.name)]} for stage in selected]}
        elif args.action == "check" or args.dry_run:
            code = check_code(context)
            assets = None if args.code_only else check_assets(context)
            result = {"passed": code["passed"] and (assets is None or assets["passed"]),
                      "code": code, "assets": assets, "training_started": False, "GT_modified": False,
                      "dry_run": args.dry_run, "full_cold_start_verified": False,
                      "stages": [stage.name for stage in selected]}
        else:
            if args.code_only:
                raise ValueError("实际run不能用code-only跳过资源校验")
            result = run(context, selected, reuse_completed=args.reuse_completed)
        if args.report:
            path = resolve(Path(context["project_root"]), args.report)
            if not any(path.is_relative_to(Path(context["project_root"]) / base) for base in ("output", "outputs")):
                raise ValueError("报告须存私有output/或outputs/目录")
            write_json(path, result)
        # 私有全量源图清单只写report；终端只输出汇总。
        display = dict(result)
        if display.get("assets"):
            display["assets"] = {key: value for key, value in result["assets"].items() if key != "files"}
        print(json.dumps(display, ensure_ascii=False, indent=2))
        return 0 if result.get("passed", True) else 1
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
