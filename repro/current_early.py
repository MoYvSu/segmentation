# -*- coding: utf-8 -*-
"""当前训练链前半段适配器；旧训练器及其默认配置保持不变。

只在显式调用 run_stage/CLI 时执行训练。build_stage 可供 CPU 预检。
SSL 的 batch=2、mask=.4 是恢复配方；新 seed 不代表恢复历史随机流。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import _deep_merge, load_config

FORMAT = "current_early_adapter_v1"
STAGES = ("lora_ssl", "stage1", "stage1_boundary", "joint", "semantic", "affinity")
LEGACY_STAGES = ("lora_ssl", "stage1", "stage1_boundary", "joint",
                 "semantic_boundary", "v6", "semantic", "affinity")
DEFAULT_CONFIGS = {
    "lora_ssl": "repro/configs/stage1_lora.yaml",
    "stage1": "repro/configs/stage1_lora.yaml",
    "stage1_boundary": "repro/configs/stage1_lora.yaml",
    "joint": "repro/configs/stage2_joint_v3.yaml",
    "semantic_boundary": "repro/configs/stage2_joint_v3.yaml",
    "v6": "repro/configs/stage2_v6.yaml",
    "semantic": "config/train/stage2_semantic_gt_aligned20.yaml",
    "affinity": "config/train/affinity_geometry_g2_skip_v6.yaml",
}
OUTPUT_NAMES = {
    "lora_ssl": "lora_state_dict.pth", "stage1": "best_model.pth",
    "stage1_boundary": "boundary_probs.npy", "joint": "best_model_stage2.pth",
    "semantic_boundary": "boundary_probs.npy", "v6": "stage2_epoch9.pth",
    "semantic": "best_model_stage2.pth", "affinity": "best_affinity.pth",
}
ARTIFACT_KEYS = {"lora_ssl": "ssl", **{s: s for s in LEGACY_STAGES if s != "lora_ssl"}}


def semantic_teacher_source(context: dict) -> str:
    mode = context.get("settings", {}).get("semantic_teacher_source", "joint")
    if mode not in {"joint", "legacy_v6_e9"}:
        raise ValueError("semantic_teacher_source must be joint or explicit legacy_v6_e9")
    return mode


def stages_for(context: dict):
    return LEGACY_STAGES if semantic_teacher_source(context) == "legacy_v6_e9" else STAGES


def _absolute(value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {value}")
    return path.resolve()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_stage(stage: str, context: dict) -> dict:
    """构造展开配置与命令，不创建目录、不读取模型、不运行 GPU。"""
    if stage not in stages_for(context):
        raise ValueError(f"unknown early stage: {stage}")
    if context.get("format") != "current_training_context_v1":
        raise ValueError("expected current_training_context_v1")
    project = _absolute(context["project_root"], "project_root")
    run = _absolute(context["run_dir"], "run_dir")
    output = run / stage
    assets, artifacts = context["assets"], context["artifacts"]
    settings = context.get("settings", {})
    teacher_source = semantic_teacher_source(context)
    spec = settings.get("stages", {}).get(stage, {})
    seed = int(settings.get("seed", 20261101))
    source = Path(spec.get("config", DEFAULT_CONFIGS[stage]))
    if not source.is_absolute():
        source = project / source
    config = _deep_merge(load_config(str(source)), spec.get("overrides", {}))
    asset = lambda key: str(_absolute(assets[key], f"assets.{key}"))
    artifact = lambda key: str(_absolute(artifacts[key], f"artifacts.{key}"))
    expected_output = str(output / OUTPUT_NAMES[stage])
    if artifact(ARTIFACT_KEYS[stage]) != expected_output:
        raise ValueError(f"artifacts.{ARTIFACT_KEYS[stage]} must be {expected_output}")
    backbone = asset("backbone")
    paths = config.setdefault("paths", {})
    paths.update(project_root=str(project), raw_data_dir=asset("raw_dir"),
                 weights_dir=str(Path(backbone).parent), sam2_ckpt=backbone,
                 output_dir=str(output), pseudo_label_dir=str(run / "stage1_boundary"),
                 active_learning_log=str(output / "active_learning.csv"))
    sam2 = config.setdefault("sam2", {})
    sam2.update(sam2_repo_path=asset("sam2_repo"), freeze=True,
                input_normalization="legacy_none")
    data = config.setdefault("data", {})
    teacher_dir = asset("teacher_gt_dir") if assets.get("teacher_gt_dir") else None
    teacher_manifest = asset("teacher_gt_manifest") if teacher_dir else None
    if teacher_dir:
        data["teacher_gt_dir"] = teacher_dir
        boundary_policy = settings.get("teacher_boundary_policy", "known_neighbors_v1")
        if boundary_policy not in {"known_neighbors_v1", "explicit_separators_v1"}:
            raise ValueError("unsupported teacher_boundary_policy")
        data["teacher_boundary_policy"] = boundary_policy
        data.update(semantic_train_completed_gt_dir="", semantic_val_completed_gt_dir="")
        config["teacher_supervision"] = dict(format="teacher_supervision_v1", source_kind="model_teacher",
            gt_dir=teacher_dir, manifest=teacher_manifest, ID0="unknown_instance_id",
            boundary_policy=boundary_policy, semantic_ID0="ignore", affinity_endpoint_ID0="ignore",
            boundary_target="known_known_4neighbors plus explicit separators only when enabled; binary_float_soft_target_v1")
    train = config.setdefault("train", {})
    train["seed"] = int(spec.get("overrides", {}).get("train", {}).get("seed", seed))
    config.setdefault("boundary", {})["gt_dir"] = teacher_dir or asset("purified_gt_dir")
    inference = config.setdefault("inference", {})
    inference.update(stage1_checkpoint=artifact("stage1"),
                     stage2_checkpoint=artifact("joint"),
                     output_dir=str(output / "inference"), test_dir=asset("monitor_dir"))
    semi = config.setdefault("semi_supervised", {})
    semi.update(unlabeled_dir=asset("unlabeled_dir"), output_dir=str(output),
                base_checkpoint=artifact("stage1"), init_from_checkpoint="",
                pseudo_label_cache_dir=str(Path(artifact("stage1_boundary")).parent))
    semi.setdefault("monitor", {}).update(image_dir=asset("monitor_dir"),
                                          manifest="", output_dir=str(output / "monitor"))
    lora = config.setdefault("lora", {})
    lora["init_from"] = artifact("ssl") if stage == "stage1" else ""
    config_path = run / "configs" / f"{stage}.yaml"
    cli: list[str] = ["--config", str(config_path)]
    notes: list[str] = []
    inputs = [backbone, asset("sam2_repo")]
    if stage == "lora_ssl":
        module = "tools.pretrain_lora_ssl"
        defaults = dict(epochs=30, batch_size=2, lr=1e-4, rank=16, alpha=32.0,
                        mask_ratio=.4, num_workers=4, checkpoint_interval=5,
                        monitor_every=5, eval_every=0)
        unknown = set(spec.get("cli", {})) - set(defaults)
        if unknown:
            raise ValueError(f"unsupported SSL CLI keys: {sorted(unknown)}")
        defaults.update(spec.get("cli", {}))
        if int(defaults["eval_every"]) != 0:
            raise ValueError("SSL fresh chain cannot probe an old decoder; eval_every must be 0")
        for key, value in defaults.items():
            cli.extend([f"--{key}", str(value)])
        cli.extend(["--outdir", str(output), "--eval_decoder", artifact("stage1")])
        inputs += [asset("unlabeled_dir")]
        notes.append("batch2 inferred conditionally from 1000 sources/30 epochs/15000 steps; mask .4 is recovered source default, not preserved CLI proof; seed is a new explicit choice")
    elif stage == "stage1":
        module = "train"
        inputs += [artifact("ssl"), asset("raw_dir"), teacher_dir or asset("purified_gt_dir")]
    elif stage == "stage1_boundary":
        module = "tools.precompute_pseudo_labels"
        semi["base_checkpoint"] = artifact("stage1")
        cli += ["--checkpoint", artifact("stage1"), "--output_dir", str(output)]
        inputs += [artifact("stage1"), asset("unlabeled_dir")]
    elif stage == "semantic_boundary":
        module = "tools.precompute_semantic_boundary"
        semi["base_checkpoint"] = artifact("joint")
        cli += ["--checkpoint", artifact("joint"), "--outdir", str(output),
                "--image_dir", asset("unlabeled_dir")]
        inputs += [artifact("joint"), asset("unlabeled_dir")]
    elif stage in {"joint", "v6", "semantic"}:
        module = "train_stage2"
        base_key = "stage1" if stage == "joint" else "v6" if stage == "semantic" and teacher_source == "legacy_v6_e9" else "joint"
        base = artifact(base_key)
        semi["base_checkpoint"] = base
        cli += ["--phase", stage, "--tag", "current", "--run_dir", str(output / "records")]
        inputs += [base, asset("raw_dir"), teacher_dir or asset("purified_gt_dir"), asset("unlabeled_dir")]
        if stage in {"v6", "semantic"}:
            lora["freeze"] = True
            if stage == "v6" or teacher_source == "legacy_v6_e9":
                semi["pseudo_label_cache_dir"] = str(Path(artifact("semantic_boundary")).parent)
                inputs += [artifact("semantic_boundary")]
            else:
                # 语义冷启动不消费边界缓存，也不需要另建Stage1边界参考模型。
                semi.update(use_cached_pseudo_labels=False, pseudo_label_cache_dir="",
                            boundary_teacher_mode="self_consistency", unsup_weight=0.)
                semi.setdefault("boundary_anchor", {})["enabled"] = False
        else:
            lora["freeze"] = False
            inputs += [artifact("stage1_boundary")]
        if stage == "v6":
            if int(semi.get("epochs", 0)) < 9:
                raise ValueError("V6 must reach epoch9 for the fixed S-align teacher")
            semi.update(save_checkpoints=True,
                        checkpoint_epochs=sorted(set(semi.get("checkpoint_epochs", [])) | {9}))
            notes.append("S-align teacher is fixed epoch9, not validation-best V6")
        if stage == "semantic":
            if not teacher_dir:
                data.update(semantic_train_completed_gt_dir=asset("completed_gt_dir"),
                            semantic_val_completed_gt_dir=asset("completed_gt_dir"))
            inputs += [artifact("joint"), teacher_dir or asset("completed_gt_dir")]
            cold = semi.get("semantic_cold_start_training", {})
            if not cold.get("enabled") or not cold.get("reset_student", True):
                raise ValueError("S-align requires fixed teacher followed by complete semantic cold reset")
            if semi.get("freeze", {}).get("seg_branch") or not semi.get("freeze", {}).get("boundary_branch"):
                raise ValueError("S-align must train semantic only and freeze boundary")
            semi["semantic_teacher_source"] = teacher_source
            notes.append("No init_from_checkpoint: capture " + teacher_source + " semantic teacher before student reset")
            notes.append("New teacher-source recipe; historical/final output equivalence is not claimed")
    else:
        module = "train_affinity_geometry_g1"
        cfg = config.setdefault("affinity_geometry_g1", {})
        # 旧 V 用同一 seed 共同控制初始化、26/6 划分与采样，整体保留该合同。
        seed = int(cfg.get("seed", 42))
        train["seed"] = seed
        cfg.update(reference_checkpoint=artifact("joint"), geometry_init_mode="reference_boundary_fpn",
                   geometry_init_checkpoint=None, init_from_v6_boundary_fpn=True,
                   output_dir=str(output), unlabeled_dir=asset("monitor_dir"), seed=seed,
                   semantic_decoder_override_base="")
        cfg.setdefault("feature_adapter", {})["enabled"] = False
        cfg.setdefault("highres_refiner", {})["enabled"] = False
        if teacher_dir:
            cfg.update(manual_teacher_gt_dir=teacher_dir, manual_train_completed_gt_dir="")
            inputs += [teacher_dir]
        pseudo = cfg.setdefault("sam2_geometry", {})
        if pseudo.get("enabled"):
            pseudo.update(dataset_dir=asset("sam2_geometry_dir"), source_dir=asset("sam2_source_dir"))
            inputs += [asset("sam2_geometry_dir"), asset("sam2_source_dir"), asset("sam2_authorization_doc")]
        # 当前 V 配方只监督原 LabelMe；completed GT 必须显式选择，不能继承旧目录。
        completed = cfg.get("manual_train_completed_gt_dir")
        if completed:
            if _absolute(completed, "manual_train_completed_gt_dir") != Path(asset("completed_gt_dir")):
                raise ValueError("V completed GT must match the explicit completed_gt_dir asset")
            inputs += [asset("completed_gt_dir")]
        challenger = config.setdefault("semantic_challenger", {})
        challenger.update(enabled=False, replace_reference_semantic=False)
        inputs += [artifact("joint"), asset("raw_dir")]
        notes.append("V initializes this run's joint boundary FPN; validation uses "
                     + ("the same teacher target version" if teacher_dir else "original LabelMe")
                     + "; legacy V checkpoints do not support exact optimizer/RNG resume")
        notes.append("V preserves its recipe seed for initialization/split/sampler; settings.seed does not replace it")
    for key in ("gda", "edge_prior"):
        if config.get(key, {}).get("enabled", False):
            raise ValueError(f"current early chain requires {key}.enabled=false")
    if teacher_dir and stage in {"stage1", "joint", "v6", "semantic", "affinity"}:
        inputs += [teacher_manifest]
        notes.append("Model-teacher only; ID0 semantic/affinity ignored; boundary domain follows explicit policy; no manual-completion provenance")
    command = [sys.executable, str(project / "repro" / "current_early.py"),
               "--legacy-module", module, "--seed", str(seed), "--", *cli]
    return dict(format=FORMAT, stage=stage, config=config, config_path=str(config_path),
                source_config=str(source.resolve()), output_dir=str(output),
                outputs={ARTIFACT_KEYS[stage]: expected_output}, inputs=list(dict.fromkeys(inputs)),
                command=command, seed=seed, notes=notes)


def verify_shared_encoder_contract(context: dict, *, include_semantic: bool = True) -> dict:
    """CPU 上核 LoRA 逐字节一致及共同 SAM2 来源；不构建模型。"""
    import torch
    from models.offset_geometry import lora_state_dict_digest

    source = semantic_teacher_source(context)
    keys = ["joint"] + (["v6"] if source == "legacy_v6_e9" else []) + (["semantic"] if include_semantic else [])
    rows = []
    for key in keys:
        path = _absolute(context["artifacts"][key], f"artifacts.{key}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config")
        if not isinstance(cfg, dict):
            raise ValueError(f"{key}: checkpoint must embed its current run config")
        if key == "v6" and int(checkpoint.get("epoch", -1)) != 8:
            raise ValueError("S-align requires V6 epoch9 (checkpoint epoch=8)")
        lora = checkpoint.get("lora_state_dict", {})
        if any("lora_A" not in str(k) and "lora_B" not in str(k) for k in lora):
            raise ValueError(f"{key}: unexpected non-LoRA encoder tensors")
        sam = cfg.get("sam2", {})
        backbone = Path(cfg["paths"]["sam2_ckpt"])
        if not backbone.is_absolute():
            backbone = Path(cfg["paths"]["project_root"]) / cfg["paths"].get("weights_dir", "weights") / backbone
        if backbone.resolve() != _absolute(context["assets"]["backbone"], "assets.backbone"):
            raise ValueError(f"{key}: backbone source differs")
        if not sam.get("freeze", True) or cfg.get("gda", {}).get("enabled") or cfg.get("edge_prior", {}).get("enabled"):
            raise ValueError(f"{key}: unsupported mutable encoder feature path")
        if key != "joint" and not cfg.get("lora", {}).get("freeze", False):
            raise ValueError(f"{key}: LoRA must remain frozen")
        signature = dict(model_version=sam.get("model_version"), config_file=sam.get("config_file"),
                         normalization=sam.get("input_normalization", "legacy_none"),
                         rank=cfg.get("lora", {}).get("rank"), alpha=cfg.get("lora", {}).get("alpha"),
                         target_layers=cfg.get("lora", {}).get("target_layers"))
        rows.append(dict(stage=key, checkpoint=str(path), checkpoint_sha256=_sha(path),
                         lora_sha256=lora_state_dict_digest(lora), signature=signature))
        del checkpoint
    if any(r["lora_sha256"] != rows[0]["lora_sha256"] or r["signature"] != rows[0]["signature"] for r in rows[1:]):
        raise ValueError("shared encoder/LoRA contract differs across the selected teacher lineage")
    return dict(format="current_shared_encoder_v2", exact=True, checkpoints=rows, semantic_teacher_source=source,
                backbone_sha256=_sha(_absolute(context["assets"]["backbone"], "assets.backbone")))


def run_stage(stage: str, context: dict) -> dict:
    """创建本阶段新目录并执行；公共前置资产/授权审查由父编排负责。"""
    plan = build_stage(stage, context)
    for value in plan["inputs"]:
        if not Path(value).exists():
            raise FileNotFoundError(f"{stage} input missing: {value}")
    proof = None
    if stage == "semantic":
        proof = verify_shared_encoder_contract(context, include_semantic=False)
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path(plan["config_path"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(plan["config"], handle, allow_unicode=True, sort_keys=False)
    env = os.environ.copy()
    env["SEGMENTATION_PROJECT_ROOT"] = str(_absolute(context["project_root"], "project_root"))
    env["PYTHONHASHSEED"] = str(plan["seed"])
    subprocess.run(plan["command"], cwd=context["project_root"], env=env, check=True)
    for value in plan["outputs"].values():
        if not Path(value).is_file():
            raise RuntimeError(f"{stage} returned without output: {value}")
    if stage in {"v6", "semantic"}:
        proof = verify_shared_encoder_contract(context, include_semantic=stage != "v6")
    if stage == "affinity":
        import torch
        checkpoint = torch.load(plan["outputs"]["affinity"], map_location="cpu", weights_only=False)
        joint = _absolute(context["artifacts"]["joint"], "artifacts.joint")
        if (Path(checkpoint.get("reference_checkpoint", "")).resolve() != joint
                or checkpoint.get("reference_checkpoint_sha256") != _sha(joint)
                or checkpoint.get("geometry_init_checkpoint") is not None
                or checkpoint.get("geometry_initialization", {}).get("mode") != "reference_boundary_fpn"):
            raise ValueError("V must bind this run's joint reference/boundary FPN exactly")
        proof = dict(format="current_v_joint_reference_v1", exact=True,
                     reference_checkpoint=str(joint), reference_checkpoint_sha256=_sha(joint))
    return {k: copy.deepcopy(v) for k, v in plan.items() if k != "config"} | {
        "complete": True, "config_sha256": _sha(config_path), "shared_encoder_contract": proof}


def run_legacy(module_name: str, seed: int, argv: list[str]) -> None:
    """导入后统一设新随机种子，调用旧 main；不声称历史随机流复原。"""
    import random
    import numpy as np
    import torch
    allowed = {"tools.pretrain_lora_ssl", "train", "tools.precompute_pseudo_labels",
               "tools.precompute_semantic_boundary", "train_stage2", "train_affinity_geometry_g1"}
    if module_name not in allowed:
        raise ValueError(f"unsupported legacy module: {module_name}")
    module = importlib.import_module(module_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    previous = sys.argv
    original_recorder = getattr(module, "RunRecorder", None)
    if module_name == "train_stage2":
        # 旧 main 在应用 --run_dir 前会先创建 recorder；约束它也留在新 run 内。
        records = Path(argv[argv.index("--run_dir") + 1])
        def recorder(*args, **kwargs):
            kwargs["output_base"] = str(records.parent / "recorder_bootstrap")
            return original_recorder(*args, **kwargs)
        module.RunRecorder = recorder
    try:
        sys.argv = [module_name, *argv]
        module.main()
    finally:
        sys.argv = previous
        if module_name == "train_stage2":
            module.RunRecorder = original_recorder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=LEGACY_STAGES)
    parser.add_argument("--context")
    parser.add_argument("--legacy-module")
    parser.add_argument("--seed", type=int, default=20261101)
    args, remainder = parser.parse_known_args()
    if args.legacy_module:
        run_legacy(args.legacy_module, args.seed, remainder[1:] if remainder[:1] == ["--"] else remainder)
        return
    if not args.stage or not args.context or remainder:
        parser.error("--stage and --context are required")
    with open(args.context, encoding="utf-8") as handle:
        context = json.load(handle)
    print(json.dumps(run_stage(args.stage, context), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
