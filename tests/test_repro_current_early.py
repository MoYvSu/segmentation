# -*- coding: utf-8 -*-
"""新编排适配层的 CPU 合同；不运行任何旧训练器。"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from repro import current_early as early


@pytest.fixture
def context(tmp_path):
    run = tmp_path / "run"
    directory_keys = ("raw_dir", "unlabeled_dir", "purified_gt_dir", "completed_gt_dir",
                      "sam2_geometry_dir", "sam2_source_dir", "sam2_repo", "monitor_dir")
    assets = {key: str(tmp_path / key) for key in directory_keys}
    for path in assets.values():
        Path(path).mkdir()
    for key in ("backbone", "completed_gt_manifest", "sam2_authorization_doc"):
        assets[key] = str(tmp_path / key)
        Path(assets[key]).write_bytes(b"fixed asset")
    artifacts = {early.ARTIFACT_KEYS[stage]: str(run / stage / early.OUTPUT_NAMES[stage])
                 for stage in early.STAGES}
    return dict(format="current_training_context_v1", project_root=str(early.ROOT),
                run_dir=str(run), assets=assets, artifacts=artifacts,
                settings={"seed": 20300102, "stages": {}})


def legacy(context):
    context["settings"]["semantic_teacher_source"] = "legacy_v6_e9"
    for stage in ("semantic_boundary", "v6"):
        context["artifacts"][stage] = str(Path(context["run_dir"]) / stage / early.OUTPUT_NAMES[stage])
    return context


def test_all_stages_are_pure_and_bind_explicit_assets(context):
    before = copy.deepcopy(context)
    for stage in early.STAGES:
        plan = early.build_stage(stage, context)
        cfg = plan["config"]
        assert plan["outputs"][early.ARTIFACT_KEYS[stage]] == context["artifacts"][early.ARTIFACT_KEYS[stage]]
        assert cfg["paths"]["raw_data_dir"] == context["assets"]["raw_dir"]
        assert cfg["boundary"]["gt_dir"] == context["assets"]["purified_gt_dir"]
        assert cfg["sam2"]["sam2_repo_path"] == context["assets"]["sam2_repo"]
        assert cfg["paths"]["sam2_ckpt"] == context["assets"]["backbone"]
        assert cfg["train"]["seed"] == (42 if stage == "affinity" else 20300102)
        assert cfg["data"]["seed"] == 42
        assert not Path(plan["output_dir"]).exists()
        assert "--legacy-module" in plan["command"]
    assert before == context
    assert not Path(context["run_dir"]).exists()


def test_default_fresh_stage1_builds_and_runs_fixed_training_monitor(context, tmp_path):
    """默认YAML经适配器传到真实monitor，覆盖原GT的e0观察路径。"""
    from utils.training_process_monitor import build_process_monitor, monitor_due

    context["settings"] = early.load_config(str(early.ROOT / "repro/current_pipeline.yaml"))
    plan = early.build_stage("stage1", context)
    config = plan["config"]
    assert "teacher_gt_dir" not in config["data"]
    assert config["train"]["epochs"] == 100 and config["data"]["batch_size"] == 8
    assert config["train"]["learning_rate"] == .00015

    class TrainingSamples:
        augment, crop_size, teacher_gt_dir = True, 512, None

        def __len__(self):
            return 3

        def __getitem__(self, index):
            assert self.augment is False and self.crop_size == 0 and index < 2
            return dict(image=torch.zeros(3, 20, 24), target=torch.zeros(3, 20, 24),
                        image_path=f"train_{index:03d}.png")

    class Model(torch.nn.Module):
        def forward(self, image, output_size=None):
            assert not self.training and not torch.is_grad_enabled()
            assert tuple(image.shape[-2:]) == output_size
            return image[:, :2]

    dataset = TrainingSamples()
    monitor = build_process_monitor(config["train"].get("process_monitor"), dataset, tmp_path)
    assert monitor is not None and monitor.dataset is not dataset
    assert monitor.indices == (0, 1) and monitor.thumbnail_size == 384
    assert [epoch for epoch in range(101) if monitor_due(epoch, 100, monitor.every_epochs)] == [
        0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    model = Model().train()
    result = monitor.maybe_save(model, 0, 100, "cpu")
    assert model.training and dataset.augment is True and dataset.crop_size == 512
    assert result["complete"] and len(result["samples"]) == 2
    assert {row["target_source"] for row in result["samples"]} == {"training_GT"}
    assert not result["validation_metrics_computed"] and not result["watershed_computed"]
    assert len(list((tmp_path / "process_monitor/epoch_000").glob("*.png"))) == 2
    assert not Path(plan["output_dir"]).exists()


def test_ssl_recovery_assumptions_are_explicit_and_overridable(context):
    plan = early.build_stage("lora_ssl", context)
    command = plan["command"]
    for key, value in (("batch_size", "2"), ("mask_ratio", "0.4"), ("eval_every", "0")):
        assert command[command.index("--" + key) + 1] == value
    assert "conditionally" in plan["notes"][0]
    context["settings"]["stages"]["lora_ssl"] = {"cli": {"batch_size": 4, "mask_ratio": .25}}
    command = early.build_stage("lora_ssl", context)["command"]
    assert command[command.index("--batch_size") + 1] == "4"
    context["settings"]["stages"]["lora_ssl"]["cli"]["eval_every"] = 5
    with pytest.raises(ValueError, match="old decoder"):
        early.build_stage("lora_ssl", context)


def test_new_v6_e9_teacher_and_joint_affinity_are_independent(context):
    legacy(context)
    v6 = early.build_stage("v6", context)["config"]
    semantic = early.build_stage("semantic", context)
    affinity_plan = early.build_stage("affinity", context)
    affinity = affinity_plan["config"]
    assert v6["semi_supervised"]["base_checkpoint"] == context["artifacts"]["joint"]
    assert v6["semi_supervised"]["checkpoint_interval"] == 5
    assert v6["semi_supervised"]["checkpoint_epochs"] == [9]
    assert v6["semi_supervised"]["save_checkpoints"] is True
    assert semantic["config"]["semi_supervised"]["base_checkpoint"] == context["artifacts"]["v6"]
    assert "--init_from_checkpoint" not in semantic["command"]
    assert semantic["config"]["semi_supervised"]["init_from_checkpoint"] == ""
    assert semantic["config"]["semi_supervised"]["semantic_cold_start_training"]["enabled"]
    assert semantic["config"]["data"]["semantic_train_completed_gt_dir"] == context["assets"]["completed_gt_dir"]
    cfg = affinity["affinity_geometry_g1"]
    assert cfg["reference_checkpoint"] == context["artifacts"]["joint"]
    assert cfg["geometry_init_checkpoint"] is None
    assert cfg["geometry_init_mode"] == "reference_boundary_fpn"
    assert not affinity["semantic_challenger"]["enabled"]
    assert cfg["seed"] == 42
    assert context["artifacts"]["v6"] not in affinity_plan["inputs"]
    assert context["artifacts"]["semantic"] not in affinity_plan["inputs"]


def test_custom_config_and_numerical_overrides_keep_contracts(context, tmp_path):
    cfg = early.build_stage("semantic", context)["config"]
    source = tmp_path / "custom.yaml"
    source.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    context["settings"]["stages"]["semantic"] = {
        "config": str(source), "overrides": {"semi_supervised": {
            "epochs": 3, "base_checkpoint": "old/E10a.pth", "init_from_checkpoint": "old/G4b.pth"}}}
    built = early.build_stage("semantic", context)["config"]
    assert built["semi_supervised"]["epochs"] == 3
    assert built["semi_supervised"]["base_checkpoint"] == context["artifacts"]["joint"]
    assert built["semi_supervised"]["init_from_checkpoint"] == ""
    context["settings"]["stages"]["semantic"]["overrides"]["semi_supervised"]["semantic_cold_start_training"] = {"enabled": False}
    with pytest.raises(ValueError, match="cold reset"):
        early.build_stage("semantic", context)


def _write_checkpoint(context, stage, *, value=1., epoch=8):
    cfg = early.build_stage(stage, context)["config"]
    path = Path(context["artifacts"][stage])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(config=cfg, epoch=epoch,
                    lora_state_dict={"blocks.0.attn.qkv.lora_A": torch.tensor([value]),
                                     "blocks.0.attn.qkv.lora_B": torch.zeros(1)}), path)


def test_shared_encoder_contract_is_exact_and_rejects_wrong_epoch(context):
    legacy(context)
    for stage in ("joint", "v6", "semantic"):
        _write_checkpoint(context, stage)
    result = early.verify_shared_encoder_contract(context)
    assert result["exact"] and len(result["checkpoints"]) == 3
    _write_checkpoint(context, "semantic", value=1.0001)
    with pytest.raises(ValueError, match="contract differs"):
        early.verify_shared_encoder_contract(context)
    _write_checkpoint(context, "semantic")
    _write_checkpoint(context, "v6", epoch=9)
    with pytest.raises(ValueError, match="epoch9"):
        early.verify_shared_encoder_contract(context)


def test_run_stage_executes_only_fresh_output_and_checks_artifact(context, monkeypatch):
    ssl = Path(context["artifacts"]["ssl"])
    ssl.parent.mkdir(parents=True)
    ssl.write_bytes(b"upstream lora")
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(context["artifacts"]["stage1"]).write_bytes(b"new output")
    monkeypatch.setattr(early.subprocess, "run", fake_run)
    result = early.run_stage("stage1", context)
    assert result["complete"]
    assert len(calls) == 1 and calls[0][1]["check"]
    assert calls[0][1]["env"]["PYTHONHASHSEED"] == "20300102"
    assert Path(result["config_path"]).exists()
    with pytest.raises(FileExistsError):
        early.run_stage("stage1", context)


def test_legacy_wrapper_seeds_before_main_and_restores_argv(monkeypatch):
    import random
    import sys
    import numpy as np
    seen = []
    module = SimpleNamespace(main=lambda: seen.append((list(sys.argv), random.random(), float(np.random.rand()), float(torch.rand(())))))
    monkeypatch.setattr(early.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    before = sys.argv
    early.run_legacy("train", 12345, ["--config", "explicit.yaml"])
    early.run_legacy("train", 12345, ["--config", "explicit.yaml"])
    assert seen[0] == seen[1]
    assert seen[0][0] == ["train", "--config", "explicit.yaml"]
    assert sys.argv is before


def test_rejects_wrong_output_binding_or_too_short_v6(context):
    legacy(context)
    context["artifacts"]["v6"] = str(Path(context["run_dir"]) / "v6" / "best_model_stage2.pth")
    with pytest.raises(ValueError, match="stage2_epoch9"):
        early.build_stage("v6", context)
    context["artifacts"]["v6"] = str(Path(context["run_dir"]) / "v6" / "stage2_epoch9.pth")
    context["settings"]["stages"]["v6"] = {"overrides": {"semi_supervised": {"epochs": 8}}}
    with pytest.raises(ValueError, match="reach epoch9"):
        early.build_stage("v6", context)


def test_canonical_extra_checkpoint_epoch_preserves_default_and_monitor_schedule():
    # 执行真实入口的保存条件表达式；无需导入或运行训练器。
    import ast
    source = (early.ROOT / "train_stage2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)
             and "checkpoint_epochs" in ast.unparse(node.test)]
    assert len(nodes) == 1
    condition = compile(ast.Expression(nodes[0].test), "checkpoint_condition", "eval")
    def selected(cfg):
        return [epoch + 1 for epoch in range(60) if eval(condition, {},
                {"epoch": epoch, "checkpoint_interval": 5, "semi_cfg": cfg})]
    periodic = list(range(5, 61, 5))
    assert selected({}) == periodic
    assert selected({"checkpoint_epochs": [9]}) == sorted(periodic + [9])
    assert selected({"checkpoint_epochs": [9], "save_checkpoints": False}) == []
    monitor_call = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "(epoch + 1) % checkpoint_interval == 0"
                        and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                                and child.func.id == "monitor_inference" for child in ast.walk(node)))
    assert "checkpoint_epochs" not in ast.unparse(monitor_call.test)


def test_teacher_assets_replace_all_early_supervised_targets_without_budget_change(context, tmp_path):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    manifest = teacher / "manifest.json"
    manifest.write_text('{"format":"teacher_supervision_v1"}', encoding="utf-8")
    context["assets"].update(teacher_gt_dir=str(teacher), teacher_gt_manifest=str(manifest))
    context["settings"]["teacher_boundary_policy"] = "explicit_separators_v1"
    for stage in ("stage1", "joint", "semantic", "affinity"):
        plan = early.build_stage(stage, context)
        cfg = plan["config"]
        assert cfg["data"]["teacher_gt_dir"] == str(teacher)
        assert cfg["data"]["teacher_boundary_policy"] == "explicit_separators_v1"
        assert cfg["data"]["seed"] == 42
        assert cfg["boundary"]["gt_dir"] == str(teacher)
        assert str(teacher) in plan["inputs"] and str(manifest) in plan["inputs"]
        assert context["assets"]["purified_gt_dir"] not in plan["inputs"]
        assert context["assets"]["completed_gt_dir"] not in plan["inputs"]
        assert cfg["data"]["semantic_train_completed_gt_dir"] == ""
        assert cfg["data"]["semantic_val_completed_gt_dir"] == ""
        if stage == "stage1":
            assert cfg["train"]["epochs"] == 100
        elif stage == "affinity":
            assert cfg["affinity_geometry_g1"]["epochs"] == 120
            assert cfg["affinity_geometry_g1"]["manual_teacher_gt_dir"] == str(teacher)
            assert cfg["affinity_geometry_g1"]["manual_train_completed_gt_dir"] == ""
        else:
            assert cfg["semi_supervised"]["epochs"] == {"joint":100, "semantic":20}[stage]


def test_default_joint_teacher_has_no_v6_cache_or_stage1_reference_dependency(context):
    assert "v6" not in early.stages_for(context) and "semantic_boundary" not in early.stages_for(context)
    plan = early.build_stage("semantic", context)
    semi = plan["config"]["semi_supervised"]
    assert semi["base_checkpoint"] == context["artifacts"]["joint"]
    assert semi["semantic_teacher_source"] == "joint"
    assert semi["use_cached_pseudo_labels"] is False
    assert semi["pseudo_label_cache_dir"] == ""
    assert semi["boundary_anchor"]["enabled"] is False
    assert semi["boundary_teacher_mode"] == "self_consistency"
    assert semi["freeze"] == {"seg_branch": False, "boundary_branch": True}
    assert plan["config"]["lora"]["freeze"] is True
    with pytest.raises(ValueError, match="unknown early stage"):
        early.build_stage("v6", context)


def test_joint_teacher_preserves_exact_lora_contract_without_v6(context):
    for stage in ("joint", "semantic"):
        _write_checkpoint(context, stage)
    proof = early.verify_shared_encoder_contract(context)
    assert proof["semantic_teacher_source"] == "joint"
    assert [row["stage"] for row in proof["checkpoints"]] == ["joint", "semantic"]
    _write_checkpoint(context, "semantic", value=1.000001)
    with pytest.raises(ValueError, match="contract differs"):
        early.verify_shared_encoder_contract(context)
