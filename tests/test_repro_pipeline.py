# -*- coding: utf-8 -*-
"""统一编排的真实风险：跨轮复用、产物篡改、半途失败和只读预检。"""
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro import pipeline as p


ROOT = Path(__file__).resolve().parents[1]


def context(tmp_path):
    run = tmp_path / "outputs" / "fresh"
    return {"format": p.CONTEXT_FORMAT, "project_root": str(tmp_path), "run_dir": str(run),
            "settings": {"seed": 20261101}, "assets": {}, "artifacts": p.artifact_paths(run)}


@pytest.fixture
def mock_checks(monkeypatch):
    monkeypatch.setattr(p, "check_code", lambda _: {"passed": True, "errors": []})
    monkeypatch.setattr(p, "check_assets", lambda _, **kw: {"passed": True, "errors": [], "sha256": "assets1"})
    monkeypatch.setattr(p, "source_snapshot", lambda _: {"sha256": "code1", "files": {}})


def fake_execute(ctx, stage):
    for path in p.stage_outputs(ctx, stage.name):
        if stage.name in ("n1_cache", "stage1_boundary", "semantic_boundary"):
            path.mkdir(parents=True)
            (path / "manifest.json").write_text('{"fake":true}', encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(stage.name, encoding="utf-8")
    p.write_json(Path(ctx["run_dir"]) / "adapter_results" / (stage.name + ".json"),
                 {"complete": True, "stage": stage.name}, exclusive=True)


@pytest.mark.parametrize("stage_name", [s.name for s in p.LEGACY_STAGES
    if s.name not in {"export_backend", "naf_export", "verify_export"}])
def test_real_gpu_stage_refuses_cpu_before_launch_or_adapter(tmp_path, monkeypatch, stage_name):
    import torch
    ctx = context(tmp_path)
    if stage_name in {"v6", "semantic_boundary"}:
        ctx["settings"]["semantic_teacher_source"] = "legacy_v6_e9"
    source = tmp_path / "existing_training_input.npz"
    source.write_bytes(b"unchanged source")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    monkeypatch.setattr(p.importlib, "import_module", lambda *a: pytest.fail("must not import adapter"))
    with pytest.raises(RuntimeError, match="sam2_env"):
        p.execute_child(ctx, p.LEGACY_BY_NAME[stage_name])
    with pytest.raises(RuntimeError, match="sam2_env"):
        p.dispatch(stage_name, ctx)
    assert source.read_bytes() == b"unchanged source"
    assert not Path(ctx["run_dir"]).exists()


def test_real_gpu_stage_keeps_existing_launch_when_cuda_available(tmp_path, monkeypatch):
    import torch
    ctx, launches = context(tmp_path), []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def launch(command, **options):
        launches.append((command, options))
        return SimpleNamespace(stdout=io.BytesIO(b"stage output\n"), wait=lambda: 0)
    monkeypatch.setattr(p.subprocess, "Popen", launch)
    p.execute_child(ctx, p.BY_NAME["lora_ssl"])
    assert len(launches) == 1 and launches[0][0][2:4] == ["_stage", "--stage"]
    assert launches[0][1]["cwd"] == ctx["project_root"]
    assert (Path(ctx["run_dir"]) / "logs/lora_ssl.log").read_bytes() == b"stage output\n"


@pytest.mark.parametrize("stage_name,options", [
    ("export_backend", {}), ("naf_export", {}), ("verify_export", {}),
    ("lora_ssl", {"import_checkpoint": "explicit_ssl.pt"}),
    ("d5a", {"import_checkpoint": "explicit_d5a.pt"}),
    ("joint", {"completed_joint_continuation": {"checkpoint": "verified_joint.pt"}}),
])
def test_cpu_exports_and_explicit_imports_do_not_require_cuda(tmp_path, monkeypatch, stage_name, options):
    import torch
    ctx = context(tmp_path)
    ctx["settings"]["stages"] = {stage_name: options}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU stage must not query CUDA"))
    monkeypatch.setattr(p.subprocess, "Popen", lambda *a, **kw:
                        SimpleNamespace(stdout=io.BytesIO(b""), wait=lambda: 0))
    monkeypatch.setattr(p, "verify_export", lambda _: "verified")
    monkeypatch.setattr(p, "import_completed_joint", lambda _: "continued")
    monkeypatch.setattr(p, "import_independent_stage", lambda stage, _: "imported:" + stage)
    monkeypatch.setattr(p.importlib, "import_module", lambda _:
                        SimpleNamespace(run_stage=lambda stage, _: "exported:" + stage))
    assert p.stage_requires_cuda(stage_name, ctx) is False
    p.execute_child(ctx, p.BY_NAME[stage_name])
    expected = ("verified" if stage_name == "verify_export" else "continued" if stage_name == "joint"
                else "imported:" + stage_name if options else "exported:" + stage_name)
    assert p.dispatch(stage_name, ctx) == expected


def test_simulated_executor_does_not_require_cuda(tmp_path, mock_checks, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU simulation must not query CUDA"))
    result = p.run(context(tmp_path), (p.BY_NAME["lora_ssl"],), executor=fake_execute)
    assert result["passed"]


@pytest.mark.parametrize("arguments", [["list"], ["plan"], ["check", "--code-only"]])
def test_readonly_actions_do_not_require_cuda(tmp_path, mock_checks, monkeypatch, arguments):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Read-only action must not query CUDA"))
    run_dir = ROOT / "outputs/repro_current/pytest_plan_only"
    assert p.main(arguments + ["--config", str(ROOT / "repro/current_pipeline.yaml"),
                               "--run-dir", str(run_dir)]) == 0
    assert not run_dir.exists()


def test_graph_has_current_branches_and_no_eliminated_ablations():
    assert p.BY_NAME["semantic"].dependencies == ("joint",)
    assert p.BY_NAME["affinity"].dependencies == ("joint",)
    assert p.BY_NAME["r1"].dependencies == ("joint", "semantic", "affinity", "d5a")
    assert not {"e10a", "affinity_g0", "affinity_g4b", "control", "native", "v6", "semantic_boundary"} & p.BY_NAME.keys()
    seen = set()
    for stage in p.STAGES:
        assert set(stage.dependencies) <= seen
        seen.add(stage.name)


def test_plan_does_not_create_run_or_import_model(tmp_path):
    ctx = p.build_context(ROOT / "repro/current_pipeline.yaml", ROOT / "outputs/repro_current/pytest_plan_only")
    assert not Path(ctx["run_dir"]).exists()
    assert p.check_code(ctx)["model_loaded"] is False
    assert not Path(ctx["run_dir"]).exists()


@pytest.mark.parametrize("run_dir", ["data/raw", "outputs", "../outside", "output/../data"])
def test_output_isolation(run_dir):
    with pytest.raises(ValueError, match="run-dir"):
        p.build_context(ROOT / "repro/current_pipeline.yaml", run_dir)


def test_completed_reuse_requires_same_sources_and_actual_files(tmp_path, mock_checks, monkeypatch):
    ctx = context(tmp_path)
    stage = (p.BY_NAME["lora_ssl"],)
    p.run(ctx, stage, executor=fake_execute)
    result = p.run(ctx, stage, reuse_completed=True, executor=lambda *_: pytest.fail("must reuse"))
    assert result["stages"] == [{"stage": "lora_ssl", "reused": True, "imported": False}]
    Path(ctx["artifacts"]["ssl"]).write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="回执/输入/产物"):
        p.run(ctx, stage, reuse_completed=True, executor=fake_execute)


def test_changed_gt_or_source_refuses_reuse(tmp_path, mock_checks, monkeypatch):
    ctx = context(tmp_path)
    p.run(ctx, (p.BY_NAME["lora_ssl"],), executor=fake_execute)
    monkeypatch.setattr(p, "check_assets", lambda _: {"passed": True, "errors": [], "sha256": "newGT"})
    with pytest.raises(RuntimeError, match="资产/运行环境"):
        p.run(ctx, (p.BY_NAME["lora_ssl"],), reuse_completed=True, executor=fake_execute)


def test_no_exists_only_shortcut(tmp_path, mock_checks):
    ctx = context(tmp_path)
    Path(ctx["run_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="同源回执"):
        p.run(ctx, (p.BY_NAME["lora_ssl"],), reuse_completed=True, executor=fake_execute)


def test_failure_is_preserved_not_automatically_resumed(tmp_path, mock_checks):
    ctx = context(tmp_path)

    def fail(context, stage):
        output = Path(context["artifacts"]["ssl"])
        output.parent.mkdir(parents=True)
        output.write_bytes(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        p.run(ctx, (p.BY_NAME["lora_ssl"],), executor=fail)
    assert p.read_json(Path(ctx["run_dir"]) / "receipts/lora_ssl.json")["status"] == "failed"
    assert Path(ctx["artifacts"]["ssl"]).read_bytes() == b"partial"
    with pytest.raises(RuntimeError, match="回执/输入/产物"):
        p.run(ctx, (p.BY_NAME["lora_ssl"],), reuse_completed=True, executor=fake_execute)


def test_range_requires_real_completed_prerequisites(tmp_path, mock_checks):
    with pytest.raises(FileNotFoundError):
        p.run(context(tmp_path), (p.BY_NAME["stage1"],), executor=fake_execute)
    with pytest.raises(ValueError):
        p.selection("r1", "joint")


def test_adapter_receipt_is_bound(tmp_path, mock_checks):
    ctx = context(tmp_path)
    p.run(ctx, (p.BY_NAME["lora_ssl"],), executor=fake_execute)
    p.write_json(Path(ctx["run_dir"]) / "adapter_results/lora_ssl.json", {"complete": False})
    with pytest.raises(RuntimeError, match="回执/输入/产物"):
        p.run(ctx, (p.BY_NAME["lora_ssl"],), reuse_completed=True, executor=fake_execute)


def test_forged_dependency_identity_rejected(tmp_path, mock_checks):
    ctx = context(tmp_path)
    p.run(ctx, (p.BY_NAME["lora_ssl"],), executor=fake_execute)
    path = Path(ctx["run_dir"]) / "receipts/lora_ssl.json"
    value = p.read_json(path)
    value["identity"]["source_sha256"] = "old code"
    p.write_json(path, value)
    with pytest.raises(RuntimeError, match="前置阶段不是本轮"):
        p.run(ctx, (p.BY_NAME["stage1"],), reuse_completed=True, executor=fake_execute)


def test_cache_indices_are_part_of_dependency_identity(tmp_path, mock_checks):
    ctx = context(tmp_path)
    p.run(ctx, p.selection(last="stage1_boundary"), executor=fake_execute)
    cache = Path(ctx["artifacts"]["stage1_boundary"]).parent
    (cache / "names.txt").write_text("changed order", encoding="utf-8")
    with pytest.raises(RuntimeError, match="前置阶段产物变更"):
        p.run(ctx, (p.BY_NAME["joint"],), reuse_completed=True, executor=fake_execute)


def test_external_stage_config_changes_are_bound(tmp_path, mock_checks, monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(p, "check_code", lambda _: {"passed": True, "errors": [], "configs": {"r1": "old"}})
    p.run(ctx, (p.BY_NAME["lora_ssl"],), executor=fake_execute)
    monkeypatch.setattr(p, "check_code", lambda _: {"passed": True, "errors": [], "configs": {"r1": "changed"}})
    with pytest.raises(RuntimeError, match="源码/资产/运行环境"):
        p.run(ctx, (p.BY_NAME["lora_ssl"],), reuse_completed=True, executor=fake_execute)


def test_standard_export_state_keys(tmp_path):
    import torch
    ctx = context(tmp_path)
    for name, key in (("backend", "model_state_dict"), ("restorer", "model"), ("naf_export", "model_state_dict")):
        path = Path(ctx["artifacts"][name]); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": name, key: {"weight": torch.ones(2, 3)}}, path)
    result = p.verify_export(ctx)
    assert result["parameter_limit_passed"] and result["state_elements_parameter_upper_bound"] == 18
    assert result["full_inference_verified"] is False


def test_imported_d5a_is_not_new_training(tmp_path, monkeypatch):
    import torch
    ctx = context(tmp_path)
    source = tmp_path / "old_d5a.pt"
    recipe = {"model": {"num_steps": 16, "kappa": .03}, "degradation": {"sigma": [1, 2]}}
    from repro import current_backend
    monkeypatch.setattr(current_backend, "prepare_d5a_config", lambda _: {"rgb_restoration":recipe})
    torch.save({"format": "rgb_restoration_diffusion_v1", "epoch": 60,
                "model_config": recipe["model"], "experiment_config": recipe,
                "model": {"weight": torch.ones(2)}}, source)
    ctx["settings"]["stages"] = {"d5a": {"import_checkpoint": str(source),
        "import_sha256": p.file_sha(source), "import_provenance": {"segmentation_GT_used": False}}}
    result = p.import_independent_stage("d5a", ctx)
    assert result["imported"] is True and result["trained_this_run"] is False
    assert p.file_sha(ctx["artifacts"]["d5a"]) == p.file_sha(source)
    assert result["original_receipt_reused"] is False
    assert Path(result["config"]).is_file()
    assert p.file_sha(result["config"]) == result["config_sha256"]
    assert result["matched_degradation_sha256"] == p.canonical_sha(recipe["degradation"])
    assert Path(result["config"]) in p.stage_outputs(ctx, "d5a")


@pytest.mark.parametrize("defect", ["missing_source", "degradation", "model"])
def test_imported_d5a_rejects_unproven_or_mismatched_recipe(tmp_path, monkeypatch, defect):
    import torch
    from repro import current_backend
    ctx = context(tmp_path)
    recipe = {"model": {"num_steps":16, "kappa":.03}, "degradation":{"sigma":[1,2]}}
    monkeypatch.setattr(current_backend, "prepare_d5a_config", lambda _: {"rgb_restoration":recipe})
    experiment = deepcopy(recipe)
    if defect == "degradation":
        experiment["degradation"]["sigma"] = [4,5]
    if defect == "model":
        experiment["model"]["kappa"] = .1
    payload = {"format":"rgb_restoration_diffusion_v1", "epoch":60,
               "model_config":recipe["model"], "experiment_config":experiment, "model":{"weight":torch.ones(2)}}
    if defect == "missing_source":
        payload.pop("experiment_config")
    source = tmp_path / "old.pt"
    torch.save(payload, source)
    ctx["settings"]["stages"] = {"d5a":{"import_checkpoint":str(source), "import_sha256":p.file_sha(source),
        "import_provenance":{"segmentation_GT_used":False}}}
    with pytest.raises(ValueError, match="experiment_config|配方"):
        p.import_independent_stage("d5a",ctx)
    assert not Path(ctx["artifacts"]["d5a"]).exists()


def test_supervised_checkpoint_cannot_be_imported(tmp_path):
    with pytest.raises(ValueError, match="监督阶段"):
        p.import_independent_stage("joint", context(tmp_path))


def test_legacy_v6_recipe_requires_explicit_selection():
    ctx = {"settings": {"semantic_teacher_source": "legacy_v6_e9"}}
    stages = p.stages_for(ctx)
    assert [s.name for s in stages if s.name in {"v6", "semantic_boundary"}] == ["semantic_boundary", "v6"]
    assert next(s for s in stages if s.name == "semantic").dependencies == ("joint", "v6", "semantic_boundary")
    assert [s.name for s in p.selection("semantic_boundary", "v6", context=ctx)] == ["semantic_boundary", "v6"]


def test_current_and_teacher_configs_default_to_joint_without_v6():
    for name in ("current_pipeline.yaml", "teacher_pipeline.yaml"):
        ctx = p.build_context(ROOT / "repro" / name, ROOT / "outputs/repro_current/new_joint_plan")
        assert ctx["settings"]["semantic_teacher_source"] == "joint"
        assert "v6" not in ctx["artifacts"] and "semantic_boundary" not in ctx["artifacts"]
        assert p.check_code(ctx)["passed"]


def completed_joint_source(tmp_path, mock_checks):
    import torch
    source = context(tmp_path / "old")
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"shared backbone")
    source["assets"] = {"backbone": str(backbone)}
    config = {"paths": {"sam2_ckpt": str(backbone)},
              "sam2": {"freeze": True, "model_version": "toy", "config_file": "toy"},
              "lora": {"rank": 1, "alpha": 2, "target_layers": ["qkv"], "freeze": False}}
    def execute(ctx, stage):
        if stage.name == "joint":
            path = Path(ctx["artifacts"]["joint"])
            path.parent.mkdir(parents=True)
            torch.save({"config": config, "epoch": 99,
                        "decoder_state_dict": {"weight": torch.ones(2)},
                        "lora_state_dict": {"block.qkv.lora_A":torch.ones(1),
                                            "block.qkv.lora_B":torch.zeros(1)}}, path)
            p.write_json(Path(ctx["run_dir"]) / "adapter_results/joint.json",
                         {"complete": True, "stage": "joint", "result": {"complete": True}}, exclusive=True)
        else:
            fake_execute(ctx, stage)
    p.run(source, p.selection(last="joint"), executor=execute)
    destination = context(tmp_path / "new")
    destination["assets"] = deepcopy(source["assets"])
    destination["settings"]["stages"] = {"joint": {"completed_joint_continuation": {
        key: str(path) for key, path in {
            "source_context": Path(source["run_dir"]) / "context.json",
            "source_receipt": Path(source["run_dir"]) / "receipts/joint.json",
            "checkpoint": Path(source["artifacts"]["joint"])}.items()}}}
    options = destination["settings"]["stages"]["joint"]["completed_joint_continuation"]
    for key in list(options):
        options[key + "_sha256"] = p.file_sha(options[key])
    return source, destination


def test_completed_joint_continuation_needs_no_new_stage1_receipts(tmp_path, mock_checks):
    source, destination = completed_joint_source(tmp_path, mock_checks)
    def execute(ctx, stage):
        result = p.import_completed_joint(ctx)
        p.write_json(Path(ctx["run_dir"]) / "adapter_results/joint.json",
                     {"complete": True, "stage": "joint", "result": result}, exclusive=True)
    result = p.run(destination, (p.BY_NAME["joint"],), executor=execute)
    assert result["full_from_scratch_training_executed"] is False
    assert result["training_executed"] is False
    assert result["stages"] == [{"stage":"joint", "reused":False, "imported":True}]
    assert not (Path(destination["run_dir"]) / "stage1").exists()
    assert p.file_sha(destination["artifacts"]["joint"]) == p.file_sha(source["artifacts"]["joint"])
    assert p.read_json(Path(destination["run_dir"]) / "identity.json") != p.read_json(Path(source["run_dir"]) / "identity.json")


@pytest.mark.parametrize("defect", ["asset_mapping", "asset_content", "boundary_policy", "checkpoint", "receipt", "incomplete", "index", "old_code"])
def test_joint_continuation_rejects_drift(tmp_path, mock_checks, monkeypatch, defect):
    source, destination = completed_joint_source(tmp_path, mock_checks)
    if defect == "asset_mapping":
        destination["assets"]["backbone"] += ".different"
    elif defect == "asset_content":
        monkeypatch.setattr(p, "check_assets", lambda _, **kw: {"passed":True, "errors":[], "sha256":"changed assets"})
    elif defect == "boundary_policy":
        destination["settings"]["teacher_boundary_policy"] = "explicit_separators_v1"
    elif defect == "checkpoint":
        Path(source["artifacts"]["joint"]).write_bytes(b"changed")
    elif defect in {"receipt", "incomplete"}:
        path = Path(source["run_dir"]) / "receipts/joint.json"
        value = p.read_json(path); value["status"] = "running"
        p.write_json(path, value)
        if defect == "incomplete":
            destination["settings"]["stages"]["joint"]["completed_joint_continuation"]["source_receipt_sha256"] = p.file_sha(path)
    elif defect == "index":
        (Path(source["artifacts"]["stage1_boundary"]).parent / "names.txt").write_text("changed",encoding="utf-8")
    else:
        monkeypatch.setattr(p, "source_snapshot", lambda _: {"sha256":"different code", "files":{}})
    with pytest.raises((ValueError, RuntimeError)):
        p.validate_joint_continuation(destination)


def test_r1_requires_frozen_d5a_recipe_not_only_unchanged_weights(tmp_path, mock_checks):
    ctx = context(tmp_path)
    p.run(ctx, p.selection(last="d5a"), executor=fake_execute)
    (Path(ctx["run_dir"]) / "configs/d5a.yaml").write_text("changed degradation", encoding="utf-8")
    with pytest.raises(RuntimeError, match="前置阶段产物变更"):
        p.run(ctx, (p.BY_NAME["r1"],), reuse_completed=True, executor=fake_execute)
