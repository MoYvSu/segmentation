# -*- coding: utf-8 -*-
"""冷启动训练完成、冻结契约与私有自动产物流程的 CPU 检查。"""
import json
from pathlib import Path

import pytest
import yaml

from tools import run_backend_coldstart as runner


@pytest.fixture
def config(tmp_path):
    return {"paths": {"project_root": str(tmp_path)}, "inference": {"test_dir": "data/test"},
            "backend_adaptation": {"output_dir": "outputs/backend_cold", "epochs": 120,
                "head_warmup_epochs": 60, "joint_epochs": 60, "expected_manual_sources": 32,
                "manual_repeats": 2, "monitor": {"every_epochs": 5,
                    "images": ["test_101", "test_091", "test_130", "test_089"]},
                "d5a_prediction_dir": "outputs/d5a_candidate/deployment",
                "simple_prediction_dir": "outputs/semantic_d5a/simple/deployment"}}


def completed_smoke(run, config):
    run.mkdir(parents=True)
    status = {"status": "completed", "epoch": 2, "updates": 4, "failed_updates": 0,
              "frozen_encoder_unchanged": True, "d5a_unchanged": True,
              "warmup_lora_unchanged": True, "strict_reload_passed": True,
              "trainable_max_changes": {"semantic": .1, "affinity": .1, "lora": .01},
              "phases": {"head_warmup": {"epochs": 1, "updates": 2},
                         "joint": {"epochs": 1, "updates": 2}}}
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    rows = [{"epoch": (i - 1) // 2 + 1, "updates": i,
             "phase": "head_warmup" if i <= 2 else "joint"} for i in range(1, 5)]
    (run / "steps.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    (run / "epochs.jsonl").write_text("\n".join(map(json.dumps, [rows[1], rows[3]])), encoding="utf-8")
    for epoch in (0, 1, 2):
        target = run / "monitor" / f"epoch_{epoch:03d}"
        target.mkdir(parents=True)
        for name in config["backend_adaptation"]["monitor"]["images"]:
            (target / f"{name}.png").write_bytes(b"thumbnail fixture")
    (run / "epoch_002.pt").write_bytes(b"checkpoint fixture")
    return status


def test_complete_smoke_checks_both_phases_and_all_process_thumbnails(tmp_path, config):
    run = tmp_path / "outputs/backend_cold_smoke/model"
    completed_smoke(run, config)
    result = runner.verify_run(run, config, smoke=True)
    assert result["updates"] == 4
    assert result["monitor_images"] == 12
    assert result["phases"]["joint"]["updates"] == 2
    assert runner.schedule(config) == (60, 60, 64)


@pytest.mark.parametrize("field,value", [
    ("updates", 3), ("failed_updates", 1), ("epoch", 1),
    ("frozen_encoder_unchanged", False), ("d5a_unchanged", False),
    ("warmup_lora_unchanged", False), ("strict_reload_passed", False),
    ("trainable_max_changes", {"semantic": .1, "affinity": .1, "lora": 0}),
    ("trainable_max_changes", {"semantic": .1, "affinity": float("nan"), "lora": .1}),
    ("phases", {"head_warmup": {"epochs": 2, "updates": 4}}),
])
def test_rejects_incomplete_or_violated_training(tmp_path, config, field, value):
    run = tmp_path / "outputs/backend_cold_smoke/model"
    status = completed_smoke(run, config)
    status[field] = value
    (run / "status.json").write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RuntimeError):
        runner.verify_run(run, config, smoke=True)
    assert not (run / "run_check.json").exists()


@pytest.mark.parametrize("broken", ["logs", "phase", "monitor", "checkpoint"])
def test_rejects_missing_evidence(tmp_path, config, broken):
    run = tmp_path / "outputs/backend_cold_smoke/model"
    completed_smoke(run, config)
    if broken == "logs":
        (run / "steps.jsonl").write_text("", encoding="utf-8")
    elif broken == "phase":
        path = run / "steps.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace('"joint"', '"head_warmup"'), encoding="utf-8")
    elif broken == "monitor":
        (run / "monitor/epoch_000/test_101.png").unlink()
    else:
        (run / "epoch_002.pt").unlink()
    with pytest.raises(RuntimeError):
        runner.verify_run(run, config, smoke=True)


@pytest.mark.parametrize("path", ["docs/result", "data/result", "outputs", "outputs/../../outside"])
def test_artifacts_cannot_leave_private_directory(config, path):
    config["backend_adaptation"]["output_dir"] = path
    with pytest.raises(ValueError, match="output"):
        runner.private_output(config)


def test_pipeline_commands_use_final_model_and_preserve_existing_run(tmp_path, config, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(runner, "verify_run", lambda *args: {"status": "passed"})
    runner.main(["--config", str(path)])
    assert [command[2] for command in calls] == ["train_backend_coldstart.py", "train_backend_coldstart.py",
                                               "tools/package_submission.py", "tools/render_backend_ablation.py"]
    assert all("--arm" not in command for command in calls)
    assert calls[1][calls[1].index("--checkpoint") + 1].endswith("epoch_120.pt")
    output = runner.private_output(config)
    assert Path(calls[2][calls[2].index("--output") + 1]).is_relative_to(output)
    status = runner.read(output / "pipeline_status.json")
    assert status["completed"] == ["train", "verify", "infer", "package", "render"]
    assert status["status"] == "completed" and not status["competition_submission"]
    with pytest.raises(FileExistsError):
        runner.main(["--config", str(path)])
    calls.clear()
    runner.main(["--config", str(path), "--postprocess-only"])
    assert len(calls) == 3 and "--checkpoint" in calls[0]


def test_failed_contract_stops_before_inference_and_records_failure(tmp_path, config, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))

    def reject(*args):
        raise RuntimeError("frozen encoder changed")

    monkeypatch.setattr(runner, "verify_run", reject)
    with pytest.raises(RuntimeError, match="frozen encoder"):
        runner.main(["--config", str(path)])
    assert len(calls) == 1
    status = runner.read(runner.private_output(config) / "pipeline_status.json")
    assert status["status"] == "failed" and status["stage"] == "verify"


def test_smoke_checks_outputs_without_creating_partial_submission(tmp_path, config, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(runner, "verify_run", lambda *args: {"status": "passed"})
    monkeypatch.setattr(runner, "verify_smoke_predictions", lambda *args: {"format_validated": True})
    runner.main(["--config", str(path), "--smoke"])
    assert len(calls) == 3
    assert "--smoke" in calls[0] and "--smoke" in calls[1]
    assert calls[1][calls[1].index("--checkpoint") + 1].endswith("epoch_002.pt")
    assert all(command[2] != "tools/package_submission.py" for command in calls)
    status = runner.read(runner.private_output(config, smoke=True) / "pipeline_status.json")
    assert "validate_smoke_predictions" in status["completed"]
    assert status["prediction_check"]["format_validated"]
