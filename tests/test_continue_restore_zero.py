# -*- coding: utf-8 -*-
"""以合成日志验证A/C续训隔离与配对门禁，不执行训练或加载GPU。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from tools import continue_restore_zero as runner
from utils.config import load_config


def _config(arm):
    value = {
        "paths": {"project_root": f"frozen/{arm}"},
        "rgb_restoration": {
            "output_dir": "outputs/original", "seed": 42,
            "model": {"num_steps": 16, "kappa": .03},
            "train": {"epochs": 60, "learning_rate": 2e-5,
                      "min_learning_rate": 2e-6, "weight_decay": 1e-4, "batch_size": 4},
        },
        "restore_consistency": {
            "stage_epochs": 5, "monitor_epochs": [0, 1, 3, 5],
            "data_epoch_offset": 60, "micro_batch": 1,
            "consistency_weight": 0. if arm == "zero" else .1,
            "monitor": {"seed": 20261008, "source_epoch": 60},
        },
    }
    if arm == "zero":
        value["restore_consistency"]["noise_mode"] = "zero"
    return value


def _metrics(epoch, arm):
    values = []
    for update in range(1, epoch * 250 + 1):
        current, batch = divmod(update - 1, 250)
        row = {
            "update": update, "epoch": current + 1, "data_epoch": 60 + current,
            "sources": [f"train_{batch * 4 + k:03d}.jpg" for k in range(4)],
            "pair_sha256": f"pair-{update}", "noise_rng_before": f"rng-{update}",
            "noise_sha256": f"draw-{update}", "profiles": ["blur"] * 4,
            "mask_mean": [0.] * 4, "learning_rate": 2e-5 - current * 1e-8,
            "samples": 4, "micro_batch": 1, "terminal_step": 16,
            "consistency_weight": 0., "loss": .02, "supervised_loss": .02,
        }
        if arm == "zero":
            row.update(training_noise_mode="zero", applied_noise_max_abs=0.)
        values.append(row)
    return values


def _write_rows(path, values):
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")


def _write_logs(directory, epoch, arm):
    directory.mkdir(parents=True, exist_ok=True)
    _write_rows(directory / "metrics.jsonl", _metrics(epoch, arm))
    _write_rows(directory / "epochs.jsonl", [
        {"epoch": e, "updates": e * 250} for e in range(1, epoch + 1)
    ])


def _parents(tmp_path):
    parents = {}
    for arm in ("control", "zero"):
        directory = tmp_path / f"parent_{arm}" / arm
        _write_logs(directory, 5, arm)
        runner.write(directory / "run_config.json", _config(arm))
        (directory / "last.pt").write_bytes(b"complete-training-state")
        (directory / "epoch_005.pt").write_bytes(arm.encode("ascii") + b"-epoch5")
        runner.write(directory / "status.json", {
            "status": "stage_complete", "arm": arm, "epoch": 5, "updates": 1250,
            "strict_reload_equal": True, "first_inference_reload_equal": True,
            "no_backend_loaded": True,
            "final_checkpoint_sha256": runner.sha(directory / "epoch_005.pt"),
        })
        runner.write(directory.parent / "source_pool.json", [f"train_{i:03d}.jpg" for i in range(1000)])
        runner.write(directory.parent / "monitor_manifest.json", {"seed": 20261008, "fixed": True})
        (directory.parent / "monitor_fixed.pt").write_bytes(b"fixed-monitor-pairs")
        parents[arm] = directory
    return parents


def _state_check(code, directory, arm, epoch):
    return {
        "epoch": epoch, "updates": epoch * 250,
        "noise_rng": "paired-noise-state", "loader_rng": "paired-loader-state",
        "next_lr": 1.9693332436601616e-5,
        "runtime": {"torch_version": "synthetic", "training_dtype": "bfloat16",
                    "matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                    "first_reload_noise_protocol": "zero" if arm == "zero" else "cuda"},
    }


def _argv(monkeypatch, tmp_path, parents, *, run):
    codes = {arm: parents[arm].parent / "code" for arm in parents}
    for path in codes.values():
        path.mkdir()
        (path / "train_restore_consistency.py").write_text("# synthetic frozen source\n", encoding="utf-8")
        runner.write(path.parent / "source_manifest.json", {
            "train_restore_consistency.py": runner.sha(path / "train_restore_consistency.py")})
    root = tmp_path / "continuation"
    arguments = ["continue_restore_zero.py", "--control-code", str(codes["control"]),
                 "--zero-code", str(codes["zero"]), "--control", str(parents["control"]),
                 "--zero", str(parents["zero"]), "--root", str(root)]
    if run:
        arguments.append("--run")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(runner, "state_check", _state_check)
    return root, codes


def test_continuation_changes_only_endpoint_and_monitor_schedule():
    original = _config("control")
    before = copy.deepcopy(original)
    changed = runner.continuation_config(original)
    assert original == before
    assert changed["restore_consistency"]["stage_epochs"] == 10
    assert changed["restore_consistency"]["monitor_epochs"] == [0, 1, 3, 5, 7, 10]
    changed["restore_consistency"]["stage_epochs"] = 5
    changed["restore_consistency"]["monitor_epochs"] = [0, 1, 3, 5]
    assert changed == original
    assert "noise_mode" not in changed["restore_consistency"]
    assert changed["rgb_restoration"]["train"]["epochs"] == 60


def test_prepare_real_copies_preserve_parents(tmp_path):
    parents = _parents(tmp_path)
    hashes = {key: value for source in parents.values() for key, value in runner.parent_files(source).items()}
    root = tmp_path / "continued"
    runner.prepare(root, parents)
    for arm in parents:
        for name in ("last.pt", "metrics.jsonl", "epochs.jsonl", "epoch_005.pt"):
            destination = root / arm / name
            assert not destination.is_symlink()
            assert destination.read_bytes() == (parents[arm] / name).read_bytes()
        with (root / arm / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("new-only\n")
        (root / arm / "last.pt").write_bytes(b"replacement-only")
    runner.verify_hashes(hashes)
    with pytest.raises(FileExistsError):
        runner.prepare(root, parents)


@pytest.mark.parametrize("arm", ["control", "zero"])
def test_prepared_config_preserves_scientific_floats_through_real_project_loader(tmp_path, arm):
    """项目入口用YAML读取；JSON的2e-05写法会被旧解析器错误读成字符串。"""
    parents = _parents(tmp_path)
    root = tmp_path / "continued"
    runner.prepare(root, parents)
    loaded = load_config(str(root / f"{arm}_config.yaml"))
    original = runner.read(parents[arm] / "run_config.json")
    for key in ("learning_rate", "min_learning_rate", "weight_decay"):
        value = loaded["rgb_restoration"]["train"][key]
        assert type(value) is float, (key, type(value), value)
        assert value == original["rgb_restoration"]["train"][key]
    expected = runner.continuation_config(original)
    expected["paths"]["project_root"] = str(Path(expected["paths"]["project_root"]).resolve())
    assert loaded == expected


def test_complete_logs_and_pairs_accept_all_originals(tmp_path):
    control, zero = tmp_path / "control", tmp_path / "zero"
    _write_logs(control, 5, "control")
    _write_logs(zero, 5, "zero")
    assert len(runner.validate_logs(zero, 5)) == 1250
    report = runner.check_pairs(control, zero, 5)
    assert report["matched_updates"] == 1250
    assert report["every_epoch_1000_unique_sources"]
    assert report["every_update_applied_c_noise_zero"]


@pytest.mark.parametrize("damage", ["missing_update", "duplicate_source", "update_order",
                                   "wrong_epoch", "wrong_data_epoch", "missing_epoch", "epoch_update"])
def test_incomplete_or_repeated_training_history_rejected(tmp_path, damage):
    directory = tmp_path / "zero"
    _write_logs(directory, 5, "zero")
    metrics = runner.rows(directory / "metrics.jsonl")
    epochs = runner.rows(directory / "epochs.jsonl")
    if damage == "missing_update":
        metrics.pop()
    elif damage == "duplicate_source":
        metrics[0]["sources"][0] = metrics[1]["sources"][0]
    elif damage == "update_order":
        metrics[0]["update"] = 2
    elif damage == "wrong_epoch":
        metrics[0]["epoch"] = 2
    elif damage == "wrong_data_epoch":
        metrics[0]["data_epoch"] = 0
    elif damage == "missing_epoch":
        epochs.pop()
    elif damage == "epoch_update":
        epochs[-1]["updates"] = 1249
    _write_rows(directory / "metrics.jsonl", metrics)
    _write_rows(directory / "epochs.jsonl", epochs)
    with pytest.raises(ValueError):
        runner.validate_logs(directory, 5)


@pytest.mark.parametrize("change", [
    {"learning_rate": .001}, {"noise_rng_before": "wrong"},
    {"noise_sha256": "wrong"}, {"pair_sha256": "wrong"},
    {"applied_noise_max_abs": .03}, {"training_noise_mode": "stochastic"},
    {"consistency_weight": .1}, {"loss": .1},
])
def test_pairing_rejects_learning_rate_rng_draw_and_actual_noise_mismatch(tmp_path, change):
    control, zero = tmp_path / "control", tmp_path / "zero"
    _write_logs(control, 5, "control")
    _write_logs(zero, 5, "zero")
    metrics = runner.rows(zero / "metrics.jsonl")
    metrics[700].update(change)
    _write_rows(zero / "metrics.jsonl", metrics)
    with pytest.raises(ValueError):
        runner.check_pairs(control, zero, 5)


def test_default_preflight_does_not_create_output_or_start_training(tmp_path, monkeypatch, capsys):
    parents = _parents(tmp_path)
    root, _ = _argv(monkeypatch, tmp_path, parents, run=False)
    def reject(*args, **kwargs):
        raise AssertionError("Preflight must not start training")
    monkeypatch.setattr(runner.subprocess, "run", reject)
    runner.main()
    assert not root.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "preflight_passed"
    assert report["old_epochs_rerun"] is False


def _fake_training(command, *, damage=None, **kwargs):
    root = Path(command[command.index("--root") + 1])
    arm = command[command.index("--arm") + 1]
    directory = root / arm
    assert len(runner.rows(directory / "metrics.jsonl")) == 1250
    assert len(runner.rows(directory / "epochs.jsonl")) == 5
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(row) + "\n" for row in _metrics(10, arm)[1250:]))
    with (directory / "epochs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("".join(json.dumps({"epoch": e, "updates": e * 250}) + "\n" for e in range(6, 11)))
    if damage == "log_prefix":
        metrics = runner.rows(directory / "metrics.jsonl")
        metrics[0]["loss"] = .333
        _write_rows(directory / "metrics.jsonl", metrics)
    elif damage == "checkpoint5":
        (directory / "epoch_005.pt").write_bytes(b"modified old epoch")
    (directory / "epoch_010.pt").write_bytes(b"epoch10")
    status = runner.read(directory / "status.json")
    status.update(epoch=10, updates=2500,
                  final_checkpoint_sha256=runner.sha(directory / "epoch_010.pt"))
    runner.write(directory / "status.json", status)
    runner.write(root / "monitor_manifest.json", {"seed": 20261008, "fixed": True})


def test_main_resumes_each_frozen_arm_to_cumulative_epoch10(tmp_path, monkeypatch):
    parents = _parents(tmp_path)
    root, codes = _argv(monkeypatch, tmp_path, parents, run=True)
    calls = []
    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        _fake_training(command, **kwargs)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.main()
    assert len(calls) == 2
    for (command, kwargs), arm in zip(calls, ("control", "zero")):
        assert command[3] == str(codes[arm] / "train_restore_consistency.py")
        assert command[command.index("--arm") + 1] == arm
        assert "--resume" in command
        assert command[command.index("--stop") + 1] == "10"
        assert command[command.index("--config") + 1] == str(root / f"{arm}_config.yaml")
        assert kwargs["cwd"] == codes[arm]
        assert command[command.index("--monitor-module") + 1] == runner.MONITORS[arm]
        assert len(runner.rows(parents[arm] / "metrics.jsonl")) == 1250
    receipt = runner.read(root / "queue.json")
    assert receipt["status"] == "stage_complete"
    assert receipt["matched_updates"] == 2500
    assert receipt["parent_artifacts_unchanged"]
    assert receipt["previous_log_prefixes_unchanged"]
    assert receipt["old_epochs_rerun"] is False


@pytest.mark.parametrize("damage, message", [("log_prefix", "Old log prefix"),
                                               ("checkpoint5", "Copied e5 checkpoint")])
def test_continuation_rejects_mutating_inherited_history(tmp_path, monkeypatch, damage, message):
    parents = _parents(tmp_path)
    root, _ = _argv(monkeypatch, tmp_path, parents, run=True)
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
                        _fake_training(command, damage=damage, **kwargs))
    with pytest.raises(RuntimeError, match=message):
        runner.main()
    assert runner.read(root / "queue.json")["status"] == "failed"


@pytest.mark.parametrize("key", ["noise_rng", "loader_rng", "next_lr"])
def test_saved_random_stream_or_lr_mismatch_stops_before_copy(tmp_path, monkeypatch, key):
    parents = _parents(tmp_path)
    root, _ = _argv(monkeypatch, tmp_path, parents, run=True)
    def mismatched(code, directory, arm, epoch):
        value = _state_check(code, directory, arm, epoch)
        if arm == "zero":
            value[key] = "different"
        return value
    monkeypatch.setattr(runner, "state_check", mismatched)
    with pytest.raises(ValueError, match="random stream or learning rate"):
        runner.main()
    assert not root.exists()
