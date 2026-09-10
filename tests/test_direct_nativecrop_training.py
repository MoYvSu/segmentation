# -*- coding: utf-8 -*-
"""局部采样实验入口合同：固定名单、损失尺度及无代理评分的阶段交接。"""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_direct_semantic_affinity as training
from utils.config import load_config


def _split_config(tmp_path, payload):
    (tmp_path / "split.json").write_text(json.dumps(payload), encoding="utf-8")
    return {
        "paths": {"project_root": str(tmp_path)},
        "direct_semantic_affinity": {"split_file": "split.json", "seed": 42},
    }


def test_explicit_split_preserves_saved_clean60_names():
    config = load_config("config/train/direct_gtv2_clean_nativecrop_60x60.yaml")
    saved = json.loads(Path("config/train/clean60_split.json").read_text(encoding="utf-8"))
    assert training.resolve_direct_split(config, saved["train"] + saved["val"]) == (
        saved["train"], saved["val"]
    )
    cfg = config["direct_semantic_affinity"]
    assert cfg["manual_samples_per_epoch"] == 64 and cfg["batch_size"] == 1
    assert cfg["head_warmup_epochs"] == cfg["joint_epochs"] == 60
    assert cfg["sam2_geometry"]["enabled"] is False
    assert config["lora"]["init_from"].endswith("20260908_133740_ssl60/best_lora.pth")


@pytest.mark.parametrize("payload", [
    {"train": ["a", "a"], "val": ["b"]},
    {"train": ["a"], "val": ["a", "b"]},
    {"train": ["a"], "val": ["c"]},
    {"train": ["a"], "val": []},
    {"train": ["a"], "val": ["b"], "seed": 7},
])
def test_explicit_split_rejects_changed_or_ambiguous_membership(tmp_path, payload):
    with pytest.raises(ValueError):
        training.resolve_direct_split(_split_config(tmp_path, payload), ["a", "b"])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "invalid"])
def test_fixed_loss_scales_require_positive_finite_values(bad):
    with pytest.raises(ValueError, match="positive finite"):
        training.validate_direct_run_options({"loss_scales": {
            "semantic": bad, "manual_affinity": 1.0, "pseudo_affinity": 1.0,
        }})


def test_deployment_disable_requires_loss_selection_and_preserves_default():
    assert training.validate_direct_run_options({}) == (True, None)
    with pytest.raises(ValueError, match="loss_selection"):
        training.validate_direct_run_options({"deployment_validation": {"enabled": False}})


def test_two_cpu_steps_keep_loss_best_without_deployment_or_calibration(tmp_path, monkeypatch):
    """用小线性头覆盖两个阶段，禁止调用模型/数据/GPU部署入口。"""
    config = load_config("config/train/direct_gtv2_clean_nativecrop_60x60.yaml")
    config["paths"]["project_root"] = str(tmp_path)
    config["sam2"]["device"] = "cpu"
    cfg = config["direct_semantic_affinity"]
    cfg.update(output_dir="run", head_warmup_epochs=1, joint_epochs=1, amp=False)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.trunk = nn.Linear(1, 1)
            self.semantic_decoder = nn.Linear(1, 1)
            self.affinity_decoder = nn.Linear(1, 1)

        def forward(self, image):
            return {
                "semantic_logits": self.semantic_decoder(image),
                "affinity_logits": self.affinity_decoder(image),
            }

    model = TinyModel()
    rows, restored, validations = [], [], []

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.manifest = {}

        def save_config(self, *args):
            pass

        def save_manifest(self):
            pass

        def append_metrics(self, row):
            rows.append(copy.deepcopy(row))

        def copy_checkpoint(self, *args):
            pass

    def forbidden(*args, **kwargs):
        pytest.fail("disabled calibration/deployment/monitor was invoked")

    def validate(*args):
        validations.append(True)
        value = 0.5 - 0.1 * len(validations)
        return {
            "val_objective_loss": value, "val_semantic_loss": value,
            "val_manual_affinity_loss": value, "semantic_iou_pearlite": 0.9,
            "semantic_iou_ferrite": 0.9, "semantic_miou": 0.9,
        }

    def restore(model, path, device):
        restored.append(Path(path).name)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.semantic_decoder.load_state_dict(payload["semantic_state_dict"])
        model.affinity_decoder.load_state_dict(payload["affinity_state_dict"])
        return payload

    batches = [{"image": torch.ones(1, 1)}]
    split = {"train": ["a"], "val": ["b"], "seed": 42, "sam2_geometry_count": 0}
    monkeypatch.setattr(sys, "argv", ["train_direct_semantic_affinity.py"])
    monkeypatch.setattr(training, "load_config", lambda _: config)
    monkeypatch.setattr(training, "check_sources", lambda _: {})
    monkeypatch.setattr(training, "build_direct_semantic_affinity_model", lambda *args: (model, {}))
    monkeypatch.setattr(training, "build_loaders", lambda *args: (
        batches, batches, None, SimpleNamespace(samples=[]), split
    ))
    monkeypatch.setattr(training, "build_semantic_criterion", lambda *args: None)
    monkeypatch.setattr(training, "configure_direct_training_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(training, "compute_semantic_loss", lambda criterion, logits, batch: logits.square().mean())
    monkeypatch.setattr(training, "compute_affinity_loss", lambda logits, *args, **kwargs: (logits.square().mean(), {}))
    monkeypatch.setattr(training, "validate_direct_objective", validate)
    monkeypatch.setattr(training, "load_checkpoint", restore)
    monkeypatch.setattr(training, "RunRecorder", Recorder)
    for name in ("calibrate_loss_scales", "evaluate_direct_deployment", "save_direct_monitor"):
        monkeypatch.setattr(training, name, forbidden)

    assert training.main() == 0
    assert len(validations) == 2
    assert restored == ["best_head_warmup_by_val_loss.pth"]
    assert [row["phase_loss_best"] for row in rows] == [1, 1]
    assert all(not any(key.startswith("deployment_") for key in row) for row in rows)
    output = tmp_path / "run"
    assert not (output / "best_deployment_direct_dual.pth").exists()
    payload = torch.load(output / "best_direct_dual.pth", map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2 and payload["phase"] == "joint_lora"
    assert payload["best_deployment_score"] is None
    assert "checkpoint_deployment_score" not in payload
    assert payload["loss_scales"] == cfg["loss_scales"]
    assert payload["loss_selection"]["joint_from_warmup_epoch"] == 1
