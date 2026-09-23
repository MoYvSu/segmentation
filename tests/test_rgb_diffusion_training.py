# -*- coding: utf-8 -*-
"""扩散训练契约：固定训练配对、完整采样验收、过程缩略图与精确续训。"""

import copy
import json

import cv2
import numpy as np
import pytest
import torch

import data.rgb_restoration_dataset as restoration_data
import train_rgb_diffusion_restoration as training
from models.rgb_restoration import build_rgb_restorer, load_rgb_restorer
from utils.config import load_config


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    yy, xx = np.mgrid[:32, :48]
    for index in range(4):
        image = np.stack(((xx * 5 + index * 23) % 256, (yy * 8) % 256,
                          ((xx // 3 + yy // 3) % 2) * 180 + 30), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(source / f"train_{index:03d}.png"), image)
    config = load_config("config/train/rgb_restoration_diffusion_d1_all60.yaml")
    cfg = config["rgb_restoration"]
    cfg.update(data_dir=str(source), image_size=32, crop_size=16, masked_pretraining=None, expected_train_sources=4,
               holdout_manifest="MUST_NOT_READ_THIS_FILE.txt")
    cfg["model"].update(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                        time_dim=16, num_steps=2)
    cfg["train"].update(batch_size=2, num_workers=0, epochs=2, amp=False, save_epochs=[1, 2])
    cfg["monitor"].update(samples=4, every_epochs=1, thumbnail_size=48, roi_size=8)
    cfg["overfit"].update(samples=4, steps=4, monitor_every=2)
    return config


def test_fixed_samples_are_deterministic_training_pairs_without_changing_recipe(tiny_config, monkeypatch):
    original = copy.deepcopy(tiny_config)

    def forbidden_manifest(*args, **kwargs):
        raise AssertionError("fixed samples must remain in the training pool, not a holdout")

    monkeypatch.setattr(restoration_data, "read_manifest", forbidden_manifest)
    first, manifest = training.fixed_training_samples(tiny_config, 4)
    second, second_manifest = training.fixed_training_samples(tiny_config, 4)
    assert tiny_config == original
    assert manifest == second_manifest
    assert first["source"] == [f"train_{i:03d}.png" for i in range(4)]
    assert manifest["profile"] == "blur" and manifest["epoch"] == 0
    assert manifest["masked_pretraining"] is manifest["training_noise"] is None
    for key in ("input", "target", "valid"):
        assert torch.equal(first[key], second[key])
    assert first["input"].shape == first["target"].shape == (4, 3, 16, 16)
    assert first["valid"].shape == (4, 1, 16, 16)
    assert not torch.equal(first["input"], first["target"])


def test_gate_requires_full_metric_improvement_and_every_image_rgb():
    report = {"mean": {"input": dict.fromkeys(training.ERROR_KEYS, 0.2),
                       "ratios": {"rgb_l1": 0.79, "gradient_l1": 0.94, "mse": 0.79}},
              "per_source": [{"input": {"rgb_l1": 0.2}, "ratios": {"rgb_l1": 0.9}}]}
    assert training.overfit_gate(report, training.DEFAULT_GATE)["passed"]
    report["per_source"][0]["ratios"]["rgb_l1"] = 1.0
    assert not training.overfit_gate(report, training.DEFAULT_GATE)["passed"]
    report["per_source"][0]["ratios"]["rgb_l1"] = 0.9
    report["mean"]["ratios"]["gradient_l1"] = 0.96
    assert not training.overfit_gate(report, training.DEFAULT_GATE)["passed"]
    with pytest.raises(ValueError, match="strictly between"):
        training.overfit_gate(report, dict.fromkeys(training.ERROR_KEYS, 1.0))


def test_monitor_full_sampling_is_reproducible_and_preserves_rng(tiny_config, tmp_path):
    cfg = tiny_config["rgb_restoration"]
    fixed, _ = training.fixed_training_samples(tiny_config, 4)
    model = build_rgb_restorer(cfg["model"]).train()
    # 非零权重避免只验证初始恒等函数；完整采样应仍可复现。
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.add_(0.0001)
    before = torch.get_rng_state().clone()
    first = training.monitor(model, fixed, tmp_path / "first", device=torch.device("cpu"), epoch=0,
                             updates=0, monitor_cfg=cfg["monitor"], use_amp=False, amp_dtype=torch.bfloat16)
    assert torch.equal(before, torch.get_rng_state()) and model.training
    second = training.monitor(model, fixed, tmp_path / "second", device=torch.device("cpu"), epoch=0,
                              updates=0, monitor_cfg=cfg["monitor"], use_amp=False, amp_dtype=torch.bfloat16)
    assert first == second
    assert first["sampling_steps"] == 2 and len(first["per_source"]) == 4
    previews = list((tmp_path / "first" / "monitor").rglob("*.png"))
    assert len(previews) == 4
    assert cv2.imread(str(previews[0])).shape == (36 + 2 * (48 + 24), 48 * 3, 3)


def test_formal_all_sources_checkpoint_and_exact_rng_resume(tiny_config, tmp_path, monkeypatch):
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=complete)
    assert summary["status"] == "completed" and summary["updates"] == 4
    assert summary["failed_updates"] == 0 and summary["gate_passed"] is None
    assert summary["selection_mode"] == "final_epoch" and summary["validation"] is None
    assert not (complete / "best.pt").exists()
    metadata = json.loads((complete / "run_config.json").read_text(encoding="utf-8"))
    assert len(metadata["train_sources"]) == 4 and metadata["holdout_sources"] == []
    assert metadata["validation_cases"] == 0 and not metadata["automatic_formal_training"]
    assert (complete / "epoch_001.pt").exists() and (complete / "epoch_002.pt").exists()
    assert len(list((complete / "monitor").glob("epoch_*"))) == 3
    fixed = torch.load(complete / "fixed_samples.pt", weights_only=True)
    assert set(fixed) == {"input", "target", "valid", "source"}
    loaded = load_rgb_restorer(str(complete / "last.pt"))
    output = loaded(fixed["input"][:1], generator=torch.Generator().manual_seed(12))
    assert output.shape == fixed["input"][:1].shape and torch.isfinite(output).all()
    original_set_epoch = restoration_data.RGBRestorationDataset.set_epoch

    def interrupt_before_second_epoch(self, epoch):
        if epoch == 1:
            raise RuntimeError("simulated diffusion interruption")
        return original_set_epoch(self, epoch)

    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", interrupt_before_second_epoch)
    with pytest.raises(RuntimeError, match="simulated diffusion interruption"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted)
    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", original_set_epoch)
    changed = copy.deepcopy(tiny_config)
    changed["rgb_restoration"]["loss"]["gradient_weight"] = 0.3
    with pytest.raises(ValueError, match="config differs"):
        training.train(changed, device=torch.device("cpu"), output_dir=interrupted, resume=interrupted / "last.pt")
    resumed = training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted,
                             resume=interrupted / "last.pt")
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert resumed["updates"] == actual["updates"] == expected["updates"] == 4
    assert expected["scheduler"] == actual["scheduler"]
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    for key in ("noise_generator", "loader_generator", "torch_cpu"):
        assert torch.equal(expected["rng"][key], actual["rng"][key])
    assert actual["output_metrics"] == expected["output_metrics"]


def test_monitor_frequency_does_not_change_training(tiny_config, tmp_path):
    frequent = tmp_path / "frequent"
    sparse = tmp_path / "sparse"
    tiny_config["rgb_restoration"]["train"]["epochs"] = 3
    training.train(tiny_config, device=torch.device("cpu"), output_dir=frequent)
    altered = copy.deepcopy(tiny_config)
    altered["rgb_restoration"]["monitor"]["every_epochs"] = 5
    training.train(altered, device=torch.device("cpu"), output_dir=sparse)
    first = torch.load(frequent / "last.pt", weights_only=True)
    second = torch.load(sparse / "last.pt", weights_only=True)
    assert all(torch.equal(value, second["model"][key]) for key, value in first["model"].items())
    assert torch.equal(first["rng"]["noise_generator"], second["rng"]["noise_generator"])


def test_overfit_uses_fixed_batch_and_resumes_from_fixed_pair_checkpoint(tiny_config, tmp_path, monkeypatch):
    complete, interrupted = tmp_path / "overfit", tmp_path / "interrupted_overfit"
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=complete, overfit=True)
    assert summary["status"] in {"completed", "failed_gate"}
    assert summary["gate_passed"] == summary["gate"]["passed"]
    assert summary["mode"] == "overfit_engineering" and summary["updates"] == 4
    assert not summary["automatic_formal_training"] and not (complete / "best.pt").exists()
    rows = [json.loads(line) for line in (complete / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(row["samples"] == 4 for row in rows)
    assert all(1 <= row["t_min"] <= row["t_max"] <= 2 for row in rows)
    metadata = json.loads((complete / "run_config.json").read_text())
    assert metadata["overfit_is_holdout"] is False and metadata["planned_epochs"] is None
    assert metadata["train_sources"] == [f"train_{i:03d}.png" for i in range(4)]
    original_append = training._append_json

    def interrupt_third_update(path, row):
        if row.get("update") == 3:
            raise RuntimeError("simulated overfit interruption")
        return original_append(path, row)

    monkeypatch.setattr(training, "_append_json", interrupt_third_update)
    with pytest.raises(RuntimeError, match="simulated overfit interruption"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted, overfit=True)
    monkeypatch.setattr(training, "_append_json", original_append)
    training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted,
                   resume=interrupted / "last.pt", overfit=True)
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert expected["updates"] == actual["updates"] == 4
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    assert expected["output_metrics"] == actual["output_metrics"]
    assert expected["scheduler"] is actual["scheduler"] is None


def test_nonfinite_training_fails_and_counts_no_successful_updates(tiny_config, tmp_path, monkeypatch):
    def nonfinite_loss(prediction, target, valid, cfg, **kwargs):
        return prediction.mean() * float("nan"), {}

    monkeypatch.setattr(training, "reconstruction_loss", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=tmp_path / "failed", overfit=True)
    summary = json.loads((tmp_path / "failed" / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["updates"] == 0 and summary["failed_updates"] == 1
