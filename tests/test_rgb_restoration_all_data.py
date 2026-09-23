# -*- coding: utf-8 -*-
"""全量修复训练不扣除源图、不运行验证，并保留真实训练与严格续训契约。"""

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import data.rgb_restoration_dataset as restoration_data
import train_rgb_restoration as training
from models.rgb_restoration import load_rgb_restorer
from utils.config import load_config


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    yy, xx = np.mgrid[:40, :64]
    for index in range(4):
        image = np.stack(((xx * 3 + index * 19) % 256, (yy * 5) % 256,
                          ((xx // 3 + yy // 5) % 2) * 180 + 30), axis=-1).astype(np.uint8)
        cv2.imwrite(str(source / f"train_{index:03d}.png"), image)
    manifest = tmp_path / "holdout.txt"
    manifest.write_text("train_003.png\n", encoding="utf-8")
    config = load_config("config/train/rgb_restoration_v1.yaml")
    cfg = config["rgb_restoration"]
    cfg.update(data_dir=str(source), holdout_manifest=str(manifest), image_size=32, crop_size=16)
    cfg["model"]["width"] = 4
    cfg["train"].update(batch_size=2, num_workers=0, epochs=2, amp=False, save_epochs=[1, 2])
    return config


def forbidden_validation(*args, **kwargs):
    raise AssertionError("all_train must not read a manifest or construct/run held-out validation")


def test_all_train_covers_every_source_without_reading_manifest(tiny_config, monkeypatch):
    cfg = tiny_config["rgb_restoration"]
    cfg["split_policy"] = "all_train"
    # 留下不存在的清单，证明确实不读；再删除配置键，证明不要求该配置。
    cfg["holdout_manifest"] = "does_not_exist.txt"
    monkeypatch.setattr(restoration_data, "read_manifest", forbidden_validation)
    train, holdout = training.build_datasets(tiny_config)
    assert holdout is None
    assert len(train) == 4
    assert {Path(path).name for path in train.samples} == {f"train_{i:03d}.png" for i in range(4)}
    assert {train[i]["source"] for i in range(len(train))} == {f"train_{i:03d}.png" for i in range(4)}
    del cfg["holdout_manifest"]
    without_manifest, holdout = training.build_datasets(tiny_config)
    assert holdout is None and without_manifest.samples == train.samples


def test_default_holdout_policy_preserves_sources_and_exact_inputs(tiny_config):
    legacy_train, legacy_holdout = training.build_datasets(tiny_config)
    explicit = copy.deepcopy(tiny_config)
    explicit["rgb_restoration"]["split_policy"] = "holdout"
    train, holdout = training.build_datasets(explicit)
    assert len(train.samples) == 3 and len(holdout.samples) == 1
    assert set(train.samples).isdisjoint(holdout.samples)
    for original, current in ((legacy_train, train), (legacy_holdout, holdout)):
        assert original.samples == current.samples
        for epoch in (0, 1):
            original.set_epoch(epoch)
            current.set_epoch(epoch)
            for index in range(len(current)):
                expected, actual = original[index], current[index]
                for key in ("input", "target", "valid"):
                    assert torch.equal(expected[key], actual[key])


@pytest.mark.parametrize("architecture", ["rgb_restorer", "nafnet"])
def test_all_train_saves_final_checkpoint_and_resumes_exactly(
        tiny_config, tmp_path, monkeypatch, architecture):
    cfg = tiny_config["rgb_restoration"]
    cfg["split_policy"] = "all_train"
    del cfg["holdout_manifest"]
    if architecture == "nafnet":
        cfg["model"].update(architecture="nafnet", encoder_blocks=[1, 1],
                            middle_blocks=1, decoder_blocks=[1, 1])
    # 即使从旧配置继承诊断项，也不能构造诊断数据或调用选优。
    cfg["diagnostics"] = {"local_damage": {"inherited": True}, "noise": {"inherited": True}}
    for name in ("evaluate", "evaluate_local_damage", "evaluate_noise", "checkpoint_selection",
                 "LocalDamageDiagnosticDataset", "NoiseDiagnosticDataset"):
        monkeypatch.setattr(training, name, forbidden_validation)
    monkeypatch.setattr(restoration_data, "read_manifest", forbidden_validation)

    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=complete)
    assert summary["status"] == "completed" and summary["total_updates"] == 4
    assert summary["selection_mode"] == "final_epoch" and summary["intended_checkpoint"] == "last.pt"
    assert summary["best_epoch"] == 0 and summary["best_loss"] is None
    assert not summary["best_checkpoint_available"]
    assert (complete / "last.pt").exists() and not (complete / "best.pt").exists()
    assert (complete / "epoch_001.pt").exists() and (complete / "epoch_002.pt").exists()
    assert not list(complete.glob("validation_epoch_*.png"))
    metadata = json.loads((complete / "run_config.json").read_text(encoding="utf-8"))
    assert len(metadata["train_sources"]) == 4 and metadata["holdout_sources"] == []
    assert metadata["validation_cases"] == metadata["local_diagnostic_cases"] == metadata["noise_diagnostic_cases"] == 0
    assert metadata["selection_mode"] == "final_epoch" and metadata["split_policy"] == "all_train"
    rows = [json.loads(line) for line in (complete / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        assert row["train_samples"] == 4 and row["updates"] == 2
        assert row["validation"] is row["local_damage"] is row["noise_diagnostic"] is None
        assert row["selection"]["loss"] is None
        assert row["selection"]["enabled"] is row["selection"]["eligible"] is False
    loaded = load_rgb_restorer(str(complete / "last.pt"))
    assert metadata["model_class"] == type(loaded).__name__
    assert metadata["format"] == loaded.checkpoint_format
    sample = training.build_datasets(tiny_config)[0][0]["input"][None]
    assert torch.isfinite(loaded(sample)).all()
    assert not torch.equal(loaded(sample), sample)

    original_set_epoch = restoration_data.RGBRestorationDataset.set_epoch

    def interrupt_before_second_epoch(self, epoch):
        if epoch == 1:
            raise RuntimeError("simulated all-data interruption")
        return original_set_epoch(self, epoch)

    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", interrupt_before_second_epoch)
    with pytest.raises(RuntimeError, match="simulated all-data"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted)
    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", original_set_epoch)
    altered = copy.deepcopy(tiny_config)
    altered["rgb_restoration"]["split_policy"] = "holdout"
    with pytest.raises(ValueError, match="config differs"):
        training.train(altered, device=torch.device("cpu"), output_dir=interrupted,
                       resume=interrupted / "last.pt")
    altered = copy.deepcopy(tiny_config)
    altered["rgb_restoration"]["loss"]["gradient_weight"] += 0.1
    with pytest.raises(ValueError, match="config differs"):
        training.train(altered, device=torch.device("cpu"), output_dir=interrupted,
                       resume=interrupted / "last.pt")
    resumed = training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted,
                             resume=interrupted / "last.pt")
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert resumed["total_updates"] == expected["updates"] == actual["updates"] == 4
    assert expected["scheduler"] == actual["scheduler"]
    assert actual["best_loss"] is None and actual["best_epoch"] == 0
    assert expected["format"] == actual["format"] == loaded.checkpoint_format
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    assert actual["metrics"]["validation"] is None
    assert not (interrupted / "best.pt").exists()
