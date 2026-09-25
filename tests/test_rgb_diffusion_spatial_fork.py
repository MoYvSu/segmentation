# -*- coding: utf-8 -*-
"""空间模糊正式分叉：保持父状态、严格变更范围和普通分段续训。"""

import copy
import json
import shutil
import sys
import types

import cv2
import numpy as np
import pytest
import torch

import data.rgb_restoration_dataset as restoration_data
import train_rgb_diffusion_restoration as training
from utils.config import load_config


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("spatial_fork_parent")
    sources = root / "sources"
    sources.mkdir()
    yy, xx = np.mgrid[:32, :48]
    for index in range(4):
        image = np.stack(((xx * 5 + index * 23) % 256, (yy * 8) % 256,
                          ((xx // 3 + yy // 3) % 2) * 180 + 30), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(sources / f"train_{index:03d}.png"), image)
    config = load_config("config/train/rgb_restoration_diffusion_d3_terminal50_all60_monitored.yaml")
    cfg = config["rgb_restoration"]
    cfg.update(data_dir=str(sources), image_size=32, crop_size=16, masked_pretraining=None,
               expected_train_sources=4)
    cfg["model"].update(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                        time_dim=16, num_steps=4)
    cfg["train"].update(batch_size=2, num_workers=0, epochs=3, amp=False, save_epochs=[1, 2, 3])
    cfg["monitor"].update(samples=4, every_epochs=1, thumbnail_size=48, roi_size=8)
    # 确保微型测试每轮都有足够增强样本，D3父配方仍与分叉前一致。
    cfg["degradation"]["profile_probabilities"] = [0.0, 1.0, 0.0, 0.0]
    parent = root / "parent"
    training.train(config, device=torch.device("cpu"), output_dir=parent, stop_after_epoch=1)
    return config, parent


def spatial_config(parent_config, *, monitor=False):
    config = copy.deepcopy(parent_config)
    cfg = config["rgb_restoration"]
    cfg["output_dir"] = "outputs/test_spatial_fork"
    cfg["degradation"]["spatial_blur"] = {
        "enabled": True, "probability": 0.5, "grid_size": [3, 5], "strength": 1.0, "levels": 8,
    }
    if monitor:
        cfg["monitor"]["spatial_blur"] = {"enabled": True, "samples": 4}
    return config


def assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key, value in left.items():
            assert_identical(value, right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for value, other in zip(left, right):
            assert_identical(value, other)
    else:
        assert left == right


def test_fork_preserves_parent_state_and_fixed_pairs_before_first_update(parent_run, tmp_path, monkeypatch):
    config, parent = parent_run
    fork_config = spatial_config(config)
    output = tmp_path / "fork"
    parent_bytes = (parent / "last.pt").read_bytes()
    expected = torch.load(parent / "last.pt", weights_only=True)
    original_set_epoch = restoration_data.RGBRestorationDataset.set_epoch

    def stop_before_first_fork_update(self, epoch):
        if epoch == expected["epoch"]:
            raise RuntimeError("stop before fork updates")
        return original_set_epoch(self, epoch)

    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", stop_before_first_fork_update)
    with pytest.raises(RuntimeError, match="stop before fork updates"):
        training.train(fork_config, device=torch.device("cpu"), output_dir=output,
                       fork_spatial_from=parent / "last.pt")
    actual = torch.load(output / "last.pt", weights_only=True)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "epoch", "updates", "failed_updates"):
        assert_identical(expected[key], actual[key])
    assert actual["elapsed_seconds"] >= expected["elapsed_seconds"]
    assert actual["experiment_config"] == fork_config["rgb_restoration"]
    assert actual["scheduler"]["T_max"] == 3 and actual["scheduler"]["last_epoch"] == 1
    assert not (output / "metrics.jsonl").exists() and not (output / "epochs.jsonl").exists()
    assert (output / "monitor/epoch_001_update_000002/errors.json").exists()
    for filename in ("fixed_samples.pt", "fixed_samples_manifest.json"):
        if filename.endswith(".pt"):
            assert_identical(torch.load(parent / filename, weights_only=True),
                             torch.load(output / filename, weights_only=True))
        else:
            assert (parent / filename).read_bytes() == (output / filename).read_bytes()
    continuation = actual["continuation"]
    assert continuation["parent_checkpoint"] == str((parent / "last.pt").resolve())
    assert continuation["parent_epoch"] == 1 and continuation["parent_updates"] == 2
    assert continuation["parent_elapsed_seconds"] == expected["elapsed_seconds"]
    assert set(continuation["recipe_changes"]) == {"output_dir", "degradation.spatial_blur"}
    assert (parent / "last.pt").read_bytes() == parent_bytes


def test_spatial_fork_and_normal_resume_have_identical_final_training_state(parent_run, tmp_path):
    config, parent = parent_run
    fork_config = spatial_config(config)
    complete, segmented = tmp_path / "complete", tmp_path / "segmented"
    training.train(fork_config, device=torch.device("cpu"), output_dir=complete,
                   fork_spatial_from=parent / "last.pt")
    partial = training.train(fork_config, device=torch.device("cpu"), output_dir=segmented,
                             fork_spatial_from=parent / "last.pt", stop_after_epoch=2)
    assert partial["status"] == "segment_completed" and partial["updates"] == 4
    assert partial["planned_updates"] == 6
    resumed = training.train(fork_config, device=torch.device("cpu"), output_dir=segmented,
                             resume=segmented / "last.pt")
    assert resumed["status"] == "completed" and resumed["epoch"] == 3 and resumed["updates"] == 6
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(segmented / "last.pt", weights_only=True)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics", "continuation"):
        assert_identical(expected[key], actual[key])
    rows = [json.loads(line) for line in (complete / "metrics.jsonl").read_text().splitlines()]
    epoch_rows = [json.loads(line) for line in (complete / "epochs.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == [3, 4, 5, 6]
    assert [row["epoch"] for row in epoch_rows] == [2, 3]
    for epoch in epoch_rows:
        selected = [row["spatial_blur"] for row in rows if row["epoch"] == epoch["epoch"]]
        aggregate = epoch["spatial_blur"]
        assert aggregate["blur_samples"] == 4
        assert aggregate["applied_samples"] == sum(row["applied_samples"] for row in selected)
        assert aggregate["base_sigma_sum"] == pytest.approx(sum(row["base_sigma_sum"] for row in selected))
        assert aggregate["base_sigma_mean"] == pytest.approx(aggregate["base_sigma_sum"] / 4)
    assert sum(row["spatial_blur"]["applied_samples"] for row in epoch_rows) > 0


@pytest.mark.parametrize("group,key,value", [
    ("loss", "gradient_weight", 0.3),
    ("train", "num_workers", 1),
    ("train", "epochs", 4),
    ("train", "timestep_sampling", "uniform"),
    ("model", "kappa", 0.1),
    ("monitor", "every_epochs", 2),
    ("degradation", "blur_sigma", [0.5, 5.2]),
])
def test_fork_rejects_any_other_recipe_change(parent_run, tmp_path, group, key, value):
    config, parent = parent_run
    altered = spatial_config(config)
    altered["rgb_restoration"][group][key] = value
    with pytest.raises(ValueError, match="config differs"):
        training.train(altered, device=torch.device("cpu"), output_dir=tmp_path / "invalid",
                       fork_spatial_from=parent / "last.pt")


def test_normal_resume_does_not_allow_spatial_recipe_changes(parent_run):
    config, parent = parent_run
    before = (parent / "last.pt").read_bytes()
    with pytest.raises(ValueError, match="config differs"):
        training.train(spatial_config(config), device=torch.device("cpu"), output_dir=parent,
                       resume=parent / "last.pt")
    assert (parent / "last.pt").read_bytes() == before


@pytest.mark.parametrize("conflict", [{"overfit": True}, {"resume": "dummy.pt"},
                                    {"extend_overfit_from": "dummy.pt"}])
def test_spatial_fork_entrypoint_is_exclusive(parent_run, tmp_path, conflict):
    config, parent = parent_run
    with pytest.raises(ValueError, match="cannot use"):
        training.train(spatial_config(config), device=torch.device("cpu"), output_dir=tmp_path / "invalid",
                       fork_spatial_from=parent / "last.pt", **conflict)


def test_spatial_fork_requires_fresh_output_and_newly_enabled_spatial_blur(parent_run, tmp_path):
    config, parent = parent_run
    with pytest.raises(ValueError, match="fresh output"):
        training.train(spatial_config(config), device=torch.device("cpu"), output_dir=parent,
                       fork_spatial_from=parent / "last.pt")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh output"):
        training.train(spatial_config(config), device=torch.device("cpu"), output_dir=occupied,
                       fork_spatial_from=parent / "last.pt")
    for altered in (copy.deepcopy(config), spatial_config(config)):
        altered["rgb_restoration"]["degradation"].get("spatial_blur", {})["enabled"] = False
        with pytest.raises(ValueError, match="newly enabled"):
            training.train(altered, device=torch.device("cpu"), output_dir=tmp_path / "disabled",
                           fork_spatial_from=parent / "last.pt")


@pytest.mark.parametrize("corrupt", ["mode", "updates", "epoch", "metrics", "scheduler"])
def test_spatial_fork_rejects_incomplete_or_nonformal_parent(parent_run, tmp_path, corrupt):
    config, parent = parent_run
    checkpoint = torch.load(parent / "last.pt", weights_only=True)
    if corrupt == "mode":
        checkpoint["mode"] = "overfit_engineering"
    elif corrupt == "updates":
        checkpoint["updates"] -= 1
    elif corrupt == "epoch":
        checkpoint["epoch"] = 0
    elif corrupt == "metrics":
        checkpoint["metrics"]["train_samples"] -= 1
    else:
        checkpoint["scheduler"]["T_max"] = 2
    corrupted = tmp_path / "corrupted_parent"
    corrupted.mkdir()
    for name in ("fixed_samples.pt", "fixed_samples_manifest.json"):
        shutil.copyfile(parent / name, corrupted / name)
    torch.save(checkpoint, corrupted / "last.pt")
    with pytest.raises(ValueError, match="training mode|completed formal epoch"):
        training.train(spatial_config(config), device=torch.device("cpu"), output_dir=tmp_path / "invalid",
                       fork_spatial_from=corrupted / "last.pt")


def test_ordinary_fixed_monitor_ignores_new_spatial_degradation(parent_run):
    config, _ = parent_run
    original, original_manifest = training.fixed_training_samples(config, 4)
    spatial, spatial_manifest = training.fixed_training_samples(spatial_config(config), 4)
    assert_identical(original, spatial)
    assert original_manifest == spatial_manifest


def test_spatial_monitor_is_saved_separately_and_loaded_on_resume(parent_run, tmp_path, monkeypatch):
    config, parent = parent_run
    config = spatial_config(config, monitor=True)
    built, monitored = [], []

    def build_spatial_fixed_samples(full_config, train_set, fixed_manifest):
        built.append(full_config)
        fixed = torch.load(parent / "fixed_samples.pt", weights_only=True)
        fixed["input"] = (fixed["input"] + 0.01).clamp(0, 1)
        # builder 即使消耗全局 RNG，正式训练也必须在构建后恢复父状态。
        torch.randn(9)
        return fixed, {"sources": fixed["source"], "scope": "spatial_training_monitor"}

    def monitor_spatial(model, fixed, output_dir, *, device, epoch, updates, monitor_cfg, use_amp, amp_dtype):
        monitored.append((epoch, updates))
        destination = output_dir / "spatial_monitor" / f"epoch_{epoch:03d}"
        destination.mkdir(parents=True, exist_ok=True)
        training.write_json(destination / "errors.json", {"epoch": epoch, "updates": updates})

    stub = types.ModuleType("tools.rgb_spatial_monitor")
    stub.build_spatial_fixed_samples = build_spatial_fixed_samples
    stub.monitor_spatial = monitor_spatial
    monkeypatch.setitem(sys.modules, "tools.rgb_spatial_monitor", stub)
    output = tmp_path / "with_spatial_monitor"
    training.train(config, device=torch.device("cpu"), output_dir=output,
                   fork_spatial_from=parent / "last.pt", stop_after_epoch=2)
    saved_fixed = (output / "spatial_fixed_samples.pt").read_bytes()
    training.train(config, device=torch.device("cpu"), output_dir=output, resume=output / "last.pt")
    assert len(built) == 1 and monitored == [(1, 2), (2, 4), (3, 6)]
    assert (output / "spatial_fixed_samples.pt").read_bytes() == saved_fixed
    assert (output / "spatial_fixed_samples_manifest.json").exists()
    checkpoint = torch.load(output / "last.pt", weights_only=True)
    assert "spatial_blur" not in checkpoint["output_metrics"]
    assert "monitor.spatial_blur" in checkpoint["continuation"]["recipe_changes"]
    reference = tmp_path / "without_spatial_monitor"
    training.train(spatial_config(parent_run[0]), device=torch.device("cpu"), output_dir=reference,
                   fork_spatial_from=parent / "last.pt")
    no_monitor = torch.load(reference / "last.pt", weights_only=True)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics"):
        assert_identical(checkpoint[key], no_monitor[key])
