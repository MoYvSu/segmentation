# -*- coding: utf-8 -*-
"""D5a 从零配对训练：相同初值、单变量配方、过程图随机隔离与裁块统计。"""

import copy
import json
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


@pytest.fixture
def tiny_config(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    yy, xx = np.mgrid[:32, :48]
    for index in range(4):
        image = np.stack(((xx * 5 + index * 23) % 256, (yy * 8) % 256,
                          ((xx // 3 + yy // 3) % 2) * 180 + 30), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(source / f"train_{index:03d}.png"), image)
    config = load_config("config/train/rgb_diffusion_d5a_control.yaml")
    cfg = config["rgb_restoration"]
    cfg.update(data_dir=str(source), image_size=32, crop_size=16, masked_pretraining=None,
               expected_train_sources=4)
    cfg["model"].update(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                        time_dim=16, num_steps=4)
    cfg["train"].update(batch_size=2, num_workers=0, epochs=2, amp=False, save_epochs=[1, 2])
    cfg["monitor"].update(samples=4, every_epochs=1, thumbnail_size=32, roi_size=8)
    for key in ("spatial_blur", "endpoint_blur", "real"):
        cfg["monitor"].pop(key)
    return config


def candidate_config(control):
    candidate = copy.deepcopy(control)
    full = load_config("config/train/rgb_diffusion_d5a.yaml")["rgb_restoration"]
    candidate["rgb_restoration"]["degradation"]["spatial_blur"]["endpoint_transition"] = (
        full["degradation"]["spatial_blur"]["endpoint_transition"])
    return candidate


def identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            identical(a, b)
    else:
        assert left == right


def test_scratch_arms_start_identically_with_no_parent(tiny_config, tmp_path, monkeypatch):
    original = restoration_data.RGBRestorationDataset.set_epoch
    calls = []

    def stop_before_update(self, epoch):
        # 固定样本生成也set_epoch(0)，仅第二次调用才是训练循环。
        calls.append(epoch)
        if len(calls) % 2 == 0:
            raise RuntimeError("stop before scratch update")
        return original(self, epoch)

    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", stop_before_update)
    initial = []
    for name, config in (("control", tiny_config), ("candidate", candidate_config(tiny_config))):
        output = tmp_path / name
        with pytest.raises(RuntimeError, match="stop before scratch update"):
            training.train(config, device=torch.device("cpu"), output_dir=output)
        checkpoint = torch.load(output / "epoch_000.pt", weights_only=True)
        initial.append(checkpoint)
        assert checkpoint["epoch"] == checkpoint["updates"] == checkpoint["failed_updates"] == 0
        assert checkpoint["optimizer"]["state"] == {}
        assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 2e-4
        assert checkpoint["scheduler"]["last_epoch"] == 0
        assert checkpoint["continuation"] is None
        assert checkpoint["metrics"]["initial_scratch_checkpoint"] is True
        assert checkpoint["output_metrics"]["epoch"] == checkpoint["output_metrics"]["updates"] == 0
        assert not (output / "metrics.jsonl").exists()
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics"):
        identical(initial[0][key], initial[1][key])
    identical(torch.load(tmp_path / "control/fixed_samples.pt", weights_only=True),
              torch.load(tmp_path / "candidate/fixed_samples.pt", weights_only=True))


def test_scratch_pair_logs_same_sources_and_profiles(tiny_config, tmp_path):
    runs = []
    for name, cfg in (("control", tiny_config), ("candidate", candidate_config(tiny_config))):
        output = tmp_path / name
        summary = training.train(cfg, device=torch.device("cpu"), output_dir=output)
        assert summary["status"] == "completed" and summary["training_complete"]
        assert summary["epoch"] == 2 and summary["updates"] == summary["planned_updates"] == 4
        metadata = json.loads((output / "run_config.json").read_text())
        assert metadata["holdout_sources"] == [] and metadata["training_pool_sources"] == 4
        assert metadata["continuation"] is None
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        runs.append(rows)
        for epoch in (1, 2):
            names = [name for row in rows if row["epoch"] == epoch for name in row["sources"]]
            assert sorted(names) == [f"train_{index:03d}.png" for index in range(4)]
    assert [row["sources"] for row in runs[0]] == [row["sources"] for row in runs[1]]
    assert [row["profiles"] for row in runs[0]] == [row["profiles"] for row in runs[1]]
    assert [row["t_histogram"] for row in runs[0]] == [row["t_histogram"] for row in runs[1]]


def test_scratch_normal_resume_remains_strict_and_exact(tiny_config, tmp_path):
    full, split = tmp_path / "full", tmp_path / "split"
    training.train(tiny_config, device=torch.device("cpu"), output_dir=full)
    training.train(tiny_config, device=torch.device("cpu"), output_dir=split, stop_after_epoch=1)
    with pytest.raises(ValueError, match="config differs"):
        training.train(candidate_config(tiny_config), device=torch.device("cpu"), output_dir=split,
                       resume=split / "last.pt")
    training.train(tiny_config, device=torch.device("cpu"), output_dir=split, resume=split / "last.pt")
    expected, actual = [torch.load(path / "last.pt", weights_only=True) for path in (full, split)]
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics", "continuation"):
        identical(expected[key], actual[key])


def test_new_monitors_do_not_change_training(tiny_config, tmp_path, monkeypatch):
    fixed, manifest = training.fixed_training_samples(tiny_config, 4)
    calls = []

    def build(*args):
        return copy.deepcopy(fixed), copy.deepcopy(manifest)

    def isolated_monitor(model, fixed, output_dir, **kwargs):
        generator = torch.Generator().manual_seed(314159)
        torch.randn(25, generator=generator)
        calls.append((kwargs["epoch"], kwargs["updates"]))

    module = types.SimpleNamespace(build_endpoint_fixed_samples=build, build_real_fixed_samples=build,
                                   monitor_endpoint=isolated_monitor, monitor_real=isolated_monitor)
    monkeypatch.setitem(sys.modules, "tools.rgb_endpoint_monitor", module)
    plain, monitored = tmp_path / "plain", tmp_path / "monitored"
    for output, enabled in ((plain, False), (monitored, True)):
        cfg = copy.deepcopy(tiny_config)
        if enabled:
            cfg["rgb_restoration"]["monitor"].update(endpoint_blur={"enabled": True, "samples": 4},
                                                     real={"enabled": True, "images": ["a", "b", "c", "d"]})
        training.train(cfg, device=torch.device("cpu"), output_dir=output)
    first, second = [torch.load(path / "last.pt", weights_only=True) for path in (plain, monitored)]
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics"):
        identical(first[key], second[key])
    assert calls == [(0, 0), (0, 0), (1, 2), (1, 2), (2, 4), (2, 4)]
    for prefix in ("endpoint", "real"):
        assert (monitored / f"{prefix}_fixed_samples.pt").exists()
        assert (monitored / f"{prefix}_fixed_samples_manifest.json").exists()


def test_pair_configs_change_only_endpoint_and_output():
    control = load_config("config/train/rgb_diffusion_d5a_control.yaml")["rgb_restoration"]
    candidate = load_config("config/train/rgb_diffusion_d5a.yaml")["rgb_restoration"]
    d4 = load_config("config/train/rgb_restoration_diffusion_d4_spatial_all60_monitored.yaml")["rgb_restoration"]
    for cfg in (control, candidate):
        assert cfg["train"]["epochs"] == 60 and cfg["train"]["batch_size"] == 4
        assert cfg["expected_train_sources"] == 1000 and cfg["split_policy"] == "all_train"
        assert cfg["seed"] == 42 and cfg["train"]["learning_rate"] == 2e-4
        assert cfg["train"]["min_learning_rate"] == 2e-6
        assert 60 * cfg["expected_train_sources"] // cfg["train"]["batch_size"] == 15000
        assert cfg["masked_pretraining"] == d4["masked_pretraining"]
        assert cfg["model"] == d4["model"] and cfg["loss"] == d4["loss"]
        assert cfg["train"]["timestep_sampling"] == "terminal_half"
        assert cfg["train"]["save_initial_checkpoint"] is True
    endpoint = candidate["degradation"]["spatial_blur"].pop("endpoint_transition")
    assert endpoint["probability"] == 0.5 and endpoint["enabled"] is True
    candidate["output_dir"] = control["output_dir"]
    assert candidate == control


def test_crop_statistics_use_actual_sample_weighted_groups():
    batch = {"profile": ["identity", "blur", "blur", "blur"],
             "spatial_blur_applied": torch.tensor([False, False, True, True]),
             "endpoint_blur_applied": torch.tensor([False, False, False, True])}
    for key in training.CROP_STAT_KEYS:
        batch[key] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    rows = training.crop_blur_statistics(batch)
    assert set(rows) == {"identity", "uniform", "oldspatial", "endpoint"}
    merged = training.merge_crop_blur_statistics([rows, {"endpoint": rows["endpoint"]}])
    assert merged["endpoint"]["samples"] == 2
    assert merged["endpoint"]["sigma_crop_mean_square_sum"] == 6
    assert merged["endpoint"]["sigma_crop_mean_square_mean"] == 3
    assert training.crop_blur_statistics({"source": ["legacy"]}) is None
