# -*- coding: utf-8 -*-
"""验证取消 SAM2 后保留预算，以及历史混合采样的随机序列兼容性。"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_affinity_geometry_g1 import build_geometry_sampler
from utils.config import load_config


def test_manual_only_explicit_budget_has_26_batches_and_replacement():
    manual = torch.arange(26)
    cfg = {
        "samples_per_epoch": 52,
        # 关闭的 SAM2 配置保留继承路径；纯人工采样无需读取该目录。
        "sam2_geometry": {"enabled": False, "dataset_dir": "not-present"},
    }
    sampler = build_geometry_sampler([manual], [1.0], cfg, 42)
    loader = DataLoader(manual, batch_size=2, sampler=sampler)
    batches = list(loader)
    sampled = torch.cat(batches)
    assert len(batches) == 26 and len(sampled) == 52
    assert len(batches) * 120 == 3120
    assert sampled.min() >= 0 and sampled.max() < 26
    assert len(torch.unique(sampled)) < len(sampled)
    assert sampler.replacement
    assert torch.equal(sampler.weights, torch.full((26,), 1 / 26).double())


def test_legacy_manual_only_keeps_unweighted_shuffle():
    # 仅继承到 SAM2 子配置中的预算不应改变历史禁用行为。
    cfg = {"sam2_geometry": {"enabled": False, "samples_per_epoch": 52}}
    assert build_geometry_sampler([range(26)], [1.0], cfg, 42) is None


@pytest.mark.parametrize("sam2_budget", [None, 52, 66])
def test_legacy_mixture_has_identical_seeded_indices_across_epochs(sam2_budget):
    parts = [range(26), range(64), range(40)]
    masses = [0.5, 0.25, 0.25]
    cfg = {"sam2_geometry": {"enabled": True}}
    if sam2_budget is not None:
        cfg["sam2_geometry"]["samples_per_epoch"] = sam2_budget
    weights = torch.cat([
        torch.full((len(part),), mass / len(part))
        for part, mass in zip(parts, masses)
    ]).double()
    original = WeightedRandomSampler(
        weights, num_samples=sam2_budget or 52, replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    actual = build_geometry_sampler(parts, masses, cfg, 42)
    assert torch.equal(actual.weights, original.weights)
    for _ in range(3):
        assert list(actual) == list(original)


def test_explicit_budget_overrides_nested_mixture_budget():
    cfg = {"samples_per_epoch": 52,
           "sam2_geometry": {"enabled": True, "samples_per_epoch": 26}}
    assert len(build_geometry_sampler([range(26), range(64)], [.5, .5], cfg, 42)) == 52


@pytest.mark.parametrize("budget", [0, -2])
def test_nonpositive_explicit_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="samples_per_epoch must be positive"):
        build_geometry_sampler([range(26)], [1.0], {"samples_per_epoch": budget}, 42)


def test_no_sam2_config_only_changes_supervision_and_explicit_sampling_budget():
    root = Path(__file__).resolve().parents[1]
    before = load_config(str(root / "config/train/affinity_geometry_g2_skip_v6.yaml"))
    after = load_config(str(root / "config/train/affinity_geometry_g2_no_sam2.yaml"))
    cfg = after["affinity_geometry_g1"]
    assert cfg["epochs"] == 120 and cfg["batch_size"] == 2
    assert cfg["reference_checkpoint"] == "outputs/stage2_joint_v3/best_model_stage2.pth"
    assert cfg["geometry_init_mode"] == "reference_boundary_fpn"
    assert cfg["selection_metric"] == "val_affinity_loss"
    assert not cfg.get("manual_train_completed_gt_dir")
    assert cfg.pop("samples_per_epoch") == 52
    assert cfg.pop("output_dir") == "outputs/20260917_align_nosam2/geometry_training"
    before["affinity_geometry_g1"].pop("output_dir")
    assert cfg["sam2_geometry"]["enabled"] is False
    assert before["affinity_geometry_g1"]["sam2_geometry"]["enabled"] is True
    cfg["sam2_geometry"]["enabled"] = True
    assert after == before
