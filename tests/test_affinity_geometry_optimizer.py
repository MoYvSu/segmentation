# -*- coding: utf-8 -*-
"""分组学习率覆盖范围、实际更新、余弦终点与旧配方兼容。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.affinity_geometry import AffinityGeometryDecoder
from train_affinity_geometry_g1 import build_optimizer
from utils.config import load_config


def _system():
    reference = nn.Linear(2, 2).requires_grad_(False)
    return SimpleNamespace(
        geometry_decoder=AffinityGeometryDecoder(
            in_channels=[8, 16, 32, 64], fpn_channels=16,
            up_channels=16, output_grid=16,
        ),
        reference_model=reference,
        geometry_feature_adapter=None,
        geometry_highres_refiner=None,
    )


def test_default_optimizer_keeps_one_geometry_group():
    system = _system()
    optimizer = build_optimizer(system, {"learning_rate": 3e-5})
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["name"] == "geometry_decoder"
    assert group["lr"] == 3e-5
    assert [id(p) for p in group["params"]] == [
        id(p) for p in system.geometry_decoder.parameters()
    ]


def test_split_optimizer_updates_both_modules_without_touching_reference():
    system = _system()
    optimizer = build_optimizer(
        system, {"learning_rate": 1e-4, "fpn_learning_rate": 5e-5},
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert groups["geometry_fpn"]["lr"] == 5e-5
    assert groups["geometry_output"]["lr"] == 1e-4
    expected_fpn = {id(p) for p in system.geometry_decoder.geometry_fpn.parameters()}
    expected_output = {id(p) for module in (
        system.geometry_decoder.upsample, system.geometry_decoder.affinity_head,
    ) for p in module.parameters()}
    assert {id(p) for p in groups["geometry_fpn"]["params"]} == expected_fpn
    assert {id(p) for p in groups["geometry_output"]["params"]} == expected_output
    actual = [id(p) for group in groups.values() for p in group["params"]]
    assert len(actual) == len(set(actual))
    assert set(actual) == expected_fpn | expected_output

    before = {name: p.detach().clone() for name, p in system.geometry_decoder.named_parameters()}
    reference_before = {name: p.detach().clone() for name, p in system.reference_model.named_parameters()}
    features = [torch.randn(1, channels, size, size) for channels, size in (
        (8, 16), (16, 8), (32, 4), (64, 2),
    )]
    loss = system.geometry_decoder(features)["affinity_logits"].square().mean()
    loss.backward()
    optimizer.step()
    changed = {name for name, p in system.geometry_decoder.named_parameters()
               if not torch.equal(p, before[name])}
    assert any(name.startswith("geometry_fpn.") for name in changed)
    assert any(name.startswith("upsample.") for name in changed)
    assert any(name.startswith("affinity_head.") for name in changed)
    for name, p in system.reference_model.named_parameters():
        assert p.grad is None and torch.equal(p, reference_before[name])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=120, eta_min=5e-6,
    )
    previous = [group["lr"] for group in optimizer.param_groups]
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        optimizer.step()
        scheduler.step()
        current = [group["lr"] for group in optimizer.param_groups]
        assert all(5e-6 <= lr <= old for lr, old in zip(current, previous))
        previous = current
    assert previous == pytest.approx([5e-6, 5e-6])


def test_long_oldgt_config_preserves_y_data_initialization_and_loss():
    root = Path(__file__).resolve().parents[1]
    before = load_config(str(root / "config/train/affinity_geometry_g2_skip_g0.yaml"))
    after = load_config(str(root / "config/train/affinity_geometry_g2_direct_long_oldgt.yaml"))
    expected_changes = {
        "output_dir": "outputs/20260914_g2_long_oldgt/training",
        "epochs": 120, "learning_rate": 1e-4, "fpn_learning_rate": 5e-5,
        "monitor_interval": 120,
    }
    for key, value in expected_changes.items():
        assert after["affinity_geometry_g1"].pop(key) == value
        before["affinity_geometry_g1"].pop(key, None)
    assert after == before
    cfg = after["affinity_geometry_g1"]
    assert cfg["geometry_init_mode"] == "v6_boundary_fpn"
    assert cfg["geometry_init_checkpoint"] is None
    assert cfg["selection_metric"] == "val_affinity_loss"
    assert cfg["sam2_geometry"]["samples_per_epoch"] == 52
    assert cfg["sam2_geometry"]["pseudo_fraction"] == .5
    assert cfg["batch_size"] == 2 and cfg["min_learning_rate"] == 5e-6
