# -*- coding: utf-8 -*-
"""验证删去 G0/G0-long 后的初始化来源及旧 checkpoint 路径。"""

import os
from pathlib import Path

import pytest
import torch
from torch import nn

import train_affinity_geometry_g1 as training
from models.affinity_geometry import AffinityGeometryDecoder
from models.fpn_decoder import FPNBackbone
from utils.config import load_config


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(2, 2)

    def get_stage_channels(self):
        return [8, 16, 32, 64]


def _reference(config, device, path):
    reference = nn.Module()
    reference.encoder = _Encoder()
    reference.decoder = nn.Module()
    reference.decoder.boundary_fpn = FPNBackbone([8, 16, 32, 64], 16)
    reference.decoder.seg_fpn = nn.Conv2d(2, 2, 1)
    return reference.to(device)


def _config(tmp_path):
    return {"paths": {"project_root": str(tmp_path)}}


def _geometry_config(**overrides):
    return {
        "reference_checkpoint": "v6.pth", "fpn_channels": 16,
        "up_channels": 16, "output_grid": 16, "seed": 42,
        **overrides,
    }


def test_skip_g0_matches_original_g0_pretraining_initialization_and_provenance(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    cfg = _geometry_config(geometry_init_mode="v6_boundary_fpn", geometry_init_checkpoint=None)
    device = torch.device("cpu")
    monkeypatch.setattr(training, "build_reference_model", _reference)

    # 按原 G0 main 的构造顺序建立独立参照，包含 reference 的随机数消耗。
    training.set_seed(42)
    reference = _reference(config, device, "v6.pth")
    expected = AffinityGeometryDecoder(
        in_channels=reference.encoder.get_stage_channels(), affinity_channels=8,
        fpn_channels=16, up_channels=16, output_grid=16,
    ).to(device)
    expected.initialize_fpn_from_boundary(reference.decoder.boundary_fpn)
    expected_rng = torch.get_rng_state().clone()

    training.set_seed(42)
    with monkeypatch.context() as context:
        def unexpected_load(*args, **kwargs):
            raise AssertionError("删去 G0 后不应读取 geometry checkpoint")
        context.setattr(torch, "load", unexpected_load)
        system, reference_path, init_path, digest = training.build_system(config, cfg, device)
    assert init_path is None
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(not p.requires_grad for p in system.reference_model.parameters())
    assert all(p.requires_grad for p in system.geometry_decoder.parameters())
    for key, value in expected.state_dict().items():
        assert torch.equal(value, system.geometry_decoder.state_dict()[key]), key

    path = tmp_path / "best.pth"
    training.save_checkpoint(
        path, system, config, epoch=0, best_score=0.5,
        reference_path=reference_path, reference_sha="v6-sha", init_path=init_path,
        digest=digest, split={}, selection_metric="val_affinity_loss",
        best_oracle_score=0.0,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["geometry_init_checkpoint"] is None
    assert payload["geometry_initialization"] == {
        "version": 1, "mode": "v6_boundary_fpn",
        "source_checkpoint": os.path.abspath(reference_path),
        "source_component": "decoder.boundary_fpn",
        "random_components": ["upsample", "affinity_head"], "seed": 42,
    }
    training.load_geometry_checkpoint_state(system, payload)


def test_existing_checkpoint_initialization_remains_default_and_strict(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "build_reference_model", _reference)
    expected = AffinityGeometryDecoder(
        in_channels=[8, 16, 32, 64], affinity_channels=8,
        fpn_channels=16, up_channels=16, output_grid=16,
    ).state_dict()
    path = tmp_path / "geometry.pth"
    torch.save({"geometry_state_dict": expected}, path)
    cfg = _geometry_config(geometry_init_checkpoint="geometry.pth")
    system, _, init_path, _ = training.build_system(_config(tmp_path), cfg, torch.device("cpu"))
    assert os.path.abspath(init_path) == str(path)
    assert system.geometry_initialization["mode"] == "checkpoint"
    for key, value in expected.items():
        assert torch.equal(value, system.geometry_decoder.state_dict()[key]), key
    expected.pop(next(iter(expected)))
    torch.save({"geometry_state_dict": expected}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        training.build_system(_config(tmp_path), cfg, torch.device("cpu"))


@pytest.mark.parametrize("cfg", [
    {},
    {"geometry_init_mode": "typo", "geometry_init_checkpoint": "old.pth"},
    {"geometry_init_mode": "v6_boundary_fpn", "geometry_init_checkpoint": "old.pth"},
    {"geometry_init_checkpoint": "old.pth", "init_from_v6_boundary_fpn": True},
    {"geometry_init_mode": "v6_boundary_fpn", "init_from_v6_boundary_fpn": False},
    {"geometry_init_mode": "v6_boundary_fpn", "feature_adapter": {"enabled": True}},
    {"geometry_init_mode": "v6_boundary_fpn", "highres_refiner": {"enabled": True}},
])
def test_ambiguous_initialization_fails_before_loading_reference(cfg):
    # 空 config 不含 paths；若过早构造模型会先抛 KeyError，无法通过此断言。
    with pytest.raises(ValueError):
        training.build_system({}, cfg, torch.device("cpu"))


def test_skip_g0_config_changes_only_initialization_and_output():
    root = Path(__file__).resolve().parents[1]
    baseline = load_config(str(root / "config/train/affinity_geometry_g2_skip_g1.yaml"))
    candidate = load_config(str(root / "config/train/affinity_geometry_g2_skip_g0.yaml"))
    before = baseline.pop("affinity_geometry_g1")
    after = candidate.pop("affinity_geometry_g1")
    assert baseline == candidate
    assert after.pop("geometry_init_mode") == "v6_boundary_fpn"
    assert after.pop("geometry_init_checkpoint") is None
    before.pop("geometry_init_checkpoint")
    assert before.pop("output_dir") != after.pop("output_dir")
    assert before == after
    assert after["epochs"] == 30
    assert after["selection_metric"] == "val_affinity_loss"
