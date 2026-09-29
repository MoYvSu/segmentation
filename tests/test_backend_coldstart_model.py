# -*- coding: utf-8 -*-
"""冷启动双头的来源隔离、冻结边界与完整 fused 保存契约。"""

import copy
import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models import direct_semantic_affinity, fused_deployment
from models.backend_adaptation import load_backend, save_backend
from models.backend_coldstart import build_cold_backend
from models.direct_semantic_affinity import configure_direct_training_phase
from models.lora import extract_lora_state_dict, inject_trunk_lora


class _TinyEncoder(nn.Module):
    """只替换需第三方库及大权重的 SAM2，双头和 LoRA 仍使用正式实现。"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.checkpoint_path = kwargs.get("ckpt_path")
        self.trainable_lora = False
        self.trunk = nn.Module()
        self.trunk.image = nn.Conv2d(3, 8, 1)
        self.trunk.attn = nn.Module()
        self.trunk.attn.qkv = nn.Linear(8, 8)
        self.trunk.attn.proj = nn.Linear(8, 8)

    def get_stage_channels(self):
        return [8, 8, 8, 8]

    def forward(self, image):
        feature = self.trunk.image(image).permute(0, 2, 3, 1)
        feature = self.trunk.attn.proj(self.trunk.attn.qkv(feature))
        feature = feature.permute(0, 3, 1, 2)
        return [F.avg_pool2d(feature, 2**index) for index in range(4)]


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cold_config(tmp_path, monkeypatch):
    monkeypatch.setattr(direct_semantic_affinity, "SAM2Encoder", _TinyEncoder)
    monkeypatch.setattr(fused_deployment, "SAM2Encoder", _TinyEncoder)
    encoder = _TinyEncoder()
    inject_trunk_lora(encoder, rank=2, alpha=4,
                      target_layers=["attn.qkv", "attn.proj"], use_grad_checkpoint=False)
    ssl_state = extract_lora_state_dict(encoder)
    for value in ssl_state.values():
        value.fill_(0.125)
    ssl_path = tmp_path / "ssl.pth"
    torch.save(ssl_state, ssl_path)
    sam2_path = tmp_path / "sam2.pt"
    sam2_path.write_bytes(b"SAM2 base fixture; encoder loading is mocked")
    return {
        "paths": {"project_root": str(tmp_path), "weights_dir": ".", "sam2_ckpt": "sam2.pt"},
        "sam2": {"config_file": "tiny.yaml", "sam2_repo_path": "sam2",
                 "input_normalization": "legacy_none"},
        "lora": {"rank": 2, "alpha": 4, "target_layers": ["attn.qkv", "attn.proj"],
                 "init_from": "ssl.pth", "gradient_checkpointing": False},
        "direct_semantic_affinity": {
            "semantic_decoder": {"highres": False, "fpn_channels": 16,
                                 "dropout": 0.2, "use_group_norm": True},
            "affinity_decoder": {"channels": 8, "fpn_channels": 8, "up_channels": 16},
            "affinity_grid": 16,
            "affinity_loss": {"manual_uncovered_as_boundary": False},
        },
        "backend_adaptation": {
            "initialization": {"mode": "ssl_random_heads", "ssl_sha256": _digest(ssl_path),
                               "sam2_sha256": _digest(sam2_path)},
        },
        # 即便基础配置保留这些旧来源，也绝不能加载。
        "semantic_challenger": {"checkpoint": "missing_old_semantic.pt"},
        "affinity_deployment": {"checkpoint": "missing_old_affinity.pt", "short_reduction": "top2"},
        "inference": {"boundary_threshold": 0.65},
    }


def test_cold_heads_only_load_pure_ssl_and_respect_phase_freezing(cold_config, monkeypatch):
    original_load = torch.load
    loaded_paths = []

    def tracked_load(path, *args, **kwargs):
        loaded_paths.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    torch.manual_seed(29)
    model, metadata = build_cold_backend(cold_config, "cpu")
    assert len(loaded_paths) == 1 and loaded_paths[0].endswith("ssl.pth")
    assert model.encoder.checkpoint_path.endswith("sam2.pt")
    assert metadata["sources"]["ssl_lora"]["loaded_tensors"] == 4
    assert metadata["sources"]["sam2"]["loaded_tensors"] == 6
    assert not model.encoder.trainable_lora
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.semantic_decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.affinity_decoder.parameters())
    assert model.semantic_decoder.semantic_residual is None
    for value in extract_lora_state_dict(model).values():
        torch.testing.assert_close(value, torch.full_like(value, .125), rtol=0, atol=0)
    for head in ("semantic", "affinity"):
        assert metadata["initial_heads"][head]["initialization"] == "random"
        assert metadata["initial_heads"][head]["loaded_tensors"] == 0

    torch.manual_seed(29)
    _, repeated = build_cold_backend(cold_config, "cpu")
    assert metadata["initial_heads"] == repeated["initial_heads"]
    torch.manual_seed(30)
    _, different = build_cold_backend(cold_config, "cpu")
    for head in ("semantic", "affinity"):
        assert metadata["initial_heads"][head]["state_sha256"] != different["initial_heads"][head]["state_sha256"]

    configure_direct_training_phase(model, train_lora=True)
    assert model.encoder.trainable_lora
    for name, parameter in model.encoder.named_parameters():
        assert parameter.requires_grad == ("lora_A" in name or "lora_B" in name)


def test_cold_fused_roundtrip_uses_actual_architecture_without_source_weights(cold_config, tmp_path):
    model, metadata = build_cold_backend(cold_config, "cpu")
    architecture = metadata["architecture"]
    assert architecture["semantic_decoder"] == {
        "fpn_channels": 16, "num_classes": 2, "dropout": .2,
        "use_bn": True, "semantic_residual": False,
    }
    assert architecture["affinity_decoder"]["fpn_channels"] == 8
    path = tmp_path / "cold.pt"
    save_backend(model, cold_config, metadata, path, epoch=0)
    # 部署 bundle 必须自包含；加载不能再要求 SSL 或原 SAM2 权重文件存在。
    (tmp_path / "ssl.pth").unlink()
    (tmp_path / "sam2.pt").unlink()
    restored, bundle = load_backend(path, cold_config, "cpu")
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        expected, actual = model(image), restored(image)
    assert expected["semantic_logits"].shape == (1, 1, 32, 32)
    assert expected["affinity_logits"].shape == (1, 8, 16, 16)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert bundle["backend_adaptation"]["initial_heads"] == metadata["initial_heads"]
    assert all(not parameter.requires_grad for parameter in restored.parameters())


@pytest.mark.parametrize("location,key,value,match", [
    (("sam2",), "input_normalization", "imagenet_v1", "归一化"),
    (("backend_adaptation", "initialization"), "mode", "joint_resume", "ssl_random_heads"),
    (("backend_adaptation", "initialization"), "ssl_sha256", "0" * 64, "不匹配"),
    (("backend_adaptation", "initialization"), "ssl_sha256", "", "ssl_sha256"),
    (("direct_semantic_affinity", "semantic_decoder"), "highres", True, "simple"),
    (("direct_semantic_affinity", "semantic_decoder"), "semantic_residual", True, "semantic_residual"),
    (("direct_semantic_affinity", "affinity_decoder"), "channels", 4, "8 通道"),
    (("direct_semantic_affinity", "affinity_loss"), "manual_uncovered_as_boundary", True, "未标注"),
    (("direct_semantic_affinity", "affinity_loss"), "manual_uncovered_as_boundary", None, "未标注"),
])
def test_cold_rejects_changed_sources_normalization_or_loss_branch(cold_config, location, key, value, match):
    config = copy.deepcopy(cold_config)
    target = config
    for part in location:
        target = target[part]
    target[key] = value
    with pytest.raises(ValueError, match=match):
        build_cold_backend(config, "cpu")


def test_cold_rejects_incomplete_lora_even_if_file_hash_is_updated(cold_config, tmp_path):
    path = tmp_path / "ssl.pth"
    state = torch.load(path, map_location="cpu", weights_only=True)
    del state[next(iter(state))]
    torch.save(state, path)
    cold_config["backend_adaptation"]["initialization"]["ssl_sha256"] = _digest(path)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        build_cold_backend(cold_config, "cpu")
