# -*- coding: utf-8 -*-
"""冻结首步前端、共享后端完整保存与严格恢复的契约测试。"""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models import fused_deployment
from models.affinity_geometry import AffinityGeometryDecoder
from models.backend_adaptation import load_backend, restore_first, save_backend
from models.fpn_decoder import FPNDecoder
from models.fused_deployment import FusedPhaseAffinityModel
from models.lora import inject_trunk_lora
from models.rgb_diffusion_restoration import PixelDiffusionRGBRestorer
from utils.semantic_challenger import SemanticChallenger


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _restorer():
    model = PixelDiffusionRGBRestorer(width=4, encoder_blocks=[1], middle_blocks=1,
                                     decoder_blocks=[1], time_dim=8, num_steps=16)
    with torch.no_grad():
        model.correction.weight.normal_(std=0.02)
    return model


def test_first_matches_terminal_estimate_without_affecting_global_rng():
    model = _restorer()
    image = torch.rand(2, 3, 8, 12, requires_grad=True)
    before = torch.get_rng_state().clone()
    actual = restore_first(model, image, seed=123)
    assert torch.equal(before, torch.get_rng_state())
    assert actual.dtype == torch.float32 and not actual.requires_grad
    assert not actual.is_inference()
    with torch.no_grad():
        time = torch.full((2,), 16, dtype=torch.long)
        state = model.q_sample(image, image, time, generator=torch.Generator().manual_seed(123))
        expected = model.denoise(state, image, time).clamp(0, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # 冻结前端的结果可正常作为可训练卷积的输入。
    head = nn.Conv2d(3, 1, 1)
    head(actual).sum().backward()
    assert head.weight.grad is not None
    assert image.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_zero_strength_returns_same_tensor_without_touching_rng_or_model():
    image = torch.rand(1, 3, 4, 4, requires_grad=True)
    before = torch.get_rng_state().clone()
    assert restore_first(None, image, seed=99, strength=0) is image
    assert torch.equal(before, torch.get_rng_state())


def test_strength_blends_unclamped_clean_estimate():
    model = _restorer()
    with torch.no_grad():
        model.correction.weight.zero_()
        model.correction.bias.fill_(2.0)
    image = torch.full((1, 3, 8, 8), 0.2)
    # raw clean=2.2，0.2+0.25*(2.2-0.2)=0.7；先clamp会错误得到0.4。
    actual = restore_first(model, image, seed=5, strength=0.25)
    torch.testing.assert_close(actual, torch.full_like(actual, 0.7))
    assert model.num_steps == 16


def test_first_remains_fp32_inside_autocast():
    model = _restorer()
    image = torch.rand(1, 3, 8, 8)
    expected = restore_first(model, image, seed=3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = restore_first(model, image, seed=3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _TinyEncoder(nn.Module):
    """只替换需第三方权重的SAM2底座；其余是正式 fused 构建与加载代码。"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.repo_path = kwargs.get("sam2_repo_path")
        self.trainable_lora = False
        self.trunk = nn.Module()
        self.trunk.proj = nn.Conv2d(3, 4, 1)
        self.trunk.attn = nn.Module()
        self.trunk.attn.qkv = nn.Linear(4, 4)

    def get_stage_channels(self):
        return [4, 4, 4, 4]

    def forward(self, image):
        value = self.trunk.proj(image).permute(0, 2, 3, 1)
        value = self.trunk.attn.qkv(value).permute(0, 3, 1, 2)
        return [F.avg_pool2d(value, 2**index) for index in range(4)]


@pytest.fixture
def saved_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(fused_deployment, "SAM2Encoder", _TinyEncoder)
    encoder = _TinyEncoder()
    inject_trunk_lora(encoder, rank=2, alpha=4, target_layers=["attn.qkv"], use_grad_checkpoint=False)
    decoder_cfg = {"fpn_channels": 8, "num_classes": 2, "dropout": 0.1, "use_bn": True}
    scaffold = FPNDecoder(in_channels=[4, 4, 4, 4], **decoder_cfg)
    model = FusedPhaseAffinityModel(
        encoder, SemanticChallenger(scaffold),
        AffinityGeometryDecoder(in_channels=[4, 4, 4, 4], fpn_channels=8, up_channels=8, output_grid=8),
    ).eval()
    architecture = {
        "sam2": {"config_file": "tiny.yaml", "sam2_repo_path": "old-server/sam2",
                 "input_normalization": "legacy_none"},
        "lora": {"rank": 2, "alpha": 4, "target_layers": ["attn.qkv"]},
        "semantic_decoder": decoder_cfg,
        "affinity_decoder": {"affinity_channels": 8, "fpn_channels": 8,
                             "up_channels": 8, "output_grid": 8},
    }
    config = {"paths": {"project_root": str(tmp_path)},
              "sam2": {"sam2_repo_path": "runtime-sam2", "input_normalization": "legacy_none"},
              "affinity_deployment": {"short_reduction": "top2"},
              "inference": {"boundary_threshold": 0.65}}
    metadata = {"architecture": architecture, "sources": {"semantic": {"epoch": 13}}}
    path = tmp_path / "backend.pth"
    bundle = save_backend(model, config, metadata, path, 5, extra={"updates": 12})
    return model, bundle, config, path


def test_full_backend_reload_preserves_logits_and_runtime_path(saved_backend):
    source, saved, config, path = saved_backend
    restored, bundle = load_backend(path, config, "cpu")
    image = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        before, after = source(image), restored(image)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    assert not restored.encoder.trainable_lora and not restored.training
    assert restored.encoder.repo_path.endswith("runtime-sam2")
    assert bundle["architecture"]["sam2"]["sam2_repo_path"] == "old-server/sam2"
    assert saved["epoch"] == 5 and bundle["extra"]["updates"] == 12


@pytest.mark.parametrize("missing", ["encoder.trunk.attn.qkv.lora_A", "encoder.trunk.proj.weight",
                                    "affinity_decoder.affinity_head.3.weight"])
def test_reload_rejects_missing_encoder_lora_or_head(saved_backend, missing):
    _, bundle, config, path = saved_backend
    del bundle["model_state_dict"][missing]
    torch.save(bundle, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_backend(path, config, "cpu")


def test_reload_rejects_changed_input_normalization(saved_backend):
    _, _, config, path = saved_backend
    config = copy.deepcopy(config)
    config["sam2"]["input_normalization"] = "imagenet_v1"
    with pytest.raises(ValueError, match="归一化"):
        load_backend(path, config, "cpu")
