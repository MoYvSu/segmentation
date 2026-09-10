# -*- coding: utf-8 -*-
"""CPU 合同检查：实例集合形状、掩码注意力、冻结边界与严格 checkpoint。"""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import models.mask_set as mask_set
from models.lora import extract_lora_state_dict, inject_trunk_lora


class TinyTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1, bias=False)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 8)
        with torch.no_grad():
            self.stem.weight.fill_(0.1)
            self.attn.qkv.weight.copy_(torch.eye(8))
            self.attn.qkv.bias.zero_()

    def forward(self, image):
        features = self.stem(image)
        features = self.attn.qkv(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return [F.avg_pool2d(features, scale) for scale in (4, 8, 16, 32)]


class TinyEncoder(nn.Module):
    def __init__(self, *, input_normalization, **kwargs):
        super().__init__()
        self.trunk = TinyTrunk()
        self.input_normalization = input_normalization
        self.trainable_lora = False

    def get_stage_channels(self):
        return [8, 8, 8, 8]

    def forward(self, image):
        self.trunk.eval()
        with nullcontext() if self.trainable_lora else torch.no_grad():
            return self.trunk(image)


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    monkeypatch.setattr(mask_set, "SAM2Encoder", TinyEncoder)
    config = {
        "paths": {"project_root": str(tmp_path), "weights_dir": "weights", "sam2_ckpt": "base.pth"},
        "sam2": {"config_file": "mock", "sam2_repo_path": "mock", "input_normalization": "imagenet_v1"},
        "lora": {"rank": 2, "alpha": 4.0, "target_layers": ["attn.qkv"],
                 "gradient_checkpointing": False, "init_from": "ssl.pth"},
        "mask_set": {"input_size": 64, "mask_grid": 32, "num_queries": 7,
                     "hidden_dim": 16, "mask_dim": 8, "num_heads": 4,
                     "decoder_layers": 3, "ffn_dim": 32},
    }
    encoder = TinyEncoder(input_normalization="imagenet_v1")
    inject_trunk_lora(encoder, rank=2, alpha=4.0, target_layers=["attn.qkv"], use_grad_checkpoint=False)
    torch.save({"lora_state_dict": extract_lora_state_dict(encoder), "config": config}, tmp_path / "ssl.pth")
    return config


def test_shapes_finite_outputs_and_low_resolution_attention(tiny_config):
    model, metadata = mask_set.build_mask_set_model(tiny_config, "cpu")
    memories = []
    hooks = [layer.cross_attention.register_forward_pre_hook(
        lambda module, inputs: memories.append(inputs[1].shape[1])
    ) for layer in model.decoder.layers]
    output = model(torch.rand(2, 3, 64, 64))
    for hook in hooks:
        hook.remove()
    assert output["pred_logits"].shape == (2, 7, 3)
    assert output["pred_masks"].shape == (2, 7, 32, 32)
    assert len(output["aux_outputs"]) == 2
    assert memories == [4, 16, 64]  # stride 32/16/8，没有 32x32 掩码网格的 attention。
    for prediction in [output, *output["aux_outputs"]]:
        assert torch.isfinite(prediction["pred_logits"]).all()
        assert torch.isfinite(prediction["pred_masks"]).all()
    assert metadata["decoder_initialization"] == "random"
    assert metadata["loaded_tensors"] == 2
    assert model.parameter_summary()["constraint_passed"]


def test_fully_blocked_query_reopens_attention_without_nan():
    embeddings = torch.ones(2, 3, 4)
    features = -torch.ones(2, 4, 2, 2)
    blocked = mask_set.MaskSetDecoder.make_attention_mask(embeddings, features, num_heads=2)
    assert blocked.shape == (4, 3, 4)
    assert blocked.dtype == torch.bool and not blocked.any()
    layer = mask_set.MaskedQueryLayer(hidden_dim=8, num_heads=2, ffn_dim=16)
    output = layer(torch.randn(2, 3, 8), torch.zeros(1, 3, 8),
                   torch.randn(2, 4, 8), torch.zeros(1, 4, 8), blocked)
    assert torch.isfinite(output).all()


def test_query_initialization_preserves_distinction_after_shared_attention():
    # 对同一组权重/特征只改变查询尺度，检查第一层后查询没有迅速趋同。
    # 这是合成特征的机制回归，不代表实际图像上的实例质量。
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        config = {"mask_set": {"input_size": 64, "mask_grid": 32, "num_queries": 128,
                               "hidden_dim": 64, "mask_dim": 32, "num_heads": 8,
                               "decoder_layers": 1, "ffn_dim": 128}}
        decoder = mask_set.MaskSetDecoder([8, 16, 32, 64], config).eval()
        features = [torch.randn(1, channels, 64 // stride, 64 // stride)
                    for channels, stride in zip([8, 16, 32, 64], [4, 8, 16, 32])]
        assert 0.9 < decoder.query_features.weight.std() < 1.1
        assert 0.9 < decoder.query_positions.weight.std() < 1.1
        old_scale = copy.deepcopy(decoder)
        with torch.no_grad():
            old_scale.query_features.weight.mul_(0.02)
            old_scale.query_positions.weight.mul_(0.02)

        def pair_cosine(model):
            measured = []
            handle = model.layers[0].register_forward_hook(
                lambda _module, _inputs, output: measured.append(output.detach()))
            with torch.no_grad():
                model(features)
            handle.remove()
            query = F.normalize(measured[0][0], dim=-1)
            cosine = query @ query.T
            count = len(query)
            return float((cosine.sum() - cosine.diag().sum()) / (count * (count - 1)))

        assert pair_cosine(decoder) < pair_cosine(old_scale) - 0.2


def test_backward_respects_frozen_encoder_then_enables_only_lora(tiny_config):
    model, _ = mask_set.build_mask_set_model(tiny_config, "cpu")
    model.train()
    output = model(torch.rand(1, 3, 64, 64))
    loss = sum(item["pred_logits"].square().mean() + item["pred_masks"].square().mean()
               for item in [output, *output["aux_outputs"]])
    loss.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert model.decoder.class_head.weight.grad.abs().sum() > 0
    assert model.decoder.mask_embedding[-1].weight.grad.abs().sum() > 0
    assert model.decoder.query_features.weight.grad.abs().sum() > 0
    assert [g["name"] for g in mask_set.mask_set_parameter_groups(model, tiny_config, False)] == ["decoder"]

    model.zero_grad(set_to_none=True)
    mask_set.configure_mask_set_phase(model, train_lora=True)
    model.train()
    output = model(torch.rand(1, 3, 64, 64))
    (output["pred_masks"].square().mean() + output["pred_logits"].square().mean()).backward()
    parameters = dict(model.encoder.trunk.named_parameters())
    assert parameters["attn.qkv.lora_B"].grad.abs().sum() > 0
    for name, parameter in parameters.items():
        if "lora_A" not in name and "lora_B" not in name:
            assert not parameter.requires_grad and parameter.grad is None
    groups = mask_set.mask_set_parameter_groups(model, tiny_config, True)
    assert [g["name"] for g in groups] == ["decoder", "lora"]
    assert len({id(p) for group in groups for p in group["params"]}) == sum(len(g["params"]) for g in groups)


@pytest.mark.parametrize("use_amp", [False, True], ids=["float32", "bfloat16"])
def test_mask_only_loss_trains_mask_mlp_and_queries(tiny_config, use_amp):
    model, _ = mask_set.build_mask_set_model(tiny_config, "cpu")
    model.train()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_amp):
        output = model(torch.rand(1, 3, 64, 64))
    # 只用最终掩码，不能让类别损失掩盖 mask MLP 的断梯度。
    output["pred_masks"].float().square().mean().backward()
    required = {
        "query_features.weight", "query_positions.weight",
        "decoder_norm.weight", "decoder_norm.bias",
        *[f"mask_embedding.{index}.{suffix}"
          for index in (0, 2, 4) for suffix in ("weight", "bias")],
        *[f"layers.{index}.{suffix}" for index in range(3)
          for suffix in ("cross_attention.in_proj_weight", "self_attention.in_proj_weight", "ffn.0.weight")],
    }
    parameters = dict(model.decoder.named_parameters())
    for name in required:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
    assert model.decoder.class_head.weight.grad is None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_checkpoint_round_trip_without_original_ssl_file(tiny_config, tmp_path):
    model, _ = mask_set.build_mask_set_model(tiny_config, "cpu")
    # 模拟修正前 std=.02 的查询权重，加载必须保留已保存值，不重新初始化。
    with torch.no_grad():
        model.decoder.query_features.weight.mul_(0.02)
        model.decoder.query_positions.weight.mul_(0.02)
    model.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(image)
    payload = mask_set.make_mask_set_checkpoint(model, tiny_config, epoch=3, phase="head_warmup")
    path = tmp_path / "model.pth"
    torch.save(payload, path)
    (tmp_path / "ssl.pth").unlink()
    restored, stored = mask_set.load_mask_set_model(path, tiny_config, "cpu")
    with torch.no_grad():
        actual = restored(image)
    assert torch.equal(actual["pred_logits"], expected["pred_logits"])
    assert torch.equal(actual["pred_masks"], expected["pred_masks"])
    assert stored["epoch"] == 3
    assert all(not p.requires_grad for p in restored.parameters())
    assert "encoder_state_dict" not in payload


@pytest.mark.parametrize("damage", ["missing_decoder", "extra_decoder", "missing_lora", "normalization", "format"])
def test_checkpoint_rejects_incompatible_state(tiny_config, tmp_path, damage):
    model, _ = mask_set.build_mask_set_model(tiny_config, "cpu")
    payload = mask_set.make_mask_set_checkpoint(model, tiny_config)
    if damage == "missing_decoder":
        payload["decoder_state_dict"].pop(next(iter(payload["decoder_state_dict"])))
    elif damage == "extra_decoder":
        payload["decoder_state_dict"]["unknown.weight"] = torch.ones(1)
    elif damage == "missing_lora":
        payload["lora_state_dict"].pop(next(iter(payload["lora_state_dict"])))
    elif damage == "normalization":
        payload["input_normalization"] = "legacy_none"
    else:
        payload["format"] = "direct_semantic_affinity_v1"
    path = tmp_path / "broken.pth"
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        mask_set.load_mask_set_model(path, tiny_config, "cpu")


def test_runtime_architecture_and_normalization_mismatch_are_rejected(tiny_config, tmp_path):
    model, _ = mask_set.build_mask_set_model(tiny_config, "cpu")
    path = tmp_path / "model.pth"
    torch.save(mask_set.make_mask_set_checkpoint(model, tiny_config), path)
    different = copy.deepcopy(tiny_config)
    different["mask_set"]["num_queries"] = 8
    with pytest.raises(RuntimeError, match="architecture"):
        mask_set.load_mask_set_model(path, different, "cpu")
    different = copy.deepcopy(tiny_config)
    different["sam2"]["input_normalization"] = "legacy_none"
    with pytest.raises(RuntimeError, match="normalization"):
        mask_set.load_mask_set_model(path, different, "cpu")


def test_invalid_dimensions_fail_before_building_encoder(tiny_config):
    bad = copy.deepcopy(tiny_config)
    bad["mask_set"]["hidden_dim"] = 15
    with pytest.raises(ValueError):
        mask_set.build_mask_set_model(bad, "cpu")
