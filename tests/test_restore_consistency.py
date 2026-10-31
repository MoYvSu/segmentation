# -*- coding: utf-8 -*-
"""双噪声实验的输入隔离、有效更新、真实严格载入和分段续训门禁。"""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from models.rgb_restoration import build_rgb_restorer, load_rgb_restorer
from train_restore_consistency import (
    file_sha, first_prediction, isolated_callback, save_rng, state_sha, train, update,
)
from train_rgb_restoration import save_checkpoint
from utils.config import load_config


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config/train/rgb_diffusion_d5a.yaml")
    cfg["paths"]["project_root"] = str(tmp_path)
    data = tmp_path / "images"
    data.mkdir()
    rng = np.random.default_rng(719)
    for index in range(8):
        assert cv2.imwrite(str(data / f"train_{index:03d}.png"),
                           rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    c = cfg["rgb_restoration"]
    c.update(data_dir="images", expected_train_sources=8, image_size=32, crop_size=16)
    c["masked_pretraining"].update(patch_size=[4, 8], lowpass_sigma=[1., 2.])
    c["model"].update(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1], time_dim=4)
    c["train"].update(num_workers=0, amp=False, learning_rate=2e-5)
    initial = tmp_path / "deployed.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_rgb_restorer(c["model"])
        # 已训练部署模型的末层不再为零；让测试真实覆盖噪声敏感性梯度。
        torch.nn.init.normal_(model.correction.weight, std=.01)
    save_checkpoint(initial, dict(format=model.checkpoint_format, model_config=model.model_config,
                                  model=model.state_dict(), epoch=60))
    cfg["restore_consistency"] = dict(initial_checkpoint="deployed.pt", initial_sha256=file_sha(initial),
                                      expected_initial_epoch=60, data_epoch_offset=60,
                                      micro_batch=1, stage_epochs=5, consistency_weight=.1,
                                      gradient_weight=.2, monitor_epochs=[0, 1, 3, 5])
    return cfg


def batch():
    rng = torch.Generator().manual_seed(66)
    return {"input": torch.rand(4, 3, 16, 16, generator=rng),
            "target": torch.rand(4, 3, 16, 16, generator=rng),
            "valid": torch.ones(4, 1, 16, 16), "source": [f"s{i}" for i in range(4)]}


def run_update(cfg, *, arm, micro=1):
    model = load_rgb_restorer(Path(cfg["paths"]["project_root"]) / "deployed.pt")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    options = {**cfg["restore_consistency"], "micro_batch": micro}
    result = update(model, batch(), optimizer, torch.amp.GradScaler("cuda", enabled=False),
                    torch.Generator().manual_seed(73), torch.device("cpu"),
                    cfg["rgb_restoration"], options, arm=arm)
    return model, result


def test_arms_share_independent_noise_and_inputs_with_only_loss_changed(config):
    left, a = run_update(config, arm="control")
    right, b = run_update(config, arm="candidate")
    for key in ("noise_sha256", "noise_rng_before", "pair_sha256", "sources", "terminal_step"):
        assert a[key] == b[key]
    assert a["terminal_step"] == 16
    assert a["supervised_loss"] == b["supervised_loss"]
    assert a["consistency_loss"] == b["consistency_loss"] > 0
    assert a["loss"] == a["supervised_loss"]
    assert b["loss"] == pytest.approx(b["supervised_loss"] + .1 * b["consistency_loss"], abs=1e-7)
    assert state_sha(left.state_dict()) != state_sha(right.state_dict())


def test_accumulation_preserves_noise_and_effective_update(config):
    a, full = run_update(config, arm="candidate", micro=4)
    b, accumulated = run_update(config, arm="candidate", micro=1)
    assert full["noise_sha256"] == accumulated["noise_sha256"]
    assert full["pair_sha256"] == accumulated["pair_sha256"]
    assert full["loss"] == pytest.approx(accumulated["loss"], abs=1e-6)
    for left, right in zip(a.parameters(), b.parameters()):
        assert torch.allclose(left, right, atol=1e-7, rtol=1e-5)


def test_terminal_network_never_receives_clean_target_and_noise_is_not_antithetic(config):
    class Recorder(torch.nn.Module):
        num_steps = 16

        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.1))
            self.noise = []
            self.outputs = []

        def q_sample(self, clean, condition, t, *, noise):
            assert torch.equal(clean, condition)
            assert bool((t == 16).all())
            self.noise.append(noise.detach().clone())
            return condition + .03 * noise

        def denoise(self, state, condition, t):
            result = condition + self.gain * (state - condition)
            result.retain_grad()
            self.outputs.append(result)
            return result

    model = Recorder()
    update(model, batch(), torch.optim.SGD(model.parameters(), lr=1e-4),
           torch.amp.GradScaler("cuda", enabled=False), torch.Generator().manual_seed(17),
           torch.device("cpu"), config["rgb_restoration"],
           {**config["restore_consistency"], "micro_batch": 4}, arm="candidate")
    assert len(model.noise) == len(model.outputs) == 2
    assert not torch.equal(model.noise[0], model.noise[1])
    assert not torch.equal(model.noise[0], -model.noise[1])
    assert all(output.grad is not None and bool(output.grad.abs().sum() > 0) for output in model.outputs)


def test_invalid_padding_has_no_direct_loss_gradient(config):
    class PixelModel(torch.nn.Module):
        num_steps = 16

        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1, 3, 16, 16))

        def q_sample(self, clean, condition, t, *, noise):
            return condition + .03 * noise

        def denoise(self, state, condition, t):
            return state + self.bias

    model = PixelModel()
    values = batch()
    values["valid"][:, :, :, 8:] = 0
    update(model, values, torch.optim.SGD(model.parameters(), lr=1e-4),
           torch.amp.GradScaler("cuda", enabled=False), torch.Generator().manual_seed(17),
           torch.device("cpu"), config["rgb_restoration"],
           {**config["restore_consistency"], "micro_batch": 4}, arm="candidate")
    assert torch.equal(model.bias.grad[..., 8:], torch.zeros_like(model.bias.grad[..., 8:]))
    assert bool(model.bias.grad[..., :8].abs().sum() > 0)


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_state(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal_state(left, right)
    else:
        assert a == b


def test_stage_resume_matches_continuous_and_allows_strict_existing_loader(config, tmp_path):
    roots = [tmp_path / "continuous", tmp_path / "segmented"]
    train(config, "candidate", roots[0], device="cpu", stop=2)
    train(config, "candidate", roots[1], device="cpu", stop=1)
    train(config, "candidate", roots[1], device="cpu", stop=2, resume=True)
    states = [torch.load(root / "candidate/last.pt", weights_only=True) for root in roots]
    assert states[0]["updates"] == states[1]["updates"] == 4
    assert states[0]["scheduler"]["T_max"] == 60
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "epoch", "updates"):
        equal_state(states[0][key], states[1][key])
    final = roots[1] / "candidate/epoch_002.pt"
    model = load_rgb_restorer(final)
    assert state_sha(model.state_dict()) == state_sha(states[1]["model"])
    assert first_prediction(model, batch()["input"][:1]).shape == (1, 3, 16, 16)
    status = json.loads((roots[1] / "candidate/status.json").read_text(encoding="utf-8"))
    assert status["strict_reload_equal"] and status["first_inference_reload_equal"]
    assert status["no_backend_loaded"] and status["holdout_sources"] == 0
    rows = [json.loads(row) for row in (roots[0] / "candidate/metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["data_epoch"] for row in rows} == {60, 61}
    assert (roots[0] / "candidate/epoch_000.pt").exists()


def test_real_training_arms_receive_identical_complete_pool(config, tmp_path):
    root = tmp_path / "paired"
    for arm in ("control", "candidate"):
        train(config, arm, root, device="cpu", stop=1)
    logs = [[json.loads(line) for line in (root / arm / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            for arm in ("control", "candidate")]
    assert len(logs[0]) == len(logs[1]) == 2
    assert len({source for row in logs[0] for source in row["sources"]}) == 8
    for a, b in zip(*logs):
        for key in ("sources", "noise_rng_before", "noise_sha256", "pair_sha256", "profiles", "mask_mean"):
            assert a[key] == b[key]


def test_monitor_cannot_advance_training_random_streams(config, tmp_path):
    def disturb(*args, **kwargs):
        random.random()
        np.random.rand(3)
        torch.rand(8)
        return {}

    module = SimpleNamespace(prepare_monitor=disturb, run_monitor=disturb)
    roots = [tmp_path / "plain", tmp_path / "monitored"]
    train(config, "candidate", roots[0], device="cpu", stop=1)
    train(config, "candidate", roots[1], device="cpu", stop=1, monitor_module=module)
    states = [torch.load(root / "candidate/last.pt", weights_only=True) for root in roots]
    for key in ("model", "optimizer", "scheduler", "rng"):
        equal_state(states[0][key], states[1][key])


def test_rejects_changed_initial_weights_and_partial_epoch_resume(config, tmp_path):
    bad = copy.deepcopy(config)
    bad["restore_consistency"]["initial_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        train(bad, "control", tmp_path / "bad", device="cpu", stop=1)
    root = tmp_path / "partial"
    train(config, "control", root, device="cpu", stop=1)
    with (root / "control/metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Partial epoch"):
        train(config, "control", root, device="cpu", stop=2, resume=True)


def test_isolated_callback_restores_state_after_failure():
    model = torch.nn.Linear(2, 2).train()
    noise, loader = torch.Generator().manual_seed(17), torch.Generator().manual_seed(18)
    device = torch.device("cpu")
    original = save_rng(noise, loader, device)

    def broken():
        torch.rand(8, generator=noise)
        torch.rand(8, generator=loader)
        random.random()
        np.random.rand(3)
        torch.rand(8)
        model.eval()
        raise RuntimeError("diagnostic failure")

    with pytest.raises(RuntimeError, match="diagnostic failure"):
        isolated_callback(broken, model, noise, loader, device)
    equal_state(original, save_rng(noise, loader, device))
    assert model.training
