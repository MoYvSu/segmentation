# -*- coding: utf-8 -*-
"""零注入消融的随机流配对、输入隔离、严格重载和分段续训门禁。"""

import copy
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from models.rgb_restoration import build_rgb_restorer, load_rgb_restorer
from train_restore_consistency import (
    file_sha, first_prediction, state_sha, tensor_sha, train, update, validate_options,
)
from train_rgb_restoration import save_checkpoint
from utils.config import load_config


@pytest.fixture
def config(tmp_path):
    cfg = load_config("config/train/restore_zero.yaml")
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
    c["train"].update(num_workers=0, amp=False)
    initial = tmp_path / "deployed.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = build_rgb_restorer(c["model"])
        torch.nn.init.normal_(model.correction.weight, std=.01)
    save_checkpoint(initial, dict(format=model.checkpoint_format, model_config=model.model_config,
                                  model=model.state_dict(), epoch=60))
    cfg["restore_consistency"].update(initial_checkpoint="deployed.pt", initial_sha256=file_sha(initial))
    return cfg


def batch():
    generator = torch.Generator().manual_seed(66)
    return {"input": torch.rand(4, 3, 16, 16, generator=generator),
            "target": torch.rand(4, 3, 16, 16, generator=generator),
            "valid": torch.ones(4, 1, 16, 16), "source": [f"s{i}" for i in range(4)]}


def no_monitor():
    return SimpleNamespace(prepare_monitor=lambda *args: None, run_monitor=lambda *args: {})


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def test_zero_recipe_only_changes_injection_and_keeps_deployed_time_model():
    original = load_config("config/train/restore_consistency.yaml")
    zero = load_config("config/train/restore_zero.yaml")
    for key in ("model", "train", "loss", "degradation", "masked_pretraining"):
        assert zero["rgb_restoration"][key] == original["rgb_restoration"][key]
    assert zero["rgb_restoration"]["model"]["num_steps"] == 16
    assert zero["rgb_restoration"]["model"]["kappa"] > 0
    assert zero["restore_consistency"]["noise_mode"] == "zero"
    assert zero["restore_consistency"]["consistency_weight"] == 0
    assert zero["restore_consistency"]["stage_epochs"] == 5
    assert zero["restore_consistency"]["data_epoch_offset"] == 60


def test_zero_consumes_same_draws_and_supervises_twice_without_target_leak(config):
    class Recorder(torch.nn.Module):
        num_steps = 16

        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(.1))
            self.noises, self.network_inputs = [], []

        def q_sample(self, clean, condition, t, *, noise):
            assert torch.equal(clean, condition)
            assert torch.equal(condition, values["input"])
            assert bool((t == 16).all())
            self.noises.append(noise.detach().clone())
            return condition + .03 * noise

        def denoise(self, state, condition, t):
            self.network_inputs.append((state.detach().clone(), condition.detach().clone()))
            return state + self.bias

    values = batch()
    result, generators, models = {}, {}, {}
    for arm in ("control", "zero"):
        options = {**config["restore_consistency"], "micro_batch": 4}
        if arm == "control":
            options.update(noise_mode="stochastic", consistency_weight=.1)
        model, generator = Recorder(), torch.Generator().manual_seed(73)
        result[arm] = update(model, values, torch.optim.SGD(model.parameters(), lr=1e-4),
                             torch.amp.GradScaler("cuda", enabled=False), generator, torch.device("cpu"),
                             config["rgb_restoration"], options, arm=arm)
        models[arm], generators[arm] = model, generator
    a, c = result["control"], result["zero"]
    for key in ("noise_sha256", "noise_rng_before", "pair_sha256", "sources", "terminal_step"):
        assert a[key] == c[key]
    assert torch.equal(generators["control"].get_state(), generators["zero"].get_state())
    zero_model = models["zero"]
    assert len(zero_model.noises) == len(zero_model.network_inputs) == 2
    assert all(torch.count_nonzero(noise) == 0 for noise in zero_model.noises)
    assert all(torch.equal(state, values["input"]) and torch.equal(condition, values["input"])
               for state, condition in zero_model.network_inputs)
    assert c["applied_noise_sha256"] == tensor_sha(*zero_model.noises)
    assert c["applied_noise_sha256"] != c["noise_sha256"]
    assert a["applied_noise_sha256"] == a["noise_sha256"]
    assert c["noise_mode"] == "zero" and c["consistency_weight"] == 0
    assert c["training_noise_mode"] == "zero" and c["applied_noise_max_abs"] == 0
    assert c["consistency_loss"] == 0 and c["loss"] == c["supervised_loss"]
    assert zero_model.bias.grad is not None and zero_model.bias.grad.abs() > 0


def test_zero_real_model_strict_reload_seed_independence_and_learned_gradient(config, tmp_path):
    initial = tmp_path / "deployed.pt"
    model = load_rgb_restorer(initial)
    before = state_sha(model.state_dict())
    values = batch()
    row = update(model, values, torch.optim.AdamW(model.parameters(), lr=2e-5),
                 torch.amp.GradScaler("cuda", enabled=False), torch.Generator().manual_seed(73),
                 torch.device("cpu"), config["rgb_restoration"], config["restore_consistency"], arm="zero")
    assert row["grad_norm"] > 0 and state_sha(model.state_dict()) != before
    checkpoint = tmp_path / "zero.pt"
    save_checkpoint(checkpoint, dict(format=model.checkpoint_format, model_config=model.model_config,
                                     model=model.state_dict(), training_noise_mode="zero"))
    reloaded = load_rgb_restorer(checkpoint)
    assert state_sha(model.state_dict()) == state_sha(reloaded.state_dict())
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        output = first_prediction(model, values["input"][:1], seed=1, noise_mode="zero")
        assert torch.equal(output, first_prediction(reloaded, values["input"][:1], seed=999, noise_mode="zero"))
        t = torch.tensor([16])
        condition = values["input"][:1]
        raw = model.denoise(condition, condition, t).float()
        assert torch.equal(output, (condition + (raw - condition)).clamp(0, 1))


def test_zero_stage_resume_equals_continuous_and_matches_control_pairs(config, tmp_path):
    roots = [tmp_path / "continuous", tmp_path / "segmented"]
    train(config, "zero", roots[0], device="cpu", stop=2, monitor_module=no_monitor())
    train(config, "zero", roots[1], device="cpu", stop=1, monitor_module=no_monitor())
    train(config, "zero", roots[1], device="cpu", stop=2, resume=True, monitor_module=no_monitor())
    states = [torch.load(root / "zero/last.pt", weights_only=True) for root in roots]
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "epoch", "updates"):
        assert_equal(states[0][key], states[1][key])
    assert states[0]["updates"] == 4 and states[0]["training_noise_mode"] == "zero"
    assert states[0]["scheduler"]["T_max"] == 60
    status = json.loads((roots[0] / "zero/status.json").read_text(encoding="utf-8"))
    assert status["training_noise_mode"] == status["runtime"]["first_reload_noise_protocol"] == "zero"
    assert status["strict_reload_equal"] and status["first_inference_reload_equal"]
    assert status["standard_first_inference_reload_equal"]
    assert status["no_backend_loaded"] and status["holdout_sources"] == 0
    control = copy.deepcopy(config)
    control["restore_consistency"].update(noise_mode="stochastic", consistency_weight=.1)
    train(control, "control", tmp_path / "control", device="cpu", stop=2, monitor_module=no_monitor())
    logs = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            for path in (roots[0] / "zero/metrics.jsonl", tmp_path / "control/control/metrics.jsonl")]
    assert {row["data_epoch"] for row in logs[0]} == {60, 61}
    assert len(logs[0]) == len(logs[1]) == 4
    for c, a in zip(*logs):
        for key in ("sources", "noise_rng_before", "noise_sha256", "pair_sha256", "profiles", "mask_mean"):
            assert c[key] == a[key]


@pytest.mark.parametrize("arm,mode,weight", [
    ("control", "zero", 0), ("candidate", "zero", 0), ("zero", "stochastic", 0),
    ("zero", "zero", .1), ("zero", "invalid", 0),
])
def test_rejects_arm_mode_or_weight_mismatch(config, arm, mode, weight):
    config["restore_consistency"].update(noise_mode=mode, consistency_weight=weight)
    with pytest.raises(ValueError):
        validate_options(config, arm=arm)
