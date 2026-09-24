# -*- coding: utf-8 -*-
"""D5b短链训练：真实后验、逐步切梯度、有效样本归一化与精确续训。"""

import copy
import json

import cv2
import numpy as np
import pytest
import torch

import train_rgb_diffusion_restoration as training
from models.rgb_diffusion_restoration import PixelDiffusionRGBRestorer
from utils.config import load_config


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config(tmp_path):
    # 沿用既有扩散训练测试的小图/小模型，只关闭与本次梯度和续训契约无关的监控。
    source = tmp_path / "sources"
    source.mkdir()
    yy, xx = np.mgrid[:32, :48]
    for index in range(4):
        image = np.stack(((xx * 5 + index * 23) % 256, (yy * 8) % 256,
                          ((xx // 3 + yy // 3) % 2) * 180 + 30), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(source / f"train_{index:03d}.png"), image)
    config = load_config("config/train/rgb_diffusion_d5a_control.yaml")
    cfg = config["rgb_restoration"]
    cfg.update(data_dir=str(source), image_size=32, crop_size=16,
               masked_pretraining=None, expected_train_sources=4)
    cfg["model"].update(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                        time_dim=16, num_steps=4)
    cfg["train"].update(batch_size=2, num_workers=0, epochs=2, amp=False, save_epochs=[1, 2])
    cfg["monitor"].update(samples=2, every_epochs=1, thumbnail_size=32, roi_size=8)
    for name in ("spatial_blur", "endpoint_blur", "real"):
        cfg["monitor"].pop(name, None)
    return config


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


class ObservedBridge(PixelDiffusionRGBRestorer):
    """复用生产后验公式，以简单可微预测器观察短链状态及梯度。"""

    def __init__(self, time_only=False):
        super().__init__(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                         time_dim=16, num_steps=4, kappa=0.2)
        self.eta.copy_(torch.tensor([0.0, 0.1, 0.25, 0.6, 1.0]))
        self.slope = torch.nn.Parameter(torch.tensor(1.0 if time_only else 0.3))
        self.time_only = time_only
        self.calls = []

    def denoise(self, state, condition, t_index):
        if self.time_only:
            prediction = self.slope * t_index[:, None, None, None] + torch.zeros_like(condition)
        else:
            prediction = self.slope * state + 0.1 * condition + t_index[:, None, None, None] * 0.01
        self.calls.append(dict(state=state.detach().clone(), condition=condition.detach().clone(),
                               time=t_index.clone(), state_requires_grad=state.requires_grad,
                               slope_grad_before=(None if self.slope.grad is None else self.slope.grad.clone()),
                               prediction=prediction.detach().clone()))
        return prediction


def short_chain(model, state, prediction, condition, target, valid, timesteps, generator, backward):
    return training.short_chain_loss(
        model, state, prediction, condition, target, valid, timesteps,
        {"rgb_weight": 1.0, "gradient_weight": 0.0}, generator=generator,
        extra_steps=2, loss_weight=0.25, backward=backward,
        use_amp=False, amp_dtype=torch.bfloat16)


def test_actual_posterior_preserves_condition_and_cuts_parent_graph_without_clamping():
    model = ObservedBridge()
    state = torch.tensor([-2.0, 2.5, -3.0])[:, None, None, None].expand(3, 3, 4, 4).clone().requires_grad_()
    prediction = (state.detach() * 0.4 + 0.2).requires_grad_()
    condition = torch.tensor([0.05, 0.55, 0.85])[:, None, None, None].expand_as(state).clone()
    target, valid = torch.zeros_like(state), torch.ones(3, 1, 4, 4)
    timesteps = torch.tensor([1, 2, 4])
    seed = 7303
    auxiliary_rng = torch.Generator().manual_seed(seed)
    replay = torch.Generator().manual_seed(seed)
    global_rng = torch.get_rng_state().clone()
    backward_values = []

    def backward(loss):
        backward_values.append(float(loss.detach()))
        # 默认不retain_graph：若跨步没detach，第二次backward应直接暴露问题。
        loss.backward()

    result = short_chain(model, state, prediction, condition, target, valid, timesteps,
                         auxiliary_rng, backward)
    assert len(model.calls) == len(backward_values) == 2
    first, second = model.calls
    assert torch.equal(first["time"], torch.tensor([1, 3]))
    assert torch.equal(second["time"], torch.tensor([2]))
    assert torch.equal(first["condition"], condition[[1, 2]])
    assert torch.equal(second["condition"], condition[[2]])
    # 独立按桥的条件高斯系数重算两次转移，避免仅验证某个helper被调用。
    eta, previous = torch.tensor([0.25, 1.0])[:, None, None, None], torch.tensor([0.1, 0.6])[:, None, None, None]
    expected_first = (previous / eta) * state.detach()[[1, 2]] + (1 - previous / eta) * prediction.detach()[[1, 2]]
    expected_first += (0.2 ** 2 * previous * (eta - previous) / eta).sqrt() * torch.randn(
        (2, 3, 4, 4), generator=replay)
    expected_second = (0.25 / 0.6) * first["state"][[1]] + (1 - 0.25 / 0.6) * first["prediction"][[1]]
    expected_second += (0.2 ** 2 * 0.25 * (0.6 - 0.25) / 0.6) ** 0.5 * torch.randn(
        (1, 3, 4, 4), generator=replay)
    torch.testing.assert_close(first["state"], expected_first, atol=1e-6, rtol=0)
    torch.testing.assert_close(second["state"], expected_second, atol=1e-6, rtol=0)
    assert first["state"].min() < 0 and first["state"].max() > 1
    assert not first["state_requires_grad"] and not second["state_requires_grad"]
    assert first["slope_grad_before"] is None and second["slope_grad_before"] is not None
    assert state.grad is None and prediction.grad is None
    assert model.slope.grad is not None and model.slope.grad.abs() > 0
    assert result["samples"] == 2 and result["step_samples"] == 3 and result["network_calls"] == 2
    assert torch.equal(torch.get_rng_state(), global_rng)
    assert torch.equal(auxiliary_rng.get_state(), replay.get_state())


def test_auxiliary_loss_is_mean_over_valid_step_samples_not_mean_of_batch_means():
    model = ObservedBridge(time_only=True)
    state = torch.zeros(3, 3, 4, 4)
    parent = torch.ones_like(state, requires_grad=True)
    valid = torch.ones(3, 1, 4, 4)
    valid[1, :, 2:] = 0
    values = []

    def backward(loss):
        values.append(float(loss.detach()))
        loss.backward()

    result = short_chain(model, state, parent, state, state, valid, torch.tensor([1, 2, 3]),
                         torch.Generator().manual_seed(8), backward)
    # 两次额外调用的有效t分别[1,2]与[1]，三个sample-step误差为1,2,1。
    assert result["aux_loss"] == pytest.approx(4 / 3)
    assert result["weighted_aux_loss"] == pytest.approx(1 / 3)
    assert values == pytest.approx([0.25, 1 / 12])
    assert model.slope.grad.item() == pytest.approx(1 / 3)
    assert parent.grad is None
    assert result["samples"] == 2 and result["step_samples"] == 3 and result["network_calls"] == 2


def test_terminal_only_batch_does_not_denoise_backward_or_advance_auxiliary_rng():
    model = ObservedBridge()
    tensor, valid = torch.zeros(2, 3, 4, 4), torch.ones(2, 1, 4, 4)
    generator = torch.Generator().manual_seed(808)
    before = generator.get_state().clone()

    def forbidden_backward(_):
        raise AssertionError("t=1 must have no auxiliary backward")

    result = short_chain(model, tensor, tensor, tensor, tensor, valid, torch.ones(2, dtype=torch.long),
                         generator, forbidden_backward)
    assert model.calls == [] and torch.equal(before, generator.get_state())
    assert result["samples"] == result["step_samples"] == result["network_calls"] == 0
    assert result["aux_loss"] == result["weighted_aux_loss"] == 0


def enable_short_chain(config, probability=1.0):
    config["rgb_restoration"]["train"]["short_chain"] = dict(
        enabled=True, probability=probability, extra_steps=2, loss_weight=0.25)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_enabled_chain_keeps_main_rng_and_actual_training_pair_order(tiny_config, tmp_path):
    off = tmp_path / "off"
    on = tmp_path / "on"
    training.train(tiny_config, device=torch.device("cpu"), output_dir=off)
    enabled = copy.deepcopy(tiny_config)
    enable_short_chain(enabled)
    training.train(enabled, device=torch.device("cpu"), output_dir=on)
    a, b = [torch.load(path / "last.pt", weights_only=True) for path in (off, on)]
    for key in ("noise_generator", "loader_generator", "torch_cpu", "torch_cuda"):
        identical(a["rng"][key], b["rng"][key])
    assert "short_chain_generator" not in a["rng"] and "short_chain_generator" in b["rng"]
    assert any(not torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    first, second = rows(off / "metrics.jsonl"), rows(on / "metrics.jsonl")
    assert len(first) == len(second) == 4
    for left, right in zip(first, second):
        for key in ("sources", "profiles", "t_histogram", "learning_rate", "epoch", "update"):
            assert left[key] == right[key]
        assert right["failed_updates"] == 0
        assert right["short_chain"]["selected_batches"] == 1
        assert right["weighted_aux_loss"] == pytest.approx(right["aux_loss"] * 0.25)
    assert sum(row["short_chain"]["step_samples"] for row in second) > 0


def test_enabled_checkpoint_restores_chain_rng_and_exact_optimization(tiny_config, tmp_path):
    enable_short_chain(tiny_config, probability=0.75)
    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    training.train(tiny_config, device=torch.device("cpu"), output_dir=complete)
    part = training.train(tiny_config, device=torch.device("cpu"), output_dir=resumed, stop_after_epoch=1)
    assert part["status"] == "segment_completed"
    saved = torch.load(resumed / "last.pt", weights_only=True)
    assert "short_chain_generator" in saved["rng"]
    # 开启短链时缺随机状态必须拒绝，不能静默重新播种后声称精确续训。
    broken = copy.deepcopy(saved)
    broken["rng"].pop("short_chain_generator")
    torch.save(broken, resumed / "last.pt")
    with pytest.raises(ValueError, match="short.chain|generator|rng|RNG"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=resumed, resume=resumed / "last.pt")
    torch.save(saved, resumed / "last.pt")
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=resumed, resume=resumed / "last.pt")
    assert summary["status"] == "completed" and summary["failed_updates"] == 0
    expected, actual = [torch.load(path / "last.pt", weights_only=True) for path in (complete, resumed)]
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics", "metrics", "short_chain_totals"):
        identical(expected[key], actual[key])
    for a, b in zip(rows(complete / "metrics.jsonl"), rows(resumed / "metrics.jsonl")):
        for key in ("loss", "base_loss", "aux_loss", "weighted_aux_loss", "short_chain", "t_histogram", "sources"):
            identical(a[key], b[key])


def test_disabled_chain_matches_old_path_and_resumes_old_rng_schema(tiny_config, tmp_path):
    full, disabled, legacy = tmp_path / "full", tmp_path / "disabled", tmp_path / "legacy"
    training.train(tiny_config, device=torch.device("cpu"), output_dir=full)
    explicit = copy.deepcopy(tiny_config)
    explicit["rgb_restoration"]["train"]["short_chain"] = {"enabled": False}
    training.train(explicit, device=torch.device("cpu"), output_dir=disabled)
    training.train(tiny_config, device=torch.device("cpu"), output_dir=legacy, stop_after_epoch=1)
    saved = torch.load(legacy / "last.pt", weights_only=True)
    assert "short_chain" not in saved["experiment_config"]["train"]
    assert "short_chain_generator" not in saved["rng"]
    assert set(saved["rng"]) == {"noise_generator", "loader_generator", "torch_cpu", "torch_cuda"}
    # 保持原配方原样续训，不能为了兼容而放松任何显式新训练选项的校验。
    training.train(tiny_config, device=torch.device("cpu"), output_dir=legacy, resume=legacy / "last.pt")
    expected = torch.load(full / "last.pt", weights_only=True)
    for path in (disabled, legacy):
        actual = torch.load(path / "last.pt", weights_only=True)
        for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics"):
            identical(expected[key], actual[key])


def test_scaler_overflow_retries_same_primary_and_short_chain_random_draws(tiny_config, tmp_path, monkeypatch):
    enable_short_chain(tiny_config)
    overflow = {"enabled": False}

    class FakeScaler:
        """CPU模拟一次AMP跳过优化器更新并降尺度，不改变真实梯度运算。"""

        def __init__(self, *args, **kwargs):
            self.pending = overflow["enabled"]
            self.value = 2.0 if self.pending else 1.0
            self.skipped = False

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def is_enabled(self):
            return True

        def get_scale(self):
            return self.value

        def step(self, optimizer):
            self.skipped = self.pending
            if not self.skipped:
                optimizer.step()

        def update(self):
            if self.skipped:
                self.pending, self.value = False, 1.0

        def state_dict(self):
            return {"scale": self.value}

    monkeypatch.setattr(torch.amp, "GradScaler", FakeScaler)
    original_chain = training.short_chain_loss
    observed = []

    def capture_chain(*args, **kwargs):
        row = {"state": args[1].detach().clone(), "time": args[6].clone(),
               "aux_rng_before": kwargs["generator"].get_state().clone()}
        row["result"] = original_chain(*args, **kwargs)
        observed.append(row)
        return row["result"]

    monkeypatch.setattr(training, "short_chain_loss", capture_chain)
    reference, retried = tmp_path / "reference", tmp_path / "retried"
    normal = training.train(tiny_config, device=torch.device("cpu"), output_dir=reference)
    observed.clear()
    overflow["enabled"] = True
    recovered = training.train(tiny_config, device=torch.device("cpu"), output_dir=retried)
    assert normal["failed_updates"] == 0 and recovered["failed_updates"] == 1
    assert normal["updates"] == recovered["updates"] == 4
    assert recovered["status"] == "completed"
    assert len(observed) == 5 and observed[0]["result"]["step_samples"] > 0
    # 首次跳步后确实重放同一主噪声状态/时间以及辅助RNG，而非恰巧最终收敛相近。
    identical(observed[0], observed[1])
    expected, actual = [torch.load(path / "last.pt", weights_only=True) for path in (reference, retried)]
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "short_chain_totals", "output_metrics"):
        identical(expected[key], actual[key])
    for left, right in zip(rows(reference / "metrics.jsonl"), rows(retried / "metrics.jsonl")):
        for key in ("loss", "base_loss", "aux_loss", "weighted_aux_loss", "short_chain", "t_histogram", "sources"):
            identical(left[key], right[key])
