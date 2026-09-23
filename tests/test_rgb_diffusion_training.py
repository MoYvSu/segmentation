# -*- coding: utf-8 -*-
"""扩散训练契约：固定训练配对、完整采样验收、过程缩略图与精确续训。"""

import copy
import json

import cv2
import numpy as np
import pytest
import torch

import data.rgb_restoration_dataset as restoration_data
import train_rgb_diffusion_restoration as training
from models.rgb_restoration import build_rgb_restorer, load_rgb_restorer
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
    config = load_config("config/train/rgb_restoration_diffusion_d1_all60.yaml")
    cfg = config["rgb_restoration"]
    cfg.update(data_dir=str(source), image_size=32, crop_size=16, masked_pretraining=None, expected_train_sources=4,
               holdout_manifest="MUST_NOT_READ_THIS_FILE.txt")
    cfg["model"].update(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                        time_dim=16, num_steps=2)
    cfg["train"].update(batch_size=2, num_workers=0, epochs=2, amp=False, save_epochs=[1, 2])
    cfg["monitor"].update(samples=4, every_epochs=1, thumbnail_size=48, roi_size=8)
    cfg["overfit"].update(samples=4, steps=4, monitor_every=2)
    return config


def test_fixed_samples_are_deterministic_training_pairs_without_changing_recipe(tiny_config, monkeypatch):
    original = copy.deepcopy(tiny_config)

    def forbidden_manifest(*args, **kwargs):
        raise AssertionError("fixed samples must remain in the training pool, not a holdout")

    monkeypatch.setattr(restoration_data, "read_manifest", forbidden_manifest)
    first, manifest = training.fixed_training_samples(tiny_config, 4)
    second, second_manifest = training.fixed_training_samples(tiny_config, 4)
    assert tiny_config == original
    assert manifest == second_manifest
    assert first["source"] == [f"train_{i:03d}.png" for i in range(4)]
    assert manifest["profile"] == "blur" and manifest["epoch"] == 0
    assert manifest["masked_pretraining"] is manifest["training_noise"] is None
    for key in ("input", "target", "valid"):
        assert torch.equal(first[key], second[key])
    assert first["input"].shape == first["target"].shape == (4, 3, 16, 16)
    assert first["valid"].shape == (4, 1, 16, 16)
    assert not torch.equal(first["input"], first["target"])


def test_gate_requires_full_metric_improvement_and_every_image_rgb():
    report = {"mean": {"input": dict.fromkeys(training.ERROR_KEYS, 0.2),
                       "ratios": {"rgb_l1": 0.79, "gradient_l1": 0.94, "mse": 0.79}},
              "per_source": [{"input": {"rgb_l1": 0.2}, "ratios": {"rgb_l1": 0.9}}]}
    assert training.overfit_gate(report, training.DEFAULT_GATE)["passed"]
    report["per_source"][0]["ratios"]["rgb_l1"] = 1.0
    assert not training.overfit_gate(report, training.DEFAULT_GATE)["passed"]
    report["per_source"][0]["ratios"]["rgb_l1"] = 0.9
    report["mean"]["ratios"]["gradient_l1"] = 0.96
    assert not training.overfit_gate(report, training.DEFAULT_GATE)["passed"]
    with pytest.raises(ValueError, match="strictly between"):
        training.overfit_gate(report, dict.fromkeys(training.ERROR_KEYS, 1.0))


def test_monitor_full_sampling_is_reproducible_and_preserves_rng(tiny_config, tmp_path):
    cfg = tiny_config["rgb_restoration"]
    fixed, _ = training.fixed_training_samples(tiny_config, 4)
    model = build_rgb_restorer(cfg["model"]).train()
    # 非零权重避免只验证初始恒等函数；完整采样应仍可复现。
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.add_(0.0001)
    before = torch.get_rng_state().clone()
    first = training.monitor(model, fixed, tmp_path / "first", device=torch.device("cpu"), epoch=0,
                             updates=0, monitor_cfg=cfg["monitor"], use_amp=False, amp_dtype=torch.bfloat16)
    assert torch.equal(before, torch.get_rng_state()) and model.training
    second = training.monitor(model, fixed, tmp_path / "second", device=torch.device("cpu"), epoch=0,
                              updates=0, monitor_cfg=cfg["monitor"], use_amp=False, amp_dtype=torch.bfloat16)
    assert first == second
    assert first["sampling_steps"] == 2 and len(first["per_source"]) == 4
    previews = list((tmp_path / "first" / "monitor").rglob("*.png"))
    assert len(previews) == 4
    assert cv2.imread(str(previews[0])).shape == (36 + 2 * (48 + 24), 48 * 3, 3)


def test_formal_all_sources_checkpoint_and_exact_rng_resume(tiny_config, tmp_path, monkeypatch):
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=complete)
    assert summary["status"] == "completed" and summary["updates"] == 4
    assert summary["failed_updates"] == 0 and summary["gate_passed"] is None
    assert summary["selection_mode"] == "final_epoch" and summary["validation"] is None
    assert not (complete / "best.pt").exists()
    metadata = json.loads((complete / "run_config.json").read_text(encoding="utf-8"))
    assert len(metadata["train_sources"]) == 4 and metadata["holdout_sources"] == []
    assert metadata["validation_cases"] == 0 and not metadata["automatic_formal_training"]
    assert (complete / "epoch_001.pt").exists() and (complete / "epoch_002.pt").exists()
    assert len(list((complete / "monitor").glob("epoch_*"))) == 3
    fixed = torch.load(complete / "fixed_samples.pt", weights_only=True)
    assert set(fixed) == {"input", "target", "valid", "source"}
    loaded = load_rgb_restorer(str(complete / "last.pt"))
    output = loaded(fixed["input"][:1], generator=torch.Generator().manual_seed(12))
    assert output.shape == fixed["input"][:1].shape and torch.isfinite(output).all()
    original_set_epoch = restoration_data.RGBRestorationDataset.set_epoch

    def interrupt_before_second_epoch(self, epoch):
        if epoch == 1:
            raise RuntimeError("simulated diffusion interruption")
        return original_set_epoch(self, epoch)

    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", interrupt_before_second_epoch)
    with pytest.raises(RuntimeError, match="simulated diffusion interruption"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted)
    monkeypatch.setattr(restoration_data.RGBRestorationDataset, "set_epoch", original_set_epoch)
    changed = copy.deepcopy(tiny_config)
    changed["rgb_restoration"]["loss"]["gradient_weight"] = 0.3
    with pytest.raises(ValueError, match="config differs"):
        training.train(changed, device=torch.device("cpu"), output_dir=interrupted, resume=interrupted / "last.pt")
    resumed = training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted,
                             resume=interrupted / "last.pt")
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert resumed["updates"] == actual["updates"] == expected["updates"] == 4
    assert expected["scheduler"] == actual["scheduler"]
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    for key in ("noise_generator", "loader_generator", "torch_cpu"):
        assert torch.equal(expected["rng"][key], actual["rng"][key])
    assert actual["output_metrics"] == expected["output_metrics"]


def test_monitor_frequency_does_not_change_training(tiny_config, tmp_path):
    frequent = tmp_path / "frequent"
    sparse = tmp_path / "sparse"
    tiny_config["rgb_restoration"]["train"]["epochs"] = 3
    training.train(tiny_config, device=torch.device("cpu"), output_dir=frequent)
    altered = copy.deepcopy(tiny_config)
    altered["rgb_restoration"]["monitor"]["every_epochs"] = 5
    training.train(altered, device=torch.device("cpu"), output_dir=sparse)
    first = torch.load(frequent / "last.pt", weights_only=True)
    second = torch.load(sparse / "last.pt", weights_only=True)
    assert all(torch.equal(value, second["model"][key]) for key, value in first["model"].items())
    assert torch.equal(first["rng"]["noise_generator"], second["rng"]["noise_generator"])


@pytest.mark.parametrize("policy", [None, "terminal_half"])
def test_segment_stop_preserves_full_schedule_and_exact_resume(tiny_config, tmp_path, monkeypatch, policy):
    complete, segmented = tmp_path / "complete", tmp_path / "segmented"
    cfg = tiny_config["rgb_restoration"]
    cfg["train"].update(epochs=4, save_epochs=[1, 4])
    # T=2时两种采样本来相同；这里用T=4确保覆盖新分布的实际训练和续训。
    cfg["model"]["num_steps"] = 4
    if policy is not None:
        cfg["train"]["timestep_sampling"] = policy
    cfg["monitor"]["every_epochs"] = 5
    observed = []
    original_sampler = training.sample_timesteps

    def capture_timesteps(count, num_steps, *, device, generator, policy="uniform"):
        sampled = original_sampler(count, num_steps, device=device, generator=generator, policy=policy)
        observed.append((policy, torch.bincount(sampled, minlength=num_steps + 1)[1:].tolist()))
        return sampled

    monkeypatch.setattr(training, "sample_timesteps", capture_timesteps)
    original = copy.deepcopy(tiny_config)
    finished = training.train(tiny_config, device=torch.device("cpu"), output_dir=complete)
    sampling = {"policy": policy or "uniform", "terminal_probability": 0.5 if policy else 0.25,
                "num_steps": 4}
    assert finished["timestep_sampling"] == sampling
    # 日志保留实际抽样数，不能由期望概率或t_min/t_max反推。
    rows = [json.loads(line) for line in (complete / "metrics.jsonl").read_text().splitlines()]
    epochs = [json.loads(line) for line in (complete / "epochs.jsonl").read_text().splitlines()]
    assert len(rows) == len(observed) == 8 and len(epochs) == 4
    for row, (actual_policy, histogram) in zip(rows, observed):
        assert actual_policy == sampling["policy"]
        assert row["t_histogram"] == histogram
        assert len(histogram) == 4 and sum(histogram) == row["samples"] == 2
    for epoch in epochs:
        update_rows = [row for row in rows if row["epoch"] == epoch["epoch"]]
        assert epoch["t_histogram"] == [sum(row["t_histogram"][t] for row in update_rows)
                                         for t in range(4)]
        assert sum(epoch["t_histogram"]) == 4
    partial = training.train(tiny_config, device=torch.device("cpu"), output_dir=segmented,
                             stop_after_epoch=2)
    assert tiny_config == original
    assert partial["status"] == "segment_completed" and partial["training_complete"] is False
    assert partial["epoch"] == partial["run_until_epoch"] == 2
    assert partial["updates"] == partial["segment_target_updates"] == 4
    assert partial["planned_updates"] == 8
    metadata = json.loads((segmented / "run_config.json").read_text(encoding="utf-8"))
    assert metadata["planned_epochs"] == 4 and metadata["planned_updates"] == 8
    assert metadata["initial_stop_after_epoch"] == 2 and metadata["initial_segment_target_updates"] == 4
    assert metadata["timestep_sampling"] == partial["timestep_sampling"] == sampling
    # 第2轮既不是常规monitor轮次也不是完整预算终点，分段退出仍须保存过程图。
    assert (segmented / "monitor/epoch_002_update_000004/errors.json").exists()
    checkpoint = torch.load(segmented / "last.pt", weights_only=True)
    assert checkpoint["epoch"] == 2 and checkpoint["updates"] == 4
    assert checkpoint["scheduler"]["T_max"] == 4
    assert checkpoint["scheduler"]["last_epoch"] == 2
    expected_lr = (cfg["train"]["learning_rate"] + cfg["train"]["min_learning_rate"]) / 2
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(expected_lr)
    # 不传stop即可用原配方继续剩余预算，无需更改学习率日程或启动新训练流。
    resumed = training.train(tiny_config, device=torch.device("cpu"), output_dir=segmented,
                             resume=segmented / "last.pt")
    assert resumed["status"] == "completed" and resumed["training_complete"] is True
    assert resumed["epoch"] == resumed["run_until_epoch"] == 4
    assert resumed["updates"] == resumed["segment_target_updates"] == resumed["planned_updates"] == 8
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(segmented / "last.pt", weights_only=True)

    def assert_identical(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key, value in left.items():
                assert_identical(value, right[key])
        elif isinstance(left, (list, tuple)):
            assert type(left) is type(right) and len(left) == len(right)
            for value, other in zip(left, right):
                assert_identical(value, other)
        else:
            assert left == right

    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "output_metrics"):
        assert_identical(expected[key], actual[key])
    assert all(actual_policy == sampling["policy"] for actual_policy, _ in observed)
    resumed_rows = [json.loads(line) for line in (segmented / "metrics.jsonl").read_text().splitlines()]
    assert [row["t_histogram"] for row in resumed_rows] == [row["t_histogram"] for row in rows]


@pytest.mark.parametrize("policy", [None, "uniform"])
def test_uniform_timestep_sampling_preserves_original_draws_and_rng(policy):
    expected_generator = torch.Generator().manual_seed(314159)
    actual_generator = torch.Generator().manual_seed(314159)
    expected = torch.randint(1, 17, (257,), generator=expected_generator)
    kwargs = {} if policy is None else {"policy": policy}
    actual = training.sample_timesteps(257, 16, device=torch.device("cpu"),
                                       generator=actual_generator, **kwargs)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_generator.get_state(), expected_generator.get_state())
    assert torch.equal(torch.randn(17, generator=actual_generator),
                       torch.randn(17, generator=expected_generator))


def test_terminal_half_sampling_gives_exact_half_mass_to_terminal_state():
    count, num_steps = 500_000, 16
    generator = torch.Generator().manual_seed(20260923)
    actual = training.sample_timesteps(count, num_steps, device=torch.device("cpu"),
                                       generator=generator, policy="terminal_half")
    assert actual.dtype == torch.int64 and actual.shape == (count,)
    counts = torch.bincount(actual, minlength=num_steps + 1)
    assert counts[0] == 0 and len(counts) == num_steps + 1
    probabilities = counts[1:].double() / count
    # 不能用“50%强制终点+50%仍均匀抽1..T”，后者实际终点概率为53.125%。
    assert float(probabilities[-1]) == pytest.approx(0.5, abs=0.003)
    assert torch.all(torch.abs(probabilities[:-1] - 0.5 / (num_steps - 1)) < 0.0015)


def test_unknown_timestep_sampling_fails_before_model_construction(tiny_config, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="timestep|sampling|policy"):
        training.sample_timesteps(4, 16, device=torch.device("cpu"),
                                   generator=torch.Generator(), policy="terminal_typo")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("invalid sampling policy must fail before constructing the model")

    monkeypatch.setattr(training, "build_rgb_restorer", forbidden_model)
    tiny_config["rgb_restoration"]["train"]["timestep_sampling"] = "terminal_typo"
    with pytest.raises(ValueError, match="timestep|sampling|policy"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=tmp_path / "invalid_policy")


@pytest.mark.parametrize("stop", [0, -1, 3, 1.5, True, "1"])
def test_segment_stop_rejects_invalid_epoch(tiny_config, tmp_path, stop):
    with pytest.raises(ValueError):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=tmp_path / "invalid",
                       stop_after_epoch=stop)


def test_segment_stop_rejects_overfit_and_completed_endpoint(tiny_config, tmp_path):
    with pytest.raises(ValueError):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=tmp_path / "overfit",
                       overfit=True, stop_after_epoch=1)
    output = tmp_path / "segmented"
    training.train(tiny_config, device=torch.device("cpu"), output_dir=output, stop_after_epoch=1)
    saved_bytes = (output / "last.pt").read_bytes()
    with pytest.raises(ValueError):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=output,
                       resume=output / "last.pt", stop_after_epoch=1)
    assert (output / "last.pt").read_bytes() == saved_bytes


def test_d2_recipe_changes_only_noise_coefficient():
    d1 = load_config("config/train/rgb_restoration_diffusion_d1_all60_monitored.yaml")
    d2 = load_config("config/train/rgb_restoration_diffusion_d2_k003_all60_monitored.yaml")
    first, second = d1["rgb_restoration"], d2["rgb_restoration"]
    assert first["model"]["kappa"] == 0.1 and second["model"]["kappa"] == 0.03
    assert first["output_dir"] != second["output_dir"]
    second["model"]["kappa"] = first["model"]["kappa"]
    second["output_dir"] = first["output_dir"]
    assert d1 == d2


def test_d3_recipe_changes_only_timestep_sampling():
    d2 = load_config("config/train/rgb_restoration_diffusion_d2_k003_all60_monitored.yaml")
    d3 = load_config("config/train/rgb_restoration_diffusion_d3_terminal50_all60_monitored.yaml")
    first, second = d2["rgb_restoration"], d3["rgb_restoration"]
    assert "timestep_sampling" not in first["train"]
    assert second["train"].pop("timestep_sampling") == "terminal_half"
    assert first["output_dir"] != second["output_dir"]
    second["output_dir"] = first["output_dir"]
    assert d2 == d3


def test_overfit_uses_fixed_batch_and_resumes_from_fixed_pair_checkpoint(tiny_config, tmp_path, monkeypatch):
    complete, interrupted = tmp_path / "overfit", tmp_path / "interrupted_overfit"
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=complete, overfit=True)
    assert summary["status"] in {"completed", "failed_gate"}
    assert summary["gate_passed"] == summary["gate"]["passed"]
    assert summary["mode"] == "overfit_engineering" and summary["updates"] == 4
    assert not summary["automatic_formal_training"] and not (complete / "best.pt").exists()
    rows = [json.loads(line) for line in (complete / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(row["samples"] == 4 for row in rows)
    assert all(1 <= row["t_min"] <= row["t_max"] <= 2 for row in rows)
    metadata = json.loads((complete / "run_config.json").read_text())
    assert metadata["overfit_is_holdout"] is False and metadata["planned_epochs"] is None
    assert metadata["train_sources"] == [f"train_{i:03d}.png" for i in range(4)]
    original_append = training._append_json

    def interrupt_third_update(path, row):
        if row.get("update") == 3:
            raise RuntimeError("simulated overfit interruption")
        return original_append(path, row)

    monkeypatch.setattr(training, "_append_json", interrupt_third_update)
    with pytest.raises(RuntimeError, match="simulated overfit interruption"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted, overfit=True)
    monkeypatch.setattr(training, "_append_json", original_append)
    training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted,
                   resume=interrupted / "last.pt", overfit=True)
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert expected["updates"] == actual["updates"] == 4
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    assert expected["output_metrics"] == actual["output_metrics"]
    assert expected["scheduler"] is actual["scheduler"] is None


def test_nonfinite_training_fails_and_counts_no_successful_updates(tiny_config, tmp_path, monkeypatch):
    def nonfinite_loss(prediction, target, valid, cfg, **kwargs):
        return prediction.mean() * float("nan"), {}

    monkeypatch.setattr(training, "reconstruction_loss", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=tmp_path / "failed", overfit=True)
    summary = json.loads((tmp_path / "failed" / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["updates"] == 0 and summary["failed_updates"] == 1


def test_first_step_monitor_does_not_change_training_or_full_sampling(tiny_config, tmp_path):
    training.train(tiny_config, device=torch.device("cpu"), output_dir=tmp_path / "full", overfit=True)
    additional = copy.deepcopy(tiny_config)
    additional["rgb_restoration"]["monitor"]["include_first_step"] = True
    training.train(additional, device=torch.device("cpu"), output_dir=tmp_path / "both", overfit=True)
    expected = torch.load(tmp_path / "full/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "both/last.pt", weights_only=True)
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    assert torch.equal(expected["rng"]["noise_generator"], actual["rng"]["noise_generator"])
    report = copy.deepcopy(actual["output_metrics"])
    first = report.pop("first_step")
    assert report == expected["output_metrics"] and first["samples"] == 4
    assert len(list((tmp_path / "both/monitor/epoch_000_update_000004").glob("*_first.png"))) == 4


def test_overfit_extension_preserves_training_stream_and_parent(tiny_config, tmp_path):
    parent, extended, complete = [tmp_path / name for name in ("parent", "extended", "complete")]
    short = copy.deepcopy(tiny_config)
    short["rgb_restoration"]["overfit"]["steps"] = 2
    training.train(short, device=torch.device("cpu"), output_dir=parent, overfit=True)
    parent_bytes = (parent / "last.pt").read_bytes()
    longer = copy.deepcopy(tiny_config)
    longer["rgb_restoration"]["overfit"]["save_steps"] = [4]
    training.train(longer, device=torch.device("cpu"), output_dir=extended, overfit=True,
                   extend_overfit_from=parent / "last.pt")
    training.train(longer, device=torch.device("cpu"), output_dir=complete, overfit=True)
    actual = torch.load(extended / "last.pt", weights_only=True)
    expected = torch.load(complete / "last.pt", weights_only=True)
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    assert torch.equal(actual["rng"]["noise_generator"], expected["rng"]["noise_generator"])
    assert actual["output_metrics"] == expected["output_metrics"]
    assert actual["continuation"]["parent_updates"] == 2
    assert (extended / "update_000004.pt").exists()
    assert (extended / "monitor/epoch_000_update_000002/errors.json").exists()
    assert (parent / "last.pt").read_bytes() == parent_bytes
    from tools.probe_rgb_diffusion import probe
    probe(extended / "last.pt", device=torch.device("cpu"), seeds=[314159, 271828],
          output_dir=tmp_path / "probe")
    diagnostic = json.loads((tmp_path / "probe/probe.json").read_text())
    assert diagnostic["updates"] == 4 and len(diagnostic["seeds"]) == 2
    assert diagnostic["seeds"][0]["full"] == {key: actual["output_metrics"][key]
                                             for key in ("samples", "mean", "per_source")}
    for key in ("input", "target", "valid"):
        assert torch.equal(torch.load(extended / "fixed_samples.pt", weights_only=True)[key],
                           torch.load(parent / "fixed_samples.pt", weights_only=True)[key])
    with pytest.raises(ValueError, match="fresh output"):
        training.train(longer, device=torch.device("cpu"), output_dir=parent, overfit=True,
                       extend_overfit_from=parent / "last.pt")
    with pytest.raises(ValueError, match="requires --overfit"):
        training.train(longer, device=torch.device("cpu"), output_dir=tmp_path / "formal",
                       extend_overfit_from=parent / "last.pt")
    for modified in (short, tiny_config):
        changed = copy.deepcopy(modified)
        if modified is tiny_config:
            changed["rgb_restoration"]["model"]["kappa"] = 0.03
        with pytest.raises(ValueError, match="config differs"):
            training.train(changed, device=torch.device("cpu"), output_dir=tmp_path / "invalid",
                           overfit=True, extend_overfit_from=parent / "last.pt")
