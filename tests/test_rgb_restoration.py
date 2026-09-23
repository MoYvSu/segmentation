# -*- coding: utf-8 -*-
"""修复器的恒等起点、源图隔离、padding 忽略及真实训练/续训契约。"""

import copy
import json

import cv2
import numpy as np
import pytest
import torch

import train_rgb_restoration as training
from data.rgb_restoration_dataset import (
    PROFILES, LocalDamageDiagnosticDataset, NoiseDiagnosticDataset, local_reconstruction_mask,
)
from models.rgb_restoration import CHECKPOINT_FORMAT, RGBRestorer, load_rgb_restorer
from utils.config import load_config


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    yy, xx = np.mgrid[:60, :100]
    for index in range(4):
        image = np.stack(((xx * 3 + index * 17) % 256, (yy * 4) % 256,
                          ((xx // 4 + yy // 7) % 2) * 180 + 30), axis=-1).astype(np.uint8)
        cv2.imwrite(str(source / f"train_{index:03d}.png"), image)
    manifest = tmp_path / "holdout.txt"
    manifest.write_text("train_003.png\n", encoding="utf-8")
    config = load_config("config/train/rgb_restoration_v1.yaml")
    cfg = config["rgb_restoration"]
    cfg.update(data_dir=str(source), holdout_manifest=str(manifest), image_size=32, crop_size=16)
    cfg["model"]["width"] = 4
    cfg["train"].update(batch_size=2, num_workers=0, epochs=2, amp=False)
    return config


def small_masking_recipe():
    return {"epochs": 1, "start_strength": 0.5, "coverage": [0.08, 0.16],
            "patch_size": [4, 8], "lowpass_sigma": [2.0, 3.0], "blend": [0.65, 0.95]}


def test_local_mask_removes_detail_without_moving_targets_or_touching_padding():
    yy, xx = np.mgrid[:64, :80]
    image = np.repeat(((xx + yy) % 2)[..., None], 3, axis=2).astype(np.float32)
    original = image.copy()
    valid = np.ones(image.shape[:2], dtype=np.float32)
    valid[48:] = 0
    recipe = small_masking_recipe()
    recipe["patch_size"] = [8, 16]
    degraded, mask = local_reconstruction_mask(image, valid, np.random.default_rng(42), recipe, 1.0)
    assert np.array_equal(image, original)
    assert np.array_equal(degraded[mask == 0], image[mask == 0])
    assert mask[valid == 0].sum() == 0 and np.count_nonzero(mask) > 0
    assert 0 <= degraded.min() <= degraded.max() <= 1
    selected = mask > 0.2
    assert np.abs(degraded[selected] - 0.5).mean() < np.abs(image[selected] - 0.5).mean()
    # 同样的掩码也覆盖均匀内部；不以清晰图强梯度筛选位置，不凭空画出纹理。
    flat = np.full_like(image, 0.75)
    flat_result, flat_mask = local_reconstruction_mask(flat, valid, np.random.default_rng(42), recipe, 1.0)
    assert np.array_equal(mask, flat_mask)
    np.testing.assert_allclose(flat_result, flat, atol=1e-7)


def test_masked_pretraining_preserves_pairing_clean_inputs_and_v3_validation(tiny_config):
    config = copy.deepcopy(tiny_config)
    config["rgb_restoration"]["degradation"]["profile_probabilities"] = [0, 1, 0, 0]
    baseline, baseline_holdout = training.build_datasets(config)
    config["rgb_restoration"]["masked_pretraining"] = small_masking_recipe()
    candidate, candidate_holdout = training.build_datasets(config)
    for index in range(len(candidate)):
        normal, masked = baseline[index], candidate[index]
        assert masked["source"] == normal["source"] and masked["profile"] == normal["profile"]
        assert torch.equal(masked["target"], normal["target"])
        assert torch.equal(masked["valid"], normal["valid"])
        assert masked["mask_mean"] > 0 and not torch.equal(masked["input"], normal["input"])
        assert torch.equal(masked["input"], candidate[index]["input"])
    candidate_holdout.set_epoch(99)
    for index in range(len(candidate_holdout)):
        assert torch.equal(baseline_holdout[index]["input"], candidate_holdout[index]["input"])
        assert torch.equal(baseline_holdout[index]["target"], candidate_holdout[index]["target"])
    # 阶段切换后逐像素恢复 v3 同轮输入，随机裁块和翻转也必须相同。
    candidate.set_epoch(1)
    baseline.set_epoch(1)
    for index in range(len(candidate)):
        assert torch.equal(baseline[index]["input"], candidate[index]["input"])
        assert candidate[index]["mask_mean"] == 0
    config["rgb_restoration"]["degradation"]["profile_probabilities"] = [1, 0, 0, 0]
    clean, _ = training.build_datasets(config)
    assert clean[0]["mask_mean"] == 0
    assert torch.equal(clean[0]["input"], clean[0]["target"])


def test_mask_curriculum_leaves_finetuning_and_uses_the_original_budget(tiny_config):
    config = copy.deepcopy(tiny_config)
    recipe = small_masking_recipe()
    recipe["epochs"] = 6
    config["rgb_restoration"]["masked_pretraining"] = recipe
    with pytest.raises(ValueError, match="leave at least one"):
        training.build_datasets(config)
    config["rgb_restoration"]["train"]["epochs"] = 20
    training_set, _ = training.build_datasets(config)
    strengths = []
    for epoch in range(7):
        training_set.set_epoch(epoch)
        strengths.append(training_set.masking_strength())
    assert strengths[:6] == pytest.approx([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert strengths[6] == 0
    original = load_config("config/train/rgb_restoration_deblur_v3.yaml")["rgb_restoration"]
    candidate = load_config("config/train/rgb_restoration_deblur_v4_masked.yaml")["rgb_restoration"]
    assert {key for key in original.keys() | candidate.keys()
            if original.get(key) != candidate.get(key)} == {"output_dir", "masked_pretraining"}


@pytest.mark.parametrize("shape", [(1, 3, 1, 1), (2, 3, 31, 47), (1, 3, 32, 48)])
def test_initial_identity_on_arbitrary_shapes(shape):
    model = RGBRestorer(width=4)
    inputs = torch.rand(shape)
    assert torch.equal(model(inputs), inputs)
    assert model(inputs, strength=0) is inputs


def test_correction_and_backbone_receive_gradients():
    model = RGBRestorer(width=4)
    inputs = torch.rand(2, 3, 17, 21)
    targets = inputs * 0.7
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = (model(inputs) - targets).abs().mean()
        loss.backward()
        optimizer.step()
    assert model.correction.weight.grad.abs().sum() > 0
    assert model.stem.weight.grad.abs().sum() > 0
    output = model(inputs)
    assert not torch.equal(output, inputs)
    assert output.min() >= 0 and output.max() <= 1


def test_holdout_source_isolation_and_repeatable_validation(tiny_config):
    train, holdout = training.build_datasets(tiny_config)
    assert len(train.samples) == 3 and len(holdout.samples) == 1
    assert set(train.samples).isdisjoint(holdout.samples)
    assert [holdout[index]["profile"] for index in range(4)] == list(PROFILES)
    reference = holdout[0]
    assert torch.equal(reference["input"], reference["target"])
    assert reference["valid"].sum() == round(60 / 100 * 32) * 32
    for index in range(1, 4):
        sample = holdout[index]
        assert not torch.equal(sample["input"], sample["target"])
        assert torch.equal(sample["target"], reference["target"])
        holdout.set_epoch(57)
        assert torch.equal(sample["input"], holdout[index]["input"])
    first = train[0]
    train.set_epoch(1)
    second = train[0]
    assert not torch.equal(first["input"], second["input"])


def test_loss_excludes_padding_and_cross_border_gradients():
    target = torch.zeros(1, 3, 4, 5)
    prediction = target.clone()
    prediction[..., 2:, :] = 100
    prediction.requires_grad_()
    valid = torch.zeros(1, 1, 4, 5)
    valid[..., :2, :] = 1
    loss = training.reconstruction_loss(prediction, target, valid,
                                        {"rgb_weight": 1.0, "gradient_weight": 0.2})
    assert loss.item() == 0
    loss.backward()
    assert prediction.grad.count_nonzero() == 0
    empty = training.reconstruction_errors(prediction, target, torch.zeros_like(valid))
    assert all(torch.isfinite(value).all() and value.item() == 0 for value in empty.values())


def test_checkpoint_rejects_other_formats_and_missing_weights(tmp_path):
    model = RGBRestorer(width=4)
    payload = {"format": CHECKPOINT_FORMAT, "model_config": model.model_config, "model": model.state_dict()}
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    loaded = load_rgb_restorer(str(path))
    assert not loaded.training
    del payload["model"]["stem.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_rgb_restorer(str(path))
    payload["format"] = "segmentation_checkpoint"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format"):
        load_rgb_restorer(str(path))


@pytest.mark.parametrize("masked", [False, True])
def test_training_smoke_and_reload_without_sam2(tiny_config, tmp_path, monkeypatch, masked):
    from models.sam2_encoder import SAM2Encoder

    def forbidden(*args, **kwargs):
        raise AssertionError("standalone restoration must not construct SAM2")

    monkeypatch.setattr(SAM2Encoder, "__init__", forbidden)
    if masked:
        tiny_config["rgb_restoration"]["masked_pretraining"] = small_masking_recipe()
    run = tmp_path / "smoke"
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=run, smoke=True)
    assert summary["status"] == "completed" and summary["total_updates"] == 2
    assert set(summary["latest_metrics"]["validation"]["profiles"]) == set(PROFILES)
    loaded = load_rgb_restorer(str(run / "best.pt"))
    sample = training.build_datasets(tiny_config)[1][3]["input"][None]
    assert torch.isfinite(loaded(sample)).all()
    assert not torch.equal(loaded(sample), sample)
    metadata = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    assert metadata["smoke"] and metadata["validation_cases"] == 4
    with pytest.raises(FileExistsError):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=run, smoke=True)


@pytest.mark.parametrize("masked", [False, True])
def test_epoch_resume_matches_uninterrupted_training(tiny_config, tmp_path, monkeypatch, masked):
    if masked:
        tiny_config["rgb_restoration"]["masked_pretraining"] = small_masking_recipe()
        tiny_config["rgb_restoration"]["degradation"]["profile_probabilities"] = [0, 1, 0, 0]
    complete = tmp_path / "complete"
    interrupted = tmp_path / "interrupted"
    training.train(tiny_config, device=torch.device("cpu"), output_dir=complete)
    original_evaluate = training.evaluate
    calls = 0

    def fail_second_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "evaluate", fail_second_validation)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted)
    monkeypatch.setattr(training, "evaluate", original_evaluate)
    altered = copy.deepcopy(tiny_config)
    altered["rgb_restoration"]["loss"]["gradient_weight"] = 0.5
    with pytest.raises(ValueError, match="config differs"):
        training.train(altered, device=torch.device("cpu"), output_dir=interrupted,
                       resume=interrupted / "last.pt")
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted,
                             resume=interrupted / "last.pt")
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert summary["total_updates"] == 4 and actual["epoch"] == 2
    assert all(torch.equal(actual["model"][key], value) for key, value in expected["model"].items())
    assert actual["scheduler"] == expected["scheduler"]
    if masked:
        rows = [json.loads(line) for line in (complete / "metrics.jsonl").read_text().splitlines()]
        assert rows[0]["phase"] == "masked_pretraining" and rows[0]["masked_samples"] == 3
        assert rows[0]["mean_mask_weight"] > 0
        assert rows[1]["phase"] == "ordinary_reconstruction" and rows[1]["masked_samples"] == 0


def test_deblur_selection_rejects_lighting_gain_and_identity_drift(tiny_config):
    cfg = tiny_config["rgb_restoration"]
    cfg["selection"] = {"profile": "blur", "max_identity_l1": 1 / 255,
                        "require_input_improvement": True}
    profiles = {name: {"input_rgb_l1": 0.1, "input_gradient_l1": 0.02,
                       "restored_rgb_l1": 0.01, "restored_gradient_l1": 0.01}
                for name in PROFILES}
    profiles["identity"]["restored_rgb_l1"] = 0.001
    profiles["blur"]["restored_rgb_l1"] = 0.2
    # 即使照明组改善很多，模糊比输入更差也不能入选。
    result = training.checkpoint_selection({"profiles": profiles}, cfg)
    assert not result["eligible"] and result["loss"] > result["baseline_loss"]
    profiles["blur"]["restored_rgb_l1"] = 0.02
    assert training.checkpoint_selection({"profiles": profiles}, cfg)["eligible"]
    profiles["identity"]["restored_rgb_l1"] = 0.02
    assert not training.checkpoint_selection({"profiles": profiles}, cfg)["eligible"]


def test_no_eligible_checkpoint_finishes_with_last_only(tiny_config, tmp_path):
    tiny_config["rgb_restoration"]["selection"] = {
        "profile": "blur", "max_identity_l1": 0.0, "require_input_improvement": True,
    }
    run = tmp_path / "no_candidate"
    summary = training.train(tiny_config, device=torch.device("cpu"), output_dir=run, smoke=True)
    assert summary["status"] == "completed"
    assert summary["best_loss"] is None and not summary["best_checkpoint_available"]
    assert (run / "last.pt").exists() and not (run / "best.pt").exists()


def small_diagnostic_recipe():
    return {"seed_epoch": 5, "strength": 1.0, "affected_threshold": 0.1,
            "mask": {key: value for key, value in small_masking_recipe().items()
                     if key not in {"epochs", "start_strength"}}}


def test_local_diagnostic_is_fixed_and_matches_previous_probe(tiny_config):
    train_set, holdout = training.build_datasets(tiny_config)
    cfg = small_diagnostic_recipe()
    with pytest.raises(ValueError, match="held-out"):
        LocalDamageDiagnosticDataset(train_set, cfg)
    diagnostic = LocalDamageDiagnosticDataset(holdout, cfg)
    actual = diagnostic[0]
    original = holdout[1]
    vh, vw = int(original["valid"][0, :, 0].sum()), int(original["valid"][0, 0, :].sum())
    size = holdout.crop_size
    y, x = max(0, (vh - size) // 2), max(0, (vw - size) // 2)
    region = original["valid"][0, y:y + size, x:x + size].numpy()
    normal = original["input"][:, y:y + size, x:x + size].permute(1, 2, 0).numpy()
    rng = np.random.default_rng(np.random.SeedSequence([42, 5, 1, 2]))
    expected, alpha = local_reconstruction_mask(normal, region, rng, cfg["mask"], 1.0)
    assert torch.equal(actual["input"], torch.from_numpy(expected.transpose(2, 0, 1).copy()))
    assert torch.equal(actual["target"], original["target"][:, y:y + size, x:x + size])
    assert torch.equal(actual["affected"], torch.from_numpy(((alpha > 0.1) * region)[None]))
    assert not torch.equal(actual["input"], torch.from_numpy(normal.transpose(2, 0, 1).copy()))
    holdout.set_epoch(79)
    assert torch.equal(actual["input"], diagnostic[0]["input"])
    assert torch.equal(original["input"], holdout[1]["input"])


@pytest.mark.parametrize("masked", [False, True])
def test_diagnostics_and_snapshots_do_not_change_updates_or_selection(tiny_config, tmp_path, masked):
    config = copy.deepcopy(tiny_config)
    if masked:
        config["rgb_restoration"]["masked_pretraining"] = small_masking_recipe()
    baseline = tmp_path / "baseline"
    training.train(config, device=torch.device("cpu"), output_dir=baseline)
    config["rgb_restoration"]["diagnostics"] = {
        "local_damage": small_diagnostic_recipe(), "noise": {"sigmas": [0.001, 0.003, 0.006]},
    }
    config["rgb_restoration"]["train"]["save_epochs"] = [1, 2]
    candidate = tmp_path / "with_diagnostics"
    training.train(config, device=torch.device("cpu"), output_dir=candidate)
    original = torch.load(baseline / "last.pt", weights_only=True)
    actual = torch.load(candidate / "last.pt", weights_only=True)
    assert original["updates"] == actual["updates"]
    assert original["best_epoch"] == actual["best_epoch"]
    assert original["metrics"]["selection"] == actual["metrics"]["selection"]
    assert original["metrics"]["train_loss"] == actual["metrics"]["train_loss"]
    assert all(torch.equal(value, actual["model"][key]) for key, value in original["model"].items())
    assert (candidate / "epoch_001.pt").exists() and (candidate / "epoch_002.pt").exists()
    restored = load_rgb_restorer(str(candidate / "epoch_001.pt"), device="cpu")
    assert restored.model_config["width"] == 4
    rows = [json.loads(line) for line in (candidate / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(row["local_damage"]["images"] == 1 for row in rows)
    assert all(row["noise_diagnostic"]["cases"] == 3 for row in rows)
    assert set(rows[0]["noise_diagnostic"]["sigmas"]) == {"0.001", "0.003", "0.006"}
    for row in rows:
        errors = row["train_errors"]
        assert row["train_loss"] == pytest.approx(errors["rgb_l1"] + 0.2 * errors["gradient_l1"])
        for region in ("full", "affected"):
            diagnostic = row["local_damage"]["regions"][region]
            assert all(np.isfinite(value) for value in diagnostic.values())
    for region in ("full", "affected"):
        for key in ("input_rgb_l1", "input_gradient_l1", "input_mse"):
            assert rows[0]["local_damage"]["regions"][region][key] == rows[1]["local_damage"]["regions"][region][key]
    for sigma in rows[0]["noise_diagnostic"]["sigmas"]:
        for key in ("input_rgb_l1", "input_gradient_l1", "input_mse"):
            assert rows[0]["noise_diagnostic"]["sigmas"][sigma][key] == rows[1]["noise_diagnostic"]["sigmas"][sigma][key]


def test_long_budget_keeps_data_model_loss_and_six_masked_epochs():
    configs = {name: load_config(f"config/train/rgb_restoration_deblur_{name}.yaml")["rgb_restoration"]
               for name in ("v3", "v4_masked", "v3_long80", "v4_masked_long80")}
    for name in ("v3", "v4_masked"):
        previous, current = copy.deepcopy(configs[name]), copy.deepcopy(configs[name + "_long80"])
        assert current["train"]["epochs"] == 80
        assert current["train"].pop("save_epochs") == [6, 20, 40, 60, 80]
        current["train"]["epochs"] = previous["train"]["epochs"]
        current.pop("diagnostics")
        for cfg in (previous, current):
            cfg.pop("output_dir")
        assert previous == current
    v3, v4 = configs["v3_long80"], configs["v4_masked_long80"]
    assert {key for key in v3.keys() | v4.keys() if v3.get(key) != v4.get(key)} == {
        "output_dir", "masked_pretraining"}
    assert v4["masked_pretraining"]["epochs"] == 6


def test_v5_configs_keep_one_training_factor_and_unchanged_selection():
    baseline = load_config("config/train/rgb_restoration_deblur_v4_masked_long80.yaml")["rgb_restoration"]
    baseline.pop("output_dir")
    for name in ("noise80", "local80"):
        cfg = load_config(f"config/train/rgb_restoration_deblur_v5_{name}.yaml")["rgb_restoration"]
        cfg.pop("output_dir")
        assert cfg["diagnostics"].pop("noise") == {"sigmas": [0.001, 0.003, 0.006]}
        if name == "noise80":
            assert cfg.pop("training_noise") == {"probability": 0.5, "sigma": [0.0, 0.006]}
        else:
            assert cfg["masked_pretraining"].pop("continuation_probability") == 0.25
        assert cfg == baseline


def test_noise_diagnostic_identity_reports_input_errors(tiny_config):
    _, holdout = training.build_datasets(tiny_config)
    dataset = NoiseDiagnosticDataset(holdout, {"sigmas": [0.001, 0.003, 0.006]})
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    result = training.evaluate_noise(torch.nn.Identity(), loader, torch.device("cpu"), False, torch.bfloat16)
    assert result["cases"] == 3 and len(result["sigmas"]) == 3
    for row in result["sigmas"].values():
        assert row["images"] == 1
        for key in ("rgb_l1", "gradient_l1", "mse", "psnr_db"):
            assert row["input_" + key] == row["restored_" + key]


@pytest.mark.parametrize("variant", ["noise", "continued_mask"])
def test_v5_resume_matches_full_training(tiny_config, tmp_path, monkeypatch, variant):
    cfg = tiny_config["rgb_restoration"]
    cfg["masked_pretraining"] = small_masking_recipe()
    cfg["degradation"]["profile_probabilities"] = [0, 1, 0, 0]
    cfg["diagnostics"] = {"noise": {"sigmas": [0.001, 0.003, 0.006]}}
    if variant == "noise":
        cfg["training_noise"] = {"probability": 1.0, "sigma": [0.003, 0.003]}
    else:
        cfg["masked_pretraining"]["continuation_probability"] = 1.0
    complete, interrupted = tmp_path / "complete", tmp_path / "interrupted"
    training.train(tiny_config, device=torch.device("cpu"), output_dir=complete)
    original_evaluate = training.evaluate
    calls = 0

    def fail_second_validation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated v5 interruption")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "evaluate", fail_second_validation)
    with pytest.raises(RuntimeError, match="simulated v5"):
        training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted)
    monkeypatch.setattr(training, "evaluate", original_evaluate)
    training.train(tiny_config, device=torch.device("cpu"), output_dir=interrupted,
                   resume=interrupted / "last.pt")
    expected = torch.load(complete / "last.pt", weights_only=True)
    actual = torch.load(interrupted / "last.pt", weights_only=True)
    assert expected["updates"] == actual["updates"] == 4
    assert expected["metrics"]["selection"] == actual["metrics"]["selection"]
    assert expected["metrics"]["noise_diagnostic"] == actual["metrics"]["noise_diagnostic"]
    assert expected["scheduler"] == actual["scheduler"]
    assert all(torch.equal(value, actual["model"][key]) for key, value in expected["model"].items())
    if variant == "noise":
        assert actual["metrics"]["noise_samples"] == 3
        assert actual["metrics"]["mean_noise_sigma"] == pytest.approx(0.003)
        assert actual["metrics"]["masked_samples"] == 0
    else:
        assert actual["metrics"]["phase"] == "mixed_local_reconstruction"
        assert actual["metrics"]["masked_samples"] == 3
        assert actual["metrics"]["noise_samples"] == 0
