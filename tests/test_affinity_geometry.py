# -*- coding: utf-8 -*-
import random

import numpy as np
import pytest
import torch

from models.affinity_geometry import (
    AffinityGeometryDecoder,
    HighResolutionShortAffinityResidual,
)
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_loss import (
    balanced_affinity_loss,
    build_affinity_targets_torch,
    negative_affinity_tail_loss,
)
from utils.affinity_fusion import (
    affinity_boundary_base_and_residual,
    affinity_boundary_probability,
)


def test_affinity_decoder_outputs_requested_channels_and_grid():
    decoder = AffinityGeometryDecoder(
        in_channels=[8, 16, 32, 64],
        affinity_channels=len(DEFAULT_AFFINITY_OFFSETS),
        fpn_channels=16,
        up_channels=16,
        output_grid=64,
    )
    features = [
        torch.randn(2, 8, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    output = decoder(features)
    assert output["affinity_logits"].shape == (2, 8, 64, 64)
    assert output["affinity_feature"].shape == (2, 16, 64, 64)


def test_highres_short_residual_starts_zero_and_preserves_long_channels():
    refiner = HighResolutionShortAffinityResidual(
        feature_channels=8,
        feature_hidden=8,
        image_hidden=8,
        fusion_hidden=8,
        max_logit_delta=1.0,
    )
    feature = torch.randn(1, 8, 8, 8)
    coarse = torch.randn(1, 8, 8, 8)
    image = torch.rand(1, 3, 16, 16)
    output = refiner(feature, coarse, image)
    expected = torch.nn.functional.interpolate(
        coarse, size=(16, 16), mode="bilinear", align_corners=False
    )
    assert output["affinity_logits"].shape == (1, 8, 16, 16)
    assert torch.count_nonzero(output["short_affinity_delta"]) == 0
    assert torch.equal(output["affinity_logits"], expected)


def test_highres_short_residual_never_changes_long_channels_after_update():
    refiner = HighResolutionShortAffinityResidual(
        feature_channels=8,
        feature_hidden=8,
        image_hidden=8,
        fusion_hidden=8,
        max_logit_delta=0.5,
    )
    with torch.no_grad():
        refiner.out.bias.fill_(0.25)
    feature = torch.randn(1, 8, 8, 8)
    coarse = torch.randn(1, 8, 8, 8)
    image = torch.rand(1, 3, 16, 16)
    output = refiner(feature, coarse, image)["affinity_logits"]
    expected = torch.nn.functional.interpolate(
        coarse, size=(16, 16), mode="bilinear", align_corners=False
    )
    assert not torch.equal(output[:, :4], expected[:, :4])
    assert torch.equal(output[:, 4:], expected[:, 4:])


def test_zero_highres_residual_preserves_coarse_fusion_contract():
    coarse = torch.randn(1, 8, 8, 8)
    refined = torch.nn.functional.interpolate(
        coarse, size=(16, 16), mode="bilinear", align_corners=False
    )
    base, residual = affinity_boundary_base_and_residual(
        {
            "affinity_logits": refined,
            "coarse_affinity_logits": coarse,
        },
        mode="gated",
    )
    expected = affinity_boundary_probability(coarse, mode="gated")
    assert torch.equal(base, expected)
    assert torch.count_nonzero(residual) == 0


def test_torch_targets_match_same_instance_pairs_and_loss_backpropagates():
    labels = torch.zeros((1, 8, 8), dtype=torch.int64)
    labels[:, 1:5, 1:5] = 1
    labels[:, 5:7, 5:7] = 2
    valid = torch.ones((1, 1, 8, 8), dtype=torch.bool)
    target, edge_valid = build_affinity_targets_torch(labels, valid)
    logits = torch.zeros_like(target, requires_grad=True)
    loss, metrics = balanced_affinity_loss(logits, target, edge_valid)
    assert torch.isfinite(loss)
    assert metrics["positive_edges"] > 0
    assert metrics["negative_edges"] > 0
    loss.backward()
    assert logits.grad is not None


def test_hard_negative_weight_penalizes_false_merge_edges():
    logits = torch.tensor([[[[0.0, -3.0, 3.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    baseline, _ = balanced_affinity_loss(logits, target, valid)
    weighted, _ = balanced_affinity_loss(
        logits,
        target,
        valid,
        negative_weight=1.5,
        hard_negative_weight=2.0,
        hard_negative_gamma=2.0,
    )
    assert weighted > baseline


def test_manual_uncovered_band_can_be_negative_without_supervising_zero_zero():
    labels = torch.tensor([[[1, 0, 0, 2]], [[1, 0, 0, 2]]], dtype=torch.int64)
    valid = torch.ones((2, 1, 1, 4), dtype=torch.bool)
    flags = torch.tensor([True, False])
    target, edge_valid, uncovered = build_affinity_targets_torch(
        labels,
        valid,
        offsets=[(0, 1)],
        uncovered_as_boundary=flags,
        return_uncovered_mask=True,
    )

    assert edge_valid[0, 0, 0].tolist() == [True, False, True, False]
    assert target[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert uncovered[0, 0, 0].tolist() == [True, False, True, False]
    assert not edge_valid[1].any()
    assert not uncovered[1].any()


def test_uncovered_boundary_flag_requires_one_value_per_sample():
    labels = torch.ones((2, 2, 2), dtype=torch.int64)
    valid = torch.ones((2, 1, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="one flag per batch sample"):
        build_affinity_targets_torch(
            labels,
            valid,
            uncovered_as_boundary=torch.tensor([True]),
        )


def test_edge_weight_can_soften_new_gap_negatives():
    logits = torch.tensor([[[[-3.0, 3.0]]]], requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    full, _ = balanced_affinity_loss(logits, target, valid)
    softened, _ = balanced_affinity_loss(
        logits,
        target,
        valid,
        edge_weight=torch.tensor([[[[1.0, 0.2]]]]),
    )
    assert softened < full


def test_absolute_edge_weight_discount_survives_pseudo_only_batch():
    logits = torch.full((1, 1, 1, 4), 2.0, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    full, _ = balanced_affinity_loss(logits, target, valid)
    discounted, _ = balanced_affinity_loss(
        logits,
        target,
        valid,
        edge_weight=torch.full_like(logits, 0.25),
        normalize_edge_weights=False,
    )
    assert torch.allclose(discounted, 0.25 * full)


def test_negative_tail_loss_focuses_sparse_false_merge_edges():
    logits = torch.tensor([[[[-3.0, -1.0, 0.0, 3.0]]]], requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss, metrics = negative_affinity_tail_loss(
        logits, target, valid, margin=0.45, top_fraction=0.25
    )
    assert metrics["tail_edges"] == 4
    assert metrics["tail_selected"] == 1
    loss.backward()
    assert logits.grad[0, 0, 0, 3] > 0
    assert torch.count_nonzero(logits.grad) == 1


def test_negative_tail_loss_can_select_manual_samples_only():
    logits = torch.full((2, 1, 1, 2), 3.0, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss, metrics = negative_affinity_tail_loss(
        logits,
        target,
        valid,
        sample_mask=torch.tensor([True, False]),
        top_fraction=1.0,
    )
    assert metrics["tail_edges"] == 2
    loss.backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].abs().sum() == 0


@pytest.mark.parametrize(
    "cfg,deployment_enabled,expected",
    [
        ({}, False, "oracle_gt_penalized_miou"),
        ({}, True, "deployment_score_total"),
        ({"selection_metric": "val_affinity_loss"}, False, "val_affinity_loss"),
        ({"selection_metric": "val_affinity_loss"}, True, "val_affinity_loss"),
        ({"selection_metric": "oracle_gt_penalized_miou"}, True,
         "oracle_gt_penalized_miou"),
        ({"selection_metric": "deployment_score_total"}, True,
         "deployment_score_total"),
    ],
)
def test_affinity_selection_preserves_legacy_defaults_and_explicit_metric(
    cfg, deployment_enabled, expected
):
    from train_affinity_geometry_g1 import resolve_selection_metric

    assert resolve_selection_metric(cfg, deployment_enabled) == expected


@pytest.mark.parametrize(
    "metric,deployment_enabled",
    [("deployment_score_total", False), ("misspelled_loss", True)],
)
def test_affinity_selection_rejects_unavailable_or_unknown_metric(
    metric, deployment_enabled
):
    from train_affinity_geometry_g1 import resolve_selection_metric

    with pytest.raises(ValueError):
        resolve_selection_metric({"selection_metric": metric}, deployment_enabled)


@pytest.mark.parametrize(
    "metric,better,worse",
    [("val_affinity_loss", 0.4, 0.6),
     ("oracle_gt_penalized_miou", 0.6, 0.4),
     ("deployment_score_total", 0.6, 0.4)],
)
def test_affinity_selection_direction_and_tie_keep_epoch_zero(metric, better, worse):
    from train_affinity_geometry_g1 import selection_improved

    assert selection_improved(better, 0.5, metric)
    assert not selection_improved(worse, 0.5, metric)
    assert not selection_improved(0.5, 0.5, metric)


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bad_is_best", [False, True])
def test_affinity_selection_rejects_nonfinite_scores(bad_score, bad_is_best):
    from train_affinity_geometry_g1 import selection_improved

    score, best = (0.5, bad_score) if bad_is_best else (bad_score, 0.5)
    with pytest.raises(ValueError):
        selection_improved(score, best, "val_affinity_loss")


class _FixedAffinitySystem(torch.nn.Module):
    """固定输出的小模型，用于核查验证契约而无需加载 SAM2。"""

    def __init__(self, logits):
        super().__init__()
        self.geometry_decoder = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer("fixed_logits", logits)
        self.forward_grad_modes = []

    def geometry_forward(self, image):
        self.forward_grad_modes.append(torch.is_grad_enabled())
        index = int(image[0, 0, 0, 0])
        logits = self.fixed_logits[index:index + 1]
        return {
            "affinity_logits": logits + self.geometry_decoder.weight.sum().to(
                dtype=logits.dtype
            ) * 0.0
        }


def _affinity_validation_sample(index=0):
    labels = torch.ones((8, 8), dtype=torch.int64)
    labels[:, 4:] = 2
    labels[3:5, 3:5] = 0
    valid = torch.ones((1, 8, 8), dtype=torch.bool)
    valid[:, 0] = False
    return {
        "image": torch.full((3, 8, 8), float(index)),
        "instance_map": labels,
        "valid_content": valid,
        "image_name": f"validation_{index}.png",
        # 验证时即使原始人工数据带此标记，也必须忽略未覆盖接缝。
        "uncovered_boundary_source": torch.tensor(True),
    }


def _affinity_validation_config():
    return {
        "loss": {
            "negative_weight": 1.5,
            "hard_negative_weight": 1.0,
            "hard_negative_gamma": 2.0,
            "normalize_edge_weights": True,
        }
    }


def test_affinity_validation_uses_raw_logits_per_image_mean_and_keeps_state():
    from train_affinity_geometry_g1 import evaluate_affinity_loss

    samples = [_affinity_validation_sample(index) for index in range(2)]
    samples[1]["instance_map"][:, :3] = 0
    logits = torch.linspace(-3.0, 3.0, 2 * 8 * 8 * 8).reshape(2, 8, 8, 8)
    system = _FixedAffinitySystem(logits).eval()
    config = _affinity_validation_config()
    expected_losses, expected_positive, expected_negative = [], [], []
    for index, sample in enumerate(samples):
        target, valid = build_affinity_targets_torch(
            sample["instance_map"].unsqueeze(0),
            sample["valid_content"].unsqueeze(0),
        )
        loss, _ = balanced_affinity_loss(
            logits[index:index + 1], target, valid, **config["loss"]
        )
        expected_losses.append(float(loss))
        expected_positive.append((valid & (target > 0.5)).sum((0, 2, 3)).tolist())
        expected_negative.append((valid & (target <= 0.5)).sum((0, 2, 3)).tolist())

    state_before = {key: value.clone() for key, value in system.state_dict().items()}
    torch_rng = torch.random.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    result = evaluate_affinity_loss(system, samples, config)

    assert result["loss"] == pytest.approx(sum(expected_losses) / 2, abs=1e-7)
    assert [item["loss"] for item in result["per_image"]] == pytest.approx(expected_losses)
    assert [item["image"] for item in result["per_image"]] == [
        sample["image_name"] for sample in samples
    ]
    assert [item["positive_edges"] for item in result["per_image"]] == expected_positive
    assert [item["negative_edges"] for item in result["per_image"]] == expected_negative
    assert result["positive_edges"] == np.sum(expected_positive, axis=0).tolist()
    assert result["negative_edges"] == np.sum(expected_negative, axis=0).tolist()
    assert result["loss_parameters"] == config["loss"]
    assert isinstance(result["protocol"], str) and result["protocol"]
    assert system.forward_grad_modes == [False, False]
    assert all(not module.training for module in system.modules())
    assert all(torch.equal(value, state_before[key]) for key, value in system.state_dict().items())
    assert all(parameter.grad is None for parameter in system.parameters())
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    current_numpy_rng = np.random.get_state()
    assert current_numpy_rng[0] == numpy_rng[0]
    assert np.array_equal(current_numpy_rng[1], numpy_rng[1])
    assert current_numpy_rng[2:] == numpy_rng[2:]
    assert random.getstate() == python_rng


def test_affinity_validation_ignores_unknown_and_padding_edge_predictions():
    from train_affinity_geometry_g1 import evaluate_affinity_loss

    sample = _affinity_validation_sample()
    _, valid = build_affinity_targets_torch(
        sample["instance_map"].unsqueeze(0),
        sample["valid_content"].unsqueeze(0),
    )
    logits = torch.full((1, 8, 8, 8), 0.25)
    changed_logits = logits.clone()
    changed_logits[~valid] = 25.0
    baseline = evaluate_affinity_loss(
        _FixedAffinitySystem(logits).eval(), [sample], _affinity_validation_config()
    )
    changed = evaluate_affinity_loss(
        _FixedAffinitySystem(changed_logits).eval(), [sample], _affinity_validation_config()
    )
    assert changed["loss"] == baseline["loss"]
    assert changed["negative_edges"] == baseline["negative_edges"]


@pytest.mark.parametrize("case", ["empty", "no_edges", "unsupervised_channel", "nan", "half"])
def test_affinity_validation_rejects_invalid_protocol_inputs(case):
    from train_affinity_geometry_g1 import evaluate_affinity_loss

    sample = _affinity_validation_sample()
    samples = [sample]
    logits = torch.zeros((1, 8, 8, 8))
    if case == "empty":
        samples = []
    elif case == "no_edges":
        sample["instance_map"].zero_()
    elif case == "unsupervised_channel":
        sample["valid_content"].zero_()
        sample["valid_content"][:, 2] = True
    elif case == "nan":
        logits.fill_(float("nan"))
    elif case == "half":
        logits = logits.half()
    with pytest.raises(ValueError):
        evaluate_affinity_loss(
            _FixedAffinitySystem(logits).eval(), samples, _affinity_validation_config()
        )


@pytest.mark.parametrize(
    "metric,direction",
    [("val_affinity_loss", "min"), ("oracle_gt_penalized_miou", "max")],
)
def test_latest_affinity_checkpoint_keeps_actual_and_best_epochs_and_loads_strictly(
    tmp_path, metric, direction
):
    from train_affinity_geometry_g1 import (
        AFFINITY_LOSS_PROTOCOL,
        load_geometry_checkpoint_state,
        save_checkpoint,
    )

    system = _FixedAffinitySystem(torch.zeros((1, 8, 8, 8))).eval()
    system.geometry_feature_adapter = None
    system.geometry_highres_refiner = None
    with torch.no_grad():
        system.geometry_decoder.weight.fill_(0.75)
    path = tmp_path / "latest_affinity.pth"
    save_checkpoint(
        path, system, config={"test": True}, epoch=30, best_score=0.4,
        reference_path=tmp_path / "reference.pth", reference_sha="reference-sha",
        init_path=tmp_path / "initialization.pth", digest="semantic-digest",
        split={"train": ["train_001"], "val": ["train_172"]},
        selection_metric=metric, best_oracle_score=0.8, best_epoch=7,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 30
    assert checkpoint["best_selection_epoch"] == 7
    assert checkpoint["best_selection_direction"] == direction
    assert checkpoint["best_selection_metric"] == metric
    assert checkpoint["best_selection_score"] == 0.4
    assert checkpoint["validation_protocol_version"] == (
        AFFINITY_LOSS_PROTOCOL if metric == "val_affinity_loss" else None
    )
    assert torch.equal(
        checkpoint["geometry_state_dict"]["weight"], system.geometry_decoder.weight
    )
    with torch.no_grad():
        system.geometry_decoder.weight.zero_()
    load_geometry_checkpoint_state(system, checkpoint)
    assert torch.equal(system.geometry_decoder.weight, torch.tensor([[0.75]]))

    incomplete = dict(checkpoint, geometry_state_dict={})
    with pytest.raises(RuntimeError, match="Missing key"):
        load_geometry_checkpoint_state(system, incomplete)
