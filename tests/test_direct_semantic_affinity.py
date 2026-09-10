# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from data.direct_dual_head_dataset import (
    DirectDualHeadDataset,
    load_direct_evaluation_target,
    load_manual_target_archive,
    mask_prediction_to_evaluation_domain,
    semantic_from_instance_classes,
)
from models.direct_semantic_affinity import (
    _checkpoint_architecture_config,
    _load_complete_lora_state,
    assert_direct_checkpoint_input_contract,
    configure_direct_affinity_recovery,
    configure_direct_semantic_recovery,
    configure_direct_training_phase,
    direct_parameter_groups,
    load_direct_ssl_lora_state,
)
from models.fused_deployment import FusedPhaseAffinityModel
from models.sam2_encoder import SAM2Encoder, normalize_sam2_trunk_input
from train_direct_semantic_affinity import (
    compute_affinity_loss,
    resolve_direct_monitor_paths,
    save_direct_monitor,
    update_phase_loss_best,
    validate_manual_target_dataset,
    validate_sam2_geometry_approval,
)
from tools.sweep_direct_deployment import evaluate_cached_semantic
from tools.pretrain_lora_ssl import MaskedUnlabeledDataset, load_excluded_stems
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config
from utils.loss import BoundaryLoss


def test_affinity_tail_short_run_keeps_five_monitor_points():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config/train/direct_ssl_semantic_affinity_hiera_l_affinity_tail_control.yaml"
    )
    config = load_config(str(config_path))
    direct_cfg = config["direct_semantic_affinity"]
    epochs = int(direct_cfg["affinity_recovery"]["epochs"])
    interval = int(direct_cfg["monitor"]["interval"])

    assert epochs == 10
    assert interval == 2
    assert direct_cfg["checkpoint_interval"] == 10
    assert [epoch for epoch in range(1, epochs + 1) if epoch % interval == 0] == [
        2,
        4,
        6,
        8,
        10,
    ]


def test_manual_target_short_run_keeps_five_monitor_points():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config/train/direct_ssl_semantic_affinity_imagenet_norm_target_v2.yaml"
    )
    config = load_config(str(config_path))
    direct_cfg = config["direct_semantic_affinity"]
    epochs = int(direct_cfg["head_warmup_epochs"]) + int(
        direct_cfg["joint_epochs"]
    )
    interval = int(direct_cfg["monitor"]["interval"])

    assert epochs == 25
    assert interval == 5
    assert [epoch for epoch in range(1, epochs + 1) if epoch % interval == 0] == [
        5,
        10,
        15,
        20,
        25,
    ]


def test_clean_gtv2_long_chain_config_contract():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config/train/direct_gtv2_clean_60x60.yaml"
    )
    config = load_config(str(config_path))
    direct_cfg = config["direct_semantic_affinity"]
    warmup_epochs = int(direct_cfg["head_warmup_epochs"])
    joint_epochs = int(direct_cfg["joint_epochs"])
    interval = int(direct_cfg["monitor"]["interval"])

    assert config["sam2"]["input_normalization"] == "imagenet_v1"
    assert config["lora"]["init_from"] == (
        "outputs/20260908_133740_ssl60/best_lora.pth"
    )
    assert warmup_epochs == 60
    assert joint_epochs == 60
    assert direct_cfg["pseudo_start_epoch"] == 60
    assert direct_cfg["pseudo_affinity_weight"] == 0.0
    assert direct_cfg["sam2_geometry"]["enabled"] is False
    assert direct_cfg["semantic_recovery"]["enabled"] is False
    assert direct_cfg["affinity_recovery"]["enabled"] is False

    manual_cfg = direct_cfg["manual_target"]
    assert manual_cfg["enabled"] is True
    assert manual_cfg["dataset_dir"] == (
        "outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps"
    )
    assert direct_cfg["affinity_loss"]["manual_uncovered_as_boundary"] is False

    assert direct_cfg["checkpoint_interval"] == 10
    assert direct_cfg["monitor"]["enabled"] is True
    assert interval == 5
    warmup_monitor_epochs = [
        epoch for epoch in range(1, warmup_epochs + 1) if epoch % interval == 0
    ]
    joint_monitor_epochs = [
        epoch
        for epoch in range(warmup_epochs + 1, warmup_epochs + joint_epochs + 1)
        if epoch % interval == 0
    ]
    assert len(warmup_monitor_epochs) == 12
    assert len(joint_monitor_epochs) == 12
    assert warmup_monitor_epochs[-1] == 60
    assert joint_monitor_epochs[-1] == 120

    assert direct_cfg["loss_selection"]["enabled"] is True
    assert direct_cfg["loss_selection"]["restore_warmup_best"] is True
    output_dir = "outputs/20260908_133740_clean60"
    assert config["paths"]["output_dir"] == output_dir
    assert direct_cfg["output_dir"] == output_dir
    assert direct_cfg["monitor"]["output_dir"] == f"{output_dir}/monitor"


def test_ssl_holdout_manifest_is_excluded_from_training(tmp_path):
    image_dir = tmp_path / "unlabeled"
    image_dir.mkdir()
    for name in ("train_001.jpg", "train_024.jpg", "train_032.png"):
        (image_dir / name).touch()
    manifest = tmp_path / "holdout.txt"
    manifest.write_text(
        "# fixed monitor images\ntrain_024.jpg\ntrain_032.png\n",
        encoding="utf-8",
    )

    excluded = load_excluded_stems(str(manifest))
    dataset = MaskedUnlabeledDataset(
        str(image_dir),
        image_size=16,
        exclude_stems=excluded,
    )

    assert excluded == {"train_024", "train_032"}
    assert [Path(path).name for path in dataset.samples] == ["train_001.jpg"]
    assert {Path(path).name for path in dataset.excluded_samples} == {
        "train_024.jpg",
        "train_032.png",
    }


def _write_labelme(path, height, width):
    payload = {
        "imageHeight": height,
        "imageWidth": width,
        "shapes": [
            {
                "label": "pearlite",
                "points": [[1, 1], [width // 2, 1], [width // 2, height - 2], [1, height - 2]],
            },
            {
                "label": "ferrite",
                "points": [[width // 2 + 1, 1], [width - 2, 1], [width - 2, height - 2], [width // 2 + 1, height - 2]],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manual_target(target_dir, stem, instance_map):
    target_dir.mkdir(parents=True, exist_ok=True)
    instance_map = np.asarray(instance_map, dtype=np.uint16)
    original = np.zeros(instance_map.shape, dtype=np.uint8)
    original[:, ::2] = instance_map[:, ::2] > 0
    filled = ((instance_map > 0) & ~original.astype(bool)).astype(np.uint8)
    unknown = (instance_map == 0).astype(np.uint8)
    np.savez_compressed(
        target_dir / f"{stem}_gt.npz",
        instance_map=instance_map,
        original_covered=original,
        filled=filled,
        residual_unknown=unknown,
    )
    class_ids = {
        str(value): index % 2
        for index, value in enumerate(
            sorted(int(value) for value in np.unique(instance_map) if value > 0)
        )
    }
    (target_dir / f"{stem}_class.json").write_text(
        json.dumps(class_ids), encoding="utf-8"
    )
    return class_ids


def test_manual_target_replaces_legacy_gt_and_ignores_only_unknown(tmp_path):
    raw = tmp_path / "raw"
    target_dir = tmp_path / "manual_target_v2"
    raw.mkdir()
    height, width = 4, 6
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(raw / "sample.jpg"), image)
    _write_labelme(raw / "sample.json", height, width)
    instances = np.array(
        [
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 2, 2],
        ],
        dtype=np.uint16,
    )
    _write_manual_target(target_dir, "sample", instances)

    dataset = DirectDualHeadDataset(
        raw,
        None,
        image_size=6,
        affinity_grid=6,
        augment=False,
        manual_target_dir=target_dir,
    )
    sample = dataset[0]

    assert sample["semantic_target"][0, 0, 0].item() == 0.0
    assert sample["semantic_target"][0, 0, 5].item() == 1.0
    assert sample["semantic_loss_weight"][0, 0, 0].item() == 1.0
    assert sample["semantic_loss_weight"][0, 0, 1].item() == 1.0
    assert sample["semantic_loss_weight"][0, 0, 2].item() == 0.0
    assert not sample["affinity_valid_content"][0, 0, 2]
    assert not bool(sample["uncovered_boundary_source"])
    assert sample["filled_pixels"] > 0


def test_manual_target_class_lut_must_exactly_match_instances():
    instances = np.array([[1, 1, 0, 2]], dtype=np.uint16)
    semantic, valid = semantic_from_instance_classes(instances, {1: 0, 2: 1})
    assert semantic.tolist() == [[0, 0, 0, 1]]
    assert valid.tolist() == [[True, True, False, True]]
    with pytest.raises(ValueError, match="exactly match"):
        semantic_from_instance_classes(instances, {1: 0})
    with pytest.raises(ValueError, match="stale instance id"):
        semantic_from_instance_classes(instances, {1: 0, 2: 1, 3: 0})
    with pytest.raises(ValueError, match="invalid semantic class"):
        semantic_from_instance_classes(instances, {1: 0, 2: 4})


def test_manual_target_manifest_and_archives_are_checked(tmp_path):
    raw = tmp_path / "raw"
    target_dir = tmp_path / "manual_target_v2"
    raw.mkdir()
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(raw / "sample.jpg"), image)
    _write_labelme(raw / "sample.json", 3, 5)
    _write_manual_target(
        target_dir,
        "sample",
        np.array([[1, 1, 0, 2, 2]] * 3, dtype=np.uint16),
    )
    manifest = {
        "format": "manual_target_v2",
        "version": 1,
        "max_fill_distance_native": 8.0,
        "blur_sigma": 1.2,
        "sample_count": 1,
        "samples": ["sample"],
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    report = validate_manual_target_dataset(target_dir, raw)
    assert report["sample_count"] == 1
    assert report["coverage"] == pytest.approx(0.8)
    loaded = load_manual_target_archive(target_dir / "sample_gt.npz")
    assert loaded[0].dtype == np.int32
    manifest["blur_sigma"] = 2.0
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="blur_sigma=1.2"):
        validate_manual_target_dataset(target_dir, raw)


def test_manual_evaluation_target_masks_unknown_without_changing_valid_pixels(
    tmp_path,
):
    raw = tmp_path / "raw"
    target_dir = tmp_path / "manual_target_v2"
    raw.mkdir()
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    assert cv2.imwrite(str(raw / "sample.jpg"), image)
    _write_labelme(raw / "sample.json", 2, 4)
    instances = np.array([[1, 1, 0, 2], [1, 1, 0, 2]], dtype=np.uint16)
    _write_manual_target(target_dir, "sample", instances)

    gt_map, classes, valid, audit = load_direct_evaluation_target(
        raw / "sample.jpg", (2, 4), target_dir
    )
    prediction = np.array([[5, 5, 9, 6], [5, 5, 9, 6]], dtype=np.uint16)
    masked = mask_prediction_to_evaluation_domain(prediction, valid)

    assert np.array_equal(gt_map, instances)
    assert classes == {1: 0, 2: 1}
    assert audit["source"] == "manual_target_v2"
    assert np.array_equal(masked[:, [0, 1, 3]], prediction[:, [0, 1, 3]])
    assert np.all(masked[:, 2] == 0)


def test_legacy_evaluation_target_keeps_full_image_scoring(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    assert cv2.imwrite(str(raw / "sample.jpg"), image)
    _write_labelme(raw / "sample.json", 4, 6)
    gt_map, classes, valid, audit = load_direct_evaluation_target(
        raw / "sample.jpg", (4, 6), None
    )

    assert gt_map.shape == (4, 6)
    assert set(classes.values()) == {0, 1}
    assert valid.all()
    assert audit["source"] == "labelme"


def test_unknown_pixels_contribute_no_semantic_loss_or_gradient():
    criterion = BoundaryLoss(
        seg_dice_weight=0.30,
        semantic_instance_weight=0.0,
        freeze_boundary=True,
        center_weight=0.0,
    )
    target = torch.zeros((1, 2, 2, 3))
    weight = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])
    prediction = torch.zeros((1, 2, 2, 3), requires_grad=True)
    loss, _, _ = criterion(prediction, target, seg_weight=weight)
    loss.backward()
    assert torch.all(prediction.grad[:, 0, :, 2] == 0)

    changed = torch.zeros((1, 2, 2, 3))
    changed[:, 0, :, 2] = 50.0
    changed_loss, _, _ = criterion(changed, target, seg_weight=weight)
    assert changed_loss.item() == pytest.approx(loss.item())


def test_manual_affinity_uses_new_boundaries_and_ignores_unknown():
    labels = torch.tensor([[[1, 1, 0, 2, 3]]], dtype=torch.long)
    valid = (labels > 0).unsqueeze(1)
    target, edge_valid = build_affinity_targets_torch(
        labels,
        valid,
        offsets=[(0, 1)],
    )

    assert edge_valid[0, 0, 0, 0] and target[0, 0, 0, 0] == 1
    assert not edge_valid[0, 0, 0, 1]
    assert not edge_valid[0, 0, 0, 2]
    assert edge_valid[0, 0, 0, 3] and target[0, 0, 0, 3] == 0


def test_loss_best_selection_is_strict_and_rejects_nonfinite():
    state = {"phase_best_loss": {}, "phase_best_epoch": {}}
    assert update_phase_loss_best(state, "head_warmup", 1.0, 0)
    assert not update_phase_loss_best(state, "head_warmup", 1.0, 1)
    assert update_phase_loss_best(state, "head_warmup", 0.9, 2)
    assert state["phase_best_epoch"]["head_warmup"] == 3
    with pytest.raises(RuntimeError, match="non-finite"):
        update_phase_loss_best(state, "head_warmup", float("nan"), 3)


def test_cached_semantic_proxy_ignores_unknown_pixels():
    logits = torch.tensor([[[[8.0, -8.0, 8.0, 8.0]]]])
    cached = [
        {
            "watershed_output": torch.cat(
                [logits, torch.zeros_like(logits)], dim=1
            ),
            "semantic_gt": np.array([[1, 0, 0, 1]], dtype=np.uint8),
            "gt_valid": np.array([[True, True, False, True]]),
        }
    ]
    metrics = evaluate_cached_semantic(cached)
    assert metrics["miou"] == pytest.approx(1.0)


def test_direct_dataset_aligns_semantic_and_affinity_targets(tmp_path):
    raw = tmp_path / "raw"
    gt = tmp_path / "gt"
    raw.mkdir()
    gt.mkdir()
    height, width = 8, 12
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = 40
    image[:, width // 2 :] = 210
    assert cv2.imwrite(str(raw / "sample.jpg"), image)
    _write_labelme(raw / "sample.json", height, width)
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[:, width // 2 :] = 1
    boundary = np.zeros((height, width), dtype=np.float32)
    boundary[:, width // 2 - 1 : width // 2 + 1] = 1.0
    np.savez_compressed(
        gt / "sample_gt.npz",
        semantic=semantic,
        boundary=boundary,
        boundary_soft=boundary,
    )

    dataset = DirectDualHeadDataset(
        raw, gt, image_size=16, affinity_grid=8, augment=False
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["semantic_target"].shape == (1, 16, 16)
    assert sample["semantic_instance_map"].shape == (16, 16)
    assert sample["affinity_instance_map"].shape == (8, 8)
    assert set(torch.unique(sample["affinity_instance_map"]).tolist()) >= {0, 1, 2}
    assert sample["semantic_target"].max().item() == 1.0


class _Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.ones(()))
        self.lora_A = nn.Parameter(torch.ones(()))
        self.lora_B = nn.Parameter(torch.ones(()))

    def forward(self, image):
        return [image]


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _Trunk()
        self.trainable_lora = True

    def forward(self, image):
        return self.trunk(image)


class _Semantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, features, image):
        return features[0][:, :1] * self.weight


class _Affinity(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, features):
        return {"affinity_logits": features[0][:, :1] * self.weight}


def test_phase_control_freezes_only_lora_during_head_warmup():
    model = FusedPhaseAffinityModel(_Encoder(), _Semantic(), _Affinity())
    configure_direct_training_phase(model, train_lora=False)
    assert not model.encoder.trainable_lora
    assert not model.encoder.trunk.lora_A.requires_grad
    assert not model.encoder.trunk.base.requires_grad
    assert model.semantic_decoder.weight.requires_grad
    assert model.affinity_decoder.weight.requires_grad
    warmup_groups = direct_parameter_groups(
        model, {"semantic": 1e-4, "affinity": 5e-5, "lora": 2e-6}
    )
    assert [group["name"] for group in warmup_groups] == ["semantic", "affinity"]

    configure_direct_training_phase(model, train_lora=True)
    assert model.encoder.trainable_lora
    assert model.encoder.trunk.lora_A.requires_grad
    assert model.encoder.trunk.lora_B.requires_grad
    assert not model.encoder.trunk.base.requires_grad
    joint_groups = direct_parameter_groups(
        model, {"semantic": 2e-5, "affinity": 1e-5, "lora": 2e-6}
    )
    assert [group["name"] for group in joint_groups] == [
        "semantic",
        "affinity",
        "lora",
    ]


def test_semantic_recovery_freezes_affinity_and_lora_path():
    model = FusedPhaseAffinityModel(_Encoder(), _Semantic(), _Affinity())
    configure_direct_semantic_recovery(model)
    assert all(not value.requires_grad for value in model.encoder.parameters())
    assert all(value.requires_grad for value in model.semantic_decoder.parameters())
    assert all(not value.requires_grad for value in model.affinity_decoder.parameters())
    assert model.encoder.trainable_lora is False


def test_affinity_recovery_updates_only_affinity_decoder():
    model = FusedPhaseAffinityModel(_Encoder(), _Semantic(), _Affinity())
    configure_direct_affinity_recovery(model)
    assert all(not value.requires_grad for value in model.encoder.parameters())
    assert all(not value.requires_grad for value in model.semantic_decoder.parameters())
    assert all(value.requires_grad for value in model.affinity_decoder.parameters())
    assert model.encoder.trainable_lora is False

    semantic_before = model.semantic_decoder.weight.detach().clone()
    lora_before = model.encoder.trunk.lora_A.detach().clone()
    affinity_before = model.affinity_decoder.weight.detach().clone()
    optimizer = torch.optim.SGD(model.affinity_decoder.parameters(), lr=0.1)
    image = torch.ones((1, 3, 4, 4))
    with torch.no_grad():
        features = model.encoder(image)
    loss = model.affinity_decoder(features)["affinity_logits"].sum()
    loss.backward()
    optimizer.step()

    assert torch.equal(model.semantic_decoder.weight, semantic_before)
    assert torch.equal(model.encoder.trunk.lora_A, lora_before)
    assert not torch.equal(model.affinity_decoder.weight, affinity_before)


def test_sam2_encoder_reports_runtime_hiera_channels():
    encoder = SAM2Encoder.__new__(SAM2Encoder)
    nn.Module.__init__(encoder)
    encoder.stage_channels = [144, 288, 576, 1152]
    assert encoder.get_stage_channels() == [144, 288, 576, 1152]
    returned = encoder.get_stage_channels()
    returned[0] = 1
    assert encoder.get_stage_channels()[0] == 144


def test_sam2_input_normalization_preserves_legacy_contract():
    image = torch.rand(1, 3, 4, 4)
    output = normalize_sam2_trunk_input(image, "legacy_none")
    assert output is image


def test_sam2_input_normalization_matches_official_constants():
    image = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    output = normalize_sam2_trunk_input(image, "imagenet_v1")
    assert torch.allclose(output, torch.zeros_like(output), atol=1.0e-7)


def test_sam2_input_normalization_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported SAM2 input normalization"):
        normalize_sam2_trunk_input(torch.zeros(1, 3, 1, 1), "automatic")


def test_direct_ssl_artifact_requires_matching_input_contract(tmp_path):
    artifact = tmp_path / "lora_state_dict.pth"
    state = {"block.lora_A": torch.ones(1)}
    torch.save(state, artifact)
    legacy = {"sam2": {"input_normalization": "legacy_none"}}
    normalized = {"sam2": {"input_normalization": "imagenet_v1"}}

    loaded, mode = load_direct_ssl_lora_state(artifact, legacy)
    assert loaded.keys() == state.keys()
    assert mode == "legacy_none"
    with pytest.raises(RuntimeError, match="requires SSL input_normalization"):
        load_direct_ssl_lora_state(artifact, normalized)

    (tmp_path / "meta.json").write_text(
        json.dumps({"input_normalization": "imagenet_v1"}), encoding="utf-8"
    )
    _, mode = load_direct_ssl_lora_state(artifact, normalized)
    assert mode == "imagenet_v1"
    with pytest.raises(RuntimeError, match="SSL/direct input_normalization mismatch"):
        load_direct_ssl_lora_state(artifact, legacy)


def test_direct_checkpoint_rejects_input_contract_mismatch():
    model = FusedPhaseAffinityModel(_Encoder(), _Semantic(), _Affinity())
    model.encoder.input_normalization = "imagenet_v1"
    with pytest.raises(RuntimeError, match="input_normalization mismatch"):
        assert_direct_checkpoint_input_contract(
            model, {"config": {"sam2": {"input_normalization": "legacy_none"}}}
        )
    assert_direct_checkpoint_input_contract(
        model, {"config": {"sam2": {"input_normalization": "imagenet_v1"}}}
    )


def test_direct_checkpoint_uses_runtime_paths_and_complete_lora():
    model = FusedPhaseAffinityModel(_Encoder(), _Semantic(), _Affinity())
    state = {
        key: torch.full_like(value, 3.0)
        for key, value in model.encoder.trunk.state_dict().items()
        if "lora_A" in key or "lora_B" in key
    }
    assert _load_complete_lora_state(model, state) == 2
    assert model.encoder.trunk.lora_A.item() == 3.0
    try:
        _load_complete_lora_state(model, {"lora_A": torch.ones(())})
    except RuntimeError as exc:
        assert "architecture mismatch" in str(exc)
    else:
        raise AssertionError("incomplete LoRA state must be rejected")

    stored = {
        "paths": {
            "project_root": "/server/project",
            "weights_dir": "weights",
            "sam2_ckpt": "base.pt",
        },
        "sam2": {
            "sam2_repo_path": "segment-anything-2",
            "input_normalization": "imagenet_v1",
        },
    }
    runtime = {
        "paths": {
            "project_root": "D:/project",
            "weights_dir": "runtime-weights",
            "sam2_ckpt": "runtime.pt",
        },
        "sam2": {
            "sam2_repo_path": "runtime-sam2",
            "input_normalization": "legacy_none",
        },
    }
    restored = _checkpoint_architecture_config(
        {"config": stored}, runtime
    )
    assert restored["paths"] == runtime["paths"]
    assert restored["sam2"]["sam2_repo_path"] == "runtime-sam2"
    assert restored["sam2"]["input_normalization"] == "imagenet_v1"
    assert stored["paths"]["project_root"] == "/server/project"


def test_pseudo_affinity_keeps_uncovered_pairs_ignored():
    labels = torch.zeros((1, 9, 9), dtype=torch.long)
    labels[:, 1:8, 1:4] = 1
    labels[:, 1:8, 4:8] = 2
    valid = torch.ones((1, 1, 9, 9), dtype=torch.bool)
    logits = torch.zeros((1, 8, 9, 9), requires_grad=True)
    batch = {
        "instance_map": labels,
        "valid_content": valid,
        "uncovered_boundary_source": torch.tensor([False]),
    }
    loss, metrics = compute_affinity_loss(
        logits,
        batch,
        {
            "negative_weight": 1.5,
            "hard_negative_weight": 1.0,
            "hard_negative_gamma": 2.0,
            "manual_uncovered_as_boundary": True,
            "manual_uncovered_boundary_weight": 0.20,
            "pseudo_negative_weight": 1.0,
        },
        pseudo=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["positive_edges"] > 0
    assert metrics["negative_edges"] > 0
    assert logits.grad is not None


def test_approval_rejects_semantic_labels(tmp_path):
    dataset_dir = tmp_path / "sam2"
    masks = dataset_dir / "masks"
    masks.mkdir(parents=True)
    digest = "a" * 64
    row = {
        "source_sha256": digest,
        "source_file": "sample.jpg",
        "mask_file": "masks/sample.npz",
        "class_label": 1,
    }
    (dataset_dir / "manifest.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    (dataset_dir / "approval.json").write_text(
        json.dumps(
            {
                "approved": True,
                "reviewed_by": "human",
                "reviewed_at": "2026-08-31",
                "source_count": 1,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(masks / "sample.npz", instance_map=np.ones((2, 2)))
    try:
        validate_sam2_geometry_approval(dataset_dir)
    except RuntimeError as exc:
        assert "class-agnostic" in str(exc)
    else:
        raise AssertionError("semantic SAM2 labels must be rejected")


def test_direct_monitor_keeps_semantic_and_boundary_heatmaps(tmp_path):
    image_dir = tmp_path / "unlabeled"
    image_dir.mkdir()
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "sample.jpg"), image)
    manifest = tmp_path / "monitor.txt"
    manifest.write_text("sample.jpg\n", encoding="utf-8")
    config = {
        "paths": {"project_root": str(tmp_path)},
        "direct_semantic_affinity": {
            "output_dir": "outputs/direct",
            "input_size": 16,
            "deployment_validation": {"fusion_mode": "mean"},
            "monitor": {
                "enabled": True,
                "interval": 10,
                "image_dir": "unlabeled",
                "manifest": "monitor.txt",
                "num_images": 1,
                "output_dir": "outputs/direct/monitor",
                "preview_max_side": 6,
            },
        },
    }

    class _MonitorModel(nn.Module):
        def forward(self, tensor):
            batch, _, height, width = tensor.shape
            return {
                "semantic_logits": torch.zeros(batch, 1, height, width),
                "affinity_logits": torch.zeros(batch, 8, height // 2, width // 2),
            }

    assert [path.name for path in resolve_direct_monitor_paths(config)] == [
        "sample.jpg"
    ]
    save_direct_monitor(_MonitorModel(), config, 9, torch.device("cpu"))
    monitor_dir = tmp_path / "outputs/direct/monitor/epoch_0010"
    semantic = cv2.imread(str(monitor_dir / "sample_seg.png"))
    boundary = cv2.imread(str(monitor_dir / "sample_boundary.png"))
    semantic_preview = cv2.imread(str(monitor_dir / "preview/sample_seg.png"))
    boundary_preview = cv2.imread(
        str(monitor_dir / "preview/sample_boundary.png")
    )
    assert semantic.shape == (8, 12, 3)
    assert boundary.shape == (8, 12, 3)
    assert semantic_preview.shape == (4, 6, 3)
    assert boundary_preview.shape == (4, 6, 3)
