# -*- coding: utf-8 -*-

import json

import cv2
import numpy as np
import torch
from torch import nn

from data.direct_dual_head_dataset import DirectDualHeadDataset
from models.direct_semantic_affinity import (
    _checkpoint_architecture_config,
    _load_complete_lora_state,
    configure_direct_training_phase,
    direct_parameter_groups,
)
from models.fused_deployment import FusedPhaseAffinityModel
from train_direct_semantic_affinity import (
    compute_affinity_loss,
    validate_sam2_geometry_approval,
)


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
        "sam2": {"sam2_repo_path": "segment-anything-2"},
    }
    runtime = {
        "paths": {
            "project_root": "D:/project",
            "weights_dir": "runtime-weights",
            "sam2_ckpt": "runtime.pt",
        },
        "sam2": {"sam2_repo_path": "runtime-sam2"},
    }
    restored = _checkpoint_architecture_config(
        {"config": stored}, runtime
    )
    assert restored["paths"] == runtime["paths"]
    assert restored["sam2"]["sam2_repo_path"] == "runtime-sam2"
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
