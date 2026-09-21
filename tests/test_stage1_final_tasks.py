# -*- coding: utf-8 -*-
"""Stage1最终任务替代的GT隔离、完整初始化及交接契约。"""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from data.stage1_final_tasks import Stage1FinalTaskDataset
from models.affinity_geometry import AffinityGeometryDecoder
from models.fpn_decoder import FPNDecoder
from models.stage1_final_tasks import initialize_from_stage1, export_final_task_checkpoints
from utils.semantic_challenger import SemanticChallenger
from train_direct_semantic_affinity import compute_semantic_loss
from utils.loss import BoundaryLoss


def test_new_semantic_targets_never_reuse_old_instances_and_unknown_has_no_gradient(tmp_path):
    raw, old, new = [tmp_path / n for n in ("raw", "old", "new")]
    for p in (raw, old, new):
        p.mkdir()
    cv2.imwrite(str(raw / "sample.png"), np.full((24, 32, 3), 100, np.uint8))
    (raw / "sample.json").write_text(json.dumps({"shapes": [{"label": "pearlite", "shape_type": "polygon",
        "points": [[0, 0], [31, 0], [31, 23], [0, 23]]}]}))
    np.savez(old / "sample_gt.npz", semantic=np.zeros((24, 32), np.uint8), boundary=np.zeros((24, 32), np.float32))
    instances = np.ones((24, 32), np.uint16)
    instances[:, 16:] = 2
    instances[5:10, 3:7] = 0
    np.savez(new / "sample_gt.npz", instance_map=instances, residual_unknown=instances == 0)
    (new / "sample_class.json").write_text('{"1": 1, "2": 0}')
    data = Stage1FinalTaskDataset(raw, old, completed_gt_dir=new, image_size=32,
                                  affinity_grid=16, augment=False)
    sample = data[0]
    sem, ids = sample["semantic_target"][0], sample["semantic_instance_map"]
    assert torch.equal(sem < 0, ids == 0)
    assert torch.all(sem[ids == 1] == 1)
    assert torch.all(sem[ids == 2] == 0)
    assert sample["affinity_instance_map"].max() == 1  # 旧人工仍是一整块
    assert not sample["semantic_valid_content"][0, 24:].any()
    batch = {k: v.unsqueeze(0) for k, v in sample.items() if torch.is_tensor(v)}
    logits = torch.randn(1, 1, 32, 32, requires_grad=True)
    criterion = BoundaryLoss(freeze_boundary=True, center_weight=0, seg_dice_weight=0.3,
                              semantic_instance_weight=0.75, semantic_instance_pool_weight=0.5)
    loss = compute_semantic_loss(criterion, logits, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert not logits.grad[batch["semantic_target"] < 0].any()
    assert logits.grad[batch["semantic_target"] >= 0].abs().sum() > 0


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(2, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


def test_stage1_initialization_is_complete_and_preserves_new_layers(tmp_path):
    channels = [8, 16, 24, 32]
    source = FPNDecoder(in_channels=channels, fpn_channels=32, num_classes=2, dropout=0.1, use_bn=True)
    scaffold = FPNDecoder(in_channels=channels, fpn_channels=32, num_classes=2, dropout=0.1, use_bn=True)
    geometry = AffinityGeometryDecoder(in_channels=channels, fpn_channels=32, up_channels=32, output_grid=16)
    model = SimpleNamespace(encoder=SimpleNamespace(trunk=Trunk()),
        semantic_decoder=SemanticChallenger(scaffold), affinity_decoder=geometry)
    checkpoint = {"epoch": 80, "decoder_state_dict": source.state_dict(),
                  "lora_state_dict": {"lora_A": torch.ones(2, 2), "lora_B": torch.full((2, 2), 2.0)}}
    output_before = {k: v.clone() for k, v in geometry.affinity_head.state_dict().items()}
    initialize_from_stage1(model, checkpoint)
    assert all(torch.equal(v, source.seg_fpn.state_dict()[k]) for k, v in model.semantic_decoder.seg_fpn.state_dict().items())
    assert all(torch.equal(v, source.boundary_fpn.state_dict()[k]) for k, v in geometry.geometry_fpn.state_dict().items())
    assert all(torch.equal(v, output_before[k]) for k, v in geometry.affinity_head.state_dict().items())
    assert model.encoder.trunk.lora_B.max() == 2
    task = {"phase": "joint_lora", "epoch": 7, "epoch_index": 6, "selection": {"metric": "val_loss"},
            "semantic_state_dict": model.semantic_decoder.state_dict(), "affinity_state_dict": geometry.state_dict(),
            "lora_state_dict": checkpoint["lora_state_dict"], "config": {}}
    reference, affinity = export_final_task_checkpoints(task, checkpoint, {"decoder": {"fpn_channels": 32}}, tmp_path)
    restored = FPNDecoder(in_channels=channels, fpn_channels=32, num_classes=2, dropout=0.1, use_bn=True)
    restored.load_state_dict(reference["decoder_state_dict"], strict=True)
    geometry.load_state_dict(affinity["geometry_state_dict"], strict=True)
    assert reference["lora_state_dict"] is task["lora_state_dict"]
    incomplete = dict(checkpoint, lora_state_dict={"lora_A": torch.ones(2, 2)})
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        initialize_from_stage1(model, incomplete)
