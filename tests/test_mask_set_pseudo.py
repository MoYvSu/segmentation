# -*- coding: utf-8 -*-
"""伪标签真实来源、留出隔离、容量、在线几何与梯度权重合同。"""
import copy
import json

import cv2
import numpy as np
import pytest
import torch

from data.mask_set_pseudo_dataset import (FixedSourceRatioSampler, MainlinePseudoDataset,
    MaskSetSampleView, PSEUDO_FORMAT, read_pseudo_manifest)
from utils.mask_set_loss import MaskSetCriterion, targets_from_direct_batch


def pseudo_fixture(tmp_path):
    raw, targets = tmp_path/"unlabeled", tmp_path/"teacher"
    raw.mkdir(); targets.mkdir(); (targets/"instances").mkdir()
    instances = np.ones((4, 8), np.uint16)
    instances[:, 4:] = 2
    instances[0, 0] = 0
    cv2.imwrite(str(raw/"train_101.png"), np.full((4, 8, 3), 127, np.uint8))
    cv2.imwrite(str(targets/"instances/train_101_inst.png"), instances)
    (targets/"instances/train_101_class.json").write_text(json.dumps({1:0, 2:1}))
    manifest = dict(format=PSEUDO_FORMAT, label_source="mainline_pseudo", status="complete", samples=["train_101"],
        sample_count=1, images=[dict(stem="train_101",image_name="train_101.png",source_sha256="fixture",native_shape=[4,8])])
    (targets/"manifest.json").write_text(json.dumps(manifest))
    return raw, targets, manifest


def test_pseudo_source_exclusion_and_capacity_are_not_silently_changed(tmp_path):
    raw, targets, manifest = pseudo_fixture(tmp_path)
    _, cm = read_pseudo_manifest(targets, raw, excluded_names=[], query_capacity=2)
    assert cm == {"train_101": {1:0, 2:1}}
    with pytest.raises(ValueError, match="leaks excluded"):
        read_pseudo_manifest(targets, raw, excluded_names=["train_101.jpg"], query_capacity=2)
    with pytest.raises(ValueError, match="query capacity"):
        read_pseudo_manifest(targets, raw, excluded_names=[], query_capacity=1)
    manifest["label_source"] = "manual"
    (targets/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicitly sourced"):
        read_pseudo_manifest(targets, raw, excluded_names=[], query_capacity=2)


def test_pseudo_letterbox_preserves_unknown_and_online_transforms(tmp_path, monkeypatch):
    raw, targets, manifest = pseudo_fixture(tmp_path)
    dataset = MainlinePseudoDataset(raw, targets, manifest, image_size=8, mask_grid=8)
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 8)
    assert not sample["affinity_valid_content"][:, 4:].any()
    assert not sample["affinity_valid_content"][0, 0, 0]
    assert sample["affinity_instance_map"][1, 1] == 1
    assert sample["affinity_instance_map"][1, 5] == 2
    dataset.augment = True
    dataset.augmentation = dict(horizontal_flip=True,vertical_flip=False,rotation90=False,brightness=0,contrast=0,gamma=0)
    monkeypatch.setattr(np.random, "rand", lambda: 0.)
    transformed = dataset[0]
    torch.testing.assert_close(transformed["affinity_instance_map"], sample["affinity_instance_map"].flip(-1))
    torch.testing.assert_close(transformed["affinity_valid_content"], sample["affinity_valid_content"].flip(-1))
    wrapped = MaskSetSampleView(dataset, "mainline_pseudo", .5)[0]
    assert wrapped["label_source"] == "mainline_pseudo" and wrapped["sample_loss_weight"] == .5


def test_fixed_source_ratio_and_reproducible_manual_stream():
    one = list(FixedSourceRatioSampler(26, 249, 64, 192, 42))
    two = list(FixedSourceRatioSampler(26, 249, 64, 192, 42))
    other = list(FixedSourceRatioSampler(26, 249, 64, 64, 42))
    assert one == two and len(one) == 256
    assert sum(i < 26 for i in one) == 64
    assert sum(i >= 26 for i in one) == 192
    assert [i for i in one if i < 26] == [i for i in other if i < 26]


def test_pseudo_half_weight_halves_actual_sgd_update():
    from train_mask_set import run_epoch

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[[0., 2., -2.], [0., 0., 1.]]]))
            self.masks = torch.nn.Parameter(torch.zeros(1, 2, 2, 2))
        def forward(self, image):
            return dict(pred_logits=self.logits, pred_masks=self.masks)

    first = TinyModel(); second = copy.deepcopy(first)
    before = {k:v.clone() for k,v in first.state_dict().items()}
    batch = dict(image=torch.zeros(1,3,2,2),image_name=["train_101.png"],
        affinity_instance_map=torch.ones(1,2,2,dtype=torch.long),
        affinity_valid_content=torch.ones(1,1,2,2,dtype=torch.bool))
    cfg = {"mask_set": {"amp":False,"grad_clip":1.e6}}
    for model, weight in ((first,1.),(second,.5)):
        batch["sample_loss_weight"] = torch.tensor([weight])
        batch["label_source"] = ["mainline_pseudo"]
        run_epoch(model,[batch],MaskSetCriterion(ownership_weight=.5),cfg,torch.device("cpu"),
                  {"train_101":{1:1}},torch.optim.SGD(model.parameters(),lr=.1),
                  torch.amp.GradScaler("cuda",enabled=False))
    for key in before:
        torch.testing.assert_close(second.state_dict()[key]-before[key],
                                   .5*(first.state_dict()[key]-before[key]),atol=2e-7,rtol=1e-5)
