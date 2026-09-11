# -*- coding: utf-8 -*-
"""在线实例一致性的几何对应、unknown零梯度、EMA独立性与A/B采样约束。"""
import copy
import json

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.mask_set_unlabeled import MaskSetUnlabeledDataset, build_unlabeled_pool
from utils.mask_set_consistency import (MaskSetConsistency, consistency_criterion, grid_partition,
    make_ema_teacher, match_partitions, stable_targets, transform_student_view, update_mask_set_ema)
from utils.mask_set_inference import postprocess_mask_set


def partition(ids, classes, confidence=1.):
    ids = torch.tensor(ids, dtype=torch.long)
    return {"instances": ids, "classes": torch.tensor(classes), "confidence": torch.full(ids.shape, confidence)}


def test_grid_competition_matches_deployment_including_ties_padding_and_empty():
    torch.manual_seed(5)
    masks = torch.randn(1, 5, 8, 8)
    logits = torch.tensor([[[2., 0., -2.], [2., 0., -2.], [0., 2., -2.], [0., 0., 5.], [0., 0., 0.]]])
    masks[:, 1] = masks[:, 0]
    domain = torch.ones(8, 8, dtype=torch.bool)
    domain[6:] = False
    grid = grid_partition(dict(pred_masks=masks, pred_logits=logits), domain)
    pred, _, info = postprocess_mask_set(masks[0], logits[0], (8, 8), (0, 0), input_size=8)
    lookup = {r["instance_id"]:r["query"]+1 for r in info["query_results"] if r["accepted"]}
    expected = np.zeros_like(pred)
    for label, query in lookup.items():
        expected[pred == label] = query
    expected[~domain.numpy()] = 0
    np.testing.assert_array_equal(grid["instances"].numpy(), expected)
    assert not (grid["instances"] == 2).any()


def test_stability_matches_instances_after_query_permutation_and_filters_pixels():
    base = partition([[1,1,2,2],[1,1,2,2],[0,0,0,0]], [0,1])
    alternate = partition([[2,2,1,1],[2,2,1,1],[0,0,0,0]], [1,0])
    domain = torch.ones(3,4,dtype=torch.bool)
    domain[2] = False
    assert match_partitions(base, alternate, domain) == {1:2,2:1}
    alternate["confidence"][0,0] = .6
    target, stats = stable_targets(base,alternate,alternate,domain)
    assert target["labels"].tolist() == [0,1]
    assert stats["selected_instances"] == 2 and stats["valid_pixels"] == 7
    assert not target["valid"][0,0] and not target["valid"][2].any()
    assert torch.all(target["masks"].sum(0)[target["valid"]] == 1)
    image = torch.arange(36).reshape(1,3,3,4).float()/36
    strong, transformed = transform_student_view(image,target,gain=1,bias=0,gamma=1,horizontal_flip=True)
    torch.testing.assert_close(strong,image.flip(-1))
    for key in ("masks","valid","content_valid"):
        torch.testing.assert_close(transformed[key],target[key].flip(-1))


def test_unstable_or_wrong_class_instances_do_not_create_background_targets():
    base = partition([[1,1,2,2],[1,1,2,2]], [0,1])
    wrong = partition([[1,1,2,2],[1,1,2,2]], [1,0])
    domain = torch.ones(2,4,dtype=torch.bool)
    target, stats = stable_targets(base,wrong,wrong,domain)
    assert stats["selected_instances"] == 0 and not target["valid"].any()
    assert target["masks"].shape == (0,2,4)


def test_consistency_has_no_unknown_pixel_or_unmatched_query_gradient():
    source = partition([[1,1,2,2],[0,0,0,0]], [0,1])
    domain = torch.ones(2,4,dtype=torch.bool)
    target,_ = stable_targets(source,source,source,domain)
    logits = torch.tensor([[[4.,0.,-2.],[0.,4.,-2.],[-2.,-2.,4.]]],requires_grad=True)
    masks = torch.tensor([[[[2.,2.,-2.,-2.],[1.,1.,1.,1.]],
                           [[-2.,-2.,2.,2.],[1.,1.,1.,1.]],
                           [[-3.,-3.,-3.,-3.],[2.,2.,2.,2.]]]],requires_grad=True)
    criterion = consistency_criterion({"num_points":8,"mask_points":8,"ownership_weight":.5})
    losses = criterion(dict(pred_masks=masks,pred_logits=logits),[target])
    losses["loss_total"].backward()
    assert logits.grad[0,:2].abs().sum() > 0 and masks.grad[0,:2,0].abs().sum() > 0
    assert not logits.grad[0,2].any() and not masks.grad[0,2].any()
    assert not masks.grad[:,:,:, :][...,1,:].any()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.trunk = nn.Linear(1,1)
        self.encoder.trainable_lora = True
        self.logits = nn.Parameter(torch.tensor([[[4.,0.,-2.],[0.,4.,-2.]]]))
        self.masks = nn.Parameter(torch.tensor([[[[4.,4.,-4.,-4.],[4.,4.,-4.,-4.]],
                                                 [[-4.,-4.,4.,4.],[-4.,-4.,4.,4.]]]]))
    def forward(self,image):
        # 几何随输入左右变化，teacher翻转恢复后仍对应相同物理区域。
        masks = self.masks if image[0,0,0,0] < image[0,0,0,-1] else self.masks.flip(-1)
        return dict(pred_masks=masks,pred_logits=self.logits)


def test_ema_teacher_is_independent_and_refuses_checkpoint_closure_copy():
    student = TinyModel()
    teacher = make_ema_teacher(student)
    before = teacher.masks.clone()
    with torch.no_grad(): student.masks.add_(2)
    torch.testing.assert_close(teacher.masks,before)
    update_mask_set_ema(teacher,student,.75)
    torch.testing.assert_close(teacher.masks,before+.5)
    assert all(not p.requires_grad for p in teacher.parameters())
    student.encoder.trunk.forward = lambda x:x
    with pytest.raises(ValueError,match="clone EMA"):
        make_ema_teacher(student)


def test_consistency_does_not_advance_supervised_rng_and_updates_ema_once():
    student = TinyModel()
    teacher = make_ema_teacher(student)
    image = torch.linspace(.2,.8,4).repeat(3,2,1)
    loader = DataLoader([dict(image=image,content_valid=torch.ones(2,4,dtype=torch.bool),image_name="train_100.png")],
        generator=torch.Generator().manual_seed(8))
    cfg = {"mask_set":{"loss":{"num_points":4,"mask_points":4},"amp":False,
        "consistency":{"weight":.2,"ramp_epochs":1,"every_n_steps":1,"ema_decay":.9,"seed":42,"iou_threshold":.8,"mask_confidence":.7}}}
    controller = MaskSetConsistency(teacher,loader,cfg,torch.device("cpu"),4)
    state = torch.get_rng_state().clone()
    loss,stats = controller.loss(student)
    assert torch.equal(state,torch.get_rng_state())
    assert stats["samples"]==1 and stats["selected_instances"]==2
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert student.masks.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())
    controller.after_step(student)
    assert controller.step==1 and controller.processed_samples==1


def test_unlabeled_pool_excludes_named_heldout_and_renamed_duplicates(tmp_path):
    raw=tmp_path/"unlabeled"; raw.mkdir()
    image=np.zeros((4,8,3),np.uint8)
    for name,value in (("train_001",image),("train_002",image),("train_003",image+128),("train_004",image+128)):
        cv2.imwrite(str(raw/f"{name}.png"),value)
    manifest=build_unlabeled_pool(raw,["train_001.png"])
    assert manifest["sample_count"]==1 and manifest["images"][0]["stem"]=="train_003"
    assert {r["reason"] for r in manifest["excluded"]}=={"excluded_name","excluded_content","duplicate_content"}
    sample=MaskSetUnlabeledDataset(raw,manifest,input_size=8,mask_grid=4)[0]
    assert sample["image"].shape==(3,8,8) and sample["content_valid"].sum()==8
    assert not sample["content_valid"][2:].any()
    cv2.imwrite(str(raw/"test_001.png"),image)
    with pytest.raises(ValueError,match="train_"):
        build_unlabeled_pool(raw,[])


def test_continuation_rejects_changed_loss_or_validation_split():
    from train_mask_set_consistency import check_continuation
    from utils.config import load_config
    cfg=load_config("config/train/mask_set_continue30.yaml")
    split={"train":["train_001"],"val":["train_002"]}
    parent={"config":copy.deepcopy(cfg),"split":split,"phase":"joint_lora"}
    check_continuation(cfg,parent,split)
    broken=copy.deepcopy(cfg); broken["mask_set"]["loss"]["ownership_weight"]=0
    with pytest.raises(ValueError,match="loss"):
        check_continuation(broken,parent,split)
    with pytest.raises(ValueError,match="split"):
        check_continuation(cfg,parent,{"train":split["val"],"val":split["train"]})


def test_cross_directory_renamed_manual_image_is_excluded_from_pool_and_generator(tmp_path):
    from tools.generate_mainline_pseudo import select_candidates
    raw,manual=tmp_path/"unlabeled",tmp_path/"manual"
    raw.mkdir();manual.mkdir()
    # 两目录同名文件内容不同；真实人工图却以另一名称存在于无标签池。
    for path,value in ((raw/"train_001.png",0),(raw/"train_100.png",90),(raw/"train_200.png",180),(manual/"train_001.png",90)):
        cv2.imwrite(str(path),np.full((8,12,3),value,np.uint8))
    pool=build_unlabeled_pool(raw,["train_001"],excluded_image_dir=manual)
    assert [r["stem"] for r in pool["images"]]==["train_200"]
    candidates,selection=select_candidates(raw,{"train_001"},1.,42,excluded_image_dir=manual)
    assert [r["stem"] for r in candidates]==["train_200"]
    assert selection["external_manual_content_checked"]


def test_pseudo_reader_rejects_actual_manual_content_under_another_name(tmp_path):
    from data.mask_set_pseudo_dataset import read_pseudo_manifest,PSEUDO_FORMAT
    from data.mask_set_unlabeled import image_digest
    raw,manual,target=tmp_path/"unlabeled",tmp_path/"manual",tmp_path/"pseudo"
    raw.mkdir();manual.mkdir();target.mkdir()
    pixels=np.full((8,12,3),90,np.uint8)
    cv2.imwrite(str(raw/"train_100.png"),pixels)
    cv2.imwrite(str(manual/"train_001.png"),pixels)
    manifest={"format":PSEUDO_FORMAT,"label_source":"mainline_pseudo","status":"complete","sample_count":1,
        "samples":["train_100"],"images":[{"stem":"train_100","image_name":"train_100.png","source_sha256":image_digest(raw/"train_100.png")} ]}
    (target/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="excluded image content"):
        read_pseudo_manifest(target,raw,excluded_names=["train_001"],query_capacity=256,excluded_image_dir=manual)


def test_evaluation_uses_saved_alias_exclusions_and_marks_explicit_override(tmp_path):
    from tools.evaluate_mask_set import resolve_evaluation_paths
    raw=tmp_path/"raw";raw.mkdir()
    for stem in ("train_172","train_889"):
        (raw/f"{stem}.png").touch()
    cfg={"paths":{"project_root":str(tmp_path),"raw_data_dir":"raw"}}
    payload={"split":{"train":[],"val":["train_172","train_889"]},"evaluation_excluded_names":["train_889"]}
    paths,override,_=resolve_evaluation_paths(cfg,payload)
    assert [p.stem for p in paths]==["train_172"] and not override
    paths,override,_=resolve_evaluation_paths(cfg,payload,names=["train_889"])
    assert [p.stem for p in paths]==["train_889"] and override
    payload["evaluation_excluded_names"]=["not_in_split"]
    with pytest.raises(ValueError,match="exclusions"):
        resolve_evaluation_paths(cfg,payload)
