# -*- coding: utf-8 -*-
"""教师诊断的来源身份、概率网格、可靠域及GT纠错方向。"""
from pathlib import Path

import numpy as np
import torch

from tools.probe_semantic_teacher import (
    native_gt_votes, pair_stats, probability_grid, pure_grid_gt,
    reliable_target, select_unlabelled, summarize,
)


def test_selection_excludes_content_mapped_source_not_manual_name():
    files = [Path(f'train_{i:03d}.jpg') for i in range(10)]
    mapping = [{'manual_name': 'train_000.jpg', 'pool_name': 'train_007.jpg'}]
    chosen = select_unlabelled(files, mapping, 9, 23)
    assert Path('train_007.jpg') not in chosen
    assert Path('train_000.jpg') in chosen
    assert select_unlabelled(files, mapping, 4, 23) == select_unlabelled(list(reversed(files)), mapping, 4, 23)
    assert len(set(chosen)) == 9


def test_probability_area_reduction_does_not_average_logits():
    logits = torch.tensor([[[[10., -2.], [10., -2.]]]])
    p = probability_grid(logits, 1)
    assert torch.allclose(p, logits.sigmoid().mean().reshape(1,1,1,1))
    assert abs(float(p) - float(logits.mean().sigmoid())) > .3


def test_reliable_target_excludes_padding_uncertainty_and_flip_disagreement():
    a = torch.full((1,1,4,4), .99)
    b, c = a.clone(), a.clone()
    valid = torch.ones_like(a)
    valid[..., 3] = 0
    b[..., 0, 0] = .94  # 高置信但三视图跨度大于允许值
    c[..., 0, 1] = .01  # 镜像复原后类别不一致
    a[..., 1, 0] = .6
    target, mask, info = reliable_target([a,b,c], valid, confidence=.9, gap=.04, erosion=0)
    assert not mask[..., 3].any()
    assert not mask[0,0,0,0] and not mask[0,0,0,1] and not mask[0,0,1,0]
    assert info['valid'] == 12 and info['accepted'] == 9
    assert info['weak_class_disagreement'] == 1


def test_reliable_erosion_is_classwise_and_has_no_padding_leak():
    p = torch.full((1,1,7,7), .99)
    p[..., :3] = .01
    _, mask, _ = reliable_target([p,p], torch.ones_like(p), erosion=1)
    assert not mask[..., 0, :].any() and not mask[..., -1, :].any()
    assert not mask[..., 2:4].any()  # 两类各自收缩，类别界面被排除
    assert int(mask.sum()) == 15


def test_grid_gt_rejects_unknown_and_mixed_class_cells():
    gt = torch.tensor([[[[0.,0.,1.,1.], [0.,0.,1.,1.],
                         [0.,1.,1.,1.], [0.,0.,1.,-1.]]]])
    assert torch.equal(pure_grid_gt(gt, 2), torch.tensor([[[[0.,1.],[-1.,-1.]]]]))


def test_gt_judgment_distinguishes_useful_and_harmful_teacher():
    t = torch.tensor([[[[.99,.01,.01,.99]]]])
    s = torch.tensor([[[[.01,.99,.01,.01]]]])
    gt = torch.tensor([[[[1.,1.,1.,-1.]]]])
    result = pair_stats(t, s, torch.ones_like(t, dtype=torch.bool), gt)
    assert result['class_disagreement'] == 3
    assert result['gt'] == {'n':3, 'teacher_wrong':2, 'student_wrong':2,
        'teacher_correct_student_wrong':1, 'teacher_wrong_student_correct':1, 'both_wrong':1}


def test_native_votes_keep_strict_tie_and_ignore_background():
    ids = np.array([[0,1,1],[0,2,2]], dtype=np.uint16)
    lookup = np.array([-1,0,1], dtype=np.float32)
    raw = np.array([[1,.5,.5],[0,.9,.9]], dtype=np.float32)
    student = np.array([[1,.6,.6],[0,.2,.2]], dtype=np.float32)
    result = native_gt_votes({'raw':raw, 'student_clean':student}, ids, lookup)
    assert result['counts']['models']['raw']['wrong'] == 0
    assert result['counts']['models']['student_clean']['wrong'] == 2
    assert result['counts']['raw_vs_student']['student_clean']['teacher_correct_student_wrong'] == 2
    assert result['counts']['instances'] == 2
