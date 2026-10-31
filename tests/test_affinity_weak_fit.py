# -*- coding: utf-8 -*-
"""复制头局部机制检查：合法固定掩码、真实梯度与归一化。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from tools.probe_affinity_weak_fit import (fixed_local_mask, supervision_record,
    parameter_change, loss_options, STEPS, SNAPSHOTS)
from utils.affinity_loss import balanced_affinity_loss


def sample():
    gt = np.ones((15, 17), np.int64)
    gt[:, 9:] = 2
    gt[3:5, 7:11] = 0
    gt[8:11, 11:13] = 3
    valid = np.ones_like(gt, bool)
    valid[0] = False
    return gt, valid


def test_local_only_known_short_pair_and_nearby_positive():
    gt, valid = sample()
    selected, target, legal, band = fixed_local_mask(gt, valid, (1,2), radius=2)
    assert selected.any() and not selected[:,4:].any()
    assert not (selected & ~legal).any()
    assert (selected & (target == 0)).any()
    assert (selected & (target == 1)).any()
    assert not selected[0,:,0].any()
    assert not selected[0,:,3:5,7:11].any()
    assert not selected[0,:,8:11,11:13].any()
    assert band.dtype == bool


def test_mask_not_model_prediction_dependent_and_empty_interface():
    gt, valid = sample()
    left = fixed_local_mask(gt, valid, (1,2), radius=2)[0]
    right = fixed_local_mask(gt.copy(), valid.copy(), (1,2), radius=2)[0]
    assert torch.equal(left,right)
    empty, _, _, band = fixed_local_mask(np.ones((9,9),np.int64), np.ones((9,9),bool),(1,2),2)
    assert not empty.any() and not band.any()


@pytest.mark.parametrize('radius', [0,-1,False,1.5])
def test_bad_mask_contract(radius):
    gt,valid=sample()
    with pytest.raises(ValueError):
        fixed_local_mask(gt,valid,(1,2),radius)


def test_real_copy_update_ignores_unknown_and_changes_only_copy():
    gt,valid=sample()
    selected,target,legal,_=fixed_local_mask(gt,valid,(1,2),2)
    source=torch.nn.Conv2d(3,8,1).eval().requires_grad_(False)
    head=deepcopy(source).eval().requires_grad_(True)
    initial={name:t.detach().clone() for name,t in source.state_dict().items()}
    optimizer=torch.optim.AdamW(head.parameters(),lr=2.e-5,eps=1.e-4)
    logits=head(torch.randn(1,3,*gt.shape))
    logits.retain_grad()
    loss,_=balanced_affinity_loss(logits,target,selected,negative_weight=1.,hard_negative_weight=1.,
                                  hard_negative_gamma=2.,normalize_edge_weights=True)
    loss.backward()
    assert not logits.grad[~selected].count_nonzero()
    assert torch.count_nonzero(head.weight.grad)>0
    optimizer.step()
    assert parameter_change(head,initial)['l2']>0
    assert parameter_change(source,initial)['l2']==0
    assert all(p.grad is None for p in source.parameters())


def test_supervision_receipt_and_original_coefficients():
    gt,valid=sample()
    selected,target,legal,_=fixed_local_mask(gt,valid,(1,2),2)
    record=supervision_record(target,selected)
    assert record['active_channels']==4
    assert record['positive_edges']+record['negative_edges']==int(selected.sum())
    assert sum(r['positive_edges'] for r in record['per_channel'])==record['positive_edges']
    cfg={'direct_semantic_affinity':{'affinity_loss':dict(negative_weight=1.,hard_negative_weight=1.,
        hard_negative_gamma=2.,normalize_edge_weights=True,manual_uncovered_as_boundary=False)}}
    assert loss_options(cfg)['hard_negative_weight']==1.
    cfg['direct_semantic_affinity']['affinity_loss']['manual_uncovered_as_boundary']=True
    with pytest.raises(RuntimeError):
        loss_options(cfg)
    assert STEPS==32 and SNAPSHOTS==(0,16,32)
