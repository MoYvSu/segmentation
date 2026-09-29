# -*- coding: utf-8 -*-
"""连续界面选位的零缝、unknown保护与双向完整部署效果。"""
import numpy as np
import torch

from tools.probe_affinity_chain import trusted_targets
from utils.affinity_interface import interface_loss
from train_affinity_connectivity import auxiliary_loss, common_config, compare_draw
from utils.affinity_connectivity import deployment_connectivity_loss
import pytest


def sample(gt, pred, trust=None):
    trust=np.ones_like(gt,bool) if trust is None else trust
    target,valid=trusted_targets(gt,trust,np.ones_like(trust))
    logits=torch.zeros_like(target,requires_grad=True)
    loss,stats=interface_loss(logits,target,valid,torch.tensor(pred)[None],torch.tensor(gt)[None],
        trusted_pixels=torch.tensor(trust)[None],band_radius=2)
    grad=torch.autograd.grad(loss,logits)[0]
    return target,valid,grad,stats


def test_zero_prediction_seam_is_supervised_as_same_gt_and_entire_interface():
    gt=np.ones((30,40),np.int64);pred=gt.copy();pred[:,20:]=2;pred[:,19:21]=0
    target,valid,grad,stats=sample(gt,pred)
    assert stats['positive']['selected_edges']>16
    assert stats['positive']['selected_zero_endpoint_edges']>0
    assert (grad[grad!=0]<0).all()
    assert (grad[:,4:]==0).all() and not (grad[~valid]!=0).any()
    # 监督需沿整条接触带分布，不能只选其中16格。
    assert (grad[0,:4,:,18:22].abs().sum((0,2))>0).all()


def test_merged_gt_contact_has_negative_gradient_and_keeps_positive_group():
    gt=np.ones((30,40),np.int64);gt[:,20:]=2
    pred=np.ones_like(gt);pred[15:,:15]=3
    target,valid,grad,stats=sample(gt,pred)
    assert stats['directions_present']==2
    assert (grad[(grad!=0)&(target==0)]>0).all()
    assert (grad[(grad!=0)&(target==1)]<0).all()


def test_unknown_hole_is_not_filled_by_nearest_prediction_ownership():
    gt=np.ones((30,40),np.int64);pred=gt.copy();pred[:,20:]=2;pred[:,19:21]=0
    trust=np.ones_like(gt,bool);trust[8:22,15:25]=False
    _,valid,grad,stats=sample(gt,pred,trust)
    assert not (grad[~valid]!=0).any()
    assert not grad[0,:,10:20,17:23].any()
    assert stats['positive']['selected_edges']>0


def test_empty_partition_gives_finite_zero_gradient():
    gt=np.ones((12,16),np.int64)
    _,_,grad,stats=sample(gt,np.zeros_like(gt))
    assert torch.isfinite(grad).all() and not grad.any()
    assert stats['selected_edges']==0


def test_correct_partition_does_not_select_gt_contact_as_error():
    gt=np.ones((20,30),np.int64);gt[:,15:]=2
    _,_,grad,stats=sample(gt,gt.copy())
    assert stats['selected_edges']==0 and not grad.any()


def test_existing_configuration_keeps_exact_old_loss_and_gradient():
    gt=torch.ones(1,20,30,dtype=torch.long);gt[:,:,15:]=2
    pred=torch.ones_like(gt);pred[:,10:,:10]=3
    target,valid=trusted_targets(gt[0].numpy(),np.ones((20,30),bool),np.ones((20,30),bool))
    logits=torch.zeros_like(target,requires_grad=True)
    trusted=torch.ones_like(gt,dtype=torch.bool)
    options=dict(max_groups=8,edges_per_group=16,minimum_group_edges=2,negative_margin=.3,positive_margin=.7)
    a,sa=deployment_connectivity_loss(logits,target,valid,pred,gt,trusted_pixels=trusted,**options)
    b,sb=auxiliary_loss({'affinity_connectivity':options},logits,target,valid,pred,gt,trusted)
    assert sa==sb and torch.equal(a,b)
    assert torch.equal(torch.autograd.grad(a,logits,retain_graph=True)[0],torch.autograd.grad(b,logits)[0])


def test_reuse_does_not_ignore_shared_training_or_postprocessing_changes():
    a={'backend_adaptation':{'output_dir':'a','epochs':20},'inference':{'threshold':.65}}
    b={'backend_adaptation':{'output_dir':'b','epochs':20},'inference':{'threshold':.65},'affinity_interface':{'enabled':True}}
    assert common_config(a)==common_config(b)
    b['inference']['threshold']=.6
    assert common_config(a)!=common_config(b)
    assert a['backend_adaptation']['output_dir']=='a'


def test_control_step_receipt_difference_is_fatal():
    row=dict(epoch=1,step=1,seed=1,manual=['a'],pseudo=['b'],manual_draw={'source':1},pseudo_draw={'source':2})
    compare_draw(row,row.copy())
    with pytest.raises(RuntimeError,match='manual_draw'):
        compare_draw(row,{**row,'manual_draw':{'source':3}})
