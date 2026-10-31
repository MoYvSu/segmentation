# -*- coding: utf-8 -*-
"""带外可信关系保留：严格GT选择、稳定KL与真实梯度。"""
import math

import pytest
import torch

from utils.affinity_loss import build_affinity_targets_torch
from utils.affinity_retain_loss import build_reliable_retention_mask, balanced_affinity_retention_kl


def sample():
    gt=torch.ones((1,9,11),dtype=torch.long)
    gt[:,:,6:]=2
    gt[:,2:4,4:7]=0
    gt[:,6:8,7:9]=3
    valid=torch.ones((1,1,9,11),dtype=torch.bool)
    valid[:,:,-1]=False
    targets,legal=build_affinity_targets_torch(gt,valid)
    teacher=torch.where(targets>.5,torch.tensor(5.),torch.tensor(-5.))
    return gt,valid,targets,legal,teacher


def test_both_endpoint_band_exclusion_short_and_long():
    gt,valid,target,legal,teacher=sample()
    band=torch.zeros((9,11),dtype=torch.bool); band[4,4]=True
    mask,counts=build_reliable_retention_mask(teacher,target,legal,band,
        instance_map=gt,valid_content=valid)
    assert not (mask&~legal).any()
    assert not mask[0,0,4,3] and not mask[0,0,4,4]
    assert not mask[0,6,4,0] and not mask[0,6,4,4]
    assert mask[0,6,4,1] and mask[0,0,4,1]
    assert counts['excluded_band_endpoint_edges']>0
    assert counts['full_path_checked']
    assert counts['active_channels']==8


def test_incorrect_or_uncertain_teacher_excluded_without_answer_imputation():
    gt,valid,target,legal,teacher=sample()
    teacher=teacher.clone()
    teacher[0,0,0,0]=-7.  # GT same-instance, teacher wrong.
    teacher[0,0,0,5]=7.  # GT different-instance, teacher wrong.
    teacher[0,1,0,1]=0.  # Teacher uncertain.
    mask,counts=build_reliable_retention_mask(teacher,target,legal,torch.zeros((9,11),dtype=torch.bool),
        instance_map=gt,valid_content=valid)
    assert not mask[0,0,0,0] and not mask[0,0,0,5] and not mask[0,1,0,1]
    assert not mask[0,:,2:4,4:7].any()
    assert not mask[0,:,-1].any()
    assert counts['selected_edges']==int(mask.sum())


@pytest.mark.parametrize('obstruction',[0,3])
def test_long_relation_path_unknown_and_third_instance_excluded(obstruction):
    gt=torch.ones((1,7,7),dtype=torch.long)
    gt[0,1,2]=obstruction
    valid=torch.ones((1,1,7,7),dtype=torch.bool)
    target,legal=build_affinity_targets_torch(gt,valid)
    teacher=torch.where(target>.5,torch.tensor(9.),torch.tensor(-9.))
    mask,counts=build_reliable_retention_mask(teacher,target,legal,torch.zeros((7,7),dtype=torch.bool),
        instance_map=gt,valid_content=valid)
    assert legal[0,6,1,0]  # Known same-instance endpoints do not alone prove path reliability.
    assert not mask[0,6,1,0]
    assert counts['excluded_unknown_or_third_instance_path_edges']>0
    loose,loose_counts=build_reliable_retention_mask(teacher,target,legal,torch.zeros((7,7),dtype=torch.bool))
    assert loose[0,6,1,0] and not loose_counts['full_path_checked']


def test_full_path_may_transition_between_its_two_endpoint_gt_ids():
    gt=torch.tensor([[[1,1,1,2,2,2,2]]],dtype=torch.long)
    valid=torch.ones((1,1,1,7),dtype=torch.bool)
    target,legal=build_affinity_targets_torch(gt,valid)
    teacher=torch.where(target>.5,torch.tensor(9.),torch.tensor(-9.))
    mask,_=build_reliable_retention_mask(teacher,target,legal,torch.zeros((1,7),dtype=torch.bool),
        instance_map=gt,valid_content=valid)
    assert legal[0,6,0,0] and target[0,6,0,0]==0 and mask[0,6,0,0]


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_equal_initial_teacher_exact_zero_value_gradient_including_saturation(dtype):
    values=torch.tensor([-1000.,-100.,-20.,-5.,5.,20.,100.,1000.],dtype=dtype).reshape(1,8,1,1)
    teacher=values.clone().requires_grad_(True)
    student=values.clone().requires_grad_(True)
    target=(values>0).to(dtype)
    mask=torch.ones_like(target,dtype=torch.bool)
    loss,counts=balanced_affinity_retention_kl(student,teacher,target,mask)
    assert loss.item()==0. and counts['loss']==0.
    loss.backward()
    assert torch.count_nonzero(student.grad)==0
    assert teacher.grad is None


def test_extreme_opposite_student_is_finite_and_teacher_has_no_gradient():
    teacher=torch.tensor([1000.,-1000.]*4).reshape(1,8,1,1).requires_grad_(True)
    student=(-teacher.detach()).requires_grad_(True)
    target=(teacher.detach()>0).float()
    loss,counts=balanced_affinity_retention_kl(student,teacher,target,torch.ones_like(target,dtype=torch.bool))
    assert torch.isfinite(loss) and loss.item()==1000.
    loss.backward()
    assert torch.isfinite(student.grad).all() and teacher.grad is None
    assert counts['direction']=='KL(teacher||student)'


def test_mask_outside_gradient_exact_zero_and_empty_mask_backpropagates_zero():
    teacher=torch.randn((1,8,3,4),requires_grad=True)
    student=torch.randn((1,8,3,4),requires_grad=True)
    target=torch.randint(2,student.shape).float()
    mask=torch.zeros_like(target,dtype=torch.bool); mask[:,[0,4],1,1]=True
    loss,_=balanced_affinity_retention_kl(student,teacher,target,mask)
    loss.backward()
    assert not student.grad[~mask].count_nonzero()
    assert student.grad[mask].count_nonzero() and teacher.grad is None
    empty=student.detach().clone().requires_grad_(True)
    loss,counts=balanced_affinity_retention_kl(empty,teacher,target,torch.zeros_like(mask))
    assert loss.item()==0. and counts['active_channels']==0
    loss.backward()
    assert not empty.grad.count_nonzero()


def test_per_class_then_channel_normalization_not_pixel_mean():
    teacher=torch.full((1,8,1,5),math.log(9.),dtype=torch.float64)
    student=torch.zeros_like(teacher)
    target=torch.ones_like(teacher)
    mask=torch.zeros_like(teacher,dtype=torch.bool)
    mask[0,0,0]=True
    target[0,0,0,4]=0
    student[0,0,0,4]=-2.
    mask[0,7,0,0]=True
    student[0,7,0,0]=1.
    loss,counts=balanced_affinity_retention_kl(student,teacher,target,mask)
    p=teacher.sigmoid()
    entropy=torch.nn.functional.binary_cross_entropy_with_logits(teacher,p,reduction='none')
    raw=torch.nn.functional.binary_cross_entropy_with_logits(student,p,reduction='none')-entropy
    expected=((raw[0,0,0,:4].mean()+raw[0,0,0,4])/2+raw[0,7,0,0])/2
    assert torch.equal(loss,expected)
    assert not torch.isclose(loss,raw[mask].mean())
    assert counts['active_channels']==2
    assert counts['per_channel'][0]['class_terms']==2
    assert counts['per_channel'][7]['class_terms']==1


def test_retention_kl_gradcheck_for_nonzero_student_difference():
    teacher=torch.randn((1,8,1,2),dtype=torch.float64)
    student=(teacher+torch.tensor([.2,-.3],dtype=torch.float64)).requires_grad_(True)
    target=torch.randint(2,student.shape).double()
    mask=torch.ones_like(target,dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda value:balanced_affinity_retention_kl(value,teacher,target,mask)[0],
                                   (student,),eps=1.e-6,atol=1.e-5)


def test_contradictory_gt_target_and_incomplete_path_contract_rejected():
    gt,valid,target,legal,teacher=sample()
    band=torch.zeros((9,11),dtype=torch.bool)
    with pytest.raises(ValueError):
        build_reliable_retention_mask(teacher,target,legal,band,instance_map=gt)
    wrong=target.clone(); wrong[0,0,0,0]=0.
    with pytest.raises(ValueError):
        build_reliable_retention_mask(teacher,wrong,legal,band,instance_map=gt,valid_content=valid)
    with pytest.raises(ValueError):
        build_reliable_retention_mask(teacher,target,legal,band,confidence=.5)
