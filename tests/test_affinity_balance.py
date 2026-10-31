# -*- coding: utf-8 -*-
"""损失方向、未知域和单变量配置约束。"""
from copy import deepcopy
import pytest
import torch
from train_affinity_balance import reference_config,comparable
from utils.affinity_loss import balanced_affinity_loss
from utils.config import load_config


@pytest.mark.parametrize('kind,base',[('native','affinity_native'),('mixed','affinity_mixed')])
def test_only_authorized_weight_changes(kind,base):
    cfg=load_config('config/train/affinity_balance_'+kind+'.yaml')
    assert comparable(reference_config(cfg))==comparable(load_config('config/train/'+base+'.yaml'))
    assert cfg['backend_adaptation']['epochs']==20
    wrong=deepcopy(cfg);wrong['direct_semantic_affinity']['affinity_loss']['negative_weight']=2.
    with pytest.raises(ValueError):reference_config(wrong)


def test_internal_disconnect_gradient_increases_unknown_stays_ignored():
    target=torch.tensor([[[[1.,0.,1.]]]]);valid=torch.tensor([[[[True,True,False]]]])
    gradients=[]
    for weight in (1.5,1.):
        logits=torch.tensor([[[[-2.,2.,-2.]]]],requires_grad=True)
        loss,_=balanced_affinity_loss(logits,target,valid,negative_weight=weight)
        loss.backward();gradients.append(logits.grad.clone())
    assert gradients[0][0,0,0,0]<0 and gradients[1][0,0,0,0]<0
    assert torch.allclose(gradients[1][...,0],gradients[0][...,0]*1.25)
    assert torch.allclose(gradients[1][...,1],gradients[0][...,1]*(5/6))
    assert gradients[0][...,2].item()==gradients[1][...,2].item()==0
