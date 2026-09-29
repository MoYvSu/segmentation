# -*- coding: utf-8 -*-
"""验证预算测量位置与真实AdamW步骤，避免短测量错训练力度。"""
import copy
import pytest
import torch
from utils.gradient_budget import parameter_weight, adamw_delta


def test_parameter_budget_reaches_target_in_nonuniform_jacobian():
    p=torch.tensor([.3,.4],requires_grad=True)
    logits=p*torch.tensor([.01,100.])
    base=logits[1].square();aux=logits[0].square()*100000
    a=torch.autograd.grad(base,p,retain_graph=True);b=torch.autograd.grad(aux,p)
    weight,stats=parameter_weight(a,b,requested=1000,target_ratio=.15)
    assert weight>0 and stats['weighted_ratio']==pytest.approx(.15)


@pytest.mark.parametrize('base,aux',[(0.,1.),(1.,0.),(0.,0.)])
def test_zero_gradients_do_not_create_budget(base,aux):
    weight,stats=parameter_weight([torch.tensor(base)],[torch.tensor(aux)],requested=.1,target_ratio=.15)
    assert weight==0 and stats['weighted_ratio']==0


def test_nominal_coefficient_is_still_an_upper_bound():
    w,s=parameter_weight([torch.tensor(100.)],[torch.tensor(1.)],requested=.1,target_ratio=.15)
    assert w==.1 and s['weighted_ratio']==pytest.approx(.001)


@pytest.mark.parametrize('history',[0,3])
def test_adamw_counterfactual_matches_real_step_and_does_not_mutate(history):
    torch.manual_seed(5)
    p=torch.nn.Parameter(torch.randn(4,6))
    opt=torch.optim.AdamW([p],lr=.002,weight_decay=.03,eps=1e-4)
    for _ in range(history):
        p.grad=torch.randn_like(p);opt.step()
    state=copy.deepcopy(opt.state_dict());before=p.detach().clone()
    grads=[torch.randn_like(p)*4]
    delta,audit=adamw_delta(opt,grads,.3)
    assert torch.equal(before,p) and audit['clip_coefficient']<1
    for key,values in state['state'].items():
        for name,value in values.items():assert torch.equal(value,opt.state_dict()['state'][key][name])
    p.grad=grads[0].clone();torch.nn.utils.clip_grad_norm_([p],.3);opt.step()
    assert torch.allclose(p-before,delta[0],atol=2e-7,rtol=1e-4)


def test_nonfinite_gradients_are_rejected():
    with pytest.raises(FloatingPointError):
        parameter_weight([torch.tensor(float('nan'))],[torch.tensor(1.)],requested=.1,target_ratio=.15)
