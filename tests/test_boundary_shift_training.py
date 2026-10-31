# -*- coding: utf-8 -*-
"""顺序对称重算与完整双图反传、microbatch及关闭兼容性。"""
from copy import deepcopy

import pytest
import torch
from torch import nn

from models.boundary_naf import BoundaryDiscriminator
from utils.boundary_rgb_training import build_tolerant_targets, rgb_gan_update, tolerant_boundary_loss
from utils.boundary_shift_consistency import (
    build_shift_masks, shift_consistency_loss, shift_consistency_settings,
    shift_for_step, shift_integer_reflect,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cfg():return shift_consistency_settings(dict(enabled=True,weight=.1,margin=8,band_radius=4))


class PhaseGenerator(nn.Module):
    """故意保留格点偏好，让一致性存在非零参数梯度。"""
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.7))
        self.bias=nn.Parameter(torch.tensor(-.2))
        self.grid=nn.Parameter(torch.tensor([[.4,-.3],[-.1,.2]]))
    def forward_logits(self,x):
        h,w=x.shape[-2:]
        return self.weight*x+self.bias+self.grid.repeat((h+1)//2,(w+1)//2)[:h,:w][None,None]
    def forward(self,x):return self.forward_logits(x).sigmoid()


def batch(n=3,size=80):
    target=torch.zeros(n,1,size,size);target[...,10:size-10,size//2]=1
    valid=torch.ones_like(target,dtype=torch.bool)
    valid[0,:,25:30,15:20]=False
    target[~valid]=float('nan')
    x=torch.rand(n,1,size,size,generator=torch.Generator().manual_seed(74))
    return dict(input=x,target=target,valid=valid)


def models():return PhaseGenerator(),BoundaryDiscriminator(width=2,seed=7)
def state(m):return {k:v.detach().clone() for k,v in m.state_dict().items()}


def test_sequential_two_branch_gradient_matches_full_symmetric_objective_and_counts_value_once():
    data=batch();settings=cfg();step=6
    g,d=models();reference=deepcopy(g)
    opt=torch.optim.SGD(g.parameters(),lr=.03)
    dop=torch.optim.SGD(d.parameters(),lr=.03)
    ropt=torch.optim.SGD(reference.parameters(),lr=.03)
    prepared=build_tolerant_targets(data['target'],data['valid'])
    shift=shift_for_step(settings,step);masks=build_shift_masks(prepared,settings,*shift)
    logits=reference.forward_logits(data['input'])
    moved=reference(shift_integer_reflect(data['input'],*shift))
    aligned=shift_integer_reflect(moved,-shift[0],-shift[1])
    base,_=tolerant_boundary_loss(logits,data['target'],data['valid'],prepared=prepared)
    regularizer,_=shift_consistency_loss(logits.sigmoid(),aligned,masks)
    objective=base+settings['weight']*regularizer
    objective.backward();ropt.step()
    stats=rgb_gan_update(g,d,opt,dop,data,micro_batch_size=3,adversarial_weight=0,grad_clip=None,
                         shift_consistency=settings,consistency_step=step)
    assert stats['shift_consistency_loss']==pytest.approx(float(regularizer.detach()),rel=1e-6)
    assert stats['loss_g']==pytest.approx(float(objective.detach()),rel=1e-6)
    assert stats['shift_second_branch_probe_grad_norm']>0
    assert stats['shift_recompute_max_error']==0
    for a,b in zip(g.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)


def test_global_region_denominators_keep_microbatch_and_gan_updates_equivalent():
    g1,d1=models();g2,d2=models();data=batch();settings=cfg()
    go1,do1=torch.optim.SGD(g1.parameters(),lr=.01),torch.optim.SGD(d1.parameters(),lr=.01)
    go2,do2=torch.optim.SGD(g2.parameters(),lr=.01),torch.optim.SGD(d2.parameters(),lr=.01)
    a=rgb_gan_update(g1,d1,go1,do1,data,micro_batch_size=3,grad_clip=None,shift_consistency=settings,consistency_step=19)
    b=rgb_gan_update(g2,d2,go2,do2,data,micro_batch_size=1,grad_clip=None,shift_consistency=settings,consistency_step=19)
    for key in ['loss_g','coverage','interior','adv_g','loss_d','grad_norm_g','shift_consistency_loss','shift_band_loss','shift_interior_loss']:
        assert a[key]==pytest.approx(b[key],rel=1e-5,abs=1e-6)
    for m1,m2 in [(g1,g2),(d1,d2)]:
        for a,b in zip(m1.parameters(),m2.parameters()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


def test_disabled_path_is_exact_and_adds_no_receipt_fields():
    g1,d1=models();g2,d2=models();data=batch(1)
    go1,do1=torch.optim.Adam(g1.parameters()),torch.optim.Adam(d1.parameters())
    go2,do2=torch.optim.Adam(g2.parameters()),torch.optim.Adam(d2.parameters())
    a=rgb_gan_update(g1,d1,go1,do1,data)
    b=rgb_gan_update(g2,d2,go2,do2,data,shift_consistency={'enabled':False},consistency_step=78)
    assert a==b and not any(k.startswith('shift_') for k in a)
    for m1,m2 in [(g1,g2),(d1,d2)]:
        assert all(torch.equal(v,state(m2)[k]) for k,v in state(m1).items())


def test_unknown_skips_every_parameter_buffer_and_optimizer_even_when_enabled():
    g,d=models();data=batch(1);data['valid'].zero_();data['target'].fill_(float('nan'))
    before_g,before_d=state(g),state(d)
    go,do=torch.optim.Adam(g.parameters()),torch.optim.Adam(d.parameters())
    stats=rgb_gan_update(g,d,go,do,data,shift_consistency=cfg(),consistency_step=2)
    assert not stats['generator_updated'] and not stats['discriminator_updated']
    assert not stats['shift_consistency_active'] and not go.state and not do.state
    assert all(torch.equal(v,state(g)[k]) for k,v in before_g.items())
    assert all(torch.equal(v,state(d)[k]) for k,v in before_d.items())


def test_settings_step_and_single_channel_contract_fail_before_updates():
    g,d=models();data=batch(1);go,do=torch.optim.Adam(g.parameters()),torch.optim.Adam(d.parameters())
    with pytest.raises(ValueError,match='step'):rgb_gan_update(g,d,go,do,data,shift_consistency=cfg())
    data['input']=data['input'].repeat(1,4,1,1)
    with pytest.raises(ValueError,match='single-channel'):
        rgb_gan_update(g,d,go,do,data,shift_consistency=cfg(),consistency_step=0)
    assert not go.state and not do.state


def test_stateless_step_schedule_survives_checkpoint_optimizer_resume():
    data=batch(1);g,d=models();go,do=torch.optim.Adam(g.parameters()),torch.optim.Adam(d.parameters())
    rgb_gan_update(g,d,go,do,data,shift_consistency=cfg(),consistency_step=0,adversarial_weight=0)
    saved=deepcopy((g.state_dict(),d.state_dict(),go.state_dict(),do.state_dict()))
    g2,d2=models();go2,do2=torch.optim.Adam(g2.parameters()),torch.optim.Adam(d2.parameters())
    for owner,value in zip((g2,d2,go2,do2),saved):owner.load_state_dict(value)
    a=rgb_gan_update(g,d,go,do,data,shift_consistency=cfg(),consistency_step=1,adversarial_weight=0)
    b=rgb_gan_update(g2,d2,go2,do2,data,shift_consistency=cfg(),consistency_step=1,adversarial_weight=0)
    assert a==b
    assert all(torch.equal(v,state(g2)[k]) for k,v in state(g).items())
