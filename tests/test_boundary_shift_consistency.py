# -*- coding: utf-8 -*-
"""平移几何、未知域、独立随机流和一致性梯度的行为验证。"""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from torch import nn

import utils.boundary_shift_consistency as shift_module
from utils.boundary_rgb_training import build_tolerant_targets, tolerant_boundary_loss
from utils.boundary_shift_consistency import (
    build_shift_masks, shift_consistency_loss, shift_consistency_settings,
    shift_for_step, shift_integer_reflect, translation_probe,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def settings(**kwargs):
    return shift_consistency_settings(dict(enabled=True, margin=4, band_radius=3, **kwargs))


def sample(count=2, size=40):
    target = torch.zeros(count, 1, size, size)
    target[..., 6:size-6, size//2] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, :, 12:16, 12:16] = False
    target[~valid] = float('nan')
    return target, valid


@pytest.mark.parametrize('dy,dx', [(1,0),(-1,0),(0,1),(0,-1),(3,-2)])
def test_translation_has_correct_orientation_no_wrap_and_exact_inverse_in_common_interior(dy,dx):
    value = torch.arange(7*9, dtype=torch.float32).reshape(1,1,7,9)
    shifted = shift_integer_reflect(value,dy,dx)
    assert shifted.shape == value.shape
    for y in range(max(0,dy), min(7,7+dy)):
        for x in range(max(0,dx), min(9,9+dx)):
            assert shifted[0,0,y,x] == value[0,0,y-dy,x-dx]
    inverse = shift_integer_reflect(shifted,-dy,-dx)
    torch.testing.assert_close(inverse[...,3:4,3:6], value[...,3:4,3:6], rtol=0, atol=0)
    if (dy,dx)==(0,1):
        assert shifted[0,0,0,0] == value[0,0,0,1]
        assert shifted[0,0,0,0] != value[0,0,0,-1]


def test_shift_stream_is_stateless_replayable_and_does_not_consume_any_training_rng():
    cfg = settings()
    py = deepcopy(random.getstate()); npstate = deepcopy(np.random.get_state()); th = torch.get_rng_state().clone()
    sequence = [shift_for_step(cfg, step) for step in range(64)]
    assert set(sequence) == {(1,0),(-1,0),(0,1),(0,-1)}
    assert sequence[16:32] == [shift_for_step(cfg, step) for step in range(16,32)]
    assert py == random.getstate()
    now = np.random.get_state()
    assert npstate[0] == now[0] and np.array_equal(npstate[1],now[1]) and npstate[2:] == now[2:]
    assert torch.equal(th,torch.get_rng_state())


def test_common_masks_exclude_unknown_frame_and_have_disjoint_regions():
    target,valid=sample()
    prepared=build_tolerant_targets(target,valid)
    masks=build_shift_masks(prepared,settings(),1,0)
    assert not bool((masks['band'] & masks['interior']).any())
    assert int(masks['band'].sum())>0 and int(masks['interior'].sum())>0
    for m in [masks['band'],masks['interior']]:
        assert not bool((m & ~prepared['trusted']).any())
        assert not bool(m[...,:4,:].any()) and not bool(m[...,-5:,:].any())
        assert not bool(m[...,:,:4].any()) and not bool(m[...,:,-4:].any())
    changed=target.clone();changed[~valid]=997
    other=build_shift_masks(build_tolerant_targets(changed,valid),settings(),1,0)
    assert all(torch.equal(masks[k],other[k]) for k in masks)


def test_region_balancing_global_denominators_and_unknown_gradients():
    target,valid=sample(); cfg=settings()
    masks=build_shift_masks(build_tolerant_targets(target,valid),cfg,0,1)
    original=torch.full_like(target,.2,requires_grad=True)
    shifted=torch.full_like(target,.8,requires_grad=True)
    loss,a=shift_consistency_loss(original,shifted,masks)
    assert float(loss.detach())==pytest.approx(.6)
    loss.backward()
    for p in [original,shifted]:
        assert torch.count_nonzero(p.grad[~(masks['band']|masks['interior'])])==0
    no_band=dict(masks,band=torch.zeros_like(masks['band']))
    half,_=shift_consistency_loss(original,shifted,no_band)
    assert float(half.detach())==pytest.approx(.3)
    denominators={k:int(masks[k].sum()) for k in ['band','interior']}
    parts=[]
    for n in range(2):
        val,_=shift_consistency_loss(original[n:n+1],shifted[n:n+1],{k:v[n:n+1] for k,v in masks.items()},denominators=denominators)
        parts.append(val)
    torch.testing.assert_close(sum(parts),loss,atol=1e-7,rtol=1e-6)


def test_constant_stability_is_not_a_reconstruction_success():
    target,valid=sample(); masks=build_shift_masks(build_tolerant_targets(target,valid),settings(),0,1)
    constant=torch.full_like(target,.01)
    consistency,_=shift_consistency_loss(constant,constant,masks)
    reconstruction,a=tolerant_boundary_loss(torch.logit(constant),target,valid)
    assert float(consistency)==0 and float(reconstruction)>4
    assert a['covered_gt_points']==0


class LinearConvolution(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(1,1,3,padding=1)
    def forward(self,x):return self.conv(x).sigmoid()


def test_translation_probe_uses_inverse_alignment_same_coordinates_and_restores_mode(monkeypatch):
    model=LinearConvolution().train();target,valid=sample(count=1)
    x=torch.nan_to_num(target)
    selected=[]
    actual_loss=shift_module.shift_consistency_loss
    def record_masks(a,b,masks,**kwargs):
        selected.append({k:v.clone() for k,v in masks.items()})
        return actual_loss(a,b,masks,**kwargs)
    monkeypatch.setattr(shift_module,'shift_consistency_loss',record_masks)
    result=translation_probe(model,x,target,valid,settings=settings(),shifts=((1,0),(0,-1),(8,0),(0,8)))
    assert model.training and result['complete']
    assert all(r['band_mae']<1e-7 and r['interior_mae']<1e-7 for r in result['rows'])
    assert len({r['band_pixels'] for r in result['rows']})==1
    assert len({r['interior_pixels'] for r in result['rows']})==1
    assert result['common_coordinates_for_all_shifts'] and result['probe_common_margin']==12
    assert len(selected)==4
    for masks in selected:
        assert all(torch.equal(masks[k],selected[0][k]) for k in masks)
        assert not bool(masks['inner'][...,:12,:].any())
        assert not bool(masks['inner'][...,-12:,:].any())


@pytest.mark.parametrize('bad',[
    {'enabled':1}, {'enabled':True,'weight':0}, {'enabled':True,'weight':float('nan')},
    {'enabled':True,'margin':0}, {'enabled':True,'band_radius':True},
    {'enabled':True,'version':'unknown'}, {'enabled':True,'typo':2},
])
def test_invalid_or_ambiguous_settings_are_rejected(bad):
    with pytest.raises(ValueError):shift_consistency_settings(bad)
