# -*- coding: utf-8 -*-
"""横截面监督的反例、梯度方向、GT资格与微批合同。"""
import numpy as np
import pytest
import torch

from data.boundary_restore import known_instance_boundary
from utils.boundary_rgb_training import build_tolerant_targets
from utils.boundary_ridge_loss import (
    build_ridge_targets, ridge_supervision_loss, ridge_supervision_settings,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def settings():
    return ridge_supervision_settings(dict(enabled=True, version='normal_hull_v1', weight=.1))


def setup(ids=None, valid=None):
    yy, xx = np.mgrid[:64, :64]
    if ids is None:
        ids = np.where(xx < 32, 1, 2)
    target = torch.from_numpy(known_instance_boundary(ids).astype(np.float32))[None, None]
    valid = torch.ones_like(target, dtype=torch.bool) if valid is None else valid
    prepared = build_tolerant_targets(target, valid)
    before = {k: v.clone() for k, v in prepared.items() if torch.is_tensor(v)}
    ridge = build_ridge_targets(prepared, torch.from_numpy(ids.astype(np.int64))[None, None], settings())
    for k, v in before.items():
        torch.testing.assert_close(prepared[k], v, equal_nan=True, rtol=0, atol=0)
    return prepared, ridge


def profile_target(batch=1):
    shape = (batch, 1, 25, 25)
    eligible = torch.zeros(shape, dtype=torch.bool)
    eligible[..., 12, 12] = True
    return dict(eligible=eligible, normal_index=torch.zeros(shape, dtype=torch.long),
                tolerance=2, profile_radius=4)


def profile_logits(high_columns, high=20., low=-20., batch=1):
    logits = torch.full((batch, 1, 25, 25), low)
    for column in high_columns:
        logits[..., 12, 12 + column] = high
    return logits.requires_grad_(True)


@pytest.mark.parametrize('columns', [[0], [-2], [2], [-2,-1,0,1,2], [], list(range(-4,5))])
def test_single_shifted_or_solid_band_has_exact_zero_loss_and_gradient(columns):
    logits = profile_logits(columns)
    loss, stats = ridge_supervision_loss(logits, profile_target())
    assert float(loss.detach()) == 0 and stats['ridge_gap_points'] == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize('columns,missing', [([-2,2],[-1,0,1]), ([-4,0,4],[-2,-1,1,2])])
def test_deep_valleys_get_strong_upward_gradients_and_peaks_are_not_suppressed(columns,missing):
    logits = profile_logits(columns)
    loss, stats = ridge_supervision_loss(logits, profile_target())
    assert loss > 1 and stats['ridge_gap_points'] == len(missing)
    loss.backward()
    assert bool((logits.grad <= 0).all())
    for delta in missing:
        assert logits.grad[0,0,12,12+delta] < -.19
    for delta in columns:
        assert logits.grad[0,0,12,12+delta] == 0
    assert torch.count_nonzero(logits.grad) == len(missing)


def test_weak_peaks_are_not_pushed_down_and_extreme_logits_stay_finite():
    logits = profile_logits([-2,2], high=float(torch.logit(torch.tensor(.2))), low=-1000.)
    loss, _ = ridge_supervision_loss(logits, profile_target())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0,0,12,12] < -.039
    assert logits.grad[0,0,12,10] == logits.grad[0,0,12,14] == 0


def test_nearly_equal_valleys_cannot_reverse_gradient_from_float_rounding():
    base=torch.linspace(-35,35,1000)
    logits=(base[:,None,None,None]-1).expand(1000,1,25,25).clone()
    logits[:,0,12,10]=base+.00001
    logits[:,0,12,14]=base+.00001
    logits[:,0,12,12]=base
    logits.requires_grad_()
    loss,_=ridge_supervision_loss(logits,profile_target(1000));loss.backward()
    assert torch.isfinite(loss) and bool((logits.grad<=0).all())
    assert not logits.grad[:,0,12,10].any() and not logits.grad[:,0,12,14].any()


@pytest.mark.parametrize('kind', ['vertical','diagonal','thin_ABA','thin_ABC','neighbor5','diagonal_thin','junction'])
def test_GT_only_mask_excludes_close_double_interfaces_and_junctions(kind):
    yy, xx = np.mgrid[:64,:64]
    ids = {
        'vertical':np.where(xx<32,1,2),
        'diagonal':np.where(xx+yy<64,1,2),
        'thin_ABA':np.where((xx>=32)&(xx<34),2,1),
        'thin_ABC':np.where(xx<32,1,np.where(xx<34,2,3)),
        'neighbor5':np.where(xx<32,1,np.where(xx<37,2,3)),
        'diagonal_thin':np.where(xx-yy<0,1,np.where(xx-yy<2,2,3)),
        'junction':np.where(yy<32,3,np.where(xx<32,1,2)),
    }[kind]
    prepared, ridge = setup(ids)
    assert prepared['positive'][0,0,32,32]
    assert not prepared['fallback'][0,0,32,32]
    assert bool(ridge['eligible'][0,0,32,32]) == (kind in ['vertical','diagonal'])


def test_unknown_in_guard_farther_than_prediction_anchors_still_excludes_extra_loss_only():
    valid = torch.ones((1,1,64,64),dtype=torch.bool)
    valid[0,0,32,38] = False
    prepared, ridge = setup(valid=valid)
    assert prepared['positive'][0,0,32,32]
    assert not ridge['eligible'][0,0,32,32]
    assert not bool(ridge['eligible'][...,:8,:].any())
    assert not bool(ridge['eligible'][...,-8:,:].any())


def test_global_denominator_matches_microbatches_with_unequal_eligibility():
    targets = profile_target(3)
    targets['eligible'][1] = False
    logits = profile_logits([-2,2],batch=3)
    loss,_ = ridge_supervision_loss(logits,targets)
    grads = torch.autograd.grad(loss,logits)[0]
    parts=[]
    for i in range(3):
        t={k:v[i:i+1] if torch.is_tensor(v) else v for k,v in targets.items()}
        part,_=ridge_supervision_loss(logits[i:i+1],t,denominator=10)
        parts.append(part)
    torch.testing.assert_close(sum(parts),loss,atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(sum(parts),logits)[0],grads,atol=0,rtol=0)
    with pytest.raises(ValueError):
        ridge_supervision_loss(logits,targets,denominator=9)


def test_disabled_is_explicit_and_settings_do_not_accept_typos():
    assert ridge_supervision_settings() == dict(enabled=False)
    assert ridge_supervision_settings(dict(enabled=False)) == dict(enabled=False)
    for item in [True,dict(enabled=1),dict(enabled=True),dict(enabled=True,version='bad'),
                 dict(enabled=True,version='normal_hull_v1',weight=float('nan')),
                 dict(enabled=True,version='normal_hull_v1',wieght=.1)]:
        with pytest.raises(ValueError):
            ridge_supervision_settings(item)
