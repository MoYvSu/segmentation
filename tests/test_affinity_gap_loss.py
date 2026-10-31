# -*- coding: utf-8 -*-
import numpy as np
import pytest
import torch
import hashlib

from utils.affinity_gap_loss import native_gap_geometry, select_gap_cuts, cut_margin_loss, continuous_gap_loss


def ring_case():
    n = 25
    ids = np.ones((n,n), np.int64); ids[6:19,6:19] = 2
    wall = np.zeros((n,n), bool)
    wall[5:7,5:20] = True; wall[18:20,5:20] = True
    wall[5:20,5:7] = True; wall[5:20,18:20] = True
    p = wall.astype(np.float32)*.8; p[5:7,11:14] = .1
    seeds = np.zeros_like(ids); seeds[11:14,11:14] = 2; seeds[1:3,1:3] = 1
    g = dict(ids=ids, original=np.ones_like(wall), valid=np.ones_like(wall),
             core_ids=seeds.copy(), seed_ids=seeds, allowed=wall)
    return p,g


def test_complete_ring_gap_cut_and_static_seeds():
    p,g = ring_case()
    a = select_gap_cuts(p,g,{'min_atom_pixels':1},seed=1)
    assert a['audit']['selected_pairs'] == 1
    assert a['cut'].any() and not (a['cut'] & ~g['allowed']).any()
    assert not (a['cut'] & (g['seed_ids']>0)).any()
    assert a['audit']['attempts'][-1]['independent_certificate']
    assert a['source_seed'].sum()+a['sink_seed'].sum() == 13


def test_unknown_route_does_not_become_a_wall():
    p,g = ring_case();g['allowed'][5:7,11:14] = False
    a = select_gap_cuts(p,g,{'min_atom_pixels':1})
    assert not a['cut'].any()
    assert a['audit']['reasons']['no_modifiable_complete_separator'] == 1


def test_blocked_static_seed_not_silently_deleted():
    p,g = ring_case();p[1,1] = .9
    a = select_gap_cuts(p,g,{'min_atom_pixels':1})
    assert not a['cut'].any()
    assert a['audit']['reasons']['static_seed_blocked_by_current_wall'] == 1


def test_no_filled_editable_corridor():
    p,g = ring_case();g['original'][5,10] = False
    with pytest.raises(ValueError, match='contract'):
        select_gap_cuts(p,g,{'min_atom_pixels':1})


def test_margin_has_no_gradient_above_target_or_outside_cut():
    x = torch.tensor([[.2,.8],[.6,.9]],requires_grad=True)
    mask = np.array([[1,1],[0,0]],bool)
    loss = cut_margin_loss(x,mask);loss.backward()
    assert x.grad[0,0] < 0 and x.grad[0,1] == 0
    assert torch.count_nonzero(x.grad[1]) == 0


def test_disabled_aux_has_exact_zero_and_no_selection():
    x = torch.full((3,3),.3,requires_grad=True)
    loss,a,cut = continuous_gap_loss(x,{},None,enabled=False)
    loss.backward()
    assert loss.item() == 0 and not cut.any() and torch.count_nonzero(x.grad)==0
    assert a['enabled'] is False


def test_native_core_ignores_fill_and_unknown_and_uses_frozen_reference():
    ids = np.ones((64,64),np.uint16);ids[:,32:] = 2;ids[:2] = 0
    original = ids>0;original[20:23,20:23] = False
    target = np.zeros((64,64),bool);target[:,31:33] = True
    valid = ids>0;ref = target.astype(np.float32)
    a = native_gap_geometry(ids,original,target,valid,ref)
    assert not ((a['seed_ids']>0)&~original).any()
    assert not (a['allowed']&~original).any()
    assert not ((a['seed_ids']>0)&a['allowed']).any()
    assert a['seed_ids'][10,10] == 1 and a['seed_ids'][10,50] == 2


def test_local_frame_cannot_fake_a_complete_separator():
    p = np.zeros((21,21),np.float32);p[:,10]=.8;p[9:12,10]=.1
    ids = np.ones(p.shape,np.int64);ids[:,11:]=2
    seeds=np.zeros_like(ids);seeds[5,5]=1;seeds[5,15]=2
    allowed=np.zeros_like(p,bool);allowed[:,10]=True
    g=dict(ids=ids,original=np.ones_like(allowed),valid=np.ones_like(allowed),
           core_ids=seeds.copy(),seed_ids=seeds,allowed=allowed)
    result=select_gap_cuts(p,g,{'min_atom_pixels':1})
    assert not result['cut'].any()
    assert result['audit']['reasons']['no_modifiable_complete_separator']==1


def test_A_zero_aux_and_C_have_identical_optimizer_update():
    a=torch.nn.Conv2d(1,1,1);b=torch.nn.Conv2d(1,1,1)
    b.load_state_dict(a.state_dict(),strict=True)
    oa=torch.optim.AdamW(a.parameters(),lr=2e-5,eps=1e-4)
    ob=torch.optim.AdamW(b.parameters(),lr=2e-5,eps=1e-4)
    x=torch.linspace(-1,1,64).reshape(1,1,8,8);target=(x>0).float()
    la=torch.nn.functional.binary_cross_entropy_with_logits(a(x),target)
    logits=b(x);lb=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    aux,_,_=continuous_gap_loss(logits.sigmoid()[0,0],{},None,enabled=False)
    la.backward();(lb+0.*aux).backward();oa.step();ob.step()
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


def test_view_loss_changes_only_certified_cut_and_keeps_geometry_unchanged():
    p,g=ring_case();g['scale']=1.
    geometry_copy={k:v.copy() for k,v in g.items() if isinstance(v,np.ndarray)}
    x=torch.from_numpy(p.copy()).requires_grad_()
    loss,a,mask=continuous_gap_loss(x,g,p,{'min_atom_pixels':1},seed=1)
    assert a['selected_pairs']==1 and mask.any()
    loss.backward()
    assert torch.count_nonzero(x.grad[torch.from_numpy(~mask)])==0
    assert all(np.array_equal(g[k],v) for k,v in geometry_copy.items())


def continuous_policy_fixture():
    ids=np.ones((64,64),np.uint16);ids[:,32:]=2;ids[:2]=0
    original=ids>0;original[20:23,20:23]=False
    valid=ids>0
    target=np.zeros(ids.shape,bool);target[:,31:33]=True
    reference=target.astype(np.float32);reference[5:55,8:20]=.8
    return ids,original,target,valid,reference


def test_default_legacy_native_arrays_match_preserved_v1_fixture_hashes():
    # Captured from unchanged v1 server snapshot SHA 8e915ee97d3a... .
    expected={
        'ids':'22a41bd07681a9d3c97502284123c07e8e0cd62ecb151b805e063b4bb94ab755',
        'original':'1e3a828e7d1372ac4ffcd462dd0e1c3fda929cd92902b91d7c04dde972f805c5',
        'valid':'f4a44536113117cd2617de3fbf54eda2346efd8d66fa41dde4df01445e37b701',
        'core_ids':'62e75a00a4a3cc6e18f94a4244fa29ed9579ebceb515d4caaa4ef2154316dc08',
        'seed_ids':'6fb9275ea034171f1bfcafa2ad57fecb11e23dbc3724354bfb9105684d2167b1',
        'allowed':'5e18b274a703e439c779ec8080847b9bc716f4acbf0d632b991ee47c7f43f8bf'}
    args=continuous_policy_fixture()
    implicit=native_gap_geometry(*args)
    explicit=native_gap_geometry(*args,{'seed_policy':'legacy_skeleton'})
    for key,digest in expected.items():
        a=np.ascontiguousarray(implicit[key]);h=hashlib.sha256()
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        assert h.hexdigest()==digest
        assert np.array_equal(implicit[key],explicit[key])


def test_continuous_low_keeps_original_core_corridor_and_ignores_fill_unknown():
    args=continuous_policy_fixture();ids,original,target,valid,ref=args
    legacy=native_gap_geometry(*args)
    new=native_gap_geometry(*args,{'seed_policy':'continuous_low'})
    for key in ['ids','original','valid','core_ids','allowed']:
        assert np.array_equal(legacy[key],new[key])
    selected=new['seed_ids']>0
    assert selected.any() and not np.array_equal(legacy['seed_ids'],new['seed_ids'])
    assert np.all(ref[selected]<=.45)
    assert not np.any(selected&~original) and not np.any(selected&~valid)
    assert not np.any(selected&(ids==0)) and not np.any(selected&new['allowed'])
    assert new['seed_policy']=='continuous_low' and new['seed_low_threshold']==.45


def test_continuous_low_includes_exact_cutoff_but_excludes_next_float():
    ids,original,target,valid,ref=continuous_policy_fixture()
    ref[10,10]=np.float32(.45)
    ref[12,10]=np.nextafter(np.float32(.45),np.float32(1.))
    new=native_gap_geometry(ids,original,target,valid,ref,{'seed_policy':'continuous_low'})
    assert new['seed_ids'][10,10]==1
    assert new['seed_ids'][12,10]==0


def test_continuous_static_seeds_are_not_trimmed_when_current_output_blocks_them():
    reference,g=ring_case()
    g.update(seed_policy='continuous_low',seed_low_threshold=.45)
    static=g['seed_ids'].copy();current=reference.copy();current[1,1]=.9
    a=select_gap_cuts(current,g,{'seed_policy':'continuous_low','min_atom_pixels':1})
    assert not a['cut'].any() and np.array_equal(g['seed_ids'],static)
    assert a['audit']['reasons']['static_seed_blocked_by_current_wall']==1
    assert a['audit']['seed_policy']=='continuous_low'
    assert a['audit']['seed_low_threshold']==.45 and a['audit']['seed_low_threshold_used']
    assert a['audit']['static_seed_metadata_verified']


def test_policy_mismatch_is_rejected_instead_of_mislabeled_audit():
    p,g=ring_case();g.update(seed_policy='continuous_low',seed_low_threshold=.45)
    with pytest.raises(ValueError,match='policy metadata'):
        select_gap_cuts(p,g,{'min_atom_pixels':1})
    with pytest.raises(ValueError,match='policy metadata'):
        select_gap_cuts(p,g,{'seed_policy':'continuous_low','seed_low_threshold':.4,'min_atom_pixels':1})
    with pytest.raises(ValueError,match='seed_policy'):
        native_gap_geometry(*continuous_policy_fixture(),{'seed_policy':'current_prediction'})
