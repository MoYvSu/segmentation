# -*- coding: utf-8 -*-
"""真实balanced BCE与小型界面反例验证；无比赛产物或GPU依赖。"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.probe_affinity_signal import (
    LOSS_CFG, analyze_signal, case_tables, contact_lengths, element_contributions,
    load_functions, native_adjacency_coverage, safe_child,
)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def functions():
    return load_functions(ROOT)


def example():
    gt=np.ones((12,16),np.int64);gt[:,8:]=2;gt[6:,12:]=3
    valid=np.ones_like(gt,bool)
    logits=torch.linspace(-2.,2.,8*gt.size,dtype=torch.float32).reshape(1,8,*gt.shape)
    return logits,gt,valid,{1:1,2:1,3:0}


def test_actual_original_compute_and_element_gradient_reconstruct(functions):
    logits,gt,valid,classes=example()
    expected=logits.clone();result=analyze_signal(logits,gt,valid,classes,deepcopy(LOSS_CFG),functions)
    assert result['reconstruction']['actual_compute_target_and_mask_exact']
    assert result['reconstruction']['gradient_max_abs_error'] < 1e-8
    assert result['actual_supervision']['negative_edges']==result['negative']['n']
    assert result['actual_supervision']['positive_edges']==result['positive']['n']
    assert result['positive']['loss_contribution']+result['negative']['loss_contribution']==pytest.approx(result['actual_bce'],rel=2e-6)
    assert torch.equal(expected,logits) and logits.grad is None
    assert {p['phase'] for p in result['pair_rows']}=={'F/F','F/P'}
    totals=[result['positive'],result['negative']]
    assert sum(t['gradient_l2_squared'] for t in totals)>0
    assert result['positive']['gradient_l2']**2==pytest.approx(result['positive']['gradient_l2_squared'])


def test_class_normalization_does_not_dilute_negative_class_with_positive_count():
    logits=torch.zeros(1,8,5,8)
    targets=torch.ones_like(logits);targets[:,:,:,0]=0
    valid=torch.ones_like(logits,dtype=torch.bool)
    loss_terms,coeff,weights,rows=element_contributions(logits,targets,valid)
    for c in range(8):
        assert coeff[:,c][targets[:,c]==0].sum().item()==pytest.approx(1/16)
        assert coeff[:,c][targets[:,c]==1].sum().item()==pytest.approx(1/16)
        assert rows[c]['class_terms']==2
        assert weights[:,c][targets[:,c]==0].unique().item()==pytest.approx(1.25)
    assert loss_terms.sum().item()==pytest.approx(np.log(2),rel=2e-6)


def test_hardness_detached_actual_loss_gradient(functions):
    z=torch.linspace(-3,3,8*8*8).reshape(1,8,8,8).requires_grad_()
    target=torch.ones_like(z);target[:,:,:,:3]=0
    valid=torch.ones_like(z,dtype=torch.bool)
    actual,_=functions['balanced_affinity_loss'](z,target,valid,negative_weight=1.,hard_negative_weight=1.,hard_negative_gamma=2.)
    terms,coeff,_,_=element_contributions(z,target,valid)
    grad=torch.autograd.grad(actual,z,retain_graph=True)[0]
    expected=coeff*(z.sigmoid()-target)
    torch.testing.assert_close(grad,expected,atol=1e-9,rtol=2e-6)
    torch.testing.assert_close(terms.sum(),actual,atol=1e-7,rtol=2e-6)


def test_actual_endpoint_only_validity_keeps_unknown_interior_long_edges(functions):
    gt=np.zeros((10,12),np.int64);gt[:,:3]=1;gt[:,5:]=2
    valid=np.ones_like(gt,bool);logits=torch.zeros(1,8,10,12)
    result=analyze_signal(logits,gt,valid,{1:1,2:0},LOSS_CFG,functions)
    pair=next(p for p in result['pair_rows'] if p['gt_ids']==[1,2])
    assert pair['four_neighbor_contact']==0
    assert pair['groups']['short']['n']==0
    assert pair['groups']['distance4']['n']>0
    assert pair['groups']['distance4']['unknown_path_n']>0
    assert result['reconstruction']['unknown_and_padding_gradient_zero']
    assert result['pair_summary']['adjacent_pairs']==0
    assert result['pair_summary']['nonadjacent_endpoint_pairs']==1


def test_padding_endpoints_are_ignored_and_single_class_channels_close(functions):
    gt=np.ones((8,10),np.int64);valid=np.ones_like(gt,bool);valid[:,7:]=False
    result=analyze_signal(torch.randn(1,8,8,10,generator=torch.Generator().manual_seed(7)),gt,valid,{1:1},LOSS_CFG,functions)
    assert result['negative']['n']==0 and result['pair_rows']==[]
    assert result['positive']['loss_contribution']==pytest.approx(result['actual_bce'],rel=2e-6)
    assert all(c['class_terms']==1 for c in result['channels'])
    assert result['reconstruction']['unknown_and_padding_gradient_zero']


def test_empty_supervision_is_zero_with_closed_budget(functions):
    gt=np.zeros((8,10),np.int64);valid=np.ones_like(gt,bool)
    result=analyze_signal(torch.zeros(1,8,8,10),gt,valid,{},LOSS_CFG,functions)
    assert result['actual_bce']==0 and result['positive']['gradient_l1']==0
    assert result['pair_rows']==result['object_rows']==[]
    assert result['pair_summary']['top_tenth_by_contact']['loss_share'] is None


def test_four_neighbor_contact_is_not_diagonal_or_fabricated_arc_length():
    gt=np.zeros((8,8),np.int64);gt[:4,:4]=1;gt[4:,4:]=2
    assert contact_lengths(gt,np.ones_like(gt,bool))=={}
    gt[:,4:]=2;gt[:,:4]=1
    assert contact_lengths(gt,np.ones_like(gt,bool))=={(1,2):8}


def test_native_contact_lost_in_training_grid_is_retained_as_no_coverage(functions):
    native=np.ones((12,16),np.int64);native[:,7]=2
    grid=np.ones((8,10),np.int64)
    signal=analyze_signal(torch.zeros(1,8,8,10),grid,np.ones_like(grid,bool),{1:1,2:0},LOSS_CFG,functions)
    coverage=native_adjacency_coverage(native,{1:1,2:0},signal)
    assert coverage['native_adjacent_pairs']==1
    assert coverage['rows'][0]['native_contact_grid_edges']==24
    assert coverage['no_training_grid_four_neighbor_contact']==[[1,2]]
    assert coverage['no_negative_edges_any_channel']==[[1,2]]


def test_failed_pair_without_peer_and_without_contact_preserved(functions):
    gt=np.zeros((10,12),np.int64);gt[:,:3]=110;gt[:,5:]=131
    result=analyze_signal(torch.zeros(1,8,10,12),gt,np.ones_like(gt,bool),{110:1,131:0},LOSS_CFG,functions)
    case=dict(kind='merge',gt_ids=[110,131])
    rows=case_tables(25,[case],result)
    assert len(rows)==1 and rows[0]['no_contact'] and rows[0]['no_peer']
    assert rows[0]['row']['groups']['distance4']['n']>0


def test_geometry_and_id_permutation_do_not_change_budgets(functions):
    z,gt,valid,classes=example();a=analyze_signal(z,gt,valid,classes,LOSS_CFG,functions)
    changed=gt.copy();mapping={1:401,2:19,3:65535}
    for k,v in mapping.items():changed[gt==k]=v
    b=analyze_signal(z,changed,valid,{mapping[k]:v for k,v in classes.items()},LOSS_CFG,functions)
    assert a['actual_bce']==b['actual_bce']
    for key in ('loss_contribution','gradient_l1','gradient_l2_squared','n'):
        assert a['negative'][key]==pytest.approx(b['negative'][key],rel=1e-12)


@pytest.mark.parametrize('mutation',["config","nan","missing_class","bool_class","float_class","content_type","shape"])
def test_malformed_inputs_fail_before_analysis(functions,mutation):
    z,gt,valid,classes=example();cfg=deepcopy(LOSS_CFG)
    if mutation=='config':cfg['negative_weight']=1.5
    if mutation=='nan':z[0,0,0,0]=float('nan')
    if mutation=='missing_class':classes.pop(2)
    if mutation=='bool_class':classes[1]=True
    if mutation=='float_class':classes[1]=1.0
    if mutation=='content_type':valid=valid.astype(np.uint8)
    if mutation=='shape':z=z[:,:,:,:-1]
    with pytest.raises(ValueError):analyze_signal(z,gt,valid,classes,cfg,functions)


@pytest.mark.parametrize('name',['../cache.npz','a/../../cache.npz'])
def test_artifact_path_escape_rejected(tmp_path,name):
    with pytest.raises(ValueError):safe_child(tmp_path,name)


def test_target_hook_restored_even_if_actual_compute_raises(functions):
    z,gt,valid,classes=example();original=functions['build_affinity_targets_torch']
    def bad(*args,**kwargs):raise RuntimeError('intentional')
    functions['compute_affinity_loss']=bad
    with pytest.raises(RuntimeError,match='intentional'):analyze_signal(z,gt,valid,classes,LOSS_CFG,functions)
    assert functions['build_affinity_targets_torch'] is original
