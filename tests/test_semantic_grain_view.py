# -*- coding: utf-8 -*-
"""整GT父对象裁窗预检：可信监督来源、完整视野、忽略域与梯度归因。"""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from tools.probe_semantic_native import verify_whole_baseline
from utils.semantic_native import prepare_native_tile
from utils.semantic_grain_view import (
    anchor_probability, build_grain_batch, grain_window, gradient_partition,
    preflight_gate, select_grain_anchors,
)


SUPPORTS = ('anchor_original','anchor_filled','other_complete_original','other_complete_filled',
            'truncated_neighbor_original','truncated_neighbor_filled','unknown','padding')


def manual_source():
    gt = np.zeros((520,760),np.uint16)
    classes = {}
    for gid, cls, y, x, side in [(1,1,20,20,100),(2,1,20,160,110),(3,1,20,310,120),
                                 (4,0,240,20,100),(5,0,240,160,110),(6,0,240,310,120),
                                 (7,1,430,20,20),(8,0,0,500,100),(9,1,380,500,100)]:
        gt[y:y+side,x:x+side] = gid; classes[gid] = cls
    original = gt>0
    original[380:401,500:600] = False  # 原标仅79%，不能进入锚点候选。
    original[20:42,160:270] = False  # 2号原标正好80%，仍合格。
    return gt,original,classes


def test_anchor_preselection_uses_only_manual_GT_balanced_lower_medians():
    gt,original,classes = manual_source()
    before = (gt.copy(),original.copy(),deepcopy(classes))
    selected = select_grain_anchors(gt,original,classes)
    assert {row['cls']:row['id'] for row in selected} == {0:5,1:2}
    assert all(row['native_area']>=10000 and row['original_fraction']>=.8 for row in selected)
    assert all(row['eligible_count']==3 and row['median_index']==1 for row in selected)
    assert all(row['frame_censored'] is False for row in selected)
    assert next(row for row in selected if row['id']==2)['original_fraction'] == .8
    np.testing.assert_array_equal(gt,before[0]); np.testing.assert_array_equal(original,before[1])
    assert classes == before[2]
    json.dumps(selected,allow_nan=False)


@pytest.mark.parametrize('failure',['only_F','no_large_P','no_original_P','empty'])
def test_missing_eligible_phase_or_anchor_does_not_silently_switch_to_predictions(failure):
    gt,original,classes = manual_source()
    if failure == 'only_F': classes = {gid:1 for gid in classes}
    elif failure == 'no_large_P':
        for gid in (4,5,6): gt[gt==gid]=0
        original &= gt>0
    elif failure == 'no_original_P':
        original[np.isin(gt,[4,5,6])] = False
    else:
        gt.fill(0); original.fill(False); classes={}
    with pytest.raises(ValueError):
        select_grain_anchors(gt,original,classes)


def test_native_square_window_moves_inward_without_cutting_complete_GT_parent():
    gt,original,classes = manual_source()
    anchors = select_grain_anchors(gt,original,classes)
    anchor = next(row for row in anchors if row['cls']==1)
    box = grain_window(gt.shape,anchor['bbox'])
    y0,y1,x0,x1 = box
    assert y0 == x0 == 0 and y1-y0 == x1-x0 == 512
    assert int((gt[y0:y1,x0:x1]==anchor['id']).sum()) == anchor['native_area']
    with pytest.raises(ValueError,match='entire'):
        build_grain_batch(gt,original,classes,(25,130,160,270),anchor['id'],64)


def test_real_rectangular_crop_preserves_RGB_GT_alignment_and_ignores_reflect_padding():
    gt = np.zeros((160,450),np.uint16)
    gt[10:120,10:120] = 1; gt[10:120,200:310] = 2
    original = gt>0; original[40:70,40:70]=False
    classes = {1:1,2:0}
    box = grain_window(gt.shape,(10,120,10,120))
    assert box == (0,160,0,450)  # 小于请求边长的轴保留真实矩形，不能挤成方形。
    rgb = np.zeros((*gt.shape,3),np.uint8)
    rgb[gt==1] = (255,0,0); rgb[gt==2]=(0,0,255)
    image,ph,pw,metadata = prepare_native_tile(rgb,box,64,'cpu')
    batch,parts,receipt = build_grain_batch(gt,original,classes,box,1,64)
    assert (ph,pw)==(41,0) and metadata['native_shape']==[160,450]
    expected_rgb = cv2.resize(rgb,(64,23),interpolation=cv2.INTER_LINEAR)
    expected_rgb = cv2.copyMakeBorder(expected_rgb,0,41,0,0,cv2.BORDER_REFLECT)
    expected_tensor = torch.from_numpy(expected_rgb).permute(2,0,1).float().unsqueeze(0)/255.
    assert torch.equal(image,expected_tensor)
    expected_ids = cv2.resize(gt,(64,23),interpolation=cv2.INTER_NEAREST)
    target = batch['semantic_target'][0,0].numpy()
    assert set(parts)==set(SUPPORTS)
    assert np.all(sum(mask.astype(np.uint8) for mask in parts.values())==1)
    assert all(mask.dtype==bool and mask.shape==(64,64) for mask in parts.values())
    assert np.all(target[parts['padding']|parts['unknown']]==-1.)
    assert np.all(parts['padding'][23:]) and not parts['padding'][:23].any()
    np.testing.assert_array_equal(target[:23][expected_ids==1],np.ones(int((expected_ids==1).sum()),np.float32))
    np.testing.assert_array_equal(target[:23][expected_ids==2],np.zeros(int((expected_ids==2).sum()),np.float32))
    assert parts['anchor_original'].any() and parts['anchor_filled'].any()
    assert parts['other_complete_original'].any()
    assert not parts['truncated_neighbor_original'].any()
    assert receipt['processed_parent_complete'] is True and receipt['physical_GT_completeness_not_claimed'] is True


def test_truncated_neighbors_are_separate_from_the_complete_anchor():
    gt = np.zeros((160,600),np.uint16)
    gt[10:120,10:120]=1; gt[10:120,150:260]=2; gt[10:120,350:460]=3
    original=gt>0; original[40:70,160:190]=False
    box=grain_window(gt.shape,(10,120,10,120),minimum_side=200)
    assert box==(0,160,0,200)
    _,parts,receipt=build_grain_batch(gt,original,{1:1,2:0,3:0},box,1,64)
    assert receipt['neighbor_truncated_ids']==[2]
    assert parts['anchor_original'].any() and parts['truncated_neighbor_original'].any()
    assert parts['truncated_neighbor_filled'].any()
    assert not parts['other_complete_original'].any()
    assert not parts['anchor_filled'].any()


def test_anchor_metrics_follow_GT_true_class_and_exclude_filled_pixels():
    gt=np.zeros((8,10),np.uint16); gt[1:7,1:9]=1
    original=gt>0; original[1,1]=False
    anchor=dict(id=1,cls=0)
    whole=np.full(gt.shape,.1,np.float32)
    crop=np.full(gt.shape,.16,np.float32); crop[1,1]=1.  # filled异常不能计入可信原标均值。
    base=anchor_probability(whole,gt,original,anchor)
    changed=anchor_probability(crop,gt,original,anchor)
    assert base['mean_true_class_probability']==pytest.approx(.9)
    assert changed['mean_true_class_probability']==pytest.approx(.84)
    assert base['mean_true_class_probability']-changed['mean_true_class_probability']>.05
    assert changed['wrong_original_pixel_fraction']==0. and changed['original_pixels']==47
    anchor['cls']=1
    F_changed=anchor_probability(crop,gt,original,anchor)
    assert F_changed['mean_true_class_probability']==pytest.approx(.16)
    assert F_changed['wrong_original_pixel_fraction']==1.


def support_case(values=(10.,1.,100.,1.,200.,1.,0.,0.)):
    parts={name:np.eye(8,dtype=bool)[i][None] for i,name in enumerate(SUPPORTS)}
    grad=torch.tensor(values,dtype=torch.float32).reshape(1,1,1,8)
    return grad,parts


def test_unknown_and_padding_output_gradients_are_exactly_zero_required():
    gradient,parts=support_case()
    receipt=gradient_partition(gradient,parts)
    assert receipt['absolute_by_support']['unknown']==receipt['absolute_by_support']['padding']==0.
    assert receipt['ancestor_interpolation_gradient_not_restricted'] is True
    assert receipt['absolute_by_support']['truncated_neighbor_original']==200.
    for index in (6,7):
        corrupted=gradient.clone(); corrupted[0,0,0,index]=torch.finfo(torch.float32).eps
        with pytest.raises(RuntimeError,match='exactly zero'):
            gradient_partition(corrupted,parts)
    json.dumps(receipt,allow_nan=False)


def test_initial_whole_gate_remains_bit_exact_and_never_tolerates_unchanged_votes():
    ids=np.array([[1,1,65535]],np.uint16)
    probability=np.array([[.8,.8,.5]],np.float32)
    classes={'1':1,'65535':0}
    digest=hashlib.sha256(probability.tobytes()).hexdigest()
    assert verify_whole_baseline('synthetic',probability,digest,ids,classes)[0]==classes
    changed=probability.copy(); changed[0,0]=np.nextafter(changed[0,0],np.float32(1.))
    with pytest.raises(RuntimeError,match='SHA'):
        verify_whole_baseline('synthetic',changed,digest,ids,classes)


def gate_row(cls=1,whole_f=.9,crop_f=.84,values=(10.,1.,100.,1.,200.,1.,0.,0.)):
    gt=np.zeros((8,10),np.uint16); gt[1:7,1:9]=1; original=gt>0
    anchor={'id':1,'cls':cls}
    gradient,parts=support_case(values)
    return dict(view='train_synthetic_'+str(cls),anchor=anchor,
        whole=anchor_probability(np.full(gt.shape,whole_f,np.float32),gt,original,anchor),
        crop=anchor_probability(np.full(gt.shape,crop_f,np.float32),gt,original,anchor),
        parameter_gradient_norms={'seg_fpn':2.,'seg_branch':3.,'semantic_residual':1.},
        anchor_parameter_gradient_norms={'seg_fpn':.5,'seg_branch':.8,'semantic_residual':.1},
        output_gradient=gradient_partition(gradient,parts))


@pytest.mark.parametrize('cls,whole_f,crop_f',[(1,.9,.84),(0,.1,.16)])
def test_gate_uses_GT_true_class_weakening_for_both_phases_without_requiring_whole_object_flip(cls,whole_f,crop_f):
    row=gate_row(cls,whole_f,crop_f)
    assert row['whole']['predicted_class']==row['crop']['predicted_class']==cls
    result=preflight_gate([row])
    assert result['passed'] is True and result['actionable_anchor_views']==[row['view']]
    assert result['test_used'] is False
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('cls,whole_f,crop_f',[(1,.84,.9),(0,.16,.1)])
def test_improvement_or_unlabeled_flips_cannot_fake_a_new_GT_weakening_signal(cls,whole_f,crop_f):
    row=gate_row(cls,whole_f,crop_f)
    row['test_large_flips']=1000; row['test_ferrite_mean_area']=12345.
    result=preflight_gate([row])
    assert result['passed'] is False and result['difficult_views']==[]
    assert result['test_used'] is False


def test_GT_original_pixel_errors_can_expose_a_signal_even_when_true_probability_mean_is_stable():
    row=gate_row(1,.6,.6)
    gt=np.zeros((8,10),np.uint16); gt[1:7,1:9]=1; original=gt>0
    probability=np.full(gt.shape,.6,np.float32)
    # 5/48原标像素变错，其余略增加；均值不降低，GT像素缺陷仍是真实新增信号。
    positions=np.argwhere(original)
    probability[original]=(48*.6-5*.4)/43
    for y,x in positions[:5]: probability[y,x]=.4
    row['crop']=anchor_probability(probability,gt,original,row['anchor'])
    assert row['whole']['mean_true_class_probability']-row['crop']['mean_true_class_probability']<.01
    assert row['crop']['wrong_original_pixel_fraction']>row['whole']['wrong_original_pixel_fraction']+.01
    assert preflight_gate([row])['passed'] is True


def test_only_neighbor_gradients_cannot_pass_without_real_anchor_coarse_gradients():
    row=gate_row()
    row['anchor_parameter_gradient_norms']['seg_fpn']=0.
    row['anchor_parameter_gradient_norms']['seg_branch']=0.
    assert row['parameter_gradient_norms']['seg_fpn']>0 and row['parameter_gradient_norms']['seg_branch']>0
    result=preflight_gate([row])
    assert result['difficult_views'] and result['actionable_anchor_views']==[] and result['passed'] is False


def test_neighbor_original_gradient_cannot_hide_filled_dominance_in_anchor_itself():
    row=gate_row(values=(.1,1.,1000.,0.,2000.,0.,0.,0.))
    assert row['output_gradient']['original_absolute']>row['output_gradient']['filled_absolute']
    assert row['output_gradient']['anchor_original_absolute']<row['output_gradient']['anchor_filled_absolute']
    assert preflight_gate([row])['passed'] is False


def test_large_filled_neighbor_mass_is_reported_but_does_not_replace_the_final_anchor_gate():
    row=gate_row(values=(10.,1.,0.,1000.,0.,2000.,0.,0.))
    assert row['output_gradient']['original_absolute']<row['output_gradient']['filled_absolute']
    assert row['output_gradient']['anchor_original_absolute']>row['output_gradient']['anchor_filled_absolute']
    assert preflight_gate([row])['passed'] is True


@pytest.mark.parametrize('damage',['common_coarse_zero','anchor_nonfinite','unknown_gradient','padding_gradient','nonfinite_support'])
def test_illegal_gradients_cannot_be_rescued_by_strong_GT_weakening(damage):
    row=gate_row()
    if damage=='common_coarse_zero': row['parameter_gradient_norms']['seg_fpn']=0.
    elif damage=='anchor_nonfinite': row['anchor_parameter_gradient_norms']['seg_branch']=float('inf')
    elif damage=='unknown_gradient': row['output_gradient']['absolute_by_support']['unknown']=1e-12
    elif damage=='padding_gradient': row['output_gradient']['absolute_by_support']['padding']=1e-12
    else: row['output_gradient']['absolute_by_support']['anchor_original']=float('nan')
    assert preflight_gate([row])['passed'] is False


def test_one_good_anchor_does_not_override_another_view_with_illegal_common_gradients():
    valid=gate_row(); invalid=gate_row(0,.1,.1)
    invalid['parameter_gradient_norms']['seg_branch']=0.
    assert preflight_gate([valid])['passed'] is True
    assert preflight_gate([valid,invalid])['passed'] is False


def test_real_common_criterion_and_anchor_diagnostic_reach_coarse_head_without_optimizer_or_ignore_gradient():
    from tools.probe_semantic_object_fit import EXPECTED_LOSS
    from train_direct_semantic_affinity import build_semantic_criterion,compute_semantic_loss
    from train_backend_adaptation import tensor_digest
    gt=np.zeros((160,450),np.uint16); gt[10:120,10:120]=1; gt[10:120,200:310]=2
    original=gt>0; original[40:70,40:70]=False
    batch,parts,_=build_grain_batch(gt,original,{1:1,2:0},(0,160,0,450),1,64)
    head=torch.nn.Module()
    head.seg_fpn=torch.nn.Conv2d(3,2,1); head.seg_branch=torch.nn.Conv2d(2,1,1)
    with torch.no_grad():
        head.seg_fpn.weight.fill_(.2); head.seg_fpn.bias.fill_(.1)
        head.seg_branch.weight.fill_(.3); head.seg_branch.bias.fill_(.2)
    head.eval()
    source=torch.ones((1,3,64,64),dtype=torch.float32,requires_grad=False)
    initial=tensor_digest(head.state_dict().items())
    criterion=build_semantic_criterion({'direct_semantic_affinity':{'semantic_loss':EXPECTED_LOSS}},'cpu')
    logits=head.seg_branch(head.seg_fpn(source))
    common=compute_semantic_loss(criterion,logits,batch)
    named=list(head.named_parameters())
    anchor_mean=logits.sigmoid()[0,0][torch.from_numpy(parts['anchor_original'])].mean()
    anchor_BCE=torch.nn.functional.binary_cross_entropy_with_logits(torch.logit(anchor_mean,eps=1e-5),anchor_mean.new_tensor(1.))
    anchor_gradients=torch.autograd.grad(anchor_BCE,[p for _,p in named],retain_graph=True)
    common_gradients=torch.autograd.grad(common,[p for _,p in named]+[logits])
    for prefix in ('seg_fpn','seg_branch'):
        assert sum(float(g.square().sum()) for (key,_),g in zip(named,anchor_gradients) if key.startswith(prefix+'.'))>0
        assert sum(float(g.square().sum()) for (key,_),g in zip(named,common_gradients[:-1]) if key.startswith(prefix+'.'))>0
    receipt=gradient_partition(common_gradients[-1],parts)
    assert receipt['absolute_by_support']['unknown']==receipt['absolute_by_support']['padding']==0.
    assert receipt['anchor_original_absolute']>0
    assert tensor_digest(head.state_dict().items())==initial
    assert all(p.grad is None for p in head.parameters()) and not source.requires_grad


def test_preflight_plan_uses_only_labeled_training_sources_and_never_starts_training():
    from tools import probe_semantic_grain_view as tool
    from utils.config import load_config
    config=load_config('config/train/semantic_grain_view.yaml')
    for all_sources,expected in ((False,8),(True,64)):
        plan=tool.plan(config,all_sources)
        assert plan['views']==expected and plan['training'] is False
        assert plan['optimizer_updates']==0 and plan['train64_implemented'] is False
        assert plan['test_used'] is False and plan['official_submission'] is False
        if not all_sources: assert plan['sources']==list(tool.TRAIN)
        json.dumps(plan,allow_nan=False)


def test_prior_control_reference_contract_is_separate_from_original_R1_initialization():
    from tools import probe_semantic_grain_view as tool
    from tools.probe_semantic_object_fit import EXPECTED_LOSS,HEAD_FORMAT
    from train_backend_adaptation import tensor_digest
    old=dict(checkpoint_sha256='synthetic_R1',learning_rate=2e-5,adam_eps=1e-4,weight_decay=1e-4)
    state={'synthetic_weight':torch.tensor([1.,2.],dtype=torch.float32)}
    cfg={'control_head_digest':tensor_digest(state.items())}
    capture=dict(identity={'synthetic_GT_identity':'same'},source_order=['train_synthetic']*64,
        embedded_loss_config=deepcopy(EXPECTED_LOSS))
    control=dict(format=HEAD_FORMAT,arm='control',step=64,terminal_only=True,
        do_not_replace_geometry_upstream=True,region_weight=0.,base_checkpoint_sha256='synthetic_R1',
        identity=deepcopy(capture['identity']),schedule=list(capture['source_order']),mode='eval',
        semantic_state_dict=state,optimizer={'param_groups':[dict(lr=2e-5,eps=1e-4,weight_decay=1e-4)]})
    before=state['synthetic_weight'].clone()
    receipt=tool.validate_control_reference(control,capture,cfg,old)
    assert receipt['verified'] is True and receipt['used_as_initialization'] is False
    assert receipt['training_repeated'] is False and torch.equal(state['synthetic_weight'],before)
    for corruption in ('GT_identity','region_objective','optimizer_recipe'):
        damaged=deepcopy(control)
        if corruption=='GT_identity': damaged['identity']={'synthetic_GT_identity':'different'}
        elif corruption=='region_objective': damaged['region_weight']=.5
        else: damaged['optimizer']['param_groups'][0]['lr']=3e-5
        with pytest.raises(RuntimeError): tool.validate_control_reference(damaged,capture,cfg,old)
