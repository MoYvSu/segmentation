# -*- coding: utf-8 -*-
"""真实原生读图路由与独立采样流；不需要任何比赛图或私有输出。"""
from copy import deepcopy
import json
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from data.affinity_coverage import CoveragePlan, CoverageNativeViewDataset, anchor_box, anchors, flags
from data.affinity_native import NativeViewDataset


def synthetic_plan(ids=None,size=8):
    ids=np.ones((12,16),np.int32) if ids is None else ids
    old=[];new=[]
    for epoch in range(1,21):
        route=iter(flags(epoch))
        for step in range(1,65):
            # draw和update刻意不同，避免错误地用draw初始化anchor流。
            draw=(step*13)%64;native=step%2==0
            row=dict(epoch=epoch,step=step,updates=(epoch-1)*64+step,seed=10000*epoch+step,
                manual=['fiction.png'],pseudo=['pseudo.png'],manual_draw=dict(draw_index=[draw],source_index=[draw%32]),
                pseudo_draw=dict(draw_index=[draw],source_index=[draw]),manual_crop=[[1,1,1+size,1+size]],
                pseudo_crop=[[1,1,1+size,1+size]],crop_attempts=[[0],[0]],
                training_view='native1024' if native else 'whole',auxiliary_weight=0.)
            candidate=deepcopy(row);candidate['counterfactual_not_executed']=True
            if native:
                selected=next(route);candidate['counterfactual_anchor']=dict(selected=selected)
                if selected:
                    box,attempt,known=anchor_box(ids,epoch,row['updates'],size=size)
                    candidate['manual_crop']=[box];candidate['crop_attempts'][0]=[attempt]
                    candidate['counterfactual_anchor'].update(accepted_attempt=attempt,known_cardinal_pairs=known)
            old.append(row);new.append(candidate)
    return old,new


def test_plan_uses_actual_update_and_exact16_each_epoch_with_unchanged_other_routes():
    old,new=synthetic_plan();plan=CoveragePlan(old,new)
    for epoch in range(1,21):
        rows=new[(epoch-1)*64:epoch*64]
        assert sum(r.get('counterfactual_anchor',{}).get('selected',False) for r in rows)==16
    selected=next(r for r in new if r.get('counterfactual_anchor',{}).get('selected'))
    draw=selected['manual_draw']['draw_index'][0]
    assert plan.row(selected['epoch'],draw)['updates']==selected['updates']!=draw
    assert plan.row(2,new[64]['manual_draw']['draw_index'][0])['updates']==65
    assert anchors(1936,1024)==[0,456,912] and anchors(2584,1024)==[0,780,1560]


@pytest.mark.parametrize('mutation',['pseudo','whole','uniform','draw','duplicate','marker','update'])
def test_plan_rejects_hidden_changes_and_invalid_execution_marker(mutation):
    old,new=synthetic_plan()
    if mutation=='pseudo':new[0]['pseudo_crop']=[[0,0,8,8]]
    elif mutation=='whole':new[0]['counterfactual_anchor']={'selected':False}
    elif mutation=='uniform':
        row=next(r for r in new if r.get('counterfactual_anchor')=={'selected':False})
        row['counterfactual_anchor']['accepted_attempt']=0
    elif mutation=='draw':new[0]['manual_draw']['draw_index']=[7]
    elif mutation=='duplicate':old[1]['manual_draw']['draw_index']=old[0]['manual_draw']['draw_index']
    elif mutation=='marker':new[0]['counterfactual_not_executed']=False
    else:old[64]['updates']=1
    with pytest.raises((ValueError,KeyError)):CoveragePlan(old,new)


def test_anchor_no_global_rng_consumption_and_no_unknown_fallback():
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    box,attempt,count=anchor_box(np.ones((12,16),np.int32),2,65,size=8)
    assert attempt==0 and count==112
    assert random.getstate()==before[0]
    after=np.random.get_state();assert after[0]==before[1][0] and np.array_equal(after[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])
    with pytest.raises(RuntimeError,match='no fallback'):
        anchor_box(np.zeros((12,16),np.int32),1,2,size=8)
    with pytest.raises(ValueError,match='Actual update'):anchor_box(np.ones((12,16),np.int32),2,1,size=8)


def test_real_old_uniform_delegate_and_anchor_crop_share_canonical_new_gt(tmp_path):
    ids=np.ones((12,16),np.int32);ids[:,8:]=2;ids[5:7,9:11]=0
    rgb=np.zeros((12,16,3),np.uint8);rgb[...,0]=np.arange(16)[None,:]*10;rgb[...,1]=np.arange(12)[:,None]*12
    path=tmp_path/'fiction.png';assert cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    np.savez(tmp_path/'fiction_gt.npz',instance_map=ids,residual_unknown=(ids==0).astype(np.uint8))
    (tmp_path/'fiction_class.json').write_text(json.dumps({'1':1,'2':0}),encoding='utf8')
    base=SimpleNamespace(samples=[path]*32,completed_gt_dir=tmp_path,image_size=8,affinity_grid=4)
    options=dict(crop_attempts=32,minimum_known_pairs=16)
    original=NativeViewDataset(base,'manual',8,17,options)
    old,new=synthetic_plan(ids);plan=CoveragePlan(old,new)
    selected=next(r for r in new if r.get('counterfactual_anchor',{}).get('selected'))
    wrapper=CoverageNativeViewDataset(original,plan)
    wrapper.epoch,wrapper.draw=selected['epoch'],selected['manual_draw']['draw_index'][0]
    sample=wrapper[selected['manual_draw']['source_index'][0]]
    from data.affinity_native import geometry_sample
    y,x,y1,x1=selected['manual_crop'][0]
    explicit=geometry_sample(rgb[y:y1,x:x1],ids[y:y1,x:x1],image_size=8,grid=4)
    for key in ('image','affinity_instance_map','affinity_valid_content','valid_content','clean_image'):
        assert torch.equal(sample[key],explicit[key])
    assert sample['coverage_update']==selected['updates'] and sample['coverage_anchor'] is True
    uniform=next(r for r in new if r.get('counterfactual_anchor')=={'selected':False})
    epoch,draw=uniform['epoch'],uniform['manual_draw']['draw_index'][0]
    original.epoch,original.draw=epoch,draw;expected=original[draw%32]
    # 合成旧收据使用真实原采样输出，路由必须直接委托同一原对象。
    uniform['manual_crop']=[expected['crop_box'].tolist()];uniform['crop_attempts'][0]=[expected['crop_attempt']]
    plan.by_draw[(epoch,draw)]=uniform;wrapper.epoch,wrapper.draw=epoch,draw
    actual=wrapper[draw%32]
    for key,value in expected.items():
        if torch.is_tensor(value):assert torch.equal(actual[key],value)
        else:assert actual[key]==value
    assert actual['coverage_anchor'] is False
