# -*- coding: utf-8 -*-
"""三组固定GT汇总与旧缓存门禁，不需要GPU。"""
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import analyze_affinity_tail_gt as subject


def sample():
    gt = np.ones((60,60),np.int32)
    gt[10:50,10:50] = 2
    return gt,{1:0,2:1}


def diagnose(gt,pred,classes,labels):
    return subject.diagnose_partition(gt,pred,classes,labels)


def test_cpu_self_test():
    subject.self_test()


def test_unknown_pixels_never_enter_phase_area():
    gt,classes=sample();gt[:5,:]=0
    pred=gt.astype(np.uint16);pred[:5,:]=3
    d=diagnose(gt,pred,classes,{'1':0,'2':1,'3':1})
    r=subject.phase_diagnostics(gt,pred,classes,{'1':0,'2':1,'3':1},d)
    assert r['seen_class_aware_mean_instance_iou']==1
    assert r['ferrite']['pred_instances_in_known_domain']==1
    assert r['ferrite']['pairs']==[dict(gt=2,pred=2,iou=1.)]


def test_wrong_semantics_are_distinct_from_perfect_geometry():
    gt,classes=sample();pred=gt.astype(np.uint16)
    d=diagnose(gt,pred,classes,{'1':1,'2':0})
    r=subject.phase_diagnostics(gt,pred,classes,{'1':1,'2':0},d)
    assert d['geometry']['matched']==2
    assert r['ferrite']['class_aware_matches']==0
    assert r['pearlite']['class_aware_matches']==0


def test_F_split_unknown_and_frame_conservative_censor():
    gt,classes=sample();pred=gt.astype(np.uint16);pred[10:50,30:50]=3
    labels={'1':0,'2':1,'3':1}
    d=diagnose(gt,pred,classes,labels)
    r=subject.phase_diagnostics(gt,pred,classes,labels,d)
    assert r['true_F_geometry']['F_split_gt_ids']==[2]
    assert r['true_F_geometry_conservative']['F_split_gt_ids']==[2]
    assert 1 in d['conservative']['censored_gt_ids']
    gt[29:31,29:31]=0
    d=diagnose(gt,pred,classes,labels)
    r=subject.phase_diagnostics(gt,pred,classes,labels,d)
    assert r['true_F_geometry']['F_split_gt_ids']==[2]
    assert r['true_F_geometry_conservative']['F_split_gt_ids']==[]


def test_pairwise_uses_common_GT_instead_of_different_matched_pools():
    gt,classes=sample();base=gt.astype(np.uint16)
    split=base.copy();split[10:50,30:50]=3
    labels={'1':0,'2':1};split_labels={'1':0,'2':1,'3':1}
    ds={n:diagnose(gt,p,classes,lab) for n,p,lab in
        [('r1',base,labels),('fused',split,split_labels),('tail',base,labels)]}
    ps={n:subject.phase_diagnostics(gt,p,classes,lab,ds[n]) for n,p,lab in
        [('r1',base,labels),('fused',split,split_labels),('tail',base,labels)]}
    labs=dict(r1=labels,fused=split_labels,tail=labels)
    row=dict(arms=ds,phase_diagnostics=ps,pairwise={a+'_to_'+b:
        subject.pairwise_diagnostic(ds[a],ds[b],classes,labs[a],labs[b])
        for a,b in combinations(subject.ARMS,2)})
    r=subject.summarize_rows([row,row])
    assert r['paired']['r1_to_tail']['known']['common_iou_delta_mean']==0
    assert r['paired']['r1_to_fused']['known']['common_iou_delta_mean']==pytest.approx(-.25)
    assert r['paired']['fused_to_tail']['known']['common_iou_delta_mean']==pytest.approx(.25)
    assert r['triple_common']['known']['common_matches']==4
    assert r['triple_common']['conservative']['common_matches']==2
    assert r['totals']['fused']['true_F_geometry']['true_F_geometry']['split_gt']==2


def test_real_inventory_rejects_mutation_and_path_escape(tmp_path):
    file=tmp_path/'code.py';file.write_text('value=1',encoding='utf8')
    receipt={'code.py':subject.sha(file)}
    subject.pinned_files(tmp_path,receipt)
    file.write_text('value=2',encoding='utf8')
    with pytest.raises(RuntimeError,match='Pinned file'):
        subject.pinned_files(tmp_path,receipt)
    with pytest.raises(RuntimeError,match='Pinned file'):
        subject.pinned_files(tmp_path,{'../code.py':'bad'})


def test_virtual_module_is_skipped_but_real_read_error_is_not_hidden(tmp_path,monkeypatch):
    real=tmp_path/'real.py';real.write_text('ok',encoding='utf8')
    virtual=tmp_path/'_ops.py'
    modules=[SimpleNamespace(__file__=str(p)) for p in (real,virtual)]
    r=subject.existing_project_sources(modules,tmp_path)
    assert r['files']=={'real.py':subject.sha(real)}
    assert r['skipped_nonexistent_virtual_paths']==[str(virtual)]
    monkeypatch.setattr(subject,'sha',lambda p: (_ for _ in ()).throw(PermissionError(str(p))))
    with pytest.raises(PermissionError):
        subject.existing_project_sources(modules,tmp_path)


def test_empty_known_domain_has_no_invented_class_score():
    gt=np.zeros((4,4),np.int32);pred=np.zeros_like(gt,dtype=np.uint16)
    d=diagnose(gt,pred,{}, {})
    r=subject.phase_diagnostics(gt,pred,{}, {},d)
    assert r['seen_class_aware_mean_instance_iou'] is None
    assert r['ferrite']['class_aware_matches']==0
