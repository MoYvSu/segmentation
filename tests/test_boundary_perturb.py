# -*- coding: utf-8 -*-
"""受扰热图混合：原采样计划/GT不变、压缩表示、独立流与完整来源合同。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.boundary_perturb import BoundaryPerturbDataset, array_sha, collate_boundary_perturb, choose_variant
from data.boundary_rgb import BoundaryRGBDataset
from test_boundary_rgb import _fixture, _sha, _json


def make(tmp_path, *, binary=True, probability=.5, shape=(80, 96)):
    processed, clean_manifest, cfg, _, clean, _ = _fixture(tmp_path, shape)
    clean[:, shape[1]//2-1:shape[1]//2+2] = .9
    record = json.loads(clean_manifest.read_text())['rows'][0]
    clean_file = tmp_path/record['heat_path']
    np.savez_compressed(clean_file, boundary=clean)
    doc = json.loads(clean_manifest.read_text())
    doc['rows'][0]['heat_sha256'] = _sha(clean_file)
    _json(clean_manifest, doc)
    changed = clean.copy()
    changed[shape[0]//4:shape[0]*3//4, shape[1]//2-8:shape[1]//2+8] = 0
    heat = tmp_path/'perturbed.npz'
    stored = (changed >= .5).astype(np.uint8) if binary else changed
    np.savez_compressed(heat, boundary=stored)
    row = dict(doc['rows'][0], heat_path=heat.name, heat_sha256=_sha(heat),
        representation='thresholded_binary' if binary else 'float32_probability')
    if binary:
        row.update(threshold=.5, continuous_boundary_sha256=array_sha(changed),
                   binary_sha256=array_sha(changed >= .5))
    manifest = tmp_path/'perturbed.json'
    _json(manifest, dict(format='boundary_naf_perturbed_heatmaps_v1', complete=True,
        no_test_training=True, clean_real_input_manifest_sha256=_sha(clean_manifest),
        processed_gt_manifest_sha256=_sha(processed),
        local_restoration={'mode':'zero','model':'restorer'},models=doc['models'],
        variants=[dict(id='moderate',augmentation={'kind':'smooth_local_blur'},rows=[row])]))
    cfg['boundary_naf'].update(input_mode='boundary', input_normalization={'mode':'skeleton_dilate','kernel_size':8},
        input_perturbation=dict(manifest=str(manifest),clean_probability=probability,variant_seed=73411))
    return processed, cfg, manifest


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_same_plan_GT_and_clean_tensor_exact(tmp_path):
    processed,cfg,_=make(tmp_path)
    old=BoundaryRGBDataset(processed,cfg)
    new=BoundaryPerturbDataset(processed,cfg)
    assert old.plan_receipt()==new.plan_receipt()
    assert len(new)==len(old)==16
    for i in range(len(old)):
        a,b=old[i],new.sample_variant(i,'clean')
        for key in ('input','input_boundary','raw_input_boundary','target','valid','rgb','image','gap_mask'):
            assert torch.equal(a[key],b[key]),key
        assert a['box']==b['box'] and a['source']==b['source'] and a['draw']==b['draw']
        assert b['meta']['input_variant']=='clean'
        assert not b['new_missing_gt_mask'].any()


@pytest.mark.parametrize('rotation,flip',[(r,f) for r in range(4) for f in (False,True)])
def test_variant_changes_only_boundary_and_preserves_D4(tmp_path,rotation,flip):
    processed,cfg,_=make(tmp_path)
    ds=BoundaryPerturbDataset(processed,cfg)
    ds.plan[0]=dict(ds.plan[0],box=[8,8,64,64],rotation=rotation,flip=flip,
                    augmentation=rotation+4*int(flip))
    a,b=ds.sample_variant(0,'clean'),ds.sample_variant(0,'moderate')
    for key in ('target','valid','rgb','image'):
        assert torch.equal(a[key],b[key])
    assert not torch.equal(a['input'],b['input'])
    assert torch.equal(b['input'],b['input_boundary'])
    assert torch.equal(b['clean_input_boundary'],a['input_boundary'])
    assert b['new_missing_gt_mask'].sum()>0
    assert not (b['new_missing_gt_mask'].bool() & ~b['valid'].bool()).any()
    assert b['meta']['geometry']==a['meta']['geometry']


def test_binary_and_float_probability_variants_are_identical(tmp_path):
    p1=tmp_path/'binary';p2=tmp_path/'float';p1.mkdir();p2.mkdir()
    source1,cfg1,_=make(p1,binary=True)
    source2,cfg2,_=make(p2,binary=False)
    a,b=BoundaryPerturbDataset(source1,cfg1),BoundaryPerturbDataset(source2,cfg2)
    assert a.plan==b.plan
    for i in range(len(a)):
        aa,bb=a.sample_variant(i,'moderate'),b.sample_variant(i,'moderate')
        for key in ('input','target','valid','raw_input_boundary','gap_mask'):
            assert torch.equal(aa[key],bb[key]),key


def test_hot_access_does_not_reload_or_change_inputs(tmp_path,monkeypatch):
    processed,cfg,_=make(tmp_path,probability=0.)
    ds=BoundaryPerturbDataset(processed,cfg)
    identity=ds.verify_sources()
    with monkeypatch.context() as p:
        p.setattr(np,'load',lambda *a,**k:pytest.fail('hot sample must use RAM cache'))
        for i in range(len(ds)):
            s=ds[i];s['input'].zero_();s['clean_input_boundary'].zero_();s['target'].zero_()
    assert ds.verify_sources()==identity
    ds.variant_boundaries['moderate']['train_001'][4,4]^=True
    with pytest.raises(RuntimeError,match='In-memory perturbed'):
        ds.verify_sources()


def test_selection_is_independent_replayable_and_epoch_specific(tmp_path):
    processed,cfg,_=make(tmp_path)
    ds=BoundaryPerturbDataset(processed,cfg)
    before_np=deepcopy(np.random.get_state());before_torch=torch.get_rng_state().clone()
    choices=[ds.selected_variant(i) for i in range(16)]
    assert set(choices)=={'clean','moderate'}
    assert choices==[ds.selected_variant(i) for i in range(16)]
    after=np.random.get_state()
    assert before_np[0]==after[0] and np.array_equal(before_np[1],after[1]) and before_np[2:]==after[2:]
    assert torch.equal(before_torch,torch.get_rng_state())
    old_plan=deepcopy(ds.plan);ds.set_epoch(2)
    reference=BoundaryRGBDataset(processed,cfg,epoch=2)
    assert ds.plan==reference.plan and ds.plan!=old_plan
    assert choices!=[ds.selected_variant(i) for i in range(16)]


def test_small_source_padding_remains_ignored(tmp_path):
    processed,cfg,_=make(tmp_path,shape=(40,48))
    ds=BoundaryPerturbDataset(processed,cfg)
    ds.plan[0]=dict(ds.plan[0],box=[0,0,40,48],rotation=1,flip=True,augmentation=5)
    s=ds.sample_variant(0,'moderate')
    assert s['input'].shape==(1,64,64)
    assert not s['valid'][0,48:].any() and not s['valid'][0,:,40:].any()
    assert not (s['new_missing_gt_mask'].bool() & ~s['valid'].bool()).any()


def test_collate_keeps_clean_and_new_missing_without_loss_changes(tmp_path):
    processed,cfg,_=make(tmp_path)
    ds=BoundaryPerturbDataset(processed,cfg)
    samples=[ds.sample_variant(0,'clean'),ds.sample_variant(1,'moderate')]
    batch=collate_boundary_perturb(samples)
    assert batch['input'].shape==(2,1,64,64)
    assert batch['clean_input_boundary'].shape==batch['input'].shape
    assert batch['new_missing_gt_mask'].shape==batch['target'].shape
    assert batch['meta'][0]['input_variant']=='clean' and batch['meta'][1]['input_variant']=='moderate'


@pytest.mark.parametrize('damage', ['clean_binding','GT_binding','model','source','shape','binary_sha','threshold','unknown_representation'])
def test_manifest_mismatch_is_rejected(tmp_path,damage):
    processed,cfg,path=make(tmp_path)
    doc=json.loads(path.read_text());row=doc['variants'][0]['rows'][0]
    if damage=='clean_binding':doc['clean_real_input_manifest_sha256']='0'*64
    elif damage=='GT_binding':doc['processed_gt_manifest_sha256']='0'*64
    elif damage=='model':doc['models']['restorer']='0'*64
    elif damage=='source':row['image_sha256']='0'*64
    elif damage=='shape':row['shape']=[1,1]
    elif damage=='binary_sha':row['binary_sha256']='0'*64
    elif damage=='threshold':row['threshold']=.45
    elif damage=='unknown_representation':row['representation']='jpeg'
    _json(path,doc)
    with pytest.raises(ValueError):BoundaryPerturbDataset(processed,cfg)


def test_manifest_tampering_after_load_is_detected(tmp_path):
    processed,cfg,path=make(tmp_path)
    ds=BoundaryPerturbDataset(processed,cfg)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(RuntimeError,match='manifest changed'):ds.verify_sources()
