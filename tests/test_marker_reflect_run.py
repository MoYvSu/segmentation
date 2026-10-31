# -*- coding: utf-8 -*-
"""反射种子候选：原forward透明性、固定rawvote、参考门禁与存盘契约。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import tools.run_marker_reflect as marker_run

from tools.affinity_connectivity_views import decode_native
from tools.semantic_crossover import revote_instances
from tools.run_marker_reflect import (VIEW, capture_original, config_gate,
    load_prediction, replay_candidates, save_prediction, selected_indices,
    validate_manifest, verify_prediction)
from utils.patch_diagnostic import phase_description


def sample():
    config={'inference':dict(boundary_threshold=.65,marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8,marker_border_seal_width=2,bridge_width=1,
        watershed_dilate_width=1,min_instance_area=50,max_instance_id=65535,
        threshold=.5,semantic_vote_mode='probability_mean')}
    boundary=np.full((128,144),.03,np.float32); boundary[:,65:79]=.9
    boundary[58:62,65:79]=.53
    semantic=torch.linspace(-2,2,144)[None,None,None].expand(1,1,128,144).clone()
    # 故意与geometry语义相反，确保候选最终类别真正沿用rawvote。
    raw=np.full((128,144),.8,np.float32); raw[:,72:]=.2
    return semantic,torch.from_numpy(boundary)[None,None],raw,config


def fake_views(semantic,boundary,probability,config):
    views=SimpleNamespace(decode_native=decode_native,revote_instances=revote_instances)
    calls=[]

    def predict(model,restorer,path,actual_config,device,seed,sizes):
        calls.append((model,restorer,path,actual_config,device,seed,sizes))
        ids,_=views.decode_native(semantic,boundary,actual_config)
        classes,_=views.revote_instances(ids,probability)
        result=dict(instances=ids,classes=classes,boundary=boundary[0,0].numpy().copy(),
                    phases=phase_description(ids,classes),blend=dict(tiles=3))
        return np.zeros((*ids.shape,3),np.uint8),{VIEW:result}

    views.predict_views=predict
    return views,calls


def test_original_forward_only_once_same_call_contract_and_restores_hooks():
    semantic,boundary,raw,config=sample()
    views,calls=fake_views(semantic,boundary,raw,config)
    expected_rgb,expected=views.predict_views('model','restorer',Path('synthetic.png'),config,'cpu',314171,[1024])
    calls.clear(); sem_before=semantic.clone(); edge_before=boundary.clone(); prob_before=raw.copy()
    rgb,captured=capture_original(views,'model','restorer',Path('synthetic.png'),config,'cpu',314171)
    assert len(calls)==1 and calls[0][-2:]==(314171,[1024])
    assert views.decode_native is decode_native and views.revote_instances is revote_instances
    assert np.array_equal(rgb,expected_rgb)
    assert np.array_equal(captured['baseline']['instances'],expected[VIEW]['instances'])
    assert captured['baseline']['classes']==expected[VIEW]['classes']
    assert torch.equal(semantic,sem_before) and torch.equal(boundary,edge_before)
    assert np.array_equal(raw,prob_before)


def test_replay_actual_full_output_and_frozen_rawvote_with32_64():
    semantic,boundary,raw,config=sample(); views,_=fake_views(semantic,boundary,raw,config)
    _,captured=capture_original(views,None,None,Path('synthetic.png'),config,'cpu',7)
    before_config=deepcopy(config)
    original,variants,gates=replay_candidates(captured,config,stability=True)
    assert gates['same_marker_gate'] is True
    assert gates['original_rawvote_exact'] and gates['original_elevation_exact']
    assert set(variants)=={'32','64'}
    for value in variants.values():
        expected,_=revote_instances(value['instances'],raw)
        assert value['classes']==expected
        assert value['elevation_sha256']==original['elevation_sha256']
        assert np.array_equal(value['original_markers'],original['actual_markers'])
        verify_prediction(value['instances'],value['classes'],raw.shape)
    assert config==before_config


def test_replay_fails_if_actual_capture_marker_or_final_output_changes():
    semantic,boundary,raw,config=sample(); views,_=fake_views(semantic,boundary,raw,config)
    _,captured=capture_original(views,None,None,Path('synthetic.png'),config,'cpu',7)
    corrupted=deepcopy(captured); corrupted['markers'][10,10]+=1
    with pytest.raises(RuntimeError,match='actual original'):
        replay_candidates(corrupted,config)
    corrupted=deepcopy(captured); corrupted['baseline']['classes']={k:1-v for k,v in corrupted['baseline']['classes'].items()}
    with pytest.raises(RuntimeError,match='rawvote'):
        replay_candidates(corrupted,config)


def minimal_config():
    return dict(paths=dict(project_root='old'),backend_adaptation=dict(output_dir='old',seed=3),
                sam2={},inference=dict(boundary_threshold=.65),affinity_native=dict(overlap=.25))


def test_config_gate_permits_only_recorded_paths_and_effective_legacy_default():
    canonical=minimal_config(); actual=deepcopy(canonical)
    actual['paths']['project_root']='new'; actual['backend_adaptation']['output_dir']='new'
    actual['sam2']['input_normalization']='legacy_none'
    result=config_gate(actual,canonical)
    assert len(result)==3 and not result[-1]['canonical_explicit'] and result[-1]['actual_explicit']
    assert 'input_normalization' not in canonical['sam2']
    for group,key,value in [('inference','boundary_threshold',.66),('sam2','input_normalization','imagenet'),
                             ('affinity_native','overlap',.5)]:
        changed=deepcopy(actual); changed[group][key]=value
        with pytest.raises(RuntimeError): config_gate(changed,canonical)


def test_manifest200_rows_are100_sources_and_full_indices_not_resampled():
    files=[Path('synthetic_%03d.png'%i) for i in range(100)]
    manifest=dict(complete=True,epoch=20,precision='FP32',sizes=[0,1024],checkpoint_sha256='digest',
                  images=[dict(source=p.stem,view=v) for p in files for v in ('global',VIEW)])
    assert len(validate_manifest(manifest,files,'digest'))==100
    assert selected_indices(None,100)==list(range(100))
    assert selected_indices('17,0',100)==[0,17]
    changed=deepcopy(manifest); changed['images'].append(changed['images'][1])
    with pytest.raises(RuntimeError): validate_manifest(changed,files,'digest')
    for value in ('-1','100','0,0','x',''):
        with pytest.raises(ValueError): selected_indices(value,100)


def test_saved_candidate16bit_and_exact_class_ids(tmp_path):
    ids=np.array([[0,1,1],[65535,65535,0]],np.uint16)
    labels={'1':0,'65535':1}
    receipt=save_prediction(tmp_path,'synthetic',ids,labels)
    actual,classes=load_prediction(tmp_path,'synthetic',ids.shape)
    assert actual.dtype==np.uint16 and np.array_equal(actual,ids) and classes==labels
    assert len(receipt['png_sha256'])==len(receipt['classes_sha256'])==64
    # PNG IHDR确实是16位灰度，不只是内存dtype正确。
    payload=(tmp_path/'synthetic_inst.png').read_bytes()
    assert payload[24:26]==bytes((16,0))
    with pytest.raises(RuntimeError): save_prediction(tmp_path,'bad',ids.astype(np.uint8),{'1':0,'255':1})
    with pytest.raises(RuntimeError): verify_prediction(ids,{'1':0},ids.shape)


def test_original_capture_rejects_missing_watershed_and_multiple_votes():
    semantic,boundary,raw,config=sample(); views,_=fake_views(semantic,boundary,raw,config)
    views.decode_native=lambda *a,**k:(np.ones(raw.shape,np.uint16),{'1':1})
    with pytest.raises(RuntimeError,match='one watershed'):
        capture_original(views,None,None,Path('synthetic.png'),config,'cpu',1)


def test_source_snapshot_records_missing_virtual_file_without_scanning_outputs(tmp_path,monkeypatch):
    root=tmp_path/'repo'; root.mkdir(); tool=root/'tools'/'synthetic.py'
    tool.parent.mkdir(); tool.write_text('# synthetic source\n',encoding='utf8')
    for name in ('train_affinity_native.py','tools/affinity_native_views.py'):
        p=root/'outputs/affinity_balance/source'/name; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('# synthetic sealed original\n',encoding='utf8')
    fake=SimpleNamespace(__file__=str(root/'models'/'virtual.py'))
    private=SimpleNamespace(__file__=str(root/'outputs'/'private.py'))
    monkeypatch.setattr(marker_run,'ROOT',root); monkeypatch.setattr(marker_run,'__file__',str(tool))
    monkeypatch.setitem(marker_run.sys.modules,'_synthetic_virtual_source',fake)
    monkeypatch.setitem(marker_run.sys.modules,'_synthetic_private_source',private)
    hashes,skipped=marker_run.source_snapshot(root/'outputs/trial')
    assert skipped==['models/virtual.py']
    assert 'tools/synthetic.py' in hashes and 'outputs/private.py' not in hashes
    assert len(hashes)==3
