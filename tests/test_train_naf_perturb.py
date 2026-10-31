# -*- coding: utf-8 -*-
"""独立NAF受扰入口的旧损失复用、固定监督分母、过程图与钩子恢复。"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

import train_boundary_rgb as base
from data.boundary_perturb import BoundaryPerturbDataset, collate_boundary_perturb
from models.boundary_restore import state_sha256
from models.boundary_rgb import BoundaryRGBNAFNet
from tools import train_naf_perturb as entry
from utils.config import load_config
from test_boundary_perturb import make


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def valid_config():
    cfg=load_config('config/train/boundary_width.yaml')
    cfg['boundary_naf']['input_perturbation']=dict(manifest='unused.json',clean_probability=.5,variant_seed=93)
    cfg['boundary_naf']['monitor']['full_readout_steps']=[]
    return cfg


def test_config_preserves_loss_and_rejects_stacked_experiments():
    cfg=valid_config();before=deepcopy(cfg)
    entry.validate_config(cfg)
    assert cfg==before
    cfg['boundary_naf']['erasure']={'enabled':True}
    with pytest.raises(ValueError):entry.validate_config(cfg)
    cfg=valid_config();cfg['boundary_naf']['monitor']['full_readout_steps']=[1000]
    with pytest.raises(ValueError,match='WS'):entry.validate_config(cfg)


def test_update_statistics_use_actual_batch_without_changing_targets(tmp_path):
    processed,cfg,_=make(tmp_path)
    ds=BoundaryPerturbDataset(processed,cfg)
    batch=collate_boundary_perturb([ds.sample_variant(0,'clean'),ds.sample_variant(1,'moderate')])
    before={k:v.clone() for k,v in batch.items() if torch.is_tensor(v)}
    result=entry.update_audit(batch)
    assert result['input_variant_counts']=={'clean':1,'moderate':1}
    assert result['new_missing_GT_points']==int(batch['new_missing_gt_mask'].sum())
    assert result['supervised_GT_positive_points']==int(((batch['target']>.5)&(batch['valid']>.5)).sum())
    for k,v in before.items():assert torch.equal(batch[k],v)


def test_adapted_hooks_restore_on_exception_and_do_not_replace_old_update_math(monkeypatch):
    events=[]
    def original_update(*args):events.append('old_update');return {'loss_g':3.5,'unchanged_sentinel':9}
    monkeypatch.setattr(base,'update',original_update)
    old_dataset,old_identity,old_monitor,old_collate=base.dataset_for,base.source_identity,base.monitor,base.collate_boundary_rgb
    monkeypatch.setattr(entry,'dataset_for',lambda *args:'new_dataset')
    batch={'target':torch.ones(1,1,4,4),'valid':torch.ones(1,1,4,4),
        'meta':[dict(input_variant='moderate',new_missing_gt_points=2,clean_gap_pixels=1,gap_pixels=3)]}
    with pytest.raises(RuntimeError,match='stop'):
        with entry.adapted_trainer():
            assert base.dataset_for(None,1)=='new_dataset'
            result=base.update(None,None,None,None,batch,None,0)
            assert result['loss_g']==3.5 and result['unchanged_sentinel']==9
            assert result['new_missing_GT_points']==2
            raise RuntimeError('stop')
    assert events==['old_update'] and base.update is original_update
    assert base.dataset_for is old_dataset and base.source_identity is old_identity
    assert base.monitor is old_monitor and base.collate_boundary_rgb is old_collate


def test_paired_monitor_preserves_weights_rng_mode_and_Gt(tmp_path):
    processed,cfg,_=make(tmp_path)
    cfg['boundary_naf']['monitor']={'pair_indices':[0],'thumbnail_width':80,'perturb_pair_indices':[0]}
    ds=BoundaryPerturbDataset(processed,cfg,epoch=0)
    g=BoundaryRGBNAFNet(width=4,encoder_blocks=(1,1),middle_blocks=1,
        decoder_blocks=(1,1),input_mode='boundary',seed=31)
    before=state_sha256(g);rng=torch.get_rng_state().clone()
    target=ds.sample_variant(0,'clean')['target'].clone()
    result=entry.save_paired_monitor(g,cfg,tmp_path/'monitor_output',0,ds)
    assert state_sha256(g)==before and torch.equal(rng,torch.get_rng_state()) and g.training
    assert torch.equal(target,ds.sample_variant(0,'clean')['target'])
    assert len(result['rows'])==1 and result['rows'][0]['variant']=='moderate'
    files=list((tmp_path/'monitor_output').rglob('*.jpg'))
    assert len(files)==1 and files[0].stat().st_size>0
    json.dumps(result,allow_nan=False)


def test_no_test_configuration_does_not_read_caches(tmp_path,monkeypatch):
    cfg=valid_config();cfg['boundary_naf']['monitor']['control_manifests']=[]
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('No test cache was configured'))
    assert entry.save_test_monitor(None,cfg,tmp_path,0)['no_test_cache_configured']


def test_test_monitor_is_unaugmented_input_only_without_WS(tmp_path,monkeypatch):
    cache=tmp_path/'test.npz';size=300
    boundary=np.zeros((size,size),np.float32);boundary[:,145:152]=.9
    rgb=np.full((size,size,3),100,np.uint8)
    np.savez_compressed(cache,boundary=boundary,rgb=rgb)
    row={'source':'test_169','path':str(cache),'sha256':base.file_sha(cache)}
    manifest=tmp_path/'control.json';manifest.write_text('{}')
    cfg=valid_config();cfg['boundary_naf']['monitor'].update(control_manifests=[str(manifest)],source_names=['test_169'])
    monkeypatch.setattr(entry,'_fixed_controls',lambda *a:[(row,{})])
    g=BoundaryRGBNAFNet(width=4,encoder_blocks=(1,1),middle_blocks=1,decoder_blocks=(1,1),input_mode='boundary',seed=31)
    before=state_sha256(g)
    result=entry.save_test_monitor(g,cfg,tmp_path/'out',0)
    assert result['no_test_GT'] and result['no_test_augmentation'] and result['no_WS']
    assert result['rows'][0]['crop_xywh']==[22,22,256,256]
    assert state_sha256(g)==before and g.training
    assert (tmp_path/'out/monitor/s000000/real_fixed/test_169.jpg').exists()
    cache.write_bytes(cache.read_bytes()+b'changed')
    with pytest.raises(RuntimeError,match='缓存改变'):entry.save_test_monitor(g,cfg,tmp_path/'later',1)
