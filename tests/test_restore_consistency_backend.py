# -*- coding: utf-8 -*-
"""局部恢复器注入、八种子统计与宏平均的CPU门禁。"""
import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools.probe_restore_consistency import (
    OFFSETS, contract, local_boundaries, macro_statistics, series_statistics, validate_cache,
)


def case(value, classes=None):
    ids=np.array([[1,1],[2,2]],dtype=np.uint16)
    return dict(boundary=np.full((2,2),value,dtype=np.float32),
                geometry_semantic=np.zeros((2,2),dtype=np.float32),
                raw_probability=np.full((2,2),.7,dtype=np.float32),instances=ids), classes or {'1':1,'2':0}


def test_series_uses_own_s0_and_predeclared_eight_seeds():
    names=[f'candidate_s{i}' for i in range(8)]
    cases={name:case(.4 if i%2==0 else .7,{'1':1,'2':1} if i%2 else None) for i,name in enumerate(names)}
    stats,std=series_statistics(cases,names)
    assert stats['complete'] and stats['sample_count']==8
    assert stats['reference_case']=='candidate_s0'
    assert np.allclose(std,.15)
    assert stats['threshold_any_seed_crossing']=={'0.45':1.,'0.65':1.}
    assert stats['each_seed_vs_own_s0']['candidate_s0']['boundary']['mae']==0
    assert stats['ferrite_count_range']==[1,2]
    assert stats['ferrite_cv']['count']==pytest.approx(1/3)
    assert stats['each_seed_vs_own_s0']['candidate_s1']['instances']['matched_class_changes']==1
    assert stats['boundary_mean_distribution']['mean']==pytest.approx(.55)
    with pytest.raises(ValueError,match='eight'):
        series_statistics({key:value for key,value in cases.items() if key!=names[-1]},names)


def test_macro_weights_source_images_equally_instead_of_pixel_area():
    names=[f'old_s{i}' for i in range(8)]
    stable,_=series_statistics({name:case(.2) for name in names},names)
    varied,_=series_statistics({name:case(.2 if i%2 else .6) for i,name in enumerate(names)},names)
    sources={'tiny':{'series':{g:stable for g in ('old','control','candidate')}},
             'huge':{'series':{g:varied for g in ('old','control','candidate')}}}
    result=macro_statistics(sources)
    assert result['old']['source_count']==2
    assert result['old']['boundary_std_mean']['mean']==pytest.approx(.1)
    assert result['old']['boundary_std_mean']['valid_sources']==2
    broken=copy.deepcopy(sources);broken['huge']['series']['old']['complete']=False
    with pytest.raises(ValueError,match='incomplete'):
        macro_statistics(broken)


def test_local_object_and_noise_schedule_cannot_modify_global_path():
    local_restorer=object();backend=object();calls=[]
    config={'affinity_native':{'overlap':.25}}
    rgb=np.zeros((4,6,3),dtype=np.uint8)
    image=torch.zeros(1,3,4,6)
    def restore(restorer,value,seed,mode):
        assert restorer is local_restorer and mode=='single'
        calls.append(seed)
        return value+float(seed%2),None
    def boundary(model,restored,actual_config):
        assert model is backend and actual_config is config
        return restored[:,:1]
    with patch('tools.probe_restore_consistency.tensor_from_rgb',return_value=(image,0,0)), \
         patch('tools.probe_restore_consistency.restore_probe',side_effect=restore), \
         patch('tools.probe_restore_consistency.boundary_grid',side_effect=boundary), \
         patch('tools.probe_restore_consistency.crop_letterbox_output',side_effect=lambda value,*args:value):
        boundaries,boxes=local_boundaries(backend,local_restorer,rgb,config,314191,device='cpu')
    assert calls==[314191+100000+1024*1000+offset for offset in OFFSETS]
    assert len(boundaries)==8 and boxes==[(0,0,4,6)]
    assert all(value.shape==(4,6) for value in boundaries)


def test_cache_contract_ignores_only_machine_root_and_rejects_backend_changes(tmp_path):
    path=tmp_path/'backend.pt';path.write_bytes(b'fixed')
    import hashlib
    checksum=hashlib.sha256(b'fixed').hexdigest()
    config={'paths':{'project_root':'/new'},'inference':{'boundary_threshold':.65}}
    old=copy.deepcopy(config);old['paths']['project_root']='/old'
    metadata=dict(format='restoration_noise_probe_v1',complete=True,offsets=list(OFFSETS),
                  config=old,sources=['test_101'],models={'backend':{'sha256':checksum}},
                  images={'test_101':{'isolation_passed':True,'cases':{'control':{'formal_control_exact':True}}}})
    validate_cache(metadata,config,{'backend':path},['test_101'])
    changed=copy.deepcopy(config);changed['inference']['boundary_threshold']=.6
    with pytest.raises(ValueError,match='configuration'):
        validate_cache(metadata,changed,{'backend':path},['test_101'])
    assert contract(old)==contract(config)
