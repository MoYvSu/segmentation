# -*- coding: utf-8 -*-
"""F0真实WS输入与独立零logit回放合同；无赛题数据或GPU。"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from tools import infer_flat_semantic as flat
from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
from utils.patch_diagnostic import BlendMap


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def baseline():
    return flat.entry.load_config(str(flat.ROOT/'config/inference/area_floor_200.yaml'))


def test_actual_WS_base128_and_F0_pre200_final200_exact_bridge():
    cfg=flat.execution_config(baseline())
    shape=(64,80)
    after=np.full(shape,.1,np.float32);after[:,39:42]=.95
    before=after.copy()
    raw=np.linspace(.15,.85,np.prod(shape),dtype=np.float32).reshape(shape)
    original=torch.linspace(-5,5,np.prod(shape)).reshape(1,1,*shape)
    semantic=flat.flat_geometry(original)
    real=cv2.watershed; observed=[]
    expected,stages=flat.expected_WS_image(after,cfg)

    def capture(image,markers):
        observed.append(image.copy())
        assert np.array_equal(image,expected)
        return real(image,markers)

    with patch.object(cv2,'watershed',new=capture):
        (ids,_),audit=flat.flat_supported_readout(semantic,after,before,cfg)
    assert len(observed)==1
    assert set(np.unique(observed[0]))=={128,166}  # 平底128；保留原barrier的30%白色叠加。
    assert audit['flat_semantic_proof']['actual_WS_elevation_exact']
    assert audit['flat_semantic_proof']['actual_WS_calls']==1
    # 按既有F0定义独立构造zeros_like(original semantic)，沿同一S0真实CPU重放。
    reflect=build_marker_variant(stages,'reflect',32)
    markers,_=flat.naf_entry.filter_unsupported_seeds(reflect['marker_labels'],before,low_threshold=.45)
    replay=decode_marker_replay(torch.from_numpy(np.zeros_like(original.numpy())),
                              torch.from_numpy(after)[None,None],cfg,markers)
    assert np.array_equal(ids,replay['instances'])
    assert audit['readout_hashes']==replay['hashes']
    classes,_=flat.entry.views.revote_instances(ids,raw)
    other_classes,_=flat.entry.views.revote_instances(replay['instances'],raw)
    current=flat.entry.finalize_instance_area(dict(instances=ids,classes=classes),cfg)
    other=flat.entry.finalize_instance_area(dict(instances=replay['instances'],classes=other_classes),cfg)
    assert np.array_equal(current['instances'],other['instances']) and current['classes']==other['classes']
    flat.entry.verify_prediction(current['instances'],current['classes'],shape)


class Encoder(nn.Module):
    def forward(self,image):return [image]


class Affinity(nn.Module):
    def forward(self,features):
        return {'affinity_logits':features[0][:,:1,::2,::2].expand(-1,8,-1,-1).contiguous()}


class Semantic(nn.Module):
    def forward(self,features,image):return image[:,:1]*2+.5


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder,self.affinity_decoder,self.semantic_decoder=Encoder(),Affinity(),Semantic()
        self.semantic_lora=self.geometry_feature_adapter=self.geometry_highres_refiner=None

    def forward(self,*args):raise AssertionError('No whole/global backend forward')


class Restorer(nn.Module):
    num_steps=16
    def q_sample(self,image,condition,step,noise):
        assert not torch.count_nonzero(noise)
        return condition
    def denoise(self,state,condition,step):return condition+.01


def test_only_raw_vote_and_local_zero_forward_and_final200_once(monkeypatch):
    raw_entry=flat.raw_entry
    rgb=np.arange(8*12*3,dtype=np.uint8).reshape(8,12,3)
    boxes=[(0,0,8,8),(0,4,8,12)]
    tensor=lambda crop:torch.from_numpy(crop.copy()).permute(2,0,1).float()[None]/255
    observed=dict(votes=0,final200=0)
    monkeypatch.setattr(raw_entry,'prepare_image',lambda *args:(rgb,tensor(rgb),0,0))
    monkeypatch.setattr(raw_entry,'tensor_from_rgb',lambda crop,device:(tensor(crop),0,0))
    monkeypatch.setattr(raw_entry,'tile_boxes',lambda *args:boxes)
    monkeypatch.setattr(raw_entry.backend_entry,'prediction_maps',lambda *a,**k:(_ for _ in ()).throw(AssertionError('global restore forbidden')))

    def supported(semantic,after,before,cfg):
        image,_=flat.expected_WS_image(after,cfg)
        return (np.ones(after.shape,np.uint16),{'1':0}),dict(readout_audit=dict(actual_watershed_calls=1,original_elevation_exact=True),
            readout_hashes=dict(elevation_sha256=flat.naf_entry.array_sha(image)))

    def decode(semantic,boundary,cfg,**kwargs):
        assert not torch.count_nonzero(semantic)
        return flat.naf_entry._supported_readout(semantic,boundary[0,0].numpy(),boundary[0,0].numpy(),cfg)[0]

    monkeypatch.setattr(flat.naf_entry,'_supported_readout',supported)
    monkeypatch.setattr(flat.entry,'decode_reflect',decode)
    original_vote,original_final=flat.entry.views.revote_instances,flat.entry.finalize_instance_area

    def vote(ids,prob):
        observed['votes']+=1;observed['probability']=prob.copy()
        return original_vote(ids,prob)

    def finalize(result,cfg):
        observed['final200']+=1
        return original_final(result,cfg)

    monkeypatch.setattr(flat.entry.views,'revote_instances',vote)
    monkeypatch.setattr(flat.entry,'finalize_instance_area',finalize)
    records,windows,fixed=[],[],{}
    model,restorer=Backend().eval(),Restorer().eval()
    _,result=flat.flat_predict(records)(model,restorer,None,Path('test_169.jpg'),baseline(),'cpu',local_audit=windows,fixed_audit=fixed)
    assert observed['votes']==observed['final200']==1 and result['classes']=={'1':1}
    expected_probability=(tensor(rgb)[:,:1]*2+.5).sigmoid()[0,0].numpy()
    assert np.array_equal(observed['probability'],expected_probability)
    row=records[0]
    assert {key:row[key] for key in flat.COUNT_KEYS}==dict(encoder_calls=3,global_encoder_calls=0,
        local_encoder_calls=2,raw_encoder_calls=1,global_zero_calls=0,local_zero_calls=2,
        semantic_head_calls=1,affinity_head_calls=2,global_model_calls=0,final200_calls=1)
    assert all(w['noise_nonzero']==0 and w['initial_equals_condition'] for w in windows)
    assert fixed['geometry_semantic_sha256']==flat.naf_entry.array_sha(np.zeros((1,1,8,12),np.float32))
    assert row['flat_WS_audit']['actual_WS_elevation_exact']
    blend=BlendMap(rgb.shape[:2])
    for y,x,y1,x1 in boxes:
        restored,_=raw_entry.restoration_probe.restore_probe(restorer,tensor(rgb[y:y1,x:x1]),0,'zero')
        boundary=raw_entry.boundary_grid(model,restored,baseline())
        boundary=raw_entry.crop_letterbox_output(boundary,1024,0,0,(y1-y,x1-x))[0,0].numpy()
        blend.add((y,x,y1,x1),boundary)
    assert np.array_equal(result['boundary'],blend.finish()[0])


def test_report_preserves_baseline_and_labels_flat_geometry_raw_vote():
    cfg=baseline()
    document=dict(format='old',config=cfg,runtime={'restoration_noise':{'name':'old'}},geometry='old',
                  images=[dict(source='test_169',boundary_naf={'raw_probability_sha256':'raw'})])
    before=deepcopy(document)
    row=dict(source='test_169',**{key:0 for key in flat.COUNT_KEYS})
    value=flat.audit_document(document,[row],'entry_sha')
    assert document==before and value['baseline_config']==cfg
    assert value['flat_semantic']['WS_semantic_base_uint8']==128
    assert value['flat_semantic']['global_restoration']=='not_computed'
    assert value['images'][0]['boundary_naf']['raw_probability_role']==flat.VOTE_ROLE
    assert flat.audit_document(value,[row],'entry_sha')==value


def test_reject_nonoriginal_model_or_semantic_delta():
    for change in (lambda c:c['submission']['models']['backend'].update(sha256='trained'),
                   lambda c:c['submission'].update(semantic_head_delta={'file':'new.pt'}),
                   lambda c:c['inference'].update(sem_edge_merge_weight=.1)):
        cfg=baseline();change(cfg)
        with pytest.raises(ValueError):flat.execution_config(cfg)
