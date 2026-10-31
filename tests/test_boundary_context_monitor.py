# -*- coding: utf-8 -*-
"""RGB上下文分块、固定最终读出及monitor隔离契约；仅CPU玩具图。"""
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from models.boundary_context import BoundaryContextNet
from tools.boundary_context_monitor import predict_context_residual_tiled, context_pair_metrics
from tools.boundary_restore_readout import predict_residual_tiled
from utils.config import load_config


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_rgb_alignment_survives_reflect_tiling_and_small_padding():
    rgb=np.arange(19*23*3,dtype=np.uint16).reshape(19,23,3).astype(np.uint8)
    boundary=(rgb[:,:,0].astype(np.float32)/255).copy()
    class Check(torch.nn.Module):
        def residual(self,heat,color):
            assert torch.equal(heat,color[:,:1])
            return torch.zeros_like(heat)
    model=Check().train()
    output,audit=predict_context_residual_tiled(boundary,rgb,model,tile_size=16,overlap=5,reflect_pad=3)
    assert output is boundary and audit['all_zero_residual'] and model.training


def test_rgb_ignored_residual_matches_original_tiled_algorithm():
    generator=np.random.default_rng(7)
    boundary=generator.uniform(.2,.8,(41,47)).astype(np.float32)
    rgb=generator.integers(0,256,(*boundary.shape,3),dtype=np.uint8)
    class Old(torch.nn.Module):
        def residual(self,value):return (value-.5)*.1
    class Context(Old):
        def residual(self,value,color):return super().residual(value)
    a,_=predict_residual_tiled(boundary,Old(),tile_size=32,overlap=8,reflect_pad=4)
    b,_=predict_context_residual_tiled(boundary,rgb,Context(),tile_size=32,overlap=8,reflect_pad=4)
    assert np.array_equal(a,b)


def test_strength_zero_bypasses_model_and_bad_alignment_rejected():
    boundary=np.full((9,13),.25,np.float32);rgb=np.zeros((9,13,3),np.uint8)
    output,audit=predict_context_residual_tiled(boundary,rgb,object(),strength=0)
    assert output is boundary and audit['bypass']
    with pytest.raises(ValueError,match='aligned'):
        predict_context_residual_tiled(boundary,rgb[:-1],object(),strength=0)


def test_pristine_is_exact_not_old_threshold_mask_complement():
    target=np.full((8,8),.4,np.float32);source=target.copy();source[3,3]-=.0001
    valid=np.ones_like(source,bool);empty=np.zeros_like(source,bool)
    result=context_pair_metrics(source,target,source,valid,empty,empty)
    assert result['untouched_pixels']==64 and result['pristine_pixels']==63


def test_monitor_passes_rgb_and_checks_final_png_classes(tmp_path,monkeypatch):
    import train_boundary_context as trainer
    import tools.boundary_context_monitor as monitoring
    config=load_config('config/train/boundary_context.yaml')
    cfg=config['boundary_restore'];cfg['crop_size']=16
    cfg['monitor'].update(pair_indices=[0],thumbnail_width=32,tile_overlap=4,
                          required_control_sources=['test_101'])
    image=np.zeros((16,20,3),np.uint8);image[:,:,0]=128
    boundary=np.full((16,20),.2,np.float32)
    instances=np.ones((16,20),np.uint16);classes={'1':1}
    cache=tmp_path/'control.npz'
    np.savez(cache,boundary=boundary,rgb=image,instances=instances,
             semantic=np.zeros_like(boundary),raw_probability=np.ones_like(boundary))
    png=tmp_path/'control.png';png.write_bytes(cv2.imencode('.png',instances)[1].tobytes())
    classfile=tmp_path/'classes.json';classfile.write_text(json.dumps(classes))
    frozen=tmp_path/'config.json';frozen.write_text('{}')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest=tmp_path/'controls.json'
    manifest.write_text(json.dumps(dict(complete=True,training=False,config=str(frozen),
        config_sha256=digest(frozen),files=[dict(source='test_101',path=str(cache),sha256=digest(cache),
        control_png=str(png),control_png_sha256=digest(png),control_classes=str(classfile),
        control_classes_sha256=digest(classfile))])))
    cfg['monitor']['control_manifest']=str(manifest)
    class Dataset:
        def set_epoch(self,epoch):assert epoch==1
        def __getitem__(self,index):
            value=torch.full((1,16,16),.2)
            return dict(input=value,target=value.clone(),valid=torch.ones_like(value),
                gap_mask=torch.zeros_like(value),noise_mask=torch.zeros_like(value),
                context_rgb=torch.full((3,16,16),.5),identity=torch.tensor(True),source='toy',meta={})
    monkeypatch.setattr(trainer,'build_dataset',lambda config,seed_offset:Dataset())
    def readout(semantic,raw,heat,cfg):
        assert np.array_equal(heat,boundary)
        return instances.copy(),dict(classes),{'fixed':True}
    monkeypatch.setattr(monitoring,'evaluate_frozen_readout',readout)
    model=BoundaryContextNet(width=4).train()
    before=torch.get_rng_state().clone()
    report=monitoring.monitor(model,config,tmp_path/'out',0,'cpu')
    assert model.training and torch.equal(before,torch.get_rng_state())
    assert report['rows'][0]['pristine_after_mae']==0
    assert report['rows'][1]['identity_epoch_exact'] and report['rows'][1]['png_pixels_exact']
    assert (tmp_path/'out/monitor/e000/test_101_whole.jpg').is_file()
