# -*- coding: utf-8 -*-
"""真实输入模式的关键接口：先整图细化再分窗，不重复幅值混合。"""
import numpy as np
import pytest
import torch
from torch import nn
from skimage.morphology import skeletonize

from tools.boundary_naf_monitor import canonical_input,predict_boundary_tiled,real_pair_metrics
from train_boundary_naf import validate_config
from utils.config import load_config


class Echo(nn.Module):
    def __init__(self):
        super().__init__();self.anchor=nn.Parameter(torch.zeros(()))
    def forward(self,x):return x+self.anchor*0


def test_global_skeleton_is_preserved_across_tiled_seams_and_reflection():
    image=np.zeros((93,117),np.float32)
    image[8:85,48:58]=.8;image[38:49,12:108]=.9
    image[63:74,60:113]=.7
    settings=dict(input_mode='skeleton',input_threshold=.5,tile_size=32,overlap=8,
                  reflect_pad=5,batch_size=4,iterations=1,revert_unconnected=False)
    before=image.copy();model=Echo().train()
    output,audit=predict_boundary_tiled(image,model,settings)
    expected=skeletonize(image>=.5).astype(np.float32)
    np.testing.assert_allclose(output,expected,atol=1e-6,rtol=0)
    np.testing.assert_array_equal(before,image)
    assert model.training and audit['global_skeletonization_before_tiling']
    assert audit['input_mode']=='skeleton'


def test_skeleton_mode_cannot_be_applied_to_each_batch_tensor():
    with pytest.raises(ValueError,match='完整原生'):
        canonical_input(torch.zeros(1,1,16,16),'skeleton',.5)


def test_disabled_skeleton_path_returns_unmodified_native_probability():
    image=np.full((40,53),.7,np.float32)
    output,audit=predict_boundary_tiled(image,None,dict(input_mode='skeleton'),enabled=False)
    assert output is image and audit['boundary_exact']


def test_real_input_config_explicitly_binds_global_skeleton_and_backend_hashes():
    config=load_config('config/train/boundary_naf_real.yaml')
    cfg=validate_config(config)
    assert cfg['inference']['input_mode']=='skeleton'
    assert cfg['real_input_models']['restorer']['sha256'].startswith('9315da')
    cfg['inference']['input_mode']='binary'
    with pytest.raises(ValueError,match='整图细化'):
        validate_config(config)


def test_real_bce_config_changes_only_gan_weight():
    previous=load_config('config/train/boundary_naf_real.yaml')['boundary_naf']
    candidate=validate_config(load_config('config/train/boundary_naf_real_bce.yaml'))
    previous['training']['adversarial_weight']=0.0
    assert candidate==previous


def test_real_pair_diagnostics_separate_preservation_missing_and_original_wrong_input():
    target=np.zeros((64,64),np.float32);target[8:56,16]=1
    source=target.copy();source[24:40,16]=0;source[8:56,46]=1
    valid=np.zeros_like(source,bool);valid[4:-4,4:-4]=True
    original=np.ones_like(valid)
    before,_=real_pair_metrics(source,target,np.full_like(source,.5),valid,original)
    after,wrong=real_pair_metrics(source,target,target,valid,original)
    assert before['natural_missing_gt_pixels']>0 and before['original_deep_input_pixels']==48
    assert after['supported_gt_probability']==1
    assert after['natural_missing_gt_probability']==1
    assert after['original_deep_input_probability']==0
    assert wrong[:,46].sum()==48
    original[:,46]=False
    restricted,_=real_pair_metrics(source,target,target,valid,original)
    assert restricted['original_deep_input_pixels']==0
