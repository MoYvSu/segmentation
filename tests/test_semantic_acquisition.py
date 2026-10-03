# -*- coding: utf-8 -*-
"""有界原生采集退化的常量、空间场、几何尺度和复现门禁。"""
from pathlib import Path

import numpy as np
import pytest

from tools.probe_semantic_acquisition import CONDITIONS, INITIAL, native_blur_variants
from utils.config import load_config


@pytest.mark.parametrize('level',[0,173,255])
def test_constant_image_preserved_and_bounded_native_network_receipt(level):
    rgb=np.full((64,96,3),level,np.uint8)
    variants,meta=native_blur_variants(rgb,seed=7)
    assert set(variants)==set(CONDITIONS[1:4])
    assert meta['sigmas_native']==[3.,8.]
    assert meta['sigmas_network']==[32.,8*1024/96]
    assert meta['border']=='BORDER_REFLECT'
    for image in variants.values():
        assert image.shape==rgb.shape and image.dtype==np.float32
        assert np.all((image>=0)&(image<=1))
        np.testing.assert_allclose(image,level/255,atol=4e-7,rtol=0)


def test_blur_applies_before_resize_and_spatial_stays_between_endpoints():
    rgb=np.zeros((80,120,3),np.uint8)
    rgb[:,::2]=255
    variants,meta=native_blur_variants(rgb,seed=13)
    low,high,mixed=[variants[k] for k in CONDITIONS[1:4]]
    assert float(low.std())<float(rgb.astype(np.float32).std()/255)*.2
    assert np.all(mixed>=np.minimum(low,high)-1e-7)
    assert np.all(mixed<=np.maximum(low,high)+1e-7)
    assert meta['spatial_field']['blend_min']==0 and meta['spatial_field']['blend_max']==1
    assert meta['nominal_second_moment_sigma_native']['0']==3
    assert meta['nominal_second_moment_sigma_native']['1']==8
    assert not np.array_equal(mixed,low) and not np.array_equal(mixed,high)


def test_seed_replay_and_invalid_recipe_rejected():
    rgb=np.random.default_rng(0).integers(0,256,(40,64,3),dtype=np.uint8)
    first,a=native_blur_variants(rgb,seed=18)
    second,b=native_blur_variants(rgb,seed=18)
    assert a==b
    for key in first:assert np.array_equal(first[key],second[key])
    _,other=native_blur_variants(rgb,seed=19)
    assert other['spatial_field']['blend_sha256']!=a['spatial_field']['blend_sha256']
    with pytest.raises(ValueError,match='Fixed bounded'):
        native_blur_variants(rgb,seed=18,sigmas=(3.,12.))
    with pytest.raises(ValueError,match='uint8 RGB'):
        native_blur_variants(rgb.astype(np.float32),seed=18)


def test_config_only_fixed_training_sources_and_no_extra_stress_views():
    cfg=load_config(str(Path(__file__).parents[1]/'config/inference/semantic_acquisition.yaml'))
    assert cfg['semantic_acquisition']['sources']==list(INITIAL)
    assert cfg['semantic_acquisition']['native_sigmas']==[3.,8.]
    assert len(CONDITIONS)==5
    assert cfg['semantic_acquisition']['global_photometric_view']==0
    assert not Path(cfg['semantic_acquisition']['output_dir']).is_absolute()
