# -*- coding: utf-8 -*-
"""保守关系标签与无GT输入通道契约。"""
import numpy as np

from utils.boundary_core_supervision import build_trusted_cores
from utils.boundary_relation_supervision import (
    IGNORE,KEEP,MERGE,adjacent_region_pairs,label_region_pair,native_relation_patch,original_components,
)

def scene():
    ids=np.ones((128,128),np.uint16);ids[:,65:]=2;ids[:,64]=0
    gt=np.ones_like(ids);original=np.ones(ids.shape,np.uint8);filled=np.zeros_like(original);unknown=np.zeros_like(original)
    return ids,gt,original,filled,unknown

def label(values):
    ids,gt,original,filled,unknown=values
    cores=build_trusted_cores(gt,original,radius=4,min_component_pixels=64)
    components,_=original_components(gt,original)
    pairs,_=adjacent_region_pairs(ids);assert len(pairs)==1
    return label_region_pair(ids,pairs[0],gt,original,filled,unknown,cores['core_map'],components),pairs[0]

def test_same_original_component_merge():
    result,pair=label(scene());assert result['label']==MERGE
    assert result['merge_corridor']['coverage']==1.
    assert len(pair['contact_flat'])>len(pair['seam_flat'])

def test_distinct_GT_keep_even_filled_seam():
    values=scene();values[1][:,65:]=2;values[2][:,64]=0;values[3][:,64]=1
    result,_=label(values);assert result['label']==KEEP

def test_single_unknown_cross_contact_is_not_hidden_in_95percent():
    values=scene();values[1][60,64]=0;values[2][60,64]=0;values[4][60,64]=1
    result,_=label(values);assert result['label']==IGNORE
    assert result['reason']=='merge_actual_contact_unknown_filled_or_other_GT'

def test_peripheral_corridor_tolerance_does_not_require_all_core_pixels():
    values=scene();values[1][60,66]=0;values[2][60,66]=0;values[4][60,66]=1
    result,_=label(values);assert result['label']==MERGE
    assert .95<=result['merge_corridor']['coverage']<1.

def test_coverage_denominator_keeps_untrusted_pixels():
    values=scene();ids,gt,original,filled,unknown=values
    cores=build_trusted_cores(gt,original,radius=4,min_component_pixels=64)
    cores['core_map'][:64]=0
    components,_=original_components(gt,original);pairs,_=adjacent_region_pairs(ids)
    result=label_region_pair(ids,pairs[0],gt,original,filled,unknown,cores['core_map'],components)
    assert result['label']==IGNORE and result['side_A']['coverage']<.8

def test_wide_zero_gap_does_not_create_pair():
    ids=np.ones((64,64),np.uint16);ids[:,34:]=2;ids[:,32:34]=0
    pairs,_=adjacent_region_pairs(ids);assert not pairs

def test_input_valid_only_reflects_image_extent_and_never_GT():
    values=scene();ids=values[0];pairs,_=adjacent_region_pairs(ids)
    rgb=np.zeros((*ids.shape,3),np.uint8);before=np.zeros(ids.shape,np.float32);after=np.ones_like(before)
    result=native_relation_patch(rgb,before,after,ids,pairs[0],(0,0),size=256)
    assert result['rgb'].dtype==np.uint8 and result['maps'].dtype==np.float16
    assert result['masks'].shape==(4,256,256)
    assert result['masks'][3].sum()==128*128
    assert result['masks'][3,128:,128:].all() and not result['masks'][3,:128].any()
    assert set(np.unique(result['masks']))<={0,1}
def test_unlabeled_monitor_is_separate_and_native_valid_only(tmp_path):
    import hashlib
    import cv2
    import pytest
    from tools.build_boundary_relation import build_unlabeled_monitors
    ids=np.ones((64,64),np.uint16);ids[:,32]=0;ids[:,33:]=2
    image=np.full((64,64,3),180,np.uint8)
    path=tmp_path/'image.png';assert cv2.imwrite(str(path),image)
    arrays=tmp_path/'cache.npz';np.savez(arrays,N1=np.full(ids.shape,.3,np.float32),NAF=np.full(ids.shape,.7,np.float32),ids=ids)
    def entry(p,key=None):
        out=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if key is not None:out['key']=key
        return out
    source=dict(kind='monitor_unlabeled',source='test_169',image=entry(path),N1=entry(arrays,'N1'),NAF=entry(arrays,'NAF'),
        C_final200=entry(arrays,'ids'),monitor_ROIs=[dict(kind='fixed',box=[0,0,4,4])])
    output=tmp_path/'out';output.mkdir()
    shards,selected=build_unlabeled_monitors([source],tmp_path,{},output)
    assert len(shards)==1 and selected[0]['ROI_fallback_to_nearest_whole_C_pair']
    with np.load(output/shards[0]['path']) as data:
        assert data['labels'].tolist()==[255]
        assert data['masks'].shape==(1,4,256,256) and data['masks'][0,3].sum()==64*64
        assert set(np.unique(data['masks']))=={0,1}
    source['GT']=dict(path='forbidden')
    with pytest.raises(AssertionError):build_unlabeled_monitors([source],tmp_path,{},output)

