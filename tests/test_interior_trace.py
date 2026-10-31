# -*- coding: utf-8 -*-
"""冻结界面恢复的关键边界条件。"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace
import tools.probe_interior_trace as runner
import tools.probe_affinity_interior as old_helper
from tools.probe_interior_trace import restore_interface, scalar_stats, rebased_tiles


def test_restore_only_fixed_original_support():
    before=np.full((3,4),.2,np.float32);reference=np.full((3,4),.8,np.float32)
    gt=np.ones((3,4),np.int32);gt[0,0]=0
    original=gt>0;original[2,3]=False
    mask=np.zeros((3,4),bool);mask[1,1]=True
    result,audit=restore_interface(before,reference,mask,gt,original)
    assert result[1,1]==reference[1,1]
    assert np.array_equal(result[~mask],before[~mask])
    assert audit['provenance']['filled_pixels']==0 and audit['changed_pixels']==1
    assert np.all(before==np.float32(.2))


def test_restore_rejects_unknown_and_records_optimistic_filled():
    gt=np.array([[0,1],[1,2]],np.int32);original=np.array([[False,True],[True,False]])
    a=np.zeros((2,2),np.float32);b=np.ones((2,2),np.float32)
    mask=gt>0
    _,audit=restore_interface(a,b,mask,gt,original)
    assert audit['provenance']['filled_pixels']==1
    mask[:]=True
    with pytest.raises(ValueError):restore_interface(a,b,mask,gt,original)


def test_restore_preserves_fp32_and_rejects_nan():
    gt=np.ones((2,2),np.int32);original=gt>0;a=np.zeros((2,2),np.float32)
    with pytest.raises(ValueError):restore_interface(a.astype(np.float64),a,original,gt,original)
    b=a.copy();b[0,0]=np.nan
    with pytest.raises(ValueError):restore_interface(a,b,original,gt,original)


def test_scalar_stats_empty_and_threshold():
    a=np.array([[.2,.5,.7]],np.float32)
    assert scalar_stats(a,np.zeros(a.shape,bool))['mean'] is None
    stats=scalar_stats(a,np.ones(a.shape,bool))
    assert stats['above_low']==2 and stats['above_high']==1


def test_connect_empty_edit_all_windows_cannot_reuse_r1_native(monkeypatch):
    monkeypatch.setattr(runner,'fusion_options',lambda config:('short',{}))
    monkeypatch.setattr(old_helper,'fusion_options',lambda config:('short',{}))
    def crop(grid,image_size,pad_h,pad_w,size):
        return torch.nn.functional.interpolate(grid,size=size,mode='bilinear',align_corners=True)
    views=SimpleNamespace(crop_letterbox_output=crop)
    old_logits=torch.zeros((1,8,4,4),dtype=torch.float32)
    logits=torch.full_like(old_logits,3.)
    old_tile=dict(index=0,box=[0,0,8,8],pad_h=0,pad_w=0,logits=old_logits,
        logits_stride=list(old_logits.stride()),grid=np.full((1,1,4,4),.5,np.float32),
        native=np.full((1,1,8,8),.5,np.float32))
    tiles=rebased_tiles(views,[old_tile],[logits],{})
    assert not np.array_equal(tiles[0]['native'],old_tile['native'])
    boundary,_,_=old_helper.recompute_positive(views,tiles,[np.zeros((1,8,4,4),bool)],{},(8,8),empty_exact=True)
    expected=runner.BlendMap((8,8));expected.add(old_tile['box'],tiles[0]['native'][0,0])
    assert np.array_equal(boundary,expected.finish()[0])
    assert np.array_equal(old_tile['native'],np.full((1,1,8,8),.5,np.float32))
