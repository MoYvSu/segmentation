# -*- coding: utf-8 -*-
"""验证拼接覆盖、分类别度量和已知像素参考，不用实现自身作答案。"""
import numpy as np
import pytest
from utils.patch_diagnostic import tile_boxes,BlendMap,phase_description,trusted_phase_metrics,reconstruction_metrics


@pytest.mark.parametrize('shape,size', [((107,233),64),((31,89),128),((128,128),128)])
def test_spatial_ramp_reconstructs_without_seams(shape,size):
    y,x=np.indices(shape);source=(.2*y/shape[0]+.8*x/shape[1]).astype(np.float32)
    blend=BlendMap(shape)
    for box in tile_boxes(shape,size):
        y,x,y1,x1=box;blend.add(box,source[y:y1,x:x1])
    result,audit=blend.finish()
    np.testing.assert_allclose(result,source,atol=2e-7)
    assert audit['covered_fraction']==1


def test_missing_tile_is_rejected():
    blend=BlendMap((100,100));blend.add((0,0,50,50),np.ones((50,50)))
    with pytest.raises(ValueError):blend.finish()


def test_pearlite_fragmentation_does_not_change_ferrite_area():
    gt=np.ones((20,30),np.uint16);gt[:,15:]=2
    pred=gt.copy();pred[:10,15:]=3
    classes={1:1,2:0};pc={'1':1,'2':0,'3':0}
    stats=phase_description(pred,pc)
    assert stats['ferrite']['mean_area']==300 and stats['ferrite']['instances']==1
    metrics=trusted_phase_metrics(gt,classes,np.ones_like(gt,bool),pred,pc)
    assert metrics['ferrite']['matched_iou']==1 and metrics['pearlite']['split_gt']==1


def test_unknown_prediction_never_becomes_false_positive():
    gt=np.ones((20,30),np.uint16);gt[:,15:]=0;pred=gt.copy();pred[:,15:]=2
    result=trusted_phase_metrics(gt,{1:1},gt>0,pred,{'1':1,'2':0})
    assert result['pearlite']['pred_instances_in_known_domain']==0 and result['ferrite']['matched_iou']==1


def test_identity_reconstruction_and_hallucinated_edge():
    image=np.zeros((40,40,3),np.float32);image[:,20:]=1
    same=reconstruction_metrics(image,image)
    assert same['rgb_mse']==0 and same['gradient_l1']==0 and same['novel_edge_fraction']==0
    changed=image.copy();changed[:,5:8]=1
    assert reconstruction_metrics(image,changed)['novel_edge_fraction']>0
