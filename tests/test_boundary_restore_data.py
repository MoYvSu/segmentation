# -*- coding: utf-8 -*-
"""补边配对的 CPU 合成契约；不使用赛题样本或模型预测。"""
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from data.boundary_restore import (
    BoundaryRestoreDataset, collate_boundary_restore, known_instance_boundary,
    make_boundary_restore_pair, native_crop, reliable_region,
)


def split_ids(size=96):
    ids=np.ones((size,size),np.int32)
    ids[:,size//2:]=2
    return ids


def toy_manifest(tmp_path):
    ids=split_ids()
    ids[32:42,10:20]=0
    original=ids>0
    original[48:52,46:50]=False
    rgb=np.full((*ids.shape,3),127,np.uint8)
    rgb[:,48:]=211
    image=tmp_path/'train_001.png'
    cv2.imwrite(str(image),rgb)
    gt=tmp_path/'train_001_gt.npz'
    np.savez_compressed(gt,instance_map=ids,original_covered=original,
                        filled=(ids>0)&~original,residual_unknown=ids==0)
    classes=tmp_path/'train_001_class.json'
    # 两个相同P类别的不同实例仍只有GT交界，不读取亮度/类别生成纹理边界。
    classes.write_text(json.dumps({'1':0,'2':0}),encoding='utf-8')
    files=dict(image=image,gt=gt,classes=classes)
    row=dict(source='train_001',**{k:str(p) for k,p in files.items()},
             hashes={k+'_sha256':hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(rows=[row])),encoding='utf-8')
    return manifest,ids,original


def test_only_known_known_instance_changes_define_boundary():
    ids=split_ids()
    ids[20:35,20:30]=0
    b=known_instance_boundary(ids)
    assert b[:,47:49].all()
    assert not b[:,:47].any() and not b[:,49:].any()
    assert not known_instance_boundary(np.ones((32,32),np.int32)).any()


def test_unknown_and_crop_rim_ignore_not_background():
    ids=split_ids()
    ids[40:50,20:30]=0
    pair=make_boundary_restore_pair(ids,np.random.default_rng(1),'gap')
    valid=pair['valid'][0].numpy().astype(bool)
    assert not valid[:8].any() and not valid[-8:].any()
    assert not valid[:, :8].any() and not valid[:, -8:].any()
    assert not valid[32:58,12:38].any()
    assert (pair['gap_mask']<=pair['valid']).all()
    assert (pair['noise_mask']<=pair['valid']).all()
    assert (pair['target'][0,40:50,20:30]==0).all()


def test_identity_is_exact_and_damage_masks_empty():
    pair=make_boundary_restore_pair(split_ids(),np.random.default_rng(2),'identity')
    assert pair['identity'].item()
    assert torch.equal(pair['input'],pair['target'])
    assert not pair['gap_mask'].any() and not pair['noise_mask'].any()


def test_gap_mask_contains_erased_true_boundary_positive_examples():
    pair=make_boundary_restore_pair(split_ids(128),np.random.default_rng(3),'gap')
    gap=pair['gap_mask'].bool()
    assert gap.any() and not pair['identity'].item()
    assert (pair['target'][gap]>=.1).all()
    # 至少部分真交界被实际抑制，不只是记录一个未作用的ellipse。
    assert ((pair['target']>.8)&gap&(pair['input']<.3)).any()
    assert not pair['noise_mask'].any()


def test_false_noise_only_on_safe_true_negative_interior():
    pair=make_boundary_restore_pair(split_ids(128),np.random.default_rng(4),'noise')
    noise=pair['noise_mask'].bool()
    assert noise.any()
    assert (pair['target'][noise]<.01).all()
    assert (pair['input'][noise]>.02).all()
    assert not pair['gap_mask'].any()


def test_native_rotation_padding_alignment_and_no_resizing():
    ids=np.ones((24,40),np.int32)
    ids[:,20:]=2
    cropped,geometry=native_crop(ids,[0,0,24,40],64,rotation=1,flip=True)
    expected=np.rot90(ids)[:,::-1]
    np.testing.assert_array_equal(cropped[:40,:24],expected)
    assert not cropped[40:].any() and not cropped[:,24:].any()
    assert geometry['resize_scale']==1 and geometry['content_shape']==[40,24]
    pair=make_boundary_restore_pair(cropped,np.random.default_rng(5),'identity',valid_margin=3)
    assert not pair['valid'][0,40:].any() and not pair['valid'][0,:,24:].any()
    assert pair['valid'][0,:40,:24].sum()>0


def test_pair_reproducibility_uses_independent_rng():
    np.random.seed(123)
    global_state=np.random.get_state()
    a=make_boundary_restore_pair(split_ids(),np.random.default_rng(42),'gap')
    b=make_boundary_restore_pair(split_ids(),np.random.default_rng(42),'gap')
    for key in ('input','target','valid','gap_mask','noise_mask','identity'):
        assert torch.equal(a[key],b[key])
    assert a['meta']==b['meta']
    after=np.random.get_state()
    assert np.array_equal(global_state[1],after[1]) and global_state[2:]==after[2:]


def test_processed_manifest_filled_known_schedule_resume_and_collate(tmp_path):
    manifest,ids,original=toy_manifest(tmp_path)
    dataset=BoundaryRestoreDataset(manifest,crop_size=96,expected_sources=1,seed=8,
                                    identity_probability=1,gap_probability=0,noise_probability=0)
    assert len(dataset)==8 and dataset.plan_receipt()['source_counts']=={'train_001':8}
    first=dataset[0]
    assert first['meta']['native_filled_pixels']==int(((ids>0)&~original).sum())
    geometry=first['meta']['geometry']
    transformed,_=native_crop(((ids>0)&~original).astype(np.int32),[0,0,96,96],96,
                               geometry['rotation'],geometry['flip'])
    assert first['valid'][0].numpy()[transformed>0].all()
    dataset.set_epoch(2)
    second=dataset[0]
    dataset.set_epoch(1)
    again=dataset[0]
    assert torch.equal(first['input'],again['input']) and first['meta']==again['meta']
    assert first['meta']['draw_seed']!=second['meta']['draw_seed']
    batch=collate_boundary_restore([first,again])
    assert batch['input'].shape==(2,1,96,96) and batch['identity'].shape==(2,)
    assert isinstance(batch['meta'],list)
    dataset.verify_sources()
    json.dumps(dataset.identity,allow_nan=False)
    json.dumps(first['meta'],allow_nan=False)


def test_reject_changed_files_test_names_and_old_target(tmp_path):
    manifest,_,_=toy_manifest(tmp_path)
    dataset=BoundaryRestoreDataset(manifest,crop_size=96,expected_sources=1)
    classes=tmp_path/'train_001_class.json'
    classes.write_text(json.dumps({'1':1,'2':0}),encoding='utf-8')
    with pytest.raises(RuntimeError,match='source changed'):
        dataset.verify_sources()
    with pytest.raises(ValueError,match='SHA256 differs'):
        BoundaryRestoreDataset(manifest,crop_size=96,expected_sources=1)
    document=json.loads(manifest.read_text())
    document['rows'][0]['source']='test_101'
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='training sources'):
        BoundaryRestoreDataset(manifest,crop_size=96,expected_sources=1)


def test_empty_known_and_invalid_probability_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='reliable known'):
        make_boundary_restore_pair(np.zeros((64,64),np.int32),np.random.default_rng(1))
    manifest,_,_=toy_manifest(tmp_path)
    with pytest.raises(ValueError,match='sum to 1'):
        BoundaryRestoreDataset(manifest,expected_sources=1,identity_probability=.5)
