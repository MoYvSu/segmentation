# -*- coding: utf-8 -*-
"""区域分隔配对的独立构造契约；不读取赛题图或权重。"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.boundary_naf_real import BoundaryNAFRealDataset
from data.boundary_naf_region import (
    BoundaryNAFRegionDataset, REGION_KEYS, collate_boundary_naf_region,
)


MODELS = {'backend':'1' * 64, 'restorer':'2' * 64}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def fixture_files(tmp_path, *, shape=(96, 128), ids=None, original=None, boundary=None):
    if ids is None:
        ids = np.full(shape, 7, np.int32)
        ids[:, shape[1]//2:] = 19
        ids[12:20, 10:18] = 0
    if original is None:
        original = ids > 0
        original[:, shape[1]//2-2:shape[1]//2+2] = False
    if boundary is None:
        boundary = np.zeros(shape, np.float32)
        boundary[:, shape[1]//2-4] = 1.
        boundary[32, 8:min(40, shape[1])] = 1.
        boundary[14:18, 12:16] = 1.
    rows, heats = [], []
    for i in range(1, 3):
        name = f'train_{i:03d}'
        image = tmp_path/f'{name}.jpg'
        image.write_bytes(b'identity-only-not-a-real-image')
        gt = tmp_path/f'{name}_gt.npz'
        np.savez_compressed(gt, instance_map=ids, original_covered=original,
                            filled=(ids > 0) & ~original, residual_unknown=ids == 0)
        classes = tmp_path/f'{name}_classes.json'
        _write(classes, {str(int(g)):int(g % 2) for g in np.unique(ids) if g > 0})
        paths = dict(image=image, gt=gt, classes=classes)
        row = dict(source=name, shape=list(ids.shape), **{k:str(v) for k,v in paths.items()},
                   hashes={k+'_sha256':_sha(v) for k,v in paths.items()})
        heat = tmp_path/f'{name}_heat.npz'
        np.savez_compressed(heat, boundary=boundary)
        rows.append(row)
        heats.append(dict(source=name, heat_path=heat.name, heat_sha256=_sha(heat),
                          shape=list(ids.shape), **row['hashes']))
    gt_manifest, heat_manifest = tmp_path/'processed.json', tmp_path/'real.json'
    _write(gt_manifest, dict(rows=rows))
    _write(heat_manifest, dict(format='boundary_naf_real_heatmaps_v1', complete=True,
        no_test_training=True, local_restoration={'mode':'zero', 'model':'restorer'},
        models={name:dict(sha256=value) for name,value in MODELS.items()}, rows=heats))
    config = dict(boundary_naf=dict(seed=20261026, real_input_manifest=str(heat_manifest),
        real_input_models=MODELS.copy(),
        data=dict(expected_sources=2, crop_size=64, stride=32, augmentations=8,
                  samples_per_epoch=32, valid_margin=4, crop_margin=4, identity_probability=.2)))
    return gt_manifest, config, ids, original


def _forced(dataset, box=(0, 0, 64, 64), rotation=0, flip=False):
    dataset.plan[0] = dict(dataset.plan[0], source='train_001', box=list(box),
                          rotation=rotation, flip=flip, augmentation=rotation + 4 * int(flip))
    return dataset[0]


def _expected(array, box, rotation, flip, size=64):
    y,x,h,w = box
    value = np.rot90(array[y:y+h, x:x+w], rotation)
    if flip:
        value = value[:, ::-1]
    if value.shape != (size, size):
        mode = 'reflect' if min(value.shape) > 1 else 'edge'
        value = np.pad(value, ((0,size-value.shape[0]), (0,size-value.shape[1])), mode=mode)
    return value


def test_parent_sampling_plan_inputs_targets_and_valid_are_unchanged(tmp_path):
    manifest, cfg, _, _ = fixture_files(tmp_path)
    parent = BoundaryNAFRealDataset(manifest, cfg, epoch=1)
    region = BoundaryNAFRegionDataset(manifest, cfg, epoch=1)
    assert parent.plan == region.plan
    assert parent.plan_receipt() == region.plan_receipt()
    for i in range(len(region)):
        a,b = parent[i],region[i]
        for key in ('input','target','valid','gap_mask','identity'):
            assert torch.equal(a[key],b[key])
        assert a['meta']['geometry'] == b['meta']['geometry']
    old = region.plan_receipt()
    region.set_epoch(2)
    assert region.plan_receipt() != old
    region.set_epoch(1)
    assert region.plan_receipt() == old


@pytest.mark.parametrize('rotation,flip', [(0,False),(1,False),(2,True),(3,True)])
def test_integer_ids_original_and_core_follow_exact_native_d4(tmp_path, rotation, flip):
    manifest,cfg,ids,_ = fixture_files(tmp_path)
    dataset = BoundaryNAFRegionDataset(manifest,cfg)
    box=(16,32,64,64)
    sample=_forced(dataset,box,rotation,flip)
    expected=_expected(ids,box,rotation,flip)
    np.testing.assert_array_equal(sample['region_ids'][0].numpy(),expected)
    assert sample['region_ids'].dtype == sample['region_core_ids'].dtype == torch.int64
    assert set(torch.unique(sample['region_ids']).tolist()) == {7,19}
    valid=sample['valid'][0].numpy().astype(bool)
    for key in REGION_KEYS[1:]:
        value=_expected(dataset.region_cache['train_001'][key],box,rotation,flip)
        value=np.where(valid,value,0)
        np.testing.assert_array_equal(sample[key][0].numpy(),value)
    assert sample['region_original'].dtype == sample['region_allowed'].dtype == torch.bool


def test_unknown_crop_rim_and_reflect_padding_never_supervised(tmp_path):
    manifest,cfg,_,_ = fixture_files(tmp_path,shape=(48,56))
    dataset=BoundaryNAFRegionDataset(manifest,cfg)
    sample=_forced(dataset,(0,0,48,56),1,True)
    valid=sample['valid'].bool()
    height,width=sample['meta']['geometry']['content_shape']
    assert not valid[:,height:].any() and not valid[:,:,width:].any()
    assert torch.any(sample['region_ids'] > 1)
    for key in REGION_KEYS[1:]:
        assert not torch.any(sample[key][~valid])
    # 输入中原本穿过unknown的预测不被新GT字段抹掉。
    parent=BoundaryNAFRealDataset(manifest,cfg)
    old=_forced(parent,(0,0,48,56),1,True)
    assert torch.equal(sample['input'],old['input'])
    assert not sample['meta']['region']['padding_supervised']


def test_filled_region_does_not_create_trusted_core(tmp_path):
    ids=np.full((96,128),7,np.int32);ids[:,64:]=19
    original=ids==7
    manifest,cfg,_,_=fixture_files(tmp_path,ids=ids,original=original)
    dataset=BoundaryNAFRegionDataset(manifest,cfg)
    arrays=dataset.region_cache['train_001']
    assert np.any(arrays['region_core_ids']==7)
    assert not np.any(arrays['region_core_ids']==19)
    assert not np.any((arrays['region_core_ids']>0)&~arrays['region_original'])
    assert dataset.region_records[0]['trusted_cores']['filled_used_for_cores'] is False


def test_real_twelve_pixel_corridor_is_not_erased_by_four_pixel_cores(tmp_path):
    manifest,cfg,_,_=fixture_files(tmp_path)
    dataset=BoundaryNAFRegionDataset(manifest,cfg)
    arrays=dataset.region_cache['train_001']
    assert arrays['region_core_ids'][45,54] == 7
    assert arrays['region_allowed'][45,54]
    assert not arrays['region_keep_low'][45,54]
    assert not np.any(arrays['region_allowed']&arrays['region_keep_low'])


def test_existing_line_inside_same_gt_is_not_forced_to_background(tmp_path):
    manifest,cfg,_,_=fixture_files(tmp_path)
    dataset=BoundaryNAFRegionDataset(manifest,cfg)
    arrays=dataset.region_cache['train_001']
    # 这条input线位于同GT内部且远离记录GT；不把“同GT”误当无物理晶界证明。
    assert arrays['region_core_ids'][32,25] == 7
    assert dataset.input_skeletons['train_001'][32,25]
    assert not arrays['region_keep_low'][32,25]
    assert arrays['region_keep_low'][45,25]
    # 靠GT的已有输入线提供高响应保持，不强迫其搬到精确GT像素。
    assert arrays['region_keep_high'][45,60]
    assert not dataset.cache['train_001'][0][45,60]


def test_full_image_distances_are_not_recomputed_after_crop(tmp_path):
    manifest,cfg,_,_=fixture_files(tmp_path)
    dataset=BoundaryNAFRegionDataset(manifest,cfg)
    box=(0,0,64,64)
    sample=_forced(dataset,box)
    # 界面在x=63/64，靠GT走廊必须根据整图计算，不能把裁窗外GT当不存在。
    assert sample['region_allowed'][0,45,53]
    assert torch.equal(sample['region_allowed'][0,4:-4,4:-4],
        torch.from_numpy(dataset.region_cache['train_001']['region_allowed'][4:60,4:60]))


def test_hot_getitem_uses_cache_and_collation_keeps_metadata(tmp_path,monkeypatch):
    manifest,cfg,_,_=fixture_files(tmp_path)
    dataset=BoundaryNAFRegionDataset(manifest,cfg)
    before=copy.deepcopy(dataset.identity)
    with monkeypatch.context() as patch:
        patch.setattr(np,'load',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('hot NPZ read')))
        samples=[dataset[i] for i in range(4)]
    batch=collate_boundary_naf_region(samples)
    for key in REGION_KEYS:
        assert batch[key].shape==(4,1,64,64)
    assert batch['source']==[s['source'] for s in samples]
    assert batch['draw']==[s['draw'] for s in samples]
    assert dataset.identity==before
    json.dumps(dataset.identity,allow_nan=False)
    dataset.verify_sources()


@pytest.mark.parametrize('key',REGION_KEYS)
def test_ram_source_identity_detects_extra_region_cache_change(tmp_path,key):
    manifest,cfg,_,_=fixture_files(tmp_path)
    dataset=BoundaryNAFRegionDataset(manifest,cfg)
    array=dataset.region_cache['train_001'][key]
    array[40,40]=not bool(array[40,40]) if array.dtype==bool else int(array[40,40])+1
    with pytest.raises(RuntimeError,match='Region in-memory source changed'):
        dataset.verify_sources()


@pytest.mark.parametrize('change', [
    {'corridor_radius':float('nan')}, {'core_radius':True}, {'min_core_component_pixels':0},
    {'keep_low_gt_margin':4}, {'typo_radius':12}, {'keep_low_distance':-1},
])
def test_invalid_or_conflicting_region_recipe_rejected(tmp_path,change):
    manifest,cfg,_,_=fixture_files(tmp_path)
    cfg['boundary_naf']['region']=dict(data=change)
    with pytest.raises(ValueError):
        BoundaryNAFRegionDataset(manifest,cfg)


def test_recipe_defaults_and_overrides_are_bound_to_identity(tmp_path):
    manifest,cfg,_,_=fixture_files(tmp_path)
    first=BoundaryNAFRegionDataset(manifest,cfg)
    cfg['boundary_naf']['region']=dict(data=dict(corridor_radius=16,core_radius=6,
        min_core_component_pixels=32,keep_high_distance=3,keep_low_distance=5))
    second=BoundaryNAFRegionDataset(manifest,cfg)
    assert first.identity['region_data']['corridor_radius']==12
    assert second.identity['region_data']==dict(core_radius=6,min_core_component_pixels=32,
        corridor_radius=16.,keep_high_distance=3.,keep_low_distance=5.,keep_low_gt_margin=16.)
    assert first.identity != second.identity
    assert first.plan==second.plan
    assert first.identity['region_sources'][0]['array_sha256']['region_ids']
