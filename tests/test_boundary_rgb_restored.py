# -*- coding: utf-8 -*-
"""冻结重建RGB的来源、精度和坐标合同；只使用合成CPU数据。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.boundary_rgb import (
    BoundaryRGBDataset, RESTORED_RGB_COORDINATES, collate_boundary_rgb,
    load_restored_rgb_manifest, restored_rgb_for_source,
)
from test_boundary_rgb import _fixture, _force, _sha
from test_boundary_erasure import _dataset_config
from utils.boundary_erasure import DEFAULTS


def _array_sha(a):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def _restored(tmp_path, raw, *, domain='train', name='train_001', image_sha=None):
    # 非1/255格点，能检测误量化、重复除255与raw回退。
    y, x = np.indices(raw.shape[:2])
    restored = np.stack([.12 + y*.0013, .22 + x*.0011, .31 + (y+x)*.0007], -1).astype(np.float32)
    restored_path = tmp_path/'restored.npy'
    np.save(restored_path, restored)
    manifest_path = tmp_path/'rgb_manifest.json'
    doc = dict(format='boundary_rgb_restored_v1', complete=True,
        coordinate_contract=RESTORED_RGB_COORDINATES,
        restoration=dict(model_file_sha256='2'*64, model_state_sha256='3'*64,
            noise_mode='zero', steps=16, output='first', native_size=1024,
            overlap=.25, cache_dtype='float32', quantization='none'),
        rows=[dict(source=name, domain=domain, image_sha256=image_sha or '1'*64,
            original_rgb_array_sha256=_array_sha(raw), shape=list(raw.shape),
            restored_path=restored_path.name, restored_sha256=_sha(restored_path),
            restored_array_sha256=_array_sha(restored))])
    _write(manifest_path, doc)
    return manifest_path, doc, restored


def _fixture_restored(tmp_path, shape=(80,96)):
    manifest, heat, cfg, raw, boundary, ids = _fixture(tmp_path,shape)
    rgb_manifest, document, restored = _restored(tmp_path, raw,
        image_sha=_sha(tmp_path/'train_001.png'))
    cfg['boundary_naf']['restored_rgb_manifest'] = str(rgb_manifest)
    return manifest, heat, cfg, raw, boundary, ids, rgb_manifest, document, restored


def test_loader_binds_lossless_arrays_and_json_receipt(tmp_path):
    _,_,_,raw,_,_,path,doc,restored = _fixture_restored(tmp_path)
    cache = load_restored_rgb_manifest(path)
    actual, receipt = restored_rgb_for_source(cache,'train_001',raw,image_sha256=doc['rows'][0]['image_sha256'])
    np.testing.assert_array_equal(actual,restored)
    assert actual.dtype == np.float32
    assert cache['domain'] == 'train' and cache['sha256'] == _sha(path)
    assert receipt['quantization'] == 'none' and receipt['array_sha256'] == _array_sha(restored)
    assert receipt['restoration'] == doc['restoration']
    json.dumps(receipt,allow_nan=False)
    assert not np.array_equal(restored,np.rint(restored*255)/255)


@pytest.mark.parametrize('rotation,flip',[(r,f) for r in range(4) for f in (False,True)])
def test_restored_conditioning_and_raw_display_share_native_D4(tmp_path,rotation,flip):
    manifest,_,cfg,raw,_,_,path,_,restored = _fixture_restored(tmp_path)
    ds=BoundaryRGBDataset(manifest,cfg)
    sample=_force(ds,rotation=rotation,flip=flip)
    expected=np.rot90(restored[8:72,8:72],rotation)
    expected_raw=np.rot90(raw[8:72,8:72],rotation)
    if flip:expected,expected_raw=expected[:,::-1],expected_raw[:,::-1]
    np.testing.assert_array_equal(sample['rgb'].numpy(),expected.transpose(2,0,1))
    np.testing.assert_array_equal(sample['image'].numpy(),expected_raw.transpose(2,0,1).astype(np.float32)/255.)
    assert torch.equal(sample['input'][:3],sample['rgb'])
    assert torch.equal(sample['input'][3:],sample['input_boundary'])
    assert not torch.equal(sample['rgb'],sample['image'])
    assert ds.identity['format']=='boundary_rgb_data_v4'
    assert ds.identity['rgb_source']=='frozen_restored_rgb_manifest'
    assert ds.identity['restored_rgb_manifest_sha256']==_sha(path)
    assert sample['meta']['rgb_quantization']=='none'
    assert ds.verify_sources()


def test_small_restored_native_source_reflects_without_supervising_padding(tmp_path):
    manifest,_,cfg,raw,_,_,_,_,restored=_fixture_restored(tmp_path,(40,48))
    ds=BoundaryRGBDataset(manifest,cfg)
    sample=_force(ds,box=(0,0,40,48),rotation=1,flip=True)
    expected=np.rot90(restored,1)[:,::-1]
    expected=np.pad(expected,((0,16),(0,24),(0,0)),mode='reflect')
    np.testing.assert_array_equal(sample['rgb'].numpy(),expected.transpose(2,0,1))
    assert not sample['valid'][0,48:].any() and not sample['valid'][0,:,40:].any()
    assert torch.isfinite(sample['rgb']).all()


def test_disabled_path_keeps_legacy_identity_meta_and_tensors_exact(tmp_path):
    manifest,_,cfg,_,_,_=_fixture(tmp_path)
    a=BoundaryRGBDataset(manifest,cfg)
    cfg2=deepcopy(cfg);cfg2['boundary_naf']['restored_rgb_manifest']=None
    b=BoundaryRGBDataset(manifest,cfg2)
    assert a.identity==b.identity and a.plan_receipt()==b.plan_receipt()
    for index in range(len(a)):
        x,y=a[index],b[index]
        assert x['meta']==y['meta'] and set(x)==set(y)
        for key in x:
            if torch.is_tensor(x[key]):assert torch.equal(x[key],y[key]),key


def test_restored_unknown_pixels_remain_input_and_GT_plan_is_unchanged(tmp_path):
    manifest,_,cfg,raw,_,ids,_,_,restored=_fixture_restored(tmp_path)
    legacy_cfg=deepcopy(cfg);legacy_cfg['boundary_naf'].pop('restored_rgb_manifest')
    before,after=BoundaryRGBDataset(manifest,legacy_cfg),BoundaryRGBDataset(manifest,cfg)
    assert before.plan_receipt()==after.plan_receipt()
    x,y=_force(before,box=(0,0,64,64)),_force(after,box=(0,0,64,64))
    for key in ('target','valid','input_boundary','raw_input_boundary','raw_gap_mask','gap_mask','image'):
        assert torch.equal(x[key],y[key]),key
    unknown=ids[:64,:64]==0
    assert not y['valid'][0].numpy()[unknown].any()
    np.testing.assert_array_equal(y['rgb'].numpy().transpose(1,2,0)[unknown],restored[:64,:64][unknown])


@pytest.mark.parametrize('edit',[
    lambda d:d.update(format='bad'),lambda d:d.update(complete=False),
    lambda d:d.update(coordinate_contract='resize_then_crop'),
    lambda d:d['restoration'].update(noise_mode='random'),
    lambda d:d['restoration'].update(output='last'),
    lambda d:d['restoration'].update(native_size=256),
    lambda d:d['restoration'].update(quantization='uint8'),
    lambda d:d['restoration'].update(cache_dtype='float16'),
    lambda d:d['rows'].append(deepcopy(d['rows'][0])),
    lambda d:d['rows'][0].update(image_sha256='bad'),
    lambda d:d['rows'][0].update(shape=[80,96,4]),
    lambda d:d['rows'][0].update(restored_path='output.png'),
])
def test_loader_rejects_invalid_provenance_or_contract(tmp_path,edit):
    _,_,_,_,_,_,path,doc,_=_fixture_restored(tmp_path)
    edit(doc);_write(path,doc)
    with pytest.raises(ValueError):load_restored_rgb_manifest(path)


def test_train_monitor_domains_cannot_be_mixed_or_substituted(tmp_path):
    manifest,_,cfg,raw,_,_,path,doc,_=_fixture_restored(tmp_path)
    mixed=deepcopy(doc);other=deepcopy(mixed['rows'][0]);other.update(source='test_101',domain='monitor_test')
    mixed['rows'].append(other);_write(path,mixed)
    with pytest.raises(ValueError,match='separate'):load_restored_rgb_manifest(path)
    doc['rows'][0].update(source='test_101',domain='monitor_test');_write(path,doc)
    cache=load_restored_rgb_manifest(path);assert cache['domain']=='monitor_test'
    assert restored_rgb_for_source(cache,'test_101',raw)[0].shape==raw.shape
    with pytest.raises(ValueError,match='training sources exactly'):BoundaryRGBDataset(manifest,cfg)


@pytest.mark.parametrize('kind',['float16','nan','range','shape','array_hash','file_hash'])
def test_wrong_cache_bytes_dtype_range_shape_or_hash_never_fallback_raw(tmp_path,kind):
    manifest,_,cfg,_,_,_,path,doc,restored=_fixture_restored(tmp_path)
    array_path=tmp_path/'restored.npy'
    if kind=='float16':restored=restored.astype(np.float16)
    if kind=='nan':restored[0,0,0]=np.nan
    if kind=='range':restored[0,0,0]=1.1
    if kind=='shape':restored=restored[:-1]
    if kind not in ('array_hash','file_hash'):
        np.save(array_path,restored);doc['rows'][0].update(restored_sha256=_sha(array_path),restored_array_sha256=_array_sha(restored))
    elif kind=='array_hash':doc['rows'][0]['restored_array_sha256']='f'*64
    else:doc['rows'][0]['restored_sha256']='f'*64
    _write(path,doc)
    with pytest.raises(ValueError):BoundaryRGBDataset(manifest,cfg)


def test_raw_array_file_binding_and_missing_source_are_strict(tmp_path):
    _,_,_,raw,_,_,path,_,_=_fixture_restored(tmp_path)
    cache=load_restored_rgb_manifest(path)
    changed=raw.copy();changed[0,0,0]^=1
    with pytest.raises(ValueError,match='array SHA256'):restored_rgb_for_source(cache,'train_001',changed)
    with pytest.raises(ValueError,match='file SHA256'):restored_rgb_for_source(cache,'train_001',raw,image_sha256='0'*64)
    with pytest.raises(ValueError,match='missing'):restored_rgb_for_source(cache,'train_002',raw)


@pytest.mark.parametrize('mutation',['manifest','file','array'])
def test_verify_sources_detects_cache_mutation(tmp_path,mutation):
    manifest,_,cfg,_,_,_,path,doc,restored=_fixture_restored(tmp_path)
    ds=BoundaryRGBDataset(manifest,cfg);ds.verify_sources()
    if mutation=='manifest':path.write_text(path.read_text()+' ')
    elif mutation=='file':np.save(tmp_path/'restored.npy',restored+.0001)
    else:ds.model_rgb_images['train_001'][0,0,0]+=.0001
    with pytest.raises(RuntimeError,match='[Rr]estored RGB'):ds.verify_sources()


def test_rgb_erasure_uses_same_masks_plan_targets_as_T8_control(tmp_path):
    manifest,cfg=_dataset_config(tmp_path)
    cfg['boundary_naf']['erasure']=dict(DEFAULTS,probability=1.,count_range=[1,1],lengths=[16])
    before=BoundaryRGBDataset(manifest,cfg)
    raw=before.rgb_images['train_001']
    path,_,_=_restored(tmp_path,raw,image_sha=before.by_source['train_001']['image_sha256'])
    rgb_cfg=deepcopy(cfg);rgb_cfg['boundary_naf'].update(input_mode='rgb_boundary',restored_rgb_manifest=str(path))
    after=BoundaryRGBDataset(manifest,rgb_cfg)
    assert before.plan_receipt()==after.plan_receipt()
    for r in range(4):
        for f in (False,True):
            a=_force(before,box=[32,32,128,128],rotation=r,flip=f)
            b=_force(after,box=[32,32,128,128],rotation=r,flip=f)
            assert a['meta']['erasure']==b['meta']['erasure'] and b['meta']['erasure']['applied']
            for key in ('input_boundary','clean_input_boundary','target','valid','erasure_mask','artificial_gap_mask',
                        'raw_input_boundary','gap_mask','raw_gap_mask','image'):
                assert torch.equal(a[key],b[key]),key
            assert b['input'].shape==(4,128,128)
            assert torch.equal(b['input'][:3],b['rgb'])
    batch=collate_boundary_rgb([b,b]);assert batch['input'].shape==(2,4,128,128)
    assert batch['rgb'].dtype==torch.float32 and not torch.equal(batch['rgb'],batch['image'])
    after.verify_sources()


@pytest.mark.parametrize('value',['',False,42,[]])
def test_explicit_bad_manifest_does_not_silently_use_raw(tmp_path,value):
    manifest,_,cfg,_,_,_=_fixture(tmp_path)
    cfg['boundary_naf']['restored_rgb_manifest']=value
    with pytest.raises(ValueError,match='restored_rgb_manifest'):BoundaryRGBDataset(manifest,cfg)


def test_restorer_model_must_match_N1_and_manifest_source_set(tmp_path):
    manifest,_,cfg,_,_,_,path,doc,_=_fixture_restored(tmp_path)
    doc['restoration']['model_file_sha256']='7'*64;_write(path,doc)
    with pytest.raises(ValueError,match='same restorer'):BoundaryRGBDataset(manifest,cfg)
    doc['restoration']['model_file_sha256']='2'*64
    doc['rows'][0]['source']='train_002';_write(path,doc)
    with pytest.raises(ValueError,match='training sources exactly'):BoundaryRGBDataset(manifest,cfg)
