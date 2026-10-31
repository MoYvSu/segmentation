# -*- coding: utf-8 -*-
"""RGB 条件只增加同源信息；合成坐标图验证位置、旧配对与随机流不变。"""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from data.boundary_context import BoundaryContextDataset, collate_boundary_context, native_rgb_crop
from data.boundary_real import BoundaryRealDataset


def fixture_files(tmp_path, shape=(88, 112)):
    yy, xx = np.indices(shape)
    rgb = np.stack((xx, yy, (xx+3*yy) % 256), axis=-1).astype(np.uint8)
    ids = np.ones(shape, np.int32); ids[:, shape[1]//2:] = 2
    ids[26:31, 22:28] = 0
    original = ids > 0; original[40:44, 54:58] = False
    image = tmp_path/'train_001.png'
    cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    gt = tmp_path/'train_001_gt.npz'
    np.savez_compressed(gt, instance_map=ids, original_covered=original,
                        filled=(ids > 0)&~original, residual_unknown=ids == 0)
    classes = tmp_path/'train_001_class.json'; classes.write_text('{"1":0,"2":1}')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    files = dict(image=image, gt=gt, classes=classes)
    manifest = tmp_path/'gt.json'
    row = dict(source='train_001', **{k:str(v) for k,v in files.items()},
               hashes={k+'_sha256':sha(v) for k,v in files.items()})
    manifest.write_text(json.dumps(dict(rows=[row])))
    heat = rgb[..., 0].astype(np.float32)/255
    heat_path = tmp_path/'heat.npz'; np.savez_compressed(heat_path, boundary=heat)
    real = tmp_path/'real.json'
    real.write_text(json.dumps(dict(format='boundary_real_heatmaps_v1', complete=True,
        training=False, frozen_models_unchanged=True, sealed_source_unchanged=True,
        normalization='none; original N0 float32 [0,1]', processed_gt_manifest_sha256=sha(manifest),
        config=dict(submission=dict(seed=17, restoration_noise='toy')),
        runtime=dict(restoration_noise=dict(name='toy')), model_states=dict(frozen='b'*64),
        files=[dict(source='train_001', heat_path=str(heat_path), heat_sha256=sha(heat_path),
                    image_sha256=sha(image), shape=list(shape), source_index=17)])))
    return manifest, real, rgb


def test_all_five_epoch_plans_and_old_pair_tensors_remain_exact(tmp_path):
    manifest, real, rgb = fixture_files(tmp_path)
    common = dict(real_manifest_path=real, expected_sources=1, crop_size=64, seed=37)
    old = BoundaryRealDataset(manifest, **common)
    new = BoundaryContextDataset(manifest, **common)
    assert new.identity['parent_real_identity_sha256'] == old.verify_sources()
    before = np.random.get_state()
    for epoch in range(1, 6):
        old.set_epoch(epoch); new.set_epoch(epoch)
        assert old.plan == new.plan and old.plan_receipt() == new.plan_receipt()
        for index in range(len(new)):
            a, b = old[index], new[index]
            for key in ('input','target','valid','gap_mask','noise_mask','identity'):
                assert torch.equal(a[key], b[key])
            context = b['meta']['context']
            meta = {k:v for k,v in b['meta'].items() if k != 'context'}
            assert a['meta'] == meta and context['geometry'] == a['meta']['geometry']
            geo = a['meta']['geometry']
            expected = native_rgb_crop(rgb, geo['box'], 64, geo['rotation'], geo['flip'])
            np.testing.assert_array_equal(b['context_rgb'].numpy(), expected.transpose(2,0,1)/np.float32(255))
            # 红通道和原N0编码同一x坐标；错转/错翻RGB会破坏此逐像素契约。
            assert torch.equal(b['context_rgb'][:1], b['target'])
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    new.verify_sources()
    json.dumps(new.identity, allow_nan=False)


def test_padding_is_reflect_but_old_heat_and_valid_stay_unchanged(tmp_path):
    manifest, real, rgb = fixture_files(tmp_path, (40,56))
    dataset = BoundaryContextDataset(manifest, real_manifest_path=real, expected_sources=1,
                                     crop_size=64, seed=21)
    sample = dataset[0]
    h, w = sample['meta']['geometry']['content_shape']
    assert not sample['valid'][:,h:,:].any() and not sample['valid'][:,:,w:].any()
    assert not sample['target'][:,h:,:].any() and not sample['target'][:,:,w:].any()
    assert sample['context_rgb'][:,h:,:].any() or sample['context_rgb'][:,:,w:].any()
    assert torch.equal(sample['context_rgb'][:1,:h,:w], sample['target'][:,:h,:w])
    batch = collate_boundary_context([sample,sample])
    assert batch['context_rgb'].shape == (2,3,64,64) and batch['input'].shape == (2,1,64,64)


def test_rotations_flips_and_misalignment_counterexample():
    yy, xx = np.indices((11,17))
    rgb = np.stack((xx,yy,xx+yy),-1).astype(np.uint8)
    for rotation in range(4):
        for flip in (False,True):
            expected = np.rot90(rgb[1:10,2:16], rotation)
            if flip:
                expected = expected[:,::-1]
            actual = native_rgb_crop(rgb,[1,2,9,14],20,rotation,flip)
            np.testing.assert_array_equal(actual[:expected.shape[0],:expected.shape[1]],expected)
    correct = native_rgb_crop(rgb,[1,2,9,14],20,1,True)
    wrong = native_rgb_crop(rgb,[1,2,9,14],20,0,True)
    assert not np.array_equal(correct,wrong)


def test_rgb_identity_is_bound_to_content_not_name_and_mutation_detected(tmp_path):
    manifest, real, _ = fixture_files(tmp_path)
    original = json.loads(real.read_text())
    bad = deepcopy(original); bad['files'][0]['image_sha256'] = '0'*64
    real.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='image identity'):
        BoundaryContextDataset(manifest,real_manifest_path=real,expected_sources=1)
    real.write_text(json.dumps(original))
    dataset = BoundaryContextDataset(manifest,real_manifest_path=real,expected_sources=1)
    path = tmp_path/'train_001.png'
    with path.open('ab') as handle:
        handle.write(b'changed_source')
    with pytest.raises(RuntimeError,match='Processed source changed'):
        dataset.verify_sources()
