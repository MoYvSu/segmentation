# -*- coding: utf-8 -*-
"""真实教师热图自损伤数据的 CPU 契约；全部使用合成图和伪协议回执。"""
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt

from data.boundary_real import BoundaryRealDataset, make_boundary_real_pair, native_heat_crop
from data.boundary_restore import BoundaryRestoreDataset, known_instance_boundary


def source_arrays(shape=(160, 192)):
    yy, xx = np.indices(shape)
    ids = np.ones(shape, np.int32); ids[:, shape[1]//2:] = 2
    ids[35:43, 24:35] = 0
    heat = (.07 + .02*yy/shape[0] + .84*np.exp(-.5*((xx-shape[1]//2)/4.5)**2)).astype(np.float32)
    heat[ids == 0] = .456789  # unknown 不是清零背景，保留其原始上下文。
    return ids, heat


def write_manifests(tmp_path):
    ids, heat = source_arrays()
    original = ids > 0; original[74:80, 92:101] = False
    filled = (ids > 0) & ~original
    rgb = np.full((*ids.shape, 3), 127, np.uint8)
    image = tmp_path/'train_001.png'; cv2.imwrite(str(image), rgb)
    gt = tmp_path/'train_001_gt.npz'
    np.savez_compressed(gt, instance_map=ids, original_covered=original, filled=filled, residual_unknown=ids==0)
    classes = tmp_path/'train_001_class.json'; classes.write_text(json.dumps({'1':0, '2':1}))
    files = dict(image=image, gt=gt, classes=classes)
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    row = dict(source='train_001', **{key:str(path) for key,path in files.items()},
               hashes={key+'_sha256':sha(path) for key,path in files.items()})
    manifest = tmp_path/'processed.json'; manifest.write_text(json.dumps(dict(rows=[row])))
    heat_path = tmp_path/'heat.npz'; np.savez_compressed(heat_path, boundary=heat)
    real = tmp_path/'real.json'
    real_document = dict(format='boundary_real_heatmaps_v1', complete=True,
        training=False, frozen_models_unchanged=True, sealed_source_unchanged=True,
        normalization='none; original N0 float32 [0,1]', processed_gt_manifest_sha256=sha(manifest),
        config=dict(submission=dict(seed=17, restoration_noise='synthetic_fixture')),
        runtime=dict(device='synthetic_fixture_cpu', restoration_noise=dict(name='synthetic_fixture')),
        model_states=dict(frozen='b'*64),
        files=[dict(source='train_001', heat_path=str(heat_path), heat_sha256=sha(heat_path),
                    image_sha256=sha(image), shape=list(ids.shape), source_index=17)])
    real.write_text(json.dumps(real_document))
    return manifest, real, heat_path, ids, heat


@pytest.mark.parametrize('mode', ['identity', 'gap', 'noise'])
def test_target_exact_unknown_context_and_damage_only_in_valid(mode):
    ids, heat = source_arrays(); before = heat.copy()
    pair = make_boundary_real_pair(ids, heat, np.random.default_rng(20), mode)
    inputs, target, valid = (pair[key][0].numpy() for key in ('input','target','valid'))
    np.testing.assert_array_equal(heat, before)
    np.testing.assert_array_equal(target, before)
    np.testing.assert_array_equal(inputs[valid == 0], target[valid == 0])
    np.testing.assert_array_equal(inputs[ids == 0], before[ids == 0])
    assert pair['meta']['coverage']['changed_outside_valid'] == 0
    assert (pair['gap_mask'] <= pair['valid']).all() and (pair['noise_mask'] <= pair['valid']).all()
    if mode == 'identity':
        assert pair['identity'].item() and torch.equal(pair['input'], pair['target'])
    else:
        assert not pair['identity'].item()
        assert pair['meta']['coverage']['changed_pixels'] > 0


def test_gap_uses_nonzero_local_floor_and_actual_effective_mask():
    ids, heat = source_arrays()
    pair = make_boundary_real_pair(ids, heat, np.random.default_rng(21), 'gap')
    inputs, target = pair['input'][0].numpy(), pair['target'][0].numpy()
    mask = pair['gap_mask'][0].numpy() > 0
    assert mask.any() and np.all((target-inputs)[mask] > .05)
    assert np.all(inputs[mask] > .06) and np.all(inputs <= target)
    assert min(row['floor'] for row in pair['meta']['gaps']) > .06
    assert all(row['floor_estimation']['strategy'] in ('local_known_deep_interior', 'crop_known_deep_interior')
               for row in pair['meta']['gaps'])
    # 断口最多在中心边界附近14px半径+1px软化内；远处没有全局调强/调宽。
    assert np.array_equal(inputs[:, :60], target[:, :60])
    assert np.array_equal(inputs[:, 130:], target[:, 130:])


def test_noise_adds_only_known_deep_low_response_pixels():
    ids, heat = source_arrays()
    pair = make_boundary_real_pair(ids, heat, np.random.default_rng(22), 'noise')
    inputs = pair['input'][0].numpy(); mask = pair['noise_mask'][0].numpy() > 0
    distance = distance_transform_edt(~known_instance_boundary(ids))
    assert mask.any() and np.all((inputs-heat)[mask] > .02)
    assert np.all(distance[mask] > 16) and np.all(heat[mask] <= .2)
    assert np.all(inputs >= heat)
    unchanged = (distance <= 16) | (heat > .2) | (pair['valid'][0].numpy() == 0)
    np.testing.assert_array_equal(inputs[unchanged], heat[unchanged])


def test_native_float_crop_rotation_flip_and_padding_no_rescaling():
    heat = np.linspace(.0123, .9876, 24*40, dtype=np.float32).reshape(24,40)
    actual = native_heat_crop(heat, [2,3,20,30], 48, rotation=1, flip=True)
    expected = np.rot90(heat[2:22,3:33])[:, ::-1]
    np.testing.assert_array_equal(actual[:30,:20], expected)
    assert not actual[30:].any() and not actual[:,20:].any()
    assert actual.dtype == np.float32 and np.unique(actual).size == expected.size+1


def test_source_plan_crop_rotation_exact_old_control_and_resume(tmp_path):
    manifest, real, _, _, heat = write_manifests(tmp_path)
    old = BoundaryRestoreDataset(manifest, crop_size=96, expected_sources=1, seed=37)
    new = BoundaryRealDataset(manifest, real_manifest_path=real, crop_size=96, expected_sources=1, seed=37)
    for epoch in (1, 2):
        old.set_epoch(epoch); new.set_epoch(epoch)
        assert new.plan == old.plan and new.plan_receipt() == old.plan_receipt()
        for index in range(len(new)):
            a, b = old[index], new[index]
            assert a['source'] == b['source'] and a['meta']['geometry'] == b['meta']['geometry']
            assert a['meta']['requested_mode'] == b['meta']['requested_mode']
            geom = b['meta']['geometry']
            expected = native_heat_crop(heat, geom['box'], 96, geom['rotation'], geom['flip'])
            np.testing.assert_array_equal(b['target'][0].numpy(), expected)
    new.set_epoch(1); first = new[0]; new.set_epoch(2); new.set_epoch(1)
    again = new[0]
    assert first['meta'] == again['meta'] and torch.equal(first['input'], again['input'])
    assert new.identity['data_mode'] == 'real_n0_heatmaps'
    assert not new.identity['target_gt_rendered'] and not new.identity['test_training']
    assert not new.load_heat('train_001').flags.writeable
    new.verify_sources(); json.dumps(new.identity, allow_nan=False)


def test_reject_wrong_image_hash_incomplete_or_changed_cache(tmp_path):
    manifest, real, heat_path, _, heat = write_manifests(tmp_path)
    original = json.loads(real.read_text())
    bad = {**original, 'complete':False}; real.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='complete frozen'):
        BoundaryRealDataset(manifest, real_manifest_path=real, expected_sources=1)
    bad = json.loads(json.dumps(original)); bad['files'][0]['image_sha256'] = '0'*64
    real.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='image identity'):
        BoundaryRealDataset(manifest, real_manifest_path=real, expected_sources=1)
    real.write_text(json.dumps(original))
    dataset = BoundaryRealDataset(manifest, real_manifest_path=real, expected_sources=1)
    dataset.load_heat('train_001')
    np.savez_compressed(heat_path, boundary=heat*.9)
    with pytest.raises(RuntimeError, match='heatmap changed'):
        dataset.verify_sources()


def test_no_effective_gap_falls_back_to_exact_identity_without_drawing_gt():
    ids = np.ones((80,80), np.int32)
    heat = np.full(ids.shape, .1234567, np.float32)
    pair = make_boundary_real_pair(ids, heat, np.random.default_rng(24), 'gap')
    assert pair['identity'].item() and torch.equal(pair['input'], pair['target'])
    assert not pair['gap_mask'].any() and pair['meta']['requested_mode'] == 'gap'
    assert pair['meta']['fallback_reason'] == 'no_effective_local_damage'


def test_reject_missing_teacher_protocol_or_different_gt_manifest(tmp_path):
    manifest, real, _, _, _ = write_manifests(tmp_path)
    original = json.loads(real.read_text())
    bad = {**original, 'processed_gt_manifest_sha256':'0'*64}
    real.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='different processed GT manifest'):
        BoundaryRealDataset(manifest, real_manifest_path=real, expected_sources=1)
    bad = {**original, 'runtime':{}}
    real.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='provenance is required'):
        BoundaryRealDataset(manifest, real_manifest_path=real, expected_sources=1)


def test_pair_rng_isolated_and_invalid_real_arrays_rejected():
    ids, heat = source_arrays()
    np.random.seed(12); before = np.random.get_state()
    a = make_boundary_real_pair(ids, heat, np.random.default_rng(2), 'gap')
    b = make_boundary_real_pair(ids, heat, np.random.default_rng(2), 'gap')
    after = np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:] == after[2:]
    assert a['meta'] == b['meta'] and torch.equal(a['input'],b['input'])
    with pytest.raises(ValueError, match='float32'):
        make_boundary_real_pair(ids, heat.astype(np.float16), np.random.default_rng(2))
    with pytest.raises(ValueError, match='Unknown real-heat recipe'):
        make_boundary_real_pair(ids, heat, np.random.default_rng(2), recipe={'target_sigma':2})
