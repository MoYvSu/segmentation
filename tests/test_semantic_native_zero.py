# -*- coding: utf-8 -*-
"""原生语义训练的来源、有效监督和可恢复采样合同。只用合成训练样本。"""
from collections import Counter, defaultdict
import json
import random

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.semantic_native_zero import NativeSemanticDataset
from data.semantic_targets import semantic_targets_from_instances


def cohort(tmp_path, *, count=2, shape=(15, 21), uniform=False):
    raw, gt = tmp_path / 'raw', tmp_path / 'gt'
    raw.mkdir(); gt.mkdir()
    h, w = shape
    ids = np.ones(shape, np.uint16)
    if not uniform:
        ids[:, w // 3:] = 2
        ids[h // 2:h // 2 + 3, w // 2:w // 2 + 4] = 3
        ids[1:3, 2:4] = 0
    classes = {'1': 0} if uniform else {'1': 0, '2': 1, '3': 0}
    lookup = np.array([-1, 0, 1, 0], np.int16)
    rgb = np.stack([np.where(ids > 0, np.maximum(lookup[ids], 0) * 255, 127),
                    np.broadcast_to(np.arange(h)[:, None], shape),
                    np.broadcast_to(np.arange(w)[None, :], shape)], axis=-1).astype(np.uint8)
    original = (ids > 0).astype(np.uint8)
    original[:, ::3] = 0
    filled = ((ids > 0) & ~original.astype(bool)).astype(np.uint8)
    names = [f'fiction_{i:02d}' for i in range(count)]
    for name in names:
        cv2.imwrite(str(raw / (name + '.png')), rgb[..., ::-1])
        (raw / (name + '.json')).write_text(json.dumps(dict(imagePath=name + '.png', imageHeight=h, imageWidth=w)))
        np.savez_compressed(gt / (name + '_gt.npz'), instance_map=ids, original_covered=original,
                            filled=filled, residual_unknown=(ids == 0).astype(np.uint8))
        (gt / (name + '_class.json')).write_text(json.dumps(classes))
    manifest = dict(format='manual_target_v2', version=1, sample_count=count, samples=names,
                    max_fill_distance_native=8., blur_sigma=1.2)
    (gt / 'manifest.json').write_text(json.dumps(manifest))
    return raw, gt, names, rgb, ids


def dataset(data, **kwargs):
    raw, gt, names, *_ = data
    return NativeSemanticDataset(raw, gt, seed=20261031, expected_sources=len(names),
                                 native_size=8, **kwargs)


def change_gt(path, key, value):
    with np.load(path) as blob:
        arrays = {k: blob[k].copy() for k in blob.files}
    arrays[key] = value(arrays[key])
    np.savez_compressed(path, **arrays)


def test_every_three_epochs_cover_actual_deployment_grid_and_full_budget(tmp_path):
    ds = dataset(cohort(tmp_path))
    all_counts = defaultdict(Counter)
    block = defaultdict(list)
    for epoch in range(1, 21):
        ds.set_epoch(epoch)
        rows = [ds.sample_spec(i) for i in range(len(ds))]
        assert Counter(r['source'] for r in rows) == {s['source']: 4 for s in ds.sources}
        for row in rows:
            all_counts[row['source']][row['window_index']] += 1
            block[row['source']].append(row['window_index'])
        if epoch % 3 == 0:
            assert all(len(values) == 12 and len(set(values)) == 12 for values in block.values())
            block.clear()
    assert all(sum(values.values()) == 80 and set(values.values()) == {6, 7} for values in all_counts.values())
    assert ds.identity()['total_updates'] == 160
    with pytest.raises(ValueError, match='budget'):
        ds.set_epoch(21)


def test_d4_preserves_rgb_class_id_unknown_and_boundary_alignment(tmp_path):
    data = cohort(tmp_path)
    ds = dataset(data)
    _, native_boundary = semantic_targets_from_instances(data[4], np.array([-1, 0, 1, 0], np.float32))
    seen = set()
    for epoch in range(1, 5):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            row = ds[i]
            seen.add(row['d4_index'])
            ids, target = row['semantic_instance_map'], row['semantic_target'][0]
            assert torch.equal(ids == 0, target < 0)
            assert torch.equal(row['semantic_valid_content'][0], ids > 0)
            known = ids > 0
            assert torch.equal(torch.round(row['image'][0][known] * 255), target[known] * 255)
            # 反变换后与原坐标裁块相同，检查图像和 ID 的实际空间对应。
            unrotated = torch.rot90(row['image'], -row['rotation_k'], (-2, -1))
            if row['horizontal_flip']:
                unrotated = unrotated.flip(-1)
            y0, x0, y1, x1 = row['crop_box']
            expected = torch.from_numpy(data[3][y0:y1, x0:x1].copy()).permute(2, 0, 1).float() / 255
            assert torch.equal(unrotated, expected)
            unrotated_boundary = torch.rot90(row['semantic_boundary'][0], -row['rotation_k'], (-2, -1))
            if row['horizontal_flip']:
                unrotated_boundary = unrotated_boundary.flip(-1)
            assert torch.equal(unrotated_boundary, torch.from_numpy(native_boundary[y0:y1, x0:x1]))
            ledger = row['ledger']
            assert ledger['original_covered_pixels'] + ledger['filled_pixels'] == ledger['known_pixels']
            assert ledger['known_pixels'] + ledger['unknown_pixels'] == 64
            assert ledger['visible_F_instances'] + ledger['visible_P_instances'] == ledger['visible_instances']
            assert row['semantic_valid_content'].dtype == torch.bool
    assert seen == set(range(8))


def test_cutting_one_uniform_instance_never_creates_a_fake_gt_frame(tmp_path):
    ds = dataset(cohort(tmp_path, uniform=True))
    for i in range(len(ds)):
        row = ds[i]
        assert row['semantic_boundary'].count_nonzero() == 0
        assert row['ledger']['truncated_instances'] == 1
        assert row['ledger']['visible_instances'] == 1


def test_resume_draws_and_monitors_do_not_consume_global_rng(tmp_path):
    data = cohort(tmp_path)
    ds = dataset(data)
    ds.set_epoch(9)
    random.seed(33); np.random.seed(34); torch.manual_seed(35)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    expected = {i: ds[i] for i in (7, 0, 3)}
    monitors = ds.monitor_samples()
    assert len(monitors) == 2 and all(r['d4_index'] == 0 and r['epoch'] == 0 for r in monitors)
    assert before[0] == random.getstate()
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert torch.equal(before[2], torch.get_rng_state())
    restored = dataset(data)
    restored.set_epoch(9)
    for i, old in expected.items():
        actual = restored[i]
        assert old['crop_box'] == actual['crop_box'] and old['d4_index'] == actual['d4_index']
        for key in ('image', 'semantic_target', 'semantic_boundary', 'semantic_instance_map', 'valid_content'):
            assert torch.equal(old[key], actual[key])
    # 训练器默认 batch1 collate 可直接消费，不含变长 GT ID 清单。
    batch = next(iter(DataLoader(restored, batch_size=1, shuffle=False)))
    assert batch['image'].shape == (1, 3, 8, 8)
    assert batch['semantic_instance_map'].shape == (1, 8, 8)


def test_reflected_rgb_padding_remains_ignored_without_gt_boundary(tmp_path):
    ds = dataset(cohort(tmp_path, count=1, shape=(4, 8), uniform=True),
                 expected_tiles_per_source=1, draws_per_source=1)
    row = ds[0]
    padding = ~row['valid_content'][0]
    assert int(padding.sum()) == 32
    assert (row['semantic_target'][0][padding] == -1).all()
    assert (row['semantic_instance_map'][padding] == 0).all()
    assert row['semantic_boundary'].count_nonzero() == 0
    assert row['ledger']['known_pixels'] == 32
    assert row['ledger']['padding_pixels'] == 32


def test_manifest_and_source_seal_reject_changed_or_mixed_inputs(tmp_path):
    data = cohort(tmp_path)
    ds = dataset(data)
    assert ds.source_seal()['source_seal_sha256'] == ds.identity()['source_seal_sha256']
    path = data[1] / (data[2][0] + '_class.json')
    path.write_text('{"1":1,"2":1,"3":0}')
    with pytest.raises(RuntimeError, match='Sealed source changed'):
        ds.identity()
    manifest = data[1] / 'manifest.json'
    value = json.loads(manifest.read_text()); value['samples'][0] = 'wrong_source'
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='manifest differs'):
        dataset(data)


@pytest.mark.parametrize('bad', ['uint8_mask', 'unknown_mismatch', 'class_bool', 'class_missing', 'shape'])
def test_invalid_supervision_is_rejected_before_training(tmp_path, bad):
    data = cohort(tmp_path, count=1)
    gt = data[1] / (data[2][0] + '_gt.npz')
    if bad == 'uint8_mask':
        change_gt(gt, 'original_covered', lambda x: x * 255)
    elif bad == 'unknown_mismatch':
        change_gt(gt, 'residual_unknown', lambda x: np.zeros_like(x))
    elif bad == 'shape':
        change_gt(gt, 'instance_map', lambda x: x[:-1])
    else:
        classes = {'1': False, '2': 1, '3': 0} if bad == 'class_bool' else {'1': 0, '2': 1}
        (data[1] / (data[2][0] + '_class.json')).write_text(json.dumps(classes))
    ds = dataset(data)
    with pytest.raises(ValueError, match='binary|provenance|Class mapping|shape/dtype'):
        ds[0]
