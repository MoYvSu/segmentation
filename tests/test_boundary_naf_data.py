# -*- coding: utf-8 -*-
"""NAF二值GT配对的CPU契约：不用赛题数据、RGB解码或模型预测。"""
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from data.boundary_naf import (
    BoundaryNAFDataset, build_training_plan, collate_boundary_naf,
    load_processed_binary, source_boxes,
)


def _manifest(tmp_path, sources=1, shape=(96, 128)):
    rows = []
    instances = np.ones(shape, np.int32)
    instances[:, shape[1] // 2:] = 2
    instances[35:43, 16:24] = 0
    original = instances > 0
    original[50:55, 60:67] = False
    for index in range(sources):
        source = f'train_{index + 1:03d}'
        image = tmp_path / (source + '.jpg')
        # 此文件刻意不是图像，证明此数据流根本不解码RGB来生成监督。
        image.write_bytes(b'checked-image-source-identity-only')
        gt = tmp_path / (source + '_gt.npz')
        np.savez_compressed(gt, instance_map=instances, original_covered=original,
                            filled=(instances > 0) & ~original, residual_unknown=instances == 0)
        classes = tmp_path / (source + '_class.json')
        classes.write_text(json.dumps({'1': 0, '2': 0}), encoding='utf-8')
        files = dict(image=image, gt=gt, classes=classes)
        rows.append(dict(source=source, **{key: str(path) for key, path in files.items()},
            hashes={key + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                    for key, path in files.items()}))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(rows=rows)), encoding='utf-8')
    return path, rows, instances, original


def _config(**kwargs):
    return {'boundary_naf': {'seed': 20261026, 'data': {
        'expected_sources': 1, 'crop_size': 96, 'stride': 50, 'augmentations': 8,
        'valid_margin': 4, 'crop_margin': 4, 'identity_probability': 0.2,
        'samples_per_epoch': 16, **kwargs}}}


def test_dense_plan_covers_last_position_and_all_eight_transforms():
    assert source_boxes((90, 130), 64, 50) == [
        [0, 0, 64, 64], [0, 50, 64, 64], [0, 66, 64, 64],
        [26, 0, 64, 64], [26, 50, 64, 64], [26, 66, 64, 64]]
    rows = [dict(source='train_001', shape=[90, 130])]
    plan = build_training_plan(rows, 64, 50, 8, seed=1)
    assert len(plan) == 6 * 8
    unique = {(tuple(item['box']), item['rotation'], item['flip']) for item in plan}
    assert len(unique) == len(plan)
    assert {item['augmentation'] for item in plan} == set(range(8))


def test_budget_schedule_is_source_balanced_and_exactly_reproducible():
    rows = [dict(source=f'train_{i:03d}', shape=[128, 128]) for i in (1, 2, 3)]
    first = build_training_plan(rows, 64, 50, seed=7, epoch=1, samples_per_epoch=32)
    again = build_training_plan(list(reversed(rows)), 64, 50, seed=7, epoch=1, samples_per_epoch=32)
    assert first == again and len(first) == 32
    counts = [sum(item['source'] == row['source'] for item in first) for row in rows]
    assert max(counts) - min(counts) == 1
    assert first != build_training_plan(rows, 64, 50, seed=7, epoch=2, samples_per_epoch=32)
    with pytest.raises(ValueError, match='training sources'):
        build_training_plan([dict(source='test_169', shape=[128, 128])])


def test_binary_targets_use_processed_instance_geometry_not_class_or_rgb(tmp_path, monkeypatch):
    path, rows, ids, original = _manifest(tmp_path)
    monkeypatch.setattr(cv2, 'imdecode', lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError('RGB must not be decoded')))
    dataset = BoundaryNAFDataset(path, _config(identity_probability=1.0), epoch=0)
    assert not dataset.identity['rgb_decoded'] and not dataset.identity['model_predictions_used']
    assert dataset.records[0]['filled_pixels'] == int(((ids > 0) & ~original).sum())
    target, valid = dataset.cache['train_001']
    assert target[:, 63:65].all()
    assert not target[:, :63].any() and not target[:, 65:].any()
    assert valid[50:55, 60:67].all()  # filled依然是处理后已知，不被误退回unknown。
    assert not valid[31:47, 12:28].any()
    item = dataset[0]
    assert item['identity'].item() and torch.equal(item['input'], item['target'])
    assert set(torch.unique(item['target']).tolist()) <= {0.0, 1.0}
    assert not item['gap_mask'].any()
    assert torch.equal(item['valid'], item['fully_known_mask'])
    assert item['meta']['resize_scale'] == 1.0


def test_gap_clears_only_true_valid_positive_pixels_and_keeps_target(tmp_path):
    path, _, _, _ = _manifest(tmp_path)
    dataset = BoundaryNAFDataset(path, _config(identity_probability=0.0,
        crop_size=96, samples_per_epoch=32, gap_count_range=[2, 2]), epoch=1)
    nonempty = []
    for index in range(len(dataset)):
        item = dataset[index]
        gap = item['gap_mask'].bool()
        assert not torch.any(gap & ~item['valid'].bool())
        assert torch.equal(item['input'][~gap], item['target'][~gap])
        if gap.any():
            assert torch.all(item['target'][gap] == 1)
            assert torch.all(item['input'][gap] == 0)
            assert not item['identity'].item()
            nonempty.append(index)
    assert nonempty


def test_rotation_padding_remain_native_and_unsupervised(tmp_path):
    path, _, _, _ = _manifest(tmp_path, shape=(64, 96))
    dataset = BoundaryNAFDataset(path, _config(crop_size=128, identity_probability=1.0,
        samples_per_epoch=None), epoch=1)
    assert len(dataset) == 8
    for item in (dataset[i] for i in range(8)):
        shape = item['meta']['geometry']['content_shape']
        valid = item['valid'][0].numpy()
        assert not valid[shape[0]:].any() and not valid[:, shape[1]:].any()
        assert not valid[:4].any() and not valid[-4:].any()
        assert valid.sum() > 0
        assert item['target'].shape == (1, 128, 128)
        assert set(torch.unique(item['target']).tolist()) <= {0.0, 1.0}


def test_cached_samples_dont_reread_npz_and_resume_plan_does_not_touch_global_rng(tmp_path, monkeypatch):
    path, _, _, _ = _manifest(tmp_path)
    dataset = BoundaryNAFDataset(path, _config(), epoch=1)
    receipt = dataset.plan_receipt()
    state = np.random.get_state()
    sample = dataset[3]
    with monkeypatch.context() as patch:
        patch.setattr(np, 'load', lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError('Cached steps must not reload NPZ')))
        assert torch.equal(sample['input'], dataset[3]['input'])
    dataset.set_epoch(2)
    assert dataset.plan_receipt()['plan_sha256'] != receipt['plan_sha256']
    dataset.set_epoch(1)
    assert dataset.plan_receipt() == receipt
    assert torch.equal(dataset[3]['input'], sample['input'])
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    batch = collate_boundary_naf([sample, dataset[3]])
    assert batch['input'].shape == (2, 1, 96, 96)
    assert batch['identity'].shape == (2,)
    assert isinstance(batch['meta'], list)
    json.dumps(dataset.identity, allow_nan=False)
    json.dumps(dataset.plan_receipt(), allow_nan=False)
    dataset.verify_sources()


def test_manifest_and_provenance_reject_unsafe_input(tmp_path):
    path, rows, _, _ = _manifest(tmp_path)
    dataset = BoundaryNAFDataset(rows, _config())
    Path = type(path)
    classes = Path(rows[0]['classes'])
    classes.write_text('{"1":1,"2":0}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='source changed'):
        dataset.verify_sources()
    with pytest.raises(ValueError, match='SHA256 differs'):
        BoundaryNAFDataset(path, _config())
    old = tmp_path / 'old_gt.npz'
    np.savez_compressed(old, instance_map=np.ones((32, 32), np.int32))
    with pytest.raises(ValueError, match='provenance'):
        load_processed_binary(old, classes)
    with pytest.raises(ValueError, match='Exactly 32'):
        BoundaryNAFDataset(rows, {'data': {'expected_sources': 32}})


def test_partial_unknown_cannot_create_edge_supervision(tmp_path):
    path, rows, ids, original = _manifest(tmp_path)
    gt = tmp_path / 'empty.npz'
    one = np.ones((40, 40), np.int32)
    one[12:24, 12:24] = 0
    np.savez_compressed(gt, instance_map=one, original_covered=one > 0,
                        filled=np.zeros_like(one), residual_unknown=one == 0)
    target, valid, metadata = load_processed_binary(gt, rows[0]['classes'])
    assert not target.any() and valid.any()
    assert metadata['boundary_pixels'] == 0


def test_no_boundary_candidate_falls_back_to_audited_identity(tmp_path):
    path, rows, ids, _ = _manifest(tmp_path)
    ids[:] = 1
    gt = tmp_path / 'single_gt.npz'
    np.savez_compressed(gt, instance_map=ids, original_covered=ids > 0,
                        filled=np.zeros_like(ids), residual_unknown=ids == 0)
    rows[0]['gt'] = str(gt)
    rows[0]['hashes']['gt_sha256'] = hashlib.sha256(gt.read_bytes()).hexdigest()
    dataset = BoundaryNAFDataset(rows, _config(identity_probability=0.0))
    item = dataset[0]
    assert item['identity'].item()
    assert item['meta']['requested_mode'] == 'gap' and item['meta']['mode'] == 'identity'
    assert item['meta']['boundary_pixels'] == item['meta']['gap_pixels'] == 0


def test_full_width_crop_margins_never_modify_preloaded_source_cache(tmp_path):
    path, _, _, _ = _manifest(tmp_path, shape=(192, 96))
    dataset = BoundaryNAFDataset(path, _config(crop_size=96, samples_per_epoch=None,
        identity_probability=1.0))
    before = [item.copy() for item in dataset.cache['train_001']]
    for index in range(len(dataset)):
        dataset[index]
    for old, current in zip(before, dataset.cache['train_001']):
        np.testing.assert_array_equal(old, current)
