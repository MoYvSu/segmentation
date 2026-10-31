# -*- coding: utf-8 -*-
"""关系数据的来源、配对增强与监督隔离契约。"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.boundary_relation import BoundaryRelationDataset, collate_boundary_relation, file_sha256


def fixture_data(root, *, labels=(0, 1), extra_field=False):
    root = Path(root)
    n, size = len(labels), 256
    rgb = np.zeros((n, 3, size, size), np.uint8)
    rgb[:, 0, 40:110, 20:90] = 255
    maps = np.zeros((n, 2, size, size), np.float16)
    maps[:, 0, 40:110, 20:90] = 1
    masks = np.zeros((n, 4, size, size), np.uint8)
    masks[:, 0, :, :127] = 1
    masks[:, 1, :, 129:] = 1
    masks[:, 2, :, 127:129] = 1
    masks[:, 3] = 1
    arrays = dict(rgb=rgb, maps=maps, masks=masks, labels=np.array(labels, np.uint8))
    if extra_field:
        arrays['GT_original_covered'] = masks[:, 3]
    path = root/'patches.npz'
    np.savez_compressed(path, **arrays)
    samples = [dict(sample_id=f'train_001:{i}', source='train_001', label=int(y)) for i, y in enumerate(labels)]
    entry = dict(path=path.name, sha256=file_sha256(path), source='train_001', count=n, samples=samples)
    manifest = dict(format='boundary_relation_data_v1', complete=True, patch_size=256,
                    shards=[entry], monitor_shards=[])
    output = root/'manifest.json'
    output.write_text(json.dumps(manifest), encoding='utf8')
    return output


def test_two_modes_share_data_and_geometry_without_mutation(tmp_path):
    path = fixture_data(tmp_path)
    rgb = BoundaryRelationDataset(path, mode='rgb')
    maps = BoundaryRelationDataset(path, mode='maps')
    assert rgb.identity_sha256 == maps.identity_sha256
    before = file_sha256(tmp_path/'patches.npz')
    for epoch in range(1, 6):
        rgb.set_epoch(epoch)
        maps.set_epoch(epoch)
        a, b = rgb[0], maps[0]
        assert torch.equal(a['input'][3:], b['input'][3:])
        assert torch.equal(a['masks'], b['masks'])
        assert torch.count_nonzero(b['input'][:3]) == 0
        assert a['meta']['geometry_D4'] == b['meta']['geometry_D4']
        # 两个记录相同形状的信号必须保持逐像素对齐。
        assert torch.equal((a['input'][0]+1)/2, a['input'][3])
    assert before == file_sha256(tmp_path/'patches.npz')
    sample = rgb.raw(0)
    sample['rgb'].fill(111)
    assert not np.all(rgb.raw(0)['rgb'] == 111)


def test_collate_and_unaugmented_monitor(tmp_path):
    dataset = BoundaryRelationDataset(fixture_data(tmp_path))
    dataset.set_epoch(4)
    batch = collate_boundary_relation([dataset.tensors(i, augment=False) for i in range(2)])
    assert batch['input'].shape == (2, 5, 256, 256)
    assert batch['masks'].shape == (2, 4, 256, 256)
    assert batch['label'].tolist() == [0., 1.]
    assert [m['geometry_D4'] for m in batch['meta']] == [0, 0]


def test_unlabeled_monitor_cannot_enter_training(tmp_path):
    path = fixture_data(tmp_path, labels=(255,))
    with pytest.raises(ValueError, match='训练仅0/1'):
        BoundaryRelationDataset(path)
    manifest = json.loads(path.read_text(encoding='utf8'))
    manifest['monitor_shards'], manifest['shards'] = manifest['shards'], []
    path.write_text(json.dumps(manifest), encoding='utf8')
    data = BoundaryRelationDataset(path, split='monitor', augment=False)
    assert data.labels.tolist() == [255]


def test_gt_input_field_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='字段不匹配'):
        BoundaryRelationDataset(fixture_data(tmp_path, extra_field=True))


def test_sha_and_memory_limits_fail_closed(tmp_path):
    path = fixture_data(tmp_path)
    with pytest.raises(MemoryError):
        BoundaryRelationDataset(path, max_ram_gib=.00001)
    shard = tmp_path/'patches.npz'
    shard.write_bytes(shard.read_bytes()+b'altered')
    with pytest.raises(ValueError, match='SHA不匹配'):
        BoundaryRelationDataset(path)


def test_manifest_cannot_escape_data_root(tmp_path):
    path = fixture_data(tmp_path)
    manifest = json.loads(path.read_text(encoding='utf8'))
    manifest['shards'][0]['path'] = '../patches.npz'
    path.write_text(json.dumps(manifest), encoding='utf8')
    with pytest.raises(ValueError, match='路径越界'):
        BoundaryRelationDataset(path)
