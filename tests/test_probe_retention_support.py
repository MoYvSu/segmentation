# -*- coding: utf-8 -*-
"""冻结cache缺失与身份门禁，用虚构文件测试。"""
import json

import numpy as np
import pytest

from tools.probe_retention_support import _load_cache, _resolve_cache_file, _sha
from utils.retention_support import array_sha


def _cache(tmp_path):
    arrays = dict(teacher_logits=np.full((1, 8, 2, 2), -3., np.float32),
                  grid=np.ones((1, 1, 2, 2), np.float32), grid_gt=np.ones((2, 2), np.int32),
                  valid=np.ones((2, 2), bool), targets=np.ones((1, 8, 2, 2), np.float32),
                  full_legal=np.ones((1, 8, 2, 2), bool), band=np.zeros((2, 2), bool),
                  native_legal=np.ones((1, 8, 2, 2), bool), retention=np.ones((1, 8, 2, 2), bool))
    file = tmp_path/'tile_04.npz'
    np.savez_compressed(file, **arrays)
    contract = dict(format='retention_support_capture_v1', source='fictional', gt_sha256='GT_SHA',
                    checkpoint_sha256='MODEL_SHA', restorer_sha256='RESTORE_SHA',
                    initial_head_sha256='HEAD_SHA', original_boundary_sha256='BOUNDARY_SHA',
                    npz_arrays=list(arrays), tiles=[dict(index=4, box=[0, 0, 2, 2],
                        teacher_sha256=array_sha(arrays['teacher_logits']), grid_sha256=array_sha(arrays['grid']),
                        band_sha256=array_sha(arrays['band']), native_legal_sha256=array_sha(arrays['native_legal']),
                        retention_sha256=array_sha(arrays['retention']))])
    manifest = {key: value for key, value in contract.items() if key not in ('tiles', 'npz_arrays')}
    manifest.update(complete_original_stitch_exact=True, training=False,
                    tiles=[dict(index=4, box=[0, 0, 2, 2], file=file.name, file_sha256=_sha(file),
                                geometry={}, pad_h=0, pad_w=0)])
    path = tmp_path/'cache.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path, contract, arrays, manifest


def test_missing_cache_returns_unknown_marker_not_fabricated_teacher(tmp_path):
    assert _load_cache(tmp_path/'missing.json', {}) is None


def test_exact_frozen_cache_loaded_without_mutating_private_arrays(tmp_path):
    path, contract, arrays, _ = _cache(tmp_path)
    before = _sha(tmp_path/'tile_04.npz')
    loaded = _load_cache(path, contract)
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[4][1][key], value)
    assert _sha(tmp_path/'tile_04.npz') == before
    assert loaded[4][0]['cache_file_resolution']['convention'] == 'manifest_relative'


def test_explicit_project_relative_prefix_maps_only_to_same_manifest_basename(tmp_path):
    path, contract, arrays, manifest = _cache(tmp_path)
    before = _sha(tmp_path/'tile_04.npz')
    manifest['tiles'][0]['file'] = 'outputs/gt_source/retention_cache/tile_04.npz'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    loaded = _load_cache(path, contract)
    resolution = loaded[4][0]['cache_file_resolution']
    assert resolution['convention'] == 'explicit_producer_project_relative'
    assert resolution['resolved_file'] == str((tmp_path/'tile_04.npz').resolve())
    assert not resolution['filename_or_array_data_changed']
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[4][1][key], value)
    assert _sha(tmp_path/'tile_04.npz') == before


@pytest.mark.parametrize('declaration', [
    'outputs/gt_source/retention_cache/tile_05.npz',
    'outputs/gt_source/retention_cache/../retention_cache/tile_04.npz',
    'outputs/gt_source/retention_cache/sub/tile_04.npz',
    'outputs/gt_source/retention_cache/tile_04.npz/extra',
    '../tile_04.npz',
])
def test_project_prefix_index_traversal_subfolder_and_outside_paths_rejected(tmp_path, declaration):
    with pytest.raises(ValueError):
        _resolve_cache_file(tmp_path/'cache.json', dict(index=4, file=declaration))


@pytest.mark.parametrize('mutation', ['teacher_identity', 'source_identity', 'stitch', 'training',
                                     'duplicate_tile', 'box', 'file_sha', 'teacher_array_sha'])
def test_frozen_cache_identity_and_each_tile_sha_must_match(tmp_path, mutation):
    path, contract, arrays, manifest = _cache(tmp_path)
    if mutation == 'teacher_identity':
        manifest['initial_head_sha256'] = 'OTHER_HEAD'
    elif mutation == 'source_identity':
        manifest['source'] = 'OTHER_SOURCE'
    elif mutation == 'stitch':
        manifest['complete_original_stitch_exact'] = False
    elif mutation == 'training':
        manifest['training'] = True
    elif mutation == 'duplicate_tile':
        manifest['tiles'].append(manifest['tiles'][0].copy())
    elif mutation == 'box':
        manifest['tiles'][0]['box'] = [0, 0, 3, 3]
    elif mutation == 'file_sha':
        manifest['tiles'][0]['file_sha256'] = 'OTHER_SHA'
    else:
        arrays['teacher_logits'][0, 0, 0, 0] = -2.
        np.savez_compressed(tmp_path/'tile_04.npz', **arrays)
        manifest['tiles'][0]['file_sha256'] = _sha(tmp_path/'tile_04.npz')
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        _load_cache(path, contract)
