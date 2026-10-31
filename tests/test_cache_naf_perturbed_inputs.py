# -*- coding: utf-8 -*-
"""缓存清单只接受同源训练图；阈值与持久化表示明确，不借GPU冒充验收。"""
from copy import deepcopy
import json

import numpy as np
import pytest

from tools.cache_naf_perturbed_inputs import atomic_json, array_sha, binary_record, register_training_rows


def cohort():
    old, gt = [], []
    for index in range(32):
        name = 'train_%03d'%(index+1)
        hashes = dict(image_sha256=str(index)*64, gt_sha256='b'*64, classes_sha256='c'*64)
        old.append(dict(source=name, source_index=index, image_sha256=hashes['image_sha256'],
            gt_sha256=hashes['gt_sha256'], classes_sha256=hashes['classes_sha256'], shape=[1936,2584], windows=[{}]))
        gt.append(dict(source=name, hashes=hashes, shape=[1936,2584]))
    return dict(format='boundary_naf_real_heatmaps_v1', complete=True,
        local_restoration=dict(mode='zero',model='restorer'), files=old), dict(rows=gt)


def test_registered_train_subset_preserves_order_and_never_accepts_test_sources():
    clean, gt = cohort(); old = deepcopy((clean,gt))
    rows = register_training_rows(clean,gt,only=['train_032','train_001'])
    assert [r['source'] for r in rows] == ['train_032','train_001'] and (clean,gt)==old
    with pytest.raises(ValueError): register_training_rows(clean,gt,only=['test_169'])
    with pytest.raises(ValueError): register_training_rows(clean,gt,only=['train_001']*2)


@pytest.mark.parametrize('change', ['GT','image','class','shape','source_index','complete','N0'])
def test_incomplete_or_mismatched_reference_is_rejected(change):
    clean, gt = cohort()
    if change=='GT':gt['rows'][0]['hashes']['gt_sha256']='d'*64
    if change=='image':gt['rows'][0]['hashes']['image_sha256']='d'*64
    if change=='class':gt['rows'][0]['hashes']['classes_sha256']='d'*64
    if change=='shape':gt['rows'][0]['shape']=[4,5]
    if change=='source_index':clean['files'][0]['source_index']=True
    if change=='complete':clean['complete']=False
    if change=='N0':clean['local_restoration']['mode']='single'
    with pytest.raises(ValueError):register_training_rows(clean,gt)


def test_binary_cache_is_bool_at_half_with_independent_continuous_and_binary_sha():
    boundary=np.array([[0,.49999,.5,1]],np.float32); original=boundary.copy()
    binary,meta=binary_record(boundary)
    assert binary.dtype==bool and np.array_equal(binary,[[False,False,True,True]])
    assert meta['binary_sha256']==array_sha(binary) and meta['continuous_boundary_sha256']==array_sha(boundary)
    assert not meta['continuous_boundary_saved'] and np.array_equal(boundary,original)
    assert meta['representation']=='thresholded_binary' and meta['threshold']==.5


@pytest.mark.parametrize('bad', [np.array([[.5]],np.float64),np.array([[np.nan]],np.float32),
    np.array([[-.1]],np.float32),np.array([[1.1]],np.float32),np.array([.5],np.float32)])
def test_invalid_continuous_probability_rejected(bad):
    with pytest.raises(ValueError):binary_record(bad)


def test_atomic_manifest_replacement_never_leaves_partial_json(tmp_path):
    path=tmp_path/'manifest.partial.json'
    atomic_json(path,dict(complete=False,rows=[]))
    atomic_json(path,dict(complete=False,rows=['train_001']))
    assert json.loads(path.read_text())['rows']==['train_001']
    assert not (tmp_path/'.manifest.partial.json.tmp').exists()
