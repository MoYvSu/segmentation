# -*- coding: utf-8 -*-
"""原生裁块的数据坐标、未知域、独立控制和最终输出契约。"""
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.affinity_native import NativeViewDataset, NativeDegradationDataset, geometry_sample, choose_box
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.direct_dual_head_dataset import _spatial_transform
from data.sam2_geometry_dataset import SAM2GeometryDataset
from tools.affinity_native_views import verify_prediction
from tools.run_affinity_native import SOURCES, verify_run
from train_affinity_connectivity import draw_receipt
from train_affinity_native import common_config, compare_draw
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config


@pytest.fixture
def cohort(tmp_path):
    raw, gt = tmp_path/'raw', tmp_path/'gt'
    raw.mkdir()
    gt.mkdir()
    yy, xx = np.mgrid[:48, :64]
    rgb = np.stack([xx*3, yy*4, (xx+yy)*2], -1).astype(np.uint8)
    cv2.imwrite(str(raw/'source.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (raw/'source.json').write_text('not used as GT')
    ids = np.ones((48,64), np.uint16)
    ids[:,32:] = 2
    ids[8:18,8:18] = 0
    np.savez(gt/'source_gt.npz', instance_map=ids, residual_unknown=ids==0)
    (gt/'source_class.json').write_text(json.dumps({1:1,2:0}))
    return CanonicalBackendDataset(raw,gt,image_size=32,affinity_grid=16), rgb, ids


def options():
    return dict(crop_attempts=32, minimum_known_pairs=16)


def degradation():
    return dict(profile_probabilities=[.2,.8,0,0], blur_sigma=[.4,1.5],
                resample_probability=0.,resample_scale=[.5,.9],
                spatial_blur=dict(enabled=True, probability=.5, grid_size=[3,5],strength=1.,levels=8))


@pytest.mark.parametrize('size',[32,16])
def test_crop_from_native_rgb_and_same_original_gt(cohort,size):
    base,rgb,ids = cohort
    native = NativeViewDataset(base,'manual',size,42,options())
    sample = native[0]
    y,x,y1,x1 = sample['crop_box'].tolist()
    expected = geometry_sample(rgb[y:y1,x:x1],ids[y:y1,x:x1],image_size=32,grid=16)
    assert torch.equal(sample['image'],expected['image'])
    assert torch.equal(sample['affinity_instance_map'],expected['affinity_instance_map'])
    assert y1-y == size and x1-x == size
    # 原生坐标颜色仍可回溯，不是整图先缩小后再裁。
    assert torch.allclose(sample['image'][:,0,0],torch.tensor(rgb[y,x]).float()/255)


def test_draws_replay_without_consuming_global_random_state(cohort):
    base,_,_ = cohort
    native = NativeViewDataset(base,'manual',16,42,options())
    dataset = NativeDegradationDataset(native,degradation(),{'enabled':False},42,repeats=8)
    np_before,py_before,torch_before = np.random.get_state(),random.getstate(),torch.get_rng_state().clone()
    boxes = []
    for epoch in [1,2]:
        dataset.set_epoch(epoch)
        for draw in range(len(dataset)):
            a,b = dataset[draw],dataset[draw]
            for key in ['image','clean_image','instance_map','crop_box']:
                assert torch.equal(a[key],b[key])
            boxes.append(tuple(a['crop_box'].tolist()))
    assert len(set(boxes)) > 8
    assert random.getstate() == py_before and torch.equal(torch.get_rng_state(),torch_before)
    after = np.random.get_state()
    assert np.array_equal(after[1],np_before[1]) and after[2:] == np_before[2:]


def test_geometry_sync_and_old_augmentation_receipt_preserved(cohort):
    base,rgb,ids = cohort
    for size in [32,16]:
        native = NativeViewDataset(base,'manual',size,18,options())
        new = NativeDegradationDataset(native,degradation(),{'enabled':True},18,repeats=8)
        old = PairedDegradationDataset(base,degradation(),{'enabled':True},18,repeats=8)
        old.set_epoch(3)
        new.set_epoch(3)
        for draw in range(8):
            a,b = old[draw],new[draw]
            assert draw_receipt(a) == draw_receipt(b)
            y,x,y1,x1 = b['crop_box'].tolist()
            original = geometry_sample(rgb[y:y1,x:x1],ids[y:y1,x:x1],image_size=32,grid=16)
            transform = b['horizontal_flip'],b['vertical_flip'],b['rotation_k']
            assert torch.equal(b['affinity_instance_map'],_spatial_transform(original['affinity_instance_map'],*transform))
            assert torch.equal(b['clean_image'],_spatial_transform(original['image'],*transform))


def test_unknown_and_crop_edges_do_not_generate_boundary_targets():
    ids = np.ones((16,32),np.uint16)
    ids[4:8,8:12] = 0
    sample = geometry_sample(np.full((16,32,3),150,np.uint8),ids,image_size=32,grid=16)
    target,valid = build_affinity_targets_torch(sample['instance_map'][None],sample['valid_content'][None])
    assert valid.any() and torch.all(target[valid] == 1)
    unknown = sample['instance_map'] == 0
    assert not valid[0,:,unknown].any()
    assert not sample['valid_content'][0,8:].any()


def test_geometry_only_sam2_keeps_uncovered_ignored(tmp_path,cohort):
    base,_,ids = cohort
    dataset_dir = tmp_path/'sam2'
    dataset_dir.mkdir()
    np.savez(dataset_dir/'mask.npz',instance_map=ids)
    row = dict(source_sha256='example',source_file=str(base.samples[0]),mask_file='mask.npz')
    (dataset_dir/'manifest.jsonl').write_text(json.dumps(row)+'\n')
    pseudo = SAM2GeometryDataset(dataset_dir,base.data_dir,image_size=32,output_grid=16)
    native = NativeViewDataset(pseudo,'pseudo',32,18,options())
    item = native[0]
    assert item['name'] == 'source.png' and item['source_kind'] == 'pseudo'
    assert not item['uncovered_boundary_source'] and 'semantic_target' not in item
    assert torch.equal(item['instance_map'],item['affinity_instance_map'])


def test_empty_crop_fails_without_dropping_source():
    with pytest.raises(RuntimeError,match='No supervised'):
        choose_box(np.zeros((16,16),np.uint16),8,np.random.default_rng(1),attempts=2)


def test_reading_native_samples_does_not_edit_source_files(cohort):
    base,_,_ = cohort
    paths = list(base.data_dir.iterdir())+list(base.completed_gt_dir.iterdir())
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    native = NativeViewDataset(base,'manual',16,42,options())
    for draw in range(4):
        native.draw = draw
        native[0]
    assert before == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_config_changes_only_native_options_and_output():
    new = load_config('config/train/affinity_native.yaml')
    old = load_config('config/train/affinity_connectivity.yaml')
    assert common_config(new) == common_config(old)
    assert new['affinity_native']['sizes'] == [1024,512]
    assert all(Path(name).is_file() for name in SOURCES)


def test_control_receipts_reject_changed_source_order():
    row = dict(epoch=1,step=1,seed=10,manual=['a'],pseudo=['b'],manual_draw={},pseudo_draw={})
    compare_draw(row,row.copy())
    with pytest.raises(RuntimeError,match='manual'):
        compare_draw(row,{**row,'manual':['c']})


def test_independent_two_arm_completion_check(tmp_path):
    state = dict(status='complete',updates=8,paired_control_updates=8,
        reused_control_verified=True,frozen_unchanged=True,restorer_unchanged=True,
        affinity_changed=True,strict_reload_equal=True,initial_state_sha256='initial',
        frozen_state_sha256='frozen',restorer_state_sha256='d5a',data={'parent_data':{'sources':32}})
    for size in [1024,512]:
        root = tmp_path/f'p{size}'
        root.mkdir()
        (root/'status.json').write_text(json.dumps(state))
        (root/'steps.jsonl').write_text('{}\n'*8)
    assert verify_run(tmp_path,8)['independent_initialization']
    (tmp_path/'p512/status.json').write_text(json.dumps({**state,'initial_state_sha256':'inherited1024'}))
    with pytest.raises(RuntimeError,match='independent initialization'):
        verify_run(tmp_path,8)


def test_submission_contract_rejects_uint8_and_mismatched_ids():
    ids = np.ones((5,7),np.uint16)
    verify_prediction(ids,{'1':1},(5,7))
    with pytest.raises(RuntimeError,match='PNG'):
        verify_prediction(ids.astype(np.uint8),{'1':1},(5,7))
    with pytest.raises(RuntimeError,match='class'):
        verify_prediction(ids,{'2':1},(5,7))
