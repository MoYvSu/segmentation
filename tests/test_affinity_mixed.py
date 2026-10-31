# -*- coding: utf-8 -*-
"""混合视野必须只改变读图路径，保持相同更新中的两种来源配对与预算。"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.affinity_mixed import MixedViewDataset, draw_views, view_plan
from data.affinity_native import NativeDegradationDataset, NativeViewDataset
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.direct_dual_head_dataset import _spatial_transform
from train_affinity_connectivity import draw_receipt
from train_affinity_native import common_config
from tools.run_affinity_mixed import SOURCES
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config


def receipts(count, epoch=1):
    manual=np.random.default_rng(11).permutation(count)
    pseudo=np.random.default_rng(12).permutation(count)
    return [dict(epoch=epoch,step=i+1,manual_draw={'draw_index':[int(a)]},
        pseudo_draw={'draw_index':[int(b)]}) for i,(a,b) in enumerate(zip(manual,pseudo))]


def test_exact_half_updates_and_same_view_for_both_sources():
    np_before=np.random.get_state();py_before=random.getstate();torch_before=torch.get_rng_state()
    for epoch in range(1,21):
        plan=view_plan(20260930,epoch,64);rows=receipts(64,epoch)
        assert plan.count(0)==plan.count(1024)==32
        a,b=[draw_views(rows,k,epoch,plan) for k in ('manual','pseudo')]
        for row,view in zip(rows,plan):
            assert a[row['manual_draw']['draw_index'][0]]==b[row['pseudo_draw']['draw_index'][0]]==view
        assert plan==view_plan(20260930,epoch,64)
    assert random.getstate()==py_before and torch.equal(torch_before,torch.get_rng_state())
    after=np.random.get_state()
    assert np.array_equal(after[1],np_before[1]) and after[2:]==np_before[2:]
    assert set(view_plan(20260930,1,64)[:8])=={0,1024}


def test_invalid_reference_schedule_rejected():
    rows=receipts(8);plan=view_plan(42,1,8)
    with pytest.raises(ValueError,match='Incomplete'):
        draw_views(rows[:-1],'manual',1,plan)
    rows[0]['manual_draw']=rows[1]['manual_draw']
    with pytest.raises(ValueError,match='Repeated'):
        draw_views(rows,'manual',1,plan)


def test_both_paths_reproduce_existing_inputs_targets_and_augmentation(tmp_path):
    raw=tmp_path/'raw';gt=tmp_path/'gt';raw.mkdir();gt.mkdir()
    yy,xx=np.mgrid[:48,:64]
    rgb=np.stack([xx*3,yy*4,(xx+yy)*2],-1).astype(np.uint8)
    cv2.imwrite(str(raw/'example.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    (raw/'example.json').write_text('source identity only')
    ids=np.ones((48,64),np.uint16);ids[:,32:]=2;ids[6:12,6:12]=0
    np.savez(gt/'example_gt.npz',instance_map=ids,residual_unknown=ids==0)
    (gt/'example_class.json').write_text(json.dumps({1:1,2:0}))
    base=CanonicalBackendDataset(raw,gt,image_size=32,affinity_grid=16)
    degradation=dict(profile_probabilities=[.2,.8,0,0],blur_sigma=[.4,1.5],resample_probability=.5,
        resample_scale=[.65,1.],spatial_blur=dict(enabled=True,probability=.5,grid_size=[3,5],strength=1.,levels=8))
    noise=dict(enabled=True,probability=.5,sigma=[0,.006])
    original=PairedDegradationDataset(base,degradation,noise,42,repeats=8)
    options=dict(mixed_view_seed=20260930,crop_attempts=32,minimum_known_pairs=16)
    mixed=MixedViewDataset(original,'manual',options,receipts(8))
    # 小型坐标合成图用16原生像素；视野分配机制保持相同。
    mixed.native.base_dataset.size=16
    native=NativeDegradationDataset(NativeViewDataset(base,'manual',16,42,options),degradation,noise,42,repeats=8)
    for ds in (original,native,mixed):ds.set_epoch(1)
    fields=('image','instance_map','affinity_instance_map','foreground','valid_content','affinity_valid_content')
    for index in range(8):
        current=mixed[index]
        expected=(original if current['training_view']=='whole' else native)[index]
        assert draw_receipt(current)==draw_receipt(expected)
        for key in fields:assert torch.equal(current[key],expected[key]),key
        assert torch.equal(current['image'],mixed[index]['image'])
        target,valid=build_affinity_targets_torch(current['instance_map'][None],current['valid_content'][None])
        assert not valid[0,:,current['instance_map']==0].any()
        if current['training_view']=='whole':
            assert current['crop_box'].tolist()==[0,0,48,64]
            clean=_spatial_transform(base[0]['image'],current['horizontal_flip'],current['vertical_flip'],current['rotation_k'])
            assert torch.equal(clean,current['clean_image'])


def test_common_configuration_and_frozen_source_list():
    cfg=load_config('config/train/affinity_mixed.yaml')
    assert common_config(cfg)==common_config(load_config('config/train/affinity_native.yaml'))
    assert common_config(cfg)==common_config(load_config('config/train/affinity_connectivity.yaml'))
    assert cfg['backend_adaptation']['epochs']==20
    assert cfg['backend_adaptation']['restoration']['checkpoint']=='outputs/d5a/candidate/epoch_060.pt'
    assert all(Path(name).is_file() for name in SOURCES)
