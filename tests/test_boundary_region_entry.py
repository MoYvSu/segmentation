# -*- coding: utf-8 -*-
"""区域训练入口的配方隔离、固定分母及严格checkpoint契约。"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from models.boundary_naf import BoundaryNAFNet
from models.boundary_restore import state_sha256
from utils.config import load_config
import train_boundary_region as training


def test_region_recipe_has_same_first1000_inputs_and_lr_but_no_pixel_gan_or_ph():
    previous=load_config('config/train/boundary_naf_real_bce.yaml')['boundary_naf']
    config=load_config('config/train/boundary_naf_region.yaml');cfg=training.validate_config(config)
    assert cfg['model']==previous['model'] and cfg['seed']==previous['seed']
    assert cfg['data']==previous['data'] and cfg['inference']==previous['inference']
    for key in ['effective_batch_size','learning_rate','main_schedule_updates','adam_betas','grad_clip']:
        assert cfg['training'][key]==previous['training'][key]
    assert cfg['training']['main_updates']==1000
    assert cfg['training']['topology_updates']==cfg['training']['bce_weight']==0
    for step in [0,99,999]:assert training.phase_and_rate(step,cfg)[2]==training.phase_and_rate(step,previous)[2]
    invalid=deepcopy(config);invalid['boundary_naf']['training']['bce_weight']=1
    with pytest.raises(ValueError,match='精确'):training.validate_config(invalid)


def test_step_record_preserves_total_elapsed_and_renames_region_internal_timing(tmp_path):
    stat=dict(elapsed_seconds=.125,loss_g=1.25,selected_pairs=2,
        generator_updated=True,nonzero_grad_g=True,
        images=[dict(sample=0,selected_cut_pixels=17)])
    before=deepcopy(stat)
    row=training.step_record(stat,step=1,learning_rate=.001,update_seconds=.75,
        sources=['train_001'],draws=[0],elapsed_seconds=9.5)
    # 回归第1步的重复elapsed_seconds关键字错误，同时区分三种计时范围。
    assert row['global_step']==1 and row['learning_rate']==.001
    assert row['elapsed_seconds']==9.5
    assert row['region_update_seconds']==.125
    assert row['update_seconds']==.75
    assert row['sources']==['train_001'] and row['draws']==[0]
    assert row['loss_g']==1.25 and row['selected_pairs']==2
    assert row['generator_updated'] and row['images']==before['images']
    assert stat==before  # 构造日志不能把训练返回值里的计时字段删掉。
    log=tmp_path/'steps.jsonl'
    log.write_text(json.dumps(row,allow_nan=False)+'\n',encoding='utf-8')
    assert json.loads(log.read_text(encoding='utf-8'))==row


def test_step_record_accepts_stats_without_internal_time_and_keeps_driver_metadata():
    stat=dict(loss_g=.5,global_step=999,learning_rate=.9,
              update_seconds=100,sources=['stale'],draws=[999])
    row=training.step_record(stat,step=2,learning_rate=.0008,update_seconds=.25,
        sources=['train_002'],draws=[16],elapsed_seconds=10)
    assert 'region_update_seconds' not in row
    assert row['elapsed_seconds']==10 and row['update_seconds']==.25
    assert row['global_step']==2 and row['learning_rate']==.0008
    assert row['sources']==['train_002'] and row['draws']==[16]
    assert stat['global_step']==999 and stat['update_seconds']==100


def test_checkpoint_cannot_promote_smoke_and_restores_actual_state(tmp_path):
    cfg=load_config('config/train/boundary_naf_region.yaml')['boundary_naf']
    cfg['model']=dict(width=4,encoder_blocks=[1],middle_blocks=1,decoder_blocks=[1])
    toy=BoundaryNAFNet(**cfg['model'],seed=cfg['seed'])
    cfg['region']['initial_generator_sha256']=state_sha256(toy)
    model=training.build_model(cfg,'cpu');optimizer=training.optimizer_for(model,cfg)
    identity={'purpose':'region_smoke_not_formal_resume','source':'synthetic_fixture'}
    path=tmp_path/'smoke.pt'
    training.save_checkpoint(path,model,optimizer,cfg,identity,2,[0])
    before=state_sha256(model)
    with torch.no_grad():next(model.parameters()).add_(1)
    assert training.load_checkpoint(path,model,optimizer,cfg,identity)==(2,[0])
    assert state_sha256(model)==before
    formal=dict(identity,purpose='formal_region_candidate')
    with pytest.raises(ValueError,match='用途'):training.load_checkpoint(path,model,optimizer,cfg,formal)


def test_separation_check_keeps_outside_route_and_rejects_painted_seeds():
    p=np.zeros((30,30),np.float32);p[:,15]=.95
    a=np.zeros_like(p,bool);a[10:15,5:10]=True
    b=np.zeros_like(p,bool);b[10:15,20:25]=True
    # 开放链只接到裁框，不能假定窗外不再相通。
    assert training.pair_status(p,a,b)['connected']
    p[7:19,3:12]=.95;p[9:17,5:10]=0
    assert training.pair_status(p,a,b)['separated']
    p[a]=.95
    status=training.pair_status(p,a,b)
    assert not status['separated'] and status['core_coverage'][0]==0
