# -*- coding: utf-8 -*-
"""随机完整头训练的CPU合同；不载赛题数据或运行CUDA。"""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tools import train_semantic_native_random as random_train
from tools import train_semantic_native_zero as base


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(nn.Conv2d(3,4,1),nn.GroupNorm(2,4),nn.ReLU())
        self.out = nn.Conv2d(4,1,1)

    def reset_output(self):
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, image):
        return self.out(self.rgb(image))


class Head(nn.Module):
    semantic_residual_version = 'highres_v1'
    def __init__(self):
        super().__init__()
        self.seg_fpn = nn.Sequential(nn.Conv2d(3,4,1),nn.GroupNorm(2,4),nn.ReLU())
        self.seg_branch = nn.Sequential(nn.Dropout(.2),nn.Conv2d(4,1,1))
        self.semantic_residual = Residual()

    def forward(self, features, image):
        assert features[0] is image
        return self.seg_branch(self.seg_fpn(features[0])) + self.semantic_residual(image)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3,3,1)
        self.affinity_decoder = nn.Conv2d(3,8,1)
        self.semantic_decoder = Head()
        self.register_buffer('frozen_buffer',torch.tensor([3.]))


def make_model():
    torch.manual_seed(777)
    return Model()


def count(model):
    return sum(p.numel() for p in model.semantic_decoder.parameters())


def test_full_reset_norm_zero_output_and_rng_unchanged():
    model = make_model()
    before = base.rng_state()
    old = base.tensor_sha(model.semantic_decoder.state_dict().items())
    frozen = base.frozen_sha(model)
    receipt = random_train.reset_semantic(model,expected_parameters=count(model))
    assert receipt['base_original_semantic_sha256'] == old
    assert receipt['random_initial_semantic_sha256'] != old
    assert base.frozen_sha(model) == frozen
    assert torch.equal(before['torch'],base.rng_state()['torch'])
    assert receipt['cpu_rng_before_sha256'] == receipt['cpu_rng_after_sha256']
    assert receipt['torch_rng_preserved']
    assert all(row['reset_applied'] for row in receipt['parameters'])
    assert len(receipt['parameters']) == len(list(model.semantic_decoder.parameters()))
    assert all(row['before_sha256'] != row['after_sha256'] for row in receipt['submodules'].values())
    for module in model.semantic_decoder.modules():
        if isinstance(module,nn.GroupNorm):
            assert torch.equal(module.weight,torch.ones_like(module.weight))
            assert not torch.count_nonzero(module.bias)
    assert not torch.count_nonzero(model.semantic_decoder.semantic_residual.out.weight)
    assert not torch.count_nonzero(model.semantic_decoder.semantic_residual.out.bias)
    twin = make_model()
    random_train.reset_semantic(twin,expected_parameters=count(twin))
    assert base.tensor_sha(twin.semantic_decoder.state_dict().items()) == receipt['random_initial_semantic_sha256']


def test_reject_partial_head_or_another_init_seed():
    model=make_model()
    with pytest.raises(ValueError,match='1101'):
        random_train.reset_semantic(model,seed=1102,expected_parameters=count(model))
    model.semantic_decoder.seg_fpn.add_module('unsupported',nn.Linear(1,1))
    with pytest.raises(ValueError,match='reset rule'):
        random_train.reset_semantic(model,expected_parameters=count(model))


def test_isolated_engine_schedule_config_and_public_module_unchanged():
    original_format, original_load, original_json = base.FORMAT,base.backend_adaptation.load_backend,base.json
    engine=random_train.make_engine()
    assert engine is not base and engine.FORMAT == random_train.FORMAT
    assert base.FORMAT == original_format and base.backend_adaptation.load_backend is original_load
    assert base.json is original_json and base.FIXED['learning_rate'] == 3e-5
    assert engine.learning_rate(1) == 5e-5
    assert engine.learning_rate(2) == engine.learning_rate(3) == 1e-4
    assert engine.learning_rate(20) == pytest.approx(1e-5)
    cfg=dict(semantic_native_zero=deepcopy(random_train.FIXED),
        semantic_native_random=dict(initialization=random_train.INITIALIZATION,initialization_seed=1101),
        direct_semantic_affinity=dict(semantic_loss=deepcopy(base.LOSS)))
    result=engine.validate_config(cfg)
    assert result['seed'] == 20261031 and result['initialization_seed'] == 1101
    cfg['semantic_native_random']['initialization_seed']=1102
    with pytest.raises(ValueError):
        engine.validate_config(cfg)
    assert engine.source_seal()['tools/train_semantic_native_random.py'] == base.file_sha(random_train.__file__)


def toy_identity(model,receipt):
    return dict(format=random_train.FORMAT,recipe={'test':'toy'},
        base_backend_file_sha256=base.BACKEND_SHA,
        base_semantic_sha256=receipt['random_initial_semantic_sha256'],
        semantic_architecture={'test':'toy_full'},
        frozen=dict(frozen_backend_sha256=base.frozen_sha(model),init_receipt=receipt))


def test_identity_json_keeps_original_base_and_random_initial_separate():
    model=make_model()
    receipt=random_train.reset_semantic(model,expected_parameters=count(model))
    raw=toy_identity(model,receipt)
    normalized=random_train.IdentityJSON.loads(json.dumps(raw))
    assert normalized['base_semantic_sha256'] == receipt['base_original_semantic_sha256']
    assert normalized['random_initial_semantic_sha256'] == receipt['random_initial_semantic_sha256']
    assert normalized['base_original_semantic_sha256'] == receipt['base_original_semantic_sha256']
    assert random_train.IdentityJSON.loads(json.dumps(normalized)) == normalized
    assert json.loads(json.dumps(raw)) == raw  # 全局json仍无改动


def small_features(model,restorer,image):
    return image,[image],dict(noise_nonzero=0,initial_equals_condition=True,q_sample_calls=1)


def criterion(prediction,target,instance_map):
    loss=(prediction[:,0]-target[:,0]).square().mean()
    return loss,float(loss.detach()),0.


def sample():
    return dict(image=torch.arange(3*8*8).reshape(3,8,8).float()/192,
                semantic_target=torch.ones(1,8,8),semantic_boundary=torch.zeros(1,8,8),
                semantic_instance_map=torch.ones(8,8,dtype=torch.long))


def test_random_checkpoint_next_update_exact_and_old_format_rejected(tmp_path):
    torch.set_num_threads(1)
    engine=random_train.make_engine()
    model=make_model()
    receipt=random_train.reset_semantic(model,expected_parameters=count(model))
    identity=random_train.normalize_identity(toy_identity(model,receipt))
    model.requires_grad_(False)
    model.semantic_decoder.requires_grad_(True)
    engine.train_mode(model)
    optimizer=torch.optim.AdamW(model.semantic_decoder.parameters(),lr=engine.learning_rate(1),
                                eps=1e-4,weight_decay=1e-4,foreach=False)
    engine.restored_features=small_features
    engine.ensure_space=lambda *args,**kwargs:None
    engine.semantic_update(model,None,criterion,optimizer,sample(),torch.device('cpu'))
    engine.save_delta(tmp_path/'last.pt',model,optimizer,identity,dict(completed_updates=1),output_dir=tmp_path)
    reference=engine.semantic_update(model,None,criterion,optimizer,sample(),torch.device('cpu'))
    head=deepcopy(model.semantic_decoder.state_dict())
    opt=deepcopy(optimizer.state_dict())
    rng=engine.rng_state()
    resumed=make_model()
    random_train.reset_semantic(resumed,expected_parameters=count(resumed))
    resumed.requires_grad_(False)
    resumed.semantic_decoder.requires_grad_(True)
    other=torch.optim.AdamW(resumed.semantic_decoder.parameters(),lr=engine.learning_rate(1),eps=1e-4,weight_decay=1e-4,foreach=False)
    loaded=torch.load(tmp_path/'last.pt',weights_only=False)
    assert loaded['format'] == random_train.FORMAT
    load_receipt=engine.load_head_delta(resumed,tmp_path/'last.pt')
    assert load_receipt['base_semantic_sha256'] == receipt['base_original_semantic_sha256']
    assert load_receipt['random_initial_semantic_sha256'] == receipt['random_initial_semantic_sha256']
    other.load_state_dict(loaded['optimizer'])
    engine.restore_rng(loaded['rng'])
    engine.train_mode(resumed)
    replay=engine.semantic_update(resumed,None,criterion,other,sample(),torch.device('cpu'))
    assert replay == reference
    assert all(torch.equal(value,resumed.semantic_decoder.state_dict()[key]) for key,value in head.items())
    assert other.state_dict()['param_groups'] == opt['param_groups']
    assert all(torch.equal(value,other.state_dict()['state'][key][field])
               for key,state in opt['state'].items() for field,value in state.items())
    assert torch.equal(rng['torch'],engine.rng_state()['torch'])
    loaded['format']=base.FORMAT
    torch.save(loaded,tmp_path/'old.pt')
    with pytest.raises(ValueError,match='random_delta_v1'):
        engine.load_head_delta(resumed,tmp_path/'old.pt')
