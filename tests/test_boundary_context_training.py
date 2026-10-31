# -*- coding: utf-8 -*-
"""RGB上下文训练的预算、共同初始化、实际更新与续训契约；仅合成CPU数据。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from models.boundary_restore import BoundaryRestoreNet, FORMAT as OLD_FORMAT, state_sha256
from train_boundary_context import (
    DATA_MODE, build_dataset, build_identity, build_model, collate_samples, load_resume,
    optimization_step, save_checkpoint, smoke, train, validate_recipe,
    deterministic_flags, deterministic_resume_probe,
)
from train_boundary_restore import _optimizer
from utils.config import load_config


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(tmp_path):
    config=load_config('config/train/boundary_context.yaml')
    cfg=config['boundary_restore'];cfg['model']['width']=4
    for key in ('manifest','real_manifest'):
        path=tmp_path/(key+'.json');path.write_text('{}');cfg[key]=str(path)
    base=BoundaryRestoreNet(**cfg['model'],seed=cfg['seed'])
    checkpoint=tmp_path/'old_e0.pt'
    torch.save(dict(format=OLD_FORMAT,architecture=base.architecture,epoch=0,global_step=0,
                    model_state_dict=base.state_dict(),model_sha256=state_sha256(base),
                    initial_state_sha256=base.initial_state_sha256),checkpoint)
    cfg['shared_e0_checkpoint']=str(checkpoint)
    return config,build_model(config)


class ToyDataset:
    identity={'data_mode':DATA_MODE,'test_fixture':True}
    epoch=1
    plan=[{'source':'toy','draw_seed':index} for index in range(256)]
    def __len__(self):return 256
    def verify_sources(self):return 'synthetic_test_fixture'
    def set_epoch(self,epoch):self.epoch=epoch
    def plan_receipt(self):return {'epoch':self.epoch,'views':256}
    def __getitem__(self,index):
        target=torch.full((1,24,24),.12)
        target[:,4:20,10:13]=.8
        image=target.clone();gap=torch.zeros_like(image);noise=torch.zeros_like(image)
        if index%3==1:
            image[:,9:14,9:14]=.05;gap[:,9:14,9:14]=1
        if index%3==2:
            image[:,9:14,17:20]=.6;noise[:,9:14,17:20]=1
        grid=torch.arange(24*24,dtype=torch.float32).reshape(1,24,24)/(24*24)
        context=torch.cat((grid,grid.flip(1),torch.full_like(grid,.25)),0)
        return dict(input=image,target=target,valid=torch.ones_like(image),gap_mask=gap,noise_mask=noise,
            identity=torch.tensor(index%3==0),context_rgb=context,source='toy',meta={'index':index})


def test_budget_and_original_supervision_unchanged():
    current=load_config('config/train/boundary_context.yaml');cfg=validate_recipe(current)
    old=load_config('config/train/boundary_real.yaml')['boundary_restore']
    for key in ('loss','real_recipe','seed','model','updates_per_epoch','effective_batch_size',
                'micro_batch_size','epochs','schedule_epochs','learning_rate','monitor_epochs'):
        assert cfg[key]==old[key]
    assert cfg['epochs']*cfg['updates_per_epoch']==320
    invalid=deepcopy(current);invalid['boundary_restore']['data_mode']='real_n0_heatmaps'
    with pytest.raises(ValueError,match='explicit'):
        validate_recipe(invalid)


def test_factory_only_adds_context_and_same_monitor_seed(monkeypatch):
    import data.boundary_context as context
    calls=[]
    monkeypatch.setattr(context,'BoundaryContextDataset',lambda *args,**kwargs:calls.append((args,kwargs)))
    config=load_config('config/train/boundary_context.yaml')
    build_dataset(config);build_dataset(config,917)
    assert calls[1][1]['seed']==calls[0][1]['seed']+917
    assert {k:v for k,v in calls[0][1].items() if k!='seed'}=={k:v for k,v in calls[1][1].items() if k!='seed'}


def test_identity_requires_checked_old_random_e0_and_binds_new_monitor(tmp_path):
    config,model=fixture(tmp_path);dataset=ToyDataset()
    identity=build_identity(model,dataset,config)
    assert identity['initialization_receipt']['source']=='sealed_old_random_e0'
    assert 'tools/boundary_context_monitor.py' in identity['runtime_sources']
    extra=tmp_path/'new_controls.json';extra.write_text('{}')
    config['boundary_restore']['monitor']['additional_control_manifest']=str(extra)
    assert build_identity(model,dataset,config)['monitor_asset_sha256']['additional_control_manifest']
    model.initialization_receipt['source']='reconstructed_same_seed_random_e0'
    with pytest.raises(ValueError,match='checked common'):
        build_identity(model,dataset,config)


def test_two_updates_train_rgb_weights_and_exact_next_step_resume(tmp_path,monkeypatch):
    import train_boundary_context as trainer
    config,model=fixture(tmp_path);dataset=ToyDataset();seen=[]
    original_flags=deterministic_flags();steps=[]
    numerical_flags=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    def tracked_step(*args,**kwargs):
        steps.append(deterministic_flags())
        assert (torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)==numerical_flags
        return optimization_step(*args,**kwargs)
    monkeypatch.setattr(trainer,'optimization_step',tracked_step)
    def callback(net,cfg,out,epoch,device):
        seen.append(epoch)
        assert not net.training and not torch.is_grad_enabled()
        assert deterministic_flags()==original_flags
        torch.rand(4)  # 监控不能改变正式随机序列。
    report=smoke(model,dataset,config,tmp_path/'smoke',callback)
    assert seen==[0,1]
    assert report['optimizer_updates']==2 and report['optimizer_resume_next_step_exact']
    assert report['strict_reload_prediction_equal']
    assert steps==[original_flags,original_flags,dict(enabled=True,warn_only=False),dict(enabled=True,warn_only=False)]
    assert report['probe_deterministic'] and report['probe_flags_restored']
    assert report['probe_flags_before']==report['probe_flags_after']==original_flags
    assert deterministic_flags()==original_flags
    assert torch.count_nonzero(model.stem.weight[:,1:])>0
    saved=torch.load(tmp_path/'smoke/smoke.pt',weights_only=False)
    assert saved['global_step']==2
    assert all(int(value['step'])==2 for value in saved['optimizer']['state'].values())


def test_resume_rejects_recipe_or_step_mismatch_and_fresh_rejects_trained(tmp_path):
    config,model=fixture(tmp_path);cfg=config['boundary_restore'];dataset=ToyDataset()
    identity=build_identity(model,dataset,config);optimizer=_optimizer(model,cfg)
    batch=collate_samples([dataset[i] for i in range(4)])
    row=optimization_step(model,optimizer,[batch],cfg,0)
    assert row['nonzero_gradient']
    path=tmp_path/'trained.pt';save_checkpoint(path,model,optimizer,identity,1,cfg,[])
    fresh=build_model(config);fresh_optimizer=_optimizer(fresh,cfg)
    assert load_resume(path,fresh,fresh_optimizer,identity,cfg)[0]==1
    wrong=deepcopy(identity);wrong['recipe']['learning_rate']*=2
    with pytest.raises(ValueError,match='identity differs'):
        load_resume(path,fresh,fresh_optimizer,wrong,cfg)
    with pytest.raises(ValueError,match='trained weights are forbidden'):
        train(model,dataset,config,tmp_path/'run',lambda *args:None)
    payload=torch.load(path,weights_only=False);payload['scheduler']['next_step']=3;torch.save(payload,path)
    with pytest.raises(ValueError,match='cosine schedule'):
        load_resume(path,fresh,fresh_optimizer,identity,cfg)


def test_resume_rejects_optimizer_step_or_missing_parameter_state(tmp_path):
    config,model=fixture(tmp_path);cfg=config['boundary_restore'];dataset=ToyDataset()
    identity=build_identity(model,dataset,config);optimizer=_optimizer(model,cfg)
    batch=collate_samples([dataset[i] for i in range(4)])
    optimization_step(model,optimizer,[batch],cfg,0)
    path=tmp_path/'step.pt';save_checkpoint(path,model,optimizer,identity,1,cfg,[])
    payload=torch.load(path,weights_only=False)
    first=next(iter(payload['optimizer']['state']))
    wrong=deepcopy(payload);wrong['optimizer']['state'][first]['step'].fill_(2)
    torch.save(wrong,path)
    with pytest.raises(ValueError,match='optimizer internal step'):
        load_resume(path,model,optimizer,identity,cfg)
    wrong=deepcopy(payload);del wrong['optimizer']['state'][first]
    torch.save(wrong,path)
    with pytest.raises(ValueError,match='optimizer parameter coverage'):
        load_resume(path,model,optimizer,identity,cfg)


@pytest.mark.parametrize('enabled,warn_only',[(False,False),(False,True),(True,False),(True,True)])
def test_deterministic_probe_restores_both_flags_even_on_exception(enabled,warn_only):
    outer=deterministic_flags()
    tf32=(torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.use_deterministic_algorithms(enabled,warn_only=warn_only)
        expected=dict(enabled=enabled,warn_only=warn_only)
        with deterministic_resume_probe():
            assert deterministic_flags()==dict(enabled=True,warn_only=False)
        assert deterministic_flags()==expected
        with pytest.raises(RuntimeError,match='probe failure'):
            with deterministic_resume_probe():
                assert deterministic_flags()==dict(enabled=True,warn_only=False)
                raise RuntimeError('probe failure')
        assert deterministic_flags()==expected
        assert (torch.backends.cudnn.allow_tf32,torch.backends.cuda.matmul.allow_tf32)==tf32
    finally:
        torch.use_deterministic_algorithms(outer['enabled'],warn_only=outer['warn_only'])
