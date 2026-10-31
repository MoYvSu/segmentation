# -*- coding: utf-8 -*-
"""结构补边入口的控制组等价、真实梯度、续训与过程图；仅 CPU 合成数据。"""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from models.boundary_restore import BoundaryRestoreNet, FORMAT as OLD_FORMAT, state_sha256
from train_boundary_context import (
    DATA_MODE, STRUCTURE_MODE, SUPERVISION_VERSION, base_supervision_batch, build_identity,
    build_model, collate_samples, combined_loss, load_resume, optimization_step,
    probe_structural_gradient, save_checkpoint, validate_recipe,
)
from train_boundary_restore import _optimizer
from utils.config import load_config


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup(tmp_path):
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
    structured=deepcopy(config)
    structured['boundary_restore'].update(data_mode=STRUCTURE_MODE,
        structure=load_config('config/train/boundary_structure.yaml')['boundary_restore']['structure'])
    return config,structured


def sample(index=0,*,structured=True,empty=False):
    target=torch.full((1,24,24),.12);target[:,3:21,3:5]=.8
    image=target.clone();gap=torch.zeros_like(image);noise=torch.zeros_like(image)
    if index%3==1:
        image[:,9:14,10:14]=.05;gap[:,9:14,10:14]=1
    if index%3==2:
        image[:,9:14,17:20]=.6;noise[:,9:14,17:20]=1
    result=dict(input=image,target=target,valid=torch.ones_like(image),gap_mask=gap,
        noise_mask=noise,identity=torch.tensor(index%3==0),
        context_rgb=torch.full((3,24,24),.4),source='toy',meta={'index':index})
    if structured:
        yy=np.arange(4,20,dtype=np.int32)
        coords=np.stack((np.repeat(yy[:,None],3,axis=1),
                         np.repeat(np.arange(11,14,dtype=np.int32)[None],len(yy),axis=0)),axis=-1)
        paths=[] if empty else [dict(coords_yx=coords,allowed=np.ones(coords.shape[:2],bool),
            closed=False,anchor_index=None,meta=dict(pair_ids=[1,2]))]
        corridor=torch.zeros_like(image,dtype=torch.bool)
        if not empty:corridor[:,4:20,11:14]=True
        result.update(structure_paths=paths,structure_corridor=corridor,
            structure_free_mask=corridor.clone(),structure_cores=torch.zeros_like(corridor))
    return result


def test_recipe_retains_exact_control_budget_and_schedule():
    old=load_config('config/train/boundary_context.yaml')['boundary_restore']
    current=validate_recipe(load_config('config/train/boundary_structure.yaml'))
    for key in old:
        if key not in ('data_mode','output_dir'):
            assert current[key]==old[key]
    assert current['epochs']==5 and current['updates_per_epoch']==64
    assert current['monitor_epochs']==[0,1,3,5]
    bad=load_config('config/train/boundary_structure.yaml')
    bad['boundary_restore']['structure']['weight']=-1
    with pytest.raises(ValueError,match='nonnegative'):validate_recipe(bad)


def test_zero_coefficient_has_exact_control_loss_gradient_and_update(tmp_path):
    control,structured=setup(tmp_path)
    structured['boundary_restore']['structure']['weight']=0.
    first,second=build_model(control),build_model(structured)
    old_batch=collate_samples([sample(i,structured=False) for i in range(4)])
    new_batch=collate_samples([sample(i) for i in range(4)])
    optimizers=[_optimizer(first,control['boundary_restore']),_optimizer(second,structured['boundary_restore'])]
    old_row=optimization_step(first,optimizers[0],[old_batch],control['boundary_restore'],0)
    new_row=optimization_step(second,optimizers[1],[new_batch],structured['boundary_restore'],0)
    assert old_row==new_row
    assert state_sha256(first)==state_sha256(second)
    assert all(torch.equal(a.grad,b.grad) for a,b in zip(first.parameters(),second.parameters()))


def test_empty_candidates_fall_back_to_original_pair_and_update(tmp_path):
    control,structured=setup(tmp_path)
    first,second=build_model(control),build_model(structured)
    old_batch=collate_samples([sample(i,structured=False) for i in range(4)])
    new_batch=collate_samples([sample(i,empty=True) for i in range(4)])
    assert base_supervision_batch(new_batch,structured['boundary_restore']) is new_batch
    old=optimization_step(first,_optimizer(first,control['boundary_restore']),[old_batch],control['boundary_restore'],0)
    new=optimization_step(second,_optimizer(second,structured['boundary_restore']),[new_batch],structured['boundary_restore'],0)
    assert old['loss']==new['loss'] and new['structure_point_count']==0
    assert state_sha256(first)==state_sha256(second)


def test_release_is_shallow_copy_and_preserves_strong_or_unknown(tmp_path):
    _,config=setup(tmp_path);cfg=config['boundary_restore']
    batch=collate_samples([sample(1)])
    original={key:value.clone() for key,value in batch.items() if torch.is_tensor(value)}
    released=base_supervision_batch(batch,cfg)
    assert released is not batch
    assert not released['valid'][batch['structure_free_mask']].any()
    assert not released['gap_mask'][batch['structure_free_mask']].any()
    for key,value in original.items():assert torch.equal(batch[key],value)
    broken=dict(batch);broken['structure_free_mask']=batch['structure_free_mask'].clone()
    broken['structure_free_mask'][0,0,4,3]=True
    with pytest.raises(ValueError,match='strong N0'):base_supervision_batch(broken,cfg)
    broken=dict(batch);broken['valid']=batch['valid'].clone();broken['valid'][0,0,4,11]=0
    with pytest.raises(ValueError,match='unknown or padded'):base_supervision_batch(broken,cfg)


def test_structural_only_gradient_is_real_and_does_not_update_parameters(tmp_path):
    _,config=setup(tmp_path);cfg=config['boundary_restore'];model=build_model(config)
    batch=collate_samples([sample(i) for i in range(4)])
    digest=state_sha256(model)
    report=probe_structural_gradient(model,[batch],cfg)
    assert report['gradient_norm']>0 and report['normalizers']['points']==64
    assert report['optimizer_updates']==0 and state_sha256(model)==digest
    assert all(parameter.grad is None for parameter in model.parameters())
    optimizer=_optimizer(model,cfg)
    row=optimization_step(model,optimizer,[batch],cfg,0)
    assert row['structure_loss']>0 and row['structure_point_count']==64
    assert state_sha256(model)!=digest


def test_structure_identity_and_resume_refuse_changed_recipe(tmp_path):
    control,config=setup(tmp_path);cfg=config['boundary_restore'];model=build_model(config)
    class Dataset:identity={'data_mode':STRUCTURE_MODE,'synthetic':True}
    identity=build_identity(model,Dataset(),config)
    assert identity['supervision_version']==SUPERVISION_VERSION
    assert 'models/boundary_structure_loss.py' in identity['runtime_sources']
    class Old:identity={'data_mode':DATA_MODE,'synthetic':True}
    old_identity=build_identity(build_model(control),Old(),control)
    assert 'supervision_version' not in old_identity
    assert 'models/boundary_structure_loss.py' not in old_identity['runtime_sources']
    batch=collate_samples([sample(i) for i in range(4)]);optimizer=_optimizer(model,cfg)
    optimization_step(model,optimizer,[batch],cfg,0)
    path=tmp_path/'structure.pt';save_checkpoint(path,model,optimizer,identity,1,cfg,[])
    fresh=build_model(config)
    assert load_resume(path,fresh,_optimizer(fresh,cfg),identity,cfg)[0]==1
    changed=deepcopy(config);changed['boundary_restore']['structure']['weight']=.5
    different=build_identity(build_model(changed),Dataset(),changed)
    with pytest.raises(ValueError,match='identity differs'):
        load_resume(path,fresh,_optimizer(fresh,cfg),different,cfg)


def test_structure_thumbnail_distinguishes_location_prior_and_selected_path():
    from tools.boundary_context_monitor import structure_pair_panels
    cfg=load_config('config/train/boundary_structure.yaml')['boundary_restore']['structure']
    row=sample();cells,stats=structure_pair_panels(row,row['target'][0].numpy(),cfg)
    assert len(cells)==6 and all(value.shape==(24,24,3) for _,value in cells)
    assert stats['path_count']==1 and stats['paths'][0]['before']['points']==16
    assert 'not ground-truth' in stats['interpretation']
    assert stats['paths'][0]['before']==stats['paths'][0]['after']


def test_effective_batch_structure_denominator_survives_microbatch_accumulation(tmp_path):
    _,config=setup(tmp_path);cfg=config['boundary_restore']
    whole,accumulated=build_model(config),build_model(config)
    samples=[sample(i) for i in range(4)]
    first=optimization_step(whole,_optimizer(whole,cfg),[collate_samples(samples)],cfg,0)
    second=optimization_step(accumulated,_optimizer(accumulated,cfg),
        [collate_samples(samples[:2]),collate_samples(samples[2:])],cfg,0)
    assert first['region_denominators']==second['region_denominators']
    assert first['structure_denominators']==second['structure_denominators']
    assert first['loss']==pytest.approx(second['loss'],abs=1e-7)
    for a,b in zip(whole.parameters(),accumulated.parameters()):
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-7)


def test_complete_structure_monitor_keeps_frozen_final_output_and_process_thumbnail(tmp_path,monkeypatch):
    import train_boundary_context as trainer
    import tools.boundary_context_monitor as monitoring
    _,config=setup(tmp_path);cfg=config['boundary_restore']
    cfg.update(crop_size=16)
    cfg['monitor'].update(pair_indices=[0],thumbnail_width=32,tile_overlap=4)
    rgb=np.zeros((24,24,3),np.uint8);heat=np.full((24,24),.12,np.float32)
    ids=np.ones((24,24),np.uint16);classes={'1':1}
    cache=tmp_path/'cache.npz'
    np.savez(cache,boundary=heat,rgb=rgb,instances=ids,semantic=np.zeros_like(heat),raw_probability=np.ones_like(heat))
    png=tmp_path/'old.png';png.write_bytes(cv2.imencode('.png',ids)[1].tobytes())
    categories=tmp_path/'old.json';categories.write_text(json.dumps(classes))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    row=dict(source='test_101',path=str(cache),sha256=digest(cache),control_png=str(png),
        control_png_sha256=digest(png),control_classes=str(categories),control_classes_sha256=digest(categories))
    monkeypatch.setattr(monitoring,'_control_groups',lambda _:[('control_manifest',{'files':[row]}, {})])
    monkeypatch.setattr(monitoring,'evaluate_frozen_readout',lambda *args:(ids.copy(),dict(classes),{'fixed':True}))
    class Dataset:
        def set_epoch(self,epoch):assert epoch==1
        def __getitem__(self,index):return sample(index)
    monkeypatch.setattr(trainer,'build_dataset',lambda config,seed_offset:Dataset())
    model=build_model(config)
    report=monitoring.monitor(model,config,tmp_path/'run',0,'cpu')
    assert report['data_mode']==STRUCTURE_MODE
    assert report['rows'][0]['structure']['path_count']==1
    assert report['rows'][0]['loss_diagnostics']['structure']['point_count']==16
    assert report['rows'][1]['identity_epoch_exact']
    assert (tmp_path/'run/monitor/e000/structure_000.jpg').is_file()
    assert (tmp_path/'run/monitor/e000/test_101_whole.jpg').is_file()
