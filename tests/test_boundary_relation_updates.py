# -*- coding: utf-8 -*-
"""精确更新预算、尾批保留与轮中严格恢复；CPU小模型，不运行主线。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_relation_update_trainer_test',ROOT/'train_boundary_relation.py')
trainer=importlib.util.module_from_spec(spec);sys.modules[spec.name]=trainer;spec.loader.exec_module(trainer)


@pytest.fixture(autouse=True)
def runtime():
    rng=trainer.capture_rng();threads=torch.get_num_threads();torch.set_num_threads(1)
    yield
    trainer.restore_rng(rng);torch.set_num_threads(threads)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.1,.2,.3,.4,.5]));self.bias=torch.nn.Parameter(torch.tensor(.1))
        self.architecture={'test_only':'tiny_relation'};self.initial_state_sha256=trainer.state_sha256(self)

    def forward(self,x,masks):
        return (x[:,:,0,0]*self.weight).sum(1)+self.bias


class TinyData:
    def __init__(self,path,mode='maps',count=3):
        self.manifest_path=path;self.mode=mode;self.count=count;self.epoch=0
        self.labels=np.asarray([i%2 for i in range(count)],np.uint8)
        self.sample_ids=[f'sample-{i}' for i in range(count)];self.sources=['synthetic']*count
        self.manifest={'monitor':{'train_sample_ids':self.sample_ids[:2]}}
        self.identity={'manifest_sha256':trainer.file_sha(path),'shards':[],'sample_count':count}
        self.identity_sha256=trainer.canonical_sha(self.identity)

    def __len__(self):return self.count
    def set_epoch(self,epoch):self.epoch=epoch

    def tensors(self,index,augment=True,mode=None):
        x=torch.arange(5*8*8,dtype=torch.float32).reshape(5,8,8)/400+index/20
        masks=torch.zeros(4,8,8);masks[0,:,:3]=1;masks[1,:,5:]=1;masks[2,:,3:5]=1;masks[3]=1
        k=(index+self.epoch)%4 if augment else 0
        x=torch.rot90(x,k,(-2,-1));masks=torch.rot90(masks,k,(-2,-1))
        if (mode or self.mode)=='maps':x=x.clone();x[:3]=0
        return dict(input=x,masks=masks,label=torch.tensor(float(self.labels[index])),sample_id=self.sample_ids[index],
                    source='synthetic',index=index,meta={'geometry_D4':k})

    def __getitem__(self,index):return self.tensors(index)

    def raw(self,index):
        sample=self.tensors(index,augment=False,mode='rgb')
        return dict(rgb=np.full((3,8,8),100+index,np.uint8),maps=sample['input'][3:].numpy().astype(np.float16),
                    masks=sample['masks'].numpy().astype(np.uint8),label=int(self.labels[index]),
                    sample_id=self.sample_ids[index],source='synthetic')


def recipe(**changes):
    cfg=dict(seed=11,epochs=3,batch_size=2,num_workers=0,precision='float32',apply_predicted_merges=False,
             no_validation_holdout=True,learning_rate=3e-4,minimum_learning_rate=3e-5,weight_decay=1e-4,
             max_ram_gib=1,augment_d4=True,monitor_epochs=[],max_updates=5,schedule_updates=5,
             monitor_updates=[0,3,5],checkpoint_updates=[0,3,5],thumbnail_width=144)
    cfg.update(changes);return trainer.validated_recipe({'boundary_relation':cfg})


def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_long_config_exact_budget_and_tail_accounting():
    cfg=trainer.validated_recipe(trainer.load_config(ROOT/'config/train/boundary_relation_long.yaml'))
    assert trainer.update_budget(cfg,43)==(1000,1000)
    assert cfg['epochs']==24 and cfg['batch_size']==64 and cfg['monitor_epochs']==[]
    assert cfg['monitor_updates']==cfg['checkpoint_updates']==[0,215,500,750,1000]
    identity={'class_counts':{'keep':2225,'merge':521}}
    cursor=trainer.sampler_cursor(identity,cfg,1000,43)
    assert (cursor['next_epoch'],cursor['batches_consumed_in_next_epoch'],cursor['samples_consumed_in_next_epoch'])==(24,11,704)
    assert 23*2746+704==63862 and len(cursor['full_epoch_order'])==2746
    assert len(set(cursor['full_epoch_order']))==2746
    assert len(cursor['full_epoch_order'][42*64:])==58
    assert trainer.learning_rate(999,1000,cfg)==cfg['minimum_learning_rate']


@pytest.mark.parametrize('changes',[{'schedule_updates':4},{'max_updates':True},{'monitor_epochs':[0,1,3]},
                                  {'monitor_updates':[0,5,3]},{'checkpoint_updates':[0,3]}])
def test_bad_update_budgets_fail_closed(changes):
    with pytest.raises(ValueError):recipe(**changes)


def test_mid_epoch_resume_exact_optimizer_RNG_and_next_batch(tmp_path):
    path=tmp_path/'data.json';path.write_text('{}');cfg=recipe();initial=tmp_path/'initial.pt'
    trainer.create_initial(initial,TinyModel());trainer.seed_runtime(11)
    full=tmp_path/'full';data=TinyData(path)
    model=trainer.train(TinyModel(),data,None,{'boundary_relation':cfg},cfg,full,'maps',initial)
    final_rng=trainer.capture_rng();expected=trainer.state_sha256(model)
    full_rows=[json.loads(line)for line in (full/'steps.jsonl').read_text().splitlines()]
    assert [r['samples']for r in full_rows]==[2,1,2,1,2]
    cp=torch.load(full/'u000003.pt',weights_only=False)
    assert (cp['global_step'],cp['epoch'],cp['batch_in_epoch'])==(3,1,1)
    assert cp['sampler']['next_epoch']==2 and cp['sampler']['samples_consumed_in_next_epoch']==2
    assert cp['monitor_epochs']==[] and cp['monitor_updates']==[0,3]
    random.random();np.random.random();torch.rand(5)
    resumed=tmp_path/'resumed'
    restored=trainer.train(TinyModel(),TinyData(path),None,{'boundary_relation':cfg},cfg,resumed,'maps',initial,resume=full/'u000003.pt')
    assert trainer.state_sha256(restored)==expected
    equal(final_rng,trainer.capture_rng())
    actual=torch.load(resumed/'last.pt',weights_only=False);expected_cp=torch.load(full/'last.pt',weights_only=False)
    equal(actual['optimizer'],expected_cp['optimizer']);equal(actual['rng'],expected_cp['rng']);equal(actual['sampler'],expected_cp['sampler'])
    rows=[json.loads(line)for line in (resumed/'steps.jsonl').read_text().splitlines()]
    for a,b in zip(rows,full_rows[3:]):
        for key in ('global_step','sample_ids','shared_draw_sha256','learning_rate','loss'):assert a[key]==b[key]
    complete=json.loads((full/'complete.json').read_text())
    assert (complete['global_step'],complete['completed_epochs'],complete['last_epoch'],complete['last_batch_in_epoch'])==(5,2,3,1)
    assert complete['samples_consumed']==8 and complete['monitor_updates']==[0,3,5]
    assert all((full/f'u{s:06d}.pt').exists()and(full/f'monitor/s{s:06d}/receipt.json').exists()for s in (0,3,5))
    report=json.loads((full/'monitor/s000003/receipt.json').read_text())
    assert report['global_step']==3 and report['keep_probability_visualized']
    assert [r['sample_id']for r in report['rows']]==data.manifest['monitor']['train_sample_ids']
    bad=deepcopy(cp);bad['sampler']['full_epoch_order'].reverse();bad_path=tmp_path/'corrupt_sampler.pt';torch.save(bad,bad_path)
    new=TinyModel();optimizer=torch.optim.AdamW(new.parameters(),lr=cfg['learning_rate'])
    with pytest.raises(ValueError,match='采样顺序'):trainer.load_resume(bad_path,new,optimizer,cp['identity'],cfg,2)


def test_legacy_identity_has_no_new_budget_keys_and_cannot_extend_old_checkpoint(tmp_path):
    cfg=recipe();legacy={k:v for k,v in cfg.items()if k not in ('max_updates','schedule_updates','monitor_updates','checkpoint_updates')}
    legacy['monitor_epochs']=[0,1,2,3];legacy=trainer.validated_recipe({'boundary_relation':legacy})
    path=tmp_path/'data.json';path.write_text('{}');data=TinyData(path);model=TinyModel();initial=tmp_path/'initial.pt';trainer.create_initial(initial,model)
    old_identity=trainer.build_identity(model,data,legacy,'maps',initial)
    assert 'update_protocol'not in old_identity and 'max_updates'not in old_identity['recipe'] and 'schedule_epochs'in old_identity['recipe']
    optimizer=torch.optim.AdamW(model.parameters(),lr=legacy['learning_rate'])
    checkpoint=tmp_path/'legacy.pt';trainer.save_checkpoint(checkpoint,model,optimizer,old_identity,legacy,0,2,[0])
    assert trainer.load_resume(checkpoint,model,optimizer,old_identity,legacy,2)==(0,[0])
    saved=torch.load(checkpoint,weights_only=False)
    assert not any(key in saved for key in ('update_budget','sampler','monitor_updates'))
    with pytest.raises(ValueError):trainer.load_resume(checkpoint,model,optimizer,trainer.build_identity(model,data,cfg,'maps',initial),cfg,2)
