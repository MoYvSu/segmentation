# -*- coding: utf-8 -*-
"""原BCE真实梯度、私有hook隔离与epoch边界状态恢复。"""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_affinity_coverage as trainer
from train_affinity_connectivity import configure_training
from train_direct_semantic_affinity import compute_affinity_loss
from utils.config import load_config
from data.affinity_coverage import CoveragePlan

_spec=importlib.util.spec_from_file_location('_coverage_test_plan',Path(__file__).with_name('test_affinity_coverage.py'))
_fixtures=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixtures)
synthetic_plan=_fixtures.synthetic_plan


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affinity_head=torch.nn.Sequential(torch.nn.Conv2d(2,4,1),torch.nn.ReLU(),torch.nn.Conv2d(4,8,1))
    def forward(self,image):return {'affinity_logits':self.affinity_head(image)}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Identity();self.affinity_decoder=Decoder()
        self.register_buffer('frozen_sentinel',torch.tensor([9.]))


def append(path,row):
    with Path(path).open('a',encoding='utf8') as file:file.write(json.dumps(row,allow_nan=False)+'\n')


def compare(row,reference):
    for key in ('epoch','step','seed','manual','pseudo','manual_draw','pseudo_draw'):
        if row[key]!=reference[key]:raise RuntimeError('Parent guard failed: '+key)


def core():
    return SimpleNamespace(configure_training=configure_training,compute_affinity_loss=compute_affinity_loss,
                           compare_draw=compare,append_json=append)


def config():return load_config(trainer.CONFIG_PATH)


def batch(pseudo=False):
    ids=torch.ones((1,20,20),dtype=torch.long);ids[:,:,10:]=2;ids[:,3:5,4:6]=0
    valid=torch.ones((1,1,20,20),dtype=torch.bool);valid[:,:,-2:]=False
    # 两种真实schema分别只保留原loss需要的字段，禁止用人工aliases遮住SAM2合同。
    fields=dict(instance_map=ids,valid_content=valid) if pseudo else dict(affinity_instance_map=ids,affinity_valid_content=valid)
    return dict(**fields,uncovered_boundary_source=torch.tensor([False]),image=torch.zeros(1,3,20,20))


@pytest.mark.parametrize('mutation',['none','gt','weight','seed','flag','bool','extra','init','view','budget'])
def test_reference_only_crop_policy_changes_and_hidden_mutations_fail(mutation):
    cfg=config();expected=load_config('config/train/affinity_balance_mixed.yaml')
    if mutation=='none':
        assert trainer.comparable(trainer.reference_config(cfg))==trainer.comparable(expected);return
    if mutation=='gt':cfg['backend_adaptation']['completed_gt_dir']='data/old_gt'
    elif mutation=='weight':cfg['direct_semantic_affinity']['affinity_loss']['negative_weight']=1.5
    elif mutation=='seed':cfg['affinity_coverage']['seed']+=1
    elif mutation=='flag':cfg['affinity_coverage']['flag_namespace']+=1
    elif mutation=='bool':cfg['direct_semantic_affinity']['affinity_loss']['negative_weight']=True
    elif mutation=='extra':cfg['affinity_coverage']['allow_fallback']=True
    elif mutation=='init':cfg['affinity_deployment']['checkpoint']='outputs/affinity_balance/mixed/final.pt'
    elif mutation=='view':cfg['affinity_native']['mixed_view_seed']+=1
    else:cfg['backend_adaptation']['epochs']=10
    with pytest.raises(ValueError):trainer.reference_config(cfg)


def test_three_real_adamw_updates_keep_original_bce_and_do_not_accumulate_probe_grads(tmp_path,monkeypatch):
    old,new=synthetic_plan();plan=CoveragePlan(old,new)
    monkeypatch.setattr(trainer,'read_plan',lambda _:plan)
    torch.manual_seed(31);model=Model();expected=deepcopy(model);reference=deepcopy(model.affinity_decoder)
    private=core();originals=vars(private).copy();cfg=config();loss_cfg=cfg['direct_semantic_affinity']['affinity_loss']
    opts=[torch.optim.AdamW(m.affinity_decoder.parameters(),lr=2.e-5,eps=1.e-4) for m in (model,expected)]
    inputs=torch.randn(3,2,20,20);saved=deepcopy(model.state_dict())
    with trainer.coverage_hooks(private,cfg,tmp_path,{'test':'same'},smoke=True,reference_head=reference) as audits:
        private.configure_training(model);configure_training(expected)
        for step in range(1,4):
            actual=new[step-1];private.compare_draw(actual,old[step-1])
            for opt in opts:opt.zero_grad(set_to_none=True)
            loss_values=[]
            for pseudo in (False,True):
                data=batch(pseudo);image=inputs[step-1:step]
                assert ('affinity_instance_map' in data) is (not pseudo)
                assert ('instance_map' in data) is pseudo
                a=model.affinity_decoder(image)['affinity_logits'];b=expected.affinity_decoder(image)['affinity_logits']
                prior_grads=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
                loss,_=private.compute_affinity_loss(a,data,loss_cfg,pseudo=pseudo)
                base,_=compute_affinity_loss(b,data,loss_cfg,pseudo=pseudo)
                assert torch.equal(loss,base)
                for p,before in zip(model.parameters(),prior_grads):
                    assert p.grad is None if before is None else torch.equal(p.grad,before)
                scale=.5 if pseudo else 1.
                (loss*scale).backward();(base*scale).backward();loss_values.append(float(loss.detach()))
            assert all(torch.equal(a.grad,b.grad) for a,b in zip(model.parameters(),expected.parameters()))
            for opt in opts:torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],1.);opt.step()
            row=deepcopy(actual);row.pop('counterfactual_not_executed');row.pop('counterfactual_anchor',None)
            row.update(manual_bce=loss_values[0],pseudo_bce=loss_values[1],grad_norm=1.)
            private.append_json(tmp_path/'steps.jsonl',row)
            assert all(torch.equal(a,b) for a,b in zip(model.parameters(),expected.parameters()))
        assert len(audits)==6 and all(r['same_candidate_input'] and r['unknown_logit_gradient_max']==0 for r in audits)
    assert vars(private)==originals and len(model.affinity_decoder._forward_pre_hooks)==0
    assert torch.equal(saved['frozen_sentinel'],model.frozen_sentinel)
    executed=trainer.rows(tmp_path/'steps.jsonl')
    assert all(r['coverage']['executed'] is True for r in executed)
    assert all('counterfactual_not_executed' not in r for r in executed)


def test_parent_guard_is_not_swallowed_and_exception_restores_private_scope(tmp_path,monkeypatch):
    old,new=synthetic_plan();monkeypatch.setattr(trainer,'read_plan',lambda _:CoveragePlan(old,new))
    private=core();originals=vars(private).copy()
    with pytest.raises(RuntimeError,match='Parent guard failed'):
        with trainer.coverage_hooks(private,config(),tmp_path,{},smoke=False):
            bad=deepcopy(new[0]);bad['seed']+=1;private.compare_draw(bad,old[0])
    assert vars(private)==originals


def test_raw_tensor_gate_rejects_rgb_gt_padding_and_undeclared_second_variable():
    original=batch();original['profile']=['blur'];actual=deepcopy(original)
    actual['coverage_anchor']=torch.tensor([True]);actual['coverage_update']=torch.tensor([2])
    assert trainer.batch_tensor_audit(actual,original)
    for key in ('image','affinity_instance_map','affinity_valid_content'):
        changed=deepcopy(actual);value=changed[key].view(-1)[0]
        changed[key].view(-1)[0]=~value if value.dtype==torch.bool else 1-value
        with pytest.raises(RuntimeError,match='declared route'):trainer.batch_tensor_audit(changed,original)
    changed=deepcopy(actual);changed['new_appearance_transform']=True
    with pytest.raises(RuntimeError,match='Unexpected candidate'):trainer.batch_tensor_audit(changed,original)


def make_checkpoint(folder):
    folder.mkdir();cfg=config();model=Model();opt=torch.optim.AdamW(model.affinity_decoder.parameters(),lr=2.e-5,eps=1.e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:1.-e/40)
    for step in range(1,65):
        opt.zero_grad();sum(p.square().mean() for p in model.parameters()).backward();opt.step()
        append(folder/'steps.jsonl',dict(epoch=1,step=step,updates=step))
    scheduler.step();append(folder/'epochs.jsonl',dict(epoch=1,updates=64))
    state=dict(initial_state_sha256='initial',frozen_state_sha256='frozen',restorer_state_sha256='d5a',data={'coverage':True})
    contract={'source':'fixed','runtime':'cpu','grid':'sealed'}
    extras=trainer.checkpoint_state(1,64,state,folder,contract)
    bundle=dict(format=trainer.HEAD_FORMAT,epoch=1,size=1024,config=cfg,geometry_state_dict=model.affinity_decoder.state_dict(),
                optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),**{k:v for k,v in extras.items() if k!='format'})
    torch.save(bundle,folder/'last_head.pt')
    return model,opt,scheduler,state,contract,bundle


def test_resume_epoch_boundary_restores_real_head_adamw_scheduler_rng_and_truncates_uncommitted_prefix(tmp_path,monkeypatch):
    old,new=synthetic_plan();monkeypatch.setattr(trainer,'read_plan',lambda _:CoveragePlan(old,new))
    model,opt,scheduler,state,contract,bundle=make_checkpoint(tmp_path/'old')
    # 模拟定时关机时下一轮已写了部分行；只能复用checkpoint提交的64行。
    append(tmp_path/'old/steps.jsonl',dict(epoch=2,step=1,updates=65))
    fresh=tmp_path/'resumed';fresh.mkdir();target=Model();opt2=torch.optim.AdamW(target.affinity_decoder.parameters(),lr=1.e-3,eps=1.e-4)
    scheduler2=torch.optim.lr_scheduler.LambdaLR(opt2,lambda e:1.-e/40)
    private=core()
    with trainer.coverage_hooks(private,config(),fresh,contract,resume=tmp_path/'old/last_head.pt'):
        assert private.coverage_resume(target,opt2,scheduler2,deepcopy(state),fresh)==(1,64)
        assert all(torch.equal(a,b) for a,b in zip(target.affinity_decoder.parameters(),model.affinity_decoder.parameters()))
        assert scheduler2.state_dict()==scheduler.state_dict()
        for a,b in zip(opt2.state.values(),opt.state.values()):
            assert all(torch.equal(a[k],b[k]) for k in a)
        assert torch.equal(torch.get_rng_state(),bundle['rng_state']['torch'])
    assert len(trainer.rows(fresh/'steps.jsonl'))==64 and len(trainer.rows(tmp_path/'old/steps.jsonl'))==65
    # 下一实际AdamW更新也必须一致，不能只比较保存字典。
    for current,optimizer in ((model,opt),(target,opt2)):
        optimizer.zero_grad();sum(p.square().mean() for p in current.parameters()).backward();optimizer.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),target.parameters()))


@pytest.mark.parametrize('mutation',['smoke','update','source','prefix','optimizer','format'])
def test_resume_rejects_smoke_mid_epoch_changed_contract_or_incomplete_state(tmp_path,mutation):
    _,_,_,_,contract,bundle=make_checkpoint(tmp_path/'old')
    if mutation=='smoke':bundle['coverage_state']['smoke']=True
    elif mutation=='update':bundle['coverage_state']['updates']=63
    elif mutation=='source':bundle['coverage_state']['contract']['source']='changed'
    elif mutation=='prefix':append(tmp_path/'old/epochs.jsonl',{}) ; (tmp_path/'old/epochs.jsonl').write_text('{}\n',encoding='utf8')
    elif mutation=='optimizer':bundle.pop('optimizer')
    else:bundle['format']='affinity_native_head_v1'
    torch.save(bundle,tmp_path/'old/last_head.pt')
    with pytest.raises(RuntimeError):trainer.resume_checkpoint(tmp_path/'old/last_head.pt',config(),contract)
