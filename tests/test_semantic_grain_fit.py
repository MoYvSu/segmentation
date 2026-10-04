# -*- coding: utf-8 -*-
"""只读特征缓存的whole/crop语义短测：预算、真实更新、门禁和严格续训。"""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from tools import fit_semantic_grain_view as fit
from train_backend_adaptation import tensor_digest
from utils.config import load_config
from utils.retention_fit import capture_rng,restore_rng
from utils.semantic_grain_view import anchor_probability,build_grain_batch,preflight_gate
from utils.semantic_native import prepare_native_tile
from utils.semantic_object_fit import array_sha,canonical_sha


def source_order():
    names=['train_659','train_558']+[f'train_{i:03d}' for i in range(30)]
    return names+names


def preflight_fixture():
    names=set(source_order())
    rows=[]
    supports={'anchor_original':10.,'anchor_filled':1.,'other_complete_original':100.,
        'other_complete_filled':1.,'truncated_neighbor_original':100.,'truncated_neighbor_filled':1.,'unknown':0.,'padding':0.}
    for name in sorted(names):
        for cls in (0,1):
            rows.append(dict(view=name+'_'+str(cls),source=name,anchor={'cls':cls,'id':cls+1},
                whole={'mean_true_class_probability':.9,'wrong_original_pixel_fraction':0.},
                crop={'mean_true_class_probability':.84,'wrong_original_pixel_fraction':0.},
                parameter_gradient_norms={'seg_fpn':1.,'seg_branch':1.,'semantic_residual':1.},
                anchor_parameter_gradient_norms={'seg_fpn':1.,'seg_branch':1.,'semantic_residual':1.},
                output_gradient={'absolute_by_support':dict(supports),'anchor_original_absolute':10.,'anchor_filled_absolute':1.}))
    report=dict(format='semantic_grain_view_preflight_v1',complete=True,training=False,optimizer_updates=0,test_used=False,
        sources=32,views=64,encoder_crop_calls=64,whole_baseline_all_exact=True,original_R1_frozen_exact=True,
        copied_head_unchanged=True,checkpoint={'sha256':'synthetic_R1'},gate=preflight_gate(rows))
    return report,rows,names


def test_old_source_order_has_two_exposures_and_each_grain_phase_is_used_once():
    order=source_order(); schedule=fit.grain_schedule(order)
    assert len(schedule)==64 and [name for name,cls in schedule]==order
    assert schedule[:2]==[('train_659',1),('train_558',1)]
    assert schedule[:32]==[(name,1) for name in order[:32]]
    assert schedule[32:]==[(name,0) for name in order[32:]]
    assert Counter(schedule)==Counter((name,cls) for name in order[:32] for cls in (0,1))
    damaged=order.copy(); damaged[-1]=damaged[-2]
    with pytest.raises(ValueError): fit.grain_schedule(damaged)


def test_formal_training_requires_complete64_view_GT_gate_and_recomputes_the_report():
    report,rows,names=preflight_fixture()
    assert fit.validate_preflight_contract(report,rows,names,'synthetic_R1')==report['gate']
    for corruption in ('small_preflight','test_used','different_R1','false_stored_gate','duplicate_view'):
        r,items=deepcopy(report),deepcopy(rows)
        if corruption=='small_preflight': r.update(sources=4,views=8)
        elif corruption=='test_used': r['test_used']=True
        elif corruption=='different_R1': r['checkpoint']['sha256']='other'
        elif corruption=='false_stored_gate': items[0]['output_gradient']['absolute_by_support']['unknown']=1.
        else: items[-1]=deepcopy(items[0])
        with pytest.raises(RuntimeError): fit.validate_preflight_contract(r,items,names,'synthetic_R1')


def _run_body():
    return next(node for node in ast.parse(inspect.getsource(fit.run)).body[0].body if isinstance(node,ast.Try)).body


def test_smoke2_snapshot_and_reload_scope_are_fixed_four_train_plus_four_test():
    train=list(fit.MONITORS)+[f'train_more_{i}' for i in range(26)]
    test=list(fit.TEST)+[f'test_extra_{i}' for i in range(4)]
    records={name:None for name in train+test}
    assert len(records)==40
    for step in (0,2):
        whole,crop=fit.evaluation_sources(records,test,step,2,True)
        assert whole==list(fit.TRAIN+fit.TEST) and crop==list(fit.TRAIN)
        assert len(whole)==8 and len(crop)*2==8 and not set(crop)&set(test)
    for step in (0,16,32,64):
        whole,crop=fit.evaluation_sources(records,test,step,64,False)
        assert whole==(list(records) if step in (0,64) else list(fit.MONITORS)+test)
        assert crop==(train if step==64 else list(fit.MONITORS))
        assert len(whole)==(40 if step in (0,64) else 14)
        assert len(crop)*2==(64 if step==64 else 12) and not set(crop)&set(test)


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__(); self.seg_fpn=nn.Linear(1,1); self.seg_branch=nn.Linear(1,1)
        with torch.no_grad():
            self.seg_fpn.weight.fill_(.7); self.seg_fpn.bias.fill_(.1)
            self.seg_branch.weight.fill_(1.3); self.seg_branch.bias.fill_(.2)
        self.eval()

    def forward(self,x): return self.seg_branch(self.seg_fpn(x))


class CountedAdamW(torch.optim.AdamW):
    def __init__(self,parameters,**kwargs):
        super().__init__(parameters,**kwargs); self.updates=0

    def step(self,*args,**kwargs):
        self.updates+=1; return super().step(*args,**kwargs)


def test_actual_two_smoke_steps_average_whole_crop_losses_then_make_one_update_each():
    """执行入口真实for-loop，CPU假缓存只替代昂贵encoder，不重写其更新顺序。"""
    with_node=next(n for n in _run_body() if isinstance(n,ast.With))
    loop=deepcopy(next(n for n in with_node.body if isinstance(n,ast.For)))
    module=ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[]))
    head=TinyHead(); reference=deepcopy(head)
    cfg=load_config('config/train/semantic_grain_fit.yaml')['semantic_grain_fit']
    optimizer=CountedAdamW(head.parameters(),lr=cfg['learning_rate'],eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'])
    expected_optimizer=torch.optim.AdamW(reference.parameters(),lr=cfg['learning_rate'],eps=cfg['adam_eps'],weight_decay=cfg['weight_decay'])
    schedule=fit.grain_schedule(source_order()); whole={};crop={}; forwards=[]
    for name,cls in schedule[:2]:
        whole[name]={'input':torch.tensor([[1.]]),'target':torch.tensor([[1.]])}
        crop[name+'_'+str(cls)]={'input':torch.tensor([[2.]]),'target':torch.tensor([[0.]])}
    def forward(model,cache):
        forwards.append(cache)
        return model(cache['input']),cache
    def loss(criterion,logits,batch):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['target'])
    for name,cls in schedule[:2]:
        expected_optimizer.zero_grad(set_to_none=True)
        expected=.5*loss(None,reference(whole[name]['input']),whole[name])+.5*loss(None,reference(crop[name+'_'+str(cls)]['input']),crop[name+'_'+str(cls)])
        expected.backward(); torch.nn.utils.clip_grad_norm_(reference.parameters(),cfg['grad_clip']); expected_optimizer.step()
    frozen=nn.Module().eval().requires_grad_(False); snapshots={};steps=[]
    env=dict(schedule=schedule,total=2,start=0,optimizer=optimizer,head=head,forward=forward,whole_cache=whole,crop_cache=crop,
        compute_semantic_loss=loss,criterion=None,torch=torch,cfg=cfg,tensor_digest=tensor_digest,
        buffers=tensor_digest(head.named_buffers()),verify_frozen_model=fit.verify_frozen_model,
        model=frozen,base_state=tensor_digest(frozen.state_dict().items()),steps=steps,snapshots=snapshots,
        evaluate=lambda step,all_sources:dict(step=step,rows={},scope='smoke8'),save=lambda step:None,
        write=lambda *args:None,out=Path('synthetic'),gc=SimpleNamespace(collect=lambda:None))
    exec(compile(module,'<actual-grain-fit-two-updates>','exec'),env)
    assert optimizer.updates==2 and len(forwards)==4 and len(steps)==2
    assert [(r['source'],r['crop_class']) for r in steps]==schedule[:2]
    assert all(r['optimizer_updates']==1 and r['head_views']==2 and r['region_extra_weight']==0 for r in steps)
    assert all(r['averaged_loss']==pytest.approx(.5*r['whole_loss']+.5*r['crop_loss'],abs=1e-7) for r in steps)
    assert all(torch.equal(a,b) for a,b in zip(head.parameters(),reference.parameters()))


def test_default_recipe_and_plan_are_R1_initialized_bounded_and_single_objective():
    config=load_config('config/train/semantic_grain_fit.yaml')
    plan=fit.plan(config); cfg=fit.validate_fit_recipe(config)
    assert plan['training'] is False and plan['smoke_steps']==2 and plan['formal_steps']==64
    assert plan['initialization']=='Original R1, smoke and oldA64 weights never inherited'
    assert cfg['whole_weight']==cfg['crop_weight']==.5 and plan['region_extra_weight']==0
    json.dumps(plan,allow_nan=False)
    damaged=deepcopy(config); damaged['semantic_grain_fit']['crop_weight']=1.
    with pytest.raises(ValueError): fit.validate_fit_recipe(damaged)


def test_configuration_identity_survives_JSON_class_keys_but_rejects_recipe_change():
    config=load_config('config/train/semantic_grain_fit.yaml')
    stored=json.loads(json.dumps(config,allow_nan=False))
    assert config['data']['classes']!=stored['data']['classes']
    assert canonical_sha(stored)==canonical_sha(config)
    stored['semantic_grain_fit']['crop_weight']=1.
    assert canonical_sha(stored)!=canonical_sha(config)


def crop_cache_fixture():
    """缓存绑定真实原生RGB裁块和最近邻GT，不能用自报摘要替代输入核验。"""
    rgb=np.arange(20*30*3,dtype=np.uint8).reshape(20,30,3)
    gt=np.zeros((20,30),np.uint16); gt[5:12,8:16]=1; gt[12:18,18:25]=2
    original=gt>0; original[7:9,9:11]=False
    classes={1:1,2:0}; box=(2,18,4,28)
    image,_,_,image_receipt=prepare_native_tile(rgb,box,32,'cpu')
    rebuilt,_,receipt=build_grain_batch(gt,original,classes,box,1,32)
    anchor=dict(id=1,cls=1,native_area=56,bbox=[5,12,8,16])
    features=[torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)]
    feature_sha=tensor_digest([(str(i),v) for i,v in enumerate(features)])
    row=dict(view='train_synthetic_1',anchor=anchor,image_receipt=image_receipt,target_receipt=receipt,
        input_sha256=array_sha(image),feature_sha256=feature_sha)
    cache=dict(view=row['view'],anchor=deepcopy(anchor),source_sha256='raw_source',processed_gt_sha256='manual_GT',
        box=list(box),image=image.clone(),image_receipt=deepcopy(image_receipt),target_receipt=deepcopy(receipt),
        image_sha256=row['input_sha256'],features=features,feature_sha256=feature_sha,
        feature_strides=[list(v.stride()) for v in features],
        batch={k:v.clone() for k,v in rebuilt.items() if torch.is_tensor(v)},
        target_sha256={k:array_sha(v) for k,v in rebuilt.items() if torch.is_tensor(v)})
    return cache,row,anchor,image,image_receipt,rebuilt,receipt


@pytest.mark.parametrize('corruption',['different_RGB','different_feature','different_feature_stride','wrong_GT',
    'wrong_source','wrong_anchor','self_consistent_wrong_cache','different_target'])
def test_crop_cache_rejects_mismatched_actual_inputs_and_e0_identity(corruption):
    cache,row,anchor,image,imreceipt,rebuilt,receipt=crop_cache_fixture()
    fit.validate_crop_cache(cache,row,anchor,'raw_source','manual_GT',image,imreceipt,rebuilt,receipt)
    if corruption=='different_RGB': image=image.clone(); image[0,0,0,0]+=1/255.
    elif corruption=='different_feature': cache['features'][0][0,0,0,0]+=1.
    elif corruption=='different_feature_stride':
        cache['features'][0]=cache['features'][0].transpose(-1,-2).contiguous().transpose(-1,-2)
    elif corruption=='wrong_GT': cache['processed_gt_sha256']='old_incomplete_GT'
    elif corruption=='wrong_source': cache['source_sha256']='other_source'
    elif corruption=='wrong_anchor': anchor={**anchor,'id':2}
    elif corruption=='self_consistent_wrong_cache':
        cache['image'][0,0,0,0]+=1/255.; cache['image_sha256']=array_sha(cache['image'])
    else:
        key=next(iter(cache['batch'])); cache['batch'][key].flatten()[0]+=1
    with pytest.raises(RuntimeError):
        fit.validate_crop_cache(cache,row,anchor,'raw_source','manual_GT',image,imreceipt,rebuilt,receipt)


def resume_state_fixture():
    head=TinyHead(); optimizer=torch.optim.AdamW(head.parameters(),lr=2e-5,eps=1e-4,weight_decay=1e-4)
    identity=dict(format=fit.FORMAT,initial_R1='synthetic_R1',smoke=True)
    architecture=dict(version='synthetic_decoder_v1'); schedule=fit.grain_schedule(source_order())
    rows=[dict(step=1,source=schedule[0][0],crop_class=schedule[0][1],optimizer_updates=1,head_views=2,region_extra_weight=0)]
    state=dict(format=fit.HEAD_FORMAT,identity=identity,complete=False,terminal_only=True,
        do_not_replace_geometry_upstream=True,architecture=architecture,step=1,total_steps=2,schedule=schedule,
        semantic_state_dict=head.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(),steps=rows,snapshots={})
    return state,identity,schedule,architecture


@pytest.mark.parametrize('corruption',['bool_step','bool_total','wrong_phase','wrong_source','extra_update',
    'wrong_views','wrong_architecture','full_model_format','upstream_replace','completed','missing_rng',
    'missing_optimizer','changed_lr','different_mode','different_initialization','row_bool_step'])
def test_resume_is_only_same_owned_run_with_exact_optimizer_rng_and_budget(corruption):
    state,identity,schedule,architecture=resume_state_fixture()
    assert fit.validate_resume_state(state,identity,schedule,architecture,2)==1
    if corruption=='bool_step': state['step']=True
    elif corruption=='bool_total': state['total_steps']=True
    elif corruption=='wrong_phase': state['steps'][0]['crop_class']=0
    elif corruption=='wrong_source': state['steps'][0]['source']=schedule[1][0]
    elif corruption=='extra_update': state['steps'][0]['optimizer_updates']=2
    elif corruption=='wrong_views': state['steps'][0]['head_views']=1
    elif corruption=='wrong_architecture': state['architecture']={'version':'other'}
    elif corruption=='full_model_format': state['terminal_only']=False
    elif corruption=='upstream_replace': state['do_not_replace_geometry_upstream']=False
    elif corruption=='completed': state['complete']=True
    elif corruption=='missing_rng': del state['rng']['torch']
    elif corruption=='missing_optimizer': del state['optimizer']['state']
    elif corruption=='changed_lr': state['optimizer']['param_groups'][0]['lr']=1e-3
    elif corruption=='different_mode': identity={**identity,'smoke':False}
    elif corruption=='different_initialization': identity={**identity,'initial_R1':'oldA64'}
    else: state['steps'][0]['step']=True
    with pytest.raises(RuntimeError): fit.validate_resume_state(state,identity,schedule,architecture,2)


def test_head_optimizer_and_rng_restore_produce_exact_next_update():
    """保存真实AdamW一更新，恢复后下一次随机输入与连续路径逐值一致。"""
    original_rng=capture_rng()
    try:
        torch.manual_seed(123); head=TinyHead()
        optimizer=torch.optim.AdamW(head.parameters(),lr=2e-5,eps=1e-4,weight_decay=1e-4)
        def step(model,opt):
            sample=torch.rand((4,1)); target=torch.randint(0,2,(4,1)).float()
            opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(sample),target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            return sample,loss.detach()
        step(head,optimizer)
        state,identity,schedule,architecture=resume_state_fixture()
        state.update(semantic_state_dict=deepcopy(head.state_dict()),optimizer=deepcopy(optimizer.state_dict()),rng=capture_rng())
        assert fit.validate_resume_state(state,identity,schedule,architecture,2)==1
        sample,loss=step(head,optimizer)
        reloaded=TinyHead(); reloaded.load_state_dict(state['semantic_state_dict'],strict=True)
        resumed=torch.optim.AdamW(reloaded.parameters(),lr=2e-5,eps=1e-4,weight_decay=1e-4)
        resumed.load_state_dict(state['optimizer']); restore_rng(state['rng'])
        second_sample,second_loss=step(reloaded,resumed)
        assert torch.equal(sample,second_sample) and torch.equal(loss,second_loss)
        assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in head.state_dict().items())
        for index,values in optimizer.state_dict()['state'].items():
            assert all(torch.equal(v,resumed.state_dict()['state'][index][k]) if torch.is_tensor(v)
                else v==resumed.state_dict()['state'][index][k] for k,v in values.items())
    finally: restore_rng(original_rng)


@pytest.mark.parametrize('location',['live_code','archived_code','live_yaml','archived_yaml'])
def test_resume_archive_rejection_never_overwrites_its_prior_evidence(tmp_path,monkeypatch,location):
    project=tmp_path/'project'; out=project/'outputs'/'owned'
    files={'live_code':project/'tools'/'owned.py','archived_code':out/'source'/'tools'/'owned.py',
        'live_yaml':project/'config'/'train.yaml','archived_yaml':out/'config_source'/'config'/'train.yaml'}
    for name,path in files.items():
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text('source' if name.endswith('code') else 'recipe',encoding='utf-8')
    monkeypatch.setattr(fit,'ROOT',project)
    identity=dict(protected_sources={'tools/owned.py':fit.file_sha(files['live_code'])},
        protected_inputs={str(files['live_yaml']):fit.file_sha(files['live_yaml'])})
    fit.verify_resume_archives(out,identity)
    files[location].write_text('changed',encoding='utf-8')
    before={name:p.read_bytes() for name,p in files.items()}
    with pytest.raises(RuntimeError): fit.verify_resume_archives(out,identity)
    assert before=={name:p.read_bytes() for name,p in files.items()}


@pytest.mark.parametrize('corruption',['output_png','source_png','probability','classes'])
def test_strict_reload_checks_actual_saved_candidate_png_not_only_source(tmp_path,corruption):
    source=tmp_path/'source.png'; output=tmp_path/'candidate.png'; classes_path=tmp_path/'classes.json'
    ids=np.zeros((5,7),np.uint16); ids[1:4,2:6]=257
    assert cv2.imwrite(str(source),ids); output.write_bytes(source.read_bytes())
    probability=np.full(ids.shape,.7,np.float32); classes={'257':1}
    classes_path.write_text(json.dumps(classes),encoding='utf-8')
    expected=dict(probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),png_sha256=fit.file_sha(source))
    fit.verify_terminal_output(probability,classes,expected,source,output,classes_path)
    if corruption=='output_png':
        changed=ids.copy();changed[0,0]=1; assert cv2.imwrite(str(output),changed)
    elif corruption=='source_png': source.write_bytes(b'not original')
    elif corruption=='probability': probability[0,0]=np.nextafter(probability[0,0],np.float32(1))
    else: classes_path.write_text(json.dumps({'257':0}),encoding='utf-8')
    with pytest.raises(RuntimeError): fit.verify_terminal_output(probability,classes,expected,source,output,classes_path)


@pytest.mark.parametrize('corruption',['probability','metrics'])
def test_crop_final_replay_requires_actual_native_probability_and_original_GT_metrics(corruption):
    gt=np.zeros((8,12),np.uint16); gt[2:7,3:9]=1; original=gt>0
    original[3:5,4:6]=False
    probability=np.full(gt.shape,.65,np.float32); anchor={'id':1,'cls':1}
    metrics=anchor_probability(probability,gt,original,anchor)
    expected=dict(metrics=metrics,probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest())
    fit.verify_crop_replay(probability,metrics,expected)
    if corruption=='probability':
        # GT原覆盖区外改动不变anchor均值，native全图SHA仍必须拒绝。
        probability[0,0]=np.nextafter(probability[0,0],np.float32(1))
        assert anchor_probability(probability,gt,original,anchor)==metrics
    else: metrics={**metrics,'original_pixels':metrics['original_pixels']+1}
    with pytest.raises(RuntimeError): fit.verify_crop_replay(probability,metrics,expected)
