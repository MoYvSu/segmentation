# -*- coding: utf-8 -*-
"""关系评分适配的真实CPU加载、训练张量一致性和零操作合同。"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from utils import boundary_relation_inference as inference
from utils.boundary_relation_supervision import adjacent_region_pairs, native_relation_patch, representative_anchor


@pytest.fixture(autouse=True)
def runtime():
    py, nr, tr = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.set_num_threads(2)
    yield
    random.setstate(py); np.random.set_state(nr); torch.set_rng_state(tr)
    torch.set_num_threads(threads)
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = flags


def scene(regions=3):
    h, width = 80, regions*42
    ids = np.zeros((h,width),np.uint16)
    for i in range(regions):
        ids[:,i*42:i*42+41] = i+1
    rng=np.random.default_rng(12)
    rgb=rng.integers(0,256,size=(h,width,3),dtype=np.uint8)
    before=np.linspace(.031,.983,h*width,dtype=np.float32).reshape(h,width)
    after=np.flip(before,1).copy()
    classes={str(i+1):(1 if i==0 else 0) for i in range(regions)}
    return rgb,before,after,ids,classes


def checkpoint(tmp_path,mode='rgb',*,step=1000,smoke=False,small=False):
    module=inference._model_module()
    model=module.BoundaryRelationNet(**({'widths':(8,16,24),'hidden':16} if small else {}))
    initial=model.initial_state_sha256
    with torch.no_grad():
        model.head[-1].weight.zero_();model.head[-1].bias.fill_(-3.)
    identity=dict(format=module.FORMAT,architecture=model.architecture,
        initial_state_sha256=initial,mode=mode,no_GT_features=True,no_C_weights_loaded=True,
        recipe=dict(max_updates=1000,schedule_updates=1000))
    payload=dict(format=module.FORMAT,architecture=model.architecture,
        initial_state_sha256=initial,model_state_dict=model.state_dict(),model_sha256=module.state_sha256(model),
        identity=identity,identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False,
            separators=(',',':')).encode('utf8')).hexdigest(),global_step=step,smoke_only=smoke)
    path=tmp_path/f'{mode}_{step}_{small}_{smoke}.pt';torch.save(payload,path)
    return path,payload


def test_zero_threshold_and_no_candidates_do_not_load_model(monkeypatch):
    monkeypatch.setattr(inference.BoundaryRelationPostprocessor,'_load',lambda self:pytest.fail('不得加载模型'))
    data=scene();old=data[3].copy();classes=data[4].copy()
    adapter=inference.BoundaryRelationPostprocessor(None,threshold=0,device='cuda')
    new,new_classes,audit=adapter.process(*data)
    assert np.array_equal(new,old) and new_classes==classes and audit['loaded_parameter_count']==0
    assert adapter.parameter_count==211865 and audit['no_op_reason']=='threshold_zero'
    assert new is not data[3] and new_classes is not data[4]
    data=scene(1);adapter=inference.BoundaryRelationPostprocessor(None,threshold=.1,device='cuda')
    new,new_classes,audit=adapter.process(*data)
    assert np.array_equal(new,data[3]) and audit['no_op_reason']=='no_adjacent_pairs'
    assert audit['loaded_parameter_count']==0


@pytest.mark.parametrize('mode',['maps','rgb'])
def test_exact_training_tensor_contract_and_streamed_real_forward(tmp_path,mode):
    data=scene(6);path,payload=checkpoint(tmp_path,mode)
    pairs,_=adjacent_region_pairs(data[3]);anchor=representative_anchor(pairs[0],data[3].shape[1])
    raw=native_relation_patch(*data[:4],pairs[0],anchor)
    # 用真实Dataset读取同一原生patch，比较反射、量化、mask与RGB归一化。
    dpath=tmp_path/'patch.npz';np.savez(dpath,rgb=raw['rgb'][None],maps=raw['maps'][None],
        masks=raw['masks'][None],labels=np.array([1],np.uint8))
    manifest=dict(format='boundary_relation_data_v1',complete=True,patch_size=256,shards=[dict(
        path=dpath.name,sha256=hashlib.sha256(dpath.read_bytes()).hexdigest(),count=1,source='synthetic',
        samples=[dict(sample_id='synthetic:1',source='synthetic',label=1)])])
    mp=tmp_path/'data.json';mp.write_text(json.dumps(manifest),encoding='utf8')
    source=Path(__file__).resolve().parents[1]/'data/boundary_relation.py'
    spec=importlib.util.spec_from_file_location('_relation_inference_test_data',source)
    dm=importlib.util.module_from_spec(spec);spec.loader.exec_module(dm)
    expected=dm.BoundaryRelationDataset(mp,mode=mode,augment=False)[0]
    xx,mm=inference._tensor_patch(raw,mode)
    assert torch.equal(xx,expected['input']) and torch.equal(mm,expected['masks'])
    assert raw['maps'].dtype==np.float16 and xx.dtype==torch.float32
    if mode=='maps':assert torch.count_nonzero(xx[:3])==0
    adapter=inference.BoundaryRelationPostprocessor(path,device='cpu',batch_size=2,
        expected_model_sha=payload['model_sha256'])
    new,classes,audit=adapter.process(*data)
    assert adapter.mode==mode and audit['loaded_parameter_count']==211865
    assert len(audit['pairs'])==len(pairs)==5 and audit['scoring_batch_sizes']==[2,2,1]
    assert all(abs(r['keep_score']-torch.sigmoid(torch.tensor(-3.)).item())<1e-7 for r in audit['pairs'])
    assert all(r['native_patch_size']==256 for r in [audit])
    assert audit['merge']['selected_count']>=1 and classes['1']==1
    assert audit['input_unchanged'] and adapter.verify_unchanged()


def test_rng_flags_autocast_and_original_inputs_remain_unchanged(tmp_path):
    data=scene();path,_=checkpoint(tmp_path)
    old=[v.copy() if isinstance(v,np.ndarray) else v.copy() for v in data]
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False
    flags=(True,True,True,False)
    py,nr,tr=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
    adapter=inference.BoundaryRelationPostprocessor(path,device='cpu')
    with torch.autocast('cpu',dtype=torch.bfloat16):
        _,_,audit=adapter.process(*data)
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],nr[1])
    assert torch.equal(torch.get_rng_state(),tr)
    assert flags==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
    assert all(np.array_equal(a,b) if isinstance(a,np.ndarray) else a==b for a,b in zip(data,old))
    assert audit['existing_RNG_and_runtime_flags_restored']
    with torch.no_grad():next(adapter._network.parameters()).view(-1)[0].add_(1.)
    with pytest.raises(RuntimeError,match='冻结关系模型'):
        adapter.verify_unchanged()


@pytest.mark.parametrize('kind',['e5','smoke','small','sha','mode','corrupt'])
def test_strict_checkpoint_gates_and_exception_runtime_restoration(tmp_path,kind):
    path,payload=checkpoint(tmp_path,step=215 if kind=='e5' else 1000,
                            smoke=kind=='smoke',small=kind=='small')
    if kind in ('mode','corrupt'):
        bad=deepcopy(payload)
        if kind=='mode':bad['identity']['mode']='other'
        else:bad['model_state_dict'][next(iter(bad['model_state_dict']))].view(-1)[0]+=1
        torch.save(bad,path)
    adapter=inference.BoundaryRelationPostprocessor(path,device='cpu',
        expected_model_sha='0'*64 if kind=='sha' else None)
    rng=torch.get_rng_state().clone();data=scene();before=data[3].copy()
    with pytest.raises((ValueError,RuntimeError)):
        adapter.process(*data)
    assert torch.equal(torch.get_rng_state(),rng) and np.array_equal(data[3],before)


def test_validation_is_strict_even_for_zero_operation():
    data=list(scene());adapter=inference.BoundaryRelationPostprocessor(threshold=0)
    data[4]={'1':1}
    with pytest.raises(ValueError):adapter.process(*data)
    data=list(scene());data[1]=data[1].astype(np.float64)
    with pytest.raises(ValueError,match='float32'):adapter.process(*data)


def test_sparse_seam_component_audit_does_not_join_row_wrap():
    assert inference._seam_components(np.array([9,10]),(3,10))==2
    assert inference._seam_components(np.array([0,1,40,41]),(10,10))==2
