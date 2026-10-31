# -*- coding: utf-8 -*-
"""原生尺度、配对随机流、梯度累积／续训及局部恢复器隔离契约。"""
import copy
import json
from pathlib import Path
from contextlib import nullcontext
from unittest import mock

import cv2
import numpy as np
import pytest
import torch

from data.restore_native import NativeRestorationDataset, crop_native
from models.rgb_restoration import build_rgb_restorer
from tools.restore_native_monitor import structure_metrics
from train_restore_native import make_initial, update, train, state_sha
from train_rgb_restoration import save_checkpoint
from utils.config import load_config


@pytest.fixture
def config(tmp_path):
    cfg=load_config('config/train/restore_native.yaml');cfg['paths']['project_root']=str(tmp_path)
    data=tmp_path/'images';data.mkdir()
    rng=np.random.default_rng(12)
    for i in range(8):
        rgb=rng.integers(0,256,(80,96,3),dtype=np.uint8)
        assert cv2.imwrite(str(data/f'train_{i:03d}.png'),rgb)
    c=cfg['rgb_restoration'];c.update(data_dir='images',expected_train_sources=8,image_size=64,crop_size=64)
    c['masked_pretraining'].update(patch_size=[4,8],lowpass_sigma=[1.,2.])
    c['model'].update(width=4,encoder_blocks=[1],middle_blocks=1,decoder_blocks=[1],time_dim=4)
    c['train'].update(num_workers=0,amp=False)
    cfg['restore_native'].update(input_size=64,arms={'native':[64],'mixed':[64,32]})
    return cfg


def test_native_crop_then_resize_and_identity(config):
    ds=NativeRestorationDataset(config,'native');item=ds.item(0,size=32,kind='identity',augment=False)
    recipe=json.loads(item['recipe']);y,x,h,w=recipe['box']
    raw=cv2.cvtColor(cv2.imread(ds.samples[0]),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    expected=cv2.resize(raw[y:y+h,x:x+w],(64,64),interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(expected,item['target'].permute(1,2,0).numpy())
    assert torch.equal(item['input'],item['target'])
    assert recipe['target_is_interpolated'] and recipe['zoom']==2
    assert recipe['downsample_scale']==1 and not recipe['local_task']


def test_exact_balanced_views_and_common_inputs(config):
    a,b=[NativeRestorationDataset(config,k) for k in ('native','mixed')]
    for epoch in (0,1,19):
        a.set_epoch(epoch);b.set_epoch(epoch)
        assert int((b.view_sizes==32).sum())==4
        for i in range(8):
            left,right=a[i],b[i];rl,rr=[json.loads(v['recipe']) for v in (left,right)]
            for key in ('source','profile','kind','base_sigma_native','downsample_scale','turns','flip','crop_fractions'):
                assert rl[key]==rr[key]
            if rr['native_size']==64:
                assert all(torch.equal(left[k],right[k]) for k in ('input','target','valid'))


def test_native_sigma_and_half_resample(config):
    ds=NativeRestorationDataset(config,'mixed')
    for size in (32,64):
        item=ds.item(2,size=size,kind='endpoint',downsample=.5,local_task=False,augment=False)
        r=json.loads(item['recipe'])
        assert r['sigma_network_max']==r['sigma_native_max']*(64/size)
        assert r['resampled_native_hw']==[size//2,size//2]
        assert r['weak_fraction']>=.05 and r['strong_fraction']>=.05
        assert item['input'].shape==(3,64,64)


def test_padding_has_no_loss_or_gradient_supervision(config):
    from train_rgb_restoration import reconstruction_loss
    raw=np.full((12,18,3),120,np.uint8)
    clean,valid,box=crop_native(raw,32,(0,0))
    assert int(valid.sum())==216 and box==(0,0,12,18)
    target=torch.from_numpy(clean.transpose(2,0,1).copy())[None]
    weight=torch.from_numpy(valid)[None,None]
    pred=target.clone().requires_grad_(True)
    changed=pred+(1-weight)*100
    loss=reconstruction_loss(changed,target,weight,config['rgb_restoration']['loss'])
    assert loss.item()==0;loss.backward()
    assert torch.equal(pred.grad,torch.zeros_like(pred))


def test_gradient_accumulation_matches_full_batch(config):
    torch.set_num_threads(1);torch.manual_seed(7)
    c=config['rgb_restoration'];a=build_rgb_restorer(c['model']);b=copy.deepcopy(a)
    batch={'input':torch.rand(4,3,32,32),'target':torch.rand(4,3,32,32),'valid':torch.ones(4,1,32,32)}
    optim=[torch.optim.AdamW(m.parameters(),lr=2e-4) for m in (a,b)]
    results=[]
    for model,opt,micro in zip((a,b),optim,(4,1)):
        results.append(update(model,batch,opt,torch.amp.GradScaler('cuda',enabled=False),
            torch.Generator().manual_seed(42),torch.device('cpu'),c,micro))
    assert results[0]['timesteps']==results[1]['timesteps']
    assert abs(results[0]['loss']-results[1]['loss'])<1e-6
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p,q,atol=2e-7,rtol=1e-5)


def test_resume_equals_continuous_and_keeps_60_epoch_schedule(config,tmp_path):
    torch.set_num_threads(1)
    roots=[tmp_path/'continuous',tmp_path/'segmented']
    fixed=NativeRestorationDataset(config,'native').item(0,kind='uniform',local_task=False)
    for root in roots:
        root.mkdir();make_initial(config,root/'initial.pt')
        save_checkpoint(root/'fixed.pt',{'pairs':[fixed],'reals':[]})
        save_checkpoint(root/'old_fixed.pt',{'pairs':[fixed['input']],'reals':[]})
    with mock.patch('train_restore_native.monitor',return_value={}):
        train(config,'native',roots[0],device='cpu',stop=2)
        train(config,'native',roots[1],device='cpu',stop=1)
        train(config,'native',roots[1],device='cpu',stop=2,resume=True)
    checkpoints=[torch.load(r/'native/last.pt',weights_only=True) for r in roots]
    assert checkpoints[0]['updates']==checkpoints[1]['updates']==4
    assert checkpoints[0]['scheduler']==checkpoints[1]['scheduler']
    assert checkpoints[0]['scheduler']['T_max']==60
    assert state_sha(checkpoints[0]['model'])==state_sha(checkpoints[1]['model'])
    assert torch.equal(checkpoints[0]['rng']['noise_generator'],checkpoints[1]['rng']['noise_generator'])


def test_clear_reference_metrics_reject_spurious_internal_edges():
    reference=np.full((96,96,3),.8,np.float32)
    cv2.circle(reference,(48,48),20,(.25,.25,.25),2)
    valid=np.ones((96,96),np.float32)
    base=structure_metrics(reference,reference,valid)
    altered=reference.copy();altered[40:43,39:58]=0
    bad=structure_metrics(reference,altered,valid)
    assert base['unsupported_edge_pixels']==0
    assert bad['unsupported_edge_pixels']>0
    assert base['small_closed_retained']==base['small_closed_reference_regions']


def test_extra_stage_monitors_preserve_full_training_state(config,tmp_path):
    """新增阶段终点真的生成缩略图，同时不改变连续训练的完整参数与随机状态。"""
    torch.set_num_threads(1)
    roots=[tmp_path/'full_monitor',tmp_path/'stage_monitor']
    fixed=NativeRestorationDataset(config,'native').item(0,kind='uniform',local_task=False)
    fixed.update(name='fixed_uniform',roi=[16,16,32,32])
    for root in roots:
        root.mkdir();make_initial(config,root/'initial.pt')
        save_checkpoint(root/'fixed.pt',{'pairs':[fixed],'reals':[]})
        save_checkpoint(root/'old_fixed.pt',{'pairs':[fixed['input']],'reals':[]})
    train(config,'native',roots[0],device='cpu',stop=6)
    for endpoint in (2,3,6):
        train(config,'native',roots[1],device='cpu',stop=endpoint,resume=endpoint!=2)
    assert (roots[1]/'native/monitor/e03/fixed_uniform.png').exists()
    checkpoints=[torch.load(r/'native/last.pt',weights_only=True) for r in roots]
    def equal(left,right):
        if isinstance(left,torch.Tensor):assert torch.equal(left,right)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:equal(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            assert len(left)==len(right)
            for a,b in zip(left,right):equal(a,b)
        else:assert left==right
    for key in ('model','optimizer','scheduler','scaler','rng','epoch','updates'):
        equal(checkpoints[0][key],checkpoints[1][key])


def test_only_local_restorer_changes_fixed_semantics():
    import tools.affinity_native_views as module
    class Backend(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Identity();self.semantic_decoder=mock.Mock(return_value=torch.zeros(1,1,8,8))
    model=Backend();old=torch.nn.Identity();candidate=torch.nn.Identity()
    rgb=np.zeros((8,8,3),np.uint8);image=torch.zeros(1,3,8,8)
    calls=[]
    def maps(m,p,c,d,r,s):
        assert r is old
        return rgb,torch.zeros(1,1,8,8),torch.zeros(1,1,8,8)
    def restore(r,t,s):
        calls.append(r);return t+(.3 if r is candidate else .1)
    with mock.patch.object(module,'prediction_maps',side_effect=maps), \
         mock.patch.object(module,'prepare_image',return_value=(rgb,image,0,0)), \
         mock.patch.object(module,'tensor_from_rgb',return_value=(image,0,0)), \
         mock.patch.object(module,'crop_letterbox_output',side_effect=lambda v,*a:v), \
         mock.patch.object(module,'restore_first',side_effect=restore), \
         mock.patch.object(module,'boundary_grid',side_effect=lambda m,t,c:t[:,:1]), \
         mock.patch.object(module,'decode_native',return_value=(np.ones((8,8),np.uint16),None)), \
         mock.patch.object(module,'revote_instances',return_value=({'1':1},None)):
        audit_a={};audit_b={};cfg={'affinity_native':{'overlap':.25}}
        _,a=module.predict_views(model,old,'unused',cfg,'cpu',7,[8],fixed_audit=audit_a)
        _,b=module.predict_views(model,old,'unused',cfg,'cpu',7,[8],local_restorer=candidate,fixed_audit=audit_b)
    assert calls==[old,candidate] and audit_a==audit_b
    assert not np.array_equal(a['patch8']['boundary'],b['patch8']['boundary'])
