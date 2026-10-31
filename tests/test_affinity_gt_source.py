# -*- coding: utf-8 -*-
"""虚构图验证共同域、配对曝光、训练隔离与完整优化器/RNG续训。"""
from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from train_affinity_connectivity import configure_training, frozen_digest, draw_receipt
from train_affinity_gt_source import (
    R1_SHA, D5A_SHA, HEAD_FORMAT, PairGeometrySource, IndependentMixedDataset,
    batch_signature, checkpoint_payload, compare_pair_rows, epoch_order,
    initial_logits_parity, load_targets, relative_input, schedule_factor,
    strict_resume, validate_config, validate_pair,
    verify_checkpoint_log, validate_peer_status, verify_canonical_rows,
)
from train_backend_adaptation import tensor_digest
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def cohort(tmp_path):
    raw = tmp_path/'data'/'raw'; gt_dir = tmp_path/'gt'; pair_dir = tmp_path/'pairs'
    for p in (raw, gt_dir, pair_dir): p.mkdir(parents=True)
    rows = []
    for index in range(2):
        stem = f'train_{index:03d}'
        yy, xx = np.mgrid[:48, :64]
        rgb = np.stack([xx*3, yy*4, (xx+yy+index)*2], -1).astype(np.uint8)
        image = raw/(stem+'.png'); cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        image.with_suffix('.json').write_text('{}')
        manual = np.ones((48, 64), np.uint16); manual[:,32:] = 2; manual[4:8,4:8] = 0
        teacher = np.ones_like(manual)*3; teacher[24:] = 4; teacher[9:13,9:13] = 0
        original = manual > 0; filled = np.zeros_like(original)
        original[0,0] = False; filled[0,0] = True
        common = (manual > 0)&(teacher > 0)
        gt = gt_dir/(stem+'_gt.npz')
        np.savez(gt, instance_map=manual, original_covered=original, filled=filled)
        (gt_dir/(stem+'_class.json')).write_text(json.dumps({'1':1, '2':0}))
        pair = pair_dir/(stem+'.npz')
        np.savez(pair, manual_ids=np.where(common, manual, 0), teacher_ids=np.where(common, teacher, 0),
                 common_known=common, original_covered=original, filled=filled)
        rows.append(dict(source=stem, image=image.relative_to(tmp_path).as_posix(), image_sha256=sha(image),
                         manual_gt=gt.relative_to(tmp_path).as_posix(), manual_gt_sha256=sha(gt),
                         pair_npz=pair.relative_to(tmp_path).as_posix(), pair_sha256=sha(pair)))
    manifest = dict(format='affinity_gt_source_targets_v1', status='complete', rows=rows,
                    quality_confirmation='pending_user_review', teacher_checkpoint_sha256=R1_SHA,
                    restorer_checkpoint_sha256=D5A_SHA, semantic_labels_used_for_teacher_training=False)
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    config = dict(paths=dict(project_root=str(tmp_path)), gt_source=dict(manifest='manifest.json',
                  approval_file='quality_review.json', seed=20261004),
                  affinity_native=dict(crop_attempts=32, minimum_known_pairs=16))
    base = CanonicalBackendDataset(raw, gt_dir, image_size=32, affinity_grid=16)
    degrade = dict(profile_probabilities=[.2,.8,0,0], blur_sigma=[.4,1.5],
                   resample_probability=0., resample_scale=[.5,.9])
    original = PairedDegradationDataset(base, degrade, dict(enabled=True, probability=.5, sigma=[0,.006]),
                                       seed=42, repeats=32)
    return config, manifest, original


def test_private_confirmation_gate_does_not_fabricate_approval(cohort):
    config, manifest, _ = cohort
    assert len(load_targets(config, 'manual', expected_sources=2)[0]['rows']) == 2
    with pytest.raises(RuntimeError, match='explicit_user_confirmed'):
        load_targets(config, 'teacher', expected_sources=2)
    path = relative_input(config, config['gt_source']['manifest'])
    (path.parent/'quality_review.json').write_text(json.dumps(dict(status='explicit_user_confirmed',
        manifest_sha256=sha(path), review_checked_target_count=2)))
    assert load_targets(config, 'teacher', expected_sources=2)[0]['semantic_labels_used_for_teacher_training'] is False
    assert json.loads(path.read_text())['quality_confirmation']=='pending_user_review'


@pytest.mark.parametrize('field,value', [('manifest_sha256','wrong'),('review_checked_target_count',1),
                                       ('status','pending_user_review')])
def test_quality_review_binds_immutable_manifest_and_full_cohort(cohort, field, value):
    config, _, _ = cohort; path=relative_input(config,config['gt_source']['manifest'])
    review=dict(status='explicit_user_confirmed',manifest_sha256=sha(path),review_checked_target_count=2)
    review[field]=value;(path.parent/'quality_review.json').write_text(json.dumps(review))
    with pytest.raises(RuntimeError,match='explicit_user_confirmed'):
        load_targets(config,'teacher',expected_sources=2)


@pytest.mark.parametrize('field', ['image', 'manual_gt', 'pair_npz'])
def test_manifest_hashes_bind_actual_source_and_both_targets(cohort, field):
    config, manifest, _ = cohort
    relative_input(config, manifest['rows'][0][field]).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='SHA'):
        load_targets(config, 'manual', expected_sources=2)


@pytest.mark.parametrize('field,value', [('semantic_labels_used_for_teacher_training',True),
    ('status','incomplete'), ('teacher_checkpoint_sha256','incorrect'), ('restorer_checkpoint_sha256','incorrect')])
def test_wrong_teacher_provenance_is_rejected(cohort, field, value):
    config, manifest, _ = cohort; manifest[field] = value
    relative_input(config, config['gt_source']['manifest']).write_text(json.dumps(manifest))
    with pytest.raises(ValueError): load_targets(config, 'manual', expected_sources=2)


@pytest.mark.parametrize('value', ['../outside.npz','C:/outside.npz','/outside.npz','data\\raw\\a.png'])
def test_manifest_path_cannot_escape_project(cohort, value):
    with pytest.raises(ValueError): relative_input(cohort[0], value)


def test_common_mask_native_crop_and_online_augmentation_are_identical(cohort):
    config, manifest, original = cohort
    a = IndependentMixedDataset(original, 'manual', config, pair_rows=manifest['rows'], arm='manual')
    b = IndependentMixedDataset(original, 'manual', config, pair_rows=manifest['rows'], arm='teacher')
    np_before = deepcopy(np.random.get_state()); py_before=random.getstate(); before=torch.get_rng_state().clone()
    for epoch in [1,2]:
        a.set_epoch(epoch); b.set_epoch(epoch)
        for draw in epoch_order(config['gt_source']['seed'],epoch,0,64)[:8]:
            x,y=a[draw],b[draw]
            for field in ['image','clean_image','foreground','valid_content','affinity_valid_content','crop_box']:
                assert torch.equal(x[field],y[field]),field
            assert draw_receipt(x)==draw_receipt(y) and x['training_view']==y['training_view']
            assert 'semantic_target' not in y
            ta,va=build_affinity_targets_torch(x['instance_map'][None],x['valid_content'][None])
            tb,vb=build_affinity_targets_torch(y['instance_map'][None],y['valid_content'][None])
            assert torch.equal(va,vb) and not torch.equal(ta,tb)
            assert not x['uncovered_boundary_source'] and not y['uncovered_boundary_source']
    assert random.getstate()==py_before and torch.equal(torch.get_rng_state(),before)
    assert np.array_equal(np.random.get_state()[1],np_before[1])


def test_crop_selection_depends_on_common_coverage_instead_of_instance_shapes(cohort):
    config, manifest, _ = cohort
    a=PairGeometrySource(config,manifest['rows'],'manual',image_size=32,grid=16,seed=1)
    b=PairGeometrySource(config,manifest['rows'],'teacher',image_size=32,grid=16,seed=1)
    for draw in range(6):
        for ds in (a,b):ds.size=16;ds.epoch=2;ds.draw=draw
        x,y=a[0],b[0]
        assert torch.equal(x['crop_box'],y['crop_box']) and torch.equal(x['image'],y['image'])
        y0,x0,y1,x1=x['crop_box'].tolist();assert y1-y0==16 and x1-x0==16


def test_view_plan_balances_actual_updates_for_independently_shuffled_streams(cohort):
    config, manifest, original = cohort
    ds=IndependentMixedDataset(original,'manual',config,pair_rows=manifest['rows'],arm='manual');ds.set_epoch(3)
    manual_order=epoch_order(config['gt_source']['seed'],3,0,64)
    pseudo_order=epoch_order(config['gt_source']['seed'],3,1,64)
    from data.affinity_mixed import view_plan
    plan=view_plan(config['gt_source']['seed'],3,64)
    assert [ds.mapping[i] for i in manual_order]==plan
    assert plan.count(0)==32 and plan.count(1024)==32 and manual_order!=pseudo_order
    assert sorted(manual_order)==list(range(64))
    # 同一个update两流共享plan；各流draw->view通过其自己的固定shuffle建立。
    assert dict(zip(pseudo_order,plan))[pseudo_order[17]]==ds.mapping[manual_order[17]]


def test_whole_sam2_signature_has_no_semantic_or_affinity_alias_dependency():
    ids=torch.ones((1,8,8),dtype=torch.long);ids[:,:,4:]=2;ids[:,0,0]=0
    batch=dict(instance_map=ids,valid_content=torch.ones((1,1,8,8),dtype=torch.bool),
               image=torch.zeros(1,3,8,8),clean_image=torch.zeros(1,3,8,8))
    sig=batch_signature(batch,batch['image']);assert sig['shape']==[1,8,8,8]


def test_pair_receipt_detects_actual_input_or_valid_relation_drift():
    from train_affinity_gt_source import COMMON_KEYS
    row={key:{} for key in COMMON_KEYS};compare_pair_rows(row,deepcopy(row))
    for field in ['inputs','manual_crop','manual_draw','training_view']:
        bad=deepcopy(row);bad[field]={'different':True}
        with pytest.raises(RuntimeError,match=field):compare_pair_rows(row,bad)


def test_stage_stop_uses_twenty_epoch_schedule_and_does_not_change_parent_config():
    cfg=load_config('config/train/affinity_gt_source.yaml');validate_config(cfg)
    old=load_config('config/train/affinity_balance_mixed.yaml');value=deepcopy(cfg);value.pop('gt_source')
    assert value==old and cfg['backend_adaptation']['epochs']==20
    assert cfg['gt_source']['epochs']*cfg['gt_source']['updates_per_epoch']==320
    assert schedule_factor(5)>schedule_factor(20)==pytest.approx(.1)
    assert schedule_factor(5)==pytest.approx(.1+.9*(1+np.cos(np.pi/4))/2)


class ToyHead(nn.Module):
    def __init__(self):
        super().__init__();self.seq=nn.Sequential(nn.Conv2d(3,8,1),nn.BatchNorm2d(8),nn.Dropout2d(.2))
    def forward(self,x):return {'affinity_logits':self.seq(x)}


def toy_model():
    return SimpleNamespace(affinity_decoder=ToyHead())


def test_actual_initial_logits_parity_preserves_train_modes_and_bn():
    torch.manual_seed(8);head=ToyHead().train();ref=deepcopy(head).eval()
    before=tensor_digest(head.state_dict().items())
    assert initial_logits_parity(head,ref,torch.rand(2,3,8,8))['passed']
    assert all(m.training for m in head.modules()) and tensor_digest(head.state_dict().items())==before
    with torch.no_grad():ref.seq[0].weight.add_(.1)
    with pytest.raises(RuntimeError,match='initial'):initial_logits_parity(head,ref,torch.rand(2,3,8,8))


def test_resume_restores_optimizer_scheduler_rng_and_next_dropout_update_exact(tmp_path):
    torch.manual_seed(7);random.seed(9);np.random.seed(4)
    m=toy_model();opt=torch.optim.AdamW(m.affinity_decoder.parameters(),lr=2e-5,eps=1e-4)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:schedule_factor(e))
    def step(model,optimizer):
        optimizer.zero_grad();x=torch.rand(2,3,8,8)
        loss=model.affinity_decoder(x)['affinity_logits'].square().mean();loss.backward();optimizer.step()
    step(m,opt);sched.step()
    contract={'frozen':'example'};saved=checkpoint_payload(m,opt,sched,{},'manual',1,64,contract)
    file=tmp_path/'head.pt';torch.save(saved,file)
    step(m,opt);expected=tensor_digest(m.affinity_decoder.state_dict().items())
    rng_after=torch.get_rng_state().clone()
    m2=toy_model();opt2=torch.optim.AdamW(m2.affinity_decoder.parameters(),lr=1.,eps=1e-4)
    sched2=torch.optim.lr_scheduler.LambdaLR(opt2,lambda e:schedule_factor(e))
    payload=torch.load(file,map_location='cpu',weights_only=False)
    assert strict_resume(payload,m2.affinity_decoder,opt2,sched2,contract,'manual')==(1,64)
    assert opt2.param_groups[0]['lr']==opt.param_groups[0]['lr']
    step(m2,opt2)
    assert tensor_digest(m2.affinity_decoder.state_dict().items())==expected
    assert torch.equal(torch.get_rng_state(),rng_after)


@pytest.mark.parametrize('change', ['arm','version','format','budget','contract','keys'])
def test_resume_rejects_wrong_arm_budget_contract_or_architecture(change):
    m=toy_model();opt=torch.optim.AdamW(m.affinity_decoder.parameters());sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    contract={'source':'same'};p=checkpoint_payload(m,opt,sched,{},'manual',1,64,contract)
    if change=='arm':p['arm']='teacher'
    elif change=='version':p['version']=2
    elif change=='format':p['format']='older'
    elif change=='budget':p['updates']=63
    elif change=='contract':p['contract']={}
    else:p['geometry_state_dict'].pop(next(iter(p['geometry_state_dict'])))
    with pytest.raises((ValueError,RuntimeError)):strict_resume(p,m.affinity_decoder,opt,sched,contract,'manual')


def test_existing_training_modes_train_only_head_and_freeze_upstream_buffers():
    class M(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,3,1),nn.BatchNorm2d(3))
            self.semantic_decoder=nn.Conv2d(3,1,1);self.affinity_decoder=ToyHead()
    m=M();configure_training(m);before=frozen_digest(m)
    with torch.no_grad():features=m.encoder(torch.rand(2,3,8,8))
    m.affinity_decoder(features)['affinity_logits'].square().mean().backward()
    assert m.affinity_decoder.training and m.affinity_decoder.seq[1].training
    assert m.affinity_decoder.seq[2].training and not m.encoder.training and not m.semantic_decoder.training
    assert frozen_digest(m)==before
    assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('affinity_decoder.'))


@pytest.mark.parametrize('change', ['wrong_labels','expanded_common','nonzero_outside','overlap','range','mask_float'])
def test_pair_validator_rejects_labels_or_provenance_contract_drift(change):
    full=np.ones((8,8),np.uint16);full[:,4:]=2
    known=np.ones((8,8),bool);known[0,0]=False
    a=dict(manual_ids=np.where(known,full,0),teacher_ids=np.where(known,3,0).astype(np.uint16),
           common_known=known.copy(),original_covered=np.ones((8,8),bool),filled=np.zeros((8,8),bool))
    if change=='wrong_labels':a['manual_ids'][1,1]=2
    elif change=='expanded_common':a['common_known'][0,0]=True
    elif change=='nonzero_outside':a['teacher_ids'][0,0]=3
    elif change=='overlap':a['filled'][1,1]=True
    elif change=='range':a['teacher_ids']=a['teacher_ids'].astype(np.int32);a['teacher_ids'][1,1]=65536
    else:a['common_known']=a['common_known'].astype(np.float32)
    with pytest.raises(ValueError):validate_pair(a,full,full.shape)


def test_resume_requires_the_exact_log_not_just_same_row_count(tmp_path):
    path=tmp_path/'steps.jsonl';path.write_text('\n'.join(json.dumps(dict(updates=i+1)) for i in range(64))+'\n')
    payload=dict(updates=64,draw_log_sha256=sha(path));assert len(verify_checkpoint_log(payload,path))==64
    text=path.read_text();path.write_text(text.replace('"updates": 1}', '"updates": 0}',1))
    with pytest.raises(ValueError,match='exact'):verify_checkpoint_log(payload,path)


def test_peer_must_be_complete_with_same_origin_and_freezing():
    from train_affinity_gt_source import FORMAT
    c={'R1':'same'};s=dict(format=FORMAT,status='complete',arm='manual',contract=c,updates=320,
                          frozen_unchanged=True,restorer_unchanged=True,strict_reload_equal=True)
    validate_peer_status(s,c,'teacher',5)
    for field,value in [('status','running'),('arm','teacher'),('contract',{}),
                         ('updates',319),('frozen_unchanged',False)]:
        bad={**s,field:value}
        with pytest.raises(ValueError):validate_peer_status(bad,c,'teacher',5)


def test_manual_gt_is_the_canonical_same_source_not_just_matching_shape(cohort):
    config, manifest, original = cohort
    verify_canonical_rows(original.base_dataset,manifest['rows'],config)
    rows=deepcopy(manifest['rows']);rows[0]['manual_gt']=rows[1]['manual_gt']
    with pytest.raises(ValueError,match='canonical processed'):
        verify_canonical_rows(original.base_dataset,rows,config)


def test_resume_can_compare_completed_peer_prefix_without_claiming_full_budget():
    from train_affinity_gt_source import FORMAT
    c={'R1':'same'};s=dict(format=FORMAT,status='complete',arm='manual',contract=c,updates=320,
                          frozen_unchanged=True,restorer_unchanged=True,strict_reload_equal=True)
    with pytest.raises(ValueError):validate_peer_status(s,c,'teacher',10)
    validate_peer_status(s,c,'teacher',10,allow_prefix=True)
    with pytest.raises(ValueError):validate_peer_status({**s,'updates':319},c,'teacher',10,allow_prefix=True)
