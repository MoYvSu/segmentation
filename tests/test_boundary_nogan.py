# -*- coding: utf-8 -*-
"""显式零GAN消融：不调用D、保留G更新、关闭分支的严格短测和续训。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

import train_boundary_rgb as trainer
from models.boundary_restore import state_sha256
from test_boundary_ridge_training import batch, models, settings, tree_equal
from test_boundary_resize_training import small_models, synthetic_batch
from utils.boundary_rgb_training import rgb_gan_update
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return load_config('config/train/boundary_nogan.yaml')


def metrics():
    before = dict(loss=1., missing_points=5, supported_points=20,
        missing_probability=.2, supported_probability=.7, recorded_interior_input_probability=.9)
    after = dict(loss=.1, missing_points=5, supported_points=20,
        missing_probability=.9, supported_probability=.95, recorded_interior_input_probability=.03)
    return before, after


def test_recipe_changes_only_explicit_GAN_weight_and_output():
    current=config();control=load_config('config/train/boundary_ridge.yaml')
    trainer.validate_config(current)
    a,b=deepcopy(current['boundary_naf']),control['boundary_naf']
    assert a['training']['adversarial_weight']==0.
    assert b['training']['adversarial_weight']==.01
    a['output_dir']=b['output_dir'];a['training']['adversarial_weight']=b['training']['adversarial_weight']
    assert a==b


@pytest.mark.parametrize('bad',[-.1,float('nan'),float('inf'),True,'0'])
def test_invalid_GAN_weight_rejected(bad):
    value=config();value['boundary_naf']['training']['adversarial_weight']=bad
    with pytest.raises(ValueError,match='GAN权重'):
        trainer.validate_config(value)


def test_zero_GAN_never_forwards_or_updates_D_and_still_trains_G(monkeypatch):
    sample=batch(2);g,d=models();old_d=deepcopy(d.state_dict());old_g=deepcopy(g.state_dict())
    go,do=torch.optim.Adam(g.parameters()),torch.optim.Adam(d.parameters())
    initial_optimizer=deepcopy(do.state_dict())
    flags=[p.requires_grad for p in d.parameters()];mode=d.training
    def forbidden(*args,**kwargs):
        raise AssertionError('Explicit zero GAN must not forward D')
    monkeypatch.setattr(d,'forward',forbidden)
    for step in range(3):
        stats=rgb_gan_update(g,d,go,do,sample,adversarial_weight=0.,ridge_supervision=settings())
        assert stats['generator_updated'] and stats['nonzero_grad_g']
        for key in ['discriminator_updated','nonzero_grad_d','adversarial_active']:
            assert not stats[key]
        for key in ['grad_norm_d','loss_d','adv_d','adv_g','spectral_power_iterations']:
            assert stats[key]==0
    assert not tree_equal(old_g,g.state_dict())
    assert tree_equal(old_d,d.state_dict()) and tree_equal(initial_optimizer,do.state_dict())
    assert d.training==mode and [p.requires_grad for p in d.parameters()]==flags


def test_zero_GAN_microbatch_update_matches_full_batch():
    sample=batch(3);g,d=models();g2,d2=deepcopy(g),deepcopy(d)
    go,do=torch.optim.SGD(g.parameters(),lr=.03),torch.optim.SGD(d.parameters(),lr=.03)
    go2,do2=torch.optim.SGD(g2.parameters(),lr=.03),torch.optim.SGD(d2.parameters(),lr=.03)
    kw=dict(adversarial_weight=0.,ridge_supervision=settings(),grad_clip=None)
    a=rgb_gan_update(g,d,go,do,sample,micro_batch_size=3,**kw)
    b=rgb_gan_update(g2,d2,go2,do2,sample,micro_batch_size=1,**kw)
    for key in ['loss_g','coverage','interior','ridge_loss']:
        assert a[key]==pytest.approx(b[key],rel=1e-6)
    for x,y in zip(g.parameters(),g2.parameters()):
        torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)
    assert tree_equal(d.state_dict(),d2.state_dict()) and not do.state and not do2.state


def test_existing_GAN_smoke_gate_remains_exact_and_cannot_pass_without_D_updates():
    before,after=metrics()
    original=trainer.smoke_checks(before,after,True,1)
    positive=trainer.smoke_checks(before,after,True,1,adversarial_weight=.01)
    assert original==positive and all(original.values())
    assert not trainer.smoke_checks(before,after,True,0)['discriminator_updated']
    assert not any('unchanged' in key or 'not_updated' in key for key in positive)


@pytest.mark.parametrize('corruption', ['state','optimizer','forward','G','D'])
def test_zero_GAN_smoke_requires_D_inertness_and_G_learning(corruption):
    before,after=metrics()
    kw=dict(adversarial_weight=0.,discriminator_state_unchanged=True,
        discriminator_optimizer_unchanged=True,discriminator_forward_calls=0,generator_updates=128)
    count=0
    assert all(trainer.smoke_checks(before,after,True,count,**kw).values())
    if corruption=='state':kw['discriminator_state_unchanged']=False
    elif corruption=='optimizer':kw['discriminator_optimizer_unchanged']=False
    elif corruption=='forward':kw['discriminator_forward_calls']=1
    elif corruption=='G':kw['generator_updates']=0
    elif corruption=='D':count=1
    assert not all(trainer.smoke_checks(before,after,True,count,**kw).values())


def test_new_identity_has_identical_model_and_sources_but_binds_GAN_recipe(monkeypatch):
    current=config();control=load_config('config/train/boundary_ridge.yaml')
    _,g,d,_,_=small_models(current['boundary_naf'])
    monkeypatch.setattr(trainer,'file_sha',lambda path:str(path))
    dataset=SimpleNamespace(identity={'synthetic':True})
    a=trainer.source_identity(current,dataset,g,d);b=trainer.source_identity(control,dataset,g,d)
    assert a['architecture']==b['architecture'] and a['discriminator_architecture']==b['discriminator_architecture']
    assert a['source_sha256']==b['source_sha256']
    assert a['initial_model_sha256']==b['initial_model_sha256']
    assert a['initial_discriminator_sha256']==b['initial_discriminator_sha256']
    assert a['recipe']['training']['adversarial_weight']==0 and b['recipe']['training']['adversarial_weight']==.01


def test_zero_GAN_smoke_records_real_D_inertness_and_restores_optimizers(tmp_path,monkeypatch):
    value=config();cfg=value['boundary_naf'];cfg['model']=dict(width=4,encoder_blocks=[1,1],middle_blocks=1,decoder_blocks=[1,1],upsampling='project_bilinear_v1')
    cfg['discriminator']=dict(width=2)
    cfg['training'].update(effective_batch_size=1,micro_batch_size=1,segment_updates=cfg['data']['samples_per_epoch'],adversarial_warmup_updates=0,minimum_free_gib=.001)
    sample=synthetic_batch();sample['ridge_instances']=torch.ones_like(sample['target'],dtype=torch.int64);sample['ridge_instances'][...,48:]=2
    class Dataset:
        identity={'synthetic':True}
        def __getitem__(self,index):return index
    monkeypatch.setattr(trainer,'dataset_for',lambda config,epoch:Dataset())
    monkeypatch.setattr(trainer,'collate_boundary_rgb',lambda rows:deepcopy(sample))
    def identity(config,dataset,g,d):
        return dict(recipe=deepcopy(config['boundary_naf']),initial_model_sha256=state_sha256(g),initial_discriminator_sha256=state_sha256(d))
    monkeypatch.setattr(trainer,'source_identity',identity)
    monkeypatch.setattr(trainer,'verify_inputs',lambda *args:None)
    # 两步只验执行与续训合同，不放宽真实恢复门槛：不可学完则必须保留失败回执。
    try:
        result=trainer.smoke(value,tmp_path,'cpu',updates=2)
    except RuntimeError as error:
        assert '短测未通过' in str(error)
        result=json.loads((tmp_path/'smoke.json').read_text(encoding='utf-8'))
    assert result['gan_disabled'] and result['discriminator_updates']==0
    assert result['discriminator_forward_calls']==0 and result['generator_updates']==2
    assert result['discriminator_state_unchanged'] and result['discriminator_optimizer_unchanged']
    assert result['discriminator_initial_sha256']==result['discriminator_after_smoke_sha256']==result['discriminator_after_resume_sha256']
    assert result['generator_state_changed']
    assert result['resumed_update']['generator_updated'] and not result['resumed_update']['discriminator_updated']
    for key in ['discriminator_not_updated','discriminator_state_unchanged','discriminator_optimizer_unchanged',
                'discriminator_never_forwarded','generator_updated','all_generator_updates_present','generator_state_changed','resumed_discriminator_not_updated']:
        assert result['checks'][key],key
    assert 'discriminator_updated' not in result['checks']
