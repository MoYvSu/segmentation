# -*- coding: utf-8 -*-
"""修复RGB＋T8擦断合同：同RGB干净对照、浮点缓存、共同初态及真实梯度。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import train_boundary_rgb as trainer
import tools.boundary_rgb_monitor as monitor
from models.boundary_restore import state_sha256
from utils.boundary_rgb_inputs import clean_model_input, model_input_from_boundary
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return load_config('config/train/boundary_rgb_erasure.yaml')


def batch():
    side = 96
    target = torch.zeros(1, 1, side, side)
    target[..., 8:88, 48] = 1
    clean = torch.zeros_like(target)
    clean[..., 8:88, 44:52] = 1
    clean[..., 18:26, :] = 0
    request = torch.zeros_like(clean)
    request[..., 40:56, 40:57] = 1
    dirty = clean.masked_fill(request.bool(), 0)
    near = lambda x: torch.nn.functional.max_pool2d(x, 5, 1, 2) > .5
    rgb = torch.linspace(.1, .9, side).view(1, 1, 1, side).expand(1, 3, side, side).clone()
    artificial = target.bool() & near(clean) & ~near(dirty)
    erasure = dict(requested=True, applied=True, reason='applied', requested_count=1, applied_count=1,
        deleted_input_pixels=int((clean.bool() & request.bool()).sum()),
        artificial_gap_gt_points=int(artificial.sum()), segments=[dict(provenance='B')])
    return dict(input=torch.cat((rgb, dirty), 1), rgb=rgb, image=torch.zeros_like(rgb),
        input_boundary=dirty, clean_input_boundary=clean, erasure_mask=request,
        artificial_gap_mask=artificial.float(), target=target, valid=torch.ones_like(target),
        gap_mask=(target.bool() & ~near(clean)).float(), raw_input_boundary=clean,
        raw_gap_mask=(target.bool() & ~near(clean)).float(),
        meta=[dict(epoch=0, geometry=dict(rotation=0, flip=False), erasure=erasure)])


def small_cfg():
    cfg = deepcopy(config()['boundary_naf'])
    cfg['model'] = dict(width=2, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    cfg['discriminator'] = dict(width=2)
    cfg['training']['micro_batch_size'] = 1
    cfg['training']['minimum_free_gib'] = .001
    return cfg


def test_new_recipe_has_only_input_cache_and_output_changes_from_erasure():
    current = config()
    trainer.validate_config(current)
    cfg = deepcopy(current['boundary_naf'])
    old = load_config('config/train/boundary_erasure.yaml')['boundary_naf']
    assert cfg.pop('restored_rgb_manifest').endswith('/train_manifest.json')
    assert cfg['monitor'].pop('restored_rgb_manifest').endswith('/monitor_manifest.json')
    cfg['input_mode'] = old['input_mode']
    cfg['inference']['input_mode'] = old['inference']['input_mode']
    assert cfg['inference'].pop('rgb_input_format') == 'float32_unit'
    cfg['output_dir'] = old['output_dir']
    assert cfg == old


@pytest.mark.parametrize('mutate', [
    lambda c: c['boundary_naf']['monitor'].pop('restored_rgb_manifest'),
    lambda c: c['boundary_naf']['monitor'].update(restored_rgb_manifest=c['boundary_naf']['restored_rgb_manifest']),
    lambda c: c['boundary_naf'].update(restored_rgb_manifest=''),
    lambda c: c['boundary_naf'].update(input_mode='boundary'),
])
def test_restored_cache_contract_cannot_silently_fall_back_or_mix_domains(mutate):
    cfg = config()
    mutate(cfg)
    with pytest.raises(ValueError):
        trainer.validate_config(cfg)


@pytest.mark.parametrize('batched', [False, True])
def test_clean_input_retains_exact_same_RGB_and_changes_only_boundary(batched):
    sample = batch()
    if not batched:
        sample = {k: v[0] if torch.is_tensor(v) else v for k, v in sample.items()}
    clean = clean_model_input(sample, input_mode='rgb_boundary')
    assert torch.equal(clean[..., :3, :, :], sample['rgb'])
    assert torch.equal(clean[..., 3:, :, :], sample['clean_input_boundary'])
    assert not torch.equal(clean[..., 3:, :, :], sample['input_boundary'])
    assert clean_model_input(sample, input_mode='boundary') is sample['clean_input_boundary']


def test_RGB_input_helper_rejects_wrong_geometry_range_or_nonbinary_T():
    sample = batch()
    with pytest.raises(ValueError, match='aligned'):
        model_input_from_boundary(sample['input_boundary'], sample['rgb'][..., :-1], input_mode='rgb_boundary')
    with pytest.raises(ValueError, match='aligned'):
        model_input_from_boundary(sample['input_boundary'], sample['rgb']+2, input_mode='rgb_boundary')
    with pytest.raises(ValueError, match='binary'):
        model_input_from_boundary(sample['input_boundary']+.01, sample['rgb'], input_mode='rgb_boundary')


def test_shared_random_parameters_and_first_RGB_update_then_strict_reload(tmp_path):
    cfg = small_cfg()
    rgb, d = trainer.build_models(cfg, 'cpu')
    base_cfg = deepcopy(cfg)
    base_cfg['input_mode'] = 'boundary'
    base, base_d = trainer.build_models(base_cfg, 'cpu')
    a, b = rgb.state_dict(), base.state_dict()
    for key in b:
        if key == 'intro.weight':
            assert torch.equal(a[key][:, 3:4], b[key])
            assert torch.count_nonzero(a[key][:, :3]) == 0
        else:
            assert torch.equal(a[key], b[key])
    assert state_sha256(d) == state_sha256(base_d)
    sample = batch()
    with torch.no_grad():
        torch.testing.assert_close(rgb(sample['input']), base(sample['input_boundary']), atol=3e-7, rtol=1e-6)
    go, do = trainer.optimizers(rgb, d, cfg)
    stat = trainer.update(rgb, d, go, do, sample, cfg, 64)
    assert stat['rgb_intro_grad_norm'] > 0 and stat['rgb_intro_weights_l2'] > 0
    assert stat['erasure_applied_samples'] == 1 and stat['discriminator_updated']
    identity = dict(recipe=deepcopy(cfg), test='same RGB/clean damaged pair')
    trainer.save_checkpoint(tmp_path/'latest.pt', rgb, d, go, do, cfg, identity, 65, [0, 1])
    fresh, fresh_d = trainer.build_models(cfg, 'cpu')
    fresh_go, fresh_do = trainer.optimizers(fresh, fresh_d, cfg)
    assert trainer.load_checkpoint(tmp_path/'latest.pt', fresh, fresh_d, fresh_go, fresh_do, cfg, identity) == (65, [0, 1])
    assert state_sha256(fresh) == state_sha256(rgb)
    assert len(fresh_go.state) == len(go.state) > 0


def test_source_identity_locks_two_separate_manifests_and_conditional_helper(monkeypatch):
    cfg = config()
    g, d = trainer.build_models(small_cfg(), 'cpu')
    monkeypatch.setattr(trainer, 'file_sha', lambda p: str(p))
    identity = trainer.source_identity(cfg, SimpleNamespace(identity={}), g, d)
    assert 'utils/boundary_rgb_inputs.py' in identity['source_sha256']
    assert any(p.endswith('train_manifest.json') for p in identity['assets_sha256'])
    assert any(p.endswith('monitor_manifest.json') for p in identity['assets_sha256'])


class RGBFirst(nn.Module):
    input_mode = 'rgb_boundary'
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.inputs = []
    def forward(self, value):
        self.inputs.append(value.detach().clone())
        return value[:, :1] + self.anchor * 0


def inference():
    return dict(input_mode='rgb_boundary', tile_size=16, overlap=4, reflect_pad=2, batch_size=3,
        input_threshold=.5, boundary_representation='binary', iterations=1, revert_unconnected=False)


def test_float32_RGB_tiler_preserves_sub_uint8_precision_and_old_uint8_equivalence():
    boundary = np.zeros((19, 23), np.float32)
    value = np.full((19, 23, 3), .1234567, np.float32)
    model = RGBFirst()
    settings = dict(inference(), rgb_input_format='float32_unit')
    result, receipt = monitor.predict_boundary_rgb_tiled(boundary, value, model, settings)
    np.testing.assert_allclose(result, .1234567, atol=5e-8, rtol=0)
    assert receipt['input_rgb_scale'] == 'unit_interval_no_requantization'
    assert receipt['input_rgb_sha256'] == hashlib.sha256(value.tobytes()).hexdigest()
    uint8 = np.full(value.shape, 91, np.uint8)
    old, old_receipt = monitor.predict_boundary_rgb_tiled(boundary, uint8, model, inference())
    float_equiv = uint8.astype(np.float32)/255.
    new, _ = monitor.predict_boundary_rgb_tiled(boundary, float_equiv, model, settings)
    np.testing.assert_array_equal(old, new)
    assert 'input_rgb_dtype' not in old_receipt


@pytest.mark.parametrize('change', [
    lambda a: a.astype(np.float64), lambda a: a+2., lambda a: np.full_like(a, np.nan),
])
def test_float_RGB_tiler_rejects_bad_cache_values(change):
    with pytest.raises(ValueError):
        monitor.predict_boundary_rgb_tiled(np.zeros((19,23),np.float32),
            change(np.zeros((19,23,3),np.float32)), RGBFirst(), dict(inference(),rgb_input_format='float32_unit'))


class RGBBoundary(RGBFirst):
    def forward(self, value):
        assert value.shape[1] == 4
        self.inputs.append(value.detach().clone())
        return value[:, -1:] + self.anchor*0


def test_pair_monitor_uses_same_restored_RGB_for_clean_and_dirty(tmp_path, monkeypatch):
    sample = batch()
    sample = {k: v[0] if torch.is_tensor(v) else v for k,v in sample.items()}
    sample['meta'] = sample['meta'][0]
    sample['source'] = 'train_001'
    cfg = config()
    cfg['boundary_naf']['monitor'].update(pair_indices=[0], full_readout_steps=[], thumbnail_width=64)
    monkeypatch.setattr(monitor, 'original_view', lambda *_: np.ones((96,96),bool))
    monkeypatch.setattr(monitor, '_fixed_controls', lambda *_: pytest.fail('pair monitor cannot read test data'))
    captured=[]
    actual=monitor.panels
    def panels(path,title,cells,*a,**kw):
        captured.append([(name,array.copy()) for name,array in cells])
        return actual(path,title,cells,*a,**kw)
    monkeypatch.setattr(monitor,'panels',panels)
    model=RGBBoundary()
    result=monitor.save_monitor(model,cfg,tmp_path,5,{0:sample})
    assert len(model.inputs)==2
    assert torch.equal(model.inputs[0][:,:3],model.inputs[1][:,:3])
    assert torch.equal(model.inputs[1][:,3:],sample['clean_input_boundary'][None])
    assert captured[0][0][0]=='RGB / training source' and captured[0][0][1].max()==0
    assert captured[0][1][0]=='D5a RGB / model condition' and captured[0][1][1].max()>0
    assert result['rows'][0]['rgb_conditioning']=='restored_float32_cache'
    assert Image.open(tmp_path/'monitor/s000005/pair_000.jpg').size[0]==9*64


def test_full_monitor_passes_restored_float_RGB_only_to_candidate_tiler(tmp_path, monkeypatch):
    import data.boundary_rgb as data
    side=16; raw=np.full((side,side,3),27,np.uint8); restored=np.full(raw.shape,.234567,np.float32)
    ids=np.ones((side,side),np.uint16); boundary=np.zeros((side,side),np.float32)
    cache=tmp_path/'raw.npz';np.savez(cache,rgb=raw,boundary=boundary,semantic=np.ones_like(boundary),raw_probability=boundary,instances=ids)
    classes=tmp_path/'classes.json';classes.write_text(json.dumps({'1':1}))
    cfg=config();b=cfg['boundary_naf'];b['input_normalization']={'mode':'none'};b['inference']['input_normalization']={'mode':'none'}
    b['monitor'].update(pair_indices=[],full_readout_steps=[0],source_names=['test_101'],roi_spec=None)
    monkeypatch.setattr(monitor,'_fixed_controls',lambda *_:[(dict(path=str(cache),control_classes=str(classes),source='test_101',sha256='cache-sha'),{})])
    monkeypatch.setattr(monitor,'evaluate_frozen_readout',lambda *a,**kw:(ids,{'1':1},{'same_readout':True}))
    def load(path):
        assert str(path).endswith('monitor_manifest.json')
        return {'domain':'monitor_test'}
    monkeypatch.setattr(data,'load_restored_rgb_manifest',load)
    def rgb(cache,name,original):
        assert name=='test_101';np.testing.assert_array_equal(original,raw)
        return restored,{'source':name,'file_sha256':'restored-sha'}
    monkeypatch.setattr(data,'restored_rgb_for_source',rgb)
    seen=[]
    def predict(B,RGB,*a,**kw):
        assert RGB is restored and RGB.dtype==np.float32
        seen.append(True)
        return boundary,{'input_mode':'rgb_boundary'}
    monkeypatch.setattr(monitor,'predict_boundary_rgb_tiled',predict)
    result=monitor.save_monitor(RGBBoundary(),cfg,tmp_path/'out',0,{})
    assert seen==[True]
    assert result['rows'][0]['rgb_conditioning']['file_sha256']=='restored-sha'


def test_full_monitor_rejects_training_RGB_table(tmp_path, monkeypatch):
    import data.boundary_rgb as data
    cfg=config()
    cfg['boundary_naf']['monitor'].update(pair_indices=[],full_readout_steps=[0])
    monkeypatch.setattr(data,'load_restored_rgb_manifest',lambda *_:{'domain':'train'})
    monkeypatch.setattr(monitor,'_fixed_controls',lambda *_:pytest.fail('wrong RGB domain must fail before test controls'))
    with pytest.raises(ValueError,match='monitor_test'):
        monitor.save_monitor(RGBBoundary(),cfg,tmp_path,0,{})
