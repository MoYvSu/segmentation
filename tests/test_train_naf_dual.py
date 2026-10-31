# -*- coding: utf-8 -*-
"""双输入NAF训练接入合同；合成CPU样本，不读取赛题GT或已有权重。"""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

import train_boundary_rgb as base
from models.boundary_dual import BoundaryDualNAFNet
from models.boundary_naf import BoundaryDiscriminator
from models.boundary_restore import state_sha256
from tools import train_naf_dual as entry
from train_boundary_restore import capture_rng
from utils.boundary_rgb_training import build_tolerant_targets, rgb_gan_update
from utils.config import load_config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    entry._TEST_CROPS.clear()
    yield
    entry._TEST_CROPS.clear()
    torch.set_num_threads(previous)


def config():
    return load_config('config/train/naf_dual.yaml')


class TinyDual(nn.Module):
    """可显式检查连续分支梯度的小网络，保持实际输入次序。"""
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(1, 1, 1)
        self.continuous_intro = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.intro.weight.zero_()
            self.intro.bias.zero_()
            self.continuous_intro.weight.zero_()

    def forward_logits(self, value):
        return self.intro(value[:, :1]) + self.continuous_intro(value[:, 1:2])

    def forward(self, value):
        return self.forward_logits(value).sigmoid()


def batch(size=40):
    target = torch.zeros(1, 1, size, size)
    target[..., 8:size - 8, size // 2] = 1
    t8 = torch.zeros_like(target)
    continuous = torch.full_like(target, .2)
    continuous[target.bool()] = .9
    return dict(input=torch.cat((t8, continuous), 1), input_boundary=t8,
                target=target, valid=torch.ones_like(target, dtype=torch.bool))


def execute_update(values, **kwargs):
    g, d = TinyDual(), BoundaryDiscriminator(width=2, seed=9)
    go, do = torch.optim.SGD(g.parameters(), lr=.1), torch.optim.SGD(d.parameters(), lr=.1)
    result = rgb_gan_update(g, d, go, do, values, adversarial_weight=0.,
                            grad_clip=None, **kwargs)
    return result, g, d


def test_config_explicit_contract_and_old_recipe_are_unchanged():
    cfg = config()
    before = deepcopy(cfg)
    settings = entry.validate_config(cfg)
    assert cfg == before and settings is cfg['boundary_naf']
    assert settings['input_mode'] == 'boundary_continuous'
    assert settings['input_contract'] == 't8_continuous_v1'
    assert settings['inference']['boundary_representation'] == 'binary_t8_and_continuous_n1'
    assert settings['training']['minimum_free_gib'] >= 10
    assert settings['training']['monitor_updates'] == [0, 1, 100, 250, 500, 750, 1000]
    assert not settings['monitor']['full_readout_steps']


@pytest.mark.parametrize('path,value', [
    (('input_mode',), 'boundary'),
    (('input_contract',), 'other'),
    (('inference', 'input_contract'), 'other'),
    (('inference', 'boundary_representation'), 'binary'),
    (('input_normalization', 'kernel_size'), 6),
    (('model', 'upsampling'), 'project_bilinear_v1'),
    (('training', 'main_updates'), 200),
    (('training', 'effective_batch_size'), 8),
    (('training', 'main_schedule_updates'), 1000),
    (('training', 'learning_rate'), .0001),
    (('training', 'adversarial_weight'), 0.),
    (('training', 'adversarial_warmup_updates'), 0),
    (('rgb_loss', 'tolerance'), 3),
    (('rgb_loss', 'interior_weight'), 2.),
    (('monitor', 'full_readout_steps'), [1000]),
    (('input_perturbation',), {'manifest': 'unused.json'}),
])
def test_config_rejects_hidden_recipe_changes(path, value):
    cfg = config()
    destination = cfg['boundary_naf']
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value
    with pytest.raises(ValueError):
        entry.validate_config(cfg)


def test_old_loss_default_rejects_two_channels_without_updates():
    values = batch()
    with pytest.raises(ValueError, match='channel contract'):
        execute_update(values)


@pytest.mark.parametrize('channels', [1, 4])
def test_old_loss_explicit_none_and_omitted_keyword_have_exact_updates(channels):
    from test_boundary_rgb_training import models, training_batch
    g1, d1 = models(channels)
    g2, d2 = models(channels)
    values = training_batch(count=1, channels=channels)
    go1, do1 = torch.optim.Adam(g1.parameters()), torch.optim.Adam(d1.parameters())
    go2, do2 = torch.optim.Adam(g2.parameters()), torch.optim.Adam(d2.parameters())
    first = rgb_gan_update(g1, d1, go1, do1, values)
    second = rgb_gan_update(g2, d2, go2, do2, values, boundary_input_channel=None)
    assert first == second
    for left, right in ((g1, g2), (d1, d2)):
        assert all(torch.equal(value, right.state_dict()[key])
                   for key, value in left.state_dict().items())
    for left, right in ((go1, go2), (do1, do2)):
        a, b = left.state_dict(), right.state_dict()
        assert a['param_groups'] == b['param_groups'] and a['state'].keys() == b['state'].keys()
        for parameter in a['state']:
            for key, value in a['state'][parameter].items():
                other = b['state'][parameter][key]
                assert torch.equal(value, other) if torch.is_tensor(value) else value == other


@pytest.mark.parametrize('channel', [-1, 1, False, True, 0.0])
def test_dual_loss_requires_explicit_integer_zero_channel(channel):
    with pytest.raises(ValueError, match='channel=0'):
        execute_update(batch(), boundary_input_channel=channel)


def test_dual_update_uses_T8_for_missing_and_learns_continuous_information():
    values = batch()
    original = {key: value.clone() for key, value in values.items()}
    prepared = build_tolerant_targets(values['target'], values['valid'], tolerance=2)
    stats, g, _ = execute_update(values, boundary_input_channel=0)
    # T8全无线，全部可信GT为missing；连续B在同处大于.5，不能替换定义。
    assert stats['missing_gt_points'] == int(prepared['positive'].sum()) > 0
    assert stats['generator_updated'] and not stats['discriminator_updated']
    gradient = g.continuous_intro.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert g.continuous_intro.weight.detach().abs().sum() > 0
    with torch.no_grad():
        normal = g(values['input'])
        zero_b = values['input'].clone()
        zero_b[:, 1:] = 0
        assert not torch.equal(normal, g(zero_b))
    for key, previous in original.items():
        assert torch.equal(values[key], previous)
    json.dumps(stats, allow_nan=False)


def test_dual_update_rejects_nonbinary_first_channel():
    values = batch()
    values['input'][:, :1] = .2
    with pytest.raises(ValueError, match='binary T8'):
        execute_update(values, boundary_input_channel=0)


def test_scoped_hooks_restore_every_replaced_attribute_after_failure(monkeypatch):
    names = ('FORMAT', 'validate_config', 'dataset_for', 'build_models',
             'collate_boundary_rgb', 'source_identity', 'update',
             'reconstruction_metrics', 'rgb_gan_update', 'monitor')
    calls = []

    def previous_update(*args):
        calls.append('old_update')
        return {'loss_g': 7., 'old_sentinel': 11}

    def previous_metrics(logits, values, cfg, **kwargs):
        assert torch.equal(values['input'], values['input_boundary'])
        calls.append('old_metrics')
        return {'sentinel': 3}

    monkeypatch.setattr(base, 'update', previous_update)
    monkeypatch.setattr(base, 'reconstruction_metrics', previous_metrics)
    previous = {name: getattr(base, name) for name in names}
    g = TinyDual()
    g.continuous_intro.weight.grad = torch.ones_like(g.continuous_intro.weight)
    with pytest.raises(RuntimeError, match='intentional'):
        with entry.adapted_trainer():
            assert base.FORMAT == entry.FORMAT != previous['FORMAT']
            values = batch()
            result = base.update(g, None, None, None, values, None, 0)
            assert result['loss_g'] == 7. and result['old_sentinel'] == 11
            assert result['continuous_intro_grad_norm'] > 0
            assert base.reconstruction_metrics(None, values, None) == {'sentinel': 3}
            raise RuntimeError('intentional')
    assert calls == ['old_update', 'old_metrics']
    assert all(getattr(base, name) is value for name, value in previous.items())


def same_rng(a, b):
    assert a.keys() == b.keys()
    for key in a:
        left, right = a[key], b[key]
        if torch.is_tensor(left):
            assert torch.equal(left, right)
        elif isinstance(left, tuple) and any(isinstance(x, np.ndarray) for x in left):
            assert left[0] == right[0] and np.array_equal(left[1], right[1])
            assert left[2:] == right[2:]
        else:
            assert left == right


def test_monitor_preserves_model_rng_targets_and_training_mode(tmp_path, monkeypatch):
    cfg = config()
    cfg['boundary_naf']['data']['crop_size'] = 32
    cfg['boundary_naf']['monitor'].update(pair_indices=[0], thumbnail_width=64,
                                         control_manifests=[], source_names=[])
    source = batch(32)
    sample = {key: value[0].float() for key, value in source.items()}
    sample.update(image=torch.full((3, 32, 32), .4), source='train_synthetic', box=[0, 0, 32, 32])
    original_target = sample['target'].clone()

    class Dataset:
        def __getitem__(self, _draw):
            # 即使读取monitor样本消耗随机数，也必须恢复训练随机流。
            random.random(); np.random.random(); torch.rand(1)
            return sample

    g = BoundaryDualNAFNet(width=4, encoder_blocks=(1, 1), middle_blocks=1,
                           decoder_blocks=(1, 1), seed=37)
    before = state_sha256(g)
    rng = capture_rng()
    monkeypatch.setattr(entry, '_fixed_controls', lambda *args: [])
    result = entry.monitor(g, cfg, tmp_path, 0, Dataset())
    assert result['complete'] and result['no_WS'] and result['no_full_test_inference']
    assert len(result['rows']) == 1 and result['rows'][0]['domain'] == 'train'
    assert result['rows'][0]['branch_effect_mae'] == 0.
    assert state_sha256(g) == before and g.training
    same_rng(rng, capture_rng())
    assert torch.equal(original_target, sample['target'])
    assert (tmp_path/'monitor/s000000/draw_00000.jpg').stat().st_size > 0
    assert (tmp_path/'monitor/index.html').exists()


def test_fixed_test_monitor_uses_continuous_native_crop_and_rejects_changed_cache(tmp_path, monkeypatch):
    cfg = config()
    (tmp_path/'manifest.json').write_text('{}', encoding='utf-8')
    cfg['boundary_naf']['data']['crop_size'] = 32
    cfg['boundary_naf']['monitor'].update(pair_indices=[], thumbnail_width=64,
        control_manifests=[str(tmp_path/'manifest.json')], source_names=['test_synthetic'])
    boundary = np.full((44, 48), .2, np.float32)
    boundary[:, 23:25] = .9
    rgb = np.full((44, 48, 3), 100, np.uint8)
    path = tmp_path/'cache.npz'
    np.savez_compressed(path, boundary=boundary, rgb=rgb)
    row = dict(source='test_synthetic', path=str(path), sha256=base.file_sha(path))
    monkeypatch.setattr(entry, '_fixed_controls', lambda *args: [(row, {})])
    g = BoundaryDualNAFNet(width=4, encoder_blocks=(1, 1), middle_blocks=1,
                           decoder_blocks=(1, 1), seed=37)
    g.eval()
    before, rng = state_sha256(g), capture_rng()
    result = entry.monitor(g, cfg, tmp_path/'out', 0, None)
    assert result['no_test_GT'] and result['no_WS']
    assert result['rows'][0]['crop'] == [8, 6, 32, 32]
    assert result['rows'][0]['domain'] == 'monitor_test'
    assert state_sha256(g) == before and not g.training
    same_rng(rng, capture_rng())
    assert (tmp_path/'out/monitor/s000000/test_synthetic.jpg').exists()
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='缓存改变'):
        entry.monitor(g, cfg, tmp_path/'later', 1, None)
    assert state_sha256(g) == before and not g.training
    same_rng(rng, capture_rng())


def test_fixed_test_crop_cache_reloads_for_changed_source_names(tmp_path, monkeypatch):
    cfg = config()
    (tmp_path/'manifest.json').write_text('{}', encoding='utf-8')
    cfg['boundary_naf']['data']['crop_size'] = 32
    settings = cfg['boundary_naf']['monitor']
    settings.update(control_manifests=[str(tmp_path/'manifest.json')], source_names=['test_first'])
    records = {}
    for name, brightness in [('test_first', .2), ('test_second', .4)]:
        path = tmp_path/(name + '.npz')
        boundary = np.full((40, 40), brightness, np.float32)
        boundary[:, 20] = .9
        np.savez_compressed(path, boundary=boundary, rgb=np.full((40, 40, 3), 100, np.uint8))
        records[name] = dict(source=name, path=str(path), sha256=base.file_sha(path))
    calls = []

    def controls(_config, current_settings):
        calls.append(tuple(current_settings['source_names']))
        return [(records[name], {}) for name in current_settings['source_names']]

    monkeypatch.setattr(entry, '_fixed_controls', controls)
    first = entry._test_crops(cfg)
    assert first[0]['source'] == 'test_first'
    settings['source_names'] = ['test_second']
    second = entry._test_crops(cfg)
    assert second[0]['source'] == 'test_second'
    assert calls == [('test_first',), ('test_second',)]
    assert first is not second and not np.array_equal(first[0]['continuous'], second[0]['continuous'])
    assert entry._test_crops(cfg) is second
    assert len(calls) == 2
