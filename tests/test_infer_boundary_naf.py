# -*- coding: utf-8 -*-
"""NAF包装器的加载、最终解码链及独立参数/状态审计；不读赛题数据、不需要GPU。"""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from models.boundary_naf import BoundaryNAFNet
from models.boundary_restore import state_sha256
from tools import infer_boundary_naf as wrapper


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def checkpoint(tmp_path, change=None):
    model_cfg = dict(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    model = BoundaryNAFNet(**model_cfg, seed=17)
    saved = dict(format=wrapper.CHECKPOINT_FORMAT, global_step=1000,
        architecture=model.architecture, model_state_dict=model.state_dict(),
        model_sha256=state_sha256(model), input_contract=deepcopy(wrapper.INPUT_CONTRACT),
        config=dict(model=model_cfg, seed=17, inference=deepcopy(wrapper.INPUT_CONTRACT)))
    if change:
        change(saved)
    path = tmp_path / 'naf.pt'
    torch.save(saved, path)
    return path, model


def test_strict_load_keeps_cpu_rng_and_freezes_model(tmp_path):
    path, expected = checkpoint(tmp_path)
    state = torch.get_rng_state().clone()
    model, receipt = wrapper.load_naf_checkpoint(path, 'cpu')
    assert torch.equal(state, torch.get_rng_state())
    assert state_sha256(model) == state_sha256(expected)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert receipt['parameter_count'] == sum(p.numel() for p in expected.parameters())
    assert receipt['global_step'] == 1000 and receipt['checkpoint_sha256'] == wrapper.entry.sha(path)


@pytest.mark.parametrize('change,pattern', [
    (lambda s: s.update(format='old_naf'), 'formal'),
    (lambda s: s.update(global_step=999), 'formal'),
    (lambda s: s['config'].update(seed=18), 'architecture/config'),
    (lambda s: s.update(model_sha256='0' * 64), 'SHA256 differs'),
    (lambda s: s['input_contract'].update(input_mode='binary'), 'input contract'),
    (lambda s: s['input_contract'].update(extra_key=True), 'input contract'),
    (lambda s: s['input_contract'].update(tile_size=256.), 'input contract'),
    (lambda s: s['config']['inference'].update(input_threshold=.4), 'input_contract/config'),
])
def test_reject_changed_or_unknown_checkpoint_contract(tmp_path, change, pattern):
    path, _ = checkpoint(tmp_path, change)
    with pytest.raises((ValueError, RuntimeError), match=pattern):
        wrapper.load_naf_checkpoint(path, 'cpu')


def install_fake_entry(tmp_path, monkeypatch, *, decode_count=1, alter_naf=False,
                       alter_input=False, base_report_extra=0):
    path, expected = checkpoint(tmp_path)
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    source = inputs / 'test_169.png'
    source.write_bytes(b'filename-only synthetic source')
    output = tmp_path / 'candidate'
    args = Namespace(input_dir=str(inputs), output_dir=str(output), model_dir=None,
        config=str(wrapper.ROOT / 'config/inference/semifinal_n1.yaml'), dataset='semifinal_150',
        only=['test_169'], naf_checkpoint=str(path), capture_names=None)
    state = dict(helper_calls=0, raw_predict_calls=0, reflect_calls=0,
                 sentinel=object(), before=torch.full((1, 1, 8, 9), .2),
                 semantic=torch.full((1, 1, 8, 9), -.3))
    backend, restorer = nn.Linear(1, 1), nn.Linear(1, 1)
    base_count = sum(p.numel() for module in [backend, restorer] for p in module.parameters())

    def fake_models(*unused):
        return backend, restorer, None

    def fake_helper(before, naf, settings, *, device):
        state['helper_calls'] += 1
        assert settings == wrapper.INPUT_CONTRACT and device == 'cpu'
        assert wrapper._frozen(naf)
        if alter_naf:
            with torch.no_grad():
                next(naf.parameters()).add_(1)
        after = np.full_like(before, .8)
        if alter_input:
            before[:] = .9
        return after, dict(output_sha256=wrapper.array_sha(after),
            global_skeletonization_before_tiling=True, input_mode='skeleton')

    def fake_reflect(semantic, boundary, config, *, original_decode=None):
        state['reflect_calls'] += 1
        assert semantic is state['semantic'] and original_decode is state['sentinel']
        assert torch.equal(boundary, torch.full_like(boundary, .8))
        return np.ones((8, 9), np.uint16), {'1': 1}

    def fake_predict(model, restorer, terminal, path, config, device, **kwargs):
        state['raw_predict_calls'] += 1
        result = (np.ones((8, 9), np.uint16), {'1': 1})
        for unused in range(decode_count):
            result = wrapper.entry.decode_reflect(state['semantic'], state['before'], config,
                                                  original_decode=state['sentinel'])
        return np.zeros((8, 9, 3), np.uint8), dict(
            instances=result[0], classes=result[1], boundary=state['before'][0, 0].numpy())

    def fake_run(actual_args):
        assert not actual_args.save_analysis
        (output / 'audit').mkdir(parents=True)
        (output / 'results').mkdir()
        wrapper.entry.write(output / 'audit/status.json', dict(complete=False, status='running'))
        config = wrapper.entry.load_config(actual_args.config)
        models = wrapper.entry.load_models(config, {}, 'cpu')
        wrapper.entry.predict(*models, source, config, 'cpu', local_restorer=None)
        wrapper.entry.write(output / 'audit/report.json', dict(complete=True,
            frozen_models_unchanged=True, parameter_count=base_count + base_report_extra,
            geometry='old frozen N1 geometry', models={'backend': {}, 'restorer': {}},
            model_states={'backend': 'fixed', 'restorer': 'fixed'}, images=[dict(source='test_169')]))
        wrapper.entry.write(output / 'audit/status.json', dict(complete=True, status='complete'))

    monkeypatch.setattr(wrapper.entry, 'load_models', fake_models)
    monkeypatch.setattr(wrapper.entry, 'predict', fake_predict)
    monkeypatch.setattr(wrapper.entry, 'decode_reflect', fake_reflect)
    monkeypatch.setattr(wrapper.entry, 'run', fake_run)
    monkeypatch.setattr(wrapper, 'predict_boundary_tiled', fake_helper)
    return args, state, output, expected, base_count, fake_predict, fake_reflect


def test_hook_calls_saved_reflect_once_keeps_inputs_and_counts_naf_once(tmp_path, monkeypatch):
    args, state, output, expected, base_count, fake_predict, fake_reflect = install_fake_entry(tmp_path, monkeypatch)
    before = state['before'].clone()
    semantic = state['semantic'].clone()
    receipt = wrapper.run(args, device='cpu')
    count = sum(p.numel() for p in expected.parameters())
    assert receipt['parameter_count'] == base_count + count
    assert receipt['upstream_parameter_count'] == base_count
    assert state['helper_calls'] == state['reflect_calls'] == state['raw_predict_calls'] == 1
    assert torch.equal(before, state['before']) and torch.equal(semantic, state['semantic'])
    assert wrapper.entry.predict is fake_predict and wrapper.entry.decode_reflect is fake_reflect
    assert not (output / 'audit/naf_captures').exists()
    report = json.loads((output / 'audit/report.json').read_text())
    assert report['parameter_count'] == receipt['parameter_count']
    assert report['parameter_count_includes_naf']
    assert report['models']['boundary_naf']['parameter_count'] == count
    assert report['upstream_geometry'] == 'old frozen N1 geometry'
    assert report['images'][0]['boundary_naf']['input_boundary_sha256'] == wrapper.array_sha(before[0, 0].numpy())
    assert report['returned_boundary_is_upstream'] and receipt['naf_frozen_unchanged']
    assert report['naf_complete']
    assert json.loads((output / 'audit/status.json').read_text())['naf_complete']


def test_only_named_source_capture_is_outside_results(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_names = ['test_169']
    receipt = wrapper.run(args, device='cpu')
    relative = receipt['images'][0]['capture']
    assert relative == 'audit/naf_captures/test_169.npz'
    with np.load(output / relative) as values:
        assert set(values.files) == {'boundary_before', 'boundary_after', 'semantic'}
        assert np.array_equal(values['boundary_before'], state['before'][0, 0].numpy())
        assert np.all(values['boundary_after'] == np.float32(.8))
        assert np.array_equal(values['semantic'], state['semantic'].numpy())
    assert not list((output / 'results').iterdir())


@pytest.mark.parametrize('count', [0, 2])
def test_missing_or_double_decode_is_rejected(tmp_path, monkeypatch, count):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch, decode_count=count)
    with pytest.raises(RuntimeError, match='exactly one'):
        wrapper.run(args, device='cpu')
    assert state['helper_calls'] == min(count, 1)
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


@pytest.mark.parametrize('kind,pattern', [('input', 'modified its input'), ('model', 'checkpoint/model/source')])
def test_input_or_naf_state_mutation_is_rejected(tmp_path, monkeypatch, kind, pattern):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch,
        alter_input=kind == 'input', alter_naf=kind == 'model')
    with pytest.raises(RuntimeError, match=pattern):
        wrapper.run(args, device='cpu')
    assert torch.all(state['before'] == .2)
    assert not json.loads((output / 'audit/status.json').read_text())['complete']
    if kind == 'model':
        report = json.loads((output / 'audit/report.json').read_text())
        assert not report['complete'] and not report['naf_complete']


def test_total_parameter_limit_checked_before_prediction(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, 'MAX_PARAMETERS', 1)
    with pytest.raises(RuntimeError, match='parameters exceed'):
        wrapper.run(args, device='cpu')
    assert state['raw_predict_calls'] == state['helper_calls'] == 0


def test_base_report_cannot_already_include_naf_parameters(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch, base_report_extra=1)
    with pytest.raises(RuntimeError, match='counted twice'):
        wrapper.run(args, device='cpu')
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


def test_unselected_capture_rejected_before_creating_output(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_names = ['test_170']
    with pytest.raises(ValueError, match='capture-names'):
        wrapper.run(args, device='cpu')
    assert not output.exists()
