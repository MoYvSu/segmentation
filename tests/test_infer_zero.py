# -*- coding: utf-8 -*-
"""全链零噪声CPU合同；不读赛题数据，不加载训练权重。"""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import infer_zero as zero


def baseline():
    return zero.entry.load_config(str(zero.ROOT / 'config/inference/area_floor_200.yaml'))


def test_explicit_zero_source_audit_does_not_require_unused_cuda_noise_resources(monkeypatch):
    original = Path.is_file

    def without_noise_resources(path):
        if path.name in ('restore_noise.cu', 'restore_noise.ptx'):
            return False
        return original(path)

    monkeypatch.setattr(Path, 'is_file', without_noise_resources)
    with pytest.raises(RuntimeError, match='noise resource missing'):
        zero.entry.local_module_sources()
    sources = zero.entry.local_module_sources(include_noise_resources=False)
    assert not any(key.endswith(('.cu', '.ptx')) for key in sources)


class Restorer:
    num_steps = 16
    kappa = .03

    def __init__(self):
        self.calls = []

    def q_sample(self, clean, condition, step, *, noise=None, generator=None):
        assert generator is None and noise is not None
        assert noise.dtype == torch.float32 and noise.shape == condition.shape
        assert torch.count_nonzero(noise).item() == 0
        self.calls.append(noise.clone())
        return condition + self.kappa * noise

    def denoise(self, state, condition, step):
        assert torch.equal(state, condition)
        return state + .125


def fake_prediction(restorer, *, attempt_noise=None, local_count=2, global_count=1):
    def predict(model, actual_restorer, terminal, path, config, device, **kwargs):
        assert actual_restorer is restorer
        assert config['submission']['restoration_noise'] == zero.ZERO_MODE
        assert config['submission']['native_sizes'] == [1024]
        image = torch.full((1, 3, 3, 4), .25)
        if attempt_noise is not None:
            attempt_noise()
        for unused in range(global_count):
            restored = zero.backend_entry.restore_first(restorer, image, 123)
            assert torch.equal(restored, torch.full_like(image, .375))
        for index in range(local_count):
            audit = {}
            value, raw = zero.entry.views.restore_probe(restorer, image, 456 + index, 'zero', audit=audit)
            assert torch.equal(value, raw)
            kwargs['local_audit'].append(dict(size=1024, window=index, noise_mode='zero', **audit))
        return np.zeros((3, 4, 3), np.uint8), dict(instances=np.ones((3, 4), np.uint16), classes={'1': 1})
    return predict


def test_global_and_local_really_pass_zero_without_calling_any_random_generator(monkeypatch):
    restorer, records, windows = Restorer(), [], []
    cfg = baseline()
    before = deepcopy(cfg)
    sentinels = []

    def forbidden(*args, **kwargs):
        sentinels.append(True)
        raise AssertionError('Prediction must not generate random noise')

    for module, name in ((torch, 'randn'), (torch, 'randn_like'), (torch, 'normal'),
            (zero.backend_adaptation, 'restoration_noise'), (zero.noise_module, 'restoration_noise'),
            (zero.restoration_probe, 'restoration_noise'), (zero.entry.views, 'restore_first')):
        monkeypatch.setattr(module, name, forbidden)
    predictor = zero.zero_predict(fake_prediction(restorer), records)
    predictor(None, restorer, None, Path('test_169.jpg'), cfg, 'cpu', local_audit=windows)
    assert sentinels == [] and len(restorer.calls) == 3
    assert cfg == before and len(windows) == 2
    record = records[0]
    assert record['global_zero_calls'] == 1 and record['local_zero_calls'] == 2
    assert record['random_noise_generated'] is False
    assert [row['branch'] for row in record['calls']] == ['global', 'local', 'local']
    for row in record['calls']:
        assert row['noise_nonzero'] == 0 and row['initial_equals_condition'] is True
        assert row['initial_max_abs'] == 0 and row['q_sample_calls'] == 1 and row['step'] == 16


@pytest.mark.parametrize('call', [
    lambda: torch.randn((1, 3, 3, 4)),
    lambda: torch.randn_like(torch.zeros((1, 3, 3, 4))),
    lambda: zero.backend_adaptation.restoration_noise((1, 3, 3, 4), 1, 'cpu'),
    lambda: zero.restoration_probe.restoration_noise((1, 3, 3, 4), 1, 'cpu'),
    lambda: zero.noise_module.restoration_noise((1, 3, 3, 4), 1, 'cpu'),
])
def test_accidental_random_generation_during_prediction_fails_closed(call):
    restorer = Restorer()
    predictor = zero.zero_predict(fake_prediction(restorer, attempt_noise=call), [])
    with pytest.raises(RuntimeError, match='noise generation is forbidden'):
        predictor(None, restorer, None, Path('test_169.jpg'), baseline(), 'cpu')


@pytest.mark.parametrize('global_count,local_count', [(0, 2), (2, 2), (1, 0)])
def test_actual_global_and_local_calls_are_required(global_count, local_count):
    restorer = Restorer()
    predictor = zero.zero_predict(fake_prediction(restorer, global_count=global_count, local_count=local_count), [])
    with pytest.raises(RuntimeError, match='one global zero restoration'):
        predictor(None, restorer, None, Path('test_169.jpg'), baseline(), 'cpu')


def test_effective_zero_contract_preserves_baseline_and_uses_native1024():
    cfg = baseline()
    original = deepcopy(cfg)
    effective = zero.execution_config(cfg)
    assert cfg == original
    assert cfg['submission']['restoration_noise'] == zero.entry.NOISE_MODE
    assert effective['submission']['restoration_noise'] == zero.ZERO_MODE
    assert effective['submission']['global_restoration'] == dict(mode='zero', model='restorer')
    assert effective['submission']['local_restoration'] == dict(mode='zero', model='restorer')
    assert effective['submission']['native_sizes'] == [1024]
    assert effective['inference'] == cfg['inference']


def test_reports_show_actual_zero_runtime_with_historical_baseline_separate():
    cfg = baseline()
    doc = dict(format=zero.entry.FORMAT, config=cfg,
               runtime=dict(restoration_noise=dict(name=zero.entry.NOISE_MODE), precision='FP32'))
    before = deepcopy(doc)
    rows = [dict(source='test_169', global_zero_calls=1, local_zero_calls=9)]
    result = zero.audit_document(doc, rows, 'a' * 64)
    assert doc == before
    assert result['format'] == zero.FORMAT
    assert result['baseline_config'] == cfg
    assert result['config']['submission']['restoration_noise'] == zero.ZERO_MODE
    assert result['runtime']['restoration_noise']['name'] == zero.ZERO_MODE
    assert result['baseline_runtime'] == doc['runtime']
    assert result['global_zero_calls'] == 1 and result['local_zero_calls'] == 9
    assert zero.audit_document(result, rows, 'a' * 64) == result


@pytest.mark.parametrize('change', [
    lambda c: c['submission'].update(native_sizes=[768]),
    lambda c: c['submission']['local_restoration'].update(mode='single'),
    lambda c: c['inference'].update(boundary_threshold=.75),
    lambda c: c['inference'].update(final_min_instance_area=50),
])
def test_other_experiment_changes_are_not_allowed(change):
    cfg = baseline()
    change(cfg)
    with pytest.raises(ValueError):
        zero.validate_baseline(cfg)


@pytest.mark.parametrize('field,value', [('global_noise', 'single'),
    ('expected_model_sha', '0' * 64), ('seed_support', 'none')])
def test_reject_nonzero_or_alternative_models_before_loading(tmp_path, field, value):
    args = Namespace(config=str(zero.ROOT / 'config/inference/area_floor_200.yaml'),
                     output_dir=str(tmp_path / 'candidate'))
    setattr(args, field, value)
    with pytest.raises(ValueError):
        zero.run(args, device='cpu')
