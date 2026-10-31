# -*- coding: utf-8 -*-
"""同视野双头融合的共享路径、参数和错误输入门槛；GPU端点另作完整复现。"""
import numpy as np
import pytest
import torch
from torch import nn

from tools.probe_affinity_balance_fusion import (
    EXPECTED_PARAMETERS, deployment_parameters, frozen_guard, fuse_probability,
    shared_boundary_grids, validate_full_gate, write,
)
from utils.affinity_fusion import affinity_boundary_probability


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.))
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return [image*self.scale]


class Head(nn.Module):
    def __init__(self, bias, *, fail=False, mutate=False):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.calls = 0
        self.received = None
        self.fail, self.mutate = fail, mutate

    def forward(self, features):
        self.calls += 1
        self.received = features
        if self.fail:
            raise RuntimeError('head failure')
        if self.mutate:
            features[0].add_(1)
        logits = features[0][:, :1].expand(-1, 8, -1, -1)+self.bias
        return {'affinity_logits': logits}


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.semantic_decoder = nn.Linear(3, 1)
        self.affinity_decoder = Head(-1.)


def frozen(module):
    return module.eval().requires_grad_(False)


def test_single_encoder_and_two_distinct_heads_reuse_same_features():
    model, mixed = frozen(Backend()), frozen(Head(1.))
    # 保留原from_numpy/permute布局，辅助记录不得把输入改成contiguous。
    image = torch.from_numpy(np.ones((5, 7, 3), np.float32)).permute(2, 0, 1).unsqueeze(0)
    stride = image.stride()
    grids, audit = shared_boundary_grids(model, mixed, image, mode='gated',
                                        fusion_kwargs={'short_reduction': 'top2'})
    assert model.encoder.calls == model.affinity_decoder.calls == mixed.calls == 1
    assert model.affinity_decoder.received is mixed.received
    assert image.stride() == stride and not image.is_contiguous()
    features = mixed.received[0]
    for key, head in (('native', model.affinity_decoder), ('mixed', mixed)):
        expected = affinity_boundary_probability(features[:, :1].expand(-1, 8, -1, -1)+head.bias,
                                                 mode='gated', short_reduction='top2')
        torch.testing.assert_close(grids[key], expected, rtol=0, atol=0)
    assert audit['encoder_calls'] == 1 and audit['shared_features_value_check']
    assert not torch.equal(grids['native'], grids['mixed'])


def test_exception_does_not_swap_head_change_input_or_module_state():
    model, mixed = frozen(Backend()), frozen(Head(1., fail=True))
    original = model.affinity_decoder
    image = torch.rand(1, 3, 5, 7)
    before = image.clone()
    with pytest.raises(RuntimeError, match='head failure'):
        with frozen_guard({'model': model, 'mixed': mixed}) as audit:
            shared_boundary_grids(model, mixed, image, mode='mean', fusion_kwargs={})
    assert model.affinity_decoder is original and audit['unchanged']
    torch.testing.assert_close(image, before, rtol=0, atol=0)


def test_frozen_guard_and_shared_features_reject_mutation():
    layer = frozen(nn.Linear(2, 1))
    with pytest.raises(RuntimeError, match='Frozen deployment'):
        with frozen_guard({'layer': layer}):
            with torch.no_grad():
                layer.weight.add_(.1)
    model, mixed = frozen(Backend()), frozen(Head(1., mutate=True))
    with pytest.raises(RuntimeError, match='changed shared features'):
        shared_boundary_grids(model, mixed, torch.ones(1, 3, 3, 3), mode='mean', fusion_kwargs={})


def test_parameters_count_two_heads_once_and_shared_modules_once():
    model, mixed, restorer = Backend(), Head(1.), nn.Linear(2, 3)
    result = deployment_parameters(model, mixed, restorer)
    assert result['shared_without_affinity'] == 1+4  # encoder scale + semantic weight/bias
    assert result['native_affinity'] == result['mixed_affinity'] == 1
    assert result['d5a'] == 9
    assert result['total'] == result['sum'] == result['actual_unique_parameters'] == 16
    with pytest.raises(ValueError, match='independent'):
        deployment_parameters(model, model.affinity_decoder, restorer)
    mixed.bias = model.affinity_decoder.bias
    with pytest.raises(ValueError, match='overlap'):
        deployment_parameters(model, mixed, restorer)


def test_continuous_fusion_endpoints_preserve_dtype_and_coordinates():
    native = np.array([[.1, .8], [.9, .4]], np.float32)
    mixed = np.array([[.7, .4], [.5, .2]], np.float32)
    for weight, expected in ((0, native), (1, mixed)):
        value = fuse_probability(native, mixed, weight)
        assert value.dtype == expected.dtype and not np.shares_memory(value, expected)
        np.testing.assert_array_equal(value, expected)
    np.testing.assert_allclose(fuse_probability(native, mixed), [[.4, .6], [.7, .3]])


@pytest.mark.parametrize('left,right,weight,message', [
    (np.zeros((2, 3)), np.zeros((3, 2)), .5, 'two-dimensional'),
    (np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), .5, 'two-dimensional'),
    (np.zeros((2, 3)), np.full((2, 3), np.nan), .5, 'probability'),
    (np.zeros((2, 3)), np.full((2, 3), 1.01), .5, 'probability'),
    (np.zeros((2, 3)), np.full((2, 3), -.01), .5, 'probability'),
    (np.zeros((2, 3)), np.zeros((2, 3)), float('nan'), 'weight'),
    (np.zeros((2, 3)), np.zeros((2, 3)), -1, 'weight'),
])
def test_invalid_probabilities_or_coordinates_rejected(left, right, weight, message):
    with pytest.raises(ValueError, match=message):
        fuse_probability(left, right, weight)


def test_wrong_network_grid_rejected_and_original_head_retained():
    model, mixed = frozen(Backend()), frozen(Head(1.))
    original = model.affinity_decoder
    with pytest.raises(ValueError, match='logits shape'):
        shared_boundary_grids(model, mixed, torch.ones(1, 3, 3, 4), mode='mean',
                              fusion_kwargs={}, output_grid=512)
    assert model.affinity_decoder is original
    with pytest.raises(ValueError, match='FP32'):
        shared_boundary_grids(model, mixed, torch.ones(1, 3, 3, 3).half(), mode='mean', fusion_kwargs={})


def test_full_gate_requires_explicit_release_and_unchanged_complete_probe(tmp_path):
    config = {'affinity_balance_fusion': {'mixed_weight': .5}}
    identities, sources, caches = {'native': 'e20'}, {'runtime': 'hash'}, {'native': 'cache'}
    cases = [{'name': 'source'+str(i)} for i in range(12)]
    arguments = (tmp_path, config, identities, sources, caches, cases)
    with pytest.raises(ValueError, match='--full-ok'):
        validate_full_gate(*arguments, False)
    report = dict(complete=True, mode='probe', samples=12, frozen_exact=True,
                  full_output_endpoint_parity=True, sources_unchanged=True,
                  checkpoints=identities, sources=sources, endpoint_caches=caches,
                  case_identities=cases, parameters={'total': EXPECTED_PARAMETERS})
    rows = [dict(source=case['name'], endpoint_parity={'native': True, 'mixed': True}) for case in cases]
    write(tmp_path/'report.json', report)
    write(tmp_path/'rows.json', rows)
    write(tmp_path/'config.json', config)
    write(tmp_path/'status.json', {'status': 'complete'})
    receipt = validate_full_gate(*arguments, True)
    assert receipt['human_stage_release'] and receipt['technical_only']
    with pytest.raises(ValueError, match='sources'):
        validate_full_gate(tmp_path, config, identities, {'runtime': 'new'}, caches, cases, True)
    rows[0]['endpoint_parity']['mixed'] = False
    write(tmp_path/'rows.json', rows)
    with pytest.raises(ValueError, match='endpoint probe'):
        validate_full_gate(*arguments, True)
