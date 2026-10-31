# -*- coding: utf-8 -*-
"""共享参数距离梯度诊断：真实BCE、有效通道分母、未知ignore及零向量。"""
from copy import deepcopy
import json

import pytest
import torch

from tools.probe_affinity_distance_gradients import probe_distance_gradients
from utils.affinity_loss import balanced_affinity_loss


OPTIONS = dict(negative_weight=1., hard_negative_weight=1., hard_negative_gamma=2.)


def sample():
    torch.manual_seed(213)
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1), torch.nn.Tanh(), torch.nn.Conv2d(3, 8, 1)).double()
    logits = model(torch.randn(2, 2, 3, 4, dtype=torch.float64))
    target = torch.randint(2, logits.shape).double()
    valid = torch.rand(logits.shape) > .25
    valid[:, :, 0, 0] = True
    weights = .25 + torch.rand(logits.shape, dtype=torch.float64)
    parameters = {'affinity_head.0.weight': model[0].weight, 'fpn.example.bias': model[0].bias}
    return model, logits, target, valid, weights, parameters


def test_real_bce_native_gradient_reconstruction_and_cshort_per_channel_formula():
    model, logits, target, valid, weights, parameters = sample()
    before = deepcopy(model.state_dict())
    for parameter in parameters.values(): parameter.grad = torch.rand_like(parameter)
    grads_before = {name: parameter.grad.clone() for name, parameter in parameters.items()}
    result = probe_distance_gradients(logits, target, valid, parameters, edge_weight=weights,
                                     rtol=1.e-10, atol=1.e-12)
    assert result['native_group_weights'] == {'short': .5, 'distance2': .25, 'distance4': .25}
    assert result['cshort_group_weights'] == {'short': 4 / 5.5, 'distance2': 1 / 5.5, 'distance4': .5 / 5.5}
    channel_losses = [balanced_affinity_loss(logits[:, i:i+1], target[:, i:i+1], valid[:, i:i+1],
                     edge_weight=weights[:, i:i+1], **OPTIONS)[0] for i in range(8)]
    explicit_native = torch.stack(channel_losses).mean()
    explicit_cshort = sum(loss * coefficient for loss, coefficient in
                         zip(channel_losses, [1.] * 4 + [.5] * 2 + [.25] * 2)) / 5.5
    actual_loss, _ = balanced_affinity_loss(logits, target, valid, edge_weight=weights, **OPTIONS)
    assert result['original_loss'] == float(actual_loss.detach())
    assert result['reconstructed_native_loss'] == pytest.approx(float(explicit_native.detach()), rel=1.e-12)
    assert result['cshort_loss'] == pytest.approx(float(explicit_cshort.detach()), rel=1.e-12)
    native = torch.autograd.grad(explicit_native, tuple(parameters.values()), retain_graph=True)
    cshort = torch.autograd.grad(explicit_cshort, tuple(parameters.values()), retain_graph=True)
    for (name, parameter), g_native, g_cshort in zip(parameters.items(), native, cshort):
        report = result['parameters'][name]
        assert report['native_mixture_norm'] == pytest.approx(float(g_native.norm()), rel=1.e-10)
        assert report['cshort_mixture_norm'] == pytest.approx(float(g_cshort.norm()), rel=1.e-10)
        assert report['native_vs_cshort_cosine'] == pytest.approx(
            float((g_native * g_cshort).sum() / (g_native.norm() * g_cshort.norm())), rel=1.e-10)
        assert report['reconstruction']['relative_l2_error'] < 1.e-12
        assert torch.equal(parameter.grad, grads_before[name])
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert result['parameter_values_exact'] and result['accumulated_gradients_untouched']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('normalize', [True, False])
def test_partial_channels_use_actual_active_count_not_fixed_half_quarter(normalize):
    _, logits, target, valid, weights, parameters = sample()
    valid.zero_(); valid[:, [0, 1, 3, 4], :, :] = True
    result = probe_distance_gradients(logits, target, valid, parameters, edge_weight=weights,
                                     loss_options={'normalize_edge_weights': normalize})
    assert result['active_channels'] == {'short': 3, 'distance2': 1, 'distance4': 0}
    assert result['native_group_weights'] == {'short': .75, 'distance2': .25, 'distance4': 0.}
    assert result['cshort_group_weights'] == {'short': 3 / 3.5, 'distance2': .5 / 3.5, 'distance4': 0.}
    assert result['group_losses']['distance4'] is None
    for report in result['parameters'].values():
        assert report['group_norms']['distance4'] is None
        assert report['pairwise_cosines']['short__distance4'] is None
        assert report['pairwise_cosines']['distance2__distance4'] is None
        assert report['reconstruction']['passed']


def test_known_conflicting_shared_gradient_and_native_cancellation():
    parameter = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
    logits = parameter.expand(1, 8, 1, 1)
    target = torch.tensor([0.] * 4 + [1.] * 4, dtype=torch.float64).reshape_as(logits)
    result = probe_distance_gradients(logits, target, torch.ones_like(logits, dtype=torch.bool), {'shared': parameter})
    report = result['parameters']['shared']
    assert report['group_norms'] == {'short': .5, 'distance2': .5, 'distance4': .5}
    assert report['pairwise_cosines'] == {'short__distance2': -1., 'short__distance4': -1., 'distance2__distance4': 1.}
    assert report['native_mixture_norm'] is None
    assert report['native_vs_cshort_cosine'] is None
    assert report['native_short_projection'] is None
    assert report['cshort_short_cosine'] == 1.
    assert report['cshort_short_projection'] == pytest.approx(1.25 / 5.5)
    assert report['cshort_mixture_norm'] == pytest.approx(1.25 / 5.5)
    assert parameter.grad is None


def test_actual_fp32_shared_convolution_gradient_reconstruction():
    torch.manual_seed(19)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv2d(5, 8, 1))
    logits = model(torch.randn(2, 3, 12, 13))
    target = torch.randint(2, logits.shape).float()
    valid = torch.rand(logits.shape) > .3
    weights = .1 + torch.rand(logits.shape)
    result = probe_distance_gradients(logits, target, valid, {'shared.0.weight': model[0].weight},
                                     edge_weight=weights)
    assert result['parameters']['shared.0.weight']['reconstruction']['relative_l2_error'] < 2.e-6
    assert model[0].weight.grad is None


def test_unknown_relation_gradients_zero_and_unknown_values_do_not_change_report():
    _, logits, target, valid, weights, parameters = sample()
    valid[:, :, :, -1] = False
    first = probe_distance_gradients(logits, target, valid, parameters, edge_weight=weights)
    full, _ = balanced_affinity_loss(logits, target, valid, edge_weight=weights, **OPTIONS)
    gradient, = torch.autograd.grad(full, logits, retain_graph=True)
    assert torch.count_nonzero(gradient[~valid]) == 0
    changed = target.clone(); changed[~valid] = 17.
    second = probe_distance_gradients(logits, changed, valid, parameters, edge_weight=weights)
    assert first == second


def test_no_valid_relations_report_none_without_accumulating_or_fabricating_cosines():
    _, logits, target, valid, weights, parameters = sample(); valid.zero_()
    result = probe_distance_gradients(logits, target, valid, parameters, edge_weight=weights)
    assert result['total_active_channels'] == 0
    assert result['original_loss'] == result['reconstructed_native_loss'] == result['cshort_loss'] == 0.
    assert result['native_group_weights'] == result['cshort_group_weights'] == dict.fromkeys(['short', 'distance2', 'distance4'])
    for report in result['parameters'].values():
        assert all(value is None for value in report['group_norms'].values())
        assert all(value is None for value in report['pairwise_cosines'].values())
        assert report['native_mixture_norm'] is report['cshort_mixture_norm'] is None
        assert report['reconstruction']['direct_norm'] == 0. and report['reconstruction']['relative_l2_error'] is None
    assert all(parameter.grad is None for parameter in parameters.values())
    json.dumps(result, allow_nan=False)


def test_zero_weight_channels_keep_original_channel_denominator_but_zero_norm_is_none():
    _, logits, target, valid, weights, parameters = sample(); weights.zero_()
    result = probe_distance_gradients(logits, target, valid, parameters, edge_weight=weights)
    assert result['total_active_channels'] == 8
    assert result['native_group_weights'] == {'short': .5, 'distance2': .25, 'distance4': .25}
    assert result['original_loss'] == 0.
    assert all(all(value is None for value in report['group_norms'].values()) for report in result['parameters'].values())


def test_edge_weight_constant_is_cancelled_by_class_normalization_cshort_is_not():
    _, logits, target, valid, _, parameters = sample()
    channel_weights = torch.tensor([1.] * 4 + [.5] * 2 + [.25] * 2, dtype=torch.float64)[None, :, None, None].expand_as(logits)
    original, _ = balanced_affinity_loss(logits, target, valid, **OPTIONS)
    wrong, _ = balanced_affinity_loss(logits, target, valid, edge_weight=channel_weights, **OPTIONS)
    assert torch.isclose(original, wrong, rtol=1.e-12, atol=1.e-12)
    result = probe_distance_gradients(logits, target, valid, parameters)
    assert abs(result['cshort_loss'] - float(original.detach())) > 1.e-4
    assert any(report['native_vs_cshort_cosine'] < .999 for report in result['parameters'].values())


def test_disconnected_parameter_and_multiple_shared_parameters_report_separately():
    _, logits, target, valid, weights, parameters = sample()
    disconnected = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))
    parameters['disconnected'] = disconnected
    result = probe_distance_gradients(logits, target, valid, parameters, edge_weight=weights)
    report = result['parameters']['disconnected']
    assert all(value is None for value in report['group_norms'].values())
    assert report['native_vs_cshort_cosine'] is report['native_mixture_norm'] is None
    assert report['reconstruction']['error_norm'] == 0.
    assert len(result['parameters']) == 3
    assert all(parameter.grad is None for parameter in parameters.values())


@pytest.mark.parametrize('mutation', ['channels', 'logits_grad', 'target_shape', 'valid_type', 'weights_negative',
    'weights_nan', 'empty_parameters', 'duplicate_parameters', 'frozen_parameters', 'nonleaf',
    'r', 'h', 'gamma', 'normalize', 'unknown_option', 'bad_atol'])
def test_bad_shapes_options_and_nonshared_parameter_contracts_rejected(mutation):
    _, logits, target, valid, weights, parameters = sample()
    kwargs = {'edge_weight': weights}
    if mutation == 'channels': logits, target, valid, weights = (value[:, :4] for value in (logits, target, valid, weights)); kwargs['edge_weight'] = weights
    elif mutation == 'logits_grad': logits = logits.detach()
    elif mutation == 'target_shape': target = target[:, :4]
    elif mutation == 'valid_type': valid = valid.float()
    elif mutation == 'weights_negative': weights[0, 0, 0, 0] = -1.
    elif mutation == 'weights_nan': weights[0, 0, 0, 0] = float('nan')
    elif mutation == 'empty_parameters': parameters = {}
    elif mutation == 'duplicate_parameters': parameters['duplicate'] = next(iter(parameters.values()))
    elif mutation == 'frozen_parameters': next(iter(parameters.values())).requires_grad_(False)
    elif mutation == 'nonleaf': parameters = {'not_parameter': next(iter(parameters.values())) * 2.}
    elif mutation in ('r', 'h', 'gamma', 'normalize', 'unknown_option'):
        key = {'r': 'negative_weight', 'h': 'hard_negative_weight', 'gamma': 'hard_negative_gamma',
               'normalize': 'normalize_edge_weights', 'unknown_option': 'channel_weight'}[mutation]
        kwargs['loss_options'] = {key: 1.25}
    else: kwargs['atol'] = float('nan')
    with pytest.raises(ValueError): probe_distance_gradients(logits, target, valid, parameters, **kwargs)
