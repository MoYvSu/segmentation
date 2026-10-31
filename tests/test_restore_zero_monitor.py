# -*- coding: utf-8 -*-
"""零噪声真实首步、固定带噪参照、有效区及无标签过程图的CPU检查。"""
import copy
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools import restore_consistency_monitor as base
from tools import restore_zero_monitor as zero
from utils.image_io import read_image


class FirstStepModel(torch.nn.Module):
    num_steps = 16

    def __init__(self, padding_delta=0.):
        super().__init__()
        self.register_buffer('eta', torch.linspace(0, 1, 17))
        self.calls = []
        self.padding_delta = padding_delta

    def q_sample(self, target, condition, step, noise):
        assert torch.equal(target, condition) and bool((step == 16).all())
        assert noise.dtype == torch.float32
        self.calls.append(noise.clone())
        return condition+.03*noise

    def denoise(self, state, condition, step):
        assert not torch.is_grad_enabled()
        value = .75*condition+.25*state
        if self.padding_delta:
            value = value.clone()
            value[..., -4:, :] += self.padding_delta
        return value


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _context():
    image = torch.linspace(.15, .85, 3*16*20).reshape(3, 16, 20)
    valid = torch.ones(1, 16, 20)
    valid[..., -4:, :] = 0
    pairs = []
    for position, condition in enumerate(base.CONDITIONS):
        pairs.append(dict(input=image.clone(), target=image.clone(), valid=valid.clone(),
                          receipt=dict(source='train_seen.png', condition=condition,
                                       noise_seeds=[71, 72], weak_reference_gradient_roi=[2, 3, 8, 8])))
    tests = [dict(input=image.clone(), valid=valid.clone(),
                  receipt=dict(source='test_unlabeled.png', has_ground_truth=False, noise_seeds=[81, 82]))]
    return dict(settings=dict(thumbnail_size=32), pairs=pairs, tests=tests,
                manifest=dict(purpose='seen_training_mechanism_diagnostic_and_unlabeled_test_monitor_only'))


def test_exact_zero_q_noise_and_inactive_seeds(tmp_path):
    context = _context()
    model = FirstStepModel().train()
    rng = torch.get_rng_state().clone()
    first = base.run_monitor(model, context, tmp_path/'first', 0, 0, 'cpu', noise_mode='zero')
    assert len(model.calls) == 8 and all(torch.count_nonzero(value) == 0 for value in model.calls)
    assert model.training and torch.equal(rng, torch.get_rng_state())
    assert first['noise_mode'] == 'zero'
    for item in context['pairs']+context['tests']:
        item['receipt']['noise_seeds'] = [123456, 876543]
    second = base.run_monitor(FirstStepModel(), context, tmp_path/'second', 1, 250, 'cpu', noise_mode='zero')
    assert first['groups'] == second['groups']
    assert all(row['noise_difference']['rgb_l1'] == 0 for row in second['training']+second['tests'])
    assert all(row['raw_noise_difference']['mse'] == 0 for row in second['training'])


def test_wrapper_keeps_original_noisy_outputs_and_global_rng(tmp_path):
    context = _context()
    original = copy.deepcopy(context)
    model = FirstStepModel().train()
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = zero.run_monitor(model, context, tmp_path/'both', 5, 1250, 'cpu')
    direct = base.run_monitor(FirstStepModel(), context, tmp_path/'direct', 5, 1250, 'cpu')
    assert result['noisy_view'] == direct
    assert model.training and torch.equal(torch_rng, torch.get_rng_state())
    assert random.getstate() == py_rng
    after_np = np.random.get_state()
    assert after_np[0] == np_rng[0] and np.array_equal(after_np[1], np_rng[1]) and after_np[2:] == np_rng[2:]
    assert all(torch.count_nonzero(noise) == 0 for noise in model.calls[:8])
    assert all(torch.count_nonzero(noise) > 0 for noise in model.calls[8:])
    assert not torch.equal(model.calls[8], model.calls[9])
    assert result['zero_view']['groups'] == result['groups']
    for before, after in zip(original['pairs']+original['tests'], context['pairs']+context['tests']):
        assert before['receipt'] == after['receipt']
        assert torch.equal(before['input'], after['input']) and torch.equal(before['valid'], after['valid'])
    for path in (tmp_path/'direct'/'monitor'/'e05').glob('*.png'):
        assert path.read_bytes() == (tmp_path/'both'/'noisy_view'/'monitor'/'e05'/path.name).read_bytes()
    gallery = tmp_path/'both'/'monitor'/'e05'
    assert len(list(gallery.glob('*.png'))) == 4
    assert read_image(gallery/'00_train_seen_identity.png').shape[:2] == (148, 128)
    assert read_image(gallery/'test_00_test_unlabeled.png').shape[:2] == (58, 96)
    assert (gallery/'metrics.json').is_file()


def test_both_modes_ignore_padding_and_test_has_no_target(tmp_path):
    context = _context()
    plain = zero.run_monitor(FirstStepModel(), context, tmp_path/'plain', 0, 0, 'cpu')
    padded = zero.run_monitor(FirstStepModel(padding_delta=10.), context, tmp_path/'padded', 0, 0, 'cpu')
    for mode in ('zero_view', 'noisy_view'):
        assert plain[mode]['groups'] == padded[mode]['groups']
        assert not plain[mode]['has_test_ground_truth']
        assert all('output_a_errors' not in row and 'target' not in row for row in plain[mode]['tests'])
        assert plain[mode]['tests'] == padded[mode]['tests']
    assert 'target' not in context['tests'][0]


def test_prepare_reuses_original_fixed_context_without_mutation(tmp_path):
    context = _context()
    config, dataset = {'unchanged': True}, object()
    with patch.object(base, 'prepare_monitor', return_value=context) as prepare:
        assert zero.prepare_monitor(config, dataset, tmp_path) is context
    prepare.assert_called_once_with(config, dataset, tmp_path)
    assert (tmp_path/'zero_monitor_protocol.json').is_file()


def test_unknown_noise_mode_fails_before_forward(tmp_path):
    model = FirstStepModel().train()
    with pytest.raises(ValueError, match='random or zero'):
        base.run_monitor(model, _context(), tmp_path, 0, 0, 'cpu', noise_mode='invalid')
    assert model.training and not model.calls
