# -*- coding: utf-8 -*-
"""固定配对、真实首步、有效区及失败行为的CPU检查。"""
import copy
import random
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tools.restore_consistency_monitor import prepare_monitor, run_monitor
from train_rgb_restoration import build_datasets
from utils.config import load_config


class FirstStepModel(torch.nn.Module):
    num_steps = 16

    def __init__(self, *, fail=False, padding_delta=0.):
        super().__init__()
        self.register_buffer('eta', torch.linspace(0, 1, 17))
        self.fail = fail
        self.padding_delta = padding_delta
        self.calls = []

    def q_sample(self, target, condition, step, noise):
        assert torch.equal(target, condition)
        assert torch.all(step == 16)
        assert noise.dtype == torch.float32
        self.calls.append(noise.detach().cpu().clone())
        return condition + .03*noise

    def denoise(self, state, condition, step):
        assert not torch.is_grad_enabled()
        assert state.dtype == condition.dtype == torch.float32
        if self.fail:
            return state*float('nan')
        result = condition + .2*(state-condition)
        if self.padding_delta:
            result = result.clone()
            result[..., -4:, :] += self.padding_delta
        return result


def _setup(tmp_path, *, with_test=False):
    directory = tmp_path/'sources'
    directory.mkdir()
    yy, xx = np.mgrid[:48, :64]
    for index in range(2):
        image = np.stack(((xx*4+index*17)%256, (yy*5+index*11)%256,
                          ((xx//5+yy//7)%2)*190), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(directory/f'img_{index}.png'), image)
    config = load_config('config/train/rgb_diffusion_d5a.yaml')
    config['rgb_restoration'].update(data_dir=str(directory), image_size=48, crop_size=32,
                                     expected_train_sources=2, masked_pretraining=None)
    settings = dict(samples=2, seed=741, thumbnail_size=48, roi_size=12)
    if with_test:
        settings.update(test_dir=str(directory), test_images=['img_0'])
    config['restore_consistency'] = dict(monitor=settings, data_epoch_offset=60)
    dataset, holdout = build_datasets(config)
    assert holdout is None
    return config, dataset


def test_fixed_pairs_match_geometry_and_preserve_dataset_and_global_rng(tmp_path):
    config, dataset = _setup(tmp_path)
    original = copy.deepcopy(dataset.__dict__)
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    first = prepare_monitor(config, dataset, tmp_path/'one')
    second = prepare_monitor(config, dataset, tmp_path/'two')
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert dataset.epoch == original['epoch'] == 0
    assert dataset.seed == original['seed']
    assert dataset.degradation == original['degradation']
    assert np.array_equal(dataset.probabilities, original['probabilities'])
    assert first['manifest'] == second['manifest']
    assert first['manifest']['source_epoch'] == 60
    assert len(first['pairs']) == 6
    for left, right in zip(first['pairs'], second['pairs']):
        assert all(torch.equal(left[key], right[key]) for key in ('input', 'target', 'valid'))
    for index in (0, 3):
        identity, uniform, endpoint = first['pairs'][index:index+3]
        assert torch.equal(identity['input'], identity['target'])
        assert torch.equal(identity['target'], uniform['target'])
        assert torch.equal(uniform['target'], endpoint['target'])
        assert torch.equal(uniform['valid'], endpoint['valid'])
        assert endpoint['receipt']['degradation']['endpoint_blur_applied']
        assert endpoint['receipt']['degradation']['sigma_crop_coexist']
        assert identity['receipt']['weak_reference_gradient_roi'] == endpoint['receipt']['weak_reference_gradient_roi']
    stored = torch.load(tmp_path/'one'/'monitor_fixed.pt', weights_only=True)
    assert stored['manifest'] == first['manifest']


def test_actual_first_step_two_independent_noise_and_state_restoration(tmp_path):
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config, dataset = _setup(tmp_path)
        context = prepare_monitor(config, dataset, tmp_path/'fixed')
        model = FirstStepModel().train()
        state = torch.get_rng_state().clone()
        first = run_monitor(model, context, tmp_path/'run', 0, 0, 'cpu')
        assert model.training and torch.equal(state, torch.get_rng_state())
        assert len(model.calls) == 12
        assert not torch.equal(model.calls[0], model.calls[1])
        # 同一图的三种退化用同一对噪声，防止退化差异与噪声差异混在一起。
        assert torch.equal(model.calls[0], model.calls[2])
        second = run_monitor(model, context, tmp_path/'run', 1, 250, torch.device('cpu'))
        assert first['training'] == second['training'] and first['groups'] == second['groups']
        assert first['actual_step'] == 16 and first['fp32']
        assert all(row['noise_difference']['rgb_l1'] > 0 for row in first['training'])
        assert first['groups']['identity']['input_errors']['rgb_l1'] == 0
        assert all(row['clear_input_drift'] is not None for row in first['training'] if row['condition'] == 'identity')
        pngs = list((tmp_path/'run'/'monitor'/'e00').glob('*.png'))
        assert len(pngs) == 6
        assert all(cv2.imread(str(path)).shape[:2] == (180, 240) for path in pngs)
        assert (tmp_path/'run'/'monitor'/'e01'/'metrics.json').is_file()
    finally:
        torch.set_num_threads(before_threads)


def test_monitor_metrics_ignore_padding_and_its_gradient_seam(tmp_path):
    config, dataset = _setup(tmp_path)
    context = prepare_monitor(config, dataset, tmp_path/'fixed')
    for item in context['pairs']:
        item['valid'][..., -4:, :] = 0
    plain = run_monitor(FirstStepModel(), context, tmp_path/'plain', 0, 0, 'cpu')
    changed = run_monitor(FirstStepModel(padding_delta=10.), context, tmp_path/'padding', 0, 0, 'cpu')
    assert plain['groups'] == changed['groups']
    for left, right in zip(plain['training'], changed['training']):
        assert left['raw_noise_difference'] == right['raw_noise_difference']


def test_unlabeled_test_monitor_has_no_target_and_uses_letterbox1024(tmp_path):
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config, dataset = _setup(tmp_path, with_test=True)
        context = prepare_monitor(config, dataset, tmp_path/'fixed')
        assert len(context['tests']) == 1
        assert 'target' not in context['tests'][0]
        assert context['tests'][0]['input'].shape == (3, 1024, 1024)
        assert not context['tests'][0]['receipt']['has_ground_truth']
        result = run_monitor(FirstStepModel().eval(), context, tmp_path/'run', 5, 1250, 'cpu')
        assert not result['has_test_ground_truth']
        assert len(result['tests']) == 1
        assert 'output_a_errors' not in result['tests'][0]
    finally:
        torch.set_num_threads(before_threads)


@pytest.mark.parametrize('failure', ['nonfinite', 'write', 'schedule'])
def test_failures_propagate_and_model_training_state_is_restored(tmp_path, failure):
    config, dataset = _setup(tmp_path)
    context = prepare_monitor(config, dataset, tmp_path/'fixed')
    model = FirstStepModel(fail=failure == 'nonfinite').train()
    if failure == 'schedule':
        model.num_steps = 8
        with pytest.raises(ValueError, match='T16'):
            run_monitor(model, context, tmp_path/'run', 0, 0, 'cpu')
    elif failure == 'write':
        with patch('tools.restore_consistency_monitor.write_image', side_effect=OSError('disk full')):
            with pytest.raises(OSError, match='disk full'):
                run_monitor(model, context, tmp_path/'run', 0, 0, 'cpu')
    else:
        with pytest.raises(FloatingPointError, match='non-finite'):
            run_monitor(model, context, tmp_path/'run', 0, 0, 'cpu')
    assert model.training
