# -*- coding: utf-8 -*-
"""半决赛局部零噪声的真实调用隔离；仅使用合成图和轻量网络。"""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from models import backend_adaptation as backend
from tools import affinity_native_views as views
from utils import restoration_probe as probe
from utils.restoration_noise import NOISE_MODE


class RecordingRestorer(nn.Module):
    """保留首步采样接口，记录真正传入去噪器的状态。"""

    def __init__(self, correction=.05):
        super().__init__()
        self.register_buffer('correction', torch.tensor(correction))
        self.num_steps, self.kappa = 16, .03
        self.inference_noise_mode = NOISE_MODE
        self.samples = []

    def q_sample(self, x0, condition, step, *, noise=None, generator=None):
        assert noise is not None and generator is None
        state = condition + self.kappa * noise
        self.samples.append(dict(noise=noise.detach().clone(),
                                 state=state.detach().clone(),
                                 condition=condition.detach().clone(),
                                 step=step.detach().clone()))
        return state

    def denoise(self, state, condition, step):
        return state + self.correction


class ToySemantic(nn.Module):
    def forward(self, features, image):
        return {'semantic_logits': features[:, :1] - .5}


class ToyBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.semantic_decoder = ToySemantic()


@pytest.fixture
def scene(monkeypatch):
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
    global_calls = []
    fixed_calls = []

    def fixed_noise(shape, seed, device):
        fixed_calls.append((tuple(shape), seed))
        return torch.full(shape, .25, device=device, dtype=torch.float32)

    def forbidden_random_noise(*args, **kwargs):
        raise AssertionError('zero policy must never request a random noise field')

    def globals_from_old(model, path, config, device, restorer, seed):
        restored = backend.restore_first(restorer, image, seed)
        global_calls.append((restorer, seed, restored.detach().clone()))
        return rgb.copy(), restored[:, :1].clone(), restored[:, :1].clone()

    def tensor_from_rgb(value, device):
        return torch.from_numpy(value.copy()).permute(2, 0, 1)[None].float() / 255, 0, 0

    def decode(semantic, boundary, config):
        ids = np.ones(rgb.shape[:2], dtype=np.uint16)
        ids[:, 4:] = 2
        return ids, {}

    monkeypatch.setattr(backend, 'restoration_noise', fixed_noise)
    monkeypatch.setattr(probe, 'restoration_noise', forbidden_random_noise)
    monkeypatch.setattr(views, 'prediction_maps', globals_from_old)
    monkeypatch.setattr(views, 'prepare_image', lambda *args: (rgb, image.clone(), 0, 0))
    monkeypatch.setattr(views, 'tensor_from_rgb', tensor_from_rgb)
    monkeypatch.setattr(views, 'tile_boxes', lambda *args: [(0, 0, 6, 4), (0, 4, 6, 8)])
    monkeypatch.setattr(views, 'boundary_grid', lambda model, restored, config: restored[:, :1])
    monkeypatch.setattr(views, 'crop_letterbox_output', lambda tensor, *args: tensor)
    monkeypatch.setattr(views, 'decode_native', decode)
    return dict(model=ToyBackend(), old=RecordingRestorer(), image=image,
                global_calls=global_calls, fixed_calls=fixed_calls,
                config={'affinity_native': {'overlap': .25}})


def run_scene(scene, **kwargs):
    audit = {}
    _, result = views.predict_views(scene['model'], scene['old'], 'synthetic.png',
                                    scene['config'], 'cpu', 314191, [0, 1024],
                                    fixed_audit=audit, **kwargs)
    return result, audit


def assert_fixed_equal(first, second):
    for name in ('geometry_semantic_sha256', 'raw_probability_sha256'):
        assert first[name] == second[name]
    if 'global_boundary_sha256' in first:
        assert first['global_boundary_sha256'] == second['global_boundary_sha256']


def test_default_single_keeps_original_call_path_seed_and_outputs(scene):
    """新增参数不能把旧控制悄悄改为另一个噪声实现。"""
    initial = deepcopy(scene['old'].state_dict())
    default, a = run_scene(scene)
    explicit, b = run_scene(scene, local_restorer=scene['old'], local_noise_mode='single')
    expected_seeds = [314191, 1438191, 1438192]
    assert [seed for _, seed in scene['fixed_calls']] == expected_seeds * 2
    assert all(restorer is scene['old'] for restorer, _, _ in scene['global_calls'])
    assert_fixed_equal(a, b)
    for name in ('global', 'patch1024'):
        assert np.array_equal(default[name]['boundary'], explicit[name]['boundary'])
        assert np.array_equal(default[name]['instances'], explicit[name]['instances'])
        assert default[name]['classes'] == explicit[name]['classes']
    assert all(torch.equal(value, scene['old'].state_dict()[name]) for name, value in initial.items())


def test_zero_reaches_actual_sampler_only_for_local_tiles(scene):
    noisy, a = run_scene(scene)
    scene['old'].samples.clear()
    scene['fixed_calls'].clear()
    zero, b = run_scene(scene, local_noise_mode='zero')
    samples = scene['old'].samples
    assert len(samples) == 3  # 一次全图，两次局部；仍然仅做首步恢复。
    assert [seed for _, seed in scene['fixed_calls']] == [314191]
    assert torch.count_nonzero(samples[0]['noise']) > 0
    for sample in samples[1:]:
        assert sample['noise'].dtype == torch.float32
        assert torch.count_nonzero(sample['noise']) == 0
        assert torch.equal(sample['state'], sample['condition'])
        assert sample['step'].tolist() == [16]
    assert scene['old'].inference_noise_mode == NOISE_MODE
    assert_fixed_equal(a, b)
    assert np.array_equal(noisy['global']['boundary'], zero['global']['boundary'])
    assert not np.array_equal(noisy['patch1024']['boundary'], zero['patch1024']['boundary'])


def test_adapted_local_weights_cannot_enter_global_or_raw_semantic_path(scene):
    zero_old, a = run_scene(scene, local_noise_mode='zero')
    scene['old'].samples.clear()
    adapted = RecordingRestorer(correction=.15)
    zero_new, b = run_scene(scene, local_restorer=adapted, local_noise_mode='zero')
    assert len(scene['old'].samples) == 1 and len(adapted.samples) == 2
    assert all(restorer is scene['old'] for restorer, _, _ in scene['global_calls'])
    assert all(torch.count_nonzero(sample['noise']) == 0 for sample in adapted.samples)
    assert all(torch.equal(sample['state'], sample['condition']) for sample in adapted.samples)
    assert_fixed_equal(a, b)
    assert np.array_equal(zero_old['global']['boundary'], zero_new['global']['boundary'])
    assert not np.array_equal(zero_old['patch1024']['boundary'], zero_new['patch1024']['boundary'])


def test_local_zero_is_not_identity_and_does_not_depend_on_seed(scene):
    """零噪声保留真实重建网络，不能被实现成直接返回原图。"""
    first, raw_first = probe.restore_probe(scene['old'], scene['image'], 1, mode='zero')
    second, raw_second = probe.restore_probe(scene['old'], scene['image'], 999, mode='zero')
    assert torch.equal(first, second) and torch.equal(raw_first, raw_second)
    assert not torch.equal(first, scene['image'])
    expected = (scene['image'] + 1.0 * (raw_first - scene['image'])).clamp(0, 1)
    assert torch.equal(first, expected)
    assert scene['fixed_calls'] == []


def test_window_audit_records_actual_zero_state_and_matching_windows(scene):
    noisy_audit, zero_audit = [], []
    run_scene(scene, local_audit=noisy_audit)
    run_scene(scene, local_noise_mode='zero', local_audit=zero_audit)
    assert len(noisy_audit) == len(zero_audit) == 2
    for control, zero in zip(noisy_audit, zero_audit):
        for name in ('size', 'window', 'box', 'seed', 'input_sha256'):
            assert control[name] == zero[name]
        assert zero['q_sample_calls'] == 1 and zero['step'] == 16
        assert zero['noise_nonzero'] == 0 and zero['initial_max_abs'] == 0
        assert zero['initial_equals_condition'] is True
        assert zero['dtype'] == 'torch.float32'
        assert control['restored_sha256'] != zero['restored_sha256']


def test_window_audit_rejects_false_zero_state_even_when_requested_noise_is_zero(scene, monkeypatch):
    original_sample = scene['old'].q_sample

    def broken_sample(*args, **kwargs):
        return original_sample(*args, **kwargs) + .001

    monkeypatch.setattr(scene['old'], 'q_sample', broken_sample)
    with pytest.raises(RuntimeError, match='actual condition state'):
        run_scene(scene, local_noise_mode='zero', local_audit=[])


@pytest.mark.parametrize('bad_mode', ['pair', 'legacy_torch', '', None])
def test_unknown_local_policy_fails_before_any_model_work(scene, bad_mode):
    with pytest.raises(ValueError, match='local.*noise|single.*zero'):
        run_scene(scene, local_noise_mode=bad_mode)
    assert scene['global_calls'] == [] and scene['old'].samples == []
