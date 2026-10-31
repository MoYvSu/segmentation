# -*- coding: utf-8 -*-
"""跨显卡噪声策略：历史默认不变、首步显式噪声及正式入口隔离。"""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from models import backend_adaptation as backend
import tools.infer_semifinal as standalone
from utils.config import load_config
from utils.restoration_noise import (
    LEGACY_MODE, NOISE_MODE, configure_restoration_noise, noise_protocol,
)


class ProbeRestorer(nn.Module):
    """只替换去噪网络，保留噪声参数与最终混合方式的可观察契约。"""

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.tensor(1.0))
        self.num_steps, self.kappa = 16, .03
        self.samples = []

    def q_sample(self, x0, condition, step, *, noise=None, generator=None):
        self.samples.append(dict(noise=noise, generator=generator, step=step))
        if noise is None:
            noise = torch.randn(x0.shape, dtype=torch.float32, device=x0.device,
                                generator=generator)
        return condition + self.kappa * noise

    def denoise(self, state, condition, step):
        return state


def test_legacy_default_and_explicit_mode_keep_original_rng_and_output():
    image = torch.full((1, 3, 4, 7), .5)
    before = torch.get_rng_state().clone()
    expected_noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(71))
    expected = (image + .03 * expected_noise).clamp(0, 1)
    for explicit in (False, True):
        restorer = ProbeRestorer()
        if explicit:
            configure_restoration_noise(restorer, LEGACY_MODE)
        actual = backend.restore_first(restorer, image, 71)
        assert torch.equal(actual, expected)
        assert restorer.samples[0]['noise'] is None
        assert restorer.samples[0]['generator'].initial_seed() == 71
    assert torch.equal(before, torch.get_rng_state())


def test_explicit_fixed_noise_reaches_full_image_and_tile_import_aliases(monkeypatch):
    from tools.affinity_native_views import restore_first as tile_restore
    from train_backend_adaptation import restore_first as whole_restore

    image = torch.full((1, 3, 4, 7), .5, dtype=torch.float16)
    fixed = torch.full(image.shape, .2, dtype=torch.float32)
    calls = []

    def fixed_noise(shape, seed, device):
        calls.append((tuple(shape), seed, device))
        return fixed

    monkeypatch.setattr(backend, 'restoration_noise', fixed_noise)
    restorer = ProbeRestorer()
    before = {name: value.clone() for name, value in restorer.state_dict().items()}
    configure_restoration_noise(restorer, NOISE_MODE)
    assert set(restorer.state_dict()) == set(before)
    assert tile_restore is backend.restore_first and whole_restore is backend.restore_first
    for function, seed in ((whole_restore, 314191), (tile_restore, 1438191)):
        actual = function(restorer, image, seed)
        assert torch.equal(actual, (image.float() + .03 * fixed).clamp(0, 1))
        sample = restorer.samples[-1]
        assert sample['noise'] is fixed and sample['generator'] is None
        assert sample['step'].tolist() == [16]
        assert actual.dtype == torch.float32 and not actual.requires_grad
    assert calls == [(image.shape, 314191, image.device),
                     (image.shape, 1438191, image.device)]
    assert all(torch.equal(value, before[name]) for name, value in restorer.state_dict().items())


def test_noise_failure_never_falls_back_to_hardware_rng(monkeypatch):
    restorer = ProbeRestorer()
    configure_restoration_noise(restorer, NOISE_MODE)

    def failure(*args):
        raise RuntimeError('CUDA driver unavailable')

    monkeypatch.setattr(backend, 'restoration_noise', failure)
    with pytest.raises(RuntimeError, match='CUDA driver unavailable'):
        backend.restore_first(restorer, torch.zeros(1, 3, 2, 2), 1)
    assert not restorer.samples


def test_unknown_runtime_policy_is_rejected_before_sampling():
    restorer = ProbeRestorer()
    restorer.inference_noise_mode = 'typo'
    with pytest.raises(ValueError, match='Unknown restoration noise mode'):
        backend.restore_first(restorer, torch.zeros(1, 3, 2, 2), 1)
    assert not restorer.samples


@pytest.mark.parametrize('mode', [None, LEGACY_MODE, 'cuda_philox4090_v2', {}])
def test_formal_entry_requires_exact_explicit_noise_version(mode):
    config = load_config('config/inference/submission_reflect.yaml')
    changed = deepcopy(config)
    if mode is None:
        changed['submission'].pop('restoration_noise')
    else:
        changed['submission']['restoration_noise'] = mode
    with pytest.raises(ValueError, match='explicit restoration_noise'):
        standalone.validate_config(changed)
    standalone.validate_config(config)


def test_formal_model_loader_sets_policy_without_changing_model_state(monkeypatch):
    model = nn.Linear(1, 1)
    restorer = ProbeRestorer()
    before = {name: value.clone() for name, value in restorer.state_dict().items()}
    monkeypatch.setattr(standalone, 'load_backend', lambda *args: (model, {}))
    monkeypatch.setattr(standalone, 'load_rgb_restorer', lambda *args: restorer)
    config = {'submission': {'restoration_noise': NOISE_MODE}}
    actual_model, actual_restorer, terminal = standalone.load_models(
        config, {'backend': Path('backend.pt'), 'restorer': Path('restorer.pt')}, 'cpu')
    assert actual_model is model and actual_restorer is restorer and terminal is None
    assert restorer.inference_noise_mode == NOISE_MODE
    assert not restorer.training and all(not p.requires_grad for p in model.parameters())
    assert all(torch.equal(value, before[name]) for name, value in restorer.state_dict().items())


def test_runtime_receipt_records_actual_noise_policy(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    fixed = standalone.runtime_record(restoration_noise_mode=NOISE_MODE)
    legacy = standalone.runtime_record(restoration_noise_mode=LEGACY_MODE)
    assert fixed['restoration_noise'] == noise_protocol(NOISE_MODE)
    assert legacy['restoration_noise'] == noise_protocol(LEGACY_MODE)
    assert fixed != legacy


def test_runtime_sources_include_both_noise_resources_and_reject_missing(tmp_path, monkeypatch):
    entry = tmp_path / 'tools/infer_semifinal.py'
    entry.parent.mkdir()
    entry.write_text('# synthetic entry', encoding='utf8')
    resource = tmp_path / 'utils/cuda'
    resource.mkdir(parents=True)
    source = resource / 'restore_noise.cu'
    ptx = resource / 'restore_noise.ptx'
    source.write_text('// synthetic CUDA source', encoding='utf8')
    ptx.write_text('// synthetic PTX', encoding='utf8')
    monkeypatch.setattr(standalone, 'ROOT', tmp_path)
    monkeypatch.setattr(standalone, '__file__', str(entry))
    before = standalone.local_module_sources()
    assert before['utils/cuda/restore_noise.cu'] == standalone.sha(source)
    assert before['utils/cuda/restore_noise.ptx'] == standalone.sha(ptx)
    ptx.write_text('// changed synthetic PTX', encoding='utf8')
    assert standalone.local_module_sources() != before
    ptx.unlink()
    with pytest.raises(RuntimeError, match='Required restoration noise resource missing'):
        standalone.local_module_sources()
