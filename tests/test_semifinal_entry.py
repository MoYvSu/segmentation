# -*- coding: utf-8 -*-
"""半决赛入口的阶段、候选身份和局部模型隔离；不读真实数据或权重。"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tools import infer_semifinal as entry
from utils.config import load_config
from utils.restoration_noise import NOISE_MODE
from utils.submission_stage import dataset_contract


ROOT = Path(__file__).resolve().parents[1]


def candidate(name):
    return load_config(str(ROOT / 'config/inference' / f'semifinal_{name.lower()}.yaml'))


def sources(folder, names):
    folder.mkdir()
    for name in names:
        (folder / (name + '.png')).write_bytes(b'synthetic filename fixture; image is not decoded')
    return folder


def synthetic_models(tmp_path, config):
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    config = deepcopy(config)
    for name, spec in config['submission']['models'].items():
        if spec is None:
            continue
        path = model_dir / spec['file']
        path.write_bytes(('synthetic ' + name).encode('ascii'))
        spec['sha256'] = entry.sha(path)
    return config, model_dir


def test_default_entry_retains_old_full_names_indices_and_seeds(tmp_path):
    original_names = tuple(f'test_{number:03d}' for number in range(69, 169))
    assert entry.FULL_NAMES == original_names
    folder = sources(tmp_path / 'old', reversed(original_names))
    assert tuple(path.stem for path in entry.input_files(folder)) == original_names
    assert [(entry.source_index(name), entry.source_seed(name)) for name in
            ('test_069', 'test_101', 'test_168')] == [(0, 314159), (32, 314191), (99, 314258)]
    old = load_config(str(ROOT / 'config/inference/submission_reflect.yaml'))
    entry.validate_config(old)
    assert entry.local_restoration(old) == {'mode': 'single', 'model': 'restorer', 'candidate': 'N0'}


def test_150_input_full_and_subset_keep_stage_global_seed_index(tmp_path):
    stage = 'semifinal_150'
    names = dataset_contract(stage).names
    folder = sources(tmp_path / 'new', reversed(names))
    full = entry.input_files(folder, dataset=stage)
    subset = entry.input_files(folder, ['test_318', 'test_201', 'test_169'], stage)
    assert len(full) == 150
    assert [path.stem for path in subset] == ['test_169', 'test_201', 'test_318']
    assert [entry.source_seed(path.stem, stage) for path in subset] == [314159, 314191, 314308]
    full_seeds = {path.stem: entry.source_seed(path.stem, stage) for path in full}
    assert all(entry.source_seed(path.stem, stage) == full_seeds[path.stem] for path in subset)
    with pytest.raises(ValueError, match='unexpected'):
        entry.input_files(folder)  # 不按150文件自动猜成新赛段。


def test_subset_directory_does_not_need_all_150_sources_but_full_mode_does(tmp_path):
    folder = sources(tmp_path / 'small', ['test_318', 'test_201'])
    paths = entry.input_files(folder, ['test_318'], 'semifinal_150')
    assert [path.stem for path in paths] == ['test_318']
    assert entry.source_seed(paths[0].stem, 'semifinal_150') == 314308
    with pytest.raises(ValueError, match='exactly 150'):
        entry.input_files(folder, dataset='semifinal_150')


@pytest.mark.parametrize('old_name', ['test_069', 'test_168'])
def test_old_stage_contamination_is_rejected_even_when_only_selects_new_image(tmp_path, old_name):
    folder = sources(tmp_path / 'mixed', ['test_169', old_name])
    with pytest.raises(ValueError, match='unexpected'):
        entry.input_files(folder, ['test_169'], 'semifinal_150')


def test_duplicate_extension_and_invalid_subset_are_rejected(tmp_path):
    folder = sources(tmp_path / 'images', ['test_169'])
    for only in ([], ['test_169', 'test_169'], ['test_170']):
        with pytest.raises(ValueError, match='distinct available'):
            entry.input_files(folder, only, 'semifinal_150')
    (folder / 'test_169.jpg').write_bytes(b'duplicate stem')
    with pytest.raises(ValueError, match='Duplicate'):
        entry.input_files(folder, ['test_169'], 'semifinal_150')


@pytest.mark.parametrize('name,mode,model', [
    ('N0', 'single', 'restorer'), ('N1', 'zero', 'restorer'), ('N2', 'zero', 'local_restorer'),
])
def test_real_candidate_configs_validate_and_keep_original_global_model(name, mode, model):
    config = candidate(name)
    entry.validate_config(config)
    assert config['submission']['dataset'] == 'semifinal_150'
    assert config['submission']['restoration_noise'] == NOISE_MODE
    assert config['submission']['models']['terminal_semantic'] is None
    assert entry.local_restoration(config) == dict(mode=mode, model=model, candidate=name)
    control = candidate('N0')
    for key in ('backend', 'restorer'):
        assert config['submission']['models'][key] == control['submission']['models'][key]
    for key in ('sam2', 'affinity_native', 'affinity_deployment', 'inference'):
        assert config[key] == control[key]


def test_n1_and_n2_have_only_the_declared_configuration_differences():
    n0, n1, n2 = (candidate(name) for name in ('N0', 'N1', 'N2'))
    n1['submission']['local_restoration']['mode'] = 'single'
    assert n1 == n0
    n2['submission']['models'].pop('local_restorer')
    n2['submission']['local_restoration']['model'] = 'restorer'
    assert n2 == candidate('N1')


@pytest.mark.parametrize('name', ['N0', 'N1'])
def test_shared_candidates_reject_an_unused_local_model(name):
    config = candidate(name)
    config['submission']['models']['local_restorer'] = candidate('N2')['submission']['models']['local_restorer']
    with pytest.raises(ValueError, match='Unused local_restorer'):
        entry.validate_config(config)


def test_n2_requires_separate_model_and_global_zero_is_rejected():
    config = candidate('N2')
    config['submission']['models'].pop('local_restorer')
    with pytest.raises(ValueError, match='separately identified'):
        entry.validate_config(config)
    config = candidate('N1')
    config['submission']['restoration_noise'] = 'zero'
    with pytest.raises(ValueError, match='explicit restoration_noise'):
        entry.validate_config(config)


@pytest.mark.parametrize('policy', [None, {}, {'mode': 'zero'},
    {'mode': 'pair', 'model': 'restorer'}, {'mode': 'single', 'model': 'local_restorer'},
    {'mode': 'zero', 'model': 'restorer', 'strength': 0},
])
def test_invalid_or_extra_local_policy_does_not_silently_change_recipe(policy):
    config = candidate('N0')
    config['submission']['local_restoration'] = policy
    with pytest.raises(ValueError):
        entry.validate_config(config)


@pytest.mark.parametrize('name,expected', [('N0', {'backend', 'restorer'}),
                                        ('N1', {'backend', 'restorer'}),
                                        ('N2', {'backend', 'restorer', 'local_restorer'})])
def test_model_paths_bind_each_candidate_to_its_exact_files_and_hashes(tmp_path, name, expected):
    config, folder = synthetic_models(tmp_path, candidate(name))
    paths = entry.model_paths(config, folder)
    assert set(paths) == expected and all(path.parent == folder for path in paths.values())
    key = 'local_restorer' if name == 'N2' else 'restorer'
    paths[key].write_bytes(b'wrong checkpoint bytes')
    with pytest.raises(RuntimeError, match='SHA differs'):
        entry.model_paths(config, folder)


def test_local_model_cannot_escape_the_model_directory(tmp_path):
    config, folder = synthetic_models(tmp_path, candidate('N2'))
    outside = tmp_path / 'outside.pt'
    outside.write_bytes(b'outside')
    config['submission']['models']['local_restorer'] = {'file': '../outside.pt', 'sha256': entry.sha(outside)}
    with pytest.raises(ValueError, match='inside --model-dir'):
        entry.model_paths(config, folder)


@pytest.mark.parametrize('name', ['N0', 'N1', 'N2'])
def test_loading_local_candidate_never_replaces_global_restorer(monkeypatch, name):
    config = candidate(name)
    model = nn.Linear(1, 1)
    old, local = nn.Linear(1, 1), nn.Linear(1, 1)
    for restorer in (old, local):
        restorer.num_steps, restorer.kappa = 16, .03
    calls = []

    def load(path, device):
        calls.append(path)
        return old if path == 'old.pt' else local

    monkeypatch.setattr(entry, 'load_backend', lambda *args: (model, {}))
    monkeypatch.setattr(entry, 'load_rgb_restorer', load)
    paths = {'backend': Path('backend.pt'), 'restorer': Path('old.pt')}
    if name == 'N2':
        paths['local_restorer'] = Path('c5.pt')
    got_model, got_old, terminal = entry.load_models(config, paths, 'cpu')
    got_local = entry.load_local_restorer(config, paths, 'cpu')
    assert got_model is model and got_old is old and terminal is None
    assert got_local is (local if name == 'N2' else None)
    assert calls == (['old.pt', 'c5.pt'] if name == 'N2' else ['old.pt'])
    assert old.inference_noise_mode == NOISE_MODE
    for value in (model, old) + ((local,) if name == 'N2' else ()):
        assert not value.training and all(not p.requires_grad for p in value.parameters())


@pytest.mark.parametrize('name', ['N0', 'N1', 'N2'])
def test_predict_dispatch_keeps_global_restorer_and_stage_seed(monkeypatch, name):
    config = candidate(name)
    model, old, local = object(), object(), object()
    calls = []
    ids = np.ones((4, 4), dtype=np.uint16)

    def fake_views(*args, **kwargs):
        calls.append((args, kwargs))
        return np.zeros((4, 4, 3), dtype=np.uint8), {'patch1024': {'instances': ids, 'classes': {'1': 1}}}

    monkeypatch.setattr(entry.views, 'predict_views', fake_views)
    supplied = local if name == 'N2' else None
    entry.predict(model, old, None, Path('test_318.png'), config, 'cpu', local_restorer=supplied)
    args, kwargs = calls[0]
    assert args[0] is model and args[1] is old and args[5] == 314308 and args[6] == [1024]
    assert kwargs['local_restorer'] is supplied
    assert kwargs['local_noise_mode'] == ('single' if name == 'N0' else 'zero')
    with pytest.raises(ValueError, match='declared candidate'):
        entry.predict(model, old, None, Path('test_318.png'), config, 'cpu',
                      local_restorer=None if name == 'N2' else local)
    assert len(calls) == 1


def test_preview_keeps_rgb_order_and_does_not_modify_model_output(monkeypatch):
    monkeypatch.setattr(entry, 'crop_letterbox_output', lambda tensor, *args: tensor)
    preview = entry.RestorationPreview()
    values = [torch.tensor([.1, .2, .3]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone(),
              torch.tensor([.7, .8, .9]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone()]
    before = [value.clone() for value in values]
    for box, value in zip(((0, 0, 4, 4), (0, 4, 4, 8)), values):
        preview(source_shape=(4, 8), box=box, restored=value, ph=0, pw=0)
    rgb = preview.finish((4, 8))
    assert rgb.shape == (4, 8, 3)
    np.testing.assert_allclose(rgb[:, :4], np.broadcast_to([.1, .2, .3], (4, 4, 3)), atol=1e-7)
    np.testing.assert_allclose(rgb[:, 4:], np.broadcast_to([.7, .8, .9], (4, 4, 3)), atol=1e-7)
    assert all(torch.equal(old, new) for old, new in zip(before, values))


def test_runtime_source_receipt_includes_new_dependencies_and_original_noise_resources():
    receipt = entry.local_module_sources()
    expected = ('utils/submission_stage.py', 'utils/restoration_probe.py',
                'utils/restoration_noise.py', 'utils/cuda/restore_noise.cu', 'utils/cuda/restore_noise.ptx')
    assert all(receipt[path] == entry.sha(ROOT / path) for path in expected)
