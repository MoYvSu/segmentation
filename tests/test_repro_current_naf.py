# -*- coding: utf-8 -*-
"""新链NAF适配CPU合同；不构造SAM2、不训练、不使用赛题数据。"""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from repro import current_naf as current
from utils.patch_diagnostic import BlendMap, tile_boxes


@pytest.fixture(autouse=True)
def preserve_CPU_runtime():
    state = (torch.get_num_threads(), cv2.getNumThreads(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    torch.set_num_threads(state[0]); cv2.setNumThreads(state[1])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = state[2:4]
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = state[4:]
    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])


def fixture_context(tmp_path):
    raw, completed, run = (tmp_path / name for name in ('raw', 'gt', 'run'))
    for path in (raw, completed, run):
        path.mkdir()
    names = [f'train_{i:03d}' for i in range(32)]
    for name in names:
        Image.fromarray(np.full((20, 24, 3), 127, np.uint8)).save(raw / (name + '.png'))
        (raw / (name + '.json')).write_text('{}', encoding='utf-8')
        ids = np.ones((20, 24), np.uint16); ids[:, 12:] = 2
        np.savez_compressed(completed / (name + '_gt.npz'), instance_map=ids,
            original_covered=np.ones(ids.shape, bool), filled=np.zeros(ids.shape, bool),
            residual_unknown=np.zeros(ids.shape, bool))
        (completed / (name + '_class.json')).write_text('{"1":0,"2":1}', encoding='utf-8')
    parent = completed / 'manifest.json'
    parent.write_text(json.dumps(dict(format='manual_target_v2', samples=names)), encoding='utf-8')
    artifacts = {name: str((run / name).resolve()) for name in ('n1_cache', 'naf', 'backend', 'restorer')}
    artifacts['naf_export'] = str((run / 'export/model/naf.pt').resolve())
    for name in ('backend', 'restorer'):
        Path(artifacts[name]).write_bytes(('new_' + name).encode())
    return dict(format='current_training_context_v1', project_root=str(current.ROOT), run_dir=str(run),
        settings={}, assets=dict(raw_dir=str(raw), completed_gt_dir=str(completed),
            completed_gt_manifest=str(parent), naf_target_manifest=None, sam2_repo=str(tmp_path / 'sam2')),
        artifacts=artifacts)


def add_cache(context):
    path, targets = current.prepare_targets(context)
    directory = Path(context['artifacts']['n1_cache']); directory.mkdir()
    files = []
    for row in targets['rows']:
        heat = directory / (row['source'] + '.npz')
        np.savez_compressed(heat, boundary=np.full(row['shape'], .25, np.float32))
        files.append(dict(source=row['source'], heat_path=heat.name, heat_sha256=current.file_sha(heat),
                          shape=row['shape'], **row['hashes']))
    value = dict(format='boundary_naf_real_heatmaps_v1', complete=True, no_test_training=True,
        frozen_models_unchanged=True, local_restoration=dict(mode='zero', model='restorer'),
        models=current.model_files(context), processed_gt_manifest_sha256=current.file_sha(path), files=files)
    current.write(directory / 'manifest.json', value)
    return value


def test_GT_wrapper_is_lossless_32source_and_idempotent(tmp_path):
    context = fixture_context(tmp_path)
    gt = Path(context['assets']['completed_gt_dir']) / 'train_000_gt.npz'
    before = current.file_sha(gt)
    path, value = current.prepare_targets(context)
    assert value['GT_arrays_modified'] is False and len(value['rows']) == 32
    assert current.file_sha(gt) == before
    assert value['rows'][0]['hashes']['gt_sha256'] == before
    assert current.prepare_targets(context)[0] == path
    with np.load(gt) as source:
        assert source['original_covered'].dtype == bool


def test_target_cohort_and_existing_wrapper_hash_reject_change(tmp_path):
    context = fixture_context(tmp_path)
    current.prepare_targets(context)
    p = Path(context['assets']['completed_gt_dir']) / 'train_000_class.json'
    p.write_text('{"1":1,"2":1}', encoding='utf-8')
    with pytest.raises(ValueError, match='已有阶段配置'):
        current.prepare_targets(context)


def test_provided_NAF_manifest_relocated_by_content_not_old_absolute_path(tmp_path):
    context = fixture_context(tmp_path)
    path, value = current.prepare_targets(context)
    path.unlink()
    provided = tmp_path / 'old_wrapper.json'
    old = deepcopy(value)
    for row in old['rows']:
        row['image'] = '/old_host/no_longer_present.png'
    current.write(provided, old)
    context['assets']['naf_target_manifest'] = str(provided)
    _, value = current.prepare_targets(context)
    assert Path(value['rows'][0]['image']).is_file()
    old['rows'][0]['hashes']['gt_sha256'] = 'f' * 64
    current.write(provided, old)
    with pytest.raises(ValueError, match='内容不同'):
        current.prepare_targets(context)


def test_native_N1_zero_path_and_full_probability_stitch():
    rgb = np.arange(6*7*3, dtype=np.uint8).reshape(6, 7, 3)
    calls = []
    def tensor(image, device):
        return torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float()/255, 0, 0
    def restore(model, image, seed, mode, audit):
        calls.append((seed, mode))
        audit.update(q_sample_calls=1, noise_nonzero=0, initial_equals_condition=True,
                     initial_max_abs=0., step=16)
        return image, image
    ops = SimpleNamespace(np=np, torch=torch, model=None, restorer=None,
        BlendMap=BlendMap, tile_boxes=lambda shape, size, overlap: tile_boxes(shape, 4, overlap),
        inference_state=lambda *args: nullcontext(), tensor_from_rgb=tensor, restore_probe=restore,
        boundary_grid=lambda model, image, config: image[:, :1],
        crop_letterbox_output=lambda value, *args: value)
    result, windows, blend = current.capture_native_boundary(rgb, {}, ops, device='cpu', source_seed=17)
    np.testing.assert_allclose(result, rgb[..., 0].astype(np.float32)/255, rtol=0, atol=6e-8)
    assert result.dtype == np.float32 and blend['covered_fraction'] == 1
    assert len(windows) == len(calls) == len(tile_boxes(rgb.shape[:2], 4, .25))
    assert all(mode == 'zero' for _, mode in calls)
    assert calls[0][0] == 17 + 100000 + 1024*1000
    def invalid(*args, **kwargs):
        value = restore(*args, **kwargs); kwargs['audit']['noise_nonzero'] = 1; return value
    ops.restore_probe = invalid
    with pytest.raises(RuntimeError, match='零噪声'):
        current.capture_native_boundary(rgb, {}, ops, device='cpu', source_seed=17)


def test_dynamic_models_and_original_width_recipe(tmp_path):
    context = fixture_context(tmp_path); add_cache(context)
    context['settings']['stages'] = dict(naf_train=dict(config='config/train/boundary_width.yaml'))
    config = current.training_config(context); cfg = config['boundary_naf']; t = cfg['training']
    assert cfg['real_input_models']['backend']['sha256'] == current.file_sha(context['artifacts']['backend'])
    assert cfg['real_input_models']['restorer']['sha256'] == current.file_sha(context['artifacts']['restorer'])
    assert (t['main_updates'], t['main_schedule_updates'], t['effective_batch_size']) == (1000, 53000, 16)
    assert (t['precision'], t['adversarial_weight'], t['adversarial_warmup_updates']) == ('float32', .01, 64)
    assert cfg['input_normalization'] == cfg['inference']['input_normalization'] == dict(mode='skeleton_dilate', kernel_size=8)
    assert cfg['monitor']['full_readout_steps'] == [] and cfg['monitor']['control_manifests'] == []
    from train_boundary_rgb import rates, dataset_for
    assert rates(999, cfg)[0] > .00099  # 53000分母，不能错误地在1000步降到最低。
    dataset = dataset_for(config, 1)
    assert len(dataset) == 16000 and set(dataset.sources) == {f'train_{i:03d}' for i in range(32)}
    sample = dataset[0]
    assert sample['input_boundary'].shape == (1, 256, 256)
    assert sample['input_boundary'].dtype == torch.float32


@pytest.mark.parametrize('which', ['model', 'heat', 'missing', 'budget'])
def test_cache_or_recipe_identity_changes_rejected(tmp_path, which):
    context = fixture_context(tmp_path); add_cache(context)
    if which == 'model':
        Path(context['artifacts']['backend']).write_bytes(b'changed_new_model')
    elif which == 'heat':
        Path(context['artifacts']['n1_cache'], 'train_000.npz').write_bytes(b'changed')
    elif which == 'missing':
        p = Path(context['artifacts']['n1_cache'], 'manifest.json'); x = current.read(p)
        x['files'].pop(); current.write(p, x)
    else:
        context['settings']['stages'] = dict(naf_train=dict(training=dict(main_updates=999)))
    with pytest.raises(ValueError):
        current.training_config(context)


def test_context_path_and_explicit_resume_guards(tmp_path):
    context = fixture_context(tmp_path)
    context['artifacts']['n1_cache'] = str(tmp_path / 'outside')
    with pytest.raises(ValueError, match='新run目录'):
        current.artifact(context, 'n1_cache')
    with pytest.raises(ValueError, match='current_training_context'):
        current.checked_context(dict(format='old'))


def test_train_dispatch_reuses_legacy_engine_and_only_completed_1000(tmp_path, monkeypatch):
    context = fixture_context(tmp_path); add_cache(context)
    import train_boundary_rgb as trainer
    observed = {}
    def train(config, output, device, resume):
        observed.update(config=config, output=Path(output), device=device, resume=resume)
        Path(output).mkdir(); Path(output, 'latest.pt').write_bytes(b'private_finished_checkpoint')
        return dict(complete=True, global_step=1000)
    monkeypatch.setattr(trainer, 'train', train)
    old_threads = torch.get_num_threads()
    try:
        result = current.run_train(context, device='cpu')
    finally:
        torch.set_num_threads(old_threads)
    assert observed['output'] == Path(context['artifacts']['naf']) and observed['resume'] is None
    assert result['complete'] is True and result['global_step'] == 1000
    with pytest.raises(ValueError, match='同run'):
        current.run_train(context, device='cpu', resume=str(tmp_path / 'other_latest.pt'))


def test_export_strips_D_optimizer_RNG_and_checks_new_state(tmp_path, monkeypatch):
    context = fixture_context(tmp_path); add_cache(context)
    config = current.training_config(context)
    folder = Path(context['artifacts']['naf']); folder.mkdir()
    cfg = config['boundary_naf']
    saved = dict(format='boundary_rgb_training_v1', global_step=1000, config=cfg,
        architecture=dict(cfg['model'], seed=cfg['seed'], input_mode='boundary'),
        model_state_dict={'weight': torch.tensor([1.25])}, model_sha256='c'*64,
        identity=dict(numerical_contract={'precision':'float32'}), identity_sha256='d'*64,
        input_contract=cfg['inference'], optimizer={'private':True}, rng={'private':True},
        discriminator_state_dict={'private':True})
    torch.save(saved, folder / 'latest.pt')
    current.write(folder / 'complete.json', dict(complete=True, global_step=1000, model_sha256='c'*64))
    from tools import infer_boundary_rgb
    calls = []
    def load(path, device, *, expected_model_sha):
        value = torch.load(path, weights_only=False)
        assert value['model_sha256'] == expected_model_sha == 'c'*64
        calls.append(Path(path))
        return object(), dict(architecture=value['architecture'], input_contract=value['input_contract'], parameter_count=1)
    monkeypatch.setattr(infer_boundary_rgb, 'load_naf_checkpoint', load)
    result = current.run_export(context)
    exported = torch.load(context['artifacts']['naf_export'], weights_only=False)
    assert len(calls) == 2 and result['model_sha256'] == 'c'*64
    assert not {'optimizer', 'rng', 'discriminator_state_dict'} & set(exported)
    assert exported['model_state_dict']['weight'].equal(torch.tensor([1.25]))
    with pytest.raises(FileExistsError): current.run_export(context)
