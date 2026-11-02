# -*- coding: utf-8 -*-
"""教师NAF真实e0过程图入口；不读取赛题数据，不训练网络。"""
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from data.boundary_rgb import BoundaryRGBDataset
from models.boundary_rgb import BoundaryRGBNAFNet
from tools import boundary_naf_monitor, boundary_rgb_monitor
from train_boundary_rgb import monitor
from utils.config import load_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def one_cpu_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def teacher_dataset(tmp_path):
    image, gt, classes = (tmp_path / name for name in ('train_001.png', 'train_001_gt.npz', 'train_001_class.json'))
    rgb = np.full((80, 96, 3), 127, np.uint8)
    assert cv2.imwrite(str(image), rgb)
    ids = np.ones((80, 96), np.uint16)
    ids[:, 49:] = 2
    ids[:, 48] = 0
    ids[10:16, 10:18] = 0
    np.savez_compressed(gt, instance_map=ids, teacher_known=ids > 0, teacher_unknown=ids == 0)
    classes.write_text('{"1":0,"2":1}', encoding='utf-8')
    row = dict(source='train_001', image=str(image), gt=str(gt), classes=str(classes), shape=list(ids.shape),
               hashes={name + '_sha256': sha(path) for name, path in (('image', image), ('gt', gt), ('classes', classes))})
    manifest = tmp_path / 'targets.json'
    manifest.write_text(json.dumps(dict(rows=[row])), encoding='utf-8')
    heat = tmp_path / 'heat.npz'
    probability = np.zeros(ids.shape, np.float32)
    probability[:, 47:50] = .9
    probability[24:28, 16:22] = .8
    np.savez_compressed(heat, boundary=probability)
    models = dict(backend={'sha256': '1' * 64}, restorer={'sha256': '2' * 64})
    heat_manifest = tmp_path / 'heat.json'
    heat_manifest.write_text(json.dumps(dict(format='boundary_naf_real_heatmaps_v1', complete=True,
        no_test_training=True, local_restoration=dict(mode='zero', model='restorer'), models=models,
        processed_gt_manifest_sha256=sha(manifest), files=[dict(source='train_001', shape=list(ids.shape),
            heat_path=heat.name, heat_sha256=sha(heat), **row['hashes'])])), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / 'config/train/boundary_width.yaml'))
    config['paths']['project_root'] = str(tmp_path)
    cfg = config['boundary_naf']
    cfg.update(manifest=str(manifest), real_input_manifest=str(heat_manifest), real_input_models=models,
               expected_sources=1)
    cfg['data'].update(crop_size=32, stride=16, samples_per_epoch=8,
                       teacher_boundary_policy='explicit_separators_v1')
    cfg['monitor'].update(pair_indices=[0], thumbnail_width=64, source_names=[], control_manifests=[],
                          full_readout_steps=[], roi_spec=None)
    dataset = BoundaryRGBDataset(manifest, cfg, epoch=0)
    return config, dataset, gt


@pytest.mark.parametrize('entry', ['current_rgb_e0', 'legacy_naf_e0'])
def test_real_teacher_dataset_passes_monitor_e0_without_manual_provenance(tmp_path, monkeypatch, entry):
    config, dataset, gt = teacher_dataset(tmp_path)
    before_file = sha(gt)
    before_target = {s: tuple(a.copy() for a in values) for s, values in dataset.cache.items()}
    model = BoundaryRGBNAFNet(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1],
                             input_mode='boundary').train()
    before_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    before_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    module = boundary_rgb_monitor if entry == 'current_rgb_e0' else boundary_naf_monitor
    monkeypatch.setattr(module, '_fixed_controls', lambda *_: (_ for _ in ()).throw(AssertionError('No old test controls')))
    if entry == 'current_rgb_e0':
        receipt = monitor(model, config, tmp_path / 'run', 0, dataset)
    else:
        with torch.no_grad():
            receipt = module.save_monitor(model, config, tmp_path / 'run', 0, dataset)
    row = receipt['rows'][0]
    assert row['target_source_kind'] == 'model_teacher'
    assert row['diagnostic_support_mask'] == 'teacher_known'
    assert row['diagnostic_support_is_manual_original'] is False
    assert 'teacher_known' in row['noise_interpretation']
    assert 'teacher_known_deep_input_pixels' in row
    assert not any('original_deep_input' in key for key in row)
    if entry == 'current_rgb_e0':
        assert 'normalized_teacher_known_deep_input_pixels' in row
        assert model.training
        assert all(torch.equal(v, before_model[k]) for k, v in model.state_dict().items())
        assert random.getstate() == before_rng[0]
        current_np = np.random.get_state()
        assert current_np[0] == before_rng[1][0] and np.array_equal(current_np[1], before_rng[1][1])
        assert current_np[2:] == before_rng[1][2:] and torch.equal(torch.get_rng_state(), before_rng[2])
    assert not receipt['full_readout']
    assert (tmp_path / 'run/monitor/s000000/pair_000.jpg').is_file()
    assert (tmp_path / 'run/monitor/s000000/pair_000_local.jpg').is_file()
    assert json.loads((tmp_path / 'run/monitor/s000000/summary.json').read_text(encoding='utf-8')) == receipt
    assert sha(gt) == before_file
    with np.load(gt) as arrays:
        assert set(arrays.files) == {'instance_map', 'teacher_known', 'teacher_unknown'}
    for source, values in dataset.cache.items():
        assert all(np.array_equal(a, b) for a, b in zip(values, before_target[source]))
    dataset.verify_sources()


def test_legacy_original_view_still_uses_original_covered_not_all_known(tmp_path):
    gt = tmp_path / 'legacy_gt.npz'
    original = np.zeros((16, 20), bool)
    original[4:12, 5:10] = True
    filled = np.zeros_like(original)
    filled[4:12, 10:15] = True
    np.savez_compressed(gt, instance_map=np.ones(original.shape, np.uint16), original_covered=original,
                        filled=filled, residual_unknown=~(original | filled))
    dataset = SimpleNamespace(records=[dict(source='train_001', gt=str(gt), source_kind='manual_completed')])
    sample = dict(source='train_001', box=[0, 0, 16, 20], input=torch.zeros(1, 20, 20),
                  meta=dict(geometry=dict(rotation=0, flip=False)))
    actual = boundary_naf_monitor.original_view(dataset, sample)
    np.testing.assert_array_equal(actual[:16], original)
    assert not actual[:16][filled].any()
    assert boundary_naf_monitor.diagnostic_support_kind(dataset, sample) == 'original_covered'
    metrics, _ = boundary_naf_monitor.real_pair_metrics(np.zeros(actual.shape), np.zeros(actual.shape),
        np.zeros(actual.shape, np.float32), np.ones(actual.shape, bool), actual)
    assert 'original_deep_input_pixels' in metrics and 'teacher_known_deep_input_pixels' not in metrics


def test_teacher_diagnostic_support_preserves_unknown_through_crop_rotation(tmp_path):
    _, dataset, _ = teacher_dataset(tmp_path)
    sample = dataset[0]
    row = dataset.records[0]
    with np.load(row['gt']) as arrays:
        expected = arrays['teacher_known']
    y, x, height, width = sample['box']
    expected = expected[y:y+height, x:x+width]
    geometry = sample['meta']['geometry']
    expected = np.rot90(expected, int(geometry['rotation']))
    if geometry['flip']:
        expected = expected[:, ::-1]
    np.testing.assert_array_equal(boundary_naf_monitor.original_view(dataset, sample), expected)
