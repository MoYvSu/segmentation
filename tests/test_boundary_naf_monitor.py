# -*- coding: utf-8 -*-
"""独立NAF读出/缩略图的CPU契约；仅使用合成图和随机小模型。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from data.boundary_naf import BoundaryNAFDataset
from tools.boundary_naf_monitor import (
    _rgb_view, canonical_input, predict_boundary_tiled, save_monitor,
)
from tools.boundary_restore_readout import evaluate_frozen_readout


class IdentityBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, value):
        self.seen.append(value.detach().cpu().clone())
        return value + self.anchor * 0


class PointwiseBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, value):
        return (value * self.scale + 0.1).clamp(0, 1)


def _settings(**overrides):
    return dict(input_mode='binary', input_threshold=0.5, tile_size=32, overlap=8,
                reflect_pad=3, batch_size=4, iterations=1, revert_unconnected=False, **overrides)


def test_disabled_path_returns_original_object_without_calling_any_model():
    boundary = np.random.default_rng(1).random((37, 55), dtype=np.float32)
    before = boundary.copy()
    output, receipt = predict_boundary_tiled(boundary, None, _settings(), enabled=False)
    assert output is boundary and receipt['bypass'] and receipt['boundary_exact']
    np.testing.assert_array_equal(boundary, before)


@pytest.mark.parametrize('mode', ['binary', 'continuous'])
def test_identity_model_matches_representation_only_not_a_second_amplitude_mix(mode):
    boundary = np.random.default_rng(2).random((73, 91), dtype=np.float32)
    model = IdentityBoundary().train()
    settings = _settings()
    settings['input_mode'] = mode
    original = boundary.copy()
    output, receipt = predict_boundary_tiled(boundary, model, settings)
    expected = canonical_input(boundary, mode, 0.5)
    np.testing.assert_allclose(output, expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(boundary, original)
    assert model.training and receipt['tiles'] > 1
    if mode == 'binary':
        # 若错把p(Bbin)-Bbin加回Braw，此断言会失败。
        assert np.abs(output - boundary).mean() > 0.1
        assert all(set(torch.unique(value).tolist()) <= {0., 1.} for value in model.seen)


def test_batch_one_and_four_have_the_same_native_result():
    boundary = np.random.default_rng(3).random((91, 77), dtype=np.float32)
    model = PointwiseBoundary()
    one = _settings(); one['batch_size'] = 1
    four = _settings(); four['batch_size'] = 4
    result_one, receipt_one = predict_boundary_tiled(boundary, model, one)
    result_four, receipt_four = predict_boundary_tiled(boundary, model, four)
    np.testing.assert_allclose(result_one, result_four, atol=1e-6, rtol=0)
    assert receipt_one['tiles'] == receipt_four['tiles']
    np.testing.assert_allclose(result_one,
        0.8 * canonical_input(boundary, 'binary', 0.5) + 0.1, atol=1e-6, rtol=0)


@pytest.mark.parametrize('shape', [(3, 7), (1, 3), (19, 11)])
def test_small_non_square_inputs_use_reflect_padding_without_resizing(shape):
    boundary = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    model = IdentityBoundary()
    settings = _settings()
    output, receipt = predict_boundary_tiled(boundary, model, settings)
    assert output.shape == shape and output.dtype == np.float32
    np.testing.assert_allclose(output, (boundary >= 0.5).astype(np.float32), atol=1e-6, rtol=0)
    padded = np.pad(boundary, 3, mode='reflect')
    expected_tile = np.pad(padded,
        ((0, 32 - padded.shape[0]), (0, 32 - padded.shape[1])), mode='reflect')
    np.testing.assert_array_equal(model.seen[0][0, 0].numpy(), expected_tile >= 0.5)
    assert receipt['tiles'] == 1


@pytest.mark.parametrize('bad', [
    np.zeros((2, 1, 8, 8), np.float32), np.zeros((8, 8, 1), np.float32),
    np.zeros((8, 8), np.float64), np.full((8, 8), np.nan, np.float32),
    np.full((8, 8), -0.1, np.float32), np.full((8, 8), 1.1, np.float32),
])
def test_invalid_native_probability_rejected(bad):
    with pytest.raises(ValueError, match='native float32 HW'):
        predict_boundary_tiled(bad, IdentityBoundary(), _settings())


@pytest.mark.parametrize('overrides', [
    {'batch_size': 0}, {'tile_size': 0}, {'overlap': 32}, {'overlap': -1},
    {'reflect_pad': -1}, {'iterations': 2}, {'revert_unconnected': True},
    {'input_mode': 'guessed'}, {'input_threshold': 0.0}, {'input_threshold': 1.0},
])
def test_invalid_tiling_and_forbidden_repair_modes_rejected(overrides):
    settings = _settings(); settings.update(overrides)
    with pytest.raises(ValueError):
        predict_boundary_tiled(np.zeros((16, 16), np.float32), IdentityBoundary(), settings)


@pytest.mark.parametrize('kind', ['wrong_shape', 'wrong_batch', 'nan', 'negative', 'above_one'])
def test_invalid_generator_outputs_rejected_and_mode_restored(kind):
    class Bad(nn.Module):
        def forward(self, value):
            if kind == 'wrong_shape': return value[..., :-1, :]
            if kind == 'wrong_batch': return value[:1]
            if kind == 'nan': return torch.full_like(value, float('nan'))
            if kind == 'negative': return torch.full_like(value, -0.1)
            return torch.full_like(value, 1.1)
    model = Bad().train()
    with pytest.raises(RuntimeError, match='NAF输出'):
        predict_boundary_tiled(np.zeros((73, 91), np.float32), model, _settings())
    assert model.training


def test_small_final_uint16_readout_matches_binary_adapter_control():
    boundary = np.full((96, 112), 0.03, np.float32)
    boundary[:, 48:57] = 0.9
    boundary[43:47, 48:57] = 0.53
    semantic = torch.linspace(-2, 2, 112)[None, None, None].expand(1, 1, 96, 112).clone()
    raw = np.full(boundary.shape, 0.8, np.float32); raw[:, 56:] = 0.2
    config = {'inference': dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_border_seal_width=2, bridge_width=1,
        watershed_dilate_width=1, min_instance_area=50, max_instance_id=65535,
        threshold=.5, semantic_vote_mode='probability_mean')}
    before = deepcopy(config)
    adapted = canonical_input(boundary, 'binary', 0.5)
    result, _ = predict_boundary_tiled(boundary, IdentityBoundary(), _settings())
    ids, classes, _ = evaluate_frozen_readout(semantic, raw, adapted, config)
    actual_ids, actual_classes, _ = evaluate_frozen_readout(semantic, raw, result, config,
        reference={'instances': ids, 'classes': classes})
    assert ids.dtype == np.uint16 and ids.max() <= 65535
    np.testing.assert_array_equal(actual_ids, ids)
    assert actual_classes == classes and config == before


def test_monitor_rgb_crop_rotation_flip_and_padding_align_exactly(tmp_path):
    y, x = np.indices((13, 19))
    rgb = np.stack((y * 9, x * 7, y + x), axis=-1).astype(np.uint8)
    image = tmp_path / 'train_001.png'
    assert cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    dataset = SimpleNamespace(records=[dict(source='train_001', image=str(image))])
    for rotation in range(4):
        for flip in (False, True):
            sample = dict(source='train_001', box=[2, 3, 5, 7], input=torch.zeros(1, 16, 16),
                meta=dict(geometry=dict(rotation=rotation, flip=flip)))
            actual = _rgb_view(dataset, sample)
            expected = np.rot90(rgb[2:7, 3:10], rotation)
            if flip: expected = expected[:, ::-1]
            h, w = expected.shape[:2]
            expected = np.pad(expected, ((0, 16-h), (0, 16-w), (0, 0)), mode='reflect')
            np.testing.assert_array_equal(actual, expected)


def test_paired_only_monitor_writes_fixed_meta_and_never_requests_full_controls(tmp_path, monkeypatch):
    import tools.boundary_naf_monitor as module
    rgb = np.zeros((80, 96, 3), np.uint8)
    rgb[:, :48] = [211, 197, 180]; rgb[:, 48:] = [71, 63, 59]
    ids = np.ones((80, 96), np.int32); ids[:, 48:] = 2
    original = ids > 0
    image = tmp_path / 'train_001.png'; assert cv2.imwrite(str(image), rgb[:, :, ::-1])
    gt = tmp_path / 'train_001_gt.npz'
    np.savez_compressed(gt, instance_map=ids, original_covered=original,
                        filled=np.zeros_like(original), residual_unknown=~original)
    labels = tmp_path / 'train_001_class.json'
    labels.write_text('{"1":0,"2":1}', encoding='utf-8')
    files = dict(image=image, gt=gt, classes=labels)
    row = dict(source='train_001', **{key: str(path) for key, path in files.items()},
        hashes={key + '_sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in files.items()})
    cfg = dict(boundary_naf=dict(seed=20261026,
        data=dict(expected_sources=1, crop_size=64, stride=50, augmentations=8,
            samples_per_epoch=8, valid_margin=4, crop_margin=4, identity_probability=0.0),
        inference=_settings(), monitor=dict(pair_indices=[0, 1], thumbnail_width=64,
            source_names=[], control_manifests=[], roi_spec=None, full_readout_steps=[])))
    dataset = BoundaryNAFDataset([row], cfg, epoch=0)
    identity = json.dumps(dataset.identity, sort_keys=True)
    sample = dataset[0]
    monkeypatch.setattr(module, '_fixed_controls', lambda *args: (_ for _ in ()).throw(
        AssertionError('Paired-only monitor must not request full test controls')))
    with torch.no_grad():
        report = save_monitor(IdentityBoundary(), cfg, tmp_path / 'run', 7, dataset)
    assert report['global_step'] == 7 and not report['full_readout']
    assert len(report['rows']) == 2 and report['no_test_training']
    assert report['rows'][0]['meta'] == sample['meta']
    assert report['rows'][0]['valid_before_mae'] == report['rows'][0]['valid_after_mae']
    folder = tmp_path / 'run/monitor/s000007'
    assert Image.open(folder / 'pair_000.jpg').size[0] == 4 * 64
    assert (folder / 'pair_000_local.jpg').is_file() and (folder / 'summary.json').is_file()
    assert json.dumps(dataset.identity, sort_keys=True) == identity
    with pytest.raises(ValueError, match='已有完整回执'):
        save_monitor(IdentityBoundary(), cfg, tmp_path / 'run', 7, dataset)
