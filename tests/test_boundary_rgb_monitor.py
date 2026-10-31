# -*- coding: utf-8 -*-
"""RGB补边分块与最终读出契约，只用合成数据。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import tools.boundary_rgb_monitor as module
from tools.boundary_restore_readout import evaluate_frozen_readout


class PointwiseRGB(nn.Module):
    def __init__(self, mode='rgb_boundary', channel=None):
        super().__init__()
        self.input_mode = mode
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout()
        self.seen = []
        self.channel = (3 if mode == 'rgb_boundary' else 0) if channel is None else channel

    def forward(self, value):
        self.seen.append(value.detach().cpu().clone())
        return value[:, self.channel:self.channel+1] + self.anchor * 0


def settings(mode='rgb_boundary', **kwargs):
    result = dict(input_mode=mode, input_threshold=.5, tile_size=32, overlap=8,
                  reflect_pad=3, batch_size=4, iterations=1, revert_unconnected=False)
    result.update(kwargs)
    return result


def rgb_for(shape):
    y, x = np.indices(shape)
    return np.stack(((y*9+x*3) % 256, (y*3+x*7) % 256, (y+x) % 256), -1).astype(np.uint8)


def frozen_config():
    return {'inference': dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_border_seal_width=2, bridge_width=1,
        watershed_dilate_width=1, min_instance_area=50, max_instance_id=65535,
        threshold=.5, semantic_vote_mode='probability_mean')}


def test_disabled_returns_original_exact_without_rgb_or_model():
    boundary = np.random.default_rng(1).random((17, 29), dtype=np.float32)
    actual, receipt = module.predict_boundary_rgb_tiled(boundary, None, None, {}, enabled=False)
    assert actual is boundary and receipt['boundary_exact'] and receipt['bypass']


@pytest.mark.parametrize('shape', [(73, 91), (3, 7), (1, 3), (19, 11)])
@pytest.mark.parametrize('mode', ['rgb_boundary', 'boundary'])
def test_native_binary_channels_padding_and_model_modes(shape, mode):
    boundary = np.random.default_rng(2).random(shape, dtype=np.float32)
    rgb = rgb_for(shape)
    original, image = boundary.copy(), rgb.copy()
    model = PointwiseRGB(mode).train()
    model.dropout.eval()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    output, receipt = module.predict_boundary_rgb_tiled(boundary, rgb, model, settings(mode))
    np.testing.assert_allclose(output, (boundary >= .5).astype(np.float32), atol=1e-6, rtol=0)
    np.testing.assert_array_equal(boundary, original)
    np.testing.assert_array_equal(rgb, image)
    assert output.dtype == np.float32 and model.training and not model.dropout.training
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert receipt['stitching'] == 'direct_probability_weighted_mean' and not receipt['native_resize']
    assert all(value.shape[1] == (4 if mode == 'rgb_boundary' else 1) for value in model.seen)


def test_rgb_and_boundary_use_same_geometry_and_batch_one_matches_four():
    boundary = np.random.default_rng(3).random((71, 83), dtype=np.float32)
    rgb = rgb_for(boundary.shape)
    model = PointwiseRGB(channel=0)
    one, _ = module.predict_boundary_rgb_tiled(boundary, rgb, model, settings(batch_size=1))
    many, _ = module.predict_boundary_rgb_tiled(boundary, rgb, model, settings(batch_size=4))
    np.testing.assert_allclose(one, rgb[..., 0].astype(np.float32)/255., atol=2e-7, rtol=0)
    np.testing.assert_allclose(one, many, atol=1e-7, rtol=0)
    # 检查同一窗口中原生坐标RGB和T的同步reflect、裁块和通道次序。
    first = model.seen[0][0].numpy()
    expected_rgb = np.pad(rgb, ((3, 3), (3, 3), (0, 0)), mode='reflect')[:32, :32]
    expected_t = np.pad(boundary >= .5, 3, mode='reflect')[:32, :32]
    np.testing.assert_array_equal(first[:3], expected_rgb.transpose(2, 0, 1).astype(np.float32)/255.)
    np.testing.assert_array_equal(first[3], expected_t)


@pytest.mark.parametrize('overrides', [dict(input_mode='wrong'), dict(input_mode='boundary'),
    dict(tile_size=0), dict(overlap=32), dict(overlap=-1), dict(reflect_pad=-1),
    dict(batch_size=0), dict(input_threshold=.4), dict(iterations=2), dict(revert_unconnected=True)])
def test_bad_settings_rejected(overrides):
    boundary = np.zeros((32, 48), np.float32)
    with pytest.raises(ValueError):
        module.predict_boundary_rgb_tiled(boundary, rgb_for(boundary.shape), PointwiseRGB(), settings(**overrides))


@pytest.mark.parametrize('bad_rgb', [np.zeros((16, 17, 3), np.float32), np.zeros((16, 17), np.uint8),
    np.zeros((17, 16, 3), np.uint8), np.zeros((16, 17, 4), np.uint8)])
def test_misaligned_or_wrong_rgb_rejected(bad_rgb):
    with pytest.raises(ValueError, match='aligned HWC uint8'):
        module.predict_boundary_rgb_tiled(np.zeros((16, 17), np.float32), bad_rgb, PointwiseRGB(), settings())


def test_bad_output_restores_mixed_training_flags():
    class Bad(PointwiseRGB):
        def forward(self, value):
            return torch.full_like(value[:, :1], float('nan'))
    model = Bad().train()
    model.dropout.eval()
    boundary = np.zeros((33, 49), np.float32)
    with pytest.raises(RuntimeError, match='finite B1HW'):
        module.predict_boundary_rgb_tiled(boundary, rgb_for(boundary.shape), model, settings())
    assert model.training and not model.dropout.training


def test_direct_probability_is_not_added_to_original_continuous_boundary():
    boundary = np.full((61, 75), .3, np.float32)
    boundary[:, 32:] = .8
    result, _ = module.predict_boundary_rgb_tiled(boundary, rgb_for(boundary.shape),
                                                 PointwiseRGB(), settings())
    np.testing.assert_allclose(result, boundary >= .5, atol=1e-6, rtol=0)
    assert np.abs(result-boundary).mean() > .2


def test_binary_adapter_final_readout_rebuilds_uint16_and_classes():
    boundary = np.full((96, 112), .03, np.float32)
    boundary[:, 48:57] = .9
    boundary[43:47, 48:57] = .53
    rgb = rgb_for(boundary.shape)
    semantic = torch.linspace(-2, 2, 112)[None, None, None].expand(1, 1, 96, 112).clone()
    raw = np.full(boundary.shape, .8, np.float32)
    raw[:, 56:] = .2
    config = frozen_config()
    original_config = deepcopy(config)
    expected = (boundary >= .5).astype(np.float32)
    ids, classes, _ = evaluate_frozen_readout(semantic, raw, expected, config)
    output, _ = module.predict_boundary_rgb_tiled(boundary, rgb, PointwiseRGB(), settings())
    actual, labels, audit = evaluate_frozen_readout(semantic, raw, output, config,
                                                   reference=dict(instances=ids, classes=classes))
    np.testing.assert_array_equal(actual, ids)
    assert labels == classes and audit['reference_final_exact']
    assert actual.dtype == np.uint16 and actual.max() <= 65535 and config == original_config
    assert {str(int(v)) for v in np.unique(actual) if v} == set(labels)
    assert set(labels.values()) <= {0, 1}


def test_paired_only_monitor_does_not_request_full_controls(tmp_path, monkeypatch):
    side = 64
    target = torch.zeros(1, side, side)
    target[:, :, 31:33] = 1
    source = target.clone()
    source[:, 25:31] = 0
    rgb = torch.from_numpy(rgb_for((side, side)).transpose(2, 0, 1).copy()).float()/255.
    sample = dict(input=torch.cat((rgb, source)), input_boundary=source, rgb=rgb,
        target=target, valid=torch.ones_like(target), gap_mask=target-source,
        source='train_001', meta=dict(geometry={'rotation': 0, 'flip': False}))
    dataset = {0: sample}
    cfg = dict(boundary_naf=dict(inference=settings(), monitor=dict(pair_indices=[0],
        thumbnail_width=64, source_names=[], control_manifests=[], full_readout_steps=[])))
    monkeypatch.setattr(module, 'original_view', lambda *_: np.ones((side, side), bool))
    monkeypatch.setattr(module, '_fixed_controls', lambda *_: (_ for _ in ()).throw(AssertionError('no full controls')))
    model = PointwiseRGB().train()
    model.dropout.eval()
    result = module.save_monitor(model, cfg, tmp_path, 7, dataset)
    assert model.training and not model.dropout.training
    assert not result['full_readout'] and result['rows'][0]['meta'] == sample['meta']
    assert result['rows'][0]['valid_before_mae'] == result['rows'][0]['valid_after_mae']
    assert Image.open(tmp_path / 'monitor/s000007/pair_000.jpg').size[0] == 256
    assert (tmp_path / 'monitor/s000007/pair_000_local.jpg').is_file()
    with pytest.raises(ValueError, match='已有完整回执'):
        module.save_monitor(model, cfg, tmp_path, 7, dataset)


def test_full_monitor_checks_n1_and_reuses_only_adapter_diagnostic(tmp_path, monkeypatch):
    boundary = np.full((64, 80), .03, np.float32)
    boundary[:, 38:43] = .9
    rgb = rgb_for(boundary.shape)
    semantic = np.zeros(boundary.shape, np.float32)
    raw = np.full(boundary.shape, .8, np.float32)
    config = frozen_config()
    ids, classes, _ = evaluate_frozen_readout(semantic, raw, boundary, config)
    cache = tmp_path / 'test_101.npz'
    np.savez_compressed(cache, boundary=boundary, rgb=rgb, semantic=semantic,
                        raw_probability=raw, instances=ids)
    class_path = tmp_path / 'test_101.json'
    class_path.write_text(json.dumps(classes), encoding='utf-8')
    row = dict(source='test_101', path=str(cache), control_classes=str(class_path),
               sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    monkeypatch.setattr(module, '_fixed_controls', lambda *_: [(row, config)])
    calls = []
    real_readout = evaluate_frozen_readout
    def readout(semantic, raw, value, cfg, **kwargs):
        calls.append((value.copy(), dict(kwargs)))
        return real_readout(semantic, raw, value, cfg, **kwargs)
    monkeypatch.setattr(module, 'evaluate_frozen_readout', readout)
    cfg = dict(boundary_naf=dict(inference=settings(), monitor=dict(pair_indices=[],
        thumbnail_width=64, source_names=['test_101'], full_readout_steps=[0, 1000])))
    model = PointwiseRGB()
    first = module.save_monitor(model, cfg, tmp_path / 'run', 0, {})
    assert len(calls) == 3  # 原N1exact、T-only表示诊断、新候选分别独立读出。
    assert calls[0][1]['reference']['classes'] == classes
    assert 'reference' not in calls[-1][1]
    second = module.save_monitor(model, cfg, tmp_path / 'run', 1000, {})
    assert len(calls) == 4  # 仅候选重算；不复用adapter实例作seed。
    assert first['rows'][0]['reference_final_exact'] and second['rows'][0]['seeds_and_barriers_regenerated']
    assert not second['rows'][0]['old_markers_reused']
    assert (tmp_path / 'run/monitor/s000000/test_101_adapter.jpg').is_file()
    assert not (tmp_path / 'run/monitor/s001000/test_101_adapter.jpg').exists()
