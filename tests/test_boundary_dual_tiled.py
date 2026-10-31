# -*- coding: utf-8 -*-
"""双输入整图T8/连续B分窗及旧概率/最终读出桥接；纯合成CPU。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from models.boundary_dual import BoundaryDualNAFNet, INPUT_MODE
from models.boundary_rgb import BoundaryRGBNAFNet
from tools.boundary_dual_monitor import predict_boundary_dual_tiled
from tools.boundary_rgb_monitor import predict_boundary_rgb_tiled
from tools.boundary_restore_readout import evaluate_frozen_readout
from utils.boundary_input import normalize_boundary_input
from utils.final_instance_cleanup import merge_final_instances


@pytest.fixture(autouse=True)
def cpu_threads():
    old_torch, old_cv = torch.get_num_threads(), cv2.getNumThreads()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    yield
    torch.set_num_threads(old_torch)
    cv2.setNumThreads(old_cv)


def _settings(**overrides):
    result = dict(input_mode=INPUT_MODE, input_threshold=.5, tile_size=32,
        overlap=8, reflect_pad=3, batch_size=4, iterations=1,
        revert_unconnected=False, input_normalization=dict(mode='skeleton_dilate', kernel_size=8))
    result.update(overrides)
    return result


class Probe(nn.Module):
    input_mode = INPUT_MODE

    def __init__(self, channel=1):
        super().__init__()
        self.channel = channel
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout()
        self.seen = []

    def forward(self, value, *, use_continuous=True):
        self.seen.append(value.detach().clone())
        assert type(use_continuous) is bool
        return value[:, self.channel:self.channel+1] + self.anchor * 0


@pytest.mark.parametrize('shape', [(73, 91), (3, 7), (1, 3)])
@pytest.mark.parametrize('channel', [0, 1])
def test_whole_t8_once_and_continuous_geometry_match(shape, channel):
    boundary = np.random.default_rng(23).random(shape, dtype=np.float32)
    before = boundary.copy()
    model = Probe(channel).train()
    model.dropout.eval()
    actual, receipt = predict_boundary_dual_tiled(boundary, model, _settings())
    normalized, normalization = normalize_boundary_input(boundary, threshold=.5,
        settings=_settings()['input_normalization'])
    expected = normalized.astype(np.float32) if channel == 0 else boundary
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    first = model.seen[0][0].numpy()
    a = np.pad(normalized.astype(np.float32), 3, mode='reflect')
    b = np.pad(boundary, 3, mode='reflect')
    a, b = a[:32, :32], b[:32, :32]
    if a.shape != (32, 32):
        a = np.pad(a, ((0, 32-a.shape[0]), (0, 32-a.shape[1])), mode='reflect')
        b = np.pad(b, ((0, 32-b.shape[0]), (0, 32-b.shape[1])), mode='reflect')
    assert np.array_equal(first[0], a) and np.array_equal(first[1], b)
    assert np.array_equal(boundary, before)
    assert model.training and not model.dropout.training
    assert receipt['input_normalization'] == normalization
    assert not receipt['continuous_input_binarized'] and not receipt['native_resize']


@pytest.mark.parametrize('off', [False, True])
def test_old_tiled_probability_and_final_readout_exact(off):
    shape = (120, 160)
    y, x = np.indices(shape)
    boundary = np.full(shape, .02, np.float32)
    for column in (40, 80, 120):
        boundary[:, column - 2:column + 3] = .9
    for row in (40, 80):
        boundary[row - 2:row + 3, :] = .9
    kwargs = dict(width=4, encoder_blocks=(1, 1), middle_blocks=1,
                  decoder_blocks=(1, 1), seed=17)
    old = BoundaryRGBNAFNet(**kwargs, input_mode='boundary').eval()
    dual = BoundaryDualNAFNet(**kwargs).eval()
    if off:
        with torch.no_grad():
            dual.continuous_intro.weight.fill_(.125)
    cfg = _settings(tile_size=64, overlap=16, reflect_pad=32, batch_size=4)
    legacy_cfg = dict(cfg, input_mode='boundary')
    expected, old_receipt = predict_boundary_rgb_tiled(boundary,
        np.zeros((*shape, 3), np.uint8), old, legacy_cfg)
    actual, receipt = predict_boundary_dual_tiled(boundary, dual, cfg, use_continuous=not off)
    assert np.array_equal(actual, expected)
    assert receipt['output_sha256'] == old_receipt['output_sha256']
    assert receipt['tiles'] == old_receipt['tiles']
    raw = np.where(x < 80, .8, .2).astype(np.float32)
    semantic = torch.from_numpy(np.where(x < 80, 2., -2.).astype(np.float32))[None, None]
    config = {'inference': dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_border_seal_width=2,
        bridge_width=1, watershed_dilate_width=1, min_instance_area=50,
        seed_min_area=50, final_min_instance_area=200,
        final_small_instance_action='neighbor_merge_v1', max_instance_id=65535,
        threshold=.5, semantic_vote_mode='probability_mean')}
    ids, classes, _ = evaluate_frozen_readout(semantic, raw, expected, config)
    now_ids, now_classes, audit = evaluate_frozen_readout(semantic, raw, actual, config,
        reference=dict(instances=ids, classes=classes))
    final, labels, cleanup = merge_final_instances(ids, classes, 200)
    current, current_labels, current_cleanup = merge_final_instances(now_ids, now_classes, 200)
    assert np.array_equal(final, current) and labels == current_labels and cleanup == current_cleanup
    assert audit['reference_final_exact'] and len(labels) >= 2


@pytest.mark.parametrize('overrides', [dict(input_mode='boundary'), dict(tile_size=0),
    dict(overlap=32), dict(reflect_pad=-1), dict(batch_size=0), dict(input_threshold=.4),
    dict(iterations=2), dict(revert_unconnected=True), dict(input_normalization=None),
    dict(input_normalization=dict(mode='skeleton_dilate', kernel_size=6)),
    dict(reflect_pad=10_000), dict(tile_size=4096), dict(tile_size=2, overlap=1)])
def test_invalid_or_unbounded_settings_rejected(overrides):
    with pytest.raises(ValueError):
        predict_boundary_dual_tiled(np.zeros((73, 91), np.float32), Probe(), _settings(**overrides))


@pytest.mark.parametrize('array', [np.zeros((32, 32), np.float64),
    np.zeros((2, 32, 32), np.float32), np.zeros((0, 2), np.float32),
    np.full((32, 32), float('nan'), np.float32), np.full((32, 32), 1.1, np.float32)])
def test_invalid_native_arrays_rejected(array):
    with pytest.raises(ValueError):
        predict_boundary_dual_tiled(array, Probe(), _settings())


def test_native_size_guard_precedes_value_checks():
    class NoRead(np.ndarray):
        def __array_ufunc__(self, *args, **kwargs):
            raise AssertionError('Value checking happened before native shape guard')

    # stride0只占一个float，不真的分配24MB以上；shape检查必须先拒绝。
    array = np.broadcast_to(np.zeros(1, np.float32), (3000, 3000)).view(NoRead)
    with pytest.raises(ValueError, match='6M'):
        predict_boundary_dual_tiled(array, Probe(), _settings())


def test_bad_output_restores_mixed_train_eval_flags():
    class Bad(Probe):
        def forward(self, value, *, use_continuous=True):
            return torch.full_like(value[:, :1], float('nan'))

    model = Bad().train()
    model.dropout.eval()
    with pytest.raises(RuntimeError, match='finite B1HW'):
        predict_boundary_dual_tiled(np.zeros((32, 32), np.float32), model, _settings())
    assert model.training and not model.dropout.training
