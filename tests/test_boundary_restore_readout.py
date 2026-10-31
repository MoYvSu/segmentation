# -*- coding: utf-8 -*-
"""合成图验证残差拼接及完整旧解码契约，不读取比赛图像。"""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from tools.boundary_restore_readout import evaluate_frozen_readout, predict_residual_tiled, tiles


class Residual(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = 0
        self.seen = []

    def residual(self, value):
        self.calls += 1
        self.seen.append(value.detach().cpu().clone())
        return torch.full_like(value, self.value)

    def forward(self, value):
        raise AssertionError('Probability forward must never be used as residual')


def fixture_maps():
    boundary = np.full((96, 112), .03, np.float32)
    boundary[:, 48:57] = .9
    boundary[43:47, 48:57] = .53
    semantic = torch.linspace(-2, 2, 112)[None, None, None].expand(1, 1, 96, 112).clone()
    raw = np.full(boundary.shape, .8, np.float32)
    raw[:, 56:] = .2
    config = {'inference': dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_border_seal_width=2, bridge_width=1,
        watershed_dilate_width=1, min_instance_area=50, max_instance_id=65535,
        threshold=.5, semantic_vote_mode='probability_mean')}
    return boundary, semantic, raw, config


def test_tiles_cover_native_domain_without_duplicate_last_tile():
    boxes = tiles((711, 941), 512, 128)
    cover = np.zeros((711, 941), bool)
    assert len(boxes) == len(set(boxes))
    for y, x, y1, x1 in boxes:
        cover[y:y1, x:x1] = True
    assert cover.all()
    with pytest.raises(ValueError):
        tiles((5, 7), 512, 512)


def test_zero_strength_preserves_same_array_and_never_calls_model():
    value = np.random.default_rng(5).random((37, 71), dtype=np.float32)
    restored, audit = predict_residual_tiled(value, None, strength=0.)
    assert restored is value and audit['tiles'] == 0 and audit['boundary_exact']


def test_zero_residual_exact_even_for_odd_shape_and_overlap():
    value = np.random.default_rng(6).random((53, 87), dtype=np.float32)
    old = value.copy()
    model = Residual(0.).train()
    restored, audit = predict_residual_tiled(value, model, tile_size=32, overlap=11, reflect_pad=7)
    assert restored is value and np.array_equal(value, old)
    assert audit['tiles'] > 1 and audit['all_zero_residual']
    assert model.training


def test_residual_is_blended_then_added_once_and_clamped():
    value = np.full((45, 79), .9, np.float32)
    restored, audit = predict_residual_tiled(value, Residual(.25), tile_size=32, overlap=9,
                                            reflect_pad=5, strength=.5)
    assert np.array_equal(restored, np.ones_like(value))
    assert np.array_equal(value, np.full_like(value, .9))
    assert audit['residual_abs_mean'] == pytest.approx(.25)


def test_small_singleton_image_reflect_padding_and_shape():
    value = np.array([[.2, .4, .7]], np.float32)
    model = Residual(0.)
    restored, _ = predict_residual_tiled(value, model, tile_size=16, overlap=4, reflect_pad=2)
    assert restored is value
    assert all(tuple(v.shape) == (1, 1, 16, 16) for v in model.seen)


def test_reject_probability_only_model():
    with pytest.raises(ValueError, match='explicit residual'):
        predict_residual_tiled(np.zeros((9, 11), np.float32), nn.Identity())


@pytest.mark.parametrize('strength', [-.1, 1.01, float('nan')])
def test_reject_unrecorded_strength(strength):
    with pytest.raises(ValueError):
        predict_residual_tiled(np.zeros((9, 11), np.float32), Residual(0.), strength=strength)


@pytest.mark.parametrize('strength', [0., 1.])
def test_final_legacy_reflect_decoder_and_raw_vote_exact(strength):
    from tools.infer_semifinal import decode_reflect
    from tools.semantic_crossover import revote_instances
    boundary, semantic, raw, config = fixture_maps()
    before = deepcopy(config)
    ids, _ = decode_reflect(semantic, torch.from_numpy(boundary)[None, None], config)
    classes, _ = revote_instances(ids, raw)
    completed, tile_audit = predict_residual_tiled(boundary, Residual(0.), tile_size=64,
        overlap=16, reflect_pad=8, strength=strength)
    actual_ids, actual_classes, audit = evaluate_frozen_readout(semantic, raw, completed, config,
        reference={'instances': ids, 'classes': classes})
    assert np.array_equal(actual_ids, ids) and actual_classes == classes
    assert audit['reference_final_exact'] and completed is boundary and tile_audit['boundary_exact']
    assert config == before


def test_final_exact_gate_rejects_wrong_reference():
    boundary, semantic, raw, config = fixture_maps()
    with pytest.raises(RuntimeError, match='exact final'):
        evaluate_frozen_readout(semantic, raw, boundary, config,
            reference={'instances': np.zeros(boundary.shape, np.uint16), 'classes': {}})


def test_refuses_resizing_frozen_semantics():
    boundary, semantic, raw, config = fixture_maps()
    with pytest.raises(ValueError, match='native boundary shape'):
        evaluate_frozen_readout(semantic[:, :, ::2, ::2], raw, boundary, config)


def test_real_zero_initialized_network_preserves_full_final_contract():
    from models.boundary_restore import BoundaryRestoreNet, state_sha256
    boundary, semantic, raw, config = fixture_maps()
    ids, classes, _ = evaluate_frozen_readout(semantic, raw, boundary, config)
    model = BoundaryRestoreNet(width=4, seed=73).train()
    before = state_sha256(model)
    completed, audit = predict_residual_tiled(boundary, model, tile_size=64, overlap=16, reflect_pad=8)
    final_ids, final_classes, final_audit = evaluate_frozen_readout(semantic, raw, completed, config,
        reference={'instances': ids, 'classes': classes})
    assert completed is boundary and audit['all_zero_residual']
    assert final_audit['reference_final_exact'] and final_classes == classes
    assert np.array_equal(final_ids, ids) and model.training
    assert state_sha256(model) == before
