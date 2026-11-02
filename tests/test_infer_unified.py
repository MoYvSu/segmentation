# -*- coding: utf-8 -*-
"""统一局部两头的CPU合同：合成数组、无赛题数据、无训练权重。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from tools import infer_unified as unified
from tools.probe_affinity_patch import boundary_grid
from utils.semantic_native import NativeLogitBlend
from utils.patch_diagnostic import BlendMap


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def baseline():
    return unified.entry.load_config(str(unified.ROOT / 'config/inference/area_floor_200.yaml'))


class Encoder(nn.Module):
    def forward(self, image):
        self.features = [image]
        return self.features


class Affinity(nn.Module):
    def forward(self, features):
        self.features = features
        return dict(affinity_logits=features[0][:, :1, ::2, ::2].expand(-1, 8, -1, -1).contiguous())


class Semantic(nn.Module):
    def forward(self, features, image):
        assert features[0] is image
        self.features = features
        return image[:, :1] * 2 - .7


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.affinity_decoder, self.semantic_decoder = Encoder(), Affinity(), Semantic()
        self.semantic_lora = self.geometry_feature_adapter = self.geometry_highres_refiner = None

    def forward(self, image):
        raise AssertionError('Whole/global/raw backend forward is forbidden in this experiment')


class Restorer(nn.Module):
    num_steps = 16

    def q_sample(self, image, condition, step, *, noise=None):
        assert noise is not None and torch.count_nonzero(noise).item() == 0
        return condition

    def denoise(self, state, condition, step):
        assert torch.equal(state, condition)
        return condition + .01


def test_same_features_for_both_heads_and_old_affinity_arithmetic_exact():
    model = Backend().eval()
    image = torch.linspace(0, 1, 3 * 8 * 12).reshape(1, 3, 8, 12)
    expected = boundary_grid(model, image, baseline())
    actual, semantic = unified.shared_tile(model, image, baseline())
    assert torch.equal(actual, expected)
    assert model.encoder.features is model.affinity_decoder.features is model.semantic_decoder.features
    assert torch.equal(semantic, image[:, :1] * 2 - .7)


@pytest.mark.parametrize('name', ['semantic_lora', 'geometry_feature_adapter', 'geometry_highres_refiner'])
def test_reject_architecture_that_needs_different_features(name):
    model = Backend()
    setattr(model, name, nn.Identity())
    with pytest.raises(ValueError, match='shared-feature backend'):
        unified.check_model(model)


def test_plain_patch_does_not_save_call_arguments():
    target = SimpleNamespace(function=lambda x: x)
    original = target.function
    replacement = lambda x: x + 1
    with unified.PlainPatch.object(target, 'function', side_effect=replacement):
        assert target.function is replacement and not isinstance(target.function, Mock)
        assert target.function(2) == 3
        assert not hasattr(target.function, 'call_args_list')
    assert target.function is original


def install_synthetic_predict(monkeypatch):
    rgb = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    boxes = [(0, 0, 8, 8), (0, 4, 8, 12)]
    monkeypatch.setattr(unified.entry, 'read_image', lambda *args: rgb[..., ::-1].copy())
    monkeypatch.setattr(unified, 'tile_boxes', lambda *args: boxes)
    monkeypatch.setattr(unified, 'tensor_from_rgb', lambda crop, device:
        (torch.from_numpy(crop.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255, 0, 0))
    monkeypatch.setattr(unified.entry.views, 'prediction_maps', lambda *a, **k:
        (_ for _ in ()).throw(AssertionError('No whole-image forward allowed')))
    observed = {}

    def decode(semantic, boundary, cfg, **kwargs):
        observed.update(semantic=semantic.clone(), boundary=boundary.clone(), cfg=deepcopy(cfg))
        return np.ones(rgb.shape[:2], np.uint16), {'1': 0}

    original_finalize = unified.entry.finalize_instance_area

    def finalize(result, cfg):
        observed['finalize_calls'] = observed.get('finalize_calls', 0) + 1
        return original_finalize(result, cfg)

    monkeypatch.setattr(unified.entry, 'decode_reflect', decode)
    monkeypatch.setattr(unified.entry, 'finalize_instance_area', finalize)
    return rgb, boxes, observed


def test_unified_route_has_no_global_forward_and_exact_original_affinity_blend(monkeypatch):
    rgb, boxes, observed = install_synthetic_predict(monkeypatch)
    model, restorer = Backend().eval(), Restorer().eval()
    records, windows, fixed = [], [], {}
    cfg, before = baseline(), baseline()
    actual_rgb, result = unified.unified_predict(records)(model, restorer, None, Path('test_169.jpg'),
        cfg, 'cpu', local_audit=windows, fixed_audit=fixed)
    assert np.array_equal(actual_rgb, rgb) and cfg == before
    bblend, sblend = BlendMap(rgb.shape[:2]), NativeLogitBlend(rgb.shape[:2], .001)
    for y, x, y1, x1 in boxes:
        condition = torch.from_numpy(rgb[y:y1, x:x1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        image, _ = unified.restoration_probe.restore_probe(Restorer(), condition, 0, 'zero')
        b = boundary_grid(model, image, cfg)
        b = unified.crop_letterbox_output(b, 1024, 0, 0, (y1-y, x1-x)).cpu()[0, 0].numpy()
        bblend.add((y, x, y1, x1), b)
        sblend.add((y, y1, x, x1), (image[:, :1] * 2 - .7)[0, 0].numpy())
    expected_boundary, _ = bblend.finish()
    expected_logits = sblend.finish()
    assert np.array_equal(result['boundary'], expected_boundary)
    assert np.array_equal(observed['semantic'][0, 0].numpy(), expected_logits)
    probability = torch.from_numpy(expected_logits).sigmoid().numpy()
    assert fixed['raw_probability_sha256'] == unified.naf_entry.array_sha(probability)
    assert result['classes'] == {'1': int(probability.mean() > .5)}
    assert result['instances'].dtype == np.uint16 and observed['finalize_calls'] == 1
    record = records[0]
    assert {name: record[name] for name in ('encoder_calls', 'semantic_head_calls', 'affinity_head_calls',
        'local_zero_calls', 'global_zero_calls', 'global_encoder_calls', 'raw_encoder_calls', 'final200_calls')} == dict(
        encoder_calls=2, semantic_head_calls=2, affinity_head_calls=2, local_zero_calls=2,
        global_zero_calls=0, global_encoder_calls=0, raw_encoder_calls=0, final200_calls=1)
    assert record['semantic_probability_sha256'] == fixed['raw_probability_sha256']
    assert len(windows) == 2 and all(row['noise_nonzero'] == 0 and row['initial_equals_condition'] for row in windows)


def test_accidental_noise_fails_closed_and_hooks_are_removed(monkeypatch):
    install_synthetic_predict(monkeypatch)
    model, restorer = Backend().eval(), Restorer().eval()
    original = model.encoder.forward

    def noisy(image):
        torch.randn((1,))
        return original(image)

    monkeypatch.setattr(model.encoder, 'forward', noisy)
    with pytest.raises(RuntimeError, match='noise generation is forbidden'):
        unified.unified_predict([])(model, restorer, None, Path('test_169.jpg'), baseline(), 'cpu')
    assert not model.encoder._forward_hooks and not model.semantic_decoder._forward_hooks


def test_report_roles_and_actual_counts_are_explicit_and_idempotent():
    cfg = baseline()
    document = dict(format='old', config=cfg, runtime=dict(restoration_noise={'name': 'old'}),
                    images=[dict(source='test_169', boundary_naf={'raw_probability_sha256': 'a' * 64})],
                    geometry='old')
    before = deepcopy(document)
    names = ('encoder_calls', 'semantic_head_calls', 'affinity_head_calls', 'local_zero_calls',
             'global_zero_calls', 'global_encoder_calls', 'raw_encoder_calls', 'final200_calls')
    rows = [dict(source='test_169', **{name: 0 for name in names})]
    actual = unified.audit_document(document, rows, 'b' * 64)
    assert document == before
    assert actual['config']['submission']['global_restoration']['mode'] == 'not_computed'
    assert actual['baseline_config'] == cfg
    assert actual['images'][0]['boundary_naf']['raw_probability_role'] == unified.SEMANTIC_ROLE
    assert actual['unified_local1024']['legacy_capture_roles']['raw_probability'].endswith('not_raw_RGB')
    assert unified.audit_document(actual, rows, 'b' * 64) == actual
