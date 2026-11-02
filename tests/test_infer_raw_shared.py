# -*- coding: utf-8 -*-
"""原图语义共用的CPU合同：无全图恢复/affinity，局部边界不变。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

from tools import infer_raw_shared as vote
from utils.patch_diagnostic import BlendMap


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def baseline():
    return vote.entry.load_config(str(vote.ROOT / 'config/inference/area_floor_200.yaml'))


class Encoder(nn.Module):
    def forward(self, image):
        return [image]


class Affinity(nn.Module):
    def forward(self, features):
        return dict(affinity_logits=features[0][:, :1, ::2, ::2].expand(-1, 8, -1, -1).contiguous())


class Semantic(nn.Module):
    def forward(self, features, image):
        return image[:, :1] * 2 + .5


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.affinity_decoder, self.semantic_decoder = Encoder(), Affinity(), Semantic()
        self.semantic_lora = self.geometry_feature_adapter = self.geometry_highres_refiner = None

    def forward(self, image):
        raise AssertionError('Full global backend forward must not execute')


class Restorer(nn.Module):
    num_steps = 16

    def q_sample(self, image, condition, step, *, noise=None):
        assert noise is not None and torch.count_nonzero(noise).item() == 0
        return condition

    def denoise(self, state, condition, step):
        assert torch.equal(state, condition)
        return condition + .01


def synthetic(monkeypatch):
    rgb = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    boxes = [(0, 0, 8, 8), (0, 4, 8, 12)]
    observed = dict(prepare_whole=0, votes=0, final200=0)

    def tensor(crop):
        return torch.from_numpy(crop.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255

    def prepare(*args):
        observed['prepare_whole'] += 1
        assert observed['prepare_whole'] == 1, 'No extra whole raw-semantic input may be prepared'
        return rgb, tensor(rgb), 0, 0

    def raw_forbidden(*args, **kwargs):
        raise AssertionError('The removed raw semantic branch must not execute')

    monkeypatch.setattr(vote, 'prepare_image', prepare)
    monkeypatch.setattr(vote.backend_entry, 'prediction_maps', raw_forbidden)
    monkeypatch.setattr(vote, 'tile_boxes', lambda *args: boxes)
    monkeypatch.setattr(vote, 'tensor_from_rgb', lambda crop, device: (tensor(crop), 0, 0))

    def decode(semantic, boundary, cfg, **kwargs):
        observed.update(semantic=semantic.clone(), boundary=boundary.clone(), cfg=deepcopy(cfg))
        return np.ones(rgb.shape[:2], np.uint16), {'1': 0}

    original_vote, original_final = vote.entry.views.revote_instances, vote.entry.finalize_instance_area

    def revote(ids, probability):
        observed['votes'] += 1
        observed['probability'] = probability.copy()
        return original_vote(ids, probability)

    def finalize(result, cfg):
        observed['final200'] += 1
        return original_final(result, cfg)

    monkeypatch.setattr(vote.entry, 'decode_reflect', decode)
    monkeypatch.setattr(vote.entry.views, 'revote_instances', revote)
    monkeypatch.setattr(vote.entry, 'finalize_instance_area', finalize)
    return rgb, boxes, observed, tensor


def test_only_raw_semantic_is_forwarded_and_reused_for_geometry_and_vote(monkeypatch):
    rgb, boxes, observed, tensor = synthetic(monkeypatch)
    records, windows, fixed = [], [], {}
    cfg, before = baseline(), baseline()
    model, restorer = Backend().eval(), Restorer().eval()
    actual_rgb, result = vote.raw_shared_predict(records)(model, restorer, None,
        Path('test_169.jpg'), cfg, 'cpu', local_audit=windows, fixed_audit=fixed)
    assert np.array_equal(actual_rgb, rgb) and cfg == before
    assert observed['prepare_whole'] == observed['votes'] == observed['final200'] == 1
    assert np.array_equal(observed['semantic'].sigmoid()[0, 0].numpy(), observed['probability'])
    assert result['classes'] == {'1': 1} and result['instances'].dtype == np.uint16
    assert observed['cfg']['submission']['semantic_route'] == vote.PROBABILITY_ROLE
    record = records[0]
    assert {key: record[key] for key in vote.COUNT_KEYS} == dict(encoder_calls=3,
        global_encoder_calls=0, local_encoder_calls=2, raw_encoder_calls=1, global_zero_calls=0,
        local_zero_calls=2, semantic_head_calls=1, affinity_head_calls=2, global_model_calls=0, final200_calls=1)
    assert record['raw_semantic_logits_sha256'] == fixed['geometry_semantic_sha256']
    assert record['raw_shared_probability_sha256'] == fixed['raw_probability_sha256']
    assert record['raw_semantic_mask_sha256'] == vote.naf_entry.array_sha(observed['probability'] > .5)
    assert torch.equal(observed['semantic'], tensor(rgb)[:, :1] * 2 + .5)
    assert len(record['zero_noise_calls']) == 2
    assert all(v['noise_nonzero'] == 0 and v['initial_equals_condition'] for v in record['zero_noise_calls'])
    # 独立旧局部路径重放：恢复、boundary_grid、原BlendMap算术完全不变。
    blend = BlendMap(rgb.shape[:2])
    for y, x, y1, x1 in boxes:
        restored, _ = vote.restoration_probe.restore_probe(restorer, tensor(rgb[y:y1, x:x1]), 0, 'zero')
        grid = vote.boundary_grid(model, restored, cfg)
        native = vote.crop_letterbox_output(grid, 1024, 0, 0, (y1-y, x1-x)).cpu()[0, 0].numpy()
        blend.add((y, x, y1, x1), native)
    expected, _ = blend.finish()
    assert np.array_equal(result['boundary'], expected)
    assert len(windows) == 2 and all(v['size'] == 1024 and v['noise_mode'] == 'zero' for v in windows)


def test_random_noise_generation_in_prediction_is_forbidden_and_hooks_restore(monkeypatch):
    synthetic(monkeypatch)
    model, restorer = Backend().eval(), Restorer().eval()
    original = model.encoder.forward

    def noisy(image):
        torch.randn((1,))
        return original(image)

    monkeypatch.setattr(model.encoder, 'forward', noisy)
    with pytest.raises(RuntimeError, match='noise generation is forbidden'):
        vote.raw_shared_predict([])(model, restorer, None, Path('test_169.jpg'), baseline(), 'cpu')
    assert not model.encoder._forward_hooks and not model._forward_hooks


@pytest.mark.parametrize('key', ['semantic_lora', 'geometry_feature_adapter', 'geometry_highres_refiner'])
def test_reject_unverified_backend_architecture(key):
    model = Backend()
    setattr(model, key, nn.Identity())
    with pytest.raises(ValueError, match='verified unadapted shared backend'):
        vote.raw_shared_predict([])(model, Restorer(), None, Path('test_169.jpg'), baseline(), 'cpu')


def test_plain_patch_uses_function_without_array_history():
    target = SimpleNamespace(function=lambda value: value)
    original = target.function
    replacement = lambda value: value + 1
    with vote.PlainPatch.object(target, 'function', side_effect=replacement):
        assert target.function is replacement and not isinstance(target.function, Mock)
        assert target.function(3) == 4 and not hasattr(target.function, 'call_args_list')
    assert target.function is original


def test_effective_report_retains_baseline_and_marks_real_vote_source():
    cfg = baseline()
    source = dict(format='legacy', config=cfg, runtime=dict(restoration_noise=dict(name='old')),
        images=[dict(source='test_169', boundary_naf={'raw_probability_sha256': 'a' * 64})], geometry='old')
    before = deepcopy(source)
    rows = [dict(source='test_169', **{key: 0 for key in vote.COUNT_KEYS})]
    actual = vote.audit_document(source, rows, 'b' * 64)
    assert source == before and actual['baseline_config'] == cfg
    assert actual['config']['submission']['semantic_route'] == vote.PROBABILITY_ROLE
    assert actual['semantic'] == actual['images'][0]['boundary_naf']['raw_probability_role'] == vote.PROBABILITY_ROLE
    assert actual['raw_shared']['raw_image_semantic_forward'] is True
    assert actual['raw_shared']['global_affinity_computed'] is False
    assert vote.audit_document(actual, rows, 'b' * 64) == actual


@pytest.mark.parametrize('change', [
    lambda c: c['submission'].update(native_sizes=[768]),
    lambda c: c['submission']['local_restoration'].update(mode='single'),
    lambda c: c['inference'].update(boundary_threshold=.75),
    lambda c: c['inference'].update(final_min_instance_area=50),
])
def test_geometry_readout_contract_cannot_be_changed(change):
    cfg = baseline()
    change(cfg)
    with pytest.raises(ValueError):
        vote.execution_config(cfg)
