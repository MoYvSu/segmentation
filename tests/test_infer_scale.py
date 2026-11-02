# -*- coding: utf-8 -*-
"""原生视野研究的CPU合同；不读赛题图或训练权重。"""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import infer_scale as scale


def baseline():
    return scale.entry.load_config(str(scale.ROOT / 'config/inference/area_floor_200.yaml'))


@pytest.mark.parametrize('size', [512, 0, True, 768., [768], None])
def test_reject_unsupported_or_ambiguous_scale(size):
    with pytest.raises(ValueError, match='exactly 1024 or 768'):
        scale.research_contract(size)


@pytest.mark.parametrize('change', [
    lambda c: c['submission'].update(native_sizes=[768]),
    lambda c: c['submission']['local_restoration'].update(mode='single'),
    lambda c: c['affinity_native'].update(overlap=.5),
    lambda c: c['inference'].update(final_min_instance_area=50),
    lambda c: c['inference'].update(final_small_instance_action='legacy_drop'),
    lambda c: c['inference'].update(boundary_threshold=.75),
])
def test_research_keeps_legacy_validation_and_frozen_readout(change):
    cfg = baseline()
    change(cfg)
    with pytest.raises(ValueError):
        scale.validate_baseline(cfg)


@pytest.mark.parametrize('size', [1024, 768])
def test_effective_configuration_is_explicit_and_baseline_is_unchanged(size):
    cfg = baseline()
    before = deepcopy(cfg)
    actual = scale.execution_config(cfg, size)
    assert cfg == before and cfg['submission']['native_sizes'] == [1024]
    assert actual['submission']['native_sizes'] == [size]
    assert actual['native_scale_research']['support_reference'] == f'own_N1_{size}_pre_NAF'
    actual.pop('native_scale_research')
    actual['submission']['native_sizes'] = [1024]
    assert actual == cfg


def test_audit_names_actual_execution_and_preserves_original_configuration():
    cfg = baseline()
    document = dict(format=scale.entry.FORMAT, config=cfg,
                    geometry='original R1 native1024, D5a first_x0')
    before = deepcopy(document)
    result = scale.audit_document(document, 768, 'a' * 64)
    assert document == before
    assert result['format'] == scale.FORMAT
    assert result['legacy_format'] == scale.entry.FORMAT
    assert result['baseline_config'] == cfg
    assert result['config']['submission']['native_sizes'] == [768]
    assert 'native768' in result['geometry']
    assert result['research_entry_sha256'] == 'a' * 64
    assert scale.audit_document(result, 768, 'a' * 64) == result


@pytest.mark.parametrize('size', [1024, 768])
def test_single_scale_predict_keeps_decode_vote_cleanup_order_and_fixed_inputs(monkeypatch, size):
    cfg, records, events = baseline(), [], []
    before = deepcopy(cfg)
    rgb = np.zeros((3, 4, 3), np.uint8)
    ids = np.ones((3, 4), np.uint16)
    boundary = torch.zeros((1, 1, 3, 4), dtype=torch.float32)
    fixed, windows, callback = {}, [], object()
    sentinel = object()

    def decode(semantic, actual_boundary, actual_cfg, *, original_decode=None):
        assert original_decode is sentinel
        assert actual_boundary is boundary
        assert actual_cfg['submission']['native_sizes'] == [size]
        events.append('decode')
        return ids, {'1': 1}

    def views(model, restorer, path, actual_cfg, device, seed, sizes, **kwargs):
        assert sizes == [size] and device == 'cpu'
        assert seed == scale.entry.source_seed('test_169', 'semifinal_150')
        assert actual_cfg['submission']['native_sizes'] == [size]
        assert actual_cfg['submission']['restoration_noise'] == cfg['submission']['restoration_noise']
        assert kwargs == dict(local_restorer=None, local_noise_mode='zero',
                              fixed_audit=fixed, local_audit=windows, analysis_callback=callback)
        events.append('frontend')
        windows.append(dict(size=size, noise_mode='zero'))
        actual_ids, classes = scale.entry.views.decode_native(None, boundary, actual_cfg)
        events.append('raw_vote')
        return rgb, {f'patch{size}': dict(instances=actual_ids, classes=classes, boundary=boundary[0, 0].numpy())}

    def finalize(result, actual_cfg):
        assert scale.resolve_instance_area_policy(actual_cfg['inference']) == scale.FINAL_POLICY
        events.append('final200')
        return result

    monkeypatch.setattr(scale.entry.views, 'decode_native', sentinel)
    monkeypatch.setattr(scale.entry.views, 'predict_views', views)
    monkeypatch.setattr(scale.entry, 'decode_reflect', decode)
    monkeypatch.setattr(scale.entry, 'finalize_instance_area', finalize)
    predictor = scale.scale_predict(size, records)
    actual_rgb, result = predictor(None, None, None, Path('test_169.jpg'), cfg, 'cpu',
        fixed_audit=fixed, local_audit=windows, analysis_callback=callback)
    assert actual_rgb is rgb and result['instances'] is ids
    assert events == ['frontend', 'decode', 'raw_vote', 'final200']
    assert scale.entry.views.decode_native is sentinel
    assert cfg == before
    assert records == [dict(source='test_169', native_size=size, view_key=f'patch{size}',
                            local_window_count=1, single_decode_and_raw_vote='enforced_by_NAF_wrapper')]


@pytest.mark.parametrize('size', [1024, 768])
def test_wrong_view_or_actual_window_scale_cannot_be_mislabeled(monkeypatch, size):
    ids = np.ones((3, 4), np.uint16)
    rgb = np.zeros((3, 4, 3), np.uint8)

    def views(*args, **kwargs):
        return rgb, {'patch512': dict(instances=ids, classes={'1': 1})}

    monkeypatch.setattr(scale.entry.views, 'predict_views', views)
    with pytest.raises(RuntimeError, match='unexpected view keys'):
        scale.scale_predict(size, [])(None, None, None, Path('test_169.jpg'), baseline(), 'cpu')


@pytest.mark.parametrize('field,value,pattern', [
    ('expected_model_sha', '0' * 64, 'old 20a59'),
    ('seed_support', 'none', 'original-zero'),
])
def test_model_or_support_replacement_rejected_before_loading(tmp_path, field, value, pattern):
    args = Namespace(native_size=768, config=str(scale.ROOT / 'config/inference/area_floor_200.yaml'),
                     output_dir=str(tmp_path / 'candidate'))
    setattr(args, field, value)
    with pytest.raises(ValueError, match=pattern):
        scale.run(args, device='cpu')
