# -*- coding: utf-8 -*-
"""擦断过程图必须同时保留干净输入输出、独立人工/天然缺口与固定坐标。"""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import tools.boundary_rgb_monitor as module


class IdentityBoundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.input_mode = 'boundary'

    def forward(self, value):
        return value + self.anchor * 0


def sample():
    side = 64
    target = torch.zeros(1, side, side)
    target[:, 6:58, 32] = 1
    clean = torch.zeros_like(target)
    clean[:, 6:58, 28:36] = 1
    clean[:, 12:20] = 0
    mask = torch.zeros_like(target)
    mask[:, 28:44, 24:41] = 1
    actual = clean.masked_fill(mask.bool(), 0)
    near = lambda a: torch.nn.functional.max_pool2d(a[None], 5, 1, 2)[0] > .5
    artificial = target.bool() & near(clean) & ~near(actual)
    return dict(input=actual, input_boundary=actual, clean_input_boundary=clean,
        erasure_mask=mask, artificial_gap_mask=artificial.float(), target=target,
        valid=torch.ones_like(target), gap_mask=(target.bool() & ~near(clean)).float(),
        raw_input_boundary=clean, raw_gap_mask=(target.bool() & ~near(clean)).float(),
        rgb=torch.zeros(3, side, side), source='train_001',
        meta=dict(epoch=0, geometry=dict(rotation=0, flip=False),
            erasure=dict(requested=True, applied=True, reason='applied', segments=[dict(provenance='B')])))


def config():
    normalization = dict(mode='skeleton_dilate', kernel_size=8)
    return dict(boundary_naf=dict(erasure=dict(enabled=True, version='native_supported_v1'),
        input_normalization=normalization,
        inference=dict(input_mode='boundary', input_normalization=deepcopy(normalization)),
        monitor=dict(pair_indices=[0], erasure_pair_indices=[1], thumbnail_width=64,
                     full_readout_steps=[], source_names=[], control_manifests=[])))


def test_enabled_monitor_compares_same_fixed_cohorts_and_keeps_clean_prediction(tmp_path, monkeypatch):
    fixed = sample()
    captured = []
    real_panels = module.panels
    def capture(path, title, cells, *args, **kwargs):
        captured.append([name for name, _ in cells])
        return real_panels(path, title, cells, *args, **kwargs)
    monkeypatch.setattr(module, 'panels', capture)
    monkeypatch.setattr(module, 'original_view', lambda *_: np.ones((64, 64), bool))
    monkeypatch.setattr(module, '_fixed_controls', lambda *_: pytest.fail('paired monitor cannot run full controls'))
    model = IdentityBoundary().train()
    result = module.save_monitor(model, config(), tmp_path, 5, {0:fixed, 1:fixed})
    assert model.training and result['erasure_training'] and not result['full_readout']
    assert [r['monitor_group'] for r in result['rows']] == ['common_control', 'erasure_extra']
    row = result['rows'][0]['erasure']
    assert row['cohorts']['artificial_missing']['points'] > 0
    assert row['cohorts']['artificial_missing']['clean_output_above065'] == 1
    assert row['cohorts']['artificial_missing']['damaged_output_above065'] == 0
    assert row['cohorts']['normalized_natural_missing']['points'] > 0
    assert row['cohorts']['original_supported']['points'] > row['cohorts']['artificial_missing']['points']
    assert 'Clean standardized T' in captured[0] and 'Erased model input T' in captured[0]
    assert 'Request gray / deleted T red' in captured[0] and 'Same model / clean T' in captured[0]
    assert Image.open(tmp_path/'monitor/s000005/pair_000.jpg').size[0] == 8*64
    assert (tmp_path/'monitor/s000005/pair_001_local.jpg').is_file()
    stored = json.loads((tmp_path/'monitor/s000005/summary.json').read_text())
    assert stored['common_pair_indices'] == [0] and stored['erasure_pair_indices'] == [1]


def test_misaligned_damage_or_artificial_cohort_is_rejected():
    fixed = sample()
    prediction = fixed['input'][0].numpy()
    clean_prediction = fixed['clean_input_boundary'][0].numpy()
    fixed['erasure_mask'] = torch.roll(fixed['erasure_mask'], 12, -2)
    with pytest.raises(ValueError, match='mask不能重建'):
        module._erasure_pair_diagnostics(fixed, prediction, clean_prediction,
            fixed['target'][0].numpy(), np.ones((64,64),bool))
    fixed = sample()
    fixed['artificial_gap_mask'].zero_()
    with pytest.raises(ValueError, match='人工缺口mask'):
        module._erasure_pair_diagnostics(fixed, prediction, clean_prediction,
            fixed['target'][0].numpy(), np.ones((64,64),bool))
