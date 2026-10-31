# -*- coding: utf-8 -*-
"""过程图不得改实例或把未知区变成训练答案。"""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from tools.semantic_dino_monitor import build_monitor, class_map, prepare_monitor_folder, vote_fixed_instances


def test_unknown_and_positive_id_semantics_are_preserved():
    ids = np.array([[0, 2, 7], [7, 2, 0]], dtype=np.uint16)
    np.testing.assert_array_equal(class_map(ids, {2: 0, 7: 1}), [[-1, 0, 1], [1, 0, -1]])
    with pytest.raises(ValueError, match='integer P=0/F=1'):
        class_map(ids, {2: True, 7: 1})


def test_fixed_vote_keeps_geometry_and_exact_half_is_pearlite():
    ids = np.array([[0, 1, 1], [2, 2, 0]], np.uint16)
    original = ids.copy()
    p = np.array([[1., .4, .6], [.6, .8, 0.]], np.float32)
    classes, scores = vote_fixed_instances(ids, p)
    assert classes == {'1': 0, '2': 1}
    assert scores['1'] == .5
    np.testing.assert_array_equal(ids, original)


class ConstantSemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))
        self.backbone = nn.Identity()

    def forward(self, rgb):
        return torch.zeros_like(rgb[:, :1]) + self.offset


def test_monitor_uses_existing_control_and_never_rewrites_its_png(monkeypatch, tmp_path):
    from data import dino_semantic as data
    rgb = np.full((12, 20, 3), 150, np.uint8)
    ids = np.ones(rgb.shape[:2], np.uint16)
    ids[:, 10:] = 2
    gt = ids.astype(np.int32)
    gt[:2] = 0

    class Dataset:
        sources = ['train_001']
        def __init__(self, config): pass
        def load_source(self, source):
            return rgb.copy(), gt.copy(), {1: 1, 2: 0}, gt > 0

    monkeypatch.setattr(data, 'ProcessedSemanticDataset', Dataset)
    test_dir, control = tmp_path/'images', tmp_path/'control'
    test_dir.mkdir(); control.mkdir()
    assert cv2.imwrite(str(test_dir/'test_116.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    png = control/'test_116_inst.png'
    assert cv2.imwrite(str(png), ids)
    labels = control/'test_116_class.json'
    labels.write_text(json.dumps({'1': 1, '2': 0}), encoding='utf-8')
    before = png.read_bytes(), labels.read_bytes()
    config = {'paths': {'project_root': str(tmp_path)}, 'semantic_dino': {'input_size': 32,
        'monitor': {'train_sources': ['train_001'], 'test_sources': ['test_116'],
                    'test_dir': str(test_dir), 'control_dir': str(control), 'thumbnail_width': 100}}}
    callback = build_monitor(config)
    model = ConstantSemantic().eval()
    state = deepcopy(model.state_dict())
    callback(model, 0, tmp_path/'run')
    folder = tmp_path/'run/monitor/e000'
    report = json.loads((folder/'monitor.json').read_text(encoding='utf-8'))
    assert len(list(folder.glob('*.jpg'))) == 2
    assert (folder/'test_116_classes.json').exists()
    row = [r for r in report['rows'] if r['kind'] == 'test'][0]
    assert row['fixed_png_sha256'] == hashlib.sha256(before[0]).hexdigest()
    assert row['instance_pixels_unchanged'] and not row['test_gt_used']
    assert (folder/'test_116_classes.json').read_text() == '{"1": 0, "2": 0}'
    assert (png.read_bytes(), labels.read_bytes()) == before
    assert all(torch.equal(state[k], model.state_dict()[k]) for k in state)
    labels.write_text('{"1": 0, "2": 0}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Original N0 control changed'):
        callback(model, 1, tmp_path/'run')


def test_monitor_requires_eval(monkeypatch, tmp_path):
    from data import dino_semantic as data
    class Dataset:
        sources = []
        def __init__(self, config): pass
    monkeypatch.setattr(data, 'ProcessedSemanticDataset', Dataset)
    callback = build_monitor({'semantic_dino': {'monitor': {'train_sources': [], 'test_sources': []}}})
    with pytest.raises(RuntimeError, match='eval'):
        callback(ConstantSemantic().train(), 0, tmp_path/'run')


def test_interrupted_monitor_is_preserved_before_redrawing(tmp_path):
    folder, backup = prepare_monitor_folder(tmp_path/'run', 1)
    assert backup is None
    (folder/'partial.jpg').write_bytes(b'preserve partial artifact')
    fresh, backup = prepare_monitor_folder(tmp_path/'run', 1)
    assert fresh == folder and backup is not None and fresh.is_dir()
    assert not list(fresh.iterdir())
    assert (backup/'partial.jpg').read_bytes() == b'preserve partial artifact'
    assert backup.parent == fresh.parent
