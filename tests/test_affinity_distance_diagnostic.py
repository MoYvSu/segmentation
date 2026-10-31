# -*- coding: utf-8 -*-
"""距离诊断协调器的状态恢复和原标可信域检查，只有虚构图样。"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn

from tools.run_affinity_distance_diagnostic import TrustedManualView, gradient_state, distance_reader_class, canonical_sample


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 3)
        self.affinity_decoder = nn.Module()
        self.affinity_decoder.affinity_head = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        self.eval().requires_grad_(False)


@pytest.mark.parametrize('raise_inside', [False, True])
def test_gradient_context_restores_all_state_and_rng(raise_inside):
    model = TinyModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    flags = [p.requires_grad for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.random.get_rng_state().clone()
    precision = torch.get_float32_matmul_precision()
    flags_before = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        with gradient_state(model, 999) as parameters:
            assert list(parameters) == ['affinity_decoder.affinity_head.0.weight']
            assert model.affinity_decoder.training and not model.encoder.training
            assert sum(p.requires_grad for p in model.parameters()) == 1
            assert torch.get_float32_matmul_precision() == 'highest'
            assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
            model.affinity_decoder.affinity_head(torch.rand(2, 3, 8, 8))
            if raise_inside:
                raise RuntimeError('synthetic')
    except RuntimeError:
        assert raise_inside
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert flags == [p.requires_grad for p in model.parameters()]
    assert modes == [m.training for m in model.modules()]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert torch.get_float32_matmul_precision() == precision
    assert flags_before == (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark)


class BaseView:
    def __init__(self, sample):
        self.sample = sample
    def __getitem__(self, index):
        return self.sample
    def __len__(self):
        return 1


def test_trusted_manual_separate_from_canonical_targets(tmp_path):
    labels = np.ones((12, 16), np.uint16)
    trusted = np.ones_like(labels, bool)
    trusted[:, 8:] = False
    np.savez(tmp_path / 'fiction_gt.npz', instance_map=labels, original_covered=trusted)
    canonical = torch.ones(512, 512, dtype=torch.int64)
    sample = {'image_path': 'fiction.png', 'crop_box': torch.tensor([0, 0, 12, 16]),
              'affinity_instance_map': canonical, 'instance_map': canonical,
              'affinity_valid_content': torch.ones(1, 512, 512, dtype=torch.bool)}
    base = BaseView(sample)
    wrapper = TrustedManualView(base, tmp_path)
    out = wrapper[0]
    assert torch.equal(out['affinity_instance_map'], canonical)
    assert out['diagnostic_instance_map'].max() == 1
    assert (out['diagnostic_instance_map'] == 0).any()
    assert 'diagnostic_instance_map' not in sample
    wrapper.epoch, wrapper.draw = 3, 7
    assert base.epoch == 3 and base.draw == 7


def test_trusted_diagnostic_maps_are_augmented_with_images():
    keys = distance_reader_class()._SPATIAL_KEYS
    assert 'diagnostic_instance_map' in keys and 'diagnostic_valid_content' in keys
    assert 'clean_image' in keys and 'affinity_instance_map' in keys


def test_sam2_whole_has_no_native_aliases():
    ids = torch.ones(8, 8, dtype=torch.int64)
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    sample = dict(instance_map=ids, valid_content=valid, source_index=0, image_name='fiction.png')
    base = SimpleNamespace(base=SimpleNamespace(rows=[dict(path='fiction.png')], _source_path=lambda row: row['path']))
    out, result_ids, result_valid = canonical_sample(sample, 'pseudo', 'whole', base)
    assert out['image_path'] == 'fiction.png'
    assert result_ids is ids and result_valid is valid
    assert 'image_path' not in sample
