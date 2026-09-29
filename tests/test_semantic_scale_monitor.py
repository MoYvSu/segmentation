# -*- coding: utf-8 -*-
"""CPU 小模型验证过程图的同区域对齐、ignore 色表与训练状态隔离。"""
import random

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import tools.semantic_scale_monitor as monitor_module


class ToyDataset:
    def __init__(self):
        self.epoch = 7
        self.base_dataset = [None, None]
        self.image = torch.linspace(.1, .9, 64)[None, None].expand(3, 64, 64).clone()
        self.valid = torch.zeros(1, 64, 64, dtype=torch.bool)
        self.valid[:, 4:60, 8:60] = True
        self.target = (self.image[:1] > .5).float()
        self.target[:, 20:24, 20:24] = -1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        # 模拟误用全局随机流的 dataset，monitor 仍不能改变后续训练随机数。
        torch.rand(1)
        np.random.rand()
        random.random()
        return {'image': self.image.clone(), 'valid_content': self.valid.clone(),
                'semantic_target': self.target.clone(),
                'semantic_valid_content': self.valid & (self.target >= 0),
                'name': f'train_{index:02d}.png'}


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(8.))
        self.dropout = nn.Dropout(.1)
        self.inputs = []
        self.fail = False

    def forward(self, features, image):
        assert not self.training and not self.dropout.training
        self.inputs.append(image.detach().clone())
        if self.fail:
            raise RuntimeError('diagnostic failure')
        return (features[:, :1] - .5) * self.weight


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.semantic_decoder = ToyDecoder()
        self.train()
        self.encoder.eval()
        self.semantic_decoder.dropout.eval()


def test_fixed_training_views_share_roi_and_preserve_modes_rng_and_dataset(monkeypatch, tmp_path):
    model, restorer, dataset = ToyModel(), nn.Sequential(nn.Identity(), nn.Dropout(.2)), ToyDataset()
    restorer[0].eval()
    modes = [(module, module.training) for root in (model, restorer) for module in root.modules()]
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    calls = []

    def fake_restore(module, image, seed):
        assert not any(layer.training for layer in module.modules())
        calls.append((seed, image.detach().clone()))
        torch.rand(1)
        return image

    monkeypatch.setattr(monitor_module, 'restore_first', fake_restore)
    cfg = {'semantic_adaptation': {'seed': 7}, 'semantic_scale': {'crop_size': 32}}
    report = monitor_module.scale_monitor(model, restorer, dataset, cfg, tmp_path, 0)
    assert dataset.epoch == 7
    assert torch.equal(rng, torch.get_rng_state())
    assert random.getstate() == python_rng
    actual_numpy = np.random.get_state()
    assert numpy_rng[0] == actual_numpy[0] and np.array_equal(numpy_rng[1], actual_numpy[1])
    assert numpy_rng[2:] == actual_numpy[2:]
    assert all(module.training == state for module, state in modes)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items())
    assert len(calls) == 2  # 每个训练源只恢复一次，局部视图不能二次恢复。
    assert len(report['samples']) == 2
    assert report['single_restoration_before_crop'] is True
    for index, row in enumerate(report['samples']):
        y, x, h, w = row['roi_top_left_height_width']
        assert (y, x, h, w) == (16, 16, 32, 32)
        expected = F.interpolate(dataset.image[None, :, y:y + h, x:x + w],
                                 size=(64, 64), mode='bilinear', align_corners=False)
        assert torch.equal(model.semantic_decoder.inputs[2 * index], dataset.image[None])
        assert torch.equal(model.semantic_decoder.inputs[2 * index + 1], expected)
        assert row['valid_gt_pixels'] == 32 * 32 - 16
        assert row['ignore_pixels'] == 16
        assert row['full_gt_disagreements'] == row['local_gt_disagreements'] == 0
    thumbnail = cv2.imread(str(tmp_path / 'scale_monitor/epoch_000/train_00.png'))
    assert thumbnail.shape == (350, 1280, 3)
    # ROI 内 (20,20) 对应第四列中坐标 (40,40)，与顶部标题错开。
    assert np.array_equal(thumbnail[30 + 45, 960 + 45], monitor_module.IGNORE_RGB)
    repeat = monitor_module.scale_monitor(model, restorer, dataset, cfg, tmp_path, 5)
    assert report['samples'] == repeat['samples']
    assert [seed for seed, _ in calls] == [9000007, 9000008, 9000007, 9000008]


def test_restore_state_even_if_diagnostic_forward_fails(monkeypatch, tmp_path):
    model, restorer, dataset = ToyModel(), nn.Identity(), ToyDataset()
    model.semantic_decoder.fail = True
    original_modes = [(module, module.training) for root in (model, restorer) for module in root.modules()]
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(monitor_module, 'restore_first', lambda module, image, seed: image)
    with pytest.raises(RuntimeError, match='diagnostic failure'):
        monitor_module.scale_monitor(model, restorer, dataset,
            {'semantic_adaptation': {'seed': 1}, 'semantic_scale': {'crop_size': 32}}, tmp_path, 1)
    assert dataset.epoch == 7 and torch.equal(rng, torch.get_rng_state())
    assert all(module.training == state for module, state in original_modes)


def test_ignore_is_grey_not_assigned_to_a_phase():
    target = np.array([[0., 1., -1., 1., np.nan]], dtype=np.float32)
    result = monitor_module.semantic_rgb(target, np.array([[True, True, True, False, True]]))
    assert np.array_equal(result[0, 0], monitor_module.PEARLITE_RGB)
    assert np.array_equal(result[0, 1], monitor_module.FERRITE_RGB)
    assert np.all(result[0, 2:] == monitor_module.IGNORE_RGB)
    assert np.array_equal(monitor_module.probability_rgb([[0., 1.]])[0],
                          np.stack([monitor_module.PEARLITE_RGB, monitor_module.FERRITE_RGB]))


def test_centre_roi_respects_shifted_content_and_four_pixel_alignment():
    valid = torch.zeros(1, 24, 24, dtype=torch.bool)
    valid[:, 7:20, 3:17] = True
    y, x, h, w = monitor_module.centre_roi(valid, (48, 48), 32)
    assert all(value % 4 == 0 for value in (y, x, h, w))
    assert y >= 14 and x >= 6 and y + h <= 40 and x + w <= 34 and h == w
    with pytest.raises(ValueError, match='nonempty'):
        monitor_module.centre_roi(torch.zeros_like(valid), (48, 48), 32)
