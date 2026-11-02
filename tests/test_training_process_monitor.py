# -*- coding: utf-8 -*-
"""固定过程图的CPU状态隔离测试；不加载SAM2、不训练、不读赛题文件。"""
from copy import deepcopy
import json
import random
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from utils.training_process_monitor import build_process_monitor, monitor_due, _target


class Dataset:
    augment, crop_size, teacher_gt_dir = True, 8, '/private/teacher/derived_maps'

    def __init__(self):
        self.reads = []

    def __len__(self):
        return 3

    def __getitem__(self, index):
        assert self.augment is False and self.crop_size == 0 and index < 2
        self.reads.append(index)
        random.random(); np.random.rand(); torch.rand(1)
        image = torch.linspace(0, 1, 3 * 32 * 40).reshape(3, 32, 40)
        target = torch.zeros(3, 32, 40)
        target[0, :, 20:] = 1
        target[1, :, 19:21] = 1
        target[:2, :2] = -1
        return dict(image=image, target=target, image_path=f'/private/train_{index:03d}.png')


class Encoder(nn.Sequential):
    trainable_lora = True


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.Dropout(.5))
        self.decoder = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2))
        self.register_buffer('frozen_buffer', torch.tensor([3.]))
        self.fail, self.mutate = False, False
        self.shapes = []

    def forward(self, image, output_size=None):
        assert not torch.is_grad_enabled()
        assert all(not layer.training for layer in self.modules())
        assert self.encoder.trainable_lora is False
        random.random(); np.random.rand(); torch.rand(1)
        self.shapes.append(list(image.shape))
        if self.fail:
            raise RuntimeError('synthetic forward failure')
        if self.mutate:
            self.frozen_buffer.add_(1)
        result = self.decoder(self.encoder(image))
        assert tuple(result.shape[-2:]) == output_size
        return result


def capture(model):
    return dict(weights={name: value.clone() for name, value in model.state_dict().items()},
        modes=[(layer, layer.training) for layer in model.modules()], lora=model.encoder.trainable_lora,
        grad=[None if p.grad is None else p.grad.clone() for p in model.parameters()],
        python=random.getstate(), numpy=deepcopy(np.random.get_state()), torch=torch.get_rng_state().clone())


def unchanged(model, before):
    assert all(torch.equal(value, before['weights'][name]) for name, value in model.state_dict().items())
    assert all(layer.training == flag for layer, flag in before['modes'])
    assert model.encoder.trainable_lora == before['lora']
    assert random.getstate() == before['python']
    now = np.random.get_state()
    assert now[0] == before['numpy'][0] and np.array_equal(now[1], before['numpy'][1])
    assert now[2:] == before['numpy'][2:]
    assert torch.equal(torch.get_rng_state(), before['torch'])
    for param, old_grad in zip(model.parameters(), before['grad']):
        assert (old_grad is None and param.grad is None) or torch.equal(param.grad, old_grad)


def test_fixed_monitor_restores_all_rng_modes_parameters_buffers_and_existing_grads(tmp_path):
    model, dataset = Model(), Dataset()
    model.train()
    model.encoder.eval()
    model.decoder[1].eval()  # 混合模式必须逐层恢复，不能只还原顶层training标志。
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = capture(model)
    monitor = build_process_monitor({'enabled': True, 'thumbnail_size': 96}, dataset, tmp_path)
    unchanged(model, before)
    assert dataset.augment is True and dataset.crop_size == 8 and dataset.reads == []
    result = monitor.maybe_save(model, 0, 100, 'cpu')
    unchanged(model, before)
    assert monitor.dataset.reads == [0, 1] and dataset.reads == []
    assert len(result['samples']) == 2 and not result['validation_metrics_computed']
    assert not result['watershed_computed'] and not result['full_arrays_saved']
    folder = tmp_path / 'process_monitor/epoch_000'
    assert len(list(folder.glob('*.png'))) == 2
    assert not list(tmp_path.rglob('*.npz'))
    image = cv2.imread(str(next(folder.glob('*.png'))))
    assert image.shape == (96 + 30, 96 * 5, 3)
    assert model.shapes == [[1, 3, 32, 40], [1, 3, 32, 40]]
    saved = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert saved['samples'][0]['target_source'] == 'model_teacher'
    assert monitor.maybe_save(model, 0, 100, 'cpu')['skipped_existing']
    assert monitor.maybe_save(model, 2, 100, 'cpu') is None
    unchanged(model, before)


def test_failure_restores_state_and_does_not_mark_complete(tmp_path):
    model = Model()
    model.train()
    model.fail = True
    monitor = build_process_monitor({'enabled': True}, Dataset(), tmp_path)
    before = capture(model)
    with pytest.raises(RuntimeError, match='synthetic forward failure'):
        monitor.maybe_save(model, 1, 100, 'cpu')
    unchanged(model, before)
    assert not (tmp_path / 'process_monitor/epoch_001/summary.json').exists()


def test_unexpected_buffer_mutation_is_restored_and_rejected(tmp_path):
    model = Model()
    model.mutate = True
    monitor = build_process_monitor({'enabled': True}, Dataset(), tmp_path)
    before = capture(model)
    with pytest.raises(RuntimeError, match='changed model buffers'):
        monitor.maybe_save(model, 1, 100, 'cpu')
    unchanged(model, before)
    assert not (tmp_path / 'process_monitor/epoch_001/summary.json').exists()


def test_disabled_monitor_does_not_copy_or_read_dataset_or_initialize_CUDA(tmp_path, monkeypatch):
    class Forbidden:
        def __len__(self):
            raise AssertionError('Disabled monitor read the dataset')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: pytest.fail('CPU monitor initialized CUDA'))
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    for options in (None, {}, {'enabled': False}):
        assert build_process_monitor(options, Forbidden(), tmp_path) is None
    monitor = build_process_monitor({'enabled': True}, Dataset(), tmp_path)
    monitor.maybe_save(Model(), 0, 1, 'cpu')


def test_initialized_CUDA_rng_is_restored_without_real_GPU(tmp_path, monkeypatch):
    state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    observed = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [value.clone() for value in state])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda saved: observed.append(saved))
    monitor = build_process_monitor({'enabled': True}, Dataset(), tmp_path)
    monitor.maybe_save(Model(), 0, 1, 'cpu')
    assert len(observed) == 2
    assert all(torch.equal(saved[0], state[0]) for saved in observed)


def test_schedule_and_ignore_palette():
    assert [epoch for epoch in range(101) if monitor_due(epoch, 100)] == [0, 1, *range(10, 101, 10)]
    assert [epoch for epoch in range(4) if monitor_due(epoch, 3)] == [0, 1, 3]
    for args in ((-1, 100), (0, 0), (1, 100, 0)):
        with pytest.raises(ValueError):
            monitor_due(*args)
    values = np.array([[-1, 0, 1]], np.float32)
    image = _target(values, semantic=True)
    assert tuple(image[0, 0]) == (128, 128, 128)
    assert tuple(image[0, 1]) != tuple(image[0, 2])


def test_stage1_main_wires_initial_and_epoch_observations_without_training(tmp_path, monkeypatch):
    import train
    import utils.training_process_monitor as module
    model = Model()
    config = dict(paths={'project_root': str(tmp_path), 'output_dir': 'stage1'}, sam2={'device': 'cpu'},
        data={'batch_size': 1}, train=dict(epochs=2, amp=False, learning_rate=1e-4, weight_decay=1e-4,
            grad_clip=1., process_monitor={'enabled': True}))
    calls = []
    fake_monitor = SimpleNamespace(maybe_save=lambda model, epoch, total, device: calls.append((epoch, total, device)))
    loader = SimpleNamespace(dataset=Dataset())
    monkeypatch.setattr(train, 'load_config', lambda path: config)
    monkeypatch.setattr(train, 'build_model', lambda config, device: model)
    monkeypatch.setattr(train, 'build_dataloaders', lambda config: (loader, loader))
    monkeypatch.setattr(train, 'train_one_epoch', lambda *args, **kwargs: dict(loss=0., seg=0., boundary=0.))
    monkeypatch.setattr(train, 'validate', lambda *args: dict(loss=0., mean_iou=0., boundary_iou=0.,
        mean_dice=0., pearlite_iou=0., ferrite_iou=0.))
    monkeypatch.setattr(train, 'build_checkpoint', lambda **kwargs: {})
    monkeypatch.setattr(train, 'extract_lora_state_dict', lambda model: {})
    monkeypatch.setattr(module, 'build_process_monitor', lambda *args: fake_monitor)
    monkeypatch.setattr('sys.argv', ['train.py'])
    with pytest.warns(UserWarning, match='lr_scheduler.step'):
        train.main()
    assert calls == [(0, 2, 'cpu'), (1, 2, 'cpu'), (2, 2, 'cpu')]
