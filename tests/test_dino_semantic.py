# -*- coding: utf-8 -*-
"""CPU 契约检查：假骨干仅用于软件验证，不代表 DINO 训练或精度证据。"""
from copy import deepcopy
import json
import random

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from data.dino_semantic import (
    ProcessedSemanticDataset, canonical_sha, collate_semantic, epoch_schedule,
    load_processed_source, prepare_semantic_view,
)
from models.dino_semantic import FrozenDinoSemantic, load_dino_backbone, state_sha256
from train_dino_semantic import (
    capture_rng, learning_rate, load_resume, optimization_step, restore_rng,
    reconcile_step_log, run_monitor, save_checkpoint, smoke, train, validate_recipe,
)
from train_direct_semantic_affinity import build_semantic_criterion, compute_semantic_loss
from utils.config import load_config


class FakeBackbone(nn.Module):
    embed_dim = 16
    patch_size = 16

    def __init__(self):
        super().__init__()
        # 固定假特征避免随机初始化消耗影响 head/resume 测试。
        self.scale = nn.Parameter(torch.linspace(.5, 1.5, 16).view(1, 16, 1, 1))
        self.bn = nn.BatchNorm2d(16)
        self.last_input = None

    def get_intermediate_layers(self, x, *, n, reshape, norm):
        assert list(n) == [2, 5, 8, 11] and reshape and norm
        self.last_input = x.detach().clone()
        f = F.avg_pool2d(x.mean(1, keepdim=True), 16).repeat(1, 16, 1, 1) * self.scale
        f = self.bn(f)
        return tuple(f + i * .1 for i in range(4))


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def toy_source(tmp_path, name='train_001'):
    image = np.full((32, 64, 3), 130, np.uint8)
    image[:, 32:] = 200
    image_path = tmp_path / (name + '.png')
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    gt = np.zeros((32, 64), np.int32)
    gt[2:30, 2:30], gt[2:30, 34:62] = 1, 2
    original = gt > 0
    original[2:4, 2:30] = False
    path = tmp_path / (name + '_gt.npz')
    np.savez_compressed(path, instance_map=gt, original_covered=original,
                        filled=(gt > 0) & ~original, residual_unknown=gt == 0)
    path.with_name(name + '_class.json').write_text(json.dumps({'1': 0, '2': 1}), encoding='utf-8')
    return image_path, path


def sample_batch():
    rgb = np.full((32, 64, 3), 128, np.uint8)
    ids = np.zeros((32, 64), np.int32)
    ids[2:30, 2:30], ids[2:30, 34:62] = 1, 2
    s = prepare_semantic_view(rgb, ids, {1: 0, 2: 1}, input_size=64)
    s.update(source='train_001', mode='whole', view_seed=4, draw=0)
    return collate_semantic([s, deepcopy(s)])


def recipe():
    config = load_config('config/train/semantic_dino.yaml')
    cfg = deepcopy(config['semantic_dino'])
    cfg.update(effective_batch_size=2, micro_batch_size=2)
    return config, cfg


def test_frozen_shape_normalization_gradient_and_random_head():
    model = FrozenDinoSemantic(FakeBackbone(), head_width=8, head_seed=5)
    identical = FrozenDinoSemantic(FakeBackbone(), head_width=8, head_seed=5)
    assert state_sha256(model.head) == state_sha256(identical.head)
    before = state_sha256(model.backbone)
    model.train()
    assert model.head.training and not model.backbone.training and not model.backbone.bn.training
    x = torch.full((2, 3, 64, 32), .5, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 1, 64, 32) and out.dtype == torch.float32
    out.square().mean().backward()
    assert any(p.grad is not None for p in model.head.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    assert x.grad is None
    assert state_sha256(model.backbone) == before
    expected = (.5 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(model.backbone.last_input[0, :, 0, 0], expected)


@pytest.mark.parametrize('shape,value', [((1, 3, 31, 32), .5), ((1, 1, 32, 32), .5), ((1, 3, 32, 32), 1.5)])
def test_invalid_rgb_rejected(shape, value):
    with pytest.raises(ValueError):
        FrozenDinoSemantic(FakeBackbone(), head_width=8)(torch.full(shape, value))


def test_reject_nonlocal_or_unapproved_backbone(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dino_backbone(tmp_path, tmp_path/'weights.pt', '0'*64)
    (tmp_path/'dinov3/hub').mkdir(parents=True)
    (tmp_path/'dinov3/hub/backbones.py').write_text('# deliberately not executable')
    with pytest.raises(ValueError, match='explicit SHA256'):
        load_dino_backbone(tmp_path, tmp_path/'weights.pt', None)
    (tmp_path/'weights.pt').write_bytes(b'not approved')
    with pytest.raises(ValueError, match='SHA256 differs'):
        load_dino_backbone(tmp_path, tmp_path/'weights.pt', '0'*64)


def test_processed_gt_and_unknown_padding(tmp_path):
    image, path = toy_source(tmp_path)
    rgb, ids, classes, original = load_processed_source(image, path)
    assert classes == {1: 0, 2: 1} and int(original.sum()) < int((ids > 0).sum())
    sample = prepare_semantic_view(rgb, ids, classes, 64)
    assert sample['geometry']['resized_height'] == 32
    assert sample['image'].shape == (3, 64, 64)
    assert (sample['semantic_target'][:, 32:] == -1).all()
    assert (sample['semantic_instance_map'][32:] == 0).all()
    assert not sample['semantic_valid_content'][:, 32:].any()
    assert torch.all(sample['semantic_target'][0][sample['semantic_instance_map'] == 0] == -1)
    np.testing.assert_array_equal(sample['image'][:, 32].numpy(), sample['image'][:, 31].numpy())


@pytest.mark.parametrize('corruption', ['old', 'unknown', 'filled', 'classbool'])
def test_corrupt_gt_is_rejected(tmp_path, corruption):
    image, path = toy_source(tmp_path)
    with np.load(path) as z:
        data = {k: z[k].copy() for k in z.files}
    if corruption == 'old':
        data.pop('original_covered')
    elif corruption == 'unknown':
        data['residual_unknown'][0, 0] = False
    elif corruption == 'filled':
        data['filled'][0, 0] = True
    else:
        path.with_name('train_001_class.json').write_text('{"1": false, "2": 1}')
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        load_processed_source(image, path)


def test_native_crop_rotation_alignment():
    ids = np.tile(np.array([0, 1, 2, 0], np.int32), (4, 1))
    rgb = np.repeat((ids * 100).astype(np.uint8)[..., None], 3, 2)
    sample = prepare_semantic_view(rgb, ids, {1: 0, 2: 1}, 16,
                                   box=[0, 1, 4, 2], rotation=1, flip=True)
    transformed = np.rot90(ids[:, 1:3])[:, ::-1]
    expected = cv2.resize(transformed, (16, 8), interpolation=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(sample['semantic_instance_map'][:8].numpy(), expected)
    assert (sample['semantic_target'][:, 8:] == -1).all()
    assert set(sample['semantic_target'].unique().tolist()) <= {-1., 0., 1.}


def test_balanced_schedule_and_dataset_manifest(tmp_path):
    names = [f'train_{i:03d}' for i in range(32)]
    plan = epoch_schedule(names, 1, 20261025)
    assert len(plan) == 256 and plan == epoch_schedule(names, 1, 20261025)
    assert plan != epoch_schedule(names, 2, 20261025)
    for name in names:
        assert sum(r['source'] == name and r['mode'] == 'whole' for r in plan) == 4
        assert sum(r['source'] == name and r['mode'] == 'native' for r in plan) == 4
    toy_source(tmp_path)
    cfg = dict(paths={'project_root': str(tmp_path)}, semantic_dino=dict(
        raw_data_dir='.', completed_gt_dir='.', expected_sources=1, input_size=64,
        seed=1, views_per_source=8, native_crop_size=16))
    dataset = ProcessedSemanticDataset(cfg)
    assert dataset.sources == ['train_001'] and dataset.identity['test_training'] is False
    assert len(dataset.identity['new_gt_manifest_sha256']) == 64
    whole, native = dataset[0], dataset[1]
    assert whole['mode'] == 'whole' and native['mode'] == 'native'
    assert native['geometry']['box'][2:] == [16, 16]
    assert (native['semantic_target'] >= 0).any()
    assert torch.equal(native['image'], dataset[1]['image'])
    assert dataset.plan_receipt()['view_counts'] == {'whole': 4, 'native': 4}
    assert dataset.verify_sources() == dataset.identity['new_gt_manifest_sha256']
    (tmp_path/'train_001_class.json').write_text('{"1": 1, "2": 1}')
    with pytest.raises(RuntimeError, match='Training input changed'):
        dataset.verify_sources()


def test_real_semantic_loss_ignores_unknown_and_padding():
    config, _ = recipe()
    criterion = build_semantic_criterion(config, 'cpu')
    batch = sample_batch()
    logits = torch.zeros_like(batch['semantic_target'], requires_grad=True)
    loss = compute_semantic_loss(criterion, logits, batch)
    loss.backward()
    unknown = batch['semantic_target'] < 0
    assert torch.isfinite(loss) and torch.count_nonzero(logits.grad[unknown]) == 0
    changed = logits.detach().clone()
    changed[unknown] = 20
    torch.testing.assert_close(compute_semantic_loss(criterion, changed, batch), loss.detach())


def test_exact_optimizer_resume_and_identity_rejection(tmp_path):
    config, cfg = recipe()
    model = FrozenDinoSemantic(FakeBackbone(), head_width=8, head_seed=42)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg['learning_rate'], eps=cfg['adam_eps'])
    criterion = build_semantic_criterion(config, 'cpu')
    batch = sample_batch()
    row = optimization_step(model, optimizer, criterion, [batch], cfg, 0)
    assert row['grad_norm_before_clip'] > 0 and not row['backbone_has_grad']
    identity = dict(backbone={'backbone_state_sha256': state_sha256(model.backbone)}, fixture='CPU only')
    path = tmp_path/'resume.pt'
    save_checkpoint(path, model, optimizer, identity, 1, cfg, [0])
    saved_head = state_sha256(model.head)
    expected_random = (random.random(), float(np.random.rand()), float(torch.rand(())))
    optimization_step(model, optimizer, criterion, [batch], cfg, 1)
    expected_final = state_sha256(model.head)
    resumed = FrozenDinoSemantic(FakeBackbone(), head_width=8, head_seed=42)
    resumed_opt = torch.optim.AdamW(resumed.head.parameters(), lr=9.)
    step, monitors = load_resume(path, resumed, resumed_opt, identity, cfg)
    assert step == 1 and monitors == [0] and state_sha256(resumed.head) == saved_head
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected_random
    optimization_step(resumed, resumed_opt, criterion, [batch], cfg, 1)
    assert state_sha256(resumed.head) == expected_final
    with pytest.raises(ValueError, match='identity differs'):
        load_resume(path, resumed, resumed_opt, {**identity, 'changed': True}, cfg)


def test_monitor_restores_rng_and_train_mode(tmp_path):
    model = FrozenDinoSemantic(FakeBackbone(), head_width=8).train()
    rng = capture_rng()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng(rng)
    def callback(m, epoch, out):
        assert not m.training and not torch.is_grad_enabled() and epoch == 1
        assert out == tmp_path
        random.random(); np.random.rand(); torch.rand(2)
        return m(torch.full((1, 3, 32, 32), .5)).mean().item()
    assert np.isfinite(run_monitor(callback, model, 1, tmp_path))
    assert model.training and not model.backbone.training
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected


def test_default_recipe_five_epoch_budget_twenty_epoch_schedule():
    config = load_config('config/train/semantic_dino.yaml')
    cfg = validate_recipe(config)
    assert cfg['epochs'] * cfg['updates_per_epoch'] == 320
    assert 320 * cfg['effective_batch_size'] == 1280
    assert learning_rate(319, cfg) > .8 * cfg['learning_rate']
    assert learning_rate(0, cfg) == cfg['learning_rate']
    with pytest.raises(ValueError):
        learning_rate(1280, cfg)


def test_empty_gt_view_is_only_preprocessing_for_unlabelled_monitor():
    sample = prepare_semantic_view(np.full((20, 40, 3), 100, np.uint8),
                                   np.zeros((20, 40), np.int32), {}, input_size=32)
    assert sample['image'].shape == (3, 32, 32)
    assert (sample['semantic_target'] == -1).all()
    assert not sample['semantic_valid_content'].any()


def test_tiny_trainer_stage_resume_and_real_smoke_path_with_fake_backbone(tmp_path):
    """整条训练循环的 CPU 集成；fixture 假骨干绝不被 CLI 正式加载。"""
    data_root = tmp_path/'data'
    data_root.mkdir()
    toy_source(data_root)
    config = load_config('config/train/semantic_dino.yaml')
    config['paths']['project_root'] = str(tmp_path)
    cfg = config['semantic_dino']
    cfg.update(raw_data_dir='data', completed_gt_dir='data', expected_sources=1,
               input_size=32, native_crop_size=16, effective_batch_size=2, micro_batch_size=1,
               updates_per_epoch=4, epochs=1, head_width=8, seed=12)
    cfg['monitor']['test_sources'] = []
    dataset = ProcessedSemanticDataset(config)
    model = FrozenDinoSemantic(FakeBackbone(), head_width=8, head_seed=12)
    identity = dict(backbone_state_sha256=state_sha256(model.backbone), fixture='CPU fake only')
    calls = []
    def monitor(m, epoch, out):
        assert not torch.is_grad_enabled() and not m.training
        calls.append(epoch)
    run = tmp_path/'run'
    train(model, dataset, config, run, identity, monitor)
    complete = json.loads((run/'complete.json').read_text())
    assert complete['updates'] == 4 and complete['windows'] == 8
    assert calls == [0, 1] and complete['geometry_modified'] is False
    cfg['epochs'] = 2
    resumed = FrozenDinoSemantic(FakeBackbone(), head_width=8, head_seed=12)
    train(resumed, dataset, config, run, identity, monitor, resume=run/'last.pt')
    assert json.loads((run/'complete.json').read_text())['updates'] == 8
    assert calls == [0, 1]  # e1 不重复渲染；e2 不在预声明 monitor 表中。
    fresh = FrozenDinoSemantic(FakeBackbone(), head_width=8, head_seed=12)
    report = smoke(fresh, dataset, config, tmp_path/'smoke', identity)
    assert report['optimizer_updates'] == 2 and report['strict_reload_prediction_equal']
    assert report['formal_training_complete'] is False


def test_lagging_checkpoint_preserves_log_tail_and_replays_prefix(tmp_path):
    path = tmp_path/'steps.jsonl'
    path.write_text(''.join(json.dumps({'global_step': i})+'\n' for i in range(1, 5)))
    reconcile_step_log(tmp_path, 2)
    assert [json.loads(line)['global_step'] for line in path.read_text().splitlines()] == [1, 2]
    receipt = json.loads((tmp_path/'resume_log_repair.json').read_text())
    assert receipt['preserved_updates'] == 2
    assert [json.loads(line)['global_step'] for line in (tmp_path/receipt['preserved_tail_file']).read_text().splitlines()] == [3, 4]
    with pytest.raises(ValueError, match='behind the resume checkpoint'):
        reconcile_step_log(tmp_path, 3)
    path.write_text('{"global_step": 1}\n{"global_step": 1}\n')
    with pytest.raises(ValueError, match='unique consecutive'):
        reconcile_step_log(tmp_path, 1)
