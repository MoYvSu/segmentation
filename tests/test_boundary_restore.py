# -*- coding: utf-8 -*-
"""边界补全 CPU 契约与机制测试；合成图不代表竞赛效果。"""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch

from models.boundary_restore import (
    BoundaryRestoreNet, boundary_restore_loss, load_boundary_restorer, region_masks, state_sha256,
)
from train_boundary_restore import (
    capture_rng, collate_samples, learning_rate, load_resume, optimization_step,
    reconcile_log, restore_rng, run_monitor, save_checkpoint, smoke, train, validate_recipe,
)
from utils.config import load_config


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def sample(identity=False, shape=(32, 32)):
    h, w = shape
    target = torch.zeros(1, h, w)
    target[:, 4:h-4, w//2] = 1.
    target[:, h//2, 4:w-4] = .8
    valid = torch.zeros_like(target)
    valid[:, 3:h-3, 3:w-3] = 1
    corrupted = target.clone()
    gap, noise = torch.zeros_like(target), torch.zeros_like(target)
    if not identity:
        corrupted[:, h//2-3:h//2+3, w//2] = 0
        gap[:, h//2-3:h//2+3, w//2] = 1
        corrupted[:, 5:9, 5] = .9
        noise[:, 5:9, 5] = 1
    # 未知标签可使用 -1；在任何损失项中都不能被解释成背景。
    target[valid == 0] = -1
    return dict(input=corrupted, target=target, valid=valid, gap_mask=gap, noise_mask=noise,
                identity=torch.tensor(identity), source='train_synthetic', meta={'synthetic': True})


def config_fixture():
    config = load_config('config/train/boundary_restore.yaml')
    cfg = config['boundary_restore']
    cfg.update(crop_size=32, effective_batch_size=2, micro_batch_size=1,
               updates_per_epoch=4, epochs=1, expected_sources=1, seed=9)
    cfg['model']['width'] = 4
    return config, cfg


@pytest.mark.parametrize('shape', [(2, 1, 32, 32), (1, 1, 31, 47), (1, 1, 1, 3)])
def test_exact_initial_identity_and_signed_residual(shape):
    model = BoundaryRestoreNet(width=4, seed=1)
    value = torch.rand(shape)
    value.reshape(-1)[0], value.reshape(-1)[-1] = 0., 1.
    model.train()
    assert torch.equal(model(value), value)
    assert torch.count_nonzero(model.residual(value)) == 0
    assert state_sha256(model) == model.initial_state_sha256
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    with torch.no_grad():
        model.correction.bias.fill_(-1.)
    assert bool((model.residual(value) < 0).all())
    assert model(value).min() >= 0 and model(value).max() <= 1


@pytest.mark.parametrize('value', [torch.zeros(1, 3, 8, 8), torch.full((1, 1, 8, 8), 1.1),
                                  torch.ones(1, 1, 8, 8, dtype=torch.int64), torch.full((1, 1, 8, 8), float('nan'))])
def test_invalid_model_inputs(value):
    with pytest.raises(ValueError):
        BoundaryRestoreNet(width=4)(value)


def test_initialization_does_not_consume_global_rng():
    state = torch.get_rng_state()
    a = BoundaryRestoreNet(width=4, seed=77)
    assert torch.equal(state, torch.get_rng_state())
    b = BoundaryRestoreNet(width=4, seed=77)
    c = BoundaryRestoreNet(width=4, seed=78)
    assert state_sha256(a) == state_sha256(b) != state_sha256(c)


def test_region_loss_not_diluted_by_sparse_gap_and_unknown_gradient_zero():
    batch = collate_samples([sample()])
    prediction = batch['input'].clone().requires_grad_(True)
    loss, stats = boundary_restore_loss(prediction, batch)
    assert stats['regions']['gap']['pixels'] == 6 and stats['regions']['noise']['pixels'] == 4
    assert stats['terms']['gap']['denominator'] == 6
    # 六个断口仍有独立平均项，不被数百个非边界背景像素冲淡。
    assert stats['terms']['gap']['contribution'] > .7
    expected = sum(stats['terms'][name]['contribution'] for name in ('valid', 'gap', 'noise', 'preserve'))
    assert float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[batch['valid'] == 0]) == 0
    changed = prediction.detach().clone()
    changed[batch['valid'] == 0] = 100
    assert float(boundary_restore_loss(changed, batch)[0]) == pytest.approx(float(loss.detach()))


def test_identity_has_no_extra_loss_and_empty_regions_zero():
    batch = collate_samples([sample(True)])
    loss, stats = boundary_restore_loss(batch['input'], batch)
    assert float(loss) == 0 and stats['identity_samples'] == 1
    assert stats['terms']['gap']['denominator'] == 0 and stats['terms']['noise']['contribution'] == 0
    broken = deepcopy(batch)
    broken['gap_mask'][0, 0, 5, 5] = 1
    with pytest.raises(ValueError, match='Identity samples'):
        boundary_restore_loss(batch['input'], broken)


def test_corruption_cannot_supervise_unknown():
    batch = collate_samples([sample()])
    batch['gap_mask'][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match='reliable valid support'):
        boundary_restore_loss(batch['input'], batch)


def test_zero_init_gradient_staging_and_microbatch_equivalence():
    _, cfg = config_fixture()
    full = collate_samples([sample(), sample(True)])
    split = [collate_samples([sample()]), collate_samples([sample(True)])]
    a, b = BoundaryRestoreNet(width=4, seed=9), BoundaryRestoreNet(width=4, seed=9)
    opt_a = torch.optim.AdamW(a.parameters(), lr=cfg['learning_rate'], weight_decay=0.)
    opt_b = torch.optim.AdamW(b.parameters(), lr=cfg['learning_rate'], weight_decay=0.)
    row_a = optimization_step(a, opt_a, [full], cfg, 0)
    row_b = optimization_step(b, opt_b, split, cfg, 0)
    assert row_a['loss'] == pytest.approx(row_b['loss'], rel=1e-6)
    assert row_a['region_denominators'] == row_b['region_denominators']
    assert torch.count_nonzero(a.correction.weight.grad) > 0
    assert torch.count_nonzero(a.stem.weight.grad) == 0  # 零末层阻断第一步的内部梯度是预期行为。
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)
    optimization_step(a, opt_a, [full], cfg, 1)
    assert torch.count_nonzero(a.stem.weight.grad) > 0


def test_checkpoint_reload_exact_and_resume_rng_optimizer(tmp_path):
    _, cfg = config_fixture()
    model = BoundaryRestoreNet(width=4, seed=9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    batch = collate_samples([sample(), sample(True)])
    optimization_step(model, optimizer, [batch], cfg, 0)
    identity = {'synthetic_cpu_test': True}
    path = tmp_path/'state.pt'
    save_checkpoint(path, model, optimizer, identity, 1, cfg, [0])
    loaded = load_boundary_restorer(path)
    with torch.no_grad():
        assert torch.equal(model(batch['input']), loaded(batch['input']))
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    optimization_step(model, optimizer, [batch], cfg, 1)
    expected_sha = state_sha256(model)
    resumed = BoundaryRestoreNet(width=4, seed=9)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=10.)
    step, monitors = load_resume(path, resumed, resumed_optimizer, identity, cfg)
    assert step == 1 and monitors == [0]
    assert expected_rng == (random.random(), float(np.random.rand()), float(torch.rand(())))
    optimization_step(resumed, resumed_optimizer, [batch], cfg, 1)
    assert state_sha256(resumed) == expected_sha
    with pytest.raises(ValueError, match='identity differs'):
        load_resume(path, resumed, resumed_optimizer, {'different': True}, cfg)
    payload = torch.load(path, weights_only=False)
    payload['model_state_dict']['unexpected_weight'] = torch.ones(1)
    torch.save(payload, tmp_path/'bad.pt')
    with pytest.raises(RuntimeError):
        load_boundary_restorer(tmp_path/'bad.pt')


def test_monitor_rng_and_model_state_are_unchanged(tmp_path):
    model = BoundaryRestoreNet(width=4).train()
    rng = capture_rng()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng(rng)
    def callback(m, cfg, out, epoch, device):
        assert not m.training and not torch.is_grad_enabled() and epoch == 0
        random.random(); np.random.rand(); torch.rand(2)
        return float(m(torch.ones(1, 1, 16, 16)).mean())
    assert run_monitor(model, {}, tmp_path, 0, 'cpu', callback) == 1.
    assert model.training and expected == (random.random(), float(np.random.rand()), float(torch.rand(())))


def test_log_tail_is_preserved_before_resume(tmp_path):
    path = tmp_path/'steps.jsonl'
    path.write_text(''.join(json.dumps({'global_step': i})+'\n' for i in range(1, 4)))
    reconcile_log(tmp_path, 1)
    assert len(path.read_text().splitlines()) == 1
    receipt = json.loads((tmp_path/'resume_log.json').read_text())
    assert receipt['replayed_updates'] == 2 and (tmp_path/receipt['preserved_tail']).is_file()


def test_default_budget_schedule_and_loss_recipe():
    cfg = validate_recipe(load_config('config/train/boundary_restore.yaml'))
    assert cfg['epochs'] * cfg['updates_per_epoch'] == 320
    assert 320 * cfg['effective_batch_size'] == 1280
    assert learning_rate(319, cfg) > .8 * cfg['learning_rate']
    assert cfg['loss'] == dict(valid_weight=1., gap_weight=1., noise_weight=.5,
                              preserve_weight=.25, preserve_threshold=.2)
    assert cfg['target_sigma'] == 2.


class SyntheticDataset:
    def __init__(self):
        self.identity = {'synthetic_cpu_only': True}
        self.set_epoch(1)
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.plan = [{'draw': i, 'epoch': epoch} for i in range(8)]
    def __len__(self):
        return 8
    def __getitem__(self, index):
        return sample(identity=bool(index % 3 == 0))
    def verify_sources(self):
        return True
    def plan_receipt(self):
        return {'epoch': self.epoch, 'synthetic_cpu_only': True}


def test_complete_small_training_resume_and_two_update_smoke(tmp_path):
    config, cfg = config_fixture()
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{"synthetic_cpu_only": true}')
    cfg['manifest'] = str(manifest)
    dataset = SyntheticDataset()
    calls = []
    def monitor(m, c, out, epoch, device):
        assert not m.training and not torch.is_grad_enabled()
        calls.append(epoch)
    model = BoundaryRestoreNet(**cfg['model'], seed=cfg['seed'])
    output = tmp_path/'run'
    train(model, dataset, config, output, monitor)
    assert calls == [0, 1]
    complete = json.loads((output/'complete.json').read_text())
    assert complete['optimizer_updates'] == 4 and complete['model_changed']
    cfg['epochs'] = 2
    resumed = BoundaryRestoreNet(**cfg['model'], seed=cfg['seed'])
    train(resumed, dataset, config, output, monitor, resume=output/'last.pt')
    assert json.loads((output/'complete.json').read_text())['optimizer_updates'] == 8
    assert calls == [0, 1]
    fresh = BoundaryRestoreNet(**cfg['model'], seed=cfg['seed'])
    report = smoke(fresh, dataset, config, tmp_path/'smoke', monitor)
    assert report['strict_reload_prediction_equal'] and not report['formal_training_complete']
