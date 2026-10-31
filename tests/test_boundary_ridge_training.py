# -*- coding: utf-8 -*-
"""单连通监督的真实更新、全批分母、关闭兼容与严格续训；仅合成CPU数据。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_boundary_rgb as trainer
from models.boundary_naf import BoundaryDiscriminator
from models.boundary_restore import state_sha256
from test_boundary_resize_training import small_models, synthetic_batch
from utils.boundary_rgb_training import build_tolerant_targets, rgb_gan_update, tolerant_boundary_loss
from utils.boundary_ridge_loss import build_ridge_targets, ridge_supervision_loss
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def settings():
    return dict(enabled=True, version='normal_hull_v1', weight=.1,
                profile_radius=4, corridor_half_width=1)


def config():
    value = load_config('config/train/boundary_resize.yaml')
    value['boundary_naf']['ridge_supervision'] = settings()
    return value


class PhaseGenerator(nn.Module):
    """用可训练格点构造多峰，确保新项对真实模型参数有独立梯度。"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.6))
        self.bias = nn.Parameter(torch.tensor(-.2))
        self.grid = nn.Parameter(torch.tensor([[2., -2.], [2., -2.]]))

    def forward_logits(self, value):
        h, w = value.shape[-2:]
        grid = self.grid.repeat((h + 1) // 2, (w + 1) // 2)[:h, :w]
        return value * self.weight + self.bias + grid[None, None]

    def forward(self, value):
        return self.forward_logits(value).sigmoid()


def batch(n=3, size=80):
    target = torch.zeros(n, 1, size, size)
    target[..., 10:size - 10, size // 2] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[0, :, 17:21, 12:16] = False
    target[~valid] = float('nan')
    instances = torch.ones_like(target, dtype=torch.int64)
    instances[..., size // 2:] = 2
    instances[~valid] = 0
    value = torch.rand(n, 1, size, size, generator=torch.Generator().manual_seed(83))
    return dict(input=value, target=target, valid=valid, ridge_instances=instances)


def models():
    return PhaseGenerator(), BoundaryDiscriminator(width=2, seed=17)


def tree_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_enabled_recipe_keeps_architecture_and_only_adds_explicit_target():
    current = config()
    trainer.validate_config(current)
    actual = deepcopy(current['boundary_naf'])
    assert actual.pop('ridge_supervision') == settings()
    assert actual == load_config('config/train/boundary_resize.yaml')['boundary_naf']


@pytest.mark.parametrize('change', [
    lambda c: c['boundary_naf']['ridge_supervision'].update(enabled='true'),
    lambda c: c['boundary_naf']['ridge_supervision'].update(weight=0),
    lambda c: c['boundary_naf']['ridge_supervision'].update(version='old'),
    lambda c: c['boundary_naf'].update(erasure={'enabled': False}),
    lambda c: c['boundary_naf'].update(input_mode='rgb_boundary'),
])
def test_invalid_or_changed_training_contract_is_rejected(change):
    value = config()
    change(value)
    with pytest.raises(ValueError):
        trainer.validate_config(value)


def test_shift_and_ridge_cannot_be_enabled_together():
    value = config()
    shift = load_config('config/train/boundary_shift.yaml')['boundary_naf']
    value['boundary_naf']['shift_consistency'] = shift['shift_consistency']
    value['boundary_naf']['monitor']['shift_pair_indices'] = shift['monitor']['shift_pair_indices']
    with pytest.raises(ValueError, match='同时启用'):
        trainer.validate_config(value)


def test_disabled_update_is_exact_including_optimizers_RNG_and_receipt(monkeypatch):
    import sys
    sample = batch(1)
    g1, d1 = models()
    g2, d2 = models()
    go1, do1 = torch.optim.Adam(g1.parameters()), torch.optim.Adam(d1.parameters())
    go2, do2 = torch.optim.Adam(g2.parameters()), torch.optim.Adam(d2.parameters())
    before = torch.get_rng_state().clone()
    a = rgb_gan_update(g1, d1, go1, do1, sample)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    # 关闭路径连helper都不应导入；多出的instance数据也不得影响旧更新。
    monkeypatch.setitem(sys.modules, 'utils.boundary_ridge_loss', None)
    b = rgb_gan_update(g2, d2, go2, do2, sample, ridge_supervision={'enabled': False})
    assert a == b and not any(key.startswith('ridge_') for key in b)
    assert torch.equal(after, torch.get_rng_state())
    for x, y in ((g1, g2), (d1, d2), (go1, go2), (do1, do2)):
        assert tree_equal(x.state_dict(), y.state_dict())


def test_gradient_matches_explicit_full_objective_without_changing_original_denominators():
    sample = batch()
    g, d = models()
    reference = deepcopy(g)
    go, do = torch.optim.SGD(g.parameters(), lr=.03), torch.optim.SGD(d.parameters(), lr=.03)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    prepared = build_tolerant_targets(sample['target'], sample['valid'])
    targets = build_ridge_targets(prepared, sample['ridge_instances'], settings())
    logits = reference.forward_logits(sample['input'])
    base, base_stats = tolerant_boundary_loss(logits, sample['target'], sample['valid'], prepared=prepared)
    shape, shape_stats = ridge_supervision_loss(logits, targets)
    objective = base + settings()['weight'] * shape
    objective.backward()
    reference_optimizer.step()
    actual = rgb_gan_update(g, d, go, do, sample, micro_batch_size=3,
                           adversarial_weight=0, grad_clip=None, ridge_supervision=settings())
    assert actual['ridge_loss'] > 0 and actual['ridge_gap_points'] > 0
    assert actual['ridge_multitrack_points'] > 0
    assert actual['loss_g'] == pytest.approx(float(objective.detach()), rel=1e-6)
    assert actual['ridge_loss'] == pytest.approx(float(shape.detach()), rel=1e-6)
    assert actual['ridge_positions'] == shape_stats['ridge_positions']
    for name in ('positive_points', 'negative_pixels', 'covered_gt_points'):
        assert actual[name] == base_stats[name]
    for x, y in zip(g.parameters(), reference.parameters()):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-7)


def test_microbatch_equivalence_includes_GAN_and_global_profile_denominator():
    sample = batch()
    g1, d1 = models()
    g2, d2 = models()
    go1, do1 = torch.optim.SGD(g1.parameters(), lr=.01), torch.optim.SGD(d1.parameters(), lr=.01)
    go2, do2 = torch.optim.SGD(g2.parameters(), lr=.01), torch.optim.SGD(d2.parameters(), lr=.01)
    a = rgb_gan_update(g1, d1, go1, do1, sample, micro_batch_size=3,
                       grad_clip=None, ridge_supervision=settings())
    b = rgb_gan_update(g2, d2, go2, do2, sample, micro_batch_size=1,
                       grad_clip=None, ridge_supervision=settings())
    for key in ('loss_g', 'loss_d', 'coverage', 'interior', 'adv_g', 'grad_norm_g',
                'ridge_loss', 'ridge_weighted', 'ridge_gap_fraction', 'ridge_multitrack_fraction'):
        assert a[key] == pytest.approx(b[key], rel=1e-5, abs=1e-6)
    for key in ('ridge_eligible_points', 'ridge_positions', 'ridge_gap_points',
                'ridge_gap_profiles', 'ridge_multitrack_points'):
        assert a[key] == b[key]
    for m1, m2 in ((g1, g2), (d1, d2)):
        for x, y in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)


def test_new_term_alone_backpropagates_to_generator_parameters():
    g, d = models()
    old = g.grid.detach().clone()
    go, do = torch.optim.SGD(g.parameters(), lr=.1), torch.optim.SGD(d.parameters(), lr=.1)
    stat = rgb_gan_update(g, d, go, do, batch(1), coverage_weight=0,
        interior_weight=0, adversarial_weight=0, grad_clip=None, ridge_supervision=settings())
    assert stat['generator_updated'] and not stat['discriminator_updated']
    assert stat['ridge_weighted'] > 0 and not torch.equal(g.grid, old)
    assert stat['loss_g'] == pytest.approx(stat['ridge_weighted'])


def test_unknown_batch_does_not_update_anything_with_new_term_enabled():
    sample = batch(1)
    sample['valid'].zero_()
    sample['target'].fill_(float('nan'))
    g, d = models()
    old_g, old_d = deepcopy(g.state_dict()), deepcopy(d.state_dict())
    go, do = torch.optim.Adam(g.parameters()), torch.optim.Adam(d.parameters())
    stat = rgb_gan_update(g, d, go, do, sample, ridge_supervision=settings())
    assert stat['ridge_eligible_points'] == stat['ridge_positions'] == 0
    assert not stat['generator_updated'] and not stat['discriminator_updated']
    assert not go.state and not do.state
    assert tree_equal(old_g, g.state_dict()) and tree_equal(old_d, d.state_dict())


def test_enabled_missing_instances_rejected_before_optimizer_changes():
    sample = batch(1)
    sample.pop('ridge_instances')
    g, d = models()
    go, do = torch.optim.Adam(g.parameters()), torch.optim.Adam(d.parameters())
    with pytest.raises(ValueError, match='ridge_instances'):
        rgb_gan_update(g, d, go, do, sample, ridge_supervision=settings())
    assert not go.state and not do.state


def test_identity_only_adds_new_helper_while_G_D_initial_states_remain_equal(monkeypatch):
    current = config()
    control = load_config('config/train/boundary_resize.yaml')
    _, g, d, _, _ = small_models(current['boundary_naf'])
    _, old_g, old_d, _, _ = small_models(control['boundary_naf'])
    monkeypatch.setattr(trainer, 'file_sha', lambda path: str(path))
    dataset = SimpleNamespace(identity={'synthetic': True})
    a = trainer.source_identity(current, dataset, g, d)
    b = trainer.source_identity(control, dataset, old_g, old_d)
    assert a['architecture'] == b['architecture']
    assert a['initial_model_sha256'] == b['initial_model_sha256']
    assert a['initial_discriminator_sha256'] == b['initial_discriminator_sha256']
    assert set(a['source_sha256']) - set(b['source_sha256']) == {'utils/boundary_ridge_loss.py'}
    assert a['assets_sha256'] == b['assets_sha256']


def test_reconstruction_diagnostic_preserves_original_metrics_and_model_gradients():
    sample = synthetic_batch()
    sample['ridge_instances'] = torch.ones_like(sample['target'], dtype=torch.int64)
    sample['ridge_instances'][..., 48:] = 2
    g = PhaseGenerator()
    logits = g.forward_logits(sample['input'])
    old = trainer.reconstruction_metrics(logits, sample,
        load_config('config/train/boundary_resize.yaml')['boundary_naf'], clean_logits=logits)
    new = trainer.reconstruction_metrics(logits, sample, config()['boundary_naf'], clean_logits=logits)
    assert all(old[key] == new[key] for key in old)
    assert new['ridge_eligible_points'] > 0 and new['ridge_logit_grad_norm'] > 0
    assert all(parameter.grad is None for parameter in g.parameters())
    assert new['ridge_gradient_scope'] == 'detached_logits_only_not_G_parameter_proof'


def test_strict_checkpoint_resume_reproduces_next_new_loss_update(tmp_path):
    cfg, g, d, go, do = small_models(config()['boundary_naf'])
    sample = synthetic_batch()
    sample['ridge_instances'] = torch.ones_like(sample['target'], dtype=torch.int64)
    sample['ridge_instances'][..., 48:] = 2
    before = state_sha256(g)
    stat = trainer.update(g, d, go, do, sample, cfg, 64)
    assert stat['generator_updated'] and stat['discriminator_updated']
    assert stat['ridge_eligible_points'] > 0 and state_sha256(g) != before
    identity = dict(recipe=deepcopy(cfg), test='normal_hull_single_variable')
    checkpoint = tmp_path / 'latest.pt'
    trainer.save_checkpoint(checkpoint, g, d, go, do, cfg, identity, 65, [0, 1])
    _, fresh, fresh_d, fresh_go, fresh_do = small_models(cfg)
    assert trainer.load_checkpoint(checkpoint, fresh, fresh_d, fresh_go, fresh_do,
                                   cfg, identity) == (65, [0, 1])
    a = trainer.update(g, d, go, do, sample, cfg, 65)
    b = trainer.update(fresh, fresh_d, fresh_go, fresh_do, sample, cfg, 65)
    assert a == b
    assert state_sha256(g) == state_sha256(fresh) and state_sha256(d) == state_sha256(fresh_d)


def test_smoke_accepts_zero_terminal_loss_after_real_earlier_supervision():
    after = dict(ridge_eligible_points=10, ridge_loss=0.)
    assert all(trainer.ridge_smoke_checks(after, 20, .03).values())
    assert not trainer.ridge_smoke_checks(after, 0, .03)['ridge_nonzero_objective']
    assert not trainer.ridge_smoke_checks(after, 20, 0.)['ridge_independent_logit_gradient']
    assert not trainer.ridge_smoke_checks({'ridge_eligible_points': 0}, 20, .03)['ridge_eligible_present']
