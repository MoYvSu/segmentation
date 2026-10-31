# -*- coding: utf-8 -*-
"""完整区域分隔训练的合成CPU检查；不读取赛题图或GT。"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import utils.boundary_region_training as training


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def batch(count=1, shape=(15, 19)):
    h, w = shape
    ids = torch.ones(count, 1, h, w, dtype=torch.long)
    ids[..., w//2:] = 2
    valid = torch.ones_like(ids, dtype=torch.bool)
    original = valid.clone()
    low = torch.zeros_like(valid)
    low[..., h//2-1:h//2+2, 3:5] = True
    low[..., h//2-1:h//2+2, w-5:w-3] = True
    allowed = torch.zeros_like(valid)
    allowed[..., 2:-2, w//2-1:w//2+2] = True
    high = torch.zeros_like(valid)
    high[..., 2, w//2] = True
    core = torch.where(low, ids, 0)
    return dict(input=torch.zeros_like(ids, dtype=torch.float32),
        target=torch.zeros_like(ids, dtype=torch.float32), valid=valid,
        region_ids=ids, region_original=original, region_core_ids=core,
        region_allowed=allowed, region_keep_high=high, region_keep_low=low,
        meta=[dict(draw_seed=100+i) for i in range(count)])


def settings(**updates):
    return dict(min_atom_pixels=2, **updates)


def closed_shell_probability(shape=(15, 19)):
    # 中央两室被外框封住；中间分隔只有两端，内部存在多条可通路径。
    p = np.full(shape, .1, np.float32)
    p[1, 1:-1] = p[-2, 1:-1] = .99
    p[1:-1, 1] = p[1:-1, -2] = .99
    p[2, shape[1]//2] = .99
    p[-3, shape[1]//2] = .99
    return p


def test_real_full_selector_backpropagates_entire_separator_not_one_bottleneck():
    data = batch()
    logits = torch.logit(torch.from_numpy(closed_shell_probability()))[None, None].requires_grad_()
    loss, audit = training.region_loss(logits, data, settings())
    assert audit['selected_pairs'] == audit['separation_samples'] == 1
    selected = audit['images'][0]['selected_pairs'][0]
    assert selected['independent_certificate']['valid']
    assert selected['cut_pixels'] > 1
    assert audit['active_cut_pixels'] == selected['cut_pixels']
    loss.backward()
    assert torch.count_nonzero(logits.grad < 0) > 1
    assert audit['no_exact_GT_pixel_BCE'] and audit['no_GAN'] and audit['no_PH']
    json.dumps(audit)


def test_high_low_normalize_independently_and_nonempty_images_are_not_diluted():
    data = batch(2)
    data['region_keep_high'].zero_(); data['region_keep_low'].zero_()
    data['region_core_ids'].zero_()
    data['region_keep_high'][0, 0, 3, 3] = True
    data['region_keep_high'][0, 0, 3, 4] = True
    data['region_keep_low'][0, 0, 5:8, 2:5] = True
    data['region_core_ids'][0, 0, 5:8, 2:5] = 1
    logits = torch.full(data['input'].shape, .7, requires_grad=True)
    loss, audit = training.region_loss(logits, data, settings(separation_weight=0))
    expected = F.softplus(torch.tensor(-.7)) + F.softplus(torch.tensor(.7))
    torch.testing.assert_close(loss, expected)
    assert audit['high_samples'] == audit['low_samples'] == 1
    assert audit['high_pixels'] == 2 and audit['low_pixels'] == 9
    loss.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_satisfied_low_background_cannot_dilute_sparse_high_errors():
    sparse = batch(shape=(64, 64))
    large = deepcopy(sparse)
    large['region_keep_low'] = (large['valid'] & ~large['region_allowed'] & ~large['region_keep_high'])
    large['region_core_ids'] = torch.where(large['region_keep_low'], large['region_ids'], 0)
    first = torch.full_like(sparse['input'], float(torch.logit(torch.tensor(.001))))
    first[0, 0, 32, 3] = float(torch.logit(torch.tensor(.99)))
    first.requires_grad_(); second = first.detach().clone().requires_grad_()
    options = settings(high_weight=0, separation_weight=0, negative_margin=.1)
    a, audit_a = training.region_loss(first, sparse, options)
    b, audit_b = training.region_loss(second, large, options)
    assert audit_b['low_pixels'] > audit_a['low_pixels'] * 100
    assert audit_a['active_low_pixels'] == audit_b['active_low_pixels'] == 1
    assert audit_a['low_active_samples'] == audit_b['low_active_samples'] == 1
    torch.testing.assert_close(a, F.softplus(first[0, 0, 32, 3]))
    torch.testing.assert_close(a, b)
    a.backward(); b.backward()
    torch.testing.assert_close(first.grad[0, 0, 32, 3], second.grad[0, 0, 32, 3])
    assert first.grad[0, 0, 32, 3] > .98
    assert torch.count_nonzero(first.grad) == torch.count_nonzero(second.grad) == 1


def test_below_negative_margin_has_zero_loss_and_gradient_without_changing_static_low_mask():
    data = batch()
    logits = torch.full_like(data['input'], float(torch.logit(torch.tensor(.05))), requires_grad=True)
    loss, audit = training.region_loss(logits, data,
        settings(high_weight=0, separation_weight=0, negative_margin=.1))
    assert loss.item() == 0 and not audit['supervised']
    assert audit['low_pixels'] == int(data['region_keep_low'].sum()) > 0
    assert audit['low_samples'] == 1 and audit['active_low_pixels'] == audit['low_active_samples'] == 0
    assert audit['images'][0]['atom_count'] == 2
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_unknown_has_zero_gradient_and_binary_gt_target_is_unused():
    first = batch()
    first['valid'][..., :1, :] = False
    for key in ['region_original', 'region_allowed', 'region_keep_high', 'region_keep_low']:
        first[key][..., :1, :] = False
    changed = deepcopy(first); changed['target'].fill_(float('nan'))
    a = torch.zeros_like(first['input'], requires_grad=True)
    b = torch.zeros_like(first['input'], requires_grad=True)
    loss_a, _ = training.region_loss(a, first, settings(separation_weight=0))
    loss_b, _ = training.region_loss(b, changed, settings(separation_weight=0))
    assert torch.equal(loss_a, loss_b)
    loss_a.backward(); loss_b.backward()
    assert torch.equal(a.grad, b.grad)
    assert torch.count_nonzero(a.grad[~first['valid']]) == 0


def test_same_GT_atoms_and_missing_pair_do_not_invent_separation_labels(monkeypatch):
    data = batch(); data['region_ids'].fill_(1)
    data['region_core_ids'][data['region_core_ids'] > 0] = 1
    def forbidden(*args, **kwargs):
        raise AssertionError('same-GT atoms must not form a pair')
    monkeypatch.setattr(training, 'select_region_cut', forbidden)
    _, audit = training.region_loss(torch.zeros_like(data['input']), data, settings())
    assert audit['selected_pairs'] == 0 and audit['pair_attempts'] == 0
    assert audit['images'][0]['atom_count'] == 2
    assert audit['reasons']['no_different_GT_atom_pair'] == 1


def test_draw_seed_is_repeatable_without_changing_global_numpy_rng(monkeypatch):
    data = batch(shape=(25, 25))
    data['region_keep_low'].zero_(); data['region_core_ids'].zero_()
    for gid, (y, x) in enumerate([(4, 4), (4, 16), (16, 4), (16, 16)], 1):
        data['region_ids'][0, 0, y:y+3, x:x+3] = gid
        data['region_keep_low'][0, 0, y:y+3, x:x+3] = True
        data['region_core_ids'][0, 0, y:y+3, x:x+3] = gid
    data['region_allowed'].zero_()
    called = []
    def no_cut(p, allowed, source, sink, **kwargs):
        called.append((tuple(np.argwhere(source)[0]), tuple(np.argwhere(sink)[0])))
        return np.zeros_like(allowed), dict(feasible=False, reason='no_modifiable_complete_separator')
    monkeypatch.setattr(training, 'select_region_cut', no_cut)
    np.random.seed(42); state = np.random.get_state()
    training.region_loss(torch.zeros_like(data['input']), data, settings())
    first = called.copy(); called.clear()
    training.region_loss(torch.zeros_like(data['input']), data, settings())
    assert first == called and len(called) == 4
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_G_cannot_redefine_static_seed_atom_by_painting_part_of_it_as_wall(monkeypatch):
    data = batch(); calls = []
    def no_cut(p, allowed, source, sink, **kwargs):
        calls.append((source.copy(), sink.copy()))
        return np.zeros_like(allowed), dict(feasible=False, reason='no_modifiable_complete_separator')
    monkeypatch.setattr(training, 'select_region_cut', no_cut)
    first = torch.zeros_like(data['input'])
    _, baseline = training.region_loss(first, data, settings())
    altered = first.clone(); altered[0, 0, 6, 3] = 6
    _, changed = training.region_loss(altered, data, settings())
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    np.testing.assert_array_equal(calls[0][1], calls[1][1])
    assert baseline['images'][0]['atom_count'] == changed['images'][0]['atom_count'] == 2
    assert changed['images'][0]['seed_wall_pixels'] == 1
    assert changed['selected_pairs'] == 0
    assert changed['reasons']['seed_coverage_blocked'] == 1


def test_uncertified_or_illegal_cut_is_not_trained(monkeypatch):
    data = batch()
    def incomplete(p, allowed, source, sink, **kwargs):
        mask = np.zeros_like(allowed); mask[7, 9] = True
        return mask, dict(feasible=True, reason='selected_complete_separator')
    monkeypatch.setattr(training, 'select_region_cut', incomplete)
    _, audit = training.region_loss(torch.zeros_like(data['input']), data, settings())
    assert audit['selected_pairs'] == 0 and audit['active_cut_pixels'] == 0
    assert audit['reasons']['uncertified_complete_cut'] > 0
    def illegal(p, allowed, source, sink, **kwargs):
        return source.copy(), dict(feasible=True)
    monkeypatch.setattr(training, 'select_region_cut', illegal)
    with pytest.raises(RuntimeError, match='illegal'):
        training.region_loss(torch.zeros_like(data['input']), data, settings())


class PixelModel(nn.Module):
    def __init__(self, shape):
        super().__init__(); self.value = nn.Parameter(torch.zeros(shape))
    def forward_logits(self, value):
        return self.value.expand_as(value)


def test_all_empty_skips_optimizer_even_when_adam_has_momentum():
    data = batch(); model = PixelModel(data['input'].shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    training.region_update(model, optimizer, data, settings(separation_weight=0))
    assert optimizer.state
    empty = deepcopy(data)
    for key in ['valid', 'region_original', 'region_allowed', 'region_keep_high', 'region_keep_low']:
        empty[key].zero_()
    empty['region_core_ids'].zero_()
    before = model.value.detach().clone()
    result = training.region_update(model, optimizer, empty, settings())
    assert not result['generator_updated'] and result['optimizer_steps'] == 0
    assert torch.equal(before, model.value)


def test_satisfied_low_only_batch_skips_existing_adam_momentum():
    data = batch(); model = PixelModel(data['input'].shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    options = settings(high_weight=0, separation_weight=0, negative_margin=.1)
    assert training.region_update(model, optimizer, data, options)['generator_updated']
    assert optimizer.state
    with torch.no_grad():
        model.value.fill_(float(torch.logit(torch.tensor(.05))))
    before = model.value.detach().clone()
    result = training.region_update(model, optimizer, data, options)
    assert result['low_pixels'] > 0 and result['active_low_pixels'] == 0
    assert not result['generator_updated'] and result['optimizer_steps'] == 0
    assert torch.equal(before, model.value)


def test_update_once_restores_modes_and_detaches_cached_input():
    data = batch(); data['input'].requires_grad_()
    model = PixelModel(data['input'].shape).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    result = training.region_update(model, optimizer, data, settings(separation_weight=0))
    assert result['generator_updated'] and result['nonzero_grad_g'] and result['optimizer_steps'] == 1
    assert result['grad_norm_g'] > 0 and not model.training
    assert data['input'].grad is None


@pytest.mark.parametrize('key,value', [('positive_margin', 1), ('high_weight', -1),
    ('negative_margin', 0), ('negative_margin', .95),
    ('min_atom_pixels', 0), ('max_pairs_per_image', 2), ('exterior', False), ('bogus', 1)])
def test_invalid_settings_rejected(key, value):
    data = batch()
    with pytest.raises(ValueError):
        training.region_loss(torch.zeros_like(data['input']), data, {key:value})


def test_conflicting_supervision_or_untrusted_core_rejected():
    data = batch(); data['region_allowed'] |= data['region_keep_low']
    with pytest.raises(ValueError, match='overlap'):
        training.region_loss(torch.zeros_like(data['input']), data, settings())
    data = batch(); data['region_original'][data['region_keep_low']] = False
    with pytest.raises(ValueError, match='original'):
        training.region_loss(torch.zeros_like(data['input']), data, settings())
