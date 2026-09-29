# -*- coding: utf-8 -*-
"""困难视图保持几何和随机流，区域诊断不把ignore或混类区域当监督。"""
import pytest
import torch

from data.semantic_hard import hard_appearance, gray_regions, region_diagnostics, region_difficulty, harder_index, gradient_budget_weight
from tools.run_semantic_consistency import control_config, same_config
from utils.config import load_config


@pytest.mark.parametrize('mode', ['global', 'local'])
def test_photometric_views_are_reproducible_without_global_rng_or_input_mutation(mode):
    image = torch.linspace(.05, .95, 64 * 64).reshape(1, 1, 64, 64).repeat(1, 3, 1, 1)
    before = image.clone()
    valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    valid[..., -8:] = False
    state = torch.get_rng_state().clone()
    left, stats = hard_appearance(image, valid, {}, 77, mode)
    right, repeat = hard_appearance(image, valid, {}, 77, mode)
    assert torch.equal(left, right) and stats == repeat
    assert torch.equal(image, before) and torch.equal(torch.get_rng_state(), state)
    assert left.shape == image.shape and torch.isfinite(left).all()
    assert 0 <= left.min() <= left.max() <= 1
    assert stats['minimum_slope'] > 0 and stats['clipped_fraction'] <= .01
    assert torch.equal(left[:, 0], left[:, 1])  # 不制造新的色偏


@pytest.mark.parametrize('mode', ['global', 'local'])
def test_identity_parameters_reproduce_input(mode):
    image = torch.rand(1, 3, 48, 64)
    valid = torch.ones(1, 1, 48, 64, dtype=torch.bool)
    actual, _ = hard_appearance(image, valid, {'contrast': [1., 1.], 'offset': [0., 0.]}, 1, mode)
    assert torch.equal(image, actual)


def test_local_field_is_smooth_and_clipping_budget_rolls_back():
    image = torch.full((1, 3, 128, 128), .8)
    valid = torch.ones(1, 1, 128, 128, dtype=torch.bool)
    local, stats = hard_appearance(image, valid, {'contrast': [.4, .4], 'offset': [.1, .1]}, 3, 'local')
    assert local.max() - local.min() > .1
    assert (local[..., 1:] - local[..., :-1]).abs().max() < .015
    assert stats['minimum_slope'] == pytest.approx(.4, abs=1e-6)
    limited, stats = hard_appearance(image, valid, {'contrast': [1., 1.], 'offset': [.5, .5],
        'max_clipped_fraction': 0.}, 3, 'global')
    assert 0 < stats['scale'] < 1 and stats['clipped_fraction'] == 0
    assert limited.max() <= 1


def test_gray_regions_exclude_unknown_small_islands_and_never_merge_classes():
    target = torch.full((1, 1, 16, 16), .95)
    target[..., :, 8:] = .05
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., 2:7, 2:14] = True
    mask[..., 12, 3] = True
    regions, classes = gray_regions(target, mask, minimum_pixels=8)
    assert classes == [0, 1]
    assert not regions[~mask].any() and regions[..., 12, 3].item() == 0
    assert torch.all(target[regions == 1] < .5) and torch.all(target[regions == 2] > .5)
    logits = torch.where(target > .5, -4., 4.)
    stats = region_diagnostics(logits, target, mask, regions, classes)
    assert stats['disagree_regions'] == 2
    assert stats['ferrite_to_pearlite_regions'] == 1 and stats['pearlite_to_ferrite_regions'] == 1
    assert stats['disagree_pixels'] == int(mask.sum())


def test_empty_prior_is_not_assigned_a_region():
    target = torch.full((1, 1, 8, 8), .5)
    mask = torch.zeros_like(target, dtype=torch.bool)
    regions, classes = gray_regions(target, mask)
    stats = region_diagnostics(torch.zeros_like(target), target, mask, regions, classes)
    assert not classes and not regions.any() and stats['disagree_regions'] == 0


def test_region_selection_prefers_a_confident_error_and_keeps_base_on_ties():
    target = torch.full((1, 1, 16, 16), .95)
    mask = torch.ones_like(target, dtype=torch.bool)
    regions, classes = gray_regions(target, mask)
    correct = region_difficulty(torch.full_like(target, 6), target, mask, regions, classes)
    wrong = region_difficulty(torch.full_like(target, -6), target, mask, regions, classes)
    assert correct == 0 and wrong > .8
    assert harder_index([correct, wrong, wrong]) == 1
    assert harder_index([0., 0., 0.]) == 0
    assert torch.all(target == .95)  # 选难例不能改先验类别


def test_gray4_only_changes_unlabeled_views_and_reuses_existing_control():
    old = load_config('config/train/semantic_gray3.yaml')
    new = load_config('config/train/semantic_gray4.yaml')
    assert same_config(control_config(old), control_config(new))
    assert new['semantic_gray'] == old['semantic_gray']
    assert new['semantic_consistency'] == old['semantic_consistency']
    assert new['semantic_adaptation']['epochs'] == 20
    assert new['semantic_hard']['variants_per_family'] == 2
    new['semantic_adaptation']['output_dir'] = old['semantic_adaptation']['output_dir']
    new.pop('semantic_hard')
    assert same_config(old, new)


@pytest.mark.parametrize('gt,prior,nominal,expected', [(.3, 50., .03, .006),
    (.3, 1., .03, .03), (.3, 0., .15, .15), (0., 2., .15, 0.), (.3, 50., 0., 0.)])
def test_prior_gradient_budget_preserves_weak_or_zero_branches(gt, prior, nominal, expected):
    weight = gradient_budget_weight(nominal, gt, prior, 1.)
    assert weight == pytest.approx(expected)
    assert weight <= nominal and weight * prior <= gt + 1e-12
