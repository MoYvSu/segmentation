# -*- coding: utf-8 -*-
"""大窗口只能增加原先零支持区，不能覆盖已有证据或降低可靠阈值。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from data.semantic_gray import build_gray_targets, _extend_gray_coverage, gray_prior_loss
from utils.config import load_config


def settings():
    return deepcopy(load_config('config/train/semantic_coverage.yaml')['semantic_gray'])


def scene(invert=False):
    image = torch.full((1, 3, 256, 256), .85)
    image[:, :, 24:104, 24:104] = .18
    if invert:
        image = 1 - image
    # 连续的小起伏避免纯常数双峰在旧直方图平滑中退化为相邻峰。
    jitter = torch.from_numpy(np.random.default_rng(11).normal(0,.008,(1,1,256,256)).astype(np.float32))
    image = image + jitter
    valid = torch.ones(1, 1, 256, 256, dtype=torch.bool)
    return image, valid, torch.zeros_like(valid)


@pytest.mark.parametrize('invert', [False, True])
def test_new_targets_add_both_classes_without_changing_any_existing_target(invert):
    cfg = settings()
    baseline = deepcopy(cfg)
    baseline.pop('coverage')
    image, valid, known = scene(invert)
    before = image.clone()
    old, old_mask, _ = build_gray_targets(image, valid, known, baseline)
    new, mask, info = build_gray_targets(image, valid, known, cfg)
    added = mask & ~old_mask
    assert added.sum() > 1000
    assert torch.equal(new[old_mask], old[old_mask]) and torch.all(mask[old_mask])
    assert torch.all((new[added] > .5) == (not invert))
    assert torch.equal(image, before)
    assert info['coverage']['added_pixels'] == int(added.sum())
    logits = torch.zeros_like(new, requires_grad=True)
    gray_prior_loss(logits, new, mask, mode='one_sided').backward()
    assert torch.all(logits.grad[~mask] == 0)
    assert torch.all((logits.grad[added] < 0) == (not invert))


def test_coverage_disabled_exactly_preserves_legacy_output_and_metadata():
    cfg = settings()
    cfg['coverage']['enabled'] = False
    legacy = deepcopy(cfg)
    legacy.pop('coverage')
    a = build_gray_targets(*scene(), cfg)
    b = build_gray_targets(*scene(), legacy)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]


def test_global_abstention_does_not_become_an_automatic_single_phase_label():
    image, valid, known = scene()
    image.fill_(.85)
    target, mask, info = build_gray_targets(image, valid, known, settings())
    assert not mask.any() and torch.all(target == .5) and info['coverage']['added_pixels'] == 0


def test_existing_local_support_blocks_fallback_even_when_old_target_was_rejected():
    cfg = settings()
    gray = np.full((256,256), .85, np.float32)
    gray[24:104,24:104] = .18
    valid = np.ones_like(gray, bool)
    known = np.zeros_like(valid)
    votes = np.ones_like(gray)  # 有过局部意见，不能借大窗口覆盖它。
    target = np.full_like(gray, .5)
    accepted = np.zeros_like(valid)
    info = _extend_gray_coverage(gray, valid, known, votes, np.full_like(gray,.99), target, accepted, cfg)
    assert info['added_pixels'] == 0 and np.all(target == .5) and not accepted.any()


def test_large_window_cannot_override_disagreeing_global_evidence():
    cfg = settings()
    gray = np.full((256,256), .85, np.float32)
    gray[24:104,24:104] = .18
    valid = np.ones_like(gray, bool)
    votes = np.ones_like(gray)
    votes[:,192:] = 0  # 缺支持处是亮区，而全图参照指定暗类；应继续忽略。
    target = np.full_like(gray, .5)
    accepted = np.zeros_like(valid)
    info = _extend_gray_coverage(gray, valid, np.zeros_like(valid), votes,
        np.full_like(gray,.01), target, accepted, cfg)
    assert info['added_pixels'] == 0 and not accepted.any()


def test_known_gt_and_padding_remain_ignored_in_added_region():
    image, valid, known = scene()
    valid[:, :, -8:] = False
    known[:, :, 20:40, 210:230] = True
    target, mask, info = build_gray_targets(image, valid, known, settings())
    assert info['coverage']['added_pixels'] > 0
    assert not mask[known | ~valid].any()
    assert torch.all(target[known | ~valid] == .5)


@pytest.mark.parametrize('window', [128,256,300])
def test_fallback_requires_larger_local_not_duplicate_global_window(window):
    cfg = settings()
    cfg['coverage']['window'] = window
    with pytest.raises(ValueError, match='larger local window'):
        build_gray_targets(*scene(), cfg)
