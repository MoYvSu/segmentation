# -*- coding: utf-8 -*-
"""灰度先验须独立、保留坐标、尊重GT，并能反对错误的高置信预测。"""
from copy import deepcopy
from pathlib import Path
import math

import cv2
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from data.direct_dual_head_dataset import _spatial_transform
from data.rgb_restoration_dataset import prepare_rgb, read_rgb
from data.semantic_gray import GrayPriorDataset, build_gray_targets, gray_prior_loss
from train_semantic_consistency import candidate_arm, verify_contract
from utils.config import load_config


def settings():
    config = deepcopy(load_config('config/train/semantic_gray.yaml')['semantic_gray'])
    config.update(local_window=32, local_stride=16, min_region_pixels=32)
    return config


def scene():
    image = torch.full((1, 3, 64, 64), .85)
    image[:, :, 16:48, 16:48] = .18
    valid = torch.ones(1, 1, 64, 64, dtype=torch.bool)
    return image, valid, torch.zeros_like(valid)


def test_brightness_targets_accept_two_interiors_and_ignore_mixed_edges():
    image, valid, known = scene()
    target, mask, info = build_gray_targets(image, valid, known, settings(), grid=64)
    assert mask[0, 0, 8, 8] and target[0, 0, 8, 8] > .9
    assert mask[0, 0, 32, 32] and target[0, 0, 32, 32] < .1
    assert not mask[0, 0, 16, 32]  # 明暗交界收缩一格，不能充当内部监督
    assert min(info['accepted_ferrite'], info['accepted_pearlite']) > 0
    assert not info['network_gating'] and not target.requires_grad


@pytest.mark.parametrize('kind', ['flat', 'low_contrast', 'continuous_ramp'])
def test_single_or_ambiguous_brightness_distribution_can_abstain(kind):
    image, valid, known = scene()
    if kind == 'flat':
        image.fill_(.7)
    elif kind == 'low_contrast':
        image = .7 + .03 * image
    else:
        image = torch.linspace(.2, .8, 64)[None, None, None, :].expand(1, 3, 64, 64).clone()
    target, mask, info = build_gray_targets(image, valid, known, settings(), grid=64)
    assert not mask.any() and info['accepted_pixels'] == 0
    assert torch.all(target == .5)


def test_reflect_padding_does_not_become_evidence_or_supervision():
    image, valid, known = scene()
    valid[:, :, -8:] = False
    a, ma, _ = build_gray_targets(image, valid, known, settings(), grid=64)
    image[:, :, -8:] = 0
    b, mb, _ = build_gray_targets(image, valid, known, settings(), grid=64)
    assert torch.equal(a, b) and torch.equal(ma, mb)
    assert not ma[:, :, -8:].any()


def test_known_gt_excludes_even_a_partially_known_output_cell():
    image, valid, known = scene()
    cfg = settings()
    cfg.update(local_window=16, local_stride=8)
    _, before, _ = build_gray_targets(image, valid, known, cfg, grid=32)
    assert before[0, 0, 4, 4]
    known[0, 0, 8, 8] = True
    target, after, info = build_gray_targets(image, valid, known, cfg, grid=32)
    assert not after[0, 0, 4, 4] and target[0, 0, 4, 4] == .5
    assert info['known_gt_excluded_pixels'] == 1


def test_independent_target_corrects_high_confidence_errors_in_both_directions():
    target = torch.tensor([[[[.98, .02, .5]]]])
    mask = torch.tensor([[[[True, True, False]]]])
    logits = torch.tensor([[[[-8., 8., 4.]]]], requires_grad=True)
    loss = gray_prior_loss(logits, target, mask)
    loss.backward()
    assert logits.grad[0, 0, 0, 0] < 0  # 亮目标，学生高置信珠光体，梯度仍须纠正
    assert logits.grad[0, 0, 0, 1] > 0
    assert logits.grad[0, 0, 0, 2] == 0


def test_loss_balances_accepted_classes_without_forcing_area_ratio():
    logits = torch.zeros(1, 1, 1, 11, requires_grad=True)
    target = torch.tensor([.95] * 10 + [.05])[None, None, None]
    mask = torch.ones_like(logits, dtype=torch.bool)
    gray_prior_loss(logits, target, mask).backward()
    assert logits.grad[0, 0, 0, :10].abs().sum().item() == pytest.approx(logits.grad[0, 0, 0, 10].abs().item())


def test_empty_target_is_differentiable_zero_and_target_has_no_gradient():
    logits = torch.randn(1, 1, 3, 3, requires_grad=True)
    target = torch.full_like(logits, .95, requires_grad=True)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    loss = gray_prior_loss(logits, target, mask)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0
    assert target.grad is None


def test_targets_follow_geometric_transforms_but_not_strong_appearance():
    image, valid, known = scene()
    # 非对称已知GT块使翻转错误不可被对称图掩盖。
    known[:, :, 4:12, 8:16] = True
    target, mask, _ = build_gray_targets(image, valid, known, settings(), grid=64)
    for geometry in [(True, False, 0), (False, True, 1), (True, True, 3)]:
        transformed = [_spatial_transform(t[0], *geometry)[None] for t in (image, valid, known)]
        q, accepted, _ = build_gray_targets(*transformed, settings(), grid=64)
        assert torch.equal(accepted, _spatial_transform(mask[0], *geometry)[None])
        assert torch.allclose(q, _spatial_transform(target[0], *geometry)[None], atol=1e-6)


def test_dataset_matches_renumbered_manual_source_and_transforms_gt_exclusion(tmp_path):
    pool, manual_dir = tmp_path / 'pool', tmp_path / 'manual'
    pool.mkdir()
    manual_dir.mkdir()
    image, _, _ = scene()
    array = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(pool / 'train_017.png'), array)
    cv2.imwrite(str(pool / 'train_002.png'), 255 - array)
    (manual_dir / 'renamed.png').write_bytes((pool / 'train_017.png').read_bytes())
    (manual_dir / 'renamed.json').write_text('{}')

    class Manual:
        data_dir = manual_dir
        samples = [manual_dir / 'renamed.png']

        def __len__(self):
            return 1

        def __getitem__(self, index):
            rgb, _ = prepare_rgb(read_rgb(str(self.samples[index])), 64)
            known = torch.zeros(1, 64, 64, dtype=torch.bool)
            known[:, 2:10, 5:17] = True
            return {'image': torch.from_numpy(rgb).permute(2, 0, 1).contiguous(), 'semantic_valid_content': known}

    ds = GrayPriorDataset(pool, {'profile_probabilities': [1, 0, 0, 0], 'blur_sigma': [.4, 5.2]},
        {'enabled': False}, 17, draws_per_epoch=8, expected_sources=2, image_size=64,
        weak_config={'enabled': False}, manual_dataset=Manual())
    for i in range(len(ds)):
        sample = ds[i]
        if sample['name'] == 'train_017.png':
            geometry = sample['horizontal_flip'], sample['vertical_flip'], sample['rotation_k']
            assert torch.equal(sample['known_gt_mask'], _spatial_transform(Manual()[0]['semantic_valid_content'], *geometry))
        else:
            assert not sample['known_gt_mask'].any()
        assert torch.equal(sample['weak_image'], sample['strong_image'])


def test_short_config_preserves_deployment_and_twenty_epoch_budget():
    config = load_config('config/train/semantic_gray.yaml')
    previous = load_config('config/train/semantic_d5a.yaml')
    verify_contract(config, {'config': previous, 'backend_adaptation': {'semantic_adaptation': {'arm': 'simple'}}})
    assert candidate_arm(config) == 'prior'
    assert candidate_arm(load_config('config/train/semantic_consistency.yaml')) == 'consistency'
    cfg = config['semantic_adaptation']
    assert cfg['epochs'] * cfg['manual_repeats'] * cfg['expected_manual_sources'] == 1280
    assert config['semantic_consistency']['expected_sources'] == 1000
    assert len(Path(cfg['output_dir']).name.split('_')) <= 4
    assert cfg['monitor']['every_epochs'] == 5 and len(cfg['monitor']['images']) == 8


def test_one_sided_stops_at_satisfied_confidence_but_corrects_errors_and_uncertainty():
    # 已满足、类别相反、同类但不足0.9、ignore，各自包含亮/暗两种方向。
    logits = torch.tensor([[[[8., -8., -8., 8., .7, -.7, -9.]]]], requires_grad=True)
    target = torch.tensor([[[[.95, .05, .98, .02, .96, .04, .95]]]], requires_grad=True)
    mask = torch.tensor([[[[True, True, True, True, True, True, False]]]])
    loss, info = gray_prior_loss(logits, target, mask, mode='one_sided', return_info=True)
    loss.backward()
    g = logits.grad.flatten()
    assert g[0] == g[1] == g[6] == 0
    assert g[2] < 0 and g[3] > 0 and g[4] < 0 and g[5] > 0
    assert target.grad is None
    assert info['active_pixels'] == 4 and info['satisfied_pixels'] == 2
    assert info['active_ferrite'] == info['active_pearlite'] == 2
    assert info['confidence_only_active_pixels'] == 2


def test_one_sided_is_zero_at_margin_and_does_not_chase_higher_soft_target():
    margin = math.log(9.)
    logits = torch.tensor([[[[margin, -margin, 3., -3.]]]], requires_grad=True)
    target = torch.tensor([[[[.995, .005, .995, .005]]]])
    loss = gray_prior_loss(logits, target, torch.ones_like(logits, dtype=torch.bool), mode='one_sided')
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0


def test_one_sided_keeps_accepted_class_denominator_and_finite_wrong_extreme_gradients():
    logits = torch.tensor([[[[-1000., 1000., 10.]]]], requires_grad=True)
    target = torch.tensor([[[[.95, .05, .95]]]])
    loss = gray_prior_loss(logits, target, torch.ones_like(logits, dtype=torch.bool), mode='one_sided')
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    # 两个接受亮像素（一个已满足）仍一起取均值，不能把未满足者突然放大两倍。
    assert logits.grad.flatten().tolist() == pytest.approx([-.25, .5, 0.])


@pytest.mark.parametrize('empty_mask', [False, True])
@pytest.mark.parametrize('gain', [1., 20.])
def test_one_sided_zero_loss_keeps_autograd_when_no_pixel_needs_penalty(empty_mask, gain):
    logits = torch.full((1, 1, 2, 2), 8., requires_grad=True)
    target = torch.full_like(logits, .95, requires_grad=True)
    mask = torch.full_like(logits, not empty_mask, dtype=torch.bool)
    loss, info = gray_prior_loss(logits, target, mask, mode='one_sided', maximum_normalization_gain=gain, return_info=True)
    loss.backward()
    assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0
    assert target.grad is None and info['active_pixels'] == 0


def test_old_soft_bce_values_and_gradients_remain_identical():
    logits = torch.tensor([[[[-3., 1., 7.]]]], requires_grad=True)
    target = torch.tensor([[[[.95, .02, .96]]]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    pixel = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    old = (pixel[target > .5].mean() + pixel[target <= .5].mean()) / 2
    new = gray_prior_loss(logits, target, mask)
    assert torch.equal(old, new)
    assert torch.equal(torch.autograd.grad(old, logits)[0], torch.autograd.grad(new, logits)[0])


def test_gray2_only_changes_loss_weight_and_output_from_previous_recipe():
    previous = load_config('config/train/semantic_gray.yaml')
    config = load_config('config/train/semantic_gray2.yaml')
    assert config['semantic_consistency']['maximum_weight'] == .15
    assert config['semantic_gray']['loss_mode'] == 'one_sided'
    assert config['semantic_gray']['minimum_confidence'] == .9
    assert len(Path(config['semantic_adaptation']['output_dir']).name.split('_')) <= 4
    config['semantic_adaptation']['output_dir'] = previous['semantic_adaptation']['output_dir']
    config['semantic_consistency']['maximum_weight'] = previous['semantic_consistency']['maximum_weight']
    config['semantic_gray'].pop('loss_mode')
    config['semantic_gray'].pop('minimum_confidence')
    assert config == previous


@pytest.mark.parametrize(('active_count', 'expected_gain'), [(1, 20.), (10, 10.), (100, 1.)])
def test_active_average_caps_sparse_error_amplification_without_moving_satisfied_pixels(active_count, expected_gain):
    # 100个亮位置，仅部分仍为错误方向；1个已满足的暗位置保持零项和原类别平均。
    logits = torch.tensor([-2.] * active_count + [8.] * (100-active_count) + [-8.])[None,None,None].requires_grad_()
    target = torch.tensor([.95]*100+[.05])[None,None,None]
    mask = torch.ones_like(logits,dtype=torch.bool)
    baseline = gray_prior_loss(logits,target,mask,mode='one_sided')
    updated, info = gray_prior_loss(logits,target,mask,mode='one_sided',maximum_normalization_gain=20.,return_info=True)
    before = torch.autograd.grad(baseline,logits,retain_graph=True)[0]
    after = torch.autograd.grad(updated,logits)[0]
    assert updated.item() == pytest.approx(baseline.item()*expected_gain)
    assert torch.allclose(after,before*expected_gain)
    assert torch.count_nonzero(after.flatten()[active_count:]) == 0
    assert info['class_normalization']['ferrite']['gain'] == expected_gain


def test_default_one_sided_preserves_historical_value_and_gradient_exactly():
    logits = torch.tensor([[[[-3., 1., 7., -7.]]]],requires_grad=True)
    target = torch.tensor([[[[.95,.02,.96,.04]]]])
    mask = torch.ones_like(logits,dtype=torch.bool)
    signed = logits * torch.where(target>.5,1.,-1.)
    pixels = F.relu(F.softplus(-signed)-F.softplus(-logits.new_tensor(math.log(9.))))
    previous = (pixels[target>.5].mean()+pixels[target<=.5].mean())/2
    actual = gray_prior_loss(logits,target,mask,mode='one_sided')
    assert torch.equal(previous,actual)
    assert torch.equal(torch.autograd.grad(previous,logits)[0],torch.autograd.grad(actual,logits)[0])


@pytest.mark.parametrize('gain', [0., .5, float('inf'), float('nan')])
def test_rejects_invalid_normalization_gain(gain):
    logits = torch.zeros(1,1,1,1)
    with pytest.raises(ValueError,match='Normalization gain'):
        gray_prior_loss(logits,torch.ones_like(logits),torch.ones_like(logits,dtype=torch.bool),
                        mode='one_sided',maximum_normalization_gain=gain)


def test_soft_target_cannot_silently_enable_active_normalization():
    logits = torch.zeros(1,1,1,1)
    with pytest.raises(ValueError,match='one_sided'):
        gray_prior_loss(logits,torch.ones_like(logits),torch.ones_like(logits,dtype=torch.bool),maximum_normalization_gain=20.)


def test_gray3_changes_only_normalization_and_output_and_reuses_original_control():
    previous = load_config('config/train/semantic_gray2.yaml')
    config = load_config('config/train/semantic_gray3.yaml')
    assert config['semantic_consistency'].pop('reuse_control_dir') == 'outputs/semantic_gray/control'
    assert config['semantic_gray'].pop('maximum_normalization_gain') == 20.
    assert config['semantic_consistency']['maximum_weight'] == .15
    assert config['semantic_adaptation']['epochs'] == 20
    assert len(Path(config['semantic_adaptation']['output_dir']).name.split('_')) <= 4
    config['semantic_adaptation']['output_dir'] = previous['semantic_adaptation']['output_dir']
    assert config == previous
