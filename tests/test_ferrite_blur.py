# -*- coding: utf-8 -*-
"""模糊全局坐标、正式uint8单位、真实分水岭及固定safe比较的CPU反例。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from tools.analyze_ferrite_native import analyze_ferrite_native
from tools.probe_ferrite_blur import (array_sha, decode_with_signed_capture, gallery_sources,
    GALLERY_FIXED, make_global_blur, matched_phase_changes, paired_diagnostic_delta, phase_pixels,
    regional_diagnostics, require_clean_parity, save_thumbnail)


def config():
    return dict(inference=dict(boundary_threshold=.65, threshold=.5, min_instance_area=50,
        max_instance_id=65535, watershed_dilate_width=1, bridge_width=1,
        marker_border_seal_width=2, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_partition_restore=dict(enabled=False),
        semantic_vote_mode='probability_mean', semantic_vote_threshold=.5,
        semantic_vote_erode_width=0))


def test_whole_blur_deterministic_overlap_units_and_no_global_rng_change():
    rgb = np.random.default_rng(4).integers(0, 256, (96, 120, 3), dtype=np.uint8)
    before = rgb.copy(); rng = np.random.get_state(); torch_rng = torch.get_rng_state().clone()
    blurred, sigma, receipt = make_global_blur(rgb, 314159)
    repeated, same_sigma, repeated_receipt = make_global_blur(rgb, 314159)
    assert np.array_equal(rgb, before) and np.array_equal(blurred, repeated)
    assert np.array_equal(sigma, same_sigma) and receipt == repeated_receipt
    assert np.array_equal(np.random.get_state()[1], rng[1]) and torch.equal(torch.get_rng_state(), torch_rng)
    assert blurred.dtype == np.uint8 and sigma.dtype == np.float32
    assert sigma.min() == 1 and sigma.max() == 4
    assert receipt['weak_fraction'] >= .05 and receipt['strong_fraction'] >= .05
    assert receipt['full_image_applications'] == 1 and not receipt['additional_illumination_or_noise']
    # 重叠窗只读同一退化整图，不各自重新生成焦面或核。
    a, b = blurred[:70, :80], blurred[30:, 40:]
    assert np.array_equal(a[30:, 40:], b[:40, :40])
    assert np.array_equal(sigma[:70, :80][30:, 40:], sigma[30:, 40:][:40, :40])


def test_constant_rgb_is_preserved_and_formal_tensor_divides255_once():
    from tools.probe_affinity_patch import tensor_from_rgb
    rgb = np.full((24, 32, 3), [17, 127, 241], dtype=np.uint8)
    blurred, _, _ = make_global_blur(rgb, 9)
    assert np.array_equal(blurred, rgb)
    tensor, _, _ = tensor_from_rgb(blurred, 'cpu')
    assert tensor.dtype == torch.float32 and tensor.min() >= 0 and tensor.max() <= 1
    assert torch.allclose(tensor[0, :, 400, 400], torch.tensor([17, 127, 241])/255, atol=1e-7, rtol=0)


@pytest.mark.parametrize('rgb,seed', [(np.zeros((16, 18, 3), np.float32), 1),
    (np.zeros((1, 18, 3), np.uint8), 1), (np.zeros((16, 18, 3), np.uint8), True)])
def test_blur_rejects_ambiguous_input_units_or_seed(rgb, seed):
    with pytest.raises(ValueError): make_global_blur(rgb, seed)


def test_full_phase_matrix_includes_P_only_F_prediction_and_ignores_unknown():
    gt = np.array([[0, 1, 1, 1], [2, 2, 2, 2]], np.uint16)
    pred = np.array([[3, 1, 0, 2], [3, 3, 0, 4]], np.uint16)
    result = phase_pixels(gt, {1: 1, 2: 0}, pred, {'1': 1, '2': 0, '3': 1, '4': 0})
    assert result['matrix'] == [[1, 2, 1], [1, 1, 1]]
    with pytest.raises(ValueError): phase_pixels(gt, {1: True, 2: 0}, pred, {1: 1, 2: 0, 3: 1, 4: 0})


def actual_decode():
    boundary = np.zeros((72, 80), np.float32); boundary[:, 39:41] = 1
    semantic = torch.zeros(1, 1, 72, 80); probability = np.full((72, 80), .75, np.float32)
    value = decode_with_signed_capture(semantic, boundary, probability, config())
    return value, boundary, semantic, probability


def test_actual_signed_watershed_kept_before_inplace_zero_and_inputs_unchanged():
    before_cv = cv2.watershed
    value, boundary, semantic, probability = actual_decode()
    assert cv2.watershed is before_cv
    assert value['instances'].dtype == np.uint16 and len(value['classes']) == 2
    assert value['signed_watershed'].dtype == np.int32 and np.any(value['signed_watershed'] == -1)
    assert np.all(value['instances'][value['signed_watershed'] == -1] == 0)
    assert value['zero_sources']['watershed_boundary'] == value['zero_sources']['final_zero']
    assert value['zero_sources']['removed_positive_region'] == 0
    assert torch.count_nonzero(semantic) == 0 and np.all(probability == .75)
    assert np.all(boundary[:, 39:41] == 1) and np.all(boundary[:, :39] == 0)


@pytest.mark.parametrize('failure', ['markers', 'watershed', 'final', 'classes', 'whole_semantics'])
def test_clean_gate_refuses_wrong_stage(failure):
    value, boundary, _, _ = actual_decode()
    fixed = dict(geometry_semantic_sha256='sem', raw_probability_sha256='raw')
    expected = dict(boundary_sha256=array_sha(boundary), actual_marker_sha256=array_sha(value['markers']),
        actual_watershed_sha256=array_sha(np.maximum(value['signed_watershed'], 0)), fixed_whole_semantics=fixed)
    original = value['instances'].copy(); classes = dict(value['classes']); markers = value['markers'].copy()
    assert require_clean_parity(value, boundary, expected, original, classes, fixed, markers)['exact']
    if failure == 'markers': expected['actual_marker_sha256'] = 'wrong'
    if failure == 'watershed': expected['actual_watershed_sha256'] = 'wrong'
    if failure == 'final': original[20, 20] = 0
    if failure == 'classes': classes[next(iter(classes))] = 0
    if failure == 'whole_semantics': expected['fixed_whole_semantics'] = dict(raw_probability_sha256='other')
    with pytest.raises(RuntimeError, match='Clean formal replay parity failed'):
        require_clean_parity(value, boundary, expected, original, classes, fixed, markers)


def test_fixed_clean_safe_cohort_records_lost_split_instead_of_reselection():
    gt = np.full((50, 60), 2, np.uint16); gt[15:35, 20:40] = 1
    clean = gt.copy(); blurred = gt.copy(); blurred[15:35, 30:40] = 3
    gc = {1: 1, 2: 0}; cc = {'1': 1, '2': 0}; bc = {'1': 1, '2': 0, '3': 1}
    original = analyze_ferrite_native(gt, clean, gc, cc)
    candidate = analyze_ferrite_native(gt, blurred, gc, bc)
    delta = paired_diagnostic_delta(original, candidate)
    assert delta['clean_safe_fixed_n'] == 1 and delta['clean_safe_survived_n'] == 0
    assert delta['clean_safe_lost_n'] == 1 and delta['clean_safe_lost'][0]['significant_split']
    changed = matched_phase_changes(original, candidate, cc, bc, gc)
    assert changed['phase_flips'] == 0
    matched = next(p['pred'] for p in candidate['pairs'] if p['gt'] == 1)
    bc[str(matched)] = 0
    changed = matched_phase_changes(original, candidate, cc, bc, gc)
    # 同一clean概率可因区域改变翻转类别；此函数记录末端匹配类别，而非固定ID。
    assert changed['phase_flips'] == 1


def test_gallery_random_source_is_fixed_before_result_and_separate_rng():
    names = list(GALLERY_FIXED) + [f'fictional_{i:03d}.jpg' for i in range(25)]
    selected, receipt = gallery_sources(names)
    assert len(selected) == 8 and receipt['random_source'] in names and receipt['random_source'] not in GALLERY_FIXED
    assert gallery_sources(list(reversed(names))) == (selected, receipt)


def test_sigma_regions_close_and_GT_groups_use_whole_median_not_sliced_area():
    gt = np.array([[0, 1, 1, 1], [2, 2, 2, 2]], np.uint16)
    pred = np.array([[3, 1, 0, 2], [3, 3, 0, 4]], np.uint16)
    sigma = np.array([[1, 1, 2, 4], [1, 2, 3, 4]], np.float32)
    gc = {1: 1, 2: 0}; pc = {'1': 1, '2': 0, '3': 1, '4': 0}
    diagnostic = analyze_ferrite_native(gt, pred, gc, pc)
    value = dict(instances=pred, classes=pc)
    out = regional_diagnostics(gt, gc, value, value, diagnostic, diagnostic, sigma)
    assert out['GT_median_sigma_assignment']['1'] == dict(median_sigma=2., region='transition')
    assert out['regions']['transition']['clean']['F_GT_group_counts']['F_GT'] == 1
    assert out['regions']['weak']['clean']['F_GT_group_counts']['F_GT'] == 0
    assert out['regions']['weak']['known_region_pixels'] == 2
    total = sum((np.array(r['clean']['phase_pixels']['matrix']) for r in out['regions'].values()), np.zeros((2, 3), int))
    assert total.tolist() == [[1, 2, 1], [1, 1, 1]]


def test_lost_clean_safe_reason_includes_matched_prediction_unknown_or_frame_censor():
    gt = np.full((50, 60), 2, np.uint16); gt[15:35, 20:40] = 1
    clean = gt.copy(); changed = gt.copy(); changed[:16, 25] = 1
    gc = {1: 1, 2: 0}; pc = {'1': 1, '2': 0}
    original = analyze_ferrite_native(gt, clean, gc, pc)
    candidate = analyze_ferrite_native(gt, changed, gc, pc)
    out = paired_diagnostic_delta(original, candidate)
    assert out['clean_safe_lost_n'] == 1
    assert out['clean_safe_lost'][0]['matched_pred_censored']
    assert not out['clean_safe_lost'][0]['GT_censored']


def test_predeclared_thumbnail_is_rgb_png_without_full_maps(tmp_path):
    rgb = np.full((16, 20, 3), 127, np.uint8); gt = np.ones((16, 20), np.uint16)
    value = dict(instances=gt, boundary=np.zeros(gt.shape, np.float32))
    path = tmp_path/'fictional.png'
    save_thumbnail(path, rgb, rgb, np.ones(gt.shape, np.float32), gt, value, value, box_yxyx=[2, 2, 10, 12])
    panel = cv2.imread(str(path))
    assert panel.shape == (284, 2560, 3) and panel.dtype == np.uint8
    assert [p.name for p in tmp_path.iterdir()] == ['fictional.png']
