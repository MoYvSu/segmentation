# -*- coding: utf-8 -*-
"""NAF输入扰动：整图重叠、原生单位、强弱覆盖、RNG隔离与uint8契约。"""
from copy import deepcopy
import json
import random

import numpy as np
import pytest

from utils import naf_input_perturbation as module


def rgb(seed=4, shape=(96, 128)):
    return np.random.default_rng(seed).integers(0, 256, (*shape, 3), dtype=np.uint8)


def test_clean_is_exact_independent_copy_and_never_calls_blur(monkeypatch):
    a = rgb(); before = a.copy()
    monkeypatch.setattr(module, '_apply_sigma_map_blur', lambda *_: pytest.fail('clean must not blur'))
    b, sigma, receipt = module.perturb_native_rgb(a, profile='clean', seed=8)
    assert np.array_equal(a, before) and np.array_equal(a, b) and not np.shares_memory(a, b)
    assert b.flags.c_contiguous and sigma.dtype == np.float32 and not sigma.any()
    assert receipt['input_sha256'] == receipt['output_sha256'] and receipt['config'] is None
    json.dumps(receipt, allow_nan=False)


def test_deterministic_without_changing_numpy_or_python_random_or_config():
    a = rgb(); before = a.copy(); py_state = random.getstate(); np_state = np.random.get_state()
    cfg = dict(strong_sigma=8.0); original = deepcopy(cfg)
    b, sigma, receipt = module.perturb_native_rgb(a, seed=11, config=cfg)
    c, same, repeated = module.perturb_native_rgb(a, seed=11, config=cfg)
    assert np.array_equal(a, before) and cfg == original
    assert np.array_equal(b, c) and np.array_equal(sigma, same) and receipt == repeated
    assert random.getstate() == py_state
    state = np.random.get_state()
    assert state[0] == np_state[0] and np.array_equal(state[1], np_state[1]) and state[2:] == np_state[2:]
    assert b.dtype == np.uint8 and b.shape == a.shape and b.flags.c_contiguous
    assert sigma.dtype == np.float32 and sigma.shape == a.shape[:2]


def test_whole_image_crop_overlap_is_exact_without_new_field_or_blur():
    b, sigma, receipt = module.perturb_native_rgb(rgb(shape=(256, 320)), seed=31, profile='strong')
    left, right = b[:180, :240], b[70:, 100:]
    np.testing.assert_array_equal(left[70:, 100:], right[:110, :140])
    np.testing.assert_array_equal(sigma[:180, :240][70:, 100:], sigma[70:, 100:][:110, :140])
    assert receipt['full_image_applications'] == 1 and not receipt['per_tile_field_regeneration']
    assert not receipt['geometry_changed'] and receipt['padding_is_not_input']


@pytest.mark.parametrize('shape', [(2, 2), (2, 31), (31, 2), (96, 128)])
def test_reflect_border_preserves_constant_colors_even_on_small_images(shape):
    a = np.empty((*shape, 3), np.uint8); a[:] = [17, 127, 241]
    b, _, receipt = module.perturb_native_rgb(a, profile='strong', seed=3)
    assert np.array_equal(a, b) and receipt['border'] == 'BORDER_REFLECT'


@pytest.mark.parametrize('profile,weak,strong', [('moderate', 1., 4.5), ('strong', 1., 7.)])
def test_original_pixel_units_and_large_endpoint_coverage(profile, weak, strong):
    a = rgb(shape=(256, 320))
    b, sigma, receipt = module.perturb_native_rgb(a, profile=profile, seed=12)
    assert b.shape == a.shape and float(sigma.min()) == weak and float(sigma.max()) == strong
    stats = receipt['sigma_statistics']
    assert stats['weak_endpoint_fraction'] >= .15 and stats['strong_endpoint_fraction'] >= .15
    assert 'native pixels' in receipt['sigma_units']
    assert not receipt['noise_or_illumination_or_resampling'] and receipt['GT_is_not_an_input']


def test_strength_profiles_share_smooth_field_geometry_and_are_not_rectangular_masks():
    a = rgb(shape=(256, 320))
    _, m, _ = module.perturb_native_rgb(a, profile='moderate', seed=22)
    _, s, _ = module.perturb_native_rgb(a, profile='strong', seed=22)
    phase_m = (m.astype(np.float64)**2-1)/(4.5**2-1)
    phase_s = (s.astype(np.float64)**2-1)/(7.**2-1)
    np.testing.assert_allclose(phase_m, phase_s, atol=3e-7, rtol=0)
    assert np.count_nonzero((s > 1.01) & (s < 6.99)) > s.size*.15
    assert np.max(np.abs(np.diff(s, axis=0))) < .35
    assert np.max(np.abs(np.diff(s, axis=1))) < .35


def test_strong_profile_reduces_stripe_contrast_more_than_moderate():
    yy, xx = np.indices((256, 320))
    a = np.repeat(((xx//12 % 2)*255).astype(np.uint8)[..., None], 3, axis=2)
    m, _, _ = module.perturb_native_rgb(a, profile='moderate', seed=16)
    s, _, _ = module.perturb_native_rgb(a, profile='strong', seed=16)
    energy = lambda value: np.abs(np.diff(value.astype(np.float64), axis=1)).mean()
    assert energy(s) < energy(m) < energy(a)


def test_distinct_seed_changes_field_but_does_not_change_input_geometry():
    a = rgb()
    b, first, _ = module.perturb_native_rgb(a, seed=4)
    c, second, _ = module.perturb_native_rgb(a, seed=5)
    assert not np.array_equal(first, second) and not np.array_equal(b, c)
    assert b.shape == c.shape == a.shape


@pytest.mark.parametrize('bad', [True, -1, 3.5, '3', 2**64])
def test_invalid_seed_rejected(bad):
    with pytest.raises(ValueError): module.perturb_native_rgb(rgb(), seed=bad)


@pytest.mark.parametrize('bad', [
    np.zeros((8, 9, 3), np.float32), np.zeros((1, 9, 3), np.uint8),
    np.zeros((8, 9), np.uint8), np.zeros((8, 9, 4), np.uint8),
    np.ma.array(np.zeros((8, 9, 3), np.uint8)), [[1, 2]],
])
def test_ambiguous_input_units_padding_geometry_or_channels_rejected(bad):
    with pytest.raises(ValueError): module.perturb_native_rgb(bad)


@pytest.mark.parametrize('cfg', [
    {'noise': .1}, {'levels': True}, {'levels': 1}, {'levels': 33},
    {'weak_sigma': 0}, {'weak_sigma': 2}, {'strong_sigma': 3}, {'strong_sigma': 17},
    {'strong_sigma': float('nan')}, {'transition_fraction': 0}, {'min_region_fraction': .5},
    {'weak_sigma': True},
])
def test_unreviewed_or_invalid_recipe_rejected(cfg):
    with pytest.raises(ValueError): module.perturb_native_rgb(rgb(), config=cfg)


def test_unknown_profile_and_nonempty_clean_config_rejected():
    with pytest.raises(ValueError): module.perturb_native_rgb(rgb(), profile='automatic_test_fit')
    with pytest.raises(ValueError): module.perturb_native_rgb(rgb(), profile='clean', config={'levels': 8})


def test_readonly_noncontiguous_rgb_is_supported_without_mutation():
    a = rgb()[:, ::-1]; a.flags.writeable = False; original = a.copy()
    b, _, receipt = module.perturb_native_rgb(a, seed=77)
    assert b.flags.c_contiguous and b.flags.writeable and np.array_equal(a, original)
    assert receipt['source_unchanged']


@pytest.mark.parametrize('profile', ['moderate', 'strong'])
@pytest.mark.parametrize('seed,shape', [(0, (2, 2)), (22, (96, 128)), (999, (37, 81))])
def test_self_contained_numeric_core_exactly_matches_existing_D5a_helpers(profile, seed, shape):
    from data.rgb_spatial_blur import apply_sigma_map_blur, make_endpoint_sigma_map
    a = rgb(shape=shape)
    actual, sigma, _ = module.perturb_native_rgb(a, profile=profile, seed=seed)
    s = module.profile_settings(profile)
    cfg = dict(enabled=True, weak_sigma=[s['weak_sigma']]*2, strong_sigma=[s['strong_sigma']]*2,
               transition_fraction=[s['transition_fraction']]*2,
               min_region_fraction=s['min_region_fraction'])
    rng = np.random.default_rng(np.random.SeedSequence([seed, module.SEED_NAMESPACE]))
    h, w = shape
    expected_sigma = make_endpoint_sigma_map(shape, (0, 0, h, w), cfg, rng,
                                            [s['weak_sigma'], s['strong_sigma']])
    expected = apply_sigma_map_blur(a.astype(np.float32), expected_sigma,
                                   [s['weak_sigma'], s['strong_sigma']], s['levels'])
    expected = np.ascontiguousarray(np.rint(np.clip(expected, 0, 255)).astype(np.uint8))
    np.testing.assert_array_equal(sigma, expected_sigma)
    np.testing.assert_array_equal(actual, expected)
