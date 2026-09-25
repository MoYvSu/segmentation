# -*- coding: utf-8 -*-
"""同网格外观诊断的数值约定。"""

import json

import numpy as np
import pytest

from tools.restoration_appearance_metrics import appearance_maps


def test_identity_has_zero_change_and_serializable_summary():
    image = np.random.default_rng(17).uniform(0.15, 0.85, (40, 48, 3)).astype(np.float32)
    maps, summary = appearance_maps(image, image.copy())
    for name, values in maps.items():
        assert values.shape == image.shape[:2]
        assert values.dtype == np.float32
        assert np.isfinite(values).all()
        if name.startswith(("delta", "abs_delta")):
            np.testing.assert_array_equal(values, 0.0)
    assert summary["pixels"] == 40 * 48
    json.dumps(summary, allow_nan=False)


def test_constant_brightness_is_low_frequency_without_added_texture():
    before = np.full((48, 48, 3), 0.35, dtype=np.float32)
    after = np.full_like(before, 0.55)
    maps, _ = appearance_maps(before, after)
    assert maps["delta_L"].mean() > 15.0
    np.testing.assert_allclose(maps["delta_low_L"], maps["delta_L"], atol=5e-5)
    for channel in "RGB":
        np.testing.assert_allclose(maps[f"delta_{channel}"], 0.2, atol=1e-6)
    for feature in ("high_L", "mid_L", "grad_L", "std_L"):
        np.testing.assert_allclose(maps[f"restored_{feature}"], 0.0, atol=2e-5)


def test_red_cast_has_signed_a_change_and_color_distance():
    before = np.full((32, 32, 3), 0.5, dtype=np.float32)
    after = before.copy()
    after[..., 0] += 0.1
    maps, _ = appearance_maps(before, after)
    assert maps["delta_a"].mean() > 5.0
    assert maps["delta_ab"].mean() > 5.0
    np.testing.assert_allclose(
        maps["deltaE76"] ** 2,
        maps["delta_L"] ** 2 + maps["delta_ab"] ** 2,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(maps["delta_G"], 0.0)
    np.testing.assert_array_equal(maps["delta_B"], 0.0)


def test_zero_mean_rgb_texture_changes_high_frequency_more_than_background():
    before = np.full((128, 128, 3), 0.5, dtype=np.float32)
    # 四像素周期，避免棋盘格被 Sobel 对称采样抵消。
    stripe = np.tile(np.array([-1.0, -1.0, 1.0, 1.0], np.float32), 32)
    after = before + np.float32(0.08) * stripe[None, :, None]
    maps, _ = appearance_maps(before, after)
    assert abs(maps["delta_R"].mean()) < 1e-6
    assert maps["delta_high_L"].mean() > 4.0
    assert maps["delta_grad_L"].mean() > 2.0
    assert maps["delta_std_L"].mean() > 4.0
    interior_low_change = np.abs(maps["delta_low_L"][32:-32, 32:-32]).mean()
    assert interior_low_change < 0.5


@pytest.mark.parametrize("case", ["shape", "nan", "range"])
def test_rejects_unmatched_or_invalid_measurement_inputs(case):
    before = np.full((8, 8, 3), 0.5, dtype=np.float32)
    after = before.copy()
    if case == "shape":
        after = after[:7]
    elif case == "nan":
        after[0, 0, 0] = np.nan
    else:
        after[0, 0, 0] = 1.1
    with pytest.raises(ValueError):
        appearance_maps(before, after)
