# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pytest

from tools.evaluate_direct_degraded_val import (
    VARIANTS,
    _resolve_output_dir,
    apply_fixed_degradation,
)


def _sample_image():
    y, x = np.mgrid[:96, :128]
    checker = ((x // 4 + y // 4) % 2) * 80
    return np.stack(
        [60 + checker, 90 + checker, 120 + checker], axis=-1
    ).clip(0, 255).astype(np.uint8)


def test_fixed_degradations_are_deterministic_and_non_mutating():
    source = _sample_image()
    before = source.copy()
    for variant in VARIANTS:
        first = apply_fixed_degradation(source, variant)
        second = apply_fixed_degradation(source, variant)
        assert first.shape == source.shape
        assert first.dtype == np.uint8
        assert np.array_equal(first, second)
        assert first is not source
    assert np.array_equal(source, before)


def test_fixed_degradations_change_expected_quality_axis():
    source = _sample_image()
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    blurred = cv2.cvtColor(
        apply_fixed_degradation(source, "defocus_v1"), cv2.COLOR_RGB2GRAY
    )
    dark = cv2.cvtColor(
        apply_fixed_degradation(source, "low_light_v1"), cv2.COLOR_RGB2GRAY
    )
    assert cv2.Laplacian(blurred, cv2.CV_32F).var() < cv2.Laplacian(
        gray, cv2.CV_32F
    ).var()
    assert float(np.median(dark)) < float(np.median(gray))


def test_combined_degradation_has_fixed_operation_order():
    source = _sample_image()
    blurred = apply_fixed_degradation(source, "defocus_v1")
    expected = apply_fixed_degradation(blurred, "low_light_v1")
    actual = apply_fixed_degradation(source, "defocus_low_light_v1")
    assert np.array_equal(actual, expected)


def test_degraded_val_rejects_output_outside_analysis(tmp_path):
    config = {
        "paths": {"project_root": str(tmp_path)},
        "degraded_validation": {"output_dir": "data/forbidden"},
    }
    with pytest.raises(ValueError, match="outputs"):
        _resolve_output_dir(config)
