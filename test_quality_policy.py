# -*- coding: utf-8 -*-
"""Physical augmentation, quality policy, and 8-bit instance-cap tests."""

import unittest

import numpy as np
import torch

from utils.config import load_config
from utils.post_process import boundary_watershed_separation
from utils.progressive_aug import ProgressiveAppearanceAug
from utils.quality_aware import (
    classify_quality,
    effective_boundary_threshold,
    enhance_weak_image,
)


class PhysicalAugmentationTest(unittest.TestCase):
    def test_forced_blur_preserves_shape_range_and_reduces_variation(self):
        config = {
            "enabled": True,
            "policy": "physical_v1",
            "min_ops": 1,
            "max_ops": 1,
            "op_weights": {"gaussian_blur": 1.0},
            "blur_sigma_range": [1.5, 1.5],
            "blur_kernel_size": 9,
        }
        augmentor = ProgressiveAppearanceAug(config, torch.device("cpu"))
        checker = (
            torch.arange(64).view(1, 64)
            + torch.arange(64).view(64, 1)
        ) % 2
        image = checker.float().unsqueeze(0).repeat(3, 1, 1)
        actual = augmentor._augment_single(image)

        self.assertEqual(actual.shape, image.shape)
        self.assertGreaterEqual(float(actual.min()), 0.0)
        self.assertLessEqual(float(actual.max()), 1.0)
        self.assertLess(float(actual.var()), float(image.var()) * 0.1)

    def test_unknown_policy_fails_fast(self):
        with self.assertRaises(ValueError):
            ProgressiveAppearanceAug(
                {"policy": "untraceable_auto_policy"}, torch.device("cpu")
            )


class QualityPolicyTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("config/inference/b2_quality_aware.yaml")[
            "inference"
        ]["quality_aware"]

    def test_policy_has_only_two_profiles_and_no_output_feedback(self):
        standard, _ = classify_quality(
            {
                "brightness": 0.8,
                "contrast": 0.4,
                "sharpness": 0.04,
                "color_cast": 0.01,
            },
            self.config,
        )
        weak, reasons = classify_quality(
            {
                "brightness": 0.4,
                "contrast": 0.05,
                "sharpness": 0.03,
                "color_cast": 0.01,
            },
            self.config,
        )
        self.assertEqual(standard, "standard")
        self.assertEqual(weak, "weak")
        self.assertEqual(reasons, ["low_brightness", "low_contrast"])
        serialized = repr(self.config).lower()
        for forbidden in ("instance_count", "mean_area", "ring", "nest"):
            self.assertNotIn(forbidden, serialized)

    def test_weak_profile_only_uses_small_bounded_threshold_offset(self):
        self.assertAlmostEqual(
            effective_boundary_threshold(0.35, "standard", self.config), 0.35
        )
        self.assertAlmostEqual(
            effective_boundary_threshold(0.35, "weak", self.config), 0.33
        )
        self.assertEqual(
            effective_boundary_threshold(0.01, "weak", self.config), 0.25
        )

    def test_weak_enhancement_is_deterministic_and_bounded(self):
        image = np.full((64, 64, 3), (70, 60, 50), dtype=np.uint8)
        metrics = {
            "brightness": 0.23,
            "contrast": 0.0,
            "sharpness": 0.0,
            "color_cast": 0.08,
        }
        first, first_ops = enhance_weak_image(image, metrics, self.config)
        second, second_ops = enhance_weak_image(image, metrics, self.config)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_ops, second_ops)
        self.assertEqual(first.dtype, np.uint8)
        self.assertGreater(float(first.mean()), float(image.mean()))


class InstanceCapTest(unittest.TestCase):
    def test_rejects_non_uint8_cap(self):
        semantic = np.ones((32, 32), dtype=np.uint8)
        boundary = np.zeros_like(semantic)
        with self.assertRaises(ValueError):
            boundary_watershed_separation(
                semantic, boundary, max_instance_id=256
            )

    def test_more_than_255_candidates_are_capped_and_classified(self):
        size = 128
        semantic = np.ones((size, size), dtype=np.uint8)
        boundary = np.zeros_like(semantic)
        center = np.zeros_like(semantic, dtype=np.float32)
        # 18 x 18 isolated peaks -> 324 candidate watershed seeds.
        coordinates = np.linspace(3, size - 4, 18, dtype=int)
        center[np.ix_(coordinates, coordinates)] = 1.0
        instance_map, class_map = boundary_watershed_separation(
            semantic,
            boundary,
            min_area=1,
            max_instance_id=255,
            bridge_width=0,
            dilate_width=0,
            center_prob=center,
            center_threshold=0.5,
            center_nms_kernel=3,
        )
        self.assertLessEqual(int(instance_map.max()), 255)
        self.assertLessEqual(len(class_map), 255)
        self.assertEqual(len(class_map), 255)
        self.assertTrue(all(value == 1 for value in class_map.values()))


if __name__ == "__main__":
    unittest.main()
