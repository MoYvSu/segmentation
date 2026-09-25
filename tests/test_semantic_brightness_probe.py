# -*- coding: utf-8 -*-
"""验证亮度探查的恒等对照，以及无裁剪加常量时的变量隔离。"""
import unittest

import torch

from tools.semantic_brightness_probe import adjust_brightness, input_metrics, luminance, variant_specs


class SemanticBrightnessTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(81)
        self.image = torch.rand((1, 3, 12, 16), generator=generator) * .5 + .25

    def test_zero_offset_and_gamma_one_are_exact_tensor_identity(self):
        self.assertIs(adjust_brightness(self.image, 'offset', 0), self.image)
        self.assertIs(adjust_brightness(self.image, 'gamma', 1), self.image)

    def test_unclipped_offset_preserves_chroma_and_local_differences(self):
        shifted = adjust_brightness(self.image, 'offset', .1)
        torch.testing.assert_close(shifted - self.image, torch.full_like(self.image, .1), atol=1e-7, rtol=0)
        torch.testing.assert_close(shifted[:, 0] - shifted[:, 1], self.image[:, 0] - self.image[:, 1], atol=1e-7, rtol=0)
        torch.testing.assert_close(shifted[..., 1:] - shifted[..., :-1], self.image[..., 1:] - self.image[..., :-1], atol=1e-7, rtol=0)
        metrics = input_metrics(self.image, shifted, 'offset', .1, (12, 16))
        self.assertEqual(metrics['clipped_pixel_fraction'], 0)
        self.assertAlmostEqual(metrics['mean_y_change'], .1, places=6)

    def test_clip_metrics_exclude_padding_and_report_fraction(self):
        image = torch.full((1, 3, 2, 3), .5)
        image[:, 0, 0, 0] = .98
        image[:, :, 1, :] = 1  # 这一行作为 padding 排除。
        shifted = adjust_brightness(image, 'offset', .1)
        metrics = input_metrics(image, shifted, 'offset', .1, (1, 3))
        self.assertAlmostEqual(metrics['clipped_pixel_fraction'], 1 / 3, places=6)
        self.assertAlmostEqual(metrics['clipped_channel_fraction'], 1 / 9, places=6)
        self.assertEqual(float(shifted.max()), 1)

    def test_gamma_preserves_luma_formula_before_clipping(self):
        shifted = adjust_brightness(self.image, 'gamma', .8)
        torch.testing.assert_close(luminance(shifted), luminance(self.image).pow(.8), atol=1e-7, rtol=1e-6)
        metrics = input_metrics(self.image, shifted, 'gamma', .8, (12, 16))
        self.assertEqual(metrics['clipped_pixel_fraction'], 0)
        self.assertGreater(metrics['y_change_std'], 0)

    def test_variants_always_include_exact_zero(self):
        specs = variant_specs([-.1, .1], [.8])
        self.assertIn({'key': 'offset_0', 'kind': 'offset', 'value': 0.0}, specs)
        with self.assertRaises(ValueError):
            variant_specs([0, 0], [])
        with self.assertRaises(ValueError):
            adjust_brightness(self.image, 'gamma', 0)


if __name__ == '__main__':
    unittest.main()
