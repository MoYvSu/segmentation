# -*- coding: utf-8 -*-
"""Regression tests for the retained G0b generative edge prior."""

import unittest

import torch

from models.edge_prior import (
    EdgePriorResidualFusion,
    GenerativeEdgePrior,
    build_structural_edge_target,
    edge_prior_loss,
)


class EdgePriorTest(unittest.TestCase):
    def test_residual_fusion_is_bounded_and_zero_initialized(self):
        fusion = EdgePriorResidualFusion(hidden_channels=16, max_logit_delta=1.0)
        boundary = torch.randn(2, 1, 32, 32)
        prior = torch.randn(2, 3, 64, 64)
        initial = fusion(boundary, prior)
        torch.testing.assert_close(initial, torch.zeros_like(initial))

        loss = (initial - 0.25).square().mean()
        loss.backward()
        self.assertIsNotNone(fusion.out.weight.grad)
        self.assertGreater(float(fusion.out.weight.grad.abs().sum()), 0.0)

    def test_structural_target_rejects_flat_field_and_detects_step(self):
        flat = torch.full((1, 3, 64, 64), 0.5)
        step = flat.clone()
        step[:, :, :, 32:] = 0.9
        flat_target = build_structural_edge_target(flat)
        step_target = build_structural_edge_target(step)
        self.assertEqual(tuple(step_target.shape), (1, 3, 64, 64))
        self.assertLess(float(flat_target[:, 0].max()), 1e-4)
        self.assertGreater(float(step_target[:, 0].max()), 0.25)

    def test_multiscale_target_suppresses_fine_periodic_texture(self):
        base = torch.full((1, 3, 64, 64), 0.2)
        step = base.clone()
        step[:, :, :, 32:] = 0.9
        stripe = torch.remainder(
            torch.div(torch.arange(64), 2, rounding_mode="floor"), 2
        ).float().view(1, 1, 1, 64).expand(1, 3, 64, 64)
        stripe = 0.2 + 0.7 * stripe
        step_edge = build_structural_edge_target(step)[:, 0]
        stripe_edge = build_structural_edge_target(stripe)[:, 0]
        self.assertLess(float(stripe_edge.max()), float(step_edge.max()) * 0.2)

    def test_prior_output_shape_and_loss_backward(self):
        prior = GenerativeEdgePrior((8, 16), hidden_channels=16)
        features = [
            torch.randn(2, 8, 16, 16),
            torch.randn(2, 16, 8, 8),
        ]
        raw = prior(features, (64, 64))
        self.assertEqual(tuple(raw.shape), (2, 3, 64, 64))
        clean = torch.rand(2, 3, 64, 64)
        target = build_structural_edge_target(clean)
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        loss, details = edge_prior_loss(raw, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(details["target_edge_mean"], 0.0)
        self.assertIsNotNone(prior.out.weight.grad)


if __name__ == "__main__":
    unittest.main()
