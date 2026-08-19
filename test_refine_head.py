# -*- coding: utf-8 -*-
"""B2 high-resolution boundary refine head regression tests."""

import unittest

import torch

from models.fpn_decoder import FPNDecoder


class BoundaryRefineHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.decoder_kwargs = {
            "in_channels": [8, 16, 32, 64],
            "fpn_channels": 32,
            "num_classes": 2,
            "dropout": 0.0,
            "use_bn": True,
        }
        self.features = [
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 16, 8, 8),
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 64, 2, 2),
        ]
        self.image = torch.randn(1, 3, 64, 64)

    def _make_pair(self):
        base = FPNDecoder(**self.decoder_kwargs, boundary_refine=False)
        refine = FPNDecoder(
            **self.decoder_kwargs,
            boundary_refine=True,
            boundary_refine_version="v2_fullres_isolated",
        )
        refine.load_state_dict(base.state_dict(), strict=False)
        return base, refine

    def test_zero_initialized_residual_matches_v6(self):
        base, refine = self._make_pair()
        base.eval()
        refine.eval()
        with torch.no_grad():
            expected = base(
                self.features, output_size=(64, 64), image=self.image
            )
            actual = refine(
                self.features, output_size=(64, 64), image=self.image
            )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)

    def test_refine_only_stage_isolates_v6_and_semantic_paths(self):
        _, refine = self._make_pair()
        refine.freeze_seg_branch()
        refine.set_boundary_base_trainable(False)
        refine.train()
        refine.zero_grad(set_to_none=True)

        boundary_logits = refine(
            self.features, output_size=(64, 64), image=self.image
        )[:, 1]
        boundary_logits.mean().backward()

        protected = (
            list(refine.seg_fpn.parameters())
            + list(refine.seg_branch.parameters())
            + list(refine.boundary_fpn.parameters())
            + list(refine.boundary_branch.parameters())
        )
        self.assertTrue(all(param.grad is None for param in protected))
        self.assertTrue(
            any(
                param.grad is not None and torch.count_nonzero(param.grad).item() > 0
                for param in refine.boundary_refine_head.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
