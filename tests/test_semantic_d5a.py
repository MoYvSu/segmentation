# -*- coding: utf-8 -*-
"""语义两组的初始化范围、冻结边界及低分辨监督对齐。"""
import unittest

import torch
from torch import nn

from train_semantic_d5a import (
    aligned_semantic_logits, configure_arm, frozen_digest,
    learning_rate_factor, semantic_train_mode,
)


class ToySemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_fpn = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.seg_branch = nn.Conv2d(4, 1, 1)
        self.semantic_residual = nn.Conv2d(3, 1, 1)
        self.semantic_residual_version = 'highres'


class ToyBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3))
        self.encoder.trainable_lora = True
        self.encoder.register_parameter('lora_A', nn.Parameter(torch.randn(1, 3)))
        self.affinity_decoder = nn.Conv2d(3, 4, 1)
        self.semantic_decoder = ToySemantic()


def metadata():
    return {'architecture': {'semantic_decoder': {'semantic_residual': True,
                                                  'semantic_residual_version': 'highres'}}}


class SemanticD5aTests(unittest.TestCase):
    def test_full_preserves_state_and_only_semantic_trains(self):
        model = ToyBackend()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        info = configure_arm(model, metadata(), 'full', 41)
        self.assertEqual(info['initialization'], 'existing_full_head')
        self.assertTrue(all(torch.equal(before[key], value) for key, value in model.state_dict().items()))
        semantic_train_mode(model)
        self.assertTrue(model.semantic_decoder.training)
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.affinity_decoder.training)
        self.assertFalse(model.encoder.trainable_lora)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith('semantic_decoder.'))

    def test_simple_reinitializes_all_semantic_state_and_removes_rgb(self):
        model = ToyBackend()
        model.semantic_decoder.seg_fpn[1].running_mean.fill_(9)
        model.semantic_decoder.seg_fpn[1].running_var.fill_(4)
        model.semantic_decoder.seg_fpn[1].num_batches_tracked.fill_(30)
        frozen_before = frozen_digest(model)
        coarse_before = model.semantic_decoder.seg_fpn[0].weight.clone()
        cfg = metadata()
        info = configure_arm(model, cfg, 'simple', 41)
        self.assertEqual(frozen_digest(model), frozen_before)
        self.assertIsNone(model.semantic_decoder.semantic_residual)
        self.assertEqual(model.semantic_decoder.semantic_residual_version, 'none')
        self.assertFalse(cfg['architecture']['semantic_decoder']['semantic_residual'])
        self.assertEqual(info['initialization'], 'random_fpn_and_classifier')
        self.assertFalse(torch.equal(coarse_before, model.semantic_decoder.seg_fpn[0].weight))
        self.assertEqual(int(model.semantic_decoder.seg_fpn[1].num_batches_tracked), 0)
        self.assertTrue(torch.all(model.semantic_decoder.seg_fpn[1].running_mean == 0))
        self.assertTrue(torch.all(model.semantic_decoder.seg_fpn[1].running_var == 1))
        self.assertFalse(any('semantic_residual' in key for key in model.state_dict()))

    def test_random_init_is_repeatable_without_advancing_rng(self):
        left, right = ToyBackend(), ToyBackend()
        state = torch.get_rng_state().clone()
        configure_arm(left, metadata(), 'simple', 41)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        configure_arm(right, metadata(), 'simple', 41)
        self.assertTrue(all(torch.equal(value, right.semantic_decoder.state_dict()[key])
                            for key, value in left.semantic_decoder.state_dict().items()))

    def test_frozen_digest_includes_lora_and_buffers(self):
        model = ToyBackend()
        original = frozen_digest(model)
        with torch.no_grad():
            model.encoder.lora_A.add_(1)
        changed_lora = frozen_digest(model)
        self.assertNotEqual(original, changed_lora)
        model.encoder[1].running_mean.add_(1)
        self.assertNotEqual(changed_lora, frozen_digest(model))

    def test_simple_logits_align_and_retain_gradient(self):
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        aligned = aligned_semantic_logits(logits, {'semantic_target': torch.zeros(1, 1, 16, 16)})
        self.assertEqual(tuple(aligned.shape), (1, 1, 16, 16))
        aligned.square().mean().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_learning_rate_warmup_and_final_floor(self):
        self.assertEqual(learning_rate_factor(1, 60, 2, .1), .5)
        self.assertEqual(learning_rate_factor(2, 60, 2, .1), 1)
        self.assertEqual(learning_rate_factor(3, 60, 2, .1), 1)
        self.assertAlmostEqual(learning_rate_factor(60, 60, 2, .1), .1)


if __name__ == '__main__':
    unittest.main()
