# -*- coding: utf-8 -*-
"""Stage-0 supervised-control training infrastructure tests."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.fpn_decoder import FPNDecoder
from train_stage2 import (
    next_restarting_batch,
    resolve_epoch_steps,
    seed_everything,
    set_student_train_modes,
    validate,
)
from utils.config import load_config


class _CountingDataset(Dataset):
    def __init__(self):
        self.calls = 0

    def __len__(self):
        return 2

    def __getitem__(self, index):
        del index
        self.calls += 1
        return self.calls


class _ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
        self.decoder = FPNDecoder(
            in_channels=[8, 16, 32, 64],
            fpn_channels=32,
            num_classes=2,
            dropout=0.1,
            use_bn=True,
            boundary_refine=True,
            boundary_refine_version="v2_fullres_isolated",
        )


class _FixedPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        semantic_prob = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
        boundary_prob = torch.tensor([[0.8, 0.4], [0.6, 0.2]])
        output = torch.stack(
            [torch.logit(semantic_prob), torch.logit(boundary_prob)], dim=0
        )
        self.register_buffer("fixed_output", output.unsqueeze(0))

    def forward(self, images, output_size=None):
        del output_size
        return self.fixed_output.expand(images.shape[0], -1, -1, -1)


def _zero_criterion(prediction, target, weight):
    del target, weight
    return prediction.sum() * 0.0, 0.0, 0.0


class Stage0TrainingControlTest(unittest.TestCase):
    def test_supervised_steps_are_independent_from_unlabeled_loader(self):
        self.assertEqual(resolve_epoch_steps(7, 0, 62), 62)
        self.assertEqual(resolve_epoch_steps(7, 40, 62), 62)
        self.assertEqual(resolve_epoch_steps(7, 80, 62), 80)
        self.assertEqual(resolve_epoch_steps(7, 0, 0), 7)

    def test_loader_restart_refetches_instead_of_replaying_cached_batches(self):
        dataset = _CountingDataset()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        iterator = iter(loader)
        values = []
        for _ in range(3):
            batch, iterator = next_restarting_batch(loader, iterator)
            values.append(int(batch.item()))
        self.assertEqual(values, [1, 2, 3])

    def test_frozen_decoder_paths_stay_in_eval_mode(self):
        student = _ToyStudent()
        student.decoder.freeze_seg_branch()
        student.decoder.set_boundary_base_trainable(False)
        set_student_train_modes(student)

        self.assertFalse(student.encoder.training)
        self.assertFalse(student.decoder.seg_fpn.training)
        self.assertFalse(student.decoder.seg_branch.training)
        self.assertFalse(student.decoder.boundary_fpn.training)
        self.assertFalse(student.decoder.boundary_branch.training)
        self.assertTrue(student.decoder.boundary_refine_head.training)

    def test_seed_repeats_torch_initialization(self):
        seed_everything(42)
        first = torch.randn(8)
        seed_everything(42)
        second = torch.randn(8)
        torch.testing.assert_close(first, second)

    def test_stage0_config_invariants(self):
        config = load_config(
            "config/train/stage2_refine_v6_stage0_control.yaml"
        )
        semi = config["semi_supervised"]
        self.assertFalse(semi["use_unlabeled"])
        self.assertEqual(semi["labeled_steps_per_epoch"], 62)
        self.assertEqual(semi["epochs"], 5)
        self.assertEqual(semi["refine_training"]["refine_only_epochs"], 5)
        self.assertTrue(semi["diagnostics"]["enabled"])
        self.assertTrue(semi["diagnostics"]["save_initial_monitor"])
        self.assertFalse(config["progressive_aug"]["enabled"])
        self.assertEqual(config["train"]["seed"], 42)
        self.assertTrue(config["train"]["deterministic"])

    def test_stage0_long_config_invariants(self):
        config = load_config(
            "config/train/stage2_refine_v6_stage0_long.yaml"
        )
        semi = config["semi_supervised"]
        self.assertEqual(semi["epochs"], 20)
        self.assertEqual(semi["labeled_steps_per_epoch"], 62)
        self.assertEqual(semi["refine_training"]["refine_only_epochs"], 20)
        self.assertEqual(semi["checkpoint_interval"], 5)
        self.assertFalse(config["progressive_aug"]["enabled"])
        self.assertFalse(semi["use_unlabeled"])

    def test_stage0_continue15_config_invariants(self):
        config = load_config(
            "config/train/stage2_refine_v6_stage0_continue15.yaml"
        )
        semi = config["semi_supervised"]
        self.assertEqual(semi["epochs"], 15)
        self.assertEqual(semi["learning_rate"], 2.0e-5)
        self.assertEqual(semi["flat_epochs"], 8)
        self.assertEqual(semi["decay_end_factor"], 0.25)
        self.assertEqual(semi["labeled_steps_per_epoch"], 62)
        self.assertEqual(semi["refine_training"]["refine_only_epochs"], 15)
        self.assertEqual(
            semi["init_from_checkpoint"],
            "outputs/stage2_refine_v6_stage0_long/best_model_stage2.pth",
        )
        self.assertFalse(config["progressive_aug"]["enabled"])
        self.assertFalse(semi["use_unlabeled"])

    def test_e1_physaug15_config_invariants(self):
        config = load_config(
            "config/train/stage2_refine_v6_e1_physaug15.yaml"
        )
        semi = config["semi_supervised"]
        aug = config["progressive_aug"]
        self.assertEqual(semi["epochs"], 15)
        self.assertEqual(semi["learning_rate"], 1.0e-5)
        self.assertEqual(semi["labeled_steps_per_epoch"], 62)
        self.assertEqual(semi["refine_training"]["refine_only_epochs"], 15)
        self.assertEqual(
            semi["init_from_checkpoint"],
            "outputs/stage2_refine_v6_stage0_continue15/best_model_stage2.pth",
        )
        self.assertFalse(semi["use_unlabeled"])
        self.assertTrue(aug["enabled"])
        self.assertEqual(aug["policy"], "physical_v1")
        self.assertEqual(aug["max_prob"], 0.60)
        self.assertEqual((aug["min_ops"], aug["max_ops"]), (1, 2))
        self.assertEqual(aug["op_weights"]["noise"], 0.0)
        self.assertEqual(aug["gaussian_noise_std"], 0.0)

    def test_validation_reports_boundary_haze_metrics(self):
        target = torch.tensor(
            [[
                [[0.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]]
        )
        loader = [{
            "image": torch.zeros(1, 3, 2, 2),
            "target": target,
            "weight": torch.ones(1, 1, 2, 2),
        }]
        metrics = validate(
            _FixedPredictionModel(), loader, _zero_criterion, torch.device("cpu")
        )
        self.assertAlmostEqual(metrics["boundary_pos_mean"], 0.7, places=6)
        self.assertAlmostEqual(metrics["boundary_bg_mean"], 0.3, places=6)
        self.assertAlmostEqual(metrics["boundary_prob_gap"], 0.4, places=6)
        self.assertAlmostEqual(metrics["boundary_recall_035"], 1.0, places=6)
        self.assertAlmostEqual(
            metrics["boundary_bg_fp_rate_035"], 0.5, places=6
        )


if __name__ == "__main__":
    unittest.main()
