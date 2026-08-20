# -*- coding: utf-8 -*-
"""GDA-MIM and SAM2-to-LabelMe regression tests."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from data.mim_dataset import MaskedMetallographyDataset
from models.gda_mim import (
    GenerativeDomainAdapterPyramid,
    MIMReconstructionDecoder,
    gda_reconstruction_loss,
)
from tools.export_sam2_labelme import (
    records_to_labelme,
    select_non_overlapping_masks,
)


class GDAMIMTest(unittest.TestCase):
    def test_zero_gates_preserve_original_features_exactly(self):
        channels = (8, 16, 32, 64)
        gda = GenerativeDomainAdapterPyramid(channels, bottleneck_ratio=4)
        features = [
            torch.randn(2, channels[i], 16 // (2 ** i), 16 // (2 ** i))
            for i in range(4)
        ]
        adapted = gda(features, gated=True)
        for original, output in zip(features, adapted):
            self.assertTrue(torch.equal(original, output))

    def test_reconstruction_loss_backpropagates(self):
        decoder = MIMReconstructionDecoder((8, 16, 32, 64), hidden_channels=16)
        features = [
            torch.randn(1, channels, 16 // (2 ** i), 16 // (2 ** i), requires_grad=True)
            for i, channels in enumerate((8, 16, 32, 64))
        ]
        prediction = decoder(features, (64, 64))
        target = torch.rand_like(prediction)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 12:52, 12:52] = 1.0
        loss, details = gda_reconstruction_loss(prediction, target, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(details["gradient"], 0.0)
        self.assertIsNotNone(features[0].grad)

    def test_holdout_manifest_is_excluded_from_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                image = np.full((80, 96, 3), 40 + 20 * index, dtype=np.uint8)
                cv2.imwrite(str(root / f"train_{index:03d}.jpg"), image)
            manifest = root / "holdout.txt"
            manifest.write_text("train_001.jpg\n", encoding="utf-8")
            train = MaskedMetallographyDataset(
                str(root), str(manifest), split="train", crop_size=64
            )
            holdout = MaskedMetallographyDataset(
                str(root), str(manifest), split="holdout", crop_size=64
            )
            self.assertEqual(len(train), 2)
            self.assertEqual(len(holdout), 1)
            self.assertEqual(Path(holdout.samples[0]).stem, "train_001")
            sample = train[0]
            self.assertEqual(tuple(sample["input"].shape), (3, 64, 64))
            self.assertGreater(float(sample["mask"].mean()), 0.45)

    def test_sam2_records_export_as_unverified_labelme_polygons(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[5:25, 6:26] = True
        record = {
            "segmentation": mask,
            "area": int(mask.sum()),
            "predicted_iou": 0.92,
            "stability_score": 0.96,
        }
        document = records_to_labelme([record], "sample.jpg", mask.shape)
        self.assertEqual(document["imagePath"], "sample.jpg")
        self.assertTrue(document["flags"]["training_use_forbidden_until_verified"])
        self.assertEqual(document["shapes"][0]["label"], "sam2_candidate")
        self.assertFalse(document["shapes"][0]["flags"]["human_verified"])

    def test_sam2_overlap_filter_rejects_duplicate_masks(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[4:20, 4:20] = True
        records = [
            {"segmentation": mask, "area": int(mask.sum()), "predicted_iou": 0.95, "stability_score": 0.98},
            {"segmentation": mask.copy(), "area": int(mask.sum()), "predicted_iou": 0.90, "stability_score": 0.98},
        ]
        selected = select_non_overlapping_masks(
            records,
            mask.shape,
            min_area=10,
            max_area_fraction=0.9,
            max_overlap_fraction=0.3,
            max_masks=255,
        )
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
