# -*- coding: utf-8 -*-
"""CPU contracts for direct mask-set instances, restoration and evaluation."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.evaluate_mask_set import evaluate_model, resolve_evaluation_paths
from utils.mask_set_inference import infer_mask_set_image, postprocess_mask_set


def test_overlap_competes_by_class_times_mask_and_keeps_both_phases_as_instances():
    masks = np.array([[[8, 0.2, -8]], [[-8, 8, 8]], [[8, 8, 8]]], dtype=np.float32)
    classes = np.array([[5, 0, -5], [0, 2, -5], [0, 0, 8]], dtype=np.float32)
    result, mapping, audit = postprocess_mask_set(masks, classes, (1, 3), (0, 0), input_size=3)
    assert result.tolist() == [[1, 2, 2]]
    assert mapping == {1: 0, 2: 1}
    assert audit["eligible_queries"] == 2
    assert audit["unassigned_pixels"] == 0


def test_two_pearlite_queries_stay_independent_and_exact_ties_are_deterministic():
    masks = np.array([[[8, 8, -8]], [[-8, 8, 8]]], dtype=np.float32)
    classes = np.array([[8, -8, -8], [8, -8, -8]], dtype=np.float32)
    first = postprocess_mask_set(masks, classes, (1, 3), (0, 0), input_size=3)
    second = postprocess_mask_set(masks, classes, (1, 3), (0, 0), input_size=3)
    assert first[0].tolist() == [[1, 1, 2]]
    assert first[1] == {1: 0, 2: 0}
    np.testing.assert_array_equal(first[0], second[0])


def test_odd_letterbox_padding_is_removed_after_input_grid_restoration():
    masks = torch.tensor([[[8.0, 8.0, -8.0, -8.0],
                           [3.0, -3.0, -8.0, -8.0],
                           [-8.0, -8.0, 8.0, 8.0],
                           [8.0, 8.0, 8.0, 8.0]]])
    classes = torch.tensor([[8.0, -8.0, -8.0]])
    result, _, audit = postprocess_mask_set(masks, classes, (4, 7), (3, 0), input_size=8)
    expanded = F.interpolate(masks[:, None], size=(8, 8), mode="bilinear", align_corners=False)
    restored = F.interpolate(expanded[:, :, :5, :8], size=(4, 7), mode="bilinear", align_corners=False)
    expected = (restored[0, 0] >= 0).numpy().astype(np.uint16)
    np.testing.assert_array_equal(result, expected)
    assert audit["padding_hw"] == [3, 0]


def test_area_and_overlap_rejection_leaves_holes_without_reassignment():
    masks = np.array([[[8, 8, 8]], [[-8, 12, -8]]], dtype=np.float32)
    classes = np.array([[8, -8, -8], [8, -8, -8]], dtype=np.float32)
    result, mapping, audit = postprocess_mask_set(
        masks, classes, (1, 3), (0, 0), input_size=3, min_area=2,
    )
    assert result.tolist() == [[1, 0, 1]]
    assert mapping == {1: 0}
    assert audit["query_results"][1]["accepted"] is False
    result, mapping, _ = postprocess_mask_set(
        masks, classes, (1, 3), (0, 0), input_size=3, min_overlap_ratio=0.9,
    )
    assert result.tolist() == [[0, 1, 0]]
    assert mapping == {1: 0}


@pytest.mark.parametrize("kind", ["empty_class", "low_class_score", "empty_mask", "zero_queries"])
def test_empty_predictions_return_uint16_and_empty_class_map(kind):
    masks = np.full((1, 2, 2), 8, dtype=np.float32)
    classes = np.array([[8, 0, -8]], dtype=np.float32)
    if kind == "empty_class":
        classes = np.array([[0, 0, 8]], dtype=np.float32)
    elif kind == "low_class_score":
        classes[:] = 0
    elif kind == "empty_mask":
        masks[:] = -8
    else:
        masks, classes = masks[:0], classes[:0]
    result, mapping, audit = postprocess_mask_set(masks, classes, (3, 5), (0, 0), input_size=5)
    assert result.dtype == np.uint16
    assert result.shape == (3, 5)
    assert not result.any()
    assert mapping == {}
    assert audit["unassigned_pixels"] == 15


def test_uint16_ids_above_255_roundtrip_png(tmp_path):
    count = 256
    masks = np.full((count, 16, 16), -8, dtype=np.float32)
    masks.reshape(count, -1)[np.arange(count), np.arange(count)] = 8
    classes = np.tile(np.array([8, -8, -8], dtype=np.float32), (count, 1))
    result, mapping, _ = postprocess_mask_set(masks, classes, (16, 16), (0, 0), input_size=16)
    assert result.dtype == np.uint16
    assert len(mapping) == 256
    assert result.max() == 256
    assert set(np.unique(result)) == set(mapping)
    path = tmp_path / "instances.png"
    assert cv2.imwrite(str(path), result)
    restored = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert restored.dtype == np.uint16
    np.testing.assert_array_equal(restored, result)


class ConstantMaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None
        self.observed_training = None

    def forward(self, image):
        self.seen = image.detach().clone()
        self.observed_training = self.training
        return {"pred_logits": torch.tensor([[[0.0, 8.0, -8.0]]], device=image.device),
                "pred_masks": torch.full((1, 1, 4, 4), 8.0, device=image.device)}


def test_inference_respects_input_size_rgb_range_and_restores_training_mode():
    model = ConstantMaskModel().train()
    rgb = np.full((3, 7, 3), 128, dtype=np.uint8)
    config = {"mask_set": {"input_size": 8}}
    output, classes, audit = infer_mask_set_image(model, rgb, config, torch.device("cpu"))
    assert model.training is True and model.observed_training is False
    assert tuple(model.seen.shape) == (1, 3, 8, 8)
    torch.testing.assert_close(model.seen, torch.full_like(model.seen, 128 / 255))
    assert output.shape == (3, 7) and output.dtype == np.uint16
    assert np.all(output == 1) and classes == {1: 1}
    assert audit["padding_hw"] == [5, 0]


def test_fixed_split_changes_fail_and_explicit_names_are_marked(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("train", "val", "test"):
        (raw / f"{name}.png").touch()
    split = {"train": ["train"], "val": ["val"]}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    config = {"paths": {"project_root": str(tmp_path), "raw_data_dir": "raw"},
              "mask_set": {"split_file": "split.json"}}
    paths, override, _ = resolve_evaluation_paths(config, {"split": split})
    assert [path.stem for path in paths] == ["val"] and not override
    paths, override, _ = resolve_evaluation_paths(config, {"split": split}, names=["test"])
    assert [path.stem for path in paths] == ["test"] and override
    split_path.write_text(json.dumps({"train": ["val"], "val": ["train"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="split differs"):
        resolve_evaluation_paths(config, {"split": split})


def test_evaluation_masks_unknown_only_for_metrics_and_saves_complete_predictions(tmp_path):
    raw, target = tmp_path / "raw", tmp_path / "gt"
    raw.mkdir()
    target.mkdir()
    rgb = np.full((4, 6, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(raw / "val.png"), rgb)
    gt = np.ones((4, 6), dtype=np.uint16)
    gt[:, -2:] = 0
    np.savez_compressed(target / "val_gt.npz", instance_map=gt, original_covered=gt > 0,
                        filled=np.zeros_like(gt, dtype=bool), residual_unknown=gt == 0)
    (target / "val_class.json").write_text('{"1": 1}', encoding="utf-8")
    config = {"paths": {"project_root": str(tmp_path), "raw_data_dir": "raw"},
              "mask_set": {"input_size": 8, "manual_target_dir": "gt"}}
    payload = {"split": {"train": [], "val": ["val"]}, "epoch": 2}
    output_dir = tmp_path / "result"
    report = evaluate_model(ConstantMaskModel(), payload, config, output_dir, torch.device("cpu"))
    stored = cv2.imread(str(output_dir / "predictions" / "val_inst.png"), cv2.IMREAD_UNCHANGED)
    assert stored.dtype == np.uint16 and np.all(stored == 1)
    assert report["aggregate"]["instance_miou_valid"] == 1.0
    assert report["aggregate"]["score_total"] == 100.0
    assert report["images"][0]["ignored_unknown_pixels"] == 8
    assert report["scope"] == "checkpoint_fixed_validation"


@pytest.mark.parametrize("keyword,value", [
    ("score_threshold", float("nan")), ("mask_threshold", 1.1),
    ("min_overlap_ratio", -0.1), ("min_area", 0), ("input_size", 0),
])
def test_invalid_postprocess_settings_are_rejected(keyword, value):
    with pytest.raises(ValueError):
        postprocess_mask_set(np.zeros((1, 2, 2)), np.zeros((1, 3)), (2, 2), (0, 0),
                             **{keyword: value})
