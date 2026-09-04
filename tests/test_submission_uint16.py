# -*- coding: utf-8 -*-
"""Official uint16 submission-format regression tests."""

import json

import cv2
import numpy as np
import pytest

from tools.package_submission import validate_pair


def _write_pair(tmp_path, dtype):
    image_path = tmp_path / "test_001.jpg"
    prediction_dir = tmp_path / "prediction"
    prediction_dir.mkdir()
    assert cv2.imwrite(str(image_path), np.zeros((8, 10, 3), dtype=np.uint8))

    instance_map = np.zeros((8, 10), dtype=dtype)
    instance_map[:, :5] = 1
    instance_map[:, 5:] = 300
    assert cv2.imwrite(
        str(prediction_dir / "test_001_inst.png"), instance_map
    )
    (prediction_dir / "test_001_class.json").write_text(
        json.dumps({"1": 1, "300": 0}), encoding="utf-8"
    )
    return image_path, prediction_dir


def test_submission_validator_accepts_uint16_id_above_255(tmp_path):
    image_path, prediction_dir = _write_pair(tmp_path, np.uint16)
    _, _, count = validate_pair(image_path, prediction_dir)
    assert count == 2


def test_submission_validator_rejects_uint8_mask(tmp_path):
    image_path, prediction_dir = _write_pair(tmp_path, np.uint16)
    instance_path = prediction_dir / "test_001_inst.png"
    uint8_map = cv2.imread(str(instance_path), cv2.IMREAD_UNCHANGED).astype(np.uint8)
    assert cv2.imwrite(str(instance_path), uint8_map)
    with pytest.raises(ValueError, match="uint16"):
        validate_pair(image_path, prediction_dir)
