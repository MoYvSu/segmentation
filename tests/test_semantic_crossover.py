# -*- coding: utf-8 -*-
"""固定最终实例图的交叉类别投票与对角门槛。"""

import json
import zipfile

import cv2
import numpy as np
import pytest

from tools.semantic_crossover import revote_instances, run_crossover
from utils.semantic_vote import instance_semantic_vote


VOTING = {"semantic_vote_mode": "probability_mean", "semantic_vote_erode_width": 0,
          "semantic_vote_threshold": 0.5}


def setup_sources(tmp_path):
    raw = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint16)
    restored = np.array([[10, 10, 10, 10], [20, 20, 20, 20]], dtype=np.uint16)
    raw_prob = np.array([[.9, .9, .1, .1], [.9, .9, .1, .1]], dtype=np.float32)
    restored_prob = np.array([[.8, .8, .8, .8], [.3, .3, .3, .3]], dtype=np.float32)
    result = {}
    for name, mask, probability, classes in (
        ("raw", raw, raw_prob, {"1": 1, "2": 0}),
        ("restored", restored, restored_prob, {"10": 1, "20": 0}),
    ):
        directory = tmp_path / (name + "_prob")
        directory.mkdir()
        np.savez(directory / "sample.npz", ferrite_probability=probability)
        package = tmp_path / (name + ".zip")
        success, png = cv2.imencode(".png", mask)
        assert success
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("sample_inst.png", png.tobytes())
            archive.writestr("sample_class.json", json.dumps(classes))
        result[name] = (package, directory)
    return result


def invoke(sources, output, voting=VOTING):
    return run_crossover(sources["raw"][0], sources["restored"][0], sources["raw"][1],
                         sources["restored"][1], output, voting)


def test_cross_only_changes_class_json_and_preserves_noncontiguous_ids_and_png_bytes(tmp_path):
    sources = setup_sources(tmp_path)
    report = invoke(sources, tmp_path / "cross")
    assert report["diagonal_exact"] and report["image_count"] == 1
    assert len(report["cells"]) == 4
    assert report["cells"]["raw_restored"]["summary"]["changed_instances"] == 1
    assert report["cells"]["restored_raw"]["summary"]["changed_instances"] == 1
    for cell, geometry, expected in (
        ("raw_restored", "raw", {"1": 1, "2": 1}),
        ("restored_raw", "restored", {"10": 0, "20": 0}),
    ):
        with zipfile.ZipFile(sources[geometry][0]) as source, zipfile.ZipFile(report["packages"][cell]["path"]) as cross:
            assert cross.namelist() == source.namelist()
            assert cross.read("sample_inst.png") == source.read("sample_inst.png")
            assert json.loads(cross.read("sample_class.json")) == expected


def test_diagonal_mismatch_writes_report_and_never_cross_packages(tmp_path):
    sources = setup_sources(tmp_path)
    np.savez(sources["raw"][1] / "sample.npz", ferrite_probability=np.ones((2, 4), np.float32))
    output = tmp_path / "cross"
    with pytest.raises(ValueError, match="Original diagonal"):
        invoke(sources, output)
    assert list(output.glob("*.zip")) == []
    report = json.loads((output / "report.json").read_text())
    assert not report["diagonal_exact"]
    assert report["diagonal_mismatches"][0]["ids"] == ["2"]


def test_cropped_vote_matches_exact_deployment_float32_mean_and_strict_tie():
    rng = np.random.default_rng(123)
    labels = rng.integers(0, 7, size=(43, 51)).astype(np.uint16)
    probability = rng.random(labels.shape).astype(np.float32)
    labels[3:5, 4:6] = 65535
    probability[labels == 65535] = .5
    before = labels.copy()
    classes, scores = revote_instances(labels, probability)
    for value in np.unique(labels):
        if value == 0:
            continue
        cls, score = instance_semantic_vote(labels == value, probability > .5,
            semantic_probability=probability, mode="probability_mean", erode_width=0, threshold=.5)
        assert classes[str(value)] == cls
        assert scores[str(value)] == score
    assert classes["65535"] == 0 and scores["65535"] == .5
    assert np.array_equal(labels, before)


@pytest.mark.parametrize("field,value", [("semantic_vote_mode", "hard_majority"),
                                        ("semantic_vote_erode_width", 1), ("semantic_vote_threshold", .49)])
def test_non_deployed_voting_rejected(tmp_path, field, value):
    sources = setup_sources(tmp_path)
    with pytest.raises(ValueError, match="unexpected deployed"):
        invoke(sources, tmp_path / "cross", {**VOTING, field: value})


@pytest.mark.parametrize("probability", [np.ones((3, 4), np.float32), np.ones((2, 4), np.float16),
                                         np.full((2, 4), np.nan, np.float32), np.full((2, 4), 1.01, np.float32)])
def test_invalid_native_cache_rejected_without_package(tmp_path, probability):
    sources = setup_sources(tmp_path)
    np.savez(sources["raw"][1] / "sample.npz", ferrite_probability=probability)
    with pytest.raises(ValueError, match="probability"):
        invoke(sources, tmp_path / "cross")
    assert not list(tmp_path.rglob("submission_*.zip"))


def test_missing_cache_image_rejected(tmp_path):
    sources = setup_sources(tmp_path)
    (sources["raw"][1] / "sample.npz").unlink()
    with pytest.raises(ValueError, match="cache cohort"):
        invoke(sources, tmp_path / "cross")
