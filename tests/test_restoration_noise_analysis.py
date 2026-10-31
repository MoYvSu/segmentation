# -*- coding: utf-8 -*-
"""噪声诊断的 CPU 合成门禁；不用赛题数据，不把标签编号变化当几何错误。"""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from tools import analyze_restoration_noise as analysis


def sample():
    instances = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint16)
    maps = dict(instances=instances, boundary=np.full(instances.shape, .5, np.float32),
                geometry_semantic=np.zeros(instances.shape, np.float32),
                raw_probability=np.full(instances.shape, .5, np.float32))
    return maps, {"1": 1, "2": 0}


def test_id_permutation_is_identical_geometry_and_phase():
    reference, classes = sample()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["instances"] = np.where(reference["instances"] == 1, 42, 17).astype(np.uint16)
    comparison = analysis.compare_maps(reference, candidate, classes, {"42": 1, "17": 0})
    assert comparison["instances"]["mean_matched_iou"] == 1
    assert comparison["instances"]["matched_class_changes"] == 0
    assert comparison["phase"]["changed_fraction"] == 0
    assert comparison["ferrite_relative_delta"]["mean_area"] == 0


def test_boundary_threshold_crossing_and_sigmoid_are_separate():
    reference, classes = sample()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["boundary"][0, 0] = .7
    candidate["boundary"][0, 1] = .4
    candidate["geometry_semantic"][0, 0] = np.log(3)
    result = analysis.compare_maps(reference, candidate, classes, classes)
    assert result["boundary"]["threshold_crossing"] == {"0.65": .125, "0.45": .125}
    assert result["boundary"]["mae"] == pytest.approx(.3 / 8)
    assert result["geometry_semantic"]["probability_mae"] == pytest.approx(.25 / 8)
    assert result["raw_semantic"]["probability_mae"] == 0


def test_threshold_equality_uses_deployed_strict_greater_than():
    reference, classes = sample()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["geometry_semantic"][0, 0] = .1
    result = analysis.compare_maps(reference, candidate, classes, classes)
    assert result["geometry_semantic"]["threshold_crossing"] == .125


def test_split_and_class_change_are_explicit_not_label_pixel_difference():
    reference, classes = sample()
    candidate = {key: value.copy() for key, value in reference.items()}
    candidate["instances"][1, :2] = 3
    candidate_classes = {"1": 0, "2": 0, "3": 1}
    result = analysis.compare_maps(reference, candidate, classes, candidate_classes)
    assert result["instances"]["matched_count"] == 2
    assert result["instances"]["unmatched_candidate"] == 1
    assert result["instances"]["ferrite_to_pearlite"] == 1
    assert result["phase"]["changed_fraction"] == .25
    assert result["ferrite"]["mean_area"] == 2


def test_no_ferrite_or_no_instances_is_undefined_not_zero_area():
    maps, _ = sample()
    stats = analysis.ferrite_stats(maps["instances"], {"1": 0, "2": 0})
    assert stats["count"] == 0 and stats["mean_area"] is None
    assert analysis.relative_delta(2, 0) is None
    empty = np.zeros((2, 2), np.uint16)
    matches = analysis.match_instances(empty, empty, {}, {})
    assert matches["matched_count"] == 0 and matches["mean_matched_iou"] is None


@pytest.mark.parametrize("classes", [{"1": 1}, {"1": 1, "2": 2}, {"1": True, "2": 0},
                                       {"01": 1, "2": 0}, {"0": 0, "1": 1, "2": 0}])
def test_invalid_or_missing_class_ids_fail(classes):
    maps, _ = sample()
    with pytest.raises(ValueError):
        analysis.class_lookup(maps["instances"], classes)


def test_missing_seed_is_reported_and_small_series_does_not_fake_zero_std():
    reference = sample()
    row, std = analysis.series_summary({"local_s0": reference}, ["local_s0", "local_s1"], reference)
    assert not row["complete"] and row["missing_cases"] == ["local_s1"]
    assert row["insufficient_repeats"] and std is None


def test_series_std_and_area_cv_use_population_definition():
    reference = sample()
    other, classes = sample()
    other["boundary"][:] = .7
    row, std = analysis.series_summary({"local_s0": reference, "local_s1": (other, classes)},
                                       ["local_s0", "local_s1"], reference)
    np.testing.assert_allclose(std, .1, atol=1e-7)
    assert row["complete"] and row["ferrite_cv"]["mean_area"] == 0
    assert analysis.coefficient_of_variation([2, 4]) == pytest.approx(1 / 3)


def store(root, source, case, maps, classes):
    (root / "maps" / source).mkdir(parents=True, exist_ok=True)
    (root / "results" / case).mkdir(parents=True, exist_ok=True)
    np.savez(root / "maps" / source / f"{case}.npz", **maps)
    (root / "results" / case / f"{source}_class.json").write_text(json.dumps(classes), encoding="utf-8")
    assert cv2.imwrite(str(root / "results" / case / f"{source}_inst.png"), maps["instances"])


def test_complete_report_and_gallery_and_explicit_missing_case(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (4, 2), "gray").save(image_dir / "synthetic.png")
    maps, classes = sample()
    for case in ("control", "local_s0", "local_s1", "local_zero"):
        value = {key: array.copy() for key, array in maps.items()}
        if case == "local_s1":
            value["boundary"][0, 0] = .7
        store(root, "synthetic", case, value, classes)
    record = dict(expected_cases=["control", "local_s0", "local_s1", "local_zero"], sources=["synthetic"])
    (root / "record.json").write_text(json.dumps(record), encoding="utf-8")
    summary = analysis.analyze(root, image_dir)
    assert summary["complete"]
    assert (root / "gallery" / "index.html").is_file()
    assert (root / "gallery" / "synthetic_local_unstable.jpg").is_file()
    assert (root / "summary.md").read_text(encoding="utf-8").startswith("# 恢复器")
    summary = analysis.analyze(root, image_dir, expected_cases=record["expected_cases"] + ["global_zero"])
    assert not summary["complete"]
    assert any(item["case"] == "global_zero" for item in summary["failures"])


def test_archived_png_is_checked_against_npz(tmp_path):
    maps, classes = sample()
    store(tmp_path, "synthetic", "control", maps, classes)
    wrong = maps["instances"].copy()
    wrong[0, 0] = 2
    cv2.imwrite(str(tmp_path / "results/control/synthetic_inst.png"), wrong)
    with pytest.raises(ValueError, match="PNG differs"):
        analysis.load_case(tmp_path, "synthetic", "control")


def test_complete_maps_cannot_hide_incomplete_runner(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (4, 2), "gray").save(image_dir / "synthetic.png")
    root = tmp_path / "run"
    store(root, "synthetic", "control", *sample())
    (root / "record.json").write_text(json.dumps(dict(format="restoration_noise_probe_v1",
        complete=False, expected_cases=["control"], sources=["synthetic"])), encoding="utf-8")
    summary = analysis.analyze(root, image_dir)
    assert not summary["complete"] and summary["failures"][0]["case"] == "record"


def test_instability_roi_is_bounded_and_tracks_noise_variation():
    std = np.zeros((8, 10), dtype=np.float32)
    std[5:, 7:] = 1
    assert analysis.instability_roi(std, 3) == (7, 5, 10, 8)
    assert analysis.instability_roi(std, 100) == (0, 0, 10, 8)


def quality_payload():
    rows = []
    for source, numbers in (("a", (1., 3.)), ("b", (10., 10., 10.))):
        for mode in ("single", "pair", "zero"):
            values = numbers if mode != "zero" else (4.,)
            for offset, number in enumerate(values):
                rows.append(dict(source=source, condition="spatial", mode=mode, offset=offset,
                                 **{key: number for key in analysis.QUALITY_METRICS}))
    return dict(complete=True, plan=[dict(source="a"), dict(source="b")], rows=rows)


def test_quality_macro_average_weights_each_source_equally_not_each_offset():
    result = analysis.quality_summary(quality_payload())
    single = result["groups"]["spatial"]["single"]
    assert single["macro_mean_metrics"]["rgb_mse"] == 6  # mean(mean(1,3), mean(10,10,10))
    assert single["rgb_mse_across_offset_std_macro_mean"] == .5  # mean(std(1,3),std(10,10,10))
    assert single["valid_source_counts"]["weak_directional_retention"] == 2
    assert "not prediction pixel variance" in result["scalar_std_warning"]
    assert result["groups"]["spatial"]["zero"]["rgb_mse_across_offset_std_macro_mean"] is None


def test_quality_missing_weak_pixels_remains_undefined_with_explicit_counts():
    payload = quality_payload()
    for row in payload["rows"]:
        if row["source"] == "a":
            row["weak_directional_retention"] = None
    result = analysis.quality_summary(payload)["groups"]["spatial"]["single"]
    assert result["macro_mean_metrics"]["weak_directional_retention"] == 10
    assert result["valid_source_counts"]["weak_directional_retention"] == 1


@pytest.mark.parametrize("change", ["incomplete", "duplicate", "missing_mode", "bad_plan", "nonfinite"])
def test_quality_incomplete_corrupt_or_inconsistent_receipt_rejected(change):
    payload = quality_payload()
    if change == "incomplete":
        payload["complete"] = False
    elif change == "duplicate":
        payload["rows"].append(payload["rows"][0])
    elif change == "missing_mode":
        payload["rows"] = [row for row in payload["rows"] if row["mode"] != "pair"]
    elif change == "bad_plan":
        payload["plan"][0]["source"] = "different"
    else:
        payload["rows"][0]["rgb_mse"] = float("nan")
    with pytest.raises(ValueError):
        analysis.quality_summary(payload)


def test_present_but_incomplete_quality_marks_whole_analysis_incomplete(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (4, 2), "gray").save(image_dir / "synthetic.png")
    root = tmp_path / "run"
    store(root, "synthetic", "control", *sample())
    (root / "quality").mkdir()
    (root / "quality/metrics.json").write_text(json.dumps(dict(complete=False)), encoding="utf-8")
    summary = analysis.analyze(root, image_dir, ["synthetic"], ["control"])
    assert not summary["complete"] and summary["quality"]["complete"] is False
    assert summary["failures"][0]["case"] == "quality"


def test_fixed_boundary_has_zero_std_but_no_fictitious_instability_roi(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (4, 2), "gray").save(image_dir / "synthetic.png")
    root = tmp_path / "run"
    for case in ("control", "global_s0", "global_s1"):
        store(root, "synthetic", case, *sample())
    summary = analysis.analyze(root, image_dir, ["synthetic"], ["control", "global_s0", "global_s1"])
    assert summary["complete"]
    assert summary["sources"]["synthetic"]["series"]["global"]["boundary_std"]["maximum"] == 0
    assert not (root / "gallery/synthetic_global_unstable.jpg").exists()


@pytest.mark.parametrize("name", ["..", "../bad", "bad/name", "bad\\name", ""])
def test_path_escape_names_rejected(name):
    with pytest.raises(ValueError):
        analysis.safe_name(name)
