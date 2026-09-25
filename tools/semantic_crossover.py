# -*- coding: utf-8 -*-
"""固定最终实例 ID 图，交叉两套原尺寸语义概率；仅重做最后的类别投票。

输入缓存为每图 <stem>.npz，其中 ferrite_probability 是 H×W float32。
概率必须来自部署的原尺寸 logits 还原后 sigmoid；这里不缩放、不推理。
最终实例图上游已经使用语义参与分水岭，不能将本实验称为纯几何头消融。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.semantic_vote import instance_semantic_vote


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_stems(archive):
    names = archive.namelist()
    _require(len(names) == len(set(names)), "duplicate ZIP entries")
    _require(all("/" not in n and "\\" not in n for n in names), "expected a flat submission ZIP")
    stems = sorted(n.removesuffix("_inst.png") for n in names if n.endswith("_inst.png"))
    expected = {stem + suffix for stem in stems for suffix in ("_inst.png", "_class.json")}
    _require(stems and set(names) == expected, "ZIP must contain only complete instance/class pairs")
    return stems


def _read_prediction(archive, stem):
    png = archive.read(stem + "_inst.png")
    _require(png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR"
             and png[24:26] == bytes((16, 0)), f"{stem}: expected 16-bit grayscale PNG")
    instances = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    _require(instances is not None and instances.ndim == 2 and instances.dtype == np.uint16,
             f"{stem}: invalid instance image")
    classes = json.loads(archive.read(stem + "_class.json"), object_pairs_hook=_unique_object)
    _require(isinstance(classes, dict), f"{stem}: expected an ID-to-class object")
    _require(all(k.isdecimal() and str(int(k)) == k and 1 <= int(k) <= 65535 for k in classes),
             f"{stem}: invalid class ID")
    _require(all(type(v) is int and v in (0, 1) for v in classes.values()),
             f"{stem}: class must be integer 0/1")
    ids, counts = np.unique(instances, return_counts=True)
    areas = {str(int(i)): int(n) for i, n in zip(ids, counts) if i > 0}
    _require(set(classes) == set(areas), f"{stem}: mask IDs differ from class IDs")
    return instances, classes, areas


def _load_probability(directory, stem, shape):
    with np.load(Path(directory) / (stem + ".npz"), allow_pickle=False) as payload:
        _require("ferrite_probability" in payload.files, f"{stem}: missing ferrite_probability")
        probability = payload["ferrite_probability"]
    _require(probability.shape == shape and probability.dtype == np.float32,
             f"{stem}: expected native-shape float32 probability, got {probability.shape}/{probability.dtype}")
    _require(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)),
             f"{stem}: invalid probability values")
    return probability


def revote_instances(instances, ferrite_probability):
    """保持所有 ID/像素不变，复用部署均值投票；返回类别及均值分数。

    对实例包围框调用原函数，仅缩小临时数组；实例内取值顺序和 float32
    np.mean 与整图调用相同，包括 score==0.5 必须归珠光体的严格阈值。
    """
    _require(instances.ndim == 2 and instances.dtype == np.uint16, "instances must be uint16 HxW")
    _require(ferrite_probability.shape == instances.shape and ferrite_probability.dtype == np.float32,
             "probability must be float32 on the native instance grid")
    _require(np.isfinite(ferrite_probability).all()
             and np.all((ferrite_probability >= 0) & (ferrite_probability <= 1)), "invalid probability")
    classes, scores = {}, {}
    for value in np.unique(instances):
        if value == 0:
            continue
        mask = instances == value
        ys, xs = np.nonzero(mask)
        region = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        probability = ferrite_probability[region]
        cls, score = instance_semantic_vote(
            mask[region], probability > 0.5, semantic_probability=probability,
            mode="probability_mean", erode_width=0, threshold=0.5,
        )
        classes[str(int(value))], scores[str(int(value))] = int(cls), float(score)
    return classes, scores


def _cell_row(classes, scores, areas, original_classes):
    ferrite_ids = [i for i, cls in classes.items() if cls == 1]
    ferrite_pixels = sum(areas[i] for i in ferrite_ids)
    flips = [i for i in classes if classes[i] != original_classes[i]]
    p2f = [i for i in flips if classes[i] == 1]
    f2p = [i for i in flips if classes[i] == 0]
    return {
        "instances": len(classes), "ferrite": len(ferrite_ids), "pearlite": len(classes) - len(ferrite_ids),
        "ferrite_pixels": ferrite_pixels, "covered_pixels": sum(areas.values()),
        "ferrite_mean_area": ferrite_pixels / len(ferrite_ids) if ferrite_ids else None,
        "changed_instances": len(flips), "changed_pixels": sum(areas[i] for i in flips),
        "pearlite_to_ferrite_ids": p2f, "ferrite_to_pearlite_ids": f2p,
        "vote_scores": scores,
    }


def _write_cross_zip(source, output, classes_by_stem):
    """PNG 文件内容逐字节复制，禁止重新编码、重编号或后处理。"""
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as target:
        for info in original.infolist():
            if info.filename.endswith("_inst.png"):
                content = original.read(info.filename)
            else:
                stem = info.filename.removesuffix("_class.json")
                content = json.dumps(classes_by_stem[stem], ensure_ascii=False, indent=2).encode("utf-8")
            target.writestr(info, content)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output) as result:
        _require(result.namelist() == original.namelist(), "cross ZIP changed the entry names/order")
        for stem in classes_by_stem:
            _require(result.read(stem + "_inst.png") == original.read(stem + "_inst.png"),
                     f"{stem}: cross ZIP changed mask bytes")
            _require(json.loads(result.read(stem + "_class.json")) == classes_by_stem[stem],
                     f"{stem}: cross ZIP class write mismatch")
    return {"path": str(output), "sha256": _sha256(output), "bytes": Path(output).stat().st_size,
            "files": 2 * len(classes_by_stem), "mask_bytes_identical": True}


def run_crossover(raw_zip, restored_zip, raw_probability_dir, restored_probability_dir,
                  output_dir, inference_config, *, provenance=None):
    """输出四格报告及两个交叉 ZIP；两个原始对角必须先逐 ID 精确复现。"""
    actual_vote = {
        "mode": inference_config.get("semantic_vote_mode", "hard_majority"),
        "erode_width": inference_config.get("semantic_vote_erode_width", 0),
        "threshold": inference_config.get("semantic_vote_threshold", 0.5),
    }
    _require(actual_vote == {"mode": "probability_mean", "erode_width": 0, "threshold": 0.5},
             f"unexpected deployed voting configuration: {actual_vote}")
    output = Path(output_dir)
    package_paths = {
        "raw_restored": output / "submission_graw_srestored.zip",
        "restored_raw": output / "submission_grestored_sraw.zip",
    }
    report_path = output / "report.json"
    _require(not any(p.exists() for p in [report_path, *package_paths.values()]), "crossover outputs already exist")
    sources = {"raw": Path(raw_zip), "restored": Path(restored_zip)}
    caches = {"raw": Path(raw_probability_dir), "restored": Path(restored_probability_dir)}
    report = {
        "status": "checking", "voting": actual_vote, "official_scores": None,
        "scope": "Only final class voting is crossed. Final instance partitions already include upstream semantic watershed effects.",
        "cells": {f"{g}_{s}": {"final_instance_source": g, "semantic_probability_source": s}
                  for g in sources for s in sources},
        "sources": {g: {"package": str(p), "sha256": _sha256(p), "probability_dir": str(caches[g])}
                    for g, p in sources.items()},
        "provenance": provenance, "images": [], "diagonal_mismatches": [],
    }
    crossed_classes = {key: {} for key in package_paths}
    with zipfile.ZipFile(raw_zip) as raw_archive, zipfile.ZipFile(restored_zip) as restored_archive:
        archives = {"raw": raw_archive, "restored": restored_archive}
        stems = _archive_stems(raw_archive)
        _require(stems == _archive_stems(restored_archive), "raw/restored ZIP cohorts differ")
        for kind, directory in caches.items():
            _require({p.stem for p in directory.glob("*.npz")} == set(stems), f"{kind}: semantic cache cohort differs from ZIP")
        for stem in stems:
            predictions = {g: _read_prediction(archive, stem) for g, archive in archives.items()}
            shape = predictions["raw"][0].shape
            _require(shape == predictions["restored"][0].shape, f"{stem}: raw/restored native shapes differ")
            probabilities = {s: _load_probability(directory, stem, shape) for s, directory in caches.items()}
            row = {"stem": stem, "shape": list(shape), "cells": {}}
            for geometry, (instances, original_classes, areas) in predictions.items():
                for semantic, probability in probabilities.items():
                    key = f"{geometry}_{semantic}"
                    classes, scores = revote_instances(instances, probability)
                    row["cells"][key] = _cell_row(classes, scores, areas, original_classes)
                    if geometry == semantic:
                        mismatches = [i for i in classes if classes[i] != original_classes[i]]
                        if mismatches:
                            report["diagonal_mismatches"].append({"stem": stem, "source": geometry,
                                "ids": mismatches, "expected": {i: original_classes[i] for i in mismatches},
                                "actual": {i: classes[i] for i in mismatches},
                                "scores": {i: scores[i] for i in mismatches}})
                    else:
                        crossed_classes[key][stem] = classes
            report["images"].append(row)
    report["image_count"] = len(stems)
    for key, metadata in report["cells"].items():
        metadata["summary"] = {field: sum(row["cells"][key][field] for row in report["images"])
            for field in ("instances", "ferrite", "pearlite", "ferrite_pixels", "covered_pixels", "changed_instances", "changed_pixels")}
    output.mkdir(parents=True, exist_ok=True)
    report["diagonal_exact"] = not report["diagonal_mismatches"]
    if not report["diagonal_exact"]:
        report["status"] = "diagonal_mismatch_no_packages"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        raise ValueError(f"Original diagonal classes did not reproduce; no cross ZIP written. See {report_path}")
    report["packages"] = {key: _write_cross_zip(sources[key.split("_")[0]], path, crossed_classes[key])
                          for key, path in package_paths.items()}
    report["status"] = "completed"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", required=True)
    parser.add_argument("--restored-zip", required=True)
    parser.add_argument("--raw-prob-dir", required=True)
    parser.add_argument("--restored-prob-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provenance", required=True, help="已有部署 provenance JSON，必须含 resolved_config.inference")
    args = parser.parse_args()
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    report = run_crossover(args.raw_zip, args.restored_zip, args.raw_prob_dir, args.restored_prob_dir,
                           args.output_dir, provenance["resolved_config"]["inference"],
                           provenance={"deployment_provenance": str(Path(args.provenance))})
    print(json.dumps({k: report[k] for k in ("status", "image_count", "diagonal_exact", "packages")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
