# -*- coding: utf-8 -*-
"""用冻结的主线部署链生成真实标明来源的实例伪标签及审核预览。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mask_set_pseudo_dataset import MAX_PSEUDO_IMAGES, PSEUDO_FORMAT
from models.fused_deployment import load_fused_deployment_model
from utils.affinity_deployment import (crop_affinity_boundary_output, crop_letterbox_output,
                                      postprocess, prepare_image, probability_to_logit)
from utils.config import load_config, project_path


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def excluded_names(config, cfg):
    split = json.loads(Path(project_path(config, cfg["split_file"])).read_text(encoding="utf-8"))
    held = Path(project_path(config, cfg["holdout_file"])).read_text(encoding="utf-8").splitlines()
    raw = Path(project_path(config, config["paths"]["raw_data_dir"]))
    return (set(split["train"] + split["val"])
            | {Path(n.strip()).stem for n in held if n.strip() and not n.lstrip().startswith("#")}
            | {p.stem for p in raw.glob("*.json")})


def select_candidates(raw, excluded, keep_fraction, seed, *, excluded_image_dir=None):
    """统一1024尺度的灰度Laplacian方差仅用于粗排清晰度，不代表分割正确率。"""
    rows, excluded_hashes = [], set()
    if excluded_image_dir is not None:
        for path in sorted(Path(excluded_image_dir).iterdir()):
            if path.stem in excluded and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                excluded_hashes.add(digest(path))
    for path in sorted(raw.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        if not path.stem.startswith("train_"):
            raise ValueError(f"unexpected training-pool filename: {path.name}")
        sha = digest(path)
        if path.stem in excluded:
            excluded_hashes.add(sha)
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"unreadable training image: {path}")
        scale = 1024 / max(gray.shape)
        resized = cv2.resize(gray, (max(1, round(gray.shape[1]*scale)), max(1, round(gray.shape[0]*scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        clarity = float(cv2.Laplacian(resized, cv2.CV_32F).var())
        rows.append(dict(stem=path.stem, image_name=path.name, source_sha256=sha,
                         native_shape=list(gray.shape), clarity_score=clarity))
    hashes, unique = set(), []
    for row in rows:
        sha = row["source_sha256"]
        if sha not in hashes and sha not in excluded_hashes:
            hashes.add(sha)
            unique.append(row)
    unique.sort(key=lambda row: (-row["clarity_score"], row["stem"]))
    retained = unique[:math.ceil(len(unique) * keep_fraction)]
    random.Random(seed).shuffle(retained)
    return retained, {"eligible_unique_images": len(unique), "clarity_pool_images": len(retained),
                      "clarity_keep_fraction": keep_fraction, "excluded_names": sorted(excluded),
                      "external_manual_content_checked": excluded_image_dir is not None,
                      "selection": "top clarity fraction, seeded shuffle; no GT or teacher score ranking"}


def render_preview(raw, directory, row):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    stem = row["stem"]
    rgb = cv2.cvtColor(cv2.imread(str(raw / row["image_name"])), cv2.COLOR_BGR2RGB)
    ids = cv2.imread(str(directory / "instances" / f"{stem}_inst.png"), cv2.IMREAD_UNCHANGED)
    classes = json.loads((directory / "instances" / f"{stem}_class.json").read_text())
    scale = min(1., 1000 / max(ids.shape))
    shape = (round(ids.shape[1]*scale), round(ids.shape[0]*scale))
    rgb = cv2.resize(rgb, shape, interpolation=cv2.INTER_AREA)
    ids = cv2.resize(ids, shape, interpolation=cv2.INTER_NEAREST)
    colors = np.random.default_rng(42).uniform(.22, .95, (int(ids.max())+1, 3))
    colors[0] = .12
    colored = colors[ids]
    colored[find_boundaries(ids, mode="inner")] = .03
    lut = np.full(int(ids.max())+1, -1)
    for key, cls in classes.items():
        if int(key) < len(lut):
            lut[int(key)] = cls
    semantic = np.zeros((*ids.shape, 3), dtype=float)
    semantic[lut[ids] == 0] = (.90, .35, .13)
    semantic[lut[ids] == 1] = (.15, .64, .90)
    overlay = .57 * (rgb / 255.) + .43 * semantic
    overlay[find_boundaries(ids, mode="inner")] = .02
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for ax, im, title in zip(axes, (rgb, colored, overlay),
                             ("Original", "Pseudo instance IDs (distinct colors)", "Classes: blue=ferrite, orange=pearlite")):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{stem} | FIXED MAINLINE PSEUDO LABEL | {row['instances']} instances | "
                 f"coverage={row['coverage']:.1%}", fontsize=13)
    fig.tight_layout()
    dest = directory / "previews"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / f"{stem}.png", dpi=130)
    plt.close(fig)


@torch.inference_mode()
def run(args):
    config = load_config(args.config)
    cfg = config["pseudo_generation"]
    count = int(cfg["target_count"])
    if not 1 <= count <= MAX_PSEUDO_IMAGES:
        raise ValueError("pseudo target_count must be in 1..249")
    fraction = float(cfg["clarity_keep_fraction"])
    if not 0 < fraction <= 1 or not 0 <= float(cfg["minimum_coverage"]) <= 1:
        raise ValueError("invalid pseudo selection limits")
    raw = Path(project_path(config, cfg["source_dir"]))
    directory = Path(project_path(config, args.output_dir or cfg["output_dir"]))
    previous = None
    if args.resume:
        previous = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("format") != PSEUDO_FORMAT or previous.get("generation_config") != cfg:
            raise ValueError("resume requires the same pseudo generation config and format")
        if not previous.get("selection", {}).get("external_manual_content_checked", False):
            raise ValueError("legacy pseudo pool lacks cross-directory content isolation; preserve it and use a reviewed disjoint subset or new output")
        if previous.get("status") == "complete":
            if previous.get("sample_count") != count:
                raise ValueError("completed dataset count changed")
            print("ALREADY_COMPLETE", directory, flush=True)
            return
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "instances").mkdir()
        (directory / "rejected").mkdir()
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    # 与已验证主线推理一致，FP32默认matmul设置，不借AMP加速改变伪标签。
    if previous is not None:
        saved = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        candidates = saved.pop("candidates")
        selection = saved
        if set(selection["excluded_names"]) != excluded_names(config, cfg):
            raise ValueError("source exclusions changed while resuming")
    else:
        candidates, selection = select_candidates(raw, excluded_names(config, cfg), fraction, int(cfg["seed"]),
            excluded_image_dir=project_path(config, config["paths"]["raw_data_dir"]))
        if len(candidates) < count:
            raise ValueError(f"only {len(candidates)} eligible source candidates for requested {count}")
        write_json(directory / "selection.json", {**selection, "candidates": candidates})
    teacher_cfg = load_config(project_path(config, cfg["teacher_config"]))
    checkpoint = Path(project_path(config, cfg["teacher_checkpoint"]))
    model, payload = load_fused_deployment_model(checkpoint, teacher_cfg, args.device)
    if model.encoder.input_normalization != "legacy_none":
        raise ValueError("mainline teacher must retain legacy_none normalization")
    inference = teacher_cfg["inference"]
    if inference["boundary_threshold"] != .65 or inference["marker_border_seal_width"] != 2:
        raise ValueError("unexpected teacher deployment contract")
    dep = teacher_cfg["affinity_deployment"]
    fusion = {key: dep[key] for key in ("distance2_weight", "distance4_weight", "support_threshold", "support_temperature")}
    fusion.update(short_reduction=dep.get("short_reduction", "mean"),
                  short_softmax_temperature=dep.get("short_softmax_temperature", .15))
    manifest = dict(format=PSEUDO_FORMAT, label_source="mainline_pseudo", status="generating",
                    teacher_checkpoint=str(checkpoint), teacher_sha256=digest(checkpoint),
                    teacher_config=cfg["teacher_config"], teacher_normalization="legacy_none",
                    inference=inference, fusion_mode=dep["fusion_mode"], fusion=fusion,
                    selection=selection, generation_config=cfg, samples=[], sample_count=0, images=[], rejected=[])
    if previous is not None:
        if previous["teacher_sha256"] != manifest["teacher_sha256"] or previous["inference"] != inference or previous["fusion"] != fusion:
            raise ValueError("teacher checkpoint or deployment changed while resuming")
        manifest = previous
    write_json(directory / "manifest.json", manifest)
    started = time.time()
    processed = {row["stem"] for row in manifest["images"] + manifest["rejected"]}
    for candidate in candidates:
        if manifest["sample_count"] >= count:
            break
        stem = candidate["stem"]
        if stem in processed:
            continue
        if digest(raw / candidate["image_name"]) != candidate["source_sha256"]:
            raise ValueError(f"source image changed after selection: {stem}")
        rgb, tensor, ph, pw = prepare_image(raw / candidate["image_name"], int(cfg["input_size"]), args.device)
        output = model(tensor)
        semantics = crop_letterbox_output(output["semantic_logits"], int(cfg["input_size"]), ph, pw, rgb.shape[:2]).cpu()
        boundary = crop_affinity_boundary_output(output, int(cfg["input_size"]), ph, pw, rgb.shape[:2], dep["fusion_mode"], fusion).cpu()
        _, prediction, classes = postprocess(torch.cat([semantics, probability_to_logit(boundary)], 1),
            rgb.shape[:2], directory / "instances", stem, inference, .65, False, image_rgb=rgb)
        if prediction.dtype != np.uint16 or prediction.ndim != 2 or prediction.shape != rgb.shape[:2]:
            raise ValueError("teacher final output contract failed")
        row = {**candidate, "instances": len(classes), "coverage": float((prediction > 0).mean()),
               "ferrite_instances": sum(int(c) == 1 for c in classes.values())}
        reasons = []
        if not classes or len(classes) > int(cfg["query_capacity"]):
            reasons.append("query_capacity")
        if row["coverage"] < float(cfg["minimum_coverage"]):
            reasons.append("low_coverage")
        if reasons:
            row["reasons"] = reasons
            manifest["rejected"].append(row)
            for path in (directory / "instances").glob(f"{stem}_*"):
                path.replace(directory / "rejected" / path.name)
        else:
            manifest["images"].append(row)
            manifest["samples"].append(stem)
            manifest["sample_count"] += 1
            if manifest["sample_count"] <= 4:
                render_preview(raw, directory, row)
        write_json(directory / "manifest.json", manifest)
        print(f"accepted={manifest['sample_count']}/{count} attempted={len(manifest['images'])+len(manifest['rejected'])} "
              f"{stem} instances={len(classes)} coverage={row['coverage']:.4f} rejected={reasons}", flush=True)
        del output, semantics, boundary, tensor
        if manifest["sample_count"] == count:
            break
    if manifest["sample_count"] != count:
        raise RuntimeError(f"only {manifest['sample_count']} eligible predictions in predeclared clarity pool")
    # 按已接受样本的清晰度低/中/高和实例数最多补充预览，避免只展示最好看的图。
    ordered = sorted(manifest["images"], key=lambda row: row["clarity_score"])
    extra = [ordered[0], ordered[len(ordered)//2], ordered[-1], max(ordered, key=lambda row: row["instances"])]
    for row in extra:
        render_preview(raw, directory, row)
    manifest.update(status="complete", elapsed_seconds=time.time()-started,
                    preview_names=sorted(p.stem for p in (directory / "previews").glob("*.png")))
    write_json(directory / "manifest.json", manifest)
    print("COMPLETE", directory, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/mainline_pseudo_generation249.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
