# -*- coding: utf-8 -*-
"""固定 E10a 语义，诊断 G4b/e110 方向连接、读出和最终实例的差异。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.direct_dual_head_dataset import load_manual_target_archive
from models.direct_semantic_affinity import load_direct_semantic_affinity_model
from models.fused_deployment import load_fused_deployment_model
from utils.affinity_deployment import (
    crop_letterbox_output, postprocess, prepare_image, probability_to_logit,
)
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_loss import _edge_slices, build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, summarize_instance_results
from utils.offset_letterbox import letterbox_instance_geometry
from utils.post_process import _boundary_skeleton_belt, reconstruct_marker_boundary


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compact(metrics):
    return {k: ({c: {a: b for a, b in v.items() if a != "matches"}
                 for c, v in value.items()} if k == "classes" else value)
            for k, value in metrics.items()}


def decomposed_fusion(logits, reduction, kwargs):
    q = 1 - logits.sigmoid()
    short = (q[:, :4].mean(1, keepdim=True) if reduction == "mean"
             else q[:, :4].topk(2, dim=1).values.mean(1, keepdim=True))
    d2, d4 = q[:, 4:6].mean(1, keepdim=True), q[:, 6:8].mean(1, keepdim=True)
    gates = [(F.max_pool2d(short, 2*r+1, stride=1, padding=r)
              - kwargs["support_threshold"]).div(kwargs["support_temperature"]).sigmoid()
             for r in (2, 4)]
    w2, w4 = kwargs["distance2_weight"] * gates[0], kwargs["distance4_weight"] * gates[1]
    boundary = (short + w2*d2 + w4*d4) / (1+w2+w4)
    official = affinity_boundary_probability(logits, mode="gated", short_reduction=reduction, **kwargs)
    torch.testing.assert_close(boundary, official, atol=1e-7, rtol=1e-6)
    return dict(short=short, d2=d2, d4=d4, support2=gates[0], support4=gates[1], boundary=official)


def diagnostic_markers(probability, cfg):
    """复用正式掩码函数并复现种子生成；CPU 测试核对正式 watershed 的输入。"""
    high = probability > float(cfg["boundary_threshold"])
    mask = reconstruct_marker_boundary(
        probability, float(cfg["marker_boundary_low_threshold"]),
        float(cfg["boundary_threshold"]), int(cfg["marker_boundary_reconstruction_steps"]),
    )
    _, belt = _boundary_skeleton_belt(mask, int(cfg["bridge_width"]), int(cfg["watershed_dilate_width"]))
    width = min(max(0, int(cfg["marker_border_seal_width"])), max(1, min(belt.shape)//2))
    if width:
        belt[:width] = belt[-width:] = 255
        belt[:, :width] = belt[:, -width:] = 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(belt), connectivity=8)
    lut = np.zeros(n, dtype=np.int32)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= int(cfg["min_instance_area"])) + 1
    lut[keep] = np.arange(1, len(keep)+1)
    markers = lut[labels]
    if int(markers.max()) > 65535:
        raise ValueError("诊断缓存的种子数超出 uint16，不能静默截断")
    return dict(high=high, marker_mask=mask > 0, belt=belt > 0, markers=markers.astype(np.uint16))


def spatial_domains(gt, valid):
    """真实界面不包含 unknown；深部排除界面、unknown 和图像边缘的 4px 邻域。"""
    interface = np.zeros(gt.shape, bool)
    all_edges = np.zeros(gt.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        source, dest = _edge_slices(*gt.shape, dy, dx)
        different = gt[source] != gt[dest]
        actual = different & valid[source] & valid[dest] & (gt[source] > 0) & (gt[dest] > 0)
        interface[source] |= actual
        interface[dest] |= actual
        all_edges[source] |= different
        all_edges[dest] |= different
    allowed = valid & (gt > 0) & ~all_edges
    distance = distance_transform_edt(np.pad(allowed, 1))[1:-1, 1:-1]
    return interface, allowed & (distance > 4)


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return dict(n=0, mean=None, p50=None, p90=None, p99=None, gt05=None, gt08=None)
    return dict(n=int(values.size), mean=float(values.mean()),
                **{k: float(v) for k, v in zip(("p50", "p90", "p99"), np.quantile(values, [.5, .9, .99]))},
                gt05=float((values > .5).mean()), gt08=float((values > .8).mean()))


def raw_statistics(prob, target, valid, original, filled):
    result = []
    for c, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        inside, cross = valid[c] & target[c], valid[c] & ~target[c]
        item = dict(offset=[int(dy), int(dx)], inside=distribution(prob[c][inside]), cross=distribution(prob[c][cross]))
        source, dest = _edge_slices(*original.shape, dy, dx)
        groups = {
            "original_original": original[source] & original[dest],
            "original_filled": (original[source] & filled[dest]) | (filled[source] & original[dest]),
            "filled_filled": filled[source] & filled[dest],
        }
        item["provenance"] = {}
        for name, group in groups.items():
            for label, mask in (("inside", inside), ("cross", cross)):
                selected = group & mask[source]
                item["provenance"][f"{name}_{label}"] = distribution(prob[c][source][selected])
        result.append(item)
    return result


def boundary_statistics(boundary, short, target, edge_valid, gt, valid):
    cross_sources = (edge_valid[:4] & ~target[:4]).any(0)
    interface, interior = spatial_domains(gt, valid)
    high = (boundary > .65) & valid
    near_interface = distance_transform_edt(~interface) <= 1 if interface.any() else np.zeros_like(valid)
    near_high = distance_transform_edt(~high) <= 1 if high.any() else np.zeros_like(valid)
    def rate(mask, values):
        return float(values[mask].mean()) if mask.any() else None
    return dict(
        cross_source_count=int(cross_sources.sum()),
        cross_source_high_rate=rate(cross_sources, high),
        interface_coverage_tol1=rate(interface, near_high),
        high_near_interface_tol1=rate(high, near_interface),
        deep_interior_count=int(interior.sum()), deep_interior_high_rate=rate(interior, high),
        high_fraction=rate(valid, high),
        cross_source_long_delta=rate(cross_sources, boundary-short),
        cross_source_long_dilution_rate=rate(cross_sources, boundary < short-1e-6),
        short_high_to_fused_low_rate=rate(cross_sources, (short > .65) & ~high),
    )


def topology(gt, prediction, valid):
    """不分语义类别的 10% 交叠诊断；零预测保留在 GT 面积分母中。"""
    g, p = gt[valid].astype(np.int64), prediction[valid].astype(np.int64)
    ng, npred = int(g.max(initial=0))+1, int(p.max(initial=0))+1
    counts = np.bincount(g*npred+p, minlength=ng*npred).reshape(ng, npred)
    ga, pa = counts.sum(1), counts.sum(0)
    intersection = counts[1:, 1:]
    gf = intersection / np.maximum(ga[1:, None], 1)
    pf = intersection / np.maximum(pa[None, 1:], 1)
    gt_present, pred_present = ga[1:] > 0, pa[1:] > 0
    significant = (gf >= .1).sum(1)
    dominant = intersection.argmax(1)+1 if intersection.shape[1] else np.zeros(ng-1, int)
    if intersection.shape[1]:
        dominant[intersection.max(1) == 0] = 0
    dominant_counts = np.bincount(dominant[gt_present], minlength=npred)[1:]
    return dict(gt_count=int(gt_present.sum()), pred_count=int(pred_present.sum()),
                split_gt_10pct=int(((significant >= 2) & gt_present).sum()),
                merged_pred_10pct=int((((pf >= .1).sum(0) >= 2) & pred_present).sum()),
                gt_without_significant_region=int(((significant == 0) & gt_present).sum()),
                regions_dominant_for_multiple_gt=int((dominant_counts >= 2).sum()),
                gt_in_shared_dominant_region=int(dominant_counts[dominant_counts >= 2].sum()),
                non_dominant_gt_fraction=float((intersection.sum()-intersection.max(0).sum()) / max(1, intersection.sum()))
                if intersection.size else 0.0)


def pick_rois(cache, content_shape, has_gt):
    h, w = content_shape
    box = min(96, h, w)
    main, direct = cache["g4b_mean_boundary"].astype(float), cache["e110_mean_boundary"].astype(float)
    if has_gt:
        interface, interior = spatial_domains(cache["gt_grid"], cache["valid_grid"])
        scores = [("missed_boundary", interface & (main > .65) & (direct <= .65)),
                  ("interior_response", interior & (direct > .65) & (main <= .65)),
                  ("agreement", interface & (main > .65) & (direct > .65))]
    else:
        scores = [("unlabeled_disagreement", np.abs(main-direct))]
    rois = []
    for kind, scores_grid in scores:
        score = cv2.boxFilter(scores_grid.astype(np.float32), -1, (box, box), normalize=False)
        half = box//2
        inner = score[half:h-half+1, half:w-half+1]
        if inner.size and inner.max() > 0:
            y, x = np.unravel_index(inner.argmax(), inner.shape)
            x, y = int(x), int(y)
        else:
            x, y = max(0, (w-box)//2), max(0, (h-box)//2)
        rois.append(dict(kind=kind, x0=x, y0=y, x1=x+box, y1=y+box,
                         selection="maximum 96-grid-window diagnostic evidence; not representative accuracy"))
    return rois


def find_image(directory, stem):
    for ext in (".jpg", ".png", ".jpeg"):
        path = directory / (stem+ext)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Image {stem} is absent from {directory}")


@torch.no_grad()
def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=args.resume)
    (out / "cache").mkdir(exist_ok=args.resume)
    main_cfg, direct_cfg = load_config(args.mainline_config), load_config(args.direct_config)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    models = {}
    models["g4b"], main_payload = load_fused_deployment_model(args.mainline_checkpoint, main_cfg, device)
    models["e110"], direct_payload = load_direct_semantic_affinity_model(args.direct_checkpoint, direct_cfg, device)
    if (int(direct_payload["epoch"]), direct_payload["phase"]) != (110, "joint_lora"):
        raise ValueError("This diagnosis expects clean60 e110 / joint_lora")
    for model in models.values():
        model.eval()
    norms = {k: m.encoder.input_normalization for k, m in models.items()}
    if norms != {"g4b": "legacy_none", "e110": "imagenet_v1"}:
        raise ValueError(f"Unexpected actual model normalization: {norms}")
    sizes = dict(g4b=int(main_cfg["affinity_geometry_g1"]["input_size"]),
                 e110=int(direct_payload["config"]["direct_semantic_affinity"]["input_size"]))
    if sizes != dict(g4b=1024, e110=1024):
        raise ValueError(f"This matched-grid diagnosis expects 1024 inputs: {sizes}")
    cfg = dict(main_cfg["inference"])
    if float(cfg["boundary_threshold"]) != .65:
        raise ValueError("Expected deployed high=.65")
    dep = main_cfg["affinity_deployment"]
    kwargs = {k: float(dep[k]) for k in ("distance2_weight", "distance4_weight", "support_threshold", "support_temperature")}
    manual_dir = Path(project_path(direct_cfg, direct_cfg["direct_semantic_affinity"]["manual_target"]["dataset_dir"]))
    raw_dir = Path(project_path(direct_cfg, direct_cfg["paths"]["raw_data_dir"]))
    test_dir = Path(project_path(direct_cfg, args.test_dir))
    names = [(s, True) for s in direct_payload["split"]["val"]]
    names += [(s, False) for s in args.test_images.split(",") if s]
    arms = [f"{model}_{reduction}" for model in models for reduction in ("mean", "top2")]
    paths = {stem: find_image(raw_dir if has_gt else test_dir, stem) for stem, has_gt in names}
    source_files = ["tools/diagnose_affinity_readout.py", "utils/affinity_fusion.py", "utils/affinity_loss.py",
                    "utils/affinity_deployment.py", "utils/post_process.py", "utils/instance_metrics.py",
                    "models/direct_semantic_affinity.py", "models/sam2_encoder.py", "models/fused_deployment.py"]
    summary = dict(format="affinity_readout_diagnosis_v1", inference=cfg, fusion=kwargs,
                   actual_normalization=norms, input_sizes=sizes, direct_epoch=110,
                   direct_phase=direct_payload["phase"], loss_selection=direct_payload["loss_selection"],
                   loss_scales=direct_payload["loss_scales"], split=direct_payload["split"],
                   checkpoint_sha256={"g4b": digest(args.mainline_checkpoint), "e110": digest(args.direct_checkpoint)},
                   checkpoints={"g4b": args.mainline_checkpoint, "e110": args.direct_checkpoint},
                   source_sha256={p: digest(ROOT/p) for p in source_files}, images=[], summaries={})
    if args.resume:
        previous = json.loads((out/"summary.json").read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "actual_normalization", "input_sizes", "inference", "fusion", "split"):
            if previous[key] != summary[key]:
                raise ValueError(f"Resume would mix different experiment contracts: {key}")
        previous.setdefault("resume_source_sha256", []).append(summary["source_sha256"])
        summary = previous
    completed = {im["stem"] for im in summary["images"]}
    for stem, has_gt in names:
        if stem in completed:
            if not (out/"cache"/f"{stem}.npz").is_file():
                raise FileNotFoundError(f"Completed image has no cache: {stem}")
            print("Reuse completed image", stem, flush=True)
            continue
        path = paths[stem]
        rgb, tensor, ph, pw = prepare_image(path, 1024, device)
        if has_gt:
            gt, classes, _, valid, provenance = load_manual_target_archive(manual_dir/f"{stem}_gt.npz")
        else:
            gt = np.zeros(rgb.shape[:2], np.int32)
            valid = np.zeros_like(gt, bool)
            classes, provenance = {}, {}
        gt_grid, content_valid, meta = letterbox_instance_geometry(gt, 1024, 512)
        valid_grid = content_valid & (gt_grid > 0)
        target, edge_valid = build_affinity_targets_torch(torch.from_numpy(gt_grid)[None], torch.from_numpy(valid_grid)[None])
        target, edge_valid = target[0].numpy().astype(bool), edge_valid[0].numpy()
        rgb_grid = np.zeros((512, 512, 3), np.uint8)
        rgb_grid[:meta.content_height, :meta.content_width] = cv2.resize(rgb, (meta.content_width, meta.content_height))
        cache = dict(rgb=rgb, gt=gt.astype(np.uint16), valid=valid, rgb_grid=rgb_grid,
                     gt_grid=gt_grid.astype(np.uint16), valid_grid=valid_grid, edge_target=target, edge_valid=edge_valid)
        original = letterbox_instance_geometry(provenance["original_covered"].astype(np.int32), 1024, 512)[0] > 0 if has_gt else valid_grid
        filled = letterbox_instance_geometry(provenance["filled"].astype(np.int32), 1024, 512)[0] > 0 if has_gt else valid_grid
        cache.update(original_grid=original, filled_grid=filled)
        entry = dict(stem=stem, has_gt=has_gt, image_path=str(path), native_shape=list(gt.shape),
                     grid_content_shape=[meta.content_height, meta.content_width], raw={}, arms={})
        outputs = {name: model(tensor) for name, model in models.items()}
        semantic = crop_letterbox_output(outputs["g4b"]["semantic_logits"], 1024, ph, pw, gt.shape).cpu()
        for name, output in outputs.items():
            logits = output["affinity_logits"]
            if logits.shape != (1, 8, 512, 512) or not torch.isfinite(logits).all():
                raise ValueError(f"Unexpected affinity output: {name} {logits.shape}")
            if "coarse_affinity_logits" in output:
                raise ValueError("High-resolution residual outputs require a different diagnostic contract")
            prob = logits.sigmoid()[0].cpu().numpy()
            cache[f"{name}_prob"] = prob
            if has_gt:
                entry["raw"][name] = raw_statistics(prob, target, edge_valid, original, filled)
            for reduction in ("mean", "top2"):
                arm = f"{name}_{reduction}"
                pieces = decomposed_fusion(logits, reduction, kwargs)
                arrays = {k: v[0, 0].cpu().numpy() for k, v in pieces.items()}
                cache.update({f"{arm}_{k}": v for k, v in arrays.items()})
                native = crop_letterbox_output(pieces["boundary"], 1024, ph, pw, gt.shape).cpu()
                native_logits = probability_to_logit(native)
                # 与正式 postprocess 内 sigmoid 的结果严格同源，避免阈值附近数值偏差。
                actual_prob = native_logits.sigmoid()[0, 0].numpy()
                masks = diagnostic_markers(actual_prob, cfg)
                dest = out/"instances"/arm
                dest.mkdir(parents=True, exist_ok=True)
                _, pred, pred_classes = postprocess(torch.cat([semantic, native_logits], 1), gt.shape,
                                                     dest, stem, cfg, .65, False, image_rgb=rgb)
                if pred.dtype != np.uint16 or int(pred.max()) > 65535:
                    raise ValueError("Final instance output violates uint16 contract")
                cache.update({f"{arm}_{k}": v for k, v in masks.items()})
                cache[f"{arm}_instances"] = pred
                record = dict(instances=len(pred_classes), seeds=int(masks["markers"].max()),
                              native_high_fraction=float(masks["high"].mean()))
                if has_gt:
                    masked = pred.copy()
                    masked[~valid] = 0
                    metrics = evaluate_instance_pair(gt, classes, masked, pred_classes)
                    record.update(metrics=compact(metrics), topology=topology(gt, pred, valid),
                                  seed_topology=topology(gt, masks["markers"], valid),
                                  boundary=boundary_statistics(arrays["boundary"], arrays["short"], target, edge_valid, gt_grid, valid_grid))
                entry["arms"][arm] = record
                print(f"{stem} {arm}: instances={len(pred_classes)} matches={record.get('metrics',{}).get('valid_matches')} seeds={record['seeds']}", flush=True)
        entry["rois"] = pick_rois(cache, entry["grid_content_shape"], has_gt)
        np.savez_compressed(out/"cache"/f"{stem}.npz", **cache)
        summary["images"].append(entry)
        (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        del outputs, cache, tensor
    metrics_all = {arm: [im["arms"][arm]["metrics"] for im in summary["images"] if im["has_gt"]] for arm in arms}
    summary["summaries"] = {arm: summarize_instance_results(rows) for arm, rows in metrics_all.items()}
    # 共同验证仅由当前 G4b split 代码推算，不能冒充历史权重的真实未见集合。
    summary["provisional_common_val_summaries"] = {
        arm: summarize_instance_results([im["arms"][arm]["metrics"] for im in summary["images"]
                                         if im["stem"] in ("train_172", "train_873")])
        for arm in arms
    }
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-config", default="config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml")
    parser.add_argument("--direct-config", default="config/train/direct_gtv2_clean_60x60.yaml")
    parser.add_argument("--mainline-checkpoint", required=True)
    parser.add_argument("--direct-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-images", default="test_009,test_026,test_045")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--resume", action="store_true", help="复用相同合同下已完成的图像")
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
