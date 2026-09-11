# -*- coding: utf-8 -*-
"""由 GT 实例直接导出 affinity，固定 E10a 和完整部署链进行诊断。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.direct_dual_head_dataset import (
    load_direct_evaluation_target, mask_prediction_to_evaluation_domain,
)
from models.cross_head_refinement import configure_mainline_precision, file_digest
from models.fused_deployment import load_fused_deployment_model
from utils.affinity_deployment import (
    crop_affinity_boundary_output, crop_letterbox_output, postprocess,
    prepare_image, probability_to_logit,
)
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, build_affinity_targets
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, summarize_instance_results
from utils.offset_letterbox import letterbox_instance_geometry
from utils.post_process import _boundary_skeleton_belt, reconstruct_marker_boundary


ARMS = ("mainline", "gt_affinity")
SAMPLES = "train_172,train_351,train_686,train_872,train_873"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def oracle_logits(baseline, labels, content):
    """只替换两端均有 GT 的连接；用无穷 logits 精确表达概率 0/1。"""
    truth, valid = build_affinity_targets(labels, content)
    target = torch.from_numpy(truth).unsqueeze(0).to(baseline.device)
    known = torch.from_numpy(valid).unsqueeze(0).to(baseline.device)
    if baseline.shape != target.shape:
        raise ValueError(f"oracle grid mismatch: {baseline.shape} vs {target.shape}")
    # 与实际训练的 torch 目标生成器交叉核对方向、源端坐标及 ignore 范围。
    train_target, train_valid = build_affinity_targets_torch(
        torch.from_numpy(labels.astype(np.int64))[None],
        torch.from_numpy(content)[None, None],
    )
    assert np.array_equal(train_target[0].numpy(), truth)
    assert np.array_equal(train_valid[0].numpy(), valid)
    exact = torch.where(target > 0.5, float("inf"), float("-inf"))
    result = torch.where(known, exact, baseline)
    assert torch.equal(result[~known], baseline[~known])
    assert torch.equal(result.sigmoid()[known], target[known])
    return result, truth, valid


def values_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "sum": float(values.sum()),
            "mean": float(values.mean()) if values.size else None,
            "above_045": int((values > .45).sum()),
            "above_065": int((values > .65).sum()),
            "quantiles": np.quantile(values, [0, .1, .5, .9, 1]).tolist()
            if values.size else None}


def marker_diagnostics(boundary, gt, infer):
    """重放当前种子生成步骤，仅观察，不把 GT 注入实际分水岭。"""
    reconstructed = reconstruct_marker_boundary(
        boundary, infer["marker_boundary_low_threshold"],
        infer["boundary_threshold"], infer["marker_boundary_reconstruction_steps"],
    )
    _, belt = _boundary_skeleton_belt(
        reconstructed, infer["bridge_width"], infer["watershed_dilate_width"],
    )
    width = int(infer["marker_border_seal_width"])
    if width:
        belt[:width, :] = belt[-width:, :] = 255
        belt[:, :width] = belt[:, -width:] = 255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(belt), connectivity=8,
    )
    keep = stats[:, cv2.CC_STAT_AREA] >= infer["min_instance_area"]
    keep[0] = False
    labels[~keep[labels]] = 0
    count = int(keep.sum())
    stride = int(labels.max()) + 1
    joint = np.bincount((gt.astype(np.int64) * stride + labels).ravel(),
                        minlength=(int(gt.max()) + 1) * stride).reshape(-1, stride)
    joint[:, 0] = 0
    ids = np.unique(gt[gt > 0])
    has_marker = joint[ids].sum(1) > 0
    dominant = joint[ids].argmax(1)
    frequency = np.bincount(dominant[has_marker], minlength=stride)
    shared = has_marker & (frequency[dominant] > 1)
    return {"marker_count": count, "gt_without_marker": int((~has_marker).sum()),
            "gt_sharing_dominant_marker": int(shared.sum()),
            "shared_groups": [ids[has_marker & (dominant == k)].tolist()
                              for k in np.flatnonzero(frequency > 1)],
            "strong_boundary_pixels": int((boundary > infer["boundary_threshold"]).sum()),
            "reconstructed_boundary_pixels": int(reconstructed.sum())}, belt


def finish(semantic, boundary, image, name, directory, infer):
    _, prediction, classes = postprocess(
        torch.cat([semantic, probability_to_logit(boundary)], 1),
        image.shape[:2], str(directory), name, infer,
        float(infer["boundary_threshold"]), False, image_rgb=image,
        semantic_challenger_logits=None,
    )
    disk = cv2.imread(str(directory / f"{name}_inst.png"), cv2.IMREAD_UNCHANGED)
    assert disk is not None and disk.dtype == np.uint16
    assert disk.shape == image.shape[:2] and np.array_equal(disk, prediction)
    assert int(disk.max()) <= 65535
    return prediction, classes


def plot_readout(path, name, image, gt_grid, content, probabilities, boundaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = content
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5), layout="constrained")
    for row, arm in enumerate(ARMS):
        for channel in range(4):
            axes[row, channel].imshow(1 - probabilities[arm][channel, :h, :w],
                                      cmap="magma", vmin=0, vmax=1)
            axes[row, channel].set_title(f"{arm}: cut {DEFAULT_AFFINITY_OFFSETS[channel]}")
        shown = axes[row, 4].imshow(boundaries[arm][:h, :w], cmap="magma", vmin=0, vmax=1)
        axes[row, 4].set_title("Gated + mean boundary")
        axes[row, 5].imshow(boundaries[arm][:h, :w] > .65, cmap="gray", vmin=0, vmax=1)
        axes[row, 5].set_title("Boundary > 0.65")
    for ax in axes.ravel():
        ax.axis("off")
    fig.colorbar(shown, ax=axes[:, :5].ravel().tolist(), shrink=.55)
    fig.suptitle(f"{name} | 512-grid direction-to-boundary readout | shared 0-1 scale")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml")
    parser.add_argument("--checkpoint", default="outputs/deployment/e10a_g4b_fused.pth")
    parser.add_argument("--manual-target-dir", default="outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps")
    parser.add_argument("--sample-names", default=SAMPLES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", help="可选：核对本次基线与已保存的主线输出逐字节一致")
    args = parser.parse_args()
    configure_mainline_precision()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = load_config(args.config)
    infer = config["inference"]
    deployment = config["affinity_deployment"]
    fusion = {key: deployment[key] for key in (
        "distance2_weight", "distance4_weight", "support_threshold",
        "support_temperature", "short_reduction", "short_softmax_temperature",
    )}
    assert deployment["fusion_mode"] == "gated" and fusion["short_reduction"] == "mean"
    assert infer["boundary_threshold"] == .65
    checkpoint = Path(project_path(config, args.checkpoint))
    target_dir = Path(project_path(config, args.manual_target_dir))
    source_dir = Path(project_path(config, config["paths"]["raw_data_dir"]))
    out = Path(project_path(config, args.output_dir))
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bundle = load_fused_deployment_model(checkpoint, config, device)
    assert model.geometry_highres_refiner is None
    assert not any(p.requires_grad for p in model.parameters())
    reference = Path(project_path(config, args.reference_dir)) if args.reference_dir else None
    names = args.sample_names.split(",")
    for arm in ARMS:
        (out / arm).mkdir()
    results = {arm: [] for arm in ARMS}
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(),
                "format": "gt_affinity_deployment_oracle_v1",
                "config": args.config, "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_digest(checkpoint), "sources": bundle["sources"],
                "manual_target_dir": args.manual_target_dir, "samples": names,
                "input_size": 1024, "affinity_grid": 512, "input_normalization": "legacy_none",
                "precision": "FP32; matmul TF32 false; cuDNN TF32 true; no autocast",
                "fusion_mode": "gated", "fusion_kwargs": fusion, "inference": infer,
                "intervention": "GT same-instance=1, different-instance=0 on known endpoint pairs; unknown/padding links retain original G4b logits. Identical normal E10a logits in both arms.",
                "metric_scope": "GT-assisted diagnostic; not deployment performance or official score",
                "runtime_versions": {"torch": torch.__version__, "opencv": cv2.__version__},
                "source_sha256": {p: file_digest(ROOT / p) for p in (
                    "tools/run_affinity_gt_deployment_oracle.py", "utils/affinity_fusion.py",
                    "utils/affinity_deployment.py", "utils/post_process.py", "models/fused_deployment.py",
                    "utils/offset_letterbox.py", "utils/affinity_graph.py", "utils/instance_metrics.py")}}
    write_json(out / "protocol.json", protocol)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.render_nativecrop_comparison import colored_partitions, match_reference
    figure, axes = plt.subplots(len(names), 4, figsize=(14, 3 * len(names)),
                                layout="constrained", squeeze=False)
    for row_index, name in enumerate(names):
        image_path = source_dir / f"{name}.jpg"
        image, tensor, pad_h, pad_w = prepare_image(image_path, 1024, device)
        gt, gt_classes, valid_native, provenance = load_direct_evaluation_target(
            image_path, image.shape[:2], target_dir,
        )
        grid, content, meta = letterbox_instance_geometry(gt, 1024, 512)
        assert meta.resized_height == 1024 - pad_h and meta.resized_width == 1024 - pad_w
        with torch.no_grad():
            original = model(tensor)
            perfect, truth, edge_valid = oracle_logits(original["affinity_logits"], grid, content)
            semantic = crop_letterbox_output(original["semantic_logits"], 1024,
                                             pad_h, pad_w, image.shape[:2]).cpu()
        logits = {"mainline": original["affinity_logits"], "gt_affinity": perfect}
        negative_sources = (edge_valid[:4] & (truth[:4] == 0)).any(0)
        # 距 unknown/padding 至少 8 像素，排除长连接及门控邻域的未知值干扰。
        interior_known = cv2.erode(((grid > 0) & content).astype(np.uint8),
                                  np.ones((17, 17), np.uint8),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        inner_sources = negative_sources & interior_known
        shared = {"image": name, "known_pixels": int(valid_native.sum()),
                  "provenance": provenance, "gt_sha256": file_digest(target_dir / f"{name}_gt.npz"),
                  "gt_grid_instance_count": int(np.unique(grid[grid > 0]).size),
                  "gt_ids_lost_on_grid": sorted(set(map(int, gt_classes)) - set(map(int, np.unique(grid[grid > 0])))),
                  "edge_known": int(edge_valid.sum()), "edge_total": int(edge_valid.size),
                  "gt_positive_edges": int((edge_valid & (truth == 1)).sum()),
                  "gt_negative_edges": int((edge_valid & (truth == 0)).sum()),
                  "oracle_exact_on_known": True, "oracle_unchanged_on_unknown": True,
                  "target_matches_training_generator": True}
        size = (640, round(gt.shape[0] * 640 / gt.shape[1]))
        small_valid = cv2.resize(valid_native.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        small_gt = cv2.resize(gt.astype(np.uint16), size, interpolation=cv2.INTER_NEAREST)
        gt_panel = colored_partitions(small_gt)
        gt_panel[~small_valid] = .65
        panels = [cv2.resize(image, size), gt_panel]
        probabilities, boundaries = {}, {}
        for arm in ARMS:
            boundary_grid = affinity_boundary_probability(logits[arm], mode="gated", **fusion)
            boundaries[arm] = boundary_grid[0, 0].cpu().numpy()
            probabilities[arm] = logits[arm][0].sigmoid().cpu().numpy()
            boundary = crop_affinity_boundary_output(
                {"affinity_logits": logits[arm]}, 1024, pad_h, pad_w,
                image.shape[:2], "gated", fusion,
            ).cpu()
            prediction, classes = finish(semantic, boundary, image, name, out / arm, infer)
            if arm == "mainline" and reference:
                for suffix in ("_inst.png", "_class.json"):
                    assert (out / arm / (name + suffix)).read_bytes() == (reference / (name + suffix)).read_bytes(), f"baseline mismatch: {name}{suffix}"
            scored = mask_prediction_to_evaluation_domain(prediction, valid_native)
            metrics = evaluate_instance_pair(gt, gt_classes, scored, classes)
            # 类别无关指标仅诊断几何，GT 类别不送入实际推理。
            geometry = evaluate_instance_pair(gt, {k: 0 for k in gt_classes}, scored, {k: 0 for k in classes})
            markers, _ = marker_diagnostics(boundary[0, 0].numpy(), gt, infer)
            item = {**shared, "metrics": metrics, "geometry": geometry, "markers": markers,
                    "unassigned_known_pixels": int((prediction[valid_native] == 0).sum()),
                    "fused_at_short_crossing_sources": values_summary(boundaries[arm][negative_sources]),
                    "fused_at_fully_known_short_crossing_sources": values_summary(boundaries[arm][inner_sources]),
                    "baseline_matches_reference": True if arm == "mainline" and reference else None}
            results[arm].append(item)
            write_json(out / arm / f"{name}_diagnostics.json", item)
            lut, _ = match_reference(scored, gt, valid_native)
            panel = colored_partitions(cv2.resize(scored, size, interpolation=cv2.INTER_NEAREST), lut)
            panel[~small_valid] = .65
            panels.append(panel)
            print(json.dumps({"image": name, "arm": arm, "matches": metrics["valid_matches"],
                              "gt_penalized_miou": metrics["gt_penalized_miou"], "markers": markers["marker_count"],
                              "gt_sharing_marker": markers["gt_sharing_dominant_marker"]}), flush=True)
        for ax, panel, title in zip(axes[row_index], panels, (name + " | image", "New GT", "E10a + G4b", "E10a + GT affinity")):
            ax.imshow(panel, interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
        if name in (names[0], names[-1]):
            plot_readout(out / f"{name}_readout.png", name, image, grid,
                         (meta.content_height, meta.content_width), probabilities, boundaries)
    figure.suptitle("Identical E10a + complete deployed postprocessing | colors aligned to GT\nBlack: partition borders. Dark gray: unassigned. Light gray: GT unknown.")
    figure.savefig(out / "comparison.png", dpi=145, facecolor="white")
    plt.close(figure)
    summary = {"protocol": protocol, "arms": {}}
    for arm, rows in results.items():
        aggregate = summarize_instance_results(r["metrics"] for r in rows)
        aggregate["known_unassigned_fraction"] = sum(r["unassigned_known_pixels"] for r in rows) / sum(r["known_pixels"] for r in rows)
        geometry = summarize_instance_results(r["geometry"] for r in rows)
        diagnostics = {key: sum(r["markers"][key] for r in rows) for key in (
            "marker_count", "gt_without_marker", "gt_sharing_dominant_marker", "strong_boundary_pixels", "reconstructed_boundary_pixels")}
        for key in ("fused_at_short_crossing_sources", "fused_at_fully_known_short_crossing_sources"):
            tally = {k: sum(r[key][k] for r in rows) for k in ("count", "sum", "above_045", "above_065")}
            tally["mean"] = tally["sum"] / tally["count"] if tally["count"] else None
            diagnostics[key] = tally
        summary["arms"][arm] = {"aggregate": aggregate, "geometry": geometry, "diagnostics": diagnostics,
                                "images": rows}
        print(json.dumps({"arm": arm, "aggregate": aggregate, "diagnostics": diagnostics}), flush=True)
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", {"status": "complete", "images": names,
                                     "oracle_is_exact_on_valid_edges": True,
                                     "native_uint16_checked": True,
                                     "baseline_reference_checked": bool(reference)})


if __name__ == "__main__":
    main()
