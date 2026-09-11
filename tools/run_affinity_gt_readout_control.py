# -*- coding: utf-8 -*-
"""主线 GT-affinity 检验失败后，只改变短程融合方式的定位对照。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from tools.run_affinity_gt_deployment_oracle import (
    configure_mainline_precision, file_digest, load_fused_deployment_model,
    load_config, project_path, load_direct_evaluation_target,
    letterbox_instance_geometry, prepare_image, crop_letterbox_output,
    oracle_logits, affinity_boundary_probability, crop_affinity_boundary_output,
    finish, mask_prediction_to_evaluation_domain, evaluate_instance_pair,
    marker_diagnostics, values_summary, summarize_instance_results, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    parent = Path(args.primary_dir)
    source = json.loads((parent / "summary.json").read_text())
    protocol = source["protocol"]
    config = load_config(protocol["config"])
    infer = protocol["inference"]
    fusion = {**protocol["fusion_kwargs"], "short_reduction": "top2"}
    configure_mainline_precision()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(protocol["checkpoint"])
    assert file_digest(checkpoint) == protocol["checkpoint_sha256"]
    for relative, digest in protocol["source_sha256"].items():
        assert file_digest(ROOT / relative) == digest, relative
    model, _ = load_fused_deployment_model(checkpoint, config, device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    target_dir = project_path(config, protocol["manual_target_dir"])
    raw = Path(project_path(config, config["paths"]["raw_data_dir"]))
    rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.render_nativecrop_comparison import colored_partitions, match_reference
    figure, axes = plt.subplots(len(protocol["samples"]), 5, figsize=(17, 14),
                                layout="constrained", squeeze=False)
    for index, name in enumerate(protocol["samples"]):
        image_path = raw / f"{name}.jpg"
        image, tensor, pad_h, pad_w = prepare_image(image_path, 1024, device)
        gt, classes_gt, valid, _ = load_direct_evaluation_target(image_path, image.shape[:2], target_dir)
        original_row = next(r for r in source["arms"]["gt_affinity"]["images"] if r["image"] == name)
        assert file_digest(Path(target_dir) / f"{name}_gt.npz") == original_row["gt_sha256"]
        grid, content, _ = letterbox_instance_geometry(gt, 1024, 512)
        with torch.no_grad():
            outputs = model(tensor)
            logits, target, edge_valid = oracle_logits(outputs["affinity_logits"], grid, content)
        semantic = crop_letterbox_output(outputs["semantic_logits"], 1024, pad_h, pad_w, image.shape[:2]).cpu()
        coarse = affinity_boundary_probability(logits, mode="gated", **fusion)[0, 0].cpu().numpy()
        boundary = crop_affinity_boundary_output({"affinity_logits": logits}, 1024, pad_h, pad_w,
                                                 image.shape[:2], "gated", fusion).cpu()
        prediction, classes = finish(semantic, boundary, image, name, out, infer)
        scored = mask_prediction_to_evaluation_domain(prediction, valid)
        metrics = evaluate_instance_pair(gt, classes_gt, scored, classes)
        geometry = evaluate_instance_pair(gt, {k: 0 for k in classes_gt}, scored, {k: 0 for k in classes})
        markers, _ = marker_diagnostics(boundary[0, 0].numpy(), gt, infer)
        negative = (edge_valid[:4] & (target[:4] == 0)).any(0)
        interior = cv2.erode(((grid > 0) & content).astype(np.uint8), np.ones((17, 17), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        row = {"image": name, "metrics": metrics, "geometry": geometry, "markers": markers,
               "known_pixels": int(valid.sum()), "unassigned_known_pixels": int((prediction[valid] == 0).sum()),
               "fused_at_short_crossing_sources": values_summary(coarse[negative]),
               "fused_at_fully_known_short_crossing_sources": values_summary(coarse[negative & interior])}
        rows.append(row)
        write_json(out / f"{name}_diagnostics.json", row)
        size = (600, round(gt.shape[0] * 600 / gt.shape[1]))
        small_valid = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        gt_panel = colored_partitions(cv2.resize(gt.astype(np.uint16), size, interpolation=cv2.INTER_NEAREST))
        gt_panel[~small_valid] = .65
        panels = [cv2.resize(image, size), gt_panel]
        for candidate in [cv2.imread(str(parent / arm / f"{name}_inst.png"), -1)
                          for arm in ("mainline", "gt_affinity")] + [prediction]:
            masked = mask_prediction_to_evaluation_domain(candidate, valid)
            lut, _ = match_reference(masked, gt, valid)
            panel = colored_partitions(cv2.resize(masked, size, interpolation=cv2.INTER_NEAREST), lut)
            panel[~small_valid] = .65
            panels.append(panel)
        for ax, panel, title in zip(axes[index], panels, (name + " | image", "New GT", "Mainline", "GT affinity + mean", "GT affinity + top2")):
            ax.imshow(panel, interpolation="nearest"); ax.set_title(title); ax.axis("off")
        print(json.dumps({"image": name, "matches": metrics["valid_matches"], "geometry_matches": geometry["valid_matches"],
                          "gt_penalized_miou": metrics["gt_penalized_miou"], "markers": markers["marker_count"],
                          "gt_sharing_marker": markers["gt_sharing_dominant_marker"]}), flush=True)
    figure.suptitle("GT-affinity readout control | only mean -> top2 changed | normal E10a and identical watershed")
    figure.savefig(out / "readout_control.png", dpi=145, facecolor="white")
    plt.close(figure)
    aggregate = summarize_instance_results(r["metrics"] for r in rows)
    aggregate["known_unassigned_fraction"] = sum(r["unassigned_known_pixels"] for r in rows) / sum(r["known_pixels"] for r in rows)
    diagnostics = {k: sum(r["markers"][k] for r in rows) for k in ("marker_count", "gt_without_marker", "gt_sharing_dominant_marker")}
    for key in ("fused_at_short_crossing_sources", "fused_at_fully_known_short_crossing_sources"):
        diagnostics[key] = {k: sum(r[key][k] for r in rows) for k in ("count", "sum", "above_045", "above_065")}
    report = {"arm": "gt_affinity_top2", "primary": str(parent), "fusion_kwargs": fusion,
              "changed_fields": {"short_reduction": ["mean", "top2"]},
              "source_sha256": file_digest(Path(__file__)), "aggregate": aggregate,
              "geometry": summarize_instance_results(r["geometry"] for r in rows),
              "diagnostics": diagnostics, "images": rows,
              "metric_scope": "GT-assisted readout diagnosis; not evidence to deploy top2 on learned affinity"}
    write_json(out / "summary.json", report)
    write_json(out / "status.json", {"status": "complete", "images": protocol["samples"]})
    print(json.dumps({k: report[k] for k in ("arm", "aggregate", "diagnostics")}), flush=True)


if __name__ == "__main__":
    main()
