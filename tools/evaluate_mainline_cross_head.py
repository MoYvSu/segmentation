# -*- coding: utf-8 -*-
"""以主线固定后处理评估 cross-head checkpoint；省略 checkpoint 则评估原主线。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.direct_dual_head_dataset import (
    load_direct_evaluation_target, mask_prediction_to_evaluation_domain,
)
from data.mim_dataset import list_images
from models.cross_head_refinement import configure_mainline_precision, file_digest, load_refined_mainline
from models.fused_deployment import load_fused_deployment_model
from utils.affinity_deployment import (
    crop_affinity_boundary_output, crop_letterbox_output, postprocess,
    prepare_image, probability_to_logit,
)
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, summarize_instance_results


def deployment_contract(config):
    deployment = config["affinity_deployment"]
    return {
        "input_size": int(config["cross_head"]["input_size"]),
        "input_normalization": "legacy_none",
        "fusion_mode": str(deployment["fusion_mode"]),
        "fusion_kwargs": {key: deployment[key] for key in (
            "distance2_weight", "distance4_weight", "support_threshold",
            "support_temperature", "short_reduction", "short_softmax_temperature",
        )},
        "inference": copy.deepcopy(config["inference"]),
    }


@torch.no_grad()
def predict_instances(model, image_path, contract, output_dir, device, *, return_maps=False):
    model.eval()
    image_size = contract["input_size"]
    image, tensor, pad_h, pad_w = prepare_image(image_path, image_size, device)
    outputs = model(tensor)
    semantic = crop_letterbox_output(
        outputs["semantic_logits"], image_size, pad_h, pad_w, image.shape[:2],
    ).cpu()
    boundary = crop_affinity_boundary_output(
        outputs, image_size, pad_h, pad_w, image.shape[:2],
        contract["fusion_mode"], contract["fusion_kwargs"],
    ).cpu()
    infer_cfg = contract["inference"]
    _, instances, classes = postprocess(
        torch.cat([semantic, probability_to_logit(boundary)], 1),
        image.shape[:2], str(output_dir), Path(image_path).stem, infer_cfg,
        float(infer_cfg["boundary_threshold"]), False, image_rgb=image,
        semantic_challenger_logits=None,
    )
    disk = cv2.imread(str(Path(output_dir) / f"{Path(image_path).stem}_inst.png"), cv2.IMREAD_UNCHANGED)
    if disk is None or disk.dtype != np.uint16 or disk.shape != image.shape[:2]:
        raise RuntimeError("final PNG must be native-sized single-channel uint16")
    if int(disk.max()) > 65535 or not np.array_equal(disk, instances):
        raise RuntimeError("saved final instances differ from the returned prediction")
    maps = None
    if return_maps:
        maps = {key: outputs[key].cpu() for key in ("semantic_logits", "affinity_logits")}
        maps["boundary_native"] = boundary
    return instances, classes, maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train/mainline_cross_head_ab.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-dir", help="显式目录仅作无标签推理；默认评估记录的人工验证名单")
    args = parser.parse_args()
    configure_mainline_precision()
    torch.set_num_threads(4)
    config = load_config(args.config)
    cfg = config["cross_head"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_digest = None
    if args.checkpoint:
        checkpoint = project_path(config, args.checkpoint)
        model, payload = load_refined_mainline(checkpoint, config, device)
        contract = payload["deployment_contract"]
        split = payload["split"]
        target_dir = payload["manual_target_dir"]
        checkpoint_digest = file_digest(checkpoint)
        epoch = payload["epoch"]
        mode = payload["context_mode"]
        base_digest = payload["base_sha256"]
    else:
        base_path = project_path(config, cfg["base_checkpoint"])
        base_digest = file_digest(base_path)
        if base_digest != cfg["base_sha256"]:
            raise ValueError("configured mainline base digest mismatch")
        model, _ = load_fused_deployment_model(base_path, config, device)
        contract = deployment_contract(config)
        split = json.loads(Path(project_path(config, cfg["split_file"])).read_text(encoding="utf-8"))
        split["val"] = [n for n in split["val"] if n not in cfg["excluded_val_names"]]
        target_dir = cfg["manual_target_dir"]
        epoch, mode = None, "mainline"
    source_dir = project_path(config, args.image_dir or cfg["data_dir"])
    paths = [Path(p) for p in list_images(source_dir)]
    if not args.image_dir:
        paths = [p for p in paths if p.stem in split["val"]]
        if {p.stem for p in paths} != set(split["val"]):
            raise ValueError("evaluation image list does not match checkpoint split")
    if not paths:
        raise ValueError("no evaluation images")
    output_dir = Path(project_path(config, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        prediction, classes, _ = predict_instances(model, path, contract, output_dir, device)
        row = {"image": path.name, "instances": len(classes),
               "unassigned_fraction": float(np.mean(prediction == 0))}
        if not args.image_dir:
            gt, gt_classes, valid, _ = load_direct_evaluation_target(
                path, prediction.shape, project_path(config, target_dir),
            )
            row["metrics"] = evaluate_instance_pair(
                gt, gt_classes, mask_prediction_to_evaluation_domain(prediction, valid), classes,
            )
            row["known_unassigned_fraction"] = float(np.mean(prediction[valid] == 0))
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "metrics"}), flush=True)
    summary = {
        "format": "mainline_cross_head_evaluation_v1", "mode": mode, "epoch": epoch,
        "checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_digest,
        "base_sha256": base_digest, "deployment_contract": contract, "split": split,
        "manual_target_dir": target_dir, "parameter_summary": model.parameter_summary(),
        "metric_scope": "unlabeled inference" if args.image_dir else "manual GT proxy; not official score",
        "images": rows,
    }
    if not args.image_dir:
        summary["aggregate"] = summarize_instance_results(r["metrics"] for r in rows)
        summary["mean_known_unassigned_fraction"] = float(np.mean([r["known_unassigned_fraction"] for r in rows]))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
