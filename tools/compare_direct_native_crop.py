# -*- coding: utf-8 -*-
"""在固定主线推理下比较 clean60/e110、局部采样候选与 G4b。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.direct_dual_head_dataset import load_manual_target_archive
from models.direct_semantic_affinity import load_direct_semantic_affinity_model
from models.fused_deployment import load_fused_deployment_model
from tools.diagnose_affinity_readout import (
    boundary_statistics, compact, decomposed_fusion, diagnostic_markers,
    digest, raw_statistics, topology,
)
from utils.affinity_deployment import crop_letterbox_output, postprocess, prepare_image, probability_to_logit
from utils.config import load_config, project_path
from utils.instance_metrics import evaluate_instance_pair, summarize_instance_results


ARMS = dict(g4b=("g4b", "g4b"), e110_hybrid=("g4b", "e110"),
            native_hybrid=("g4b", "native"), e110_direct=("e110", "e110"),
            native_direct=("native", "native"))


def semantic_iou(logit, gt, classes, valid):
    lut = np.zeros(int(gt.max())+1, np.uint8)
    for key, value in classes.items():
        lut[int(key)] = int(value)
    target, pred = lut[gt], logit > 0
    scores = []
    for cls in (0, 1):
        a, b = (target == cls) & valid, (pred == cls) & valid
        scores.append(float((a & b).sum() / max(1, (a | b).sum())))
    return dict(pearlite=scores[0], ferrite=scores[1], mean=float(np.mean(scores)))


@torch.inference_mode()
def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "cache").mkdir()
    old_dir = Path(args.previous_dir)
    previous = json.loads((old_dir / "summary.json").read_text(encoding="utf-8"))
    main_cfg = load_config("config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml")
    direct_cfg = load_config("config/train/direct_gtv2_clean_60x60.yaml")
    native_cfg = load_config(args.candidate_config)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    models, payloads = {}, {}
    paths = dict(previous["checkpoints"], native=str(Path(args.candidate_checkpoint).resolve()))
    models["g4b"], payloads["g4b"] = load_fused_deployment_model(paths["g4b"], main_cfg, device)
    for name, cfg in (("e110", direct_cfg), ("native", native_cfg)):
        models[name], payloads[name] = load_direct_semantic_affinity_model(paths[name], cfg, device)
        payload = payloads[name]
        if payload["phase"] != "joint_lora" or payload["epoch"] != payload["loss_selection"]["phase_best_epoch"]["joint_lora"]:
            raise ValueError(f"Expected joint validation-loss best checkpoint: {name}")
        if payload["split"] != previous["split"]:
            raise ValueError(f"Different evaluation split: {name}")
    if payloads["native"]["loss_scales"] != payloads["e110"]["loss_scales"]:
        raise ValueError("Different loss calibration scales")
    norms = {name: model.encoder.input_normalization for name, model in models.items()}
    if norms != dict(g4b="legacy_none", e110="imagenet_v1", native="imagenet_v1"):
        raise ValueError(norms)
    if any(int(payloads[name]["config"]["direct_semantic_affinity"]["input_size"]) != 1024
           for name in ("e110", "native")):
        raise ValueError("This experiment requires 1024 inputs")
    cfg = dict(main_cfg["inference"])
    if cfg != previous["inference"] or cfg["boundary_threshold"] != .65:
        raise ValueError("Mainline postprocessing changed")
    dep = main_cfg["affinity_deployment"]
    kwargs = {key: float(dep[key]) for key in ("distance2_weight", "distance4_weight", "support_threshold", "support_temperature")}
    if kwargs != previous["fusion"]:
        raise ValueError("Mainline fusion changed")
    hashes = {name: digest(path) for name, path in paths.items()}
    if any(hashes[name] != previous["checkpoint_sha256"][name] for name in ("g4b", "e110")):
        raise ValueError("Reference checkpoint changed")
    for model in models.values():
        model.eval()
    source_files = ["tools/compare_direct_native_crop.py", "tools/diagnose_affinity_readout.py",
                    "utils/affinity_fusion.py", "utils/affinity_deployment.py", "utils/post_process.py",
                    "utils/instance_metrics.py", "models/direct_semantic_affinity.py", "models/sam2_encoder.py"]
    result = dict(format="nativecrop_fixed_deployment_v1", arms=ARMS, inference=cfg, fusion=kwargs,
                  input_size=1024, output_grid=512, actual_normalization=norms, split=previous["split"],
                  checkpoints=paths, checkpoint_sha256=hashes,
                  checkpoint_metadata={name: {key: payloads[name][key] for key in ("epoch", "phase", "loss_scales", "loss_selection")}
                                       for name in ("e110", "native")},
                  source_sha256={name: digest(ROOT / name) for name in source_files}, images=[], summaries={})
    manual_dir = Path(project_path(native_cfg, native_cfg["direct_semantic_affinity"]["manual_target"]["dataset_dir"]))
    for prior in previous["images"]:
        stem, has_gt = prior["stem"], prior["has_gt"]
        rgb, tensor, ph, pw = prepare_image(prior["image_path"], 1024, device)
        with np.load(old_dir / "cache" / f"{stem}.npz") as old:
            if not np.array_equal(rgb, old["rgb"]):
                raise ValueError(f"RGB changed: {stem}")
            cache = {key: old[key] for key in ("rgb", "gt", "valid", "rgb_grid", "gt_grid", "valid_grid",
                                               "edge_target", "edge_valid", "original_grid", "filled_grid")}
            cache["input_padding_hw"] = np.asarray([ph, pw], dtype=np.int32)
            gt, valid = cache["gt"], cache["valid"]
            classes = {}
            if has_gt:
                actual_gt, classes, _, actual_valid, _ = load_manual_target_archive(manual_dir / f"{stem}_gt.npz")
                if not np.array_equal(gt, actual_gt) or not np.array_equal(valid, actual_valid):
                    raise ValueError(f"New GT differs from prior evaluation: {stem}")
            entry = dict(stem=stem, has_gt=has_gt, image_path=prior["image_path"], native_shape=list(gt.shape),
                         grid_content_shape=prior["grid_content_shape"], rois=prior["rois"], raw={}, boundary={},
                         semantic_native_iou={}, arms={}, reference_checks={})
            semantic, boundary, masks = {}, {}, {}
            for name, model in models.items():
                output = model(tensor)
                logits = output["affinity_logits"]
                if logits.shape != (1, 8, 512, 512) or not torch.isfinite(logits).all():
                    raise ValueError(f"Bad affinity output: {name}")
                if "coarse_affinity_logits" in output:
                    raise ValueError("Residual architecture requires a separate comparison")
                semantic[name] = crop_letterbox_output(output["semantic_logits"], 1024, ph, pw, gt.shape).cpu()
                cache[f"{name}_semantic_logits"] = output["semantic_logits"][0, 0].cpu().numpy()
                if not torch.isfinite(semantic[name]).all():
                    raise ValueError(f"Bad semantic output: {name}")
                cache[f"{name}_semantic_grid"] = F.interpolate(output["semantic_logits"], size=(512, 512), mode="bilinear", align_corners=True).sigmoid()[0, 0].cpu().numpy()
                cache[f"{name}_prob"] = logits.sigmoid()[0].cpu().numpy()
                pieces = decomposed_fusion(logits, "mean", kwargs)
                arrays = {key: value[0, 0].cpu().numpy() for key, value in pieces.items()}
                cache[f"{name}_boundary"] = arrays["boundary"]
                boundary[name] = probability_to_logit(crop_letterbox_output(pieces["boundary"], 1024, ph, pw, gt.shape).cpu())
                masks[name] = diagnostic_markers(boundary[name].sigmoid()[0, 0].numpy(), cfg)
                if name in ("g4b", "e110"):
                    delta = float(np.abs(cache[f"{name}_prob"] - old[f"{name}_prob"]).max())
                    entry["reference_checks"][f"{name}_max_probability_delta"] = delta
                    np.testing.assert_allclose(cache[f"{name}_prob"], old[f"{name}_prob"], atol=1e-6, rtol=1e-6)
                if has_gt:
                    entry["raw"][name] = raw_statistics(cache[f"{name}_prob"], cache["edge_target"], cache["edge_valid"], cache["original_grid"], cache["filled_grid"])
                    entry["boundary"][name] = boundary_statistics(arrays["boundary"], arrays["short"], cache["edge_target"], cache["edge_valid"], cache["gt_grid"], cache["valid_grid"])
                    entry["semantic_native_iou"][name] = semantic_iou(semantic[name][0, 0].numpy(), gt, classes, valid)
                del output, pieces, logits
            for arm, (sem_name, aff_name) in ARMS.items():
                destination = out / "instances" / arm
                destination.mkdir(parents=True, exist_ok=True)
                _, prediction, predicted_classes = postprocess(torch.cat([semantic[sem_name], boundary[aff_name]], 1),
                    gt.shape, destination, stem, cfg, .65, False, image_rgb=rgb)
                if prediction.dtype != np.uint16 or prediction.ndim != 2 or prediction.shape != gt.shape or int(prediction.max()) > 65535:
                    raise ValueError("Final output violates instance contract")
                cache[f"{arm}_instances"] = prediction
                cache[f"{arm}_markers"] = masks[aff_name]["markers"]
                record = dict(instances=len(predicted_classes), seeds=int(masks[aff_name]["markers"].max()))
                if has_gt:
                    masked = prediction.copy()
                    masked[~valid] = 0
                    record.update(metrics=compact(evaluate_instance_pair(gt, classes, masked, predicted_classes)),
                                  topology=topology(gt, prediction, valid), seed_topology=topology(gt, masks[aff_name]["markers"], valid))
                entry["arms"][arm] = record
                if arm in ("g4b", "e110_hybrid"):
                    old_arm = "g4b_mean" if arm == "g4b" else "e110_mean"
                    same = bool(np.array_equal(prediction, old[f"{old_arm}_instances"]))
                    entry["reference_checks"][f"{arm}_same_instances"] = same
                    if not same:
                        raise ValueError(f"Reference final instances changed: {stem} {arm}")
                    old_classes = json.loads((old_dir / "instances" / old_arm / f"{stem}_class.json").read_text(encoding="utf-8"))
                    new_classes = json.loads((destination / f"{stem}_class.json").read_text(encoding="utf-8"))
                    entry["reference_checks"][f"{arm}_same_classes"] = old_classes == new_classes
                    if old_classes != new_classes or not np.array_equal(masks[aff_name]["markers"], old[f"{old_arm}_markers"]):
                        raise ValueError(f"Reference classes/markers changed: {stem} {arm}")
                    if has_gt and record["metrics"] != prior["arms"][old_arm]["metrics"]:
                        raise ValueError(f"Reference evaluation changed: {stem} {arm}")
                print(stem, arm, json.dumps({key: record.get("metrics", {}).get(key) for key in
                     ("score_total", "instance_miou_valid", "valid_matches", "pred_count")}), flush=True)
            np.savez_compressed(out / "cache" / f"{stem}.npz", **cache)
        result["images"].append(entry)
        result["summaries"] = {arm: summarize_instance_results([im["arms"][arm]["metrics"] for im in result["images"] if im["has_gt"]]) for arm in ARMS}
        (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-dir", default="outputs/20260910_111210_diag")
    parser.add_argument("--candidate-config", default="config/train/direct_gtv2_clean_nativecrop_60x60.yaml")
    parser.add_argument("--candidate-checkpoint", default="outputs/20260910_clean60_nativecrop/best_direct_dual.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
