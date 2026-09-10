# -*- coding: utf-8 -*-
"""固定三个局部，比较全图与原尺寸裁剪的边界响应；不是最终分割评估。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import letterbox
from models.direct_semantic_affinity import load_direct_semantic_affinity_model
from models.fused_deployment import load_fused_deployment_model
from tools.diagnose_affinity_readout import decomposed_fusion, digest, spatial_domains
from utils.affinity_deployment import crop_letterbox_output
from utils.config import load_config


# 在运行前固定：两处既有漏界证据 + 第三张图的既有一致区域。
# 坐标来自上一轮全图512网格，不能根据本次裁剪输出重新选择。
REGIONS = [
    dict(stem="train_172", role="failed_pair_107_168", box=[324, 328, 377, 384]),
    dict(stem="train_873", role="failed_pair_141_142", box=[201, 260, 306, 315]),
    dict(stem="train_351", role="agreement_control", box=[373, 175, 469, 271]),
]


def rectangle_mask(shape, box):
    x0, y0, x1, y1 = box
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def metric(boundary, interface, deep, roi):
    selected = interface & roi
    interior = deep & roi
    high = boundary > .65
    return dict(
        interface_pixels=int(selected.sum()), deep_pixels=int(interior.sum()),
        interface_mean=float(boundary[selected].mean()),
        interface_high_fraction=float(high[selected].mean()),
        interface_within6_high_fraction=float(maximum_filter(high, size=13)[selected].mean()),
        deep_high_fraction=float(high[interior].mean()),
        deep_mean=float(boundary[interior].mean()),
    )


@torch.inference_mode()
def run(args):
    old = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((old / "summary.json").read_text(encoding="utf-8"))
    cfg = load_config("config/train/direct_gtv2_clean_60x60.yaml")
    main_cfg = load_config("config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml")
    device = torch.device(args.device)
    models = {}
    models["g4b"], _ = load_fused_deployment_model(previous["checkpoints"]["g4b"], main_cfg, device)
    models["e110"], payload = load_direct_semantic_affinity_model(previous["checkpoints"]["e110"], cfg, device)
    assert payload["epoch"] == 110 and payload["phase"] == "joint_lora"
    for name, model in models.items():
        model.eval()
        assert model.encoder.input_normalization == previous["actual_normalization"][name]
        assert digest(previous["checkpoints"][name]) == previous["checkpoint_sha256"][name]
    result = dict(
        format="native_scale_probe_v1", regions_fixed_before_run=REGIONS,
        checkpoint_sha256=previous["checkpoint_sha256"],
        actual_normalization=previous["actual_normalization"],
        source_sha256=digest(__file__), crop_size=1024, input_size=1024,
        fusion=previous["fusion"], short_reduction="mean", high=.65,
        rules=[
            "Native scalar boundaries are compared on exactly the same GT interface/deep pixel masks.",
            "Fusion is applied on each 512 output grid, then official bilinear align_corners=True restoration.",
            "Six-native-pixel tolerance uses a 13x13 square, not different tolerances per model/scale.",
            "Deep interior excludes GT interfaces, unknown and original frame by >21 native pixels.",
            "Scored ROI excludes 32 native pixels at artificial crop edges and original image frame.",
            "Each direction spans different native distance after cropping; raw edge accuracy across scales is not compared.",
            "Scale and context change together. These preselected local diagnostics do not establish training or full deployment gains.",
        ], images=[],
    )
    metadata = {entry["stem"]: entry for entry in previous["images"]}
    for region in REGIONS:
        stem = region["stem"]
        with np.load(old / "cache" / f"{stem}.npz") as cache:
            rgb, gt, valid = cache["rgb"], cache["gt"], cache["valid"]
            h, w = gt.shape
            gh, gw = metadata[stem]["grid_content_shape"]
            x0, y0, x1, y1 = region["box"]
            box = [int(x0*w/gw), int(y0*h/gh), int(np.ceil(x1*w/gw)), int(np.ceil(y1*h/gh))]
            side = min(1024, h, w)
            left = int(np.clip((box[0]+box[2]-side)//2, 0, w-side))
            top = int(np.clip((box[1]+box[3]-side)//2, 0, h-side))
            scored = [max(box[0], left+32), max(box[1], top+32),
                      min(box[2], left+side-32), min(box[3], top+side-32)]
            roi = rectangle_mask(gt.shape, scored)
            interface, _ = spatial_domains(gt, valid)
            excluded = interface | ~valid
            excluded[[0, -1], :] = True
            excluded[:, [0, -1]] = True
            deep = valid & (distance_transform_edt(~excluded) > 21)
            image_crop = rgb[top:top+side, left:left+side]
            lb, _, ph, pw = letterbox(image_crop, 1024)
            tensor = torch.from_numpy(lb).permute(2, 0, 1)[None].float().to(device)/255.
            entry = dict(**region, native_roi=scored, native_crop=[left, top, left+side, top+side],
                         native_pixels_per_grid_step=dict(full=[w/gw, h/gh], crop=[side/512, side/512]), arms={})
            arrays = dict(rgb=rgb[scored[1]:scored[3], scored[0]:scored[2]],
                          gt=gt[scored[1]:scored[3], scored[0]:scored[2]],
                          interface=interface[scored[1]:scored[3], scored[0]:scored[2]])
            for name, model in models.items():
                full_grid = torch.from_numpy(cache[f"{name}_mean_boundary"])[None, None]
                # 用上一轮实测content尺寸精确移除padding，再走相同恢复函数。
                full = crop_letterbox_output(full_grid, 1024, 1024-2*gh, 1024-2*gw, (h, w))[0, 0].numpy()
                output = model(tensor)
                pieces = decomposed_fusion(output["affinity_logits"], "mean", previous["fusion"])
                crop = crop_letterbox_output(pieces["boundary"], 1024, ph, pw, (side, side))[0, 0].cpu().numpy()
                native = np.zeros((h, w), np.float32)
                native[top:top+side, left:left+side] = crop
                for kind, boundary in (("full", full), ("crop", native)):
                    arm = f"{name}_{kind}"
                    entry["arms"][arm] = metric(boundary, interface, deep, roi)
                    arrays[arm] = boundary[scored[1]:scored[3], scored[0]:scored[2]]
                del output, pieces
            np.savez_compressed(out / f"{stem}.npz", **arrays)
            result["images"].append(entry)
            (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(entry), flush=True)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
