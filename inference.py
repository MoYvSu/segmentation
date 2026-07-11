# -*- coding: utf-8 -*-
"""
Test inference entry (dual-task distance field - multi-resolution adaptive)
=============================================================================
Load trained FPN decoder weights, run inference on test images,
perform full post-processing (upsample + threshold + distance compensation +
watershed topo separation + instance ID assignment).

Multi-resolution adaptive inference:
- Case A (max dim <= tile_threshold): Letterbox to 1024x1024, single forward.
- Case B (max dim > tile_threshold): Tiled inference (1024 window, 512 stride),
  Gaussian weight blending for seamless stitching.
"""

import argparse
import glob
import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from data.dataset import letterbox
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.sam2_encoder import SAM2Encoder
from utils.post_process import (
    post_process_prediction,
    output_to_binary_mask,
    output_to_distance_field,
    topo_instance_separation,
    compensate_distance_field,
    gaussian_weight_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config, device, checkpoint_path=None):
    """Build model and load trained decoder weights."""
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]

    ckpt_path = os.path.join(paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    if not os.path.exists(ckpt_path):
        logger.warning(f"SAM 2 checkpoint not found: {ckpt_path}")

    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
        device=device,
        freeze=True,
        sam2_repo_path=os.path.join(paths_cfg["project_root"], sam2_cfg["sam2_repo_path"]),
    )

    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=decoder_cfg["fpn_channels"],
        num_classes=decoder_cfg["num_classes"],
        dropout=decoder_cfg["dropout"],
        use_bn=decoder_cfg["use_bn"],
    )

    model = SegmentationModel(encoder, decoder)
    model = model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading decoder weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        logger.info(
            f"Weights loaded (epoch={checkpoint.get('epoch', '?')}, "
            f"best_val_iou={checkpoint.get('best_val_iou', '?')})"
        )
    else:
        logger.warning("No decoder weights loaded, using random init!")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Multi-resolution adaptive inference
# ---------------------------------------------------------------------------

def predict_single_patch(model, image_patch, device):
    """Forward pass on a single 1024x1024 patch."""
    image_tensor = torch.from_numpy(image_patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output


def tiled_inference(model, image, device, tile_size=1024, stride=512):
    """Sliding window tiled inference with Gaussian weight blending."""
    h, w = image.shape[:2]
    n_channels = 2

    output_sum = torch.zeros((1, n_channels, h, w), dtype=torch.float32)
    weight_sum = torch.zeros((1, 1, h, w), dtype=torch.float32)

    weight_map = gaussian_weight_map(tile_size)
    weight_tensor = torch.from_numpy(weight_map).float().unsqueeze(0).unsqueeze(0)

    positions_h = list(range(0, max(1, h - tile_size + 1), stride))
    if positions_h[-1] + tile_size < h:
        positions_h.append(h - tile_size)
    positions_w = list(range(0, max(1, w - tile_size + 1), stride))
    if positions_w[-1] + tile_size < w:
        positions_w.append(w - tile_size)

    total_patches = len(positions_h) * len(positions_w)
    patch_idx = 0

    for top in positions_h:
        for left in positions_w:
            patch_idx += 1
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)

            patch = image[top:bottom, left:right]
            patch_h, patch_w = patch.shape[:2]

            if patch_h < tile_size or patch_w < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=patch.dtype)
                padded[:patch_h, :patch_w] = patch
                patch = padded

            output_patch = predict_single_patch(model, patch, device)
            output_patch = output_patch.cpu()
            output_patch = output_patch[:, :, :patch_h, :patch_w]
            weight_patch = weight_tensor[:, :, :patch_h, :patch_w]

            output_sum[:, :, top:bottom, left:right] += output_patch * weight_patch
            weight_sum[:, :, top:bottom, left:right] += weight_patch

            if patch_idx % 3 == 0 or patch_idx == total_patches:
                logger.info(f"  Tile {patch_idx}/{total_patches} "
                            f"(top={top}, left={left}, size={patch_h}x{patch_w})")

    eps = 1e-8
    output_full = output_sum / (weight_sum + eps)
    return output_full


def predict_single_image(
    model, image_path, device,
    image_size=1024,
    min_instance_area=50, max_instance_id=255, connectivity=8,
    interpolate_mode="bilinear", align_corners=True,
    threshold=0.5,
    tile_size=1024, tile_stride=512, tile_threshold=1224,
    train_max_dim=2584,
    dist_scale_factor=10.0,
    output_dir=None, save_visualization=True,
):
    """Multi-resolution adaptive inference + post-processing for a single image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    basename = os.path.splitext(os.path.basename(image_path))[0]

    train_scale = image_size / train_max_dim
    spatial_scale = 1.0 / train_scale

    max_dim = max(h_orig, w_orig)

    if max_dim <= tile_threshold:
        logger.info(f"  Case A (letterbox): {h_orig}x{w_orig} <= {tile_threshold}")
        image_lb, scale, pad_h, pad_w = letterbox(image_rgb, image_size)
        image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            output = model(image_tensor, output_size=(h_orig, w_orig))
    else:
        logger.info(f"  Case B (tiled): {h_orig}x{w_orig} > {tile_threshold}, "
                    f"tile={tile_size}, stride={tile_stride}")
        output = tiled_inference(
            model, image_rgb, device,
            tile_size=tile_size, stride=tile_stride,
        )

    if output_dir is not None:
        output_paths = post_process_prediction(
            output=output,
            original_size=(h_orig, w_orig),
            output_dir=output_dir,
            image_basename=basename,
            min_instance_area=min_instance_area,
            max_instance_id=max_instance_id,
            connectivity=connectivity,
            interpolate_mode=interpolate_mode,
            align_corners=align_corners,
            threshold=threshold,
            save_visualization=save_visualization,
            dist_scale_factor=dist_scale_factor,
            spatial_scale=spatial_scale,
            use_watershed=True,
        )
    else:
        output_paths = {}

    mask = output_to_binary_mask(
        output, threshold=threshold, original_size=(h_orig, w_orig),
        mode=interpolate_mode, align_corners=align_corners,
    )
    dist_field = output_to_distance_field(
        output, original_size=(h_orig, w_orig),
        mode=interpolate_mode, align_corners=align_corners,
    )
    dist_field = compensate_distance_field(
        dist_field,
        spatial_scale=spatial_scale,
        scale_factor=dist_scale_factor,
    )

    inst_map, class_map = topo_instance_separation(
        mask,
        dist_field=dist_field,
        min_instance_area=min_instance_area,
        max_instance_id=max_instance_id,
        connectivity=connectivity,
        use_watershed=True,
    )

    n_ferrite = sum(1 for v in class_map.values() if v == 1)
    n_pearlite = sum(1 for v in class_map.values() if v == 0)

    return {
        "image_path": image_path,
        "original_size": (h_orig, w_orig),
        "num_instances": len(class_map),
        "num_ferrite": n_ferrite,
        "num_pearlite": n_pearlite,
        "output_paths": output_paths,
        "spatial_scale": spatial_scale,
        "case": "A" if max_dim <= tile_threshold else "B",
    }


def main():
    parser = argparse.ArgumentParser(description="Metallographic segmentation inference (multi-resolution)")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth")
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]
    infer_cfg = config["inference"]
    post_cfg = config["post_process"]
    data_cfg = config["data"]

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    test_dir = args.test_dir or os.path.join(paths_cfg["project_root"], infer_cfg["test_dir"])
    output_dir = args.output_dir or os.path.join(paths_cfg["project_root"], infer_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Test dir: {test_dir}")
    logger.info(f"Output dir: {output_dir}")

    model = build_model(config, device, args.checkpoint)

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        logger.error(f"No images found in test dir: {test_dir}")
        return

    logger.info(f"Found {len(image_paths)} test images")

    all_results = []
    total_time = 0.0

    for img_path in image_paths:
        start_time = time.time()
        logger.info(f"Processing: {os.path.basename(img_path)}")

        result = predict_single_image(
            model, img_path, device,
            image_size=data_cfg["image_size"],
            min_instance_area=infer_cfg["min_instance_area"],
            max_instance_id=infer_cfg["max_instance_id"],
            connectivity=post_cfg["connectivity"],
            interpolate_mode=post_cfg["interpolate_mode"],
            align_corners=post_cfg["align_corners"],
            threshold=infer_cfg.get("threshold", 0.5),
            tile_size=infer_cfg.get("tile_size", 1024),
            tile_stride=infer_cfg.get("tile_stride", 512),
            tile_threshold=infer_cfg.get("tile_threshold", 1224),
            train_max_dim=infer_cfg.get("train_max_dim", 2584),
            dist_scale_factor=data_cfg.get("dist_scale_factor", 10.0),
            output_dir=output_dir,
            save_visualization=True,
        )

        elapsed = time.time() - start_time
        total_time += elapsed
        all_results.append(result)

        logger.info(
            f"  Done ({elapsed:.2f}s): "
            f"Case={result['case']} "
            f"instances={result['num_instances']} "
            f"(ferrite={result['num_ferrite']}, pearlite={result['num_pearlite']}) "
            f"spatial_scale={result['spatial_scale']:.4f}"
        )

    logger.info("=" * 60)
    logger.info("Inference complete:")
    logger.info(f"  Total images: {len(all_results)}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time / len(all_results):.2f}s/img)")
    total_instances = sum(r["num_instances"] for r in all_results)
    total_ferrite = sum(r["num_ferrite"] for r in all_results)
    total_pearlite = sum(r["num_pearlite"] for r in all_results)
    logger.info(f"  Total instances: {total_instances} (ferrite={total_ferrite}, pearlite={total_pearlite})")
    logger.info(f"  Output saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()