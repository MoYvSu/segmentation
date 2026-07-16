# -*- coding: utf-8 -*-
"""
Test inference entry (vector field version - unified letterbox)
================================================================
Load trained FPN decoder weights, run inference on test images,
perform vector field centroid collapse + DBSCAN clustering for instance separation.

All images are processed via Letterbox to 1024x1024 (same as training),
ensuring consistent preprocessing between train and inference.
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
    post_process_prediction_vector,
    centroid_collapse_clustering,
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


def predict_single_image(
    model, image_path, device,
    image_size=1024,
    min_instance_area=50, max_instance_id=255,
    threshold=0.5,
    output_dir=None, save_visualization=True,
    dbscan_eps=5.0, dbscan_min_samples=3, downsample_grid=4,
):
    """
    Unified Letterbox inference + vector field post-processing for a single image.

    All images are Letterboxed to image_size x image_size (same as training),
    regardless of original dimensions. This ensures the model sees the same
    preprocessing it was trained on.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Letterbox to image_size x image_size (same as training)
    image_lb, scale, pad_h, pad_w = letterbox(image_rgb, image_size)
    image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Forward pass without output_size: get native FPN resolution (image_size//4)
        output = model(image_tensor)

    # Inverse Letterbox: crop padding from model output, then upsample to original size
    # FPN native output is image_size//4 (e.g. 256 for 1024 input)
    content_h = int(round(h_orig * scale / 4))  # content rows in native resolution
    content_w = int(round(w_orig * scale / 4))  # content cols in native resolution
    output = output[:, :, :content_h, :content_w]
    output = F.interpolate(output, size=(h_orig, w_orig), mode="bilinear", align_corners=True)

    if output_dir is not None:
        output_paths, inst_map, class_map = post_process_prediction_vector(
            output=output,
            original_size=(h_orig, w_orig),
            output_dir=output_dir,
            image_basename=basename,
            image_size=image_size,
            min_instance_area=min_instance_area,
            max_instance_id=max_instance_id,
            threshold=threshold,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            downsample_grid=downsample_grid,
            save_visualization=save_visualization,
        )
    else:
        # Fallback: no output dir, return inst_map directly
        from utils.post_process import CLASS_FERRITE, CLASS_PEARLITE

        if output.ndim == 3:
            output = output.unsqueeze(0)

        seg_logits = output[0, 0].cpu()
        vx_field = output[0, 1].cpu().numpy().astype(np.float32)
        vy_field = output[0, 2].cpu().numpy().astype(np.float32)
        seg_prob = torch.sigmoid(seg_logits)
        mask = (seg_prob > threshold).numpy().astype(np.uint8)

        ferrite_binary = (mask == CLASS_FERRITE).astype(np.uint8)
        if ferrite_binary.sum() > 0:
            inst_map, class_map = centroid_collapse_clustering(
                ferrite_binary, vx_field, vy_field,
                image_size=image_size,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                downsample_grid=downsample_grid,
                min_instance_area=min_instance_area,
                max_instance_id=max_instance_id,
            )
        else:
            inst_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
            class_map = {}

        # Pearlite via connected components
        pearlite_binary = (mask == CLASS_PEARLITE).astype(np.uint8)
        if pearlite_binary.sum() > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pearlite_binary, connectivity=8)
            current_id = max(class_map.keys()) + 1 if class_map else 1
            for label_id in range(1, num_labels):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < min_instance_area:
                    continue
                if current_id > max_instance_id:
                    current_id = max_instance_id
                inst_map[labels == label_id] = current_id
                class_map[current_id] = CLASS_PEARLITE
                if current_id < max_instance_id:
                    current_id += 1

        output_paths = {}

    n_ferrite = sum(1 for v in class_map.values() if v == 1)
    n_pearlite = sum(1 for v in class_map.values() if v == 0)

    return {
        "image_path": image_path,
        "original_size": (h_orig, w_orig),
        "num_instances": len(class_map),
        "num_ferrite": n_ferrite,
        "num_pearlite": n_pearlite,
        "output_paths": output_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Metallographic segmentation inference (vector field version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Default: use config checkpoint_stage, run on data/test/
  python inference.py

  # Override checkpoint path directly (highest priority)
  python inference.py --checkpoint outputs/stage1/best_model.pth

  # Custom test images and output directory
  python inference.py --test_dir data/my_test_images --output_dir outputs/my_results

Output files (per image <basename>):
  <basename>_inst.png   - Instance map (uint8, 1-255 by descending area)
  <basename>_class.json - Instance ID -> class mapping (1=ferrite, 0=pearlite)
  <basename>_mask.png   - Binary mask visualization (green=ferrite, red=pearlite)
  <basename>_vec.png    - Vector field visualization (color-coded by direction)

Processing:
  All images are Letterboxed to 1024x1024 (same as training), then a single
  forward pass is performed. Output is upsampled to original resolution for
  post-processing (vector field centroid collapse + DBSCAN clustering).
""",
    )
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to YAML config file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained decoder checkpoint .pth file (overrides config checkpoint_stage)",
    )
    parser.add_argument(
        "--test_dir", type=str, default=None,
        help="Directory containing test images (default: from config inference.test_dir)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for output files (default: from config inference.output_dir)",
    )
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

    # ------------------------------------------------------------------
    # 模型权重路径解析
    # ------------------------------------------------------------------
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        logger.info(f"Using checkpoint from CLI: {checkpoint_path}")
    else:
        checkpoint_stage = infer_cfg.get("checkpoint_stage", "stage1")
        if checkpoint_stage == "stage2":
            checkpoint_path = os.path.join(
                paths_cfg["project_root"],
                infer_cfg.get("stage2_checkpoint", "outputs/stage2/best_model_stage2.pth"),
            )
            logger.info(f"Using Stage-2 checkpoint from config: {checkpoint_path}")
        else:
            checkpoint_path = os.path.join(
                paths_cfg["project_root"],
                infer_cfg.get("stage1_checkpoint", "outputs/stage1/best_model.pth"),
            )
            logger.info(f"Using Stage-1 checkpoint from config: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Test dir: {test_dir}")
    logger.info(f"Output dir: {output_dir}")

    model = build_model(config, device, checkpoint_path)

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
            threshold=infer_cfg.get("threshold", 0.5),
            output_dir=output_dir,
            save_visualization=post_cfg.get("save_visualization", False),
            dbscan_eps=post_cfg.get("dbscan_eps", 5.0),
            dbscan_min_samples=post_cfg.get("dbscan_min_samples", 3),
            downsample_grid=post_cfg.get("downsample_grid", 4),
        )

        elapsed = time.time() - start_time
        total_time += elapsed
        all_results.append(result)

        logger.info(
            f"  Done ({elapsed:.2f}s): "
            f"instances={result['num_instances']} "
            f"(ferrite={result['num_ferrite']}, pearlite={result['num_pearlite']})"
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