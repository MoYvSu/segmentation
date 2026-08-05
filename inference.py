# -*- coding: utf-8 -*-
"""
推理入口（边界预测版本）
========================
加载训练好的 FPN decoder 权重，对测试图像执行前向推理，
执行边界骨架化 + 受阻分水岭 + 语义投票实现实例分割。

所有图像通过 Letterbox 处理到 1024x1024（与训练一致）。
"""

import argparse
import glob
import logging
import os
import sys
import time

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
from utils.post_process import post_process_prediction_boundary

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
        # 含 lora_state_dict 的检查点：注入并加载 LoRA（trunk 域适配）
        from models.lora import load_lora_from_checkpoint
        load_lora_from_checkpoint(model, checkpoint)
        model.to(device)   # 注入发生在 .to() 之后，需再移动一次 LoRA 参数
        # 兼容新旧 checkpoint key
        best_score = checkpoint.get(
            "best_composite_score", checkpoint.get("best_val_iou", "?")
        )
        logger.info(
            f"Weights loaded (epoch={checkpoint.get('epoch', '?')}, "
            f"best_score={best_score})"
        )
    else:
        logger.warning("No decoder weights loaded, using random init!")

    model.eval()
    return model


def predict_single_image(
    model, image_path, device,
    image_size=1024,
    min_instance_area=50, max_instance_id=255,
    threshold=0.5, boundary_threshold=0.5,
    boundary_logit_scale=1.0,
    sem_edge_boost_alpha=0.0,
    use_tta=False,
    watershed_dilate_width=2,
    output_dir=None, save_visualization=True,
):
    """Letterbox inference + boundary watershed post-processing."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Letterbox
    image_lb, scale, pad_h, pad_w = letterbox(image_rgb, image_size)
    image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    if use_tta:
        # 4 视角 TTA：hflip / vflip / rot180，logits 平均后逆变换回原方向
        views = torch.cat(
            [
                image_tensor,
                torch.flip(image_tensor, dims=[3]),
                torch.flip(image_tensor, dims=[2]),
                torch.rot90(image_tensor, 2, dims=[2, 3]),
            ],
            dim=0,
        )
        with torch.no_grad():
            outs = model(views)  # [4, 2, H, W]
        output = (
            outs[0:1]
            + torch.flip(outs[1:2], dims=[3])
            + torch.flip(outs[2:3], dims=[2])
            + torch.rot90(outs[3:4], 2, dims=[2, 3])
        ) / 4.0
    else:
        with torch.no_grad():
            output = model(image_tensor)

    # Inverse Letterbox: crop padding, upsample to original size
    content_h = int(round(h_orig * scale / 4))
    content_w = int(round(w_orig * scale / 4))
    output = output[:, :, :content_h, :content_w]
    output = F.interpolate(output, size=(h_orig, w_orig), mode="bilinear", align_corners=True)

    if output_dir is not None:
        output_paths, inst_map, class_map = post_process_prediction_boundary(
            output=output,
            original_size=(h_orig, w_orig),
            output_dir=output_dir,
            image_basename=basename,
            min_instance_area=min_instance_area,
            max_instance_id=max_instance_id,
            threshold=threshold,
            boundary_threshold=boundary_threshold,
            boundary_logit_scale=boundary_logit_scale,
            sem_edge_boost_alpha=sem_edge_boost_alpha,
            watershed_dilate_width=watershed_dilate_width,
            save_visualization=save_visualization,
        )
    else:
        output_paths = {}
        inst_map = np.zeros((h_orig, w_orig), dtype=np.uint8)
        class_map = {}

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
        description="Metallographic segmentation inference (boundary prediction version)",
    )
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tta", action="store_true",
                        help="推理 TTA（hflip/vflip/rot180 平均）")
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

    # Checkpoint path
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
            boundary_threshold=infer_cfg.get("boundary_threshold", 0.5),
            boundary_logit_scale=infer_cfg.get("boundary_logit_scale", 1.0),
            sem_edge_boost_alpha=infer_cfg.get("sem_edge_boost_alpha", 0.0),
            use_tta=infer_cfg.get("tta", False) or args.tta,
            watershed_dilate_width=infer_cfg.get("watershed_dilate_width", 2),
            output_dir=output_dir,
            save_visualization=post_cfg.get("save_visualization", False),
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
