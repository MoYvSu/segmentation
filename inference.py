# -*- coding: utf-8 -*-
"""
测试集交卷预测主入口（双任务距离场版本）
=========================================
加载训练好的 FPN 解码头权重，对测试集图像进行推理，
执行完整的后处理流程（动态上采样 + 二分类阈值化 + 拓扑剥离 + 实例 ID 分配）。

使用方法：
    conda activate sam2_env
    python inference.py --config config/default_config.yaml --checkpoint outputs/best_model.pth

输出：
    - {basename}_inst.png   : 单通道 uint8 实例图 (1~255)
    - {basename}_class.json : {"实例ID": 类别标签} 映射
    - {basename}_mask.png   : 二分类可视化掩码（可选）
    - {basename}_dist.png   : 距离场可视化（可选）
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
    topo_instance_separation,
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
    """构建模型并加载训练好的 decoder 权重。"""
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]

    ckpt_path = os.path.join(paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    if not os.path.exists(ckpt_path):
        logger.warning(f"SAM 2 权重文件不存在: {ckpt_path}")

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
        logger.info(f"加载 decoder 权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        logger.info(
            f"权重加载成功 (epoch={checkpoint.get('epoch', '?')}, "
            f"best_val_iou={checkpoint.get('best_val_iou', '?')})"
        )
    else:
        logger.warning("未加载 decoder 权重，将使用随机初始化的 decoder！")

    model.eval()
    return model


def predict_single_image(
    model, image_path, device, image_size=1024,
    min_instance_area=50, max_instance_id=255, connectivity=8,
    interpolate_mode="bilinear", align_corners=True,
    threshold=0.5,
    output_dir=None, save_visualization=True,
):
    """对单张图像进行推理 + 后处理（双任务距离场版本）。"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    image_lb, scale, pad_h, pad_w = letterbox(image_rgb, image_size)

    image_tensor = torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    basename = os.path.splitext(os.path.basename(image_path))[0]
    with torch.no_grad():
        output = model(image_tensor, output_size=(h_orig, w_orig))

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
        )
    else:
        output_paths = {}

    # 提取二值掩码和实例分割结果
    mask = output_to_binary_mask(
        output, threshold=threshold, original_size=(h_orig, w_orig),
        mode=interpolate_mode, align_corners=align_corners,
    )
    inst_map, class_map = topo_instance_separation(
        mask, min_instance_area=min_instance_area,
        max_instance_id=max_instance_id, connectivity=connectivity,
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
    }


def main():
    parser = argparse.ArgumentParser(description="低碳钢金相分割推理 (双任务距离场版本)")
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

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA 不可用，切换到 CPU")
        device = "cpu"

    test_dir = args.test_dir or os.path.join(paths_cfg["project_root"], infer_cfg["test_dir"])
    output_dir = args.output_dir or os.path.join(paths_cfg["project_root"], infer_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"测试目录: {test_dir}")
    logger.info(f"输出目录: {output_dir}")

    model = build_model(config, device, args.checkpoint)

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        logger.error(f"测试目录中未找到图像: {test_dir}")
        return

    logger.info(f"找到 {len(image_paths)} 张测试图像")

    all_results = []
    total_time = 0.0

    for img_path in image_paths:
        start_time = time.time()
        logger.info(f"处理: {os.path.basename(img_path)}")

        result = predict_single_image(
            model, img_path, device,
            image_size=config["data"]["image_size"],
            min_instance_area=infer_cfg["min_instance_area"],
            max_instance_id=infer_cfg["max_instance_id"],
            connectivity=post_cfg["connectivity"],
            interpolate_mode=post_cfg["interpolate_mode"],
            align_corners=post_cfg["align_corners"],
            threshold=infer_cfg.get("threshold", 0.5),
            output_dir=output_dir,
            save_visualization=True,
        )

        elapsed = time.time() - start_time
        total_time += elapsed
        all_results.append(result)

        logger.info(
            f"  完成 ({elapsed:.2f}s): "
            f"实例数={result['num_instances']} "
            f"(铁素体={result['num_ferrite']}, 珠光体={result['num_pearlite']})"
        )

    logger.info("=" * 60)
    logger.info("推理完成汇总:")
    logger.info(f"  总图像数: {len(all_results)}")
    logger.info(f"  总耗时: {total_time:.1f}s ({total_time / len(all_results):.2f}s/张)")
    total_instances = sum(r["num_instances"] for r in all_results)
    total_ferrite = sum(r["num_ferrite"] for r in all_results)
    total_pearlite = sum(r["num_pearlite"] for r in all_results)
    logger.info(f"  总实例数: {total_instances} (铁素体={total_ferrite}, 珠光体={total_pearlite})")
    logger.info(f"  输出文件保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()