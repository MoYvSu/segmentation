# -*- coding: utf-8 -*-
"""
Stage-1 边界伪标签离线预计算
============================
在 Stage 2 训练前，用冻结的 Stage-1 模型对全部无标签图像一次性前向，
生成边界概率图缓存（memmap .npy + names.txt + report.csv + exclude.txt）。

作用：
  1. 训练时免去每个 step 的 ref_model 前向（stage1_direct 模式）；
  2. 离线统计伪标签质量（max / >0.3 / >0.5 占比），剔除"无边界响应"的低质量图；
  3. 可选 TTA（原图/水平翻转/垂直翻转/旋转180 平均），锚点伪标签更稳。

用法:
    python tools/precompute_pseudo_labels.py --config config/default_config.yaml
    python tools/precompute_pseudo_labels.py --no_tta
    python tools/precompute_pseudo_labels.py --exclude_max_threshold 0.3
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from data.dataset import letterbox
from inference import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_tta(model, image_lb, device, use_tta: bool) -> np.ndarray:
    """对单张 letterbox 图像预测边界概率（可选 4 视角 TTA 平均）。

    Returns:
        boundary_prob: [1024, 1024] float32，sigmoid 后概率图
    """
    views = [image_lb]
    if use_tta:
        views += [
            image_lb[:, ::-1].copy(),
            image_lb[::-1, :].copy(),
            image_lb[::-1, ::-1].copy(),
        ]

    batch = (
        torch.from_numpy(np.stack(views)).float().permute(0, 3, 1, 2) / 255.0
    ).to(device)

    with torch.no_grad():
        output = model(batch)  # [V, 2, 1024, 1024]
        probs = torch.sigmoid(output[:, 1]).cpu().numpy()  # [V, H, W]

    if use_tta:
        # 逆变换后取平均
        inv = [probs[0], probs[1][:, ::-1], probs[2][::-1, :], probs[3][::-1, ::-1]]
        return np.mean(inv, axis=0).astype(np.float32)
    return probs[0].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Stage-1 boundary pseudo-labels for unlabeled set",
    )
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Stage-1 checkpoint（默认取 config.inference.stage1_checkpoint）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="缓存输出目录（默认取 config.semi_supervised.pseudo_label_cache_dir）")
    parser.add_argument("--no_tta", action="store_true",
                        help="禁用 TTA（默认启用 4 视角平均）")
    parser.add_argument("--exclude_max_threshold", type=float, default=0.3,
                        help="边界概率 max 低于该阈值的图像写入 exclude.txt（默认 0.3）")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    semi_cfg = config.get("semi_supervised", {})
    sam2_cfg = config["sam2"]

    device = args.device or sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    unlabeled_dir = os.path.join(
        paths_cfg["project_root"],
        semi_cfg.get("unlabeled_dir", "data/unlabeled"),
    )
    checkpoint_path = args.checkpoint or os.path.join(
        paths_cfg["project_root"],
        config["inference"].get("stage1_checkpoint", "outputs/stage1/best_model.pth"),
    )
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            paths_cfg["project_root"],
            semi_cfg.get(
                "pseudo_label_cache_dir", "outputs/pseudo_labels/stage1_boundary"
            ),
        )

    if not os.path.exists(checkpoint_path):
        logger.error(f"Stage-1 checkpoint not found: {checkpoint_path}")
        return

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(unlabeled_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        logger.error(f"No images found in unlabeled dir: {unlabeled_dir}")
        return

    logger.info(f"Unlabeled images: {len(image_paths)}")
    logger.info(f"Stage-1 checkpoint: {checkpoint_path}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"TTA: {'ENABLED' if not args.no_tta else 'DISABLED'}")

    model = build_model(config, device, checkpoint_path)

    os.makedirs(output_dir, exist_ok=True)
    image_size = config["data"]["image_size"]
    n = len(image_paths)

    probs_path = os.path.join(output_dir, "boundary_probs.npy")
    names_path = os.path.join(output_dir, "names.txt")
    report_path = os.path.join(output_dir, "report.csv")
    exclude_path = os.path.join(output_dir, "exclude.txt")

    mem = np.lib.format.open_memmap(
        probs_path, mode="w+", dtype=np.float16,
        shape=(n, image_size, image_size),
    )

    basenames = []
    report_rows = []
    exclude_rows = []
    t0 = time.time()

    for i, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning(f"  Skip unreadable: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lb, _, _, _ = letterbox(image_rgb, image_size)

        prob = predict_tta(model, image_lb, device, use_tta=not args.no_tta)
        mem[i] = prob.astype(np.float16)
        basenames.append(basename)

        p_max = float(prob.max())
        f03 = float((prob > 0.3).mean())
        f05 = float((prob > 0.5).mean())
        report_rows.append(f"{basename},{p_max:.4f},{f03:.4f},{f05:.4f}")
        if p_max < args.exclude_max_threshold:
            exclude_rows.append(basename)

        if (i + 1) % 100 == 0 or i == n - 1:
            logger.info(
                f"  [{i + 1}/{n}] {basename}: max={p_max:.3f} "
                f">0.3={f03 * 100:.1f}% >0.5={f05 * 100:.1f}% "
                f"({(time.time() - t0) / (i + 1):.2f}s/img)"
            )

    mem.flush()

    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(basenames) + "\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("basename,max_prob,frac_gt0.3,frac_gt0.5\n")
        f.write("\n".join(report_rows) + "\n")
    with open(exclude_path, "w", encoding="utf-8") as f:
        f.write("\n".join(exclude_rows) + ("\n" if exclude_rows else ""))

    logger.info("=" * 60)
    logger.info(
        f"Done! {len(basenames)} images -> {probs_path} "
        f"({os.path.getsize(probs_path) / 1e9:.2f} GB)"
    )
    logger.info(f"Excluded (max < {args.exclude_max_threshold}): {len(exclude_rows)}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
