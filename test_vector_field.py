# -*- coding: utf-8 -*-
"""
向量场回归可视化测试脚本
========================
加载训练好的 FPN decoder，对测试图像执行前向推理，将向量场回归结果
以多种方式叠加到原始图像上，保持原始分辨率输出，用于诊断：
  - 向量场是否正确指向各实例质心
  - 相邻铁素体晶粒之间的向量场是否有清晰边界
  - 坍塌坐标空间的聚类是否紧凑

输出四面板拼接图（2×2 布局）：
  A. 原图 + 向量箭头叠加（网格采样，仅铁素体前景区域）
  B. HSV 方向编码图（角度→H，幅度→V）
  C. 坍塌坐标散点图（所有铁素体像素的坍塌位置）
  D. 分类 mask 可视化（绿=铁素体，红=珠光体）

用法:
  # 单张图
  python test_vector_field.py --image data/test/sample.jpg

  # 批量目录
  python test_vector_field.py --image_dir data/test/ --output_dir outputs/vec_vis

  # 指定 checkpoint
  python test_vector_field.py --image data/test/sample.jpg --checkpoint outputs/stage1/best_model.pth
"""

import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import letterbox
from inference import build_model, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASS_PEARLITE = 0
CLASS_FERRITE = 1


# ---------------------------------------------------------------------------
# 推理：获取原始分辨率的分类 + 向量场
# ---------------------------------------------------------------------------

def predict_vector_field(
    model, image_path, device, image_size=1024, threshold=0.5,
):
    """
    对单张图像执行前向推理，返回原始分辨率的分类 mask 和向量场。

    复用 inference.py 的 Letterbox + 裁剪 + 上采样逻辑，但跳过后处理，
    直接返回中间结果供可视化使用。

    Returns:
        image_rgb: [H, W, 3] 原始 RGB 图像
        mask: [H, W] 二值分类 (0=pearlite, 1=ferrite)
        seg_prob: [H, W] 分类概率图 [0, 1]
        vx_field: [H, W] Vx 偏移量（归一化 [-1,1]）
        vy_field: [H, W] Vy 偏移量（归一化 [-1,1]）
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    # Letterbox to image_size x image_size
    image_lb, scale, pad_h, pad_w = letterbox(image_rgb, image_size)
    image_tensor = (
        torch.from_numpy(image_lb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    )
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

    # Inverse Letterbox: crop padding, upsample to original size
    content_h = int(round(h_orig * scale / 4))
    content_w = int(round(w_orig * scale / 4))
    output = output[:, :, :content_h, :content_w]
    output = F.interpolate(
        output, size=(h_orig, w_orig), mode="bilinear", align_corners=True
    )

    seg_logits = output[0, 0].cpu()
    vx_field = output[0, 1].cpu().numpy().astype(np.float32)
    vy_field = output[0, 2].cpu().numpy().astype(np.float32)

    seg_prob = torch.sigmoid(seg_logits).numpy()
    mask = (seg_prob > threshold).astype(np.uint8)

    return image_rgb, mask, seg_prob, vx_field, vy_field


# ---------------------------------------------------------------------------
# 可视化面板
# ---------------------------------------------------------------------------

def draw_vector_arrows(
    image_rgb, mask, vx_field, vy_field, image_size=1024,
    grid_step=16, arrow_scale=1.0, thickness=1, tip_length=0.3,
):
    """
    面板 A：在原图上以网格采样点绘制向量箭头。

    仅在铁素体前景区域绘制。箭头方向和长度 = 反归一化后的偏移向量。
    箭头颜色按方向编码（HSV → BGR）。

    Args:
        image_rgb: [H, W, 3] 原始 RGB 图像
        mask: [H, W] 二值分类 (1=ferrite)
        vx_field: [H, W] 归一化 Vx [-1,1]
        vy_field: [H, W] 归一化 Vy [-1,1]
        image_size: 归一化因子（反归一化用）
        grid_step: 箭头采样网格间距（像素）
        arrow_scale: 箭头长度缩放系数
        thickness: 箭头线宽
        tip_length: 箭头尖端长度比例

    Returns:
        canvas: [H, W, 3] BGR 图像（带箭头叠加）
    """
    h, w = image_rgb.shape[:2]
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    ferrite_mask = (mask == CLASS_FERRITE)

    for y in range(0, h, grid_step):
        for x in range(0, w, grid_step):
            if not ferrite_mask[y, x]:
                continue

            # 反归一化偏移
            denorm_factor = float(max(image_rgb.shape[0], image_rgb.shape[1]))
            dx = vx_field[y, x] * denorm_factor * arrow_scale
            dy = vy_field[y, x] * denorm_factor * arrow_scale

            # 跳过零向量
            mag = np.sqrt(dx * dx + dy * dy)
            if mag < 1.0:
                continue

            end_x = int(round(x + dx))
            end_y = int(round(y + dy))

            # 方向编码颜色 (HSV)
            angle = np.arctan2(dy, dx)  # [-pi, pi]
            hue = ((angle + np.pi) / (2 * np.pi)) * 180  # [0, 180] for OpenCV
            hue = int(hue) % 180
            sat = 255
            val = 255
            color_bgr = cv2.cvtColor(
                np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
            )[0, 0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

            cv2.arrowedLine(
                canvas, (x, y), (end_x, end_y), color,
                thickness=thickness, tipLength=tip_length,
            )

    return canvas


def draw_hsv_direction(
    image_rgb, mask, vx_field, vy_field, alpha=0.6,
):
    """
    面板 B：HSV 方向编码图。

    向量角度 → H 通道，向量幅度 → V 通道（归一化后乘以 image_size）。
    仅铁素体区域显示彩色编码，叠加在半透明原图上。

    Args:
        image_rgb: [H, W, 3] 原始 RGB 图像
        mask: [H, W] 二值分类
        vx_field: [H, W] 归一化 Vx [-1,1]
        vy_field: [H, W] 归一化 Vy [-1,1]
        alpha: 原图透明度 (0=完全覆盖, 1=原图)

    Returns:
        canvas: [H, W, 3] BGR 图像
    """
    h, w = image_rgb.shape[:2]

    # 计算角度和幅度
    angle = np.arctan2(vy_field, vx_field)  # [-pi, pi]
    magnitude = np.sqrt(vx_field ** 2 + vy_field ** 2)  # [0, ~1.4]

    # HSV 编码
    hue = ((angle + np.pi) / (2 * np.pi)) * 180  # [0, 180]
    sat = np.ones_like(hue) * 255
    val = np.clip(magnitude / 1.4 * 255, 0, 255)  # 归一化到 [0, 255]

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    dir_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 仅铁素体区域显示
    ferrite_mask = (mask == CLASS_FERRITE)
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    # 混合：铁素体区域用方向编码，非铁素体保持原图
    overlay = canvas.copy()
    overlay[ferrite_mask] = dir_bgr[ferrite_mask]
    canvas = cv2.addWeighted(overlay, 1 - alpha, canvas, alpha, 0)

    return canvas


def draw_collapsed_scatter(
    mask, vx_field, vy_field, image_size=1024, bg_color=(40, 40, 40),
    point_size=1,
):
    """
    面板 C：坍塌坐标散点图。

    所有铁素体像素的坍塌坐标 (x + Vx*1024, y + Vy*1024) 散点图。
    如果向量场训练良好，同一实例的像素坍塌到同一个紧凑簇。
    簇之间有明显间隙 = 实例可分；簇之间无间隙 = 实例不可分。

    Args:
        mask: [H, W] 二值分类
        vx_field: [H, W] 归一化 Vx
        vy_field: [H, W] 归一化 Vy
        image_size: 归一化因子
        bg_color: 背景色 (B, G, R)
        point_size: 散点大小

    Returns:
        canvas: [H, W, 3] BGR 图像
    """
    h, w = mask.shape[:2]
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    ferrite_mask = (mask == CLASS_FERRITE)
    ys, xs = np.where(ferrite_mask)

    if len(ys) == 0:
        # 写提示文字
        cv2.putText(canvas, "No ferrite pixels", (w // 4, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        return canvas

    # 反归一化坍塌坐标
    denorm_factor = float(max(mask.shape[0], mask.shape[1]))
    collapsed_x = xs.astype(np.float64) + vx_field[ys, xs] * denorm_factor
    collapsed_y = ys.astype(np.float64) + vy_field[ys, xs] * denorm_factor

    # 按方向编码颜色
    angles = np.arctan2(vy_field[ys, xs], vx_field[ys, xs])
    hues = ((angles + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

    # 绘制散点
    for i in range(len(ys)):
        cx = int(round(collapsed_x[i]))
        cy = int(round(collapsed_y[i]))
        # 限制在画布范围内
        if 0 <= cx < w and 0 <= cy < h:
            hue = int(hues[i])
            color_bgr = cv2.cvtColor(
                np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
            )[0, 0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
            if point_size <= 1:
                canvas[cy, cx] = color
            else:
                cv2.circle(canvas, (cx, cy), point_size, color, -1)

    return canvas


def draw_mask_visualization(image_rgb, mask, alpha=0.5):
    """
    面板 D：分类 mask 可视化。

    绿色 = 铁素体 (class=1)，红色 = 珠光体 (class=0)。
    叠加在半透明原图上。

    Args:
        image_rgb: [H, W, 3] 原始 RGB 图像
        mask: [H, W] 二值分类
        alpha: 原图透明度

    Returns:
        canvas: [H, W, 3] BGR 图像
    """
    h, w = mask.shape[:2]
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()

    overlay = canvas.copy()
    overlay[mask == CLASS_PEARLITE] = [0, 0, 200]    # 红色
    overlay[mask == CLASS_FERRITE] = [0, 200, 0]     # 绿色

    canvas = cv2.addWeighted(overlay, 1 - alpha, canvas, alpha, 0)
    return canvas


# ---------------------------------------------------------------------------
# 拼接面板
# ---------------------------------------------------------------------------

def add_label(canvas, text, font_scale=0.7, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """在画布左上角添加标签文字。"""
    h, w = canvas.shape[:2]
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    # 黑色背景条
    cv2.rectangle(canvas, (0, 0), (tw + 10, th + baseline + 5), bg_color, -1)
    cv2.putText(
        canvas, text, (5, th + 3),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2,
    )
    return canvas


def create_composite(
    panel_a, panel_b, panel_c, panel_d, labels=None, gap=4,
):
    """
    将四个面板拼成 2×2 布局。

    所有面板会裁剪到最小公共尺寸。

    Args:
        panel_a, panel_b, panel_c, panel_d: [H, W, 3] BGR 图像
        labels: 面板标签列表 ["A", "B", "C", "D"] 或自定义
        gap: 面板间距（像素）

    Returns:
        composite: [2H+gap, 2W+gap, 3] BGR 图像
    """
    if labels is None:
        labels = ["A: Vector Arrows", "B: HSV Direction",
                  "C: Collapsed Scatter", "D: Classification Mask"]

    h = min(p.shape[0] for p in [panel_a, panel_b, panel_c, panel_d])
    w = min(p.shape[1] for p in [panel_a, panel_b, panel_c, panel_d])

    panels = [panel_a, panel_b, panel_c, panel_d]
    for i, p in enumerate(panels):
        if p.shape[0] != h or p.shape[1] != w:
            panels[i] = cv2.resize(p, (w, h))
        add_label(panels[i], labels[i])

    composite = np.full(
        (h * 2 + gap, w * 2 + gap, 3), 0, dtype=np.uint8
    )
    composite[0:h, 0:w] = panels[0]
    composite[0:h, w + gap:2 * w + gap] = panels[1]
    composite[h + gap:2 * h + gap, 0:w] = panels[2]
    composite[h + gap:2 * h + gap, w + gap:2 * w + gap] = panels[3]

    return composite


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_single_image(
    model, image_path, device, output_dir,
    image_size=1024, threshold=0.5,
    grid_step=16, arrow_scale=1.0,
):
    """
    处理单张图像：推理 + 四面板可视化 + 保存。

    Args:
        model: 已加载权重的 SegmentationModel
        image_path: 输入图像路径
        device: torch.device
        output_dir: 输出目录
        image_size: Letterbox 目标尺寸
        threshold: 分类阈值
        grid_step: 箭头采样间距
        arrow_scale: 箭头长度缩放

    Returns:
        output_path: 保存的拼接图路径
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    logger.info(f"Processing: {basename}")

    image_rgb, mask, seg_prob, vx_field, vy_field = predict_vector_field(
        model, image_path, device, image_size=image_size, threshold=threshold,
    )

    h_orig, w_orig = image_rgb.shape[:2]
    logger.info(f"  Image size: {w_orig}x{h_orig}")
    ferrite_ratio = (mask == CLASS_FERRITE).sum() / (h_orig * w_orig) * 100
    logger.info(f"  Ferrite ratio: {ferrite_ratio:.1f}%")

    # 面板 A: 向量箭头
    panel_a = draw_vector_arrows(
        image_rgb, mask, vx_field, vy_field,
        image_size=image_size, grid_step=grid_step, arrow_scale=arrow_scale,
    )

    # 面板 B: HSV 方向编码
    panel_b = draw_hsv_direction(image_rgb, mask, vx_field, vy_field)

    # 面板 C: 坍塌散点
    panel_c = draw_collapsed_scatter(
        mask, vx_field, vy_field, image_size=image_size,
    )

    # 面板 D: 分类 mask
    panel_d = draw_mask_visualization(image_rgb, mask)

    # 拼接
    composite = create_composite(panel_a, panel_b, panel_c, panel_d)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{basename}_vecfield.png")
    cv2.imwrite(output_path, composite)
    logger.info(f"  Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Vector field regression visualization (original resolution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single image
  python test_vector_field.py --image data/test/sample.jpg

  # Batch directory
  python test_vector_field.py --image_dir data/test/ --output_dir outputs/vec_vis

  # Custom checkpoint + grid step
  python test_vector_field.py --image data/test/sample.jpg \\
      --checkpoint outputs/stage1/best_model.pth --grid_step 12

Panels:
  A: Original image + vector arrows (grid sampled, ferrite only)
  B: HSV direction-coded overlay
  C: Collapsed coordinate scatter plot
  D: Classification mask (green=ferrite, red=pearlite)
""",
    )
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained decoder checkpoint (overrides config)",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single test image",
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Directory of test images (batch mode)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/vec_vis",
        help="Output directory for visualization images",
    )
    parser.add_argument(
        "--grid_step", type=int, default=16,
        help="Grid sampling step for vector arrows (pixels, default: 16)",
    )
    parser.add_argument(
        "--arrow_scale", type=float, default=1.0,
        help="Arrow length scale factor (default: 1.0 = true scale)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Classification threshold (default: from config)",
    )
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("Either --image or --image_dir must be specified")

    config = load_config(args.config)
    sam2_cfg = config["sam2"]
    paths_cfg = config["paths"]
    infer_cfg = config["inference"]
    data_cfg = config["data"]

    device = sam2_cfg["device"]
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available, switching to CPU")
        device = "cpu"

    image_size = data_cfg["image_size"]
    threshold = args.threshold if args.threshold is not None else infer_cfg.get("threshold", 0.5)

    # 解析 checkpoint 路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
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

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Grid step: {args.grid_step}, Arrow scale: {args.arrow_scale}")

    model = build_model(config, device, checkpoint_path)

    # 收集图像列表
    if args.image:
        image_paths = [args.image]
    else:
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths.sort()

    if len(image_paths) == 0:
        logger.error("No images found!")
        return

    logger.info(f"Found {len(image_paths)} image(s)")

    for img_path in image_paths:
        try:
            process_single_image(
                model, img_path, device, args.output_dir,
                image_size=image_size, threshold=threshold,
                grid_step=args.grid_step, arrow_scale=args.arrow_scale,
            )
        except Exception as e:
            logger.error(f"  ERROR: {os.path.basename(img_path)}: {e}")

    logger.info(f"Done! {len(image_paths)} image(s) processed. Output: {args.output_dir}")


if __name__ == "__main__":
    main()