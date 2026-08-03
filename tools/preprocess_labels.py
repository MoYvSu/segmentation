# -*- coding: utf-8 -*-
"""
离线边界净化与数据集真值重构
============================
从 data/raw/ 中已标注的实例掩码（Labelme JSON）提取纯净、无划痕的晶界目标。

流程（单张图像）：
  1. 全图边缘获取：CLAHE 预处理 → Canny 边缘检测 → E_total
  2. 内部约束剪裁：所有晶粒实例（铁素体+珠光体）多边形并集 → M_internal
     → 形态学腐蚀 r 像素 → M_eroded（收缩内部区域，确保真实边界不落入违禁区）
  3. 位运算交集净化：GB_pure = E_total ∧ ¬M_eroded
  4. 目标膨胀：GB_pure 膨胀 2-3px → GB_belt（带状目标，缓解类别不平衡）

输出：
  data/purified_gt/{image_basename}_gt.npz
    - semantic: [H, W] uint8 二值掩码（0=珠光体, 1=铁素体）
    - boundary: [H, W] uint8 二值掩码（0=非边界, 1=边界带）

用法:
  python tools/preprocess_labels.py
  python tools/preprocess_labels.py --raw_dir data/raw --output_dir data/purified_gt
  python tools/preprocess_labels.py --erode_radius 3 --dilate_width 2 --canny_low 30 --canny_high 100
  python tools/preprocess_labels.py --visualize
"""

import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import parse_labelme_json, create_binary_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def preprocess_gray(image_bgr: np.ndarray, clahe_clip: float = 2.0,
                    clahe_grid: int = 8, median_ksize: int = 5) -> np.ndarray:
    """灰度化 → CLAHE 对比度增强 → 中值滤波去噪。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    gray_clahe = clahe.apply(gray)
    if median_ksize > 1:
        gray_enhanced = cv2.medianBlur(gray_clahe, median_ksize)
    else:
        gray_enhanced = gray_clahe
    return gray_enhanced


def extract_canny_edges(gray: np.ndarray, canny_low: int = 30,
                        canny_high: int = 100) -> np.ndarray:
    """Canny 边缘检测。"""
    return cv2.Canny(gray, canny_low, canny_high)


def build_internal_mask(json_path: str, height: int, width: int) -> np.ndarray:
    """
    构建所有晶粒实例（铁素体+珠光体）的并集掩码 M_internal。

    Args:
        json_path: Labelme JSON 文件路径
        height: 图像高度
        width: 图像宽度

    Returns:
        M_internal: [H, W] uint8 二值掩码（1=晶粒内部, 0=背景/边界）
    """
    masks = parse_labelme_json(json_path, height, width)
    ferrite_mask = masks["ferrite"]
    pearlite_mask = masks["pearlite"]

    # 铁素体 + 珠光体并集 = 全图所有晶粒区域
    M_internal = np.maximum(ferrite_mask, pearlite_mask)
    return M_internal


def purify_boundary(edges: np.ndarray, M_internal: np.ndarray,
                    erode_radius: int = 2) -> np.ndarray:
    """
    位运算交集净化：GB_pure = E_total ∧ ¬(M_internal ⊖ r)

    1. 对 M_internal 腐蚀 r 像素 → M_eroded
    2. ¬M_eroded → 允许保留边缘的区域（晶界 + 背景外）
    3. E_total ∧ ¬M_eroded → 净化后的边界

    Args:
        edges: [H, W] Canny 边缘图（255=边缘）
        M_internal: [H, W] 晶粒内部并集掩码（1=内部）
        erode_radius: 腐蚀半径（像素）

    Returns:
        GB_pure: [H, W] uint8 净化边界（255=边界, 0=非边界）
    """
    if erode_radius < 1:
        # 不腐蚀，直接取反
        M_eroded = M_internal
    else:
        kernel_size = 2 * erode_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        M_eroded = cv2.erode(M_internal, kernel)

    # ¬M_eroded：晶粒内部腐蚀后，边缘区域变为允许保留的区域
    # M_eroded 是 0/1 二值掩码，取反后：0=内部（禁止），1=边界区（允许）
    not_eroded = (M_eroded == 0).astype(np.uint8)

    # 按位与：仅保留腐蚀后露出的边缘
    GB_pure = cv2.bitwise_and(edges, edges, mask=not_eroded)
    return GB_pure


def dilate_boundary(GB_pure: np.ndarray, dilate_width: int = 2) -> np.ndarray:
    """对净化后的边界进行膨胀，形成带状目标。"""
    if dilate_width < 1:
        return GB_pure.copy()
    kernel_size = 2 * dilate_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    GB_belt = cv2.dilate(GB_pure, kernel)
    return GB_belt


def process_single_image(
    image_path: str,
    json_path: str,
    output_dir: str,
    erode_radius: int = 2,
    dilate_width: int = 2,
    canny_low: int = 30,
    canny_high: int = 100,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    median_ksize: int = 5,
    visualize: bool = False,
) -> str:
    """处理单张图像：边界净化 + 保存 .npz。

    Args:
        image_path: 输入图像路径
        json_path: 对应的 JSON 标注路径
        output_dir: 输出目录
        erode_radius: 内部掩码腐蚀半径
        dilate_width: 边界膨胀宽度
        canny_low: Canny 低阈值
        canny_high: Canny 高阈值
        clahe_clip: CLAHE 对比度限制
        clahe_grid: CLAHE 分块大小
        median_ksize: 中值滤波核大小
        visualize: 是否生成可视化对比图

    Returns:
        npz_path: 保存的 .npz 文件路径
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]

    # 加载图像
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = image_bgr.shape[:2]

    # Step 1: 全图边缘获取
    gray = preprocess_gray(image_bgr, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
                           median_ksize=median_ksize)
    edges = extract_canny_edges(gray, canny_low=canny_low, canny_high=canny_high)

    # Step 2: 内部约束掩码
    M_internal = build_internal_mask(json_path, h, w)

    # 语义掩码（铁素体=1，珠光体=0）
    masks = parse_labelme_json(json_path, h, w)
    semantic = create_binary_mask(masks["ferrite"], masks["pearlite"])

    # Step 3: 边界净化
    GB_pure = purify_boundary(edges, M_internal, erode_radius=erode_radius)

    # Step 4: 边界膨胀
    GB_belt = dilate_boundary(GB_pure, dilate_width=dilate_width)

    # 二值化（0/1）
    boundary = (GB_belt > 0).astype(np.uint8)

    # 保存 .npz
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{basename}_gt.npz")
    np.savez_compressed(
        npz_path,
        semantic=semantic,
        boundary=boundary,
    )

    edge_pixels = int((edges > 0).sum())
    pure_pixels = int((GB_pure > 0).sum())
    belt_pixels = int(boundary.sum())
    logger.info(
        f"  {basename}: edges={edge_pixels}, purified={pure_pixels}, "
        f"belt={belt_pixels} ({belt_pixels / (h * w) * 100:.2f}%)"
    )

    # 可视化
    if visualize:
        vis_dir = os.path.join(output_dir, "visualize")
        os.makedirs(vis_dir, exist_ok=True)

        # 四面板拼接
        panel_a = image_bgr.copy()
        panel_a[edges > 0] = [0, 0, 255]  # 红色边缘叠加

        panel_b = np.zeros((h, w, 3), dtype=np.uint8)
        panel_b[M_internal > 0] = [0, 200, 0]  # 绿色=内部
        panel_b[M_internal == 0] = [0, 0, 0]  # 黑色=边界区

        panel_c = np.zeros((h, w, 3), dtype=np.uint8)
        panel_c[GB_pure > 0] = [255, 255, 255]  # 白色=净化边界

        panel_d = np.zeros((h, w, 3), dtype=np.uint8)
        panel_d[semantic > 0] = [0, 200, 0]  # 绿色=铁素体
        panel_d[semantic == 0] = [0, 0, 200]  # 红色=珠光体
        panel_d[boundary > 0] = [255, 255, 255]  # 白色=边界带

        # 标签
        for panel, label in [
            (panel_a, "A: Original + Canny"),
            (panel_b, "B: M_internal (green)"),
            (panel_c, "C: GB_pure (white)"),
            (panel_d, "D: Semantic + Boundary"),
        ]:
            cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 缩放到合理尺寸拼接
        scale = 1024 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        panels = [cv2.resize(p, (new_w, new_h)) for p in [panel_a, panel_b, panel_c, panel_d]]
        gap = 4
        composite = np.full(
            (new_h * 2 + gap, new_w * 2 + gap, 3), 0, dtype=np.uint8
        )
        composite[0:new_h, 0:new_w] = panels[0]
        composite[0:new_h, new_w + gap:] = panels[1]
        composite[new_h + gap:, 0:new_w] = panels[2]
        composite[new_h + gap:, new_w + gap:] = panels[3]

        vis_path = os.path.join(vis_dir, f"{basename}_purify.png")
        cv2.imwrite(vis_path, composite)
        logger.info(f"  Visualization saved: {vis_path}")

    return npz_path


def main():
    parser = argparse.ArgumentParser(
        description="Offline boundary purification & dataset GT reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Default: process data/raw -> data/purified_gt
  python tools/preprocess_labels.py

  # Custom dirs + visualization
  python tools/preprocess_labels.py --visualize

  # Custom parameters
  python tools/preprocess_labels.py --erode_radius 3 --dilate_width 2 --canny_low 30 --canny_high 100
""",
    )
    parser.add_argument(
        "--raw_dir", type=str, default="data/raw",
        help="Raw data directory with .jpg + .json pairs (default: data/raw)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/purified_gt",
        help="Output directory for .npz files (default: data/purified_gt)",
    )
    parser.add_argument(
        "--erode_radius", type=int, default=2,
        help="Erosion radius for internal mask (default: 2)",
    )
    parser.add_argument(
        "--dilate_width", type=int, default=2,
        help="Boundary dilation width (default: 2)",
    )
    parser.add_argument(
        "--canny_low", type=int, default=30,
        help="Canny low threshold (default: 30)",
    )
    parser.add_argument(
        "--canny_high", type=int, default=100,
        help="Canny high threshold (default: 100)",
    )
    parser.add_argument(
        "--clahe_clip", type=float, default=2.0,
        help="CLAHE clip limit (default: 2.0)",
    )
    parser.add_argument(
        "--clahe_grid", type=int, default=8,
        help="CLAHE grid size (default: 8)",
    )
    parser.add_argument(
        "--median_ksize", type=int, default=5,
        help="Median filter kernel size (default: 5)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate 4-panel visualization for each image",
    )
    args = parser.parse_args()

    # 收集所有 .jpg + .json 对
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(args.raw_dir, ext)))
    image_paths.sort()

    if len(image_paths) == 0:
        logger.error(f"No images found in: {args.raw_dir}")
        return

    logger.info(f"Found {len(image_paths)} images in {args.raw_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(
        f"Parameters: erode_radius={args.erode_radius}, "
        f"dilate_width={args.dilate_width}, "
        f"canny=[{args.canny_low},{args.canny_high}]"
    )

    success_count = 0
    for img_path in image_paths:
        json_path = os.path.splitext(img_path)[0] + ".json"
        if not os.path.exists(json_path):
            logger.warning(f"  Skipping {os.path.basename(img_path)}: no JSON found")
            continue

        basename = os.path.splitext(os.path.basename(img_path))[0]
        logger.info(f"Processing: {basename}")

        try:
            npz_path = process_single_image(
                img_path,
                json_path,
                output_dir=args.output_dir,
                erode_radius=args.erode_radius,
                dilate_width=args.dilate_width,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                clahe_clip=args.clahe_clip,
                clahe_grid=args.clahe_grid,
                median_ksize=args.median_ksize,
                visualize=args.visualize,
            )
            success_count += 1
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    logger.info(
        f"Done! {success_count}/{len(image_paths)} images processed. "
        f"Output: {args.output_dir}"
    )


if __name__ == "__main__":
    main()