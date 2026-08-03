# -*- coding: utf-8 -*-
"""
骨架 + 受阻分水岭分割测试脚本（纯图像处理，不依赖模型）
==========================================================

流程：
  1. 预处理：灰度化 → CLAHE 对比度增强 → 中值滤波
  2. 边缘提取：Canny
  3. 断裂修复：形态学闭运算
  4. 骨架化：细化为 1px 线条
  5. 骨架带加粗：膨胀形成"晶界骨架带"（防线）
  6. 空间核心剥离：整图减去骨架带 → 独立小岛
  7. 独立种子标记：连通域编号
  8. 受阻分水岭缝合：种子蔓延 + 骨架带高阻 → 实例分割
  9. 面积过滤

可视化输出（四面板拼接）：
  A: 原图 + Canny 边缘叠加
  B: 骨架带（白=骨架带）
  C: 剥离后的独立晶核（彩色连通域）
  D: 最终分水岭实例图（彩色实例 + 黑色晶界线）

用法:
  python test_skeleton_watershed.py
  python test_skeleton_watershed.py --input data/smoketest --output_dir outputs/skeleton_watershed
  python test_skeleton_watershed.py --input data/smoketest --canny_low 30 --canny_high 100 --dilate_width 3
"""

import argparse
import glob
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: 预处理
# ---------------------------------------------------------------------------

def preprocess(image_bgr: np.ndarray, clahe_clip: float = 2.0,
               clahe_grid: int = 8, median_ksize: int = 5) -> np.ndarray:
    """灰度化 → CLAHE 对比度增强 → 中值滤波去噪。

    Args:
        image_bgr: [H, W, 3] BGR 图像
        clahe_clip: CLAHE 对比度限制
        clahe_grid: CLAHE 分块大小
        median_ksize: 中值滤波核大小（奇数）

    Returns:
        gray_enhanced: [H, W] uint8 增强灰度图
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE 对比度增强
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    gray_clahe = clahe.apply(gray)

    # 中值滤波去噪
    if median_ksize > 1:
        gray_enhanced = cv2.medianBlur(gray_clahe, median_ksize)
    else:
        gray_enhanced = gray_clahe

    return gray_enhanced


# ---------------------------------------------------------------------------
# Step 2: 边缘提取
# ---------------------------------------------------------------------------

def extract_edges(gray: np.ndarray, canny_low: int = 50, canny_high: int = 150,
                  use_auto: bool = True) -> np.ndarray:
    """Canny 边缘检测。

    Args:
        gray: [H, W] 灰度图
        canny_low: Canny 低阈值（手动模式）
        canny_high: Canny 高阈值（手动模式）
        use_auto: True=自动阈值（OTSU），False=手动阈值

    Returns:
        edges: [H, W] uint8 二值边缘图（255=边缘, 0=背景）
    """
    if use_auto:
        # 自动阈值：基于图像中值
        sigma = 0.33
        v = np.median(gray)
        low = int(max(0, (1.0 - sigma) * v))
        high = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, low, high)
        logger.info(f"  Auto Canny thresholds: low={low}, high={high}")
    else:
        edges = cv2.Canny(gray, canny_low, canny_high)

    return edges


# ---------------------------------------------------------------------------
# Step 3: 断裂修复（形态学闭运算）
# ---------------------------------------------------------------------------

def repair_breaks(edges: np.ndarray, close_kernel: int = 3) -> np.ndarray:
    """形态学闭运算连接断裂的边缘。

    Args:
        edges: [H, W] 二值边缘图
        close_kernel: 闭运算核大小（奇数）

    Returns:
        repaired: [H, W] 修复后的边缘图
    """
    if close_kernel < 3:
        return edges.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    # 闭运算：先膨胀后腐蚀，连接小断裂
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed


# ---------------------------------------------------------------------------
# Step 4: 骨架化（细化为 1px 线条）
# ---------------------------------------------------------------------------

def skeletonize(edges: np.ndarray) -> np.ndarray:
    """将边缘细化为 1px 宽骨架。

    优先使用 cv2.ximgproc.thinning，不可用时 fallback 到 skimage。

    Args:
        edges: [H, W] 二值边缘图

    Returns:
        skeleton: [H, W] uint8 骨架图（255=骨架, 0=背景）
    """
    # 尝试 cv2.ximgproc
    try:
        skeleton = cv2.ximgproc.thinning(
            edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        logger.info("  Skeletonization: cv2.ximgproc.thinning (Zhang-Suen)")
        return skeleton
    except (AttributeError, cv2.error):
        pass

    # Fallback: skimage.morphology.skeletonize
    try:
        from skimage.morphology import skeletonize as sk_skeletonize
        skeleton_bool = sk_skeletonize(edges > 0)
        skeleton = (skeleton_bool * 255).astype(np.uint8)
        logger.info("  Skeletonization: skimage.morphology.skeletonize")
        return skeleton
    except ImportError:
        logger.warning("  Neither cv2.ximgproc nor skimage available, using morphological thinning")
        # 最后 fallback：迭代腐蚀
        skeleton = edges.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(skeleton, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(skeleton, opened)
            skeleton = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skeleton


# ---------------------------------------------------------------------------
# Step 5: 骨架带加粗（膨胀形成防线）
# ---------------------------------------------------------------------------

def dilate_skeleton(skeleton: np.ndarray, dilate_width: int = 2) -> np.ndarray:
    """将 1px 骨架膨胀为指定宽度的"骨架带"。

    Args:
        skeleton: [H, W] 骨架图
        dilate_width: 骨架带宽度（像素），膨胀核大小 = 2*width+1

    Returns:
        belt: [H, W] 骨架带二值图（255=骨架带, 0=背景）
    """
    if dilate_width < 1:
        return skeleton.copy()

    kernel_size = 2 * dilate_width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    belt = cv2.dilate(skeleton, kernel)
    return belt


# ---------------------------------------------------------------------------
# Step 6: 空间核心剥离
# ---------------------------------------------------------------------------

def extract_cores(skeleton_belt: np.ndarray) -> np.ndarray:
    """整图减去骨架带，得到独立的晶核小岛。

    Args:
        skeleton_belt: [H, W] 骨架带二值图

    Returns:
        cores: [H, W] 晶核二值图（255=晶核, 0=骨架带/背景）
    """
    cores = cv2.bitwise_not(skeleton_belt)
    return cores


# ---------------------------------------------------------------------------
# Step 7: 独立种子标记
# ---------------------------------------------------------------------------

def label_cores(cores: np.ndarray, min_area: int = 30) -> tuple:
    """对独立晶核进行连通域标记。

    Args:
        cores: [H, W] 晶核二值图
        min_area: 最小种子面积（过滤噪声）

    Returns:
        markers: [H, W] int32 标记图（0=背景, 1..N=种子）
        num_valid: 有效种子数量
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)

    h, w = cores.shape
    markers = np.zeros((h, w), dtype=np.int32)
    valid_id = 0

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        valid_id += 1
        markers[labels == label_id] = valid_id

    logger.info(f"  Found {valid_id} valid seed regions (min_area={min_area})")
    return markers, valid_id


# ---------------------------------------------------------------------------
# Step 8: 受阻分水岭缝合
# ---------------------------------------------------------------------------

def watershed_stitching(
    markers: np.ndarray,
    skeleton_belt: np.ndarray,
    image_bgr: np.ndarray,
    min_area: int = 30,
) -> tuple:
    """以种子为发源地，骨架带为高阻地形，执行受阻分水岭。

    使用 cv2.watershed：以 BGR 图像为梯度地形，markers 为种子。
    骨架带区域在 markers 中标记为 -1（边界），阻止蔓延跨越。

    Args:
        markers: [H, W] int32 种子标记图
        skeleton_belt: [H, W] 骨架带二值图
        image_bgr: [H, W, 3] 原始 BGR 图像（提供梯度信息）
        min_area: 最小实例面积过滤

    Returns:
        labels: [H, W] int32 最终实例标记（0=背景, 1..N=实例）
        num_instances: 实例数量
    """
    h, w = markers.shape

    # cv2.watershed 需要 3 通道 uint8 图像
    # 使用原始图像作为梯度信息
    img_for_watershed = image_bgr.copy()

    # 准备 markers：cv2.watershed 要求 0=未知, >0=种子, -1=边界
    ws_markers = markers.copy()

    # 骨架带设为边界标记（-1 不可直接用，需用不同策略）
    # 策略：骨架带区域设为 0（未知但高阻），让分水岭自然在此处形成分水岭线
    # 但 cv2.watershed 用图像梯度而非外部地形作为地形，所以需增强骨架带处的梯度

    # 在骨架带位置叠加白色线条，增加图像梯度
    overlay = img_for_watershed.copy()
    overlay[skeleton_belt > 0] = [255, 255, 255]  # 骨架带变白
    img_for_watershed = cv2.addWeighted(img_for_watershed, 0.7, overlay, 0.3, 0)

    # 执行 watershed
    # cv2.watershed 输出：-1=边界, 0=未分配, >0=区域
    ws_result = cv2.watershed(img_for_watershed, ws_markers.copy())

    labels = ws_result.copy()
    labels[labels < 0] = 0  # 边界 → 0

    # 面积过滤：移除过小的实例
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    filtered = np.zeros_like(labels)
    new_id = 0
    for old_id in unique_labels:
        mask = labels == old_id
        area = int(mask.sum())
        if area < min_area:
            continue
        new_id += 1
        filtered[mask] = new_id

    logger.info(f"  Watershed: {new_id} instances after area filter (min_area={min_area})")
    return filtered, new_id


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def random_colormap(num_colors: int, seed: int = 42) -> np.ndarray:
    """生成随机颜色表。

    Args:
        num_colors: 颜色数量
        seed: 随机种子

    Returns:
        colors: [num_colors+1, 3] BGR 颜色表（索引 0=黑色=背景）
    """
    rng = np.random.RandomState(seed)
    colors = np.zeros((num_colors + 1, 3), dtype=np.uint8)
    for i in range(1, num_colors + 1):
        # HSV 随机色 → BGR
        hue = rng.randint(0, 180)
        sat = rng.randint(100, 255)
        val = rng.randint(100, 255)
        hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors[i] = bgr
    return colors


def render_labels(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """将标记图渲染为彩色实例图。

    Args:
        labels: [H, W] int32 标记图（0=背景, >0=实例）
        seed: 随机种子

    Returns:
        color_img: [H, W, 3] BGR 彩色图
    """
    max_label = int(labels.max())
    if max_label == 0:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)

    colors = random_colormap(max_label, seed=seed)
    h, w = labels.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id in range(1, max_label + 1):
        color_img[labels == label_id] = colors[label_id]
    return color_img


def render_cores(markers: np.ndarray, seed: int = 42) -> np.ndarray:
    """将种子标记渲染为彩色图。

    Args:
        markers: [H, W] int32 种子标记
        seed: 随机种子

    Returns:
        color_img: [H, W, 3] BGR 彩色图
    """
    return render_labels(markers, seed=seed)


def overlay_edges(image_bgr: np.ndarray, edges: np.ndarray,
                  color: tuple = (0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    """在原图上叠加边缘（红色）。

    Args:
        image_bgr: [H, W, 3] 原图 BGR
        edges: [H, W] 边缘二值图
        color: 边缘颜色 BGR
        alpha: 原图透明度

    Returns:
        overlay: [H, W, 3] BGR 叠加图
    """
    canvas = image_bgr.copy()
    overlay = canvas.copy()
    overlay[edges > 0] = color
    canvas = cv2.addWeighted(overlay, 1 - alpha, canvas, alpha, 0)
    return canvas


def render_watershed_result(labels: np.ndarray, skeleton_belt: np.ndarray,
                            seed: int = 42) -> np.ndarray:
    """渲染分水岭最终结果：彩色实例 + 黑色骨架带边界线。

    Args:
        labels: [H, W] int32 实例标记
        skeleton_belt: [H, W] 骨架带二值图
        seed: 随机种子

    Returns:
        result: [H, W, 3] BGR 彩色图
    """
    color_img = render_labels(labels, seed=seed)
    # 在骨架带位置画黑色边界线
    color_img[skeleton_belt > 0] = [0, 0, 0]
    return color_img


def add_label(canvas: np.ndarray, text: str, font_scale: float = 0.7,
              color: tuple = (255, 255, 255), bg_color: tuple = (0, 0, 0)) -> np.ndarray:
    """在画布左上角添加标签文字。"""
    h, w = canvas.shape[:2]
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
    )
    cv2.rectangle(canvas, (0, 0), (tw + 10, th + baseline + 5), bg_color, -1)
    cv2.putText(
        canvas, text, (5, th + 3),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2,
    )
    return canvas


def create_composite(panels: list, labels: list, gap: int = 4) -> np.ndarray:
    """将多个面板拼成 2×2 布局。

    Args:
        panels: 面板列表（4 个 [H, W, 3] BGR 图像）
        labels: 面板标签列表
        gap: 面板间距

    Returns:
        composite: 拼接图
    """
    h = min(p.shape[0] for p in panels)
    w = min(p.shape[1] for p in panels)

    resized = []
    for i, p in enumerate(panels):
        if p.shape[0] != h or p.shape[1] != w:
            p = cv2.resize(p, (w, h))
        add_label(p, labels[i])
        resized.append(p)

    composite = np.full((h * 2 + gap, w * 2 + gap, 3), 0, dtype=np.uint8)
    composite[0:h, 0:w] = resized[0]
    composite[0:h, w + gap:2 * w + gap] = resized[1]
    composite[h + gap:2 * h + gap, 0:w] = resized[2]
    composite[h + gap:2 * h + gap, w + gap:2 * w + gap] = resized[3]

    return composite


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_single_image(
    image_path: str,
    output_dir: str,
    canny_low: int = 50,
    canny_high: int = 150,
    use_auto_canny: bool = True,
    close_kernel: int = 3,
    dilate_width: int = 2,
    min_area: int = 30,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    median_ksize: int = 5,
) -> str:
    """处理单张图像：完整骨架+分水岭流水线 + 四面板可视化。

    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        canny_low: Canny 低阈值
        canny_high: Canny 高阈值
        use_auto_canny: 是否使用自动 Canny 阈值
        close_kernel: 闭运算核大小
        dilate_width: 骨架带宽度
        min_area: 最小实例面积
        clahe_clip: CLAHE 对比度限制
        clahe_grid: CLAHE 分块大小
        median_ksize: 中值滤波核大小

    Returns:
        output_path: 保存的拼接图路径
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    logger.info(f"Processing: {basename}")

    # 加载图像
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = image_bgr.shape[:2]
    logger.info(f"  Image size: {w}x{h}")

    # Step 1: 预处理
    logger.info("Step 1: Preprocessing (CLAHE + median filter)")
    gray = preprocess(image_bgr, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
                      median_ksize=median_ksize)

    # Step 2: 边缘提取
    logger.info("Step 2: Canny edge detection")
    edges = extract_edges(gray, canny_low=canny_low, canny_high=canny_high,
                          use_auto=use_auto_canny)
    logger.info(f"  Edge pixels: {cv2.countNonZero(edges)}")

    # Step 3: 断裂修复
    logger.info(f"Step 3: Morphological closing (kernel={close_kernel})")
    repaired = repair_breaks(edges, close_kernel=close_kernel)

    # Step 4: 骨架化
    logger.info("Step 4: Skeletonization")
    skeleton = skeletonize(repaired)
    logger.info(f"  Skeleton pixels: {cv2.countNonZero(skeleton)}")

    # Step 5: 骨架带加粗
    logger.info(f"Step 5: Dilation (width={dilate_width}px)")
    skeleton_belt = dilate_skeleton(skeleton, dilate_width=dilate_width)
    logger.info(f"  Belt pixels: {cv2.countNonZero(skeleton_belt)}")

    # Step 6: 空间核心剥离
    logger.info("Step 6: Core extraction (subtract skeleton belt)")
    cores = extract_cores(skeleton_belt)

    # Step 7: 独立种子标记
    logger.info("Step 7: Connected component labeling")
    markers, num_seeds = label_cores(cores, min_area=min_area)
    if num_seeds == 0:
        logger.warning("  No valid seeds found! Try adjusting parameters.")
        markers = np.ones_like(markers, dtype=np.int32)

    # Step 8: 受阻分水岭
    logger.info("Step 8: Watershed stitching")
    labels, num_instances = watershed_stitching(
        markers, skeleton_belt, image_bgr, min_area=min_area
    )

    # Step 9: 面积过滤已在 watershed_stitching 中完成
    logger.info(f"Result: {num_instances} instances")

    # --- 可视化 ---
    # Panel A: 原图 + Canny 边缘叠加
    panel_a = overlay_edges(image_bgr, edges, color=(0, 0, 255), alpha=0.5)

    # Panel B: 骨架带
    panel_b = np.zeros((h, w, 3), dtype=np.uint8)
    panel_b[skeleton_belt > 0] = [255, 255, 255]

    # Panel C: 独立晶核
    panel_c = render_cores(markers, seed=42)

    # Panel D: 分水岭最终结果
    panel_d = render_watershed_result(labels, skeleton_belt, seed=42)

    # 拼接
    panels = [panel_a, panel_b, panel_c, panel_d]
    panel_labels = [
        f"A: Original + Canny Edges",
        f"B: Skeleton Belt (width={dilate_width}px)",
        f"C: Seed Cores ({num_seeds} regions)",
        f"D: Watershed Result ({num_instances} instances)",
    ]
    composite = create_composite(panels, panel_labels)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{basename}_skeleton_watershed.png")
    cv2.imwrite(output_path, composite)
    logger.info(f"  Saved: {output_path}")

    # 额外保存：单独的实例图
    inst_path = os.path.join(output_dir, f"{basename}_sw_inst.png")
    cv2.imwrite(inst_path, panel_d)
    logger.info(f"  Saved instance map: {inst_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Skeleton + Watershed segmentation test (pure image processing, no model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Default: process data/smoketest
  python test_skeleton_watershed.py

  # Custom input/output
  python test_skeleton_watershed.py --input data/smoketest --output_dir outputs/skeleton_watershed

  # Manual Canny thresholds + wider belt
  python test_skeleton_watershed.py --canny_low 30 --canny_high 100 --dilate_width 3

  # Auto Canny (default)
  python test_skeleton_watershed.py --auto_canny

Panels:
  A: Original + Canny edges overlay
  B: Skeleton belt (white = belt, black = background)
  C: Seed cores (colored connected components)
  D: Watershed result (colored instances + black boundary)
""",
    )
    parser.add_argument(
        "--input", type=str, default="data/smoketest",
        help="Input image or directory (default: data/smoketest)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/skeleton_watershed",
        help="Output directory (default: outputs/skeleton_watershed)",
    )
    parser.add_argument(
        "--canny_low", type=int, default=50,
        help="Canny low threshold (default: 50, used when --no_auto_canny)",
    )
    parser.add_argument(
        "--canny_high", type=int, default=150,
        help="Canny high threshold (default: 150, used when --no_auto_canny)",
    )
    parser.add_argument(
        "--auto_canny", action="store_true", default=True,
        help="Use auto Canny thresholds based on image median (default: True)",
    )
    parser.add_argument(
        "--no_auto_canny", dest="auto_canny", action="store_false",
        help="Use manual Canny thresholds (--canny_low, --canny_high)",
    )
    parser.add_argument(
        "--close_kernel", type=int, default=3,
        help="Morphological closing kernel size (default: 3)",
    )
    parser.add_argument(
        "--dilate_width", type=int, default=2,
        help="Skeleton belt dilation width in pixels (default: 2)",
    )
    parser.add_argument(
        "--min_area", type=int, default=30,
        help="Minimum instance area in pixels (default: 30)",
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
        help="Median filter kernel size (default: 5, set 1 to disable)",
    )
    args = parser.parse_args()

    # 收集图像列表
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        image_paths.sort()

    if len(image_paths) == 0:
        logger.error(f"No images found in: {args.input}")
        return

    logger.info(f"Found {len(image_paths)} image(s)")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Parameters: auto_canny={args.auto_canny}, "
                f"canny=[{args.canny_low},{args.canny_high}], "
                f"close_kernel={args.close_kernel}, "
                f"dilate_width={args.dilate_width}, min_area={args.min_area}")

    for img_path in image_paths:
        try:
            process_single_image(
                img_path,
                output_dir=args.output_dir,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                use_auto_canny=args.auto_canny,
                close_kernel=args.close_kernel,
                dilate_width=args.dilate_width,
                min_area=args.min_area,
                clahe_clip=args.clahe_clip,
                clahe_grid=args.clahe_grid,
                median_ksize=args.median_ksize,
            )
        except Exception as e:
            logger.error(f"  ERROR processing {os.path.basename(img_path)}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"Done! {len(image_paths)} image(s) processed. Output: {args.output_dir}")


if __name__ == "__main__":
    main()