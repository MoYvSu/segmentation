# -*- coding: utf-8 -*-
"""
数据管线诊断脚本（边界预测版）
============================
验证当前数据管线的三个关键环节：
1. Letterbox 填充比例与可视化
2. 语义/边界掩码 + EDT 边界权重图生成（与 BoundaryDataset 同路径）
3. 边界骨架化 + 受阻分水岭实例分割（合成数据）

用法:
    D:\\Anaconda\\envs\\sam2_env\\python.exe debug_pipeline.py
"""

import os
import sys
import glob

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import (
    letterbox,
    letterbox_mask,
    parse_labelme_json,
    create_binary_mask,
    compute_boundary_weight,
)
from utils.post_process import boundary_watershed_separation

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "debug_pipeline")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_letterbox_padding():
    """Test 1: Letterbox padding ratio visualization"""
    print("\n" + "=" * 60)
    print("Test 1: Letterbox padding ratio")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))[:3]

    for json_path in json_files:
        img_path = json_path.replace(".json", ".jpg")
        if not os.path.exists(img_path):
            continue

        basename = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        image_lb, scale, pad_h, pad_w = letterbox(image_rgb, 1024)

        vis = image_lb.copy()
        if pad_h > 0:
            vis[1024 - pad_h:, :] = [255, 0, 0]
        if pad_w > 0:
            vis[:, 1024 - pad_w:] = [255, 0, 0]

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(OUTPUT_DIR, basename + "_letterbox_vis.png")
        cv2.imwrite(out_path, vis_bgr)

        content_ratio = (1024 - pad_h) * (1024 - pad_w) / (1024 * 1024) * 100
        print("  %s: %dx%d -> 1024x1024, pad_h=%d (%.1f%%), pad_w=%d, content=%.1f%%" % (
            basename, h, w, pad_h, pad_h / 1024 * 100, pad_w, content_ratio))


def test_mask_and_boundary_weight():
    """Test 2: 语义掩码 + 净化边界 + EDT 权重（与 BoundaryDataset 同路径）"""
    print("\n" + "=" * 60)
    print("Test 2: Semantic mask + boundary weight")
    print("=" * 60)

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    gt_dir = os.path.join(PROJECT_ROOT, "data", "purified_gt")
    json_files = sorted(glob.glob(os.path.join(raw_dir, "*.json")))[:3]

    for json_path in json_files:
        img_path = json_path.replace(".json", ".jpg")
        basename = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(gt_dir, basename + "_gt.npz")
        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        masks = parse_labelme_json(json_path, h, w)
        semantic = create_binary_mask(masks["ferrite"], masks["pearlite"])
        gt_data = np.load(gt_path)
        boundary = gt_data["boundary"]

        image_lb, _, _, _ = letterbox(image_rgb, 1024)
        semantic_lb, _, _, _ = letterbox_mask(semantic, 1024)
        boundary_lb, _, _, _ = letterbox_mask(boundary, 1024)
        weight_lb = compute_boundary_weight(
            boundary_lb, scale_factor=10.0, weight_floor=1.0, weight_ceil=4.0
        )

        ferrite_ratio = semantic_lb.sum() / semantic_lb.size * 100
        boundary_ratio = boundary_lb.sum() / boundary_lb.size * 100
        print("  %s: ferrite=%.1f%%, boundary=%.2f%%, weight_range=[%.2f, %.2f]" % (
            basename, ferrite_ratio, boundary_ratio,
            weight_lb.min(), weight_lb.max()))

        mask_vis = np.zeros((1024, 1024, 3), dtype=np.uint8)
        mask_vis[semantic_lb == 1] = [0, 128, 0]
        mask_vis[semantic_lb == 0] = [0, 0, 128]
        cv2.imwrite(os.path.join(OUTPUT_DIR, basename + "_mask.png"),
                    cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

        weight_vis = (weight_lb / weight_lb.max() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, basename + "_boundary_weight.png"),
                    cv2.applyColorMap(weight_vis, cv2.COLORMAP_JET))


def test_boundary_watershed():
    """Test 3: 边界受阻分水岭（合成数据：两个被边界环封闭的晶粒）

    注意：背景区域（非语义区域）本身也是一个 core 连通域，会作为种子参与
    分水岭并成为一个珠光体实例，因此只断言分离出 2 个铁素体实例。
    """
    print("\n" + "=" * 60)
    print("Test 3: Boundary watershed separation (synthetic)")
    print("=" * 60)

    h, w = 512, 512
    semantic = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(semantic, (140, 256), 90, 1, -1)
    cv2.circle(semantic, (380, 256), 90, 1, -1)

    boundary = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(boundary, (140, 256), 90, 1, thickness=6)
    cv2.circle(boundary, (380, 256), 90, 1, thickness=6)

    inst_map, class_map = boundary_watershed_separation(
        semantic, boundary, dilate_width=1, min_area=100,
    )

    n_ferrite = sum(1 for v in class_map.values() if v == 1)
    n_pearlite = sum(1 for v in class_map.values() if v == 0)
    print("  instances: %d (ferrite=%d, pearlite=%d), expected ferrite=2" % (
        len(class_map), n_ferrite, n_pearlite))

    vis = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [(0, 128, 0), (128, 0, 0), (0, 0, 128), (128, 128, 0)]
    for i in range(1, len(class_map) + 1):
        vis[inst_map == i] = colors[i % len(colors)]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_inst.png"), vis)

    sem_vis = np.zeros((h, w, 3), dtype=np.uint8)
    sem_vis[semantic == 1] = [0, 128, 0]
    sem_vis[semantic == 0] = [0, 0, 128]
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_semantic.png"),
                cv2.cvtColor(sem_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_boundary.png"), boundary * 255)


def main():
    print("=" * 60)
    print("Pipeline diagnostic (boundary version)")
    print("Output dir: " + OUTPUT_DIR)
    print("=" * 60)

    test_letterbox_padding()
    test_mask_and_boundary_weight()
    test_boundary_watershed()

    print("\n" + "=" * 60)
    print("Diagnostic complete! Results saved to:")
    print("  " + OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
