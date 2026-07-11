# -*- coding: utf-8 -*-
"""
Pipeline diagnostic script
============================
Verifies:
1. Letterbox padding ratio
2. Binary mask + distance field generation
3. Distance field compensation (inverse transform vs linear)
4. Tiled inference positions (Case A / Case B)
5. Gaussian weight blending
6. Watershed separation

Usage:
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
    letterbox_distance_field,
    parse_labelme_json,
    create_binary_mask,
    create_distance_field,
)
from utils.post_process import (
    compensate_distance_field,
    _get_dynamic_kernel_size,
    watershed_separation,
    gaussian_weight_map,
)

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

    print("\n  --- Test images ---")
    test_dir = os.path.join(PROJECT_ROOT, "data", "smoketest")
    for img_path in sorted(glob.glob(os.path.join(test_dir, "*.jpg"))):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        image_lb, scale, pad_h, pad_w = letterbox(image_rgb, 1024)
        content_ratio = (1024 - pad_h) * (1024 - pad_w) / (1024 * 1024) * 100
        print("  %s: %dx%d -> 1024x1024, pad_h=%d (%.1f%%), pad_w=%d, content=%.1f%%" % (
            basename, h, w, pad_h, pad_h / 1024 * 100, pad_w, content_ratio))


def test_mask_generation():
    """Test 2: Binary mask and distance field generation"""
    print("\n" + "=" * 60)
    print("Test 2: Binary mask and distance field generation")
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

        masks = parse_labelme_json(json_path, h, w)
        ferrite_mask = masks["ferrite"]
        pearlite_mask = masks["pearlite"]

        binary_mask = create_binary_mask(ferrite_mask, pearlite_mask)
        dist_field = create_distance_field(ferrite_mask, pearlite_mask, scale_factor=10.0)

        ferrite_ratio = binary_mask.sum() / binary_mask.size * 100
        pearlite_ratio = pearlite_mask.sum() / pearlite_mask.size * 100
        dist_max = dist_field.max()
        dist_mean = dist_field[ferrite_mask > 0].mean() if ferrite_mask.sum() > 0 else 0

        print("  %s: ferrite=%.1f%%, pearlite=%.1f%%, dist_max=%.1f, dist_mean=%.1f" % (
            basename, ferrite_ratio, pearlite_ratio, dist_max, dist_mean))

        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        mask_vis[binary_mask == 1] = [0, 128, 0]
        mask_vis[binary_mask == 0] = [0, 0, 128]
        cv2.imwrite(os.path.join(OUTPUT_DIR, basename + "_mask.png"),
                    cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

        dist_vis = (dist_field / max(dist_max, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, basename + "_dist.png"),
                    cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET))


def test_distance_compensation():
    """Test 3: Distance field compensation numerical verification"""
    print("\n" + "=" * 60)
    print("Test 3: Distance field compensation")
    print("=" * 60)

    scale_factor = 10.0
    spatial_scale = 2584.0 / 1024.0

    print("  spatial_scale: %.4f" % spatial_scale)
    print("  scale_factor: %.1f" % scale_factor)
    print()
    print("  %10s | %10s | %14s | %16s | %14s" % (
        "raw_dist", "normalized", "linear(wrong)", "inverse(correct)", "corrected_raw"))
    print("  %s" % ("-" * 70))

    for dist_raw in [5, 10, 20, 50, 100, 200, 500]:
        dist_norm = dist_raw / (dist_raw + scale_factor)
        linear_compensated = min(dist_norm * spatial_scale, 1.0)

        dist_compensated = compensate_distance_field(
            np.array([[dist_norm]], dtype=np.float32),
            spatial_scale=spatial_scale,
            scale_factor=scale_factor,
        )[0, 0]

        dist_raw_corrected = dist_compensated * scale_factor / (1 - dist_compensated + 1e-7)

        print("  %10.1f | %10.4f | %14.4f | %16.4f | %14.1f" % (
            dist_raw, dist_norm, linear_compensated, dist_compensated, dist_raw_corrected))

    print()
    print("  Conclusion: linear compensation causes [0,1] overflow and distortion;")
    print("  inverse transform correctly recovers distance values at different scales.")


def test_tiled_positions():
    """Test 4: Tiled inference position calculation"""
    print("\n" + "=" * 60)
    print("Test 4: Tiled inference positions")
    print("=" * 60)

    tile_size = 1024
    stride = 512

    test_cases = [
        ("test_002 (1024x1224)", 1024, 1224),
        ("test_001 (2048x2448)", 2048, 2448),
    ]

    for name, h, w in test_cases:
        max_dim = max(h, w)
        case = "A" if max_dim <= 1224 else "B"
        print("\n  %s: Case %s" % (name, case))

        if case == "A":
            _, _, pad_h, pad_w = letterbox(np.zeros((h, w, 3)), 1024)
            print("    Letterbox: -> 1024x1024, pad_h=%d, pad_w=%d" % (pad_h, pad_w))
        else:
            positions_h = list(range(0, max(1, h - tile_size + 1), stride))
            if positions_h[-1] + tile_size < h:
                positions_h.append(h - tile_size)
            positions_w = list(range(0, max(1, w - tile_size + 1), stride))
            if positions_w[-1] + tile_size < w:
                positions_w.append(w - tile_size)

            total = len(positions_h) * len(positions_w)
            print("    H positions: %s" % positions_h)
            print("    W positions: %s" % positions_w)
            print("    total patches: %dx%d = %d" % (len(positions_h), len(positions_w), total))


def test_gaussian_weight():
    """Test 5: Gaussian weight blending visualization"""
    print("\n" + "=" * 60)
    print("Test 5: Gaussian weight blending")
    print("=" * 60)

    weight_map = gaussian_weight_map(1024, sigma_scale=0.25)

    weight_vis = (weight_map * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "gaussian_weight.png"),
                cv2.applyColorMap(weight_vis, cv2.COLORMAP_JET))

    print("  weight map shape: %s" % str(weight_map.shape))
    print("  center value: %.4f" % weight_map[512, 512])
    print("  corner value: %.6f" % weight_map[0, 0])
    print("  quarter value: %.4f" % weight_map[256, 256])

    print("\n  Overlap region weight verification (50%% overlap):")
    stride = 512
    for pos in [stride, (stride + 1024) // 2, 1024 - 1]:
        w1 = weight_map[pos, 512]
        w2 = weight_map[pos - stride, 512]
        print("    pos %d: w1=%.4f, w2=%.4f, sum=%.4f" % (pos, w1, w2, w1 + w2))


def test_watershed():
    """Test 6: Watershed separation verification (synthetic data)"""
    print("\n" + "=" * 60)
    print("Test 6: Watershed separation")
    print("=" * 60)

    h, w = 512, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (180, 256), 100, 1, -1)
    cv2.circle(mask, (340, 256), 100, 1, -1)

    yy, xx = np.meshgrid(np.arange(w), np.arange(h))
    dist_field = np.maximum(
        np.exp(-((xx - 180) ** 2 + (yy - 256) ** 2) / (2 * 50 ** 2)),
        np.exp(-((xx - 340) ** 2 + (yy - 256) ** 2) / (2 * 50 ** 2)),
    ).astype(np.float32)

    num_cc, labels_cc = cv2.connectedComponents(mask, connectivity=8)
    print("  connected components: %d (expected 1, circles overlap)" % (num_cc - 1))

    kernel_size = _get_dynamic_kernel_size(h * w)
    print("  dynamic kernel size: %d" % kernel_size)

    labels_ws = watershed_separation(mask, dist_field, kernel_size=kernel_size)
    num_ws = len(np.unique(labels_ws)) - 1
    print("  watershed instances: %d (expected 2)" % num_ws)

    vis_cc = np.zeros((h, w, 3), dtype=np.uint8)
    vis_ws = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [(0, 128, 0), (128, 0, 0), (0, 0, 128), (128, 128, 0)]

    for i in range(1, num_cc):
        vis_cc[labels_cc == i] = colors[i % len(colors)]
    for i in range(1, num_ws + 1):
        vis_ws[labels_ws == i] = colors[i % len(colors)]

    cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_input_mask.png"), mask * 255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_input_dist.png"),
                cv2.applyColorMap((dist_field * 255).astype(np.uint8), cv2.COLORMAP_JET))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_cc_result.png"), vis_cc)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "watershed_ws_result.png"), vis_ws)

    print("\n  Dynamic kernel size vs image area:")
    for area_name, area in [
        ("1024x1024", 1024 * 1024),
        ("1224x1024", 1224 * 1024),
        ("2448x2048", 2448 * 2048),
        ("1936x2584", 1936 * 2584),
    ]:
        ks = _get_dynamic_kernel_size(area)
        print("    %s (area=%d): kernel_size=%d" % (area_name, area, ks))


def main():
    print("=" * 60)
    print("Pipeline diagnostic")
    print("Output dir: " + OUTPUT_DIR)
    print("=" * 60)

    test_letterbox_padding()
    test_mask_generation()
    test_distance_compensation()
    test_tiled_positions()
    test_gaussian_weight()
    test_watershed()

    print("\n" + "=" * 60)
    print("Diagnostic complete! Results saved to:")
    print("  " + OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()