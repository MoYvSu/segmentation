# -*- coding: utf-8 -*-
"""从 Labelme 掩码并集的补集生成边界候选 GT。

与 Canny 边界不同，本脚本只把 ferrite/pearlite 多边形未覆盖、且靠近
已标注晶粒区域的像素作为晶界候选，避免把晶粒内部纹理当成监督信号。
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def polygon_mask(shapes, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        points = np.asarray(shape.get("points", []), dtype=np.int32).reshape((-1, 1, 2))
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def make_targets(
    json_path,
    image_path,
    near_radius=6,
    close_radius=1,
    min_component=16,
    soft_decay=4.0,
):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ferrite_shapes = []
    pearlite_shapes = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "").lower().strip()
        if label in ("ferrite", "ferrite_core", "铁素体", "1"):
            ferrite_shapes.append(shape)
        elif label in ("pearlite", "珠光体", "0"):
            pearlite_shapes.append(shape)

    ferrite = polygon_mask(ferrite_shapes, height, width)
    pearlite = polygon_mask(pearlite_shapes, height, width)
    covered = np.maximum(ferrite, pearlite)
    semantic = ferrite.astype(np.uint8)

    # 只保留靠近已标注晶粒的补集，去掉远离晶粒的外部空白区域。
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * near_radius + 1, 2 * near_radius + 1)
    )
    near_grain = cv2.dilate(covered, kernel) > 0
    boundary = ((covered == 0) & near_grain).astype(np.uint8)

    # 轻微闭运算填补栅格化造成的单像素断点，不描绘多边形轮廓。
    if close_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
        )
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, k)

    # 清除孤立小孔洞/噪点，但保留与晶界网络相连的分支。
    if min_component > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(boundary, 8)
        keep = np.zeros_like(boundary)
        for idx in range(1, n):
            if stats[idx, cv2.CC_STAT_AREA] >= min_component:
                keep[labels == idx] = 1
        boundary = keep

    # 软目标：未覆盖区域仍是候选边界，但越远离已标注晶粒，置信度越低。
    # 这样不会把整条 near_radius 带状区域都当成同等确定的晶界。
    dist_to_covered = cv2.distanceTransform(
        (covered == 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    boundary_soft = np.exp(-dist_to_covered / max(float(soft_decay), 1e-3)).astype(
        np.float32
    )
    boundary_soft *= boundary.astype(np.float32)

    return semantic, boundary, boundary_soft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/purified_gt_uncovered")
    parser.add_argument("--near_radius", type=int, default=6)
    parser.add_argument("--close_radius", type=int, default=1)
    parser.add_argument("--min_component", type=int, default=16)
    parser.add_argument("--soft_decay", type=float, default=4.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for json_path in sorted(glob.glob(os.path.join(args.raw_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(json_path))[0]
        image_path = os.path.join(args.raw_dir, name + ".jpg")
        semantic, boundary, boundary_soft = make_targets(
            json_path,
            image_path,
            near_radius=args.near_radius,
            close_radius=args.close_radius,
            min_component=args.min_component,
            soft_decay=args.soft_decay,
        )
        np.savez_compressed(
            os.path.join(args.output_dir, name + "_gt.npz"),
            semantic=semantic,
            boundary=boundary,
            boundary_soft=boundary_soft,
        )
        rows.append((name, float(semantic.mean()), float(boundary.mean()), float(boundary_soft.mean())))

    print("name,semantic_rate,boundary_rate")
    for row in rows:
        print(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f}")
    if rows:
        print(
            f"mean,{np.mean([r[1] for r in rows]):.6f},"
            f"{np.mean([r[2] for r in rows]):.6f},"
            f"{np.mean([r[3] for r in rows]):.6f}"
        )


if __name__ == "__main__":
    main()
