# -*- coding: utf-8 -*-
"""
实例级管线对照评估工具
======================
用 labelme GT（人工标注多边形）评估两条推理管线的实例分类准确率：
  1. 基线：语义投票（outputs/inference_baseline 或任意 inst.png+class.json）
  2. 新：实例级分类器（outputs/inference_instance）

匹配方式：每个 GT 实例与其重叠最大的预测实例配对（IoU 最大），
取该预测实例的类别为预测类别；GT 实例的类别为多边形标注。

用法：
    python tools/eval_instance_pipelines.py \
        --gt_dir data/raw --images /tmp/val_imgs \
        --baseline_dir outputs/val_baseline \
        --instance_dir outputs/val_instance
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np


def parse_gt_instances(json_path):
    """返回 [(mask, cls), ...]：每个多边形一个实例。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    h, w = data.get("imageHeight"), data.get("imageWidth")
    out = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        if label in ("ferrite", "ferrite_core", "铁素体", "1"):
            cls = 1
        elif label in ("pearlite", "珠光体", "0"):
            cls = 0
        else:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        out.append((mask, cls))
    return out


def evaluate_dir(gt_insts, inst_path, class_path):
    """对一个预测目录评估实例分类准确率。"""
    inst_map = cv2.imread(inst_path, cv2.IMREAD_GRAYSCALE)
    with open(class_path, "r", encoding="utf-8") as f:
        class_map = {int(k): int(v) for k, v in json.load(f).items()}

    pred_cls_arr = np.zeros_like(inst_map, dtype=np.int8)
    for iid, cls in class_map.items():
        pred_cls_arr[inst_map == iid] = cls

    tp = tn = 0
    n = 0
    matched_ids = set()
    for mask, cls in gt_insts:
        gt_px = mask > 0
        if gt_px.sum() < 10:
            continue
        overlap = np.where(gt_px, inst_map, 0)
        counts = np.bincount(overlap.ravel())   # 线性计数，避免 np.unique 全图排序
        ids = np.nonzero(counts)[0]
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue  # 预测无覆盖，跳过
        best_id = ids[np.argmax(counts[ids])]
        pred_cls = int(pred_cls_arr[inst_map == best_id][0]) if (inst_map == best_id).any() else -1
        n += 1
        matched_ids.add(best_id)
        if pred_cls == cls:
            tp += 1
        # 混淆矩阵
        if cls == 1:
            tn += 1 if pred_cls == 1 else 0
    # 简化：统计准确率 + 两类召回
    acc = tp / max(n, 1)
    return acc, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--instance_dir", required=True)
    args = ap.parse_args()

    images = sorted(os.listdir(args.images))
    print(f"评估 {len(images)} 张图")
    agg = {"base": [0, 0], "inst": [0, 0]}
    for fn in images:
        bn = os.path.splitext(fn)[0]
        gt_path = os.path.join(args.gt_dir, bn + ".json")
        if not os.path.exists(gt_path):
            continue
        gt_insts = parse_gt_instances(gt_path)
        a_b, n_b = evaluate_dir(gt_insts,
                                os.path.join(args.baseline_dir, f"{bn}_inst.png"),
                                os.path.join(args.baseline_dir, f"{bn}_class.json"))
        a_i, n_i = evaluate_dir(gt_insts,
                                os.path.join(args.instance_dir, f"{bn}_inst.png"),
                                os.path.join(args.instance_dir, f"{bn}_class.json"))
        agg["base"][0] += a_b * n_b; agg["base"][1] += n_b
        agg["inst"][0] += a_i * n_i; agg["inst"][1] += n_i
        print(f"{bn}: GT实例={len(gt_insts)} 基线acc={a_b:.3f}(n={n_b}) 分类器acc={a_i:.3f}(n={n_i})")

    print("\n汇总:")
    for k, (s, n) in agg.items():
        print(f"  {k}: acc={s/max(n,1):.4f} (n={n})")


if __name__ == "__main__":
    main()
