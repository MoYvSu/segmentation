# -*- coding: utf-8 -*-
"""
语义梯度生成单线边界伪标签（v6 训练目标源）
=============================================
用 v3 语义头对无标签/任意图像生成"单线锐利"边界伪标签：

  语义概率(256) -> 上采样 1024 -> |∇seg| -> 每图自适应阈值(P80)
  -> 膨胀(桥接双梯度线) -> 骨架化(单线) -> 膨胀1px -> 二值单线目标

输出与 stage1_direct 缓存同格式（boundary_probs.npy + names.txt + report.csv），
可直接作为 semi_supervised.pseudo_label_cache_dir 使用（UnlabeledDataset 按名对齐）。

用法：
    python tools/precompute_semantic_boundary.py --config config/default_config.yaml \
        --checkpoint outputs/stage2_joint_v3/best_model_stage2.pth \
        --outdir outputs/pseudo_labels/semantic_boundary
    # val 质量检查（对 data/raw 的 val 划分对比 GT）
    python tools/precompute_semantic_boundary.py --config ... --checkpoint ... --check_val
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import yaml

from data.dataset import letterbox, letterbox_mask, split_train_val_indices
from skimage.morphology import skeletonize

CLASS_FERRITE = 1


def semantic_gradient(seg_prob, smooth=1.0):
    if smooth > 0:
        seg_prob = cv2.GaussianBlur(seg_prob, (0, 0), smooth)
    gx = cv2.Sobel(seg_prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(seg_prob, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def gradient_to_single_line(grad, percentile=80, bridge_dilate=2, out_dilate=1):
    """梯度图 -> 每图自适应阈值 -> 桥接膨胀 -> 骨架化 -> 窄膨胀 -> 单线二值。

    bridge_dilate: 骨架化前膨胀宽度，把铁素体-铁素体的双梯度线桥接成带，
                   骨架取中轴 -> 单线（对应双线中间的黑色晶界）。
    out_dilate: 骨架后膨胀宽度，目标带 2*out_dilate+1 px。
    """
    thr = np.percentile(grad, percentile)
    bm = (grad > thr).astype(np.uint8) * 255
    if bridge_dilate > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bridge_dilate + 1, 2 * bridge_dilate + 1))
        bm = cv2.dilate(bm, k)
    sk = (skeletonize(bm > 0) * 255).astype(np.uint8)
    if out_dilate > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * out_dilate + 1, 2 * out_dilate + 1))
        sk = cv2.dilate(sk, k)
    return (sk > 0).astype(np.uint8)


def build_model(config, checkpoint, device):
    """复用 inference.py 的模型构建（含 LoRA 加载）。"""
    import inference
    return inference.build_model(config, device, checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_config.yaml")
    ap.add_argument("--checkpoint", default="outputs/stage2_joint_v3/best_model_stage2.pth")
    ap.add_argument("--outdir", default="outputs/pseudo_labels/semantic_boundary")
    ap.add_argument("--image_dir", default=None,
                    help="处理目录（默认 config.semi_supervised.unlabeled_dir）")
    ap.add_argument("--percentile", type=float, default=80.0)
    ap.add_argument("--bridge_dilate", type=int, default=2)
    ap.add_argument("--out_dilate", type=int, default=1)
    ap.add_argument("--check_val", action="store_true",
                    help="只做 val 质量检查（对比 GT），不生成缓存")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = config["paths"]["project_root"]
    model = build_model(config, args.checkpoint, device)
    model.eval()

    if args.check_val:
        # ---- val 质量检查 ----
        raw_dir = os.path.join(root, config["paths"]["raw_data_dir"])
        jsons = sorted(glob.glob(os.path.join(raw_dir, "*.json")))
        val_idx = split_train_val_indices(len(jsons), 0.8, 42, "val")
        val_names = [os.path.splitext(os.path.basename(jsons[i]))[0] for i in val_idx]
        print(f"val images: {len(val_names)}")
        iou_list, dbl_list, recall3_list = [], [], []
        for vn in val_names:
            img = cv2.cvtColor(cv2.imread(os.path.join(raw_dir, vn + ".jpg")), cv2.COLOR_BGR2RGB)
            lb, _, _, _ = letterbox(img, 1024)
            t = torch.from_numpy(lb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                out = model(t)
            seg = cv2.resize(torch.sigmoid(out[0, 0]).cpu().numpy(), (1024, 1024),
                             interpolation=cv2.INTER_LINEAR)
            grad = semantic_gradient(seg)
            pred = gradient_to_single_line(grad, args.percentile, args.bridge_dilate, args.out_dilate)
            gt, _, _, _ = letterbox_mask(np.load(
                os.path.join(root, config["boundary"]["gt_dir"], vn + "_gt.npz"))["boundary"], 1024)
            # 边界 IoU（各膨胀1px 容差）
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            pd = cv2.dilate(pred, k3)
            gd = cv2.dilate(gt, k3)
            inter = int((pd & gd).sum()); union = int((pd | gd).sum())
            iou_list.append(inter / max(union, 1))
            # GT 召回@3px
            recall3_list.append(float((gd[pred > 0] > 0).mean()) if pred.sum() else 0.0)
            # 预测单线率（骨架双线占比）
            sk = (skeletonize(pred > 0) * 255).astype(np.uint8)
            n, labels = cv2.connectedComponents(sk, connectivity=8)
            fracs = []
            for lab in range(1, n):
                comp = (labels == lab).astype(np.uint8)
                d = cv2.dilate(comp, np.ones((5, 5), np.uint8))
                others = (labels > 0) & (labels != lab)
                ov = (d & others).sum()
                fracs.append(ov / max(int(comp.sum()), 1))
            dbl_list.append(float(np.mean(fracs)) if fracs else 0.0)
            print(f"  {vn}: IoU@1px={iou_list[-1]:.3f} 单线率={1-dbl_list[-1]:.3f} GT召回@3px={recall3_list[-1]:.3f}")
        print("\n==== 语义伪标签 val 质量 ====")
        print(f"边界IoU(容差1px)  均值: {np.mean(iou_list):.3f}")
        print(f"单线率(1-双线占比) 均值: {1-np.mean(dbl_list):.3f}")
        print(f"GT召回@3px         均值: {np.mean(recall3_list):.3f}")
        return

    # ---- 生成缓存 ----
    os.makedirs(args.outdir, exist_ok=True)
    image_dir = args.image_dir or os.path.join(
        root, config["semi_supervised"]["unlabeled_dir"])
    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for ext in valid_exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths.sort()
    print(f"images: {len(paths)}")

    probs_path = os.path.join(args.outdir, "boundary_probs.npy")
    names_path = os.path.join(args.outdir, "names.txt")
    report_path = os.path.join(args.outdir, "report.csv")
    mm = np.lib.format.open_memmap(probs_path, mode="w+", dtype=np.float16,
                                   shape=(len(paths), 1024, 1024))
    names = []
    report = [["basename", "line_frac"]]
    for i, p in enumerate(paths):
        bn = os.path.splitext(os.path.basename(p))[0]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        lb, _, _, _ = letterbox(img, 1024)
        t = torch.from_numpy(lb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            out = model(t)
        seg = cv2.resize(torch.sigmoid(out[0, 0]).cpu().numpy(), (1024, 1024),
                         interpolation=cv2.INTER_LINEAR)
        grad = semantic_gradient(seg)
        pred = gradient_to_single_line(grad, args.percentile, args.bridge_dilate, args.out_dilate)
        mm[i] = pred.astype(np.float16)
        names.append(bn)
        report.append([bn, f"{pred.mean():.5f}"])
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(paths)}] {bn}")
    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(",".join(r) for r in report) + "\n")
    meta = {"percentile": args.percentile, "bridge_dilate": args.bridge_dilate,
            "out_dilate": args.out_dilate, "checkpoint": args.checkpoint}
    json.dump(meta, open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"done. {len(paths)} images -> {args.outdir}")


if __name__ == "__main__":
    main()
