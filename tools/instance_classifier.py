# -*- coding: utf-8 -*-
"""
实例级分类器实验（对照管线）
============================
目标：用 labelme 实例标注（每个多边形 = 一个晶粒实例）训练一个实例级分类器，
替代"分水岭实例掩码 + 逐像素语义投票"的类别判定，作为推理对照实验。

特征来源（同一 forward 的冻结 SAM2 trunk + best_bnd 检查点 FPN）：
  1. SAM2 四尺度 trunk 特征：masked mean + std（112/224/448/896 ch）
  2. seg_fpn 特征（256ch @ 1/4）：masked mean + std
  3. boundary_fpn 特征（256ch @ 1/4）：masked mean + std
  4. 实例灰度统计（原图：mean/std/q25/q75/面积/长宽比）

用法：
    python tools/instance_classifier.py --config config/default_config.yaml \
        --checkpoint outputs/stage2/best_bnd_model/stage2_epoch100.pth \
        --outdir outputs/instance_clf

输出：
    features_train.npz / features_val.npz（特征缓存，可复用）
    report.txt / report.csv（分类器对比 + 语义投票基线）
    classifier.joblib + scaler.joblib + meta.json（最优分类器）
"""

import argparse
import csv
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

from data.dataset import letterbox, split_train_val_indices
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.sam2_encoder import SAM2Encoder

CLASS_PEARLITE = 0
CLASS_FERRITE = 1


def parse_labelme_instances(json_path):
    """解析 labelme JSON，返回 [(polygon_points, cls), ...]，保持每个多边形一个实例。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    h, w = data.get("imageHeight"), data.get("imageWidth")
    instances = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        if label in ("ferrite", "ferrite_core", "铁素体", "1"):
            cls = CLASS_FERRITE
        elif label in ("pearlite", "珠光体", "0"):
            cls = CLASS_PEARLITE
        else:
            continue
        instances.append((points, cls))
    return instances, h, w


def pooled_stats(feat, mask_f):
    """掩码索引 + 逐通道 mean/std。

    mask_f: [H,W] bool（已对齐 feat 分辨率）。只取掩码覆盖的像素，避免
    [C,H,W] 全张量乘法（FPN 特征 256x256x256 会带来 TB 级内存流量）。
    返回 (mean[C], std[C])；掩码为空返回 (None, None)。
    """
    ys, xs = np.nonzero(mask_f)
    if len(ys) == 0:
        return None, None
    vals = feat[:, ys, xs]          # [C, n]
    mean = vals.mean(axis=1)
    var = vals.var(axis=1)
    return mean, np.sqrt(np.maximum(var, 0.0))


def extract_features(model, image_path, instances, device, target=1024):
    """对一张图的所有实例提取特征（高效版：一次性构建实例 ID 图再池化）。

    instances: [(points, cls), ...]（原图像素坐标）
    返回 (feat_matrix [N, D], labels [N])；跳过面积过小实例。
    """
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    img_lb, scale, pad_h, pad_w = letterbox(image_rgb, target)
    gray_lb = cv2.cvtColor(img_lb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    t = torch.from_numpy(img_lb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        feats = model.encoder(t)                     # 4 尺度
        seg_feat = model.decoder.seg_fpn(feats)      # [B,256,256,256]
        bnd_feat = model.decoder.boundary_fpn(feats)
        out = model.decoder(feats, output_size=(target, target))
        seg_prob = torch.sigmoid(out[0, 0]).cpu().numpy()

    # 一次性构建 1024 实例 ID 图（多边形按 letterbox scale 缩放后填充）
    idmap = np.zeros((target, target), dtype=np.int32)
    valid = []  # (id, cls)
    for inst_id, (points, cls) in enumerate(instances, start=1):
        pts = np.array(points, dtype=np.float32)
        pts[:, 0] *= scale
        pts[:, 1] *= scale
        cv2.fillPoly(idmap, [pts.astype(np.int32).reshape((-1, 1, 2))], inst_id)
        valid.append((inst_id, cls))

    scale_sizes = [(f.shape[-2], f.shape[-1]) for f in feats]
    seg_feat_np = seg_feat[0].cpu().numpy()
    bnd_feat_np = bnd_feat[0].cpu().numpy()
    feats_np = [f[0].cpu().numpy() for f in feats]
    seg_prob_gray = cv2.resize(seg_prob, (target, target), interpolation=cv2.INTER_LINEAR)

    rows = []
    labels = []
    for inst_id, cls in valid:
        mask_lb = (idmap == inst_id)
        area_lb = int(mask_lb.sum())
        if area_lb < 20:                 # 过滤过小（letterbox 空间）
            continue

        vec = []
        ok = True
        for fi, fmap in enumerate(feats_np):
            fh, fw = scale_sizes[fi]
            mf = cv2.resize(mask_lb.astype(np.uint8), (fw, fh),
                            interpolation=cv2.INTER_NEAREST) > 0
            mean, std = pooled_stats(fmap, mf)
            if mean is None:
                ok = False
                break
            vec.extend(mean.tolist())
            vec.extend(std.tolist())
        if not ok:
            continue
        for fmap in (seg_feat_np, bnd_feat_np):
            fh, fw = fmap.shape[-2:]
            mf = cv2.resize(mask_lb.astype(np.uint8), (fw, fh),
                            interpolation=cv2.INTER_NEAREST) > 0
            mean, std = pooled_stats(fmap, mf)
            if mean is None:
                ok = False
                break
            vec.extend(mean.tolist())
            vec.extend(std.tolist())
        if not ok:
            continue

        # 灰度统计（letterbox 灰度图，均值/std；面积与长宽比）
        vals = gray_lb[mask_lb]
        ys, xs = np.where(mask_lb)
        bbox_aspect = (ys.max() - ys.min() + 1) / max(xs.max() - xs.min() + 1, 1)
        vec.extend([
            float(vals.mean()), float(vals.std()),
            float(area_lb) / (target * target), float(bbox_aspect),
        ])
        # 语义概率均值（用于投票基线）
        vec.append(float(seg_prob_gray[mask_lb].mean()))

        rows.append(vec)
        labels.append(cls)
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--checkpoint", default="outputs/stage2/best_bnd_model/stage2_epoch100.pth")
    parser.add_argument("--outdir", default="outputs/instance_clf")
    parser.add_argument("--reuse_features", action="store_true",
                        help="复用已缓存的特征，不重新提取")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = config["paths"]["project_root"]

    os.makedirs(args.outdir, exist_ok=True)

    encoder = SAM2Encoder(
        config_file=config["sam2"]["config_file"],
        ckpt_path=os.path.join(root, "weights", config["paths"]["sam2_ckpt"]),
        device=device, freeze=True,
        sam2_repo_path=os.path.join(root, config["sam2"]["sam2_repo_path"]),
    )
    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=config["decoder"]["fpn_channels"],
        num_classes=config["decoder"]["num_classes"],
        dropout=config["decoder"]["dropout"],
        use_bn=config["decoder"]["use_bn"],
    )
    model = SegmentationModel(encoder, decoder).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    model.eval()

    # ---- 数据划分（与模型训练完全一致：seed 42, ratio 0.8）----
    raw_dir = os.path.join(root, config["paths"]["raw_data_dir"])
    json_paths = sorted([
        os.path.join(raw_dir, p) for p in os.listdir(raw_dir) if p.endswith(".json")
    ])
    idx_map = split_train_val_indices(len(json_paths), 0.8, 42, "train")
    val_idx = split_train_val_indices(len(json_paths), 0.8, 42, "val")
    train_json = [json_paths[i] for i in idx_map]
    val_json = [json_paths[i] for i in val_idx]
    print(f"train images: {len(train_json)}, val images: {len(val_json)}", flush=True)

    def load_or_extract(split_list, tag):
        cache = os.path.join(args.outdir, f"features_{tag}.npz")
        if args.reuse_features and os.path.exists(cache):
            d = np.load(cache)
            print(f"[{tag}] 复用缓存 {cache}: {d['X'].shape}")
            return d["X"], d["y"]
        Xs, ys = [], []
        for jp in split_list:
            instances, h, w = parse_labelme_instances(jp)
            img_path = os.path.splitext(jp)[0] + ".jpg"
            if not os.path.exists(img_path):
                img_path = os.path.splitext(jp)[0] + ".png"
            X, y = extract_features(model, img_path, instances, device)
            Xs.append(X)
            ys.append(y)
            print(f"  {os.path.basename(jp)}: {len(y)} 实例", flush=True)
        X = np.vstack(Xs) if Xs else np.zeros((0, 0), dtype=np.float32)
        y = np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64)
        np.savez(cache, X=X, y=y)
        print(f"[{tag}] 提取完成: X={X.shape}, 类别分布={np.bincount(y).tolist()}")
        return X, y

    X_tr, y_tr = load_or_extract(train_json, "train")
    X_va, y_va = load_or_extract(val_json, "val")

    # ---- 基线：语义投票（最后一维是语义概率均值，>0.5 判 ferrite）----
    sem_vote = (X_va[:, -1] > 0.5).astype(int)
    acc_base = float((sem_vote == y_va).mean())
    # 语义投票按实例像素加权版（这里用概率均值近似，真正的投票用逐像素）
    print(f"\n基线 语义投票(val): 准确率={acc_base:.4f}")
    print("  混淆(预测->真实): ferrite/pearlite")
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_va, sem_vote, labels=[CLASS_FERRITE, CLASS_PEARLITE]))

    # ---- 分类器对比 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr = X_tr[:, :-1]  # 去掉语义概率列（分类器只用特征）
    Xva = X_va[:, :-1]

    candidates = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "SVM-RBF": make_pipeline(StandardScaler(), PCA(n_components=128), SVC(C=1.0, gamma="scale", class_weight="balanced")),
        "MLP(128-64)": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800, early_stopping=True)),
    }

    report = [["method", "val_acc", "ferrite_recall", "pearlite_recall", "macro_f1", "train_acc"]]
    best = None
    for name, pipe in candidates.items():
        pipe.fit(Xtr, y_tr)
        pred = pipe.predict(Xva)
        acc = float((pred == y_va).mean())
        cm = confusion_matrix(y_va, pred, labels=[CLASS_FERRITE, CLASS_PEARLITE])
        fr = cm[0, 0] / max(cm[0].sum(), 1)
        pr = cm[1, 1] / max(cm[1].sum(), 1)
        f1 = 2 * fr * pr / max(fr + pr, 1e-9)
        tr_acc = float((pipe.predict(Xtr) == y_tr).mean())
        report.append([name, f"{acc:.4f}", f"{fr:.4f}", f"{pr:.4f}", f"{f1:.4f}", f"{tr_acc:.4f}"])
        print(f"{name}: val_acc={acc:.4f} ferrite_recall={fr:.4f} pearlite_recall={pr:.4f} "
              f"macro_f1={f1:.4f} train_acc={tr_acc:.4f}")
        if best is None or acc > best[1]:
            best = (name, acc, pipe)

    with open(os.path.join(args.outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"基线 语义投票 val_acc={acc_base:.4f}\n")
        for row in report:
            f.write(",".join(row) + "\n")
    with open(os.path.join(args.outdir, "report.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(report)

    # 保存最优分类器
    import joblib
    name, acc, pipe = best
    joblib.dump(pipe, os.path.join(args.outdir, "classifier.joblib"))
    meta = {
        "method": name,
        "val_acc": acc,
        "baseline_semantic_vote_acc": acc_base,
        "feature_dim": int(Xtr.shape[1]),
        "feature_spec": "trunk4x 逐通道(mean,std)+seg_fpn 逐通道(mean,std)+bnd_fpn 逐通道(mean,std)+gray(4)+sem_prob(1)",
        "checkpoint": os.path.abspath(args.checkpoint),
        "train_instances": int(len(y_tr)),
        "val_instances": int(len(y_va)),
    }
    json.dump(meta, open(os.path.join(args.outdir, "meta.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n最优: {name} val_acc={acc:.4f} -> 已保存 {args.outdir}")


if __name__ == "__main__":
    main()
