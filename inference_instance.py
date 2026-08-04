# -*- coding: utf-8 -*-
"""
实例级分类器推理管线（对照实验）
=================================
与 inference.py（语义投票管线）并行的新管线：
  1. best_bnd 检查点 -> 边界头 -> 骨架化 + 受阻分水岭 -> 实例掩码
  2. 冻结 SAM2 trunk + FPN 特征 -> 按实例掩码池化（与训练特征完全一致）
  3. 实例级分类器（tools/instance_classifier.py 训练产物）-> 实例类别

输出与 inference.py 相同格式：{basename}_inst.png + {basename}_class.json。

用法：
    python inference_instance.py --config config/default_config.yaml \
        --checkpoint outputs/stage2/best_bnd_model/stage2_epoch100.pth \
        --classifier outputs/instance_clf/classifier.joblib \
        --test_dir data/test --output_dir outputs/inference_instance
"""

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.dataset import letterbox
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.sam2_encoder import SAM2Encoder
from tools.instance_classifier import pooled_stats
from utils.post_process import boundary_watershed_separation

CLASS_PEARLITE = 0
CLASS_FERRITE = 1


def build_model(config, checkpoint, device):
    encoder = SAM2Encoder(
        config_file=config["sam2"]["config_file"],
        ckpt_path=os.path.join(config["paths"]["project_root"], "weights",
                               config["paths"]["sam2_ckpt"]),
        device=device, freeze=True,
        sam2_repo_path=os.path.join(config["paths"]["project_root"],
                                    config["sam2"]["sam2_repo_path"]),
    )
    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=config["decoder"]["fpn_channels"],
        num_classes=config["decoder"]["num_classes"],
        dropout=config["decoder"]["dropout"],
        use_bn=config["decoder"]["use_bn"],
    )
    model = SegmentationModel(encoder, decoder).to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(ck["decoder_state_dict"])
    model.eval()
    return model


def pool_instance_features(feats_np, seg_feat_np, bnd_feat_np, gray_lb,
                           mask_lb, target):
    """与训练一致的实例特征向量（不含语义概率列）。"""
    vec = []
    for fmap in feats_np:
        fh, fw = fmap.shape[-2:]
        mf = cv2.resize(mask_lb, (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
        mean, std = pooled_stats(fmap, mf)
        if mean is None:
            return None
        vec.extend(mean.tolist())
        vec.extend(std.tolist())
    for fmap in (seg_feat_np, bnd_feat_np):
        fh, fw = fmap.shape[-2:]
        mf = cv2.resize(mask_lb, (fw, fh), interpolation=cv2.INTER_NEAREST) > 0
        mean, std = pooled_stats(fmap, mf)
        if mean is None:
            return None
        vec.extend(mean.tolist())
        vec.extend(std.tolist())
    gm = mask_lb > 0
    vals = gray_lb[gm]
    ys, xs = np.where(gm)
    bbox_aspect = (ys.max() - ys.min() + 1) / max(xs.max() - xs.min() + 1, 1)
    vec.extend([
        float(vals.mean()), float(vals.std()),
        float(mask_lb.sum()) / (target * target), float(bbox_aspect),
    ])
    return np.array(vec, dtype=np.float32)


def predict_single(model, classifier, image_path, device, infer_cfg,
                 target=1024):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]
    basename = os.path.splitext(os.path.basename(image_path))[0]

    img_lb, scale, pad_h, pad_w = letterbox(image_rgb, target)
    gray_lb = cv2.cvtColor(img_lb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    t = torch.from_numpy(img_lb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        feats = model.encoder(t)
        seg_feat = model.decoder.seg_fpn(feats)
        bnd_feat = model.decoder.boundary_fpn(feats)
        out = model.decoder(feats)   # 与 inference.py 一致：原生 256 空间输出

    feats_np = [f[0].cpu().numpy() for f in feats]
    seg_feat_np = seg_feat[0].cpu().numpy()
    bnd_feat_np = bnd_feat[0].cpu().numpy()

    # 与 inference.py 完全一致的掩码生成路径：原生 256 输出 -> 裁剪内容区 ->
    # F.interpolate(bilinear, align_corners=True) 回原图，保证实例掩码与基线逐位一致
    content_h = int(round(h_orig * scale / 4))
    content_w = int(round(w_orig * scale / 4))
    output = out[:, :, :content_h, :content_w]
    output = F.interpolate(output, size=(h_orig, w_orig),
                           mode="bilinear", align_corners=True)
    seg_logits = output[0, 0].cpu()
    bnd_logits = output[0, 1].cpu()
    if infer_cfg.get("boundary_logit_scale", 1.0) != 1.0:
        bnd_logits = bnd_logits * infer_cfg["boundary_logit_scale"]

    seg_prob = torch.sigmoid(seg_logits).numpy()
    bnd_prob = torch.sigmoid(bnd_logits).numpy()

    semantic_mask = (seg_prob > infer_cfg.get("threshold", 0.5)).astype(np.uint8)
    from utils.post_process import hysteresis_threshold
    bt = infer_cfg.get("boundary_threshold", 0.5)
    bth = infer_cfg.get("boundary_threshold_high", None)
    if bth is not None and bth > bt:
        boundary_mask = hysteresis_threshold(bnd_prob, low=bt, high=bth)
    else:
        boundary_mask = (bnd_prob > bt).astype(np.uint8)

    def classify_fn(inst_mask_orig):
        # 原图实例掩码 -> letterbox 空间 -> 特征池化 -> 分类
        new_h = target - pad_h
        new_w = target - pad_w
        resized = cv2.resize(inst_mask_orig.astype(np.uint8), (new_w, new_h),
                             interpolation=cv2.INTER_NEAREST)
        mask_lb = np.zeros((target, target), dtype=np.uint8)
        mask_lb[:new_h, :new_w] = resized
        vec = pool_instance_features(feats_np, seg_feat_np, bnd_feat_np,
                                     gray_lb, mask_lb, target)
        if vec is None or vec.shape[0] != classifier.n_features_in_:
            return CLASS_PEARLITE
        return int(classifier.predict(vec.reshape(1, -1))[0])

    inst_map, class_map = boundary_watershed_separation(
        semantic_mask, boundary_mask,
        dilate_width=infer_cfg.get("watershed_dilate_width", 1),
        min_area=infer_cfg.get("min_instance_area", 500),
        max_instance_id=infer_cfg.get("max_instance_id", 255),
        classify_fn=classify_fn,
    )
    return inst_map, class_map, basename, (h_orig, w_orig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="边界模型检查点（默认取 instance_classifier.checkpoint）")
    parser.add_argument("--classifier", default=None,
                        help="实例分类器 joblib（默认取 instance_classifier.classifier）")
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    ic_cfg = config.get("instance_classifier", {})
    infer_cfg = config.get("inference", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = args.checkpoint or ic_cfg.get(
        "checkpoint", "outputs/stage2/best_bnd_model/stage2_epoch100.pth")
    classifier_path = args.classifier or ic_cfg.get(
        "classifier", "outputs/instance_clf/classifier.joblib")
    test_dir = args.test_dir or ic_cfg.get("test_dir", "data/test")
    output_dir = args.output_dir or ic_cfg.get(
        "output_dir", "outputs/inference_instance")
    target_size = int(ic_cfg.get("target_size", 1024))

    model = build_model(config, checkpoint, device)
    classifier = joblib.load(classifier_path)
    print(f"checkpoint: {checkpoint}")
    print(f"classifier: {os.path.basename(classifier_path)} "
          f"feat_dim={classifier.n_features_in_}")
    os.makedirs(output_dir, exist_ok=True)

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for ext in valid_exts:
        paths.extend(glob.glob(os.path.join(test_dir, ext)))
    paths.sort()
    print(f"test images: {len(paths)}")

    n_ferrite = n_pearlite = 0
    for i, img_path in enumerate(paths):
        inst_map, class_map, basename, (h, w) = predict_single(
            model, classifier, img_path, device, infer_cfg, target=target_size)
        inst_path = os.path.join(output_dir, f"{basename}_inst.png")
        cv2.imwrite(inst_path, inst_map)
        class_json = {str(k): int(v) for k, v in class_map.items()}
        with open(os.path.join(output_dir, f"{basename}_class.json"),
                  "w", encoding="utf-8") as f:
            json.dump(class_json, f, ensure_ascii=False, indent=2)
        n_ferrite += sum(1 for v in class_map.values() if v == CLASS_FERRITE)
        n_pearlite += sum(1 for v in class_map.values() if v == CLASS_PEARLITE)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(paths)}: {basename} instances={len(class_map)}")

    print(f"done. total instances: ferrite={n_ferrite}, pearlite={n_pearlite}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
