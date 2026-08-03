# -*- coding: utf-8 -*-
"""
零 Epoch 预测图与指标计算硬审计脚本
==================================
诊断 ferrite_iou 始终为 0.0000 的根因。

三项闭环审计：
1. 数据源审计：target[:, 0].sum() 是否 > 0
2. 前向传播数值范围审计：pred[:, 0] logits 是否有正有负
3. 评估函数与二值化门限审计：preds_binary.sum() 是否为 0
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from data.dataset import MetallographicDataset, collate_fn
from models.fpn_decoder import FPNDecoder, SegmentationModel
from models.sam2_encoder import SAM2Encoder
from utils.metrics import SegMetrics


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "config/default_config.yaml"))
    sam2_cfg = config["sam2"]
    decoder_cfg = config["decoder"]
    paths_cfg = config["paths"]
    data_cfg = config["data"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # ================================================================
    # 构建 Dataset（验证集，无增强）
    # ================================================================
    data_dir = os.path.join(paths_cfg["project_root"], paths_cfg["raw_data_dir"])
    dataset = MetallographicDataset(
        data_dir=data_dir,
        image_size=data_cfg["image_size"],
        augment=False,
        split="val",
        train_ratio=data_cfg["train_ratio"],
        seed=data_cfg["seed"],
        dist_scale_factor=data_cfg.get("dist_scale_factor", 10.0),
    )
    print(f"验证集大小: {len(dataset)}")

    if len(dataset) == 0:
        print("错误: 验证集为空！尝试使用训练集...")
        dataset = MetallographicDataset(
            data_dir=data_dir,
            image_size=data_cfg["image_size"],
            augment=False,
            split="train",
            train_ratio=data_cfg["train_ratio"],
            seed=data_cfg["seed"],
            dist_scale_factor=data_cfg.get("dist_scale_factor", 10.0),
        )
        print(f"训练集大小: {len(dataset)}")

    n_samples = min(3, len(dataset))
    print(f"将审计前 {n_samples} 张图像")
    print("=" * 80)

    # ================================================================
    # 构建 Model（随机权重，不加载任何训练 checkpoint）
    # ================================================================
    ckpt_path = os.path.join(paths_cfg["weights_dir"], paths_cfg["sam2_ckpt"])
    print(f"SAM 2 权重: {ckpt_path}")
    print(f"权重存在: {os.path.exists(ckpt_path)}")

    encoder = SAM2Encoder(
        config_file=sam2_cfg["config_file"],
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None,
        device=device,
        freeze=sam2_cfg["freeze"],
        sam2_repo_path=os.path.join(paths_cfg["project_root"], sam2_cfg["sam2_repo_path"]),
    )

    decoder = FPNDecoder(
        in_channels=encoder.get_stage_channels(),
        fpn_channels=decoder_cfg["fpn_channels"],
        num_classes=decoder_cfg["num_classes"],
        dropout=decoder_cfg["dropout"],
        use_bn=decoder_cfg["use_bn"],
    )

    model = SegmentationModel(encoder, decoder).to(device)
    model.eval()  # 随机权重，eval 模式

    # ================================================================
    # 审计 1: 数据源审计
    # ================================================================
    print("\n" + "=" * 80)
    print("审计 1: 数据源审计 (target[:, 0].sum())")
    print("=" * 80)

    samples = []
    for i in range(n_samples):
        item = dataset[i]
        target = item["target"]  # [2, H, W]
        image = item["image"]    # [3, H, W]
        img_path = item["image_path"]

        mask_ch0 = target[0]  # 二值掩码
        dist_ch1 = target[1]  # 距离场

        ferrite_pixels = mask_ch0.sum().item()
        total_pixels = mask_ch0.numel()
        pearlite_pixels = total_pixels - ferrite_pixels
        ferrite_ratio = ferrite_pixels / total_pixels

        print(f"\n  [图像 {i+1}] {os.path.basename(img_path)}")
        print(f"    target shape: {target.shape}")
        print(f"    通道0 (二值掩码):")
        print(f"      铁素体像素数 (sum): {ferrite_pixels}")
        print(f"      总像素数: {total_pixels}")
        print(f"      铁素体占比: {ferrite_ratio:.4f} ({ferrite_ratio*100:.2f}%)")
        print(f"      珠光体像素数: {pearlite_pixels}")
        print(f"      mask unique values: {torch.unique(mask_ch0).tolist()}")
        print(f"    通道1 (距离场):")
        print(f"      min={dist_ch1.min().item():.6f}, max={dist_ch1.max().item():.6f}, mean={dist_ch1.mean().item():.6f}")

        if ferrite_pixels == 0:
            print(f"      [警告] 铁素体标签全黑! Dataset 载入的铁素体标签本身为全 0!")
        else:
            print(f"      [通过] 铁素体标签非空")

        samples.append({
            "image": image,
            "target": target,
            "img_path": img_path,
        })

    # ================================================================
    # 审计 2: 前向传播数值范围审计
    # ================================================================
    print("\n" + "=" * 80)
    print("审计 2: 前向传播数值范围审计 (pred[:, 0] logits)")
    print("=" * 80)

    images = torch.stack([s["image"] for s in samples]).to(device)  # [N, 3, H, W]
    targets = torch.stack([s["target"] for s in samples]).to(device)  # [N, 2, H, W]

    with torch.no_grad():
        output = model(images, output_size=targets.shape[-2:])  # [N, 2, H, W]

    print(f"\n  模型输出 shape: {output.shape}")

    for i in range(n_samples):
        seg_logits = output[i, 0]  # [H, W] 分类 logits
        dist_pred = output[i, 1]   # [H, W] 距离场预测

        print(f"\n  [图像 {i+1}] {os.path.basename(samples[i]['img_path'])}")
        print(f"    通道0 (分类 logits):")
        print(f"      min={seg_logits.min().item():.6f}")
        print(f"      max={seg_logits.max().item():.6f}")
        print(f"      mean={seg_logits.mean().item():.6f}")
        print(f"      std={seg_logits.std().item():.6f}")
        print(f"      正值像素数: {(seg_logits > 0).sum().item()}")
        print(f"      负值像素数: {(seg_logits < 0).sum().item()}")
        print(f"      零值像素数: {(seg_logits == 0).sum().item()}")
        print(f"    通道1 (距离场预测, 经 Sigmoid):")
        print(f"      min={dist_pred.min().item():.6f}")
        print(f"      max={dist_pred.max().item():.6f}")
        print(f"      mean={dist_pred.mean().item():.6f}")

        if seg_logits.min().item() > 0:
            print(f"      [警告] 所有 logits 均为正数! 二值化后全部预测为铁素体(1)")
        elif seg_logits.max().item() < 0:
            print(f"      [警告] 所有 logits 均为负数! 二值化后全部预测为珠光体(0)")
        else:
            print(f"      [通过] logits 有正有负，分布正常")

    # ================================================================
    # 审计 3: 评估函数与二值化门限审计
    # ================================================================
    print("\n" + "=" * 80)
    print("审计 3: 评估函数与二值化门限审计")
    print("=" * 80)

    # 复现 train.py validate 函数中的二值化逻辑
    metrics = SegMetrics(num_classes=2)

    for i in range(n_samples):
        seg_logits = output[i, 0:1]  # [1, H, W]
        mask_gt = targets[i, 0:1].long()  # [1, H, W]

        # train.py 中的二值化逻辑: torch.sigmoid(seg_logits) > 0.5
        seg_prob = torch.sigmoid(seg_logits)
        pred_binary = (seg_prob > 0.5).long()  # [1, H, W]

        print(f"\n  [图像 {i+1}] {os.path.basename(samples[i]['img_path'])}")
        print(f"    seg_logits range: [{seg_logits.min().item():.4f}, {seg_logits.max().item():.4f}]")
        print(f"    sigmoid(logits) range: [{seg_prob.min().item():.4f}, {seg_prob.max().item():.4f}]")
        print(f"    二值化阈值: 0.5 (等价于 logits > 0.0)")
        print(f"    preds_binary.sum() = {pred_binary.sum().item()}")
        print(f"    target_mask.sum() = {mask_gt.sum().item()}")
        print(f"    预测前景像素数: {pred_binary.sum().item()}")
        print(f"    真实前景像素数: {mask_gt.sum().item()}")

        if pred_binary.sum().item() == 0:
            print(f"    [警告] 预测前景像素数为 0! 二值化后全部被过滤为背景!")
            print(f"    这意味着 ferrite_iou 将为 0.0 (无预测正样本与真实正样本的交集)")
        elif pred_binary.sum().item() == pred_binary.numel():
            print(f"    [警告] 预测全部为前景! 二值化后全部激活为 1!")
        else:
            print(f"    [通过] 预测前景像素数非 0 且非全量")

        # 手动计算混淆矩阵
        single_metrics = SegMetrics(num_classes=2)
        single_metrics.update_tensor(pred_binary, mask_gt)
        cm = single_metrics.confusion_matrix
        print(f"    混淆矩阵 (行=真实, 列=预测):")
        print(f"      真实珠光体(0) -> 预测珠光体(0): {cm[0, 0]}")
        print(f"      真实珠光体(0) -> 预测铁素体(1): {cm[0, 1]}")
        print(f"      真实铁素体(1) -> 预测珠光体(0): {cm[1, 0]}")
        print(f"      真实铁素体(1) -> 预测铁素体(1): {cm[1, 1]}")

        m = single_metrics.get_metrics()
        print(f"    ferrite_iou = {m['ferrite_iou']:.6f}")
        print(f"    pearlite_iou = {m['pearlite_iou']:.6f}")
        print(f"    mean_iou = {m['mean_iou']:.6f}")

        # 累积到全局 metrics
        metrics.update_tensor(pred_binary, mask_gt)

    # ================================================================
    # 汇总审计结果
    # ================================================================
    print("\n" + "=" * 80)
    print("汇总审计结果 (3 张图像合并)")
    print("=" * 80)

    cm = metrics.confusion_matrix
    print(f"\n  全局混淆矩阵 (行=真实, 列=预测):")
    print(f"    真实珠光体(0) -> 预测珠光体(0): {cm[0, 0]}")
    print(f"    真实珠光体(0) -> 预测铁素体(1): {cm[0, 1]}")
    print(f"    真实铁素体(1) -> 预测珠光体(0): {cm[1, 0]}")
    print(f"    真实铁素体(1) -> 预测铁素体(1): {cm[1, 1]}")

    m = metrics.get_metrics()
    print(f"\n  全局指标:")
    print(f"    ferrite_iou  = {m['ferrite_iou']:.6f}")
    print(f"    pearlite_iou = {m['pearlite_iou']:.6f}")
    print(f"    mean_iou     = {m['mean_iou']:.6f}")
    print(f"    mean_dice    = {m['mean_dice']:.6f}")

    print("\n" + "=" * 80)
    print("审计结论:")
    print("=" * 80)

    all_logits_neg = all(output[i, 0].max().item() <= 0 for i in range(n_samples))
    all_logits_pos = all(output[i, 0].min().item() > 0 for i in range(n_samples))

    if all_logits_neg:
        print("  [根因定位] 随机初始化模型的分类 logits 全部 <= 0")
        print("  -> sigmoid(logits) 全部 <= 0.5 -> 二值化后全部为 0 (珠光体)")
        print("  -> 铁素体预测为空 -> ferrite_iou = 0.0")
        print("  -> 这不是 Bug，而是随机初始化的数值偏移问题")
        print("  -> 修复方案: 将二值化阈值从 sigmoid>0.5 改为 logits>0.0 (等价但更直接)")
        print("     或在训练初期使用 logits>0.0 作为阈值（已确认当前代码逻辑正确）")
        print("  -> 真正的根因可能是: 随机初始化的 bias 导致 logits 整体偏负")
    elif all_logits_pos:
        print("  [根因定位] 随机初始化模型的分类 logits 全部 > 0")
        print("  -> 二值化后全部为 1 (铁素体) -> pearlite_iou 应为 0 而非 ferrite_iou")
    else:
        print("  [通过] logits 分布正常（有正有负），ferrite_iou 不应为 0.0")
        print("  -> 如果训练后 ferrite_iou 仍为 0.0，问题出在训练过程而非评估逻辑")

    print("\n审计完成。")


if __name__ == "__main__":
    main()