# -*- coding: utf-8 -*-
"""低成本诊断合同：有效连接、融合数值、正式种子与几何计数。"""

import cv2
import numpy as np
import pytest
import torch

from tools.diagnose_affinity_readout import (
    decomposed_fusion,
    diagnostic_markers,
    raw_statistics,
    spatial_domains,
    topology,
)
from utils.affinity_deployment import postprocess, probability_to_logit
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
from utils.affinity_loss import build_affinity_targets_torch


FUSION = dict(
    distance2_weight=0.50,
    distance4_weight=0.25,
    support_threshold=0.20,
    support_temperature=0.05,
)


def test_direction_statistics_exclude_unknown_and_padding_and_partition_provenance():
    gt = np.zeros((6, 8), dtype=np.int32)
    gt[:, :3], gt[:, 3:6] = 1, 2
    gt[2, 1] = 0  # 内容中的 unknown。
    content = np.ones(gt.shape, bool)
    content[:, 7] = False  # 最后一列是 padding，前一列是 unknown。
    targets, edge_valid = build_affinity_targets_torch(
        torch.from_numpy(gt)[None], torch.from_numpy(content)[None]
    )
    targets = targets[0].numpy().astype(bool)
    edge_valid = edge_valid[0].numpy()
    expected_valid = np.zeros_like(edge_valid)
    expected_inside = np.zeros_like(targets)
    # 独立按像素端点定义检查方向和边缘裁切，不调用生产切片辅助函数。
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        for y in range(gt.shape[0]):
            for x in range(gt.shape[1]):
                yy, xx = y + dy, x + dx
                if not (0 <= yy < gt.shape[0] and 0 <= xx < gt.shape[1]):
                    continue
                allowed = content[y, x] and content[yy, xx] and gt[y, x] > 0 and gt[yy, xx] > 0
                expected_valid[channel, y, x] = allowed
                expected_inside[channel, y, x] = allowed and gt[y, x] == gt[yy, xx]
    np.testing.assert_array_equal(edge_valid, expected_valid)
    np.testing.assert_array_equal(targets, expected_inside)

    probability = np.where(targets, 0.90, 0.10).astype(np.float32)
    probability[~edge_valid] = 0.99  # 如果 ignore 泄漏，会明显污染结果。
    probability[0, 0, 2] = 0.85  # 一个真实跨界连接被错误预测为高 affinity。
    filled = (gt > 0) & (np.indices(gt.shape)[1] == 3)
    original = (gt > 0) & ~filled
    statistics = raw_statistics(probability, targets, edge_valid, original, filled)
    assert statistics[0]["inside"]["n"] == 22
    assert statistics[0]["cross"]["n"] == 6
    assert statistics[0]["inside"]["mean"] == pytest.approx(0.90)
    assert statistics[0]["cross"]["mean"] == pytest.approx((0.85 + 5 * 0.10) / 6)
    assert statistics[0]["cross"]["gt08"] == pytest.approx(1 / 6)
    for item in statistics:
        for kind in ("inside", "cross"):
            assert sum(
                item["provenance"][f"{group}_{kind}"]["n"]
                for group in ("original_original", "original_filled", "filled_filled")
            ) == item[kind]["n"]


def test_spatial_domains_do_not_call_unknown_a_boundary_or_near_edge_an_interior():
    gt = np.ones((28, 32), np.int32)
    gt[:, 16:] = 2
    gt[3:7, 3:7] = 0
    gt[-2:] = 0
    valid = gt > 0
    interface, interior = spatial_domains(gt, valid)
    expected = np.zeros(gt.shape, bool)
    expected[:26, 15:17] = True
    np.testing.assert_array_equal(interface, expected)
    assert interior[13, 8]
    assert not interior[13, 14]  # 离晶界不足 4px。
    assert not interior[8, 4]  # 离 unknown 不足 4px。
    assert not interior[2, 10]  # 离图像上沿不足 4px。
    assert not interior[23, 8]  # 离下方无效内容不足 4px。
    assert not np.any(interior[~valid])


@pytest.mark.parametrize("reduction", ["mean", "top2"])
def test_fusion_decomposition_matches_deployment_and_exposes_long_range_dilution(reduction):
    q = torch.full((1, 8, 13, 17), 0.10)
    q[:, :4, 6, 8] = 0.90
    q[:, 4:, 6, 8] = 0.02
    q[:, 0, 2, 2] = 0.90  # 单方向强响应。
    logits = probability_to_logit(1 - q)
    pieces = decomposed_fusion(logits, reduction, FUSION)
    expected = affinity_boundary_probability(
        logits, mode="gated", short_reduction=reduction, **FUSION
    )
    torch.testing.assert_close(pieces["boundary"], expected, atol=1e-7, rtol=1e-6)
    assert float(pieces["boundary"][0, 0, 6, 8]) < float(pieces["short"][0, 0, 6, 8])
    assert float(pieces["short"][0, 0, 2, 2]) == pytest.approx(0.30 if reduction == "mean" else 0.50)

    # 邻点的 top2 提高 support，能够在当前短程值不变处带入更多低值长程。
    q = torch.full((1, 8, 13, 17), 0.10)
    q[:, 4:] = 0.01
    q[:, 0, 6, 9] = 0.70
    logits = probability_to_logit(1 - q)
    mean = decomposed_fusion(logits, "mean", FUSION)
    top2 = decomposed_fusion(logits, "top2", FUSION)
    assert torch.all(top2["short"] >= mean["short"] - 1e-7)
    assert float(top2["boundary"][0, 0, 6, 8]) < float(mean["boundary"][0, 0, 6, 8])


def test_diagnostic_markers_equal_actual_watershed_input_after_probability_round_trip(monkeypatch, tmp_path):
    cfg = dict(
        boundary_threshold=0.65,
        marker_boundary_low_threshold=0.45,
        marker_boundary_reconstruction_steps=8,
        bridge_width=1,
        watershed_dilate_width=1,
        marker_border_seal_width=2,
        min_instance_area=20,
        max_instance_id=65535,
        semantic_vote_mode="probability_mean",
    )
    probability = np.full((48, 64), 0.05, np.float32)
    probability[:, 31:33] = 0.90  # 强竖线贯穿图像，避免骨架端点收缩引入额外开口。
    probability[22:24, 31:33] = 0.50  # 局部重建可弥合的弱断口。
    probability[6:10, 10:14] = 0.50  # 独立弱响应不应生成阻挡。
    probability[14, 31] = np.float32(0.65)
    probability[15, 31] = np.nextafter(np.float32(0.65), np.float32(1))
    probability[16, 31] = np.nextafter(np.float32(0.65), np.float32(0))
    probability[23, 32] = np.float32(0.45)
    logits = probability_to_logit(torch.from_numpy(probability)[None, None])
    actual_probability = logits.sigmoid()[0, 0].numpy()
    diagnosis = diagnostic_markers(actual_probability, cfg)
    captured = []
    real_watershed = cv2.watershed

    def capture_watershed(image, markers):
        captured.append(markers.copy())
        return real_watershed(image, markers)

    monkeypatch.setattr(cv2, "watershed", capture_watershed)
    semantic = torch.full_like(logits, 2.0)
    _, instances, classes = postprocess(
        torch.cat([semantic, logits], dim=1), probability.shape, tmp_path,
        "marker_contract", cfg, cfg["boundary_threshold"], False,
    )
    assert len(captured) == 1
    np.testing.assert_array_equal(diagnosis["markers"], captured[0])
    assert int(diagnosis["markers"].max()) >= 2
    assert not diagnosis["marker_mask"][6:10, 10:14].any()
    assert instances.dtype == np.uint16
    assert int(instances.max()) <= 65535
    assert classes


def test_geometry_topology_counts_merges_splits_missing_regions_and_ignores_unknown():
    gt = np.ones((8, 8), np.int32)
    valid = np.ones(gt.shape, bool)
    prediction = np.full(gt.shape, 7, np.uint16)
    prediction[:, 4:] = 11
    split = topology(gt, prediction, valid)
    assert split["split_gt_10pct"] == 1
    assert split["merged_pred_10pct"] == 0

    gt[:, :4], gt[:, 4:] = 3, 8
    prediction[:] = 9
    merged = topology(gt, prediction, valid)
    assert merged["merged_pred_10pct"] == 1
    assert merged["split_gt_10pct"] == 0
    assert merged["gt_in_shared_dominant_region"] == 2
    assert merged["non_dominant_gt_fraction"] == pytest.approx(0.50)

    prediction[:] = 0
    missing = topology(gt, prediction, valid)
    assert missing["gt_without_significant_region"] == 2
    assert missing["pred_count"] == 0

    prediction[:, :4], prediction[:, 4:] = 12, 2
    valid[:2, :2] = False
    gt[:2, :2] = 0
    prediction[:2, :2] = 21  # 无效域独立预测不能增加 region 数。
    perfect = topology(gt, prediction, valid)
    assert perfect["gt_count"] == perfect["pred_count"] == 2
    assert perfect["merged_pred_10pct"] == perfect["split_gt_10pct"] == 0
    assert perfect["non_dominant_gt_fraction"] == 0
