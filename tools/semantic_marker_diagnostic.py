# -*- coding: utf-8 -*-
"""只读重现部署的分水岭种子阶段，检查相反语义预测是否已经共用种子。

必须先对整幅原尺寸边界图运行 marker_stages，再裁剪返回数组用于显示或统计。
裁剪后重新求连通域会丢失窗口外的通路，不能用于判断原部署种子是否相连。
本模块不修改输入数组或默认部署；可返回独立的marker诊断候选，不产生测试标签。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from tools.semantic_instance_metrics import instance_bands
from utils.affinity_deployment import probability_to_logit
from utils.post_process import _boundary_skeleton_belt, reconstruct_marker_boundary


def marker_stages(boundary: np.ndarray, config: dict) -> dict:
    """输入 prediction_maps 返回的原尺寸融合边界概率，返回实际部署种子。

    boundary_probability 包含 save_prediction 的 logit 转换及后处理 sigmoid；
    high_mask/marker_boundary_mask 为 uint8 0/1；belt/cores 为 uint8 0/255；
    core_labels 是面积过滤前连通域，marker_labels 是用于 watershed 的 int32 种子。
    当前范围仅限无中心头、无语义边缘增强的部署契约。
    """
    infer = config['inference']
    if (float(infer.get('sem_edge_merge_weight', 0.0)) != 0.0
            or float(infer.get('sem_edge_boost_alpha', 0.0)) != 0.0):
        raise ValueError('Marker diagnostic requires disabled semantic edge augmentation')
    if bool(infer.get('center_seeds', False)) or bool(infer.get('use_center_seeds', False)):
        raise ValueError('Marker diagnostic requires absent center seeds')
    value = np.asarray(boundary, dtype=np.float32)
    if value.ndim != 2 or not value.size or not np.all(np.isfinite(value)):
        raise ValueError('Boundary must be a nonempty finite two-dimensional probability map')
    if np.any((value < 0) | (value > 1)):
        raise ValueError('Boundary probability must lie in [0, 1]')
    # 与 save_prediction -> postprocess 一致；原尺寸输入的第二次插值不改变值。
    probability = torch.sigmoid(probability_to_logit(torch.from_numpy(value))).numpy()
    high = float(infer['boundary_threshold'])
    bridge = int(infer.get('bridge_width', 1))
    dilate = int(infer.get('watershed_dilate_width', 1))
    minimum = int(infer.get('min_instance_area', 50))
    seal = max(0, int(infer.get('marker_border_seal_width', 0)))
    low = infer.get('marker_boundary_low_threshold')
    steps = int(infer.get('marker_boundary_reconstruction_steps', 0))
    high_mask = (probability > high).astype(np.uint8)
    _, barrier_belt = _boundary_skeleton_belt(high_mask, bridge, dilate)
    marker_boundary = high_mask
    marker_belt = barrier_belt
    if low is not None and steps > 0:
        marker_boundary = reconstruct_marker_boundary(probability, float(low), high, steps)
        _, marker_belt = _boundary_skeleton_belt(marker_boundary, bridge, dilate)
    # 有意保持原函数别名行为：未独立重建 marker 时，封边也作用于 barrier_belt。
    if seal > 0:
        seal = min(seal, max(1, min(value.shape) // 2))
        marker_belt[:seal, :] = 255
        marker_belt[-seal:, :] = 255
        marker_belt[:, :seal] = 255
        marker_belt[:, -seal:] = 255
    cores = cv2.bitwise_not(marker_belt)
    num, core_labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
    marker_labels = np.zeros(value.shape, dtype=np.int32)
    count = 0
    for label_id in range(1, num):
        if int(stats[label_id, cv2.CC_STAT_AREA]) < minimum:
            continue
        count += 1
        marker_labels[core_labels == label_id] = count
    return {
        'boundary_probability': probability,
        'high_mask': high_mask,
        'barrier_belt': barrier_belt,
        'marker_boundary_mask': marker_boundary,
        'marker_belt': marker_belt,
        'cores': cores,
        'core_labels': core_labels,
        'marker_labels': marker_labels,
        'marker_count': count,
        'resolved': {
            'boundary_threshold': high, 'bridge_width': bridge,
            'watershed_dilate_width': dilate, 'min_instance_area': minimum,
            'marker_border_seal_width': seal, 'marker_boundary_low_threshold': low,
            'marker_boundary_reconstruction_steps': steps,
            'probability_roundtrip': 'native -> probability_to_logit -> CPU sigmoid',
        },
    }


def anchored_marker_stages(stages: dict) -> dict:
    """只读候选：在 marker 桥接/骨架化前增加同宽封边框，保留原来的后置封边。

    唯一变化是让触及图边的粗边界先与封边框连成整体，再经过既有骨架流程，
    避免独立线段末端收缩后绕过封边框。此函数不改真实 watershed barrier，
    不写部署配置；用于固定代表点及完整输出反事实对照，不能据此宣称分割准确率提升。
    返回结构兼容 marker_stages；所有改写数组新建，输入 stages 保持不变。
    """
    resolved = stages['resolved']
    source = np.asarray(stages['marker_boundary_mask'], dtype=np.uint8).copy()
    seal = max(0, int(resolved['marker_border_seal_width']))
    if seal > 0:
        seal = min(seal, max(1, min(source.shape) // 2))
        source[:seal, :] = 1
        source[-seal:, :] = 1
        source[:, :seal] = 1
        source[:, -seal:] = 1
    _, marker_belt = _boundary_skeleton_belt(source, int(resolved['bridge_width']),
                                            int(resolved['watershed_dilate_width']))
    if seal > 0:
        marker_belt[:seal, :] = 255
        marker_belt[-seal:, :] = 255
        marker_belt[:, :seal] = 255
        marker_belt[:, -seal:] = 255
    cores = cv2.bitwise_not(marker_belt)
    num, core_labels, stats, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)
    marker_labels = np.zeros(source.shape, dtype=np.int32)
    count = 0
    for label_id in range(1, num):
        if int(stats[label_id, cv2.CC_STAT_AREA]) < int(resolved['min_instance_area']):
            continue
        count += 1
        marker_labels[core_labels == label_id] = count
    return {
        **stages,
        'marker_boundary_mask': source,
        'marker_belt': marker_belt,
        'cores': cores,
        'core_labels': core_labels,
        'marker_labels': marker_labels,
        'marker_count': count,
        'resolved': {**resolved, 'diagnostic_variant': 'marker_preseal_v1'},
        'diagnostic_only': True,
        'real_barrier_changed': False,
    }


def component_seed_relation(mask: np.ndarray, probability: np.ndarray,
                            marker_labels: np.ndarray, *, bands: dict | None = None) -> dict:
    """检查内部两类最大强预测连通块是否主要位于同一个已有种子。

    mask/probability/marker_labels 可以同框裁剪，但 marker_labels 必须来自整图。
    内部定义沿用 instance_bands。dominant_fraction 的分母包含 marker=0 像素；
    每类最大连通块均 >=32像素、>=内部10%，且各自 >=90% 位于同一个正种子时，
    same_seed 才为 True；充分覆盖但不同种子为 False，其余情况为 None。
    相反预测只是结构线索，不代表两块真实组织类别不同。
    """
    bands = instance_bands(mask) if bands is None else bands
    probability = np.asarray(probability)
    marker_labels = np.asarray(marker_labels)
    if probability.shape != mask.shape or marker_labels.shape != mask.shape:
        raise ValueError('Mask, probability, and marker labels must have the same shape')
    if not np.issubdtype(marker_labels.dtype, np.integer) or np.any(marker_labels < 0):
        raise ValueError('Marker labels must be nonnegative integers')
    if not np.all(np.isfinite(probability[mask])) or np.any((probability[mask] < 0) | (probability[mask] > 1)):
        raise ValueError('Instance probabilities must be finite and lie in [0, 1]')
    interior = bands['interior']
    interior_area = int(interior.sum())

    def largest(selected):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), connectivity=8)
        if len(stats) <= 1:
            return {'area': 0, 'interior_fraction': 0.0, 'qualifies': False,
                    'dominant_marker': None, 'dominant_pixels': 0, 'dominant_fraction': 0.0,
                    'unseeded_pixels': 0, 'positive_marker_count': 0}
        iid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = labels == iid
        area = int(stats[iid, cv2.CC_STAT_AREA])
        ids, counts = np.unique(marker_labels[component], return_counts=True)
        positive = ids > 0
        positive_ids, positive_counts = ids[positive], counts[positive]
        index = int(np.argmax(positive_counts)) if positive_counts.size else None
        dominant = int(positive_ids[index]) if index is not None else None
        dominant_pixels = int(positive_counts[index]) if index is not None else 0
        fraction = area / max(1, interior_area)
        return {
            'area': area, 'interior_fraction': fraction,
            'qualifies': area >= 32 and fraction >= .1,
            'dominant_marker': dominant, 'dominant_pixels': dominant_pixels,
            'dominant_fraction': dominant_pixels / area,
            'unseeded_pixels': int(counts[ids == 0].sum()),
            'positive_marker_count': int(positive_ids.size),
        }

    ferrite = largest(interior & (probability >= .8))
    pearlite = largest(interior & (probability <= .2))
    both_covered = all(c['qualifies'] and c['dominant_fraction'] >= .9
                       for c in (ferrite, pearlite))
    return {
        'interior_area': interior_area,
        'interior_distance_cutoff': bands['interior_distance_cutoff'],
        'ferrite_component': ferrite, 'pearlite_component': pearlite,
        'same_seed': bool(ferrite['dominant_marker'] == pearlite['dominant_marker'])
        if both_covered else None,
        'dominant_fraction_threshold': .9,
        'interpretation': 'prediction-component connectivity only; no ground truth inferred',
    }


def connectivity_maps(stages: dict) -> dict:
    """每图一次计算各阶段的全图8连通标签，供多个实例复用。

    单独保留纯骨架、骨架膨胀后但封边前两阶段，从而避免把骨架化、膨胀和
    封边的作用混为一谈。返回 {阶段名: {labels: int32数组, component_count: int}}。
    labels 不按面积过滤，也不是最终实例ID；每张图完成后即可释放该缓存。
    """
    bridge = int(stages['resolved']['bridge_width'])
    dilate = int(stages['resolved']['watershed_dilate_width'])
    marker_boundary = np.asarray(stages['marker_boundary_mask'])
    bridged = (marker_boundary > 0).astype(np.uint8) * 255
    if bridge > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
        bridged = cv2.dilate(bridged, kernel)
    skeleton, _ = _boundary_skeleton_belt(marker_boundary, bridge, 0)
    before_seal = skeleton
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        before_seal = cv2.dilate(skeleton, kernel)
    barriers = {
        'high_mask': stages['high_mask'],
        'marker_boundary_mask': marker_boundary,
        'bridged_marker_mask': bridged,
        'skeleton_only': skeleton,
        'dilated_skeleton_before_seal': before_seal,
        'marker_belt': stages['marker_belt'],
    }
    result = {}
    for name, barrier in barriers.items():
        if np.asarray(barrier).shape != marker_boundary.shape:
            raise ValueError('All connectivity stages must retain full image dimensions')
        count, labels = cv2.connectedComponents((np.asarray(barrier) == 0).astype(np.uint8), connectivity=8)
        result[name] = {'labels': labels, 'component_count': int(count - 1)}
    return result


def connectivity_stages(stages: dict, mask_full: np.ndarray,
                        probability_full: np.ndarray, *, maps: dict | None = None) -> dict:
    """用固定两个内部代表点追踪阈值、重建、桥接及骨架后的全图连通性。

    每类在其最大强预测连通块中选择离边缘最远的点；若该块与已过滤种子有
    重叠，优先在这些正种子像素内取点。两点一经选定，在全部阶段保持不变。
    连通域按全图8连通计算，不进行面积过滤，因而这里的标签不是部署实例ID。
    同通路只说明当前预测与边界的拓扑关系，不证明真实类别或边界有误。
    """
    mask_full = np.asarray(mask_full)
    probability = np.asarray(probability_full)
    marker_labels = np.asarray(stages['marker_labels'])
    if probability.shape != mask_full.shape or marker_labels.shape != mask_full.shape:
        raise ValueError('Connectivity diagnostic requires full-image arrays with identical shape')
    if mask_full.ndim != 2 or mask_full.dtype != np.bool_ or not mask_full.any():
        raise ValueError('Instance mask must be a nonempty two-dimensional bool array')
    ys = np.flatnonzero(mask_full.any(axis=1))
    xs = np.flatnonzero(mask_full.any(axis=0))
    y0, x0 = int(ys[0]), int(xs[0])
    crop = (slice(y0, int(ys[-1]) + 1), slice(x0, int(xs[-1]) + 1))
    # 实例距离与连通块只在紧包围框中计算；缓存标签仍来自整幅图。
    # instance_bands 自带零边框，故与原整图实例的距离定义一致。
    bands = instance_bands(mask_full[crop])
    probability = probability[crop]
    local_markers = marker_labels[crop]
    interior = bands['interior']
    interior_area = int(interior.sum())

    def representative(selected):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), connectivity=8)
        if len(stats) <= 1:
            return None
        iid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = labels == iid
        area = int(stats[iid, cv2.CC_STAT_AREA])
        supported = component & (local_markers > 0)
        has_support = bool(supported.any())
        eligible = supported if has_support else component
        distance = cv2.distanceTransform(np.pad(eligible.astype(np.uint8), 1),
                                         cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        fraction = area / max(1, interior_area)
        return {
            'x': int(x) + x0, 'y': int(y) + y0, 'component_area': area,
            'interior_fraction': fraction, 'qualifies': area >= 32 and fraction >= .1,
            'selected_from_positive_marker': has_support,
            'final_marker_id': int(local_markers[y, x]),
            'distance_to_selection_edge': float(distance[y, x]),
        }

    points = {'ferrite': representative(interior & (probability >= .8)),
              'pearlite': representative(interior & (probability <= .2))}
    maps = connectivity_maps(stages) if maps is None else maps
    result = {}
    for name, entry in maps.items():
        labels = entry['labels']
        if labels.shape != mask_full.shape:
            raise ValueError('All connectivity stages must retain full image dimensions')
        labels_at_points = {phase: int(labels[point['y'], point['x']]) if point is not None else None
                           for phase, point in points.items()}
        usable = all(label is not None and label > 0 for label in labels_at_points.values())
        same = labels_at_points['ferrite'] == labels_at_points['pearlite'] if usable else None
        result[name] = {
            'component_count': int(entry['component_count']),
            'ferrite_label': labels_at_points['ferrite'],
            'pearlite_label': labels_at_points['pearlite'],
            'same_component': same,
            'relation': ('same' if same else 'different') if same is not None else None,
        }
    return {
        'points': points, 'stages': result, 'interior_area': interior_area,
        'qualifying_components': all(point is not None and point['qualifies'] for point in points.values()),
        'interpretation': 'fixed predicted-component centers and full-image connectivity; no ground truth inferred',
    }
