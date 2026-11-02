# -*- coding: utf-8 -*-
"""冻结模型教师实例监督；与人工补缝 GT 的来源字段严格分开。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

FORMAT = "teacher_supervision_v1"
TEACHER_KEYS = {"instance_map", "teacher_known", "teacher_unknown"}
BOUNDARY_POLICIES = {"known_neighbors_v1", "explicit_separators_v1"}


def is_teacher_supervision(path):
    with np.load(path, allow_pickle=False) as payload:
        return bool({"teacher_known", "teacher_unknown"} & set(payload.files))


def load_teacher_instance_source(gt_path, image_shape=None, class_path=None):
    """返回 ids(int32)、classes(int→int)、known、unknown、来源摘要。"""
    path = Path(gt_path)
    with np.load(path, allow_pickle=False) as payload:
        if not TEACHER_KEYS <= set(payload.files):
            raise ValueError(f"teacher target requires {sorted(TEACHER_KEYS)}: {path}")
        if {"original_covered", "filled", "residual_unknown"} & set(payload.files):
            raise ValueError("teacher supervision must not claim manual completion provenance")
        ids = payload["instance_map"].copy()
        known, unknown = payload["teacher_known"].copy(), payload["teacher_unknown"].copy()
    if ids.dtype != np.uint16 or ids.ndim != 2 or not ids.size:
        raise ValueError("teacher instance_map must be nonempty HW uint16")
    if image_shape is not None and tuple(ids.shape) != tuple(image_shape):
        raise ValueError(f"teacher target/image shape mismatch: {path}")
    for mask in (known, unknown):
        if mask.shape != ids.shape or not np.isin(mask, (0, 1)).all():
            raise ValueError("teacher masks must be binary HW matching instance_map")
    known, unknown = known.astype(bool), unknown.astype(bool)
    if not np.array_equal(known, ids > 0) or not np.array_equal(unknown, ~known):
        raise ValueError("teacher known/unknown must equal positive IDs / ID0")
    if not known.any():
        raise ValueError("teacher target has no known instances")
    if class_path is None:
        class_path = path.with_name(path.name.removesuffix("_gt.npz") + "_class.json")
    raw = json.loads(Path(class_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(
        not isinstance(k, str) or not k.isdecimal() or str(int(k)) != k or int(k) <= 0
        or type(v) is not int or v not in (0, 1) for k, v in raw.items()
    ):
        raise ValueError("teacher classes require canonical positive ID keys and integer P0/F1")
    classes = {int(k): v for k, v in raw.items()}
    if set(classes) != set(map(int, np.unique(ids))) - {0}:
        raise ValueError("teacher classes and positive IDs must match exactly")
    return ids.astype(np.int32), classes, known, unknown, dict(
        format=FORMAT, source_kind="model_teacher", known_pixels=int(known.sum()),
        unknown_pixels=int(unknown.sum()), instance_count=len(classes), ID0="ignore")


def teacher_boundary_targets(instances, valid_margin=1, policy="known_neighbors_v1"):
    """边界监督域可显式包含A-0-B单像素分界；不填ID、不赋予零线相别。"""
    if policy not in BOUNDARY_POLICIES:
        raise ValueError(f"unsupported teacher boundary policy: {policy}")
    if type(valid_margin) is not int or valid_margin < 0:
        raise ValueError("valid_margin must be a nonnegative integer")
    if instances.ndim != 2 or not instances.size or instances.dtype.kind not in "iu" or np.any(instances < 0):
        raise ValueError("teacher boundary target requires a nonnegative HW integer instance map")
    known = instances > 0
    edge = np.zeros(instances.shape, dtype=bool)
    h = (instances[:, 1:] != instances[:, :-1]) & known[:, 1:] & known[:, :-1]
    v = (instances[1:, :] != instances[:-1, :]) & known[1:, :] & known[:-1, :]
    edge[:, 1:] |= h
    edge[:, :-1] |= h
    edge[1:, :] |= v
    edge[:-1, :] |= v
    seams = np.zeros(instances.shape, dtype=bool)
    if policy == "explicit_separators_v1":
        seams[:, 1:-1] |= ((instances[:, 1:-1] == 0) & known[:, :-2] & known[:, 2:]
                            & (instances[:, :-2] != instances[:, 2:]))
        seams[1:-1, :] |= ((instances[1:-1, :] == 0) & known[:-2, :] & known[2:, :]
                            & (instances[:-2, :] != instances[2:, :]))
    geometry_known = known | seams
    boundary = edge | seams
    if valid_margin:
        width = 2 * valid_margin + 1
        valid = cv2.erode(geometry_known.astype(np.uint8), np.ones((width, width), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    else:
        valid = geometry_known.copy()
        valid[[0, -1], :] = False
        valid[:, [0, -1]] = False
    metadata = dict(source_kind="model_teacher", policy=policy, valid_margin=valid_margin,
        semantic_known_pixels=int(known.sum()), semantic_unknown_pixels=int((~known).sum()),
        separator_pixels=int(seams.sum()), geometry_known_pixels=int(geometry_known.sum()),
        boundary_pixels=int(boundary.sum()), valid_pixels=int(valid.sum()),
        valid_boundary_pixels=int((boundary & valid).sum()),
        ID0_semantic="ignore", ID0_affinity_endpoint="ignore",
        ID0_boundary="explicit_axis_single_pixel_different_ID_separator_only" if policy == "explicit_separators_v1" else "ignore")
    return boundary, valid, metadata


def teacher_targets(instances, classes, *, center_sigma=0., boundary_policy="known_neighbors_v1"):
    """分别派生相别与边界监督域；零ID始终不获得相别或affinity端点标签。"""
    known = instances > 0
    lookup = np.full(max(int(instances.max()), max(classes, default=0)) + 1, -1, dtype=np.float32)
    for key, value in classes.items():
        lookup[key] = value
    semantic = lookup[instances]
    edge, valid, _ = teacher_boundary_targets(instances, policy=boundary_policy)
    boundary = edge.astype(np.float32)
    boundary[~valid] = -1
    center = np.zeros(instances.shape, dtype=np.float32)
    if center_sigma > 0:
        radius = max(1, int(np.ceil(3 * center_sigma)))
        for key in np.unique(instances):
            if key == 0:
                continue
            distance = cv2.distanceTransform((instances == key).astype(np.uint8), cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(distance.argmax()), distance.shape)
            y0, y1 = max(0, y - radius), min(center.shape[0], y + radius + 1)
            x0, x1 = max(0, x - radius), min(center.shape[1], x + radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            gaussian = np.exp(-((yy-y)**2 + (xx-x)**2) / (2 * center_sigma**2))
            center[y0:y1, x0:x1] = np.maximum(center[y0:y1, x0:x1], gaussian)
    center[~known] = -1
    return semantic, boundary, edge.astype(np.float32), center


def teacher_sample(dataset, image_path, gt_path, *, with_instances=False):
    """复用旧图像缩放/裁窗/D4，所有监督只由教师实例派生。"""
    from data.dataset import letterbox, letterbox_mask, compute_boundary_weight, random_crop
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = image.shape[:2]
    ids, classes, _, _, provenance = load_teacher_instance_source(gt_path, shape)
    image, scale, pad_h, pad_w = letterbox(image, dataset.image_size)
    ids, _, _, _ = letterbox_mask(ids, dataset.image_size)
    ids = ids.astype(np.int32)
    if pad_h:
        ids[-pad_h:, :] = 0
    if pad_w:
        ids[:, -pad_w:] = 0
    semantic, boundary, core, center = teacher_targets(ids, classes,
        center_sigma=dataset.center_sigma if dataset.teacher_center_supervision else 0.,
        boundary_policy=dataset.teacher_boundary_policy)
    weight = compute_boundary_weight(core, scale_factor=dataset.boundary_scale_factor,
        weight_floor=dataset.boundary_weight_floor, weight_ceil=dataset.boundary_weight_ceil)
    if dataset.augment and dataset.crop_size > 0:
        if dataset.crop_size > dataset.image_size:
            raise ValueError("teacher crop must fit the letterbox grid; padding is ignored, not mirrored supervision")
        image, semantic, boundary, weight, center, ids = random_crop(
            image, semantic, boundary, weight, center, dataset.crop_size, instance_map=ids)
    if dataset.augment:
        if with_instances:
            image, semantic, boundary, weight, center, ids = dataset._augment(
                image, semantic, boundary, weight, center, ids)
        else:
            image, semantic, boundary, weight, center = dataset._augment(image, semantic, boundary, weight, center)
    item = dict(image=torch.from_numpy(np.ascontiguousarray(image)).float().permute(2, 0, 1) / 255.,
        target=torch.from_numpy(np.stack([semantic, boundary, center]).copy()).float(),
        weight=torch.from_numpy(np.ascontiguousarray(weight)).float().unsqueeze(0),
        original_size=shape, scale=scale, pad_h=pad_h, pad_w=pad_w,
        image_path=str(image_path), source_crop_size=0, requested_source_crop_size=0,
        supervision_source=FORMAT)
    item["teacher_boundary_policy"] = dataset.teacher_boundary_policy
    if with_instances:
        item["instance_map"] = torch.from_numpy(np.ascontiguousarray(ids)).long()
    return item
