# -*- coding: utf-8 -*-
"""从人工补缝或显式教师实例派生语义；来源分开，未知语义使用 -1。"""

import json
from pathlib import Path

import numpy as np

from data.offset_geometry_dataset import load_completed_instance_map
from data.teacher_supervision import is_teacher_supervision, load_teacher_instance_source


def load_completed_semantic_source(path, image_shape):
    path = Path(path)
    if is_teacher_supervision(path):
        instances, classes, _, _, _ = load_teacher_instance_source(path, image_shape)
        lookup = np.full(int(instances.max()) + 1, -1, dtype=np.float32)
        for key, value in classes.items():
            lookup[key] = value
        return instances, lookup
    instances = load_completed_instance_map(path, image_shape)
    class_path = path.with_name(path.name.removesuffix("_gt.npz") + "_class.json")
    with class_path.open(encoding="utf-8") as stream:
        classes = {int(key): value for key, value in json.load(stream).items()}
    ids = set(int(value) for value in np.unique(instances) if value > 0)
    if not ids.issubset(classes) or any(classes[value] not in (0, 1) for value in ids):
        raise ValueError(f"completed GT needs class 0/1 for every instance: {class_path}")
    lookup = np.full(int(instances.max()) + 1, -1, dtype=np.float32)
    for instance_id in ids:
        lookup[instance_id] = classes[instance_id]
    return instances, lookup


def semantic_targets_from_instances(instances, class_lookup, *, known_boundaries_only=False):
    """在最终网格派生目标；边缘只用于排除 core，不是边界分支的监督。"""
    semantic = class_lookup[instances]
    boundary = np.zeros(instances.shape, dtype=bool)
    horizontal = instances[:, 1:] != instances[:, :-1]
    vertical = instances[1:, :] != instances[:-1, :]
    if known_boundaries_only:
        horizontal &= (instances[:, 1:] > 0) & (instances[:, :-1] > 0)
        vertical &= (instances[1:, :] > 0) & (instances[:-1, :] > 0)
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary &= instances > 0
    return semantic, boundary.astype(np.float32)
