# -*- coding: utf-8 -*-
"""从同一份补缝实例 GT 派生语义与内部区域；未知语义使用 -1。"""

import json
from pathlib import Path

import numpy as np

from data.offset_geometry_dataset import load_completed_instance_map


def load_completed_semantic_source(path, image_shape):
    path = Path(path)
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


def semantic_targets_from_instances(instances, class_lookup):
    """在最终网格派生目标；边缘只用于排除 core，不是边界分支的监督。"""
    semantic = class_lookup[instances]
    boundary = np.zeros(instances.shape, dtype=bool)
    horizontal = instances[:, 1:] != instances[:, :-1]
    vertical = instances[1:, :] != instances[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    boundary &= instances > 0
    return semantic, boundary.astype(np.float32)
