# -*- coding: utf-8 -*-
"""最终任务适配：新GT语义与原人工affinity各自保持独立、空间对齐。"""

from pathlib import Path

import numpy as np
import torch

from data.direct_dual_head_dataset import DirectDualHeadDataset
from data.semantic_targets import (
    load_completed_semantic_source,
    semantic_targets_from_instances,
)
from utils.offset_letterbox import letterbox_instance_geometry


class Stage1FinalTaskDataset(DirectDualHeadDataset):
    """语义全部从补缝实例派生；affinity仍用原LabelMe，保持V的数据口径。"""

    def __init__(self, *args, completed_gt_dir, augment=False, **kwargs):
        super().__init__(*args, augment=False, **kwargs)
        self.task_augment = bool(augment)
        self.completed_gt_dir = Path(completed_gt_dir)
        for image_path, _ in self.samples:
            for suffix in ("_gt.npz", "_class.json"):
                path = self.completed_gt_dir / (image_path.stem + suffix)
                if not path.is_file():
                    raise FileNotFoundError(path)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image_path, _ = self.samples[int(index)]
        # 读取原尺寸，不从旧语义/实例图生成任何新语义辅助监督。
        import cv2
        shape = cv2.imread(str(image_path), cv2.IMREAD_COLOR).shape[:2]
        instances, lookup = load_completed_semantic_source(
            self.completed_gt_dir / (image_path.stem + "_gt.npz"), shape
        )
        instances, valid, _ = letterbox_instance_geometry(
            instances, input_size=self.image_size, output_grid=self.image_size
        )
        instances[~valid.astype(bool)] = 0
        semantic, boundary = semantic_targets_from_instances(instances, lookup)
        sample["semantic_target"] = torch.from_numpy(semantic).unsqueeze(0)
        sample["semantic_boundary"] = torch.from_numpy(boundary).unsqueeze(0)
        sample["semantic_instance_map"] = torch.from_numpy(instances.astype(np.int64))
        sample["semantic_valid_content"] = torch.from_numpy(
            valid.astype(bool) & (instances > 0)
        ).unsqueeze(0)
        return self._augment(sample) if self.task_augment else sample
