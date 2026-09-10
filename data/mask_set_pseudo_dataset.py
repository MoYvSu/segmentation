# -*- coding: utf-8 -*-
"""主线生成的类别感知伪实例；与人工 GT 分开存储、采样和记录。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.dataset import letterbox
from data.direct_dual_head_dataset import DirectDualHeadDataset, semantic_from_instance_classes
from utils.instance_metrics import load_class_map
from utils.offset_letterbox import letterbox_instance_geometry


PSEUDO_FORMAT = "mainline_pseudo_instances_v1"
MAX_PSEUDO_IMAGES = 249


def read_pseudo_manifest(directory, raw_dir, *, excluded_names, query_capacity):
    directory, raw_dir = Path(directory), Path(raw_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != PSEUDO_FORMAT or manifest.get("label_source") != "mainline_pseudo":
        raise ValueError("expected explicitly sourced mainline pseudo labels")
    names = manifest.get("samples", [])
    if not names or len(names) != len(set(names)) or not len(names) <= MAX_PSEUDO_IMAGES:
        raise ValueError("pseudo samples must be unique and contain 1..249 images")
    if manifest.get("sample_count") != len(names) or manifest.get("status") != "complete":
        raise ValueError("pseudo manifest is incomplete")
    overlap = set(names) & {Path(n).stem for n in excluded_names}
    if overlap:
        raise ValueError(f"pseudo training leaks excluded images: {sorted(overlap)}")
    rows = manifest["images"]
    if {r["stem"] for r in rows} != set(names) or len(rows) != len(names):
        raise ValueError("pseudo manifest image records mismatch")
    hashes = [r["source_sha256"] for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate source image content in pseudo dataset")
    classes = {}
    for row in rows:
        stem = row["stem"]
        filename = row["image_name"]
        if Path(filename).name != filename or Path(filename).stem != stem:
            raise ValueError("invalid pseudo image filename")
        if not (raw_dir / filename).is_file():
            raise FileNotFoundError(raw_dir / filename)
        mask = cv2.imread(str(directory / "instances" / f"{stem}_inst.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.dtype != np.uint16 or mask.ndim != 2:
            raise ValueError(f"pseudo instance PNG must be single-channel uint16: {stem}")
        if list(mask.shape) != row["native_shape"]:
            raise ValueError(f"pseudo native shape mismatch: {stem}")
        cm = load_class_map(directory / "instances" / f"{stem}_class.json")
        semantic_from_instance_classes(mask, cm)
        if not cm or len(cm) > int(query_capacity):
            raise ValueError(f"pseudo target exceeds query capacity: {stem} {len(cm)}")
        classes[stem] = cm
    return manifest, classes


class MainlinePseudoDataset(Dataset):
    """只读最终实例 PNG；id=0 保持 unknown，无需伪造 LabelMe 或 manual_target_v2。"""

    def __init__(self, raw_dir, directory, manifest, *, image_size=1024, mask_grid=512,
                 augment=False, augmentation=None):
        self.raw_dir, self.directory = Path(raw_dir), Path(directory)
        self.rows = sorted(manifest["images"], key=lambda row: row["stem"])
        self.image_size, self.mask_grid = int(image_size), int(mask_grid)
        self.augment, self.augmentation = bool(augment), augmentation or {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[int(index)]
        bgr = cv2.imread(str(self.raw_dir / row["image_name"]), cv2.IMREAD_COLOR)
        instances = cv2.imread(str(self.directory / "instances" / f"{row['stem']}_inst.png"),
                               cv2.IMREAD_UNCHANGED)
        if bgr is None or instances is None or bgr.shape[:2] != instances.shape:
            raise ValueError(f"pseudo image/target shape mismatch: {row['stem']}")
        rgb, _, _, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.image_size)
        grid, content, _ = letterbox_instance_geometry(instances, self.image_size, self.mask_grid)
        sample = {
            "image": torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.,
            "affinity_instance_map": torch.from_numpy(grid.astype(np.int64)),
            "affinity_valid_content": torch.from_numpy(content & (grid > 0)).unsqueeze(0),
            "image_name": row["image_name"],
        }
        if self.augment:
            # 复用既有在线空间/外观增强，图像与标签接受同一个变换。
            sample = DirectDualHeadDataset._augment(self, sample)
        return sample


class MaskSetSampleView(Dataset):
    """混合数据仅返回任务所需字段，避免把伪标签包装成人工来源。"""

    def __init__(self, dataset, source, loss_weight=1.):
        self.dataset, self.source, self.loss_weight = dataset, str(source), float(loss_weight)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        result = {k: sample[k] for k in ("image", "image_name", "affinity_instance_map", "affinity_valid_content")}
        result.update(label_source=self.source, sample_loss_weight=self.loss_weight)
        return result


class FixedSourceRatioSampler(Sampler):
    """每轮固定人工/伪标签抽样数；独立 RNG 保持两组对照的样本顺序一致。"""

    def __init__(self, manual_count, pseudo_count, manual_samples, pseudo_samples, seed):
        values = (manual_count, pseudo_count, manual_samples, pseudo_samples)
        if any(int(v) != v or v < 1 for v in values):
            raise ValueError("source counts and samples must be positive integers")
        self.manual_count, self.pseudo_count, self.manual_samples, self.pseudo_samples = map(int, values)
        self.manual_rng = torch.Generator().manual_seed(int(seed))
        self.pseudo_rng = torch.Generator().manual_seed(int(seed) + 1)

    def __len__(self):
        return self.manual_samples + self.pseudo_samples

    def __iter__(self):
        manual = torch.randint(self.manual_count, (self.manual_samples,), generator=self.manual_rng).tolist()
        pseudo = (torch.randint(self.pseudo_count, (self.pseudo_samples,), generator=self.pseudo_rng)
                  + self.manual_count).tolist()
        # 均匀穿插；同一人工批次后的伪标签数可相差一，不改变总数。
        indices = []
        for i, sample in enumerate(manual):
            indices.append(sample)
            indices.extend(pseudo[i * len(pseudo) // len(manual):(i + 1) * len(pseudo) // len(manual)])
        return iter(indices)
