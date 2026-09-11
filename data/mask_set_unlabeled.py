# -*- coding: utf-8 -*-
"""一致性训练只读图像池；人工划分、留出图及重复内容在入口排除。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox
from utils.offset_letterbox import geometry_letterbox_metadata


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_unlabeled_pool(image_dir, excluded_names, *, excluded_image_dir=None):
    """内容摘要用于防止换名的留出图泄漏，不依赖清晰度或模型分数挑选。"""
    image_dir = Path(image_dir)
    excluded = {Path(name).stem for name in excluded_names}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths or any(not p.stem.startswith("train_") for p in paths):
        raise ValueError("consistency pool must contain only competition train_* images")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("duplicate image stems in consistency pool")
    rows = [{"stem": p.stem, "image_name": p.name, "source_sha256": image_digest(p)} for p in paths]
    banned_hashes = {row["source_sha256"] for row in rows if row["stem"] in excluded}
    if excluded_image_dir is not None:
        for path in sorted(Path(excluded_image_dir).iterdir()):
            if path.stem in excluded and path.suffix.lower() in IMAGE_SUFFIXES:
                banned_hashes.add(image_digest(path))
    seen, accepted, rejected = set(), [], []
    for row in rows:
        digest = row["source_sha256"]
        reason = ("excluded_name" if row["stem"] in excluded else
                  "excluded_content" if digest in banned_hashes else
                  "duplicate_content" if digest in seen else None)
        if reason:
            rejected.append({**row, "reason": reason})
        else:
            accepted.append(row)
            seen.add(digest)
    if not accepted:
        raise ValueError("no eligible unlabeled training images")
    return {"format": "mask_set_unlabeled_pool_v1", "label_source": "online_ema_consistency",
            "sample_count": len(accepted), "images": accepted, "excluded": rejected,
            "excluded_names": sorted(excluded), "selection": "all eligible unique training images; no labels saved"}


def disjoint_pseudo_subset(source_dir, image_dir, manual_dir, split, holdout_names, output_dir):
    """复用原PNG，只写排除人工/留出别名后的清单；原始伪标签归档保持不变。"""
    source_dir, image_dir, manual_dir, output_dir = map(Path, (source_dir, image_dir, manual_dir, output_dir))
    original = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    excluded_names = set(split["train"] + split["val"]) | {Path(n).stem for n in holdout_names}
    banned = {}
    # 必须同时检查人工目录的真实内容；两个目录的同名图像未必相同。
    for directory in (image_dir, manual_dir):
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.stem in excluded_names:
                banned.setdefault(image_digest(path), []).append({"stem": path.stem,
                    "source": "manual" if directory == manual_dir else "unlabeled",
                    "manual_validation": directory == manual_dir and path.stem in split["val"]})
    removed, kept = [], []
    for row in original["images"]:
        aliases = banned.get(row["source_sha256"], [])
        if row["stem"] in excluded_names or aliases:
            removed.append({"stem": row["stem"], "source_sha256": row["source_sha256"], "aliases": aliases})
        else:
            kept.append(row)
    if not kept:
        raise ValueError("no pseudo labels remain after content isolation")
    names = {row["stem"] for row in kept}
    audit = {"format": "manual_content_disjoint_v1", "source_manifest_sha256": image_digest(source_dir / "manifest.json"),
             "original_count": len(original["images"]), "retained_count": len(kept), "excluded_images": removed,
             "exposed_validation_names": sorted({alias["stem"] for row in removed for alias in row["aliases"] if alias["manual_validation"]})}
    manifest = {**original, "sample_count": len(kept), "images": kept,
                "samples": [name for name in original["samples"] if name in names], "content_isolation": audit}
    if output_dir.exists():
        saved = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if saved != manifest or (output_dir / "instances").resolve() != (source_dir / "instances").resolve():
            raise ValueError("preserving a different existing pseudo subset")
    else:
        output_dir.mkdir(parents=True)
        (output_dir / "instances").symlink_to((source_dir / "instances").resolve(), target_is_directory=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


class MaskSetUnlabeledDataset(Dataset):
    def __init__(self, image_dir, manifest, *, input_size=1024, mask_grid=512):
        self.image_dir = Path(image_dir)
        self.rows = manifest["images"]
        self.input_size, self.mask_grid = int(input_size), int(mask_grid)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        bgr = cv2.imread(str(self.image_dir / row["image_name"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"unreadable consistency image: {row['image_name']}")
        image, _, _, _ = letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.input_size)
        meta = geometry_letterbox_metadata(bgr.shape[:2], self.input_size, self.mask_grid)
        domain = np.zeros((self.mask_grid, self.mask_grid), dtype=bool)
        domain[:meta.content_height, :meta.content_width] = True
        return {"image": torch.from_numpy(image).permute(2, 0, 1).float() / 255,
                "content_valid": torch.from_numpy(domain), "image_name": row["image_name"]}
