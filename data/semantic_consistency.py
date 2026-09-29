# -*- coding: utf-8 -*-
"""赛方训练源的语义一致性视图、采集外观增强和可靠像素监督。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from data.backend_adaptation import PairedDegradationDataset
from data.direct_dual_head_dataset import _spatial_transform
from data.mim_dataset import list_images
from data.rgb_restoration_dataset import prepare_rgb, read_rgb
from data.semantic_illumination import training_illumination


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _range(config, name, default, low, high, *, positive=False):
    value = np.asarray(config.get(name, default), dtype=np.float64)
    if (value.shape != (2,) or not np.isfinite(value).all()
            or not low <= value[0] <= value[1] <= high
            or (positive and value[0] <= 0)):
        raise ValueError(f"invalid {name} range")
    return value


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and within [0,1]")
    return float(value)


def _valid_mask(mask, reference, *, conservative=False):
    if not torch.is_tensor(mask) or mask.ndim not in (3, 4):
        raise ValueError("valid_content must be B1HW or BHW")
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[:2] != (reference.shape[0], 1):
        raise ValueError("valid_content batch/channel shape differs")
    if not bool(torch.isfinite(mask).all()) or bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError("valid_content must be finite and within [0,1]")
    mask = mask.to(device=reference.device, dtype=torch.float32)
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask, size=reference.shape[-2:], mode="area" if conservative else "nearest")
    return mask >= 1.0 if conservative else mask > .5


def full_content_mask(batch, image):
    """由完整输入尺寸和共享几何重建内容mask，不读取GT覆盖或低分辨率mask。

    PairedDegradationDataset已在奇数次rot90后交换input_content_shape；
    先还原原始letterbox有效尺寸，再重放翻转和旋转。
    """
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("image must be BCHW")
    size = image.shape[0]
    shapes = torch.as_tensor(batch["input_content_shape"])
    if shapes.shape != (size, 2):
        raise ValueError("input_content_shape must be Bx2")
    geometry = {}
    for key in ("horizontal_flip", "vertical_flip", "rotation_k"):
        value = torch.as_tensor(batch[key])
        if value.numel() != size:
            raise ValueError(f"{key} requires one value per image")
        geometry[key] = value.reshape(size)
    masks = []
    for index in range(size):
        height, width = map(int, shapes[index].tolist())
        rotation = int(geometry["rotation_k"][index])
        if rotation not in (0, 1, 2, 3):
            raise ValueError("rotation_k must be in 0..3")
        canvas_height, canvas_width = image.shape[-2:]
        if rotation % 2:
            height, width = width, height
            canvas_height, canvas_width = canvas_width, canvas_height
        if not (0 < height <= canvas_height and 0 < width <= canvas_width):
            raise ValueError("input content dimensions exceed the image canvas")
        mask = torch.zeros((1, canvas_height, canvas_width), device=image.device, dtype=torch.bool)
        mask[:, :height, :width] = True
        masks.append(_spatial_transform(mask, bool(geometry["horizontal_flip"][index]),
                                        bool(geometry["vertical_flip"][index]), rotation))
    return torch.stack(masks)


def apply_acquisition_appearance(image, valid_content, config, seed):
    """D5a前的全局正斜率仿射；独立随机流与有效区截断预算。

    通道偏移去均值；回退沿原图到候选的线段进行，保持正斜率。
    填充区应用同一全局变换，不参与截断比例。批内共享一次采集条件。
    """
    if (not torch.is_tensor(image) or image.ndim != 4 or image.shape[1] != 3
            or not image.is_floating_point()):
        raise ValueError("expected floating BCHW RGB")
    if not bool(torch.isfinite(image).all()) or bool(((image < 0) | (image > 1)).any()):
        raise ValueError("image must be finite and within [0,1]")
    valid = _valid_mask(valid_content, image)
    if not bool(valid.flatten(1).any(1).all()):
        raise ValueError("valid_content must include content in each image")
    seed = _integer(seed, "seed")
    config = {"enabled": False} if config is None else config
    if not isinstance(config, dict):
        raise ValueError("appearance config must be a mapping")
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("appearance.enabled must be boolean")
    probability = _unit(config.get("probability", .5), "probability")
    budget = _unit(config.get("max_clipped_fraction", .01), "max_clipped_fraction")
    contrast_range = _range(config, "contrast", [.5, 1.0], 0, 2, positive=True)
    offset_range = _range(config, "offset", [0, .15], -1, 1)
    max_shift = _unit(config.get("max_rgb_shift", .03), "max_rgb_shift")
    shift_range = _range(config, "rgb_shift", [-max_shift, max_shift], -1, 1)
    stats = {"seed": seed, "selected": False, "applied": False,
             "probability_draw": 0.0, "requested_contrast": 1.0, "requested_offset": 0.0,
             "requested_rgb_shift": [0.0, 0.0, 0.0], "scale": 0.0,
             "actual_contrast": 1.0, "actual_offset": 0.0,
             "actual_rgb_shift": [0.0, 0.0, 0.0], "clipped_fraction": 0.0}
    if not enabled:
        return image, stats
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20260927, 71]))
    draw = float(rng.random())
    stats["probability_draw"] = draw
    if draw >= probability:
        return image, stats
    contrast = float(rng.uniform(*contrast_range))
    offset = float(rng.uniform(*offset_range))
    shift = rng.uniform(*shift_range, size=3)
    shift -= shift.mean()
    stats.update(selected=True, requested_contrast=contrast, requested_offset=offset,
                 requested_rgb_shift=shift.tolist())
    delta = (contrast - 1.0) * image + offset + image.new_tensor(shift)[None, :, None, None]
    denominator = valid.sum().item()

    def clipped_fraction(scale):
        changed = image + scale * delta
        clipped = ((changed < 0) | (changed > 1)).any(1, keepdim=True)
        return float((clipped & valid).sum().item() / denominator)

    scale = 1.0
    fraction = clipped_fraction(scale)
    if fraction > budget:
        low, high = 0.0, 1.0
        for _ in range(16):
            middle = (low + high) / 2
            if clipped_fraction(middle) <= budget:
                low = middle
            else:
                high = middle
        scale, fraction = low, clipped_fraction(low)
    stats.update(scale=scale, actual_contrast=1 + scale * (contrast - 1),
                 actual_offset=scale * offset, actual_rgb_shift=(scale * shift).tolist(),
                 clipped_fraction=fraction, applied=bool(scale > 0 and torch.any(delta != 0)))
    if not stats["applied"]:
        return image, stats
    return (image + scale * delta).clamp(0, 1), stats


class _TrainingSourceDataset(Dataset):
    """仅图像和有效内容；不读取同名标注或历史留出清单。"""
    def __init__(self, data_dir, expected_sources, image_size):
        self.samples = [Path(path) for path in list_images(str(data_dir))]
        if len(self.samples) != expected_sources:
            raise ValueError(f"expected {expected_sources} training sources, got {len(self.samples)}")
        if len({path.stem for path in self.samples}) != len(self.samples):
            raise ValueError("training source stems must be unique")
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        image, valid = prepare_rgb(read_rgb(str(path)), self.image_size)
        ys, xs = np.nonzero(valid)
        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return {"image": tensor, "clean_image": tensor,
                "valid_content": torch.from_numpy(valid.astype(bool))[None],
                "clean_valid_content": torch.from_numpy(valid.astype(bool))[None],
                "input_content_shape": torch.tensor([ys.max() + 1, xs.max() + 1], dtype=torch.int32),
                "content_shape": torch.tensor([ys.max() + 1, xs.max() + 1], dtype=torch.int32),
                "name": path.name}


class SemanticConsistencyDataset(Dataset):
    """跨epoch不断开的全源置乱周期；每1000次抽取恰好覆盖1000源。"""
    def __init__(self, data_dir, degradation, noise, seed, *, draws_per_epoch,
                 expected_sources=1000, image_size=1024, weak_config=None):
        self.seed = _integer(seed, "seed")
        self.draws_per_epoch = _integer(draws_per_epoch, "draws_per_epoch", 1)
        expected_sources = _integer(expected_sources, "expected_sources", 1)
        image_size = _integer(image_size, "image_size", 1)
        self.base_dataset = _TrainingSourceDataset(data_dir, expected_sources, image_size)
        self.samples = self.base_dataset.samples
        self.paired = PairedDegradationDataset(self.base_dataset, degradation, noise,
                                             seed=self.seed + 20260927)
        self.weak_config = {
            "enabled": True, "probability": 1.0, "local_probability": 0.0,
            "max_offset": .02, "max_clipped_fraction": .005}
        if weak_config is not None:
            if not isinstance(weak_config, dict):
                raise ValueError("weak_config must be a mapping")
            self.weak_config.update(deepcopy(weak_config))
        self._permutations = {}
        self.epoch = 1

    def __len__(self):
        return self.draws_per_epoch

    def set_epoch(self, epoch):
        self.epoch = _integer(epoch, "epoch", 1)

    def source_index_for_draw(self, global_draw):
        global_draw = _integer(global_draw, "global_draw")
        cycle, position = divmod(global_draw, len(self.samples))
        if cycle not in self._permutations:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 20260927, 13, cycle]))
            self._permutations[cycle] = rng.permutation(len(self.samples))
        return int(self._permutations[cycle][position])

    def __getitem__(self, index):
        index = _integer(index, "index")
        if index >= len(self):
            raise IndexError(index)
        global_draw = (self.epoch - 1) * self.draws_per_epoch + index
        source_index = self.source_index_for_draw(global_draw)
        # 复用原退化实现，以独立global draw命名随机流；不触碰有标签dataset。
        self.paired.set_epoch(global_draw)
        sample = self.paired[source_index]
        clean = sample.pop("clean_image")
        clean_valid = sample.pop("clean_valid_content")
        seed_rng = np.random.default_rng(np.random.SeedSequence([self.seed, global_draw, 20260927, 29]))
        weak_seed, restoration_seed, strong_seed = map(int, seed_rng.integers(0, 2**31 - 1, size=3))
        weak_check, weak_stats = training_illumination(
            clean[None], self.weak_config, weak_seed, clean_valid[None])
        geometry = (sample["horizontal_flip"], sample["vertical_flip"], sample["rotation_k"])
        sample.update(weak_image=_spatial_transform(clean, *geometry),
                      weak_check_image=_spatial_transform(weak_check[0], *geometry),
                      strong_image=sample.pop("image"), global_draw=global_draw,
                      source_index=source_index, weak_stats=weak_stats,
                      restoration_seed=restoration_seed, strong_restoration_seed=strong_seed)
        return sample


def _logits(logits, name):
    if (not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[1] != 1
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())):
        raise ValueError(f"{name} must be finite floating B1HW logits")


@torch.no_grad()
def build_reliable_semantic_targets(weak_logits, weak_check_logits, valid_content, *,
                                    confidence=.90, max_probability_gap=.05, erosion_radius=1):
    """双弱视图同类高置信且稳定的内部像素；不生成实例或强制类别比例。"""
    _logits(weak_logits, "weak_logits")
    _logits(weak_check_logits, "weak_check_logits")
    if weak_logits.shape != weak_check_logits.shape or weak_logits.device != weak_check_logits.device:
        raise ValueError("teacher views must have identical shapes and devices")
    confidence = _unit(confidence, "confidence")
    if confidence <= .5:
        raise ValueError("confidence must exceed .5")
    gap = _unit(max_probability_gap, "max_probability_gap")
    radius = _integer(erosion_radius, "erosion_radius")
    left, right = weak_logits.float().sigmoid(), weak_check_logits.float().sigmoid()
    valid = _valid_mask(valid_content, left, conservative=True)
    same_class = (left >= .5) == (right >= .5)
    high = (torch.maximum(left, 1 - left) >= confidence) & (torch.maximum(right, 1 - right) >= confidence)
    stable = (left - right).abs() <= gap
    candidate = valid & same_class & high & stable
    target = (left + right) / 2
    reliable = torch.zeros_like(candidate)
    counts = {}
    for label, name in ((0, "pearlite"), (1, "ferrite")):
        class_pixels = (target >= .5) == bool(label)
        selected = candidate & class_pixels
        if radius:
            padded = F.pad(selected.float(), (radius,) * 4, value=0)
            selected = -F.max_pool2d(-padded, 2 * radius + 1, stride=1) > .5
        reliable |= selected
        valid_count, accepted = int((valid & class_pixels).sum()), int(selected.sum())
        counts.update({f"{name}_valid_pixels": valid_count,
                       f"{name}_accepted_pixels": accepted,
                       f"{name}_accepted_fraction": accepted / max(1, valid_count)})
    valid_count, accepted = int(valid.sum()), int(reliable.sum())
    counts.update(valid_pixels=valid_count, candidate_pixels=int(candidate.sum()),
                  accepted_pixels=accepted, accepted_fraction=accepted / max(1, valid_count),
                  disagreement_pixels=int((valid & ~same_class).sum()),
                  unstable_pixels=int((valid & ~stable).sum()),
                  mean_accepted_confidence=float(torch.maximum(target, 1 - target)[reliable].mean()) if accepted else 0.0)
    return target.detach(), reliable, counts


def masked_semantic_consistency(student_logits, target_probability, reliable_mask):
    """按接受像素归一soft BCE；空监督返回可反传的零，不锐化、不类重加权。"""
    _logits(student_logits, "student_logits")
    if (target_probability.shape != student_logits.shape or reliable_mask.shape != student_logits.shape
            or target_probability.device != student_logits.device or reliable_mask.device != student_logits.device):
        raise ValueError("consistency tensors must have identical shapes and devices")
    if (not bool(torch.isfinite(target_probability).all())
            or bool(((target_probability < 0) | (target_probability > 1)).any())):
        raise ValueError("target_probability must be finite and within [0,1]")
    if reliable_mask.dtype != torch.bool:
        raise ValueError("reliable_mask must be boolean")
    count = int(reliable_mask.sum())
    if not count:
        return student_logits.float().sum() * 0, {"accepted_pixels": 0}
    loss = F.binary_cross_entropy_with_logits(student_logits.float(), target_probability.detach().float(), reduction="none")
    return loss[reliable_mask].mean(), {"accepted_pixels": count}
