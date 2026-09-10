# -*- coding: utf-8 -*-
"""两类实例集合监督：有效点匹配、掩码损失和部分标注的类别监督。

每个实例先抽一个有效正点，再补充均匀有效点。这是保证小实例可见的采样，
不是对全图像素均匀积分的无偏估计。unknown 和 padding 均不参加匹配或掩码
损失；当前数据不能可靠区分两者，因此不从 content_shape 猜测内容区域。
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def _spatial_batch(value, name):
    value = torch.as_tensor(value)
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W]")
    return value


def targets_from_direct_batch(batch, device):
    """转成 labels/masks/valid；ID=0 的未知像素绝不转换成珠光体或背景。

    优先使用 trainer 从原始 class JSON 提供的 instance_classes (list[dict])。
    无映射时，从同 ID 的语义像素推导类别，遇到类混合即报错，不用多数票掩盖
    对齐错误。semantic_instance_map 存在时先在其原网格推导，避免跨网格舍入。
    """
    maps = _spatial_batch(batch["affinity_instance_map"], "affinity_instance_map").to(device)
    valid = _spatial_batch(batch["affinity_valid_content"], "affinity_valid_content").to(
        device=device, dtype=torch.bool
    )
    if maps.shape != valid.shape or maps.is_floating_point():
        raise ValueError("instance maps must be integer and align with valid")
    if torch.any(valid & (maps <= 0)):
        raise ValueError("valid pixels must carry a positive instance ID")
    mappings = batch.get("instance_classes")
    if mappings is not None and len(mappings) != len(maps):
        raise ValueError("instance_classes must contain one mapping per image")
    semantic = semantic_maps = semantic_valid = None
    if mappings is None:
        semantic = _spatial_batch(batch["semantic_target"], "semantic_target").to(device)
        if "semantic_instance_map" in batch:
            semantic_maps = _spatial_batch(batch["semantic_instance_map"], "semantic_instance_map").to(device)
            if semantic_maps.shape != semantic.shape:
                raise ValueError("semantic_instance_map and semantic_target must align")
            semantic_valid = semantic_maps > 0
            if "semantic_valid_content" in batch:
                semantic_valid &= _spatial_batch(batch["semantic_valid_content"], "semantic_valid_content").to(
                    device=device, dtype=torch.bool
                )
        else:
            semantic = F.interpolate(semantic[:, None].float(), size=maps.shape[-2:], mode="nearest")[:, 0]
            semantic_maps, semantic_valid = maps, valid
        if len(semantic) != len(maps):
            raise ValueError("semantic batch size does not match instance maps")
    targets = []
    for index, (instance_map, allowed) in enumerate(zip(maps, valid)):
        ids = torch.unique(instance_map[allowed], sorted=True).long()
        if torch.any(ids > 65535):
            raise ValueError("instance ID exceeds the uint16 contract")
        class_map = None
        if mappings is not None:
            if not isinstance(mappings[index], Mapping):
                raise ValueError("instance_classes entries must be mappings")
            class_map = {}
            for key, value in mappings[index].items():
                instance_id = int(key)
                if not 1 <= instance_id <= 65535 or value not in (0, 1):
                    raise ValueError("instance_classes requires positive uint16 IDs and classes 0/1")
                class_map[instance_id] = int(value)
        labels = []
        for instance_id in ids.tolist():
            if class_map is not None:
                if instance_id not in class_map:
                    raise ValueError(f"missing class for instance ID {instance_id}")
                labels.append(class_map[instance_id])
            else:
                pixels = (semantic_maps[index] == instance_id) & semantic_valid[index]
                values = torch.unique(semantic[index][pixels])
                if values.numel() != 1 or values[0].item() not in (0, 1):
                    raise ValueError(f"instance ID {instance_id} has missing or mixed semantic classes")
                labels.append(int(values[0].item()))
        targets.append({
            "labels": torch.tensor(labels, device=device, dtype=torch.long),
            "masks": (instance_map[None] == ids[:, None, None]) & allowed[None],
            "valid": allowed,
            "instance_ids": ids,
        })
    return targets


class MaskSetCriterion(nn.Module):
    """Hungarian 一对一匹配，pred_logits 为 [B,Q,3]，pred_masks 为 logits。

    类别 0/1 对应珠光体/铁素体，类别 2 为 no-object。未匹配 query 的正区域
    中至少 no_object_valid_fraction 落在 valid 才监督为 no-object；全空掩码
    也监督。只有未知区域正响应的 query 不受此监督。门控完全 detach，无法
    经该权重向掩码回传梯度，但仍是启发式，不能保证识别所有未知实例。

    可选准确的 target['content_valid'] 仅供调用方已有内容 mask 时使用；它
    排除 padding 后计算正区域支持率。未提供时 unknown/padding 合并处理。
    aux_outputs 的各层独立匹配，辅助损失取层均值，避免层数放大权重。
    所有随机采样使用 generator；验证应每次从同一 seed 重建 generator。
    """

    DEFAULTS = {
        "class_weight": 2.0, "mask_weight": 5.0, "dice_weight": 5.0,
        "matching_cost_class": 2.0, "matching_cost_mask": 5.0, "matching_cost_dice": 5.0,
        "eos_coef": 0.1, "num_points": 4096, "mask_points": 4096,
        "aux_weight": 1.0, "no_object_valid_fraction": 0.5,
        "ownership_weight": 0.0,
    }

    def __init__(self, loss_config=None, **overrides):
        super().__init__()
        config = dict(loss_config or {})
        config.update(overrides)
        unknown = set(config) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown mask-set loss options: {sorted(unknown)}")
        self.config = {**self.DEFAULTS, **config}
        for key in ("num_points", "mask_points"):
            if int(self.config[key]) != self.config[key] or self.config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.config["no_object_valid_fraction"] <= 1:
            raise ValueError("no_object_valid_fraction must be in (0,1]")
        for key, value in self.config.items():
            if value < 0:
                raise ValueError(f"{key} must be nonnegative")

    @staticmethod
    def _randint(high, size, device, generator):
        # CPU generator 可控制 CUDA 目标；避免依赖两个设备各自的隐式 RNG。
        rng_device = generator.device if generator is not None else device
        return torch.randint(high, size, device=rng_device, generator=generator).to(device)

    def _sample_points(self, target, count, generator):
        valid = target["valid"].bool().flatten()
        available = torch.nonzero(valid, as_tuple=False).flatten()
        if not len(available):
            if len(target["labels"]):
                raise ValueError("nonempty targets require valid pixels")
            return available
        masks = target["masks"].flatten(1)
        positive = []
        for mask in masks:
            candidates = torch.nonzero(mask.bool() & valid, as_tuple=False).flatten()
            if not len(candidates):
                raise ValueError("each target instance requires at least one valid positive pixel")
            pick = self._randint(len(candidates), (1,), available.device, generator)
            positive.append(candidates[pick])
        # 小图直接使用全部有效点，既准确又避免重复采样。
        count = max(int(count), len(positive))
        if count >= len(available):
            return available
        remaining = count - len(positive)
        uniform = available[self._randint(len(available), (remaining,), available.device, generator)]
        return torch.cat([*positive, uniform])

    @staticmethod
    def _validate(outputs, targets):
        logits, masks = outputs["pred_logits"], outputs["pred_masks"]
        if logits.ndim != 3 or logits.shape[-1] != 3 or masks.ndim != 4:
            raise ValueError("expected pred_logits [B,Q,3] and pred_masks [B,Q,H,W]")
        if logits.shape[:2] != masks.shape[:2] or len(targets) != len(logits):
            raise ValueError("prediction batch/query dimensions do not match")
        for target in targets:
            labels, gt_masks, valid = target["labels"], target["masks"], target["valid"]
            if labels.ndim != 1 or torch.any((labels != 0) & (labels != 1)):
                raise ValueError("target labels must contain classes 0/1")
            if gt_masks.ndim != 3 or valid.ndim != 2 or gt_masks.shape != (len(labels), *valid.shape):
                raise ValueError("target masks/valid shapes do not align")
            if len(labels) > logits.shape[1]:
                raise ValueError(f"query capacity {logits.shape[1]} is below target instance count {len(labels)}")
            if "content_valid" in target:
                content = target["content_valid"]
                if content.shape != valid.shape or torch.any(valid.bool() & ~content.bool()):
                    raise ValueError("content_valid must align with and include all valid pixels")

    @staticmethod
    def _grid_masks(masks, target):
        if masks.shape[-2:] != target["valid"].shape:
            masks = F.interpolate(masks[None].float(), size=target["valid"].shape,
                                  mode="bilinear", align_corners=False)[0]
        return masks

    @torch.no_grad()
    def _match_one(self, logits, masks, target, points):
        device = masks.device
        if not len(target["labels"]):
            empty = torch.empty(0, device=device, dtype=torch.long)
            return empty, empty
        # 即使 trainer 在 AMP 上下文中调用，也以 float32 计算匹配成本。
        with torch.autocast(device_type=masks.device.type, enabled=False):
            pred = masks.flatten(1)[:, points].float()
            truth = target["masks"].flatten(1)[:, points].float()
            class_cost = -logits.float().softmax(-1)[:, target["labels"].long()]
            # 矩阵乘法得到 [Q,N]，不建立 [Q,N,P] 的逐点中间张量。
            bce = F.softplus(pred).mean(1)[:, None] - pred @ truth.T / len(points)
            probability = pred.sigmoid()
            dice = 1 - (2 * (probability @ truth.T) + 1) / (
                probability.sum(1)[:, None] + truth.sum(1)[None] + 1
            )
            cost = (self.config["matching_cost_class"] * class_cost
                    + self.config["matching_cost_mask"] * bce
                    + self.config["matching_cost_dice"] * dice)
        rows, columns = linear_sum_assignment(cost.cpu().numpy())
        return (torch.as_tensor(rows, device=device, dtype=torch.long),
                torch.as_tensor(columns, device=device, dtype=torch.long))

    @torch.no_grad()
    def match(self, outputs, targets, generator=None):
        """可审计的匹配入口；使用相同 seed 可与 forward 的主层匹配对照。"""
        self._validate(outputs, targets)
        points = [self._sample_points(t, self.config["num_points"], generator) for t in targets]
        return [self._match_one(logits, self._grid_masks(masks, target), target, point)
                for logits, masks, target, point in zip(outputs["pred_logits"], outputs["pred_masks"], targets, points)]

    @torch.no_grad()
    def _no_object_support(self, masks, target):
        valid = target["valid"].bool()
        if not torch.any(valid):
            return torch.zeros(len(masks), device=masks.device, dtype=torch.bool)
        positive = masks.detach() > 0  # sigmoid > 0.5，与 logit > 0 完全等价。
        if "content_valid" in target:
            positive &= target["content_valid"].bool()[None]
        total = positive.flatten(1).sum(1)
        known = (positive & valid[None]).flatten(1).sum(1)
        return (total == 0) | ((known > 0) & (known.float() >= total * self.config["no_object_valid_fraction"]))

    def _layer_loss(self, outputs, targets, match_points, mask_points):
        self._validate(outputs, targets)
        # 零监督批次仍保留梯度图，不对 ignored 像素制造梯度。
        zero = (outputs["pred_logits"].sum(dtype=torch.float32) * 0
                + outputs["pred_masks"].sum(dtype=torch.float32) * 0)
        class_sum = mask_sum = dice_sum = zero
        class_normalizer = zero.detach().clone()
        count = ignored = supervised = 0
        for logits, raw_masks, target, points, loss_points in zip(
            outputs["pred_logits"], outputs["pred_masks"], targets, match_points, mask_points
        ):
            masks = self._grid_masks(raw_masks, target)
            rows, columns = self._match_one(logits, masks, target, points)
            labels = torch.full((len(logits),), 2, device=logits.device, dtype=torch.long)
            labels[rows] = target["labels"][columns].long()
            support = self._no_object_support(masks, target)
            support[rows] = False  # 此处只统计未匹配 query。
            weights = support.float() * self.config["eos_coef"]
            weights[rows] = 1
            class_sum = class_sum + (F.cross_entropy(logits.float(), labels, reduction="none") * weights).sum()
            class_normalizer = class_normalizer + weights.sum()
            count += len(rows)
            supervised += int(support.sum().item())
            ignored += len(logits) - len(rows) - int(support.sum().item())
            if len(rows):
                pred = masks.flatten(1)[:, loss_points][rows].float()
                truth = target["masks"].flatten(1)[:, loss_points][columns].float()
                mask_sum = mask_sum + F.binary_cross_entropy_with_logits(pred, truth, reduction="none").mean(1).sum()
                probability = pred.sigmoid()
                dice_sum = dice_sum + (1 - (2 * (probability * truth).sum(1) + 1) / (
                    probability.sum(1) + truth.sum(1) + 1
                )).sum()
        losses = {"loss_class": class_sum / class_normalizer.clamp_min(1),
                  "loss_mask": mask_sum / max(count, 1), "loss_dice": dice_sum / max(count, 1)}
        losses["weighted"] = sum(losses[f"loss_{name}"] * self.config[f"{name}_weight"]
                                  for name in ("class", "mask", "dice"))
        losses.update(matched_count=zero.detach() + count, ignored_no_object=zero.detach() + ignored,
                      supervised_no_object=zero.detach() + supervised)
        return losses

    def forward(self, outputs, targets, generator=None):
        self._validate(outputs, targets)
        match_points = [self._sample_points(t, self.config["num_points"], generator) for t in targets]
        if self.config["mask_points"] == self.config["num_points"]:
            mask_points = match_points
        else:
            mask_points = [self._sample_points(t, self.config["mask_points"], generator) for t in targets]
        result = self._layer_loss(outputs, targets, match_points, mask_points)
        main = result.pop("weighted")
        auxiliaries = outputs.get("aux_outputs", [])
        auxiliary = main * 0
        for layer in auxiliaries:
            auxiliary = auxiliary + self._layer_loss(layer, targets, match_points, mask_points)["weighted"]
        result["loss_aux"] = auxiliary / max(len(auxiliaries), 1)
        result["loss_supervised"] = main + self.config["aux_weight"] * result["loss_aux"]
        # 像素归属只作用于最后一层。与原匹配使用同一批点，不新增随机采样。
        # BCE/Dice继续约束绝对掩码概率；这里明确比较同一像素的不同query。
        ownership = main * 0
        if self.config["ownership_weight"] > 0:
            for logits, raw_masks, target, points, sampled in zip(
                outputs["pred_logits"], outputs["pred_masks"], targets, match_points, mask_points
            ):
                if not len(target["labels"]):
                    continue
                truth = target["masks"].flatten(1)[:, sampled].bool()
                if not torch.all(truth.sum(0) == 1):
                    raise ValueError("ownership requires exactly one target instance at each valid sampled pixel")
                masks = self._grid_masks(raw_masks, target)
                rows, columns = self._match_one(logits, masks, target, points)
                target_to_query = torch.empty(len(columns), device=masks.device, dtype=torch.long)
                target_to_query[columns] = rows
                owners = target_to_query[truth.long().argmax(0)]
                # 所有query均参与竞争，包括未匹配query；只在已确认像素上抑制抢占。
                ownership = ownership + F.cross_entropy(masks.flatten(1)[:, sampled].float().T, owners)
        result["loss_ownership"] = ownership / max(1, len(targets))
        result["loss_total"] = result["loss_supervised"] + self.config["ownership_weight"] * result["loss_ownership"]
        return result
