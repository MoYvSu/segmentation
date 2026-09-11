# -*- coding: utf-8 -*-
"""EMA 实例目标：互斥分配、跨视图配对、可靠区域监督；query 编号不作对应。"""
from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from utils.mask_set_loss import MaskSetCriterion


@torch.no_grad()
def grid_partition(outputs, domain, *, score_threshold=.5, mask_threshold=.5):
    """与正式输出相同的候选竞争，直接在原掩码网格计算，无尺寸恢复。"""
    masks, logits = outputs["pred_masks"][0].float(), outputs["pred_logits"][0].float()
    if masks.shape[-2:] != domain.shape or logits.shape != (len(masks), 3):
        raise ValueError("consistency outputs and content domain do not align")
    if not torch.isfinite(masks).all() or not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite teacher output")
    scores, labels = logits.softmax(-1).max(-1)
    probability = masks.sigmoid()
    eligible = ((labels != 2) & (scores >= score_threshold))[:, None, None] & (probability >= mask_threshold)
    competing = (probability * scores[:, None, None]).masked_fill(~eligible, -1)
    best, owners = competing.max(0)  # 并列时保留较小 query 编号。
    pred = torch.where((best >= 0) & domain, owners + 1, 0).long()
    confidence = probability.gather(0, owners[None])[0]
    return {"instances": pred, "classes": labels, "confidence": confidence}


@torch.no_grad()
def match_partitions(left, right, domain, threshold=.8):
    """返回基准 ID -> 另一视图 ID；先同类匈牙利匹配，再应用固定 IoU 门槛。"""
    a, b = left["instances"][domain], right["instances"][domain]
    width = len(right["classes"]) + 1
    hist = torch.bincount(a * width + b, minlength=(len(left["classes"]) + 1) * width).reshape(-1, width)
    li, ri = hist.sum(1).nonzero().flatten(), hist.sum(0).nonzero().flatten()
    li, ri = li[li > 0], ri[ri > 0]
    intersection = hist[li[:, None], ri[None]].float()
    union = hist.sum(1)[li, None] + hist.sum(0)[None, ri] - intersection
    iou = (intersection / union.clamp_min(1)).cpu().numpy()
    lc, rc = left["classes"][li - 1], right["classes"][ri - 1]
    pairs = {}
    for label in (0, 1):
        rows, cols = (lc == label).nonzero().flatten().tolist(), (rc == label).nonzero().flatten().tolist()
        if not rows or not cols:
            continue
        x, y = linear_sum_assignment(-iou[np.ix_(rows, cols)])
        for r, c in zip(x, y):
            if iou[rows[r], cols[c]] >= threshold:
                pairs[int(li[rows[r]])] = int(ri[cols[c]])
    return pairs


@torch.no_grad()
def stable_targets(base, appearance, flipped, domain, *, iou_threshold=.8, mask_confidence=.7):
    """只监督三视图对同一实例达成一致且绝对掩码概率可靠的像素。"""
    first = match_partitions(base, appearance, domain, iou_threshold)
    second = match_partitions(base, flipped, domain, iou_threshold)
    stable = sorted(first.keys() & second.keys())
    masks, labels = [], []
    for instance_id in stable:
        support = (base["instances"] == instance_id) & domain
        support &= appearance["instances"] == first[instance_id]
        support &= flipped["instances"] == second[instance_id]
        for view in (base, appearance, flipped):
            support &= view["confidence"] >= mask_confidence
        if support.any():
            masks.append(support)
            labels.append(base["classes"][instance_id - 1])
    masks = torch.stack(masks) if masks else torch.empty((0, *domain.shape), device=domain.device, dtype=torch.bool)
    valid = masks.any(0)
    target = {"masks": masks, "labels": torch.stack(labels).long() if labels else
              torch.empty(0, device=domain.device, dtype=torch.long), "valid": valid, "content_valid": domain}
    return target, {"selected_instances": len(labels), "stable_instances_before_pixel_filter": len(stable),
                    "valid_pixels": int(valid.sum()), "content_pixels": int(domain.sum())}


def transform_student_view(image, target, *, gain, bias, gamma, horizontal_flip):
    strong = (image * gain + bias).clamp(0, 1).pow(gamma)
    result = dict(target)
    if horizontal_flip:
        strong = strong.flip(-1)
        for key in ("masks", "valid", "content_valid"):
            result[key] = result[key].flip(-1)
    return strong, result


def consistency_criterion(supervised_config):
    # 未配对 query 不设背景；只在已筛选实例的可靠区域监督最后一层。
    return MaskSetCriterion(supervised_config, eos_coef=0., ownership_weight=0., aux_weight=0.)


def make_ema_teacher(model):
    # 必须在给学生安装捕获 trunk 的 checkpoint forward 闭包之前复制。
    if "forward" in vars(model.encoder.trunk):
        raise ValueError("clone EMA before installing the student's gradient checkpoint closure")
    teacher = copy.deepcopy(model).eval()
    teacher.encoder.trainable_lora = False
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_mask_set_ema(teacher, student, decay):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0,1)")
    source = dict(student.named_parameters())
    for name, parameter in teacher.named_parameters():
        current = source[name]
        if current.requires_grad:
            parameter.lerp_(current.detach(), 1 - decay)
    buffers = dict(student.named_buffers())
    for name, value in teacher.named_buffers():
        value.copy_(buffers[name])


class MaskSetConsistency:
    """每若干监督步附加一个无标签损失，EMA 在同一次优化器更新后移动。"""
    def __init__(self, teacher, loader, config, device, steps_per_epoch):
        self.teacher, self.loader = teacher, loader
        self.iterator = None
        self.config = config["mask_set"]["consistency"]
        self.criterion = consistency_criterion(config["mask_set"]["loss"])
        self.device, self.steps_per_epoch = device, steps_per_epoch
        self.step = 0
        self.amp_enabled = bool(config["mask_set"].get("amp", True)) and device.type == "cuda"
        self.amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["mask_set"].get("amp_dtype", "bfloat16")]
        self.samples_seen = set()
        self.processed_samples = 0

    def weight(self):
        ramp_steps = max(1, int(self.config["ramp_epochs"]) * self.steps_per_epoch)
        return float(self.config["weight"]) * min(1., (self.step + 1) / ramp_steps)

    @torch.no_grad()
    def _teacher_view(self, image, domain, undo_flip=False):
        # 教师用已诊断的 FP32，退出上下文后学生才进行自己的 AMP 前向。
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            with torch.autocast(device_type=self.device.type, enabled=False):
                result = self.teacher(image)
        finally:
            torch.set_float32_matmul_precision(precision)
        result.pop("aux_outputs", None)
        if undo_flip:
            result["pred_masks"] = result["pred_masks"].flip(-1)
        return grid_partition(result, domain)

    def loss(self, student):
        stats = dict(samples=0, selected_instances=0, stable_instances_before_pixel_filter=0,
                     valid_pixels=0, content_pixels=0, loss=0., weighted_loss=0., weight=self.weight())
        if (self.step + 1) % int(self.config["every_n_steps"]):
            return None, stats
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        self.samples_seen.add(batch["image_name"][0])
        self.processed_samples += 1
        image = batch["image"].to(self.device, non_blocking=True)
        domain = batch["content_valid"][0].to(self.device)
        # 辅助前向与匹配点不推进监督训练的 RNG，便于对齐 A/B 的采样。
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.config["seed"]) + self.step)
            base = self._teacher_view(image, domain)
            appearance = self._teacher_view((image * 1.08 - .04).clamp(0, 1).pow(.94), domain)
            flipped = self._teacher_view(image.flip(-1), domain, undo_flip=True)
            target, quality = stable_targets(base, appearance, flipped, domain,
                                            iou_threshold=self.config["iou_threshold"],
                                            mask_confidence=self.config["mask_confidence"])
            stats.update(samples=1, **quality)
            del base, appearance, flipped
            if not len(target["labels"]):
                return None, stats
            random = torch.rand(4).tolist()
            strong, target = transform_student_view(image, target, gain=1 + (random[0] * 2 - 1) * .08,
                bias=(random[1] * 2 - 1) * .04, gamma=1 + (random[2] * 2 - 1) * .06,
                horizontal_flip=random[3] < .5)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                outputs = student(strong)
            outputs.pop("aux_outputs", None)
            losses = self.criterion(outputs, [target])
            loss = losses["loss_total"] * self.weight()
            stats.update(loss=float(losses["loss_total"].detach()), weighted_loss=float(loss.detach()))
        return loss, stats

    def after_step(self, student):
        update_mask_set_ema(self.teacher, student, float(self.config["ema_decay"]))
        self.step += 1
