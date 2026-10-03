# -*- coding: utf-8 -*-
"""固定GT可信的带外关系保留，不生成伪GT，不改变原完整GT损失。

教师为固定旧头。仅保留与处理后GT一致的高置信关系，修正带两端外
的8种距离均可参与；可显式启用整条关系路径的已知/第三实例检查。
KL方向为teacher||student，按通道内GT正负分别均值再平均。
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F

from utils.affinity_connectivity import _line_points
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices


def _logits(value, name):
    if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[:2] != (1, 8)
            or min(value.shape[-2:]) < 1 or not value.is_floating_point()
            or not torch.isfinite(value).all()):
        raise ValueError(name+' must be finite floating [1,8,H,W]')


def _target_mask(reference, targets, mask, mask_name):
    if (not torch.is_tensor(targets) or targets.shape != reference.shape
            or targets.device != reference.device or not targets.is_floating_point()
            or not torch.isfinite(targets).all()):
        raise ValueError('targets must be finite floating and match logits shape/device')
    if (not torch.is_tensor(mask) or mask.shape != reference.shape
            or mask.device != reference.device or mask.dtype != torch.bool):
        raise ValueError(mask_name+' must be boolean and match logits shape/device')
    if torch.any(mask & (targets != 0) & (targets != 1)):
        raise ValueError('Selected GT affinity targets must be exactly zero or one')


def _pixel_map(value, shape, device, name, *, boolean):
    value = torch.as_tensor(value, device=device)
    if value.ndim == 4 and value.shape[:2] == (1,1):
        value = value[0,0]
    elif value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.shape != shape:
        raise ValueError(name+' must match HW or singleton BHW/B1HW')
    if boolean:
        if value.dtype != torch.bool:
            raise ValueError(name+' must be boolean')
    elif value.is_floating_point() or value.is_complex() or value.dtype == torch.bool or torch.any(value < 0):
        raise ValueError(name+' must be nonnegative integer labels')
    return value


def _counts(targets, mask):
    per_channel = []
    for channel in range(mask.shape[1]):
        selected = mask[:,channel]
        positive = int((selected & (targets[:,channel] > .5)).sum())
        negative = int((selected & (targets[:,channel] <= .5)).sum())
        per_channel.append(dict(channel=channel, positive_edges=positive, negative_edges=negative,
            class_terms=int(positive>0)+int(negative>0),
            positive_fraction=positive/(positive+negative) if positive+negative else None))
    return dict(selected_edges=int(mask.sum()), positive_edges=sum(r['positive_edges'] for r in per_channel),
        negative_edges=sum(r['negative_edges'] for r in per_channel),
        active_channels=sum(r['class_terms']>0 for r in per_channel), per_channel=per_channel,
        normalization='GT positive/negative means -> present-class mean -> active-channel mean')


@torch.no_grad()
def build_reliable_retention_mask(teacher_logits, targets, legal, correction_band, confidence=.9, *,
                                  instance_map=None, valid_content=None):
    """返回固定bool mask与计数；教师错误、任一端在修正带、非法边均排除。

    targets的1/0表示同/异实例，而非材料类别。instance_map与valid_content
    必须同时提供，才能确认长边中间没有未知、padding或第三个实例。
    不提供时仅使用调用者legal的端点合同，不声称通过整路径检查。
    """
    _logits(teacher_logits, 'teacher_logits')
    _target_mask(teacher_logits, targets, legal, 'legal')
    if (isinstance(confidence, bool) or not isinstance(confidence, Real)
            or not math.isfinite(confidence) or not .5 < confidence <= 1.):
        raise ValueError('confidence must be finite and within (.5,1]')
    if (instance_map is None) != (valid_content is None):
        raise ValueError('instance_map and valid_content must be provided together')
    height,width = teacher_logits.shape[-2:]
    shape = torch.Size((height,width)); device=teacher_logits.device
    band = _pixel_map(correction_band,shape,device,'correction_band',boolean=True)
    gt = content = None
    if instance_map is not None:
        gt = _pixel_map(instance_map,shape,device,'instance_map',boolean=False)
        content = _pixel_map(valid_content,shape,device,'valid_content',boolean=True)
    probability = teacher_logits.detach().float().sigmoid()
    agreement = ((targets == 1) & (probability >= confidence)) | ((targets == 0) & (probability <= 1.-confidence))
    result = torch.zeros_like(legal)
    removed_band=removed_path=removed_unknown_endpoints=0
    for channel,(dy,dx) in enumerate(DEFAULT_AFFINITY_OFFSETS):
        if height <= abs(dy) or width <= abs(dx):
            continue
        source,dest = _edge_slices(height,width,dy,dx)
        initial = legal[0,channel][source] & agreement[0,channel][source]
        outside = ~band[source] & ~band[dest]
        removed_band += int((initial & ~outside).sum())
        take = initial & outside
        if gt is not None:
            left,right = gt[source],gt[dest]
            known = content & (gt>0)
            endpoints = known[source] & known[dest]
            removed_unknown_endpoints += int((take & ~endpoints).sum())
            expected = (left==right).to(targets.dtype)
            if torch.any(legal[0,channel][source] & endpoints & (targets[0,channel][source] != expected)):
                raise ValueError('Legal affinity target does not match supplied processed GT endpoints')
            take &= endpoints
            y0,y1,x0,x1 = source[0].start,source[0].stop,source[1].start,source[1].stop
            entire = torch.ones_like(take)
            for sy,sx in _line_points(dy,dx):
                path_gt = gt[y0+sy:y1+sy,x0+sx:x1+sx]
                entire &= known[y0+sy:y1+sy,x0+sx:x1+sx] & ((path_gt==left)|(path_gt==right))
            removed_path += int((take & ~entire).sum())
            take &= entire
        result[0,channel][source] = take
    counts = _counts(targets,result)
    counts.update(confidence=float(confidence), eligible_legal_edges=int(legal.sum()),
        legal_and_teacher_agree_edges=int((legal&agreement).sum()),
        excluded_band_endpoint_edges=removed_band, excluded_unknown_endpoint_edges=removed_unknown_endpoints,
        excluded_unknown_or_third_instance_path_edges=removed_path,
        full_path_checked=gt is not None, correction_band_rule='both relation endpoints outside fixed band',
        selection_fixed_by='detached old teacher + processed GT + fixed correction band')
    return result,counts


def balanced_affinity_retention_kl(student_logits, teacher_logits, targets, retention_mask):
    """返回class/channel归一KL(teacher||student)及可JSON序列化计数。

    用稳定BCEWithLogits减教师自身熵，避免计算log(0)。不夹改教师soft
    target，因此极端饱和且相同logits的初态也value=0、gradient=0。
    teacher总是detach；mask外的student梯度为零；空mask保持可反传零。
    """
    _logits(student_logits,'student_logits'); _logits(teacher_logits,'teacher_logits')
    if teacher_logits.shape != student_logits.shape or teacher_logits.device != student_logits.device:
        raise ValueError('Teacher/student logits must match shape/device')
    _target_mask(student_logits,targets,retention_mask,'retention_mask')
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    student = student_logits.to(dtype=dtype)
    teacher = teacher_logits.detach().to(dtype=dtype)
    probability = teacher.sigmoid()
    entropy = F.binary_cross_entropy_with_logits(teacher,probability,reduction='none')
    raw = F.binary_cross_entropy_with_logits(student,probability,reduction='none')-entropy
    # Numerical cancellation may produce a tiny negative value; KL is nonnegative.
    raw = raw.clamp_min(0.)
    channels = []
    counts = _counts(targets,retention_mask)
    channel_reports = []
    for channel in range(student.shape[1]):
        selected = retention_mask[:,channel]
        terms=[]; report=dict(channel=channel,positive_mean_kl=None,negative_mean_kl=None)
        for name,mask in (('positive',selected&(targets[:,channel]>.5)),
                          ('negative',selected&(targets[:,channel]<=.5))):
            if mask.any():
                term=raw[:,channel][mask].mean(); terms.append(term)
                report[name+'_mean_kl']=float(term.detach())
        if terms:
            channels.append(torch.stack(terms).mean())
        channel_reports.append(report)
    loss=torch.stack(channels).mean() if channels else student.sum()*0.
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite affinity retention KL')
    counts.update(direction='KL(teacher||student)',teacher_detached=True,
                  probability_clamping=False,stability='BCEWithLogits entropy difference',
                  per_channel_kl=channel_reports,loss=float(loss.detach()))
    return loss,counts
