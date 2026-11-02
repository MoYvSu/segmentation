# -*- coding: utf-8 -*-
"""NAF边界横截面的连通响应约束；原覆盖、负集、定位容差均不改。

仅在处理后GT的局部窄走廊确实只有一次A/B界面切换时启用。
停止梯度的区间上包络只抬两峰间的暗谷，不通过压低弱峰降低损失。
这是横截面连通约束，不是物理精确中心线或完整分水岭拓扑保证。
"""
from __future__ import annotations

from functools import lru_cache
import math
from numbers import Real

import torch
from torch.nn import functional as F


VERSION = 'normal_hull_v1'
DIRECTIONS = ((0, 1), (1, 1), (1, 0), (-1, 1))


def ridge_supervision_settings(value=None):
    """缺省严格关闭；新增参数只服务显式实验，不改旧配方。"""
    if value is None:
        return dict(enabled=False)
    if not isinstance(value, dict) or type(value.get('enabled', False)) is not bool:
        raise ValueError('ridge_supervision must be a mapping with boolean enabled')
    allowed = {'enabled', 'version', 'weight', 'profile_radius', 'corridor_half_width'}
    if set(value) - allowed:
        raise ValueError('Unknown ridge_supervision fields: ' + str(sorted(set(value) - allowed)))
    if not value.get('enabled', False):
        return dict(enabled=False)
    if value.get('version') != VERSION:
        raise ValueError('ridge_supervision requires explicit version=' + VERSION)
    weight = value.get('weight', .1)
    if (not isinstance(weight, Real) or isinstance(weight, bool)
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('Ridge weight must be positive finite')
    radius, width = value.get('profile_radius', 4), value.get('corridor_half_width', 1)
    if type(radius) is not int or radius < 1 or type(width) is not int or width < 1:
        raise ValueError('Ridge profile radius and corridor half width must be positive integers')
    return dict(enabled=True, version=VERSION, weight=float(weight),
                profile_radius=radius, corridor_half_width=width)


@lru_cache(maxsize=32)
def _corridor_rays(normal_index, radius, half_width):
    """枚举整个整数走廊；斜向包含半格错位射线，避免稀疏对角漏掉细通路。"""
    ny, nx = DIRECTIONS[normal_index]
    ty, tx = nx, -ny
    scale, extent = ny * ny + nx * nx, radius + half_width
    rays = {}
    for dy in range(-extent, extent + 1):
        for dx in range(-extent, extent + 1):
            normal, tangent = dy * ny + dx * nx, dy * ty + dx * tx
            if abs(normal) <= radius * scale and abs(tangent) <= half_width * scale:
                rays.setdefault(tangent, []).append((normal, dy, dx))
    return tuple(tuple((dy, dx) for _, dy, dx in sorted(ray))
                 for _, ray in sorted(rays.items()))


@torch.no_grad()
def build_ridge_targets(prepared, instances, settings):
    """GT-only资格；额外掩码绝不替换原positive/negative或其分母。

    ±profile_radius为预测锚点范围。另一条GT界面允许在原tolerance内偏移，
    故检查半径为profile_radius+tolerance+1；最后1格覆盖GT双侧标记的差异。
    每条离散射线必须只出现同一对正ID且恰好切换一次，排除A-B-A、A-B-C与交叉。
    """
    settings = ridge_supervision_settings(settings)
    if not settings['enabled']:
        raise ValueError('Cannot build ridge targets when disabled')
    valid = prepared['valid']
    if (not torch.is_tensor(instances) or instances.shape != valid.shape
            or instances.device != valid.device or instances.dtype != torch.int64):
        raise ValueError('ridge_instances must be aligned int64 B1HW on the GT device')
    tolerance = prepared['tolerance']
    radius, width = settings['profile_radius'], settings['corridor_half_width']
    if radius < tolerance:
        raise ValueError('Ridge anchors must cover the original tolerance interval')
    guard = radius + tolerance + 1
    extent = guard + width
    # 显式图外0，不能把padding当known；对角射线也完整落在这个方形内。
    known = valid & (instances > 0)
    padded = F.pad(known.float(), (extent,) * 4, value=0)
    trusted = F.avg_pool2d(padded, 2 * extent + 1, stride=1) == 1
    candidates = prepared['positive'] & ~prepared['fallback'] & trusted
    normal = prepared['normal_index']
    eligible = torch.zeros_like(valid)
    for index in range(4):
        b, c, y, x = (candidates & (normal == index)).nonzero(as_tuple=True)
        if not b.numel():
            continue
        rays = _corridor_rays(index, guard, width)
        first = rays[0]
        left = instances[b, c, y + first[0][0], x + first[0][1]]
        right = instances[b, c, y + first[-1][0], x + first[-1][1]]
        good = (left > 0) & (right > 0) & (left != right)
        for ray in rays:
            values = torch.stack([instances[b, c, y + dy, x + dx] for dy, dx in ray], dim=1)
            good &= ((values == left[:, None]) | (values == right[:, None])).all(1)
            good &= (values[:, 0] == left) & (values[:, -1] == right)
            good &= ((values[:, 1:] != values[:, :-1]).sum(1) == 1)
        eligible[b[good], c[good], y[good], x[good]] = True
    return dict(eligible=eligible, normal_index=normal, tolerance=tolerance,
                profile_radius=radius, corridor_half_width=width,
                guard_radius=guard, valid_extent=extent,
                eligible_points=int(eligible.sum()), candidate_points=int(candidates.sum()),
                original_positive_points=int(prepared['positive'].sum()), version=VERSION)


def ridge_supervision_loss(logits, targets, *, denominator=None):
    """中央容差带KL，上包络取较宽法向剖面；只向上填谷。

    denominator为完整有效批的eligible_points*(2*tolerance+1)，不是微批分母。
    外部锚点仅决定停止梯度的目标，自身不接受本项梯度；可信内部仍由原negative抑制。
    """
    eligible = targets['eligible']
    if (not torch.is_tensor(logits) or logits.shape != eligible.shape
            or logits.device != eligible.device or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())):
        raise ValueError('Ridge logits must be finite floating B1HW aligned with targets')
    count = int(eligible.sum())
    tolerance, radius = targets['tolerance'], targets['profile_radius']
    positions = count * (2 * tolerance + 1)
    if denominator is None:
        denominator = positions
    if type(denominator) is not int or denominator < positions:
        raise ValueError('Ridge global denominator must be integer >= local supervised positions')
    zero = logits.sum() * 0
    numerators = []
    gap_points = gap_profiles = multitrack = 0
    for index, (ny, nx) in enumerate(DIRECTIONS):
        b, c, y, x = (eligible & (targets['normal_index'] == index)).nonzero(as_tuple=True)
        if not b.numel():
            continue
        samples = torch.stack([logits[b, c, y + delta * ny, x + delta * nx]
                               for delta in range(-radius, radius + 1)], dim=1)
        detached = samples.detach()
        left = detached.cummax(1).values
        right = detached.flip(1).cummax(1).values.flip(1)
        hull = torch.minimum(left, right)
        low, high = radius - tolerance, radius + tolerance + 1
        z, a = samples[:, low:high], hull[:, low:high]
        q = a.sigmoid()
        # 稳定KL：同剖面单线/单峰/实心带逐项严格0；深谷不受sigmoid饱和的弱梯度限制。
        divergence = q * (F.logsigmoid(a) - F.logsigmoid(z))
        divergence += (1 - q) * (F.logsigmoid(-a) - F.logsigmoid(-z))
        # 数值值用稳定log-sigmoid差，反向用融合BCE的解析p-q。
        # 直接反传上式在几乎同高的峰/谷可能因舍入产生约1e-8的反向推峰；
        # 下面与KL解析梯度相同，且q>=p时浮点梯度也严格不为正。
        bce = F.binary_cross_entropy_with_logits(z, q, reduction='none')
        divergence = divergence.detach().clamp_min(0) + (bce - bce.detach())
        # 同值峰/单调段显式无梯度：sigmoid在大正logit舍入成1，
        # 单靠KL数值为0仍可能留下logsigmoid的极小导数。
        fill = a > z.detach()
        numerators.append(torch.where(fill, divergence,
                                      torch.zeros_like(divergence)).sum())
        with torch.no_grad():
            gaps = q - z.sigmoid() > .05
            gap_points += int(gaps.sum())
            gap_profiles += int(gaps.any(1).sum())
            strong = detached.sigmoid() > .65
            starts = strong[:, :1].sum(1) + (strong[:, 1:] & ~strong[:, :-1]).sum(1)
            multitrack += int((starts >= 2).sum())
    loss = (sum(numerators, zero) / denominator) if positions else zero
    return loss, dict(ridge_loss=loss, ridge_eligible_points=count,
        ridge_positions=positions, ridge_gap_points=gap_points,
        ridge_gap_fraction=gap_points / max(1, positions),
        ridge_gap_profiles=gap_profiles,
        ridge_multitrack_points=multitrack,
        ridge_multitrack_fraction=multitrack / max(1, count))
