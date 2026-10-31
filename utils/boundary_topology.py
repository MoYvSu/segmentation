# -*- coding: utf-8 -*-
"""补边器的有限H1持久同调损失；PH配对在CPU，临界像素误差在原张量反传。

这是Dual-CGB式拓扑微调的明确受限适配，不是全部Betti维度的精确复现：
只检查二维边界超水平集的有限H1（闭环），不把裁框/unknown补成背景，不优化
最终marker或watershed。缺少预测持久点时不凭空制造可微拓扑答案，另行记录缺口。

实现依据GUDHI CubicalComplex/cofaces_of_persistence_pairs官方接口：
https://gudhi.inria.fr/python/3.11.0/cubical_complex_ref.html
GUDHI输入平铺索引使用Fortran顺序，不能直接当作PyTorch的C顺序索引。
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import hashlib

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn


TOPOLOGY_FORMAT = 'boundary_finite_h1_gudhi_v1'
ORIGIN_POLICY = 'integral_valid_grid_component_boundary_center_v2'


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length < tile_size:
        return []
    values = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if values[-1] != final:
        values.append(final)
    return values


@lru_cache(maxsize=128)
def _cached_eligible_origins(height, width, tile_size, stride, packed_valid):
    """先精确求全部合法起点，再压缩为与规则网格及合法域边界对齐的候选。"""
    if height < tile_size or width < tile_size:
        return (), 0, 0
    valid = np.unpackbits(np.frombuffer(packed_valid, dtype=np.uint8), count=height * width).reshape(height, width)
    integral = np.pad(valid.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    t = tile_size
    sums = integral[t:, t:] - integral[:-t, t:] - integral[t:, :-t] + integral[:-t, :-t]
    legal = sums == t * t
    legal_count = int(legal.sum())
    if not legal_count:
        return (), int(legal.size), 0
    candidates = {
        (y, x)
        for y in _positions(height, t, stride)
        for x in _positions(width, t, stride)
        if legal[y, x]
    }
    # 仅从0起的规则网格会错过4..252这类合法域；每个合法起点连通域都补入
    # bbox两端/中心对齐点。若bbox角本身不合法，则选该域内最近的真实合法起点。
    # 这里不会改变valid、补padding或读取模型预测来挑监督。
    labels, _ = ndimage.label(legal)
    for component_id, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        coords = np.argwhere(labels[region] == component_id)
        coords[:, 0] += region[0].start
        coords[:, 1] += region[1].start
        y0, y1 = region[0].start, region[0].stop - 1
        x0, x1 = region[1].start, region[1].stop - 1
        for anchor_y in sorted({y0, (y0 + y1) // 2, y1}):
            for anchor_x in sorted({x0, (x0 + x1) // 2, x1}):
                distance = (coords[:, 0] - anchor_y) ** 2 + (coords[:, 1] - anchor_x) ** 2
                y, x = coords[int(distance.argmin())]
                candidates.add((int(y), int(x)))
    center_y, center_x = (height - t) / 2.0, (width - t) / 2.0
    origins = tuple(sorted(candidates, key=lambda point: (
        (point[0] - center_y) ** 2 + (point[1] - center_x) ** 2, point[0], point[1])))
    return origins, int(legal.size), legal_count


def _eligible_origins_info(valid, tile_size: int, stride: int | None = None):
    if torch.is_tensor(valid):
        valid = valid.detach().cpu().numpy()
    array = np.asarray(valid)
    if array.ndim != 2 or not np.isfinite(array).all() or not np.logical_or(array == 0, array == 1).all():
        raise ValueError('eligible_origins的valid必须为二维bool或0/1数组')
    tile_size = int(tile_size)
    stride = max(1, tile_size // 2) if stride is None else int(stride)
    if tile_size < 1 or not 1 <= stride <= tile_size:
        raise ValueError('tile_size须为正且stride须位于[1,tile_size]')
    height, width = array.shape
    packed = np.packbits(np.asarray(array, dtype=bool).reshape(-1)).tobytes()
    return _cached_eligible_origins(height, width, tile_size, stride, packed)


def eligible_origins(valid, tile_size: int, stride: int | None = None) -> list[tuple[int, int]]:
    """返回合法、有限且确定排序的PH候选起点，供训练和GT密度预检共同使用。

    积分图先检测每一个整数起点；返回的是规则网格加合法起点连通域的bbox边界/中心
    代表集，不是把全部整数起点都交给PH。中心优先、之后按距离和坐标固定排序；
    forward再按step轮换。只由valid决定，不按GT环数量或模型预测筛选。
    """
    return list(_eligible_origins_info(valid, tile_size, stride)[0])


def _finite_h1(
    probability: np.ndarray,
    *,
    min_persistence: float,
    border_margin: int,
    max_features: int,
) -> dict:
    """对1-p的cubical下水平过滤算边界p的超水平集H1。"""
    try:
        import gudhi
    except ImportError as exc:
        raise RuntimeError('拓扑微调需要显式安装 gudhi==3.11.0') from exc
    field = np.asfortranarray(1.0 - np.asarray(probability, dtype=np.float64))
    height, width = field.shape
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=field)
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0.0)
    regular, essential = complex_.cofaces_of_persistence_pairs()
    pairs = regular[1] if len(regular) > 1 else np.empty((0, 2), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    flat = field.ravel(order='F')
    rows = pairs % height
    cols = pairs // height
    diagrams = flat[pairs] if len(pairs) else np.empty((0, 2), dtype=np.float64)
    if len(diagrams) and np.any(diagrams[:, 1] < diagrams[:, 0] - 1e-12):
        raise RuntimeError('GUDHI临界像素顺序/映射与持久图不一致')
    persistence = diagrams[:, 1] - diagrams[:, 0]
    keep = persistence > float(min_persistence)
    below_threshold = int((~keep).sum())
    near_frame = (
        (rows < border_margin) | (rows >= height - border_margin)
        | (cols < border_margin) | (cols >= width - border_margin)
    ).any(axis=1)
    ignored_border = int((keep & near_frame).sum())
    keep &= ~near_frame
    indices = np.flatnonzero(keep)
    # 保留最持久的点；截断显式记录，避免稠密噪声导致非受控匹配开销。
    indices = indices[np.argsort(-persistence[indices], kind='stable')]
    truncated = max(0, len(indices) - max_features)
    indices = indices[:max_features]
    selected = pairs[indices]
    return {
        'diagram': diagrams[indices],
        'rows': selected % height,
        'cols': selected // height,
        'all_finite_h1': len(pairs),
        'below_persistence': below_threshold,
        'ignored_border_h1': ignored_border,
        'truncated_h1': truncated,
        'essential_h1': len(essential[1]) if len(essential) > 1 else 0,
    }


def _diagram_matching(prediction: np.ndarray, target: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """由GT有限H1数量约束的平方欧氏持久图匹配，多余预测点送对角线。

    二值GT的有限H1通常为(0,1)。先强制匹配min(P,T)个应有结构；否则自由
    Wasserstein匹配会把真实弱环送对角线并接受漏GT的常量代价，给出与补口相反的
    梯度。这里明确不是自由Wasserstein距离，也不是逐晶粒空间对应。
    """
    count_p, count_t = len(prediction), len(target)
    if not count_t:
        return [], list(range(count_p)), []
    if not count_p:
        return [], [], list(range(count_t))
    pair_cost = ((target[:, None, :] - prediction[None, :, :]) ** 2).sum(axis=2)
    target_rows, choices = linear_sum_assignment(pair_cost)
    matches = [(int(p), int(t)) for t, p in zip(target_rows, choices)]
    used_p = {p for p, _ in matches}
    used_t = {t for _, t in matches}
    return matches, [p for p in range(count_p) if p not in used_p], [t for t in range(count_t) if t not in used_t]


class BoundaryTopologyLoss(nn.Module):
    """完全已知小窗上的有限H1损失，返回(loss, audit)。

    默认每图仅取一个合法窗，按step循环固定候选，不消耗训练的随机数流。
    积分图检测全部合法起点，再用规则网格及合法域边界/中心构造有限候选。
    loss为“每窗匹配误差之和”的窗均值；不再按全图像素数稀释关键点梯度。
    """

    def __init__(
        self,
        tile_size: int = 64,
        max_tiles_per_image: int = 1,
        min_persistence: float = 0.01,
        border_margin: int = 2,
        stride: int | None = None,
        max_pred_features: int = 256,
        target_cache_size: int = 256,
    ):
        super().__init__()
        if tile_size < 8 or max_tiles_per_image < 1:
            raise ValueError('tile_size须至少8，max_tiles_per_image须为正整数')
        if not 0 <= min_persistence < 1:
            raise ValueError('min_persistence必须位于[0,1)')
        if border_margin < 0 or 2 * border_margin >= tile_size:
            raise ValueError('border_margin不能吞掉整个tile')
        stride = tile_size // 2 if stride is None else int(stride)
        if stride < 1 or stride > tile_size:
            raise ValueError('stride须位于[1,tile_size]')
        if max_pred_features < 1 or target_cache_size < 0:
            raise ValueError('特征上限须为正，缓存容量不能为负')
        self.tile_size = int(tile_size)
        self.max_tiles_per_image = int(max_tiles_per_image)
        self.min_persistence = float(min_persistence)
        self.border_margin = int(border_margin)
        self.stride = stride
        self.max_pred_features = int(max_pred_features)
        self.target_cache_size = int(target_cache_size)
        self._target_cache: OrderedDict[str, dict] = OrderedDict()
        self.architecture = dict(
            format=TOPOLOGY_FORMAT, tile_size=self.tile_size,
            max_tiles_per_image=self.max_tiles_per_image,
            min_persistence=self.min_persistence, border_margin=self.border_margin,
            stride=self.stride, max_pred_features=self.max_pred_features,
            target_cache_size=self.target_cache_size,
            origin_policy=ORIGIN_POLICY,
            filtration='1-boundary_probability', homology_dimensions=[1],
            matching='gt_count_constrained_squared_persistence_assignment',
            normalization='mean_tiles_sum_pairs',
            missing_target='constant_diagonal_cost_no_fabricated_gradient',
        )

    def _target(self, target: np.ndarray) -> tuple[dict, bool]:
        key = hashlib.sha256(np.ascontiguousarray(target, dtype=np.uint8).tobytes()).hexdigest()
        if key in self._target_cache:
            value = self._target_cache.pop(key)
            self._target_cache[key] = value
            return value, True
        value = _finite_h1(
            target, min_persistence=1e-8, border_margin=self.border_margin,
            max_features=self.tile_size * self.tile_size,
        )
        if self.target_cache_size:
            self._target_cache[key] = value
            while len(self._target_cache) > self.target_cache_size:
                self._target_cache.popitem(last=False)
        return value, False

    @staticmethod
    def _validate(probability, target, valid):
        if not torch.is_tensor(probability) or probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError('probability必须为B1HW张量')
        if not probability.is_floating_point() or not bool(torch.isfinite(probability).all()):
            raise ValueError('probability必须为有限浮点张量')
        if bool(((probability < 0) | (probability > 1)).any()):
            raise ValueError('probability必须位于[0,1]')
        if target.shape != probability.shape or valid.shape != probability.shape:
            raise ValueError('probability/target/valid必须同为B1HW且shape一致')
        if not bool(torch.isfinite(target).all()) or not bool(torch.isfinite(valid).all()):
            raise ValueError('target/valid必须有限')
        if not bool(((target == 0) | (target == 1)).all()):
            raise ValueError('PH target必须是二值边界，而非软热图')
        if not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError('valid必须为0/1或bool，不能隐式阈值未知域')

    def forward(self, probability, target, valid, *, step: int = 0):
        self._validate(probability, target, valid)
        if int(step) < 0:
            raise ValueError('step不能为负')
        batch, _, height, width = probability.shape
        target_np = target.detach().cpu().numpy()[:, 0].astype(np.uint8, copy=False)
        valid_np = valid.detach().cpu().numpy()[:, 0].astype(bool, copy=False)
        probability_np = probability.detach().float().cpu().numpy()[:, 0]
        audit = dict(
            format=TOPOLOGY_FORMAT, images=int(batch), step=int(step),
            candidate_tiles=0, eligible_tiles=0, selected_tiles=0, used_tiles=0,
            all_legal_origins=0, origin_policy=ORIGIN_POLICY,
            skipped_unknown_tiles=0, images_without_eligible_tiles=0,
            pred_h1=0, gt_h1=0, matched_h1=0, spurious_h1=0,
            unmatched_target_h1=0, critical_points=0,
            pred_ignored_border_h1=0, gt_ignored_border_h1=0,
            pred_below_persistence=0, pred_truncated_h1=0,
            pred_essential_h1_ignored=0, target_cache_hits=0,
            matched_loss=0.0, spurious_loss=0.0, missing_target_constant_loss=0.0,
            selected_origins=[],
        )
        tiles = []
        zeros = probability.sum() * 0.0
        t = self.tile_size
        for image_index in range(batch):
            eligible, candidate_count, legal_count = _eligible_origins_info(valid_np[image_index], t, self.stride)
            audit['candidate_tiles'] += candidate_count
            audit['all_legal_origins'] += legal_count
            audit['skipped_unknown_tiles'] += candidate_count - legal_count
            audit['eligible_tiles'] += len(eligible)
            if not eligible:
                audit['images_without_eligible_tiles'] += 1
                continue
            offset = (int(step) * self.max_tiles_per_image + image_index) % len(eligible)
            selection = [eligible[(offset + j) % len(eligible)] for j in range(min(self.max_tiles_per_image, len(eligible)))]
            for y, x in selection:
                audit['selected_tiles'] += 1
                audit['selected_origins'].append([image_index, y, x])
                pred_info = _finite_h1(
                    probability_np[image_index, y:y+t, x:x+t],
                    min_persistence=self.min_persistence, border_margin=self.border_margin,
                    max_features=self.max_pred_features,
                )
                gt_info, hit = self._target(target_np[image_index, y:y+t, x:x+t])
                audit['target_cache_hits'] += int(hit)
                pd, gd = pred_info['diagram'], gt_info['diagram']
                audit['pred_h1'] += len(pd)
                audit['gt_h1'] += len(gd)
                audit['pred_ignored_border_h1'] += pred_info['ignored_border_h1']
                audit['gt_ignored_border_h1'] += gt_info['ignored_border_h1']
                audit['pred_below_persistence'] += pred_info['below_persistence']
                audit['pred_truncated_h1'] += pred_info['truncated_h1']
                audit['pred_essential_h1_ignored'] += pred_info['essential_h1']
                matches, spurious, missing = _diagram_matching(pd, gd)
                audit['matched_h1'] += len(matches)
                audit['spurious_h1'] += len(spurious)
                audit['unmatched_target_h1'] += len(missing)
                if not len(pd) and not len(gd):
                    continue
                audit['used_tiles'] += 1
                tile = probability[image_index, 0, y:y+t, x:x+t]
                matched_loss, spurious_loss = zeros, zeros
                # 一窗一次向量化gather；逐临界点Python循环会产生大量小CUDA核。
                if len(pd):
                    rows = torch.as_tensor(pred_info['rows'], device=tile.device, dtype=torch.long)
                    cols = torch.as_tensor(pred_info['cols'], device=tile.device, dtype=torch.long)
                    critical_values = 1.0 - tile[rows, cols]
                    if matches:
                        matched_indices = torch.as_tensor([p for p, _ in matches], device=tile.device)
                        expected = torch.as_tensor(gd[[g for _, g in matches]], device=tile.device, dtype=tile.dtype)
                        matched_loss = (critical_values[matched_indices] - expected).square().sum()
                    if spurious:
                        extra = critical_values[torch.as_tensor(spurious, device=tile.device)]
                        spurious_loss = 0.5 * (extra[:, 0] - extra[:, 1]).square().sum()
                missing_cost = float(sum((gd[index, 1] - gd[index, 0]) ** 2 / 2.0 for index in missing))
                audit['critical_points'] += 2 * (len(matches) + len(spurious))
                audit['matched_loss'] += float(sum(((pd[p] - gd[g]) ** 2).sum() for p, g in matches))
                audit['spurious_loss'] += float(sum(0.5 * (pd[p, 0] - pd[p, 1]) ** 2 for p in spurious))
                audit['missing_target_constant_loss'] += missing_cost
                tiles.append(matched_loss + spurious_loss + missing_cost)
        # 空窗不当作零拓扑误差加入均值，且是否完全无信号由audit明确区分。
        loss = torch.stack(tiles).mean() if tiles else zeros
        audit['loss'] = float(loss.detach())
        audit['has_differentiable_signal'] = audit['critical_points'] > 0
        return loss, audit

