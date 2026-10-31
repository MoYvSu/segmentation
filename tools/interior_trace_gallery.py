# -*- coding: utf-8 -*-
"""固定ROI的冻结末态对照图；实例色按记录GT角色而非预测ID选取。

只渲染传入数组，不读取模型、源文件或服务器；不增强/锐化原图。
GT仅用来固定视野及颜色角色，不能解释为物理组织真值。
"""
from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path

import cv2
import numpy as np


def _ascii(value, name):
    if not isinstance(value, str) or not value.isascii():
        raise ValueError(name+' must be ASCII text')
    return value


def _probability(value, shape, name, *, signed=False):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError(name+' must be matching finite float32 HW')
    if not signed and np.any((value < 0) | (value > 1)):
        raise ValueError(name+' must lie in [0,1]')
    return value


def _color(gid):
    # 仅GT编号选固定色；不读取或哈希任何预测ID。
    hue = (int(gid)*.6180339887498949+.07) % 1.
    return np.rint(np.asarray(colorsys.hsv_to_rgb(hue, .63, .88))*255).astype(np.uint8)


def _role_colors(instances, gt):
    """局部dominant GT定色；同GT最大重叠主块用原色，其它片用固定浅色。

    重叠数量并列时按块的首个空间位置决定主块，不以预测ID排序打破并列。
    merge覆盖多个GT仍只有一个预测色，避免跨界分色伪装成已分开。
    """
    flat_pred, flat_gt = instances.ravel(), gt.ravel()
    predicted = np.unique(flat_pred)
    roles, by_gt = {}, {}
    for pred in predicted:
        if not pred:
            continue
        locations = np.flatnonzero(flat_pred == pred)
        known = flat_gt[locations]
        ids, counts = np.unique(known[known > 0], return_counts=True)
        if not len(ids):
            gid, overlap = 0, 0
        else:
            maximum = counts.max()
            # GT并列用同一固定记录ID顺序；预测ID不参与判定。
            index = int(np.flatnonzero(counts == maximum)[0])
            gid, overlap = int(ids[index]), int(counts[index])
        roles[int(pred)] = dict(pred=int(pred), recorded_dominant_gt=gid,
                                dominant_overlap_pixels=overlap,
                                local_pixels=len(locations), first_flat_position=int(locations[0]), role='unknown')
        if gid:
            by_gt.setdefault(gid, []).append(int(pred))
    for gid, predictions in by_gt.items():
        main = min(predictions, key=lambda pred:(-roles[pred]['dominant_overlap_pixels'], roles[pred]['first_flat_position']))
        for pred in predictions:
            roles[pred]['role'] = 'main' if pred == main else 'fragment'
    colors = np.zeros((65536, 3), np.uint8)
    for pred, row in roles.items():
        if row['recorded_dominant_gt'] == 0:
            color = np.array([142, 142, 142], np.uint8)
        else:
            color = _color(row['recorded_dominant_gt'])
            if row['role'] == 'fragment':
                color = np.rint(.65*color.astype(float)+.35*255.).astype(np.uint8)
        colors[pred] = color
    return colors, list(roles.values())


def _instance_overlay(rgb, instances, gt):
    colors, roles = _role_colors(instances, gt)
    result = rgb.copy()
    visible = instances > 0
    result[visible] = np.rint(.45*rgb[visible].astype(float)+.55*colors[instances[visible]].astype(float)).astype(np.uint8)
    edges = np.zeros(instances.shape, bool)
    horizontal, vertical = instances[:, :-1] != instances[:, 1:], instances[:-1] != instances[1:]
    # 双侧真实预测ID差异；不从GT描出预测边界。
    edges[:, :-1] |= horizontal; edges[:, 1:] |= horizontal
    edges[:-1] |= vertical; edges[1:] |= vertical
    result[edges] = [18, 18, 18]
    return result, roles


def _heat(value):
    bgr = cv2.applyColorMap(np.rint(value*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _shift(value, scale):
    magnitude = np.clip(np.abs(value)/scale, 0., 1.)[..., None]
    positive = np.array([202., 50., 50.])
    negative = np.array([47., 84., 203.])
    end = np.where((value >= 0)[..., None], positive, negative)
    return np.rint(238.*(1.-magnitude)+end*magnitude).astype(np.uint8)


def _text(image, text, position, scale=.46):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (15, 15, 15), 1, cv2.LINE_AA)


def save_gallery(path, rgb, gt, gt_ids, rows, title):
    """保存多行PNG，返回可JSON封存的视野、颜色角色及文件哈希。

    rows每项需label,boundary,instances,marker_belt；short与long_shift可选。
    所有图保留完整原生HW，统一裁GT目标bbox外扩50px；缺少可选列写n/a。
    原图列保持真实RGB；实例列同GT主块/片块角色跨行稳定，黑边为预测ID。
    long_shift用全部行统一对称色阶，蓝表示负、红表示正，不逐行自动缩放。
    目的地应由调用者置于忽略目录；已有文件拒绝覆盖。
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError('Fresh gallery path required: '+str(path))
    if path.suffix.lower() != '.png':
        raise ValueError('Gallery destination must be PNG')
    _ascii(title, 'title')
    rgb, gt = np.asarray(rgb), np.asarray(gt)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError('rgb must be nonempty uint8 HWC RGB')
    shape = rgb.shape[:2]
    if (gt.shape != shape or not np.issubdtype(gt.dtype, np.integer)
            or gt.dtype == np.bool_ or np.any(gt < 0)):
        raise ValueError('gt must be matching nonnegative integer HW')
    if (not isinstance(gt_ids, (list, tuple)) or not gt_ids
            or any(isinstance(gid, (bool, np.bool_)) or not isinstance(gid, (int, np.integer))
                   or gid <= 0 or not np.any(gt == gid) for gid in gt_ids)):
        raise ValueError('gt_ids must name present positive recorded GT IDs')
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('At least one comparison row required')
    region = np.isin(gt, gt_ids)
    yy, xx = np.nonzero(region)
    y0, y1 = max(0, int(yy.min())-50), min(shape[0], int(yy.max())+51)
    x0, x1 = max(0, int(xx.min())-50), min(shape[1], int(xx.max())+51)
    crop = (slice(y0, y1), slice(x0, x1))
    raw, labels = rgb[crop], gt[crop]
    prepared, include_short, include_shift = [], False, False
    shift_scale = .05
    for row in rows:
        _ascii(row['label'], 'row label')
        boundary = _probability(row['boundary'], shape, 'boundary')
        instances = np.asarray(row['instances'])
        belt = np.asarray(row['marker_belt'])
        if instances.shape != shape or instances.dtype != np.uint16:
            raise ValueError('instances must be matching single-channel uint16 HW')
        if belt.shape != shape or belt.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
            raise ValueError('marker_belt must be matching bool/uint8 HW')
        item = dict(label=row['label'], boundary=boundary[crop], instances=instances[crop], belt=belt[crop] > 0)
        if row.get('short') is not None:
            item['short'] = _probability(row['short'], shape, 'short')[crop]
            include_short = True
        if row.get('long_shift') is not None:
            item['long_shift'] = _probability(row['long_shift'], shape, 'long_shift', signed=True)[crop]
            include_shift = True
            shift_scale = max(shift_scale, float(np.abs(item['long_shift']).max()))
        prepared.append(item)
    column_names = ['raw RGB', 'boundary [0,1]', 'marker belt on raw', 'instances on raw']
    if include_short:
        column_names.append('short [0,1]')
    if include_shift:
        column_names.append('long shift +/-'+format(shift_scale, '.3f'))
    width = 320
    height = max(1, round(raw.shape[0]*width/raw.shape[1]))
    gap, row_header, top_header = 5, 27, 78
    canvas = np.full((top_header+len(prepared)*(height+row_header+gap),
                      len(column_names)*width+(len(column_names)-1)*gap, 3), 255, np.uint8)
    _text(canvas, title, (5, 20), .55)
    _text(canvas, 'Recorded GT roles only, not physical truth. Black edges: predicted IDs. Raw RGB unchanged.', (5, 41), .42)
    _text(canvas, 'GT targets: '+','.join(str(int(g)) for g in gt_ids)+' | Main: fixed color; fragments: pale; unknown: gray.', (5, 61), .42)
    report_rows = []
    for index, row in enumerate(prepared):
        belt_image = raw.copy()
        belt_image[row['belt']] = np.rint(.45*raw[row['belt']].astype(float)+.55*np.array([255., 52., 157.])).astype(np.uint8)
        instance_image, roles = _instance_overlay(raw, row['instances'], labels)
        panels = [raw, _heat(row['boundary']), belt_image, instance_image]
        if include_short:
            panels.append(_heat(row['short']) if 'short' in row else None)
        if include_shift:
            panels.append(_shift(row['long_shift'], shift_scale) if 'long_shift' in row else None)
        top = top_header+index*(height+row_header+gap)
        for column, panel in enumerate(panels):
            left = column*(width+gap)
            _text(canvas, row['label']+' | '+column_names[column], (left+4, top+18), .4)
            if panel is None:
                _text(canvas, 'n/a', (left+width//2-15, top+row_header+height//2), .55)
                continue
            interpolation = cv2.INTER_AREA if width <= raw.shape[1] else cv2.INTER_LINEAR
            resized = cv2.resize(panel, (width, height), interpolation=interpolation)
            canvas[top+row_header:top+row_header+height, left:left+width] = resized
        report_rows.append(dict(label=row['label'], instance_roles=roles))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Could not save gallery PNG')
    return dict(format='interior_trace_gallery_v1', path=str(path.resolve()),
                PNG_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bbox_native=[y0, x0, y1, x1], shape=list(canvas.shape), columns=column_names,
                long_shift_symmetric_scale=shift_scale if include_shift else None,
                role_rule='ROI dominant recorded GT; spatially tie-broken largest local overlap is main; predicted IDs never select colors',
                GT_not_physical_truth=True, raw_unsharpened=True, rows=report_rows)
