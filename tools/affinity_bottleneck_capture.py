# -*- coding: utf-8 -*-
"""透明捕获任意源图的正式原生1024窗口，供冻结关系反事实使用。

不训练、不写文件、不选择GT关系，也不改原forward的返回值。GPU logits保留
实际设备和stride，禁止从CPU连续数组重建logits后冒充原CUDA融合。
"""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from tools.run_marker_reflect import capture_original
from train_backend_adaptation import fusion_options
from utils.affinity_fusion import affinity_boundary_probability
from utils.offset_letterbox import letterbox_instance_geometry
from utils.patch_diagnostic import BlendMap, tile_boxes


def capture_native_tiles(views, model, restorer, path, config, gt, device, seed):
    """返回 ``rgb, captured, tiles, audit``；所有状态在退出/异常时恢复。

    captured原样沿用capture_original的实际decode/marker/WS/rawvote记录。
    tiles的logits为原设备detach tensor；grid/native为保留BCHW的CPU numpy
    副本，geometry为letterbox元数据dict，grid_gt/valid仅编码实际有效内容。
    logits/grid/grid_gt/valid使用512输出网格坐标；box/native使用原图像素坐标。
    本模块不计算路径，调用者必须按geometry显式映射，不能混用两种坐标。
    CPU仅用于虚构smoke；真实运行要求传入实际CUDA设备，dtype均严格为FP32。
    调用者负责封存配置、源码、输入与模型哈希，并确保原49 views版本正确。
    """
    gt = np.asarray(gt)
    if (gt.ndim != 2 or not gt.size or not np.issubdtype(gt.dtype, np.integer)
            or gt.dtype == np.bool_ or np.any(gt < 0)
            or int(gt.max()) > np.iinfo(np.int32).max):
        raise ValueError('gt must be a nonempty nonnegative integer HW map')
    overlap = config['affinity_native']['overlap']
    if isinstance(overlap, bool) or float(overlap) != .25:
        raise ValueError('Capture requires the fixed native1024 overlap .25')
    requested_device = torch.device(device)
    boxes = tile_boxes(gt.shape, 1024, .25)
    original_grid = views.boundary_grid
    original_crop = views.crop_letterbox_output
    mode, options = fusion_options(config)
    tiles, pending = [], {}
    skipped_crops = 0

    def observe_head(_module, _inputs, result):
        if not pending.get('active'):
            return None
        if 'logits' in pending:
            raise RuntimeError('More than one affinity head forward inside an actual tile')
        if not isinstance(result, dict) or 'affinity_logits' not in result:
            raise RuntimeError('Actual affinity head must return affinity_logits')
        logits = result['affinity_logits']
        if (not torch.is_tensor(logits) or tuple(logits.shape) != (1, 8, 512, 512)
                or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all())):
            raise RuntimeError('Actual tile logits must be finite FP32 [1,8,512,512]')
        if (logits.device.type != requested_device.type
                or (requested_device.index is not None
                    and logits.device.index != requested_device.index)):
            raise RuntimeError('Actual tile logits device differs from requested device')
        # detach保留原storage/stride；不返回任何值替换原模型输出。
        pending['logits'] = logits.detach()
        return None

    def observed_grid(*args, **kwargs):
        if pending or len(tiles) >= len(boxes):
            raise RuntimeError('Actual native tile order or crop order differs')
        pending['active'] = True
        try:
            grid = original_grid(*args, **kwargs)
        finally:
            pending['active'] = False
        if 'logits' not in pending:
            raise RuntimeError('Actual tile did not call the observed affinity head')
        logits = pending.pop('logits')
        if (not torch.is_tensor(grid) or tuple(grid.shape) != (1, 1, 512, 512)
                or grid.dtype != torch.float32 or grid.device != logits.device):
            raise RuntimeError('Actual tile boundary must be FP32 [1,1,512,512] on logits device')
        empty = torch.zeros_like(logits, dtype=torch.bool)
        empty_logits = torch.where(empty, logits.new_tensor(-16.), logits)
        empty_grid = affinity_boundary_probability(empty_logits.float(), mode=mode, **options)
        if not torch.equal(empty_grid, grid):
            raise RuntimeError('Actual empty-mask fusion is not exact')
        index = len(tiles)
        y, x, y1, x1 = boxes[index]
        grid_gt, valid, geometry = letterbox_instance_geometry(gt[y:y1, x:x1], 1024, 512)
        tile = dict(index=index, box=list(boxes[index]),
                    seed=int(seed) + 100000 + 1024 * 1000 + index,
                    logits=logits, logits_shape=list(logits.shape),
                    logits_stride=list(logits.stride()),
                    grid=grid.detach().cpu().numpy().copy(),
                    grid_gt=grid_gt, valid=valid, geometry=geometry.to_dict(),
                    empty_grid_exact=True)
        pending.update(tile=tile, grid=grid, empty_grid=empty_grid)
        return grid

    def observed_crop(*args, **kwargs):
        nonlocal skipped_crops
        native = original_crop(*args, **kwargs)
        if 'tile' not in pending:
            # 原全图恢复语义与raw语义先于局部边界；不把这些crop误当窗口。
            skipped_crops += 1
            return native
        if len(args) < 5 or args[0] is not pending['grid']:
            raise RuntimeError('Actual native crop differs from captured tile grid')
        tile = pending['tile']
        y, x, y1, x1 = tile['box']
        geometry = tile['geometry']
        if (int(args[1]) != 1024 or tuple(args[4]) != (y1-y, x1-x)
                or int(args[2]) != 1024-geometry['resized_height']
                or int(args[3]) != 1024-geometry['resized_width']):
            raise RuntimeError('Actual RGB crop geometry differs from tile GT letterbox')
        if (not torch.is_tensor(native) or tuple(native.shape) != (1, 1, y1-y, x1-x)
                or native.dtype != torch.float32):
            raise RuntimeError('Actual native tile crop must be FP32 [1,1,H,W]')
        empty_native = original_crop(pending['empty_grid'], *args[1:], **kwargs)
        if not torch.equal(empty_native, native):
            raise RuntimeError('Actual empty-mask native crop is not exact')
        tile.update(native=native.detach().cpu().numpy().copy(),
                    empty_native=empty_native.detach().cpu().numpy().copy(),
                    pad_h=int(args[2]), pad_w=int(args[3]), empty_native_exact=True)
        tiles.append(tile)
        pending.clear()
        return native

    hook = model.affinity_decoder.register_forward_hook(observe_head)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(views, 'boundary_grid', side_effect=observed_grid))
            stack.enter_context(patch.object(views, 'crop_letterbox_output', side_effect=observed_crop))
            with torch.no_grad():
                rgb, captured = capture_original(views, model, restorer, path, config, device, seed)
    finally:
        hook.remove()
    if pending or len(tiles) != len(boxes) or tuple(rgb.shape[:2]) != tuple(gt.shape):
        raise RuntimeError('Actual source shape or complete native tile count differs')

    original_blend, empty_blend = BlendMap(gt.shape), BlendMap(gt.shape)
    for tile in tiles:
        original_blend.add(tile['box'], tile['native'][0, 0])
        empty_blend.add(tile['box'], tile['empty_native'][0, 0])
    original_boundary, original_audit = original_blend.finish()
    empty_boundary, empty_audit = empty_blend.finish()
    original_audit['tiles'] = empty_audit['tiles'] = len(tiles)
    baseline = captured['baseline']
    if (not np.array_equal(original_boundary, baseline['boundary'])
            or original_audit != baseline['blend']):
        raise RuntimeError('Actual original-order native BlendMap is not exact')
    if (not np.array_equal(empty_boundary, baseline['boundary'])
            or empty_audit != baseline['blend']):
        raise RuntimeError('Actual empty-mask native BlendMap is not exact')
    audit = dict(format='affinity_bottleneck_capture_v1', tile_count=len(tiles),
                 output_shape=list(gt.shape), seed=int(seed), input_size=1024,
                 output_grid=512, overlap=.25, device=str(requested_device), dtype='FP32',
                 single_predict_views_call=True, no_hook_return_change=True,
                 original_order_baseline_stitch_exact=True,
                 empty_mask_grid_native_and_stitch_exact=True,
                 baseline_blend=original_audit, empty_blend=empty_audit,
                 skipped_non_tile_crops=skipped_crops,
                 intervention_fusion_device=str(tiles[0]['logits'].device))
    return rgb, captured, tiles, audit
