# -*- coding: utf-8 -*-
"""连接性适配的原部署解码、坐标回放和固定过程缩略图。"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data.affinity_connectivity import undo_spatial
from data.direct_dual_head_dataset import _spatial_transform
from data.mim_dataset import list_images
from tools.render_backend_ablation import overlay, text_line, write_png
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import fusion_options, prediction_maps
from utils.affinity_deployment import crop_letterbox_output, prepare_image, probability_to_logit
from utils.affinity_fusion import affinity_boundary_probability
from utils.config import project_path
from utils.offset_letterbox import geometry_letterbox_metadata, letterbox_instance_geometry
from utils.post_process import boundary_watershed_separation, reconstruct_marker_boundary


def _vote_options(cfg):
    # 与 utils.affinity_deployment.postprocess 的显式默认值相同。
    defaults = {
        'core_fraction': .40, 'core_min_pixels': 8, 'core_distance_power': 2.,
        'color_uncertain_low': .35, 'color_uncertain_high': .65,
        'color_weight': .25, 'color_min_separation': 1.,
        'dual_p2f_base_min': .35, 'dual_p2f_candidate_min': .85,
        'dual_f2p_base_max': .65, 'dual_f2p_candidate_max': .15,
        'dual_p2f_min_core_gain': .08,
    }
    return {key: type(value)(cfg.get('semantic_vote_' + key, value))
            for key, value in defaults.items()}


@torch.no_grad()
def decode_native(semantic_logits, boundary_probability, config, *, marker_audit=None):
    """无文件副作用地复现原 affinity postprocess，输入已为原图尺寸。"""
    if (semantic_logits.ndim != 4 or semantic_logits.shape[:2] != (1, 1)
            or boundary_probability.shape != semantic_logits.shape):
        raise ValueError('Native maps must have the same [1,1,H,W] shape')
    cfg = config['inference']
    # 部署在 CPU 将概率转 logit 后再次 sigmoid；不能直接拿原概率做阈值。
    semantic = semantic_logits.detach().float().cpu()
    boundary = probability_to_logit(boundary_probability.detach().float().cpu())
    packed = F.interpolate(torch.cat([semantic, boundary], dim=1),
                           size=semantic.shape[-2:], mode='bilinear', align_corners=True)
    sem_prob = torch.sigmoid(packed[0, 0]).numpy()
    edge_prob = torch.sigmoid(packed[0, 1]).numpy()
    high = float(cfg['boundary_threshold'])
    marker = None
    low = cfg.get('marker_boundary_low_threshold')
    steps = int(cfg.get('marker_boundary_reconstruction_steps', 0))
    if low is not None and steps > 0:
        marker = reconstruct_marker_boundary(edge_prob, float(low), high, steps)
    return boundary_watershed_separation(
        (sem_prob > float(cfg.get('threshold', .5))).astype(np.uint8),
        (edge_prob > high).astype(np.uint8),
        dilate_width=int(cfg.get('watershed_dilate_width', 1)),
        min_area=int(cfg.get('min_instance_area', 50)),
        max_instance_id=int(cfg.get('max_instance_id', 65535)),
        bridge_width=int(cfg.get('bridge_width', 1)),
        center_prob=None, center_threshold=.25, center_nms_kernel=9,
        marker_boundary_mask=marker,
        marker_border_seal_width=int(cfg.get('marker_border_seal_width', 0)),
        semantic_probability=sem_prob, candidate_semantic_probability=None,
        semantic_vote_mode=str(cfg.get('semantic_vote_mode', 'hard_majority')),
        semantic_vote_erode_width=int(cfg.get('semantic_vote_erode_width', 0)),
        semantic_vote_threshold=float(cfg.get('semantic_vote_threshold', .5)),
        semantic_vote_options=_vote_options(cfg), semantic_lab_prior=None,
        semantic_vote_audit={}, marker_partition_restore=cfg.get('marker_partition_restore'),
        marker_restoration_audit={} if marker_audit is None else marker_audit,
    )


def _scalar(batch, name):
    value = torch.as_tensor(batch[name])
    if value.numel() != 1:
        raise ValueError(f'{name} must contain exactly one augmentation value')
    return value.item()


@torch.no_grad()
def training_partition(output, batch, config):
    """将当前输出按原尺寸完整解码，再回到增强后的 512 GT 坐标。

    output 使用 FusedPhaseAffinityModel 的 semantic_logits / affinity_logits。
    本轮仅支持 batch=1、1024 输入、512 affinity，返回同设备 long[1,512,512]。
    """
    affinity = output['affinity_logits'].detach().float()
    semantic = output['semantic_logits'].detach().float()
    if tuple(affinity.shape) != (1, 8, 512, 512):
        raise ValueError('Training partition requires affinity [1,8,512,512]')
    if semantic.ndim != 4 or semantic.shape[:2] != (1, 1):
        raise ValueError('Training partition requires semantic [1,1,H,W]')
    shape = torch.as_tensor(batch['native_shape']).detach().cpu()
    if tuple(shape.shape) != (1, 2) or bool((shape <= 0).any()):
        raise ValueError('native_shape must be canonical original height/width [1,2]')
    hflip = bool(_scalar(batch, 'horizontal_flip'))
    vflip = bool(_scalar(batch, 'vertical_flip'))
    turns = int(_scalar(batch, 'rotation_k'))
    native_shape = tuple(map(int, shape[0].tolist()))
    metadata = geometry_letterbox_metadata(native_shape, 1024, 512)
    pad_h, pad_w = 1024 - metadata.resized_height, 1024 - metadata.resized_width
    mode, kwargs = fusion_options(config)
    # 融合的局部支持与 top2 是非线性的，必须先在原 512 格点执行。
    boundary = affinity_boundary_probability(affinity, mode=mode, **kwargs)
    boundary = undo_spatial(boundary, hflip, vflip, turns)
    semantic = undo_spatial(semantic, hflip, vflip, turns)
    native_boundary = crop_letterbox_output(boundary, 1024, pad_h, pad_w, native_shape).cpu()
    native_semantic = crop_letterbox_output(semantic, 1024, pad_h, pad_w, native_shape).cpu()
    instances, _ = decode_native(native_semantic, native_boundary.clamp(0., 1.), config)
    ids, _, _ = letterbox_instance_geometry(instances, input_size=1024, output_grid=512)
    result = _spatial_transform(torch.from_numpy(ids.astype(np.int64)), hflip, vflip, turns)
    return result.unsqueeze(0).to(affinity.device)


@contextmanager
def _inference_state(model, restorer, device):
    """过程观察不改变混合 train/eval 状态或训练随机流。"""
    modules = {id(module): (module, module.training)
               for root in (model, restorer) if root is not None for module in root.modules()}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            model.eval()
            if restorer is not None:
                restorer.eval()
            yield
    finally:
        for module, training in modules.values():
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def predict_cross(model, restorer, path, config, device, seed):
    """D5a 图确定完整划分；相同冻结 S-align 的 raw 概率仅作末端分类。"""
    infer = config['inference']
    if (infer.get('semantic_vote_mode', 'hard_majority') != 'probability_mean'
            or int(infer.get('semantic_vote_erode_width', 0)) != 0
            or float(infer.get('semantic_vote_threshold', .5)) != .5):
        raise ValueError('Cross deployment requires probability_mean / erode0 / threshold0.5')
    with _inference_state(model, restorer, device):
        image, semantic, boundary = prediction_maps(model, path, config, device, restorer, int(seed))
        instances, _ = decode_native(semantic, boundary, config)
        _, raw, pad_h, pad_w = prepare_image(path, 1024, device)
        raw_logits = model.semantic_decoder(model.encoder(raw.float()), raw.float())
        raw_native = crop_letterbox_output(raw_logits.float(), 1024, pad_h, pad_w, image.shape[:2]).cpu()
        raw_probability = torch.sigmoid(raw_native)[0, 0].numpy()
        classes, _ = revote_instances(instances, raw_probability)
    return {'image_rgb': image, 'instances': instances, 'classes': classes,
            'raw_probability': raw_probability, 'boundary': boundary}


def save_monitor(model, restorer, config, device, outdir, epoch, names):
    """固定色阶三格过程图；种子始终取全部测试图排序中的位置。"""
    files = [Path(path) for path in list_images(project_path(config, config['inference']['test_dir']))]
    requested = [Path(name).stem for name in names]
    if len(set(requested)) != len(requested):
        raise ValueError('Monitor names must be unique')
    available = {path.stem for path in files}
    if set(requested) - available:
        raise ValueError(f'Missing monitor images: {sorted(set(requested) - available)}')
    target = Path(outdir) / 'monitor' / f'epoch_{int(epoch):03d}'
    target.mkdir(parents=True, exist_ok=True)
    records = []
    base_seed = int(config['backend_adaptation']['restoration'].get('inference_seed', 314159))
    for index, path in enumerate(files):
        if path.stem not in requested:
            continue
        result = predict_cross(model, restorer, path, config, device, base_seed + index)
        image = cv2.cvtColor(result['image_rgb'], cv2.COLOR_RGB2BGR)
        size = (320, max(1, round(image.shape[0] * 320 / image.shape[1])))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        ids = cv2.resize(result['instances'], size, interpolation=cv2.INTER_NEAREST)
        classes = {int(key): value for key, value in result['classes'].items()}
        edge = cv2.resize(result['boundary'][0, 0].cpu().numpy(), size, interpolation=cv2.INTER_AREA)
        heat = cv2.applyColorMap(np.rint(np.clip(edge, 0., 1.) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        panels = []
        for title, body in ((f'{path.stem} e{epoch}: original', image),
                            ('Affinity boundary: fixed 0..1', heat),
                            ('Final: F gold / P blue', overlay(image, ids, classes))):
            panel = cv2.copyMakeBorder(body, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            text_line(panel, title, (6, 20), size[0], size=.44)
            panels.append(panel)
        filename = target / (path.stem + '.png')
        write_png(filename, np.concatenate(panels, axis=1))
        records.append({'image': path.stem, 'seed': base_seed + index, 'path': str(filename),
                        'instances': len(classes), 'ferrite': sum(c == 1 for c in classes.values()),
                        'pearlite': sum(c == 0 for c in classes.values())})
    summary = {'epoch': int(epoch), 'source_count': len(files), 'images': records,
               'scope': 'unlabeled fixed process thumbnails; no checkpoint selection'}
    (target / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
