# -*- coding: utf-8 -*-
"""原生裁块适配的固定全图语义、局部边界拼接和过程缩略图。"""
from __future__ import annotations

from pathlib import Path
from contextlib import ExitStack
import hashlib
import cv2
import numpy as np
import torch

from data.mim_dataset import list_images
from models.backend_adaptation import restore_first
from tools.affinity_connectivity_views import _inference_state, decode_native
from tools.analyze_affinity_connectivity import load_directory, render
from tools.probe_affinity_patch import boundary_grid, tensor_from_rgb
from tools.render_backend_ablation import text_line, write_png
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import prediction_maps, write_json
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.config import project_path
from utils.patch_diagnostic import BlendMap, phase_description, tile_boxes


def verify_prediction(ids, classes, shape):
    if ids.shape != tuple(shape) or ids.ndim != 2 or ids.dtype != np.uint16 or int(ids.max()) > 65535:
        raise RuntimeError('Invalid native final instance PNG')
    if {str(int(i)) for i in np.unique(ids) if i} != set(classes) or any(c not in (0, 1) for c in classes.values()):
        raise RuntimeError('Instance and class contract differs')


@torch.no_grad()
def predict_views(model, restorer, path, config, device, seed, sizes, *, local_restorer=None, fixed_audit=None):
    """各尺度复用同一全图两路语义；连续边界拼回原图后各作一次全局划分。"""
    results = {}
    with ExitStack() as stack:
        stack.enter_context(_inference_state(model, restorer, device))
        if local_restorer is not None and local_restorer is not restorer:
            stack.enter_context(_inference_state(local_restorer, None, device))
        rgb, semantic, global_boundary = prediction_maps(model, path, config, device, restorer, seed)
        _, tensor, ph, pw = prepare_image(path, 1024, device)
        raw = model.semantic_decoder(model.encoder(tensor.float()), tensor.float())
        if isinstance(raw, dict):
            raw = raw['semantic_logits']
        probability = crop_letterbox_output(raw.float(), 1024, ph, pw, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()
        if fixed_audit is not None:
            fixed_audit.update(
                geometry_semantic_sha256=hashlib.sha256(semantic.contiguous().numpy().tobytes()).hexdigest(),
                raw_probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest())
        for size in sizes:
            if int(size) == 0:
                boundary, audit = global_boundary[0, 0].numpy(), dict(tiles=1)
            else:
                blend = BlendMap(rgb.shape[:2])
                boxes = tile_boxes(rgb.shape[:2], int(size), config['affinity_native']['overlap'])
                for index, box in enumerate(boxes):
                    y, x, y1, x1 = box
                    image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], device)
                    restored = restore_first(restorer if local_restorer is None else local_restorer,
                                             image, seed + 100000 + int(size) * 1000 + index)
                    grid = boundary_grid(model, restored, config)
                    native = crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
                    blend.add(box, native)
                boundary, audit = blend.finish()
                audit['tiles'] = len(boxes)
            ids, _ = decode_native(semantic, torch.from_numpy(boundary)[None, None], config)
            classes, _ = revote_instances(ids, probability)
            verify_prediction(ids, classes, rgb.shape[:2])
            name = 'global' if size == 0 else f'patch{size}'
            results[name] = dict(instances=ids, classes=classes, boundary=boundary,
                                 phases=phase_description(ids, classes), blend=audit)
    return rgb, results


def save_monitor(model, restorer, config, device, output, epoch, size, *, smoke=False,
                 reference_folder=None, reference_label='control whole'):
    names = config['backend_adaptation']['monitor']['images'][:2] if smoke else config['backend_adaptation']['monitor']['images']
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    if not set(names).issubset({p.stem for p in files}):
        raise RuntimeError('Missing fixed monitor images')
    folder = Path(output) / 'monitor' / f'epoch_{epoch:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(files):
        if path.stem not in names:
            continue
        seed = config['backend_adaptation']['restoration']['inference_seed'] + index
        rgb, views = predict_views(model, restorer, path, config, device, seed, [0, size])
        reference = (project_path(config, config['affinity_native']['reused_control'], 'deployment')
                     if reference_folder is None else reference_folder)
        control = load_directory(reference, path.stem)
        predictions = [control] + [(v['instances'], v['classes']) for v in views.values()]
        labels = [reference_label, f'e{epoch} whole', f'e{epoch} native{size}']
        for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
            render(path, predictions, labels, folder / f'{path.stem}_{suffix}.png',
                   f'{path.stem} e{epoch} F gold / P blue', crop, config['affinity_native']['monitor_width'])
        boundary = views[f'patch{size}']['boundary']
        h, w = boundary.shape
        thumbnail = cv2.resize(boundary, (400, round(h*400/w)), interpolation=cv2.INTER_AREA)
        heat = cv2.applyColorMap(np.rint(np.clip(thumbnail, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        write_png(folder / f'{path.stem}_boundary.png', heat)
        rows.append(dict(source=path.stem, seed=seed,
                         views={k: dict(phases=v['phases'], blend=v['blend']) for k, v in views.items()}))
    write_json(folder / 'summary.json', dict(epoch=epoch, images=rows,
               scope='fixed unlabeled process thumbnails, no validation or checkpoint selection'))


def save_training_draw(folder, batch, restored, epoch, step):
    """实际训练视图：清晰输入／退化／恢复／实例界面，不额外随机采样。"""
    ids = batch['affinity_instance_map'][0].detach().cpu().numpy()
    edge = np.zeros(ids.shape, bool)
    edge[1:] |= (ids[1:] != ids[:-1]) & (ids[1:] > 0) & (ids[:-1] > 0)
    edge[:, 1:] |= (ids[:, 1:] != ids[:, :-1]) & (ids[:, 1:] > 0) & (ids[:, :-1] > 0)
    panels = []
    for title, value in [('Clear crop', batch['clean_image']), ('Degraded', batch['image']), ('D5a', restored)]:
        rgb = value[0].detach().cpu().permute(1, 2, 0).numpy()
        panels.append((title, cv2.cvtColor(np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)))
    label = np.full((*ids.shape, 3), 230, np.uint8)
    label[ids == 0] = (80, 80, 80)
    label[edge] = (0, 0, 0)
    panels.append(('GT interface / unknown gray', label))
    strips = []
    for title, body in panels:
        body = cv2.resize(body, (256, 256), interpolation=cv2.INTER_AREA)
        body = cv2.copyMakeBorder(body, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        text_line(body, title, (4, 18), 256, size=.4)
        strips.append(body)
    folder = Path(folder) / 'draws' / f'epoch_{epoch:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    write_png(folder / f'draw_{step:03d}.png', np.concatenate(strips, axis=1))
    write_json(folder / f'draw_{step:03d}.json', dict(source=batch['name'],
               crop_box=batch['crop_box'].tolist(), source_shape=batch['native_source_shape'].tolist()))
