# -*- coding: utf-8 -*-
"""补边短训固定过程图；训练配对诊断与无标签终态变化分开报告。"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from tools.semantic_dino_monitor import prepare_monitor_folder
from utils.config import load_config, project_path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _heat(value):
    u8 = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def _edges(ids):
    ids = np.asarray(ids)
    edge = np.zeros(ids.shape, bool)
    edge[1:] |= ids[1:] != ids[:-1]
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    return edge


def _overlay(rgb, ids):
    result = rgb.copy()
    result[_edges(ids)] = (20, 245, 245)
    return result


def panels(path, title, cells, width=300, footer=''):
    """固定色阶、等比缩略，不逐图拉伸概率。"""
    resized = []
    for name, array in cells:
        height = max(1, round(array.shape[0] * width / array.shape[1]))
        resized.append((name, cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)))
    height = max(a.shape[0] for _, a in resized)
    canvas = Image.new('RGB', (width * len(resized), height + 100), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill='black')
    for i, (name, array) in enumerate(resized):
        draw.text((i*width + 5, 30), name, fill='black')
        canvas.paste(Image.fromarray(array), (i*width, 52))
    draw.text((6, height + 64), footer, fill='black')
    canvas.save(path, quality=90)


def paired_metrics(source, target, output, valid, gap, noise):
    """分母仅为相应可靠区域；没有独立准确率或物理正确性含义。"""
    result = {'valid_pixels': int(np.count_nonzero(valid))}
    for name, mask in [('valid', valid), ('gap', gap & valid), ('noise', noise & valid),
                       ('untouched', valid & ~gap & ~noise)]:
        count = int(np.count_nonzero(mask))
        result[name + '_pixels'] = count
        for label, value in [('before', source), ('after', output)]:
            result[name + '_' + label + '_mae'] = float(np.abs(value-target)[mask].mean()) if count else None
    result['valid_output_change_mae'] = float(np.abs(output-source)[valid].mean()) if valid.any() else None
    return result


def zoom_box(target, valid, gap, noise, side=128):
    """选真实受损像素，避免多个断口的坐标中位数落在两个断口之间。"""
    if np.any(gap & valid):
        mask, label = gap & valid, 'Gap'
    elif np.any(noise & valid):
        mask, label = noise & valid, 'Noise'
    else:
        mask, label = (target > .2) & valid, 'Unchanged'
    points = np.argwhere(mask)
    cy, cx = points[len(points)//2] if len(points) else (target.shape[0]//2, target.shape[1]//2)
    side = min(int(side), *target.shape)
    y = int(np.clip(cy-side//2, 0, target.shape[0]-side))
    x = int(np.clip(cx-side//2, 0, target.shape[1]-side))
    return (x, y, x+side, y+side), label


@contextmanager
def preserved_state(model):
    """过程图不消耗训练随机流，也不改变模式。"""
    import random
    python = random.getstate()
    numpy = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    training = model.training
    try:
        model.eval()
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(torch_state)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        model.train(training)


@torch.no_grad()
def monitor(model, config, output_dir, epoch, device):
    from train_boundary_restore import build_dataset, data_mode
    from tools.boundary_restore_readout import predict_residual_tiled, evaluate_frozen_readout

    cfg = config['boundary_restore']
    settings = cfg.get('monitor', {})
    width = int(settings.get('thumbnail_width', 300))
    mode = data_mode(config)
    dataset = build_dataset(config, seed_offset=917)
    dataset.set_epoch(1)
    folder, backup = prepare_monitor_folder(output_dir, int(epoch))
    rows = []
    with preserved_state(model):
        for index in settings.get('pair_indices', [0, 1, 2, 3, 4, 5]):
            sample = dataset[int(index)]
            source = sample['input'][0].numpy()
            target = sample['target'][0].numpy()
            valid = sample['valid'][0].numpy() > 0
            gap = sample['gap_mask'][0].numpy() > 0
            noise = sample['noise_mask'][0].numpy() > 0
            output = model(sample['input'][None].to(device))[0, 0].float().cpu().numpy()
            target_title = ('Original N0 teacher / NOT GT' if mode == 'real_n0_heatmaps'
                            else 'Known target / gray ignored')
            cells = [(target_title, _heat(target)), ('Corrupted boundary [0,1]', _heat(source)),
                     ('Restored boundary [0,1]', _heat(output))]
            for _, array in cells:
                array[~valid] = 96
            # 只定位已知训练配对的损伤，不定位测试答案。
            (x,y,x1,y1), label = zoom_box(target, valid, gap, noise)
            cells += [(label+' target zoom', _heat(target[y:y1, x:x1])),
                      (label+' input zoom', _heat(source[y:y1, x:x1])),
                      (label+' output zoom', _heat(output[y:y1, x:x1]))]
            identity = bool(sample['identity'])
            footer = ('Target is the original N0 response, NOT ground truth. Unknown support ignored.'
                      if mode == 'real_n0_heatmaps'
                      else 'Training-source mechanism probe. No holdout accuracy; target unknown is ignored.')
            panels(folder/f'pair_{int(index):03d}.jpg', f'{sample["source"]} | fixed paired draw {index} | e{epoch:03d}',
                   cells, width, footer)
            rows.append(dict(kind='paired_training_source', source=sample['source'], draw=int(index),
                             identity=identity, data_mode=mode, target_kind=('frozen_N0_teacher_response' if mode == 'real_n0_heatmaps'
                                                                         else 'synthetic_processed_GT_boundary'),
                             meta=sample['meta'],
                             **paired_metrics(source, target, output, valid, gap, noise)))

        manifest_path = settings.get('control_manifest')
        if manifest_path:
            manifest_path = Path(project_path(config, manifest_path))
            receipt = json.loads(manifest_path.read_text(encoding='utf-8'))
            if receipt.get('complete') is not True or _sha(receipt['config']) != receipt.get('config_sha256'):
                raise RuntimeError('Fixed control capture is incomplete or its decoder configuration changed')
            frozen_config = load_config(receipt['config'])
            roi_spec = {}
            if settings.get('roi_spec'):
                roi_spec = json.loads(Path(project_path(config, settings['roi_spec'])).read_text(encoding='utf-8')).get('images', {})
            for row in receipt['files']:
                path = Path(row['path'])
                if _sha(path) != row['sha256']:
                    raise RuntimeError('Fixed N0 cache changed: ' + str(path))
                class_path = Path(row['control_classes'])
                if row.get('control_classes_sha256') and _sha(class_path) != row['control_classes_sha256']:
                    raise RuntimeError('Fixed N0 classes changed: ' + str(class_path))
                old_classes = json.loads(class_path.read_text(encoding='utf-8'))
                with np.load(path, allow_pickle=False) as arrays:
                    values = {k: arrays[k].copy() for k in arrays.files}
                boundary = values['boundary']
                restored, audit = predict_residual_tiled(boundary, model, device=device,
                    tile_size=int(cfg.get('crop_size', 512)), overlap=int(settings.get('tile_overlap', 128)), strength=1.)
                ids, classes, readout = evaluate_frozen_readout(values['semantic'], values['raw_probability'], restored, frozen_config)
                old_ids = values['instances']
                rgb = values['rgb']
                stem = row['source']
                panels(folder/(stem+'_whole.jpg'), f'{stem} | e{epoch:03d} | original N0 readout frozen',
                    [('RGB / no test GT', rgb), ('Original boundary [0,1]', _heat(boundary)),
                     ('Completed boundary [0,1]', _heat(restored)), ('N0 instance outlines', _overlay(rgb, old_ids)),
                     ('Candidate outlines', _overlay(rgb, ids))], width,
                     'Only stitched boundary changes. Instance-count changes are not accuracy or official area scores.')
                roi = roi_spec.get(stem, {}).get('roi')
                if roi:
                    x0,y0,x1,y1 = roi
                    cut = np.s_[y0:y1, x0:x1]
                    panels(folder/(stem+'_local.jpg'), f'{stem} | fixed diagnostic ROI | e{epoch:03d}',
                        [('RGB', rgb[cut]), ('N0 boundary', _heat(boundary[cut])), ('Completed boundary', _heat(restored[cut])),
                         ('N0 outlines', _overlay(rgb[cut], old_ids[cut])), ('Candidate outlines', _overlay(rgb[cut], ids[cut]))], width,
                        'Fixed ROI for observation only; not a target, loss, or epoch-selection signal.')
                if epoch == 0 and (not np.array_equal(boundary, restored) or not np.array_equal(old_ids, ids)
                                   or old_classes != classes):
                    raise RuntimeError('Zero-initialized completion changed final N0 output')
                rows.append(dict(kind='unlabelled_test_monitor', source=stem, no_test_gt=True,
                    output_change_mae=float(np.abs(restored-boundary).mean()),
                    original_instances=int(old_ids.max()), candidate_instances=int(ids.max()),
                    identity_epoch_exact=bool(np.array_equal(old_ids, ids) and old_classes == classes),
                    classes_changed=bool(old_classes != classes), inference=audit, readout=readout))
    result = dict(format='boundary_restore_monitor_v1', data_mode=mode, epoch=int(epoch), rows=rows,
                  interrupted_backup=str(backup) if backup else None,
                  status='mechanism_diagnostics_only; no accuracy or official score; no epoch selection')
    (folder/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result
