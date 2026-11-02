# -*- coding: utf-8 -*-
"""DINO末端语义过程图：固定旧实例，只重新投票，不读测试GT。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from utils.affinity_deployment import crop_letterbox_output
from utils.config import project_path
from utils.image_io import read_image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def class_map(instances, classes):
    """仅按明确类别映射；背景保持未知，不把未标注当珠光体。"""
    ids = np.asarray(instances)
    if ids.ndim != 2 or ids.dtype.kind not in 'iu' or np.any(ids < 0):
        raise ValueError('Nonnegative integer instance map required')
    maximum = max(int(ids.max(initial=0)), max((int(key) for key in classes), default=0))
    lookup = np.full(maximum + 1, -1, np.int8)
    for key, value in classes.items():
        if type(value) is not int or value not in (0, 1):
            raise ValueError('Classes must be integer P=0/F=1')
        if int(key) <= 0:
            raise ValueError('Instance classes require positive IDs')
        lookup[int(key)] = value
    return lookup[ids]


def vote_fixed_instances(instances, probability):
    """复用既有概率均值投票；实例像素和编号保持不变。"""
    from tools.semantic_crossover import revote_instances
    before = np.asarray(instances).copy()
    result, scores = revote_instances(instances, np.asarray(probability, np.float32))
    if not np.array_equal(before, instances):
        raise RuntimeError('Fixed instance pixels changed while revoting')
    return result, scores


def _font(size=16):
    for name in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def _edges(ids):
    result = np.zeros(ids.shape, bool)
    result[1:] |= ids[1:] != ids[:-1]
    result[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    return result


def _overlay(rgb, semantic, width, instances=None):
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    size = (width, height)
    image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    target = cv2.resize(np.asarray(semantic, np.float32), size, interpolation=cv2.INTER_NEAREST)
    colors = np.full_like(image, 100.)
    colors[target == 0] = (52, 124, 204)
    colors[target == 1] = (244, 205, 57)
    image = .55 * image + .45 * colors
    if instances is not None:
        ids = cv2.resize(np.asarray(instances, np.int32), size, interpolation=cv2.INTER_NEAREST)
        image[_edges(ids)] = (15, 15, 15)
    return np.rint(image).clip(0, 255).astype(np.uint8)


def _probability_panel(rgb, probability, width):
    size = (width, max(1, round(rgb.shape[0] * width / rgb.shape[1])))
    p = cv2.resize(probability, size, interpolation=cv2.INTER_LINEAR)
    # 三组/各epoch固定0..1色阶，不能逐图自动拉伸。
    value = np.rint(np.clip(p, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(value, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def _save_panels(path, stem, epoch, panels, footer):
    panel_h = max(image.shape[0] for _, image in panels)
    width = sum(image.shape[1] for _, image in panels)
    canvas = Image.new('RGB', (width, panel_h + 100), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f'{stem} | e{epoch:03d} | frozen geometry', font=_font(18), fill='black')
    x = 0
    for title, image in panels:
        draw.text((x + 6, 34), title, font=_font(15), fill='black')
        canvas.paste(Image.fromarray(image), (x, 60))
        x += image.shape[1]
    draw.text((8, panel_h + 69), footer, font=_font(13), fill='black')
    canvas.save(path, quality=90)


def prepare_monitor_folder(outdir, epoch):
    """中断后保留上一份不完整过程图，再从同一权重重绘，避免续训被旧目录卡住。"""
    root = Path(outdir).resolve() / 'monitor'
    if root.is_symlink():
        raise ValueError('Monitor root cannot be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f'e{epoch:03d}'
    backup = None
    if folder.exists():
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError('Existing monitor must be an ordinary directory')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = root / (folder.name + '.interrupted_' + stamp)
        if folder.resolve().parent != root.resolve() or backup.resolve().parent != root.resolve():
            raise ValueError('Monitor backup escaped the intended monitor directory')
        folder.rename(backup)
    folder.mkdir()
    return folder, backup


def build_monitor(config):
    """构建一次固定输入快照；返回无训练、无选epoch行为的回调。"""
    from data.dino_semantic import ProcessedSemanticDataset, prepare_semantic_view

    cfg = config['semantic_dino']
    settings = cfg.get('monitor', {})
    width = int(settings.get('thumbnail_width', 320))
    dataset = ProcessedSemanticDataset(config)
    train_names = tuple(settings.get('train_sources', ('train_001', 'train_173', 'train_558', 'train_889', 'train_175', 'train_873')))
    if not set(train_names) <= set(dataset.sources):
        raise ValueError('Monitor requires available processed-GT train sources')
    train = {name: dataset.load_source(name) for name in train_names}
    tests = {}
    test_names = tuple(settings.get('test_sources', ()))
    if test_names:
        test_dir = Path(project_path(config, settings['test_dir']))
        control_dir = Path(project_path(config, settings['control_dir']))
        for name in test_names:
            matches = [path for path in test_dir.iterdir() if path.stem == name and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.bmp')]
            if len(matches) != 1:
                raise ValueError('Exactly one fixed test source required: ' + name)
            path = matches[0]
            rgb = cv2.cvtColor(read_image(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            png = control_dir / (name + '_inst.png')
            labels = control_dir / (name + '_class.json')
            ids = read_image(png, cv2.IMREAD_UNCHANGED)
            classes = json.loads(labels.read_text(encoding='utf-8'))
            if (ids.dtype != np.uint16 or ids.ndim != 2 or ids.shape != rgb.shape[:2]
                    or set(classes) != {str(int(i)) for i in np.unique(ids) if i}):
                raise ValueError('Fixed N0 control contract differs: ' + name)
            tests[name] = dict(rgb=rgb, ids=ids, classes=classes, png=png, labels=labels,
                               png_sha256=sha(png), classes_sha256=sha(labels), source_sha256=sha(path))

    @torch.no_grad()
    def probability(model, rgb):
        # 测试图只构建空ignore目标以复用同一等比预处理；不读取任何测试标注。
        blank = np.zeros(rgb.shape[:2], np.int32)
        view = prepare_semantic_view(rgb, blank, {}, input_size=int(cfg.get('input_size', 1024)))
        image = view['image'].unsqueeze(0).to(next(model.parameters()).device)
        logits = model(image).float()
        h, w = rgb.shape[:2]
        scale = image.shape[-1] / max(h, w)
        rh, rw = round(h * scale), round(w * scale)
        native = crop_letterbox_output(logits, image.shape[-1], image.shape[-1]-rh,
                                       image.shape[-1]-rw, (h, w))
        return native.sigmoid()[0, 0].cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def callback(model, epoch, outdir):
        if model.training or getattr(model, 'backbone', model).training:
            raise RuntimeError('Monitor requires eval mode')
        folder, interrupted = prepare_monitor_folder(outdir, epoch)
        rows = []
        for name, (rgb, ids, classes, _) in train.items():
            target = class_map(ids, classes)
            p = probability(model, rgb)
            pred = (p > .5).astype(np.int8)
            size = (width, max(1, round(rgb.shape[0] * width / rgb.shape[1])))
            panels = [('RGB / training source', cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)),
                      ('Processed GT / gray unknown', _overlay(rgb, target, width)),
                      ('DINO F probability [0,1]', _probability_panel(rgb, p, width)),
                      ('DINO pixel class', _overlay(rgb, pred, width))]
            _save_panels(folder/(name+'.jpg'), name, epoch, panels,
                         'F yellow, P blue. Training images, not independent validation; unknown GT gray.')
            rows.append(dict(source=name, kind='train', participated_in_training=True,
                             known_pixels=int((target >= 0).sum()), gt_used_only_for_display=True))
        for name, record in tests.items():
            if sha(record['png']) != record['png_sha256'] or sha(record['labels']) != record['classes_sha256']:
                raise RuntimeError('Original N0 control changed during training')
            rgb, ids, classes = record['rgb'], record['ids'], record['classes']
            p = probability(model, rgb)
            new, scores = vote_fixed_instances(ids, p)
            size = (width, max(1, round(rgb.shape[0] * width / rgb.shape[1])))
            panels = [('RGB / unlabelled test', cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)),
                      ('Reused N0 / fixed instances', _overlay(rgb, class_map(ids, classes), width, ids)),
                      ('DINO F probability [0,1]', _probability_panel(rgb, p, width)),
                      ('DINO / same instance pixels', _overlay(rgb, class_map(ids, new), width, ids))]
            _save_panels(folder/(name+'.jpg'), name, epoch, panels,
                         'Same N0 geometry, only final class vote changes. Test RGB is monitor only; no GT or accuracy claim.')
            flips = [key for key in classes if new[key] != classes[key]]
            (folder/(name+'_classes.json')).write_text(json.dumps(new, sort_keys=True), encoding='utf-8')
            rows.append(dict(source=name, kind='test', participated_in_training=False, test_gt_used=False,
                             fixed_png_sha256=record['png_sha256'], control_class_sha256=record['classes_sha256'],
                             source_sha256=record['source_sha256'], instance_pixels_unchanged=True,
                             instances=len(classes), changed_instance_count=len(flips), scores=scores))
        report = dict(epoch=int(epoch), display_only=True, test_gt_used=False, accuracy_evaluated=False,
                      changes_training_rng=False, rows=rows,
                      previous_monitor_backup=interrupted.name if interrupted is not None else None)
        (folder/'monitor.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return callback
