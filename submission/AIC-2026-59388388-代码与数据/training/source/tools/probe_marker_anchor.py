# -*- coding: utf-8 -*-
"""固定全部模型，只在实际分水岭入口替换前置封边种子；输出仅用于诊断。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from models.backend_adaptation import load_backend
from tools.analyze_semantic_split import (class_rgb, contours, load_prediction,
    prob_rgb, tile, verify_votes, write_png)
from tools.semantic_marker_diagnostic import anchored_marker_stages, marker_stages
from train_backend_adaptation import prediction_maps, save_prediction, write_json
from train_semantic_d5a import get_restorer, sha
from utils.config import load_config, project_path


def partition_changes(old, new):
    """按重叠面积关联，避免新编号排序导致假变化；5%/50px只作描述。"""
    n = int(new.max()) + 1
    overlap = np.bincount((old.astype(np.int64) * n + new).ravel(),
                          minlength=(int(old.max()) + 1) * n).reshape(-1, n)
    old_area, new_area = overlap.sum(1), overlap.sum(0)
    split_links = (overlap >= 50) & (overlap >= .05 * old_area[:, None])
    merge_links = (overlap >= 50) & (overlap >= .05 * new_area[None, :])
    split_links[0, :] = split_links[:, 0] = False
    merge_links[0, :] = merge_links[:, 0] = False
    split_ids = np.flatnonzero(split_links.sum(1) >= 2).tolist()
    merge_ids = np.flatnonzero(merge_links.sum(0) >= 2).tolist()
    objects = ndimage.find_objects(old)
    border_ids = []
    for iid in split_ids:
        y, x = objects[iid - 1]
        if min(y.start, x.start, old.shape[0] - y.stop, old.shape[1] - x.stop) <= 3:
            border_ids.append(iid)
    return dict(old_count=int(np.count_nonzero(old_area[1:])),
        new_count=int(np.count_nonzero(new_area[1:])),
        split_old_ids=split_ids, split_border_old_ids=border_ids,
        merged_new_ids=merge_ids,
        old_small_50_199=int(np.count_nonzero((old_area[1:] >= 50) & (old_area[1:] < 200))),
        new_small_50_199=int(np.count_nonzero((new_area[1:] >= 50) & (new_area[1:] < 200))),
        foreground_lost=int(np.count_nonzero((old > 0) & (new == 0))),
        foreground_added=int(np.count_nonzero((old == 0) & (new > 0))),
        contour_changed_pixels=int(np.count_nonzero(contours(old) != contours(new))))


def point_relation(labels, points):
    ids = [int(labels[p['y'], p['x']]) if p is not None else 0
           for p in (points['ferrite'], points['pearlite'])]
    return dict(ids=ids, same=bool(ids[0] == ids[1]) if min(ids) > 0 else None)


def class_field(labels, classes):
    lut = np.full(int(labels.max()) + 1, -1, np.int8)
    for iid, value in classes.items():
        lut[iid] = value
    return lut[labels]


def small_cores(stages):
    areas = np.bincount(stages['core_labels'].ravel())[1:]
    return int(np.count_nonzero((areas > 0) & (areas < 50)))


def border_small(labels, upper):
    areas = np.bincount(labels.ravel())
    border = np.unique(np.concatenate([labels[:4].ravel(), labels[-4:].ravel(),
                                       labels[:, :4].ravel(), labels[:, -4:].ravel()]))
    border = border[border > 0]
    return int(np.count_nonzero((areas[border] >= 50) & (areas[border] < upper)))


def render(path, image, probability, old, old_classes, new, new_classes, base, candidate, crop=None):
    sl = crop or (slice(None), slice(None))
    def binary(array):
        return np.repeat(((array > 0) * 255).astype(np.uint8)[..., None], 3, axis=2)
    panels = [(image, 'Raw image'), (class_rgb(image, old, old_classes), 'Original final: gold F / blue P'),
              (class_rgb(image, new, new_classes), 'Preseal final: gold F / blue P'),
              (prob_rgb(probability), 'Unchanged semantic P(F)'),
              (binary(base['marker_belt']), 'Original seed barrier'),
              (binary(candidate['marker_belt']), 'Preseal seed barrier')]
    height = min(420, max(200, 32 + round(400 * image[sl].shape[0] / image[sl].shape[1])))
    cells = [tile(rgb[sl], title, width=400, height=height) for rgb, title in panels]
    write_png(path, np.concatenate([np.concatenate(cells[:3], 1), np.concatenate(cells[3:], 1)], 0))


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    config = load_config(args.config)
    out = Path(project_path(config, args.output_dir))
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    for relative in ['tools/probe_marker_anchor.py', 'tools/semantic_marker_diagnostic.py',
                     'tools/analyze_semantic_split.py', 'utils/post_process.py', args.config]:
        shutil.copy2(ROOT / relative, source / Path(relative).name)
    write_json(out / 'config.json', config)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    entry = config['semantic_split']['models']['lora']
    checkpoint = Path(project_path(config, entry['checkpoint']))
    if sha(checkpoint) != entry['sha256']:
        raise ValueError('Checkpoint lineage mismatch')
    model, _ = load_backend(checkpoint, config, device)
    model.eval().requires_grad_(False)
    restorer = get_restorer(config, device)
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    channel_report = json.loads(Path(project_path(config, args.channels)).read_text(encoding='utf-8'))
    cases = {}
    for case in channel_report['cases']:
        cases.setdefault(case['stem'], []).append(case)
    render_images = {'test_101', 'test_116', 'test_123', 'test_073', 'test_088', 'test_160'}
    render_cases = {'test_101': [56], 'test_116': [29], 'test_135': [30], 'test_084': [44]}
    rows, point_rows = [], []
    vote_delta = 0.0
    backend = 'opencv_ximgproc' if hasattr(getattr(cv2, 'ximgproc', None), 'thinning') else 'skimage'
    for index, path in enumerate(files):
        if args.images and path.stem not in args.images:
            continue
        image, semantic, boundary = prediction_maps(model, path, config, device, restorer,
            config['semantic_adaptation']['restoration']['inference_seed'] + index)
        probability = torch.sigmoid(semantic)[0, 0].numpy()
        old, old_classes, audit = load_prediction(Path(project_path(config, entry['deployment'])), path.stem)
        vote_delta = max(vote_delta, verify_votes(old, probability, old_classes, audit))
        base = marker_stages(boundary[0, 0].numpy(), config)
        candidate = anchored_marker_stages(base)
        if not base['marker_count'] or not candidate['marker_count']:
            raise ValueError('Zero seeds not supported by this isolated replacement')
        expected = np.full((*old.shape, 3), 128, np.uint8)
        expected[probability > float(config['inference'].get('threshold', .5))] = 200
        overlay = expected.copy()
        overlay[base['barrier_belt'] > 0] = 255
        expected = cv2.addWeighted(expected, .7, overlay, .3, 0)
        original_watershed = cv2.watershed
        captured = []
        def replace_markers(ws_image, ws_markers):
            if not np.array_equal(ws_markers, base['marker_labels']):
                raise AssertionError('Diagnostic seeds differ from actual deployed seeds')
            if not np.array_equal(ws_image, expected):
                raise AssertionError('Actual watershed image differs from frozen barrier reconstruction')
            captured.append(True)
            return original_watershed(ws_image, candidate['marker_labels'].copy())
        try:
            cv2.watershed = replace_markers
            new, new_classes = save_prediction(image, semantic, boundary, config, out / 'deployment', path.stem)
        finally:
            cv2.watershed = original_watershed
        if len(captured) != 1 or new.dtype != np.uint16 or int(new.max()) > 65535:
            raise AssertionError('Actual postprocessing contract violated')
        changes = partition_changes(old, new)
        changes.update(stem=path.stem, old_marker_count=base['marker_count'],
            new_marker_count=candidate['marker_count'],
            old_filtered_small_cores=small_cores(base), new_filtered_small_cores=small_cores(candidate),
            old_ferrite_count=sum(old_classes.values()), new_ferrite_count=sum(new_classes.values()),
            semantic_pixel_changes=int(np.count_nonzero((old > 0) & (new > 0) &
                (class_field(old, old_classes) != class_field(new, new_classes)))))
        for label, value in [('old', old), ('new', new)]:
            for upper in [200, 500]:
                changes[f'{label}_border_small_50_{upper-1}'] = border_small(value, upper)
        rows.append(changes)
        for case in cases.get(path.stem, []):
            before = point_relation(base['marker_labels'], case['points'])
            if before['same'] is not True:
                raise AssertionError('Previously selected same-seed points do not reproduce')
            reopened = (case['stages']['bridged_marker_mask']['same_component'] is False and
                        case['stages']['skeleton_only']['same_component'] is True)
            point_rows.append(dict(stem=path.stem, id=case['id'], reopened_by_thinning=reopened,
                original=before, preseal_marker=point_relation(candidate['marker_labels'], case['points']),
                preseal_final=point_relation(new, case['points'])))
        if path.stem in render_images:
            render(out / 'figures' / f'{path.stem}.png', image, probability, old, old_classes,
                   new, new_classes, base, candidate)
        objects = ndimage.find_objects(old)
        for iid in render_cases.get(path.stem, []):
            y, x = objects[iid - 1]
            sl = (slice(max(0, y.start-24), min(old.shape[0], y.stop+24)),
                  slice(max(0, x.start-24), min(old.shape[1], x.stop+24)))
            render(out / 'figures' / f'{path.stem}_{iid}.png', image, probability, old, old_classes,
                   new, new_classes, base, candidate, sl)
        print(json.dumps(changes), flush=True)
    scalar_keys = [k for k, v in rows[0].items() if isinstance(v, int)]
    totals = {key: sum(row[key] for row in rows) for key in scalar_keys}
    totals.update(split_old_instances=sum(len(r['split_old_ids']) for r in rows),
        split_border_old_instances=sum(len(r['split_border_old_ids']) for r in rows),
        merged_new_instances=sum(len(r['merged_new_ids']) for r in rows))
    group_metrics = {}
    for name, selected in [('all_mixed', point_rows),
                           ('reopened', [r for r in point_rows if r['reopened_by_thinning']]),
                           ('other_mixed', [r for r in point_rows if not r['reopened_by_thinning']])]:
        group_metrics[name] = dict(count=len(selected), **{
            field: {str(state): sum(r[field]['same'] is state for r in selected) for state in (True, False, None)}
            for field in ['preseal_marker', 'preseal_final']})
    report = dict(complete=True, images=len(rows), checkpoint=entry, variant='marker_preseal_v1',
        no_training=True, no_test_labels=True, official_score=None, totals=totals, groups=group_metrics,
        vote_max_absolute_delta=vote_delta, actual_markers_and_watershed_image_verified=True,
        thinning_backend=backend, opencv_version=cv2.__version__, elapsed_seconds=time.monotonic()-started,
        images_detail=rows, points=point_rows,
        overlap_rule='at least 50px and 5% of old area for split / new area for merge; diagnostic only')
    write_json(out / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['images_detail', 'points']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiments/semantic_split.yaml')
    parser.add_argument('--output-dir', default='outputs/marker_anchor')
    parser.add_argument('--channels', default='outputs/semantic_channels/report.json')
    parser.add_argument('--images', nargs='+')
    run(parser.parse_args())
