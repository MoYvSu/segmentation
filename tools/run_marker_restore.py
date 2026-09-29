# -*- coding: utf-8 -*-
"""冻结网络的完整种子恢复／统一面积过滤推理，复用已有控制结果。"""
import argparse
from copy import deepcopy
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
from tools.analyze_semantic_split import class_rgb, load_prediction, tile, verify_votes, write_png
from tools.probe_marker_anchor import border_small, class_field, partition_changes, point_relation
from tools.semantic_marker_diagnostic import marker_stages
from train_backend_adaptation import prediction_maps, save_prediction, write_json
from train_semantic_d5a import get_restorer, sha
from utils.config import load_config, project_path
from utils.marker_restoration import restore_marker_partitions


def render(path, image, old, old_classes, new, new_classes, crop=None):
    sl = crop or (slice(None), slice(None))
    height = min(460, max(230, 32 + round(440 * image[sl].shape[0] / image[sl].shape[1])))
    panels = [(image, 'Raw image'), (class_rgb(image, old, old_classes), 'Original | gold F / blue P'),
              (class_rgb(image, new, new_classes), 'Restore + uniform area filter')]
    write_png(path, np.concatenate([tile(value[sl], title, width=440, height=height)
                                   for value, title in panels], 1))


@torch.inference_mode()
def run(args):
    started = time.monotonic()
    config = load_config(args.config)
    options = config['inference']['marker_partition_restore']
    if not options['enabled']:
        raise ValueError('Candidate configuration must enable marker restoration')
    out = Path(project_path(config, args.output_dir or config['marker_restore']['output_dir']))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source').mkdir()
    for relative in ['tools/run_marker_restore.py', 'utils/marker_restoration.py', 'utils/post_process.py',
                     'utils/affinity_deployment.py', 'tools/probe_marker_anchor.py', args.config]:
        shutil.copy2(ROOT / relative, out / 'source' / Path(relative).name)
    write_json(out / 'config.json', config)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    entry = config['semantic_split']['models']['lora']
    checkpoint = Path(project_path(config, entry['checkpoint']))
    if sha(checkpoint) != entry['sha256']:
        raise ValueError('Checkpoint differs from the fixed LoRA e20 control')
    model, _ = load_backend(checkpoint, config, device)
    model.eval().requires_grad_(False)
    restorer = get_restorer(config, device)
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    if len(files) != config['semantic_split']['expected_images']:
        raise ValueError('Incomplete test cohort')
    case_data = json.loads(Path(project_path(config, args.channels)).read_text(encoding='utf-8'))
    cases = {}
    for row in case_data['cases']:
        cases.setdefault(row['stem'], []).append(row)
    render_images = {'test_101', 'test_116', 'test_123', 'test_073', 'test_088', 'test_160'}
    render_cases = {'test_101': [56], 'test_116': [29], 'test_135': [30], 'test_084': [44]}
    rows, points = [], []
    vote_delta = 0.0
    for index, path in enumerate(files):
        if args.images and path.stem not in args.images:
            continue
        image, semantic, boundary = prediction_maps(model, path, config, device, restorer,
            config['semantic_adaptation']['restoration']['inference_seed'] + index)
        probability = torch.sigmoid(semantic)[0, 0].numpy()
        old, old_classes, old_audit = load_prediction(Path(project_path(config, entry['deployment'])), path.stem)
        vote_delta = max(vote_delta, verify_votes(old, probability, old_classes, old_audit))
        if not rows:
            disabled = deepcopy(config)
            disabled['inference']['marker_partition_restore']['enabled'] = False
            replay, replay_classes = save_prediction(image, semantic, boundary, disabled,
                                                     out / 'disabled_check', path.stem)
            if not np.array_equal(replay, old) or replay_classes != old_classes:
                raise AssertionError('Disabled feature failed complete original-output replay')
        stages = marker_stages(boundary[0, 0].numpy(), config)
        initial, _ = restore_marker_partitions(stages['marker_labels'], stages['marker_boundary_mask'],
            stages['resolved']['bridge_width'], int(options['min_core_area']))
        expected = np.full((*old.shape, 3), 128, np.uint8)
        expected[probability > float(config['inference'].get('threshold', .5))] = 200
        overlay = expected.copy()
        overlay[stages['barrier_belt'] > 0] = 255
        expected = cv2.addWeighted(expected, .7, overlay, .3, 0)
        watershed = cv2.watershed
        capture = {}
        def observe(ws_image, ws_markers):
            if not np.array_equal(ws_image, expected):
                raise AssertionError('Candidate changed the actual watershed barrier image')
            if not capture:
                if not np.array_equal(ws_markers, initial):
                    raise AssertionError('Deployment seeds differ from the independent restore replay')
                capture['initial_markers'] = ws_markers.copy()
            capture['final_markers'] = ws_markers.copy()
            value = watershed(ws_image, ws_markers)
            if 'initial_regions' not in capture:
                capture['initial_regions'] = value.copy()
            return value
        try:
            cv2.watershed = observe
            new, new_classes = save_prediction(image, semantic, boundary, config, out / 'deployment', path.stem)
        finally:
            cv2.watershed = watershed
        if new.dtype != np.uint16 or int(new.max()) > 65535:
            raise AssertionError('Output contract violated')
        audit = json.loads((out / 'deployment' / f'{path.stem}_marker_restore.json').read_text(encoding='utf-8'))
        new_areas = np.bincount(new.ravel())[1:]
        if np.any((new_areas > 0) & (new_areas < audit['area_threshold_native'])):
            raise AssertionError('Final output still contains a below-threshold instance')
        changes = partition_changes(old, new)
        changes.update(stem=path.stem, split_parent_count=audit['split_parent_count'],
            added_markers_before_filter=audit['added_marker_count'],
            initial_region_count=audit['initial_watershed']['region_count'],
            initial_small_50_199=audit['initial_watershed']['small_50_199'],
            filtered_regions=audit['filtered_region_count'],
            watershed_passes=audit['watershed_passes'], area_threshold_native=audit['area_threshold_native'],
            old_ferrite_count=sum(old_classes.values()), new_ferrite_count=sum(new_classes.values()),
            semantic_pixel_changes=int(np.count_nonzero((old > 0) & (new > 0) &
                (class_field(old, old_classes) != class_field(new, new_classes)))))
        for label, value in [('old', old), ('new', new)]:
            changes[f'{label}_border_small_50_199'] = border_small(value, 200)
        rows.append(changes)
        for case in cases.get(path.stem, []):
            if point_relation(stages['marker_labels'], case['points'])['same'] is not True:
                raise AssertionError('Original selected diagnostic points changed')
            final_markers = capture['final_markers'] if new.any() else np.zeros_like(old)
            points.append(dict(stem=path.stem, id=case['id'],
                reopened_by_thinning=case['stages']['bridged_marker_mask']['same_component'] is False and
                                     case['stages']['skeleton_only']['same_component'] is True,
                restored_before_filter=point_relation(capture['initial_regions'], case['points']),
                filtered_seeds=point_relation(final_markers, case['points']),
                final_regions=point_relation(new, case['points'])))
        if path.stem in render_images:
            render(out / 'figures' / f'{path.stem}.png', image, old, old_classes, new, new_classes)
        boxes = ndimage.find_objects(old)
        for iid in render_cases.get(path.stem, []):
            y, x = boxes[iid-1]
            crop = (slice(max(0, y.start-24), min(old.shape[0], y.stop+24)),
                    slice(max(0, x.start-24), min(old.shape[1], x.stop+24)))
            render(out / 'figures' / f'{path.stem}_{iid}.png', image, old, old_classes, new, new_classes, crop)
        print(json.dumps(changes), flush=True)
    if not rows:
        raise ValueError('No selected images found')
    summed = [key for key, value in rows[0].items()
              if isinstance(value, int) and key not in ('watershed_passes', 'area_threshold_native')]
    totals = {key: sum(row[key] for row in rows) for key in summed}
    totals.update(split_old_instances=sum(len(r['split_old_ids']) for r in rows),
        split_border_old_instances=sum(len(r['split_border_old_ids']) for r in rows),
        merged_new_instances=sum(len(r['merged_new_ids']) for r in rows),
        maximum_watershed_passes=max(r['watershed_passes'] for r in rows))
    groups = {}
    for name, selected in [('all_mixed', points), ('reopened', [p for p in points if p['reopened_by_thinning']])]:
        groups[name] = dict(count=len(selected), **{
            key: {str(state): sum(p[key]['same'] is state for p in selected) for state in (True, False, None)}
            for key in ['restored_before_filter', 'filtered_seeds', 'final_regions']})
    report = dict(complete=True, images=len(rows), variant='restore_global_area_v1', checkpoint=entry,
        area_filter=options, area_filter_scope='all_instances', totals=totals, groups=groups,
        area_threshold_native=dict(minimum=min(r['area_threshold_native'] for r in rows),
            median=float(np.median([r['area_threshold_native'] for r in rows])),
            maximum=max(r['area_threshold_native'] for r in rows)),
        vote_max_absolute_delta=vote_delta, first_disabled_full_output_equal=True,
        actual_watershed_image_and_restored_markers_verified=True,
        no_training=True, no_test_labels=True, official_score=None,
        elapsed_seconds=time.monotonic()-started, images_detail=rows, points=points)
    write_json(out / 'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['images_detail', 'points']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiments/marker_restore.yaml')
    parser.add_argument('--output-dir')
    parser.add_argument('--channels', default='outputs/semantic_channels/report.json')
    parser.add_argument('--images', nargs='+')
    run(parser.parse_args())
