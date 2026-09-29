# -*- coding: utf-8 -*-
"""最高分control的纯种子恢复消融；不训练、不用GT决定恢复位置、不加面积离群过滤。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from itertools import combinations
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from scipy.ndimage import find_objects
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.affinity_connectivity import undo_spatial
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.mim_dataset import list_images
from models.backend_adaptation import load_backend, restore_first
from tools.affinity_connectivity_views import decode_native
from tools.analyze_affinity_connectivity import description, load_directory
from tools.probe_affinity_chain import relations_at
from tools.probe_marker_anchor import partition_changes
from tools.render_backend_ablation import overlay, text_line, write_png
from tools.semantic_crossover import revote_instances
from tools.semantic_marker_diagnostic import marker_stages
from train_affinity_connectivity import get_restorer
from train_backend_adaptation import fusion_options, prediction_maps, sha, tensor_digest
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.affinity_fusion import affinity_boundary_probability
from utils.config import load_config, project_path
from utils.marker_restoration import restore_marker_partitions
from utils.offset_letterbox import geometry_letterbox_metadata


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def decode_pair(semantic, boundary, raw_probability, config):
    """在同一预测图上解码两组，截获真实分水岭，核对只改变种子。"""
    control = deepcopy(config)
    control['inference']['marker_partition_restore']['enabled'] = False
    stages = marker_stages(boundary[0, 0].numpy(), control)
    native_watershed = cv2.watershed
    captured = {}

    def capture_base(image, markers):
        if not np.array_equal(markers, stages['marker_labels']):
            raise AssertionError('Actual baseline markers differ from independent replay')
        captured['image'] = image.copy()
        return native_watershed(image, markers)

    with patch('utils.post_process.cv2.watershed', side_effect=capture_base):
        before, _ = decode_native(semantic, boundary, control)
    if 'image' not in captured:
        raise AssertionError('Expected connected-core baseline seeds')
    seeds, expected_audit = restore_marker_partitions(stages['marker_labels'], stages['marker_boundary_mask'],
        stages['resolved']['bridge_width'], int(config['inference']['marker_partition_restore']['min_core_area']))
    calls = []

    def capture_candidate(image, markers):
        if not np.array_equal(image, captured['image']):
            raise AssertionError('Candidate changed the actual watershed image/barrier')
        if not np.array_equal(markers, seeds):
            raise AssertionError('Candidate markers differ from isolated restoration')
        calls.append(True)
        return native_watershed(image, markers)

    audit = {}
    with patch('utils.post_process.cv2.watershed', side_effect=capture_candidate):
        after, _ = decode_native(semantic, boundary, config, marker_audit=audit)
    if len(calls) != 1 or audit['watershed_passes'] != 1 or audit['area_filter_enabled']:
        raise AssertionError('Expected one watershed pass with no extra area filtering')
    if audit['splits'] != expected_audit['splits']:
        raise AssertionError('Restoration audit mismatch')
    bc, _ = revote_instances(before, raw_probability)
    ac, _ = revote_instances(after, raw_probability)
    for ids, classes in [(before, bc), (after, ac)]:
        if ids.dtype != np.uint16 or int(ids.max()) > 65535:
            raise AssertionError('Instance output type/cap violated')
        if {str(int(i)) for i in np.unique(ids) if i} != set(classes) or not set(classes.values()).issubset({0, 1}):
            raise AssertionError('Invalid class mapping')
    return before, after, bc, ac, audit, stages['marker_labels'], seeds


def _point(mask, box):
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return [int(y + box[0].start), int(x + box[1].start)]


def trusted_changes(gt, trusted, before, after):
    """GT只用于事后诊断；同时寻找新增误合并和显著误切，未知像素不参与。"""
    records, anchors, new_split_pairs = [], [], []
    for iid, box in enumerate(find_objects(gt.astype(np.int32)), 1):
        if box is None:
            continue
        known = (gt[box] == iid) & trusted[box]
        area = int(known.sum())
        if area < 64:
            continue
        significant = {}
        for name, ids in [('base', before), ('candidate', after)]:
            values, counts = np.unique(ids[box][known], return_counts=True)
            pieces = sorted([(int(n), int(i)) for i, n in zip(values, counts)
                             if i > 0 and n >= max(50, .10 * area)], reverse=True)
            significant[name] = [(n, pid, _point(known & (ids[box] == pid), box)) for n, pid in pieces]
        old, new = significant['base'], significant['candidate']
        if old:
            anchors.append({'gt': iid, 'point': old[0][2]})
        records.append({'gt': iid, 'known_area': area, 'base_fragments': len(old), 'candidate_fragments': len(new),
            'base_uncovered_known_pixels': int((known & (before[box] == 0)).sum()),
            'candidate_uncovered_known_pixels': int((known & (after[box] == 0)).sum())})
        for a, b in combinations(new, 2):
            p, q = a[2], b[2]
            left, right = int(before[tuple(p)]), int(before[tuple(q)])
            if left > 0 and left == right:
                new_split_pairs.append({'gt': iid, 'points': [p, q], 'base_id': left,
                                        'candidate_ids': [a[1], b[1]]})
    merges = Counter()
    new_merge_pairs = []
    for a, b in combinations(anchors, 2):
        pa, pb = tuple(a['point']), tuple(b['point'])
        oa, ob = int(before[pa]), int(before[pb])
        na, nb = int(after[pa]), int(after[pb])
        was_same = oa > 0 and oa == ob
        is_same = na > 0 and na == nb
        if was_same:
            merges['base_pairs'] += 1
            if min(na, nb) <= 0:
                merges['blocked_after'] += 1
            elif not is_same:
                merges['fixed_pairs'] += 1
        elif is_same:
            merges['new_pairs'] += 1
            new_merge_pairs.append({'gt_ids': [a['gt'], b['gt']], 'points': [a['point'], b['point']]})
    summary = {'known_gt_regions': len(records), 'fixed_anchor_merges': dict(merges),
        'gt_regions_more_fragments': sum(r['candidate_fragments'] > r['base_fragments'] for r in records),
        'gt_regions_fewer_fragments': sum(r['candidate_fragments'] < r['base_fragments'] for r in records),
        'new_split_pairs': len(new_split_pairs), 'gt_regions_with_new_split': len({r['gt'] for r in new_split_pairs}),
        'base_uncovered_known_pixels': sum(r['base_uncovered_known_pixels'] for r in records),
        'candidate_uncovered_known_pixels': sum(r['candidate_uncovered_known_pixels'] for r in records)}
    return {'summary': summary, 'regions': records, 'new_split_pairs': new_split_pairs, 'new_merge_pairs': new_merge_pairs}


def old_cases(reference, before, after, seeds):
    relations = reference['relations']
    old = relations_at(before, relations)
    if old != [r['stages']['base']['final']['full'] for r in relations]:
        raise AssertionError('Previous fixed-anchor baseline is not reproduced')
    new, marker = relations_at(after, relations), relations_at(seeds, relations)
    rows = []
    for relation, old_state, new_state, seed_state in zip(relations, old, new, marker):
        desired = 'different' if relation['kind'] == 'merge' else 'same'
        reopened = (relation['kind'] == 'merge' and relation['stages']['base']['bridge']['full'] == 'different'
                    and relation['stages']['base']['skeleton']['full'] == 'same')
        rows.append({'kind': relation['kind'], 'points': relation['points'], 'gt_ids': relation['gt_ids'],
            'base': old_state, 'candidate': new_state, 'candidate_marker': seed_state,
            'fixed': new_state == desired, 'skeleton_reopened': reopened,
            'trusted_route': reopened and relation['stages']['base']['sealed_core']['trusted_only'] == 'same'})
    return rows


def render(path, rgb, before, after, bc, ac, title, crop=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    region = crop or np.s_[:, :]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bodies = [bgr, overlay(bgr, before, {int(k): v for k, v in bc.items()}),
              overlay(bgr, after, {int(k): v for k, v in ac.items()})]
    panels = []
    for label, body in zip([title, f'Control e20: N={len(bc)}', f'Restore only: N={len(ac)}'], bodies):
        small = body[region]
        small = cv2.resize(small, (430, max(1, round(430 * small.shape[0] / small.shape[1]))), interpolation=cv2.INTER_AREA)
        small = cv2.copyMakeBorder(small, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        text_line(small, label, (7, 23), 430, size=.49)
        panels.append(small)
    write_png(path, np.concatenate(panels, 1))


@torch.no_grad()
def native_training_maps(model, restorer, tensor, shape, config, seed):
    metadata = geometry_letterbox_metadata(shape, 1024, 512)
    ph, pw = 1024 - metadata.resized_height, 1024 - metadata.resized_width
    native = lambda value: crop_letterbox_output(value, 1024, ph, pw, shape).cpu()
    image = tensor[None].cuda().float()
    raw = native(model.semantic_decoder(model.encoder(image), image))
    output = model(restore_first(restorer, image, seed))
    mode, kwargs = fusion_options(config)
    boundary = native(affinity_boundary_probability(output['affinity_logits'].float(), mode=mode, **kwargs))
    semantic = native(output['semantic_logits'].float())
    rgb = np.rint(native(tensor[None])[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return rgb, semantic, boundary, torch.sigmoid(raw)[0, 0].numpy()


def summarize(rows):
    result = {}
    for view in sorted({r['view'] for r in rows}):
        group = [r for r in rows if r['view'] == view]
        cases = [c for r in group for c in r.get('old_cases', [])]
        result[view] = {'inputs': len(group),
            'base': {key: sum(r['base'][key] for r in group) for key in ['instances', 'ferrite', 'tiny_50_199', 'covered_pixels']},
            'candidate': {key: sum(r['candidate'][key] for r in group) for key in ['instances', 'ferrite', 'tiny_50_199', 'covered_pixels']},
            'restoration': {key: sum(r['audit'][key] for r in group) for key in ['split_parent_count', 'added_marker_count', 'released_marker_pixels', 'filtered_region_count']},
            'changes': {key: sum(r['changes'][key] for r in group) for key in ['foreground_lost', 'foreground_added', 'contour_changed_pixels']},
            'split_old_instances': sum(len(r['changes']['split_old_ids']) for r in group),
            'split_border_instances': sum(len(r['changes']['split_border_old_ids']) for r in group),
            'merged_new_instances': sum(len(r['changes']['merged_new_ids']) for r in group)}
        area = [r['candidate']['ferrite_mean_area'] / r['base']['ferrite_mean_area'] - 1 for r in group
                if r['base']['ferrite_mean_area'] and r['candidate']['ferrite_mean_area']]
        result[view]['ferrite_mean_area_relative_quantiles'] = dict(zip(['p10', 'median', 'p90'], map(float, np.quantile(area, [.1, .5, .9])))) if area else {}
        if cases:
            result[view]['previous_errors'] = {kind: {'cases': sum(c['kind'] == kind for c in cases),
                'fixed': sum(c['kind'] == kind and c['fixed'] for c in cases),
                'blocked': sum(c['kind'] == kind and c['candidate'] == 'blocked' for c in cases)} for kind in ['merge', 'split']}
            for key in ['skeleton_reopened', 'trusted_route']:
                selected = [c for c in cases if c[key]]
                result[view][key] = {'cases': len(selected), 'fixed': sum(c['fixed'] for c in selected)}
            result[view]['trusted_changes'] = {key: sum(r['gt']['summary'][key] for r in group) for key in
                ['known_gt_regions', 'gt_regions_more_fragments', 'gt_regions_fewer_fragments', 'new_split_pairs',
                 'gt_regions_with_new_split', 'base_uncovered_known_pixels', 'candidate_uncovered_known_pixels']}
            result[view]['trusted_changes']['new_merge_pairs'] = sum(r['gt']['summary']['fixed_anchor_merges'].get('new_pairs', 0) for r in group)
    return result


def run(args):
    started = time.monotonic()
    config = load_config(args.config)
    options = config['affinity_preserve']
    restore = config['inference']['marker_partition_restore']
    if not restore['enabled'] or restore['area_filter_enabled']:
        raise ValueError('This experiment requires restoration only')
    out = Path(project_path(config, args.output or (options['output_dir'] + '/' + args.mode)))
    out.mkdir(parents=True, exist_ok=False)
    sources = ['tools/run_affinity_preserve.py', 'tools/affinity_connectivity_views.py', 'utils/marker_restoration.py',
               'utils/post_process.py', args.config]
    for relative in sources:
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    checkpoint = project_path(config, options['checkpoint'])
    if sha(checkpoint) != options['checkpoint_sha256']:
        raise ValueError('Winning checkpoint hash mismatch')
    device = torch.device('cuda')
    model, bundle = load_backend(checkpoint, config, device)
    model.eval().requires_grad_(False)
    restorer = get_restorer(config, device)
    initial, ri = tensor_digest(model.state_dict().items()), tensor_digest(restorer.state_dict().items())
    cfg = config['backend_adaptation']
    dump(out / 'provenance.json', {'config': config, 'checkpoint_epoch': bundle['epoch'],
        'checkpoint_sha256': sha(checkpoint), 'source_sha256': {p: sha(ROOT / p) for p in sources},
        'no_training': True, 'no_test_labels': True, 'only_change': 'restore_marker_partitions with extra area filter disabled'})
    rows = []

    def process(stem, view, rgb, semantic, boundary, raw_probability, reference=None, gt=None, trusted=None, control_path=None):
        before, after, bc, ac, audit, _, seeds = decode_pair(semantic, boundary, raw_probability, config)
        if control_path:
            expected, classes = load_directory(control_path, stem)
            if not np.array_equal(before, expected) or bc != classes:
                raise AssertionError('Existing winning submission is not reproduced exactly: ' + stem)
        row = {'source': stem, 'view': view, 'base': description(before, bc), 'candidate': description(after, ac),
            'changes': partition_changes(before, after), 'audit': audit,
            'actual_watershed_image_equal': True, 'actual_markers_equal_isolated_restore': True}
        if reference is not None:
            if len(bc) != reference['variants']['base']['instances']:
                raise AssertionError('Previous training-view baseline count differs')
            row['old_cases'] = old_cases(reference, before, after, seeds)
            row['gt'] = trusted_changes(gt, trusted, before, after)
        else:
            row['existing_submission_exact'] = True
        if args.mode == 'test':
            directory = out / 'deployment'
            directory.mkdir(exist_ok=True)
            write_png(directory / (stem + '_inst.png'), after)
            dump(directory / (stem + '_class.json'), ac)
        if args.mode == 'train' or stem in render_names:
            render(out / 'figures' / (stem + '_' + view + '.png'), rgb, before, after, bc, ac, stem + ' ' + view)
        if args.mode == 'train' and stem in detail_crops and view == 'clean':
            y0, y1, x0, x1 = detail_crops[stem]
            render(out / 'figures' / (stem + '_detail.png'), rgb, before, after, bc, ac, stem + ' known gap',
                   (slice(y0, y1), slice(x0, x1)))
        if args.mode == 'test' and stem in render_names and audit['splits']:
            parent_id = max(audit['splits'], key=lambda a: a['parent_area'])['parent_id']
            marker = marker_stages(boundary[0, 0].numpy(), config)['marker_labels']
            box = find_objects(marker)[parent_id - 1]
            y, x = box
            crop = (slice(max(0, y.start - 24), min(rgb.shape[0], y.stop + 24)),
                    slice(max(0, x.start - 24), min(rgb.shape[1], x.stop + 24)))
            render(out / 'figures' / (stem + '_detail.png'), rgb, before, after, bc, ac, stem + ' largest changed seed', crop)
        rows.append(row)
        with (out / 'views.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps({'source': stem, 'view': view, 'base': len(bc), 'candidate': len(ac),
            'split_parents': audit['split_parent_count'], 'added_seeds': audit['added_marker_count'],
            'gt': row.get('gt', {}).get('summary')}, ensure_ascii=False), flush=True)

    render_names, detail_crops = set(), {}
    if args.mode == 'train':
        trace_dir = Path(project_path(config, options['trace_dir']))
        references = {(r['source'], r['view']): r for p in sorted(trace_dir.glob('part*/views.jsonl'))
                      for r in [json.loads(s) for s in p.read_text().splitlines()]}
        if len(references) != 64:
            raise ValueError('Need the complete frozen 64-view trace')
        detail_crops = {r['source']: r['path_bounds'] for r in json.loads((trace_dir / 'detail_routes.json').read_text())}
        base = CanonicalBackendDataset(project_path(config, config['paths']['raw_data_dir']), project_path(config, cfg['completed_gt_dir']))
        degradation = load_config(project_path(config, cfg['restoration']['degradation_config']))['rgb_restoration']['degradation']
        paired = PairedDegradationDataset(base, {**degradation, 'profile_probabilities': [0., 1., 0., 0.]}, cfg['noise'], cfg['seed'], repeats=1)
        paired.set_epoch(20)
        if len(base) != 32:
            raise ValueError('Manual cohort changed')
        for index in range(args.start, min(args.start + (args.limit or len(base)), len(base))):
            stem = base.samples[index].stem
            if args.images and stem not in args.images:
                continue
            clean, sample = base[index], paired[index]
            with np.load(base.completed_gt_dir / (stem + '_gt.npz'), allow_pickle=False) as source:
                gt = source['instance_map'].copy()
                trusted = source['original_covered'].astype(bool) & (gt > 0)
            degraded = undo_spatial(sample['image'], sample['horizontal_flip'], sample['vertical_flip'], sample['rotation_k'])
            for view, tensor in [('clean', clean['image']), ('degraded', degraded)]:
                maps = native_training_maps(model, restorer, tensor, gt.shape, config, cfg['restoration']['inference_seed'] + index)
                process(stem, view, *maps, reference=references[(stem, view)], gt=gt, trusted=trusted)
    else:
        files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
        if len(files) != options['expected_test_images']:
            raise ValueError('Test cohort incomplete')
        fixed = set(cfg['monitor']['images'])
        render_names = fixed | set(np.random.default_rng(options['render_seed']).choice([p.stem for p in files if p.stem not in fixed], 4, replace=False).tolist())
        dump(out / 'render_selection.json', {'names': sorted(render_names), 'seed': options['render_seed']})
        for index, path in enumerate(files):
            if not args.start <= index < args.start + (args.limit or len(files)):
                continue
            if args.images and path.stem not in args.images:
                continue
            with torch.no_grad():
                rgb, semantic, boundary = prediction_maps(model, path, config, device, restorer, cfg['restoration']['inference_seed'] + index)
                _, raw, ph, pw = prepare_image(path, 1024, device)
                output = model.semantic_decoder(model.encoder(raw.float()), raw.float())
                probability = torch.sigmoid(crop_letterbox_output(output.float(), 1024, ph, pw, rgb.shape[:2]).cpu())[0, 0].numpy()
            process(path.stem, 'test', rgb, semantic, boundary, probability,
                    control_path=Path(project_path(config, options['control_deployment'])))
    if not rows:
        raise ValueError('No selected inputs')
    if tensor_digest(model.state_dict().items()) != initial or tensor_digest(restorer.state_dict().items()) != ri:
        raise AssertionError('Frozen weights changed')
    report = {'complete': True, 'mode': args.mode, 'sources': len({r['source'] for r in rows}), 'inputs': len(rows),
        'frozen_model_exact': True, 'frozen_restorer_exact': True, 'no_training': True,
        'scope': 'mechanism diagnostic on seen GT or unlabeled test output differences; not official accuracy',
        'extra_area_filter_disabled': True, 'unchanged_minimum_area': int(config['inference'].get('min_instance_area', 50)),
        'summary': summarize(rows), 'elapsed_seconds': time.monotonic() - started}
    dump(out / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/experiments/affinity_preserve.yaml')
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--output')
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int)
    run(parser.parse_args())
