# -*- coding: utf-8 -*-
"""固定四个已见训练源，拆开连接权重扫参的几何变化与实例语义变化。"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.backend_adaptation import load_backend
from tools.affinity_native_views import predict_views, verify_prediction
from tools.analyze_affinity_connectivity import load_directory, read, render, write
from tools.probe_affinity_patch import load_gt
from tools.probe_semantic_fixed import class_independent_pairs, fixed_confusion
from train_affinity_balance import comparable
from train_affinity_connectivity import frozen_digest, get_restorer
from train_affinity_sweep import reference_config
from train_backend_adaptation import sha, tensor_digest
from utils.config import load_config, project_path
from utils.patch_diagnostic import trusted_phase_metrics


CHECKPOINTS = {
    'r1': ('outputs/affinity_balance/mixed/final.pt',
           '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'),
    'n075': ('outputs/affinity_sweep/n075/final.pt',
             'e6ef317af4b4f86427807143cd5fa426c5b40bedb64e5b0d16b1fdff3bad4394'),
    'n125': ('outputs/affinity_sweep/n125/final.pt',
             'e20aac61eb1e45bc0f253ce999691a3ae60ad4fd7e7fe4d4e3fb7c1a4a2fb138'),
}
ARMS = tuple(CHECKPOINTS)


def geometry(gt, trust, pred, pairs, summary):
    """未知像素从交集及双方面积中移除；显著拆分/粘连与类别无关。"""
    keep = np.asarray(trust, bool) & (gt > 0)
    g, gi = np.unique(gt[keep], return_inverse=True)
    p, pi = np.unique(pred[keep], return_inverse=True)
    matrix = np.bincount(gi * len(p) + pi, minlength=len(g) * len(p)).reshape(len(g), len(p))
    ga, pa = matrix.sum(1), matrix.sum(0)
    indices = np.flatnonzero(p > 0)
    inter = matrix[:, indices]
    split_support = inter >= np.maximum(50, .1 * ga[:, None])
    merge_support = inter >= np.maximum(50, .1 * pa[None, indices])
    split_ids = [int(g[i]) for i in np.flatnonzero(split_support.sum(1) >= 2)]
    merges = [dict(pred=int(p[indices[j]]), gt_ids=[int(g[i]) for i in np.flatnonzero(merge_support[:, j])])
              for j in np.flatnonzero(merge_support.sum(0) >= 2)]
    merge_pairs = sorted({tuple(pair) for row in merges for pair in combinations(row['gt_ids'], 2)})
    total = sum(pair['iou'] for pair in pairs)
    return dict(**summary, iou_sum=total, matched_iou=total / max(1, len(pairs)),
                gt_penalized_iou=total / max(1, len(g)), split_gt=len(split_ids), merged_pred=len(merges),
                split_gt_ids=split_ids, merged_predictions=merges, merged_gt_pairs=[list(p) for p in merge_pairs])


def paired_changes(reference, candidate, gt_classes, reference_classes, candidate_classes):
    """以共同GT ID比较，不把各自不同的匹配集合当相同评估总体。"""
    a = {p['gt']: p for p in reference['pairs']}
    b = {p['gt']: p for p in candidate['pairs']}
    shared = sorted(set(a) & set(b))
    corrected, introduced, unchanged_wrong = [], [], []
    for g in shared:
        old_correct = int(reference_classes[str(a[g]['pred'])]) == gt_classes[g]
        new_correct = int(candidate_classes[str(b[g]['pred'])]) == gt_classes[g]
        if not old_correct and new_correct:
            corrected.append(g)
        elif old_correct and not new_correct:
            introduced.append(g)
        elif not old_correct and not new_correct:
            unchanged_wrong.append(g)
    old_split, new_split = set(reference['geometry']['split_gt_ids']), set(candidate['geometry']['split_gt_ids'])
    old_merge = {tuple(p) for p in reference['geometry']['merged_gt_pairs']}
    new_merge = {tuple(p) for p in candidate['geometry']['merged_gt_pairs']}
    return dict(reference_only_matched_gt=sorted(set(a) - set(b)), candidate_only_matched_gt=sorted(set(b) - set(a)),
                common_matches=len(shared), common_iou_delta_sum=sum(b[g]['iou'] - a[g]['iou'] for g in shared),
                common_iou_deltas=[dict(gt=g, reference_iou=a[g]['iou'], candidate_iou=b[g]['iou'],
                                       delta=b[g]['iou'] - a[g]['iou']) for g in shared],
                removed_split_gt=sorted(old_split - new_split), new_split_gt=sorted(new_split - old_split),
                removed_merged_gt_pairs=[list(p) for p in sorted(old_merge - new_merge)],
                new_merged_gt_pairs=[list(p) for p in sorted(new_merge - old_merge)],
                semantic_corrected_on_common_gt=corrected, semantic_new_wrong_on_common_gt=introduced,
                semantic_unchanged_wrong_on_common_gt=unchanged_wrong)


def preflight(root):
    pipeline = read(root / 'pipeline_status.json')
    if pipeline['status'] != 'complete' or pipeline.get('control_retrained') is not False:
        raise RuntimeError('Only the completed, control-reusing sweep may be analyzed')
    sources = pipeline['sources']
    if len(sources) != 116:
        raise RuntimeError('Expected the registered 116 training runtime sources')
    for name, digest in sources.items():
        if sha(ROOT / name) != digest or sha(root / 'source' / name) != digest:
            raise RuntimeError('Current source or archived training source changed: ' + name)
    configs = {name: load_config('config/train/affinity_sweep_' + name + '.yaml') for name in ARMS[1:]}
    if json.loads(json.dumps(configs)) != pipeline['configs']:
        raise RuntimeError('Actual training configurations differ from current configurations')
    configs['r1'] = reference_config(configs['n075'])
    states = {}
    for name in ARMS:
        path, digest = CHECKPOINTS[name]
        folder = ROOT / Path(path).parent
        state = read(folder / 'status.json')
        if (state['status'] != 'complete' or state['epoch'] != 20 or state['updates'] != 1280
                or state.get('smoke') is not False or state.get('size') != 1024
                or state.get('precision') != 'FP32' or state['final_checkpoint_sha256'] != digest
                or sha(ROOT / path) != digest or comparable(state['config']) != comparable(configs[name])):
            raise RuntimeError('Checkpoint, e20 budget or configuration differs: ' + name)
        for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'):
            if state.get(key) is not True:
                raise RuntimeError('Training audit failed: ' + name + '/' + key)
        states[name] = state
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data'):
        if any(states[name][key] != states['r1'][key] for name in ARMS[1:]):
            raise RuntimeError('Shared initial/frozen/data contract differs: ' + key)
    reference_training = read(ROOT / 'outputs/affinity_balance/analysis/training.json')
    if reference_training['mixed']['checkpoint_sha256'] != CHECKPOINTS['r1'][1]:
        raise RuntimeError('Cached reference GT report belongs to another model')
    return configs, states, sources


def cached_reference(case):
    parent = ROOT / 'outputs/affinity_balance/analysis/train' / case['name']
    paths = [p for p in (parent / 'mixed_local', parent / 'mixed_new_local') if p.is_dir()]
    if not paths:
        raise RuntimeError('Missing cached mixed balance native1024 output: ' + case['name'])
    values = [load_directory(path, case['name']) for path in paths]
    if any(not np.array_equal(values[0][0], value[0]) or values[0][1] != value[1] for value in values[1:]):
        raise RuntimeError('Conflicting cached mixed balance folders')
    return values[0], dict(path=paths[0].relative_to(ROOT).as_posix(),
                          inst_sha256=sha(paths[0] / (case['name'] + '_inst.png')),
                          class_sha256=sha(paths[0] / (case['name'] + '_class.json')))


def totals(records):
    result = {}
    for name in ARMS:
        values = [r['views'][name] for r in records]
        geom = {k: sum(v['geometry'][k] for v in values) for k in
                ('gt_instances', 'predicted_instances', 'matched', 'unmatched_gt', 'unmatched_pred',
                 'iou_sum', 'split_gt', 'merged_pred')}
        geom.update(matched_iou=geom['iou_sum'] / max(1, geom['matched']),
                    gt_penalized_iou=geom['iou_sum'] / max(1, geom['gt_instances']))
        confusion = {k: sum(v['fixed_confusion'][k] for v in values) for k in
                     ('true_F_pred_F', 'true_F_pred_P', 'true_P_pred_F', 'true_P_pred_P', 'correct', 'wrong',
                      'wrong_gt_known_pixels', 'evaluated_gt_known_pixels')}
        common_confusion = {k: sum(r['three_way_common_confusion'][name][k] for r in records) for k in confusion}
        phases = {}
        for phase in ('ferrite', 'pearlite'):
            aggregate = {k: sum(v['class_aware'][phase][k] for v in values) for k in
                         ('gt_instances', 'pred_instances_in_known_domain', 'matches', 'iou_sum', 'split_gt', 'merged_pred')}
            aggregate.update(matched_iou=aggregate['iou_sum'] / max(1, aggregate['matches']),
                             gt_penalized_iou=aggregate['iou_sum'] / max(1, aggregate['gt_instances']))
            phases[phase] = aggregate
        result[name] = dict(geometry=geom, fixed_confusion=confusion, three_way_common_confusion=common_confusion,
                            class_aware=phases)
    return result


def run(args):
    root = Path(project_path(load_config(args.config), 'outputs/affinity_sweep'))
    output = root / 'analysis/gt'
    output.mkdir(parents=True, exist_ok=True)
    configs, states, source_hashes = preflight(root)
    config = configs['r1']
    prior = Path(project_path(config, config['affinity_native']['prior_probe']))
    cases = [c for c in read(prior / 'cases.json') if c['kind'] == 'train']
    if len(cases) != 4 or len({c['name'] for c in cases}) != 4:
        raise RuntimeError('Expected the same four distinct seen training sources')
    reference_report = read(ROOT / 'outputs/affinity_balance/analysis/gt.json')
    reference_metrics = {(r['source'], r['view']): r['metrics'] for r in reference_report['rows']}
    predictions, fixed_inputs, records = {}, {}, {}
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    restorer = get_restorer(config, args.device)
    restorer_before = tensor_digest(restorer.state_dict().items())
    if restorer_before != states['r1']['restorer_state_sha256']:
        raise RuntimeError('Actual fixed D5a state differs from training')
    for name in ARMS:
        model, _ = load_backend(ROOT / CHECKPOINTS[name][0], configs[name], args.device)
        model.eval().requires_grad_(False)
        before = tensor_digest(model.state_dict().items())
        if before != states[name]['final_state_sha256'] or frozen_digest(model) != states[name]['frozen_state_sha256']:
            raise RuntimeError('Loaded state or non-affinity frozen tensors differ: ' + name)
        for case in cases:
            target, trust, classes = load_gt(case, config)
            audit = {}
            with torch.no_grad():
                rgb, views = predict_views(model, restorer, case['path'], configs[name], args.device,
                                           case['seed'], [1024], fixed_audit=audit)
            value = views['patch1024']
            current = (value['instances'], value['classes'])
            verify_prediction(*current, rgb.shape[:2])
            phase = trusted_phase_metrics(target, classes, trust, *current)
            if name == 'r1':
                cached, cache = cached_reference(case)
                if not np.array_equal(cached[0], current[0]) or cached[1] != current[1]:
                    raise RuntimeError('R1 final native1024 output no longer reproduces cache: ' + case['name'])
                if phase != reference_metrics[(case['name'], 'mixed_new_local')]:
                    raise RuntimeError('Cached GT domain/metrics changed: ' + case['name'])
                fixed_inputs[case['name']] = audit
                gt_root = Path(project_path(config, config['backend_adaptation']['completed_gt_dir']))
                records[case['name']] = dict(source=case['name'], seed=case['seed'],
                    trust_sha256=tensor_digest([('trust', torch.from_numpy(trust))]),
                    image_sha256=sha(case['path']), gt_npz_sha256=sha(gt_root / (case['name'] + '_gt.npz')),
                    gt_classes_sha256=sha(gt_root / (case['name'] + '_class.json')), cached_reference=cache,
                    fixed_semantic_inputs=audit, views={})
            elif audit != fixed_inputs[case['name']]:
                raise RuntimeError('Geometry semantic or raw probability changed: ' + name + '/' + case['name'])
            pairs, summary = class_independent_pairs(target, trust, current[0])
            records[case['name']]['views'][name] = dict(class_aware=phase,
                geometry=geometry(target, trust, current[0], pairs, summary),
                fixed_confusion=fixed_confusion(pairs, classes, current[1]), pairs=pairs, fixed_semantic_inputs=audit)
            predictions.setdefault(case['name'], {})[name] = current
            folder = output / 'predictions' / case['name'] / name
            folder.mkdir(parents=True, exist_ok=True)
            encoded, data = cv2.imencode('.png', current[0])
            if not encoded:
                raise RuntimeError('Cannot encode final PNG')
            data.tofile(folder / (case['name'] + '_inst.png'))
            write(folder / (case['name'] + '_class.json'), current[1])
            print('GT', name, case['name'], flush=True)
        if tensor_digest(model.state_dict().items()) != before:
            raise RuntimeError('Read-only diagnostic changed a backend model')
        del model
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    if tensor_digest(restorer.state_dict().items()) != restorer_before:
        raise RuntimeError('Read-only diagnostic changed D5a')
    for case in cases:
        target, trust, classes = load_gt(case, config)
        record, predicted = records[case['name']], predictions[case['name']]
        common = set.intersection(*(set(p['gt'] for p in record['views'][name]['pairs']) for name in ARMS))
        record['three_way_common_gt'] = sorted(common)
        record['three_way_common_confusion'] = {
            name: fixed_confusion([p for p in record['views'][name]['pairs'] if p['gt'] in common], classes, predicted[name][1])
            for name in ARMS}
        record['paired_changes'] = {
            name: paired_changes(record['views']['r1'], record['views'][name], classes, predicted['r1'][1], predicted[name][1])
            for name in ARMS[1:]}
        shown_gt = np.where(trust, target, 0).astype(np.uint16)
        shown_classes = {str(int(g)): classes[int(g)] for g in np.unique(shown_gt) if g > 0}
        for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
            render(case['path'], [(shown_gt, shown_classes), *(predicted[name] for name in ARMS)],
                   ['GT known only', 'mixed balance r1 e20', 'N075 e20', 'N125 e20'],
                   output / (case['name'] + '_' + suffix + '.png'),
                   case['name'] + ' SEEN training / unknown ignored / native1024', crop=crop, width=380)
    record_list = [records[c['name']] for c in cases]
    report = dict(scope='Four SEEN training sources; original covered GT only. Not held-out or official accuracy.',
        checkpoints={name: dict(path=CHECKPOINTS[name][0], sha256=CHECKPOINTS[name][1], epoch=20, updates=1280) for name in ARMS},
        source_hashes=source_hashes, source_snapshot_count=len(source_hashes), actual_runtime_matches_training=True,
        shared_initial_sha256=states['r1']['initial_state_sha256'], frozen_sha256=states['r1']['frozen_state_sha256'],
        restorer_sha256=restorer_before, cached_reference_final_output_equal=True, fixed_semantic_inputs_equal=True,
        unknown_ignored=True, models_frozen_unchanged=True,
        matching='Geometry: class-independent maximum number of IoU>=0.5 pairs, then IoU. Class-aware: unchanged cached GT metric.',
        split_merge='Intersections >=50 known pixels and >=10% of corresponding known area; geometry ignores classes.',
        paired_comparison='Corrections/new errors reported on common matched GT; gained/lost matches and split/merge sets reported separately.',
        rows=record_list, totals=totals(record_list))
    write(output / 'report.json', report)
    print('TOTALS', json.dumps(report['totals']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_sweep_n075.yaml')
    parser.add_argument('--device', default='cuda')
    run(parser.parse_args())
