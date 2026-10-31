# -*- coding: utf-8 -*-
"""成对视野e20的四源完整输出：原标可信域、几何匹配与语义混淆分开。"""
from __future__ import annotations

import argparse
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
from train_affinity_connectivity import frozen_digest, get_restorer
from train_backend_adaptation import sha, tensor_digest
from utils.config import load_config, project_path
from utils.patch_diagnostic import trusted_phase_metrics


def geometry(gt, trust, pred, pairs, summary):
    keep = trust & (gt > 0)
    g, gi = np.unique(gt[keep], return_inverse=True)
    p, pi = np.unique(pred[keep], return_inverse=True)
    matrix = np.bincount(gi * len(p) + pi, minlength=len(g) * len(p)).reshape(len(g), len(p))
    ga, pa = matrix.sum(1), matrix.sum(0)
    positive = p > 0
    inter = matrix[:, positive]
    split = (inter >= np.maximum(50, .1 * ga[:, None])).sum(1) >= 2
    merge = (inter >= np.maximum(50, .1 * pa[None, positive])).sum(0) >= 2
    total = sum(pair['iou'] for pair in pairs)
    return dict(**summary, iou_sum=total, matched_iou=total / max(1, len(pairs)),
                gt_penalized_iou=total / max(1, len(g)), split_gt=int(split.sum()),
                merged_pred=int(merge.sum()))


def run(args):
    config = load_config(args.config)
    root = Path(project_path(config, config['backend_adaptation']['output_dir']))
    output = root/'analysis/gt'
    output.mkdir(parents=True, exist_ok=True)
    state = read(root/'candidate/status.json')
    pipeline = read(root/'pipeline_status.json')
    if pipeline['status'] != 'complete' or state['epoch'] != 20 or state['updates'] != 1280:
        raise RuntimeError('The candidate must have finished the authorized e20 pipeline')
    checkpoint = root/'candidate/final.pt'
    if sha(checkpoint) != state['final_checkpoint_sha256']:
        raise RuntimeError('Candidate checkpoint changed')
    source_hashes = {name:sha(ROOT/name) for name in pipeline['sources']}
    if source_hashes != pipeline['sources']:
        raise RuntimeError('Training runtime source changed')
    reference_report = read(ROOT/'outputs/affinity_balance/analysis/gt.json')
    reference_training = read(ROOT/'outputs/affinity_balance/analysis/training.json')
    expected = {'native':'2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b',
                'mixed':config['affinity_pair']['reference_sha256']}
    for name, digest in expected.items():
        if (reference_training[name]['checkpoint_sha256'] != digest or
                sha(ROOT/'outputs/affinity_balance'/name/'final.pt') != digest):
            raise RuntimeError('Cached labeled reference differs: '+name)
    cases = [c for c in read(Path(project_path(config, config['affinity_native']['prior_probe']))/'cases.json')
             if c['kind'] == 'train']
    if len(cases) != 4:
        raise RuntimeError('Expected the same four seen training sources')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    model, _ = load_backend(checkpoint, config, 'cuda')
    restorer = get_restorer(config, 'cuda')
    before = tensor_digest(model.state_dict().items())
    restorer_before = tensor_digest(restorer.state_dict().items())
    if frozen_digest(model) != state['frozen_state_sha256'] or restorer_before != state['restorer_state_sha256']:
        raise RuntimeError('Frozen inputs differ')
    records = []
    for case in cases:
        target, trust, classes = load_gt(case, config)
        predicted = {name:load_directory(ROOT/'outputs/affinity_balance/analysis/train'/case['name']/(name+'_local'), case['name'])
                     for name in expected}
        fixed = {}
        with torch.no_grad():
            rgb, views = predict_views(model, restorer, case['path'], config, 'cuda', case['seed'], [1024], fixed_audit=fixed)
        values = views['patch1024']
        predicted['paired'] = (values['instances'], values['classes'])
        folder = output/'predictions'/case['name']
        folder.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(folder/(case['name']+'_inst.png')), values['instances']):
            raise RuntimeError('Cannot save final PNG')
        write(folder/(case['name']+'_class.json'), values['classes'])
        per_view = {}
        for name, (ids, prediction_classes) in predicted.items():
            verify_prediction(ids, prediction_classes, rgb.shape[:2])
            phase = trusted_phase_metrics(target, classes, trust, ids, prediction_classes)
            if name in expected:
                previous = next(row['metrics'] for row in reference_report['rows']
                                if row['source'] == case['name'] and row['view'] == name+'_new_local')
                if phase != previous:
                    raise RuntimeError('Cached GT domain/metric changed: '+name+'/'+case['name'])
            pairs, summary = class_independent_pairs(target, trust, ids)
            per_view[name] = dict(class_aware=phase, geometry=geometry(target, trust, ids, pairs, summary),
                                  fixed_confusion=fixed_confusion(pairs, classes, prediction_classes), pairs=pairs)
        common = set.intersection(*(set(p['gt'] for p in value['pairs']) for value in per_view.values()))
        deltas = {}
        for name in expected:
            a = {p['gt']:p for p in per_view[name]['pairs']}
            b = {p['gt']:p for p in per_view['paired']['pairs']}
            shared = set(a) & set(b)
            deltas[name] = dict(reference_only=sorted(set(a)-set(b)), paired_only=sorted(set(b)-set(a)),
                                common_matches=len(shared), common_iou_delta_sum=sum(b[g]['iou']-a[g]['iou'] for g in shared))
        records.append(dict(source=case['name'], seed=case['seed'], trust_sha256=tensor_digest([('trust', torch.from_numpy(trust))]),
                            fixed_semantic_inputs=fixed, views=per_view, paired_deltas=deltas, three_way_common=len(common)))
        shown_gt = np.where(trust, target, 0).astype(np.uint16)
        shown_classes = {str(g):classes[int(g)] for g in np.unique(shown_gt) if g > 0}
        for suffix, crop in [('full', None), ('detail', [.25,.25,.75,.75])]:
            render(case['path'], [(shown_gt,shown_classes),*predicted.values()],
                   ['GT known only','native balance','mixed balance','paired views'],
                   output/(case['name']+'_'+suffix+'.png'),case['name']+' SEEN training / unknown ignored',crop=crop,width=380)
        print('GT',case['name'],flush=True)
    if tensor_digest(model.state_dict().items()) != before or tensor_digest(restorer.state_dict().items()) != restorer_before:
        raise RuntimeError('Read-only diagnostic changed a model')
    totals = {}
    for name in ('native','mixed','paired'):
        geometry_rows = [r['views'][name]['geometry'] for r in records]
        geom = {key:sum(r[key] for r in geometry_rows) for key in
                ('gt_instances','predicted_instances','matched','unmatched_gt','unmatched_pred','iou_sum','split_gt','merged_pred')}
        geom.update(matched_iou=geom['iou_sum']/max(1,geom['matched']),gt_penalized_iou=geom['iou_sum']/max(1,geom['gt_instances']))
        confusion = {key:sum(r['views'][name]['fixed_confusion'][key] for r in records)
                     for key in ('true_F_pred_F','true_F_pred_P','true_P_pred_F','true_P_pred_P','correct','wrong')}
        phases = {}
        for phase in ('ferrite','pearlite'):
            values = [r['views'][name]['class_aware'][phase] for r in records]
            aggregate = {k:sum(r[k] for r in values) for k in
                         ('gt_instances','pred_instances_in_known_domain','matches','iou_sum','split_gt','merged_pred')}
            aggregate.update(matched_iou=aggregate['iou_sum']/max(1,aggregate['matches']),
                             gt_penalized_iou=aggregate['iou_sum']/max(1,aggregate['gt_instances']))
            phases[phase] = aggregate
        totals[name] = dict(geometry=geom, fixed_confusion=confusion, class_aware=phases)
    report = dict(scope='Four SEEN training sources; original covered GT only. Not held-out or official accuracy.',
                  candidate_checkpoint_sha256=state['final_checkpoint_sha256'],references=expected,
                  matching='Class-independent maximum eligible pairs, then IoU; semantic confusion on those pairs.',
                  split_merge='Geometry ignores predicted classes; intersections >=50 known pixels and >=10% of corresponding known area.',
                  unknown_ignored=True, models_frozen_unchanged=True, rows=records,totals=totals)
    write(output/'report.json',report)
    print('TOTALS',json.dumps(totals),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/affinity_pair.yaml')
    run(parser.parse_args())
