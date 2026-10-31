# -*- coding: utf-8 -*-
"""H0固定e20诊断：同输入边界辨识、原标可信域和既有100图；不训练或调部署。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from models.backend_adaptation import load_backend, restore_first
from tools.affinity_native_views import predict_views, verify_prediction
from tools.analyze_affinity_connectivity import relation, render, write
from tools.analyze_affinity_sweep_gt import geometry, paired_changes
from tools.analyze_affinity_sweep_test import (
    PHASES, BINS, read, quantiles, load_prediction, areas_by_phase, area_bins,
    changes, pixel_class_changes, summarize_totals, summarize_comparison)
from tools.probe_affinity_equal_recall import (
    LAYERS, make_cases, prepared_view, score_histogram, fused_targets, summarize, plot)
from tools.probe_affinity_patch import load_gt
from tools.probe_semantic_fixed import class_independent_pairs, fixed_confusion
from tools.run_affinity_hard import source_names
from tools.run_affinity_sweep import runtime_contract
from train_affinity_balance import comparable
from train_affinity_connectivity import frozen_digest, get_restorer
from train_affinity_hard import REFERENCE, REFERENCE_SHA256, reference_config, verify_run
from train_backend_adaptation import sha, tensor_digest, fusion_options
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_loss import balanced_affinity_loss
from utils.config import load_config, project_path
from utils.patch_diagnostic import phase_description, trusted_phase_metrics

ARMS = ('r1', 'h0')
ANALYSIS_SOURCES = (
    'tools/analyze_affinity_hard.py', 'tools/analyze_affinity_sweep_gt.py',
    'tools/analyze_affinity_sweep_test.py', 'tools/probe_affinity_equal_recall.py',
    'tools/probe_semantic_fixed.py', 'tools/probe_affinity_patch.py',
    'tools/analyze_affinity_connectivity.py', 'tools/render_backend_ablation.py',
    'tools/affinity_native_views.py', 'utils/patch_diagnostic.py',
    'utils/affinity_loss.py', 'utils/affinity_fusion.py')


def tensor_items(value, prefix='features'):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from tensor_items(item, prefix + '/' + str(name))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_items(item, prefix + '/' + str(index))


def verify_sources(run, sources):
    if len(sources) != 119 or set(sources) != set(source_names()):
        raise RuntimeError('Expected the unchanged registered 119-source training closure')
    for name, expected in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('Invalid source path: ' + name)
        if sha(ROOT / name) != expected or sha(run / 'source' / name) != expected:
            raise RuntimeError('Training source or archived source changed: ' + name)


def preflight(config, run):
    pipeline = read(run / 'pipeline_status.json')
    if (pipeline.get('status') != 'complete' or pipeline.get('smoke') is not False
            or pipeline.get('control_retrained') is not False
            or pipeline.get('completed') != ['h0', 'h0_infer']
            or pipeline.get('config') != json.loads(json.dumps(config))):
        raise RuntimeError('Complete isolated H0 training and final inference are required')
    verify_sources(run, pipeline['sources'])
    runtime = runtime_contract(config)
    if runtime != pipeline['runtime']:
        raise RuntimeError('Current Python/SAM2 runtime differs from actual training')
    audit = verify_run(config, run / 'h0', 1280)
    if audit != pipeline['audits'].get('h0') or audit.get('passed') is not True:
        raise RuntimeError('Actual complete 1280-update audit differs from the pipeline receipt')
    configs = {'r1': reference_config(config), 'h0': config}
    folders = {'r1': Path(project_path(config, REFERENCE)), 'h0': run / 'h0'}
    states, identities = {}, {}
    for name, folder in folders.items():
        state = read(folder / 'status.json')
        actual = sha(folder / 'final.pt')
        expected = REFERENCE_SHA256 if name == 'r1' else audit['final_checkpoint_sha256']
        if (state['status'] != 'complete' or state['updates'] != 1280 or state['epoch'] != 20
                or state.get('smoke') is not False or state.get('size') != 1024
                or state.get('precision') != 'FP32' or state['final_checkpoint_sha256'] != actual
                or actual != expected or comparable(state['config']) != comparable(configs[name])):
            raise RuntimeError('Actual e20 checkpoint/config identity differs: ' + name)
        states[name] = state
        identities[name] = dict(checkpoint=folder.joinpath('final.pt').relative_to(ROOT).as_posix(),
            checkpoint_sha256=actual, epoch=20, updates=1280,
            negative_weight=1., hard_negative_weight=1. if name == 'r1' else 0.,
            hard_negative_gamma=2., pseudo_weight=.5, status_sha256=sha(folder / 'status.json'))
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data'):
        if states['r1'][key] != states['h0'][key]:
            raise RuntimeError('Shared initial/frozen/restorer/data differs: ' + key)
    return dict(configs=configs, folders=folders, states=states, identities=identities,
        audit=audit, sources=pipeline['sources'], runtime=runtime,
        pipeline_sha256=sha(run / 'pipeline_status.json'),
        analysis_sources={name: sha(ROOT / name) for name in ANALYSIS_SOURCES})


def load_checked(name, checks, device):
    model, bundle = load_backend(checks['folders'][name] / 'final.pt', checks['configs'][name], device)
    model.eval().requires_grad_(False)
    before = tensor_digest(model.state_dict().items())
    state = checks['states'][name]
    if (bundle['epoch'] != 20 or before != state['final_state_sha256']
            or frozen_digest(model) != state['frozen_state_sha256']
            or any(p.requires_grad for p in model.parameters())):
        raise RuntimeError('Loaded model/frozen tensors differ: ' + name)
    return model, before


def probe(config, checks, out, device, seed):
    """同一次D5a及encoder输出进入两个头，统一h1参考loss；分界是负连接关系。"""
    out.mkdir()
    cases = make_cases(config, 'all', seed)
    if len(cases) != 32:
        raise RuntimeError('Expected the original 32 seen manual sources')
    write(out / 'cases.json', cases)
    models, before = {}, {}
    for name in ARMS:
        models[name], before[name] = load_checked(name, checks, device)
    restorer = get_restorer(config, device)
    restorer_before = tensor_digest(restorer.state_dict().items())
    if restorer_before != checks['states']['r1']['restorer_state_sha256']:
        raise RuntimeError('Actual D5a differs from the training frozen state')
    parameters = {name: sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in restorer.parameters()) for name, model in models.items()}
    if any(count >= 500_000_000 for count in parameters.values()):
        raise RuntimeError('Deployment parameter limit violated')
    mode, fusion = fusion_options(config)
    reference = dict(negative_weight=1., hard_negative_weight=1.,
                     hard_negative_gamma=2., normalize_edge_weights=True)
    records, losses, receipts = [], [], []
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        for case in cases:
            for profile in ('clean', 'blur'):
                image, target, valid, receipt = prepared_view(case, profile, config, seed)
                tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()[None].to(device)
                restored = restore_first(restorer, tensor, receipt['restoration_seed'])
                features = models['r1'].encoder(restored)
                receipt['restored_sha256'] = tensor_digest([('image', restored)])
                receipt['shared_features_sha256'] = tensor_digest(tensor_items(features))
                if not receipts:
                    other = models['h0'].encoder(restored)
                    if tensor_digest(tensor_items(other)) != receipt['shared_features_sha256']:
                        raise RuntimeError('Shared encoder features do not reproduce H0 exactly')
                    del other
                receipts.append(receipt)
                target_t = torch.from_numpy(target)[None].to(device, dtype=torch.float32)
                valid_t = torch.from_numpy(valid)[None].to(device)
                for name, model in models.items():
                    logits = model.affinity_decoder(features)['affinity_logits'].float()
                    if tuple(logits.shape) != (1, 8, 512, 512) or not torch.isfinite(logits).all():
                        raise RuntimeError('Nonfinite or wrong-shape affinity grid')
                    loss, metrics = balanced_affinity_loss(logits, target_t, valid_t, **reference)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite unified reference loss')
                    losses.append(dict(model=name, source=case['name'], profile=profile,
                        reference_loss=float(loss), connection_recall=metrics['recall'],
                        separation_specificity=metrics['specificity']))
                    scores = (1 - logits.sigmoid())[0].cpu().numpy()
                    fused = affinity_boundary_probability(logits, mode=mode, **fusion)[0, 0].cpu().numpy()
                    histograms = {}
                    for layer, selection in (('relations_all', slice(None)), ('relations_short', slice(0, 4)),
                            ('relations_distance2', slice(4, 6)), ('relations_distance4', slice(6, 8))):
                        histograms[layer] = score_histogram(scores[selection], ~target[selection], valid[selection])
                    for layer, strict in (('fused_boundary', False), ('fused_deep_interior', True)):
                        boundary, eligible = fused_targets(target, valid, strict_interior=strict)
                        histograms[layer] = score_histogram(fused, boundary, eligible)
                    records.append(dict(model='control' if name == 'r1' else 'balance',
                        source=case['name'], profile=profile, histograms=histograms))
                write(out / 'status.json', dict(status='running', completed_views=len(receipts),
                                              expected_views=64, training=False))
                print('PROBE', case['name'], profile, len(receipts), flush=True)
    if any(tensor_digest(model.state_dict().items()) != before[name] for name, model in models.items()):
        raise RuntimeError('Read-only probe changed a backend')
    if tensor_digest(restorer.state_dict().items()) != restorer_before:
        raise RuntimeError('Read-only probe changed D5a')
    pooled, paired = summarize(records, [.90, .95, .97, .98])
    np.savez_compressed(out / 'histograms.npz', **{
        record['model'] + '/' + record['source'] + '/' + record['profile'] + '/' + layer: histogram
        for record in records for layer, histogram in record['histograms'].items()})
    plot(records, out)
    loss_summary = {profile: {name: {key: float(np.mean([row[key] for row in losses
        if row['model'] == name and row['profile'] == profile]))
        for key in ('reference_loss', 'connection_recall', 'separation_specificity')}
        for name in ARMS} for profile in ('clean', 'blur')}
    report = dict(complete=True, training=False, sources=32, input_views=64, affinity_head_views=128,
        shared_encoder_views=64, extra_encoder_parity_views=1, encoder_parity_exact=True,
        frozen_exact=True, d5a_frozen_exact=True, parameters_including_d5a=parameters,
        seed=seed, checkpoints=checks['identities'], receipts=receipts, layers=list(LAYERS),
        scope='32 SEEN manual sources, original trusted pairs; no holdout or official accuracy',
        histogram_model_names=dict(control='r1/h1 mixed balance', balance='h0 candidate'),
        directions='target=1 means inside connection; boundary score=1-sigmoid(logits). '
            'Loss recall is connection recall, specificity is true separation. Histogram row0=boundary,row1=interior.',
        reference_loss_formula=reference, loss_summary=loss_summary, loss_rows=losses,
        comparisons={'h0': dict(pooled=pooled, source_paired=paired)},
        caveats=['Equal-recall thresholds are diagnostic and do not change deployment.',
                 'Fused boundary labels are mechanism diagnostics, not final-instance scores.',
                 'Incomplete GT unknown pixels remain ignored. No epoch/model selection.'])
    write(out / 'report.json', report)
    write(out / 'status.json', dict(status='complete', completed_views=64, training=False))
    del models, features, restored, restorer
    torch.cuda.empty_cache()
    return report


def gt_totals(rows):
    totals = {}
    for name in ARMS:
        values = [row['views'][name] for row in rows]
        geom = {key: sum(v['geometry'][key] for v in values) for key in
            ('gt_instances', 'predicted_instances', 'matched', 'unmatched_gt', 'unmatched_pred',
             'iou_sum', 'split_gt', 'merged_pred')}
        geom.update(matched_iou=geom['iou_sum'] / max(1, geom['matched']),
                    gt_penalized_iou=geom['iou_sum'] / max(1, geom['gt_instances']))
        keys = ('true_F_pred_F', 'true_F_pred_P', 'true_P_pred_F', 'true_P_pred_P', 'correct', 'wrong',
                'wrong_gt_known_pixels', 'evaluated_gt_known_pixels')
        confusion = {k: sum(v['fixed_confusion'][k] for v in values) for k in keys}
        common = {k: sum(row['two_way_common_confusion'][name][k] for row in rows) for k in keys}
        phases = {}
        for phase in PHASES:
            data = {k: sum(v['class_aware'][phase][k] for v in values) for k in
                ('gt_instances', 'pred_instances_in_known_domain', 'matches', 'iou_sum', 'split_gt', 'merged_pred')}
            data.update(matched_iou=data['iou_sum'] / max(1, data['matches']),
                        gt_penalized_iou=data['iou_sum'] / max(1, data['gt_instances']))
            phases[phase] = data
        totals[name] = dict(geometry=geom, fixed_confusion=confusion,
                           two_way_common_confusion=common, class_aware=phases)
    return totals


def gt(config, checks, out, device, reference_report_path):
    out.mkdir()
    prior = Path(project_path(config, config['affinity_native']['prior_probe']))
    cases = [c for c in read(prior / 'cases.json') if c['kind'] == 'train']
    if len(cases) != 4 or len({c['name'] for c in cases}) != 4:
        raise RuntimeError('Expected the unchanged four seen GT sources')
    reference_report = read(reference_report_path)
    if (reference_report['checkpoints']['r1']['sha256'] != REFERENCE_SHA256
            or reference_report.get('unknown_ignored') is not True
            or reference_report.get('fixed_semantic_inputs_equal') is not True):
        raise RuntimeError('Cached GT metadata belongs to another control/domain')
    refrows = {r['source']: r for r in reference_report['rows']}
    if len(refrows) != 4 or set(refrows) != {c['name'] for c in cases}:
        raise RuntimeError('Cached GT metadata cohort differs')
    model, before = load_checked('h0', checks, device)
    restorer = get_restorer(config, device)
    restorer_before = tensor_digest(restorer.state_dict().items())
    if restorer_before != checks['states']['h0']['restorer_state_sha256']:
        raise RuntimeError('GT inference D5a state differs')
    rows = []
    for case in cases:
        target, trust, classes = load_gt(case, config)
        if not trust.any():
            raise RuntimeError('Original trusted GT domain is empty')
        fixed = {}
        with torch.no_grad():
            rgb, views = predict_views(model, restorer, case['path'], config, device,
                                       case['seed'], [1024], fixed_audit=fixed)
        value = views['patch1024']
        h0 = (value['instances'], value['classes'])
        verify_prediction(*h0, rgb.shape[:2])
        cache = ROOT / 'outputs/affinity_balance/analysis/train' / case['name'] / 'mixed_local'
        r1 = load_prediction(cache, case['name'], rgb.shape[:2])
        gtroot = Path(project_path(config, config['backend_adaptation']['completed_gt_dir']))
        receipt = dict(seed=case['seed'], image_sha256=sha(case['path']),
            trust_sha256=tensor_digest([('trust', torch.from_numpy(trust))]),
            gt_npz_sha256=sha(gtroot / (case['name'] + '_gt.npz')),
            gt_classes_sha256=sha(gtroot / (case['name'] + '_class.json')))
        refrow = refrows[case['name']]
        if any(refrow[k] != v for k, v in receipt.items()) or fixed != refrow['fixed_semantic_inputs']:
            raise RuntimeError('GT input/trusted domain/fixed semantic hashes changed: ' + case['name'])
        row = dict(source=case['name'], **receipt, fixed_semantic_inputs=fixed,
            cached_reference=dict(path=cache.relative_to(ROOT).as_posix(),
                inst_sha256=sha(cache / (case['name'] + '_inst.png')),
                class_sha256=sha(cache / (case['name'] + '_class.json'))), views={})
        predictions = {'r1': r1, 'h0': h0}
        for name, current in predictions.items():
            pairs, summary = class_independent_pairs(target, trust, current[0])
            row['views'][name] = dict(class_aware=trusted_phase_metrics(target, classes, trust, *current),
                geometry=geometry(target, trust, current[0], pairs, summary), pairs=pairs,
                fixed_confusion=fixed_confusion(pairs, classes, current[1]), fixed_semantic_inputs=fixed)
        if row['views']['r1'] != refrow['views']['r1']:
            raise RuntimeError('Cached r1 GT output/metrics no longer reproduce reference metadata: ' + case['name'])
        row['reference_metrics_exact'] = True
        common = set(p['gt'] for p in row['views']['r1']['pairs']) & set(p['gt'] for p in row['views']['h0']['pairs'])
        row['two_way_common_gt'] = sorted(common)
        row['two_way_common_confusion'] = {name: fixed_confusion(
            [p for p in row['views'][name]['pairs'] if p['gt'] in common], classes, predictions[name][1]) for name in ARMS}
        row['paired_changes'] = {'h0': paired_changes(row['views']['r1'], row['views']['h0'],
                                                     classes, r1[1], h0[1])}
        folder = out / 'predictions' / case['name'] / 'h0'
        folder.mkdir(parents=True)
        success, encoded = cv2.imencode('.png', h0[0])
        if not success:
            raise RuntimeError('Cannot save GT candidate uint16 output')
        encoded.tofile(folder / (case['name'] + '_inst.png'))
        write(folder / (case['name'] + '_class.json'), h0[1])
        row['candidate_output'] = dict(png_sha256=sha(folder / (case['name'] + '_inst.png')),
                                      classes_sha256=sha(folder / (case['name'] + '_class.json')))
        shown = np.where(trust, target, 0).astype(np.uint16)
        shown_classes = {str(int(g)): classes[int(g)] for g in np.unique(shown) if g > 0}
        for suffix, crop in (('full', None), ('detail', [.25, .25, .75, .75])):
            render(case['path'], [(shown, shown_classes), r1, h0],
                ['GT known only', 'mixed r1 h1 e20', 'H0 r1 h0 e20'], out / (case['name'] + '_' + suffix + '.png'),
                case['name'] + ' SEEN training / unknown ignored', crop=crop, width=380)
        rows.append(row)
        write(out / 'status.json', dict(status='running', completed_sources=len(rows), training=False))
        print('GT', case['name'], flush=True)
    if (tensor_digest(model.state_dict().items()) != before
            or tensor_digest(restorer.state_dict().items()) != restorer_before):
        raise RuntimeError('GT diagnostic changed frozen model/restorer')
    report = dict(complete=True, scope='Four SEEN training sources; original covered GT only. Not held-out or official accuracy.',
        checkpoints=checks['identities'], cached_reference_report_sha256=sha(reference_report_path),
        cached_reference_metrics_equal=True, fixed_semantic_inputs_equal=True, unknown_ignored=True,
        models_frozen_unchanged=True, rows=rows, totals=gt_totals(rows),
        matching='Geometry: maximize number of class-independent IoU>=0.5 pairs, then IoU. '
            'Class-aware metrics retain the previous trusted GT implementation.',
        split_merge='Intersection>=50 known pixels and >=10% of corresponding known area; unknown ignored.',
        paired_comparison='Corrected/new errors and IoU deltas use the same matched GT set; gained/lost GT and split/merge sets separately.')
    write(out / 'report.json', report)
    write(out / 'status.json', dict(status='complete', completed_sources=4, training=False))
    del model, restorer
    torch.cuda.empty_cache()
    return report


def test(config, checks, out, seed):
    """只读最终输出，不重推100图，不把预测之间的重叠当GT准确率。"""
    out.mkdir()
    gallery = out / 'gallery'
    gallery.mkdir()
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    expected = {path.stem for path in files}
    if len(files) != 100 or len(expected) != 100:
        raise RuntimeError('Expected exactly 100 distinct semifinal sources')
    manifests, directories = {}, {}
    for arm, key in (('mixed', 'r1'), ('h0', 'h0')):
        folder = checks['folders'][key]
        manifest = read(folder / 'deployment/manifest.json')
        selected = [r for r in manifest['images'] if r.get('view') == 'patch1024']
        if (manifest.get('complete') is not True or manifest['epoch'] != 20 or manifest['precision'] != 'FP32'
                or manifest['checkpoint_sha256'] != checks['identities'][key]['checkpoint_sha256']
                or 1024 not in manifest['sizes'] or len(selected) != 100
                or {r['source'] for r in selected} != expected or (arm == 'h0' and manifest['sizes'] != [1024])):
            raise RuntimeError('Final deployment manifest differs: ' + arm)
        directory = folder / 'deployment/patch1024'
        for suffix in ('_inst.png', '_class.json'):
            if {p.name for p in directory.glob('*' + suffix)} != {s + suffix for s in expected}:
                raise RuntimeError('Exactly 100 PNG/JSON members required: ' + arm)
        directories[arm] = directory
        manifests[arm] = {r['source']: r for r in selected}
        checks['identities'][key]['manifest_sha256'] = sha(folder / 'deployment/manifest.json')
    fixed = ['test_101', 'test_116']
    excluded = set(config['backend_adaptation']['monitor']['images']) | set(fixed)
    random = sorted(np.random.default_rng(seed).choice([p.stem for p in files if p.stem not in excluded], 6, replace=False).tolist())
    if not set(fixed).issubset(expected):
        raise RuntimeError('Missing fixed visual sources')
    report = dict(scope='100 unlabeled final predictions; composition/topology, not GT accuracy or official area score',
        inference_rerun=False, training=False, epoch_selection=False, postprocessing_tuned=False,
        selection=dict(fixed=fixed, random=random, seed=seed, excluded_original_monitor=sorted(excluded)),
        area_bins=dict(reference='Each source mixed-r1 ferrite median; identical disjoint cuts for both arms',
            intervals={name: [low, None if np.isinf(high) else high] for name, low, high in BINS},
            interpretation='Small predictions are not labelled GT errors or fragments.'),
        relation_definition='Prediction-to-prediction maximum-total-IoU Hungarian matching, IoU50/95. '
            'Split/merge support>=50 pixels and >=10% of corresponding full predicted area. No correctness claim.',
        images=[], output_receipts=[])
    for index, path in enumerate(files):
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read source dimensions: ' + path.stem)
        predictions = {arm: load_prediction(folder, path.stem, image.shape[:2]) for arm, folder in directories.items()}
        area = {arm: areas_by_phase(*pred) for arm, pred in predictions.items()}
        median = float(np.median(area['mixed']['ferrite'])) if len(area['mixed']['ferrite']) else None
        row = dict(source=path.stem, shape=list(image.shape[:2]), original_sha256=sha(path),
                   mixed_ferrite_median=median, views={}, changes={}, relations={})
        for arm, pred in predictions.items():
            description = phase_description(*pred)
            if description != manifests[arm][path.stem]['phases']:
                raise RuntimeError('Final prediction statistics differ from manifest: ' + arm + '/' + path.stem)
            for phase in PHASES:
                description[phase]['mean_area'] = float(area[arm][phase].mean()) if len(area[arm][phase]) else None
                description[phase]['area_quantiles'] = quantiles(area[arm][phase])
            description['ferrite'].update(fixed_reference_bins=area_bins(area[arm]['ferrite'], median),
                tiny_under200_pixels=int(area[arm]['ferrite'][area[arm]['ferrite'] < 200].sum()))
            row['views'][arm] = description
            report['output_receipts'].append(dict(source=path.stem, arm=arm,
                png_sha256=sha(directories[arm] / (path.stem + '_inst.png')),
                classes_sha256=sha(directories[arm] / (path.stem + '_class.json'))))
        row['changes']['h0'] = changes(row['views']['mixed'], row['views']['h0'])
        row['relations']['h0'] = relation(predictions['mixed'][0], predictions['h0'][0],
                                         predictions['mixed'][1], predictions['h0'][1])
        row['relations']['h0'].update(pixel_class_changes(predictions['mixed'], predictions['h0']),
            final_ids_exact=bool(np.array_equal(predictions['mixed'][0], predictions['h0'][0])),
            class_json_exact=predictions['mixed'][1] == predictions['h0'][1])
        report['images'].append(row)
        if path.stem in fixed + random:
            for suffix, crop in (('full', None), ('detail', [.25, .25, .75, .75])):
                render(path, list(predictions.values()), ['mixed r1 h1 e20', 'H0 r1 h0 e20'],
                    gallery / (path.stem + '_' + suffix + '.png'), path.stem + ' fixed e20, no test GT', crop=crop, width=380)
        if index % 20 == 0:
            write(out / 'status.json', dict(status='running', completed_sources=index + 1, inference_rerun=False))
            print('CPU_TEST', index + 1, path.stem, flush=True)
    report['totals'] = {arm: summarize_totals(report['images'], arm) for arm in ('mixed', 'h0')}
    report['comparisons'] = {'h0': summarize_comparison(report['images'], 'h0')}
    report['pooled_changes'] = {'h0': changes(
        {phase: dict(report['totals']['mixed'][phase], mean_area=report['totals']['mixed'][phase]['pooled_mean_area']) for phase in PHASES},
        {phase: dict(report['totals']['h0'][phase], mean_area=report['totals']['h0'][phase]['pooled_mean_area']) for phase in PHASES})}
    ordered = sorted((r for r in report['images'] if r['changes']['h0']['ferrite']['mean_area_percent'] is not None),
                     key=lambda r: r['changes']['h0']['ferrite']['mean_area_percent'])
    report['ferrite_mean_area_change_tails'] = {tail: [dict(source=r['source'], shape=r['shape'],
        changes=r['changes']['h0']['ferrite'], relations=r['relations']['h0']) for r in chosen]
        for tail, chosen in (('lower_six', ordered[:6]), ('upper_six', ordered[-6:]))}
    report['complete'] = len(report['images']) == 100 and len(report['output_receipts']) == 200
    write(out / 'test.json', report)
    write(out / 'summary.json', {k: v for k, v in report.items() if k not in ('images', 'output_receipts')})
    write(out / 'status.json', dict(status='complete', completed_sources=100, inference_rerun=False))
    return report


def self_test():
    """无GPU/无赛题数据的统计方向和未知域短测。"""
    target = np.ones((8, 4, 4), bool)
    target[0, :, 2] = False
    valid = np.ones_like(target)
    valid[:, 0] = False
    boundary, eligible = fused_targets(target, valid)
    scores = np.where(boundary, .9, .1).astype(np.float32)
    hist = score_histogram(scores, boundary, eligible, bins=128)
    if list(hist.sum(1)) != [3, 9]:
        raise AssertionError('Boundary versus interior histogram direction or unknown mask failed')
    from tools.probe_affinity_equal_recall import equal_recall
    if equal_recall(hist, .95)['interior_false_boundary_rate'] != 0:
        raise AssertionError('Equal-recall false boundary direction failed')
    labels = np.ones((32, 32), np.uint16)
    labels[:, 16:] = 2
    trust = np.ones_like(labels, bool)
    trust[24:] = False
    baseline = labels.copy()
    baseline[24:] = 9
    predicted = baseline.copy()
    predicted[:24, :8] = 3
    truth_classes = {1: 1, 2: 0}
    reference_classes, candidate_classes = {'1': 1, '2': 0, '9': 0}, {'1': 0, '2': 0, '3': 1, '9': 1}
    views = []
    for value, classes in ((baseline, reference_classes), (predicted, candidate_classes)):
        pairs, summary = class_independent_pairs(labels, trust, value)
        geom = geometry(labels, trust, value, pairs, summary)
        if any(p['pred'] == 9 for p in pairs) or geom['gt_instances'] != 2:
            raise AssertionError('Unknown GT area incorrectly counted')
        views.append(dict(pairs=pairs, geometry=geom))
    delta = paired_changes(views[0], views[1], truth_classes, reference_classes, candidate_classes)
    if delta['new_split_gt'] != [1] or delta['semantic_new_wrong_on_common_gt'] != [1]:
        raise AssertionError('Common GT changes or split statistics failed')
    bins = area_bins(np.array([1, 10, 25, 100]), 100.)
    if [bins[name]['instances'] for name, _, _ in BINS] != [0, 2, 2]:
        raise AssertionError('Disjoint median-relative area bins failed')
    flips = pixel_class_changes((baseline, reference_classes), (predicted, candidate_classes))
    if flips['ferrite_to_pearlite_pixels'] != 192 or flips['pearlite_to_ferrite_pixels'] != 256:
        raise AssertionError('Prediction-only pixel class changes failed')
    print('CPU_SYNTHETIC_PASS: boundary direction, equal recall, unknown ignored, common GT, split, bins, class pixels')


def run(args):
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Actual probe/GT require the server sam2_env GPU; --self-test is CPU only')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = load_config(args.config)
    runroot = Path(project_path(config, args.run))
    output = Path(project_path(config, args.output))
    started = time.time()
    checks = preflight(config, runroot)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'config.json', config)
    write(output / 'preflight.json', {k: checks[k] for k in ('audit', 'sources', 'runtime',
        'pipeline_sha256', 'analysis_sources', 'identities')})
    try:
        write(output / 'status.json', dict(status='running', stage='probe', training=False))
        probe_report = probe(config, checks, output / 'probe', args.device, args.seed)
        write(output / 'status.json', dict(status='running', stage='gt', training=False))
        gt_report = gt(config, checks, output / 'gt', args.device,
                       Path(project_path(config, args.reference_gt)))
        write(output / 'status.json', dict(status='running', stage='test', training=False))
        test_report = test(config, checks, output / 'test', args.seed)
        verify_sources(runroot, checks['sources'])
        if any(sha(ROOT / name) != value for name, value in checks['analysis_sources'].items()):
            raise RuntimeError('Analysis dependency changed during diagnostics')
        report = dict(complete=True, training=False, test_inference_rerun=False, epoch_selection=False,
            postprocessing_tuned=False, official_accuracy=False, checkpoint_selection='Fixed e20; no test/GT epoch selection',
            identities=checks['identities'], audit=checks['audit'], sources=checks['sources'],
            source_count=119, current_and_snapshot_exact=True, runtime=checks['runtime'],
            pipeline_sha256=checks['pipeline_sha256'], analysis_sources=checks['analysis_sources'],
            probe=probe_report, gt=gt_report, test={k: v for k, v in test_report.items() if k not in ('images', 'output_receipts')},
            elapsed_seconds=time.time() - started,
            caveats=['32/4 GT sources participated in training; mechanism checks do not establish generalization.',
                     'Unknown original GT is ignored; test sources have no GT.',
                     'Prediction counts/area/split/merge or equal-recall thresholds do not choose deployment or tune postprocessing.'])
        write(output / 'report.json', report)
        pages = ['<!doctype html><meta charset="utf-8"><title>H0 affinity fixed e20</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>mixed r1 h1 versus H0 r1 h0, fixed e20</h1>',
            '<p>Only hard-negative weight changes. Gold=F, blue=P. GT is seen training/original covered only. Test has no GT.</p>',
            '<p><a href="report.json">Complete report</a> | <a href="test/test.json">100 test rows</a> | <a href="gt/report.json">Four seen GT</a></p>',
            '<p>Equal-recall plot labels: control=r1 h1, balance=H0 h0; thresholds only diagnose, never change deployment.</p>',
            '<img src="probe/equal_recall.png">']
        for folder in (output / 'gt', output / 'test/gallery'):
            for path in sorted(folder.glob('*.png')):
                rel = html.escape(path.relative_to(output).as_posix())
                pages.append(f'<p>{html.escape(path.stem)}<br><img loading="lazy" src="{rel}"></p>')
        (output / 'index.html').write_text('\n'.join(pages), encoding='utf8')
        write(output / 'status.json', dict(status='complete', stage='done', training=False,
            probe_sources=32, gt_sources=4, test_sources=100, elapsed_seconds=time.time() - started))
        with zipfile.ZipFile(output / 'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.html', '.png', '.npz'):
                    archive.write(path, path.relative_to(output).as_posix())
        print('COMPLETE H0 analysis', output, flush=True)
    except Exception as error:
        write(output / 'status.json', dict(status='failed', training=False, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_hard_h0.yaml')
    parser.add_argument('--run', default='outputs/affinity_hard')
    parser.add_argument('--output', default='outputs/affinity_hard/analysis')
    parser.add_argument('--reference-gt', default='outputs/affinity_sweep/analysis/gt/report.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--self-test', action='store_true')
    options = parser.parse_args()
    self_test() if options.self_test else run(options)
