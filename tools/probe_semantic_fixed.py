# -*- coding: utf-8 -*-
"""固定新最佳实例PNG，比较四条已有语义路线；仅12源末端分类诊断。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROUTES = ('salign_raw', 'salign_d5a', 'lora_d5a', 'full_d5a')
CHECKPOINTS = {
    'baseline': ('outputs/affinity_balance/native/final.pt', '2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b', 20),
    'lora': ('outputs/semantic_lora/candidate/epoch_020.pt', '23b23d1f0cecb268091943610ceddffd1dffc4371ca548ce725639e2dfc4068b', 20),
    'full': ('outputs/semantic_d5a/full/epoch_060.pt', 'cce91a6643b919e69f6cd1f6f4b331265a33c80842e1587a241386d964f7d2a7', 60),
}


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_fixed_mask(source, target):
    """仅复制原始字节，禁止用图像库重新编码。"""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if file_sha(source) != file_sha(target):
        raise RuntimeError('Fixed PNG bytes changed')


def class_independent_pairs(gt, trust, pred, minimum_iou=.5):
    """仅原标可信域的一对一匹配；优先最大化合格配对数，再最大化IoU。

    不接受任何预测类别参数，同一PNG的配对集合由几何一次确定，供全部路线共用。
    """
    if gt.shape != pred.shape or trust.shape != gt.shape:
        raise ValueError('Matching grids must be identical')
    keep = np.asarray(trust, bool) & (gt > 0)
    if not keep.any():
        return [], dict(gt_instances=0, predicted_instances=0, matched=0, unmatched_gt=0, unmatched_pred=0)
    g, gi = np.unique(gt[keep], return_inverse=True)
    p, pi = np.unique(pred[keep], return_inverse=True)
    matrix = np.bincount(gi*len(p)+pi, minlength=len(g)*len(p)).reshape(len(g), len(p))
    ga, pa = matrix.sum(1), matrix.sum(0)
    indices = np.flatnonzero(p > 0)
    intersection = matrix[:, indices]
    iou = intersection / np.maximum(1, ga[:, None]+pa[None, indices]-intersection)
    eligible = iou >= float(minimum_iou)
    # 合格配对的主目标权重大于所有次级IoU之和，不将类别引入分配。
    score = eligible.astype(float)*(min(iou.shape)+1) + np.where(eligible, iou, 0)
    left, right = linear_sum_assignment(-score)
    pairs = [dict(gt=int(g[i]), pred=int(p[indices[j]]), iou=float(iou[i, j]),
                  gt_known_pixels=int(ga[i]), pred_known_pixels=int(pa[indices[j]]))
             for i, j in zip(left, right) if eligible[i, j]]
    return pairs, dict(gt_instances=len(g), predicted_instances=len(indices), matched=len(pairs),
                       unmatched_gt=len(g)-len(pairs), unmatched_pred=len(indices)-len(pairs))


def fixed_confusion(pairs, gt_classes, pred_classes):
    result = dict(true_F_pred_F=0, true_F_pred_P=0, true_P_pred_F=0, true_P_pred_P=0,
                  wrong_gt_known_pixels=0, evaluated_gt_known_pixels=0)
    for pair in pairs:
        truth = int(gt_classes[pair['gt']])
        guess = int(pred_classes[str(pair['pred'])])
        key = 'true_' + ('F' if truth else 'P') + '_pred_' + ('F' if guess else 'P')
        result[key] += 1
        result['evaluated_gt_known_pixels'] += pair['gt_known_pixels']
        if truth != guess:
            result['wrong_gt_known_pixels'] += pair['gt_known_pixels']
    result['correct'] = result['true_F_pred_F'] + result['true_P_pred_P']
    result['wrong'] = result['true_F_pred_P'] + result['true_P_pred_F']
    return result


def probability_details(ids, probability, scores):
    from utils.semantic_vote import adaptive_instance_core
    details = {}
    for value in np.unique(ids):
        if value == 0:
            continue
        mask = ids == value
        yy, xx = np.nonzero(mask)
        region = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
        local = mask[region]
        core, _ = adaptive_instance_core(local, fraction=.4, min_pixels=8, distance_power=2.)
        ring = local & ~core
        p = probability[region]
        score = scores[str(int(value))]
        details[str(int(value))] = dict(area=int(local.sum()), probability_all=score,
            probability_core=float(p[core].mean()), probability_outer=float(p[ring].mean()) if ring.any() else None,
            core_pixels=int(core.sum()), outer_pixels=int(ring.sum()), signed_margin=2*score-1,
            confidence=abs(2*score-1), low_confidence=abs(score-.5)<.1,
            core_class=int(float(p[core].mean())>.5),
            outer_class=int(float(p[ring].mean())>.5) if ring.any() else None)
    return details


def changes(original, current, details, base_details):
    changed = [key for key in current if current[key] != original[key]]
    p2f = [key for key in changed if current[key] == 1]
    f2p = [key for key in changed if current[key] == 0]
    ferrite = [key for key, cls in current.items() if cls == 1]
    pixels = sum(details[key]['area'] for key in ferrite)
    low = [key for key in changed if base_details[key]['low_confidence']]
    return dict(instances=len(current), ferrite=len(ferrite), ferrite_pixels=pixels,
                ferrite_mean_area=pixels/len(ferrite) if ferrite else None,
                changed_instances=len(changed), changed_pixels=sum(details[key]['area'] for key in changed),
                P_to_F_instances=len(p2f), P_to_F_pixels=sum(details[key]['area'] for key in p2f),
                F_to_P_instances=len(f2p), F_to_P_pixels=sum(details[key]['area'] for key in f2p),
                low_confidence_changed=len(low), high_confidence_changed=len(changed)-len(low),
                changed_ids=changed)


def run(args):
    import torch
    from models.backend_adaptation import load_backend, restore_first
    from models.semantic_lora import semantic_features
    from tools.affinity_native_views import verify_prediction
    from tools.analyze_affinity_connectivity import load_directory, render
    from tools.probe_affinity_patch import load_gt
    from tools.probe_affinity_scale_fusion import sources
    from tools.semantic_crossover import revote_instances
    from train_affinity_connectivity import get_restorer
    from train_backend_adaptation import tensor_digest
    from utils.affinity_deployment import crop_letterbox_output, prepare_image
    from utils.config import load_config, project_path

    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env GPU')
    config = load_config(args.config)
    out = Path(project_path(config, args.output))
    out.mkdir(parents=True, exist_ok=False)
    gallery = out/'gallery'
    gallery.mkdir()
    cases, selection = sources(config)
    write(out/'cases.json', cases)
    write(out/'status.json', dict(status='running', training=False))
    deploy = Path(project_path(config, 'outputs/affinity_balance/native/deployment'))
    manifest = json.loads((deploy/'manifest.json').read_text())
    if (not manifest['complete'] or manifest['checkpoint_sha256'] != CHECKPOINTS['baseline'][1]
        or manifest['epoch'] != 20 or manifest['precision'] != 'FP32' or 1024 not in manifest['sizes']):
        raise RuntimeError('Fixed best mask deployment identity differs')
    models, identities, digests = {}, {}, {}
    for key, (relative, expected, epoch) in CHECKPOINTS.items():
        path = Path(project_path(config, relative))
        if file_sha(path) != expected:
            raise RuntimeError('Semantic source checkpoint mismatch: '+key)
        model, bundle = load_backend(path, config, 'cuda')
        if bundle['epoch'] != epoch:
            raise RuntimeError('Unexpected source epoch')
        model.eval().requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Semantic source is not frozen')
        models[key] = model
        digests[key] = tensor_digest(model.state_dict().items())
        identities[key] = dict(path=relative, sha256=expected, epoch=epoch,
                               architecture=bundle['architecture'], parameters=sum(p.numel() for p in model.parameters()))
        del bundle
    encoder_digests = {key:tensor_digest(model.encoder.state_dict().items()) for key, model in models.items()}
    if len(set(encoder_digests.values())) != 1:
        raise RuntimeError('Shared geometry encoder/LoRA changed between semantic routes')
    if models['baseline'].semantic_lora is not None or models['full'].semantic_lora is not None:
        raise RuntimeError('Unexpected private LoRA in baseline/full route')
    adapter = models['lora'].semantic_lora
    if adapter is None or adapter.VERSION != 'independent_lora_v1':
        raise RuntimeError('Independent semantic LoRA missing')
    private_calls = []
    hook = adapter.register_forward_hook(lambda module, inp, output: private_calls.append(True))
    geometry_lora = dict(models['lora'].encoder.trunk.named_parameters())
    if all(torch.equal(value, geometry_lora[name]) for name, value in zip(adapter.parameter_names, adapter.weights)):
        raise RuntimeError('Private semantic LoRA equals geometry LoRA; wrong source or untrained adapter')
    restorer = get_restorer(config, 'cuda')
    rdigest = tensor_digest(restorer.state_dict().items())
    restoration_count = sum(p.numel() for p in restorer.parameters())
    resident = sum(row['parameters'] for row in identities.values()) + restoration_count
    # 即便候选需要独立第二编码器，按完整两个bundle计数的保守部署上界仍须合规。
    deployment_upper = {key:identities['baseline']['parameters'] + row['parameters'] + restoration_count
                        for key, row in identities.items() if key != 'baseline'}
    deployment_upper['baseline'] = identities['baseline']['parameters'] + restoration_count
    if max([resident, *deployment_upper.values()]) >= 500000000:
        raise RuntimeError('Combined parameter limit exceeded')
    source_names = ['tools/probe_semantic_fixed.py', 'models/semantic_lora.py',
                    'tools/probe_affinity_scale_fusion.py', 'tools/semantic_crossover.py', 'utils/semantic_vote.py']
    source = out/'source'
    for name in source_names:
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    records = []
    started = time.time()
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        for case in cases:
            directory = (deploy/'patch1024' if case['kind'] == 'test' else
                         Path(project_path(config, 'outputs/affinity_balance/analysis/train', case['name'], 'native_local')))
            mask_path = directory/(case['name']+'_inst.png')
            mask_sha = file_sha(mask_path)
            ids, original_classes = load_directory(directory, case['name'])
            rgb, image, ph, pw = prepare_image(case['path'], 1024, 'cuda')
            # 保留原部署from_numpy/permute产生的布局；强制contiguous可改变GPU数值。
            image = image.float()
            if ids.shape != rgb.shape[:2]:
                raise RuntimeError('Fixed mask and original image coordinates differ')
            restored = restore_first(restorer, image, case['seed']).float()
            restored_digest = tensor_digest([('restored', restored)])
            gt = load_gt(case, config)
            pairs, pair_stats = (class_independent_pairs(gt[0], gt[1], ids) if gt is not None else (None, None))
            record = dict(source=case['name'], kind=case['kind'], seed=case['seed'],
                          fixed_png_path=str(mask_path), fixed_png_sha256=mask_sha,
                          shape=list(ids.shape), original_instances=len(original_classes),
                          restored_tensor_sha256=restored_digest, image_stride=list(image.stride()),
                          restored_stride=list(restored.stride()), routes={}, fixed_pair_stats=pair_stats)
            predictions, base_details = {}, None
            for route in ROUTES:
                key = 'baseline' if route.startswith('salign') else route.split('_')[0]
                value = image if route == 'salign_raw' else restored
                model = models[key]
                before_calls = len(private_calls)
                logits = model.semantic_decoder(semantic_features(model, value), value)
                if isinstance(logits, dict):
                    logits = logits['semantic_logits']
                probability = crop_letterbox_output(logits.float(), 1024, ph, pw, rgb.shape[:2]).cpu().sigmoid()[0, 0].numpy()
                probability = np.ascontiguousarray(probability, dtype=np.float32)
                if route == 'lora_d5a' and len(private_calls) != before_calls+1:
                    raise RuntimeError('Semantic forward bypassed private LoRA')
                classes, scores = revote_instances(ids, probability)
                verify_prediction(ids, classes, rgb.shape[:2])
                details = probability_details(ids, probability, scores)
                if route == 'salign_raw':
                    if classes != original_classes:
                        raise RuntimeError('Baseline semantic classes fail exact reproduction: '+case['name'])
                    base_details = details
                folder = out/'predictions'/case['name']/route
                folder.mkdir(parents=True)
                copy_fixed_mask(mask_path, folder/mask_path.name)
                write(folder/(case['name']+'_class.json'), classes)
                write(folder/'instances.json', details)
                row = dict(changes=changes(original_classes, classes, details, base_details),
                           probability_sha256=hashlib.sha256(probability.tobytes()).hexdigest(),
                           copied_png_sha256=file_sha(folder/mask_path.name),
                           input='raw' if route == 'salign_raw' else 'same_D5a_first_x0_tensor',
                           semantic_features_route='private_LoRA' if route == 'lora_d5a' else 'original_encoder')
                if pairs is not None:
                    row['fixed_pair_confusion'] = fixed_confusion(pairs, gt[2], classes)
                record['routes'][route] = row
                predictions[route] = ids, classes
            if file_sha(mask_path) != mask_sha or tensor_digest([('restored', restored)]) != restored_digest:
                raise RuntimeError('Fixed source PNG or shared restored input changed')
            if pairs is not None:
                write(out/(case['name']+'_fixed_pairs.json'), pairs)
            for suffix, crop in [('full', None), ('detail', [.25, .25, .75, .75])]:
                render(case['path'], [predictions[r] for r in ROUTES],
                       ['best S-align/raw', 'S-align/D5a', 'private LoRA/D5a', 'full e60/D5a'],
                       gallery/(case['name']+'_'+suffix+'.png'),
                       case['name']+' FIXED balance e20 PNG; only classes change', crop=crop, width=380)
            records.append(record)
            write(out/'rows.json', records)
            write(out/'status.json', dict(status='running', completed=len(records), total=len(cases), training=False))
            print(json.dumps(dict(source=case['name'], fixed_png=True, baseline_exact=True,
                                  routes={r:v['changes'] for r,v in record['routes'].items()})), flush=True)
    hook.remove()
    if len(private_calls) != len(cases):
        raise RuntimeError('Incomplete private-LoRA forward audit')
    if any(tensor_digest(model.state_dict().items()) != digests[key] for key, model in models.items()):
        raise RuntimeError('Frozen semantic weights changed')
    if tensor_digest(restorer.state_dict().items()) != rdigest:
        raise RuntimeError('Frozen D5a state changed')
    totals = {}
    for route in ROUTES:
        training = [r['routes'][route]['fixed_pair_confusion'] for r in records if r['kind'] == 'train']
        testing = [r['routes'][route]['changes'] for r in records if r['kind'] == 'test']
        totals[route] = dict(known_domain_confusion={key:sum(r[key] for r in training) for key in training[0]},
            unlabeled_test_changes={key:sum(r[key] for r in testing) for key in
                ('instances', 'ferrite', 'ferrite_pixels', 'changed_instances', 'changed_pixels',
                 'P_to_F_instances', 'P_to_F_pixels', 'F_to_P_instances', 'F_to_P_pixels',
                 'low_confidence_changed', 'high_confidence_changed')})
    report = dict(complete=True, training=False, full_test_inference=False, official_submission=False,
        scope='Fixed best instance geometry. Four seen incomplete-GT training sources original covered domain only; eight unlabeled test sources. No test truth inferred.',
        samples=len(cases), selection=selection, checkpoints=identities, restorer=config['backend_adaptation']['restoration'],
        parameter_counts=dict(diagnostic_resident=resident, restorer=restoration_count,
                              conservative_combined_deployment_upper_bound=deployment_upper),
        private_lora_forward_calls=len(private_calls), private_lora_distinct_from_geometry=True,
        common_geometry_encoder_sha256=encoder_digests,
        baseline_classes_exact=True, original_png_bytes_unchanged=True, frozen_exact=True,
        restored_once_per_source=True, cpu_sigmoid_after_native_logit_restore=True,
        votes='float32 arithmetic mean >0.5; equal 0.5 is pearlite',
        probability_core='adaptive_instance_core fraction .4 min8; diagnostic only, not used to change class',
        fixed_matching='class-independent one-to-one IoU>=.5, same pairs across all routes, unknown ignored; unmatched geometry excluded from semantic confusion',
        low_confidence='baseline instance probability abs(P(F)-.5)<.1; predetermined',
        manifest_sha256=file_sha(deploy/'manifest.json'), totals=totals,
        source_sha256={name:file_sha(ROOT/name) for name in source_names},
        elapsed_seconds=time.time()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2)
    write(out/'report.json', report)
    write(out/'status.json', dict(status='complete', completed=len(cases), training=False))
    page = ['<!doctype html><meta charset="utf-8"><title>固定实例语义诊断</title>',
        '<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:100%;background:white}section{margin:28px 0}</style>',
        '<h1>固定新最佳balance e20实例图，仅比较语义</h1>',
        '<p>全图原图/S-align、D5a/S-align、D5a/私有LoRA、D5a/完整头e60。金=铁素体，蓝=珠光体。</p>',
        '<p>测试图没有GT；四张已见GT仅原标域、配对集合固定，不能代表泛化。</p>',
        '<p><a href="report.json">回执与汇总</a> · <a href="rows.json">逐图诊断</a></p>']
    for path in sorted(gallery.glob('*_detail.png')):
        rel = html.escape(path.relative_to(out).as_posix())
        page.append(f'<section><h2>{html.escape(path.stem)}</h2><img src="{rel}"></section>')
    (out/'index.html').write_text('\n'.join(page), encoding='utf-8')
    with zipfile.ZipFile(out/'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in [*out.glob('*.json'), out/'index.html', *gallery.glob('*.png'), *out.glob('predictions/*/*/instances.json')]:
            archive.write(path, path.relative_to(out).as_posix())
    print('SEMANTIC FIXED COMPLETE '+json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_balance_native.yaml')
    parser.add_argument('--output', default='outputs/semantic_fixed')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        from utils.config import load_config, project_path
        out = Path(project_path(load_config(args.config), args.output))
        if out.is_dir() and not isinstance(error, FileExistsError):
            write(out/'status.json', dict(status='failed', training=False, error=repr(error)))
        raise
