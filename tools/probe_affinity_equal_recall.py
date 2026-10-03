# -*- coding: utf-8 -*-
"""已见人工源的等真分界召回诊断；不训练、不改部署阈值、不生成测试标签。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path
from utils.offset_letterbox import letterbox_instance_geometry

LAYERS = ('relations_all', 'relations_short', 'relations_distance2',
          'relations_distance4', 'fused_boundary', 'fused_deep_interior')


def select_checkpoints(config, args, status_reader=None):
    """默认旧native／balance不变；覆盖时两路径与两SHA必须同时明确提供。"""
    names = ('control_checkpoint', 'control_sha256', 'balance_checkpoint', 'balance_sha256')
    supplied = {name: getattr(args, name, None) for name in names}
    if any(value is not None for value in supplied.values()):
        if not all(isinstance(value, str) and value.strip() for value in supplied.values()):
            raise ValueError('Explicit comparison requires both checkpoint paths and both SHA256 values')
        paths = {role: supplied[role+'_checkpoint'] for role in ('control', 'balance')}
        expected = {role: supplied[role+'_sha256'] for role in paths}
        selection = 'explicit_pair'
    else:
        paths = {'control': config['affinity_native']['penalty_reference'] + '/final.pt',
                 'balance': config['backend_adaptation']['output_dir'] + '/final.pt'}
        if status_reader is None:
            status_reader = lambda path: json.loads(Path(path).read_text(encoding='utf8'))
        status = status_reader(project_path(config, config['backend_adaptation']['output_dir'], 'status.json'))
        expected = {'control': config['affinity_native']['penalty_sha256'],
                    'balance': status['final_checkpoint_sha256']}
        selection = 'default_old_native_vs_native_balance'
    if any(re.fullmatch('[0-9a-fA-F]{64}', value) is None for value in expected.values()):
        raise ValueError('Each expected SHA256 must contain exactly 64 hexadecimal characters')
    return paths, {role: value.lower() for role, value in expected.items()}, selection


def pair_targets(ids, trusted, grid=512):
    """两个端点均在原标可信域；补缝像素、零ID、窗外及padding均忽略。"""
    labels = np.where(np.asarray(trusted, bool), np.asarray(ids), 0)
    mapped, valid, metadata = letterbox_instance_geometry(labels, 1024, grid)
    target, edge_valid = build_affinity_targets_torch(
        torch.from_numpy(mapped.astype(np.int64))[None],
        torch.from_numpy(valid)[None, None])
    return target[0].bool().numpy(), edge_valid[0].numpy(), metadata.to_dict()


def fused_targets(target, valid, strict_interior=False):
    """融合像素须有全部四个可信短距关系；任一跨实例为边界，全同实例为内部。"""
    boundary = (~target[:4]).any(axis=0)
    trustworthy = valid[:4].all(axis=0)
    if strict_interior:
        trustworthy = (boundary & trustworthy) | (target.all(axis=0) & valid.all(axis=0))
    return boundary, trustworthy


def score_histogram(scores, boundary, valid, bins=8192):
    scores = np.asarray(scores)
    boundary, valid = np.asarray(boundary, bool), np.asarray(valid, bool)
    if scores.shape != boundary.shape or scores.shape != valid.shape:
        raise ValueError('Scores/targets/valid shapes differ')
    if not np.isfinite(scores[valid]).all() or np.any(scores[valid] < 0) or np.any(scores[valid] > 1):
        raise ValueError('Boundary scores must be finite probabilities')
    index = np.minimum((scores[valid] * bins).astype(np.int64), bins - 1)
    truth = boundary[valid]
    return np.stack([np.bincount(index[truth], minlength=bins),
                     np.bincount(index[~truth], minlength=bins)]).astype(np.int64)


def operating_curve(hist):
    """阈值由1向0下降；8192格量化，分界保留与内部误断都按原始计数累计。"""
    hist = np.asarray(hist, np.int64)
    if hist.ndim != 2 or hist.shape[0] != 2 or np.any(hist < 0):
        raise ValueError('Expected nonnegative [2,bins] histogram')
    positive, negative = map(int, hist.sum(axis=1))
    tp = np.r_[0, np.cumsum(hist[0, ::-1])]
    fp = np.r_[0, np.cumsum(hist[1, ::-1])]
    recall = tp / max(positive, 1)
    false_boundary = fp / max(negative, 1)
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float), where=(tp + fp) > 0)
    thresholds = np.arange(hist.shape[1], -1, -1) / hist.shape[1]
    return dict(threshold=thresholds, recall=recall, false_boundary=false_boundary,
                precision=precision, true_boundaries=positive, interior_relations=negative)


def equal_recall(hist, requested):
    curve = operating_curve(hist)
    if not curve['true_boundaries'] or not curve['interior_relations']:
        return None
    index = int(np.searchsorted(curve['recall'], float(requested), side='left'))
    return dict(requested_true_boundary_recall=float(requested),
                attained_true_boundary_recall=float(curve['recall'][index]),
                interior_false_boundary_rate=float(curve['false_boundary'][index]),
                diagnostic_threshold=float(curve['threshold'][index]),
                true_boundaries=curve['true_boundaries'], interior_relations=curve['interior_relations'],
                quantization_width=1 / hist.shape[1])


def curve_metrics(hist):
    curve = operating_curve(hist)
    if not curve['true_boundaries'] or not curve['interior_relations']:
        return dict(roc_auc=None, boundary_average_precision=None)
    auc = np.sum(np.diff(curve['false_boundary']) * (curve['recall'][:-1] + curve['recall'][1:]) / 2)
    ap = np.sum(np.diff(curve['recall']) * curve['precision'][1:])
    midpoint = hist.shape[1] // 2
    retained = hist[0, midpoint:].sum() / curve['true_boundaries']
    false = hist[1, midpoint:].sum() / curve['interior_relations']
    return dict(roc_auc=float(auc), boundary_average_precision=float(ap),
                threshold_0_5=dict(true_boundary_recall=float(retained), interior_false_boundary_rate=float(false)))


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def make_cases(config, subset, seed):
    from data.backend_adaptation import CanonicalBackendDataset
    from data.affinity_native import choose_box
    from data.rgb_restoration_dataset import read_rgb
    from tools.probe_affinity_patch import load_gt
    dataset = CanonicalBackendDataset(project_path(config, config['paths']['raw_data_dir']),
                                      project_path(config, config['backend_adaptation']['completed_gt_dir']))
    if len(dataset) != 32:
        raise RuntimeError('Expected all original 32 manual sources')
    previous = json.loads(Path(project_path(config, config['affinity_native']['prior_probe'], 'cases.json')).read_text(encoding='utf8'))
    names = {c['name'] for c in previous if c['kind'] == 'train'}
    result = []
    for index, path in enumerate(dataset.samples):
        if subset == 'four' and path.stem not in names:
            continue
        case = dict(kind='train', name=path.stem, path=str(path), source_index=index,
                    prior_four=path.stem in names)
        ids, trusted, _ = load_gt(case, config)
        image = read_rgb(str(path))
        if image.shape[:2] != ids.shape or trusted.shape != ids.shape:
            raise RuntimeError('Original GT/image shape differs')
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 90]))
        box, attempt = choose_box(np.where(trusted, ids, 0), 1024, rng,
                                   attempts=32, minimum_pairs=16)
        y, x, y1, x1 = box
        case.update(box=list(box), crop_attempt=attempt, crop_shape=[y1-y, x1-x],
                    original_shape=list(ids.shape), original_sha256=digest(image),
                    gt_sha256=digest(ids), trusted_sha256=digest(trusted))
        result.append(case)
    if len(result) != (32 if subset == 'all' else 4):
        raise RuntimeError('Case cohort incomplete')
    return result


def prepared_view(case, profile, config, seed):
    from data.dataset import letterbox
    from data.rgb_restoration_dataset import read_rgb, degrade_rgb, _add_achromatic_noise
    from data.rgb_spatial_blur import make_endpoint_sigma_map
    from tools.probe_affinity_patch import load_gt
    raw = read_rgb(case['path'])
    ids, trusted, _ = load_gt(case, config)
    y, x, y1, x1 = case['box']
    image, _, ph, pw = letterbox(raw[y:y1, x:x1], 1024)
    vh, vw = 1024-ph, 1024-pw
    clean = image.astype(np.float32) / 255
    degradation = load_config(project_path(config, config['backend_adaptation']['restoration']['degradation_config']))['rgb_restoration']['degradation']
    stream = lambda n: np.random.default_rng(np.random.SeedSequence([seed, case['source_index'], n]))
    rng, spatial_rng = stream(0), stream(6)
    saved = deepcopy(rng.bit_generator.state)
    info = {}
    condition = degrade_rgb(clean[:vh, :vw], rng, 'identity' if profile == 'clean' else 'blur',
                            degradation, spatial_rng=spatial_rng, spatial_info=info)
    endpoint = degradation.get('spatial_blur', {}).get('endpoint_transition', {})
    applied, noise_sigma = False, 0.0
    if profile == 'blur' and info['selected'] and endpoint.get('enabled', False):
        endpoint_rng = stream(7)
        applied = bool(endpoint_rng.random() < endpoint.get('probability', .5))
        if applied:
            sigma_map = make_endpoint_sigma_map((vh, vw), (0, 0, vh, vw), endpoint,
                                                endpoint_rng, degradation['blur_sigma'])
            replay = np.random.default_rng(0)
            replay.bit_generator.state = saved
            condition = degrade_rgb(clean[:vh, :vw], replay, 'blur', degradation,
                                    spatial_rng=stream(6), spatial_info=info, sigma_map_override=sigma_map)
    noise = config['backend_adaptation']['noise']
    if profile == 'blur' and noise.get('enabled', True):
        noise_rng = stream(3)
        if noise_rng.random() < noise.get('probability', .5):
            noise_sigma = float(noise_rng.uniform(*noise.get('sigma', [0, .006])))
            condition = _add_achromatic_noise(condition, np.ones((vh, vw), np.float32), noise_rng, noise_sigma)
    condition = cv2.copyMakeBorder(condition, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    target, valid, metadata = pair_targets(ids[y:y1, x:x1], trusted[y:y1, x:x1])
    receipt = dict(source=case['name'], profile=profile, input_sha256=digest(condition),
                   target_sha256=digest(target), valid_sha256=digest(valid), layout='contiguous NCHW',
                   restoration_seed=seed + case['source_index'] * 100 + (profile == 'blur'),
                   spatial=bool(info['selected']), endpoint=applied, base_sigma=float(info['base_sigma']),
                   sigma_min=float(info['sigma_map'].min()) if info['sigma_map'] is not None else None,
                   sigma_max=float(info['sigma_map'].max()) if info['sigma_map'] is not None else None,
                   noise_sigma=noise_sigma, geometry=metadata)
    return condition, target, valid, receipt


def summarize(records, recalls):
    pooled, comparisons = {}, []
    for profile in ('clean', 'blur'):
        pooled[profile] = {}
        for layer in LAYERS:
            pair = {}
            source_maps = {}
            for model in ('control', 'balance'):
                chosen = [r for r in records if r['profile'] == profile and r['model'] == model]
                hist = sum((r['histograms'][layer] for r in chosen), np.zeros_like(chosen[0]['histograms'][layer]))
                pair[model] = dict(**curve_metrics(hist), equal_recall=[equal_recall(hist, v) for v in recalls])
                source_maps[model] = {r['source']: r['histograms'][layer] for r in chosen}
            pooled[profile][layer] = pair
            for requested in recalls:
                source_rows = []
                for source in source_maps['control']:
                    a, b = (equal_recall(source_maps[m][source], requested) for m in ('control', 'balance'))
                    if a is None or b is None:
                        continue
                    source_rows.append(dict(source=source, control=a, balance=b,
                                            false_boundary_delta=b['interior_false_boundary_rate'] - a['interior_false_boundary_rate']))
                deltas = np.array([r['false_boundary_delta'] for r in source_rows])
                comparisons.append(dict(profile=profile, layer=layer, requested_recall=requested,
                                        paired_sources=len(source_rows), fewer_false_boundaries=int((deltas < 0).sum()),
                                        more_false_boundaries=int((deltas > 0).sum()),
                                        median_delta=float(np.median(deltas)) if len(deltas) else None,
                                        mean_delta=float(deltas.mean()) if len(deltas) else None,
                                        sources=source_rows))
    return pooled, comparisons


def plot(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, profile in enumerate(('clean', 'blur')):
        for col, layer in enumerate(('relations_all', 'relations_short', 'fused_deep_interior')):
            ax = axes[row, col]
            for model, color in (('control', '#2878b5'), ('balance', '#d95319')):
                hist = sum(r['histograms'][layer] for r in records if r['model'] == model and r['profile'] == profile)
                curve = operating_curve(hist)
                ax.plot(curve['recall'], curve['false_boundary'], label=model, color=color)
            ax.set(xlim=(.85, 1), ylim=(0, .35), xlabel='True boundary recall',
                   ylabel='Interior false boundary rate', title=f'{profile}: {layer}')
            ax.grid(alpha=.25); ax.legend()
    figure.suptitle('SEEN training sources only; thresholds are diagnostic, not deployed')
    figure.tight_layout(); figure.savefig(out/'equal_recall.png', dpi=150); plt.close(figure)


def run(args):
    from models.backend_adaptation import load_backend, restore_first
    from train_backend_adaptation import sha, tensor_digest, fusion_options, write_json
    from train_affinity_connectivity import get_restorer
    from utils.affinity_fusion import affinity_boundary_probability
    config = load_config(args.config)
    if args.bins < 128 or args.bins > 65536:
        raise ValueError('Use 128..65536 bins')
    if not torch.cuda.is_available():
        raise RuntimeError('Use sam2_env on the GPU server')
    out = Path(project_path(config, args.output)); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    cases = make_cases(config, args.sources, args.seed)
    checkpoint_paths, expected, checkpoint_selection = select_checkpoints(config, args)
    write_json(out/'cases.json', cases); write_json(out/'config.json', config)
    write_json(out/'status.json', dict(status='running', training=False))
    restorer = get_restorer(config, 'cuda')
    rdigest = tensor_digest(restorer.state_dict().items())
    records, receipts, checkpoints = [], {}, {}
    shared_digest = None
    started = time.time()
    mode, kwargs = fusion_options(config)
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        for model_name, path in checkpoint_paths.items():
            path = project_path(config, path)
            if sha(path) != expected[model_name]:
                raise RuntimeError('Checkpoint hash mismatch: ' + model_name)
            model, bundle = load_backend(path, config, 'cuda')
            model.eval().requires_grad_(False)
            before = tensor_digest(model.state_dict().items())
            shared = tensor_digest((k, v) for k, v in model.state_dict().items() if not k.startswith('affinity_decoder.'))
            if shared_digest is not None and shared != shared_digest:
                raise RuntimeError('Encoder/LoRA/semantic states differ between models')
            shared_digest = shared
            if any(p.requires_grad for p in model.parameters()) or any(p.requires_grad for p in restorer.parameters()):
                raise RuntimeError('Diagnostic models must be fully frozen')
            checkpoints[model_name] = dict(path=checkpoint_paths[model_name], sha256=sha(path),
                                          epoch=int(bundle['epoch']), frozen_exact=False, shared_state_sha256=shared,
                                          training_output_dir=bundle.get('config', {}).get('backend_adaptation', {}).get('output_dir'),
                                          training_sampling=bundle.get('config', {}).get('affinity_native', {}).get('penalty_sampling'))
            for case in cases:
                for profile in ('clean', 'blur'):
                    image, target, valid, receipt = prepared_view(case, profile, config, args.seed)
                    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()[None].to('cuda')
                    restored = restore_first(restorer, tensor, receipt['restoration_seed'])
                    receipt['restored_sha256'] = tensor_digest([('image', restored)])
                    key = case['name'] + '/' + profile
                    if key in receipts and receipts[key] != receipt:
                        raise RuntimeError('Model inputs or supervision differ: ' + key)
                    receipts[key] = receipt
                    logits = model.affinity_decoder(model.encoder(restored))['affinity_logits'].float()
                    if tuple(logits.shape) != (1, 8, 512, 512):
                        raise RuntimeError('Expected original 512-grid affinity decoder')
                    scores = (1 - logits.sigmoid())[0].cpu().numpy()
                    fused = affinity_boundary_probability(logits, mode=mode, **kwargs)[0, 0].cpu().numpy()
                    histogram = {}
                    for layer, selection in [('relations_all', slice(None)), ('relations_short', slice(0, 4)),
                                             ('relations_distance2', slice(4, 6)), ('relations_distance4', slice(6, 8))]:
                        histogram[layer] = score_histogram(scores[selection], ~target[selection], valid[selection], args.bins)
                    boundary, eligible = fused_targets(target, valid)
                    histogram['fused_boundary'] = score_histogram(fused, boundary, eligible, args.bins)
                    boundary, eligible = fused_targets(target, valid, strict_interior=True)
                    histogram['fused_deep_interior'] = score_histogram(fused, boundary, eligible, args.bins)
                    records.append(dict(model=model_name, source=case['name'], profile=profile, histograms=histogram))
                    if case['prior_four']:
                        panels = []
                        for value in (image, restored[0].permute(1, 2, 0).cpu().numpy()):
                            panels.append(cv2.cvtColor(np.rint(np.clip(cv2.resize(value, (320, 320)), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        heat = cv2.applyColorMap(np.rint(cv2.resize(fused, (320, 320))*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                        panels.append(heat)
                        cv2.imwrite(str(out/(case['name']+'_'+profile+'_'+model_name+'.png')), np.concatenate(panels, axis=1))
                    print(json.dumps(dict(model=model_name, source=case['name'], profile=profile,
                                          views_completed=len(records))), flush=True)
            if tensor_digest(model.state_dict().items()) != before:
                raise RuntimeError('Backend state changed during diagnostic')
            checkpoints[model_name]['frozen_exact'] = True
            del model, bundle, logits, scores, fused, tensor, restored
            gc.collect(); torch.cuda.empty_cache()
    if tensor_digest(restorer.state_dict().items()) != rdigest:
        raise RuntimeError('D5a state changed during diagnostic')
    pooled, paired = summarize(records, args.recalls)
    np.savez_compressed(out/'histograms.npz', **{r['model']+'/'+r['source']+'/'+r['profile']+'/'+layer: histogram
                                             for r in records for layer, histogram in r['histograms'].items()})
    report = dict(complete=True, training=False, scope='SEEN original 32 human training sources; mechanism diagnostic only, not held-out/official accuracy',
                  source_count=len(cases), input_views=len(cases)*2, model_views=len(records), seed=args.seed, bins=args.bins,
                  layers=dict(relations='Offset relation true boundary = different positive instance IDs, endpoints both originally covered',
                              fused_boundary='512-grid fusion width-sensitive layer: all four short endpoints originally covered, any different-instance short relation = boundary',
                              fused_deep_interior='Same trustworthy short boundary positives; strict negatives require all eight offsets originally covered and same instance, near-interface long-offset responses excluded'),
                  caveats=['Fusion-layer pixel labels are a mechanism surrogate, not a final-instance metric.',
                           'Incomplete/incorrect original labels can bias this seen-source diagnostic.',
                           'Histogram thresholds are diagnostic only and must not be deployed or selected on test images.'],
                  comparison_selection=checkpoint_selection,
                  model_aliases='control/balance are comparison slots; actual identities are recorded in checkpoints',
                  checkpoints=checkpoints, d5a=config['backend_adaptation']['restoration'], d5a_frozen_exact=True,
                  fusion=dict(mode=mode, **kwargs), layout='contiguous NCHW, FP32, no autocast',
                  pooled=pooled, source_paired=paired, receipts=list(receipts.values()),
                  elapsed_seconds=time.time()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,
                  source_sha256=sha(__file__))
    write_json(out/'report.json', report); plot(records, out)
    write_json(out/'status.json', dict(status='complete', training=False))
    print('COMPLETE', json.dumps(dict(sources=len(cases), views=len(records), seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train/affinity_balance_native.yaml')
    parser.add_argument('--output', default='outputs/equal_recall')
    parser.add_argument('--sources', choices=('all', 'four'), default='all')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--bins', type=int, default=8192)
    parser.add_argument('--recalls', nargs='+', type=float, default=[.90, .95, .97, .98])
    parser.add_argument('--control-checkpoint', help='Optional first checkpoint; requires all four override arguments')
    parser.add_argument('--control-sha256', help='Expected first checkpoint SHA256')
    parser.add_argument('--balance-checkpoint', help='Optional second checkpoint; requires all four override arguments')
    parser.add_argument('--balance-sha256', help='Expected second checkpoint SHA256')
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.recalls):
        parser.error('Recalls must be in (0,1]')
    run(args)
