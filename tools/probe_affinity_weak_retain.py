# -*- coding: utf-8 -*-
"""冻结真实特征，比较弱界局部修正与可靠旧关系保留；各32步。

具体源图、界面、坐标和随机种子只读取被忽略的既有捕获回执。
两组都用完整新GT＋局部弱界监督，唯一差别为保留项系数0／1。
结果是已见训练图的机制诊断，不是验证集或提交候选。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_affinity_partition import correct_local_relations, decode_capture
from tools.probe_affinity_weak_path import array_sha, verify_protected
from train_backend_adaptation import fusion_options, tensor_digest
from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS, _edge_slices
from utils.affinity_loss import balanced_affinity_loss, build_affinity_targets_torch
from utils.affinity_fusion import affinity_boundary_probability
from utils.offset_letterbox import letterbox_instance_geometry
from utils.patch_diagnostic import BlendMap

from utils.affinity_retain_loss import (build_reliable_retention_mask,
    balanced_affinity_retention_kl)

FULL_WEIGHT = LOCAL_WEIGHT = RETAIN_WEIGHT = 1.0
CONFIDENCE = 0.9
GRADIENT_STEPS = (1, 16, 32)
STEPS = 32
SNAPSHOTS = (0, 16, 32)
LR = 2.e-5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fixed_local_mask(grid_gt, valid, pair, radius=16):
    """固定合法负界面＋两侧同GT正连接；范围与模型预测无关。"""
    gt = np.asarray(grid_gt)
    content = np.asarray(valid)
    if (gt.ndim != 2 or not np.issubdtype(gt.dtype, np.integer) or np.any(gt < 0)
            or content.shape != gt.shape or content.dtype != np.bool_):
        raise ValueError('Expected integer nonnegative HW GT and boolean matching valid_content')
    if (len(pair) != 2 or pair[0] == pair[1] or min(pair) <= 0
            or isinstance(radius, bool) or not isinstance(radius, int) or radius < 1):
        raise ValueError('Two distinct positive GT IDs and positive integer radius required')
    dummy = torch.zeros((1, 8, *gt.shape), dtype=torch.float32)
    _, negatives, _ = correct_local_relations(dummy, gt, content,
        dict(kind='merge', gt_ids=list(pair)), channels='short')
    interface = np.zeros(gt.shape, bool)
    height, width = gt.shape
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        source, dest = _edge_slices(height, width, dy, dx)
        chosen = negatives[0, channel].numpy()[source]
        interface[source] |= chosen
        interface[dest] |= chosen
    band = cv2.dilate(interface.astype(np.uint8), np.ones((radius*2+1, radius*2+1), np.uint8)).astype(bool)
    targets, legal = build_affinity_targets_torch(torch.from_numpy(gt.astype(np.int64))[None],
                                                torch.from_numpy(content)[None, None])
    result = negatives.clone()
    for channel, (dy, dx) in enumerate(DEFAULT_AFFINITY_OFFSETS[:4]):
        source, dest = _edge_slices(height, width, dy, dx)
        positive = (legal[0, channel].numpy()[source] & (targets[0, channel].numpy()[source] > .5)
                    & np.isin(gt[source], pair) & band[source] & band[dest])
        result[0, channel][source] |= torch.from_numpy(positive)
    if torch.any(result & ~legal) or result[:, 4:].any():
        raise RuntimeError('Fixed local mask contains unknown/padding or long edges')
    return result, targets, legal, band


def supervision_record(target, mask):
    records = []
    for c in range(target.shape[1]):
        pos = int((mask[:, c] & (target[:, c] > .5)).sum())
        neg = int((mask[:, c] & (target[:, c] <= .5)).sum())
        records.append(dict(channel=c, positive_edges=pos, negative_edges=neg,
                            positive_fraction=pos/(pos+neg) if pos+neg else None,
                            class_terms=int(pos > 0)+int(neg > 0)))
    return dict(per_channel=records, positive_edges=sum(r['positive_edges'] for r in records),
        negative_edges=sum(r['negative_edges'] for r in records),
        active_channels=sum(bool(r['positive_edges']+r['negative_edges']) for r in records),
        normalization='per-channel class-weighted means, then mean over active channels',
        caveat='Different edge counts, class ratios and per-edge gradient budgets; not equal-gradient-budget ablation')


def loss_options(config):
    cfg = config['direct_semantic_affinity']['affinity_loss']
    if cfg.get('manual_uncovered_as_boundary', True):
        raise RuntimeError('This diagnostic requires processed GT unknown as ignore')
    return dict(negative_weight=float(cfg['negative_weight']),
                hard_negative_weight=float(cfg['hard_negative_weight']),
                hard_negative_gamma=float(cfg['hard_negative_gamma']),
                normalize_edge_weights=bool(cfg['normalize_edge_weights']))


def parameter_change(head, initial):
    square, maximum, changed = 0., 0., 0
    for name, tensor in head.state_dict().items():
        delta = tensor.detach().float().cpu() - initial[name].float().cpu()
        square += float(delta.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        changed += int(torch.count_nonzero(delta))
    return dict(l2=square**.5, maximum_absolute=maximum, changed_values=changed)


def relation_fit_statistics(logits, target, selected):
    """固定同一局部关系报告拟合，不以变化后的预测选择样本。"""
    import torch.nn.functional as F
    raw = F.binary_cross_entropy_with_logits(logits.detach(), target, reduction='none')
    probability = logits.detach().sigmoid()
    groups = {}
    for name, mask in (('negative_interface', selected & (target <= .5)),
                       ('positive_nearby', selected & (target > .5))):
        values = probability[mask].cpu().numpy()
        groups[name] = dict(edges=int(mask.sum()), raw_bce_mean=float(raw[mask].mean()) if mask.any() else None,
            same_instance_probability_quantiles=np.quantile(values, [0., .1, .5, .9, 1.]).tolist() if values.size else None,
            disconnection_above_half=int((values < .5).sum()), same_instance_above_half=int((values > .5).sum()))
    return groups


def save_thumbnail(path, rgb, boundary, instances, gt, pair, title):
    region = np.isin(gt, pair)
    yy, xx = np.nonzero(region)
    y0, y1 = max(0, int(yy.min())-60), min(gt.shape[0], int(yy.max())+61)
    x0, x1 = max(0, int(xx.min())-60), min(gt.shape[1], int(xx.max())+61)
    pieces = [cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
              cv2.applyColorMap(np.rint(boundary[y0:y1, x0:x1]*255).astype(np.uint8), cv2.COLORMAP_INFERNO)]
    crop = instances[y0:y1, x0:x1]
    rng = np.random.default_rng(1882)
    colors = rng.integers(50, 240, (int(instances.max())+1, 3), dtype=np.uint8)
    colors[0] = 0
    pieces.append(colors[crop])
    panels = []
    for piece in pieces:
        h, w = piece.shape[:2]
        piece = cv2.resize(piece, (360, max(1, round(h*360/w))), interpolation=cv2.INTER_AREA)
        panels.append(piece)
    strip = np.concatenate(panels, 1)
    strip = cv2.copyMakeBorder(strip, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.putText(strip, title, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,0,0), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), strip):
        raise RuntimeError('Cannot save thumbnail')


def component_gradient_record(terms, head):
    """记录真实参数梯度，不用初态零KL推定保留项力度。"""
    parameters = tuple(head.parameters())
    grads = {}
    for name, loss in terms.items():
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        grads[name] = [torch.zeros_like(p) if g is None else g.detach()
                       for p, g in zip(parameters, values)]
    norms = {name: sum(float(g.float().square().sum()) for g in values)**.5
             for name, values in grads.items()}
    fit = [FULL_WEIGHT*a + LOCAL_WEIGHT*b for a, b in zip(grads['full'], grads['local'])]
    fit_norm = sum(float(g.float().square().sum()) for g in fit)**.5
    dot = sum(float((a.float()*b.float()).sum()) for a, b in zip(fit, grads['retention']))
    denominator = fit_norm*norms['retention']
    return dict(unweighted_parameter_gradient_norms=norms,
                full_local_gradient_norm=fit_norm,
                full_local_retention_cosine=dot/denominator if denominator > 0 else None)


def scalar_record(value):
    """损失统计可包含标量Tensor；回执不保存图或GPU对象。"""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError('Loss receipt tensors must be scalar')
        return float(value.detach())
    if isinstance(value, dict):
        return {key: scalar_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_record(item) for item in value]
    return value


def prior_failure_guard(prior, baseline):
    """从已见图私有回执确定需保护的F-P对象，不内嵌真实逐图ID。"""
    original = {r['gt']: r for r in baseline['complete_ferrite_diagnostic']['gt_rows']}
    rows = prior['arms']['local']['full_ferrite']['gt_rows']
    candidates = [(r['F_pixels_predicted_P'] - original[r['gt']]['F_pixels_predicted_P'], r)
                  for r in rows if r['phase'] == 1]
    delta, worst = max(candidates, key=lambda item: item[0])
    if delta <= 0:
        raise RuntimeError('Prior local arm has no diagnosed F-to-P loss')
    pieces = [r for r in worst['contributions'] if r['predicted_phase'] == 0]
    other = int(max(pieces, key=lambda r:r['pixels'])['dominant_gt'])
    if other == int(worst['gt']):
        raise RuntimeError('Prior F-to-P failure is not the foreign P merge')
    return dict(gt_ids=[int(worst['gt']), other], delta_F_to_P_pixels=int(delta))


def run(args):
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend, restore_first
    from tools.affinity_connectivity_views import _inference_state
    from tools.semantic_crossover import revote_instances
    from tools.run_affinity_short import precision_record
    from tools.run_affinity_sweep import runtime_contract
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core, reference_config

    capture_root, output = Path(args.capture).resolve(), Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'outputs/weak_retain') or output.exists():
        raise RuntimeError('Fresh private outputs/weak_retain output required')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual sam2_env CUDA required')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    receipt = read(capture_root/'capture.json')
    if (receipt.get('complete') is not True or receipt.get('protected_before_after_exact') is not True
            or receipt.get('format') != 'affinity_weak_capture_v1'):
        raise RuntimeError('Completed original49 fixed capture required')
    protected = dict(receipt['protected_sha256'])
    capture_source_relocation = None
    prior_entry = ROOT/'tools/probe_affinity_weak_path.py'
    old_sha = '6abcc4cf12a0afad7828941e4699685ddb2cc1ce0412817adb317260311a9cb8'
    current_sha = '85aadec3a48dec33f704f81b2c6b82fe5ca25be0e850eec415507f9f359abd0d'
    if protected.get(str(prior_entry)) == old_sha:
        archived = ROOT/'outputs/affinity_weak/capture_source.py'
        if sha(archived) != old_sha or sha(prior_entry) != current_sha:
            raise RuntimeError('Known capture-source privacy relocation does not match sealed originals')
        protected.pop(str(prior_entry))
        protected[str(archived)] = old_sha
        protected[str(prior_entry)] = current_sha
        capture_source_relocation = dict(original_sha256=old_sha, archived_path=str(archived),
            current_sha256=current_sha, reason='After original GPU capture, private per-image coordinates moved into receipt')
    for name, digest in read(capture_root/'capture_manifest.json')['files'].items():
        protected[str(capture_root/name)] = digest
    protected[str(Path(__file__).resolve())] = sha(__file__)
    prior_root = Path(args.prior).resolve()
    for manifest in (prior_root/'manifest.json', prior_root/'analysis/manifest.json'):
        metadata = read(manifest)
        if metadata.get('complete') is not True:
            raise RuntimeError('Complete previous weak-fit manifest required')
        for name, digest in metadata['files'].items():
            protected[str(manifest.parent/name)] = digest
        protected[str(manifest)] = sha(manifest)
    for entry in ('utils/affinity_retain_loss.py', 'tools/probe_affinity_weak_fit.py'):
        protected[str(ROOT/entry)] = sha(ROOT/entry)
    verify_protected(protected)
    prior = read(prior_root/'analysis/report.json')
    baseline_path = capture_root/'analysis/report.json'
    if (sha(baseline_path) != 'b8f687d2d3dd8f3dd42e9fba6d69175d953d0ffbc6d9996eff50955b4dc38031'
            or sha(prior_root/'analysis/report.json') != '2fff1191e8136a4a35a8ca4469c2504c7e5a1107f9e50f25e73802b874dcb6bb'):
        raise RuntimeError('Previously sealed complete baseline/fit analysis changed')
    protected[str(baseline_path)] = sha(baseline_path)
    baseline = read(baseline_path)['arms']['r1']['baseline']
    guard = prior_failure_guard(prior, baseline)
    coverage_config = receipt['config']; config = reference_config(coverage_config)
    record = receipt['arms']['r1']; source = receipt['source']
    # Source-specific pair identities are derived from existing private crossing coordinates.
    rgb = read_rgb(receipt['source_path'])
    gt, lookup = load_completed_semantic_source(receipt['gt_path'], rgb.shape[:2])
    pair = [int(gt[tuple(point)]) for point in receipt['crossing']]
    if pair[0] == pair[1] or any(g <= 0 for g in pair):
        raise RuntimeError('Private receipt crossing does not identify two known GT instances')
    points = receipt['points']; core = load_original_core(coverage_config); views = core.coverage_original_views
    options = loss_options(config)
    if any(g <= 0 or g >= len(lookup) or lookup[g] != c
           for g,c in zip(guard['gt_ids'], (1,0))):
        raise RuntimeError('Prior F-P guard phase identity differs from completed newGT')
    guard['points'] = []
    for g in guard['gt_ids']:
        distance = cv2.distanceTransform((gt == g).astype(np.uint8), cv2.DIST_L2, 5)
        point = np.unravel_index(int(distance.argmax()), gt.shape)
        guard['points'].append([int(v) for v in point])
    runtime, precision = runtime_contract(config), precision_record(torch)
    model = load_backend(record['checkpoint'], config, 'cuda')[0].eval().requires_grad_(False)
    restorer = get_restorer(config, 'cuda').eval().requires_grad_(False)
    frozen_model = tensor_digest(model.state_dict().items())
    frozen_restorer = tensor_digest(restorer.state_dict().items())
    if frozen_model != record['model_state_sha256'] or frozen_restorer != receipt['restorer_state_sha256']:
        raise RuntimeError('Actual R1/D5a states differ from sealed capture')
    mode, fusion = fusion_options(config)
    with np.load(capture_root/'r1/maps.npz', allow_pickle=False) as maps:
        semantic = torch.from_numpy(maps['semantic'].copy())
        probability = maps['raw_probability'].copy()
        expected_boundary = maps['baseline'].copy()
        expected_markers = maps['markers'].copy()
        expected_watershed = maps['watershed'].copy()
    expected_ids = cv2.imread(str(capture_root/'r1/baseline_inst.png'), cv2.IMREAD_UNCHANGED)
    expected_classes = read(capture_root/'r1/baseline_class.json')
    if dict(geometry_semantic_sha256=array_sha(semantic.numpy()),
            raw_probability_sha256=array_sha(probability)) != record['fixed_whole_semantics']:
        raise RuntimeError('Fixed whole semantic/raw probability identity differs')
    output.mkdir(parents=True)
    caches, masks, relevant = [], {}, []
    with _inference_state(model, restorer, 'cuda'):
        for tile in record['tiles']:
            y, x, y1, x1 = tile['box']; index = tile['index']
            image, ph, pw = views.tensor_from_rgb(rgb[y:y1, x:x1], 'cuda')
            restored = restore_first(restorer, image, tile['seed'])
            features = [value.detach() for value in model.encoder(restored)]
            logits = model.affinity_decoder(features)['affinity_logits'].float()
            grid = affinity_boundary_probability(logits, mode=mode, **fusion)
            native = views.crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x))
            with np.load(capture_root/tile['map_file'], allow_pickle=False) as old:
                if (not np.array_equal(grid.cpu().numpy(), old['grid'])
                        or not np.array_equal(native.cpu().numpy(), old['native'])
                        or tuple(logits.stride()) != tuple(tile['logits_stride'])):
                    raise RuntimeError('Fresh actual feature replay differs at tile '+str(index))
                if tile['relevant'] and not np.array_equal(logits.cpu().numpy(), old['logits']):
                    raise RuntimeError('Fresh relevant logits differ from sealed capture')
            grid_gt, valid, metadata = letterbox_instance_geometry(gt[y:y1,x:x1], 1024, 512)
            local, target, legal, band = fixed_local_mask(grid_gt, valid, pair)
            if tile['relevant'] != bool(local.any()):
                raise RuntimeError('Fixed training interface presence differs from capture')
            if tile['relevant']:
                relevant.append(index)
                gpu_target, gpu_legal = target.to('cuda'), legal.to('cuda')
                teacher = logits.detach().clone()
                retention, retention_counts = build_reliable_retention_mask(
                    teacher, gpu_target, gpu_legal, band, confidence=CONFIDENCE,
                    instance_map=grid_gt, valid_content=valid)
                _, guard_mask, _ = correct_local_relations(torch.zeros_like(logits).cpu(),
                    grid_gt, valid, dict(kind='merge', gt_ids=guard['gt_ids']), channels='all')
                guard_mask = guard_mask.to('cuda')
                guard_counts = dict(per_channel=[dict(channel=c,
                    legal_interface_edges=int(guard_mask[:,c].sum()),
                    retained_interface_edges=int((guard_mask[:,c]&retention[:,c]).sum()))
                    for c in range(logits.shape[1])])
                masks[index] = dict(local=local.to('cuda'), target=gpu_target, legal=gpu_legal,
                    teacher=teacher, retention=retention, retention_counts=retention_counts,
                    guard_counts=guard_counts, guard_mask=guard_mask,
                    teacher_sha256=tensor_digest([('teacher',teacher)]),
                    retention_mask_sha256=array_sha(retention.cpu().numpy()))
                np.savez_compressed(output/f'mask_{index:02d}.npz', grid_gt=grid_gt, valid=valid,
                    local=local.numpy(), full=legal.numpy(), target=target.numpy(), band=band,
                    retention=retention.cpu().numpy(), guard_interface=guard_mask.cpu().numpy(),
                    teacher_logits=teacher.cpu().numpy())
            caches.append(dict(features=features, box=tile['box'], ph=ph, pw=pw,
                native_shape=(y1-y,x1-x), original_grid=grid.detach(),
                features_sha256=tensor_digest([(str(i), v) for i,v in enumerate(features)])))
            print(json.dumps(dict(stage='fixed_features', tile=index, total=len(record['tiles']))), flush=True)
    if len(relevant) != 2:
        raise RuntimeError('Exactly the two existing relevant tiles required')
    initial = {name: value.detach().cpu().clone() for name,value in model.affinity_decoder.state_dict().items()}
    cfg = config['backend_adaptation']
    summaries = {}; started = time.time()

    def evaluate(head, arm, step):
        blend = BlendMap(gt.shape)
        folder = output/arm/f'step_{step:02d}'; folder.mkdir(parents=True)
        fits = {}
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            for index, item in enumerate(caches):
                logits = head(item['features'])['affinity_logits'].float()
                grid = affinity_boundary_probability(logits, mode=mode, **fusion)
                if step == 0 and not torch.equal(grid, item['original_grid']):
                    raise RuntimeError('Copied head step0 logits/fusion differs at tile '+str(index))
                if index in relevant:
                    chosen = masks[index]
                    fits[str(index)] = relation_fit_statistics(logits, chosen['target'], chosen['local'])
                    _, statistics = balanced_affinity_retention_kl(logits,
                        chosen['teacher'], chosen['target'], chosen['retention'])
                    fits[str(index)]['retention'] = statistics
                    fits[str(index)]['guard_interface'] = relation_fit_statistics(logits,
                        chosen['target'], chosen['guard_mask'])
                    np.savez_compressed(folder/f'tile_{index:02d}.npz', logits=logits.cpu().numpy(),
                        grid=grid.cpu().numpy(), logits_stride=np.asarray(logits.stride(), np.int64))
                native = views.crop_letterbox_output(grid, 1024, item['ph'], item['pw'], item['native_shape'])
                blend.add(item['box'], native.cpu()[0,0].numpy())
        boundary, blend_audit = blend.finish(); blend_audit['tiles'] = len(caches)
        decoded = decode_capture(semantic, torch.from_numpy(boundary)[None,None], config)
        classes, _ = revote_instances(decoded['instances'], probability)
        if step == 0 and (not np.array_equal(boundary, expected_boundary)
                or not np.array_equal(decoded['actual_markers'], expected_markers)
                or not np.array_equal(decoded['watershed'], expected_watershed)
                or not np.array_equal(decoded['instances'], expected_ids) or classes != expected_classes
                or blend_audit != record['blend']['baseline']):
            raise RuntimeError('Copied head full12 step0 final deployment parity failed')
        np.savez_compressed(folder/'maps.npz', boundary=boundary, markers=decoded['actual_markers'],
                            watershed=decoded['watershed'])
        cv2.imwrite(str(folder/'instances.png'), decoded['instances']); write(folder/'classes.json', classes)
        save_thumbnail(folder/'detail.png', rgb, boundary, decoded['instances'], gt, pair, arm+' step '+str(step))
        save_thumbnail(folder/'guard.png', rgb, boundary, decoded['instances'], gt,
                       guard['gt_ids'], arm+' preserved interface step '+str(step))
        contributions = {}
        for g in sorted(set(pair + guard['gt_ids'])):
            ids, counts = np.unique(decoded['instances'][gt==g], return_counts=True)
            contributions[str(g)] = sorted([dict(pred=int(p), pixels=int(n),
                pred_class=classes.get(str(int(p)))) for p,n in zip(ids,counts)], key=lambda r:-r['pixels'])
        summary = dict(step=step, all_tiles_recomputed=len(caches), blend=blend_audit,
            final_contract=decoded['capture_audit'], point_markers=[int(decoded['actual_markers'][tuple(p)]) for p in points],
            point_instances=[int(decoded['instances'][tuple(p)]) for p in points], gt_contributions=contributions,
            guard_point_markers=[int(decoded['actual_markers'][tuple(p)]) for p in guard['points']],
            guard_point_instances=[int(decoded['instances'][tuple(p)]) for p in guard['points']],
            marker_count=int(decoded['actual_markers'].max()), final_instances=int(decoded['instances'].max()),
            final_boundary_sha256=array_sha(boundary), final_instance_sha256=array_sha(decoded['instances']),
            fixed_local_fit=fits, step0_exact=True if step==0 else None, parameter_change=parameter_change(head, initial))
        write(folder/'summary.json', summary)
        print(json.dumps(dict(stage='evaluated', arm=arm, step=step, markers=summary['marker_count'])), flush=True)
        return summary

    for arm in ('combined', 'retained'):
        folder = output/arm; folder.mkdir()
        head = deepcopy(model.affinity_decoder).eval().requires_grad_(True)
        if tensor_digest(head.state_dict().items()) != tensor_digest(initial.items()):
            raise RuntimeError('Copied head initial state differs')
        optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=float(cfg['weight_decay']), eps=1.e-4)
        retain_weight = RETAIN_WEIGHT if arm == 'retained' else 0.0
        supervision = {str(i): dict(full=supervision_record(masks[i]['target'], masks[i]['legal']),
            local=supervision_record(masks[i]['target'], masks[i]['local']),
            retention=masks[i]['retention_counts'], guard=masks[i]['guard_counts'],
            teacher_sha256=masks[i]['teacher_sha256'],
            retention_mask_sha256=masks[i]['retention_mask_sha256']) for i in relevant}
        snapshots = {'0': evaluate(head, arm, 0)}
        rows = []
        for step in range(1, STEPS+1):
            index = relevant[(step-1)%len(relevant)]; selected = masks[index]
            mask = selected['legal']
            optimizer.zero_grad(set_to_none=True)
            logits = head(caches[index]['features'])['affinity_logits'].float()
            logits.retain_grad()
            full, details = balanced_affinity_loss(logits, selected['target'], mask, **options)
            local, local_details = balanced_affinity_loss(logits, selected['target'], selected['local'], **options)
            retention, retention_details = balanced_affinity_retention_kl(logits,
                selected['teacher'], selected['target'], selected['retention'])
            terms = dict(full=full, local=local, retention=retention)
            gradient_terms = component_gradient_record(terms, head) if step in GRADIENT_STEPS else None
            loss = FULL_WEIGHT*full + LOCAL_WEIGHT*local + retain_weight*retention
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite real local fit loss')
            # 分项autograd.grad会触发中间Tensor的retain_grad，清掉其诊断累加。
            logits.grad = None
            loss.backward()
            if torch.count_nonzero(logits.grad[~mask]):
                raise RuntimeError('Unselected/unknown/padding logit gradient must be exactly zero')
            grad = float(torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg['grad_clip']), error_if_nonfinite=True))
            if grad <= 0 or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Missing real head gradient or frozen gradient present')
            optimizer.step()
            row = dict(step=step, tile=index, loss=float(loss.detach()), gradient_norm_before_clip=grad,
                       logit_gradient_l2=float(logits.grad.float().norm()), outside_supervision_gradient_max=0.,
                       parameter_change=parameter_change(head, initial),
                       positive_edges=details['positive_edges'], negative_edges=details['negative_edges'],
                        component_losses={name:float(value.detach()) for name,value in terms.items()},
                        component_gradients=gradient_terms, retention_weight=retain_weight,
                        gradient_clip_multiplier=min(1.,float(cfg['grad_clip'])/(grad+1.e-6)),
                        local_details=scalar_record(local_details), retention_details=retention_details,
                        head_state_sha256=tensor_digest(head.state_dict().items()) if step in GRADIENT_STEPS else None)
            del logits, loss, full, local, retention, terms
            if row['parameter_change']['l2'] <= 0:
                raise RuntimeError('Optimizer failed to change actual head parameters')
            rows.append(row)
            with (folder/'steps.jsonl').open('a',encoding='utf8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(dict(stage='update', arm=arm, step=step, loss=row['loss'], grad=grad)), flush=True)
            if step in SNAPSHOTS:
                snapshots[str(step)] = evaluate(head, arm, step)
                torch.save(dict(format='affinity_weak_retain_head_v1', step=step, arm=arm,
                    geometry_state_dict=head.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    initial_head_sha256=tensor_digest(initial.items()), supervision=supervision,
                    loss_weights=dict(full=FULL_WEIGHT, local=LOCAL_WEIGHT, retention=retain_weight),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), folder/f'head_{step:02d}.pt')
        reloaded = deepcopy(model.affinity_decoder).eval().requires_grad_(False)
        saved = torch.load(folder/'head_32.pt', map_location='cuda', weights_only=False)
        reloaded.load_state_dict(saved['geometry_state_dict'], strict=True)
        if tensor_digest(reloaded.state_dict().items()) != tensor_digest(head.state_dict().items()):
            raise RuntimeError('Saved head strict reload differs')
        if tensor_digest(model.state_dict().items()) != frozen_model or tensor_digest(restorer.state_dict().items()) != frozen_restorer:
            raise RuntimeError('Fixed model/D5a changed during diagnostic')
        if any(item['features_sha256'] != tensor_digest([(str(i),v) for i,v in enumerate(item['features'])]) for item in caches):
            raise RuntimeError('Detached feature cache changed')
        for item in masks.values():
            if tensor_digest([('teacher',item['teacher'])]) != item['teacher_sha256']:
                raise RuntimeError('Fixed retention teacher changed')
            if array_sha(item['retention'].cpu().numpy()) != item['retention_mask_sha256']:
                raise RuntimeError('Fixed retention mask changed')
        with np.load(folder/'step_32/maps.npz', allow_pickle=False) as arrays:
            np.savez_compressed(folder/'maps.npz', semantic=semantic.numpy(), raw_probability=probability,
                                **{name: arrays[name].copy() for name in arrays.files})
        import shutil
        shutil.copyfile(folder/'step_32/instances.png', folder/'inst.png')
        shutil.copyfile(folder/'step_32/classes.json', folder/'class.json')
        summaries[arm] = dict(steps=STEPS, supervision=supervision, snapshots=snapshots,
            final_head_sha256=tensor_digest(head.state_dict().items()), strict_reload_equal=True,
            non_head_frozen_exact=True, feature_cache_exact=True, fixed_teacher_and_mask_exact=True,
            loss_weights=dict(full=FULL_WEIGHT, local=LOCAL_WEIGHT, retention=retain_weight),
            training_module_mode='eval; gradients enabled')
        del head,reloaded,saved,optimizer; gc.collect(); torch.cuda.empty_cache()
    verify_protected(protected)
    if runtime_contract(config) != runtime or precision_record(torch) != precision:
        raise RuntimeError('Runtime/precision changed')
    write(output/'report.json', dict(complete=True, format='affinity_weak_retain_v1', source=source,
        steps_per_arm=STEPS, seed=receipt['seed'], pair=pair, points=points, relevant_tiles=relevant,
        optimizer=dict(name='AdamW', lr=LR, eps=1.e-4, weight_decay=float(cfg['weight_decay']),
                       grad_clip=float(cfg['grad_clip']), scheduler='constant diagnostic'),
        config=config, source_path=receipt['source_path'], gt_path=receipt['gt_path'],
        prior_fit_root=str(prior_root), prior_analysis_sha256=sha(prior_root/'analysis/report.json'),
        retention_confidence=CONFIDENCE, guard=guard,
        common_loss_weights=dict(full=FULL_WEIGHT,local=LOCAL_WEIGHT),
        unique_variable='retention weight 0 versus 1; inputs and initial states identical',
        loss_options=options, local_band_radius_grid_pixels=16, local_band_radius_native_pixels=32,
        common_initial_head_sha256=tensor_digest(initial.items()),
        capture_sha256=sha(capture_root/'capture.json'), protected_inputs_exact=True,
        capture_source_privacy_relocation=capture_source_relocation,
        fixed_model_sha256=frozen_model, fixed_restorer_sha256=frozen_restorer, arms=summaries,
        runtime=runtime, precision=precision, elapsed_seconds=time.time()-started,
        caveat='Seen-source fixed-input retention check; U/R differ only in retention, versus prior B also full-GT term changes; not generalization or promotion'))
    write(output/'manifest.json', dict(complete=True, files={p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file()}))
    print('WEAK_RETAIN_COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run',), default='run')
    parser.add_argument('--capture', default='outputs/affinity_weak')
    parser.add_argument('--output', default='outputs/weak_retain')
    parser.add_argument('--prior', default='outputs/weak_fit')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
