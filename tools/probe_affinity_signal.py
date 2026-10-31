# -*- coding: utf-8 -*-
"""从四个封存R1输出审计实际BCE监督落点；纯CPU，无参数更新。"""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
R1_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
CAPTURE_SHA = 'cd7c5c0a479fa09335389342256e4827eddee03ef24746b22877a9f6e13910fe'
CAPTURE_MANIFEST_SHA = '665b1e92fb8210e12499a3b27a9a24ace5d1b84b1a73f0237f1de2087b50d2c5'
R1_PIPELINE_SHA = 'dfda0cbe675dfc8d92cf327e484be9af7eda3cf9f1d8fb334e17f0672225c7ef'
EXPECTED_DRAWS = {24: ('train_686.jpg', 'whole'), 25: ('train_707.jpg', 'whole'),
                  28: ('train_874.jpg', 'native1024'), 37: ('train_006.jpg', 'native1024')}
PRIMARY = {(24, 'split', (109, 109)), (24, 'split', (154, 154)),
           (25, 'split', (46, 46)), (25, 'merge', (21, 22)),
           (25, 'merge', (110, 131))}
LOSS_CFG = dict(negative_weight=1., hard_negative_weight=1., hard_negative_gamma=2.,
                manual_uncovered_as_boundary=False, manual_uncovered_boundary_weight=.2,
                pseudo_negative_weight=1., normalize_edge_weights=True)
FUNCTIONS = {'train_direct_semantic_affinity.py': ('compute_affinity_loss',),
             'utils/affinity_loss.py': ('_edge_slices', 'build_affinity_targets_torch', 'balanced_affinity_loss'),
             'data/direct_dual_head_dataset.py': ('_spatial_transform',),
             'utils/offset_letterbox.py': ('GeometryLetterbox', 'geometry_letterbox_metadata', 'letterbox_instance_geometry')}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, body):
    Path(path).write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def safe_child(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Expected relative contained path: '+str(name))
    return Path(root)/path


def nodes(path, names):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'), filename=str(path))
    result = []
    for name in names:
        matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        if len(matches) != 1:
            raise RuntimeError('Exactly one original function/class required: '+name)
        result.append(matches[0])
    return result


def load_functions(snapshot):
    """只执行原快照的纯函数AST；不导入现工作树trainer、SAM2或模型。"""
    snapshot = Path(snapshot)
    tree = ast.parse((snapshot/'utils/affinity_graph.py').read_text(encoding='utf-8-sig'))
    offset_nodes = [n for n in tree.body if isinstance(n, ast.AnnAssign)
                    and isinstance(n.target, ast.Name) and n.target.id == 'DEFAULT_AFFINITY_OFFSETS']
    if len(offset_nodes) != 1:
        raise RuntimeError('Original offsets declaration missing')
    offsets = ast.literal_eval(offset_nodes[0].value)
    if offsets != ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (0, 4), (4, 0)):
        raise RuntimeError('Original eight-channel order changed')
    env = dict(__name__=__name__, torch=torch, F=F, cv2=cv2, np=np, Dict=Dict, Sequence=Sequence, Tuple=Tuple,
               dataclass=dataclass, asdict=asdict, DEFAULT_AFFINITY_OFFSETS=offsets)
    ordered = ('utils/affinity_loss.py', 'data/direct_dual_head_dataset.py',
               'utils/offset_letterbox.py', 'train_direct_semantic_affinity.py')
    for name in ordered:
        fragment = ast.Module(body=nodes(snapshot/name, FUNCTIONS[name]), type_ignores=[])
        exec(compile(fragment, str(snapshot/name), 'exec'), env)
    return env


def validate_inputs(logits, grid_gt, content, classes, loss_cfg):
    if loss_cfg != LOSS_CFG:
        raise ValueError('This diagnostic is fixed to actual R1 r1/h1/gamma2 manual loss')
    if (not isinstance(logits, torch.Tensor) or logits.device.type != 'cpu'
            or logits.dtype != torch.float32 or logits.ndim != 4 or logits.shape[:2] != (1, 8)):
        raise ValueError('CPU float32 logits [1,8,H,W] required')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    gt, valid = np.asarray(grid_gt), np.asarray(content)
    if (gt.ndim != 2 or min(gt.shape, default=0) < 1 or gt.shape != tuple(logits.shape[-2:]) or not np.issubdtype(gt.dtype, np.integer)
            or gt.min() < 0 or gt.max() > 65535 or valid.dtype != np.bool_ or valid.shape != gt.shape):
        raise ValueError('Aligned integer GT and bool content required')
    mapping = {}
    for k, v in classes.items():
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v not in (0, 1):
            raise ValueError('Class convention must be integer P=0/F=1')
        if isinstance(k, (bool, np.bool_)):
            raise ValueError('Class key cannot be bool')
        key = int(k)
        if key <= 0 or key > 65535 or key in mapping:
            raise ValueError('Invalid or duplicated class key')
        mapping[key] = int(v)
    if not set(np.unique(gt[gt > 0])).issubset(mapping):
        raise ValueError('Every positive GT needs canonical phase lookup')
    return gt, valid, mapping


def element_contributions(logits, target, edge_valid):
    """重构真实loss每项系数，hard权重仅依赖detach后的响应。"""
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    coefficient = torch.zeros_like(logits)
    effective_weight = torch.zeros_like(logits)
    active = sum(bool(edge_valid[:, c].any()) for c in range(8))
    channel_rows = []
    for c in range(8):
        positive = edge_valid[:, c] & (target[:, c] > .5)
        negative = edge_valid[:, c] & ~positive
        terms = int(positive.any())+int(negative.any())
        wp = torch.ones_like(logits[:, c][positive])
        wn = 1.+logits[:, c][negative].sigmoid().detach().square()
        if terms:
            if wp.numel():
                coefficient[:, c][positive] = wp/wp.sum() / (active*terms)
                effective_weight[:, c][positive] = wp
            if wn.numel():
                coefficient[:, c][negative] = wn/wn.sum() / (active*terms)
                effective_weight[:, c][negative] = wn
        channel_rows.append(dict(channel=c, positive_n=int(positive.sum()), negative_n=int(negative.sum()),
                                 positive_weight=float(wp.sum()), negative_weight=float(wn.sum()),
                                 class_terms=terms, active_channels=active))
    return raw*coefficient, coefficient, effective_weight, channel_rows


def _sum_stats(rows):
    n = sum(r['n'] for r in rows)
    result = dict(n=n, raw_bce_sum=sum(r['raw_bce_sum'] for r in rows),
                  weighted_bce_sum=sum(r['weighted_bce_sum'] for r in rows),
                  effective_weight=sum(r['effective_weight'] for r in rows),
                  loss_contribution=sum(r['loss_contribution'] for r in rows),
                  gradient_l1=sum(r['gradient_l1'] for r in rows),
                  gradient_l2_squared=sum(r['gradient_l2_squared'] for r in rows),
                  violations_half=sum(r['violations_half'] for r in rows),
                  unknown_path_n=sum(r['unknown_path_n'] for r in rows))
    result.update(mean_raw_bce=result['raw_bce_sum']/n if n else None,
                  mean_weighted_bce=result['weighted_bce_sum']/result['effective_weight'] if result['effective_weight'] else None,
                  gradient_l2=math.sqrt(result['gradient_l2_squared']),
                  violation_fraction=result['violations_half']/n if n else None)
    return result


def _path_unknown(gt, valid, source, destination, dy, dx):
    """实际目标只检查端点；中间未知不新增ignore，仅单列计数。"""
    distance = max(abs(dy), abs(dx))
    sy, sx = source
    mask = np.zeros(gt[source].shape, bool)
    for step in range(1, distance):
        ty, tx = dy*step//distance, dx*step//distance
        iy = slice(sy.start+ty, sy.stop+ty)
        ix = slice(sx.start+tx, sx.stop+tx)
        mask |= (gt[iy, ix] == 0) | ~valid[iy, ix]
    return mask


def contact_lengths(gt, valid):
    """四邻真实GT直接接触的网格边数；不是物理弧长。"""
    contacts = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = gt[:gt.shape[0]-dy, :gt.shape[1]-dx]
        b = gt[dy:, dx:]
        keep = valid[:gt.shape[0]-dy, :gt.shape[1]-dx] & valid[dy:, dx:] & (a > 0) & (b > 0) & (a != b)
        pair = np.sort(np.stack((a[keep], b[keep]), axis=1), axis=1)
        if not len(pair):
            continue
        unique, counts = np.unique(pair, axis=0, return_counts=True)
        for (left, right), n in zip(unique, counts):
            key = (int(left), int(right)); contacts[key] = contacts.get(key, 0)+int(n)
    return contacts


def _tail_budget(rows, value_key):
    if not rows:
        return dict(pairs=0, selected=0, count_share=None, loss_share=None, gradient_l1_share=None)
    count = max(1, math.ceil(.1*len(rows)))
    chosen = sorted(rows, key=lambda r: (-r[value_key], tuple(r['gt_ids'])))[:count]
    totals = _sum_stats([r['groups']['all'] for r in rows]); selected = _sum_stats([r['groups']['all'] for r in chosen])
    def ratio(key):
        return selected[key]/totals[key] if totals[key] else None
    return dict(pairs=len(rows), selected=count, selected_gt_pairs=[r['gt_ids'] for r in chosen],
                count_share=ratio('n'), loss_share=ratio('loss_contribution'), gradient_l1_share=ratio('gradient_l1'))


def summarize_pairs(pairs):
    adjacent = [r for r in pairs if r['four_neighbor_contact'] > 0]
    report = dict(adjacent_pairs=len(adjacent), nonadjacent_endpoint_pairs=len(pairs)-len(adjacent))
    report['phase_groups'] = {phase:dict(pairs=len(rows), **_sum_stats([r['groups']['all'] for r in rows]))
        for phase in ('F/F', 'F/P', 'P/P') for rows in [[r for r in adjacent if r['phase'] == phase]]}
    report['top_tenth_by_contact'] = _tail_budget(adjacent, 'four_neighbor_contact')
    ranked = [dict(r, _loss=r['groups']['all']['loss_contribution'], _gradient=r['groups']['all']['gradient_l1']) for r in adjacent]
    report['top_tenth_by_loss'] = _tail_budget(ranked, '_loss')
    report['top_tenth_by_gradient_l1'] = _tail_budget(ranked, '_gradient')
    report['quartiles'] = {}
    for name in ('min_grid_area', 'four_neighbor_contact'):
        cuts = np.quantile([r[name] for r in adjacent], [.25, .5, .75]).tolist() if adjacent else []
        bins = []
        for index in range(4):
            chosen = [r for r in adjacent if int(np.searchsorted(cuts, r[name], side='right')) == index]
            bins.append(dict(bin=index, pairs=len(chosen), **_sum_stats([r['groups']['all'] for r in chosen])))
        report['quartiles'][name] = dict(cuts=cuts, bins=bins, domain='within this view, no physical scale pooling')
    return report


def analyze_signal(logits, grid_gt, content, classes, loss_cfg, functions):
    """返回小数表；clone叶张量，不改输入，梯度为输出logit梯度。"""
    gt, valid, lookup = validate_inputs(logits, grid_gt, content, classes, loss_cfg)
    leaf = logits.detach().clone().requires_grad_(True)
    labels = torch.from_numpy(gt.astype(np.int64, copy=True))[None]
    pixel_valid = torch.from_numpy(valid.copy())[None, None]
    batch = dict(affinity_instance_map=labels, affinity_valid_content=pixel_valid,
                 uncovered_boundary_source=torch.ones(1, dtype=torch.bool))
    target, edge_valid, uncovered = functions['build_affinity_targets_torch'](
        labels, pixel_valid, uncovered_as_boundary=torch.zeros(1, dtype=torch.bool), return_uncovered_mask=True)
    if bool(uncovered.any()):
        raise RuntimeError('Unexpected uncovered supervision')
    observed=[]
    original_builder=functions['build_affinity_targets_torch']
    def capture_builder(actual_ids, actual_valid, **kwargs):
        if not torch.equal(actual_ids,labels) or not torch.equal(actual_valid,pixel_valid):
            raise RuntimeError('Actual compute changed cached full-batch GT/grid/content')
        result=original_builder(actual_ids,actual_valid,**kwargs)
        observed.append(result)
        return result
    functions['build_affinity_targets_torch']=capture_builder
    try:
        actual, metrics = functions['compute_affinity_loss'](leaf, batch, loss_cfg, pseudo=False)
    finally:
        functions['build_affinity_targets_torch']=original_builder
    if len(observed)!=1 or any(not torch.equal(a,b) for a,b in zip(observed[0],(target,edge_valid,uncovered))):
        raise RuntimeError('Actual compute target/mask differs from explicit audit target')
    gradient = torch.autograd.grad(actual, leaf, retain_graph=True)[0]
    terms, coefficients, effective, channel_rows = element_contributions(leaf, target, edge_valid)
    rebuilt = terms.sum()
    rebuilt_gradient = torch.autograd.grad(rebuilt, leaf)[0]
    torch.testing.assert_close(rebuilt, actual, atol=1e-7, rtol=2e-6)
    torch.testing.assert_close(rebuilt_gradient, gradient, atol=1e-10, rtol=2e-6)
    if not torch.equal(gradient[~edge_valid], torch.zeros_like(gradient[~edge_valid])):
        raise RuntimeError('Unknown/padding output gradient is nonzero')
    # 用double作分组累加，避免大量小贡献的float32统计相消。
    raw = F.binary_cross_entropy_with_logits(leaf, target, reduction='none').detach().numpy()[0]
    losses = terms.detach().numpy()[0]; grads = gradient.detach().numpy()[0]
    weights = effective.numpy()[0]; probability = leaf.detach().sigmoid().numpy()[0]
    masks = edge_valid.numpy()[0]; targets = target.numpy()[0]
    grouped_negative, grouped_positive = {}, {}
    areas = {int(k):int(n) for k,n in zip(*np.unique(gt[valid & (gt > 0)], return_counts=True))}
    contacts = contact_lengths(gt, valid)
    for c, (dy, dx) in enumerate(functions['DEFAULT_AFFINITY_OFFSETS']):
        if gt.shape[0] <= abs(dy) or gt.shape[1] <= abs(dx):
            continue
        source, destination = functions['_edge_slices'](*gt.shape, dy, dx)
        a, b = gt[source], gt[destination]
        unknown_path = _path_unknown(gt, valid, source, destination, dy, dx)
        for positive, table in ((False, grouped_negative), (True, grouped_positive)):
            keep = masks[c][source] & ((targets[c][source] > .5) == positive)
            if not keep.any():
                continue
            keys = np.stack((np.minimum(a[keep], b[keep]), np.maximum(a[keep], b[keep])), axis=1)
            unique, inverse = np.unique(keys, axis=0, return_inverse=True)
            values = dict(raw_bce_sum=raw[c][source][keep], effective_weight=weights[c][source][keep],
                          weighted_bce_sum=raw[c][source][keep].astype(np.float64)*weights[c][source][keep].astype(np.float64),
                          loss_contribution=losses[c][source][keep], gradient_l1=np.abs(grads[c][source][keep]),
                          gradient_l2_squared=np.square(grads[c][source][keep].astype(np.float64)),
                          violations_half=(probability[c][source][keep] < .5) if positive else (probability[c][source][keep] >= .5),
                          unknown_path_n=unknown_path[keep])
            accumulated = {key:np.bincount(inverse, weights=val.astype(np.float64), minlength=len(unique)) for key,val in values.items()}
            ns = np.bincount(inverse, minlength=len(unique))
            for index, pair in enumerate(unique):
                key = tuple(map(int, pair))
                record = dict(channel=c, n=int(ns[index]), **{k:float(v[index]) for k,v in accumulated.items()})
                record['violations_half'] = int(record['violations_half']); record['unknown_path_n'] = int(record['unknown_path_n'])
                table.setdefault(key, []).append(record)
    pairs, objects = [], []
    for key, rows in sorted(grouped_negative.items()):
        phases = sorted((lookup[key[0]], lookup[key[1]]))
        phase = {(0, 0):'P/P', (0, 1):'F/P', (1, 1):'F/F'}[tuple(phases)]
        groups = {name:_sum_stats([r for r in rows if r['channel'] in channels])
                  for name, channels in [('short', range(4)), ('distance2', range(4,6)), ('distance4', range(6,8)), ('all', range(8))]}
        pairs.append(dict(gt_ids=list(key), phase=phase, grid_areas=[areas[key[0]], areas[key[1]]],
                          min_grid_area=min(areas[key[0]], areas[key[1]]),
                          four_neighbor_contact=contacts.get(key, 0), channels=rows, groups=groups))
    for (key, _), rows in sorted(grouped_positive.items()):
        objects.append(dict(gt=key, phase=lookup[key], grid_area=areas[key], channels=rows,
                            groups={name:_sum_stats([r for r in rows if r['channel'] in channels])
                                    for name, channels in [('short', range(4)), ('distance2', range(4,6)), ('distance4', range(6,8)), ('all', range(8))]}))
    neg_total = _sum_stats([r['groups']['all'] for r in pairs])
    pos_total = _sum_stats([r['groups']['all'] for r in objects])
    for group, denominator in ((pairs, neg_total), (objects, pos_total)):
        for row in group:
            for value in row['groups'].values():
                value['class_loss_share'] = value['loss_contribution']/denominator['loss_contribution'] if denominator['loss_contribution'] else None
                value['class_gradient_l1_share'] = value['gradient_l1']/denominator['gradient_l1'] if denominator['gradient_l1'] else None
    grouped_loss = neg_total['loss_contribution']+pos_total['loss_contribution']
    grouped_l1 = neg_total['gradient_l1']+pos_total['gradient_l1']
    grouped_l2sq = neg_total['gradient_l2_squared']+pos_total['gradient_l2_squared']
    actual_value=float(actual.detach())
    if not math.isclose(grouped_loss, actual_value, rel_tol=3e-6, abs_tol=1e-7):
        raise RuntimeError('Pair/object loss does not reconstruct actual BCE')
    if not math.isclose(grouped_l1, float(gradient.abs().double().sum()), rel_tol=2e-6, abs_tol=1e-10):
        raise RuntimeError('Pair/object output-gradient L1 does not close')
    if not math.isclose(grouped_l2sq, float(gradient.double().square().sum()), rel_tol=2e-6, abs_tol=1e-16):
        raise RuntimeError('Pair/object squared output-gradient L2 does not close')
    return dict(actual_bce=actual_value, positive=pos_total, negative=neg_total,
                actual_supervision={k:int(metrics[k]) for k in ('positive_edges', 'negative_edges')},
                channels=channel_rows, pair_rows=pairs, object_rows=objects,
                pair_summary=summarize_pairs(pairs),
                reconstruction=dict(passed=True, loss_abs_error=abs(float((rebuilt-actual).detach())),
                    gradient_max_abs_error=float((rebuilt_gradient-gradient).abs().max()),
                    grouped_loss_abs_error=abs(grouped_loss-actual_value),
                    unknown_and_padding_gradient_zero=True, no_internal_target_resizing=True,
                    actual_compute_target_and_mask_exact=True),
                scope='actual manual BCE on cached frozen-e20 output logits; not head parameter or historical update gradients')


def native_adjacency_coverage(native_gt, classes, signal):
    """原裁图直接接触与训练网格对照；仅记录重采样损失，不补造目标。"""
    native=np.asarray(native_gt)
    contacts=contact_lengths(native,np.ones_like(native,dtype=bool))
    pairs={tuple(r['gt_ids']):r for r in signal['pair_rows']}
    rows=[]
    for key,n in sorted(contacts.items()):
        pair=pairs.get(key)
        phases=tuple(sorted((classes[key[0]],classes[key[1]])))
        rows.append(dict(gt_ids=list(key),phase={(0,0):'P/P',(0,1):'F/P',(1,1):'F/F'}[phases],
            native_contact_grid_edges=n,actual_training_grid_contact=pair['four_neighbor_contact'] if pair else 0,
            short_negative_edges=pair['groups']['short']['n'] if pair else 0,
            all_negative_edges=pair['groups']['all']['n'] if pair else 0))
    return dict(native_adjacent_pairs=len(rows),rows=rows,
        no_training_grid_four_neighbor_contact=[r['gt_ids'] for r in rows if r['actual_training_grid_contact']==0],
        no_short_negative_edges=[r['gt_ids'] for r in rows if r['short_negative_edges']==0],
        no_negative_edges_any_channel=[r['gt_ids'] for r in rows if r['all_negative_edges']==0],
        training_grid_adjacency_without_native_contact=[r['gt_ids'] for r in pairs.values()
            if r['four_neighbor_contact']>0 and tuple(r['gt_ids']) not in contacts],
        scope='native cropped processed GT versus actual downsampled training target; neither is a new supervision mask')


def case_tables(draw, cases, signal):
    pairs = {tuple(r['gt_ids']):r for r in signal['pair_rows']}
    objects = {r['gt']:r for r in signal['object_rows']}
    results = []
    for case in cases:
        key = (draw, case['kind'], tuple(sorted(case['gt_ids'])))
        if key not in PRIMARY:
            continue
        row = pairs.get(key[2]) if case['kind'] == 'merge' else objects.get(key[2][0])
        peers = []
        if row is not None and case['kind'] == 'merge' and row['four_neighbor_contact']:
            peers = [p for p in pairs.values() if p['gt_ids'] != row['gt_ids'] and p['phase'] == row['phase']
                     and .5*row['four_neighbor_contact'] <= p['four_neighbor_contact'] <= 2*row['four_neighbor_contact']
                     and all(.5*a <= b <= 2*a for a,b in zip(sorted(row['grid_areas']),sorted(p['grid_areas'])))]
        results.append(dict(case=case, row=row, no_contact=row is None or (case['kind']=='merge' and row['four_neighbor_contact']==0),
                            peers=peers, no_peer=len(peers)==0,
                            peer_rule='same phase; each sorted GT grid area and four-neighbor contact within 0.5..2x; no forced match; peers not certified error-free',
                            split_scope='whole GT positive relations; does not isolate the actual erroneous partition seam' if case['kind']=='split' else None))
    return results


def verify_runtime(expected):
    actual = dict(python_executable=sys.executable, python_version=sys.version, torch=torch.__version__,
                  cuda=torch.version.cuda, numpy=np.__version__, cv2=cv2.__version__)
    if any(actual[k] != expected[k] for k in actual):
        raise RuntimeError('Original runtime scalar differs')
    files = {Path(expected['sam2_config']):expected['sam2_config_sha256']}
    files.update({safe_child(expected['sam2_package'], n):h for n,h in expected['sam2_python_sha256'].items()})
    for path, digest in files.items():
        if sha(path) != digest:
            raise RuntimeError('SAM2 runtime file changed: '+str(path))
    process = subprocess.run(['git', '-C', str(Path(expected['sam2_package']).parent), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=False)
    if (process.stdout.strip() if process.returncode==0 else None) != expected['sam2_repo_head']:
        raise RuntimeError('SAM2 repository revision differs')
    return dict(passed=True, files=len(files), scalars=actual, sam2_not_imported=True, no_gpu_calls=True)


def run(capture, r1_root, output):
    capture, r1_root, output = map(lambda p:Path(p).resolve(), (capture, r1_root, output))
    if output.exists() or output.parent != ROOT/'outputs' or output.name != 'affinity_signal':
        raise RuntimeError('Fresh outputs/affinity_signal required; never overwrite an accepted analysis')
    manifest = read(capture/'capture_manifest.json'); record = read(capture/'capture.json')
    pipeline_path = r1_root.parent/'pipeline_status.json'; r1_pipeline = read(pipeline_path)
    r1_status = read(r1_root/'status.json'); snapshot = r1_root.parent/'source'
    if (manifest.get('complete') is not True or sha(capture/'capture.json') != CAPTURE_SHA
            or sha(capture/'capture_manifest.json') != CAPTURE_MANIFEST_SHA or sha(pipeline_path) != R1_PIPELINE_SHA
            or manifest['files'].get('capture.json') != CAPTURE_SHA or record.get('complete') is not True
            or record.get('no_training') is not True or record['checkpoints']['r1'] != R1_SHA
            or record['epoch'] != 20 or r1_status['status'] != 'complete' or r1_status['updates'] != 1280
            or r1_status['final_checkpoint_sha256'] != R1_SHA or sha(r1_root/'final.pt') != R1_SHA
            or r1_pipeline['status'] != 'complete' or r1_pipeline['stage'] != 'done'
            or len(r1_pipeline['sources']) != 49 or len(record['sealed_sources']) != 148):
        raise RuntimeError('Only accepted original R1 and complete four-draw capture are permitted')
    for key in ('frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if r1_status.get(key) is not True:
            raise RuntimeError('Original R1 freeze/reload audit missing: '+key)
    if record['config']['direct_semantic_affinity']['affinity_loss'] != LOSS_CFG or r1_status['config']['direct_semantic_affinity']['affinity_loss'] != LOSS_CFG:
        raise RuntimeError('R1 loss configuration differs')
    if (record['newgt']['original_covered_is_supervision_mask'] is not False
            or record['newgt']['filled_and_original_equal_weight'] is not True or record['newgt']['passed'] is not True):
        raise RuntimeError('Full processed-newGT supervision required')
    if {r['draw']:(r['source'][0],r['view']) for r in record['rows']} != EXPECTED_DRAWS or len(record['rows']) != 4:
        raise RuntimeError('Four predeclared draw cohort differs')
    protected = {Path(__file__):sha(__file__), capture/'capture_manifest.json':sha(capture/'capture_manifest.json'),
                 pipeline_path:sha(pipeline_path), r1_root/'status.json':sha(r1_root/'status.json'),
                 r1_root/'steps.jsonl':sha(r1_root/'steps.jsonl'), r1_root/'final.pt':R1_SHA}
    protected.update({safe_child(capture,n):h for n,h in manifest['files'].items()})
    protected.update({Path(n):h for n,h in record['protected_sha256'].items()})
    protected.update({safe_child(snapshot,n):h for n,h in r1_pipeline['sources'].items()})
    for n,h in record['sealed_sources'].items():
        protected[safe_child(ROOT,n)] = h
        protected[safe_child(ROOT/'outputs/affinity_tail/source',n)] = h
    for sample in record['newgt']['samples']:
        protected.update({safe_child(ROOT,n):h for n,h in sample['files_sha256'].items()})
    def check():
        for path, digest in protected.items():
            if sha(path) != digest:
                raise RuntimeError('Protected parent/source/data/core changed: '+str(path))
        return verify_runtime(record['sealed_runtime'])
    runtime_before = check()
    compatibility = {}
    for name,names in FUNCTIONS.items():
        if name not in r1_pipeline['sources']:
            raise RuntimeError('R1 original source closure lacks actual dependency: '+name)
        old = nodes(snapshot/name,names); new = nodes(ROOT/'outputs/affinity_tail/source'/name,names)
        if [ast.dump(n,include_attributes=False) for n in old] != [ast.dump(n,include_attributes=False) for n in new]:
            raise RuntimeError('R1 and cached148 pure loss/target/geometry implementations differ: '+name)
        compatibility[name] = dict(original_sha256=r1_pipeline['sources'][name], cached148_sha256=record['sealed_sources'][name],
                                   function_ast_equal=True, functions=list(names))
    if ast.dump(ast.parse((snapshot/'utils/affinity_graph.py').read_text()),include_attributes=False) != ast.dump(ast.parse((ROOT/'outputs/affinity_tail/source/utils/affinity_graph.py').read_text()),include_attributes=False):
        raise RuntimeError('Original affinity channel dependency changed')
    functions = load_functions(snapshot)
    steps = [json.loads(line) for line in (r1_root/'steps.jsonl').read_text().splitlines()]
    if len(steps)!=1280:
        raise RuntimeError('Original complete draw receipts required')
    output.mkdir(); (output/'tables').mkdir()
    rows=[]; seen=set(); torch.set_num_threads(4); cv2.setNumThreads(2)
    start = __import__('time').perf_counter()
    for item in sorted(record['rows'], key=lambda r:r['draw']):
        index=item['draw']; name=Path(item['source'][0]).stem
        receipt = next(r for r in steps[:64] if r['manual_draw']['draw_index']==[index])
        if receipt['manual']!=item['source'] or receipt['training_view']!=item['view'] or receipt['auxiliary_weight']!=0:
            raise RuntimeError('Original epoch1 draw receipt differs')
        sample=next(s for s in record['newgt']['samples'] if s['source']==name)
        gt_path=next(safe_child(ROOT,n) for n in sample['files_sha256'] if n.endswith('_gt.npz'))
        class_path=gt_path.with_name(name+'_class.json'); canonical_classes=read(class_path)
        classes={int(k):int(v) for k,v in item['gt_classes'].items()}
        if any(str(k) not in canonical_classes or canonical_classes[str(k)] != v for k,v in classes.items()):
            raise RuntimeError('Cached phase lookup differs from canonical processed newGT')
        with np.load(gt_path,allow_pickle=False) as original:
            full_gt=original['instance_map']
            y,x,y1,x1=receipt['manual_crop'][0]; native_gt=full_gt[y:y1,x:x1].copy()
        with np.load(capture/item['input_file'],allow_pickle=False) as blob:
            gt=blob['grid_gt'].copy(); content=blob['content'].copy(); logits=torch.from_numpy(blob['logits_r1'].copy())
            if not np.array_equal(blob['gt'],native_gt):
                raise RuntimeError('Cached native full processed GT does not match original draw crop')
        grid, valid, meta=functions['letterbox_instance_geometry'](native_gt,1024,512)
        draw=receipt['manual_draw']; hf=draw['horizontal_flip'][0]; vf=draw['vertical_flip'][0]; k=draw['rotation_k'][0]
        aligned=functions['_spatial_transform'](torch.from_numpy(grid),hf,vf,k).numpy()
        aligned_valid=functions['_spatial_transform'](torch.from_numpy(valid),hf,vf,k).numpy()
        if not np.array_equal(aligned,gt) or not np.array_equal(aligned_valid,content):
            raise RuntimeError('Actual full-batch letterbox/augmentation GT is not exactly cached grid/content')
        signal=analyze_signal(logits,gt,content,classes,LOSS_CFG,functions)
        coverage=native_adjacency_coverage(native_gt,classes,signal)
        signal['native_adjacency_coverage']=coverage
        fixed=case_tables(index,item['cases'],signal)
        seen.update((index,c['case']['kind'],tuple(sorted(c['case']['gt_ids']))) for c in fixed)
        filename=f'tables/draw_{index:03d}.json'; write(output/filename,signal)
        with (output/'tables'/f'draw_{index:03d}_pairs.csv').open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=['gt_a','gt_b','phase','contact_grid_edges','group','n','loss','gradient_l1','gradient_l2_squared','mean_bce','violation_fraction','unknown_path_n'])
            writer.writeheader()
            for row in signal['pair_rows']:
                for group,values in row['groups'].items():
                    writer.writerow(dict(gt_a=row['gt_ids'][0],gt_b=row['gt_ids'][1],phase=row['phase'],contact_grid_edges=row['four_neighbor_contact'],group=group,n=values['n'],
                        loss=values['loss_contribution'],gradient_l1=values['gradient_l1'],gradient_l2_squared=values['gradient_l2_squared'],mean_bce=values['mean_raw_bce'],violation_fraction=values['violation_fraction'],unknown_path_n=values['unknown_path_n']))
        rows.append(dict(draw=index,source=item['source'],view=item['view'],original_epoch1_draw_receipt=receipt,
            input_file_sha256=manifest['files'][item['input_file']],grid_reconstruction_exact=True,geometry=meta.to_dict(),
            actual_bce=signal['actual_bce'],positive=signal['positive'],negative=signal['negative'],
            pair_summary=signal['pair_summary'],native_adjacency_coverage=coverage,
            reconstruction=signal['reconstruction'],fixed_cases=fixed,table=filename))
        print(json.dumps(dict(draw=index,actual_bce=signal['actual_bce'],pairs=len(signal['pair_rows']),cpu_only=True)),flush=True)
    if seen != PRIMARY:
        raise RuntimeError('Five predeclared diagnostic objects not covered')
    runtime_after=check()
    if runtime_after != runtime_before:
        raise RuntimeError('Runtime changed during CPU diagnostic')
    report=dict(complete=True,format='affinity_signal_v1',no_training=True,no_gpu_inference=True,no_optimizer=True,
        checkpoint_sha256=R1_SHA,epoch=20,updates=1280,capture_sha256=CAPTURE_SHA,
        capture_manifest_sha256=protected[capture/'capture_manifest.json'],r1_source_count=49,cached_source_count=148,
        r1_original_source_functions=compatibility,loss_cfg=LOSS_CFG,full_processed_gt_including_filled=True,
        protected_sha256={str(p):h for p,h in protected.items()},protected_before_after_exact=True,runtime=runtime_after,
        model_and_restorer_not_loaded=True,
        rows=rows,elapsed_cpu_analysis_seconds=__import__('time').perf_counter()-start,
        scope='four seen augmented epoch1 draw inputs evaluated with frozen R1 e20; 2 whole + 2 native1024 from different sources',
        caveats=['output-logit gradients are not head/shared-parameter or historical optimizer gradients',
                 'GT adjacency and violations do not prove a physically weak boundary or annotation correctness',
                 'no same-source scale A/B, no test labels, no train-GT target count or area fitting',
                 'long relations use original endpoint-only validity; unknown interior paths counted, not masked',
                 'L1 and squared L2 budgets close; individual L2 norms are not additive',
                 'split object positive budget does not isolate the actual erroneous predicted seam',
                 'class-balanced BCE protects negative class total budget; no per-interface balance within that class'])
    write(output/'report.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',default='outputs/affinity_tail/analysis/partition')
    parser.add_argument('--r1-root',default='outputs/affinity_balance/mixed')
    parser.add_argument('--output',default='outputs/affinity_signal')
    args=parser.parse_args();run(args.capture,args.r1_root,args.output)
