# -*- coding: utf-8 -*-
"""只读缓存中的理想完整分隔预检；不运行模型、不生成GT或竞赛代理评分。

预先按OO界面弱响应排序，每源至多两个不同GT对/256原生窗口。每窗只使用共享
affinity_gap_loss的静态原子及最多四对尝试，不因失败扩大窗口或放宽来源资格。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def choose_interface_views(ids, original, valid, boundary, *, size=256, limit=2, minimum_contact=8):
    """选择只依赖固定GT/N1的两视图，不按理想干预结果追选病例。"""
    records = {}
    for dy, dx in [(0, 1), (1, 0)]:
        aa = np.s_[:ids.shape[0]-dy or None, :ids.shape[1]-dx or None]
        bb = np.s_[dy:, dx:]
        good = original[aa] & original[bb] & valid[aa] & valid[bb]
        good &= (ids[aa] > 0) & (ids[bb] > 0) & (ids[aa] != ids[bb])
        yy, xx = np.nonzero(good)
        first, second = ids[aa][good], ids[bb][good]
        pairs = np.stack([np.minimum(first, second), np.maximum(first, second)], axis=1)
        for pair in np.unique(pairs, axis=0):
            selected = np.all(pairs == pair, axis=1)
            key = tuple(map(int, pair))
            r = records.setdefault(key, dict(y=[], x=[], p=[]))
            r['y'].extend(yy[selected].tolist()); r['x'].extend(xx[selected].tolist())
            r['p'].extend(np.maximum(boundary[aa][good][selected], boundary[bb][good][selected]).tolist())
    views = []
    ranked = sorted(records.items(), key=lambda item: (float(np.mean(item[1]['p'])), item[0]))
    for pair, r in ranked:
        if len(r['p']) < minimum_contact:
            continue
        cy, cx = int(np.median(r['y'])), int(np.median(r['x']))
        # 坐标中位数未必恰好在弯曲界面上，选距离中位数最近的实际OO落点。
        nearest = int(np.argmin((np.asarray(r['y'])-cy)**2+(np.asarray(r['x'])-cx)**2))
        cy, cx = r['y'][nearest], r['x'][nearest]
        if any((cy-v['center_yx'][0])**2+(cx-v['center_yx'][1])**2 < (size//2)**2 for v in views):
            continue
        h, w = ids.shape
        y, x = max(0, min(cy-size//2, h-size)), max(0, min(cx-size//2, w-size))
        y1, x1 = min(h, y+size), min(w, x+size)
        views.append(dict(gt_pair=list(pair), center_yx=[cy, cx], box_yxyx=[y, x, y1, x1],
                          OO_contact_pixels=len(r['p']), OO_contact_mean_B=float(np.mean(r['p']))))
        if len(views) == limit:
            break
    return views


def crop_geometry(geometry, box):
    y, x, y1, x1 = box
    return {k: np.ascontiguousarray(v[y:y1, x:x1]) if isinstance(v, np.ndarray) else v
            for k, v in geometry.items()}


def fixed_views_for_source(cases, source, shape):
    """显式v2复用原病例，不调用困难程度排序或追选新的坐标。"""
    fields = ('gt_pair', 'center_yx', 'box_yxyx', 'OO_contact_pixels', 'OO_contact_mean_B')
    rows = [row for row in cases if row['source'] == source]
    if [row['view'] for row in rows] != list(range(len(rows))):
        raise ValueError('Fixed source view order changed')
    result = []
    for row in rows:
        y, x, y1, x1 = row['box_yxyx']
        if not (0 <= y < y1 <= shape[0] and 0 <= x < x1 <= shape[1]):
            raise ValueError('Fixed view no longer matches source shape')
        result.append({key: row[key] for key in fields})
    return result


def seed_assignment(ids, source_seed, sink_seed):
    first = set(map(int, np.unique(ids[source_seed]))) - {0}
    second = set(map(int, np.unique(ids[sink_seed]))) - {0}
    return dict(source_ids=sorted(first), sink_ids=sorted(second),
                shared_ids=sorted(first & second),
                separated=bool(first and second and not (first & second)))


def full_seed_assignment(ids, source_seed, sink_seed, box):
    y, x, y1, x1 = box
    return seed_assignment(ids[y:y1, x:x1], source_seed, sink_seed)


def core_assignments(ids, core_ids, minimum=64):
    good = (core_ids > 0) & (ids > 0)
    scale = int(ids.max()) + 1
    values, counts = np.unique(core_ids[good].astype(np.int64)*scale+ids[good], return_counts=True)
    result = {}
    for value, pixels in zip(values, counts):
        if pixels >= minimum:
            result.setdefault(int(value//scale), set()).add(int(value % scale))
    return result


def compare_core_assignments(before, after, adjacent):
    split_ids = [gid for gid, a in before.items() if len(after.get(gid, set())) > len(a)]
    lost = []
    for first, second in adjacent:
        a, b = before.get(first, set()), before.get(second, set())
        c, d = after.get(first, set()), after.get(second, set())
        if a and b and not a & b and c and d and c & d:
            lost.append([first, second])
    return dict(additional_known_core_fragment_GT=split_ids, lost_known_core_adjacent_separations=lost,
                interpretation='Known original cores only; not whole-grain accuracy or a local score.')


def instance_stats(ids, classes):
    sizes = np.bincount(ids.ravel())
    result = dict(instances=len(classes))
    for name, label in [('F', 1), ('P', 0)]:
        keys = [int(key) for key, value in classes.items() if value == label]
        a = sizes[keys]
        result[name] = dict(count=len(keys), pixels=int(a.sum()), under500=int((a < 500).sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--capture-manifest', type=Path)
    parser.add_argument('--sources', nargs='*')
    parser.add_argument('--fixed-view-manifest', type=Path)
    parser.add_argument('--seed-policy', choices=['legacy_skeleton', 'continuous_low'])
    parser.add_argument('--seed-low-threshold', type=float, default=.45)
    args = parser.parse_args()
    if args.code_root:
        sys.path.insert(0, str(args.code_root))
    from data.boundary_naf import load_processed_binary
    from utils.affinity_gap_loss import native_gap_geometry, select_gap_cuts, adjacent_pairs
    from tools.boundary_restore_readout import evaluate_frozen_readout
    from tools.boundary_restore_monitor import panels, _heat, _overlay
    from utils.config import load_config
    import torch
    torch.set_num_threads(2); cv2.setNumThreads(1)
    out = args.output.resolve()
    if (out/'complete.json').exists():
        raise FileExistsError('Do not overwrite completed preflight')
    out.mkdir(parents=True, exist_ok=True)
    (out/'cases').mkdir(exist_ok=True); (out/'gallery').mkdir(exist_ok=True)
    config = load_config(args.config)['boundary_naf']
    gts = read(config['manifest'])['rows']
    heats = {r['source']: r for r in read(config['real_input_manifest'])['files']}
    captures = {} if not args.capture_manifest else {r['source']:r for r in read(args.capture_manifest)['rows']
                    if r.get('source_group') == 'seen_training'}
    fixed = None if args.fixed_view_manifest is None else read(args.fixed_view_manifest)
    if fixed is not None and fixed.get('complete') is not True:
        raise ValueError('Fixed view manifest must bind a completed preflight')
    selector_settings = {} if args.seed_policy is None else dict(
        seed_policy=args.seed_policy, seed_low_threshold=args.seed_low_threshold)
    inputs = {args.config:sha(args.config), config['manifest']:sha(config['manifest']),
              config['real_input_manifest']:sha(config['real_input_manifest'])}
    if args.capture_manifest:
        inputs[str(args.capture_manifest)] = sha(args.capture_manifest)
    if args.fixed_view_manifest:
        inputs[str(args.fixed_view_manifest)] = sha(args.fixed_view_manifest)
    all_cases = []; counts = Counter(); start = time.monotonic(); source_reports = []
    # 全链缓存来源先处理，仅改变执行顺序，不改变各源两个视图选择。
    gts.sort(key=lambda row: (row['source'] not in captures, row['source']))
    for row in gts:
        name = row['source']
        if args.sources and name not in args.sources:
            continue
        for key in ['gt', 'classes']:
            expected = row['hashes'][key+'_sha256']
            assert sha(row[key]) == expected; inputs[row[key]] = expected
        heat = heats[name]
        assert sha(heat['heat_path']) == heat['heat_sha256']; inputs[heat['heat_path']] = heat['heat_sha256']
        with np.load(heat['heat_path'], allow_pickle=False) as z:
            B = z['boundary'].copy()
        with np.load(row['gt'], allow_pickle=False) as z:
            gt = {k:z[k].copy() for k in z.files}
        ids = gt['instance_map']; original = gt['original_covered'].astype(bool)
        target, valid, _ = load_processed_binary(row['gt'], row['classes'], boundary_dilation=0, valid_margin=4)
        geometry = native_gap_geometry(ids, original, target, valid, B, selector_settings)
        views = (choose_interface_views(ids, original, valid, B) if fixed is None
                 else fixed_views_for_source(fixed['cases'], name, B.shape))
        if fixed is not None:
            for path in [row['gt'], row['classes'], heat['heat_path']]:
                if fixed['inputs_sha256'].get(path) != inputs[path]:
                    raise RuntimeError('Fixed view source identity changed: '+path)
        adjacent = adjacent_pairs(ids, original, valid)
        full_cache = None; baseline_core = None; baseline_stats = None
        if name in captures:
            cap = captures[name]
            assert sha(cap['cache']) == cap['cache_sha256']; inputs[cap['cache']] = cap['cache_sha256']
            with np.load(cap['cache'], allow_pickle=False) as z:
                full_cache = {k:z[k].copy() for k in z.files if k != 'boundary_after'}
            assert np.array_equal(B, full_cache['boundary_before'])
            assert sha(cap['N1_png']) == cap['N1_png_sha256']
            assert sha(cap['N1_classes']) == cap['N1_classes_sha256']
            inputs[cap['N1_png']] = cap['N1_png_sha256']; inputs[cap['N1_classes']] = cap['N1_classes_sha256']
            baseline = cv2.imread(cap['N1_png'], cv2.IMREAD_UNCHANGED)
            base_classes = read(cap['N1_classes']); frozen = load_config(cap['config'])
            assert sha(cap['config']) == cap['config_sha256']; inputs[cap['config']] = cap['config_sha256']
            evaluate_frozen_readout(full_cache['semantic'], full_cache['raw_probability'], B, frozen,
                reference=dict(instances=baseline, classes=base_classes))
            baseline_core = core_assignments(baseline, geometry['core_ids'])
            baseline_stats = instance_stats(baseline, base_classes)
        for index, view in enumerate(views):
            box = view['box_yxyx']; y, x, y1, x1 = box
            local = crop_geometry(geometry, box); local_B = B[y:y1, x:x1]
            selected = select_gap_cuts(local_B, local, selector_settings, seed=20261027+index, gt_pair=view['gt_pair'])
            audit = selected['audit']; counts.update(audit['reasons'])
            case = dict(source=name, view=index, **view, selector=audit, full_readout_available=full_cache is not None)
            if selected['cut'].any():
                assert not np.any(selected['cut'] & ~local['allowed'])
                assert not np.any(selected['cut'] & (selected['source_seed'] | selected['sink_seed']))
                candidate = B.copy(); section = candidate[y:y1, x:x1]
                section[selected['cut']] = np.maximum(section[selected['cut']], np.float32(.70))
                assert int(((candidate != B) & ~original).sum()) == 0
                before_labels = cv2.connectedComponents((B < .65).astype(np.uint8), connectivity=8)[1]
                after_labels = cv2.connectedComponents((candidate < .65).astype(np.uint8), connectivity=8)[1]
                case['full_raw_before'] = full_seed_assignment(before_labels, selected['source_seed'], selected['sink_seed'], box)
                case['full_raw_after'] = full_seed_assignment(after_labels, selected['source_seed'], selected['sink_seed'], box)
                assert case['full_raw_after']['separated']
                case['raw8_new_separation'] = not case['full_raw_before']['separated']
                case['changed_pixels'] = int((candidate != B).sum())
                path = out/'cases'/f'{name}_{index}.npz'
                np.savez_compressed(path, box_yxyx=np.array(box), cut=selected['cut'],
                    source_seed=selected['source_seed'], sink_seed=selected['sink_seed'])
                case['case_npz'] = str(path); case['case_npz_sha256'] = sha(path)
                if full_cache is not None:
                    pred, classes, readout = evaluate_frozen_readout(full_cache['semantic'], full_cache['raw_probability'], candidate, frozen)
                    case['final_before'] = full_seed_assignment(baseline, selected['source_seed'], selected['sink_seed'], box)
                    case['final_after'] = full_seed_assignment(pred, selected['source_seed'], selected['sink_seed'], box)
                    case['baseline_stats'] = baseline_stats; case['candidate_stats'] = instance_stats(pred, classes)
                    case['known_core_safety'] = compare_core_assignments(baseline_core, core_assignments(pred, geometry['core_ids']), adjacent)
                    case['final_readout'] = readout
                    panels(out/'gallery'/f'{name}_{index}.jpg',name+' | ideal .70 complete cut | seen training GT',
                        [('RGB', full_cache['rgb'][y:y1, x:x1]), ('Recorded GT',_heat(target[y:y1,x:x1].astype(np.float32))),
                         ('Original B',_heat(local_B)), ('Ideal cut',_heat(candidate[y:y1,x:x1])),
                         ('Before outlines',_overlay(full_cache['rgb'][y:y1,x:x1],baseline[y:y1,x:x1])),
                         ('After outlines',_overlay(full_cache['rgb'][y:y1,x:x1],pred[y:y1,x:x1]))],240,
                         'Mechanism only; no test GT, no model, no local competition score.')
                else:
                    panels(out/'gallery'/f'{name}_{index}.jpg',name+' | ideal .70 complete cut | no final cache',
                        [('Recorded GT',_heat(target[y:y1,x:x1].astype(np.float32))),('Original B',_heat(local_B)),
                         ('Allowed original corridor',_heat(local['allowed'].astype(np.float32))),
                         ('Ideal cut B',_heat(candidate[y:y1,x:x1]))],240)
            all_cases.append(case)
            write(out/'progress.json',dict(complete=False,sources_completed=len(source_reports),cases=all_cases,reasons=dict(counts)))
            print(json.dumps(dict(source=name,view=index,selected=audit['selected_pairs'],reasons=audit['reasons'],
                final_after=case.get('final_after')),ensure_ascii=False),flush=True)
        source_reports.append(dict(source=name,views=len(views),full_cache_available=full_cache is not None,
            original_seed_pixels=int((geometry['seed_ids']>0).sum()), allowed_pixels=int(geometry['allowed'].sum())))
    for path, expected in inputs.items():
        assert sha(path) == expected, 'Read-only input changed: '+path
    selected_cases = [r for r in all_cases if r['selector']['selected_pairs']]
    final_cases = [r for r in selected_cases if 'final_after' in r]
    success = [r for r in final_cases if not r['final_before']['separated'] and r['final_after']['separated']]
    safe_success = [r for r in success if not r['known_core_safety']['lost_known_core_adjacent_separations']
                    and not r['known_core_safety']['additional_known_core_fragment_GT']]
    if fixed is not None:
        comparable = lambda c: (c['source'], c['view'], c['gt_pair'], c['box_yxyx'])
        expected = [comparable(c) for c in fixed['cases'] if not args.sources or c['source'] in args.sources]
        if [comparable(c) for c in all_cases] != expected:
            raise RuntimeError('Fixed case list/order changed')
    payload = dict(complete=True,source_count=len(source_reports),view_count=len(all_cases),
        selected_case_count=len(selected_cases),actual_final_cases=len(final_cases),
        raw8_newly_separated_cases=sum(r.get('raw8_new_separation',False) for r in selected_cases),
        final_newly_separated_cases=len(success),final_newly_separated_without_known_core_regression=len(safe_success),
        gate_status='requires_root_review' if safe_success else 'not_passed_actual_final_gate',
        cases=all_cases,sources=source_reports,reasons=dict(counts),inputs_sha256=inputs,
        source_selection=('Exact original completed case manifest, no new view selection.' if fixed is not None else
                          'Each source top two weak OO interfaces, distinct centers >=128px; chosen before intervention; at most4 static atom pair attempts per view.'),
        fixed_view_manifest_sha256=sha(args.fixed_view_manifest) if fixed is not None else None,
        settings=dict(native_roi_size=256,threshold=.65,target_margin=.70,gt_margin=12,
                      source='original only; filled/unknown not editable',seed=20261027,
                      seed_policy=args.seed_policy or 'legacy_skeleton',seed_low_threshold=args.seed_low_threshold),
        elapsed_seconds=time.monotonic()-start,script_sha256=sha(__file__),
        selector_code_sha256=sha(Path(select_gap_cuts.__code__.co_filename)),
        no_network_forward=True,no_training=True,no_test_GT=True,no_full150=True,
        no_new_proxy_score=True,all_GT_seen_in_training=True,
        limitations=['Known-core safety is not complete physical truth; processed GT may omit boundaries.',
                     'No full-output evidence for sources without a pre-existing capture; raw8 is not actual watershed.',
                     'Perfect intervention is a mechanism gate, not demonstrated learning or official gain.'])
    write(out/'complete.json',payload)
    print(json.dumps({k:payload[k] for k in ['complete','source_count','view_count','selected_case_count','actual_final_cases',
        'final_newly_separated_cases','final_newly_separated_without_known_core_regression','gate_status','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
