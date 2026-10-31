# -*- coding: utf-8 -*-
"""原R1与固定锚点反事实的真实训练网格监督覆盖；仅CPU，不训练。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import probe_affinity_signal as original

BASELINE_SHA = 'da9e312417152e8f7da25bfb4018ca7972fe659129438537824bf66fa47b5e03'
TRIAL_SHA = '9a207cc9bd8953eb0d807c288c059a1aff951a06fd26c0a1507a163b4d56178b'
METADATA_SHA = '33dd94a24ad720d9e52edaa83facf39733210593b9770b0d290a8978ec6c3b07'
SIGNAL_CODE_SHA = 'b890ac1341f34f01a99c0f94e7ae06f2add55c2868038887e712bfe5a6cfd3d9'
METRICS = ('crop_hits', 'crop_contact_edges', 'cardinal_hits', 'short_hits', 'any_hits',
           'cardinal_edges', 'short_edges', 'any_edges', 'short_border_edges',
           'short_interior_edges', 'any_border_edges', 'any_interior_edges')
SIZE_BINS = ((3, '1-3'), (15, '4-15'), (63, '16-63'), (255, '64-255'), (65536**2, '256+'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


def validate_labels(ids, classes):
    ids = np.asarray(ids)
    if ids.ndim != 2 or min(ids.shape) < 1 or not np.issubdtype(ids.dtype, np.integer) or ids.min() < 0 or ids.max() > 65535:
        raise ValueError('GT必须是二维0..65535整数图')
    lookup = np.full(65536, -1, dtype=np.int8)
    for key, phase in classes.items():
        if isinstance(key, (bool, np.bool_)) or isinstance(phase, (bool, np.bool_)):
            raise ValueError('GT ID及类别不能为bool')
        if not isinstance(phase, (int, np.integer)) or int(phase) not in (0, 1):
            raise ValueError('类别必须P=0/F=1整数')
        index = int(key)
        if index <= 0 or index > 65535 or lookup[index] != -1:
            raise ValueError('重复或无效类别ID')
        lookup[index] = int(phase)
    if np.any(lookup[np.unique(ids[ids > 0])] < 0):
        raise ValueError('完整正GT必须有canonical类别')
    return lookup


def packed_pairs(a, b):
    lo = np.minimum(a, b).astype(np.uint64, copy=False)
    hi = np.maximum(a, b).astype(np.uint64, copy=False)
    return lo*65536+hi


def counted(codes):
    return np.unique(codes, return_counts=True)


def cardinal_contacts(ids, origin=(0, 0), full_shape=None):
    """真实原生四邻接触边；返回pair码及all/border256/interior数量。"""
    ids = np.asarray(ids)
    height, width = ids.shape
    full_shape = tuple(full_shape or ids.shape)
    keys, bands = [], []
    for dy, dx in ((0, 1), (1, 0)):
        if height <= dy or width <= dx:
            continue
        a = ids[:height-dy, :width-dx]
        b = ids[dy:, dx:]
        ys, xs = np.nonzero((a > 0) & (b > 0) & (a != b))
        keys.append(packed_pairs(a[ys, xs], b[ys, xs]))
        yy, xx = ys+origin[0]+dy/2, xs+origin[1]+dx/2
        distance = np.minimum.reduce((yy, xx, full_shape[0]-1-yy, full_shape[1]-1-xx))
        bands.append(distance < 256)
    if not keys:
        return np.zeros(0, np.uint64), np.zeros((0, 3), np.int64)
    codes, inverse = np.unique(np.concatenate(keys), return_inverse=True)
    band = np.concatenate(bands)
    all_counts = np.bincount(inverse, minlength=len(codes))
    border = np.bincount(inverse[band], minlength=len(codes))
    return codes, np.column_stack((all_counts, border, all_counts-border)).astype(np.int64)


def geometry_grid(full_gt, box, draw, functions, *, input_size=1024, output_grid=512):
    """严格按原geometry函数和原spatial函数，不建立中间1024标签。"""
    if len(box) != 4 or any(isinstance(x, bool) or int(x) != x for x in box):
        raise ValueError('原生框必须是四个整数')
    y, x, y1, x1 = map(int, box)
    if not (0 <= y < y1 <= full_gt.shape[0] and 0 <= x < x1 <= full_gt.shape[1]):
        raise ValueError('裁框越界')
    native = full_gt[y:y1, x:x1]
    grid, valid, metadata = functions['letterbox_instance_geometry'](native, input_size, output_grid)
    hf, vf, k = draw['horizontal_flip'][0], draw['vertical_flip'][0], draw['rotation_k'][0]
    if not isinstance(hf, bool) or not isinstance(vf, bool) or int(k) not in range(4):
        raise ValueError('原增强回执非法')
    transform = functions['_spatial_transform']
    aligned = transform(torch.from_numpy(grid), hf, vf, k).numpy()
    content = transform(torch.from_numpy(valid), hf, vf, k).numpy()
    # cv2.INTER_NEAREST取样的原生位置；仅用于关系中点的空间分层，不造target。
    h, w = metadata.content_height, metadata.content_width
    ymap = np.full(grid.shape, -1, np.int32); xmap = np.full(grid.shape, -1, np.int32)
    ymap[:h, :w] = (y+np.floor(np.arange(h)*native.shape[0]/h).astype(np.int32))[:, None]
    xmap[:h, :w] = (x+np.floor(np.arange(w)*native.shape[1]/w).astype(np.int32))[None, :]
    ymap = transform(torch.from_numpy(ymap), hf, vf, k).numpy()
    xmap = transform(torch.from_numpy(xmap), hf, vf, k).numpy()
    if not np.array_equal(aligned[content], full_gt[ymap[content], xmap[content]]):
        raise RuntimeError('nearest原像素坐标映射与实际grid不一致')
    return native, aligned, content, metadata, ymap, xmap


def supervision_counts(grid, valid, ymap, xmap, full_shape, functions):
    """调用实际8offset target构造器并逐通道校核，聚合unordered GT pair。"""
    grid = np.asarray(grid); valid = np.asarray(valid)
    if valid.dtype != np.bool_ or valid.shape != grid.shape:
        raise ValueError('content必须是对齐bool图')
    target, edge_valid, unknown = functions['build_affinity_targets_torch'](
        torch.from_numpy(grid.astype(np.int64, copy=True))[None], torch.from_numpy(valid.copy())[None, None],
        uncovered_as_boundary=torch.zeros(1, dtype=torch.bool), return_uncovered_mask=True)
    if bool(unknown.any()):
        raise RuntimeError('未知区不应进入监督')
    target, edge_valid = target.numpy()[0], edge_valid.numpy()[0]
    tables = []
    for channel, (dy, dx) in enumerate(functions['DEFAULT_AFFINITY_OFFSETS']):
        src, dst = functions['_edge_slices'](*grid.shape, dy, dx)
        a, b = grid[src], grid[dst]
        expected = valid[src] & valid[dst] & (a > 0) & (b > 0)
        whole_expected = np.zeros(grid.shape, bool); whole_expected[src] = expected
        whole_target = np.zeros(grid.shape, np.float32); whole_target[src] = expected & (a == b)
        if not np.array_equal(edge_valid[channel], whole_expected) or not np.array_equal(target[channel], whole_target):
            raise RuntimeError('实际8offset目标/有效域不符合原已知端点合同')
        negative = expected & (a != b)
        keys = packed_pairs(a[negative], b[negative])
        yy = (ymap[src][negative].astype(np.float64)+ymap[dst][negative])/2
        xx = (xmap[src][negative].astype(np.float64)+xmap[dst][negative])/2
        border = np.minimum.reduce((yy, xx, full_shape[0]-1-yy, full_shape[1]-1-xx)) < 256
        codes, inv = np.unique(keys, return_inverse=True)
        n = np.bincount(inv, minlength=len(codes)); nb = np.bincount(inv[border], minlength=len(codes))
        tables.append((codes, n, nb, channel))
    union = np.unique(np.concatenate([v[0] for v in tables]))
    # cardinal,short,any及short/all的端点空间分层，均是关系条目数。
    values = np.zeros((len(union), 7), np.int64)
    for codes, n, nb, channel in tables:
        pos = np.searchsorted(union, codes)
        values[pos, 2] += n; values[pos, 5] += nb; values[pos, 6] += n-nb
        if channel < 4:
            values[pos, 1] += n; values[pos, 3] += nb; values[pos, 4] += n-nb
        if channel < 2:
            values[pos, 0] += n
    return union, values


def view_coverage(full_gt, box, draw, classes, functions, *, output_grid=512, input_size=1024, validated=False):
    if not validated:
        validate_labels(full_gt, classes)
    native, grid, valid, meta, yy, xx = geometry_grid(full_gt, box, draw, functions, input_size=input_size, output_grid=output_grid)
    cropped_codes, cropped_counts = cardinal_contacts(native, box[:2], full_gt.shape)
    codes, values = supervision_counts(grid, valid, yy, xx, full_gt.shape, functions)
    native_sizes = np.bincount(native.ravel().astype(np.int64), minlength=65536)
    grid_sizes = np.bincount(grid[valid].ravel().astype(np.int64), minlength=65536)
    lost_ids = np.flatnonzero((native_sizes > 0) & (grid_sizes == 0)); lost_ids = lost_ids[lost_ids > 0]
    return dict(crop_codes=cropped_codes, crop_counts=cropped_counts, codes=codes, values=values,
                native_sizes=native_sizes, grid_sizes=grid_sizes, lost_ids=lost_ids,
                grid_sha256=array_sha(grid), content_sha256=array_sha(valid), geometry=meta.to_dict(),
                exact_factor2=bool(native.shape == (1024, 1024) and meta.content_height == meta.content_width == 512))


def phase_names(codes, lookup):
    a, b = lookup[(codes//65536).astype(np.int64)], lookup[(codes%65536).astype(np.int64)]
    if np.any(a < 0) or np.any(b < 0):
        raise RuntimeError('关系端点缺canonical类别')
    return np.where((a == 1) & (b == 1), 'F/F', np.where((a == 0) & (b == 0), 'P/P', 'F/P'))


def align_counts(cohort, result):
    out = np.zeros((len(cohort), len(METRICS)), np.int64)
    for codes, vals, kind in ((result['crop_codes'], result['crop_counts'], 'crop'), (result['codes'], result['values'], 'grid')):
        pos = np.searchsorted(cohort, codes)
        keep = pos < len(cohort)
        keep[keep] &= cohort[pos[keep]] == codes[keep]
        pos, vals = pos[keep], vals[keep]
        if kind == 'crop':
            out[pos, 0] = vals[:, 0] > 0; out[pos, 1] = vals[:, 0]
        else:
            out[pos, 2:5] = vals[:, :3] > 0; out[pos, 5:8] = vals[:, :3]
            out[pos, 8:] = vals[:, 3:]
    return out


def pair_summary(old, candidate, mask):
    a, b = old[mask], candidate[mask]
    result = dict(pairs=int(mask.sum()), baseline={}, candidate={}, transitions={})
    for arm, data in (('baseline', a), ('candidate', b)):
        result[arm] = {name: int(data[:, i].sum()) for i, name in enumerate(METRICS)}
        for name in ('crop_hits', 'cardinal_hits', 'short_hits', 'any_hits'):
            index = METRICS.index(name)
            result[arm]['never_'+name] = int((data[:, index] == 0).sum())
    for name in ('crop_hits', 'cardinal_hits', 'short_hits', 'any_hits'):
        i = METRICS.index(name)
        result['transitions'][name] = dict(gained_from_zero=int(((a[:, i] == 0) & (b[:, i] > 0)).sum()),
            new_zero=int(((a[:, i] > 0) & (b[:, i] == 0)).sum()),
            increase=int((b[:, i] > a[:, i]).sum()), decrease=int((b[:, i] < a[:, i]).sum()), same=int((b[:, i] == a[:, i]).sum()))
    # 原生crop的新获覆盖不保证在512格仍有短/长负关系，显式保留子集。
    for label, selected in (('raw_crop_rescued', (a[:, 0] == 0) & (b[:, 0] > 0)),
                            ('raw_crop_new_zero', (a[:, 0] > 0) & (b[:, 0] == 0))):
        result[label] = dict(pairs=int(selected.sum()),
            candidate_cardinal_zero=int((selected & (b[:, 2] == 0)).sum()),
            candidate_short_zero=int((selected & (b[:, 3] == 0)).sum()),
            candidate_any_zero=int((selected & (b[:, 4] == 0)).sum()),
            baseline_short_nonzero=int((selected & (a[:, 3] > 0)).sum()),
            baseline_any_nonzero=int((selected & (a[:, 4] > 0)).sum()),
            baseline_short_edges=int(a[selected, 6].sum()), candidate_short_edges=int(b[selected, 6].sum()),
            baseline_any_edges=int(a[selected, 7].sum()), candidate_any_edges=int(b[selected, 7].sum()))
    return result


def check_files(files):
    for path, digest in files.items():
        if original.sha(path) != digest:
            raise RuntimeError('封存父来源/GT/回执改变: '+str(path))


def preflight(r1_root, baseline, trial, capture):
    if original.sha(baseline/'report.json') != BASELINE_SHA or original.sha(trial/'report.json') != TRIAL_SHA or original.sha(trial/'metadata_receipts.jsonl') != METADATA_SHA:
        raise RuntimeError('仅允许已验收的单预设坐标反事实')
    if original.sha(ROOT/'tools/probe_affinity_signal.py') != SIGNAL_CODE_SHA:
        raise RuntimeError('封存AST提取器改变')
    manifest = read(capture/'capture_manifest.json'); record = read(capture/'capture.json')
    if original.sha(capture/'capture.json') != original.CAPTURE_SHA or original.sha(capture/'capture_manifest.json') != original.CAPTURE_MANIFEST_SHA:
        raise RuntimeError('原四draw捕获身份改变')
    status = read(r1_root/'status.json'); pipeline = read(r1_root.parent/'pipeline_status.json')
    tail_pipeline = read(ROOT/'outputs/affinity_tail/pipeline_status.json')
    if status['status'] != 'complete' or status['updates'] != 1280 or status['epoch'] != 20 or status['final_checkpoint_sha256'] != original.R1_SHA or len(pipeline['sources']) != 49 or len(tail_pipeline['sources']) != 148:
        raise RuntimeError('R1完成状态或原来源闭包不同')
    for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'):
        if status.get(key) is not True:
            raise RuntimeError('R1冻结/重载门禁缺失')
    if original.sha(r1_root.parent/'pipeline_status.json') != original.R1_PIPELINE_SHA or original.sha(r1_root/'final.pt') != original.R1_SHA:
        raise RuntimeError('R1管线/完整权重身份不同')
    parent = read(baseline/'report.json'); candidate = read(trial/'report.json')
    if candidate['seed'] != 20271002 or candidate['newgt'] != parent['newgt'] or not candidate['fixed_receipts_unchanged'] or not candidate['native640_anchor320_uniform320']:
        raise RuntimeError('反事实seed/数据/1280采样合同不同')
    if record['newgt']['original_covered_is_supervision_mask'] is not False or record['newgt']['filled_and_original_equal_weight'] is not True:
        raise RuntimeError('必须完整processed新GT且含filled')
    for name in ('npz_cohort_sha256', 'class_cohort_sha256', 'manifest_sha256'):
        if record['newgt'][name] != parent['newgt'][name]:
            raise RuntimeError('完整GT cohort身份不符')
    snapshot = r1_root.parent/'source'
    protected = {Path(__file__): original.sha(__file__), ROOT/'tools/probe_affinity_signal.py': SIGNAL_CODE_SHA,
        capture/'capture.json': original.CAPTURE_SHA, capture/'capture_manifest.json': original.CAPTURE_MANIFEST_SHA,
        r1_root/'final.pt': original.R1_SHA, r1_root/'steps.jsonl': original.sha(r1_root/'steps.jsonl'),
        r1_root/'status.json': original.sha(r1_root/'status.json'), r1_root.parent/'pipeline_status.json': original.R1_PIPELINE_SHA,
        ROOT/'outputs/affinity_tail/pipeline_status.json': original.sha(ROOT/'outputs/affinity_tail/pipeline_status.json'),
        trial/'report.json': TRIAL_SHA, trial/'metadata_receipts.jsonl': METADATA_SHA, baseline/'report.json': BASELINE_SHA}
    for rel, digest in candidate['protected_sha256'].items():
        protected[original.safe_child(ROOT, rel)] = digest
    for rel, digest in pipeline['sources'].items():
        protected[original.safe_child(snapshot, rel)] = digest
    for rel, digest in tail_pipeline['sources'].items():
        protected[original.safe_child(ROOT, rel)] = digest
        protected[original.safe_child(ROOT/'outputs/affinity_tail/source', rel)] = digest
    for rel, digest in manifest['files'].items():
        protected[original.safe_child(capture, rel)] = digest
    for sample in record['newgt']['samples']:
        for rel, digest in sample['files_sha256'].items():
            protected[original.safe_child(ROOT, rel)] = digest
    check_files(protected)
    runtime = original.verify_runtime(record['sealed_runtime'])
    for name, names in original.FUNCTIONS.items():
        if [ast.dump(x, include_attributes=False) for x in original.nodes(snapshot/name, names)] != [ast.dump(x, include_attributes=False) for x in original.nodes(ROOT/'outputs/affinity_tail/source'/name, names)]:
            raise RuntimeError('原49与缓存148纯函数不同')
    old = [json.loads(x) for x in (r1_root/'steps.jsonl').read_text().splitlines()]
    new = [json.loads(x) for x in (trial/'metadata_receipts.jsonl').read_text().splitlines()]
    if len(old) != 1280 or len(new) != 1280 or [x['updates'] for x in old] != list(range(1, 1281)):
        raise RuntimeError('原完整1280回执缺失')
    epochs = defaultdict(Counter)
    for a, b in zip(old, new):
        anchor = b.get('counterfactual_anchor', {}).get('selected', False)
        for key in ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw', 'pseudo_crop', 'training_view', 'updates', 'auxiliary_weight'):
            if a[key] != b[key]:
                raise RuntimeError('反事实改变固定字段: '+key)
        if a['crop_attempts'][1] != b['crop_attempts'][1] or (not anchor and (a['manual_crop'] != b['manual_crop'] or a['crop_attempts'] != b['crop_attempts'])):
            raise RuntimeError('保留的框/伪标签门禁改变')
        if anchor and (a['training_view'] != 'native1024' or b['counterfactual_anchor']['known_cardinal_pairs'] < 16):
            raise RuntimeError('锚点门禁不符')
        if a['training_view'] == 'native1024':
            epochs[a['epoch']]['anchor' if anchor else 'uniform'] += 1
    if len(epochs) != 20 or any(v != Counter(anchor=16, uniform=16) for v in epochs.values()):
        raise RuntimeError('预定每轮16+16变化')
    return record, old, new, original.load_functions(snapshot), protected, runtime


def run(r1_root, baseline, trial, capture, output):
    r1_root, baseline, trial, capture, output = [Path(x).resolve() for x in (r1_root, baseline, trial, capture, output)]
    if output.exists() or output != ROOT/'outputs/affinity_coverage_grid':
        raise RuntimeError('只允许新的outputs/affinity_coverage_grid，不覆盖已接受结果')
    record, old_receipts, new_receipts, functions, protected, runtime_before = preflight(r1_root, baseline, trial, capture)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    output.mkdir(); (output/'sources').mkdir(); start = time.perf_counter()
    full_old = read(baseline/'report.json'); full_trial = read(trial/'report.json')
    csv_old = {}; csv_trial = {}
    for folder, table in ((baseline, csv_old), (trial, csv_trial)):
        with (folder/'contacts.csv').open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if row['zone'] == 'all':
                    table[(row['source'], int(row['gt_a']), int(row['gt_b']))] = row
    if len(csv_old) != 12821 or set(csv_old) != set(csv_trial):
        raise RuntimeError('完整原生接触cohort改变')
    draws = []; pairs = []; summaries = []; parity = defaultdict(Counter); disappearing = defaultdict(Counter)
    exact_draws = []; cache_hits = 0; actual_builder_calls = 0
    for sample in sorted(record['newgt']['samples'], key=lambda x:x['source']):
        source = sample['source']; raw_path = next(original.safe_child(ROOT, n) for n in sample['files_sha256'] if n.endswith('_gt.npz'))
        with np.load(raw_path, allow_pickle=False) as data:
            gt = data['instance_map'].copy()
        classes = read(raw_path.with_name(source+'_class.json')); lookup = validate_labels(gt, classes)
        full_sizes = np.bincount(gt.ravel().astype(np.int64), minlength=65536)
        cohort, contacts = cardinal_contacts(gt); phases = phase_names(cohort, lookup)
        totals = {arm: np.zeros((len(cohort), len(METRICS)), np.int64) for arm in ('baseline', 'candidate')}
        # Whole曝露单独，避免native零曝被误说成从未监督。
        whole_totals = np.zeros((len(cohort), len(METRICS)), np.int64)
        extra = {arm: defaultdict(lambda: np.zeros(4, np.int64)) for arm in totals}
        cache = {}; rows = [(a, b) for a, b in zip(old_receipts, new_receipts) if a['manual'] == [source+'.jpg']]
        if len(rows) != 40:
            raise RuntimeError('每源原40曝光改变')
        for a, b in rows:
            results = []
            for arm, receipt in (('baseline', a), ('candidate', b)):
                box = receipt['manual_crop'][0]; draw = receipt['manual_draw']
                key = (*box, draw['horizontal_flip'][0], draw['vertical_flip'][0], draw['rotation_k'][0])
                if key not in cache:
                    cache[key] = view_coverage(gt, box, draw, classes, functions, validated=True); actual_builder_calls += 1
                else:
                    cache_hits += 1
                result = cache[key]; results.append(result)
                aligned = align_counts(cohort, result)
                if receipt['training_view'] == 'native1024':
                    totals[arm] += aligned
                    parity[(arm, 'all_native')][f'{box[0]%2}{box[1]%2}'] += 1
                    if b.get('counterfactual_anchor', {}).get('selected', False):
                        parity[(arm, 'anchor_replaced_slots')][f'{box[0]%2}{box[1]%2}'] += 1
                    for code, values in zip(result['codes'], result['values']):
                        i = np.searchsorted(cohort, code)
                        if i == len(cohort) or cohort[i] != code:
                            extra[arm][int(code)] += np.array([values[0] > 0, values[1] > 0, values[2] > 0, values[2]], np.int64)
                elif arm == 'baseline':
                    whole_totals += aligned
                for iid in np.flatnonzero(result['native_sizes'][1:] > 0)+1:
                    size = int(result['native_sizes'][iid]); label = next(name for bound, name in SIZE_BINS if size <= bound)
                    phase = 'F' if lookup[iid] == 1 else 'P'
                    c = disappearing[(arm, receipt['training_view'], phase, label)]
                    c['native_visible_id_exposures'] += 1
                    c['grid_lost_id_exposures'] += int(result['grid_sizes'][iid] == 0)
                    c['crop_fragment_id_exposures'] += int(size < int(full_sizes[iid]))
                    c['grid_lost_full_gt_support_id_exposures'] += int(result['grid_sizes'][iid] == 0 and size == int(full_sizes[iid]))
                    c['grid_lost_crop_fragment_id_exposures'] += int(result['grid_sizes'][iid] == 0 and size < int(full_sizes[iid]))
                item = next((q for q in record['rows'] if q['draw'] == draw['draw_index'][0] and receipt['epoch'] == 1), None)
                if item is not None and arm == 'baseline':
                    with np.load(capture/item['input_file'], allow_pickle=False) as blob:
                        _, grid, valid, _, _, _ = geometry_grid(gt, box, draw, functions)
                        if not np.array_equal(blob['grid_gt'], grid) or not np.array_equal(blob['content'], valid):
                            raise RuntimeError('原四draw真实grid/content不能精确回放')
                    exact_draws.append(item['draw'])
            if a['manual_crop'] == b['manual_crop'] and any(results[0][k] != results[1][k] for k in ('grid_sha256', 'content_sha256')):
                raise RuntimeError('不变框grid身份改变')
            draws.append(dict(source=source, epoch=a['epoch'], update=a['updates'], view=a['training_view'],
                anchor=b.get('counterfactual_anchor', {}).get('selected', False),
                baseline_grid_sha256=results[0]['grid_sha256'], candidate_grid_sha256=results[1]['grid_sha256'],
                baseline_content_sha256=results[0]['content_sha256'], candidate_content_sha256=results[1]['content_sha256'],
                baseline_lost_ids=len(results[0]['lost_ids']), candidate_lost_ids=len(results[1]['lost_ids']),
                baseline_lost_gt_ids=results[0]['lost_ids'].tolist(), candidate_lost_gt_ids=results[1]['lost_ids'].tolist(),
                baseline_exact_factor2=results[0]['exact_factor2'], candidate_exact_factor2=results[1]['exact_factor2']))
        for i, code in enumerate(cohort):
            ga, gb = int(code//65536), int(code%65536); key = (source, ga, gb)
            for arm, field in (('baseline', ''), ('candidate', 'candidate_')):
                expected = csv_old[key] if arm == 'baseline' else csv_trial[key]
                if int(contacts[i, 0]) != int(expected['contact_edges']) or totals[arm][i, 0] != int(expected[field+'any_draws']) or totals[arm][i, 1] != int(expected[field+'edge_exposure_total']):
                    raise RuntimeError('原生接触counts与已封存CSV不一致')
            row = dict(source=source, gt_a=ga, gt_b=gb, phase=str(phases[i]),
                original_contact_edges=int(contacts[i, 0]), original_border_contact_edges=int(contacts[i, 1]), original_interior_contact_edges=int(contacts[i, 2]))
            for arm in totals:
                row.update({arm+'_'+name: int(totals[arm][i, j]) for j, name in enumerate(METRICS)})
            row.update({'whole_'+name: int(whole_totals[i, j]) for j, name in enumerate(METRICS)})
            pairs.append(row)
        summary = dict(source=source, shape=list(gt.shape), native_draws=sum(a['training_view']=='native1024' for a,b in rows),
            whole_draws=sum(a['training_view']=='whole' for a,b in rows),
            phases={phase: pair_summary(totals['baseline'], totals['candidate'], phases == phase) for phase in ('F/F', 'F/P', 'P/P')},
            grid_relations_without_original_cardinal_contact={arm:[dict(gt_a=c//65536, gt_b=c%65536,
                phase=str(phase_names(np.array([c], np.uint64), lookup)[0]),
                cardinal_exposures=int(v[0]), short_exposures=int(v[1]), any_exposures=int(v[2]), any_edges=int(v[3])) for c,v in sorted(extra[arm].items())] for arm in totals})
        write(output/'sources'/f'{source}.json', summary); summaries.append(summary)
        print(json.dumps(dict(source=source, complete=True, cpu_only=True, builder_calls=actual_builder_calls)), flush=True)
    if sorted(exact_draws) != sorted(original.EXPECTED_DRAWS) or len(draws) != 1280 or len(pairs) != 12821:
        raise RuntimeError('实际1280/32源/四draw gate不完整')
    with (output/'pairs.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    write(output/'draws.json', draws)
    aggregate = {}
    old = np.array([[q['baseline_'+k] for k in METRICS] for q in pairs], np.int64)
    new = np.array([[q['candidate_'+k] for k in METRICS] for q in pairs], np.int64)
    for phase in ('F/F', 'F/P', 'P/P'):
        mask = np.array([q['phase'] == phase for q in pairs])
        aggregate[phase] = {zone: pair_summary(old, new, mask & zone_mask) for zone, zone_mask in (
            ('all', np.ones(len(pairs), bool)), ('has_original_border_contact', np.array([q['original_border_contact_edges'] > 0 for q in pairs])),
            ('has_original_interior_contact', np.array([q['original_interior_contact_edges'] > 0 for q in pairs])))}
    check_files(protected); runtime_after = original.verify_runtime(record['sealed_runtime'])
    if runtime_before != runtime_after:
        raise RuntimeError('实际CPU环境改变')
    report = dict(complete=True, format='affinity_coverage_grid_v1', no_gpu=True, no_training=True, no_optimizer=True, no_model_load=True,
        checkpoint_sha256=original.R1_SHA, epochs=20, original_updates=1280, original49_sources=49, cached148_sources=148,
        parents=dict(crop_audit_sha256=BASELINE_SHA, trial_sha256=TRIAL_SHA, trial_metadata_sha256=METADATA_SHA,
            original_steps_sha256=original.sha(r1_root/'steps.jsonl'), capture_sha256=original.CAPTURE_SHA),
        full_processed_newgt_including_filled=True, newgt=full_trial['newgt'], actual_four_draw_grid_content_exact=sorted(exact_draws),
        actual_target_mask_gate=True, builder_calls=actual_builder_calls, cache_hits=cache_hits,
        sources=summaries, paired_native_coverage=aggregate,
        origin_parity={arm:{group:dict(parity[(arm, group)]) for group in ('all_native', 'anchor_replaced_slots')} for arm in ('baseline', 'candidate')},
        grid_id_disappearance=[dict(arm=k[0], view=k[1], phase=k[2], native_crop_area_bin=k[3], **dict(v)) for k,v in sorted(disappearing.items())],
        protected_sha256={str(k):v for k,v in protected.items()}, protected_before_after_exact=True, runtime=runtime_after,
        elapsed_seconds=time.perf_counter()-start,
        scope='actual GT crop->original1024 letterbox metadata->512 nearest grid->original hf/vf/rotation->actual8offset targets, metadata counterfactual only',
        limits=['No image/restoration/encoder forward, optimizer or parameters; not evidence of model improvement.',
            'Native GT cardinal-contact cohort is fixed before cropping; generated grid endpoint pairs without original contact are separate.',
            'Pair cohorts having border and interior contact overlap; do not add those cohort counts.',
            'Actual negative relation spatial bands use original endpoint midpoints, not a traced physical interface or arc length.',
            'Grid loss of an ID may be a tiny cropped fragment; not automatically disappearance of a complete physical grain.',
            'Origin parity is descriptive; no added jitter, phase correction, or changed predeclared seed.',
            'Unknown/padding pairs are ignored; long relation validity remains known endpoints only.',
            'Whole exposures are separate; zero native target exposure does not mean never trained.'])
    write(output/'report.json', report)
    print(json.dumps(dict(complete=True, report_sha256=original.sha(output/'report.json'), elapsed_seconds=report['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r1-root', default='outputs/affinity_balance/mixed')
    parser.add_argument('--baseline', default='outputs/affinity_crop_audit')
    parser.add_argument('--trial', default='outputs/affinity_crop_trial')
    parser.add_argument('--capture', default='outputs/affinity_tail/analysis/partition')
    parser.add_argument('--output', default='outputs/affinity_coverage_grid')
    args = parser.parse_args()
    run(*(getattr(args, name) for name in ('r1_root', 'baseline', 'trial', 'capture', 'output')))


if __name__ == '__main__':
    main()
