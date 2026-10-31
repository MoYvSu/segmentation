# -*- coding: utf-8 -*-
"""固定训练GT界面的网格保真诊断；单源CPU重放，不加载模型、不训练。"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import probe_affinity_coverage as coverage
from tools import probe_affinity_signal as original

GRID_REPORT_SHA = 'af42aaaec32f036f031207a39e7f4d3d7a01076b5fd5663e6c3299e8dbe2b759'
RADII = (0., 1., 2., 4., 8.)
GROUPS = {'cardinal': (0, 1), 'short': (0, 1, 2, 3),
          'long': (4, 5, 6, 7), 'any': tuple(range(8))}


def contact_edges(ids, pair, origin=(0, 0)):
    """每条原生四邻接触只计一次；未知0绝不成为接触端点。"""
    ids = np.asarray(ids)
    a, b = sorted(map(int, pair))
    if a <= 0 or a == b or ids.ndim != 2:
        raise ValueError('需要两个不同正GT ID及二维实例图')
    rows = []
    for dy, dx in ((0, 1), (1, 0)):
        if ids.shape[0] <= dy or ids.shape[1] <= dx:
            continue
        left = ids[:ids.shape[0]-dy, :ids.shape[1]-dx]
        right = ids[dy:, dx:]
        y, x = np.nonzero(((left == a) & (right == b)) | ((left == b) & (right == a)))
        first = np.column_stack((y + origin[0], x + origin[1]))
        rows.append(np.stack((first, first + (dy, dx)), axis=1))
    return np.concatenate(rows) if rows else np.zeros((0, 2, 2), np.int64)


def in_box(edges, box):
    y, x, y1, x1 = map(int, box)
    return ((edges[:, :, 0] >= y) & (edges[:, :, 0] < y1)
            & (edges[:, :, 1] >= x) & (edges[:, :, 1] < x1)).all(axis=1)


def nearest_distance(query, reference):
    """两组原生坐标中点的欧氏最近距离；无参照时保留inf，汇总时记缺失。"""
    query = np.asarray(query, np.float64).reshape(-1, 2)
    reference = np.asarray(reference, np.float64).reshape(-1, 2)
    if not len(reference):
        return np.full(len(query), np.inf)
    if not len(query):
        return np.zeros(0, np.float64)
    # 仅固定界面的稀疏点，不建立全图距离矩阵。
    chunks = []
    for first in range(0, len(query), 512):
        delta = query[first:first+512, None] - reference[None]
        chunks.append(np.sqrt(np.min(np.sum(delta * delta, axis=-1), axis=1)))
    return np.concatenate(chunks)


def distance_summary(values):
    values = np.asarray(values, np.float64)
    finite = values[np.isfinite(values)]
    return dict(count=len(values), finite_count=len(finite), no_reference_count=int((~np.isfinite(values)).sum()),
                quantiles={str(q): float(np.quantile(finite, q)) if len(finite) else None
                           for q in (0., .25, .5, .75, .9, 1.)},
                within_native_pixels={str(r): dict(count=int((values <= r).sum()),
                    fraction=float((values <= r).mean()) if len(values) else None) for r in RADII})


def inverse_spatial(grid, draw):
    """先撤销旋转，再撤销翻转，恢复增强前的训练网格。"""
    result = np.rot90(np.asarray(grid), -int(draw['rotation_k'][0]))
    if draw['vertical_flip'][0]:
        result = np.flip(result, axis=-2)
    if draw['horizontal_flip'][0]:
        result = np.flip(result, axis=-1)
    return np.ascontiguousarray(result)


def pair_targets(grid, valid, ymap, xmap, full_gt, original_covered, pair, functions):
    """使用封存的真实目标构造器；长距仍按原合同只检查两个端点。"""
    target, edge_valid, unknown = functions['build_affinity_targets_torch'](
        torch.from_numpy(grid.astype(np.int64, copy=True))[None],
        torch.from_numpy(valid.copy())[None, None], uncovered_as_boundary=torch.zeros(1, dtype=torch.bool),
        return_uncovered_mask=True)
    if bool(unknown.any()):
        raise RuntimeError('未知区域不得成为boundary负监督')
    target, edge_valid = target.numpy()[0], edge_valid.numpy()[0]
    rows, points = [], []
    a, b = sorted(map(int, pair))
    for channel, (dy, dx) in enumerate(functions['DEFAULT_AFFINITY_OFFSETS']):
        src, dst = functions['_edge_slices'](*grid.shape, dy, dx)
        left, right = grid[src], grid[dst]
        expected = valid[src] & valid[dst] & (left > 0) & (right > 0)
        whole_expected = np.zeros(grid.shape, bool); whole_expected[src] = expected
        whole_target = np.zeros(grid.shape, np.float32); whole_target[src] = expected & (left == right)
        if not np.array_equal(edge_valid[channel], whole_expected) or not np.array_equal(target[channel], whole_target):
            raise RuntimeError('实际目标不符合封存已知端点合同')
        selected = expected & (((left == a) & (right == b)) | ((left == b) & (right == a)))
        ends = np.stack((np.column_stack((ymap[src][selected], xmap[src][selected])),
                         np.column_stack((ymap[dst][selected], xmap[dst][selected]))), axis=1)
        if len(ends) and (np.any(ends < 0) or not np.all(full_gt[ends[:, :, 0], ends[:, :, 1]] > 0)):
            raise RuntimeError('实际目标端点落入未知或padding')
        trusted = original_covered[ends[:, :, 0], ends[:, :, 1]].all(axis=1)
        # 不改变target，只审计端点之间是否经过未知或第三ID。
        crosses_unknown = crosses_third = 0
        for segment in ends:
            length = int(np.max(np.abs(segment[1] - segment[0]))) + 1
            path = np.rint(np.linspace(segment[0], segment[1], length)).astype(np.int64)
            labels = full_gt[path[:, 0], path[:, 1]]
            crosses_unknown += int(np.any(labels == 0))
            crosses_third += int(np.any((labels > 0) & (labels != a) & (labels != b)))
        rows.append(dict(channel=channel, offset=[dy, dx], negative_relations=int(selected.sum()),
            both_original_covered=int(trusted.sum()), includes_filled_endpoint=int((~trusted).sum()),
            segment_has_unknown=crosses_unknown, segment_has_third_id=crosses_third))
        points.append(ends.mean(axis=1))
    return rows, points


def view_fidelity(gt, trust, pair, edges, box, draw, functions, *, input_size=1024, output_grid=512):
    native, grid, valid, meta, yy, xx = coverage.geometry_grid(
        gt, box, draw, functions, input_size=input_size, output_grid=output_grid)
    selected = in_box(edges, box)
    original_edges = trust[edges[:, :, 0], edges[:, :, 1]].all(axis=1)
    source_points = edges[selected].mean(axis=1)
    channels, points = pair_targets(grid, valid, yy, xx, gt, trust, pair, functions)
    unaugmented = inverse_spatial(grid, draw)
    restored = cv2.resize(unaugmented[:meta.content_height, :meta.content_width],
                          (native.shape[1], native.shape[0]), interpolation=cv2.INTER_NEAREST)
    restored_edges = contact_edges(restored, pair, box[:2])
    restored_points = restored_edges.mean(axis=1)
    ref_to_restored = nearest_distance(source_points, restored_points)
    # 距离数组始终对应固定原生169条接触；没进入视野与进视野但丢失分开记录。
    distances = {'restored_cardinal': np.full(len(edges), np.inf)}
    distances['restored_cardinal'][selected] = ref_to_restored
    groups = {}
    for name, indices in GROUPS.items():
        midpoints = np.concatenate([points[i] for i in indices])
        dist = nearest_distance(source_points, midpoints)
        distances[name] = np.full(len(edges), np.inf); distances[name][selected] = dist
        groups[name] = dict(negative_relations=sum(channels[i]['negative_relations'] for i in indices),
            both_original_covered=sum(channels[i]['both_original_covered'] for i in indices),
            includes_filled_endpoint=sum(channels[i]['includes_filled_endpoint'] for i in indices),
            native_contact_to_relation_midpoint=distance_summary(dist),
            reference_cohort_coverage={label: distance_summary(dist[mask[selected]]) for label, mask in
                (('both_original_covered', original_edges), ('includes_filled_endpoint', ~original_edges))},
            relation_midpoint_to_native_contact=distance_summary(nearest_distance(midpoints, source_points)))
    rim = np.min(np.stack((edges[:, :, 0]-box[0], edges[:, :, 1]-box[1],
                          box[2]-1-edges[:, :, 0], box[3]-1-edges[:, :, 1])), axis=(0, 2))
    row = dict(box_yxyx=list(map(int, box)), geometry=meta.to_dict(),
        grid_sha256=coverage.array_sha(grid), content_sha256=coverage.array_sha(valid),
        interface_edges_in_view=int(selected.sum()), interface_edges_within8_of_crop_frame=int((selected & (rim < 8)).sum()),
        original_covered_interface_edges_in_view=int((selected & original_edges).sum()),
        filled_endpoint_interface_edges_in_view=int((selected & ~original_edges).sum()),
        visible_pair_gt_pixels={str(i): int((native == i).sum()) for i in pair},
        sampled_pair_gt_pixels={str(i): int(((grid == i) & valid).sum()) for i in pair},
        channels=channels, groups=groups,
        nearest_restored_cardinal=dict(contact_edges=len(restored_edges),
            native_contact_to_restored_contact=distance_summary(ref_to_restored),
            reference_cohort_coverage={label: distance_summary(ref_to_restored[mask[selected]]) for label, mask in
                (('both_original_covered', original_edges), ('includes_filled_endpoint', ~original_edges))},
            restored_contact_to_native_contact=distance_summary(nearest_distance(restored_points, source_points))))
    return row, distances, selected


def aggregate(rows, count, reference_cohorts=None):
    result = dict(views=len(rows), interface_edges_ever_in_view=0, interface_edge_exposures=0, groups={})
    seen = np.zeros(count, np.int64)
    for _, _, selected in rows:
        seen += selected
    result.update(interface_edges_ever_in_view=int((seen > 0).sum()), interface_edge_exposures=int(seen.sum()),
                  interface_edge_exposures_within8_of_crop_frame=sum(r[0]['interface_edges_within8_of_crop_frame'] for r in rows),
                  native_interface_exposure_quantiles={str(q): float(np.quantile(seen, q)) for q in (0., .5, 1.)})
    if reference_cohorts is not None:
        result['reference_cohort_exposures'] = {label: dict(native_contact_edges=int(mask.sum()),
            interface_edges_ever_in_view=int((seen[mask] > 0).sum()), interface_edge_exposures=int(seen[mask].sum()))
            for label, mask in reference_cohorts.items()}
    for name in (*GROUPS, 'restored_cardinal'):
        distances = np.min(np.stack([item[1][name] for item in rows]), axis=0) if rows else np.full(count, np.inf)
        record = dict(unique_native_interface_coverage=distance_summary(distances))
        if reference_cohorts is not None:
            record['reference_cohort_coverage'] = {label: distance_summary(distances[mask])
                                                 for label, mask in reference_cohorts.items()}
        if name in GROUPS:
            record.update(negative_relations=sum(row[0]['groups'][name]['negative_relations'] for row in rows),
                          views_with_relations=sum(row[0]['groups'][name]['negative_relations'] > 0 for row in rows),
                          both_original_covered=sum(row[0]['groups'][name]['both_original_covered'] for row in rows),
                          includes_filled_endpoint=sum(row[0]['groups'][name]['includes_filled_endpoint'] for row in rows))
        result['groups'][name] = record
    return result


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line]


def verify_old_totals(summary, old, prefix):
    if summary['interface_edge_exposures'] != int(old[prefix+'_crop_contact_edges']):
        raise RuntimeError('旧CSV原生接触曝光数量不一致: '+prefix)
    for name in ('cardinal', 'short', 'any'):
        actual = summary['groups'][name]
        for key, suffix in (('negative_relations', '_edges'), ('views_with_relations', '_hits')):
            if actual[key] != int(old[prefix+'_'+name+suffix]):
                raise RuntimeError('旧CSV真实网格关系数量不一致: '+prefix+' '+name+' '+key)


def run(args):
    start = time.perf_counter()
    output = Path(args.output).resolve()
    if output.exists() or output.relative_to(ROOT).parts[0] not in {'output', 'outputs'}:
        raise ValueError('仅写入项目内output/或outputs/的新私有目录')
    paths = {name: Path(getattr(args, name)).resolve() for name in
             ('gt', 'classes', 'pipeline', 'snapshot', 'r1_steps', 'coverage_steps', 'grid_report', 'grid_pairs', 'grid_draws')}
    protected = {path: original.sha(path) for name, path in paths.items() if name != 'snapshot'}
    protected[Path(__file__)] = original.sha(__file__)
    protected[Path(coverage.__file__)] = original.sha(coverage.__file__)
    protected[Path(original.__file__)] = original.sha(original.__file__)
    if protected[paths['grid_report']] != GRID_REPORT_SHA or protected[Path(original.__file__)] != coverage.SIGNAL_CODE_SHA:
        raise RuntimeError('已验收网格回执或封存纯函数提取器身份不同')
    pipeline = original.read(paths['pipeline']); contract = pipeline['contract']
    if contract['original_pipeline_sha256'] != original.R1_PIPELINE_SHA or len(contract['original_sources']) != 49:
        raise RuntimeError('原49来源绑定不符')
    if (contract['grid'].get('report_sha256') != GRID_REPORT_SHA
            or contract['grid'].get('actual_target_mask_gate') is not True):
        raise RuntimeError('完整512训练网格回放门禁绑定不同')
    source = args.source
    cohort = [x for x in contract['newgt']['samples'] if x['source'] == source]
    if len(cohort) != 1 or not source.startswith('train_'):
        raise ValueError('仅允许原32个训练来源，禁止测试GT')
    sample = cohort[0]
    for name, suffix in (('gt', '_gt.npz'), ('classes', '_class.json')):
        expected = [v for k, v in sample['files_sha256'].items() if k.endswith(source+suffix)]
        if len(expected) != 1 or protected[paths[name]] != expected[0]:
            raise RuntimeError('完整processed训练GT实文件身份不同: '+name)
    funcs = paths['snapshot']
    for name in (*original.FUNCTIONS, 'utils/affinity_graph.py', 'utils/patch_diagnostic.py', 'config/train/affinity_native.yaml'):
        path = funcs/name; digest = original.sha(path); protected[path] = digest
        expected = contract['original_sources'].get(name)
        # 后续graph文件增加实现；只读已封存八offset声明，并由load_functions严格核值。
        if name != 'utils/affinity_graph.py' and digest != expected:
            raise RuntimeError('原49几何/目标/正式窗口配置不同: '+name)
        if digest != pipeline['sources'].get(name):
            raise RuntimeError('分析源码快照与coverage来源回执不同: '+name)
    functions = original.load_functions(funcs)
    tile_env = {}
    node = original.nodes(funcs/'utils/patch_diagnostic.py', ('tile_boxes',))
    exec(compile(ast.Module(body=node, type_ignores=[]), str(funcs/'utils/patch_diagnostic.py'), 'exec'), tile_env)
    overlap = yaml.safe_load((funcs/'config/train/affinity_native.yaml').read_text(encoding='utf8'))['affinity_native']['overlap']
    with np.load(paths['gt'], allow_pickle=False) as blob:
        gt = blob['instance_map'].copy()
        masks = {key: blob[key].copy() for key in ('original_covered', 'filled', 'residual_unknown')}
    classes = original.read(paths['classes']); coverage.validate_labels(gt, classes)
    if list(gt.shape) != sample['shape'] or any(value.shape != gt.shape or not np.all((value == 0) | (value == 1)) for value in masks.values()):
        raise ValueError('完整GT或provenance掩码形状/取值不同')
    trust = masks['original_covered'].astype(bool)
    if np.any(trust & (gt == 0)) or not np.array_equal(masks['filled'].astype(bool), (gt > 0) & ~trust) or not np.array_equal(masks['residual_unknown'].astype(bool), gt == 0):
        raise ValueError('未知/补区合同不一致，不自动修补')
    pair = tuple(args.pair)
    if any(classes.get(str(i)) != 1 for i in pair):
        raise ValueError('固定诊断需要两个F类别实例')
    edges = contact_edges(gt, pair)
    if not len(edges):
        raise ValueError('原生完整GT中没有该四邻接触')
    original_edges = trust[edges[:, :, 0], edges[:, :, 1]].all(axis=1)
    reference_cohorts = dict(both_original_covered=original_edges, includes_filled_endpoint=~original_edges)
    existing_report = original.read(paths['grid_report'])
    if existing_report['parents']['original_steps_sha256'] != protected[paths['r1_steps']]:
        raise RuntimeError('原R1完整训练回执与既存网格父回执不同')
    old_rows = list(csv.DictReader(paths['grid_pairs'].open(encoding='utf8', newline='')))
    old = [r for r in old_rows if r['source'] == source and (int(r['gt_a']), int(r['gt_b'])) == tuple(sorted(pair))]
    if len(old) != 1 or len(edges) != int(old[0]['original_contact_edges']):
        raise RuntimeError('旧CSV固定原生GT对与本地真实接触不符')
    draws = {int(r['update']): r for r in original.read(paths['grid_draws']) if r['source'] == source}
    r1 = read_jsonl(paths['r1_steps']); candidate = read_jsonl(paths['coverage_steps'])
    if len(r1) != 1280 or len(candidate) != 1280:
        raise RuntimeError('需要完整1280条实际训练回执')
    for a, b in zip(r1, candidate):
        for key in ('epoch', 'step', 'seed', 'manual', 'manual_draw', 'training_view', 'updates'):
            if a[key] != b[key]:
                raise RuntimeError('两臂固定训练字段不同: '+key)
        if b.get('coverage', {}).get('executed') is not True:
            raise RuntimeError('候选必须是实际执行训练回执，不能用metadata反事实冒充')
    cache = {}; detail = []; training_rows = defaultdict(list)
    for arm, receipts, prefix in (('r1', r1, 'baseline'), ('coverage', candidate, 'candidate')):
        chosen = [r for r in receipts if r['manual'] == [source+'.jpg']]
        if len(chosen) != len(draws):
            raise RuntimeError('单源40次实际回执与旧draw集合不符')
        for receipt in chosen:
            box = receipt['manual_crop'][0]; draw = receipt['manual_draw']
            key = (*box, draw['horizontal_flip'][0], draw['vertical_flip'][0], draw['rotation_k'][0])
            if key not in cache:
                cache[key] = view_fidelity(gt, trust, pair, edges, box, draw, functions)
            value = cache[key]
            saved = draws[receipt['updates']]
            for kind in ('grid', 'content'):
                if value[0][kind+'_sha256'] != saved[prefix+'_'+kind+'_sha256']:
                    raise RuntimeError('单源实际训练目标不能精确重放: '+arm+' '+str(receipt['updates']))
            row = dict(value[0], arm=arm, coordinate_system='actual_training_crop_augmented512_mapped_to_native',
                       view=receipt['training_view'], epoch=receipt['epoch'], update=receipt['updates'], augmentation=draw)
            detail.append(row); training_rows[(arm, receipt['training_view'])].append(value)
        print(json.dumps(dict(arm=arm, actual_training_views=len(chosen), single_source=True, cpu_only=True)), flush=True)
    training = {arm: {view: aggregate(training_rows[(arm, view)], len(edges), reference_cohorts) for view in ('whole', 'native1024')}
                for arm in ('r1', 'coverage')}
    for arm, prefix in (('r1', 'baseline'), ('coverage', 'candidate')):
        verify_old_totals(training[arm]['native1024'], old[0], prefix)
        verify_old_totals(training[arm]['whole'], old[0], 'whole')
        training[arm]['combined'] = aggregate(training_rows[(arm, 'whole')]+training_rows[(arm, 'native1024')], len(edges), reference_cohorts)
    neutral = dict(horizontal_flip=[False], vertical_flip=[False], rotation_k=[0])
    inference_rows = []
    boxes = tile_env['tile_boxes'](gt.shape, 1024, overlap)
    for index, box in enumerate(boxes):
        value = view_fidelity(gt, trust, pair, edges, box, neutral, functions)
        inference_rows.append(value)
        detail.append(dict(value[0], tile_index=index, coordinate_system='formal_native1024_inference_tile_unaugmented512_mapped_to_native',
                           view='native1024', arm='formal_geometry_only'))
    inference = dict(whole_geometry=aggregate([view_fidelity(gt, trust, pair, edges, (0, 0, *gt.shape), neutral, functions)], len(edges), reference_cohorts),
        native1024_geometry=aggregate(inference_rows, len(edges), reference_cohorts), overlap=float(overlap), tiles=len(boxes),
        tiles_with_original_contact=int(sum(r[2].any() for r in inference_rows)), boxes_yxyx=[list(b) for b in boxes],
        does_not_claim_inference_uses_gt_targets=True, no_model_response_or_blending_measured=True)
    for path, digest in protected.items():
        if original.sha(path) != digest:
            raise RuntimeError('诊断期间输入/源码改变: '+str(path))
    report = dict(complete=True, format='affinity_grid_fidelity_v1', source=source, gt_pair=list(pair),
        no_gpu=True, no_training=True, no_model_load=True, single_source_replay=True,
        original49_geometry_targets_and_tile_function_bound=True, original_graph_offset_literal_checked=True,
        actual_training_grids_exact_equal_existing=True, old_pair_counts_exact_equal=True,
        native_gt=dict(shape=list(gt.shape), contact_edges=len(edges),
            both_original_covered_edges=int(trust[edges[:, :, 0], edges[:, :, 1]].all(axis=1).sum()),
            includes_filled_endpoint_edges=int((~trust[edges[:, :, 0], edges[:, :, 1]].all(axis=1)).sum()),
            contact_endpoint_pixels=int(len(np.unique(edges.reshape(-1, 2), axis=0)))),
        training=training, formal_inference_geometry=inference,
        unique_training_geometries_replayed=len(cache), actual_single_source_draws_replayed=len(detail)-len(boxes),
        protected_before_after_exact=True,
        runtime=dict(torch=torch.__version__, opencv=cv2.__version__, numpy=np.__version__,
                     cuda_initialized=torch.cuda.is_initialized(),
                     training_runtime_exact_equal_not_claimed=True),
        protected_sha256={str(p.relative_to(ROOT)): h for p, h in protected.items()},
        elapsed_cpu_seconds=time.perf_counter()-start,
        definitions=dict(short='original channels0..3: horizontal/vertical and two diagonals',
            long='original channels4..7: cardinal distance2/distance4',
            relation_location='midpoint of the two nearest-sampled native endpoints; proximity does not prove a closed boundary',
            restored_contact='GT512 nearest-upsampled back to native crop; compare unordered direct4-neighbor GT-pair contact midpoints',
            coverage='unique fixed native contact-edge midpoints with a nearest candidate at radius in native pixels; counts are not interface arc length',
            provenance='filled endpoints remain valid in actual target, but are reported separately; GT0 and padding never negatives'),
        caveats=['All diagnostics use a seen training source; no official, validation, or test GT metrics.',
            'Formal inference tiles are geometry probes only. Inference does not consume GT targets; model features, fusion, blend, and watershed responses are not measured.',
            'Long target validity follows the accepted two-known-endpoint contract; unknown or third-ID interiors are separately counted, not silently removed.',
            'Frame-adjacent contact exposures are separately reported and are not assumed to imply good model response.'])
    output.mkdir(parents=True)
    coverage.write(output/'report.json', report)
    coverage.write(output/'views.json', detail)
    print(json.dumps(dict(complete=True, output=str(output), contact_edges=len(edges), old_counts_exact=True,
                         seconds=report['elapsed_cpu_seconds']), ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='train_659')
    parser.add_argument('--pair', type=int, nargs=2, default=[69, 70])
    parser.add_argument('--gt', default='outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps/train_659_gt.npz')
    parser.add_argument('--classes', default='outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps/train_659_class.json')
    parser.add_argument('--pipeline', default='output/affinity_coverage/completed/pipeline_status.json')
    parser.add_argument('--snapshot', default='output/affinity_coverage/completed/source')
    parser.add_argument('--r1-steps', default='output/affinity_coverage/completed/reference/steps.jsonl')
    parser.add_argument('--coverage-steps', default='output/affinity_coverage/completed/candidate/steps.jsonl')
    parser.add_argument('--grid-report', default='output/affinity_coverage_grid/analysis/report.json')
    parser.add_argument('--grid-pairs', default='output/affinity_coverage_grid/analysis/pairs.csv')
    parser.add_argument('--grid-draws', default='output/affinity_coverage_grid/analysis/draws.json')
    parser.add_argument('--output', default='output/affinity_grid_fidelity/train659')
    args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(2)
    run(args)


if __name__ == '__main__':
    main()
