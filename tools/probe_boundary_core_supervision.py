# -*- coding: utf-8 -*-
"""只用已见训练GT检查完整异实例界面，不训练、不产生竞赛提交结果。

先对整图执行封存N0的reflect marker，再看原标深部内核；不在裁窗内
重算连通性，不把未知区删除后得到的分隔冒充实际部署分隔。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def load_helpers(path):
    spec = importlib.util.spec_from_file_location('core_supervision_probe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_bundle(root, expected):
    manifest = root / 'manifest.json'
    if sha(manifest) != expected['manifest_sha256']:
        raise ValueError('Frozen deployment manifest changed')
    rows = json.loads(manifest.read_text(encoding='utf-8'))['files']
    for row in rows:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or sha(path) != row['sha256']:
            raise ValueError('Frozen deployment source/model changed: ' + row['path'])
    if len(rows) != expected['checked_files']:
        raise ValueError('Frozen deployment file count changed')
    return dict(manifest_sha256=sha(manifest), checked_files=len(rows))


def load_processed_source(image_path, gt_path, class_path):
    """独立CPU读取器；不为了读GT引入封存推理包未包含的DINO训练模块。"""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode source RGB')
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with np.load(gt_path, allow_pickle=False) as payload:
        ids = payload['instance_map'].copy()
        original = payload['original_covered'].copy()
        filled = payload['filled'].copy()
        unknown = payload['residual_unknown'].copy()
    if (ids.shape != rgb.shape[:2] or ids.dtype.kind not in 'iu'
            or ids.min() < 0 or ids.max() > 65535):
        raise ValueError('Invalid processed GT labels/shape')
    for mask in (original, filled, unknown):
        if mask.shape != ids.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError('Invalid processed GT provenance')
    original = original.astype(bool)
    if (np.any(original & (ids == 0))
            or not np.array_equal(filled.astype(bool), (ids > 0) & ~original)
            or not np.array_equal(unknown.astype(bool), ids == 0)):
        raise ValueError('Processed GT provenance disagrees')
    values = json.loads(Path(class_path).read_text(encoding='utf-8'))
    if not isinstance(values, dict) or any(not k.isdecimal() or str(int(k)) != k
            or int(k) < 1 or type(v) is not int or v not in (0, 1) for k, v in values.items()):
        raise ValueError('Invalid processed GT class mapping')
    classes = {int(k): v for k, v in values.items()}
    if not set(np.unique(ids))-{0} <= set(classes):
        raise ValueError('Known GT instance missing a class')
    return rgb, ids.astype(np.int32), classes, original


def restricted_relation(marker, ids, cores, pair, support=None):
    """仅用于辨认实际全图通路是否也能在某个支持域内找到。

    限域不连通不能证明全图分隔；实际成功标准始终来自原全图marker。
    """
    selected = np.isin(ids, pair) & (marker > 0)
    if support is not None:
        selected &= support
    # 保留原marker之间的屏障，不把不同正marker因相邻而重新粘合。
    labels = np.zeros(marker.shape, np.int32)
    count = 0
    a = marker[cores == pair[0]]
    b = marker[cores == pair[1]]
    shared = np.intersect1d(a[a > 0], b[b > 0])
    for value in shared:
        number, components = cv2.connectedComponents(
            (selected & (marker == value)).astype(np.uint8), connectivity=8)
        components[components > 0] += count
        labels[components > 0] = components[components > 0]
        count += number - 1
    first = np.unique(labels[cores == pair[0]])
    second = np.unique(labels[cores == pair[1]])
    common = np.intersect1d(first[first > 0], second[second > 0])
    return dict(shared_components=int(common.size), connected=bool(common.size))


def gallery(path, rgb, ids, cores, pair, boundary, marker, variants):
    selected = np.isin(cores, pair)
    yy, xx = np.nonzero(selected)
    y0, y1 = max(0, int(yy.min())-48), min(ids.shape[0], int(yy.max())+49)
    x0, x1 = max(0, int(xx.min())-48), min(ids.shape[1], int(xx.max())+49)
    panels = [('RGB + original GT cores', rgb.copy(), None)]
    panels[0][1][cores == pair[0]] = (
        .55*rgb[cores == pair[0]] + .45*np.array([250, 90, 40])).astype(np.uint8)
    panels[0][1][cores == pair[1]] = (
        .55*rgb[cores == pair[1]] + .45*np.array([30, 180, 255])).astype(np.uint8)
    for title, heat, labels, changed in [('N0 boundary / marker', boundary, marker, None), *variants]:
        color = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1)*255),
                                             cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        if changed is not None:
            contours, _ = cv2.findContours(changed.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color, contours, -1, (255, 255, 255), 1)
        panels.append((title, color, labels))
    width = 340
    height = max(80, min(430, round((y1-y0)*width/max(1, x1-x0))))
    canvas = np.full((2*height+115, width*len(panels), 3), 245, np.uint8)
    for index, (title, image, labels) in enumerate(panels):
        image = cv2.resize(image[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_AREA)
        canvas[40:40+height, index*width:(index+1)*width] = image
        cv2.putText(canvas, title, (index*width+6, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (20, 20, 20), 1, cv2.LINE_AA)
        view = np.full((*ids.shape, 3), 65, np.uint8)
        if labels is None:
            view[ids == 0] = 0
            view[ids == pair[0]] = (245, 95, 40)
            view[ids == pair[1]] = (30, 180, 255)
        else:
            first = np.unique(labels[cores == pair[0]])
            second = np.unique(labels[cores == pair[1]])
            a, b = np.isin(labels, first[first > 0]), np.isin(labels, second[second > 0])
            view[labels == 0] = 0
            view[a & ~b] = (245, 95, 40)
            view[b & ~a] = (30, 180, 255)
            view[a & b] = (200, 60, 210)
        canvas[height+65:2*height+65, index*width:(index+1)*width] = cv2.resize(
            view[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.putText(canvas, 'Bottom: orange=A / cyan=B / purple=shared positive marker / black=unseeded; NOT final instances',
                (8, height+57), cv2.FONT_HERSHEY_SIMPLEX, .47, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Seen training GT; whole native marker replay; white=edited interface; no test GT / no training',
                (8, 2*height+97), cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imencode('.jpg', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                 [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(path)
    return [y0, x0, y1, x1]


def inspect_source(helper, stem, ids, original, classes, boundary, marker,
                   replay, rgb, output, pair_budget):
    cores = helper.build_trusted_cores(ids, original, radius=4, min_component_pixels=64)
    stats = helper.core_marker_statistics(cores, marker)
    pairs = []
    for pair in helper.adjacent_instance_pairs(ids):
        relation = helper.pair_core_relation(stats, pair)
        pairs.append(dict(pair=list(map(int, pair)), classes=[classes[int(k)] for k in pair],
                          relation=relation))
    # 每个关系的候选名单由baseline固定；不根据干预成功与否重新选病例。
    eligible = [p for p in pairs if p['relation']['significant']['marker_count'] > 0]
    eligible.sort(key=lambda p: (sum(c == 0 for c in p['classes']), *p['pair']))
    item = dict(source=stem, shape=list(ids.shape), core_summary=cores['summary'],
                core_gt_stats=cores['gt_stats'], baseline_core_summary=stats['summary'],
                pairs=pairs, eligible_pairs=len(eligible), candidates=[])
    original_boundary_sha = hashlib.sha256(boundary.tobytes()).hexdigest()
    for case_index, selected in enumerate(eligible[:pair_budget]):
        pair = selected['pair']
        record = dict(pair=pair, classes=selected['classes'], before=selected['relation'],
                      path_within_known_pair=restricted_relation(marker, ids, cores['core_map'], pair),
                      path_within_original_pair=restricted_relation(marker, ids, cores['core_map'], pair, original),
                      arms=[])
        panels = []
        for policy in ('original', 'thin4', 'processed_known_optimistic'):
            interface = helper.candidate_interface(ids, original, pair, support_policy=policy,
                                                    radius=2, thin_radius=4)
            mask = interface['mask']
            modified = boundary.copy()
            modified[mask] = np.maximum(modified[mask], np.float32(.9))
            changed = modified != boundary
            if np.any(changed & (ids == 0)):
                raise RuntimeError('Unknown changed by candidate')
            labels = replay(modified) if changed.any() else marker
            after = helper.core_marker_statistics(cores, labels)
            comparison = helper.compare_core_separation(stats, after, pair, min_coverage=.9)
            atoms = helper.compare_marker_atoms(cores, marker, labels)
            arm = dict(policy=policy, interface_summary=interface['summary'],
                       provenance=interface['provenance'], modified_pixels=int(changed.sum()),
                       comparison=comparison, atoms=atoms,
                       safe_marker_intervention=core_intervention_is_stable(comparison, atoms))
            record['arms'].append(arm)
            if case_index == 0:
                panels.append((policy, modified, labels, changed))
            del modified, labels, after
        if case_index == 0:
            filename = f'{stem}_{pair[0]}_{pair[1]}.jpg'
            record['gallery_box_yxyx'] = gallery(output/'gallery'/filename, rgb, ids, cores['core_map'],
                                                pair, boundary, marker, panels)
            record['gallery'] = 'gallery/' + filename
        item['candidates'].append(record)
    if hashlib.sha256(boundary.tobytes()).hexdigest() != original_boundary_sha:
        raise RuntimeError('Probe mutated input N0 boundary')
    return item


def aggregate(sources):
    counts = Counter()
    per_class = {key: Counter() for key in ('FF', 'FP', 'PP')}
    arms = {key: Counter() for key in ('original', 'thin4', 'processed_known_optimistic')}
    for source in sources:
        counts['sources'] += 1
        counts['adjacent_pairs'] += len(source['pairs'])
        counts['eligible_pairs'] += source['eligible_pairs']
        counts['sources_with_eligible'] += bool(source['eligible_pairs'])
        counts['selected_pairs'] += len(source['candidates'])
        for row in source['pairs']:
            name = 'FF' if row['classes'] == [1, 1] else 'PP' if row['classes'] == [0, 0] else 'FP'
            per_class[name]['adjacent'] += 1
            per_class[name]['significant_shared'] += row['relation']['significant']['marker_count'] > 0
        for row in source['candidates']:
            for arm in row['arms']:
                result = arms[arm['policy']]
                result['pairs'] += 1
                result['modified_pixels'] += arm['modified_pixels']
                result['separated_and_covered'] += arm['comparison']['target_separated_and_covered']
                result['no_new_atom_side_effects'] += arm['safe_marker_intervention']
                result['split_atoms'] += arm['atoms']['summary']['atom_splits']
                result['old_positive_to_zero_pixels'] += arm['atoms']['summary']['zero_pixels_from_old_positive']
                result['new_merged_atom_pairs'] += arm['atoms']['summary']['new_merged_atom_pairs']
    return dict(counts=dict(counts), per_class={k: dict(v) for k, v in per_class.items()},
                arms={k: dict(v) for k, v in arms.items()},
                training_started=False, official_accuracy_measured=False)


def core_intervention_is_stable(comparison, atoms):
    """登记原标内部上的稳定检查；不保证内部以外的新种子没有副作用。

    atoms只来源于旧正种子，必须额外检查旧0内部变正后出现的新异GT共享。
    """
    return bool(comparison['target_separated_and_covered']
                and atoms['summary']['atom_splits'] == 0
                and atoms['summary']['zero_pixels_from_old_positive'] == 0
                and atoms['summary']['new_merged_atom_pairs'] == 0
                and not comparison['new_cross_gt_shared_pairs'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-pairs-per-source', type=int, default=2)
    parser.add_argument('--sources', type=int, default=32)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    if (args.max_pairs_per_source < 1 or not 1 <= args.sources <= 32
            or not 0 <= args.shard_index < args.shard_count <= 8):
        raise ValueError('Positive pair budget and 1..32 sources required')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh output directory, preserve old receipts')
    output.mkdir(parents=True)
    (output/'gallery').mkdir()
    module_path = Path(__file__).resolve().parents[1]/'utils/boundary_core_supervision.py'
    helper = load_helpers(module_path)
    manifest = Path(args.manifest).resolve()
    source = json.loads(manifest.read_text(encoding='utf-8'))
    if (not source.get('complete') or source.get('semantic_forward')
            or source.get('final_decode') or len(source['files']) != 32):
        raise ValueError('Expected original complete32 boundary-only N0 capture')
    root = Path(source['source_root']).resolve()
    bundle_before = validate_bundle(root, source['source_manifest'])
    # 使用热图实际封存配置，不读恢复器训练配置中的基础inference默认值。
    config = source['config']
    sys.path.insert(0, str(root))
    from tools.semantic_marker_diagnostic import marker_stages
    from tools.probe_affinity_marker_replay import build_marker_variant
    import torch
    import skimage
    for name in ('tools.semantic_marker_diagnostic', 'tools.probe_affinity_marker_replay',
                 'utils.post_process', 'utils.affinity_deployment'):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(root):
            raise RuntimeError('Marker probe imported source outside frozen root: ' + name)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)

    def replay(heat):
        return build_marker_variant(marker_stages(heat, config), 'reflect',
                                    int(config['submission']['marker_reflection_padding']))['marker_labels']

    report = dict(format='boundary_core_supervision_v1', complete=False,
                  training=False, final_decode=False, official_accuracy=False,
                  requested_sources=args.sources,
                  shard=dict(index=args.shard_index, count=args.shard_count),
                  selection='all adjacent GT pairs inventoried; max2 significant shared-core pairs/source; FF first, then FP/PP, numeric IDs',
                  source_manifest_sha256=sha(manifest), source_bundle=bundle_before,
                  executed_sources={str(Path(__file__).name): sha(__file__), module_path.name: sha(module_path)},
                  marker_runtime=dict(python=sys.version, torch=torch.__version__,
                                      numpy=np.__version__, opencv=cv2.__version__,
                                      skimage=skimage.__version__,
                                      ximgproc_thinning=hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning')),
                  marker_imports={name: dict(path=sys.modules[name].__file__,
                                              sha256=sha(sys.modules[name].__file__))
                                  for name in ('tools.semantic_marker_diagnostic',
                                               'tools.probe_affinity_marker_replay',
                                               'utils.post_process', 'utils.affinity_deployment')},
                  recipe=dict(core_margin=4, minimum_component_pixels=64, interface_radius=2,
                              thin_fill_radius=4, strength=.9, minimum_shared_pixels=32,
                              minimum_shared_fraction=.1, minimum_seed_coverage=.9),
                  limitations=['seen training32, no independent validation',
                               'marker connectivity only, no semantic or final watershed',
                               'processed-known is an optimistic position reference, not independent ground truth',
                               'same-GT internal borders may be omitted; no forced same-GT connectivity target'],
                  sources=[])
    start = time.monotonic()
    rows = source['files'][:args.sources][args.shard_index::args.shard_count]
    for index, row in enumerate(rows):
        stem = row['source']
        gt = row['processed_gt']
        for path, expected in [(row['heat_path'], row['heat_sha256']),
                               (row['image_path'], row['image_sha256']),
                               (gt['path'], gt['sha256']), (gt['classes_path'], gt['classes_sha256'])]:
            if sha(path) != expected:
                raise RuntimeError('Input changed: ' + str(path))
        rgb, ids, classes, original = load_processed_source(row['image_path'], gt['path'], gt['classes_path'])
        with np.load(row['heat_path'], allow_pickle=False) as payload:
            boundary = payload['boundary'].copy()
        if boundary.dtype != np.float32 or boundary.shape != ids.shape:
            raise ValueError('Native N0 boundary contract mismatch')
        marker = replay(boundary)
        # helper API is adapted below after its focused synthetic tests.
        item = inspect_source(helper, stem, ids, original, classes, boundary, marker,
                              replay, rgb, output, args.max_pairs_per_source)
        item['source_hashes'] = dict(heat=row['heat_sha256'], image=row['image_sha256'],
                                     gt=gt['sha256'], classes=gt['classes_sha256'])
        report['sources'].append(item)
        report['elapsed_seconds'] = time.monotonic()-start
        write(output/'progress.json', report)
        print(json.dumps(dict(source=stem, progress=[index+1, args.sources],
                              adjacent_pairs=len(item['pairs']), candidates=len(item['candidates']),
                              seconds=round(report['elapsed_seconds'], 1))), flush=True)
    report['source_bundle_after'] = validate_bundle(root, source['source_manifest'])
    if report['source_bundle_after'] != bundle_before or sha(manifest) != report['source_manifest_sha256']:
        raise RuntimeError('Frozen capture/deployment changed during CPU probe')
    for row in rows:
        gt = row['processed_gt']
        for path, expected in [(row['heat_path'], row['heat_sha256']),
                               (row['image_path'], row['image_sha256']),
                               (gt['path'], gt['sha256']), (gt['classes_path'], gt['classes_sha256'])]:
            if sha(path) != expected:
                raise RuntimeError('Input changed after probe: ' + str(path))
    report['input_files_unchanged'] = True
    report['complete'] = True
    report['summary'] = aggregate(report['sources'])
    write(output/'report.json', report)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
