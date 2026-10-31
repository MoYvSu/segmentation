# -*- coding: utf-8 -*-
"""冻结N1缓存，仅分开种子核心门槛与最终面积门槛；不训练、不生成测试GT。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def statistics(ids, classes):
    if ids.dtype != np.uint16 or ids.ndim != 2 or int(ids.max(initial=0)) > 65535:
        raise ValueError('Invalid final instance contract')
    sizes = np.bincount(ids.ravel())
    if set(classes) != {str(int(v)) for v in np.unique(ids) if v}:
        raise ValueError('Instance IDs and class keys differ')
    if any(type(value) is not int or value not in (0, 1) for value in classes.values()):
        raise ValueError('Invalid instance class')
    result = dict(instances=len(classes), zero_pixels=int(sizes[0]))
    for name, category in [('F', 1), ('P', 0)]:
        values = sizes[[int(key) for key, value in classes.items() if value == category]]
        result[name] = dict(count=len(values), pixels=int(values.sum()),
                            mean_area=float(values.mean()) if len(values) else None,
                            under500=int((values < 500).sum()))
    return result


def distribution(values):
    labels, counts = np.unique(values, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def processed_gt(row, shape, build_cores):
    if not row.get('gt'):
        return None
    if not row['source'].startswith('train_'):
        raise ValueError('Test ground truth is forbidden')
    with np.load(row['gt'], allow_pickle=False) as payload:
        ids = payload['instance_map'].copy()
        original = payload['original_covered'].astype(bool)
        filled = payload['filled'].astype(bool)
        unknown = payload['residual_unknown'].astype(bool)
    if (ids.shape != shape or not np.array_equal(unknown, ids == 0)
            or not np.array_equal(filled, (ids > 0) & ~original)):
        raise ValueError('Processed GT provenance differs')
    return dict(ids=ids, original=original, filled=filled, unknown=unknown,
                cores=build_cores(ids, original, radius=4, min_component_pixels=64))


def candidate_details(core_labels, old_markers, markers, before, after, watershed,
                      classes, boundary, gt):
    """只描述新增核心去向；不把新增实例自动称为真晶粒或误切。"""
    added = (markers > 0) & (old_markers == 0)
    rows = []
    for source_label in np.unique(core_labels[added]):
        if not source_label:
            continue
        mask = core_labels == source_label
        ys, xs = np.nonzero(mask)
        marker_ids = np.unique(markers[mask])
        if len(marker_ids) != 1 or marker_ids[0] <= 0:
            raise ValueError('An admitted core must have exactly one positive seed')
        marker_id = int(marker_ids[0])
        final = distribution(after[mask])
        positive = [int(key) for key in final if int(key) > 0]
        final_ids = {str(key): dict(area=int((after == key).sum()),
                                    category=int(classes[str(key)])) for key in positive}
        item = dict(source_core_label=int(source_label), seed_id=marker_id,
                    core_area=int(mask.sum()),
                    box_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                    boundary_min=float(boundary[mask].min()),
                    boundary_mean=float(boundary[mask].mean()),
                    old_final_on_core=distribution(before[mask]),
                    new_final_on_core=final, final_regions=final_ids,
                    watershed_area_before_final_filter=int((watershed == marker_id).sum()),
                    survives_final_filter=bool(positive))
        if gt is not None:
            item['GT_provenance'] = dict(
                original_pixels=int((mask & gt['original']).sum()),
                filled_pixels=int((mask & gt['filled']).sum()),
                unknown_pixels=int((mask & gt['unknown']).sum()),
                original_GT=distribution(gt['ids'][mask & gt['original']]),
                trusted_GT_core=distribution(gt['cores']['core_map'][mask]))
        rows.append(item)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--core-module', type=Path, required=True)
    parser.add_argument('--core-area', type=int, default=25)
    args = parser.parse_args()
    if not 1 <= args.core_area < 50:
        parser.error('Candidate core area must be in [1,49]; final minimum remains 50')
    root, out = args.run_root.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Preserve previous candidate outputs: ' + str(out))
    out.mkdir(parents=True)
    (out / 'gallery').mkdir()
    # 真实解码器来自既有封存环境，新工具不覆盖任何生产文件。
    sys.path.insert(0, str(root / 'code'))
    from utils.config import load_config
    from utils.image_io import read_image, write_image
    from utils.boundary_core_supervision import build_trusted_cores, compare_marker_atoms
    from tools.semantic_marker_diagnostic import marker_stages
    from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
    from tools.semantic_crossover import revote_instances
    from tools.boundary_restore_monitor import panels, _heat, _overlay

    spec = importlib.util.spec_from_file_location('independent_marker_core', args.core_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuild = module.rebuild_core_markers
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    manifest_path = root / 'capture/manifest.json'
    manifest = read(manifest_path)
    if not manifest.get('complete') or len(manifest['rows']) != 8:
        raise ValueError('Expected the existing sealed eight-source control manifest')
    hashes = {str(manifest_path): sha(manifest_path), str(args.core_module): sha(args.core_module),
              str(Path(__file__).resolve()): sha(__file__)}
    for name in ['utils.post_process', 'tools.semantic_marker_diagnostic',
                 'tools.probe_affinity_marker_replay', 'tools.affinity_connectivity_views',
                 'tools.probe_affinity_partition', 'tools.semantic_crossover']:
        path = Path(sys.modules[name].__file__).resolve()
        if not path.is_relative_to(root / 'code'):
            raise ValueError('Decoder import escaped the frozen control: ' + str(path))
        hashes[str(path)] = sha(path)
    records, started = [], time.monotonic()
    try:
        for row in manifest['rows']:
            name = row['source']
            for key in ['cache', 'config', 'N1_png', 'N1_classes', 'gt', 'gt_classes']:
                if row.get(key):
                    actual = sha(row[key])
                    if row.get(key + '_sha256') and row[key + '_sha256'] != actual:
                        raise ValueError(name + ': input hash changed: ' + key)
                    hashes[row[key]] = actual
            config = load_config(row['config'])
            if int(config['inference']['min_instance_area']) != 50:
                raise ValueError('Final instance minimum must remain 50')
            config_before = json.dumps(config, sort_keys=True)
            with np.load(row['cache'], allow_pickle=False) as payload:
                arrays = {key: payload[key].copy() for key in
                          ['rgb', 'semantic', 'raw_probability', 'boundary_before']}
            B = arrays['boundary_before']
            frozen = {key: array_sha(value) for key, value in arrays.items()}
            stages = build_marker_variant(marker_stages(B, config), 'reflect', 32)
            old, control_audit = rebuild(stages['core_labels'], 50)
            candidate, candidate_audit = rebuild(stages['core_labels'], args.core_area)
            if not np.array_equal(old, stages['marker_labels']):
                raise ValueError(name + ': core=50 did not reproduce reflect markers exactly')
            gt = processed_gt(row, B.shape, build_trusted_cores)
            original_png = read_image(row['N1_png'], cv2.IMREAD_UNCHANGED)
            original_classes = read(row['N1_classes'])
            outputs, baseline_height = {}, None
            record = dict(source=name, source_group=row.get('source_group'), arms={})
            for arm, selected, audit in [('N1', old, control_audit),
                                         ('core' + str(args.core_area), candidate, candidate_audit)]:
                if arm != 'N1' and np.array_equal(selected, old):
                    previous = outputs['N1']
                    ids, classes, ws = previous['ids'], previous['classes'], previous['watershed']
                    height = baseline_height
                    capture = dict(unchanged_markers_reuse_exact_control=True)
                else:
                    replay = decode_marker_replay(arrays['semantic'],
                        torch.from_numpy(B)[None, None], config, selected)
                    if (not np.array_equal(replay['actual_markers'], selected)
                            or replay['capture_audit']['real_barrier_changed']):
                        raise ValueError('Actual watershed did not use the isolated candidate')
                    ids, ws = replay['instances'], replay['watershed']
                    classes, _ = revote_instances(ids, arrays['raw_probability'])
                    height, capture = replay['elevation_sha256'], replay['capture_audit']
                if arm == 'N1':
                    if not np.array_equal(ids, original_png) or classes != original_classes:
                        raise ValueError(name + ': baseline final PNG/classes differ')
                    baseline_height = height
                if height != baseline_height:
                    raise ValueError('Candidate changed the real watershed elevation')
                outputs[arm] = dict(ids=ids, classes=classes, watershed=ws)
                item = dict(statistics=statistics(ids, classes), core_audit=audit,
                            final_min_area=50, elevation_sha256=height, capture_audit=capture,
                            boundary_unchanged=True, semantic_and_raw_unchanged=True)
                if gt is not None:
                    item['GT_core_tracking'] = compare_marker_atoms(gt['cores'], original_png, ids)
                    item['coverage_provenance'] = {
                        key: dict(pixels=int(gt[key].sum()),
                                  old_positive_to_zero=int((gt[key] & (original_png > 0) & (ids == 0)).sum()))
                        for key in ['original', 'filled']}
                dest = out / 'results' / arm
                dest.mkdir(parents=True, exist_ok=True)
                write_image(dest / (name + '_inst.png'), ids)
                write(dest / (name + '_class.json'), classes)
                record['arms'][arm] = item
            new = outputs['core' + str(args.core_area)]
            details = candidate_details(stages['core_labels'], old, candidate, original_png,
                new['ids'], new['watershed'], new['classes'], B, gt)
            record['added_cores'] = details
            record['candidate_cores_added'] = len(details)
            record['added_cores_surviving_final_filter'] = sum(r['survives_final_filter'] for r in details)
            record['final_partition_exact'] = np.array_equal(original_png, new['ids'])
            x0, y0, x1, y1 = map(int, row['roi'])
            crops = [('fixed', (x0, y0, x1, y1))]
            if details:
                target = max(details, key=lambda value: value['watershed_area_before_final_filter'])
                x, y, right, bottom = target['box_xyxy']
                side = min(384, *B.shape)
                left = int(np.clip((x + right)//2 - side//2, 0, B.shape[1]-side))
                top = int(np.clip((y + bottom)//2 - side//2, 0, B.shape[0]-side))
                crops.append(('largest_added', (left, top, left+side, top+side)))
            marked = arrays['rgb'].copy()
            added_mask = (candidate > 0) & (old == 0)
            marked[added_mask] = (255, 60, 30)
            for title, (left, top, right, bottom) in crops:
                region = np.s_[top:bottom, left:right]
                panels(out / 'gallery' / (name + '_' + title + '.jpg'),
                    name + ' | core minimum only 50 -> ' + str(args.core_area),
                    [('RGB', arrays['rgb'][region]), ('B (fixed)', _heat(B[region])),
                     ('New core pixels', marked[region]),
                     ('N1 outlines', _overlay(arrays['rgb'][region], original_png[region])),
                     ('Candidate outlines', _overlay(arrays['rgb'][region], new['ids'][region]))],
                    260, 'Final area >=50; same B, watershed height and semantic. No test GT.')
            if (frozen != {key: array_sha(value) for key, value in arrays.items()}
                    or config_before != json.dumps(config, sort_keys=True)):
                raise ValueError('Input arrays or control configuration changed')
            records.append(record)
            write(out / 'progress.json', dict(complete=False, records=records))
            print(json.dumps(dict(source=name, admitted=len(details),
                survive=record['added_cores_surviving_final_filter'],
                before=record['arms']['N1']['statistics'],
                after=record['arms']['core' + str(args.core_area)]['statistics'])), flush=True)
        for path, expected in hashes.items():
            if sha(path) != expected:
                raise ValueError('Protected input/source changed: ' + path)
        result = dict(complete=True, source_count=len(records), records=records,
                      inputs_sha256=hashes, elapsed_seconds=time.monotonic()-started,
                      candidate_core_min_area=args.core_area, final_min_area=50,
                      no_model_forward=True, no_training=True, no_full150=True,
                      no_proxy_score=True, no_test_GT=True, no_package=True,
                      existing_deployment_unchanged=True)
        write(out / 'complete.json', result)
        print(json.dumps(dict(complete=True, seconds=result['elapsed_seconds'])), flush=True)
    except Exception as error:
        write(out / 'failed.json', dict(complete=False, error=repr(error), completed_sources=len(records)))
        raise


if __name__ == '__main__':
    main()
