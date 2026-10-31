# -*- coding: utf-8 -*-
"""既有N1缓存的单高阈值种子对照；不改变实际分水岭，不训练或出包。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def overlap_change(before, after):
    """空间重叠描述，不将再生种子的ID或拆合数量当作准确率。"""
    result = {}
    for name, source, target in [('old_regions_split', before, after),
                                  ('new_regions_overlap_multiple_old', after, before)]:
        rows = []
        for label in np.unique(source):
            if not label:
                continue
            mask = source == label
            labels, counts = np.unique(target[mask], return_counts=True)
            selected = (labels > 0) & (counts >= 50) & (counts >= .05 * mask.sum())
            if selected.sum() > 1:
                rows.append(dict(source_id=int(label), source_area=int(mask.sum()),
                                 target_counts={str(int(k)): int(v)
                                                for k, v in zip(labels[selected], counts[selected])}))
        result[name] = rows
    result['old_positive_to_zero'] = int(((before > 0) & (after == 0)).sum())
    result['old_zero_to_positive'] = int(((before == 0) & (after > 0)).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--marker-module', type=Path, required=True)
    parser.add_argument('--common-module', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_root.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError('Preserve previous outputs: ' + str(out))
    out.mkdir(parents=True)
    (out / 'gallery').mkdir()
    sys.path.insert(0, str(root / 'code'))
    common = load_module('marker_cache_common', args.common_module)
    from utils.config import load_config
    from utils.image_io import read_image, write_image
    from utils.boundary_core_supervision import build_trusted_cores, compare_marker_atoms
    from tools.semantic_marker_diagnostic import marker_stages
    from tools.probe_affinity_marker_replay import build_marker_variant, decode_marker_replay
    from tools.semantic_crossover import revote_instances
    from tools.boundary_restore_monitor import panels, _overlay, _heat
    candidate_module = load_module('independent_marker_single', args.marker_module)
    build_candidate = candidate_module.high_only_reflect_markers
    read, write, sha = common.read, common.write, common.sha
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    manifest_path = root / 'capture/manifest.json'
    manifest = read(manifest_path)
    if not manifest.get('complete') or len(manifest['rows']) != 8:
        raise ValueError('Expected the existing sealed eight-source manifest')
    hashes = {str(path): sha(path) for path in [manifest_path, args.marker_module,
              args.common_module, Path(__file__).resolve()]}
    for name in ['utils.post_process', 'tools.semantic_marker_diagnostic',
                 'tools.probe_affinity_marker_replay', 'tools.affinity_connectivity_views',
                 'tools.probe_affinity_partition', 'tools.semantic_crossover']:
        path = Path(sys.modules[name].__file__).resolve()
        if not path.is_relative_to(root / 'code'):
            raise ValueError('Import escaped the sealed decoder: ' + str(path))
        hashes[str(path)] = sha(path)
    records, started = [], time.monotonic()
    try:
        for row in manifest['rows']:
            name = row['source']
            for key in ['cache', 'config', 'N1_png', 'N1_classes', 'gt', 'gt_classes']:
                if row.get(key):
                    actual = sha(row[key])
                    if row.get(key + '_sha256') and row[key + '_sha256'] != actual:
                        raise ValueError(name + ': changed input ' + key)
                    hashes[row[key]] = actual
            config = load_config(row['config'])
            config_before = json.dumps(config, sort_keys=True)
            if int(config['inference']['min_instance_area']) != 50:
                raise ValueError('Preserve original final minimum 50')
            with np.load(row['cache'], allow_pickle=False) as payload:
                arrays = {key: payload[key].copy() for key in
                          ['rgb', 'semantic', 'raw_probability', 'boundary_before']}
            B = arrays['boundary_before']
            fixed = {key: common.array_sha(value) for key, value in arrays.items()}
            stages = marker_stages(B, config)
            baseline = build_marker_variant(stages, 'reflect', 32)
            candidate = build_candidate(stages, reflection_padding=32)
            if not np.array_equal(candidate['barrier_belt'], baseline['barrier_belt']):
                raise ValueError('High-only seed generation changed the real barrier')
            gt = common.processed_gt(row, B.shape, build_trusted_cores)
            original = read_image(row['N1_png'], cv2.IMREAD_UNCHANGED)
            original_classes = read(row['N1_classes'])
            record = dict(source=name, source_group=row.get('source_group'), arms={})
            outputs, baseline_height = {}, None
            for arm, current in [('N1', baseline), ('high_only', candidate)]:
                markers = current['marker_labels']
                if arm != 'N1' and np.array_equal(markers, baseline['marker_labels']):
                    ids, classes = outputs['N1']['ids'], outputs['N1']['classes']
                    height = baseline_height
                    capture = dict(unchanged_seeds_reuse_exact_baseline=True)
                else:
                    replay = decode_marker_replay(arrays['semantic'],
                        torch.from_numpy(B)[None, None], config, markers)
                    if (not np.array_equal(replay['actual_markers'], markers)
                            or replay['capture_audit']['real_barrier_changed']):
                        raise ValueError('The real watershed did not use the isolated seeds')
                    ids = replay['instances']
                    classes, _ = revote_instances(ids, arrays['raw_probability'])
                    height, capture = replay['elevation_sha256'], replay['capture_audit']
                if arm == 'N1':
                    if not np.array_equal(ids, original) or classes != original_classes:
                        raise ValueError(name + ': baseline final output differs')
                    baseline_height = height
                if height != baseline_height:
                    raise ValueError('Real watershed height changed')
                outputs[arm] = dict(ids=ids, classes=classes)
                stats = common.statistics(ids, classes)
                labels, sizes = np.unique(ids, return_counts=True)
                if np.any(sizes[labels > 0] < 50):
                    raise ValueError('Final minimum area changed')
                item = dict(statistics=stats, marker_count=int(current['marker_count']),
                    actual_marker_sha256=common.array_sha(markers),
                    elevation_sha256=height, capture_audit=capture, final_min_area=50,
                    only_seed_generation_changed=True)
                if gt is not None:
                    item['GT_core_tracking'] = compare_marker_atoms(gt['cores'], original, ids)
                    item['coverage_provenance'] = {
                        key: dict(pixels=int(gt[key].sum()),
                            old_positive_to_zero=int((gt[key] & (original > 0) & (ids == 0)).sum()))
                        for key in ['original', 'filled']}
                dest = out / 'results' / arm
                dest.mkdir(parents=True, exist_ok=True)
                write_image(dest / (name + '_inst.png'), ids)
                write(dest / (name + '_class.json'), classes)
                record['arms'][arm] = item
            changed = outputs['high_only']['ids']
            record['overlap_change'] = overlap_change(original, changed)
            record['single_threshold_audit'] = candidate['single_threshold_audit']
            record['final_exact'] = bool(np.array_equal(original, changed)
                and outputs['high_only']['classes'] == original_classes)
            record['marker_rule'] = dict(
                original_low=config['inference'].get('marker_boundary_low_threshold'),
                original_steps=config['inference'].get('marker_boundary_reconstruction_steps'),
                candidate_threshold=float(config['inference']['boundary_threshold']),
                candidate_weak_reconstruction=False, reflection_padding=32,
                candidate_core_min_area=50, final_min_area=50,
                configuration_unchanged=True, real_watershed_unchanged=True)
            x0, y0, x1, y1 = map(int, row['roi'])
            region = np.s_[y0:y1, x0:x1]
            panels(out / 'gallery' / (name + '_fixed.jpg'), name + ' | remove weak seed reconstruction only',
                [('RGB', arrays['rgb'][region]), ('B (fixed)', _heat(B[region])),
                 ('N1 outlines', _overlay(arrays['rgb'][region], original[region])),
                 ('High-only outlines', _overlay(arrays['rgb'][region], changed[region]))], 280,
                'Same real watershed / semantic / reflect32 / area50. No test GT.')
            # 用差异边界最多的固定网格窗口展示变化；不选择或推断测试答案。
            def edges(ids):
                mask = np.zeros(ids.shape, bool)
                mask[1:] |= ids[1:] != ids[:-1]
                mask[:, 1:] |= ids[:, 1:] != ids[:, :-1]
                return mask
            difference = edges(original) != edges(changed)
            if difference.any():
                side = min(384, *B.shape)
                ys = sorted({*range(0, B.shape[0]-side+1, max(1, side//2)), B.shape[0]-side})
                xs = sorted({*range(0, B.shape[1]-side+1, max(1, side//2)), B.shape[1]-side})
                choices = [(int(difference[y:y+side, x:x+side].sum()), y, x) for y in ys for x in xs]
                _, y, x = max(choices)
                region = np.s_[y:y+side, x:x+side]
                record['largest_difference_roi'] = [x, y, x+side, y+side]
                panels(out / 'gallery' / (name + '_changed.jpg'), name + ' | largest contour-change grid crop',
                    [('RGB', arrays['rgb'][region]), ('B (fixed)', _heat(B[region])),
                     ('N1 outlines', _overlay(arrays['rgb'][region], original[region])),
                     ('High-only outlines', _overlay(arrays['rgb'][region], changed[region]))], 280,
                    'Diagnostic change selection, not accuracy or test-GT selection.')
            if (fixed != {key: common.array_sha(value) for key, value in arrays.items()}
                    or config_before != json.dumps(config, sort_keys=True)):
                raise ValueError('Control configuration or input arrays mutated')
            records.append(record)
            write(out / 'progress.json', dict(complete=False, records=records))
            print(json.dumps(dict(source=name,
                N1=record['arms']['N1']['statistics'],
                high_only=record['arms']['high_only']['statistics'],
                GT=record['arms']['high_only'].get('GT_core_tracking', {}).get('summary'))), flush=True)
        for path, expected in hashes.items():
            if sha(path) != expected:
                raise ValueError('Protected input/source changed: ' + path)
        result = dict(complete=True, source_count=len(records), records=records,
            elapsed_seconds=time.monotonic()-started, inputs_sha256=hashes,
            no_model_forward=True, no_training=True, no_full150=True, no_proxy_score=True,
            no_test_GT=True, no_package=True, existing_deployment_unchanged=True)
        write(out / 'complete.json', result)
        print(json.dumps(dict(complete=True, seconds=result['elapsed_seconds'])), flush=True)
    except Exception as error:
        write(out / 'failed.json', dict(complete=False, error=repr(error), completed_sources=len(records)))
        raise


if __name__ == '__main__':
    main()
