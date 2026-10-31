# -*- coding: utf-8 -*-
"""正式native1024的32训练原图铁素体定位；冻结推理，不训练、不扫阈值。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analyze_affinity_tail_gt import CHECKPOINTS, existing_project_sources, pinned_files, runtime_files_contract
from train_backend_adaptation import sha, tensor_digest, write_json

NATIVE_REPORT_SHA = 'c224b6fd0290f0bffda76e6f44f9eeccd9d4c7d6389b65066a7bd0fea400e447'
FOUR_GT_REPORT_SHA = '93edc44f1b69051c0d9271b85c43dfb2d46a066cc985eb716ae65cd4eb62a634'
PARTITION_REPORT_SHA = '6a0ff413c1676c330c44d1cfd094cde1c7926ecaeeaa8c4080aeae5d758b08c1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def protect_manifest(folder, protected):
    manifest = read(folder / 'download_manifest.json')
    if manifest.get('complete') is not True:
        raise RuntimeError('Completed parent manifest required')
    pinned_files(folder, manifest['files'])
    protected[str(folder / 'download_manifest.json')] = sha(folder / 'download_manifest.json')
    protected.update({str(folder / name): value for name, value in manifest['files'].items()})


def verify_protected(protected):
    if any(sha(Path(name)) != value for name, value in protected.items()):
        raise RuntimeError('Protected source, parent, weight or data changed')


def capture(output):
    from data.affinity_mixed import build_datasets
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views, verify_prediction
    from tools.run_affinity_tail import verify_newgt, verify_snapshot
    from tools.run_affinity_short import precision_record
    from tools.verify_affinity_newgt import audit_new_gt
    from train_affinity_connectivity import frozen_digest, get_restorer
    from train_affinity_tail import CONFIG_PATH, REFERENCE
    from utils.config import load_config, project_path

    output = Path(output).resolve()
    if output != ROOT / 'outputs/ferrite_native/analysis' or output.exists():
        raise RuntimeError('Fresh outputs/ferrite_native/analysis required')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual server sam2_env GPU required')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    parent = ROOT / 'outputs/affinity_tail'
    native_folder = parent / 'analysis/partition_native'
    if (sha(native_folder / 'report.json') != NATIVE_REPORT_SHA
            or sha(parent / 'analysis/four_gt/report.json') != FOUR_GT_REPORT_SHA
            or sha(parent / 'analysis/partition/report.json') != PARTITION_REPORT_SHA):
        raise RuntimeError('Accepted parent report identity differs')
    old = read(native_folder / 'report.json')
    if not all(old.get(k) is True for k in ('complete', 'no_training', 'no_intervention',
                                           'source148_unchanged', 'formal_pipeline_unchanged')):
        raise RuntimeError('Completed formal native parent required')
    config = load_config(CONFIG_PATH)
    pipeline = read(parent / 'pipeline_status.json')
    if (pipeline['status'] != 'complete' or pipeline['stage'] != 'done'
            or len(pipeline['sources']) != 148 or pipeline.get('control_retrained') is not False
            or pipeline.get('ranking_parent_retrained') is not False
            or json.loads(json.dumps(config)) != pipeline['config']
            or json.loads(json.dumps(config)) != old['config']):
        raise RuntimeError('Accepted fixed configuration/source148 contract differs')
    verify_snapshot(parent, pipeline['sources'])
    runtime = runtime_files_contract(pipeline['runtime'], config)
    precision = precision_record(torch)
    if runtime != old['runtime_audit'] or precision != old['precision'] or precision != pipeline['precision']:
        raise RuntimeError('Fixed runtime or precision differs')
    gt_audit = audit_new_gt(config)
    verify_newgt(config, gt_audit)
    if gt_audit != old['newgt']:
        raise RuntimeError('Processed newGT cohort changed')
    checkpoint = Path(project_path(config, REFERENCE, 'final.pt'))
    if sha(checkpoint) != CHECKPOINTS['r1']:
        raise RuntimeError('Only best mixed-balance R1 e20 is accepted')
    protected = {str(Path(__file__)): sha(Path(__file__)), str(checkpoint): sha(checkpoint),
                 str(parent / 'pipeline_status.json'): sha(parent / 'pipeline_status.json'),
                 str(parent / 'candidate/status.json'): sha(parent / 'candidate/status.json'),
                 str(parent / 'analysis/partition/report.json'): PARTITION_REPORT_SHA}
    protect_manifest(native_folder, protected)
    protect_manifest(parent / 'analysis/four_gt', protected)
    imported = existing_project_sources(list(sys.modules.values()), ROOT)
    protected.update({str(ROOT / name): value for name, value in imported['files'].items()})
    model = load_backend(checkpoint, config, 'cuda')[0]
    restorer = get_restorer(config, 'cuda')
    status = read(parent / 'candidate/status.json')
    if (frozen_digest(model) != status['frozen_state_sha256']
            or tensor_digest(restorer.state_dict().items()) != status['restorer_state_sha256']):
        raise RuntimeError('Frozen encoder/LoRA/semantic/D5a identity differs')
    states = dict(model=tensor_digest(model.state_dict().items()),
                  restorer=tensor_digest(restorer.state_dict().items()))
    if states['model'] != old['frozen_states_sha256']['r1']:
        raise RuntimeError('R1 model state differs from actual formal parent')
    manual, _, _ = build_datasets(config, 1024)
    source_map = {Path(p).name: Path(p) for p in manual.whole.base_dataset.base.samples}
    previous = {row['source']: row for row in old['rows']}
    names = list(old['sources']) + sorted(set(source_map) - set(old['sources']))
    if len(source_map) != 32 or len(names) != 32 or len(set(names)) != 32:
        raise RuntimeError('Exactly all 32 allowed labelled sources required')
    seed_base = int(config['backend_adaptation']['restoration']['inference_seed'])
    seed_map = {name: seed_base + index for index, name in enumerate(names)}
    if any(seed_map[name] != previous[name]['seed'] for name in previous):
        raise RuntimeError('Previous two source seeds must remain exact')
    output.mkdir(parents=True)
    (output / 'masks').mkdir()
    (output / 'maps').mkdir()
    (output / 'sources').mkdir()
    shutil.copyfile(__file__, output / 'sources/probe_ferrite_native.py')
    rows = []
    original = cv2.watershed
    for name in names:
        path = source_map[name]
        rgb = read_rgb(str(path))
        gt_path = Path(project_path(config, config['backend_adaptation']['completed_gt_dir'], path.stem + '_gt.npz'))
        gt, lookup = load_completed_semantic_source(gt_path, rgb.shape[:2])
        classes = {int(g): int(lookup[g]) for g in np.unique(gt) if g > 0}
        for protected_path in (path, gt_path, gt_path.with_name(path.stem + '_class.json')):
            protected[str(protected_path)] = sha(protected_path)
        captures = []

        def capture_watershed(image, markers):
            before = markers.copy()
            value = original(image, markers)
            captures.append(dict(markers=before, watershed=np.maximum(value, 0).copy()))
            return value

        fixed = {}
        with patch('utils.post_process.cv2.watershed', side_effect=capture_watershed):
            actual_rgb, views = predict_views(model, restorer, path, config, 'cuda', seed_map[name], [1024], fixed_audit=fixed)
        if (set(views) != {'patch1024'} or not np.array_equal(rgb, actual_rgb)
                or len(captures) != 1 or set(fixed) != {'geometry_semantic_sha256', 'raw_probability_sha256'}):
            raise RuntimeError('Actual one-view formal inference/capture differs')
        value = views['patch1024']
        verify_prediction(value['instances'], value['classes'], gt.shape)
        marker = captures[0]['markers']
        replay = dict(source_file_sha256=sha(path), gt_file_sha256=sha(gt_path),
                      boundary_sha256=array_sha(value['boundary']), actual_marker_sha256=array_sha(marker),
                      actual_watershed_sha256=array_sha(captures[0]['watershed']))
        reuse = None
        if name in previous:
            before = previous[name]
            accepted = before['arms']['r1']
            old_mask = cv2.imread(str(native_folder / 'masks' / (path.stem + '_r1_inst.png')), cv2.IMREAD_UNCHANGED)
            old_classes = read(native_folder / 'masks' / (path.stem + '_r1_class.json'))
            if (not np.array_equal(value['instances'], old_mask) or value['classes'] != old_classes
                    or fixed != before['fixed_whole_semantics']['r1'] or value['blend'] != accepted['blend']
                    or replay['source_file_sha256'] != before['source_file_sha256']
                    or replay['gt_file_sha256'] != before['gt_file_sha256']
                    or any(replay[k] != accepted[k] for k in ('boundary_sha256', 'actual_marker_sha256', 'actual_watershed_sha256'))):
                raise RuntimeError('Previous native two-source exact parity failed')
            reuse = dict(parent_sha256=NATIVE_REPORT_SHA, final_exact=True, whole_semantics_exact=True,
                         boundary_and_actual_markers_exact=True,
                         reason='old cache has boundary hash but no continuous array; frozen R1 forward recaptures missing array')
        mask_file = 'masks/' + path.stem + '_inst.png'
        class_file = 'masks/' + path.stem + '_class.json'
        map_file = 'maps/' + path.stem + '.npz'
        if not cv2.imwrite(str(output / mask_file), value['instances']):
            raise RuntimeError('Mask write failed')
        write_json(output / class_file, value['classes'])
        np.savez_compressed(output / map_file, boundary=value['boundary'], markers=marker)
        rows.append(dict(source=name, source_path=str(path.relative_to(ROOT)), gt_path=str(gt_path.relative_to(ROOT)),
                         shape=list(gt.shape), seed=seed_map[name], **replay, gt_classes=classes,
                         fixed_whole_semantics=fixed, blend=value['blend'], mask_file=mask_file,
                         class_file=class_file, map_file=map_file, previous_exact_parity=reuse,
                         actual_watershed_calls=1, dtype='uint16', maximum_id=int(value['instances'].max())))
        write_json(output / 'capture_progress.json', dict(complete=False, rows=rows))
        print(json.dumps(dict(stage='captured', index=len(rows), total=32, source=name,
                              prior_exact_parity=reuse is not None)), flush=True)
    final = dict(model=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if states != final:
        raise RuntimeError('Frozen model state changed')
    verify_protected(protected)
    verify_snapshot(parent, pipeline['sources'])
    if (runtime_files_contract(pipeline['runtime'], config) != runtime
            or precision_record(torch) != precision or audit_new_gt(config) != gt_audit):
        raise RuntimeError('Runtime/precision/newGT changed')
    write_json(output / 'capture.json', dict(complete=True, no_training=True, no_intervention=True,
        arm='r1', epoch=20, checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=CHECKPOINTS['r1'],
        config=config, precision=precision, runtime_audit=runtime, sealed_runtime=pipeline['runtime'],
        sealed_sources=pipeline['sources'], source148_unchanged=True, formal_pipeline_unchanged=True,
        newgt=gt_audit, seed_map=seed_map, rows=rows, frozen_states_sha256=states,
        protected_sha256=protected, imported_sources=existing_project_sources(list(sys.modules.values()), ROOT),
        scope='all 32 seen labelled original training images; frozen best R1 formal native1024; not validation or official scores'))
    write_json(output / 'capture_manifest.json', dict(complete=True, files={
        p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*') if p.is_file()}))
    print('CAPTURE_COMPLETE', output, flush=True)


def analyze(output):
    from data.rgb_restoration_dataset import read_rgb
    from data.semantic_targets import load_completed_semantic_source
    from tools.affinity_native_views import verify_prediction
    from tools.analyze_ferrite_native import analyze_ferrite_native, select_ferrite_cases, render_ferrite_cases
    from tools.run_affinity_tail import verify_snapshot
    from tools.run_affinity_short import precision_record

    output = Path(output).resolve()
    if output != ROOT / 'outputs/ferrite_native/analysis' or (output / 'report.json').exists():
        raise RuntimeError('Fresh diagnosis over fixed ferrite capture required')
    capture_record = read(output / 'capture.json')
    manifest = read(output / 'capture_manifest.json')
    if not manifest['complete'] or not capture_record['complete'] or len(capture_record['rows']) != 32:
        raise RuntimeError('Complete exact 32-source frozen capture required')
    pinned_files(output, manifest['files'])
    verify_protected(capture_record['protected_sha256'])
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    core = Path(ROOT / 'tools/analyze_ferrite_native.py')
    core_sha = sha(core)
    shutil.copyfile(core, output / 'sources/analyze_ferrite_native.py')
    rows = []
    for row in capture_record['rows']:
        gt, lookup = load_completed_semantic_source(ROOT / row['gt_path'], row['shape'])
        gt_classes = {int(g): int(lookup[g]) for g in np.unique(gt) if g > 0}
        if gt_classes != {int(g): int(v) for g, v in row['gt_classes'].items()}:
            raise RuntimeError('Captured GT classes differ')
        pred = cv2.imread(str(output / row['mask_file']), cv2.IMREAD_UNCHANGED)
        pred_classes = read(output / row['class_file'])
        verify_prediction(pred, pred_classes, gt.shape)
        diagnostic = analyze_ferrite_native(gt, pred, gt_classes, pred_classes)
        cases = select_ferrite_cases(diagnostic, max_per_kind=1)
        rows.append(dict(source=row['source'], seed=row['seed'], diagnostic=diagnostic, cases=cases))
        write_json(output / 'analysis_progress.json', dict(complete=False, rows=rows))
        print(json.dumps(dict(stage='analyzed', index=len(rows), total=32, source=row['source'])), flush=True)
    # 展示规则预先固定：按源图名排序，每种机制最多2源；不按修复或观感挑样本。
    selected = []
    kind_count = {}
    for row in sorted(rows, key=lambda r: r['source']):
        cases = []
        for case in row['cases']:
            kind = case['kind']
            if kind_count.get(kind, 0) < 2:
                kind_count[kind] = kind_count.get(kind, 0) + 1
                cases.append(case)
        if cases:
            selected.append((row['source'], cases))
    captured = {r['source']: r for r in capture_record['rows']}
    gallery = output / 'gallery'
    gallery.mkdir(exist_ok=True)
    for name, cases in selected:
        row = captured[name]
        rgb = read_rgb(str(ROOT / row['source_path']))
        gt, _ = load_completed_semantic_source(ROOT / row['gt_path'], row['shape'])
        pred = cv2.imread(str(output / row['mask_file']), cv2.IMREAD_UNCHANGED)
        diagnostic = next(r['diagnostic'] for r in rows if r['source'] == name)
        with np.load(output / row['map_file'], allow_pickle=False) as arrays:
            render_ferrite_cases(gallery, Path(name).stem, rgb, gt, pred, diagnostic, cases,
                                 boundary=arrays['boundary'], markers=arrays['markers'])
    protected = capture_record['protected_sha256']
    verify_protected(protected)
    verify_snapshot(ROOT / 'outputs/affinity_tail', capture_record['sealed_sources'])
    if (sha(core) != core_sha or runtime_files_contract(capture_record['sealed_runtime'], capture_record['config'])
            != capture_record['runtime_audit'] or precision_record(torch) != capture_record['precision']):
        raise RuntimeError('Diagnostic core/runtime/precision changed')
    report = dict(complete=True, no_training=True, no_intervention=True, arm='r1', epoch=20,
                  checkpoint=capture_record['checkpoint'], checkpoint_sha256=CHECKPOINTS['r1'],
                  capture_sha256=sha(output / 'capture.json'), diagnostic_core_sha256=core_sha,
                  source148_unchanged=True, formal_pipeline_unchanged=True, rows=rows,
                  gallery_selection=dict(rule='sorted source, first at most two sources per predeclared mechanism; max one case/kind/source',
                                         selected=[dict(source=name, cases=cases) for name, cases in selected]),
                  scope=capture_record['scope'],
                  caveats=['Seen training originals with complete processed known GT including filled; unknown ignored.',
                           'Whole GT/pred frame/unknown censor and topology/neighbor leakage are separate.',
                           'Safe matched ferrite object area ratios are diagnostic, not official whole-image mean-area scores.',
                           'No teacher replacement, new holdout, training, threshold search, test GT, blackbox or deployment change.'])
    write_json(output / 'report.json', report)
    pages = ['<!doctype html><meta charset="utf-8"><title>Ferrite native1024 mechanisms</title>',
             '<p>Best R1; seen processed-newGT originals; unknown ignored; diagnostic, not official scores.</p>']
    for path in sorted(gallery.glob('*.png')):
        pages.append('<p>' + path.name + '</p><img style="max-width:100%" src="gallery/' + path.name + '">')
    (output / 'index.html').write_text('\n'.join(pages), encoding='utf8')
    write_json(output / 'download_manifest.json', dict(complete=True, files={
        p.relative_to(output).as_posix(): sha(p) for p in output.rglob('*')
        if p.is_file() and p.name != 'download_manifest.json'}))
    print('ANALYSIS_COMPLETE', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/ferrite_native/analysis')
    parser.add_argument('--mode', choices=('capture', 'analyze', 'both'), default='both')
    args = parser.parse_args()
    target = ROOT / args.output
    if args.mode in ('capture', 'both'):
        capture(target)
    if args.mode in ('analyze', 'both'):
        analyze(target)
