# -*- coding: utf-8 -*-
"""封存语义诊断输入：复用评分测试PNG，补四源当前reflect训练输出。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_marker_reflect import (capture_original, config_gate, load_prediction,
    precision_record, read, replay_candidates, save_prediction, sha, source_snapshot, write)

R1_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
ZIP_SHA = 'c812fbf32a87ae3a22973b2cd5548a4ab4bf9643dfc4b3136a584942d54da1e2'


def run(args):
    from data.mim_dataset import list_images
    from models.backend_adaptation import load_backend
    from train_affinity_connectivity import get_restorer
    from train_affinity_coverage import load_original_core
    from train_backend_adaptation import tensor_digest
    from tools.run_affinity_sweep import runtime_contract
    from utils.config import load_config, project_path

    out = (ROOT / args.output).resolve()
    if out.exists():
        raise FileExistsError('Fresh ignored outputs directory required')
    if not out.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Ignored outputs directory required')
    if not torch.cuda.is_available():
        raise RuntimeError('sam2_env GPU required only for four missing train captures')
    torch.set_num_threads(4); cv2.setNumThreads(2)
    config = deepcopy(read(ROOT / args.capture)['arms']['r1']['config'])
    canonical = load_config(str(ROOT / 'config/train/affinity_balance_mixed.yaml'))
    config_gate(config, canonical)
    receipt = read(ROOT / args.capture)
    r1 = receipt['arms']['r1']
    if (not receipt.get('complete') or r1['epoch'] != 20
            or r1['checkpoint_sha256'] != R1_SHA or sha(r1['checkpoint']) != R1_SHA
            or Path(project_path(config)).resolve() != ROOT):
        raise RuntimeError('Actual registered mixed R1 identity differs')
    scored = ROOT / 'outputs/marker_trial/reflect1024.zip'
    report_path = ROOT / 'outputs/marker_trial/report.json'
    scored_report = read(report_path)
    if (sha(scored) != ZIP_SHA or not scored_report.get('complete')
            or not scored_report.get('full100_complete') or scored_report['checkpoint_sha256'] != R1_SHA
            or scored_report['reflection_padding'] != 32 or not scored_report['no_seams']):
        raise RuntimeError('Scored reflect package identity differs')
    prior = ROOT / 'outputs/semantic_fixed'
    old_report = read(prior / 'report.json')
    cases = read(prior / 'cases.json')
    if (not old_report.get('complete') or len(cases) != 12
            or sum(c['kind'] == 'train' for c in cases) != 4
            or sum(c['kind'] == 'test' for c in cases) != 8):
        raise RuntimeError('Original predetermined semantic cohort differs')
    precision = precision_record(); runtime = runtime_contract(config)
    if precision != receipt['precision'] or runtime != receipt.get('current_runtime', receipt.get('runtime')):
        raise RuntimeError('Actual original49 runtime/precision differs')
    protected = {}
    def pin(p):
        p = Path(p).resolve()
        if not p.is_relative_to(ROOT):
            raise RuntimeError('Input outside current project')
        protected[str(p)] = sha(p)
        return p
    for p in (ROOT / args.capture, scored, report_path, prior / 'report.json', prior / 'cases.json', prior / 'rows.json'):
        pin(p)
    pin(r1['checkpoint'])
    pin(project_path(config, config['backend_adaptation']['restoration']['checkpoint']))
    files = [Path(p).resolve() for p in list_images(project_path(config, config['inference']['test_dir']))]
    if len(files) != 100 or files != sorted(files):
        raise RuntimeError('Original sorted test cohort differs')
    scored_rows = {row['source']: row for row in scored_report['images']}
    originals = ROOT / 'outputs/affinity_balance/mixed/deployment/patch1024'
    gt_root = Path(project_path(config, config['backend_adaptation']['completed_gt_dir']))
    for c in cases:
        path = Path(c['path']).resolve(); pin(path)
        if c['seed'] != 314159 + c['index']:
            raise RuntimeError('Original semantic source seed/index differs')
        if c['kind'] == 'test':
            if files[c['index']] != path or sha(path) != scored_rows[c['name']]['source_sha256']:
                raise RuntimeError('Scored test source identity differs')
            old = originals
        else:
            old = ROOT / 'outputs/affinity_balance/analysis/train' / c['name'] / 'mixed_local'
            pin(gt_root / (c['name'] + '_gt.npz')); pin(gt_root / (c['name'] + '_class.json'))
        pin(old / (c['name'] + '_inst.png')); pin(old / (c['name'] + '_class.json'))
    out.mkdir(); started = time.time()
    write(out / 'status.json', dict(status='running', training=False, stage='four_train_capture'))
    core = load_original_core(config); views = core.coverage_original_views
    model, bundle = load_backend(r1['checkpoint'], config, 'cuda')
    model.eval().requires_grad_(False); restorer = get_restorer(config, 'cuda')
    before = tensor_digest(model.state_dict().items()); rb = tensor_digest(restorer.state_dict().items())
    if (bundle['epoch'] != 20 or before != r1['model_state_sha256']
            or rb != receipt['restorer_state_sha256']):
        raise RuntimeError('Actual frozen R1/D5a state differs')
    parameters = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in restorer.parameters())
    if parameters >= 500_000_000 or any(m.training for m in [*model.modules(), *restorer.modules()]):
        raise RuntimeError('Actual deployment parameter/eval contract differs')
    sources, _ = source_snapshot(out)
    self_name = 'tools/prepare_semantic_reflect.py'
    (out / 'source' / self_name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / self_name, out / 'source' / self_name); sources[self_name] = sha(ROOT / self_name)
    train_report = dict(format='semantic_reflect_train_cache_v1', complete=False, no_training=True,
        checkpoint_sha256=R1_SHA, model_state_sha256=before,
        restorer_checkpoint_sha256=receipt['restorer_checkpoint_sha256'], restorer_state_sha256=rb,
        reflection_padding=32, no_seams=True, config=config, precision=precision, runtime=runtime,
        parameter_count=parameters, images=[], protected_inputs=protected, protected_code=sources,
        scope='Four registered training sources only. Original49 capture followed by actual reflect marker replay. Main semantic probe never reruns geometry.')
    train_report_path = out / 'train_cache.json'
    records = []
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False), zipfile.ZipFile(scored) as archive:
        for c in cases:
            name = c['name']; old = originals if c['kind'] == 'test' else ROOT / 'outputs/affinity_balance/analysis/train' / name / 'mixed_local'
            fixed = out / 'fixed' / name; fixed.mkdir(parents=True)
            if c['kind'] == 'train':
                rgb, captured = capture_original(views, model, restorer, c['path'], config, 'cuda', c['seed'])
                ids, classes = load_prediction(old, name, rgb.shape[:2])
                if not np.array_equal(ids, captured['baseline']['instances']) or classes != captured['baseline']['classes']:
                    raise RuntimeError('Original mixed training output does not reproduce exactly: ' + name)
                original, variants, gates = replay_candidates(captured, config)
                candidate = variants['32']
                output_hashes = save_prediction(fixed, name, candidate['instances'], candidate['classes'])
                train_report['images'].append(dict(source=name, seed=c['seed'], output=output_hashes,
                    reference=dict(png_sha256=sha(old / (name + '_inst.png')), classes_sha256=sha(old / (name + '_class.json'))),
                    baseline_exact=True, real_barrier_elevation_exact=candidate['elevation_sha256'] == original['elevation_sha256'],
                    original_replay=gates, reflect_hashes=candidate['hashes']))
                del rgb, captured, original, variants, candidate
            else:
                for suffix in ('_inst.png', '_class.json'):
                    target = fixed / (name + suffix)
                    target.write_bytes(archive.read(name + suffix))
                    cached = ROOT / 'outputs/marker_trial/deployment/patch1024' / target.name
                    if sha(target) != sha(cached):
                        raise RuntimeError('Scored ZIP byte content differs from reflect output')
                    pin(cached)
            current = deepcopy(c)
            current.update(fixed_dir=str(fixed.relative_to(ROOT)), old_dir=str(old.relative_to(ROOT)),
                source_sha256=sha(c['path']), fixed_png_sha256=sha(fixed / (name + '_inst.png')),
                fixed_classes_sha256=sha(fixed / (name + '_class.json')),
                old_png_sha256=sha(old / (name + '_inst.png')), old_classes_sha256=sha(old / (name + '_class.json')))
            if c['kind'] == 'train':
                current['reflect_origin'] = dict(path=str(train_report_path.relative_to(ROOT)), sha256=None)
                current.update(gt_path=str((gt_root / (name + '_gt.npz')).relative_to(ROOT)),
                    gt_npz_sha256=sha(gt_root / (name + '_gt.npz')), gt_classes_sha256=sha(gt_root / (name + '_class.json')))
            records.append(current)
            print('SEMANTIC INPUT ' + name, flush=True)
    if tensor_digest(model.state_dict().items()) != before or tensor_digest(restorer.state_dict().items()) != rb:
        raise RuntimeError('Frozen model/restorer changed during four-source preparation')
    if any(sha(p) != digest for p, digest in protected.items()) or any(sha(ROOT / p) != digest for p, digest in sources.items()):
        raise RuntimeError('Protected inputs/source changed during preparation')
    train_report.update(complete=True, model_unchanged=True, restorer_unchanged=True,
        protected_inputs_exact=True, protected_code_exact=True, elapsed_seconds=time.time() - started)
    write(train_report_path, train_report)
    for c in records:
        if c['kind'] == 'train': c['reflect_origin']['sha256'] = sha(train_report_path)
    inputs = dict(format='semantic_reflect_inputs_v1', complete=True, cases=records, selection=old_report['selection'],
        scored_zip=dict(path=str(scored.relative_to(ROOT)), sha256=sha(scored)),
        reflect_report=dict(path=str(report_path.relative_to(ROOT)), sha256=sha(report_path)),
        prior_report=dict(path=str((prior / 'report.json').relative_to(ROOT)), sha256=sha(prior / 'report.json')),
        prior_rows=dict(path=str((prior / 'rows.json').relative_to(ROOT)), sha256=sha(prior / 'rows.json')),
        prior_semantic_rows=dict(path=str((prior / 'rows.json').relative_to(ROOT)), sha256=sha(prior / 'rows.json')),
        prior_cases=dict(path=str((prior / 'cases.json').relative_to(ROOT)), sha256=sha(prior / 'cases.json')),
        native_checkpoint=dict(path='outputs/affinity_balance/native/final.pt'),
        preparation=dict(train_source_captures=4, test_source_captures=0, training=False))
    write(out / 'inputs.json', inputs)
    write(out / 'status.json', dict(status='complete', training=False, cases=len(records), train_source_captures=4))
    print('SEMANTIC INPUTS COMPLETE ' + str(out / 'inputs.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', default='outputs/affinity_weak/capture.json')
    parser.add_argument('--output', default='outputs/semantic_reflect_cache')
    try:
        run(parser.parse_args())
    except Exception as error:
        args = parser.parse_args(); out = (ROOT / args.output).resolve()
        if out.is_dir() and out.is_relative_to(ROOT / 'outputs') and not isinstance(error, FileExistsError):
            write(out / 'status.json', dict(status='failed', training=False, error=repr(error)))
        raise
