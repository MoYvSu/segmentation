# -*- coding: utf-8 -*-
"""成对视野短测、独立20轮与原生1024全量推理；已有控制只读复用。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_affinity_balance import SOURCES as BALANCE_SOURCES
from train_affinity_pair import audit_zero, verify_reference, verify_run
from train_backend_adaptation import sha, write_json
from utils.config import load_config, project_path

SOURCES = tuple(dict.fromkeys((*BALANCE_SOURCES, 'data/affinity_pair.py', 'train_affinity_pair.py',
    'config/train/affinity_pair.yaml', 'tools/run_affinity_pair.py',
    'tools/probe_affinity_equal_recall.py', 'tools/confidential_artifacts_guard.py')))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def diagnostic_gate(config, path):
    report = read(path)
    pair = config['affinity_pair']
    expected = {pair['reference_sha256'],
                '2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b'}
    actual = {item['sha256'] for item in report['checkpoints'].values()}
    if (not report['complete'] or report['training'] or report['source_count'] != 32 or
            report['input_views'] != 64 or report['model_views'] != 128 or actual != expected or
            report['seed'] != 20261001 or not report['d5a_frozen_exact'] or
            any(not item['frozen_exact'] for item in report['checkpoints'].values())):
        raise RuntimeError('Native/mixed balance fixed-input diagnostic is incomplete or differs')
    if report['source_sha256'] != sha(ROOT/'tools/probe_affinity_equal_recall.py'):
        raise RuntimeError('Diagnostic source differs from the reviewed comparison code')
    if len({item['shared_state_sha256'] for item in report['checkpoints'].values()}) != 1:
        raise RuntimeError('Diagnostic shared encoder/LoRA/semantic differs')
    if len(report['receipts']) != 64 or any('restored_sha256' not in r for r in report['receipts']):
        raise RuntimeError('Missing exact paired RGB/GT/D5a receipts')
    return dict(passed=True, path=str(path), sha256=sha(path),
                scope=report['scope'], evidence='mechanism only; not held-out or official performance')


def audit_outputs(config, output):
    import numpy as np
    import torch
    from data.mim_dataset import list_images
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views
    from tools.analyze_affinity_connectivity import load_directory
    from train_affinity_connectivity import get_restorer
    torch.set_num_threads(4)
    restorer = get_restorer(config, 'cuda')
    references = [('mixed', config['affinity_pair']['reference']),
                  ('native', 'outputs/affinity_balance/native')]
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    rows = []
    for kind, folder in references:
        model, _ = load_backend(project_path(config, folder, 'final.pt'), config, 'cuda')
        for index, path in enumerate(files):
            if path.stem not in config['affinity_connectivity']['parity_images']:
                continue
            seed = config['backend_adaptation']['restoration']['inference_seed'] + index
            _, views = predict_views(model, restorer, path, config, 'cuda', seed, [1024])
            ids, classes = load_directory(project_path(config, folder, 'deployment/patch1024'), path.stem)
            current = views['patch1024']
            if not np.array_equal(ids, current['instances']) or classes != current['classes']:
                raise RuntimeError('Final control output differs: ' + kind + '/' + path.stem)
            rows.append(dict(reference=kind, source=path.stem, final_output_equal=True))
        del model
        torch.cuda.empty_cache()
    if len(rows) != 2 * len(config['affinity_connectivity']['parity_images']):
        raise RuntimeError('Incomplete native/mixed final-output parity')
    write_json(Path(output)/'parity.json', dict(passed=True, control_retrained=False, rows=rows))


def render_final(config, root):
    from data.mim_dataset import list_images
    from tools.analyze_affinity_connectivity import load_directory, render
    from utils.patch_diagnostic import phase_description
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    gallery = root/'gallery'
    gallery.mkdir(exist_ok=False)
    folders = [Path(project_path(config, 'outputs/affinity_balance/native/deployment/patch1024')),
               Path(project_path(config, config['affinity_pair']['reference'], 'deployment/patch1024')),
               root/'candidate/deployment/patch1024']
    rows = []
    pages = ['<!doctype html><meta charset="utf-8"><title>Affinity paired views</title>',
             '<p>e20/native1024. Gold=ferrite, blue=pearlite; no test GT.</p>']
    for path in files:
        predictions = [load_directory(folder, path.stem) for folder in folders]
        render(path, predictions, ['native balance', 'mixed balance', 'paired views'],
               gallery/(path.stem+'.png'), path.stem, width=320)
        rows.append(dict(source=path.stem, phases=[phase_description(*v) for v in predictions]))
        pages.append(f'<p>{path.stem}<br><img loading="lazy" width="1280" src="gallery/{path.stem}.png"></p>')
    if len(rows) != 100:
        raise RuntimeError('Incomplete final gallery')
    write_json(root/'summary.json', dict(scope='Predicted distributions, not accuracy', images=rows))
    (root/'index.html').write_text('\n'.join(pages), encoding='utf8')


def run(args):
    config = load_config(args.config)
    if config['backend_adaptation']['epochs'] != 20 or config['affinity_pair']['auxiliary_weight'] != .25:
        raise ValueError('Authorized experiment is 20 epochs with auxiliary weight .25')
    verify_reference(config)
    diagnostic = diagnostic_gate(config, project_path(config, args.diagnostic))
    root = Path(project_path(config, config['backend_adaptation']['output_dir'] + ('_smoke' if args.smoke else '')))
    if shutil.disk_usage(ROOT).free < config['affinity_native']['minimum_free_gib'] * 1024**3:
        raise RuntimeError('Insufficient fast-disk space')
    if not args.smoke:
        gate = Path(str(root)+'_smoke')
        prior = read(gate/'pipeline_status.json')
        if prior['status'] != 'complete' or prior['config'] != json.loads(json.dumps(config)):
            raise RuntimeError('The exact config must pass smoke first')
        if not read(gate/'parity.json')['passed'] or not prior['zero_audit']['passed']:
            raise RuntimeError('Zero-auxiliary or final-output parity failed')
        for name in SOURCES:
            if (gate/'source'/name).read_bytes() != (ROOT/name).read_bytes():
                raise RuntimeError('Source changed after smoke: ' + name)
    root.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        if Path(name).suffix not in ('.py', '.yaml'):
            raise RuntimeError('Source snapshot must contain code/config only')
        dest = root/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    status = dict(status='running', pid=os.getpid(), smoke=args.smoke, config=config,
                  stage='preflight', completed=[], diagnostic=diagnostic,
                  sources={name:sha(ROOT/name) for name in SOURCES},
                  control_retrained=False, automatic_extension=False, competition_submission=False)
    def save():
        write_json(root/'pipeline_status.json', status)
    def execute(stage, command):
        status['stage'] = stage
        save()
        with (root/(stage+'.log')).open('w', encoding='utf8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            save()
            if child.wait():
                raise RuntimeError(stage + ' failed; inspect its log')
        status['completed'].append(stage)
        save()
    save()
    try:
        if args.smoke:
            execute('parity', [sys.executable, '-u', __file__, '--config', args.config, '--audit', str(root)])
            execute('zero', [sys.executable, '-u', 'train_affinity_pair.py', '--config', args.config,
                            '--output', str(root/'zero'), '--smoke', '--auxiliary-weight', '0'])
            status['zero_audit'] = audit_zero(config, root/'zero', expected=8)
            save()
        execute('train', [sys.executable, '-u', 'train_affinity_pair.py', '--config', args.config,
                          '--output', str(root/'candidate')] + (['--smoke'] if args.smoke else []))
        status['training_audit'] = verify_run(config, root/'candidate', 8 if args.smoke else 1280)
        save()
        if not args.smoke:
            execute('inference', [sys.executable, '-u', 'train_affinity_native.py', '--config', args.config,
                                 '--mode', 'infer', '--checkpoint', str(root/'candidate/final.pt'),
                                 '--output', str(root/'candidate/deployment'), '--views', '1024'])
            status['stage'] = 'render'
            save()
            render_final(config, root)
        status.update(status='complete', stage='done')
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_pair.yaml')
    parser.add_argument('--diagnostic', default='outputs/pair_probe/checked/report.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--audit')
    args = parser.parse_args()
    if args.audit:
        audit_outputs(load_config(args.config), args.audit)
    else:
        run(args)
