# -*- coding: utf-8 -*-
"""单独关闭困难负关系加权；短测后训练H0，不重训控制或自动提交。"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_affinity_sweep as prior_runner
from train_affinity_hard import verify_reference, verify_run
from train_affinity_native import read
from train_backend_adaptation import sha, write_json
from utils.config import load_config, project_path

CONFIG_PATH = 'config/train/affinity_hard_h0.yaml'
NEW_SOURCES = (CONFIG_PATH, 'train_affinity_hard.py', 'tools/run_affinity_hard.py')


def source_names():
    """保留原闭包，并递归补齐本轮新入口的静态import与配置继承。"""
    found = set(prior_runner.source_names())
    pending = list(NEW_SOURCES)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / name
        if not path.is_file() or path.suffix not in ('.py', '.yaml', '.yml'):
            raise RuntimeError('Missing or invalid source: ' + name)
        found.add(name)
        if path.suffix in ('.yaml', '.yml'):
            base = (yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('_base')
            if base:
                pending.append((path.parent / base).resolve().relative_to(ROOT).as_posix())
            continue
        package, modules = Path(name).parent.parts, []
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'), filename=name)):
            if isinstance(node, ast.Import):
                modules.extend(alias.name.split('.') for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                module = prefix + (node.module.split('.') if node.module else [])
                modules.append(module)
                modules.extend(module + alias.name.split('.') for alias in node.names if alias.name != '*')
        for parts in modules:
            if not parts:
                continue
            for candidate in (Path(*parts).with_suffix('.py'), Path(*parts) / '__init__.py'):
                if (ROOT / candidate).is_file():
                    pending.append(candidate.as_posix())
            for length in range(1, len(parts)):
                init = Path(*parts[:length]) / '__init__.py'
                if (ROOT / init).is_file():
                    pending.append(init.as_posix())
    return tuple(sorted(found))


def verify_historical(sources, runtime):
    """旧r实验不重跑，也不覆盖它绑定的服务器依赖。"""
    root = ROOT / 'outputs/affinity_sweep'
    status = read(root / 'pipeline_status.json')
    if status['status'] != 'complete' or status.get('control_retrained') is not False:
        raise RuntimeError('The prior isolated sweep must be complete')
    if status.get('runtime') != runtime or not set(status['sources']).issubset(sources):
        raise RuntimeError('Historical runtime/source inventory changed')
    for name, expected in status['sources'].items():
        if sha(ROOT / name) != expected or sha(root / 'source' / name) != expected:
            raise RuntimeError('Historical trained source changed: ' + name)
    return dict(passed=True, historical_sources=len(status['sources']),
                pipeline_sha256=sha(root / 'pipeline_status.json'))


def verify_gate(root, config, sources, runtime):
    status = read(root / 'pipeline_status.json')
    if (status['status'] != 'complete' or status.get('smoke') is not True
            or status['config'] != json.loads(json.dumps(config))
            or status.get('control_retrained') is not False
            or status.get('runtime') != runtime
            or set(status['sources']) != set(sources)
            or not read(root / 'parity.json').get('passed')):
        raise RuntimeError('Exact H0 smoke/runtime/control parity is required')
    for name in sources:
        if sha(ROOT / name) != status['sources'][name] or sha(root / 'source' / name) != status['sources'][name]:
            raise RuntimeError('Source changed after smoke: ' + name)
    verify_run(config, root / 'h0', 8)
    return status


def render_final(config, root):
    import cv2
    from data.mim_dataset import list_images
    from tools.affinity_native_views import verify_prediction
    from tools.analyze_affinity_connectivity import load_directory, render
    from utils.patch_diagnostic import phase_description

    control = Path(project_path(config, config['affinity_hard']['reference'], 'deployment/patch1024'))
    candidate = root / 'h0/deployment/patch1024'
    gallery = root / 'gallery'
    gallery.mkdir(exist_ok=False)
    rows = []
    pages = ['<!doctype html><meta charset="utf-8"><title>H0 affinity diagnostic</title>',
             '<p>r=1 fixed, hard-negative weight 1 versus 0. Gold=F, blue=P. No test GT.</p>']
    for path in map(Path, list_images(project_path(config, config['inference']['test_dir']))):
        shape = cv2.imread(str(path)).shape[:2]
        predictions = [load_directory(folder, path.stem) for folder in (control, candidate)]
        for ids, classes in predictions:
            verify_prediction(ids, classes, shape)
        render(path, predictions, ['mixed balance e20 h1', 'H0 e20 native1024'],
               gallery / (path.stem + '.png'), path.stem, width=360)
        rows.append(dict(source=path.stem, phases=[phase_description(*pred) for pred in predictions]))
        pages.append(f'<p>{path.stem}<br><img loading="lazy" width="1080" src="gallery/{path.stem}.png"></p>')
    manifest = read(root / 'h0/deployment/manifest.json')
    if (len(rows) != 100 or manifest['complete'] is not True or manifest['epoch'] != 20
            or manifest['sizes'] != [1024] or len(manifest['images']) != 100
            or manifest['checkpoint_sha256'] != sha(root / 'h0/final.pt')):
        raise RuntimeError('H0 complete final output contract differs')
    write_json(root / 'summary.json', dict(scope='Unlabeled predictions, not accuracy', images=rows))
    (root / 'index.html').write_text('\n'.join(pages), encoding='utf8')


def run(smoke=False):
    config = load_config(CONFIG_PATH)
    verify_reference(config)
    sources = source_names()
    runtime = prior_runner.runtime_contract(config)
    historical = verify_historical(sources, runtime)
    if shutil.disk_usage(ROOT).free < config['affinity_native']['minimum_free_gib'] * 1024**3:
        raise RuntimeError('Insufficient fast-disk space')
    if not smoke:
        verify_gate(ROOT / 'outputs/affinity_hard_smoke', config, sources, runtime)
    root = ROOT / 'outputs' / ('affinity_hard_smoke' if smoke else 'affinity_hard')
    root.mkdir(parents=True, exist_ok=False)
    for name in sources:
        dest = root / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    status = dict(status='running', stage='preflight', pid=os.getpid(), smoke=smoke,
        config=config, sources={name: sha(ROOT/name) for name in sources}, runtime=runtime,
        historical=historical, completed=[], audits={}, control_retrained=False,
        automatic_extension=False, competition_submission=False, automatic_package=False)

    def save():
        write_json(root / 'pipeline_status.json', status)

    def execute(stage, command):
        status['stage'] = stage
        save()
        with (root / (stage + '.log')).open('x', encoding='utf8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            save()
            if child.wait():
                raise RuntimeError(stage + ' failed; inspect its log')
        status['completed'].append(stage)
        status.pop('child_pid', None)
        save()

    save()
    try:
        if smoke:
            execute('parity', [sys.executable, '-u', __file__, '--audit', str(root)])
        execute('h0', [sys.executable, '-u', 'train_affinity_hard.py', '--config', CONFIG_PATH,
            '--output', str(root/'h0')] + (['--smoke'] if smoke else []))
        status['audits']['h0'] = verify_run(config, root/'h0', 8 if smoke else 1280)
        save()
        if not smoke:
            execute('h0_infer', [sys.executable, '-u', 'train_affinity_native.py', '--config', CONFIG_PATH,
                '--mode', 'infer', '--checkpoint', str(root/'h0/final.pt'),
                '--output', str(root/'h0/deployment'), '--views', '1024'])
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
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--audit')
    args = parser.parse_args()
    if args.audit:
        # 复用旧入口验证同一已评分r1两图最终输出；不训练或新增控制。
        prior_runner.audit_control(args.audit)
    else:
        run(args.smoke)
