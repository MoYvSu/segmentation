# -*- coding: utf-8 -*-
"""两侧连接权重短测放行、依次训练与固定1024最终输出；已有控制只读复用。"""
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
from tools.run_affinity_balance import SOURCES as BALANCE_SOURCES
from train_affinity_sweep import reference_config, verify_reference, verify_run
from train_affinity_native import read
from train_backend_adaptation import sha, write_json
from utils.config import load_config, project_path

CONFIGS = {name: f'config/train/affinity_sweep_{name}.yaml' for name in ('n075', 'n125')}
SEEDS = tuple(dict.fromkeys((*BALANCE_SOURCES, *CONFIGS.values(),
    'train_affinity_sweep.py', 'tools/run_affinity_sweep.py',
    'config/train/rgb_diffusion_d5a.yaml', 'tools/confidential_artifacts_guard.py')))


def source_names():
    """补齐本地静态import和YAML继承链，只快照代码配置，不读取赛题产物。"""
    pending, found = list(SEEDS), set()
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
                inherited = (path.parent / base).resolve()
                pending.append(inherited.relative_to(ROOT).as_posix())
            continue
        package = Path(name).parent.parts
        modules = []
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
                initializer = Path(*parts[:length]) / '__init__.py'
                if (ROOT / initializer).is_file():
                    pending.append(initializer.as_posix())
    return tuple(sorted(found))


def runtime_contract(config):
    """记录实际第三方SAM2代码／配置和环境，不记录环境变量或凭据。"""
    import cv2
    import numpy as np
    import torch

    repository = Path(project_path(config, config['sam2']['sam2_repo_path'])).resolve()
    sys.path.insert(0, str(repository))
    import sam2
    package = Path(sam2.__file__).resolve().parent
    configured = config['sam2']['config_file']
    filename = configured if configured.endswith(('.yaml', '.yml')) else configured + '.yaml'
    candidates = [package / filename, package / 'configs' / filename, repository / filename,
                  repository / 'sam2' / filename]
    matches = {path.resolve() for path in candidates if path.is_file()}
    if len(matches) != 1:
        raise RuntimeError('Cannot uniquely locate actual SAM2 configuration: ' + configured)
    config_path = next(iter(matches))
    commit = subprocess.run(['git', '-C', str(repository), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=False)
    return dict(python_executable=sys.executable, python_version=sys.version, torch=torch.__version__,
        cuda=torch.version.cuda, numpy=np.__version__, cv2=cv2.__version__, sam2_package=str(package),
        sam2_repo_head=commit.stdout.strip() if commit.returncode == 0 else None,
        sam2_config=str(config_path), sam2_config_sha256=sha(config_path),
        sam2_python_sha256={path.relative_to(package).as_posix(): sha(path)
                           for path in sorted(package.rglob('*.py'))})


def verify_gate(root, configs, sources, runtime=None):
    gate = read(root / 'pipeline_status.json')
    if gate['status'] != 'complete' or gate['configs'] != json.loads(json.dumps(configs)):
        raise RuntimeError('Both exact candidates must pass smoke before training')
    if gate.get('control_retrained') is not False or not read(root / 'parity.json')['passed']:
        raise RuntimeError('Control reuse or final-output parity gate failed')
    if runtime is not None and gate.get('runtime') != runtime:
        raise RuntimeError('SAM2 source/configuration or runtime changed after smoke')
    if set(gate['sources']) != set(sources):
        raise RuntimeError('Source inventory changed after smoke')
    for name in sources:
        if sha(ROOT / name) != gate['sources'][name] or (root / 'source' / name).read_bytes() != (ROOT / name).read_bytes():
            raise RuntimeError('Source changed after smoke: ' + name)
    for name, config in configs.items():
        verify_run(config, root / name, 8)
    return gate


def audit_control(output):
    import numpy as np
    import torch
    from data.mim_dataset import list_images
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views
    from tools.analyze_affinity_connectivity import load_directory
    from train_affinity_connectivity import get_restorer
    from train_backend_adaptation import tensor_digest

    torch.set_num_threads(4)
    config = load_config(CONFIGS['n075'])
    verify_reference(config)
    base = reference_config(config)
    folder = Path(project_path(config, config['affinity_sweep']['reference']))
    model, _ = load_backend(folder / 'final.pt', base, 'cuda')
    restorer = get_restorer(base, 'cuda')
    before = tensor_digest(model.state_dict().items())
    rows = []
    for index, path in enumerate(map(Path, list_images(project_path(base, base['inference']['test_dir'])))):
        if path.stem not in base['affinity_connectivity']['parity_images']:
            continue
        _, views = predict_views(model, restorer, path, base, 'cuda',
            base['backend_adaptation']['restoration']['inference_seed'] + index, [1024])
        ids, classes = load_directory(folder / 'deployment/patch1024', path.stem)
        if not np.array_equal(ids, views['patch1024']['instances']) or classes != views['patch1024']['classes']:
            raise RuntimeError('Existing mixed balance final output differs: ' + path.stem)
        rows.append(dict(source=path.stem, final_output_equal=True))
    if len(rows) != len(base['affinity_connectivity']['parity_images']) or tensor_digest(model.state_dict().items()) != before:
        raise RuntimeError('Incomplete or mutated control parity')
    write_json(Path(output) / 'parity.json', dict(passed=True, control_retrained=False,
        checkpoint_sha256=sha(folder / 'final.pt'), rows=rows))


def render_final(configs, root):
    import cv2
    from data.mim_dataset import list_images
    from tools.affinity_native_views import verify_prediction
    from tools.analyze_affinity_connectivity import load_directory, render
    from utils.patch_diagnostic import phase_description

    config = configs['n075']
    folder = Path(project_path(config, config['affinity_sweep']['reference'], 'deployment/patch1024'))
    folders = [folder] + [root / name / 'deployment/patch1024' for name in CONFIGS]
    labels = ['mixed balance e20 r1', 'N075 e20 native1024', 'N125 e20 native1024']
    gallery = root / 'gallery'
    gallery.mkdir(exist_ok=False)
    rows = []
    pages = ['<!doctype html><meta charset="utf-8"><title>Affinity supervision sweep</title>',
        '<p>Only r changes; all e20/native1024. Gold=ferrite, blue=pearlite. No test GT.</p>']
    for path in map(Path, list_images(project_path(config, config['inference']['test_dir']))):
        shape = cv2.imread(str(path)).shape[:2]
        predictions = [load_directory(directory, path.stem) for directory in folders]
        for ids, classes in predictions:
            verify_prediction(ids, classes, shape)
        render(path, predictions, labels, gallery / (path.stem + '.png'), path.stem, width=320)
        rows.append(dict(source=path.stem, phases=[phase_description(*prediction) for prediction in predictions]))
        pages.append(f'<p>{path.stem}<br><img loading="lazy" width="1280" src="gallery/{path.stem}.png"></p>')
    if len(rows) != 100:
        raise RuntimeError('Incomplete final output/gallery cohort')
    for name, cfg in configs.items():
        manifest = read(root / name / 'deployment/manifest.json')
        expected = sha(root / name / 'final.pt')
        if (not manifest['complete'] or manifest['epoch'] != 20 or manifest['sizes'] != [1024]
                or manifest['checkpoint_sha256'] != expected or len(manifest['images']) != 100):
            raise RuntimeError('Final inference identity differs: ' + name)
    write_json(root / 'summary.json', dict(scope='Prediction distributions, not accuracy', images=rows))
    (root / 'index.html').write_text('\n'.join(pages), encoding='utf8')


def run(smoke=False):
    root = ROOT / 'outputs' / ('affinity_sweep_smoke' if smoke else 'affinity_sweep')
    configs = {name: load_config(path) for name, path in CONFIGS.items()}
    for config in configs.values():
        verify_reference(config)
    sources = source_names()
    runtime = runtime_contract(configs['n075'])
    required = max(config['affinity_native']['minimum_free_gib'] for config in configs.values())
    if shutil.disk_usage(ROOT).free < required * 1024**3:
        raise RuntimeError('Insufficient fast-disk space')
    if not smoke:
        verify_gate(ROOT / 'outputs/affinity_sweep_smoke', configs, sources, runtime)
    root.mkdir(parents=True, exist_ok=False)
    for name in sources:
        dest = root / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    status = dict(status='running', pid=os.getpid(), smoke=smoke, configs=configs, stage='preflight',
        sources={name: sha(ROOT / name) for name in sources}, runtime=runtime, completed=[], audits={},
        control_retrained=False, automatic_extension=False, competition_submission=False)

    def save():
        write_json(root / 'pipeline_status.json', status)

    def execute(stage, command):
        status['stage'] = stage
        save()
        with (root / (stage + '.log')).open('w', encoding='utf8') as log:
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
        # 先接续训练两组，均重新构造共同初态；既有控制不进入训练调度。
        for name, path in CONFIGS.items():
            execute(name, [sys.executable, '-u', 'train_affinity_sweep.py', '--config', path,
                '--output', str(root / name)] + (['--smoke'] if smoke else []))
            status['audits'][name] = verify_run(configs[name], root / name, 8 if smoke else 1280)
            save()
        for key in ('initialization_sha256', 'frozen_state_sha256', 'restorer_state_sha256'):
            if len({audit[key] for audit in status['audits'].values()}) != 1:
                raise RuntimeError('Two candidates differ in shared state: ' + key)
        if not smoke:
            for name, path in CONFIGS.items():
                execute(name + '_infer', [sys.executable, '-u', 'train_affinity_native.py', '--config', path,
                    '--mode', 'infer', '--checkpoint', str(root / name / 'final.pt'),
                    '--output', str(root / name / 'deployment'), '--views', '1024'])
            status['stage'] = 'render'
            save()
            render_final(configs, root)
        status.update(status='complete', stage='done')
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--audit')
    args = parser.parse_args()
    if args.audit:
        audit_control(args.audit)
    else:
        run(args.smoke)
