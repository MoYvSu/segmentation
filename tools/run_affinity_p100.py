# -*- coding: utf-8 -*-
"""P100独立放行与部署：来源比翻倍，旧控制只读复用，不自动打包或提交。"""
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
from tools import analyze_affinity_source as source_analysis
from tools import run_affinity_source as original_runner
from train_affinity_balance import comparable
from train_affinity_native import read
from train_affinity_p100 import reference_config, verify_config, verify_reference, verify_run
from train_backend_adaptation import sha, write_json
from utils.config import load_config, project_path

CONFIG_PATH = 'config/train/affinity_source_p100.yaml'
LEGACY_CONFIG = 'config/train/affinity_source_p025.yaml'
NEW_SOURCES = (CONFIG_PATH, 'train_affinity_p100.py', 'tools/run_affinity_p100.py',
               'tools/analyze_affinity_p100.py', 'tools/analyze_affinity_source.py',
               'tools/resume_affinity_source.py')
SMOKE_STAGES = ['parity', 'reuse_control', 'p100']


def source_names():
    """完整绑定旧122源和新入口；静态补齐import与配置继承，不读取数据。"""
    found = set(original_runner.source_names())
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


def check_snapshots(root, expected, inventory):
    """校验当前文件和训练快照；新闭包不能漏掉任何旧依赖。"""
    if not set(expected).issubset(inventory):
        raise RuntimeError('Historical source inventory is not included in the new closure')
    for name, digest in expected.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('Invalid source path: ' + name)
        if sha(ROOT / name) != digest or sha(Path(root) / 'source' / name) != digest:
            raise RuntimeError('Protected current source or archived snapshot changed: ' + name)


def technical_config(config):
    """只返回已保存技术控制的原配置；不能用新配置掩盖旧权重身份。"""
    legacy = load_config(LEGACY_CONFIG)
    result = original_runner.technical_config(legacy)
    # 两者都去掉来源实验的登记块后比较；output_dir由comparable忽略。
    if comparable(reference_config(config)) != comparable(source_analysis.reference_config(legacy)):
        raise RuntimeError('P100 differs from the reused control beyond the registered source ratio')
    return result


def verify_historical(sources, runtime):
    """既有P025的原e16、恢复门禁、e20及旧122源均必须仍可完整核验。"""
    legacy = load_config(LEGACY_CONFIG)
    run = ROOT / 'outputs/affinity_source_resume'
    pipeline = read(run / 'pipeline_status.json')
    if pipeline.get('runtime') != runtime:
        raise RuntimeError('Recovered P025 runtime differs from P100')
    check_snapshots(run, pipeline['sources'], sources)
    checks = source_analysis.preflight(legacy, run)
    lineage = checks.get('resume_lineage')
    if not lineage or checks['audit'].get('passed') is not True:
        raise RuntimeError('Completed P025 recovery lineage or full-budget audit is missing')
    report_path = run / 'analysis/report.json'
    report = read(report_path)
    analysis_sources = report.get('analysis_sources', {})
    if (report.get('complete') is not True or report.get('pipeline_sha256') != checks['pipeline_sha256']
            or set(analysis_sources) != set(source_analysis.ANALYSIS_SOURCES)):
        raise RuntimeError('Completed P025 analysis identity or dependency inventory differs')
    for name, digest in analysis_sources.items():
        if name not in sources or sha(ROOT / name) != digest:
            raise RuntimeError('Protected historical analysis implementation changed: ' + name)
    return dict(passed=True, historical_sources=len(pipeline['sources']),
        original_pipeline_sha256=sha(ROOT / 'outputs/affinity_source/pipeline_status.json'),
        resumed_pipeline_sha256=sha(run / 'pipeline_status.json'),
        resumed_receipt_sha256=sha(run / 'resume_receipt.json'),
        resumed_training_receipt_sha256=sha(run / 'p025/resume_training_receipt.json'),
        resumed_final_sha256=checks['audit']['final_checkpoint_sha256'],
        resumed_audit=checks['audit'], old_history=checks['historical'], resume_lineage=lineage,
        old_smoke_pipeline_sha256=checks['smoke_pipeline_sha256'],
        historical_analysis_report_sha256=sha(report_path), historical_analysis_sources=analysis_sources)


def control_reuse_receipt(config, runtime):
    """复用原8步技术控制；绑定其完整审计和当前100图缓存的字节哈希。"""
    legacy = load_config(LEGACY_CONFIG)
    old_root = ROOT / 'outputs/affinity_source_smoke'
    old_gate = original_runner.verify_gate(old_root, legacy, original_runner.source_names(), runtime)
    audit = source_analysis.verify_run(technical_config(config), old_root / 'technical_control', 8)
    if audit != old_gate['audits'].get('technical_control') or audit.get('passed') is not True:
        raise RuntimeError('The reused eight-update technical-control audit differs')
    folder = Path(project_path(config, config['affinity_source']['reference']))
    deployment = folder / 'deployment'
    manifest = read(deployment / 'manifest.json')
    expected = config['affinity_source']['reference_sha256']
    rows = [row for row in manifest.get('images', []) if row.get('view') == 'patch1024']
    names = [row.get('source') for row in rows]
    if (manifest.get('complete') is not True or manifest.get('epoch') != 20
            or 1024 not in manifest.get('sizes', []) or manifest.get('checkpoint_sha256') != expected
            or len(names) != 100 or len(set(names)) != 100
            or any(row.get('view') != 'patch1024' for row in rows) or sha(folder / 'final.pt') != expected):
        raise RuntimeError('The reused final control is not the registered e20/100 native1024 output')
    cache = deployment / 'patch1024'
    wanted = {stem + suffix for stem in names for suffix in ('_inst.png', '_class.json')}
    actual = {item.name for item in cache.iterdir() if item.is_file()}
    if wanted != actual:
        raise RuntimeError('The reused control output filenames or cohort differ')
    return dict(format='affinity_p100_control_reuse_v1', passed=True,
        control_retrained=False, technical_updates_reused=8,
        legacy_smoke_pipeline_sha256=sha(old_root / 'pipeline_status.json'),
        legacy_parity_sha256=sha(old_root / 'parity.json'),
        legacy_sources=old_gate['sources'], runtime=runtime, technical_audit=audit,
        technical_status_sha256=sha(old_root / 'technical_control/status.json'),
        technical_final_sha256=sha(old_root / 'technical_control/final.pt'),
        technical_head_sha256=sha(old_root / 'technical_control/last_head.pt'),
        final_control_checkpoint_sha256=expected,
        final_control_manifest_sha256=sha(deployment / 'manifest.json'),
        final_control_output_sha256={name: sha(cache / name) for name in sorted(wanted)},
        scope='Historical technical training is reused; current two-image inference parity is read-only')


def verify_gate(root, config, sources, runtime):
    gate = read(root / 'pipeline_status.json')
    if (gate.get('status') != 'complete' or gate.get('stage') != 'done' or gate.get('smoke') is not True
            or gate.get('completed') != SMOKE_STAGES or gate.get('control_retrained') is not False
            or gate.get('config') != json.loads(json.dumps(config)) or gate.get('runtime') != runtime
            or set(gate.get('sources', {})) != set(sources)):
        raise RuntimeError('Exact completed P100 smoke/config/runtime with reused control is required')
    check_snapshots(root, gate['sources'], sources)
    parity = read(root / 'parity.json')
    if (parity.get('passed') is not True or parity.get('control_retrained') is not False
            or parity.get('checkpoint_sha256') != config['affinity_source']['reference_sha256']
            or not parity.get('rows') or any(row.get('final_output_equal') is not True for row in parity['rows'])
            or [row.get('source') for row in parity['rows']] != config['affinity_connectivity']['parity_images']):
        raise RuntimeError('Current read-only final-control parity is missing or differs')
    receipt = control_reuse_receipt(config, runtime)
    if receipt != read(root / 'control_reuse.json') or receipt != gate.get('reused_control'):
        raise RuntimeError('Historical technical-control or current final-output receipt changed after smoke')
    if verify_historical(sources, runtime) != gate.get('historical'):
        raise RuntimeError('Recovered P025 lineage changed after smoke')
    audit = verify_run(config, root / 'p100', 8)
    if audit != gate.get('audits', {}).get('p100') or audit.get('passed') is not True:
        raise RuntimeError('Recorded P100 eight-update audit differs')
    return gate


def render_final(config, root):
    import cv2
    from data.mim_dataset import list_images
    from tools.affinity_native_views import verify_prediction
    from tools.analyze_affinity_connectivity import load_directory, render
    from utils.patch_diagnostic import phase_description

    control = Path(project_path(config, config['affinity_source']['reference'], 'deployment/patch1024'))
    candidate = root / 'p100/deployment/patch1024'
    gallery = root / 'gallery'
    gallery.mkdir(exist_ok=False)
    rows = []
    pages = ['<!doctype html><meta charset="utf-8"><title>P100 affinity diagnostic</title>',
        '<p>Only source ratio beta0.5 versus beta1. Gold=F, blue=P. No test GT or accuracy claim.</p>']
    for path in map(Path, list_images(project_path(config, config['inference']['test_dir']))):
        shape = cv2.imread(str(path)).shape[:2]
        predictions = [load_directory(folder, path.stem) for folder in (control, candidate)]
        for ids, classes in predictions:
            verify_prediction(ids, classes, shape)
        render(path, predictions, ['mixed balance e20 beta0.5', 'P100 e20 beta1 native1024'],
            gallery / (path.stem + '.png'), path.stem, width=360)
        rows.append(dict(source=path.stem, phases=[phase_description(*pred) for pred in predictions]))
        pages.append(f'<p>{path.stem}<br><img loading="lazy" width="1080" src="gallery/{path.stem}.png"></p>')
    manifest = read(root / 'p100/deployment/manifest.json')
    if (len(rows) != 100 or manifest.get('complete') is not True or manifest.get('epoch') != 20
            or manifest.get('sizes') != [1024] or len(manifest.get('images', [])) != 100
            or manifest.get('checkpoint_sha256') != sha(root / 'p100/final.pt')):
        raise RuntimeError('P100 complete final output contract differs')
    write_json(root / 'summary.json', dict(scope='Unlabeled predictions, not accuracy', images=rows))
    (root / 'index.html').write_text('\n'.join(pages), encoding='utf8')


def run_analysis(root):
    """正式完成回执保持不动；分析有单独状态，避免完成后哈希漂移。"""
    status = dict(status='running', stage='analysis', pid=os.getpid(), automatic_package=False,
        competition_submission=False, formal_pipeline_sha256=sha(root / 'pipeline_status.json'))
    path = root / 'analysis_pipeline_status.json'
    write_json(path, status)
    command = [sys.executable, '-u', 'tools/analyze_affinity_p100.py', '--run', str(root),
               '--output', str(root / 'analysis')]
    try:
        with (root / 'analysis.log').open('x', encoding='utf8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            write_json(path, status)
            if child.wait():
                raise RuntimeError('P100 final analysis failed; inspect analysis.log')
        if sha(root / 'pipeline_status.json') != status['formal_pipeline_sha256']:
            raise RuntimeError('Formal completed receipt changed during analysis')
        status.update(status='complete', stage='done')
        status.pop('child_pid', None)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(path, status)


def run(smoke=False):
    config = load_config(CONFIG_PATH)
    verify_config(config)
    verify_reference(config)
    sources = source_names()
    runtime = original_runner.prior_runner.runtime_contract(config)
    historical = verify_historical(sources, runtime)
    reuse = control_reuse_receipt(config, runtime)
    if shutil.disk_usage(ROOT).free < config['affinity_native']['minimum_free_gib'] * 1024**3:
        raise RuntimeError('Insufficient fast-disk space')
    smoke_gate = None
    if not smoke:
        smoke_root = ROOT / 'outputs/affinity_p100_smoke'
        verify_gate(smoke_root, config, sources, runtime)
        smoke_gate = dict(path='outputs/affinity_p100_smoke/pipeline_status.json',
            sha256=sha(smoke_root / 'pipeline_status.json'),
            control_reuse_sha256=sha(smoke_root / 'control_reuse.json'),
            parity_sha256=sha(smoke_root / 'parity.json'))
    root = ROOT / 'outputs' / ('affinity_p100_smoke' if smoke else 'affinity_p100')
    root.mkdir(parents=True, exist_ok=False)
    for name in sources:
        dest = root / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    status = dict(status='running', stage='preflight', pid=os.getpid(), smoke=smoke,
        config=config, sources={name: sha(ROOT / name) for name in sources}, runtime=runtime,
        historical=historical, reused_control=reuse, smoke_gate=smoke_gate,
        completed=[], audits={}, control_retrained=False, automatic_extension=False,
        competition_submission=False, automatic_package=False)

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
            # 新计算仅为两张最终控制输出的只读推理；不重训技术控制。
            execute('parity', [sys.executable, '-u', __file__, '--audit', str(root)])
            write_json(root / 'control_reuse.json', reuse)
            status['completed'].append('reuse_control')
            save()
        execute('p100', [sys.executable, '-u', 'train_affinity_p100.py', '--config', CONFIG_PATH,
            '--output', str(root / 'p100')] + (['--smoke'] if smoke else []))
        status['audits']['p100'] = verify_run(config, root / 'p100', 8 if smoke else 1280)
        save()
        if not smoke:
            execute('p100_infer', [sys.executable, '-u', 'train_affinity_native.py', '--config', CONFIG_PATH,
                '--mode', 'infer', '--checkpoint', str(root / 'p100/final.pt'),
                '--output', str(root / 'p100/deployment'), '--views', '1024'])
            status['stage'] = 'render'
            save()
            render_final(config, root)
        # 控制缓存与恢复谱系在本轮完成时再次核验，源closure也不允许变化。
        check_snapshots(root, status['sources'], sources)
        if control_reuse_receipt(config, runtime) != reuse or verify_historical(sources, runtime) != historical:
            raise RuntimeError('Protected historical evidence changed during this experiment')
        status.update(status='complete', stage='done')
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        save()
    if not smoke:
        run_analysis(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--audit')
    args = parser.parse_args()
    if args.audit:
        # 旧入口固定beta.5、同两张图及最终划分契约，现状只读核验。
        original_runner.prior_runner.audit_control(args.audit)
    else:
        run(args.smoke)
