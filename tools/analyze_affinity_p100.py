# -*- coding: utf-8 -*-
"""P100固定e20诊断；来源系数0.75/0.75，复用旧统计而不改写历史模块。

只在完整20轮、1280更新和100图最终输出后运行。32/4张有标签诊断图均
参加过训练，测试图仅作无标签变化分析；不选择epoch或调整部署阈值。
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import time
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import analyze_affinity_hard as hard
from tools import run_affinity_p100 as runner
from tools.run_affinity_sweep import runtime_contract
from train_affinity_balance import comparable
from train_affinity_p100 import (
    REFERENCE, REFERENCE_SHA256, reference_config, source_coefficients, verify_run)
from train_backend_adaptation import sha
from utils.config import load_config, project_path

read, write, torch = hard.read, hard.write, hard.torch
ARMS = ('r1', 'p100')
ANALYSIS_SOURCES = tuple(sorted(set(hard.ANALYSIS_SOURCES) | {
    'tools/analyze_affinity_p100.py', 'tools/run_affinity_p100.py',
    'train_affinity_p100.py', 'train_affinity_balance.py',
    'train_affinity_source.py', 'train_affinity_connectivity.py',
    'train_backend_adaptation.py', 'config/train/affinity_source_p100.yaml'}))


def require_complete_pipeline(pipeline, config):
    """拒绝中间权重、技术短测和重训控制；只接受固定e20最终流水线。"""
    if (pipeline.get('status') != 'complete' or pipeline.get('stage') != 'done'
            or pipeline.get('smoke') is not False
            or pipeline.get('control_retrained') is not False
            or pipeline.get('completed') != ['p100', 'p100_infer']
            or pipeline.get('config') != json.loads(json.dumps(config))):
        raise RuntimeError('Complete isolated P100 e20/1280-update training and final 100-image inference are required')


def require_coefficients(config):
    weights = source_coefficients(config)
    expected = dict(beta=1., alpha=.75, manual_coefficient=.75,
                    pseudo_coefficient=.75, nominal_coefficient_sum=1.5)
    if weights != expected:
        raise RuntimeError('Only P100 beta=1, manual=.75/SAM2=.75 is supported')
    return weights


def verify_sources(run, sources):
    names = runner.source_names()
    if len(sources) != len(names) or set(sources) != set(names):
        raise RuntimeError('Expected unchanged registered P100 training-source closure')
    for name, expected in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('Invalid source path: ' + name)
        if sha(ROOT / name) != expected or sha(run / 'source' / name) != expected:
            raise RuntimeError('P100 current or archived training source changed: ' + name)


def require_manifest_record(manifest, checkpoint_sha256, *, candidate):
    """预先核验100图清单；真实PNG及JSON契约随后逐图检查。"""
    selected = [row for row in manifest.get('images', []) if row.get('view') == 'patch1024']
    names = [row.get('source') for row in selected]
    if (manifest.get('complete') is not True or manifest.get('epoch') != 20
            or manifest.get('precision') != 'FP32'
            or manifest.get('checkpoint_sha256') != checkpoint_sha256
            or 1024 not in manifest.get('sizes', [])
            or len(selected) != 100 or len(set(names)) != 100
            or any(not isinstance(name, str) or not name for name in names)
            or (candidate and manifest['sizes'] != [1024])):
        raise RuntimeError('Complete fixed-e20 native1024 deployment manifest differs')
    return set(names)


def require_smoke_binding(pipeline, smoke, actual_receipt):
    """正式运行必须绑定放行时那份短测及复用控制，不能补换门禁。"""
    if (pipeline.get('smoke_gate') != actual_receipt
            or pipeline.get('reused_control') != smoke.get('reused_control')
            or not pipeline.get('reused_control', {}).get('passed')):
        raise RuntimeError('Formal P100 launch does not bind the exact smoke/reused-control receipt')


def preflight(config, run):
    pipeline_path = run / 'pipeline_status.json'
    pipeline = read(pipeline_path)
    require_complete_pipeline(pipeline, config)
    require_coefficients(config)
    verify_sources(run, pipeline['sources'])
    runtime = runtime_contract(config)
    if runtime != pipeline['runtime']:
        raise RuntimeError('Current Python/SAM2 runtime differs from actual P100 training')
    historical = runner.verify_historical(pipeline['sources'], runtime)
    if historical != pipeline['historical']:
        raise RuntimeError('Historical controls/P025 lineage or reused smoke receipts changed')
    smoke_root = ROOT / 'outputs/affinity_p100_smoke'
    smoke = runner.verify_gate(smoke_root, config, tuple(pipeline['sources']), runtime)
    smoke_binding = dict(path='outputs/affinity_p100_smoke/pipeline_status.json',
        sha256=sha(smoke_root / 'pipeline_status.json'),
        control_reuse_sha256=sha(smoke_root / 'control_reuse.json'),
        parity_sha256=sha(smoke_root / 'parity.json'))
    require_smoke_binding(pipeline, smoke, smoke_binding)
    smoke_audit = verify_run(config, smoke_root / 'p100', 8)
    if smoke_audit != smoke['audits'].get('p100') or smoke_audit.get('passed') is not True:
        raise RuntimeError('Recorded exact P100 smoke audit differs')
    audit = verify_run(config, run / 'p100', 1280)
    if audit != pipeline['audits'].get('p100') or audit.get('passed') is not True:
        raise RuntimeError('Actual 1280-update P100 audit differs from pipeline receipt')
    configs = {'r1': reference_config(config), 'p100': config}
    folders = {'r1': Path(project_path(config, REFERENCE)), 'p100': run / 'p100'}
    states, identities, cohorts = {}, {}, {}
    for name, folder in folders.items():
        state = read(folder / 'status.json')
        actual = sha(folder / 'final.pt')
        expected = REFERENCE_SHA256 if name == 'r1' else audit['final_checkpoint_sha256']
        if (state['status'] != 'complete' or state['updates'] != 1280 or state['epoch'] != 20
                or state.get('smoke') is not False or state.get('size') != 1024
                or state.get('precision') != 'FP32' or state['final_checkpoint_sha256'] != actual
                or actual != expected or comparable(state['config']) != comparable(configs[name])):
            raise RuntimeError('Actual e20 checkpoint/config identity differs: ' + name)
        states[name] = state
        coefficients = (dict(beta=.5, alpha=1., manual_coefficient=1., pseudo_coefficient=.5,
                             nominal_coefficient_sum=1.5) if name == 'r1' else require_coefficients(config))
        manifest_path = folder / 'deployment/manifest.json'
        cohorts[name] = require_manifest_record(read(manifest_path), actual, candidate=name == 'p100')
        identities[name] = dict(checkpoint=folder.joinpath('final.pt').relative_to(ROOT).as_posix(),
            checkpoint_sha256=actual, epoch=20, updates=1280, negative_weight=1.,
            hard_negative_weight=1., hard_negative_gamma=2., pseudo_weight=coefficients['beta'],
            source_coefficients=coefficients, status_sha256=sha(folder / 'status.json'),
            manifest_sha256=sha(manifest_path))
    if cohorts['r1'] != cohorts['p100']:
        raise RuntimeError('Candidate/control final-output source cohort differs')
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data'):
        if states['r1'][key] != states['p100'][key]:
            raise RuntimeError('Shared initial/frozen/restorer/data differs: ' + key)
    return dict(configs=configs, folders=folders, states=states, identities=identities,
        audit=audit, sources=pipeline['sources'], runtime=runtime, historical=historical,
        smoke_binding=smoke_binding, reused_control=pipeline['reused_control'],
        smoke_pipeline_sha256=sha(smoke_root / 'pipeline_status.json'),
        pipeline_sha256=sha(pipeline_path),
        analysis_sources={name: sha(ROOT / name) for name in ANALYSIS_SOURCES})


def _map_constant(value):
    """只改候选报告键和显示字样，保留数字、统计和部署运算。"""
    if isinstance(value, types.CodeType):
        return value.replace(co_consts=tuple(_map_constant(v) for v in value.co_consts))
    if isinstance(value, tuple):
        return tuple(_map_constant(v) for v in value)
    if isinstance(value, str):
        exact = {'H0 r1 h0 e20': 'P100 r1 h1 beta1 e20',
                 'h0 candidate': 'p100 beta1 candidate', 'H0 r1 h0': 'P100 r1 h1 beta1'}
        return exact.get(value, value.replace('H0', 'P100').replace('h0', 'p100'))
    return value


def _diagnostics():
    namespace = dict(hard.__dict__)
    namespace.update(ARMS=ARMS, REFERENCE=REFERENCE, REFERENCE_SHA256=REFERENCE_SHA256)
    for name in ('load_checked', 'gt_totals', 'probe', 'gt', 'test'):
        original = getattr(hard, name)
        adapted = types.FunctionType(_map_constant(original.__code__), namespace, name,
            original.__defaults__, original.__closure__)
        adapted.__kwdefaults__ = original.__kwdefaults__
        namespace[name] = adapted
    return namespace


_DIAGNOSTICS = _diagnostics()
probe, gt, test = (_DIAGNOSTICS[name] for name in ('probe', 'gt', 'test'))


def self_test():
    hard.self_test()
    config = {'sample': True}
    complete = dict(status='complete', stage='done', smoke=False, control_retrained=False,
                    completed=['p100', 'p100_infer'], config=config)
    require_complete_pipeline(complete, config)
    for key, value in (('status', 'running'), ('completed', ['p100']), ('smoke', True),
                       ('control_retrained', True), ('config', {})):
        try:
            require_complete_pipeline(dict(complete, **{key: value}), config)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incomplete or changed P100 accepted: ' + key)
    if hard.ARMS != ('r1', 'h0') or _DIAGNOSTICS['ARMS'] != ARMS:
        raise AssertionError('Diagnostic reuse modified the old module namespace')
    for name in ('load_checked', 'gt_totals', 'probe', 'gt', 'test'):
        original, adapted = getattr(hard, name), _DIAGNOSTICS[name]
        if adapted.__code__.co_code != original.__code__.co_code or adapted.__defaults__ != original.__defaults__:
            raise AssertionError('Diagnostic operations/defaults changed')
    print('P100_CPU_PASS: incomplete output rejected, labels isolated, old diagnostics unchanged')


def run(args):
    config = load_config(args.config)
    runroot = Path(project_path(config, args.run))
    require_complete_pipeline(read(runroot / 'pipeline_status.json'), config)
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('Actual probe/GT require server sam2_env GPU; --self-test is CPU only')
    torch.set_num_threads(4)
    hard.cv2.setNumThreads(2)
    output = Path(project_path(config, args.output))
    started = time.time()
    checks = preflight(config, runroot)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'config.json', config)
    write(output / 'preflight.json', {key: checks[key] for key in ('audit', 'sources', 'runtime',
        'historical', 'smoke_binding', 'reused_control', 'smoke_pipeline_sha256',
        'pipeline_sha256', 'analysis_sources', 'identities')})
    try:
        write(output / 'status.json', dict(status='running', stage='probe', training=False))
        probe_report = probe(config, checks, output / 'probe', args.device, args.seed)
        write(output / 'status.json', dict(status='running', stage='gt', training=False))
        gt_report = gt(config, checks, output / 'gt', args.device,
                       Path(project_path(config, args.reference_gt)))
        write(output / 'status.json', dict(status='running', stage='test', training=False))
        test_report = test(config, checks, output / 'test', args.seed)
        verify_sources(runroot, checks['sources'])
        if sha(runroot / 'pipeline_status.json') != checks['pipeline_sha256']:
            raise RuntimeError('Completed training pipeline changed during diagnostics')
        if runner.verify_historical(checks['sources'], checks['runtime']) != checks['historical']:
            raise RuntimeError('Reused control/P025 lineage changed during diagnostics')
        if (runner.control_reuse_receipt(config, checks['runtime']) != checks['reused_control']
                or sha(ROOT / checks['smoke_binding']['path']) != checks['smoke_pipeline_sha256']):
            raise RuntimeError('Reused final control or short-test gate changed during diagnostics')
        if any(sha(ROOT / name) != value for name, value in checks['analysis_sources'].items()):
            raise RuntimeError('Analysis dependency changed during diagnostics')
        report = dict(complete=True, training=False, test_inference_rerun=False, epoch_selection=False,
            postprocessing_tuned=False, official_accuracy=False,
            checkpoint_selection='Fixed e20; no test/GT epoch selection', identities=checks['identities'],
            audit=checks['audit'], sources=checks['sources'], source_count=len(checks['sources']),
            current_and_snapshot_exact=True, runtime=checks['runtime'], historical=checks['historical'],
            smoke_binding=checks['smoke_binding'], reused_control=checks['reused_control'],
            pipeline_sha256=checks['pipeline_sha256'], smoke_pipeline_sha256=checks['smoke_pipeline_sha256'],
            analysis_sources=checks['analysis_sources'],
            diagnostic_reuse='H0 statistics with isolated labels; P100 independent strict training/final-output preflight',
            probe=probe_report, gt=gt_report,
            test={key: value for key, value in test_report.items() if key not in ('images', 'output_receipts')},
            elapsed_seconds=time.time() - started,
            caveats=['32/4 GT sources participated in training; mechanism checks do not establish generalization.',
                     'Unknown original GT is ignored; test sources have no GT.',
                     'Prediction counts/areas or equal-recall thresholds do not choose deployment or tune postprocessing.'])
        write(output / 'report.json', report)
        pages = ['<!doctype html><meta charset="utf-8"><title>P100 affinity fixed e20</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>mixed balance beta0.5 versus P100 beta1, fixed e20</h1>',
            '<p>Only SAM2/manual supervision coefficients change: 1/0.5 to 0.75/0.75. r1/h1/gamma2 fixed. Gold=F, blue=P. Seen GT known regions only; test has no GT.</p>',
            '<p><a href="report.json">Complete report</a> | <a href="test/test.json">100 test rows</a> | <a href="gt/report.json">Four seen GT</a></p>',
            '<p>Equal-recall plot: control=beta0.5, balance=P100 beta1; thresholds only diagnose.</p>',
            '<img src="probe/equal_recall.png">']
        for folder in (output / 'gt', output / 'test/gallery'):
            for path in sorted(folder.glob('*.png')):
                rel = html.escape(path.relative_to(output).as_posix())
                pages.append(f'<p>{html.escape(path.stem)}<br><img loading="lazy" src="{rel}"></p>')
        (output / 'index.html').write_text('\n'.join(pages), encoding='utf8')
        write(output / 'status.json', dict(status='complete', stage='done', training=False,
            probe_sources=32, gt_sources=4, test_sources=100, elapsed_seconds=time.time() - started))
        with zipfile.ZipFile(output / 'report.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.html', '.png', '.npz'):
                    archive.write(path, path.relative_to(output).as_posix())
        print('COMPLETE P100 analysis', output, flush=True)
    except Exception as error:
        write(output / 'status.json', dict(status='failed', training=False, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_source_p100.yaml')
    parser.add_argument('--run', default='outputs/affinity_p100')
    parser.add_argument('--output', default='outputs/affinity_p100/analysis')
    parser.add_argument('--reference-gt', default='outputs/affinity_sweep/analysis/gt/report.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--self-test', action='store_true')
    options = parser.parse_args()
    self_test() if options.self_test else run(options)
