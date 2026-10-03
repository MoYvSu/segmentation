# -*- coding: utf-8 -*-
"""P025固定e20诊断；复用H0统计实现，但独立核验122源及完整来源比例实验。

未完成20轮/1280更新和100图推理时拒绝实际分析。三段诊断复用旧实现的
字节码，只在独立命名空间中把候选报告键和显示名称由h0映射为p025；
旧模块、训练源和历史产物均不写入，旧H0的119源preflight也不被调用。
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
from tools import run_affinity_source as runner
from tools.run_affinity_sweep import runtime_contract
from train_affinity_balance import comparable
from train_affinity_source import (
    REFERENCE, REFERENCE_SHA256, reference_config, source_coefficients, verify_run)
from train_backend_adaptation import sha
from utils.config import load_config, project_path

read, write = hard.read, hard.write
torch = hard.torch
ARMS = ('r1', 'p025')
SOURCE_COUNT = 122
ANALYSIS_SOURCES = tuple(sorted(set(hard.ANALYSIS_SOURCES) | {
    'tools/analyze_affinity_source.py', 'tools/run_affinity_source.py',
    'train_affinity_source.py', 'train_affinity_balance.py',
    'train_affinity_connectivity.py', 'train_backend_adaptation.py',
    'config/train/affinity_source_p025.yaml'}))


def require_complete_pipeline(pipeline, config):
    """先拒绝partial，不能用中间checkpoint或缺失推理冒充e20分析。"""
    if (pipeline.get('status') != 'complete' or pipeline.get('stage') != 'done'
            or pipeline.get('smoke') is not False
            or pipeline.get('control_retrained') is not False
            or pipeline.get('completed') != ['p025', 'p025_infer']
            or pipeline.get('config') != json.loads(json.dumps(config))):
        raise RuntimeError('Complete isolated P025 e20/1280-update training and final 100-image inference are required; partial output is not analyzable')


def verify_sources(run, sources):
    if len(sources) != SOURCE_COUNT or set(sources) != set(runner.source_names()):
        raise RuntimeError('Expected unchanged registered 122-source P025 training closure')
    for name, expected in sources.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('Invalid source path: ' + name)
        if sha(ROOT / name) != expected or sha(run / 'source' / name) != expected:
            raise RuntimeError('P025 current or archived training source changed: ' + name)


def _prefix_bytes(path, rows):
    with Path(path).open('rb') as stream:
        lines = [stream.readline() for _ in range(rows)]
    if any(not line or not line.endswith(b'\n') for line in lines):
        raise RuntimeError('Incomplete original text receipt prefix: ' + str(path))
    return b''.join(lines)


def require_resume_smoke_records(config, formal_pipeline, formal_receipt, gate, smoke_receipt, probe):
    """单步恢复门禁和正式续训必须绑定同一原e16，而不是另一套初态。"""
    if (gate.get('status') != 'complete' or gate.get('stage') != 'done'
            or gate.get('smoke') is not True or gate.get('completed') != ['resume_smoke']
            or gate.get('control_retrained') is not False
            or gate.get('config') != json.loads(json.dumps(config))
            or gate.get('runtime') != formal_pipeline['runtime']
            or gate.get('sources') != formal_pipeline['sources']
            or gate.get('resume_sources') != formal_pipeline['resume_sources']
            or smoke_receipt.get('identity') != formal_receipt['identity']
            or smoke_receipt.get('original_run_preserved') is not True
            or smoke_receipt.get('resumed_new_updates') != 1
            or smoke_receipt.get('finished_training') is not False):
        raise RuntimeError('Complete one-update resume smoke belongs to another original/config/runtime/source')
    expected = dict(passed=True, epoch=17, step=1, updates=1025, new_updates=1,
        original_total_epochs=20, scheduler_last_epoch=16, scheduler_not_restarted=True,
        restored_optimizer_scheduler_rng=True, strict_head_reload_equal=True,
        frozen_unchanged=True, restorer_unchanged=True)
    if any(probe.get(key) != value for key, value in expected.items()):
        raise RuntimeError('One-update resume smoke state/freeze/reload/scheduler gate failed')


def verify_resume_smoke(config, pipeline, receipt, original_head):
    gate_receipt = receipt.get('smoke_gate')
    expected_path = 'outputs/affinity_source_resume_smoke/resume_receipt.json'
    if not isinstance(gate_receipt, dict) or gate_receipt.get('path') != expected_path:
        raise RuntimeError('Formal resume is missing the exact one-update smoke receipt path')
    smoke_root = ROOT / 'outputs/affinity_source_resume_smoke'
    if (sha(smoke_root / 'resume_receipt.json') != gate_receipt.get('sha256')
            or sha(smoke_root / 'p025/smoke_receipt.json') != gate_receipt.get('probe_sha256')):
        raise RuntimeError('Resume smoke or probe receipt changed after formal launch')
    gate = read(smoke_root / 'pipeline_status.json')
    smoke_receipt = read(smoke_root / 'resume_receipt.json')
    probe = read(smoke_root / 'p025/smoke_receipt.json')
    require_resume_smoke_records(config, pipeline, receipt, gate, smoke_receipt, probe)
    verify_sources(smoke_root, pipeline['sources'])
    for name, expected in pipeline['resume_sources'].items():
        if (sha(ROOT / name) != expected
                or sha(smoke_root / 'resume_source' / name) != expected):
            raise RuntimeError('Resume smoke implementation or source snapshot changed')
    checkpoint = smoke_root / 'p025/probe_head.pt'
    if sha(checkpoint) != probe.get('probe_head_sha256'):
        raise RuntimeError('Actual restored one-update checkpoint SHA differs from the gate')
    head = torch.load(checkpoint, map_location='cpu', weights_only=False)
    extra = head.get('source_state', {})
    optimizer, scheduler = head.get('optimizer', {}), head.get('scheduler', {})
    if (head.get('format') != 'affinity_source_resume_smoke_v1'
            or head.get('epoch') != 17 or head.get('step') != 1 or head.get('updates') != 1025
            or extra.get('epoch') != 17 or extra.get('updates') != 1025
            or extra.get('continuation_step') != 1
            or any(extra.get(key) != value for key, value in source_coefficients(config).items())
            or scheduler.get('last_epoch') != 16
            or scheduler != original_head['scheduler']
            or not optimizer.get('state') or len(optimizer.get('param_groups', [])) != 1
            or optimizer['param_groups'][0]['lr'] != original_head['optimizer']['param_groups'][0]['lr']
            or not head.get('geometry_state_dict')
            or any(not torch.isfinite(value).all() for value in head['geometry_state_dict'].values())):
        raise RuntimeError('Actual one-update checkpoint did not preserve e16 scheduler or e17/1 update state')
    from train_affinity_sweep import _verify_rng
    _verify_rng(extra.get('rng', {}))
    for value in optimizer['state'].values():
        if (float(value['step']) != 1025
                or any(not torch.isfinite(value[key]).all() for key in ('exp_avg', 'exp_avg_sq'))):
            raise RuntimeError('Resume smoke AdamW state is not the finite update1025 state')
    return dict(passed=True, receipt_path=expected_path,
        receipt_sha256=gate_receipt['sha256'], probe_sha256=gate_receipt['probe_sha256'],
        probe_head_sha256=sha(checkpoint), pipeline_sha256=sha(smoke_root / 'pipeline_status.json'),
        original_identity_exact=True, source_snapshots_exact=True, scheduler_not_restarted=True)


def verify_resume_lineage(config, run, pipeline):
    """允许独立e16续训目录，但原运行和前1024更新必须完整、不可改写。"""
    if 'resume_sources' not in pipeline:
        if run.name == 'affinity_source_resume':
            raise RuntimeError('Resume run is missing an explicit lineage receipt')
        return None
    import hashlib
    receipt_path = run / 'resume_receipt.json'
    receipt = read(receipt_path)
    identity = receipt.get('identity', {})
    if (not identity or any(receipt.get(key) != value for key, value in identity.items())
            or identity.get('config') != json.loads(json.dumps(config))
            or identity.get('runtime') != pipeline['runtime']
            or identity.get('control_retrained') is not False
            or receipt.get('original_run_preserved') is not True
            or receipt.get('finished_training') is not True
            or receipt.get('resumed_new_updates') != 256
            or receipt.get('final_updates') != 1280):
        raise RuntimeError('Resume formal identity/preservation/completion budget receipt differs')
    if receipt.get('original_run') != 'outputs/affinity_source':
        raise RuntimeError('Resume lineage must originate from the exact interrupted P025 run')
    original = ROOT / receipt['original_run']
    if (receipt.get('epoch') != 16 or receipt.get('prefix_updates') != 1024
            or receipt.get('new_updates') != 256):
        raise RuntimeError('Resume must retain 1024 updates and add exactly 256 to e20')
    old_pipeline = read(original / 'pipeline_status.json')
    if (old_pipeline.get('config') != json.loads(json.dumps(config))
            or old_pipeline.get('smoke') is not False
            or old_pipeline.get('control_retrained') is not False
            or old_pipeline.get('runtime') != pipeline['runtime']
            or old_pipeline['sources'] != pipeline['sources']
            or receipt['original_source_sha256'] != pipeline['sources']
            or sha(original / 'pipeline_status.json') != receipt['original_pipeline_sha256']
            or sha(original / 'p025/status.json') != receipt['original_status_sha256']):
        raise RuntimeError('Interrupted run source/runtime/config/status lineage changed')
    verify_sources(original, pipeline['sources'])
    head_path = Path(receipt['original_head_path'])
    if (head_path.is_absolute() or '..' in head_path.parts
            or not head_path.as_posix().startswith('outputs/affinity_source/p025/')
            or sha(ROOT / head_path) != receipt['original_head_sha256']):
        raise RuntimeError('Original e16 training head identity changed')
    head = torch.load(ROOT / head_path, map_location='cpu', weights_only=False)
    weights = source_coefficients(config)
    extra = head.get('source_state', {})
    if (head.get('epoch') != 16 or head.get('size') != 1024
            or json.loads(json.dumps(head.get('config'))) != json.loads(json.dumps(config))
            or not head.get('optimizer') or not head.get('scheduler')
            or extra.get('epoch') != 16 or extra.get('updates') != 1024
            or any(extra.get(key) != value for key, value in weights.items())
            or not extra.get('rng')):
        raise RuntimeError('Original head is not a resumable exact e16 P025 checkpoint')
    resume_sources = pipeline['resume_sources']
    if (set(resume_sources) != {'tools/resume_affinity_source.py'}
            or receipt['resume_source_sha256'] != resume_sources):
        raise RuntimeError('Unexpected or unbound resume implementation')
    for name, expected in resume_sources.items():
        if (sha(ROOT / name) != expected
                or sha(run / 'resume_source' / name) != expected):
            raise RuntimeError('Resume current or archived implementation changed: ' + name)
    # 新恢复文件也绑定SHA后再复用它的完整e16 optimizer/scheduler/RNG检查。
    from tools.resume_affinity_source import check_head
    check_head(config, head)
    prefixes = {'steps.jsonl': 1024, 'epochs.jsonl': 16, 'weighted_logits_grads.jsonl': 2048}
    if set(receipt['retained_prefix_sha256']) != set(prefixes):
        raise RuntimeError('Resume is missing a full retained-training-prefix receipt')
    for name, rows in prefixes.items():
        old_bytes = _prefix_bytes(original / 'p025' / name, rows)
        new_bytes = _prefix_bytes(run / 'p025' / name, rows)
        if (old_bytes != new_bytes
                or hashlib.sha256(old_bytes).hexdigest() != receipt['retained_prefix_sha256'][name]):
            raise RuntimeError('Resume rewrote or lost an original completed-training prefix: ' + name)
        if sha(original / 'p025' / name) != receipt['original_log_sha256'][name]:
            raise RuntimeError('Original interrupted raw log changed: ' + name)
    smoke = verify_resume_smoke(config, pipeline, receipt, head)
    training_path = run / 'p025/resume_training_receipt.json'
    training = read(training_path)
    required_training = dict(new_updates=256, total_updates=1280, original_total_epochs=20,
        restored_optimizer_scheduler_rng=True, scheduler_not_restarted=True)
    if (training != receipt.get('training')
            or any(training.get(key) != value for key, value in required_training.items())):
        raise RuntimeError('Formal continuation did not complete exactly 256 updates with the original scheduler')
    state = read(run / 'p025/status.json')
    if (state.get('resumed_from_epoch') != 16 or state.get('new_updates') != 256
            or state.get('restored_optimizer_scheduler_rng') is not True
            or state.get('strict_reload_equal') is not True):
        raise RuntimeError('Actual completed model status does not bind the e16 continuation')
    return dict(complete=True, original_run=receipt['original_run'], original_epoch=16,
        retained_updates=1024, additional_updates=256,
        receipt_sha256=sha(receipt_path), resume_sources=resume_sources,
        smoke=smoke, training_sha256=sha(training_path),
        completed_prefix_bytes_exact=True, original_checkpoint_and_sources_exact=True)


def preflight(config, run):
    pipeline = read(run / 'pipeline_status.json')
    require_complete_pipeline(pipeline, config)
    weights = source_coefficients(config)
    if weights['beta'] != .25 or weights['manual_coefficient'] != 1.2 or weights['pseudo_coefficient'] != .3:
        raise RuntimeError('Only the formal beta=.25, manual=1.2/SAM2=.3 candidate is supported')
    verify_sources(run, pipeline['sources'])
    runtime = runtime_contract(config)
    if runtime != pipeline['runtime']:
        raise RuntimeError('Current Python/SAM2 runtime differs from actual P025 training')
    resume_lineage = verify_resume_lineage(config, run, pipeline)
    historical = runner.verify_historical(pipeline['sources'], runtime)
    if historical != pipeline['historical']:
        raise RuntimeError('Historical H0/r1 source or pipeline receipt changed')
    smoke_root = ROOT / 'outputs/affinity_source_smoke'
    smoke = runner.verify_gate(smoke_root, config, tuple(pipeline['sources']), runtime)
    for name, cfg in (('p025', config), ('technical_control', runner.technical_config(config))):
        if verify_run(cfg, smoke_root / name, 8) != smoke['audits'].get(name):
            raise RuntimeError('Recorded exact source smoke audit differs: ' + name)
    audit = verify_run(config, run / 'p025', 1280)
    if audit != pipeline['audits'].get('p025') or audit.get('passed') is not True:
        raise RuntimeError('Actual 1280-update P025 audit differs from pipeline receipt')
    configs = {'r1': reference_config(config), 'p025': config}
    folders = {'r1': Path(project_path(config, REFERENCE)), 'p025': run / 'p025'}
    states, identities = {}, {}
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
        coefficients = source_coefficients(runner.technical_config(config) if name == 'r1' else config)
        identities[name] = dict(checkpoint=folder.joinpath('final.pt').relative_to(ROOT).as_posix(),
            checkpoint_sha256=actual, epoch=20, updates=1280, negative_weight=1.,
            hard_negative_weight=1., hard_negative_gamma=2., pseudo_weight=coefficients['beta'],
            source_coefficients=coefficients, status_sha256=sha(folder / 'status.json'))
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data'):
        if states['r1'][key] != states['p025'][key]:
            raise RuntimeError('Shared initial/frozen/restorer/data differs: ' + key)
    return dict(configs=configs, folders=folders, states=states, identities=identities,
        audit=audit, sources=pipeline['sources'], runtime=runtime, historical=historical,
        resume_lineage=resume_lineage,
        smoke_pipeline_sha256=sha(smoke_root / 'pipeline_status.json'),
        pipeline_sha256=sha(run / 'pipeline_status.json'),
        analysis_sources={name: sha(ROOT / name) for name in ANALYSIS_SOURCES})


def _map_constant(value):
    """只改候选标签和报告键；运算、阈值、输入及统计定义保留旧实现。"""
    if isinstance(value, types.CodeType):
        return value.replace(co_consts=tuple(_map_constant(v) for v in value.co_consts))
    if isinstance(value, tuple):
        return tuple(_map_constant(v) for v in value)
    if isinstance(value, str):
        exact = {
            'H0 r1 h0 e20': 'P025 r1 h1 beta0.25 e20',
            'h0 candidate': 'p025 beta0.25 candidate',
            'H0 r1 h0': 'P025 r1 h1 beta0.25',
        }
        return exact.get(value, value.replace('H0', 'P025').replace('h0', 'p025'))
    return value


def _diagnostics():
    """独立globals，避免重绑定旧模块；明确只复用以下五个统计函数。"""
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
    import tempfile
    config = {'sample': True}
    complete = dict(status='complete', stage='done', smoke=False, control_retrained=False,
                    completed=['p025', 'p025_infer'], config=config)
    require_complete_pipeline(complete, config)
    for key, value in (('status', 'running'), ('completed', ['p025']),
                       ('smoke', True), ('control_retrained', True), ('config', {})):
        partial = dict(complete, **{key: value})
        try:
            require_complete_pipeline(partial, config)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incomplete or changed P025 accepted: ' + key)
    try:
        verify_sources(Path('unused'), {'one.py': 'not-a-hash'})
    except RuntimeError:
        pass
    else:
        raise AssertionError('Wrong source count accepted')
    try:
        verify_resume_lineage(config, Path('affinity_source_resume'), complete)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Resume run accepted without explicit original lineage')
    with tempfile.TemporaryDirectory() as scratch:
        text = Path(scratch) / 'steps.jsonl'
        text.write_bytes(b'{"updates":1}\n{"updates":2}\npartial')
        if _prefix_bytes(text, 2) != b'{"updates":1}\n{"updates":2}\n':
            raise AssertionError('Retained completed prefix was reserialized or extended')
        try:
            _prefix_bytes(text, 3)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Incomplete log row accepted into completed prefix')
    formal = dict(runtime={'test': 'CPU'}, sources={'old.py': 'old'},
                  resume_sources={'tools/resume_affinity_source.py': 'new'})
    receipt = dict(identity={'original_run': 'outputs/affinity_source'})
    gate = dict(status='complete', stage='done', smoke=True, completed=['resume_smoke'],
        control_retrained=False, config=config, **formal)
    smoke_receipt = dict(identity=receipt['identity'], original_run_preserved=True,
                         resumed_new_updates=1, finished_training=False)
    probe_receipt = dict(passed=True, epoch=17, step=1, updates=1025, new_updates=1,
        original_total_epochs=20, scheduler_last_epoch=16, scheduler_not_restarted=True,
        restored_optimizer_scheduler_rng=True, strict_head_reload_equal=True,
        frozen_unchanged=True, restorer_unchanged=True)
    require_resume_smoke_records(config, formal, receipt, gate, smoke_receipt, probe_receipt)
    for name in ('passed', 'scheduler_not_restarted', 'strict_head_reload_equal',
                 'frozen_unchanged', 'restorer_unchanged'):
        try:
            require_resume_smoke_records(config, formal, receipt, gate, smoke_receipt,
                                          dict(probe_receipt, **{name: False}))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Failed resume smoke accepted: ' + name)
    for new_identity in ({'original_run': 'another_run'}, {}):
        try:
            require_resume_smoke_records(config, formal, receipt, gate,
                dict(smoke_receipt, identity=new_identity), probe_receipt)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Smoke from a different e16 identity accepted')
    if _map_constant(('h0', 'H0 r1 h0 e20', (119, 'mixed'))) != (
            'p025', 'P025 r1 h1 beta0.25 e20', (119, 'mixed')):
        raise AssertionError('Nested candidate label mapping changed a number/reference')
    if hard.ARMS != ('r1', 'h0') or _DIAGNOSTICS['ARMS'] != ARMS:
        raise AssertionError('Diagnostic reuse modified the old module namespace')
    def strings(code):
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                yield from strings(value)
            elif isinstance(value, tuple):
                yield from (v for v in value if isinstance(v, str))
            elif isinstance(value, str):
                yield value
    for name in ('probe', 'gt', 'test'):
        if (_DIAGNOSTICS[name].__code__.co_code != getattr(hard, name).__code__.co_code
                or _DIAGNOSTICS[name].__defaults__ != getattr(hard, name).__defaults__):
            raise AssertionError('Diagnostic operations/defaults changed while adapting candidate labels')
        if any('h0' in value or 'H0' in value for value in strings(_DIAGNOSTICS[name].__code__)):
            raise AssertionError('Unadapted candidate label: ' + name)
    print('P025_CPU_PASS: incomplete pipeline rejected, 122-source count enforced, labels isolated, original diagnostics unchanged')


def run(args):
    config = load_config(args.config)
    runroot = Path(project_path(config, args.run))
    # 先检查流水线，避免partial被误报成缺GPU或被写出分析目录。
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
    write(output / 'preflight.json', {k: checks[k] for k in ('audit', 'sources', 'runtime',
        'historical', 'resume_lineage', 'smoke_pipeline_sha256', 'pipeline_sha256', 'analysis_sources', 'identities')})
    try:
        write(output / 'status.json', dict(status='running', stage='probe', training=False))
        probe_report = probe(config, checks, output / 'probe', args.device, args.seed)
        write(output / 'status.json', dict(status='running', stage='gt', training=False))
        gt_report = gt(config, checks, output / 'gt', args.device,
                       Path(project_path(config, args.reference_gt)))
        write(output / 'status.json', dict(status='running', stage='test', training=False))
        test_report = test(config, checks, output / 'test', args.seed)
        verify_sources(runroot, checks['sources'])
        if verify_resume_lineage(config, runroot, read(runroot / 'pipeline_status.json')) != checks['resume_lineage']:
            raise RuntimeError('Resume lineage changed during diagnostics')
        if any(sha(ROOT / name) != value for name, value in checks['analysis_sources'].items()):
            raise RuntimeError('Analysis dependency changed during diagnostics')
        report = dict(complete=True, training=False, test_inference_rerun=False, epoch_selection=False,
            postprocessing_tuned=False, official_accuracy=False,
            checkpoint_selection='Fixed e20; no test/GT epoch selection',
            identities=checks['identities'], audit=checks['audit'], sources=checks['sources'],
            source_count=SOURCE_COUNT, current_and_snapshot_exact=True, runtime=checks['runtime'],
            pipeline_sha256=checks['pipeline_sha256'], historical=checks['historical'],
            resume_lineage=checks['resume_lineage'],
            smoke_pipeline_sha256=checks['smoke_pipeline_sha256'], analysis_sources=checks['analysis_sources'],
            diagnostic_reuse='H0 probe/GT/test statistics with isolated candidate-key/display-label adaptation; P025 independent strict preflight',
            probe=probe_report, gt=gt_report,
            test={k: v for k, v in test_report.items() if k not in ('images', 'output_receipts')},
            elapsed_seconds=time.time() - started,
            caveats=['32/4 GT sources participated in training; mechanism checks do not establish generalization.',
                     'Unknown original GT is ignored; test sources have no GT.',
                     'Prediction counts/area/split/merge or equal-recall thresholds do not choose deployment or tune postprocessing.'])
        write(output / 'report.json', report)
        pages = ['<!doctype html><meta charset="utf-8"><title>P025 affinity fixed e20</title>',
            '<style>body{font:16px system-ui;margin:24px}img{max-width:100%}</style>',
            '<h1>mixed balance beta0.5 versus P025 beta0.25, fixed e20</h1>',
            '<p>Only whole SAM2/manual supervision coefficients change: 1/0.5 to 1.2/0.3. r1/h1/gamma2 fixed. Gold=F, blue=P. GT is seen training/original covered only. Test has no GT.</p>',
            '<p><a href="report.json">Complete report</a> | <a href="test/test.json">100 test rows</a> | <a href="gt/report.json">Four seen GT</a></p>',
            '<p>Equal-recall plot labels: control=beta0.5, balance=P025 beta0.25; thresholds only diagnose, never change deployment.</p>',
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
        print('COMPLETE P025 analysis', output, flush=True)
    except Exception as error:
        write(output / 'status.json', dict(status='failed', training=False, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_source_p025.yaml')
    parser.add_argument('--run', default='outputs/affinity_source')
    parser.add_argument('--output', default='outputs/affinity_source/analysis')
    parser.add_argument('--reference-gt', default='outputs/affinity_sweep/analysis/gt/report.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--self-test', action='store_true')
    options = parser.parse_args()
    self_test() if options.self_test else run(options)
