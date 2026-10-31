# -*- coding: utf-8 -*-
"""P100正式放行绑定8步候选、只读控制缓存及旧P025完整谱系。"""
from copy import deepcopy
import json

import pytest

from tools import run_affinity_p100 as runner
from train_backend_adaptation import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf8')


def gate_fixture(tmp_path, monkeypatch):
    (tmp_path / 'old.py').write_text('old = 1\n', encoding='utf8')
    (tmp_path / 'new.py').write_text('new = 1\n', encoding='utf8')
    root = tmp_path / 'smoke'
    (root / 'source').mkdir(parents=True)
    sources = ('old.py', 'new.py')
    for name in sources:
        (root / 'source' / name).write_bytes((tmp_path / name).read_bytes())
    config = {'affinity_source': {'reference_sha256': 'control-sha'},
              'affinity_connectivity': {'parity_images': ['test_101', 'test_116']}}
    runtime = {'torch': 'test'}
    receipt = {'passed': True, 'technical_updates_reused': 8, 'cache_sha': 'same'}
    historical = {'passed': True, 'old122_unchanged': True, 'resumed_final_sha': 'p025-final'}
    audit = {'passed': True, 'updates': 8, 'source_coefficients': [.75, .75]}
    status = dict(status='complete', stage='done', smoke=True, completed=runner.SMOKE_STAGES,
        control_retrained=False, config=config, runtime=runtime,
        sources={name: sha(tmp_path / name) for name in sources},
        reused_control=receipt, historical=historical, audits={'p100': audit})
    write(root / 'pipeline_status.json', status)
    write(root / 'control_reuse.json', receipt)
    write(root / 'parity.json', dict(passed=True, control_retrained=False,
        checkpoint_sha256='control-sha', rows=[dict(source=name, final_output_equal=True)
            for name in config['affinity_connectivity']['parity_images']]))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'control_reuse_receipt', lambda c, r: receipt)
    monkeypatch.setattr(runner, 'verify_historical', lambda s, r: historical)
    calls = []
    monkeypatch.setattr(runner, 'verify_run', lambda c, p, u: calls.append((p.name, u)) or audit)
    return root, config, sources, runtime, status, calls


def test_gate_only_audits_new_eight_step_candidate_control_is_reused(tmp_path, monkeypatch):
    root, config, sources, runtime, status, calls = gate_fixture(tmp_path, monkeypatch)
    assert runner.verify_gate(root, config, sources, runtime) == status
    assert calls == [('p100', 8)]
    assert runner.SMOKE_STAGES == ['parity', 'reuse_control', 'p100']


@pytest.mark.parametrize('key,value', [
    ('status', 'running'), ('stage', 'p100'), ('smoke', False), ('control_retrained', True),
    ('completed', ['p100']), ('runtime', {'torch': 'changed'}), ('config', {'changed': True})])
def test_gate_rejects_partial_or_changed_run(tmp_path, monkeypatch, key, value):
    root, config, sources, runtime, status, _ = gate_fixture(tmp_path, monkeypatch)
    write(root / 'pipeline_status.json', dict(status, **{key: value}))
    with pytest.raises(RuntimeError, match='Exact completed P100'):
        runner.verify_gate(root, config, sources, runtime)


@pytest.mark.parametrize('name', ['old.py', 'new.py'])
def test_gate_protects_old_and_new_source_snapshots(tmp_path, monkeypatch, name):
    root, config, sources, runtime, _, _ = gate_fixture(tmp_path, monkeypatch)
    (root / 'source' / name).write_text('changed\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='archived snapshot changed'):
        runner.verify_gate(root, config, sources, runtime)


def test_gate_rejects_changed_control_output_cache_or_technical_receipt(tmp_path, monkeypatch):
    root, config, sources, runtime, _, _ = gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'control_reuse_receipt', lambda c, r: {'cache_sha': 'changed'})
    with pytest.raises(RuntimeError, match='receipt changed after smoke'):
        runner.verify_gate(root, config, sources, runtime)


def test_gate_rejects_changed_recovered_p025_lineage(tmp_path, monkeypatch):
    root, config, sources, runtime, _, _ = gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'verify_historical', lambda s, r: {'resumed_final_sha': 'changed'})
    with pytest.raises(RuntimeError, match='lineage changed after smoke'):
        runner.verify_gate(root, config, sources, runtime)


def test_gate_requires_actual_recorded_candidate_audit(tmp_path, monkeypatch):
    root, config, sources, runtime, _, _ = gate_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'verify_run', lambda c, p, n: {'passed': True, 'updates': 7})
    with pytest.raises(RuntimeError, match='eight-update audit differs'):
        runner.verify_gate(root, config, sources, runtime)


def test_gate_requires_current_two_image_final_output_parity(tmp_path, monkeypatch):
    root, config, sources, runtime, _, _ = gate_fixture(tmp_path, monkeypatch)
    value = json.loads((root / 'parity.json').read_text())
    value['rows'][1]['final_output_equal'] = False
    write(root / 'parity.json', value)
    with pytest.raises(RuntimeError, match='read-only final-control parity'):
        runner.verify_gate(root, config, sources, runtime)


def control_fixture(tmp_path, monkeypatch):
    root = tmp_path / 'outputs/affinity_source_smoke'
    folder = tmp_path / 'reference'
    cache = folder / 'deployment/patch1024'
    cache.mkdir(parents=True)
    for name in ('status.json', 'final.pt', 'last_head.pt'):
        path = root / 'technical_control' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    for name in ('pipeline_status.json', 'parity.json'):
        (root / name).write_bytes(name.encode())
    (folder / 'final.pt').write_bytes(b'control')
    audit = dict(passed=True, updates=8)
    old_gate = {'sources': {'old.py': 'old-sha'}, 'audits': {'technical_control': audit}}
    rows = []
    for index in range(100):
        stem = f'test_{index:03d}'
        for suffix in ('_inst.png', '_class.json'):
            (cache / (stem + suffix)).write_bytes(stem.encode())
        rows.extend([dict(source=stem, view='global'), dict(source=stem, view='patch1024')])
    manifest = dict(complete=True, epoch=20, sizes=[0, 1024],
        checkpoint_sha256=sha(folder / 'final.pt'), images=rows)
    write(folder / 'deployment/manifest.json', manifest)
    config = {'affinity_source': {'reference': 'reference', 'reference_sha256': sha(folder / 'final.pt')}}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load_config', lambda p: {'legacy': True})
    monkeypatch.setattr(runner, 'technical_config', lambda c: {'beta': .5})
    monkeypatch.setattr(runner.original_runner, 'source_names', lambda: ('old.py',))
    monkeypatch.setattr(runner.original_runner, 'verify_gate', lambda *args: old_gate)
    monkeypatch.setattr(runner.source_analysis, 'verify_run', lambda *args: audit)
    monkeypatch.setattr(runner, 'project_path', lambda *args: str(folder))
    return folder, config, manifest


def test_control_receipt_reuses_global_plus_native_manifest_without_retraining(tmp_path, monkeypatch):
    _, config, _ = control_fixture(tmp_path, monkeypatch)
    receipt = runner.control_reuse_receipt(config, {'torch': 'test'})
    assert receipt['passed'] and not receipt['control_retrained']
    assert receipt['technical_updates_reused'] == 8
    assert len(receipt['final_control_output_sha256']) == 200


def test_control_receipt_rejects_incomplete_native_cohort(tmp_path, monkeypatch):
    folder, config, manifest = control_fixture(tmp_path, monkeypatch)
    manifest['images'].pop()
    write(folder / 'deployment/manifest.json', manifest)
    with pytest.raises(RuntimeError, match='e20/100 native1024 output'):
        runner.control_reuse_receipt(config, {})


def test_control_receipt_binds_cached_final_output_bytes(tmp_path, monkeypatch):
    folder, config, _ = control_fixture(tmp_path, monkeypatch)
    before = runner.control_reuse_receipt(config, {})
    (folder / 'deployment/patch1024/test_000_inst.png').write_bytes(b'changed')
    after = runner.control_reuse_receipt(config, {})
    assert before['final_control_output_sha256'] != after['final_control_output_sha256']
