# -*- coding: utf-8 -*-
"""H0自己的短测、历史源码和配置绑定，不增加训练控制。"""
import json

import pytest

from tools import run_affinity_hard as runner
from train_backend_adaptation import sha


def make_gate(tmp_path, monkeypatch):
    root = tmp_path/'workspace'
    root.mkdir()
    (root/'entry.py').write_text('pass\n', encoding='utf8')
    gate = root/'outputs/smoke'
    (gate/'source').mkdir(parents=True)
    (gate/'source/entry.py').write_bytes((root/'entry.py').read_bytes())
    config, runtime = {'h': 0}, {'torch': 'synthetic'}
    status = dict(status='complete', smoke=True, config=config, runtime=runtime,
                  control_retrained=False, sources={'entry.py': sha(root/'entry.py')})
    (gate/'pipeline_status.json').write_text(json.dumps(status), encoding='utf8')
    (gate/'parity.json').write_text(json.dumps({'passed': True}), encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', root)
    calls = []
    monkeypatch.setattr(runner, 'verify_run', lambda cfg, folder, count: calls.append((folder.name, count)))
    return root, gate, config, runtime, status, calls


def test_gate_binds_h0_smoke_and_source_bytes(tmp_path, monkeypatch):
    root, gate, config, runtime, _, calls = make_gate(tmp_path, monkeypatch)
    assert runner.verify_gate(gate, config, ('entry.py',), runtime)['status'] == 'complete'
    assert calls == [('h0', 8)]
    (gate/'source/entry.py').write_text('changed\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='Source changed'):
        runner.verify_gate(gate, config, ('entry.py',), runtime)


@pytest.mark.parametrize('key,value', [('status', 'failed'), ('smoke', False),
    ('control_retrained', True), ('config', {'h': 2}), ('runtime', {'torch': 'changed'})])
def test_gate_rejects_unapproved_execution(tmp_path, monkeypatch, key, value):
    _, gate, config, runtime, status, _ = make_gate(tmp_path, monkeypatch)
    status[key] = value
    (gate/'pipeline_status.json').write_text(json.dumps(status), encoding='utf8')
    with pytest.raises(RuntimeError):
        runner.verify_gate(gate, config, ('entry.py',), runtime)


def test_new_import_closure_keeps_historical_and_yaml_ancestors(tmp_path, monkeypatch):
    (tmp_path/'entry.py').write_text('import additional\n', encoding='utf8')
    (tmp_path/'additional.py').write_text('pass\n', encoding='utf8')
    (tmp_path/'base.yaml').write_text('h: 1\n', encoding='utf8')
    (tmp_path/'candidate.yaml').write_text('_base: base.yaml\n', encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'NEW_SOURCES', ('entry.py', 'candidate.yaml'))
    monkeypatch.setattr(runner.prior_runner, 'source_names', lambda: ('historical.py',))
    assert set(runner.source_names()) == {
        'historical.py', 'entry.py', 'additional.py', 'base.yaml', 'candidate.yaml'}


def test_old_trained_snapshot_is_not_silently_replaced(tmp_path, monkeypatch):
    (tmp_path/'entry.py').write_text('pass\n', encoding='utf8')
    prior = tmp_path/'outputs/affinity_sweep'
    (prior/'source').mkdir(parents=True)
    (prior/'source/entry.py').write_bytes((tmp_path/'entry.py').read_bytes())
    runtime = {'torch': 'synthetic'}
    status = dict(status='complete', control_retrained=False, runtime=runtime,
                  sources={'entry.py': sha(tmp_path/'entry.py')})
    (prior/'pipeline_status.json').write_text(json.dumps(status), encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    assert runner.verify_historical(('entry.py',), runtime)['passed']
    (tmp_path/'entry.py').write_text('changed\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='Historical trained source changed'):
        runner.verify_historical(('entry.py',), runtime)
