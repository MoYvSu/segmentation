# -*- coding: utf-8 -*-
"""P025放行绑定本轮两种技术配方，并保护已完成H0的源码快照。"""
import json

import pytest

from tools import run_affinity_source as runner
from train_backend_adaptation import sha


def gate_fixture(tmp_path, monkeypatch):
    (tmp_path/'entry.py').write_text('pass\n', encoding='utf8')
    gate = tmp_path/'smoke'
    (gate/'source').mkdir(parents=True)
    (gate/'source/entry.py').write_bytes((tmp_path/'entry.py').read_bytes())
    config = {'backend_adaptation': {'pseudo_weight': .25}}
    runtime = {'torch': 'test'}
    state = dict(status='complete', smoke=True, config=config, runtime=runtime,
                 control_retrained=False, sources={'entry.py': sha(tmp_path/'entry.py')})
    (gate/'pipeline_status.json').write_text(json.dumps(state), encoding='utf8')
    (gate/'parity.json').write_text(json.dumps({'passed': True}), encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    calls = []
    monkeypatch.setattr(runner, 'verify_run', lambda c, path, n:
                        calls.append((path.name, c['backend_adaptation']['pseudo_weight'], n)))
    return gate, config, runtime, state, calls


def test_gate_requires_both_candidate_and_technical_zero_change(tmp_path, monkeypatch):
    gate, config, runtime, _, calls = gate_fixture(tmp_path, monkeypatch)
    runner.verify_gate(gate, config, ('entry.py',), runtime)
    assert calls == [('p025', .25, 8), ('technical_control', .5, 8)]
    assert config['backend_adaptation']['pseudo_weight'] == .25


def test_gate_does_not_accept_changed_source_snapshot(tmp_path, monkeypatch):
    gate, config, runtime, _, _ = gate_fixture(tmp_path, monkeypatch)
    (gate/'source/entry.py').write_text('changed\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='Source changed'):
        runner.verify_gate(gate, config, ('entry.py',), runtime)


def test_gate_rejects_new_runtime(tmp_path, monkeypatch):
    gate, config, _, _, _ = gate_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError):
        runner.verify_gate(gate, config, ('entry.py',), {'torch': 'changed'})


def test_preflight_preserves_both_existing_experiment_sources(tmp_path, monkeypatch):
    runtime = {'torch': 'test'}
    (tmp_path/'base.py').write_text('pass\n', encoding='utf8')
    (tmp_path/'hard.py').write_text('pass\n', encoding='utf8')
    for label, names in [('affinity_sweep', ['base.py']), ('affinity_hard', ['base.py', 'hard.py'])]:
        root = tmp_path/'outputs'/label
        (root/'source').mkdir(parents=True)
        for name in names:
            (root/'source'/name).write_bytes((tmp_path/name).read_bytes())
        state = dict(status='complete', control_retrained=False, runtime=runtime,
                     sources={name: sha(tmp_path/name) for name in names})
        (root/'pipeline_status.json').write_text(json.dumps(state), encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    assert runner.verify_historical(('base.py', 'hard.py'), runtime)['historical_sources'] == 2
    (tmp_path/'hard.py').write_text('changed\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='Historical H0 source changed'):
        runner.verify_historical(('base.py', 'hard.py'), runtime)
