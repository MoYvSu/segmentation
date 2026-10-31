# -*- coding: utf-8 -*-
"""放行门槛绑定两组短测、服务器源码以及完整控制，不放宽既有训练检查。"""
import json
from pathlib import Path

import pytest

from tools import run_affinity_sweep as runner
from train_backend_adaptation import sha


def fixture_gate(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    root.mkdir()
    path = root / 'entry.py'
    path.write_text('# -*- coding: utf-8 -*-\npass\n', encoding='utf8')
    gate = root / 'outputs/smoke'
    (gate / 'source').mkdir(parents=True)
    (gate / 'source/entry.py').write_bytes(path.read_bytes())
    configs = {'n075': {'r': .75}, 'n125': {'r': 1.25}}
    status = dict(status='complete', configs=configs, control_retrained=False,
                  sources={'entry.py': sha(path)})
    (gate / 'pipeline_status.json').write_text(json.dumps(status), encoding='utf8')
    (gate / 'parity.json').write_text(json.dumps({'passed': True}), encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', root)
    calls = []
    monkeypatch.setattr(runner, 'verify_run', lambda config, folder, count:
                        calls.append((Path(folder).name, count)))
    return root, gate, configs, status, calls


def test_gate_requires_both_candidates_and_detects_source_tampering(tmp_path, monkeypatch):
    root, gate, configs, _, calls = fixture_gate(tmp_path, monkeypatch)
    assert runner.verify_gate(gate, configs, ('entry.py',))['status'] == 'complete'
    assert calls == [('n075', 8), ('n125', 8)]
    (root / 'entry.py').write_text('raise RuntimeError()\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='Source changed'):
        runner.verify_gate(gate, configs, ('entry.py',))


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('control_retrained', True),
    ('configs', {'n075': {'r': .7}, 'n125': {'r': 1.25}})])
def test_gate_rejects_changed_contract(tmp_path, monkeypatch, field, value):
    _, gate, configs, status, _ = fixture_gate(tmp_path, monkeypatch)
    status[field] = value
    (gate / 'pipeline_status.json').write_text(json.dumps(status), encoding='utf8')
    with pytest.raises(RuntimeError):
        runner.verify_gate(gate, configs, ('entry.py',))


def test_inventory_includes_static_dependencies_and_yaml_ancestors(tmp_path, monkeypatch):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg/__init__.py').write_text('', encoding='utf8')
    (tmp_path / 'pkg/a.py').write_text('from . import b\n', encoding='utf8')
    (tmp_path / 'pkg/b.py').write_text('from pathlib import Path\n', encoding='utf8')
    (tmp_path / 'entry.py').write_text('from pkg.a import value\n', encoding='utf8')
    (tmp_path / 'base.yaml').write_text('value: 1\n', encoding='utf8')
    (tmp_path / 'candidate.yaml').write_text('_base: base.yaml\n', encoding='utf8')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'SEEDS', ('entry.py', 'candidate.yaml'))
    assert set(runner.source_names()) == {
        'entry.py', 'pkg/__init__.py', 'pkg/a.py', 'pkg/b.py', 'candidate.yaml', 'base.yaml'}


def test_gate_rejects_changed_third_party_runtime(tmp_path, monkeypatch):
    _, gate, configs, status, _ = fixture_gate(tmp_path, monkeypatch)
    status['runtime'] = {'sam2_config_sha256': 'a'*64}
    (gate / 'pipeline_status.json').write_text(json.dumps(status), encoding='utf8')
    assert runner.verify_gate(gate, configs, ('entry.py',), status['runtime'])
    with pytest.raises(RuntimeError, match='SAM2 source'):
        runner.verify_gate(gate, configs, ('entry.py',), {'sam2_config_sha256': 'b'*64})
