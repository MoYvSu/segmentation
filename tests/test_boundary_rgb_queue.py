# -*- coding: utf-8 -*-
"""队列完成衔接回归：训练完成回执含checkpoint时仍能进入控制组。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_boundary_rgb as queue


def prepare(tmp_path):
    configs = []
    for name, mode in [('rgb', 'rgb_boundary'), ('boundary', 'boundary')]:
        path = tmp_path / (name + '.json')
        path.write_text(json.dumps({'boundary_naf': {'input_mode': mode}}))
        configs.append(str(path))
    return SimpleNamespace(output=str(tmp_path / 'run'), configs=configs, device='cpu', smoke_updates=128)


def fake_phases(monkeypatch, *, fail_control=False):
    calls = []
    def invoke(command, **kwargs):
        destination = Path(command[command.index('--output') + 1])
        phase = command[command.index('--mode') + 1]
        destination.mkdir(parents=True, exist_ok=True)
        mode = 'rgb_boundary' if destination.name.endswith('rgb') else 'boundary'
        name = 'rgb' if mode == 'rgb_boundary' else 'boundary'
        calls.append((phase, name))
        if phase == 'preflight':
            document = dict(complete=True, no_parameter_updates=True,
                plan=dict(plan_sha256='fixed_plan'), initialization_identity=dict(
                    dataset=dict(real_input_sources=['same_training_source_binding']),
                    initial_discriminator_sha256='same_initial_D', recipe=dict(
                        input_mode=mode, output_dir=name, inference=dict(input_mode=mode))))
            filename = 'preflight.json'
        elif phase == 'smoke':
            document, filename = dict(passed=True), 'smoke.json'
        else:
            if fail_control and name == 'boundary':
                raise RuntimeError('simulated control interruption')
            document = dict(complete=True, global_step=1000, checkpoint='latest.pt')
            filename = 'complete.json'
        (destination / filename).write_text(json.dumps(document))
    monkeypatch.setattr(queue.subprocess, 'run', invoke)
    return calls


def test_completed_rgb_receipt_with_checkpoint_does_not_block_control(tmp_path, monkeypatch):
    args = prepare(tmp_path)
    calls = fake_phases(monkeypatch)
    result = queue.run(args)
    assert calls == [('preflight', 'rgb'), ('preflight', 'boundary'),
                     ('smoke', 'rgb'), ('smoke', 'boundary'), ('train', 'rgb'), ('train', 'boundary')]
    assert result['complete']
    assert [g['group'] for g in result['groups']] == ['rgb', 'boundary']
    assert all(Path(g['checkpoint']).is_absolute() for g in result['groups'])
    assert not (Path(args.output) / 'queue_failed.json').exists()


def test_real_control_interruption_keeps_completed_rgb_receipt(tmp_path, monkeypatch):
    args = prepare(tmp_path)
    fake_phases(monkeypatch, fail_control=True)
    with pytest.raises(RuntimeError, match='simulated'):
        queue.run(args)
    failed = json.loads((Path(args.output) / 'queue_failed.json').read_text())
    assert failed['completed'][0]['group'] == 'rgb'
    assert failed['completed'][0]['global_step'] == 1000
    with pytest.raises(ValueError, match='已有启动'):
        queue.run(args)
