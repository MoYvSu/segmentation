# -*- coding: utf-8 -*-
"""训练放行必须绑定已审计的两端权重与完整同输入诊断。"""
from copy import deepcopy
import json

import pytest

from tools.run_affinity_pair import ROOT, diagnostic_gate
from train_backend_adaptation import sha
from utils.config import load_config


def report():
    return dict(complete=True, training=False, source_count=32, input_views=64, model_views=128,
                seed=20261001, d5a_frozen_exact=True, scope='synthetic gate fixture',
                source_sha256=sha(ROOT/'tools/probe_affinity_equal_recall.py'),
                receipts=[{'restored_sha256': 'f'*64} for _ in range(64)],
                checkpoints={
                    'control': dict(sha256='2bbae197f4716ced694309b9c5d49b5e24a4154f74755d4652d2beb0e1318d5b',
                                    frozen_exact=True, shared_state_sha256='a'*64),
                    'balance': dict(sha256='43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434',
                                    frozen_exact=True, shared_state_sha256='a'*64)})


def write(tmp_path, value):
    path = tmp_path/'probe.json'
    path.write_text(json.dumps(value), encoding='utf8')
    return path


def test_gate_binds_registered_pair_and_exact_diagnostic_source(tmp_path):
    config = load_config('config/train/affinity_pair.yaml')
    value = report()
    assert diagnostic_gate(config, write(tmp_path, value))['passed']
    value['checkpoints']['balance']['sha256'] = 'b'*64
    with pytest.raises(RuntimeError, match='incomplete or differs'):
        diagnostic_gate(config, write(tmp_path, value))
    value = report()
    value['source_sha256'] = 'b'*64
    with pytest.raises(RuntimeError, match='Diagnostic source differs'):
        diagnostic_gate(config, write(tmp_path, value))


@pytest.mark.parametrize('field,value', [('input_views', 63), ('model_views', 127),
                                       ('training', True), ('d5a_frozen_exact', False)])
def test_gate_rejects_partial_or_trainable_probe(tmp_path, field, value):
    current = report()
    current[field] = value
    with pytest.raises(RuntimeError):
        diagnostic_gate(load_config('config/train/affinity_pair.yaml'), write(tmp_path, current))


def test_gate_rejects_changed_shared_encoder_and_missing_input_receipts(tmp_path):
    config = load_config('config/train/affinity_pair.yaml')
    value = report()
    changed = deepcopy(value)
    changed['checkpoints']['balance']['shared_state_sha256'] = 'c'*64
    with pytest.raises(RuntimeError, match='shared encoder'):
        diagnostic_gate(config, write(tmp_path, changed))
    value['receipts'].pop()
    with pytest.raises(RuntimeError, match='paired RGB'):
        diagnostic_gate(config, write(tmp_path, value))
