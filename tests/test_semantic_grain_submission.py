# -*- coding: utf-8 -*-
"""末端语义提交的CPU合同：固定PNG、严格投票、身份拒绝及只计划默认。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from tools import package_semantic_grain as package
from utils.config import load_config


def fixture_zip(tmp_path, *, dtype=np.uint16, labels=None, extra=None):
    ids = np.asarray([[0, 1, 1], [2, 2, 3]], dtype=dtype)
    ok, png = cv2.imencode('.png', ids)
    assert ok
    path = tmp_path/'source.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('synthetic_inst.png', png.tobytes())
        archive.writestr('synthetic_class.json', json.dumps(labels or {'1': 0, '2': 1, '3': 0}))
        if extra:
            archive.writestr(*extra)
    return path, ids


def test_original_zero_change_vote_and_strict_half_threshold():
    ids = np.array([[0, 1, 1], [2, 2, 3]], np.uint16)
    probability = np.array([[0, .4, .6], [.8, .9, .1]], np.float32)
    expected = {'1': 0, '2': 1, '3': 0}
    digest = hashlib.sha256(probability.tobytes()).hexdigest()
    scores = package.require_baseline(ids, probability, expected, digest)
    assert scores['1'] == .5
    assert scores['2'] > .5
    with pytest.raises(RuntimeError):
        package.require_baseline(ids, probability, {'1': 1, '2': 1, '3': 0}, digest)
    with pytest.raises(RuntimeError):
        package.require_baseline(ids, probability, expected, '0'*64)


def test_package_replaces_classes_and_preserves_every_png_byte(tmp_path):
    source, ids = fixture_zip(tmp_path)
    new = {'synthetic': {'1': 1, '2': 0, '3': 0}}
    report = package.write_package(source, tmp_path/'grain64.zip', new)
    assert report['files'] == 2 and report['mask_bytes_identical']
    assert package.archive_contract(tmp_path/'grain64.zip', 1) == ['synthetic']
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(tmp_path/'grain64.zip') as after:
        assert before.read('synthetic_inst.png') == after.read('synthetic_inst.png')
        assert json.loads(after.read('synthetic_class.json')) == new['synthetic']
        decoded = cv2.imdecode(np.frombuffer(after.read('synthetic_inst.png'), np.uint8), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(decoded, ids) and decoded.dtype == np.uint16


@pytest.mark.parametrize('bad', [{'1': 0, '2': 1}, {'1': 0, '2': 1, '3': True}, {'1': 0, '2': 1, '3': 2}])
def test_package_rejects_missing_ids_bool_or_nonbinary_classes(tmp_path, bad):
    source, _ = fixture_zip(tmp_path)
    with pytest.raises(ValueError):
        package.write_package(source, tmp_path/'bad.zip', {'synthetic': bad})


def test_archive_rejects_uint8_and_extra_payloads(tmp_path):
    source, _ = fixture_zip(tmp_path, dtype=np.uint8)
    with pytest.raises(ValueError):
        package.archive_contract(source, 1)
    source, _ = fixture_zip(tmp_path, extra=('report.json', '{}'))
    with pytest.raises(ValueError):
        package.archive_contract(source, 1)
    source, _ = fixture_zip(tmp_path)
    with pytest.raises(RuntimeError):
        package.archive_contract(source, 100)


def test_sha_rejects_changed_source_before_inference(tmp_path):
    source = tmp_path/'weight.bin'; source.write_bytes(b'old')
    digest = package.sha(source)
    assert package.require_sha(source, digest) == digest
    source.write_bytes(b'new')
    with pytest.raises(RuntimeError):
        package.require_sha(source, digest)


def test_receipt_sha_is_from_unique_training_sealed_path_and_rejects_changed_receipt(tmp_path):
    receipt = tmp_path/'outputs'/'inputs.json'; receipt.parent.mkdir(); receipt.write_text('{}')
    digest = package.sha(receipt)
    archived = {'/prior/project/outputs/inputs.json': digest}
    expected = package.sealed_input_sha(archived, receipt, tmp_path)
    assert expected == digest
    assert package.require_sha(receipt, expected) == digest
    receipt.write_text('{"untrusted_sha": "replacement"}')
    with pytest.raises(RuntimeError):
        package.require_sha(receipt, expected)
    with pytest.raises(RuntimeError):
        package.sealed_input_sha({}, receipt, tmp_path)
    archived['/other/project/outputs/inputs.json'] = digest
    with pytest.raises(RuntimeError):
        package.sealed_input_sha(archived, receipt, tmp_path)
    with pytest.raises(RuntimeError):
        package.sealed_input_sha(archived, tmp_path.parent/'outside.json', tmp_path)


def test_smoke_is_registered_eight_and_never_first_eight():
    stems = [f'synthetic_{i:03d}' for i in range(100)]
    tests = stems[35:43]
    report = {'snapshots': {'64': {'rows': {name: {'kind': 'test'} for name in tests}}}}
    selected, fixed = package.select_sources(stems, report, 8)
    assert selected == fixed == tests and selected != stems[:8]
    assert package.select_sources(stems, report, None)[0] == stems
    with pytest.raises(ValueError):
        package.select_sources(stems, report, 5)


def test_training_gate_requires_identity_and_architecture_but_not_checkpoint_complete_flag():
    cfg = dict(package.PINS)
    identity = dict(mode='formal', total_steps=64, base_checkpoint_sha256=cfg['checkpoint_sha256'],
                    base_state_sha256=cfg['base_state_sha256'], initial_head_sha256=cfg['initial_head_sha256'])
    report = dict(format='semantic_grain_fit_v1', complete=True, steps=64, optimizer_updates=64,
                  effective_head_views=128, final_sources=40, final_crop_views=64, training=False,
                  final_head_sha256=cfg['final_head_sha256'], strict_reload_final40_exact=True,
                  strict_reload_crop_scope_exact=True, upstream_frozen_exact=True, fixed_png_bytes_exact=True,
                  identity=identity, snapshots={'64': {'rows': {}}})
    architecture = {'synthetic': True}
    state = dict(format=package.HEAD_FORMAT, step=64, total_steps=64, complete=False,
                 terminal_only=True, do_not_replace_geometry_upstream=True, identity=deepcopy(identity),
                 architecture=architecture, semantic_state_dict={}, snapshots=deepcopy(report['snapshots']))
    package.validate_training(report, state, architecture, cfg)
    for key, value in [('step', 63), ('architecture', {'other': True}), ('identity', {}), ('terminal_only', False)]:
        bad = deepcopy(state); bad[key] = value
        with pytest.raises(RuntimeError):
            package.validate_training(report, bad, architecture, cfg)
    bad = deepcopy(report); bad['strict_reload_final40_exact'] = False
    with pytest.raises(RuntimeError):
        package.validate_training(bad, state, architecture, cfg)


def test_default_plan_never_touches_weights_or_cuda(monkeypatch, capsys):
    def forbid(*args, **kwargs):
        raise AssertionError('Plan must not load actual model/data or inspect CUDA')
    monkeypatch.setattr(package.torch.cuda, 'is_available', forbid)
    monkeypatch.setattr(package, 'read', forbid)
    monkeypatch.setattr(package, 'sha', forbid)
    package.run(SimpleNamespace(config='config/inference/semantic_grain_submission.yaml', run=False, limit=None, output=None))
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'plan_only' and report['training'] is False and report['images'] == 100
    assert report['official_submission'] is False and report['best_deployment_changed'] is False
    cfg = load_config('config/inference/semantic_grain_submission.yaml')['semantic_grain_submission']
    cfg['head_checkpoint_sha256'] = '0'*64
    with pytest.raises(ValueError):
        package.validate_recipe(cfg)
