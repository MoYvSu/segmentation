# -*- coding: utf-8 -*-
"""独立固定P路线：原C后一次合并、零新参数及异常后共享函数恢复。"""
from copy import deepcopy
import json

import numpy as np
import pytest

from tools import infer_pearlite as fixed
from tools import infer_boundary_rgb as naf
from models.boundary_restore import state_sha256
from test_infer_boundary_rgb import checkpoint, install_fake_entry, one_cpu_thread, install_support_readout


def scene():
    ids = np.zeros((8, 9), np.uint16)
    ids[1:4, 1:3] = 1
    ids[1:4, 4:7] = 2
    ids[5:7, 1:7] = 3
    return np.zeros((8, 9, 3), np.uint8), dict(instances=ids,
        classes={'1': 0, '2': 0, '3': 1}, instance_scores={'1': .8, '2': .3, '3': .9},
        boundary=np.zeros((8, 9), np.float32), sentinel={'unchanged': [1, 2]})


def test_final_apply_preserves_F_and_original_result_and_inherits_score():
    rgb, before = scene()
    saved = deepcopy(before)
    after, audit = fixed.apply_result(rgb, before)
    assert after['classes'] == {'1': 0, '3': 1}
    assert after['instance_scores'] == {'1': .8, '3': .9}
    assert np.array_equal(before['instances'], saved['instances'])
    assert before['classes'] == saved['classes'] and before['instance_scores'] == saved['instance_scores']
    assert np.array_equal(after['instances'] == 3, before['instances'] == 3)
    assert after['boundary'] is before['boundary'] and after['sentinel'] == before['sentinel']
    assert audit['F_final_ID_map_and_classes_exact'] and audit['removed_P_instance_count'] == 1
    assert after['phases'] == naf.entry.phase_description(after['instances'], after['classes'])


def install_C(tmp_path, monkeypatch):
    args, state, output, model, count, *rest = install_fake_entry(tmp_path, monkeypatch)
    args.config = str(fixed.FIXED_CONFIG)
    observed, _ = install_support_readout(monkeypatch, state)
    # 仅把本次选择的候选改成真实toy RGB模型SHA，不篡改loader或其provenance。
    # 严格格式/架构/T8/数值合同及真实state SHA仍由正式RGB loader逐项校验。
    monkeypatch.setattr(fixed, 'NAF_MODEL_SHA', state_sha256(model))
    monkeypatch.setattr(naf, 'EXPECTED_MODEL_SHA', state_sha256(model))
    original_load = naf.load_naf_checkpoint
    original_predict = naf.entry.predict
    events = []

    def original_C(*values, **options):
        rgb, result = original_predict(*values, **options)
        _, prepared = scene()
        prepared.pop('boundary')
        result.update(prepared)
        state['C_result'] = result
        events.append('C_returned')
        return rgb, result

    monkeypatch.setattr(naf.entry, 'predict', original_C)
    return args, state, output, model, count, original_load, original_C, observed, events


def test_fixed_full_wrapper_calls_original_C_then_merges_once_with_no_new_parameters(tmp_path, monkeypatch):
    args, state, output, model, base_count, original_load, original_predict, observed, events = install_C(tmp_path, monkeypatch)
    actual_apply = fixed.apply_result

    def checked_apply(rgb, result):
        assert events == ['C_returned']
        events.append('P_once')
        return actual_apply(rgb, result)

    monkeypatch.setattr(fixed, 'apply_result', checked_apply)
    receipt = fixed.run(args, device='cpu')
    assert events == ['C_returned', 'P_once'] and observed['replay'] == 1
    assert state['raw_predict_calls'] == 1
    assert naf.load_naf_checkpoint is original_load and naf.entry.predict is original_predict
    assert receipt['parameter_count'] == base_count + sum(p.numel() for p in model.parameters())
    assert receipt['added_parameter_count'] == 0 and receipt['exactly_one_merge_per_source']
    assert receipt['no_relation_model'] and receipt['no_revote'] and receipt['no_WS_rerun']
    assert receipt['seed_support'] == 'original-zero' and receipt['fixed_config'].endswith('area_floor_200.yaml')
    assert receipt['F_final_ID_map_and_classes_exact'] and len(receipt['images']) == 1
    assert receipt['upstream_interface'] == 'tools/infer_boundary_rgb.py'
    assert receipt['naf_checkpoint_format'] == 'boundary_rgb_training_v1'
    assert receipt['naf_input_contract'] == naf.INPUT_CONTRACT
    assert receipt['naf_numerical_contract'] == naf.NUMERICAL_CONTRACT
    record = receipt['images'][0]
    assert record['audit'] == 'audit/pearlite/test_169.json'
    assert record['removed_P_instance_count'] == 1
    assert record['audit_sha256'] == naf.entry.sha(output/record['audit'])
    report = json.loads((output/'audit/report.json').read_text())
    assert report['complete'] and report['pearlite_merge_complete']
    assert 'pearlite_merge' not in report['models'] and report['parameter_count'] == receipt['parameter_count']
    assert json.loads((output/'audit/pearlite_report.json').read_text()) == receipt
    assert json.loads((output/'audit/status.json').read_text())['pearlite_merge_complete']
    assert not list((output/'results').iterdir())


def test_wrong_NAF_is_rejected_before_source_or_output(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.config = str(fixed.FIXED_CONFIG)
    original_load, original_predict = naf.load_naf_checkpoint, naf.entry.predict
    with pytest.raises(ValueError, match='selected candidate'):
        fixed.run(args, device='cpu')
    assert state['raw_predict_calls'] == 0 and not output.exists()
    assert naf.load_naf_checkpoint is original_load and naf.entry.predict is original_predict


@pytest.mark.parametrize('name,value', [('relation_checkpoint', 'r.pt'), ('relation_threshold', .1)])
def test_relation_settings_not_accepted(tmp_path, monkeypatch, name, value):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    setattr(args, name, value)
    with pytest.raises(ValueError, match='关系'):
        fixed.run(args, device='cpu')
    assert not output.exists() and state['raw_predict_calls'] == 0


def test_programmatic_config_cannot_change_fixed_recipe(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='area_floor_200'):
        fixed.run(args, device='cpu')
    assert not output.exists() and state['raw_predict_calls'] == 0


def test_merge_error_marks_failed_and_restores_every_shared_hook(tmp_path, monkeypatch):
    args, state, output, _, _, original_load, original_predict, _, _ = install_C(tmp_path, monkeypatch)

    def fail(*unused):
        raise RuntimeError('synthetic merge failure')

    monkeypatch.setattr(fixed, 'apply_result', fail)
    with pytest.raises(RuntimeError, match='synthetic merge failure'):
        fixed.run(args, device='cpu')
    assert naf.load_naf_checkpoint is original_load and naf.entry.predict is original_predict
    assert not json.loads((output/'audit/status.json').read_text())['complete']
    assert json.loads((output/'audit/pearlite_failed.json').read_text())['complete'] is False


def test_CLI_has_no_recipe_or_P_feature_switch(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(fixed, 'run', lambda args: calls.append(args) or dict(images=[], parameter_count=1))
    values = ['--input-dir', 'i', '--output-dir', 'o', '--naf-checkpoint', 'n']
    fixed.main(values)
    assert len(calls) == 1 and calls[0].dataset == 'semifinal_150'
    assert not hasattr(calls[0], 'merge_adjacent_pearlite') and not hasattr(calls[0], 'config')
    for flag in ('--merge-adjacent-pearlite', '--relation-checkpoint', '--seed-support', '--config'):
        with pytest.raises(SystemExit) as error:
            fixed.main(values+[flag, 'x'])
        assert error.value.code == 2
    assert len(calls) == 1
    capsys.readouterr()


def test_actual_CPU_RGB_loader_accepts_its_real_format_and_rejects_region_checkpoint(tmp_path):
    # CPU实际构造/保存/严格加载，而非用mock把Region provenance改成20a59。
    from tools import infer_boundary_naf as region
    path, original = checkpoint(tmp_path)
    loaded, proof = fixed.naf_entry.load_naf_checkpoint(path, 'cpu', expected_model_sha=state_sha256(original))
    assert state_sha256(loaded) == state_sha256(original)
    assert proof['checkpoint_format'] == fixed.NAF_CHECKPOINT_FORMAT == 'boundary_rgb_training_v1'
    assert proof['input_contract']['input_normalization'] == dict(mode='skeleton_dilate', kernel_size=8)
    with pytest.raises(ValueError, match='boundary_region_training_v1'):
        region.load_naf_checkpoint(path, 'cpu')


def test_fixed_route_fails_fast_if_wrong_checkpoint_interface_is_reintroduced(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.config = str(fixed.FIXED_CONFIG)
    monkeypatch.setattr(naf, 'CHECKPOINT_FORMAT', 'boundary_region_training_v1')
    with pytest.raises(RuntimeError, match='boundary_rgb_training_v1'):
        fixed.run(args, device='cpu')
    assert not output.exists() and state['raw_predict_calls'] == 0
