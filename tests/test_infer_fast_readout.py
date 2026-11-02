# -*- coding: utf-8 -*-
"""独立CLI的单函数替换、真实WS、审计和异常恢复；不加载模型。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import infer_fast_readout as infer


def args(output, mode='C'):
    return SimpleNamespace(output_dir=str(output), geometry_mode=mode)


def test_reuses_existing_C_default_parser_without_changing_old_entry():
    assert infer.parse_args is infer.compact.parse_args
    value = infer.parse_args(['--input-dir', 'i', '--output-dir', 'o', '--naf-checkpoint', 'n'])
    assert value.geometry_mode == 'C' and value.global_noise == 'zero'


def test_only_finish_is_scoped_and_saved_reports_identify_fast_source(monkeypatch, tmp_path):
    output = tmp_path / 'candidate'
    target = infer.compact.compact
    original_finish, original_write = target.finish_watershed, infer.entry.write
    watershed = np.array([[1, 1, -1, 77], [1, 1, 0, 77]], np.int32)
    policy = dict(min_instance_area=1)
    expected = original_finish(watershed, policy)
    document = dict(format='legacy_compact', complete=True, config=dict(inference=policy),
                    images=[dict(source='synthetic')])
    before = deepcopy(document)

    def run(values, *, device):
        assert device == 'cpu' and target.finish_watershed is not original_finish
        assert infer.entry.write is not original_write
        actual = target.finish_watershed(watershed, policy)
        assert np.array_equal(actual[0], expected[0]) and actual[1:] == expected[1:]
        (output / 'audit').mkdir(parents=True)
        for name in ('report.json', 'status.json', 'compact_report.json'):
            infer.entry.write(output / 'audit' / name, document)
        return document

    monkeypatch.setattr(infer.compact, 'run', run)
    result = infer.run(args(output), device='cpu')
    assert target.finish_watershed is original_finish and infer.entry.write is original_write
    assert document == before
    for name in ('report.json', 'status.json', 'compact_report.json', 'fast_readout_report.json'):
        saved = json.loads((output / 'audit' / name).read_text(encoding='utf-8'))
        assert saved['format'] == infer.FORMAT and saved['training'] is False
        assert saved['no_training'] is True
        assert saved['fast_readout_base_format'] == 'legacy_compact'
        assert saved['fast_readout_entry_sha256'] == infer.entry.sha(infer.__file__)
        assert saved['fast_readout_utility_sha256'] == infer.entry.sha(infer.fast.__file__)
        assert saved['fast_readout']['finish_watershed_calls'] == 1
        assert saved['fast_readout']['completed_finish_calls'] == 1
        assert saved['fast_readout']['raw_float32_vote_unchanged'] is True
        assert saved['fast_readout']['original_cap_input_and_function_unchanged'] is True
        assert saved['config']['inference'] == policy
    assert result == json.loads((output / 'audit/fast_readout_report.json').read_text())


@pytest.mark.parametrize('failure', ['branch', 'missing_finish'])
def test_exception_restores_both_patches_and_marks_candidate_failed(monkeypatch, tmp_path, failure):
    output = tmp_path / failure
    original_finish = infer.compact.compact.finish_watershed
    original_write = infer.entry.write

    def run(values, *, device):
        (output / 'audit').mkdir(parents=True)
        infer.entry.write(output / 'audit/status.json',
                          dict(complete=True, status='complete', images=[dict(source='x')]))
        if failure == 'branch':
            raise RuntimeError('deliberate branch failure')
        return dict(complete=True, images=[dict(source='x')])

    monkeypatch.setattr(infer.compact, 'run', run)
    with pytest.raises(RuntimeError, match='branch failure|exactly once'):
        infer.run(args(output), device='cpu')
    assert infer.compact.compact.finish_watershed is original_finish
    assert infer.entry.write is original_write
    failed = json.loads((output / 'audit/fast_readout_failed.json').read_text())
    status = json.loads((output / 'audit/status.json').read_text())
    assert failed['format'] == infer.FORMAT and failed['complete'] is False
    assert status['status'] == 'failed' and status['complete'] is False
    assert status['fast_readout']['finish_watershed_calls'] == 0


def test_existing_output_is_not_modified_or_annotated(monkeypatch, tmp_path):
    output = tmp_path / 'existing'
    (output / 'audit').mkdir(parents=True)
    path = output / 'audit/report.json'
    path.write_bytes(b'{"old":true}')
    monkeypatch.setattr(infer.compact, 'run', lambda *a, **k: pytest.fail('Must reject first'))
    with pytest.raises(FileExistsError, match='fresh'):
        infer.run(args(output), device='cpu')
    assert path.read_bytes() == b'{"old":true}'
    assert sorted(p.name for p in (output / 'audit').iterdir()) == ['report.json']


@pytest.mark.parametrize('mode', ['C', 'F0'])
def test_actual_small_watershed_exact_with_candidate_CLI_scope(monkeypatch, tmp_path, mode):
    config = dict(inference=dict(seed_min_area=1, final_min_instance_area=200,
        final_small_instance_action='neighbor_merge_v1', max_instance_id=65535,
        boundary_threshold=.65, bridge_width=1, watershed_dilate_width=1,
        marker_border_seal_width=2, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, semantic_vote_mode='probability_mean',
        semantic_vote_erode_width=0, semantic_vote_threshold=.5))
    after = np.full((64, 80), .1, np.float32)
    after[:, 39:42] = .95
    semantic = torch.linspace(-3, 3, after.size).reshape(1, 1, *after.shape)
    if mode == 'F0':
        semantic.zero_()
    output = tmp_path / mode
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        expected, old_proof = infer.compact.compact.compact_supported_readout(
            semantic, after, after, config, geometry_mode=mode)

        def run(values, *, device):
            actual, proof = infer.compact.compact.compact_supported_readout(
                semantic, after, after, config, geometry_mode=mode)
            assert np.array_equal(actual[0], expected[0]) and actual[1] == expected[1]
            assert proof == old_proof
            (output / 'audit').mkdir(parents=True)
            document = dict(complete=True, images=[dict(source='synthetic')])
            infer.entry.write(output / 'audit/report.json', document)
            return document

        monkeypatch.setattr(infer.compact, 'run', run)
        result = infer.run(args(output, mode), device='cpu')
        assert result['fast_readout']['geometry_mode'] == mode
        assert result['fast_readout']['completed_finish_calls'] == 1
    finally:
        torch.set_num_threads(old_threads)
