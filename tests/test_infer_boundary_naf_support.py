# -*- coding: utf-8 -*-
"""NAF零支撑种子分支、原概率缓存和完整流水线计时的CPU契约。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import infer_boundary_naf as wrapper
from test_infer_boundary_naf import install_fake_entry, one_cpu_thread


def install_support(monkeypatch, state, *, bad_replay=None, all_unsupported=False):
    markers = np.zeros((8, 9), np.int32)
    markers[1:3, 1:3] = 2
    markers[5:7, 5:7] = 7
    state['before'][0, 0, 5:7, 5:7] = .9
    if all_unsupported:
        state['before'][0, 0, 1:3, 1:3] = .9
    observed = dict(stages=0, variant=0, replay=0)

    def stages(after, config):
        observed['stages'] += 1
        assert after.dtype == np.float32 and np.all(after == np.float32(.8))
        return dict(marker_labels=np.full_like(markers, 99), token='original_stages')

    def variant(base, mode, padding):
        observed['variant'] += 1
        assert base['token'] == 'original_stages' and mode == 'reflect' and padding == 32
        return dict(marker_labels=markers.copy())

    def replay(semantic, boundary, config, filtered):
        observed['replay'] += 1
        assert semantic is state['semantic']
        assert boundary.shape == (1, 1, 8, 9) and torch.all(boundary == .8)
        assert filtered.dtype == np.int32 and np.array_equal(filtered == 2, markers == 2)
        assert not np.any(filtered == 7)
        observed['filtered'] = filtered.copy()
        capture = dict(actual_watershed_calls=1, original_marker_exact=True,
                       original_elevation_exact=True, real_barrier_changed=False)
        if bad_replay:
            capture[bad_replay] = True if bad_replay == 'real_barrier_changed' else False
        return dict(instances=np.ones((8, 9), np.uint16), classes={'1': 0},
                    actual_markers=filtered.copy(), capture_audit=capture,
                    hashes=dict(elevation_sha256='fixed'))

    monkeypatch.setattr(wrapper, 'marker_stages', stages)
    monkeypatch.setattr(wrapper, 'build_marker_variant', variant)
    monkeypatch.setattr(wrapper, 'decode_marker_replay', replay)
    return observed, markers


def test_original_zero_uses_original_n1_reference_actual_reflect_markers_and_one_replay(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.seed_support = 'original-zero'
    observed, markers = install_support(monkeypatch, state)
    before = state['before'].clone()
    receipt = wrapper.run(args, device='cpu')
    audit = receipt['images'][0]['seed_support']
    assert observed['stages'] == observed['variant'] == observed['replay'] == 1
    assert state['reflect_calls'] == 0 and state['helper_calls'] == state['raw_predict_calls'] == 1
    assert audit['low_threshold'] == .45 and audit['removed_ids'] == [7] and audit['retained_ids'] == [2]
    assert audit['reference'] == 'original_N1_pre_NAF_boundary'
    assert audit['reference_raw_array_sha256'] == wrapper.array_sha(before[0, 0].numpy())
    assert audit['removed_seed_count'] == 1 and audit['pre_watershed_seed_only']
    assert torch.equal(state['before'], before)
    assert receipt['seed_support'] == 'original-zero'
    assert json.loads((output / 'audit/report.json').read_text())['seed_support'] == 'original-zero'


@pytest.mark.parametrize('mode', [None, 'none'])
def test_default_and_explicit_none_never_invoke_support_replay(tmp_path, monkeypatch, mode):
    args, state, *_ = install_fake_entry(tmp_path, monkeypatch)
    if mode is not None:
        args.seed_support = mode
    def forbidden(*a, **k):
        pytest.fail('default branch must call saved reflect only')
    monkeypatch.setattr(wrapper, '_supported_readout', forbidden)
    receipt = wrapper.run(args, device='cpu')
    assert receipt['seed_support'] == 'none' and state['reflect_calls'] == 1
    assert receipt['images'][0]['seed_support']['unchanged_saved_reflect']


@pytest.mark.parametrize('field', ['original_marker_exact', 'original_elevation_exact', 'real_barrier_changed'])
def test_filtered_replay_rejects_changed_watershed_contract(tmp_path, monkeypatch, field):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.seed_support = 'original-zero'
    install_support(monkeypatch, state, bad_replay=field)
    with pytest.raises(RuntimeError, match='watershed contract'):
        wrapper.run(args, device='cpu')
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


def test_all_seeds_removed_is_error_and_never_falls_back(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.seed_support = 'original-zero'
    observed, _ = install_support(monkeypatch, state, all_unsupported=True)
    with pytest.raises(ValueError, match='All positive seeds'):
        wrapper.run(args, device='cpu')
    assert observed['replay'] == state['reflect_calls'] == 0
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


def test_unknown_support_mode_rejected_before_model_or_output(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.seed_support = 'arbitrary'
    with pytest.raises(ValueError, match='seed-support'):
        wrapper.run(args, device='cpu')
    assert not output.exists() and not state['helper_calls']


def test_supported_readout_matches_actual_frozen_watershed_on_synthetic_grid():
    config = wrapper.entry.load_config(str(wrapper.ROOT / 'config/inference/semifinal_n1.yaml'))
    after = np.full((96, 96), .1, np.float32)
    after[44:52] = .95
    after[:, 44:52] = .95
    stages = wrapper.marker_stages(after, config)
    markers = wrapper.build_marker_variant(stages, 'reflect', 32)['marker_labels']
    ids = np.unique(markers[markers > 0])
    assert len(ids) >= 2
    reference = np.full_like(after, .1)
    reference[markers == ids[0]] = .9
    filtered, expected_audit = wrapper.filter_unsupported_seeds(markers, reference, .45)
    semantic = torch.zeros(1, 1, 96, 96)
    expected = wrapper.decode_marker_replay(semantic, torch.from_numpy(after)[None, None], config, filtered)
    before = after.copy()
    (instances, classes), audit = wrapper._supported_readout(semantic, after, reference, config)
    assert np.array_equal(instances, expected['instances']) and classes == expected['classes']
    assert audit['removed_ids'] == expected_audit['removed_ids']
    assert audit['readout_hashes'] == expected['hashes']
    assert not audit['readout_audit']['real_barrier_changed'] and np.array_equal(after, before)


def test_cuda_synchronization_is_explicit_but_cpu_never_touches_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(str(device)))
    wrapper._synchronize('cpu')
    wrapper._synchronize(torch.device('cpu'))
    assert calls == []
    wrapper._synchronize('cuda:1')
    assert calls == ['cuda:1']


def test_timing_summaries_cover_full_predict_and_preserve_upstream_runtime(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    original_run = wrapper.entry.run
    runtime = dict(device_name='synthetic CPU', precision='FP32', autocast=False)
    def run(actual):
        original_run(actual)
        path = output / 'audit/report.json'
        report = json.loads(path.read_text())
        report['runtime'] = runtime
        wrapper.entry.write(path, report)
    monkeypatch.setattr(wrapper.entry, 'run', run)
    receipt = wrapper.run(args, device='cpu')
    image = receipt['images'][0]
    timing = receipt['timing']
    assert timing['upstream_runtime'] == runtime and not timing['cuda_synchronized']
    assert timing['model_load_seconds'] == timing['naf_load_seconds'] + timing['backend_load_seconds']
    assert timing['run_wall_seconds'] >= timing['entry_run_wall_seconds'] >= image['predict_seconds']
    assert image['predict_seconds'] >= image['naf_seconds'] + image['readout_seconds'] >= 0
    for key in ['predict', 'naf', 'readout']:
        assert timing[key]['count'] == 1
        assert timing[key]['sum_seconds'] == timing[key]['mean_seconds'] == timing[key]['median_seconds'] == timing[key]['p90_seconds']
    assert 'not pure GPU' in timing['predict_scope']
    assert json.loads((output / 'audit/report.json').read_text())['naf_timing'] == timing


def install_raw_vote(monkeypatch, state):
    predict = wrapper.entry.predict
    raw = np.full((8, 9), .7, np.float32)
    vote_result = ({'1': 1}, {'1': .7})
    observed = dict(calls=0)
    def vote(instances, probability, *, synthetic_option=False):
        observed['calls'] += 1
        assert probability is raw and synthetic_option
        return vote_result
    def with_vote(*args, **kwargs):
        rgb, result = predict(*args, **kwargs)
        actual = wrapper.entry.views.revote_instances(result['instances'], raw, synthetic_option=True)
        assert actual is vote_result
        result['classes'], result['instance_scores'] = actual
        return rgb, result
    monkeypatch.setattr(wrapper.entry, 'predict', with_vote)
    monkeypatch.setattr(wrapper.entry.views, 'revote_instances', vote)
    return raw, observed


def test_capture_all_maps_preserves_raw_vote_and_excludes_npz_save_from_predict_timer(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_all_maps = True
    raw, observed = install_raw_vote(monkeypatch, state)
    clock = dict(now=0.)
    monkeypatch.setattr(wrapper.time, 'perf_counter', lambda: clock['now'])
    original_save = np.savez_compressed
    def slow_save(*a, **k):
        clock['now'] += 60.
        return original_save(*a, **k)
    monkeypatch.setattr(wrapper.np, 'savez_compressed', slow_save)
    receipt = wrapper.run(args, device='cpu')
    record = receipt['images'][0]
    with np.load(output / record['capture']) as saved:
        assert set(saved.files) == {'boundary_before', 'boundary_after', 'semantic', 'raw_probability'}
        assert np.array_equal(saved['raw_probability'], raw)
    assert observed['calls'] == 1 and record['predict_seconds'] == 0.
    assert receipt['timing']['entry_run_wall_seconds'] == receipt['timing']['run_wall_seconds'] == 60.
    assert record['raw_probability_sha256'] == wrapper.array_sha(raw)
    assert receipt['capture_all_maps']


def test_capture_all_and_selected_names_are_mutually_exclusive(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_all_maps = True
    args.capture_names = ['test_169']
    with pytest.raises(ValueError, match='mutually exclusive'):
        wrapper.run(args, device='cpu')
    assert not output.exists()


def test_capture_all_requires_exactly_one_actual_raw_vote(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_all_maps = True
    with pytest.raises(RuntimeError, match='exactly one raw vote'):
        wrapper.run(args, device='cpu')
    assert not list((output / 'audit/naf_captures').glob('*.npz'))


def test_capture_refuses_disk_full_without_deleting_files(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_all_maps = True
    install_raw_vote(monkeypatch, state)
    monkeypatch.setattr(wrapper.shutil, 'disk_usage', lambda path: SimpleNamespace(free=512 * 1024**2))
    with pytest.raises(OSError, match='Insufficient capture disk space'):
        wrapper.run(args, device='cpu')
    assert (output / 'audit/status.json').is_file()
    assert not list((output / 'audit/naf_captures').glob('*.npz'))


def test_summary_uses_all_images_not_only_mean():
    values = [dict(predict_seconds=n) for n in [1., 2., 9.]]
    summary = wrapper._timing_summary(values, 'predict_seconds')
    assert summary['count'] == 3 and summary['sum_seconds'] == 12.
    assert summary['mean_seconds'] == 4. and summary['median_seconds'] == 2.
    assert summary['p90_seconds'] == pytest.approx(7.6)
