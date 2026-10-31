# -*- coding: utf-8 -*-
"""标准线宽T-only正式入口的CPU合同；不加载赛题数据或已训练分割权重。"""
from argparse import Namespace
from copy import deepcopy
import json

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from models.boundary_rgb import BoundaryRGBNAFNet
from models.boundary_restore import state_sha256
from tools import infer_boundary_rgb as wrapper


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def checkpoint(tmp_path, change=None):
    model_cfg = dict(width=4, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    model = BoundaryRGBNAFNet(**model_cfg, seed=17, input_mode='boundary')
    saved = dict(format=wrapper.CHECKPOINT_FORMAT, global_step=1000,
        architecture=model.architecture, model_state_dict=model.state_dict(),
        model_sha256=state_sha256(model), input_contract=deepcopy(wrapper.INPUT_CONTRACT),
        config=dict(model=model_cfg, seed=17, input_mode='boundary',
                    input_normalization=deepcopy(wrapper.INPUT_CONTRACT['input_normalization']),
                    inference=deepcopy(wrapper.INPUT_CONTRACT)),
        identity=dict(numerical_contract=deepcopy(wrapper.NUMERICAL_CONTRACT)))
    if change:
        change(saved)
    path = tmp_path / 'naf.pt'
    torch.save(saved, path)
    return path, model


def test_strict_load_keeps_cpu_rng_and_freezes_model(tmp_path):
    path, expected = checkpoint(tmp_path)
    state = torch.get_rng_state().clone()
    model, receipt = wrapper.load_naf_checkpoint(path, 'cpu', expected_model_sha=state_sha256(expected))
    assert torch.equal(state, torch.get_rng_state())
    assert state_sha256(model) == state_sha256(expected)
    assert wrapper._frozen(model)
    assert receipt['parameter_count'] == sum(p.numel() for p in expected.parameters())
    assert receipt['global_step'] == 1000 and receipt['checkpoint_sha256'] == wrapper.entry.sha(path)
    assert receipt['input_contract'] == wrapper.INPUT_CONTRACT
    assert receipt['numerical_contract'] == wrapper.NUMERICAL_CONTRACT


@pytest.mark.parametrize('change,pattern', [
    (lambda s: s.update(format='old_naf'), 'formal'),
    (lambda s: s.update(global_step=999), 'formal'),
    (lambda s: s['config'].update(seed=18), 'architecture/config'),
    (lambda s: s['config'].update(input_mode='rgb_boundary'), 'architecture/config'),
    (lambda s: s.update(model_sha256='0' * 64), 'SHA256 differs'),
    (lambda s: s['model_state_dict'].pop('intro.bias'), 'Missing key'),
    (lambda s: s['input_contract'].update(input_mode='rgb_boundary'), 'input contract'),
    (lambda s: s['input_contract'].update(extra_key=True), 'input contract'),
    (lambda s: s['input_contract'].update(tile_size=256.), 'input contract'),
    (lambda s: s['input_contract']['input_normalization'].update(kernel_size=7), 'input contract'),
    (lambda s: s['config']['inference'].update(input_threshold=.4), 'input_contract/config'),
    (lambda s: s['config']['input_normalization'].update(kernel_size=6), 'input_contract/config'),
    (lambda s: s['identity']['numerical_contract'].update(cudnn_tf32=True), 'numerical contract'),
])
def test_reject_changed_or_unknown_checkpoint_contract(tmp_path, change, pattern):
    path, _ = checkpoint(tmp_path, change)
    with pytest.raises((ValueError, RuntimeError), match=pattern):
        wrapper.load_naf_checkpoint(path, 'cpu')


def test_reject_wrong_selected_model(tmp_path):
    path, _ = checkpoint(tmp_path)
    with pytest.raises(ValueError, match='selected candidate'):
        wrapper.load_naf_checkpoint(path, 'cpu', expected_model_sha='0' * 64)


def runtime_flags():
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)


def set_runtime_flags(values):
    (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
     torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = values


@pytest.mark.parametrize('fails', [False, True])
def test_local_precision_context_restores_frontend_even_after_exception(fails):
    before = runtime_flags()
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    try:
        set_runtime_flags((True, True, True, False))
        try:
            with wrapper.naf_runtime_context():
                assert runtime_flags() == (False, False, False, True)
                assert torch.are_deterministic_algorithms_enabled() == deterministic_algorithms
                if fails:
                    raise ArithmeticError('synthetic NAF failure')
        except ArithmeticError:
            assert fails
        assert runtime_flags() == (True, True, True, False)
    finally:
        set_runtime_flags(before)


def install_fake_entry(tmp_path, monkeypatch, *, decode_count=1, vote_count=1,
                       alter_naf=False, alter_input=False, base_report_extra=0,
                       helper_fails=False, source_channels=3):
    path, expected = checkpoint(tmp_path)
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    source = inputs / 'test_169.png'
    rgb = np.zeros((8, 9, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 19, 73, 127
    bgr = rgb[..., ::-1]
    if source_channels == 4:
        bgr = np.dstack((bgr, np.full((8, 9), 255, np.uint8)))
    assert cv2.imwrite(str(source), bgr)
    output = tmp_path / 'candidate'
    args = Namespace(input_dir=str(inputs), output_dir=str(output), model_dir=None,
        config=str(wrapper.ROOT / 'config/inference/semifinal_n1.yaml'), dataset='semifinal_150',
        only=['test_169'], naf_checkpoint=str(path), capture_names=None)
    state = dict(helper_calls=0, raw_predict_calls=0, reflect_calls=0, vote_calls=0,
                 sentinel=object(), before=torch.full((1, 1, 8, 9), .2),
                 semantic=torch.full((1, 1, 8, 9), -.3),
                 raw_probability=np.full((8, 9), .7, np.float32), rgb=rgb)
    backend, restorer = nn.Linear(1, 1), nn.Linear(1, 1)
    base_count = sum(p.numel() for module in [backend, restorer] for p in module.parameters())
    outer_flags = runtime_flags()

    def fake_models(*unused):
        return backend, restorer, None

    def fake_helper(before, actual_rgb, naf, settings, *, device):
        state['helper_calls'] += 1
        assert settings == wrapper.INPUT_CONTRACT and device == 'cpu'
        assert np.array_equal(actual_rgb, rgb)
        assert runtime_flags() == (False, False, False, True)
        assert wrapper._frozen(naf)
        if helper_fails:
            raise ArithmeticError('synthetic NAF failure')
        if alter_naf:
            with torch.no_grad():
                next(naf.parameters()).add_(1)
        after = np.full_like(before, .8)
        if alter_input:
            before[:] = .9
        return after, dict(output_sha256=wrapper.array_sha(after), input_mode='boundary',
                          input_normalization=deepcopy(wrapper.INPUT_CONTRACT['input_normalization']))

    def fake_reflect(semantic, boundary, config, *, original_decode=None):
        state['reflect_calls'] += 1
        assert runtime_flags() == outer_flags
        assert semantic is state['semantic'] and original_decode is state['sentinel']
        assert torch.equal(boundary, torch.full_like(boundary, .8))
        return np.ones((8, 9), np.uint16), {'1': 1}

    def fake_vote(instances, probability):
        state['vote_calls'] += 1
        assert runtime_flags() == outer_flags
        assert probability is state['raw_probability']
        return {'1': 1}, {'1': .7}

    def fake_predict(model, restorer, terminal, path, config, device, **kwargs):
        state['raw_predict_calls'] += 1
        assert runtime_flags() == outer_flags
        result = (np.ones((8, 9), np.uint16), {'1': 1})
        for unused in range(decode_count):
            result = wrapper.entry.decode_reflect(state['semantic'], state['before'], config,
                                                  original_decode=state['sentinel'])
        for unused in range(vote_count):
            voted, _ = wrapper.entry.views.revote_instances(result[0], state['raw_probability'])
            result = (result[0], voted)
        wrapper.entry.verify_prediction(result[0], result[1], rgb.shape[:2])
        return rgb, dict(instances=result[0], classes=result[1], boundary=state['before'][0, 0].numpy())

    def fake_run(actual_args):
        assert not actual_args.save_analysis
        (output / 'audit').mkdir(parents=True)
        (output / 'results').mkdir()
        wrapper.entry.write(output / 'audit/status.json', dict(complete=False, status='running'))
        config = wrapper.entry.load_config(actual_args.config)
        models = wrapper.entry.load_models(config, {}, 'cpu')
        _, state['last_result'] = wrapper.entry.predict(*models, source, config, 'cpu', local_restorer=None)
        wrapper.entry.write(output / 'audit/report.json', dict(complete=True,
            frozen_models_unchanged=True, parameter_count=base_count + base_report_extra,
            geometry='old frozen N1 geometry', models={'backend': {}, 'restorer': {}},
            model_states={'backend': 'fixed', 'restorer': 'fixed'}, images=[dict(source='test_169')]))
        wrapper.entry.write(output / 'audit/status.json', dict(complete=True, status='complete'))

    monkeypatch.setattr(wrapper.entry, 'load_models', fake_models)
    monkeypatch.setattr(wrapper.entry, 'predict', fake_predict)
    monkeypatch.setattr(wrapper.entry, 'decode_reflect', fake_reflect)
    monkeypatch.setattr(wrapper.entry.views, 'revote_instances', fake_vote)
    monkeypatch.setattr(wrapper.entry, 'run', fake_run)
    monkeypatch.setattr(wrapper, 'predict_boundary_rgb_tiled', fake_helper)
    return args, state, output, expected, base_count, fake_predict, fake_reflect


@pytest.mark.parametrize('channels', [3, 4])
def test_hook_preloads_native_rgb_calls_once_keeps_frontend_and_counts_naf_once(tmp_path, monkeypatch, channels):
    args, state, output, expected, base_count, fake_predict, fake_reflect = install_fake_entry(
        tmp_path, monkeypatch, source_channels=channels)
    before, semantic = state['before'].clone(), state['semantic'].clone()
    old_flags = runtime_flags()
    receipt = wrapper.run(args, device='cpu')
    count = sum(p.numel() for p in expected.parameters())
    assert receipt['parameter_count'] == receipt['total_parameters'] == base_count + count
    assert receipt['upstream_parameter_count'] == base_count
    assert state['helper_calls'] == state['reflect_calls'] == state['raw_predict_calls'] == state['vote_calls'] == 1
    assert torch.equal(before, state['before']) and torch.equal(semantic, state['semantic'])
    assert wrapper.entry.predict is fake_predict and wrapper.entry.decode_reflect is fake_reflect
    assert runtime_flags() == old_flags
    assert not (output / 'audit/naf_captures').exists()
    report = json.loads((output / 'audit/report.json').read_text())
    assert report['parameter_count'] == receipt['parameter_count']
    assert report['parameter_count_includes_naf']
    assert report['models']['boundary_naf']['parameter_count'] == count
    assert report['upstream_geometry'] == 'old frozen N1 geometry'
    row = report['images'][0]['boundary_naf']
    assert row['input_boundary_sha256'] == wrapper.array_sha(before[0, 0].numpy())
    assert row['raw_probability_sha256'] == wrapper.array_sha(state['raw_probability'])
    assert report['returned_boundary_is_upstream'] and receipt['naf_frozen_unchanged']
    assert report['naf_complete']
    assert json.loads((output / 'audit/status.json').read_text())['naf_complete']


def test_only_named_source_capture_is_outside_results_including_raw_vote(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_names = ['test_169']
    receipt = wrapper.run(args, device='cpu')
    relative = receipt['images'][0]['capture']
    assert relative == 'audit/naf_captures/test_169.npz'
    with np.load(output / relative) as values:
        assert set(values.files) == {'boundary_before', 'boundary_after', 'semantic', 'raw_probability'}
        assert np.array_equal(values['boundary_before'], state['before'][0, 0].numpy())
        assert np.all(values['boundary_after'] == np.float32(.8))
        assert np.array_equal(values['semantic'], state['semantic'].numpy())
        assert np.array_equal(values['raw_probability'], state['raw_probability'])
    assert not list((output / 'results').iterdir())


@pytest.mark.parametrize('decode_count,vote_count', [(0, 0), (2, 1), (1, 0), (1, 2)])
def test_missing_or_double_decode_or_vote_rejected(tmp_path, monkeypatch, decode_count, vote_count):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch,
        decode_count=decode_count, vote_count=vote_count)
    with pytest.raises(RuntimeError, match='exactly one'):
        wrapper.run(args, device='cpu')
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


@pytest.mark.parametrize('kind,pattern', [('input', 'modified its input'), ('model', 'checkpoint/model/source')])
def test_input_or_naf_state_mutation_is_rejected(tmp_path, monkeypatch, kind, pattern):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch,
        alter_input=kind == 'input', alter_naf=kind == 'model')
    with pytest.raises(RuntimeError, match=pattern):
        wrapper.run(args, device='cpu')
    assert torch.all(state['before'] == .2)
    assert not json.loads((output / 'audit/status.json').read_text())['complete']
    if kind == 'model':
        report = json.loads((output / 'audit/report.json').read_text())
        assert not report['complete'] and not report['naf_complete']


def test_helper_exception_restores_runtime_and_all_hooks(tmp_path, monkeypatch):
    args, state, output, _, _, fake_predict, fake_reflect = install_fake_entry(tmp_path, monkeypatch, helper_fails=True)
    flags = runtime_flags()
    with pytest.raises(ArithmeticError, match='NAF failure'):
        wrapper.run(args, device='cpu')
    assert runtime_flags() == flags
    assert wrapper.entry.predict is fake_predict and wrapper.entry.decode_reflect is fake_reflect
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


def test_total_parameter_limit_checked_before_prediction(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, 'MAX_PARAMETERS', 1)
    with pytest.raises(RuntimeError, match='parameters exceed'):
        wrapper.run(args, device='cpu')
    assert state['raw_predict_calls'] == state['helper_calls'] == 0


def test_base_report_cannot_already_include_naf_parameters(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch, base_report_extra=1)
    with pytest.raises(RuntimeError, match='counted twice'):
        wrapper.run(args, device='cpu')
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


def test_unselected_capture_rejected_before_creating_output(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.capture_names = ['test_170']
    with pytest.raises(ValueError, match='capture-names'):
        wrapper.run(args, device='cpu')
    assert not output.exists()


def test_full_map_capture_cannot_be_enabled(tmp_path):
    with pytest.raises(ValueError, match='bounded'):
        wrapper.run(Namespace(save_analysis=True), device='cpu')


def test_existing_output_cannot_be_marked_failed_by_new_run(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    (output / 'audit').mkdir(parents=True)
    target = output / 'audit/status.json'
    target.write_bytes(b'preserved previous audit')
    with pytest.raises(FileExistsError, match='fresh output'):
        wrapper.run(args, device='cpu')
    assert target.read_bytes() == b'preserved previous audit'
    assert state['raw_predict_calls'] == 0


def install_support_readout(monkeypatch, state, *, bad_capture=None, all_unsupported=False):
    markers = np.zeros((8, 9), np.int32)
    markers[1:3, 1:3] = 2
    markers[5:7, 5:7] = 7
    state['before'][0, 0, 5:7, 5:7] = .9
    if all_unsupported:
        state['before'][0, 0, 1:3, 1:3] = .9
    observed = dict(stages=0, reflected=0, replay=0)
    outer_flags = runtime_flags()

    def stages(after, config):
        observed['stages'] += 1
        assert np.all(after == np.float32(.8))
        return dict(token='frozen_NAF_stages')

    def reflected(actual, mode, padding):
        observed['reflected'] += 1
        assert actual['token'] == 'frozen_NAF_stages' and mode == 'reflect' and padding == 32
        return dict(marker_labels=markers.copy())

    def replay(semantic, boundary, config, filtered):
        observed['replay'] += 1
        assert runtime_flags() == outer_flags and semantic is state['semantic']
        assert torch.all(boundary == .8) and boundary.shape == (1, 1, 8, 9)
        assert np.array_equal(filtered == 2, markers == 2) and not (filtered == 7).any()
        observed['filtered'] = filtered.copy()
        capture = dict(actual_watershed_calls=1, original_marker_exact=True,
                       original_elevation_exact=True, real_barrier_changed=False)
        if bad_capture:
            capture[bad_capture] = True if bad_capture == 'real_barrier_changed' else False
        return dict(instances=np.ones((8, 9), np.uint16), classes={'1': 0},
            actual_markers=filtered.copy(), capture_audit=capture,
            hashes=dict(elevation_sha256='fixed_high_elevation', actual_marker_sha256=wrapper.array_sha(filtered)))

    monkeypatch.setattr(wrapper, 'marker_stages', stages)
    monkeypatch.setattr(wrapper, 'build_marker_variant', reflected)
    monkeypatch.setattr(wrapper, 'decode_marker_replay', replay)
    return observed, markers


@pytest.mark.parametrize('mode', [None, 'none'])
def test_default_and_explicit_none_preserve_saved_final_reflect_and_raw_vote(tmp_path, monkeypatch, mode):
    args, state, _, _, _, _, fake_reflect = install_fake_entry(tmp_path, monkeypatch)
    if mode is not None:
        args.seed_support = mode
    def forbidden(*args, **kwargs):
        pytest.fail('none must not enter S0 or repeat independent N1 inference')
    monkeypatch.setattr(wrapper, '_supported_readout', forbidden)
    receipt = wrapper.run(args, device='cpu')
    assert state['reflect_calls'] == state['helper_calls'] == state['raw_predict_calls'] == state['vote_calls'] == 1
    assert receipt['seed_support'] == receipt['images'][0]['seed_support'] == 'none'
    assert receipt['removed_seed_count'] == 0
    assert state['last_result']['classes'] == {'1': 1}
    assert np.array_equal(state['last_result']['instances'], np.ones((8, 9), np.uint16))
    assert wrapper.entry.decode_reflect is fake_reflect
    assert receipt['geometry'].endswith('fresh reflect32 seeds/barriers -> unchanged WS/raw vote')


def test_s0_uses_original_reference_once_then_keeps_one_final_raw_vote(tmp_path, monkeypatch):
    args, state, output, expected, *_ = install_fake_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, 'EXPECTED_MODEL_SHA', state_sha256(expected))
    args.seed_support = 'original-zero'
    observed, markers = install_support_readout(monkeypatch, state)
    before = state['before'].clone()
    flags = runtime_flags()
    receipt = wrapper.run(args, device='cpu')
    audit = receipt['images'][0]['seed_support']
    assert observed['stages'] == observed['reflected'] == observed['replay'] == 1
    assert state['reflect_calls'] == 0
    assert state['helper_calls'] == state['raw_predict_calls'] == state['vote_calls'] == 1
    assert audit['removed_ids'] == [7] and audit['retained_ids'] == [2] and audit['low_threshold'] == .45
    assert np.array_equal(observed['filtered'][markers == 2], markers[markers == 2])
    assert audit['reference'] == 'original_N1_pre_NAF_boundary'
    assert audit['reference_raw_array_sha256'] == wrapper.array_sha(before[0, 0].numpy())
    assert audit['retained_marker_pixels_and_IDs_exact'] and not audit['readout_audit']['real_barrier_changed']
    assert audit['readout_hashes']['elevation_sha256'] == 'fixed_high_elevation'
    assert receipt['seed_support'] == 'original-zero' and receipt['removed_seed_count'] == 1
    assert receipt['geometry'] == ('N1 stitched B -> whole threshold/skeleton/ellipse8 -> fixed20a59 NAF -> '
        'fresh reflect32 seeds/barriers -> remove only seeds with zero original_N1<=.45 support -> unchanged WS/raw vote')
    assert 'zero original_N1<=.45 support' in receipt['geometry']
    assert torch.equal(before, state['before']) and runtime_flags() == flags
    assert state['last_result']['classes'] == {'1': 1}  # 末端raw票覆盖geometry类别0。
    assert receipt['imported_sources']['utils/marker_seed_support.py'] == wrapper.entry.sha(wrapper.ROOT / 'utils/marker_seed_support.py')
    for name in ('report.json', 'status.json'):
        saved = json.loads((output / 'audit' / name).read_text())
        assert saved['seed_support'] == 'original-zero' and saved['removed_seed_count'] == 1


@pytest.mark.parametrize('field', ['original_marker_exact', 'original_elevation_exact', 'real_barrier_changed'])
def test_s0_contract_failure_restores_hooks_marks_failed_and_does_not_fallback(tmp_path, monkeypatch, field):
    args, state, output, expected, _, fake_predict, fake_reflect = install_fake_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, 'EXPECTED_MODEL_SHA', state_sha256(expected))
    args.seed_support = 'original-zero'
    observed, _ = install_support_readout(monkeypatch, state, bad_capture=field)
    flags = runtime_flags()
    with pytest.raises(RuntimeError, match='watershed contract'):
        wrapper.run(args, device='cpu')
    assert observed['replay'] == 1 and state['reflect_calls'] == state['vote_calls'] == 0
    assert wrapper.entry.predict is fake_predict and wrapper.entry.decode_reflect is fake_reflect
    assert runtime_flags() == flags
    status = json.loads((output / 'audit/status.json').read_text())
    assert not status['complete'] and not status['naf_complete'] and status['status'] == 'failed'


def test_s0_rejects_all_seed_removal_without_original_reflect_fallback(tmp_path, monkeypatch):
    args, state, output, expected, *_ = install_fake_entry(tmp_path, monkeypatch)
    monkeypatch.setattr(wrapper, 'EXPECTED_MODEL_SHA', state_sha256(expected))
    args.seed_support = 'original-zero'
    observed, _ = install_support_readout(monkeypatch, state, all_unsupported=True)
    with pytest.raises(ValueError, match='All positive seeds'):
        wrapper.run(args, device='cpu')
    assert observed['replay'] == state['reflect_calls'] == state['vote_calls'] == 0
    assert not json.loads((output / 'audit/status.json').read_text())['complete']


@pytest.mark.parametrize('mode', ['arbitrary', True, None])
def test_unknown_support_mode_rejected_before_output_or_forward(tmp_path, monkeypatch, mode):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.seed_support = mode
    with pytest.raises(ValueError, match='seed-support'):
        wrapper.run(args, device='cpu')
    assert not output.exists() and state['helper_calls'] == state['raw_predict_calls'] == 0


def test_s0_cannot_override_selected_20a59_checkpoint(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.seed_support = 'original-zero'
    with pytest.raises(ValueError, match='selected candidate'):
        wrapper.run(args, device='cpu')
    args.expected_model_sha = '0' * 64
    with pytest.raises(ValueError, match='fixed 20a59'):
        wrapper.run(args, device='cpu')
    assert not output.exists() and state['helper_calls'] == 0


@pytest.mark.parametrize('explicit,expected', [
    (None, wrapper.EXPECTED_MODEL_SHA),
    (wrapper.EXPECTED_MODEL_SHA, wrapper.EXPECTED_MODEL_SHA),
    (wrapper.NOGAN_RIDGE_MODEL_SHA, wrapper.NOGAN_RIDGE_MODEL_SHA),
])
def test_s0_model_identity_keeps_legacy_default_and_requires_explicit_nogan(explicit, expected):
    assert wrapper._s0_expected_model_sha(explicit) == expected


@pytest.mark.parametrize('unknown', ['0' * 64, '', True, '8eb7c8da'])
def test_s0_model_identity_rejects_unverified_or_partial_hash(unknown):
    with pytest.raises(ValueError, match='selected model SHA'):
        wrapper._s0_expected_model_sha(unknown)


def test_explicit_verified_nogan_s0_keeps_reference_and_records_actual_identity(tmp_path, monkeypatch):
    args, state, output, expected, *_ = install_fake_entry(tmp_path, monkeypatch)
    # 以合成checkpoint模拟第二个已核验候选，不改变旧缺省或实际严格加载。
    digest = state_sha256(expected)
    monkeypatch.setattr(wrapper, 'NOGAN_RIDGE_MODEL_SHA', digest)
    args.seed_support = 'original-zero'
    args.expected_model_sha = digest
    observed, markers = install_support_readout(monkeypatch, state)
    before = state['before'].clone()
    receipt = wrapper.run(args, device='cpu')
    assert receipt['naf']['model_sha256'] == digest
    assert 'fixed' + digest[:8] + ' NAF' in receipt['geometry']
    assert 'fixed20a59' not in receipt['geometry']
    assert observed['replay'] == 1 and state['reflect_calls'] == 0
    assert state['helper_calls'] == state['raw_predict_calls'] == state['vote_calls'] == 1
    audit = receipt['images'][0]['seed_support']
    assert audit['removed_ids'] == [7] and audit['retained_ids'] == [2]
    assert np.array_equal(observed['filtered'][markers == 2], markers[markers == 2])
    assert audit['reference'] == 'original_N1_pre_NAF_boundary'
    assert audit['reference_raw_array_sha256'] == wrapper.array_sha(before[0, 0].numpy())
    assert torch.equal(state['before'], before)
    saved = json.loads((output / 'audit/report.json').read_text())
    assert saved['model_states']['boundary_naf'] == digest
    assert saved['geometry'] == receipt['geometry']


def test_nogan_s0_hash_does_not_accept_a_different_checkpoint(tmp_path, monkeypatch):
    args, state, output, *_ = install_fake_entry(tmp_path, monkeypatch)
    args.seed_support = 'original-zero'
    args.expected_model_sha = wrapper.NOGAN_RIDGE_MODEL_SHA
    with pytest.raises(ValueError, match='selected candidate'):
        wrapper.run(args, device='cpu')
    assert not output.exists() and state['helper_calls'] == state['raw_predict_calls'] == 0


@pytest.mark.parametrize('remove_one', [False, True])
def test_s0_real_frozen_watershed_matches_exact_expected_output_and_uint16(remove_one):
    cfg = wrapper.entry.load_config(str(wrapper.ROOT / 'config/inference/semifinal_n1.yaml'))
    after = np.full((96, 96), .1, np.float32)
    after[44:52] = .95
    after[:, 44:52] = .95
    semantic = torch.zeros(1, 1, 96, 96)
    stages = wrapper.marker_stages(after, cfg)
    markers = wrapper.build_marker_variant(stages, 'reflect', 32)['marker_labels']
    marker_ids = np.unique(markers[markers > 0])
    assert len(marker_ids) == 4
    reference = np.full_like(after, .1)
    if remove_one:
        reference[markers == marker_ids[0]] = .9
    original_after, original_reference = after.copy(), reference.copy()
    filtered, wanted = wrapper.filter_unsupported_seeds(markers, reference, .45)
    control = wrapper.decode_marker_replay(semantic, torch.from_numpy(after)[None, None], cfg, markers)
    expected = wrapper.decode_marker_replay(semantic, torch.from_numpy(after)[None, None], cfg, filtered)
    (instances, classes), audit = wrapper._supported_readout(semantic, after, reference, cfg)
    assert np.array_equal(instances, expected['instances']) and classes == expected['classes']
    assert audit['removed_ids'] == wanted['removed_ids']
    assert audit['readout_hashes'] == expected['hashes']
    assert audit['readout_hashes']['elevation_sha256'] == control['elevation_sha256']
    if not remove_one:
        original_ids, original_classes = wrapper.entry.decode_reflect(semantic, torch.from_numpy(after)[None, None], cfg)
        assert np.array_equal(instances, original_ids) and classes == original_classes
    # geometry类别沿用原int键；真正提交契约在唯一的末端raw复投票之后核验。
    voted, _ = wrapper.entry.views.revote_instances(instances, np.full_like(after, .7))
    wrapper.entry.verify_prediction(instances, voted, after.shape)
    areas = np.bincount(instances.ravel())[1:]
    assert instances.dtype == np.uint16 and (areas[areas > 0] >= 50).all()
    assert np.array_equal(after, original_after) and np.array_equal(reference, original_reference)


def test_seed_support_cli_defaults_and_explicit_mode(monkeypatch):
    observed = []
    def run(args):
        observed.append(args.seed_support)
        return dict(images=[], parameter_count=1)
    monkeypatch.setattr(wrapper, 'run', run)
    common = ['--input-dir', 'input', '--output-dir', 'output', '--naf-checkpoint', 'model.pt']
    wrapper.main(common)
    wrapper.main(common + ['--seed-support', 'original-zero'])
    assert observed == ['none', 'original-zero']
    with pytest.raises(SystemExit):
        wrapper.main(common + ['--seed-support', 'invalid'])


def test_s0_cli_legacy_sha_default_and_explicit_verified_nogan(monkeypatch):
    observed = []
    def run(args):
        assert args.seed_support == 'original-zero'
        observed.append(wrapper._s0_expected_model_sha(args.expected_model_sha))
        return dict(images=[], parameter_count=1)
    monkeypatch.setattr(wrapper, 'run', run)
    common = ['--input-dir', 'input', '--output-dir', 'output', '--naf-checkpoint', 'model.pt',
              '--seed-support', 'original-zero']
    wrapper.main(common)
    wrapper.main(common + ['--expected-model-sha', wrapper.NOGAN_RIDGE_MODEL_SHA])
    assert observed == [wrapper.EXPECTED_MODEL_SHA, wrapper.NOGAN_RIDGE_MODEL_SHA]
