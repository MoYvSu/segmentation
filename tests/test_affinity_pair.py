# -*- coding: utf-8 -*-
"""同源坐标、原主流逐像素兼容、独立随机流和一次更新的真实约束。"""
from copy import deepcopy
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import data.affinity_pair as pair_module
from data.affinity_mixed import MixedViewDataset, view_plan
from data.affinity_native import geometry_sample
from data.affinity_pair import PairViewDataset, validate_weight
from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from data.direct_dual_head_dataset import _spatial_transform
from data.sam2_geometry_dataset import SAM2GeometryDataset
from train_affinity_connectivity import draw_receipt
from train_affinity_pair import (branch_coefficients, capture_rng_state, compare_pair, compare_primary,
    gradient_diagnostics, isolated_random_stream, pair_receipt, reference_config, restore_rng_state)
from train_affinity_pair import verify_diagnostics, verify_monitor
from train_affinity_pair import zero_difference, repeatability_limits, verify_zero_numeric
from train_affinity_balance import comparable
from utils.affinity_loss import balanced_affinity_loss, build_affinity_targets_torch
from utils.config import load_config


@pytest.fixture
def cohort(tmp_path):
    raw, gt, sam = [tmp_path / name for name in ('raw', 'gt', 'sam2')]
    for folder in (raw, gt, sam):
        folder.mkdir()
    yy, xx = np.mgrid[:48, :64]
    rgb = np.stack([xx * 3, yy * 4, (xx + yy) * 2], -1).astype(np.uint8)
    cv2.imwrite(str(raw / 'source.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (raw / 'source.json').write_text('unused as GT')
    ids = np.ones((48, 64), np.uint16)
    ids[:, 32:] = 2
    ids[8:18, 8:18] = 0
    np.savez(gt / 'source_gt.npz', instance_map=ids, residual_unknown=ids == 0)
    (gt / 'source_class.json').write_text(json.dumps({1: 1, 2: 0}))
    manual = CanonicalBackendDataset(raw, gt, image_size=32, affinity_grid=16)
    np.savez(sam / 'mask.npz', instance_map=ids)
    (sam / 'manifest.jsonl').write_text(json.dumps(dict(source_sha256='synthetic',
        source_file=str(raw / 'source.png'), mask_file='mask.npz')) + '\n')
    pseudo = SAM2GeometryDataset(sam, raw, image_size=32, output_grid=16)
    return manual, pseudo, rgb, ids


def make_mixed(base, kind, count=8, epochs=2):
    rows = []
    for epoch in range(1, epochs + 1):
        m, p = [np.random.default_rng(seed + epoch).permutation(count) for seed in (10, 20)]
        rows += [dict(epoch=epoch, step=i + 1, manual_draw={'draw_index': [int(a)]},
            pseudo_draw={'draw_index': [int(b)]}) for i, (a, b) in enumerate(zip(m, p))]
    degradation = dict(profile_probabilities=[.2, .8, 0, 0], blur_sigma=[.4, 1.5],
        resample_probability=.5, resample_scale=[.65, 1.], spatial_blur=dict(
            enabled=True, probability=.5, grid_size=[3, 5], strength=1., levels=8))
    seed = 42 if kind == 'manual' else 59
    original = PairedDegradationDataset(base, degradation,
        dict(enabled=True, probability=.5, sigma=[0, .006]), seed, repeats=count)
    result = MixedViewDataset(original, kind,
        dict(mixed_view_seed=20260930, crop_attempts=32, minimum_known_pairs=16), rows)
    result.native.base_dataset.size = 16
    return result


def same_rng(a, b):
    assert a['python'] == b['python']
    assert a['numpy'][0] == b['numpy'][0] and np.array_equal(a['numpy'][1], b['numpy'][1])
    assert a['numpy'][2:] == b['numpy'][2:]
    assert torch.equal(a['torch_cpu'], b['torch_cpu'])
    assert len(a['torch_cuda']) == len(b['torch_cuda'])
    assert all(torch.equal(x, y) for x, y in zip(a['torch_cuda'], b['torch_cuda']))


@pytest.mark.parametrize('kind', ['manual', 'pseudo'])
def test_exact_main_pixels_targets_and_replay_with_auxiliary(cohort, kind):
    base = cohort[0 if kind == 'manual' else 1]
    control = make_mixed(base, kind)
    pair = PairViewDataset(make_mixed(base, kind), 101 if kind == 'manual' else 118)
    pair.auxiliary_native.base_dataset.size = 16
    before = capture_rng_state()
    fields = ('image', 'clean_image', 'instance_map', 'affinity_instance_map', 'foreground',
              'valid_content', 'affinity_valid_content', 'crop_box')
    for epoch in (1, 2):
        control.set_epoch(epoch)
        pair.set_epoch(epoch)
        for draw in range(8):
            expected, actual = control[draw], pair[draw]
            main, auxiliary = actual['main'], actual['auxiliary']
            assert draw_receipt(main) == draw_receipt(expected)
            assert main['training_view'] == expected['training_view']
            assert {main['training_view'], auxiliary['training_view']} == {'whole', 'native1024'}
            for key in fields:
                assert (key in main) == (key in expected)
                if key not in main:
                    continue
                assert torch.equal(main[key], expected[key]), (kind, epoch, draw, key)
                assert torch.equal(main[key], pair[draw]['main'][key]), key
            for key in ('name', 'image_path', 'source_index', 'draw_index'):
                assert main[key] == auxiliary[key]
            target, valid = build_affinity_targets_torch(auxiliary['instance_map'][None],
                auxiliary['valid_content'][None])
            assert not valid[0, :, auxiliary['instance_map'] == 0].any()
            assert valid.any()
            assert set(target[valid].tolist()) <= {0., 1.}
    same_rng(before, capture_rng_state())


@pytest.mark.parametrize('kind', ['manual', 'pseudo'])
def test_auxiliary_clean_coordinates_and_transform_are_same_native_source(cohort, kind):
    base, rgb, ids = cohort[0 if kind == 'manual' else 1], cohort[2], cohort[3]
    pair = PairViewDataset(make_mixed(base, kind), 101)
    pair.auxiliary_native.base_dataset.size = 16
    pair.set_epoch(1)
    for draw in range(8):
        auxiliary = pair[draw]['auxiliary']
        transform = tuple(auxiliary[k] for k in ('horizontal_flip', 'vertical_flip', 'rotation_k'))
        if auxiliary['training_view'] == 'native1024':
            y, x, y1, x1 = auxiliary['crop_box'].tolist()
            expected = geometry_sample(rgb[y:y1, x:x1], ids[y:y1, x:x1], image_size=32, grid=16)
            assert (y1 - y, x1 - x) == (16, 16)
        else:
            expected = base[0]
            assert auxiliary['crop_box'].tolist() == [0, 0, 48, 64]
        assert torch.equal(auxiliary['clean_image'], _spatial_transform(expected['image'], *transform))
        assert torch.equal(auxiliary.get('affinity_instance_map', auxiliary['instance_map']),
            _spatial_transform(expected.get('affinity_instance_map', expected['instance_map']), *transform))


def test_zero_does_not_construct_or_read_auxiliary(monkeypatch, cohort):
    mixed = make_mixed(cohort[0], 'manual')
    def forbidden(*args, **kwargs):
        raise AssertionError('alpha zero touched an auxiliary constructor')
    monkeypatch.setattr(pair_module, 'NativeViewDataset', forbidden)
    monkeypatch.setattr(pair_module, 'WholeViewDataset', forbidden)
    pair = PairViewDataset(mixed, 101, auxiliary_enabled=False)
    pair.set_epoch(1)
    assert pair.auxiliary_whole is None and pair.auxiliary_native is None
    for index in range(8):
        assert set(pair[index]) == {'main'}


def test_changing_auxiliary_seed_leaves_main_bitwise_fixed(cohort):
    pairs = [PairViewDataset(make_mixed(cohort[0], 'manual'), seed) for seed in (101, 202)]
    for pair in pairs:
        pair.auxiliary_native.base_dataset.size = 16
        pair.set_epoch(1)
    differences = 0
    for index in range(8):
        a, b = [pair[index] for pair in pairs]
        assert torch.equal(a['main']['image'], b['main']['image'])
        assert draw_receipt(a['main']) == draw_receipt(b['main'])
        differences += not torch.equal(a['auxiliary']['image'], b['auxiliary']['image'])
    assert differences > 0
    with pytest.raises(ValueError, match='independent'):
        PairViewDataset(make_mixed(cohort[0], 'manual'), 42)


def test_full_epoch_has_balanced_complements_and_same_manual_pseudo_views(cohort):
    pairs = [PairViewDataset(make_mixed(base, kind, count=64, epochs=1), seed)
        for base, kind, seed in ((cohort[0], 'manual', 101), (cohort[1], 'pseudo', 118))]
    for pair in pairs:
        pair.auxiliary_native.base_dataset.size = 16
        pair.set_epoch(1)
    # 回执已定义main draw->view，按对应源draw配对，不另写一个随机表。
    rows = pairs[0].main.receipts
    totals = {'whole': 0, 'native1024': 0}
    for row in rows:
        a, b = [pair[row[kind+'_draw']['draw_index'][0]]
                for pair, kind in zip(pairs, ('manual', 'pseudo'))]
        assert a['main']['training_view'] == b['main']['training_view']
        assert a['auxiliary']['training_view'] == b['auxiliary']['training_view']
        totals[a['auxiliary']['training_view']] += 1
    assert totals == {'whole': 32, 'native1024': 32}


def test_rng_isolation_restores_even_on_exception_and_checkpoint_roundtrip(tmp_path):
    initial = capture_rng_state()
    with pytest.raises(RuntimeError, match='intentional'):
        with isolated_random_stream(817):
            random.random(); np.random.random(5); torch.rand(5)
            raise RuntimeError('intentional')
    same_rng(initial, capture_rng_state())
    torch.save(dict(rng_state=initial), tmp_path/'state.pt')
    expected = (random.random(), np.random.random(), torch.rand(4))
    loaded = torch.load(tmp_path/'state.pt', weights_only=False)['rng_state']
    restore_rng_state(loaded)
    actual = (random.random(), np.random.random(), torch.rand(4))
    assert expected[:2] == actual[:2] and torch.equal(expected[2], actual[2])
    restore_rng_state(initial)


def test_normalized_source_weighting_has_one_equivalent_parameter_update():
    main_coef, aux_coef = branch_coefficients(.25)
    assert (main_coef, aux_coef) == (.8, .2)
    p = torch.nn.Parameter(torch.tensor([.3, -.2]))
    q = torch.nn.Parameter(p.detach().clone())
    main = (p - torch.tensor([1., 2.])).square().sum() + .5 * (p + 1).square().sum()
    (main * main_coef).backward()
    before = [p.grad.clone()]
    auxiliary = (p - torch.tensor([-.3, .4])).square().sum() + .5 * (p + 2).square().sum()
    (auxiliary * aux_coef).backward()
    metrics = gradient_diagnostics([p], before, main_coef, aux_coef)
    direct = (((q-torch.tensor([1., 2.])).square().sum()+.5*(q+1).square().sum())
              + .25*((q-torch.tensor([-.3, .4])).square().sum()+.5*(q+2).square().sum())) / 1.25
    direct.backward()
    assert torch.allclose(p.grad, q.grad)
    assert metrics['combined_parameter_norm'] == pytest.approx(float(q.grad.norm()))
    assert -1 <= metrics['parameter_gradient_cosine'] <= 1
    for parameter in (p, q):
        torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
        torch.optim.AdamW([parameter], lr=2e-5, eps=1e-4).step()
    assert torch.equal(p, q)


def test_unknown_padding_and_outside_endpoints_have_zero_loss_gradient():
    ids = np.ones((16, 32), np.uint16)
    ids[4:8, 8:12] = 0
    sample = geometry_sample(np.full((16, 32, 3), 150, np.uint8), ids, image_size=32, grid=16)
    target, valid = build_affinity_targets_torch(sample['instance_map'][None], sample['valid_content'][None])
    assert valid.any() and (target[valid] == 1).all()
    logits = torch.full_like(target, -.3, requires_grad=True)
    loss, _ = balanced_affinity_loss(logits, target, valid, negative_weight=1.)
    loss.backward()
    assert not logits.grad[~valid].any()
    assert not valid[0, :, sample['instance_map'] == 0].any()


def test_only_new_branch_config_is_removed_and_invalid_weights_rejected():
    config = load_config('config/train/affinity_pair.yaml')
    assert comparable(reference_config(config)) == comparable(load_config('config/train/affinity_balance_mixed.yaml'))
    assert config['backend_adaptation']['epochs'] == 20
    for weight in (True, .5, float('nan')):
        with pytest.raises(ValueError):
            validate_weight(weight)
    assert branch_coefficients(0) == (1., 0.)
    wrong = deepcopy(config); wrong['affinity_pair']['extra_loss'] = 1
    with pytest.raises(ValueError, match='Unknown'):
        reference_config(wrong)
    wrong = deepcopy(config); wrong['affinity_pair']['auxiliary_seed'] = config['backend_adaptation']['seed']
    with pytest.raises(ValueError, match='independent'):
        reference_config(wrong)


def test_receipts_reject_source_crop_or_complement_changes(cohort):
    pair = PairViewDataset(make_mixed(cohort[0], 'manual'), 101)
    pair.set_epoch(1)
    raw = next(iter(DataLoader(pair, batch_size=1, shuffle=False)))
    main = pair_receipt(raw['main'], raw['main'], 1, 1, 123)
    auxiliary = pair_receipt(raw['auxiliary'], raw['auxiliary'], 1, 1, 987)
    compare_primary(main, deepcopy(main)); compare_pair(main, auxiliary)
    wrong = deepcopy(main); wrong['manual_crop'][0][0] += 1
    with pytest.raises(RuntimeError, match='manual_crop'):
        compare_primary(wrong, main)
    wrong = deepcopy(auxiliary); wrong['manual_image_path'] = ['another.png']
    with pytest.raises(RuntimeError, match='path'):
        compare_pair(main, wrong)
    wrong = deepcopy(auxiliary); wrong['training_view'] = main['training_view']
    with pytest.raises(RuntimeError, match='complement'):
        compare_pair(main, wrong)


def test_launcher_source_manifest_contains_runtime_dependencies():
    from tools.run_affinity_pair import SOURCES
    for path in ('data/affinity_pair.py', 'train_affinity_pair.py', 'config/train/affinity_pair.yaml'):
        assert path in SOURCES
    assert all(Path(name).is_file() for name in SOURCES)


def test_diagnostic_gate_allows_finite_conflict_but_rejects_missing_updates():
    metrics = dict(main_weighted_parameter_norm=2., auxiliary_weighted_parameter_norm=.5,
                   combined_parameter_norm=1.8, parameter_gradient_cosine=-.4)
    rows = [dict(gradient_diagnostics=metrics.copy(), parameter_update_norm=.01,
                 grad_norm=1.8, head_state_sha256='real-parameter-summary') for _ in range(8)]
    assert verify_diagnostics(rows, .25)['parameter_cosine_mean'] == pytest.approx(-.4)
    wrong = deepcopy(rows); wrong[1]['parameter_update_norm'] = 0
    with pytest.raises(RuntimeError, match='update'):
        verify_diagnostics(wrong, .25)
    with pytest.raises(RuntimeError, match='Auxiliary'):
        verify_diagnostics(rows, 0)
    with pytest.raises(RuntimeError, match='Incomplete'):
        verify_diagnostics(rows[:7], .25)


def test_monitor_gate_checks_real_png_not_just_summary(tmp_path):
    config = dict(affinity_pair={'monitor_width': 30})
    folder = tmp_path/'monitor/epoch_005'; folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps(dict(epoch=5, images=[dict(source='example')])))
    with pytest.raises(RuntimeError, match='thumbnail'):
        verify_monitor(config, tmp_path, 5, 1)
    cv2.imwrite(str(folder/'example.png'), np.zeros((40, 120, 3), np.uint8))
    verify_monitor(config, tmp_path, 5, 1)
    cv2.imwrite(str(folder/'example.png'), np.zeros((40, 120), np.uint16))
    with pytest.raises(RuntimeError, match='thumbnail'):
        verify_monitor(config, tmp_path, 5, 1)


def zero_rows():
    return [dict(epoch=1, step=i+1, seed=100+i, manual=['a.png'], pseudo=['b.png'],
        manual_draw={'draw_index':[i], 'source_index':[0]},
        pseudo_draw={'draw_index':[i], 'source_index':[0]},
        manual_crop=[[0,0,16,16]], pseudo_crop=[[0,0,16,16]], crop_attempts=[[0],[0]],
        training_view='whole', auxiliary_weight=0., manual_bce=.3, pseudo_bce=.2, grad_norm=.6,
        supervision=[dict(positive_edges=1000, negative_edges=100, precision=.9,recall=.9,specificity=.8)]*2)
        for i in range(8)]


def repeatability_fixture():
    history = zero_rows()
    legacy = deepcopy(history)
    legacy[2]['pseudo_bce'] += 2e-6
    legacy[3]['grad_norm'] += 2e-5
    head = {'conv.weight':torch.ones(1024)*.01, 'last.bias':torch.zeros(2)}
    legacy_head = deepcopy(head)
    legacy_head['conv.weight'][0] += 1e-6
    return history, legacy, head, legacy_head


def test_bitwise_zero_path_and_small_measured_fallback_are_distinct():
    history, legacy, head, legacy_head = repeatability_fixture()
    exact = zero_difference(history,deepcopy(history),head,deepcopy(head))
    assert exact['exact'] and exact['supervision_counts_exact']
    observed = zero_difference(history,legacy,head,legacy_head)
    limits = repeatability_limits(observed)
    assert not observed['exact'] and limits['engineering_margin'] == 3
    assert limits['statistical_confidence_interval'] is False
    pair = deepcopy(history); pair[1]['manual_bce'] += 3e-6; pair[5]['grad_norm'] += 3e-5
    pair_head = deepcopy(head); pair_head['conv.weight'][0] += 3e-7
    verify_zero_numeric(zero_difference(history,pair,head,pair_head),limits)
    verify_zero_numeric(zero_difference(pair,legacy,pair_head,legacy_head),limits)
    with pytest.raises(RuntimeError,match='exact legacy'):
        repeatability_limits(exact)


@pytest.mark.parametrize('change', ['loss', 'gradient', 'weight_peak', 'weight_dense'])
def test_numerical_fallback_still_rejects_changed_training(change):
    history, legacy, head, legacy_head = repeatability_fixture()
    limits = repeatability_limits(zero_difference(history,legacy,head,legacy_head))
    pair, pair_head = deepcopy(history), deepcopy(head)
    if change == 'loss':
        pair[1]['manual_bce'] += 2e-5
    elif change == 'gradient':
        pair[1]['grad_norm'] += .001
    elif change == 'weight_peak':
        pair_head['conv.weight'][0] += 1e-5
    else:
        # 最大偏差仍小于单点包络，但整层一起漂移会被RMS拒绝。
        pair_head['conv.weight'] += 1e-6
    difference = zero_difference(history,pair,head,pair_head)
    with pytest.raises(RuntimeError,match='exceeds'):
        verify_zero_numeric(difference,limits)


def test_repeatability_cannot_open_caps_using_bad_legacy():
    history, legacy, head, legacy_head = repeatability_fixture()
    legacy[0]['pseudo_bce'] += .01
    with pytest.raises(RuntimeError,match='fixed absolute caps'):
        repeatability_limits(zero_difference(history,legacy,head,legacy_head))


@pytest.mark.parametrize('change',['source','crop','count','nan','dtype'])
def test_zero_input_counts_and_finiteness_always_remain_exact(change):
    rows, head = zero_rows(), {'conv.weight':torch.ones(4)}
    pair, pair_head = deepcopy(rows), deepcopy(head)
    if change == 'source':
        pair[0]['manual'] = ['different.png']
    elif change == 'crop':
        pair[0]['manual_crop'][0][0] += 1
    elif change == 'count':
        pair[0]['supervision'][0]['positive_edges'] += 1
    elif change == 'nan':
        pair[0]['grad_norm'] = float('nan')
    else:
        pair_head['conv.weight'] = pair_head['conv.weight'].double()
    with pytest.raises(RuntimeError):
        zero_difference(rows,pair,head,pair_head)
