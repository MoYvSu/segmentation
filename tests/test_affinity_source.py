# -*- coding: utf-8 -*-
"""来源比例包装的CPU检查：原公式、反传/裁剪/AdamW、异常恢复及保存合同。"""
from copy import deepcopy
import json
from pathlib import Path
import random
import threading

import cv2
import numpy as np
import pytest
import torch

import train_affinity_source as source
from train_affinity_balance import comparable
from train_backend_adaptation import sha, tensor_digest
from utils.affinity_loss import balanced_affinity_loss
from utils.config import load_config


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf8')


def write_rows(path, rows):
    Path(path).write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf8')


@pytest.mark.parametrize('beta,manual,pseudo', [(.25, 1.2, .3), (.5, 1., .5)])
def test_single_variable_coefficients_and_original_initialization(beta, manual, pseudo):
    cfg = load_config('config/train/affinity_source_p025.yaml')
    cfg['backend_adaptation']['pseudo_weight'] = beta
    original = load_config('config/train/affinity_balance_mixed.yaml')
    assert comparable(source.reference_config(cfg)) == comparable(original)
    coefficients = source.source_coefficients(cfg)
    assert coefficients['manual_coefficient'] == manual and coefficients['pseudo_coefficient'] == pseudo
    assert coefficients['nominal_coefficient_sum'] == 1.5
    for key in ('sam2', 'affinity_geometry_g1', 'semantic_challenger', 'affinity_deployment'):
        assert cfg[key] == original[key]
    assert cfg['affinity_deployment']['checkpoint'] != source.REFERENCE + '/final.pt'
    assert cfg['backend_adaptation']['pseudo_weight'] == beta and 'affinity_source' in cfg


@pytest.mark.parametrize('beta', [0., 1., -.25, True, float('nan')])
def test_unauthorized_source_ratios_rejected(beta):
    cfg = load_config('config/train/affinity_source_p025.yaml')
    cfg['backend_adaptation']['pseudo_weight'] = beta
    with pytest.raises(ValueError):
        source.reference_config(cfg)


@pytest.mark.parametrize('key,value', [('negative_weight', .75), ('hard_negative_weight', 0.), ('hard_negative_gamma', 3.)])
def test_other_supervision_variables_cannot_change(key, value):
    cfg = load_config('config/train/affinity_source_p025.yaml')
    cfg['direct_semantic_affinity']['affinity_loss'][key] = value
    with pytest.raises(ValueError):
        source.reference_config(cfg)


def test_registered_control_and_fixed_budget_cannot_change():
    cfg = load_config('config/train/affinity_source_p025.yaml')
    bad = deepcopy(cfg); bad['affinity_source']['reference'] = 'outputs/affinity_hard/h0'
    with pytest.raises(ValueError): source.reference_config(bad)
    bad = deepcopy(cfg); bad['affinity_source']['alpha'] = 9.
    with pytest.raises(ValueError): source.reference_config(bad)
    bad = deepcopy(cfg); bad['backend_adaptation']['epochs'] = 40
    with pytest.raises(ValueError): source.reference_config(bad)


@pytest.mark.parametrize('beta', [.5, .25])
def test_loss_hook_matches_explicit_objective_through_clip_and_two_adamw_updates(tmp_path, monkeypatch, beta):
    """比较真实BCE图的两次更新，observer不改梯度，未知位置保持零。"""
    cfg = load_config('config/train/affinity_source_p025.yaml')
    cfg['backend_adaptation']['pseudo_weight'] = beta
    weights = source.source_coefficients(cfg)
    torch.manual_seed(17)
    initial = torch.nn.Linear(2, 1)
    def compute(logits, batch, loss_cfg, *, pseudo):
        return balanced_affinity_loss(logits, batch['target'], batch['valid'], negative_weight=1.,
                                      hard_negative_weight=1., hard_negative_gamma=2.)
    monkeypatch.setattr(source.native, 'compute_affinity_loss', compute)
    captured = []
    def run(hooked):
        model = deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.e-5, weight_decay=1.e-4, eps=1.e-4)
        logs, raw_gradients, clipped_gradients, norms = [], [], [], []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for pseudo, inputs, target in [
                (False, [[1., 0.], [0., 1.], [1., 1.], [9., 9.]], [1., 0., 1., 0.]),
                (True, [[-1., 1.], [1., -1.], [2., 1.], [9., 9.]], [0., 1., 0., 1.])]:
                logits = model(torch.tensor(inputs)).reshape(1, 1, 1, 4)
                logits.retain_grad()
                batch = dict(name=['sam2.png' if pseudo else 'manual.png'],
                             target=torch.tensor(target).reshape(1, 1, 1, 4),
                             valid=torch.tensor([True, True, True, False]).reshape(1, 1, 1, 4))
                loss, details = source.native.compute_affinity_loss(logits, batch, {}, pseudo=pseudo)
                logs.append(float(loss.detach()))
                multiplier = beta if pseudo else 1.
                if not hooked: multiplier *= weights['alpha']
                (loss * multiplier).backward()
                assert logits.grad[..., -1].item() == 0.
                assert details['positive_edges'] + details['negative_edges'] == 3
            raw_gradients.append([p.grad.clone() for p in model.parameters()])
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), .1)))
            clipped_gradients.append([p.grad.clone() for p in model.parameters()])
            optimizer.step()
        return raw_gradients, clipped_gradients, [p.detach().clone() for p in model.parameters()], logs, norms
    explicit = run(False)
    with source.source_loss_scaling(cfg, tmp_path) as counter:
        hooked = run(True)
        captured.append(counter.copy())
    assert source.native.compute_affinity_loss is compute
    assert captured == [{'calls': 4, 'backwards': 4}]
    for left, right in zip(explicit[:2], hooked[:2]):
        for a, b in zip(left, right):
            assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert all(torch.equal(a, b) for a, b in zip(explicit[2], hooked[2]))
    assert explicit[3:] == hooked[3:] and all(n > .1 for n in hooked[4])
    receipts = source._rows(tmp_path / source.SIDECAR)
    assert [r['raw_bce'] for r in receipts] == hooked[3]
    assert [r['source_coefficient'] for r in receipts] == [weights['manual_coefficient'], weights['pseudo_coefficient']] * 2
    assert all(r['weighted_logits_grad_norm'] == r['raw_logits_grad_norm'] * r['source_coefficient']
               or np.isclose(r['weighted_logits_grad_norm'], r['raw_logits_grad_norm'] * r['source_coefficient']) for r in receipts)
    assert all('not parameter gradient' in r['scope'] for r in receipts)


def test_function_restored_on_exception_and_nested_context_rejected(tmp_path):
    cfg = load_config('config/train/affinity_source_p025.yaml')
    original = source.native.compute_affinity_loss
    with pytest.raises(RuntimeError, match='failure'):
        with source.source_loss_scaling(cfg, tmp_path):
            with pytest.raises(RuntimeError, match='Nested'):
                with source.source_loss_scaling(cfg, tmp_path): pass
            raise RuntimeError('failure')
    assert source.native.compute_affinity_loss is original


def test_scope_cannot_modify_trainer_from_a_worker_thread(tmp_path):
    cfg = load_config('config/train/affinity_source_p025.yaml')
    errors = []
    original = source.native.compute_affinity_loss
    def worker():
        try:
            with source.source_loss_scaling(cfg, tmp_path): pass
        except RuntimeError as error: errors.append(str(error))
    thread = threading.Thread(target=worker); thread.start(); thread.join()
    assert errors and 'isolated training process' in errors[0]
    assert source.native.compute_affinity_loss is original


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    """合成回执/微小checkpoint，仅验证保存与核验，不调用GPU或赛题原图。"""
    cfg = load_config('config/train/affinity_source_p025.yaml')
    cfg['paths']['project_root'] = str(tmp_path)
    reference = tmp_path / source.REFERENCE
    reference.mkdir(parents=True)
    (reference / 'final.pt').write_bytes(b'synthetic-reference')
    actual_sha = source.sha
    monkeypatch.setattr(source, 'sha', lambda path: source.REFERENCE_SHA256
                        if Path(path) == reference / 'final.pt' else actual_sha(path))
    ancestor_calls = []
    monkeypatch.setattr(source, 'verify_balance_reference', lambda c, state=None: ancestor_calls.append((deepcopy(c), state)))
    rows = []
    for index in range(1280):
        epoch, step = index // 64 + 1, index % 64 + 1
        rows.append(dict(epoch=epoch, step=step, updates=index + 1,
            seed=cfg['backend_adaptation']['seed'] + epoch * 10000 + step - 1,
            manual=['manual.png'], pseudo=['sam2.png'], manual_draw={'index': index}, pseudo_draw={'index': index},
            manual_crop=[[0, 0, 1024, 1024]], pseudo_crop=[[1, 2, 1025, 1026]], crop_attempts=[[0], [0]],
            training_view='whole' if step <= 32 else 'native1024', manual_bce=.5, pseudo_bce=.6,
            grad_norm=.2, auxiliary_weight=0., supervision=[
                dict(positive_edges=100, negative_edges=20, precision=.9, recall=.8, specificity=.7),
                dict(positive_edges=80, negative_edges=10, precision=.8, recall=.9, specificity=.8)]))
    write_rows(reference / 'steps.jsonl', rows)
    tensors = {'encoder.synthetic': torch.tensor([3., 4.]), 'affinity_decoder.synthetic': torch.tensor([1., 2.])}
    base = dict(status='complete', config=source.reference_config(cfg), epoch=20, updates=1280,
        smoke=False, size=1024, precision='FP32', initial_state_sha256='a' * 64,
        frozen_state_sha256=tensor_digest([('encoder.synthetic', tensors['encoder.synthetic'])]),
        restorer_state_sha256='b' * 64, data={'parent_data': {'manual_sources': 32, 'sam2_sources': 64}},
        **{key: True for key in source.AUDIT_FLAGS})
    write(reference / 'status.json', base)
    def create(updates=8, beta=.25):
        candidate = deepcopy(cfg); candidate['backend_adaptation']['pseudo_weight'] = beta
        output = tmp_path / ('run_' + str(updates) + '_' + str(beta)); output.mkdir()
        epoch = 1 if updates == 8 else 20
        state = dict(base, config=candidate, epoch=epoch, updates=updates, smoke=updates == 8,
                     final_state_sha256=tensor_digest(tensors.items()))
        bundle = dict(format=source.FUSED_DEPLOYMENT_FORMAT, config=candidate, epoch=epoch,
                      model_state_dict=deepcopy(tensors), extra={'updates': updates, 'native_size': 1024, 'smoke': updates == 8})
        torch.save(bundle, output / 'final.pt')
        state['final_checkpoint_sha256'] = sha(output / 'final.pt')
        torch.save(dict(format='affinity_native_head_v1', epoch=epoch, size=1024, config=candidate,
            geometry_state_dict={'synthetic': tensors['affinity_decoder.synthetic'].clone()},
            optimizer={'state': {}, 'param_groups': [{'lr': 2.e-6}]}, scheduler={'last_epoch': epoch},
            **source.checkpoint_extras(candidate, epoch, smoke=updates == 8)), output / 'last_head.pt')
        write(output / 'status.json', state)
        write_rows(output / 'steps.jsonl', rows[:updates])
        weights = source.source_coefficients(candidate)
        receipts = []
        for i, row in enumerate(rows[:updates]):
            for pseudo in (False, True):
                key = 'pseudo' if pseudo else 'manual'
                coefficient = weights['pseudo_coefficient'] if pseudo else weights['manual_coefficient']
                receipts.append(dict(call=2 * i + int(pseudo) + 1, epoch=row['epoch'], step=row['step'], update=row['updates'],
                    source_kind='sam2' if pseudo else 'manual', source=row[key], raw_bce=row[key + '_bce'],
                    source_coefficient=coefficient, **weights, raw_logits_grad_norm=.1,
                    weighted_logits_grad_norm=.1 * coefficient))
        write_rows(output / source.SIDECAR, receipts)
        epochs = (0, 1) if updates == 8 else (0, 1, 5, 10, 15, 20)
        names = candidate['backend_adaptation']['monitor']['images'][:2] if updates == 8 else candidate['backend_adaptation']['monitor']['images']
        for e in epochs:
            folder = output / 'monitor' / f'epoch_{e:03d}'
            write(folder / 'summary.json', {'epoch': e, 'images': [{'source': name} for name in names]})
            for name in names:
                for suffix in ('full', 'detail', 'boundary'):
                    assert cv2.imwrite(str(folder / f'{name}_{suffix}.png'), np.zeros((3, 4, 3), np.uint8))
        for e in epochs[1:]:
            for step in (1, 2):
                folder = output / 'draws' / f'epoch_{e:03d}'
                row = rows[(e - 1) * 64 + step - 1]
                write(folder / f'draw_{step:03d}.json', {'source': row['manual'], 'crop_box': row['manual_crop']})
                assert cv2.imwrite(str(folder / f'draw_{step:03d}.png'), np.zeros((282, 1024, 3), np.uint8))
        return candidate, output
    return cfg, reference, ancestor_calls, create


def test_ancestor_reference_is_retained_and_hidden_mutations_rejected(run_fixture):
    cfg, _, calls, _ = run_fixture
    source.verify_reference(cfg)
    assert calls[0][0]['backend_adaptation']['pseudo_weight'] == .5
    for field, value in [('normalize_edge_weights', False), ('pseudo_negative_weight', .25)]:
        changed = deepcopy(cfg); changed['direct_semantic_affinity']['affinity_loss'][field] = value
        with pytest.raises(RuntimeError, match='configuration changed'): source.verify_reference(changed)
    changed = deepcopy(cfg); changed['backend_adaptation']['learning_rates']['affinity'] *= 1.2
    with pytest.raises(RuntimeError, match='configuration changed'): source.verify_reference(changed)


@pytest.mark.parametrize('updates,beta', [(8, .25), (1280, .25), (8, .5)])
def test_complete_short_formal_and_technical_control(run_fixture, updates, beta):
    _, _, _, create = run_fixture
    cfg, output = create(updates, beta)
    result = source.verify_run(cfg, output, updates)
    assert result['passed'] and result['control_retrained'] is False
    assert result['weighted_logits_receipts'] == 2 * updates and result['raw_bce_logs_unscaled']
    assert result['continuation_state_saved'] and result['head_bundle_exact'] and result['process_monitor_complete']
    assert result['technical_control'] == (beta == .5)


@pytest.mark.parametrize('key', source.DEPENDENCY_KEYS)
def test_initialization_frozen_data_dependencies_are_exact(run_fixture, key):
    _, _, _, create = run_fixture
    cfg, output = create()
    state = source.read(output / 'status.json'); state[key] = {} if key == 'data' else 'c' * 64
    write(output / 'status.json', state)
    with pytest.raises(RuntimeError, match='dependency or data differs'): source.verify_run(cfg, output, 8)


@pytest.mark.parametrize('mutation,match', [('metadata', 'continuation metadata'), ('rng', 'random state'),
    ('sidecar', 'actual draw/loss/coefficient'), ('missing_sidecar', 'Missing or extra'),
    ('unknown_norm', 'Invalid weighted'), ('crop', 'Sampling/crop/view')])
def test_saved_coefficients_rng_and_gradient_receipts_are_checked(run_fixture, mutation, match):
    _, _, _, create = run_fixture
    cfg, output = create()
    if mutation in ('metadata', 'rng'):
        head = torch.load(output / 'last_head.pt', weights_only=False)
        if mutation == 'metadata': head['source_state']['manual_coefficient'] = 1.
        else: head['source_state']['rng'].pop('torch_cpu')
        torch.save(head, output / 'last_head.pt')
    elif mutation == 'crop':
        steps = source._rows(output / 'steps.jsonl'); steps[0]['manual_crop'][0][0] = 1
        write_rows(output / 'steps.jsonl', steps)
    else:
        receipts = source._rows(output / source.SIDECAR)
        if mutation == 'sidecar': receipts[0]['raw_bce'] *= 1.2
        elif mutation == 'missing_sidecar': receipts.pop()
        else: receipts[0]['weighted_logits_grad_norm'] = float('inf')
        write_rows(output / source.SIDECAR, receipts)
    with pytest.raises(RuntimeError, match=match): source.verify_run(cfg, output, 8)


def test_technical_control_cannot_repeat_formal_training():
    cfg = load_config('config/train/affinity_source_p025.yaml'); cfg['backend_adaptation']['pseudo_weight'] = .5
    with pytest.raises(ValueError, match='short technical'): source.train(cfg, 'unused', smoke=False)
    with pytest.raises(ValueError, match='eight-update'): source.verify_run(cfg, 'unused', 1280)


def test_checkpoint_capture_does_not_consume_rng():
    cfg = load_config('config/train/affinity_source_p025.yaml')
    random.seed(42); np.random.seed(43); torch.manual_seed(44)
    before = source.capture_rng_state()
    extra = source.checkpoint_extras(cfg, 5)['source_state']; source._verify_rng(extra['rng'])
    after = source.capture_rng_state()
    assert extra['updates'] == 320 and extra['manual_coefficient'] == 1.2 and extra['pseudo_coefficient'] == .3
    assert before['python'] == after['python'] and np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:] and torch.equal(before['torch_cpu'], after['torch_cpu'])


def test_train_wrapper_preserves_native_options_and_restores_function(tmp_path, monkeypatch):
    cfg = load_config('config/train/affinity_source_p025.yaml')
    original = source.native.compute_affinity_loss
    calls = []
    monkeypatch.setattr(source, 'verify_reference', lambda c: None)
    monkeypatch.setattr(source, 'verify_run', lambda c, out, updates: {'passed': True, 'updates': updates})
    def compute(logits, batch, loss_cfg, *, pseudo):
        return logits.square().mean(), {}
    monkeypatch.setattr(source.native, 'compute_affinity_loss', compute)
    def native_train(config, size, output, **kwargs):
        calls.append((config, size, kwargs)); Path(output).mkdir()
        for _ in range(8):
            for pseudo in (False, True):
                logits = torch.ones((1, 1, 1, 2), requires_grad=True)
                loss, _ = source.native.compute_affinity_loss(logits, {'name': ['source.png']}, {}, pseudo=pseudo)
                (loss * (config['backend_adaptation']['pseudo_weight'] if pseudo else 1.)).backward()
    monkeypatch.setattr(source.native, 'train', native_train)
    result = source.train(cfg, tmp_path / 'run', smoke=True)
    assert result['updates'] == 8 and source.native.compute_affinity_loss is compute
    assert len(calls) == 1 and calls[0][0] is cfg and calls[0][1] == 1024
    options = calls[0][2]
    assert options['dataset_builder'] is source.mixed_data and options['control_verifier'] is source.verify_reference
    assert options['keep_smoke_checkpoint'] is True
    assert options['checkpoint_extras'](1)['source_state']['manual_coefficient'] == 1.2
    assert options['monitor_saver'].keywords['reference_folder'].replace('\\', '/').endswith('affinity_balance/mixed/deployment/patch1024')
    # 当前测试monkeypatch清理时恢复真正的原函数，包装器本身已先恢复进入时的函数。
    assert original is not source.native.compute_affinity_loss
