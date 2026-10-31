# -*- coding: utf-8 -*-
"""P100 CPU检查：单变量隔离、真实BCE反传及独立保存/审计合同。"""
from copy import deepcopy
import json
from pathlib import Path
import random
import threading

import cv2
import numpy as np
import pytest
import torch

import train_affinity_p100 as p100
import train_affinity_source as previous
from train_affinity_balance import comparable
from train_backend_adaptation import sha, tensor_digest
from utils.affinity_loss import balanced_affinity_loss
from utils.config import load_config


def configuration():
    return load_config(p100.CONFIG_PATH)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf8')


def write_rows(path, rows):
    Path(path).write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf8')


def test_coefficients_initialization_and_previous_namespace_are_preserved():
    cfg = configuration()
    assert comparable(p100.reference_config(cfg)) == comparable(load_config('config/train/affinity_balance_mixed.yaml'))
    assert p100.verify_config(cfg) == dict(format='affinity_source_p100_v1', beta=1., alpha=.75,
        manual_coefficient=.75, pseudo_coefficient=.75, nominal_coefficient_sum=1.5,
        reference_beta=.5, reference_sha256=p100.REFERENCE_SHA256, control_retrained=False)
    old = load_config('config/train/affinity_source_p025.yaml')
    assert previous.SOURCE_FORMAT == 'affinity_source_ratio_v1'
    assert previous.source_coefficients(old)['manual_coefficient'] == 1.2
    assert previous.source_coefficients(old)['pseudo_coefficient'] == .3
    with pytest.raises(ValueError): previous.source_coefficients(cfg)
    assert p100.verify_run.__globals__ is p100._core.__dict__
    assert previous.verify_run.__globals__ is previous.__dict__
    assert previous.native.compute_affinity_loss is p100.native.compute_affinity_loss
    assert p100.SOURCE_IMPLEMENTATION_SHA256 == sha('train_affinity_source.py')
    for key in ('sam2', 'affinity_geometry_g1', 'semantic_challenger', 'affinity_deployment'):
        assert cfg[key] == old[key]
    assert cfg['affinity_deployment']['checkpoint'] != p100.REFERENCE + '/final.pt'


@pytest.mark.parametrize('beta', [0., .25, .5, -1., 2., True, '1', None, float('nan'), float('inf')])
def test_only_numeric_p100_ratio_allowed(beta):
    cfg = configuration(); cfg['backend_adaptation']['pseudo_weight'] = beta
    with pytest.raises(ValueError, match='numeric beta=1'): p100.verify_config(cfg)


@pytest.mark.parametrize('mutation', ['missing', 'options', 'reference', 'sha', 'r', 'h', 'gamma',
    'epochs', 'repeats', 'sampling', 'lr', 'initialization', 'unknown_loss', 'bool_r', 'bool_epochs'])
def test_hidden_variables_and_malformed_config_rejected(mutation):
    cfg = configuration()
    loss = cfg['direct_semantic_affinity']['affinity_loss']
    if mutation == 'missing': cfg.pop('direct_semantic_affinity')
    elif mutation == 'options': cfg['affinity_source']['alpha'] = .75
    elif mutation == 'reference': cfg['affinity_source']['reference'] = 'outputs/affinity_hard/h0'
    elif mutation == 'sha': cfg['affinity_source']['reference_sha256'] = 'a' * 64
    elif mutation in ('r', 'h', 'gamma', 'bool_r'):
        key = {'r': 'negative_weight', 'h': 'hard_negative_weight', 'gamma': 'hard_negative_gamma',
               'bool_r': 'negative_weight'}[mutation]
        loss[key] = True if mutation == 'bool_r' else .75
    elif mutation in ('epochs', 'bool_epochs'): cfg['backend_adaptation']['epochs'] = True if mutation == 'bool_epochs' else 21
    elif mutation == 'repeats': cfg['backend_adaptation']['manual_repeats'] = 1
    elif mutation == 'sampling': cfg['affinity_native']['penalty_sampling'] = 'native'
    elif mutation == 'lr': cfg['backend_adaptation']['learning_rates']['affinity'] *= 1.2
    elif mutation == 'initialization': cfg['affinity_deployment']['checkpoint'] = p100.REFERENCE + '/final.pt'
    else: loss['pseudo_negative_weight'] = .25
    with pytest.raises(ValueError): p100.verify_config(cfg)


def test_eight_real_bce_updates_equal_explicit_objective_and_unknowns_ignored(tmp_path, monkeypatch):
    """对比裁剪前/后梯度、AdamW权重及原始日志，不用镜像实现的平方损失。"""
    cfg = configuration(); torch.manual_seed(17)
    initial = torch.nn.Linear(2, 1)
    def compute(logits, batch, loss_cfg, *, pseudo):
        return balanced_affinity_loss(logits, batch['target'], batch['valid'], negative_weight=1.,
            hard_negative_weight=1., hard_negative_gamma=2.)
    monkeypatch.setattr(p100.native, 'compute_affinity_loss', compute)
    def run(hooked):
        model = deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.e-5, weight_decay=1.e-4, eps=1.e-4)
        logs, raw, clipped, norms = [], [], [], []
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            for pseudo, inputs, target in [
                (False, [[1., 0.], [0., 1.], [1., 1.], [9., 9.]], [1., 0., 1., 0.]),
                (True, [[-1., 1.], [1., -1.], [2., 1.], [9., 9.]], [0., 1., 0., 1.])]:
                logits = model(torch.tensor(inputs)).reshape(1, 1, 1, 4); logits.retain_grad()
                batch = dict(name=['sam2.png' if pseudo else 'manual.png'],
                    target=torch.tensor(target).reshape(1, 1, 1, 4),
                    valid=torch.tensor([True, True, True, False]).reshape(1, 1, 1, 4))
                loss, details = p100.native.compute_affinity_loss(logits, batch, {}, pseudo=pseudo)
                logs.append(float(loss.detach()))
                # 既有trainer在pseudo分支乘beta=1；hook再乘alpha=.75。
                (loss * (1. if hooked else .75)).backward()
                assert logits.grad[..., -1].item() == 0.
                assert details['positive_edges'] + details['negative_edges'] == 3
            raw.append([parameter.grad.clone() for parameter in model.parameters()])
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), .1)))
            clipped.append([parameter.grad.clone() for parameter in model.parameters()]); optimizer.step()
        return raw, clipped, [parameter.detach().clone() for parameter in model.parameters()], logs, norms
    explicit = run(False)
    with p100.source_loss_scaling(cfg, tmp_path) as counter:
        hooked = run(True)
    assert p100.native.compute_affinity_loss is compute and counter == {'calls': 16, 'backwards': 16}
    for left, right in zip(explicit[:2], hooked[:2]):
        for actual, expected in zip(left, right):
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(torch.equal(a, b) for a, b in zip(explicit[2], hooked[2]))
    assert explicit[3:] == hooked[3:] and all(value > .1 for value in hooked[4])
    receipts = p100._core._rows(tmp_path / p100.SIDECAR)
    assert [row['raw_bce'] for row in receipts] == hooked[3]
    assert [row['source_kind'] for row in receipts] == ['manual', 'sam2'] * 8
    assert all(row['source_coefficient'] == .75 and np.isclose(row['weighted_logits_grad_norm'],
               .75 * row['raw_logits_grad_norm']) for row in receipts)


def test_exception_nested_and_thread_context_restore_native_function(tmp_path):
    cfg = configuration(); original = p100.native.compute_affinity_loss
    with pytest.raises(RuntimeError, match='failure'):
        with p100.source_loss_scaling(cfg, tmp_path):
            with pytest.raises(RuntimeError, match='Nested'):
                with p100.source_loss_scaling(cfg, tmp_path): pass
            raise RuntimeError('failure')
    assert p100.native.compute_affinity_loss is original
    errors = []
    def worker():
        try:
            with p100.source_loss_scaling(cfg, tmp_path): pass
        except RuntimeError as error: errors.append(str(error))
    thread = threading.Thread(target=worker); thread.start(); thread.join()
    assert len(errors) == 1 and 'isolated training process' in errors[0]
    assert p100.native.compute_affinity_loss is original


def test_checkpoint_metadata_identity_and_rng_are_preserved():
    random.seed(42); np.random.seed(43); torch.manual_seed(44)
    before = p100._core.capture_rng_state()
    extra = p100.checkpoint_extras(configuration(), 5)['source_state']
    p100._core._verify_rng(extra['rng'])
    after = p100._core.capture_rng_state()
    assert extra['format'] == p100.SOURCE_FORMAT and extra['epoch'] == 5 and extra['updates'] == 320
    assert extra['manual_coefficient'] == extra['pseudo_coefficient'] == .75
    assert before['python'] == after['python'] and np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:] and torch.equal(before['torch_cpu'], after['torch_cpu'])


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    """小型合成checkpoint和完整1280回执，不使用私有图像或GPU。"""
    core = p100._core
    cfg = configuration(); cfg['paths']['project_root'] = str(tmp_path)
    reference = tmp_path / p100.REFERENCE; reference.mkdir(parents=True)
    (reference / 'final.pt').write_bytes(b'synthetic-reference')
    actual_sha = core.sha
    monkeypatch.setattr(core, 'sha', lambda path: p100.REFERENCE_SHA256 if Path(path) == reference / 'final.pt' else actual_sha(path))
    monkeypatch.setattr(core, 'verify_balance_reference', lambda c, state=None: None)
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
    base = dict(status='complete', config=p100.reference_config(cfg), epoch=20, updates=1280,
        smoke=False, size=1024, precision='FP32', initial_state_sha256='a' * 64,
        frozen_state_sha256=tensor_digest([('encoder.synthetic', tensors['encoder.synthetic'])]),
        restorer_state_sha256='b' * 64, data={'parent_data': {'manual_sources': 32, 'sam2_sources': 64}},
        **{key: True for key in core.AUDIT_FLAGS})
    write(reference / 'status.json', base)
    def create(updates=8):
        output = tmp_path / ('run_' + str(updates)); output.mkdir()
        epoch = 1 if updates == 8 else 20
        state = dict(base, config=cfg, epoch=epoch, updates=updates, smoke=updates == 8,
            final_state_sha256=tensor_digest(tensors.items()))
        bundle = dict(format=core.FUSED_DEPLOYMENT_FORMAT, config=cfg, epoch=epoch,
            model_state_dict=deepcopy(tensors), extra={'updates': updates, 'native_size': 1024, 'smoke': updates == 8})
        torch.save(bundle, output / 'final.pt'); state['final_checkpoint_sha256'] = sha(output / 'final.pt')
        torch.save(dict(format='affinity_native_head_v1', epoch=epoch, size=1024, config=cfg,
            geometry_state_dict={'synthetic': tensors['affinity_decoder.synthetic'].clone()},
            optimizer={'state': {}, 'param_groups': [{'lr': 2.e-6}]}, scheduler={'last_epoch': epoch},
            **p100.checkpoint_extras(cfg, epoch, smoke=updates == 8)), output / 'last_head.pt')
        write(output / 'status.json', state); write_rows(output / 'steps.jsonl', rows[:updates])
        weights = p100.source_coefficients(cfg); receipts = []
        for index, row in enumerate(rows[:updates]):
            for pseudo in (False, True):
                key = 'pseudo' if pseudo else 'manual'
                receipts.append(dict(call=2 * index + int(pseudo) + 1, epoch=row['epoch'], step=row['step'], update=row['updates'],
                    source_kind='sam2' if pseudo else 'manual', source=row[key], raw_bce=row[key + '_bce'],
                    source_coefficient=.75, **weights, raw_logits_grad_norm=.1, weighted_logits_grad_norm=.075))
        write_rows(output / p100.SIDECAR, receipts)
        epochs = (0, 1) if updates == 8 else (0, 1, 5, 10, 15, 20)
        names = cfg['backend_adaptation']['monitor']['images'][:2] if updates == 8 else cfg['backend_adaptation']['monitor']['images']
        for epoch in epochs:
            folder = output / 'monitor' / f'epoch_{epoch:03d}'
            write(folder / 'summary.json', {'epoch': epoch, 'images': [{'source': name} for name in names]})
            for name in names:
                for suffix in ('full', 'detail', 'boundary'):
                    assert cv2.imwrite(str(folder / f'{name}_{suffix}.png'), np.zeros((3, 4, 3), np.uint8))
        for epoch in epochs[1:]:
            for step in (1, 2):
                folder = output / 'draws' / f'epoch_{epoch:03d}'; row = rows[(epoch - 1) * 64 + step - 1]
                write(folder / f'draw_{step:03d}.json', {'source': row['manual'], 'crop_box': row['manual_crop']})
                assert cv2.imwrite(str(folder / f'draw_{step:03d}.png'), np.zeros((282, 1024, 3), np.uint8))
        return cfg, output
    return create


@pytest.mark.parametrize('updates', [8, 1280])
def test_short_and_full_saved_receipts_are_identified_as_p100(run_fixture, updates):
    cfg, output = run_fixture(updates)
    result = p100.verify_run(cfg, output, updates)
    assert result['format'] == p100.SOURCE_FORMAT and result['beta'] == 1.
    assert result['manual_coefficient'] == result['pseudo_coefficient'] == .75
    assert result['passed'] and result['control_retrained'] is False and result['technical_control'] is False
    assert result['weighted_logits_receipts'] == 2 * updates and result['raw_bce_logs_unscaled']
    assert result['continuation_state_saved'] and result['head_bundle_exact'] and result['process_monitor_complete']


@pytest.mark.parametrize('mutation,match', [('metadata', 'continuation metadata'), ('rng', 'random state'),
    ('receipt', 'actual draw/loss/coefficient'), ('crop', 'Sampling/crop/view'), ('frozen', 'dependency or data differs')])
def test_old_identity_wrong_coefficients_unknown_rng_or_sampling_fail(run_fixture, mutation, match):
    cfg, output = run_fixture()
    if mutation in ('metadata', 'rng'):
        head = torch.load(output / 'last_head.pt', weights_only=False)
        if mutation == 'metadata': head['source_state']['format'] = previous.SOURCE_FORMAT
        else: head['source_state']['rng'].pop('torch_cpu')
        torch.save(head, output / 'last_head.pt')
    elif mutation == 'receipt':
        receipts = p100._core._rows(output / p100.SIDECAR); receipts[0]['source_coefficient'] = 1.2
        write_rows(output / p100.SIDECAR, receipts)
    elif mutation == 'crop':
        steps = p100._core._rows(output / 'steps.jsonl'); steps[0]['manual_crop'][0][0] = 1
        write_rows(output / 'steps.jsonl', steps)
    else:
        state = p100._core.read(output / 'status.json'); state['frozen_state_sha256'] = 'c' * 64
        write(output / 'status.json', state)
    with pytest.raises(RuntimeError, match=match): p100.verify_run(cfg, output, 8)


def test_train_adapter_keeps_frozen_native_monitor_and_checkpoint_contract(tmp_path, monkeypatch):
    cfg = configuration(); core = p100._core; calls = []
    monkeypatch.setattr(core, 'verify_reference', lambda c: None)
    monkeypatch.setattr(core, 'verify_run', lambda c, out, updates: {'passed': True, 'updates': updates})
    def compute(logits, batch, loss_cfg, *, pseudo): return logits.square().mean(), {}
    monkeypatch.setattr(p100.native, 'compute_affinity_loss', compute)
    def native_train(config, size, output, **options):
        calls.append((config, size, options)); Path(output).mkdir()
        for _ in range(8):
            for pseudo in (False, True):
                logits = torch.ones((1, 1, 1, 2), requires_grad=True)
                loss, _ = p100.native.compute_affinity_loss(logits, {'name': ['source.png']}, {}, pseudo=pseudo)
                loss.backward()
    monkeypatch.setattr(p100.native, 'train', native_train)
    result = p100.train(cfg, tmp_path / 'run', smoke=True)
    assert result['updates'] == 8 and p100.native.compute_affinity_loss is compute
    assert len(calls) == 1 and calls[0][0] is cfg and calls[0][1] == 1024
    options = calls[0][2]
    assert options['dataset_builder'] is core.mixed_data and options['control_verifier'] is core.verify_reference
    assert options['keep_smoke_checkpoint'] is True
    assert options['checkpoint_extras'](1)['source_state']['format'] == p100.SOURCE_FORMAT
    assert options['checkpoint_extras'](1)['source_state']['manual_coefficient'] == .75
    assert options['monitor_saver'].keywords['reference_folder'].replace('\\', '/').endswith('affinity_balance/mixed/deployment/patch1024')
