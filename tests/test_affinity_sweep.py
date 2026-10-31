# -*- coding: utf-8 -*-
"""单变量继承、祖先控制、真实抽样/有效域回执和保存状态的CPU门槛。"""
from copy import deepcopy
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pytest
import torch

import train_affinity_sweep as sweep
from train_affinity_balance import comparable
from train_backend_adaptation import sha, tensor_digest
from utils.affinity_loss import balanced_affinity_loss
from utils.config import load_config


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf8')


def write_rows(path, rows):
    Path(path).write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf8')


@pytest.mark.parametrize('name,weight', [('n075', .75), ('n125', 1.25)])
def test_resolved_config_has_only_authorized_change(name, weight):
    cfg = load_config('config/train/affinity_sweep_' + name + '.yaml')
    original = load_config('config/train/affinity_balance_mixed.yaml')
    assert cfg['direct_semantic_affinity']['affinity_loss']['negative_weight'] == weight
    assert comparable(sweep.reference_config(cfg)) == comparable(original)
    assert cfg['backend_adaptation']['output_dir'] == 'outputs/affinity_sweep/' + name
    assert cfg['backend_adaptation']['epochs'] == 20
    assert cfg['backend_adaptation']['manual_repeats'] == 2
    assert 'affinity_sweep' in cfg  # reference_config must not modify its input.


@pytest.mark.parametrize('value', [.5, 1., 1.5, 2., True, float('nan')])
def test_rejects_unauthorized_weight(value):
    cfg = load_config('config/train/affinity_sweep_n075.yaml')
    cfg['direct_semantic_affinity']['affinity_loss']['negative_weight'] = value
    with pytest.raises(ValueError):
        sweep.reference_config(cfg)


@pytest.mark.parametrize('key,value', [('reference', 'outputs/affinity_pair/candidate'),
                                     ('reference_sha256', 'a' * 64), ('hard_negative_weight', 0)])
def test_reference_metadata_has_no_hidden_training_switch(key, value):
    cfg = load_config('config/train/affinity_sweep_n075.yaml')
    cfg['affinity_sweep'][key] = value
    with pytest.raises(ValueError):
        sweep.reference_config(cfg)


def make_rows(config):
    result = []
    for index in range(1280):
        epoch, step = index // 64 + 1, index % 64 + 1
        result.append(dict(epoch=epoch, step=step, updates=index + 1,
            seed=config['backend_adaptation']['seed'] + epoch * 10000 + step - 1,
            manual=[f'manual_{index % 32}.png'], pseudo=[f'pseudo_{step - 1}.png'],
            manual_draw={'source_index': [index % 32], 'draw_index': [step - 1], 'blur_seed': [index]},
            pseudo_draw={'source_index': [step - 1], 'draw_index': [step - 1], 'blur_seed': [index + 17]},
            manual_crop=[[0, 0, 1024, 1024]], pseudo_crop=[[1, 2, 1025, 1026]],
            crop_attempts=[[0], [0]], training_view='whole' if step <= 32 else 'native1024',
            manual_bce=.5, pseudo_bce=.6, grad_norm=.2, auxiliary_weight=0.,
            supervision=[dict(positive_edges=100, negative_edges=20, precision=.9, recall=.8, specificity=.7),
                         dict(positive_edges=80, negative_edges=10, precision=.8, recall=.9, specificity=.8)]))
    return result


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    """合成数据回执，不读取赛题图、真实模型或GPU。"""
    cfg = load_config('config/train/affinity_sweep_n075.yaml')
    cfg['paths']['project_root'] = str(tmp_path)
    reference = tmp_path / sweep.REFERENCE
    reference.mkdir(parents=True)
    (reference / 'final.pt').write_bytes(b'synthetic-reference-checkpoint')
    actual_sha = sweep.sha
    monkeypatch.setattr(sweep, 'sha', lambda path: sweep.REFERENCE_SHA256
                        if Path(path) == reference / 'final.pt' else actual_sha(path))
    ancestor_calls = []
    monkeypatch.setattr(sweep, 'verify_balance_reference', lambda c, state=None: ancestor_calls.append((deepcopy(c), state)))
    rows = make_rows(cfg)
    write_rows(reference / 'steps.jsonl', rows)
    tensors = {'encoder.synthetic': torch.tensor([3., 4.]), 'affinity_decoder.synthetic': torch.tensor([1., 2.])}
    base = dict(status='complete', config=sweep.reference_config(cfg), epoch=20, updates=1280,
        smoke=False, size=1024, precision='FP32', initial_state_sha256='a' * 64,
        frozen_state_sha256=tensor_digest([('encoder.synthetic', tensors['encoder.synthetic'])]),
        restorer_state_sha256='b' * 64, data={'parent_data': {'manual_sources': 32, 'sam2_sources': 64}},
        **{key: True for key in sweep.AUDIT_FLAGS})
    write(reference / 'status.json', base)

    def create(updates=8):
        output = tmp_path / ('smoke' if updates == 8 else 'formal')
        output.mkdir()
        epoch = 1 if updates == 8 else 20
        state = dict(base, config=deepcopy(cfg), epoch=epoch, updates=updates, smoke=updates == 8,
                     final_state_sha256=tensor_digest(tensors.items()))
        bundle = dict(format=sweep.FUSED_DEPLOYMENT_FORMAT, config=deepcopy(cfg), epoch=epoch,
                      model_state_dict=deepcopy(tensors), extra={'updates': updates, 'native_size': 1024, 'smoke': updates == 8})
        torch.save(bundle, output / 'final.pt')
        state['final_checkpoint_sha256'] = sha(output / 'final.pt')
        head = dict(format='affinity_native_head_v1', epoch=epoch, size=1024, config=deepcopy(cfg),
                    geometry_state_dict={'synthetic': tensors['affinity_decoder.synthetic'].clone()},
                    optimizer={'state': {}, 'param_groups': [{'lr': 2.e-6}]}, scheduler={'last_epoch': epoch},
                    **sweep.checkpoint_extras(cfg, epoch, smoke=updates == 8))
        torch.save(head, output / 'last_head.pt')
        write(output / 'status.json', state)
        write_rows(output / 'steps.jsonl', rows[:updates])
        epochs = (0, 1) if updates == 8 else (0, 1, 5, 10, 15, 20)
        names = cfg['backend_adaptation']['monitor']['images'][:2] if updates == 8 else cfg['backend_adaptation']['monitor']['images']
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
        return output
    return cfg, reference, rows, ancestor_calls, create


def test_ancestor_verifier_is_retained_and_config_mutation_rejected(run_fixture):
    cfg, _, _, calls, _ = run_fixture
    assert sweep.verify_reference(cfg)[0]['updates'] == 1280
    assert len(calls) == 1
    assert calls[0][0]['direct_semantic_affinity']['affinity_loss']['negative_weight'] == 1.
    for key, value in [('hard_negative_weight', 2.), ('pseudo_negative_weight', .5), ('normalize_edge_weights', False)]:
        altered = deepcopy(cfg)
        altered['direct_semantic_affinity']['affinity_loss'][key] = value
        with pytest.raises(RuntimeError, match='configuration changed'):
            sweep.verify_reference(altered)
    altered = deepcopy(cfg)
    altered['backend_adaptation']['pseudo_weight'] = .25
    with pytest.raises(RuntimeError, match='configuration changed'):
        sweep.verify_reference(altered)


@pytest.mark.parametrize('key', sweep.AUDIT_FLAGS)
def test_incomplete_or_unfrozen_reference_cannot_be_reused(run_fixture, key):
    cfg, reference, _, _, _ = run_fixture
    status = sweep.read(reference / 'status.json')
    status[key] = False
    write(reference / 'status.json', status)
    with pytest.raises(RuntimeError, match='control audit failed'):
        sweep.verify_reference(cfg)


@pytest.mark.parametrize('updates', [8, 1280])
def test_complete_smoke_and_formal_exact_sampling_gate(run_fixture, updates):
    cfg, _, _, _, create = run_fixture
    output = create(updates)
    result = sweep.verify_run(cfg, output, updates)
    assert result['passed'] and result['updates'] == updates and not result['control_retrained']
    assert result['paired_sources_crops_augmentation_views_and_counts_exact']
    assert result['head_bundle_exact'] and result['process_monitor_complete']
    assert result['continuation_state_saved']


@pytest.mark.parametrize('mutation,match', [
    ('crop', 'Sampling/crop/view'), ('source', 'Control source/augmentation'),
    ('view', 'Sampling/crop/view'), ('counts', 'Supervised edge counts'),
    ('aux', 'auxiliary'), ('seed', 'random seed'), ('nan', 'Nonfinite'),
    ('gradient', 'gradient'), ('missing', 'Missing or extra')])
def test_changed_input_supervision_or_numerics_is_rejected(run_fixture, mutation, match):
    cfg, _, _, _, create = run_fixture
    output = create()
    rows = sweep._rows(output / 'steps.jsonl')
    if mutation == 'crop': rows[0]['manual_crop'][0][0] = 2
    elif mutation == 'source': rows[0]['manual_draw']['source_index'] = [17]
    elif mutation == 'view': rows[0]['training_view'] = 'native1024'
    elif mutation == 'counts': rows[0]['supervision'][1]['negative_edges'] += 1
    elif mutation == 'aux': rows[0]['auxiliary_weight'] = .25
    elif mutation == 'seed': rows[0]['seed'] += 1
    elif mutation == 'nan': rows[0]['manual_bce'] = float('nan')
    elif mutation == 'gradient': rows[0]['grad_norm'] = 0.
    else: rows.pop()
    write_rows(output / 'steps.jsonl', rows)
    with pytest.raises(RuntimeError, match=match):
        sweep.verify_run(cfg, output, 8)


@pytest.mark.parametrize('key', sweep.DEPENDENCY_KEYS)
def test_shared_initialization_or_data_change_is_rejected(run_fixture, key):
    cfg, _, _, _, create = run_fixture
    output = create()
    status = sweep.read(output / 'status.json')
    status[key] = {} if key == 'data' else 'c' * 64
    write(output / 'status.json', status)
    with pytest.raises(RuntimeError, match='dependency or data differs'):
        sweep.verify_run(cfg, output, 8)


@pytest.mark.parametrize('mutation,match', [('bytes', 'checkpoint hash'), ('frozen', 'frozen tensor'),
                                         ('head', 'training head'), ('rng', 'random state'),
                                         ('monitor', 'process monitor PNG')])
def test_saved_checkpoint_and_real_monitor_are_not_just_status_flags(run_fixture, mutation, match):
    cfg, _, _, _, create = run_fixture
    output = create()
    if mutation == 'bytes':
        with (output / 'final.pt').open('ab') as stream:
            stream.write(b'changed')
    elif mutation == 'frozen':
        bundle = torch.load(output / 'final.pt', weights_only=False)
        bundle['model_state_dict']['encoder.synthetic'][0] += 1
        torch.save(bundle, output / 'final.pt')
        state = sweep.read(output / 'status.json')
        state['final_state_sha256'] = tensor_digest(bundle['model_state_dict'].items())
        state['final_checkpoint_sha256'] = sha(output / 'final.pt')
        write(output / 'status.json', state)
    elif mutation in ('head', 'rng'):
        head = torch.load(output / 'last_head.pt', weights_only=False)
        if mutation == 'head': head['geometry_state_dict']['synthetic'][0] += 1
        else: head['sweep_state']['rng'].pop('torch_cuda')
        torch.save(head, output / 'last_head.pt')
    else:
        (output / 'monitor' / 'epoch_001' / (cfg['backend_adaptation']['monitor']['images'][0] + '_boundary.png')).unlink()
    with pytest.raises(RuntimeError, match=match):
        sweep.verify_run(cfg, output, 8)


def test_both_weight_directions_change_expected_gradient_unknown_remains_zero():
    target = torch.tensor([[[[1., 0., 1.]]]])
    valid = torch.tensor([[[[True, True, False]]]])
    gradients = []
    for weight in (.75, 1., 1.25):
        logits = torch.tensor([[[[-2., 2., -2.]]]], requires_grad=True)
        loss, _ = balanced_affinity_loss(logits, target, valid, negative_weight=weight)
        loss.backward()
        gradients.append(logits.grad)
    assert gradients[0][..., 0].abs() > gradients[1][..., 0].abs() > gradients[2][..., 0].abs()
    assert gradients[0][..., 1].abs() < gradients[1][..., 1].abs() < gradients[2][..., 1].abs()
    assert all(value[..., 2].item() == 0 for value in gradients)


def test_checkpoint_state_capture_does_not_consume_random_stream():
    cfg = load_config('config/train/affinity_sweep_n075.yaml')
    random.seed(42)
    np.random.seed(43)
    torch.manual_seed(44)
    before = sweep.capture_rng_state()
    extra = sweep.checkpoint_extras(cfg, 5)
    sweep._verify_rng(extra['sweep_state']['rng'])
    after = sweep.capture_rng_state()
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:]
    assert torch.equal(before['torch_cpu'], after['torch_cpu'])


def test_wrapper_uses_original_train_and_explicit_save_only_options(tmp_path, monkeypatch):
    cfg = load_config('config/train/affinity_sweep_n125.yaml')
    monkeypatch.setattr(sweep, 'verify_reference', lambda config: None)
    calls = []
    def fake_train(config, size, output, **kwargs):
        calls.append((config, size, kwargs))
        Path(output).mkdir()
    monkeypatch.setattr(sweep, 'native_train', fake_train)
    monkeypatch.setattr(sweep, 'verify_run', lambda config, output, updates: {'passed': True, 'updates': updates})
    assert sweep.train(cfg, tmp_path / 'run', smoke=True)['updates'] == 8
    assert len(calls) == 1
    config, size, options = calls[0]
    assert config is cfg and size == 1024
    assert options['dataset_builder'] is sweep.mixed_data
    assert options['control_verifier'] is sweep.verify_reference
    assert options['keep_smoke_checkpoint'] is True
    assert options['checkpoint_extras'](1)['sweep_state']['negative_weight'] == 1.25
    assert options['monitor_saver'].keywords['reference_folder'].replace('\\', '/').endswith('affinity_balance/mixed/deployment/patch1024')
