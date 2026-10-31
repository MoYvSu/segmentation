# -*- coding: utf-8 -*-
"""H0单变量、同原初态回执、保存重载和过程monitor的CPU检查。"""
from copy import deepcopy
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pytest
import torch

import train_affinity_hard as hard
from train_affinity_balance import comparable
from train_backend_adaptation import sha, tensor_digest
from utils.affinity_loss import balanced_affinity_loss
from utils.config import load_config


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding='utf8')


def write_rows(path, rows):
    Path(path).write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf8')


def make_rows(config):
    rows = []
    for index in range(1280):
        epoch, step = index // 64 + 1, index % 64 + 1
        rows.append(dict(epoch=epoch, step=step, updates=index + 1,
            seed=config['backend_adaptation']['seed'] + epoch * 10000 + step - 1,
            manual=[f'manual_{index % 32}.png'], pseudo=[f'pseudo_{step - 1}.png'],
            manual_draw={'source_index': [index % 32], 'draw_index': [step - 1], 'blur_seed': [index]},
            pseudo_draw={'source_index': [step - 1], 'draw_index': [step - 1], 'blur_seed': [index + 17]},
            manual_crop=[[0, 0, 1024, 1024]], pseudo_crop=[[1, 2, 1025, 1026]],
            crop_attempts=[[0], [0]], training_view='whole' if step <= 32 else 'native1024',
            manual_bce=.5, pseudo_bce=.6, grad_norm=.2, auxiliary_weight=0.,
            supervision=[dict(positive_edges=100, negative_edges=20, precision=.9, recall=.8, specificity=.7),
                         dict(positive_edges=80, negative_edges=10, precision=.8, recall=.9, specificity=.8)]))
    return rows


def test_resolved_config_has_only_h0_change_and_retains_original_initialization():
    cfg = load_config('config/train/affinity_hard_h0.yaml')
    original = load_config('config/train/affinity_balance_mixed.yaml')
    assert comparable(hard.reference_config(cfg)) == comparable(original)
    assert cfg['direct_semantic_affinity']['affinity_loss']['hard_negative_weight'] == 0.
    assert cfg['backend_adaptation']['output_dir'] == 'outputs/affinity_hard/h0'
    assert cfg['backend_adaptation']['epochs'] == 20
    assert cfg['backend_adaptation']['manual_repeats'] == 2
    for key in ('sam2', 'affinity_geometry_g1', 'semantic_challenger', 'affinity_deployment'):
        assert cfg[key] == original[key]
    assert cfg['affinity_deployment']['checkpoint'] != hard.REFERENCE + '/final.pt'
    assert 'affinity_hard' in cfg and cfg['direct_semantic_affinity']['affinity_loss']['hard_negative_weight'] == 0.


@pytest.mark.parametrize('value', [1., 2., -.1, True, float('nan')])
def test_rejects_unauthorized_h(value):
    cfg = load_config('config/train/affinity_hard_h0.yaml')
    cfg['direct_semantic_affinity']['affinity_loss']['hard_negative_weight'] = value
    with pytest.raises(ValueError):
        hard.reference_config(cfg)


@pytest.mark.parametrize('key,value', [('reference', 'outputs/affinity_sweep/n075'),
                                     ('reference_sha256', 'a' * 64), ('other_variable', 1)])
def test_reference_metadata_cannot_hide_a_switch(key, value):
    cfg = load_config('config/train/affinity_hard_h0.yaml')
    cfg['affinity_hard'][key] = value
    with pytest.raises(ValueError):
        hard.reference_config(cfg)


@pytest.mark.parametrize('section,key,value', [
    ('loss', 'negative_weight', .75), ('loss', 'hard_negative_gamma', 3.),
    ('backend', 'pseudo_weight', .25), ('backend', 'epochs', 40), ('backend', 'manual_repeats', 3)])
def test_budget_and_other_supervision_axes_cannot_change(section, key, value):
    cfg = load_config('config/train/affinity_hard_h0.yaml')
    selected = cfg['direct_semantic_affinity']['affinity_loss'] if section == 'loss' else cfg['backend_adaptation']
    selected[key] = value
    with pytest.raises(ValueError):
        hard.reference_config(cfg)


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    """合成回执和微小模型，不读取赛题图、真实模型或GPU。"""
    cfg = load_config('config/train/affinity_hard_h0.yaml')
    cfg['paths']['project_root'] = str(tmp_path)
    reference = tmp_path / hard.REFERENCE
    reference.mkdir(parents=True)
    (reference / 'final.pt').write_bytes(b'synthetic-reference-checkpoint')
    actual_sha = hard.sha
    monkeypatch.setattr(hard, 'sha', lambda path: hard.REFERENCE_SHA256
                        if Path(path) == reference / 'final.pt' else actual_sha(path))
    ancestor_calls = []
    monkeypatch.setattr(hard, 'verify_balance_reference', lambda c, state=None: ancestor_calls.append((deepcopy(c), state)))
    rows = make_rows(cfg)
    write_rows(reference / 'steps.jsonl', rows)
    tensors = {'encoder.synthetic': torch.tensor([3., 4.]), 'affinity_decoder.synthetic': torch.tensor([1., 2.])}
    base = dict(status='complete', config=hard.reference_config(cfg), epoch=20, updates=1280,
        smoke=False, size=1024, precision='FP32', initial_state_sha256='a' * 64,
        frozen_state_sha256=tensor_digest([('encoder.synthetic', tensors['encoder.synthetic'])]),
        restorer_state_sha256='b' * 64, data={'parent_data': {'manual_sources': 32, 'sam2_sources': 64}},
        **{key: True for key in hard.AUDIT_FLAGS})
    write(reference / 'status.json', base)

    def create(updates=8):
        output = tmp_path / ('smoke' if updates == 8 else 'formal')
        output.mkdir()
        epoch = 1 if updates == 8 else 20
        state = dict(base, config=deepcopy(cfg), epoch=epoch, updates=updates, smoke=updates == 8,
                     final_state_sha256=tensor_digest(tensors.items()))
        bundle = dict(format=hard.FUSED_DEPLOYMENT_FORMAT, config=deepcopy(cfg), epoch=epoch,
                      model_state_dict=deepcopy(tensors), extra={'updates': updates, 'native_size': 1024, 'smoke': updates == 8})
        torch.save(bundle, output / 'final.pt')
        state['final_checkpoint_sha256'] = sha(output / 'final.pt')
        head = dict(format='affinity_native_head_v1', epoch=epoch, size=1024, config=deepcopy(cfg),
                    geometry_state_dict={'synthetic': tensors['affinity_decoder.synthetic'].clone()},
                    optimizer={'state': {}, 'param_groups': [{'lr': 2.e-6}]}, scheduler={'last_epoch': epoch},
                    **hard.checkpoint_extras(cfg, epoch, smoke=updates == 8))
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


def test_ancestor_contract_is_retained_and_other_config_change_rejected(run_fixture):
    cfg, _, _, calls, _ = run_fixture
    assert hard.verify_reference(cfg)[0]['updates'] == 1280
    assert len(calls) == 1
    assert calls[0][0]['direct_semantic_affinity']['affinity_loss']['hard_negative_weight'] == 1.
    changed = deepcopy(cfg)
    changed['direct_semantic_affinity']['affinity_loss']['normalize_edge_weights'] = False
    with pytest.raises(RuntimeError, match='configuration changed'):
        hard.verify_reference(changed)


@pytest.mark.parametrize('key', hard.AUDIT_FLAGS)
def test_reference_audit_cannot_be_skipped(run_fixture, key):
    cfg, reference, _, _, _ = run_fixture
    state = hard.read(reference / 'status.json')
    state[key] = False
    write(reference / 'status.json', state)
    with pytest.raises(RuntimeError, match='control audit failed'):
        hard.verify_reference(cfg)


@pytest.mark.parametrize('updates', [8, 1280])
def test_smoke_and_formal_complete_contract(run_fixture, updates):
    cfg, _, _, _, create = run_fixture
    result = hard.verify_run(cfg, create(updates), updates)
    assert result['passed'] and result['updates'] == updates and result['control_retrained'] is False
    assert result['hard_negative_weight'] == 0. and result['reference_hard_negative_weight'] == 1.
    assert result['paired_sources_crops_augmentation_views_and_counts_exact']
    assert result['continuation_state_saved'] and result['head_bundle_exact'] and result['process_monitor_complete']


@pytest.mark.parametrize('key', hard.DEPENDENCY_KEYS)
def test_initialization_frozen_or_data_change_rejected(run_fixture, key):
    cfg, _, _, _, create = run_fixture
    output = create()
    state = hard.read(output / 'status.json')
    state[key] = {} if key == 'data' else 'c' * 64
    write(output / 'status.json', state)
    with pytest.raises(RuntimeError, match='dependency or data differs'):
        hard.verify_run(cfg, output, 8)


@pytest.mark.parametrize('mutation,match', [('crop', 'Sampling/crop/view'), ('view', 'Sampling/crop/view'),
    ('source', 'Control source/augmentation'), ('counts', 'Supervised edge counts'), ('aux', 'auxiliary'),
    ('seed', 'random seed'), ('nan', 'Nonfinite'), ('gradient', 'gradient')])
def test_input_and_supervision_receipts_are_exact(run_fixture, mutation, match):
    cfg, _, _, _, create = run_fixture
    output = create()
    rows = hard._rows(output / 'steps.jsonl')
    if mutation == 'crop': rows[0]['manual_crop'][0][0] = 1
    elif mutation == 'view': rows[0]['training_view'] = 'native1024'
    elif mutation == 'source': rows[0]['manual_draw']['source_index'] = [17]
    elif mutation == 'counts': rows[0]['supervision'][1]['negative_edges'] += 1
    elif mutation == 'aux': rows[0]['auxiliary_weight'] = .25
    elif mutation == 'seed': rows[0]['seed'] += 1
    elif mutation == 'nan': rows[0]['manual_bce'] = float('nan')
    else: rows[0]['grad_norm'] = 0.
    write_rows(output / 'steps.jsonl', rows)
    with pytest.raises(RuntimeError, match=match):
        hard.verify_run(cfg, output, 8)


@pytest.mark.parametrize('mutation,match', [('bytes', 'checkpoint hash'), ('frozen', 'frozen tensor'),
    ('head', 'training head'), ('rng', 'random state'), ('metadata', 'continuation metadata'),
    ('monitor', 'process monitor PNG')])
def test_actual_checkpoints_and_monitor_required(run_fixture, mutation, match):
    cfg, _, _, _, create = run_fixture
    output = create()
    if mutation == 'bytes':
        with (output / 'final.pt').open('ab') as stream:
            stream.write(b'changed')
    elif mutation == 'frozen':
        bundle = torch.load(output / 'final.pt', weights_only=False)
        bundle['model_state_dict']['encoder.synthetic'][0] += 1
        torch.save(bundle, output / 'final.pt')
        state = hard.read(output / 'status.json')
        state['final_state_sha256'] = tensor_digest(bundle['model_state_dict'].items())
        state['final_checkpoint_sha256'] = sha(output / 'final.pt')
        write(output / 'status.json', state)
    elif mutation in ('head', 'rng', 'metadata'):
        head = torch.load(output / 'last_head.pt', weights_only=False)
        if mutation == 'head': head['geometry_state_dict']['synthetic'][0] += 1
        elif mutation == 'rng': head['hard_state']['rng'].pop('torch_cuda')
        else: head['hard_state']['hard_negative_weight'] = 1.
        torch.save(head, output / 'last_head.pt')
    else:
        (output / 'monitor' / 'epoch_001' / (cfg['backend_adaptation']['monitor']['images'][0] + '_boundary.png')).unlink()
    with pytest.raises(RuntimeError, match=match):
        hard.verify_run(cfg, output, 8)


def test_h0_removes_hardness_redistribution_without_supervising_unknown():
    target = torch.tensor([[[[1., 0., 0., 1.]]]])
    valid = torch.tensor([[[[True, True, True, False]]]])
    gradients = []
    for h in (0., 1.):
        logits = torch.tensor([[[[-2., 2., -2., 8.]]]], requires_grad=True)
        loss, metrics = balanced_affinity_loss(logits, target, valid, negative_weight=1.,
                                              hard_negative_weight=h, hard_negative_gamma=2.)
        loss.backward()
        gradients.append(logits.grad)
        assert metrics['positive_edges'] == 1 and metrics['negative_edges'] == 2
    assert gradients[0][..., 0].item() == gradients[1][..., 0].item()
    assert gradients[0][..., 1].abs() < gradients[1][..., 1].abs()
    assert gradients[0][..., 2].abs() > gradients[1][..., 2].abs()
    assert all(gradient[..., 3].item() == 0. for gradient in gradients)


def test_checkpoint_rng_capture_leaves_random_streams_unchanged():
    cfg = load_config('config/train/affinity_hard_h0.yaml')
    random.seed(42); np.random.seed(43); torch.manual_seed(44)
    before = hard.capture_rng_state()
    state = hard.checkpoint_extras(cfg, 5)['hard_state']
    hard._verify_rng(state['rng'])
    after = hard.capture_rng_state()
    assert state['updates'] == 320 and state['hard_negative_weight'] == 0.
    assert before['python'] == after['python']
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    assert before['numpy'][2:] == after['numpy'][2:]
    assert torch.equal(before['torch_cpu'], after['torch_cpu'])


def test_train_wrapper_uses_original_trainer_and_original_initialization(tmp_path, monkeypatch):
    cfg = load_config('config/train/affinity_hard_h0.yaml')
    original_initialization = {key: deepcopy(cfg[key]) for key in
                               ('sam2', 'affinity_geometry_g1', 'semantic_challenger', 'affinity_deployment')}
    monkeypatch.setattr(hard, 'verify_reference', lambda config: None)
    calls = []
    def fake_train(config, size, output, **kwargs):
        calls.append((config, size, kwargs))
        Path(output).mkdir()
    monkeypatch.setattr(hard, 'native_train', fake_train)
    monkeypatch.setattr(hard, 'verify_run', lambda config, output, updates: {'passed': True, 'updates': updates})
    assert hard.train(cfg, tmp_path / 'run', smoke=True)['updates'] == 8
    assert len(calls) == 1
    config, size, options = calls[0]
    assert config is cfg and size == 1024
    assert {key: config[key] for key in original_initialization} == original_initialization
    assert options['dataset_builder'] is hard.mixed_data
    assert options['control_verifier'] is hard.verify_reference
    assert options['keep_smoke_checkpoint'] is True
    assert options['checkpoint_extras'](1)['hard_state']['hard_negative_weight'] == 0.
    assert options['monitor_saver'].keywords['reference_folder'].replace('\\', '/').endswith('affinity_balance/mixed/deployment/patch1024')
