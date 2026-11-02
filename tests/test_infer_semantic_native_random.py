# -*- coding: utf-8 -*-
"""正式语义增量推理包装的CPU合同；不读赛题数据或载入真实模型。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tools import infer_semantic_native_random as infer


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.affinity_decoder = nn.Linear(2, 8)
        self.semantic_decoder = nn.Linear(3, 1)
        self.semantic_lora = self.geometry_feature_adapter = self.geometry_highres_refiner = None
        self.register_buffer('frozen_buffer', torch.tensor([3.]))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(341)
    model = Backend().eval().requires_grad_(False)
    monkeypatch.setattr(infer, 'EXPECTED_HEAD_PARAMETERS', 4)
    head = {name: value.clone()+.05 for name, value in model.semantic_decoder.state_dict().items()}
    identity = dict(format=infer.random_trainer.FORMAT, base_backend_file_sha256=infer.trainer.BACKEND_SHA,
        base_restorer_file_sha256=infer.trainer.RESTORER_SHA,
        base_semantic_sha256=infer.trainer.tensor_sha(model.semantic_decoder.state_dict().items()),
        semantic_architecture={'test': 'linear'}, total_updates=2560,
        recipe=deepcopy(infer.random_trainer.FIXED),
        frozen=dict(trainable_parameters=4, frozen_backend_sha256=infer.trainer.frozen_sha(model)),
        data=dict(source_seal_sha256='synthetic_test_only'))
    progress = dict(completed_updates=2560, next_epoch=21, next_index=0, affinity_calls=0,
                    training_zero_calls=2560, training_encoder_calls=2560, training_semantic_calls=2560)
    initial_sha = 'synthetic_random_initial_distinct_from_original'
    receipt = dict(initialization='random_full_v1', initialization_seed=1101, parameter_count=4,
        base_original_semantic_sha256=identity['base_semantic_sha256'],
        random_initial_semantic_sha256=initial_sha, previous_trained_delta_loaded=False,
        torch_rng_preserved=True, nonsemantic_before_sha256=identity['frozen']['frozen_backend_sha256'],
        nonsemantic_after_sha256=identity['frozen']['frozen_backend_sha256'])
    identity['frozen'].update(init_receipt=deepcopy(receipt),
        base_original_semantic_sha256=identity['base_semantic_sha256'], random_initial_semantic_sha256=initial_sha)
    infer.random_trainer.normalize_identity(identity)
    bundle = dict(format=infer.random_trainer.FORMAT, identity=identity, progress=progress,
                  semantic_decoder=head, semantic_sha256=infer.trainer.tensor_sha(head.items()))
    path = tmp_path/'final.pt'
    torch.save(bundle, path)
    return model, path, bundle


def test_formal_checkpoint_metadata_and_cli(checkpoint):
    model, path, bundle = checkpoint
    metadata = infer.inspect_checkpoint(path)
    assert metadata['completed_updates'] == 2560 and metadata['epoch'] == 20
    assert metadata['head_sha256'] == bundle['semantic_sha256']
    assert metadata['training_performed_in_this_run'] is False
    assert metadata['optimizer_or_rng_restored'] is False
    with pytest.raises(SystemExit):
        infer.parse_args(['--input-dir','i','--output-dir','o','--naf-checkpoint','naf'])
    args = infer.parse_args(['--input-dir','i','--output-dir','o','--naf-checkpoint','naf',
                             '--semantic-checkpoint',str(path), '--only','test_169'])
    assert args.only == ['test_169'] and args.global_noise == 'zero'


@pytest.mark.parametrize('completed', [0, 3, 2559])
def test_reject_initial_or_smoke_even_if_filename_final(checkpoint, completed):
    model, path, bundle = checkpoint
    bundle['progress']['completed_updates'] = completed
    torch.save(bundle, path)
    with pytest.raises(ValueError, match='2560'):
        infer.inspect_checkpoint(path)


def test_reject_wrong_base_recipe_or_training_counts(checkpoint):
    model, path, original = checkpoint
    for change in ('base', 'recipe', 'count', 'tensor'):
        bundle = deepcopy(original)
        if change == 'base':
            bundle['identity']['base_backend_file_sha256'] = 'different'
        elif change == 'recipe':
            bundle['identity']['recipe']['restoration_noise'] = 'single'
        elif change == 'count':
            bundle['progress']['affinity_calls'] = 1
        else:
            bundle['semantic_decoder']['bias'].add_(1)
        torch.save(bundle, path)
        with pytest.raises(ValueError):
            infer.inspect_checkpoint(path)


def patch_base_hashes(monkeypatch, tmp_path):
    paths = dict(backend=tmp_path/'backend.pt', restorer=tmp_path/'restorer.pt')
    for path in paths.values():
        path.write_bytes(b'synthetic_test_only')
    original = infer.trainer.file_sha
    def file_sha(path):
        if Path(path) == paths['backend']:
            return infer.trainer.BACKEND_SHA
        if Path(path) == paths['restorer']:
            return infer.trainer.RESTORER_SHA
        return original(path)
    monkeypatch.setattr(infer.trainer, 'file_sha', file_sha)
    return paths


def test_actual_delta_loader_changes_only_head_without_mock_history(checkpoint, tmp_path, monkeypatch):
    model, path, bundle = checkpoint
    metadata = infer.inspect_checkpoint(path)
    paths = patch_base_hashes(monkeypatch, tmp_path)
    restorer = nn.Linear(1, 1).eval().requires_grad_(False)
    restorer_before = infer.trainer.tensor_sha(restorer.state_dict().items())
    state = dict(loads=0)
    loader = infer.delta_loader(lambda *args: (model, restorer, None), metadata, state)
    assert not isinstance(loader, Mock) and not hasattr(loader, 'call_args_list')
    result = loader({}, paths, 'cpu')
    assert result == (model, restorer, None)
    assert infer.trainer.tensor_sha(model.semantic_decoder.state_dict().items()) == bundle['semantic_sha256']
    assert infer.trainer.tensor_sha(restorer.state_dict().items()) == restorer_before
    assert infer.trainer.frozen_sha(model) == bundle['identity']['frozen']['frozen_backend_sha256']
    assert state['receipt']['strict_load'] and state['receipt']['completed_updates'] == 2560
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(RuntimeError, match='one'):
        loader({}, paths, 'cpu')


def test_run_report_records_effective_trained_identity_and_unchanged_counts(checkpoint, tmp_path, monkeypatch):
    model, path, bundle = checkpoint
    paths = patch_base_hashes(monkeypatch, tmp_path)
    restorer = nn.Linear(1, 1).eval().requires_grad_(False)
    monkeypatch.setattr(infer.entry, 'load_models', lambda *args: (model, restorer, None))
    output = tmp_path/'inference'
    counts = dict(final200_calls=1, global_zero_calls=0, global_encoder_calls=0, raw_encoder_calls=0,
                  encoder_calls=2, semantic_head_calls=2, affinity_head_calls=2, local_zero_calls=2)
    source = dict(source='synthetic', raw_probability_sha256='probability',
                  fixed_paths=dict(vote_probability_role='old'))
    original = dict(format=infer.unified.FORMAT, complete=True, training=False,
                    unified_local1024=infer.unified.contract(), actual_forward_counts=counts,
                    models=dict(backend=dict(sha256=infer.trainer.BACKEND_SHA)),
                    config={'submission':{}}, baseline_config={'submission':{}}, images=[source])
    def fake_run(args, device):
        assert not isinstance(infer.entry.load_models, Mock)
        infer.entry.load_models({}, paths, device)
        (output/'audit').mkdir(parents=True)
        infer.entry.write(output/'audit/report.json', original)
        infer.entry.write(output/'audit/unified_report.json', original)
        return deepcopy(original)
    monkeypatch.setattr(infer.unified, 'run', fake_run)
    args = SimpleNamespace(semantic_checkpoint=str(path), output_dir=str(output))
    result = infer.run(args, device='cpu')
    assert original['format'] == infer.unified.FORMAT  # audit不原地修改上游文件对象
    assert result['format'] == infer.FORMAT and result['semantic_delta_loaded']
    assert result['semantic_delta']['head_sha256'] == bundle['semantic_sha256']
    assert result['models']['backend']['sha256'] == infer.trainer.BACKEND_SHA
    assert 'replaced' in result['models']['backend']['role']
    assert result['models']['semantic_head_delta']['sha256'] == infer.trainer.file_sha(path)
    assert result['config']['submission']['semantic_head_delta']['head_sha256'] == bundle['semantic_sha256']
    assert 'semantic_head_delta' not in result['baseline_config']['submission']
    assert result['actual_forward_counts'] == counts
    assert result['training'] is False and result['optimizer_steps'] == 0
    assert result['images'][0]['raw_probability_role'] == infer.SEMANTIC_ROLE
    for name in ('report.json', 'unified_report.json', 'semantic_native_random_report.json'):
        saved = json.loads((output/'audit'/name).read_text(encoding='utf-8'))
        assert saved['format'] == infer.FORMAT and saved['semantic_delta_loaded']
        assert saved['semantic_delta_load_receipt']['strict_load']


@pytest.mark.parametrize('corruption', ['old_format', 'missing_random_identity', 'original_alias', 'seed'])
def test_reject_ambiguous_random_provenance(checkpoint, corruption):
    _, path, bundle = checkpoint
    if corruption == 'old_format':
        bundle['format'] = 'semantic_native_zero_delta_v1'
    elif corruption == 'missing_random_identity':
        del bundle['identity']['random_initial_semantic_sha256']
    elif corruption == 'original_alias':
        bundle['identity']['base_semantic_sha256'] = 'not_original_or_initial'
    else:
        bundle['identity']['initialization_seed'] = 1102
    torch.save(bundle, path)
    with pytest.raises((ValueError, KeyError)):
        infer.inspect_checkpoint(path)
