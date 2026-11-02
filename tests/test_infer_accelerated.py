# -*- coding: utf-8 -*-
"""全五引擎入口CPU合同：原位方法适配、无回退、真实整数读出及异常恢复。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.fused_deployment import FusedPhaseAffinityModel
from tools import infer_accelerated as infer


def forbidden(*args, **kwargs):
    raise AssertionError('Original network computation must not execute as fallback')


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    forward = forbidden


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Block()

    def forward(self, image):
        return self.trunk(image)


class Restorer(Block):
    denoise = forbidden

    def _check_inputs(self, xt, condition, t_index):
        self.checks = getattr(self, 'checks', 0) + 1
        if xt.shape != condition.shape or t_index.dtype != torch.int64:
            raise ValueError('Original restorer input check rejected inputs')


class NAF(Block):
    input_channels, boundary_channel = 1, 0
    forward_logits = forbidden

    def forward(self, inputs):
        return self.forward_logits(inputs).sigmoid()


@pytest.fixture
def case(monkeypatch, tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    h = SimpleNamespace(root=tmp_path, output=tmp_path / 'candidate', loaded={},
                        image_input=True, semantic_feature=True, request_features=True,
                        unknown_input=False, fail_after_report=False)
    h.model = FusedPhaseAffinityModel(Encoder(), Block(), Block()).eval().requires_grad_(False)
    h.restorer, h.naf = Restorer(), NAF()
    h.manifest = dict(format=infer.MANIFEST_FORMAT, cpu_integer_readout=True, precision='float32',
                      trt_flags=dict(TF32=False, FP16=False, INT8=False), engines={})
    source_modules = dict(trunk=h.model.encoder.trunk, semantic=h.model.semantic_decoder,
                          affinity=h.model.affinity_decoder, d5a=h.restorer, naf=h.naf)
    for role in infer.ROLES:
        path = tmp_path / (role + '.engine')
        path.write_bytes(('mock ' + role).encode())
        h.manifest['engines'][role] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_state_sha256=infer.entry.tensor_digest(source_modules[role].state_dict().items()))
    h.manifest_path = tmp_path / 'engines.json'

    def save_manifest():
        h.manifest_path.write_text(json.dumps(h.manifest), encoding='utf-8')

    h.save_manifest = save_manifest
    save_manifest()

    class Engine:
        def __init__(self, path, expected_sha256):
            self.role = path.stem
            assert expected_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
            self.input_names = dict(trunk=('image',), semantic=infer.FEATURES,
                affinity=infer.FEATURES, naf=('inputs',), d5a=('xt', 'condition', 't_index'))[self.role]
            if self.role == 'semantic':
                if h.image_input:
                    self.input_names += ('image',)
                if h.unknown_input:
                    self.input_names += ('unexpected',)
            self.output_names = dict(trunk=infer.FEATURES, semantic=('semantic_logits',),
                affinity=('affinity_logits', 'affinity_feature'), naf=('logits',), d5a=('x0',))[self.role]
            if self.role == 'semantic' and h.semantic_feature:
                self.output_names += ('semantic_feature',)
            self.signature = {name:dict(mode='INPUT' if name in self.input_names else 'OUTPUT',
                dtype='int64' if name == 't_index' else 'float32', shape=[1])
                for name in (*self.input_names, *self.output_names)}
            self.execution_count, self.last_execution = 0, None
            h.loaded[self.role] = self

        def __call__(self, **inputs):
            assert set(inputs) == set(self.input_names)
            self.execution_count += 1
            self.last_execution = dict(execution_count=self.execution_count)
            if self.role == 'trunk':
                return {name:inputs['image'] + index for index, name in enumerate(infer.FEATURES)}
            if self.role == 'semantic':
                result = dict(semantic_logits=inputs['f0'][:, :1])
                if h.semantic_feature:
                    result['semantic_feature'] = inputs['f0'] * 2
                return result
            if self.role == 'affinity':
                return dict(affinity_logits=inputs['f0'][:, :1], affinity_feature=inputs['f0'])
            if self.role == 'd5a':
                return dict(x0=inputs['condition'] + .01)
            return dict(logits=torch.full_like(inputs['inputs'], 2.0))

    monkeypatch.setattr(infer, 'Engine', Engine)
    monkeypatch.setattr(infer.entry, 'load_models', lambda *a, **k: (h.model, h.restorer, None))
    monkeypatch.setattr(infer.naf, 'load_naf_checkpoint', lambda *a, **k: (h.naf, {'unchanged':True}))
    h.original_methods = [(h.model.encoder.trunk, 'forward', h.model.encoder.trunk.forward),
        (h.model.semantic_decoder, 'forward', h.model.semantic_decoder.forward),
        (h.model.affinity_decoder, 'forward', h.model.affinity_decoder.forward),
        (h.restorer, 'denoise', h.restorer.denoise), (h.naf, 'forward_logits', h.naf.forward_logits)]
    h.original_load, h.original_naf_load = infer.entry.load_models, infer.naf.load_naf_checkpoint
    h.original_write = infer.entry.write

    def pipeline(args, *, device):
        assert device == 'cpu'
        model, restorer, terminal = infer.entry.load_models(None, None, device)
        naf, provenance = infer.naf.load_naf_checkpoint('unused', device=device)
        assert model is h.model and restorer is h.restorer and naf is h.naf and terminal is None
        image = torch.full((1, 3, 4, 4), .25)
        t = torch.tensor([16], dtype=torch.int64)
        restored = restorer.denoise(image, image, t)
        global_prediction = model(restored, return_features=h.request_features)
        assert 'affinity_logits' in global_prediction
        if h.request_features:
            assert torch.equal(global_prediction['semantic_feature'], restored * 2)
        raw = model.semantic_decoder(model.encoder(image), image)
        assert torch.equal(raw, image[:, :1])
        local = restorer.denoise(image, image, t)
        assert set(model.affinity_decoder(model.encoder(local))) == {'affinity_logits', 'affinity_feature'}
        probability = naf(torch.zeros((1, 1, 256, 256)))
        assert torch.equal(probability, torch.full_like(probability, 2.0).sigmoid())
        # 真正调用内层fast.run已替换的整数函数，而非绕过CPU快路径。
        ids, classes, proof = infer.fast.compact.compact.finish_watershed(
            np.array([[1, 1, 0, 7], [1, 1, 0, 7]], np.int32), dict(min_instance_area=1))
        assert ids.dtype == np.uint16 and len(classes) == 2
        (h.output / 'audit').mkdir(parents=True)
        document = dict(complete=True, format='compact_base', config={},
            compact_readout=dict(model_forward_unchanged=True),
            images=[dict(source='synthetic', inference=dict(tiles=1, batch_size=16))],
            global_zero_calls=1, local_zero_calls=1)
        infer.entry.write(h.output / 'audit/report.json', document)
        if h.fail_after_report:
            raise RuntimeError('deliberate inference failure')
        return document

    monkeypatch.setattr(infer.fast.compact, 'run', pipeline)
    h.args = SimpleNamespace(output_dir=str(h.output), engine_manifest=str(h.manifest_path), geometry_mode='C')
    yield h
    torch.set_num_threads(old_threads)


def assert_restored(h):
    for model, name, method in h.original_methods:
        assert getattr(model, name) == method
    assert infer.entry.load_models is h.original_load
    assert infer.naf.load_naf_checkpoint is h.original_naf_load
    assert infer.entry.write is h.original_write


@pytest.mark.parametrize('image_input', [True, False])
def test_full_adapters_forward_protocol_CPU_fast_audit_and_state_identity(case, image_input):
    h = case
    h.image_input = image_input
    modules = list(h.model.named_modules())
    parameters = list(h.model.parameters())
    original_state = {k:v.clone() for k,v in h.model.state_dict().items()}
    result = infer.run(h.args, device='cpu')
    assert_restored(h)
    assert list(h.model.named_modules()) == modules
    assert all(a is b for a,b in zip(parameters, h.model.parameters()))
    assert set(original_state) == set(h.model.state_dict())
    assert all(torch.equal(value,h.model.state_dict()[name]) for name,value in original_state.items())
    assert h.restorer.checks == 2
    expected = dict(trunk=3, semantic=2, affinity=2, naf=1, d5a=2)
    assert result['format'] == infer.FORMAT and result['training'] is False
    assert result['accelerated_runtime']['no_fallback'] is True
    assert result['accelerated_runtime']['cpu_integer_readout'] is True
    assert result['accelerated_runtime']['all_engine_source_states_verified'] is True
    for role, record in result['accelerated_runtime']['engines'].items():
        assert record['source_state_verified'] is True
        assert record['verified_source_state_sha256'] == h.manifest['engines'][role]['source_state_sha256']
    assert {k:v['actual_calls'] for k,v in result['accelerated_runtime']['engines'].items()} == expected
    assert {k:v['attempts'] for k,v in result['accelerated_runtime']['engines'].items()} == expected
    for name in ('report.json', 'fast_readout_report.json', 'accelerated_report.json'):
        saved = json.loads((h.output / 'audit' / name).read_text())
        assert saved['format'] == infer.FORMAT and saved['no_training']
        assert saved['accelerated_runtime']['engine_manifest_sha256'] == infer.entry.sha(h.manifest_path)
        assert saved['accelerated_entry_sha256'] == infer.entry.sha(infer.__file__)
        assert saved['tensorrt_runtime_sha256'] == infer.entry.sha(infer.tensorrt_runtime.__file__)
        assert saved['compact_readout']['model_forward_unchanged'] is False
        assert saved['fast_readout']['model_forward_unchanged'] is False


def test_exception_restores_original_methods_and_marks_all_audits_failed(case):
    h = case
    h.fail_after_report = True
    with pytest.raises(RuntimeError, match='deliberate inference failure'):
        infer.run(h.args, device='cpu')
    assert_restored(h)
    for path in (h.output / 'audit').glob('*.json'):
        value = json.loads(path.read_text())
        assert value['complete'] is False and value['format'] == infer.FORMAT
        assert value['accelerated_runtime']['no_fallback'] is True
    assert (h.output / 'audit/accelerated_failed.json').is_file()


def test_semantic_missing_requested_feature_fails_without_original_computation(case):
    case.semantic_feature = False
    with pytest.raises(ValueError, match='semantic_feature'):
        infer.run(case.args, device='cpu')
    assert_restored(case)
    assert case.loaded['semantic'].execution_count == 0


def test_unknown_semantic_input_rejected_and_patches_restore(case):
    case.unknown_input = True
    with pytest.raises(ValueError, match='I/O names'):
        infer.run(case.args, device='cpu')
    assert_restored(case)


@pytest.mark.parametrize('failure', ['parent', 'absolute', 'sha', 'missing_role', 'flag', 'missing_source'])
def test_manifest_failures_happen_before_any_engine_load(case, failure):
    h = case
    if failure == 'parent': h.manifest['engines']['trunk']['file'] = '../trunk.engine'
    elif failure == 'absolute': h.manifest['engines']['trunk']['file'] = '/tmp/trunk.engine'
    elif failure == 'sha': h.manifest['engines']['trunk']['sha256'] = '0' * 64
    elif failure == 'missing_role': del h.manifest['engines']['naf']
    elif failure == 'flag': h.manifest['trt_flags']['TF32'] = True
    elif failure == 'missing_source': del h.manifest['engines']['trunk']['source_state_sha256']
    h.save_manifest()
    with pytest.raises(ValueError):
        infer.run(h.args, device='cpu')
    assert not h.loaded and not h.output.exists()
    assert_restored(h)


@pytest.mark.parametrize('role', infer.ROLES)
def test_wrong_source_state_rejected_before_that_engine_is_created_or_patched(case, role):
    h = case
    h.manifest['engines'][role]['source_state_sha256'] = '0' * 64
    h.save_manifest()
    with pytest.raises(ValueError, match='source module state SHA256 differs: ' + role):
        infer.run(h.args, device='cpu')
    assert role not in h.loaded
    assert_restored(h)


@pytest.mark.parametrize('kind', ['nan', 'range', 'binary', 'dtype', 'shape'])
def test_NAF_preserves_original_input_checks_before_engine(kind):
    inputs = torch.zeros((1, 1, 2, 2))
    if kind == 'nan': inputs[0,0,0,0] = float('nan')
    elif kind == 'range': inputs[0,0,0,0] = 2
    elif kind == 'binary': inputs[0,0,0,0] = .5
    elif kind == 'dtype': inputs = inputs.half()
    elif kind == 'shape': inputs = inputs[0]
    with pytest.raises((ValueError, TypeError)):
        infer.naf_input_checks(SimpleNamespace(input_channels=1,boundary_channel=0), inputs)


def test_explicit_C_default_and_required_manifest_parser():
    required = ['--input-dir','i','--output-dir','o','--naf-checkpoint','n']
    with pytest.raises(SystemExit): infer.parse_args(required)
    args = infer.parse_args(required + ['--engine-manifest','engines.json'])
    assert args.geometry_mode == 'C' and args.global_noise == 'zero'
    with pytest.raises(SystemExit): infer.parse_args(required + ['--engine-manifest','e','--geometry-mode','F0'])
