# -*- coding: utf-8 -*-
"""公开builder的CPU合同：固定适配、profile、生命周期和原子发布；无GPU构建。"""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import weakref

import pytest
import torch
from torch.nn import functional as F

from models.boundary_naf import BoundaryChannelLayerNorm
from tools import build_inference_engines as build


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def test_help_without_tensorrt_and_no_1024_batch_override():
    command = [sys.executable, '-c',
        "import sys; from tools import build_inference_engines as b; "
        "assert 'tensorrt' not in sys.modules; b.parse_args(['--help'])"]
    result = subprocess.run(command, cwd=build.ROOT, capture_output=True, check=True)
    assert b'naf-checkpoint' in result.stdout
    args = build.parse_args(['build', '--naf-checkpoint', 'naf.pt'])
    assert 'outputs' in args.output_dir and not hasattr(args, 'batch_size')
    with pytest.raises(SystemExit):
        build.parse_args(['build', '--naf-checkpoint', 'n', '--batch-size', '2'])


def test_private_fresh_destination_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    target = tmp_path / 'outputs/new/engines'
    assert build.destination(target) == target
    for bad in (tmp_path, tmp_path / 'outputs', tmp_path / 'outside/new'):
        with pytest.raises(ValueError): build.destination(bad)
    target.mkdir(parents=True)
    sentinel = target / 'trunk.engine'
    sentinel.write_bytes(b'preserve')
    with pytest.raises(FileExistsError): build.destination(target)
    assert sentinel.read_bytes() == b'preserve'


class ToyTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('position', torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2))

    def _get_pos_embed(self, hw):
        return F.interpolate(self.position, size=hw, mode='bicubic', align_corners=False)


@pytest.mark.parametrize('fail', [False, True])
def test_exact_bicubic_constant_and_method_restore_on_export_failure(fail):
    model = ToyTrunk()
    wrapper = build.Trunk(model)
    original = model._get_pos_embed
    before = build.app.entry.tensor_digest(model.state_dict().items())
    expected = original((256, 256))
    try:
        with build.fixed_trunk_position(wrapper) as proof:
            assert torch.equal(model._get_pos_embed((256, 256)), expected)
            assert proof['position_torch_equal'] is True
            with pytest.raises(ValueError): model._get_pos_embed((128, 128))
            if fail: raise RuntimeError('simulated exporter failure')
    except RuntimeError:
        assert fail
    assert model._get_pos_embed == original and proof['original_method_restored']
    assert build.app.entry.tensor_digest(model.state_dict().items()) == before


class ToyNAF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = BoundaryChannelLayerNorm(3)

    def forward_logits(self, value):
        return self.norm(value)


@pytest.mark.parametrize('fail', [False, True])
def test_original_LayerNorm_equal_at_batch_bounds_and_restore(fail):
    model = ToyNAF().eval().requires_grad_(False)
    sample = torch.arange(16 * 3 * 8 * 8, dtype=torch.float32).reshape(16, 3, 8, 8) / 100
    before = build.app.entry.tensor_digest(model.state_dict().items())
    original = model.norm.forward
    try:
        with build.fixed_naf_channels(model, sample) as proof:
            assert proof['adaptation_logits_torch_equal'] is True and proof['checked_batches'] == [1, 16]
            if fail: raise RuntimeError('simulated NAF exporter failure')
    except RuntimeError:
        assert fail
    assert model.norm.forward == original and proof['original_methods_restored']
    assert build.app.entry.tensor_digest(model.state_dict().items()) == before


@pytest.fixture
def trt_mock():
    h = SimpleNamespace(events=[], bad_profile=False, parse_ok=True, serialize_ok=True, refs={})

    def track(name, value):
        h.refs[name] = weakref.ref(value)
        weakref.finalize(value, h.events.append, name)

    class Logger:
        WARNING = 1
        def __init__(self, severity): track('logger', self)

    class Network:
        num_inputs = num_outputs = 1
        def __init__(self):
            self.input = SimpleNamespace(name='inputs', shape=(-1, 1, 256, 256), dtype='float32')
            self.output = SimpleNamespace(name='logits', shape=(-1, 1, 256, 256), dtype='float32')
            track('network', self)
        def get_input(self, index): return self.input
        def get_output(self, index): return self.output

    class Parser:
        num_errors = 1
        def __init__(self, network, logger):
            self.network, self.logger = network, logger
            track('parser', self)
        def parse_from_file(self, path): return h.parse_ok
        def get_error(self, index): return 'mock ONNX parse error'

    class Profile:
        def __init__(self): track('profile', self)
        def set_shape(self, name, *bounds): self.bounds = bounds; return None
        def get_shape(self, name): return ((2, 1, 256, 256), *self.bounds[1:]) if h.bad_profile else self.bounds

    class Config:
        def __init__(self):
            self.flags = dict(TF32=True, FP16=False, INT8=False)
            track('config', self)
        def clear_flag(self, flag): self.flags[flag] = False
        def get_flag(self, flag): return self.flags[flag]
        def set_memory_pool_limit(self, pool, limit): self.workspace = limit
        def add_optimization_profile(self, profile): self.profile = profile; return 0

    class Serialized:
        def __init__(self): track('serialized', self)
        def __bytes__(self): return b'mock TensorRT engine'

    class Builder:
        def __init__(self, logger): self.logger = logger; track('builder', self)
        def create_network(self, flags): return Network()
        def create_builder_config(self): return Config()
        def create_optimization_profile(self): return Profile()
        def build_serialized_network(self, network, config):
            assert not any(config.flags.values()) and config.workspace == 2 * 2**30
            assert config.max_aux_streams == 0 and config.profile.bounds == build.NAF_BOUNDS
            assert network.input.allowed_formats == network.output.allowed_formats == 1
            return Serialized() if h.serialize_ok else None

    h.trt = SimpleNamespace(Logger=Logger, Builder=Builder, OnnxParser=Parser,
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
        BuilderFlag=SimpleNamespace(TF32='TF32',FP16='FP16',INT8='INT8'),
        MemoryPoolType=SimpleNamespace(WORKSPACE=0), TensorFormat=SimpleNamespace(LINEAR=0),
        float32='float32', int64='int64', __version__='10.12.mock')
    return h


def test_profile_set_shape_None_is_valid_and_native_children_release_first(trt_mock, tmp_path):
    h = trt_mock
    record = build.build_engine('naf', tmp_path/'n.onnx', tmp_path/'n.engine', h.trt)
    assert record['profile']['set_shape_return_repr'] == 'None'
    assert record['profile']['get_shape_verified'] and record['profile']['opt'][0] == 16
    assert record['sha256'] == build.app.entry.sha(tmp_path/'n.engine')
    assert all(ref() is None for ref in h.refs.values())
    assert h.events.index('serialized') < h.events.index('builder') < h.events.index('logger')
    assert h.events.index('network') < h.events.index('builder')


@pytest.mark.parametrize('failure', ['parse', 'profile', 'serialize'])
def test_failed_TRT_build_preserves_no_engine_and_releases_native_owners(trt_mock, tmp_path, failure):
    h = trt_mock
    if failure == 'parse': h.parse_ok = False
    elif failure == 'profile': h.bad_profile = True
    else: h.serialize_ok = False
    with pytest.raises(RuntimeError):
        build.build_engine('naf', tmp_path/'n.onnx', tmp_path/'n.engine', h.trt)
    assert not (tmp_path/'n.engine').exists()
    assert all(ref() is None for ref in h.refs.values())
    assert h.events[-2:] == ['builder', 'logger']


@pytest.fixture
def fake_build(monkeypatch, tmp_path):
    h = SimpleNamespace(target=tmp_path/'outputs/new_engines', fail_role=None, built=[])
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU simulated GPU')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda: (8, 9))
    monkeypatch.setattr(build, 'importlib', SimpleNamespace(import_module=lambda name: SimpleNamespace(__version__='mock')))
    modules = {role:torch.nn.Linear(1,1,bias=False).eval().requires_grad_(False) for role in build.ROLES}
    config, weight = tmp_path/'config.json', tmp_path/'model.pt'
    config.write_text('{}'); weight.write_bytes(b'fake source checkpoint')
    source = dict(config_sha256=build.app.entry.sha(config), model_file_sha256=dict(backend=build.app.entry.sha(weight)))
    monkeypatch.setattr(build, '_load_sources', lambda args: (modules,source,dict(backend=weight),config))
    monkeypatch.setattr(build, 'synthetic_inputs', lambda modules: {role:(torch.zeros(1),) for role in build.ROLES})

    def export(role, wrapper, sample, path, onnx):
        path.write_bytes(('ONNX '+role).encode())
        return dict(onnx_file=path.name, onnx_sha256=build.app.entry.sha(path))

    def engine(role, onnx_path, engine_path, trt):
        assert not h.target.exists()
        if role == h.fail_role: raise RuntimeError('deliberate '+role+' build failure')
        engine_path.write_bytes(('engine '+role).encode())
        h.built.append(role)
        return dict(file=engine_path.name,sha256=build.app.entry.sha(engine_path))

    monkeypatch.setattr(build, 'export_onnx', export)
    monkeypatch.setattr(build, 'build_engine', engine)
    h.args = SimpleNamespace(output_dir=str(h.target))
    h.modules = modules
    return h


def test_publish_only_after_all_five_with_v2_source_and_relative_paths(fake_build):
    h = fake_build
    result = build.run(h.args)
    assert result['complete'] and Path(result['manifest']) == h.target/'engines.json'
    manifest, files = build.accelerated.load_manifest(h.target/'engines.json')
    assert manifest['format'] == 'C_TensorRT_acceleration_v2' and len(files) == 5
    assert h.built == list(build.ROLES) and not list(h.target.parent.glob('.*.building-*'))
    for role, row in manifest['engines'].items():
        assert row['source_state_sha256'] == build.app.entry.tensor_digest(h.modules[role].state_dict().items())
        assert not Path(row['file']).is_absolute()
    assert manifest['versions']['GPU_compute_capability'] == [8,9]
    assert manifest['build']['full_inference_verified'] is False
    with pytest.raises(FileExistsError): build.run(h.args)


def test_failure_keeps_diagnostics_but_never_publishes_success_manifest(fake_build):
    h = fake_build
    h.fail_role = 'naf'
    with pytest.raises(RuntimeError, match='deliberate naf'):
        build.run(h.args)
    assert not h.target.exists()
    folders = list(h.target.parent.glob('.new_engines.building-*'))
    assert len(folders) == 1 and (folders[0]/'build_failed.json').is_file()
    assert not (folders[0]/'engines.json').exists()
    state = json.loads((folders[0]/'build_state.json').read_text())
    assert state['complete'] is False and state['phase'] == 'failed'


def test_no_CUDA_rejects_before_importing_GPU_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(build, 'importlib', SimpleNamespace(import_module=lambda name: pytest.fail('No TRT import')))
    with pytest.raises(RuntimeError, match='requires CUDA'):
        build.run(SimpleNamespace(output_dir=str(tmp_path/'outputs/new')))
    assert not (tmp_path/'outputs').exists()
