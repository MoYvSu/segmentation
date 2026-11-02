# -*- coding: utf-8 -*-
"""独立包引擎构建的CPU合同；使用小模块和TensorRT替身。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F


CODE = Path(__file__).resolve().parents[1] / 'submission/AIC-2026-59388388-代码与数据'
spec = importlib.util.spec_from_file_location('package_engine_export_tested', CODE / 'training/engine_export.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_help_has_relative_package_defaults_without_source_or_TRT_imports():
    command = [sys.executable, '-c',
        "import sys; from training import engine_export as b; "
        "assert 'tensorrt' not in sys.modules and 'models.backend_adaptation' not in sys.modules; "
        "a=b.parse_args([]); assert a.model_dir == str(b.MODEL_ROOT/'weights'); b.parse_args(['--help'])"]
    result = subprocess.run(command, cwd=CODE, capture_output=True, check=True)
    assert b'--model-dir' in result.stdout and b'--output-dir' in result.stdout
    with pytest.raises(SystemExit): build.parse_args(['--batch-size', '2'])


def test_packaged_model_imports_use_source_without_inference_wrappers():
    command = [sys.executable, '-c',
        "import importlib,sys; from training import engine_export as b; source=b._activate_source(); "
        "mods=[importlib.import_module(x) for x in "
        "('models.backend_adaptation','models.rgb_restoration','models.boundary_rgb')]; "
        "assert all(b.Path(x.__file__).resolve().is_relative_to(source) for x in mods); "
        "assert not any(x.startswith('tools.infer_') for x in sys.modules); "
        "print('PACKAGED_MODEL_IMPORTS_OK')"]
    result = subprocess.run(command, cwd=CODE, capture_output=True, check=True)
    assert b'PACKAGED_MODEL_IMPORTS_OK' in result.stdout


def test_output_is_fresh_and_separate_from_weights_and_source(tmp_path, monkeypatch):
    weights = tmp_path/'weights'; weights.mkdir()
    source = tmp_path/'source'; source.mkdir()
    monkeypatch.setattr(build, 'SOURCE_ROOT', source)
    assert build.destination(tmp_path/'engines', weights) == tmp_path/'engines'
    for target in (weights/'engines', source/'engines', tmp_path):
        with pytest.raises((ValueError, FileExistsError)): build.destination(target, weights)
    engines = tmp_path/'engines'; engines.mkdir()
    sentinel = engines/'trunk.engine'; sentinel.write_bytes(b'keep')
    with pytest.raises(FileExistsError): build.destination(engines, weights)
    assert sentinel.read_bytes() == b'keep'


class ToyTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('position', torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2))

    def _get_pos_embed(self, hw):
        return F.interpolate(self.position, size=hw, mode='bicubic', align_corners=False)


@pytest.mark.parametrize('failure', [False, True])
def test_bicubic_constant_is_exact_and_method_restores(failure):
    model = ToyTrunk()
    original = model._get_pos_embed
    expected = original((256, 256))
    try:
        with build.fixed_trunk_position(build.Trunk(model)):
            assert torch.equal(model._get_pos_embed((256, 256)), expected)
            with pytest.raises(ValueError): model._get_pos_embed((128, 128))
            if failure: raise RuntimeError('synthetic exporter failure')
    except RuntimeError:
        assert failure
    assert model._get_pos_embed == original


class BoundaryChannelLayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.bias = torch.nn.Parameter(torch.zeros(3))

    def forward(self, value):
        value = value.permute(0, 2, 3, 1)
        return F.layer_norm(value, (value.shape[-1],), self.weight, self.bias, 1e-6).permute(0, 3, 1, 2)


class ToyNAF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = BoundaryChannelLayerNorm()

    def forward_logits(self, value):
        return self.norm(value)


@pytest.mark.parametrize('failure', [False, True])
def test_NAF_dynamic_batch_adaptation_equal_and_restored(failure):
    model = ToyNAF().eval().requires_grad_(False)
    original = model.norm.forward
    sample = torch.arange(16*3*8*8, dtype=torch.float32).reshape(16, 3, 8, 8)/100
    expected = model.forward_logits(sample)
    try:
        with build.fixed_naf_channels(model, sample) as proof:
            assert proof['checked_batches'] == [1, 16]
            assert torch.equal(model.forward_logits(sample), expected)
            if failure: raise RuntimeError('synthetic exporter failure')
    except RuntimeError:
        assert failure
    assert model.norm.forward == original


@pytest.fixture
def trt_mock():
    h = SimpleNamespace(parse=True, serialize=True, deserialize=True, bad_shape=False, flags=None)

    class Logger:
        WARNING = 1
        def __init__(self, level): pass

    class Network:
        num_inputs = num_outputs = 1
        def __init__(self):
            self.input = SimpleNamespace(name='inputs', shape=(-1, 1, 256, 256), dtype='float32')
            self.output = SimpleNamespace(name='logits', shape=(-1, 1, 256, 256), dtype='float32')
        def get_input(self, index): return self.input
        def get_output(self, index): return self.output

    class Parser:
        num_errors = 1
        def __init__(self, network, logger): pass
        def parse_from_file(self, path): return h.parse
        def get_error(self, index): return 'synthetic parse failure'

    class Profile:
        def set_shape(self, name, *bounds): self.bounds = bounds; return None
        def get_shape(self, name):
            return ((2, 1, 256, 256), *self.bounds[1:]) if h.bad_shape else self.bounds

    class Config:
        def __init__(self): self.flags = {'TF32':True, 'FP16':True, 'INT8':True}
        def clear_flag(self, name): self.flags[name] = False
        def get_flag(self, name): return self.flags[name]
        def set_memory_pool_limit(self, pool, limit): self.workspace = limit
        def add_optimization_profile(self, profile): self.profile = profile; return 0

    class Builder:
        def __init__(self, logger): pass
        def create_network(self, flags): return Network()
        def create_builder_config(self): return Config()
        def create_optimization_profile(self): return Profile()
        def build_serialized_network(self, network, config):
            assert not any(config.flags.values())
            assert config.workspace == 2*2**30 and config.max_aux_streams == 0
            assert config.profile.bounds == build.NAF_BOUNDS
            assert network.input.allowed_formats == network.output.allowed_formats == 1
            h.flags = dict(config.flags)
            return b'new engine weights' if h.serialize else None

    class Runtime:
        def __init__(self, logger): pass
        def deserialize_cuda_engine(self, payload):
            assert payload == b'new engine weights'
            return object() if h.deserialize else None

    h.trt = SimpleNamespace(Logger=Logger, Builder=Builder, Runtime=Runtime, OnnxParser=Parser,
        NetworkDefinitionCreationFlag=SimpleNamespace(EXPLICIT_BATCH=0),
        BuilderFlag=SimpleNamespace(TF32='TF32', FP16='FP16', INT8='INT8'),
        TensorFormat=SimpleNamespace(LINEAR=0), MemoryPoolType=SimpleNamespace(WORKSPACE=0),
        float32='float32', int64='int64')
    return h


def test_None_profile_set_return_is_valid_and_engine_deserializes(trt_mock, tmp_path):
    path = tmp_path/'naf.engine'
    result = build.build_engine('naf', tmp_path/'naf.onnx', path, trt_mock.trt)
    assert result['profile'] == {'min':[1,1,256,256], 'opt':[16,1,256,256], 'max':[16,1,256,256]}
    assert result['file'] == 'naf.engine' and path.read_bytes() == b'new engine weights'


def test_semantic_contract_keeps_verified_highres_logits_and_coarse_feature():
    # 已验收GPU引擎：highres_v1输出全分辨率logits，FPN特征仍是256网格。
    io = {name:dict(mode='INPUT', shape=list(shape))
          for name,shape in zip(build.FEATURES, build.FEATURE_SHAPES)}
    io.update(image=dict(mode='INPUT', shape=[1,3,1024,1024]),
              semantic_logits=dict(mode='OUTPUT', shape=[1,1,1024,1024]),
              semantic_feature=dict(mode='OUTPUT', shape=[1,256,256,256]))
    build._check_io('semantic', io)
    io['semantic_logits']['shape'] = [1,1,256,256]
    with pytest.raises(ValueError, match=r'actual=\(1, 1, 256, 256\).*expected=\(1, 1, 1024, 1024\)'):
        build._check_io('semantic', io)


@pytest.mark.parametrize('failure', ['parse', 'bad_shape', 'serialize', 'deserialize'])
def test_invalid_engine_never_writes_published_file(trt_mock, tmp_path, failure):
    setattr(trt_mock, failure, failure == 'bad_shape')
    path = tmp_path/'naf.engine'
    with pytest.raises(RuntimeError): build.build_engine('naf', tmp_path/'naf.onnx', path, trt_mock.trt)
    assert not path.exists()


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    weights = tmp_path/'weights'; weights.mkdir()
    for name in ('backend', 'restorer', 'naf'): (weights/(name+'.pt')).write_bytes(b'original weights')
    args = SimpleNamespace(model_dir=str(weights), output_dir=str(tmp_path/'engines'))
    modules = {role: torch.nn.Linear(1,1).eval() for role in build.ROLES}
    monkeypatch.setattr(build.torch.cuda, 'is_available', lambda:True)
    monkeypatch.setattr(build.importlib, 'import_module', lambda name:SimpleNamespace())
    monkeypatch.setattr(build, 'load_sources', lambda path:modules)
    monkeypatch.setattr(build, 'synthetic_inputs', lambda modules:{role:(torch.zeros(1),) for role in build.ROLES})
    def export(role, wrapper, sample, path, onnx): path.write_bytes(b'intermediate ONNX')
    monkeypatch.setattr(build, 'export_onnx', export)
    def engine(role, onnx_path, path, trt):
        assert onnx_path.is_file()
        path.write_bytes(role.encode())
        return dict(file=path.name, profile={'min':[1,1,256,256]} if role=='naf' else None)
    monkeypatch.setattr(build, 'build_engine', engine)
    return SimpleNamespace(args=args, weights=weights, output=Path(args.output_dir), engine=engine)


def test_all_five_publish_without_ONNX_or_historical_identity_fields(fake_build):
    h = fake_build
    result = build.run(h.args)
    assert Path(result['engine_dir']) == h.output
    assert {path.name for path in h.output.iterdir()} == {'engines.json', *(role+'.engine' for role in build.ROLES)}
    manifest = json.loads((h.output/'engines.json').read_text())
    assert set(manifest['engines']) == set(build.ROLES)
    assert manifest['trt_flags'] == {'TF32':False, 'FP16':False, 'INT8':False}
    assert all(row['file'] == role+'.engine' for role,row in manifest['engines'].items())
    assert not list(h.output.parent.glob('.engine-build-*'))
    assert all(path.read_bytes() == b'original weights' for path in h.weights.iterdir())


def test_mid_build_failure_publishes_nothing_and_preserves_inputs(fake_build, monkeypatch):
    h = fake_build
    def fail(role, *args):
        if role == 'affinity': raise RuntimeError('synthetic build failure')
        return h.engine(role, *args)
    monkeypatch.setattr(build, 'build_engine', fail)
    with pytest.raises(RuntimeError, match='synthetic'): build.run(h.args)
    assert not h.output.exists() and not list(h.output.parent.glob('.engine-build-*'))
    assert all(path.read_bytes() == b'original weights' for path in h.weights.iterdir())


def test_loader_uses_packaged_source_and_strict_NAF_state_without_old_identity(tmp_path, monkeypatch):
    source = tmp_path/'source'
    (source/'segment-anything-2/sam2').mkdir(parents=True)
    weights = tmp_path/'weights'; weights.mkdir()
    (weights/'backend.pt').write_bytes(b'backend delegated to strict loader')
    (weights/'restorer.pt').write_bytes(b'restorer delegated to strict loader')
    torch.save({'architecture':{'input_mode':'boundary'},
                'model_state_dict':{'weight':torch.tensor([.75])}}, weights/'naf.pt')
    calls = []
    class Backend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module(); self.encoder.trunk = torch.nn.Linear(1,1)
            self.semantic_decoder = torch.nn.Linear(1,1)
            self.affinity_decoder = torch.nn.Linear(1,1)
    class Restorer(torch.nn.Module):
        num_steps, kappa = 16, .03
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
    class NAFModel(torch.nn.Module):
        def __init__(self, input_mode):
            super().__init__()
            self.input_mode, self.input_channels = input_mode, 1
            self.weight = torch.nn.Parameter(torch.ones(1))
        def load_state_dict(self, state, strict):
            calls.append(('NAF_strict', strict))
            return super().load_state_dict(state, strict=strict)
    def backend(path, config, device):
        assert Path(path) == weights/'backend.pt' and device == 'cpu'
        assert config == {'paths':{'project_root':str(source)},
                          'sam2':{'input_normalization':'legacy_none',
                                  'sam2_repo_path':str(source/'segment-anything-2')}}
        return Backend(), {}
    apis = {'models.backend_adaptation':SimpleNamespace(load_backend=backend),
            'models.rgb_restoration':SimpleNamespace(load_rgb_restorer=lambda path,device:Restorer()),
            'models.boundary_rgb':SimpleNamespace(BoundaryRGBNAFNet=NAFModel)}
    monkeypatch.setattr(build, '_activate_source', lambda:source)
    monkeypatch.setattr(build.importlib, 'import_module', lambda name:apis[name])
    modules = build.load_sources(weights, device='cpu')
    assert set(modules) == set(build.ROLES) and calls == [('NAF_strict', True)]
    assert torch.equal(modules['naf'].weight, torch.tensor([.75]))
    assert all(not module.training and not any(p.requires_grad for p in module.parameters())
               for module in modules.values())
