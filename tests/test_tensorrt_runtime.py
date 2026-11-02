# -*- coding: utf-8 -*-
"""CPU模拟TensorRT/CUDA接口，验证运行桥合同；不导入真实TensorRT或访问GPU。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from utils import tensorrt_runtime as runtime


class Tensor:
    def __init__(self, harness, shape, dtype=torch.float32, device='cuda:0', contiguous=True):
        self.harness, self.shape, self.dtype = harness, tuple(shape), dtype
        self.device = torch.device(device)
        self.is_contiguous = contiguous
        harness.pointer += 1
        self.pointer = harness.pointer
        harness.tensors[self.pointer] = self

    def contiguous(self):
        if self.is_contiguous:
            return self
        return Tensor(self.harness, self.shape, self.dtype, str(self.device))

    def data_ptr(self):
        return self.pointer


class Context:
    def __init__(self, harness, engine):
        self.h, self.engine = harness, engine
        self.shapes, self.addresses, self.bound_refs = {}, {}, {}

    def set_optimization_profile_async(self, profile, stream):
        self.h.events.append(('profile', profile, stream))
        return self.h.failure != 'profile'

    def set_input_shape(self, name, shape):
        self.h.events.append(('shape', name, shape))
        self.shapes[name] = shape
        return self.h.failure != 'shape'

    def set_tensor_address(self, name, address):
        self.h.events.append(('bind', name))
        self.addresses[name] = address
        self.bound_refs[name] = weakref.ref(self.h.tensors[address])
        return self.h.failure != 'address_' + name

    def infer_shapes(self):
        self.h.events.append(('infer',))
        return ['x'] if self.h.failure == 'infer' else []

    def get_tensor_shape(self, name):
        if name in self.shapes:
            return self.shapes[name]
        declared = self.engine.rows[name]['shape']
        if self.engine.rows[name]['mode'] == 'OUTPUT' and -1 in declared:
            if self.h.failure == 'output_shape':
                return declared
            return (self.shapes['x'][0], *declared[1:])
        return declared

    def execute_async_v3(self, *, stream_handle):
        self.h.events.append(('execute', stream_handle))
        assert set(self.addresses) == set(self.engine.rows)
        assert all(ref() is not None for ref in self.bound_refs.values())
        return self.h.failure != 'execute'


class NativeEngine:
    num_optimization_profiles = 1

    def __init__(self, harness, rows):
        self.h, self.rows = harness, rows
        self.context = Context(harness, self)
        self.num_io_tensors = len(rows)

    def create_execution_context(self):
        return self.context

    def get_tensor_name(self, index):
        return list(self.rows)[index]

    def get_tensor_mode(self, name):
        return self.rows[name]['mode']

    def get_tensor_dtype(self, name):
        return self.rows[name].get('dtype', 'float32')

    def get_tensor_shape(self, name):
        return self.rows[name]['shape']

    def get_tensor_location(self, name):
        return self.rows[name].get('location', 'DEVICE')

    def is_shape_inference_io(self, name):
        return self.rows[name].get('shape_io', False)

    def get_tensor_format(self, name, profile_index):
        assert profile_index == 0
        return self.rows[name].get('format', 'LINEAR')

    def get_tensor_profile_shape(self, name, profile_index):
        assert profile_index == 0
        return self.rows[name]['profile']


@pytest.fixture
def harness(monkeypatch, tmp_path):
    h = SimpleNamespace(path=tmp_path / 'sample.engine', pointer=0,
        tensors=weakref.WeakValueDictionary(), events=[], imports=[], failure=None,
        current_device=0, on_sync=None)
    h.path.write_bytes(b'CPU mock engine bytes')

    def sync():
        h.events.append(('sync',))
        if h.on_sync is not None:
            h.on_sync()
        if h.failure == 'sync':
            raise RuntimeError('deliberate CUDA synchronization failure')

    h.stream = SimpleNamespace(cuda_stream=9876, synchronize=sync)
    fake_torch = SimpleNamespace(device=torch.device,
        is_tensor=lambda value: isinstance(value, Tensor),
        empty=lambda shape, *, dtype, device: Tensor(h, shape, dtype, str(device)),
        cuda=SimpleNamespace(current_device=lambda: h.current_device,
                             current_stream=lambda device: h.stream))
    names = ('float32', 'float16', 'bfloat16', 'int8', 'int32', 'int64', 'uint8', 'bool')
    for name in names:
        setattr(fake_torch, name, getattr(torch, name))
    logger = type('Logger', (), dict(WARNING=1, __init__=lambda self, severity: None))
    h.trt = SimpleNamespace(__version__='10.12.mock', Logger=logger,
        Runtime=lambda log: SimpleNamespace(deserialize_cuda_engine=lambda payload: h.native),
        TensorIOMode=SimpleNamespace(INPUT='INPUT', OUTPUT='OUTPUT'),
        TensorLocation=SimpleNamespace(DEVICE='DEVICE'), TensorFormat=SimpleNamespace(LINEAR='LINEAR'))
    for name in names:
        setattr(h.trt, name, name)

    def load(name):
        h.imports.append(name)
        assert name == 'tensorrt'
        return h.trt

    monkeypatch.setattr(runtime, 'torch', fake_torch)
    monkeypatch.setattr(runtime, 'importlib', SimpleNamespace(import_module=load))

    def make(*, dynamic=False, rows=None, **kwargs):
        if rows is None:
            shape = (-1, 1, 256, 256) if dynamic else (1, 3, 4, 4)
            rows = dict(x=dict(mode='INPUT', shape=shape), y=dict(mode='OUTPUT', shape=shape))
            if dynamic:
                rows['x']['profile'] = ((1, 1, 256, 256), (8, 1, 256, 256), (16, 1, 256, 256))
        h.native = NativeEngine(h, rows)
        return runtime.Engine(h.path, **kwargs)

    h.make = make
    h.tensor = lambda shape=(1, 3, 4, 4), **kw: Tensor(h, shape, **kw)
    return h


def test_plain_import_does_not_import_tensorrt_or_require_cuda():
    command = [sys.executable, '-c',
        "import sys; import utils.tensorrt_runtime; assert 'tensorrt' not in sys.modules"]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)


def test_file_sha_binds_actual_deserialized_bytes_and_mismatch_fails_before_import(harness):
    h = harness
    digest = hashlib.sha256(h.path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='file SHA256 differs'):
        h.make(expected_sha256='0' * 64)
    assert h.imports == []
    engine = h.make(expected_sha256=digest.upper())
    assert engine.sha256 == digest and h.imports == ['tensorrt']
    assert engine.logger is not None and engine.runtime is not None and engine.context is h.native.context
    assert engine.io_signature is engine.signature
    assert json.loads(json.dumps(engine.signature))['x']['dtype'] == 'float32'


def test_static_multiple_outputs_int64_input_and_independent_output_buffers(harness):
    h = harness
    rows = dict(x=dict(mode='INPUT', shape=(1, 3, 4, 4)),
                t=dict(mode='INPUT', shape=(1,), dtype='int64'),
                y=dict(mode='OUTPUT', shape=(1, 1, 4, 4)),
                feature=dict(mode='OUTPUT', shape=(1, 2, 2, 2)))
    engine = h.make(rows=rows)
    inputs = dict(x=h.tensor(), t=h.tensor((1,), dtype=torch.int64))
    a, b = engine(**inputs), engine(**inputs)
    assert set(a) == {'y', 'feature'}
    assert a['y'].shape == (1, 1, 4, 4) and a['feature'].shape == (1, 2, 2, 2)
    assert a['y'].data_ptr() != b['y'].data_ptr()
    assert a['feature'].data_ptr() != b['feature'].data_ptr()
    assert engine.execution_count == 2 and engine.last_execution['synchronized']
    assert engine.last_execution['inputs']['t']['dtype'] == 'int64'
    assert [event[0] for event in h.events].count('profile') == 1
    json.dumps(engine.last_execution)


def test_dynamic_NAF_batch_min_max_and_tail_change_without_padding(harness):
    h = harness
    engine = h.make(dynamic=True)
    assert engine.signature['x']['profile']['max'] == [16, 1, 256, 256]
    for batch in (1, 16, 10, 15):
        output = engine(x=h.tensor((batch, 1, 256, 256)))
        assert output['y'].shape == (batch, 1, 256, 256)
        assert engine.last_execution['inputs']['x']['shape'][0] == batch
    assert len([e for e in h.events if e[0] == 'shape']) == 4
    assert len([e for e in h.events if e[0] == 'profile']) == 1


@pytest.mark.parametrize('invalid', ['missing', 'extra', 'not_tensor', 'dtype', 'rank',
                                   'static_shape', 'cpu', 'other_gpu', 'current_gpu'])
def test_invalid_inputs_fail_before_enqueue_or_cuda_work(harness, invalid):
    h = harness
    engine = h.make()
    inputs = dict(x=h.tensor())
    if invalid == 'missing': inputs.clear()
    elif invalid == 'extra': inputs['unused'] = h.tensor()
    elif invalid == 'not_tensor': inputs['x'] = [1]
    elif invalid == 'dtype': inputs['x'] = h.tensor(dtype=torch.float16)
    elif invalid == 'rank': inputs['x'] = h.tensor((3, 4, 4))
    elif invalid == 'static_shape': inputs['x'] = h.tensor((1, 3, 4, 5))
    elif invalid == 'cpu': inputs['x'] = h.tensor(device='cpu')
    elif invalid == 'other_gpu': inputs['x'] = h.tensor(device='cuda:1')
    elif invalid == 'current_gpu': h.current_device = 1
    with pytest.raises((ValueError, TypeError)):
        engine(**inputs)
    assert h.events == [] and engine.execution_count == 0 and engine.last_execution is None


@pytest.mark.parametrize('shape,match', [((17, 1, 256, 256), 'min/max'),
                                       ((0, 1, 256, 256), 'shape differs'),
                                       ((4, 1, 128, 256), 'shape differs')])
def test_dynamic_profile_and_static_spatial_bounds_are_strict(harness, shape, match):
    h = harness
    engine = h.make(dynamic=True)
    with pytest.raises(ValueError, match=match):
        engine(x=h.tensor(shape))
    assert h.events == []


def test_contiguous_copy_and_outputs_stay_alive_until_current_stream_sync(harness):
    h = harness
    engine = h.make()
    source = h.tensor(contiguous=False)

    def check_alive():
        refs = h.native.context.bound_refs
        assert set(refs) == {'x', 'y'} and all(ref() is not None for ref in refs.values())
        assert refs['x']().data_ptr() != source.data_ptr()

    h.on_sync = check_alive
    result = engine(x=source)
    assert h.events[-2:] == [('execute', 9876), ('sync',)]
    assert h.native.context.bound_refs['x']() is None
    assert h.native.context.bound_refs['y']() is result['y']


@pytest.mark.parametrize('failure', ['profile', 'shape', 'address_x', 'address_y',
                                   'infer', 'output_shape', 'execute', 'sync'])
def test_failures_always_sync_before_releasing_buffers_and_never_fallback(harness, failure):
    h = harness
    engine = h.make(dynamic=True)
    h.failure = failure

    def alive_on_failure():
        assert all(ref() is not None for ref in h.native.context.bound_refs.values())

    h.on_sync = alive_on_failure
    with pytest.raises(RuntimeError):
        engine(x=h.tensor((10, 1, 256, 256), contiguous=False))
    assert h.events[-1] == ('sync',)
    assert engine.execution_count == 0 and engine.last_execution is None
    assert len([e for e in h.events if e[0] == 'execute']) == (1 if failure in ('execute', 'sync') else 0)
    h.failure, h.on_sync = None, None
    assert engine(x=h.tensor((1, 1, 256, 256)))['y'].shape == (1, 1, 256, 256)


@pytest.mark.parametrize('change,match', [('location', 'CUDA device'), ('shape_io', 'Shape tensor'),
                                       ('format', 'LINEAR'), ('dtype', 'dtype')])
def test_unsupported_engine_io_is_rejected_explicitly(harness, change, match):
    rows = dict(x=dict(mode='INPUT', shape=(1, 1, 4, 4)), y=dict(mode='OUTPUT', shape=(1, 1, 4, 4)))
    rows['x'][change] = dict(location='HOST', shape_io=True, format='CHW32', dtype='float8')[change]
    with pytest.raises((ValueError, TypeError), match=match):
        harness.make(rows=rows)


@pytest.mark.parametrize('profile_index', [-1, 1, True])
def test_profile_index_is_bounded_without_silent_selection(harness, profile_index):
    with pytest.raises(ValueError, match='profile_index'):
        harness.make(profile_index=profile_index)


def test_invalid_profile_static_dimension_is_rejected_at_load(harness):
    rows = dict(x=dict(mode='INPUT', shape=(-1, 1, 256, 256),
                       profile=((1, 1, 128, 256), (8, 1, 256, 256), (16, 1, 256, 256))),
                y=dict(mode='OUTPUT', shape=(-1, 1, 256, 256)))
    with pytest.raises(ValueError, match='static input dimension'):
        harness.make(rows=rows)
