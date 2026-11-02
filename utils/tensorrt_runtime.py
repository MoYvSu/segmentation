# -*- coding: utf-8 -*-
"""TensorRT命名CUDA张量运行桥；不导出引擎、不改精度、不提供回退。

仅在Engine构造时导入TensorRT。每个实例持有独立execution context，调用
同步完成后才返回，因此输入临时连续副本和本次独立输出在执行中始终存活。
支持LINEAR设备张量和由输入形状确定的输出，不接受host/shape tensor I/O。
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import re
from threading import Lock

import torch


FORMAT = 'named_cuda_tensorrt_runtime_v1'


class Engine:
    """Engine(path, SHA)(**named_cuda_inputs) -> 独立CUDA输出字典。

    在当前CUDA设备加载引擎，此后输入和当前设备必须与它一致。signature
    与last_execution仅含可写入JSON的元数据，不持有历次输入或输出张量。
    """

    def __init__(self, path, expected_sha256=None, *, profile_index=0):
        self.path = Path(path).resolve(strict=True)
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None:
            if (not isinstance(expected_sha256, str)
                    or re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha256) is None):
                raise ValueError('Expected engine SHA256 must contain 64 hex digits')
            if self.sha256 != expected_sha256.lower():
                raise ValueError('TensorRT engine file SHA256 differs')
        if type(profile_index) is not int or profile_index < 0:
            raise ValueError('profile_index must be a nonnegative integer')

        # 延迟依赖：普通CPU测试或旧推理入口import本模块不需要TensorRT。
        self._trt = importlib.import_module('tensorrt')
        trt = self._trt
        self.runtime_version = str(trt.__version__)
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(payload)
        if self.engine is None:
            raise RuntimeError('TensorRT engine deserialization failed')
        if profile_index >= self.engine.num_optimization_profiles:
            raise ValueError('profile_index is outside engine optimization profiles')
        self.profile_index = profile_index
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('TensorRT execution context creation failed')

        dtypes = {}
        for name in ('float32', 'float16', 'bfloat16', 'int8', 'int32', 'int64', 'uint8', 'bool'):
            if hasattr(trt, name) and hasattr(torch, name):
                dtypes[getattr(trt, name)] = getattr(torch, name)
        self.signature, self._dtypes = {}, {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if not isinstance(name, str) or not name or name in self.signature:
                raise ValueError('Engine I/O names must be unique nonempty strings')
            mode = self.engine.get_tensor_mode(name)
            if mode not in (trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT):
                raise ValueError('Unknown TensorRT I/O mode: ' + name)
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError('Only CUDA device I/O tensors are supported: ' + name)
            if self.engine.is_shape_inference_io(name):
                raise ValueError('Shape tensor I/O is unsupported: ' + name)
            if self.engine.get_tensor_format(name, profile_index) != trt.TensorFormat.LINEAR:
                raise ValueError('Only LINEAR tensor format is supported: ' + name)
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in dtypes:
                raise TypeError('Unsupported TensorRT dtype: ' + str(dtype))
            self._dtypes[name] = dtypes[dtype]
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            if any(value == 0 or value < -1 for value in shape):
                raise ValueError('Invalid TensorRT tensor dimensions: ' + name)
            row = dict(mode='INPUT' if mode == trt.TensorIOMode.INPUT else 'OUTPUT',
                       dtype=str(dtypes[dtype]).removeprefix('torch.'), shape=list(shape))
            if row['mode'] == 'INPUT' and -1 in shape:
                bounds = self.engine.get_tensor_profile_shape(name, profile_index)
                if len(bounds) != 3:
                    raise ValueError('Expected min/opt/max profile shapes: ' + name)
                bounds = [tuple(int(value) for value in bound) for bound in bounds]
                if any(len(bound) != len(shape) for bound in bounds):
                    raise ValueError('Profile rank differs from engine input: ' + name)
                for dimension, (low, opt, high) in enumerate(zip(*bounds)):
                    if not 1 <= low <= opt <= high:
                        raise ValueError('Invalid min/opt/max profile bounds: ' + name)
                    if shape[dimension] != -1 and (low != shape[dimension] or opt != low or high != low):
                        raise ValueError('Profile changes a static input dimension: ' + name)
                row['profile'] = dict(zip(('min', 'opt', 'max'), map(list, bounds)))
            self.signature[name] = row
        self.input_names = tuple(name for name, row in self.signature.items() if row['mode'] == 'INPUT')
        self.output_names = tuple(name for name, row in self.signature.items() if row['mode'] == 'OUTPUT')
        if not self.input_names or not self.output_names:
            raise ValueError('Engine needs at least one input and one output')
        self.io_signature = self.signature
        self.last_execution = None
        self.execution_count = 0
        self._profile_selected = False
        self._lock = Lock()

    def _validate_inputs(self, inputs):
        if set(inputs) != set(self.input_names):
            missing = sorted(set(self.input_names) - set(inputs))
            extra = sorted(set(inputs) - set(self.input_names))
            raise ValueError(f'Engine input names differ; missing={missing}, extra={extra}')
        for name, value in inputs.items():
            if not torch.is_tensor(value):
                raise TypeError('Engine input must be a Torch tensor: ' + name)
            if value.device != self.device or value.device.type != 'cuda':
                raise ValueError('Engine input must use its bound CUDA device: ' + name)
            if value.dtype != self._dtypes[name]:
                raise TypeError('Engine input dtype differs: ' + name)
            actual = tuple(value.shape)
            declared = self.signature[name]['shape']
            if len(actual) != len(declared) or any(
                    size < 1 or (fixed != -1 and size != fixed)
                    for size, fixed in zip(actual, declared)):
                raise ValueError('Engine input shape differs: ' + name)
            if 'profile' in self.signature[name]:
                bounds = self.signature[name]['profile']
                if any(size < low or size > high for size, low, high in
                       zip(actual, bounds['min'], bounds['max'])):
                    raise ValueError('Engine input exceeds profile min/max bounds: ' + name)
        if torch.cuda.current_device() != self.device.index:
            raise ValueError('Current CUDA device differs from the loaded engine device')

    def __call__(self, **inputs):
        with self._lock:
            self._validate_inputs(inputs)
            stream = torch.cuda.current_stream(self.device)
            buffers, outputs = {}, {}
            # finally同步也覆盖profile切换、连续副本和enqueue失败；局部字典在
            # 同步完成前不释放，从而避免异步执行仍在使用已回收的输入内存。
            try:
                if not self._profile_selected:
                    if not self.context.set_optimization_profile_async(self.profile_index, stream.cuda_stream):
                        raise RuntimeError('TensorRT optimization profile selection failed')
                    self._profile_selected = True
                for name in self.input_names:
                    buffers[name] = inputs[name].contiguous()
                    if -1 in self.signature[name]['shape']:
                        if not self.context.set_input_shape(name, tuple(buffers[name].shape)):
                            raise RuntimeError('TensorRT input shape selection failed: ' + name)
                    if not self.context.set_tensor_address(name, int(buffers[name].data_ptr())):
                        raise RuntimeError('TensorRT input address binding failed: ' + name)
                unresolved = self.context.infer_shapes()
                if unresolved:
                    raise RuntimeError('TensorRT shape inference unresolved: ' + ', '.join(unresolved))
                for name in self.input_names:
                    if tuple(self.context.get_tensor_shape(name)) != tuple(inputs[name].shape):
                        raise RuntimeError('TensorRT resolved input shape differs: ' + name)
                for name in self.output_names:
                    shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
                    declared = self.signature[name]['shape']
                    if len(shape) != len(declared) or any(size < 1 or (fixed != -1 and size != fixed)
                                                            for size, fixed in zip(shape, declared)):
                        raise RuntimeError('TensorRT output shape unresolved or invalid: ' + name)
                    outputs[name] = torch.empty(shape, dtype=self._dtypes[name], device=self.device)
                    if not self.context.set_tensor_address(name, int(outputs[name].data_ptr())):
                        raise RuntimeError('TensorRT output address binding failed: ' + name)
                if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                    raise RuntimeError('TensorRT execute_async_v3 failed')
            finally:
                stream.synchronize()
            self.execution_count += 1
            self.last_execution = dict(
                format=FORMAT, engine_sha256=self.sha256, profile_index=self.profile_index,
                device=str(self.device), synchronized=True, execution_count=self.execution_count,
                inputs={name: dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix('torch.'))
                        for name, value in inputs.items()},
                outputs={name: dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix('torch.'))
                         for name, value in outputs.items()})
            return outputs
