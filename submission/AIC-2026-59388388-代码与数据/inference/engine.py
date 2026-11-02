# -*- coding: utf-8 -*-
"""命名CUDA张量与TensorRT引擎之间的独立运行桥。"""
from __future__ import annotations

import importlib
from pathlib import Path

import torch


class EngineCompatibilityError(RuntimeError):
    """TensorRT明确报告当前平台、GPU或版本与引擎不兼容。"""


def load_logger(trt):
    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.errors = []
            self.forward = trt.Logger(trt.Logger.WARNING)

        def log(self, severity, message):
            if severity <= trt.ILogger.ERROR:
                self.errors.append(message)
            self.forward.log(severity, message)

    return Logger()


class Engine:
    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError('Engine file not found: ' + str(path))
        trt = importlib.import_module('tensorrt')
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.logger = load_logger(trt)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            details = '\n'.join(self.logger.errors)
            incompatible = any(text in details.lower() for text in (
                'platform specific tag mismatch',
                'only supported on the target runtime platform',
                'generated on an incompatible device',
                'generated on an incompatible platform',
                'expecting compute',
                'version tag does not match',
                'not compatible with this version of tensorrt',
                'expecting library version',
            ))
            error = EngineCompatibilityError if incompatible else RuntimeError
            raise error('Could not deserialize TensorRT engine: ' + str(path) + '\n' + details)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create TensorRT execution context')
        self.inputs, self.outputs, self.signature, self.dtypes = [], [], {}, {}
        dtype_map = {trt.float32:torch.float32, trt.int64:torch.int64}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode not in (trt.TensorIOMode.INPUT, trt.TensorIOMode.OUTPUT):
                raise ValueError('Unsupported engine I/O mode: ' + name)
            if (self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                    or self.engine.is_shape_inference_io(name)
                    or self.engine.get_tensor_format(name, 0) != trt.TensorFormat.LINEAR):
                raise ValueError('Engine requires unsupported host/shape/nonlinear I/O: ' + name)
            dtype = self.engine.get_tensor_dtype(name)
            if dtype not in dtype_map:
                raise TypeError('Engine I/O must be FP32 or INT64: ' + name)
            shape = tuple(self.engine.get_tensor_shape(name))
            bounds = None
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
                if -1 in shape:
                    bounds = tuple(tuple(row) for row in self.engine.get_tensor_profile_shape(name, 0))
            else:
                self.outputs.append(name)
            self.signature[name] = (shape, bounds)
            self.dtypes[name] = dtype_map[dtype]
        if not self.inputs or not self.outputs:
            raise ValueError('Engine has no input or output tensors')
        self.profile_selected = False

    def __call__(self, **inputs):
        if set(inputs) != set(self.inputs):
            raise ValueError('Engine input names differ: ' + ', '.join(self.inputs))
        if torch.cuda.current_device() != self.device.index:
            raise ValueError('Current CUDA device differs from the engine device')
        for name, value in inputs.items():
            if not torch.is_tensor(value) or value.device != self.device or value.dtype != self.dtypes[name]:
                raise TypeError('Engine input tensor device/dtype differs: ' + name)
            declared, bounds = self.signature[name]
            actual = tuple(value.shape)
            if len(actual) != len(declared) or any(size < 1 or (fixed != -1 and size != fixed)
                                                   for size, fixed in zip(actual, declared)):
                raise ValueError('Engine input shape differs: ' + name)
            if bounds is not None and any(size < low or size > high
                                          for size, low, high in zip(actual, bounds[0], bounds[2])):
                raise ValueError('Engine input exceeds its dynamic profile: ' + name)
        stream = torch.cuda.current_stream(self.device)
        buffers, outputs = {}, {}
        try:
            if not self.profile_selected:
                if not self.context.set_optimization_profile_async(0, stream.cuda_stream):
                    raise RuntimeError('Could not select engine profile')
                self.profile_selected = True
            for name in self.inputs:
                buffers[name] = inputs[name].contiguous()
                if -1 in self.signature[name][0]:
                    if not self.context.set_input_shape(name, tuple(buffers[name].shape)):
                        raise RuntimeError('Could not set engine input shape: ' + name)
                if not self.context.set_tensor_address(name, int(buffers[name].data_ptr())):
                    raise RuntimeError('Could not bind engine input: ' + name)
            unresolved = self.context.infer_shapes()
            if unresolved:
                raise RuntimeError('Unresolved engine shapes: ' + ', '.join(unresolved))
            for name in self.outputs:
                shape = tuple(self.context.get_tensor_shape(name))
                declared = self.signature[name][0]
                if len(shape) != len(declared) or any(size < 1 or (fixed != -1 and size != fixed)
                                                      for size, fixed in zip(shape, declared)):
                    raise RuntimeError('Invalid engine output shape: ' + name)
                outputs[name] = torch.empty(shape, dtype=self.dtypes[name], device=self.device)
                if not self.context.set_tensor_address(name, int(outputs[name].data_ptr())):
                    raise RuntimeError('Could not bind engine output: ' + name)
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError('TensorRT execution failed')
        finally:
            # 临时输入和独立输出都保留至当前stream完成；没有原模型回退。
            stream.synchronize()
        return outputs
