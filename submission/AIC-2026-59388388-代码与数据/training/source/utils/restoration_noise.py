# -*- coding: utf-8 -*-
"""固定D5a首步噪声的像素定位，不依赖当前显卡的SM数量。

随包PTX是CUDA驱动可即时编译的GPU程序，最低目标为Volta（compute_70）。
部署无需NVCC、NVRTC或C++编译器；需支持CUDA12.8的NVIDIA驱动及CUDA版PyTorch。
仅复现fresh CUDA generator offset=0的float32 normal4，不替代通用torch.randn。
"""
from __future__ import annotations

import ctypes as C
import hashlib
import math
import operator
import os
from pathlib import Path
import threading

import torch

NOISE_MODE = 'cuda_philox4090_v1'
LEGACY_MODE = 'legacy_torch'
REFERENCE_BLOCK = 256
REFERENCE_GRID_LIMIT = 768
PTX_PATH = Path(__file__).resolve().parent / 'cuda' / 'restore_noise.ptx'
# 由CUDA12.8 NVRTC生成；内容摘要同时约束版本与编译选项（fmad=true）。
PTX_SHA256 = 'e89e8641e977ec28952576d37a7a54ebaacb66754567f3e521dd1637e689afaa'
_LOCK = threading.RLock()
_DRIVER = None
_MODULES = {}  # context -> (module, function)，一直存活至进程结束，避免异步kernel被卸载。


def noise_protocol(mode=NOISE_MODE):
    if mode == LEGACY_MODE:
        return dict(name=mode, dependency='pytorch_device_launch_layout')
    if mode != NOISE_MODE:
        raise ValueError(f'Unknown restoration noise protocol: {mode!r}')
    return dict(name=mode, reference_block=REFERENCE_BLOCK,
                reference_grid_limit=REFERENCE_GRID_LIMIT, generator_offset=0,
                storage='float32_contiguous', ptx_target='compute_70', ptx_sha256=PTX_SHA256)


def configure_restoration_noise(restorer, mode):
    """显式开启推理协议；普通Python属性不改变权重/训练检查点。"""
    noise_protocol(mode)
    restorer.inference_noise_mode = mode


def reference_stride(numel):
    numel = operator.index(numel)
    if numel < 0 or numel > (1 << 63) - 1:
        raise ValueError('Noise size must fit a nonnegative signed int64')
    return REFERENCE_BLOCK * min((numel + REFERENCE_BLOCK - 1) // REFERENCE_BLOCK,
                                 REFERENCE_GRID_LIMIT)


def _shape_seed(shape, seed):
    dimensions = tuple(operator.index(d) for d in shape)
    if any(d < 0 for d in dimensions):
        raise ValueError('Noise shape contains a negative dimension')
    reference_stride(math.prod(dimensions))
    seed = operator.index(seed)
    if not -(1 << 63) <= seed < (1 << 64):
        raise ValueError('Noise seed must be in the torch.manual_seed int64/uint64 range')
    return dimensions, seed & ((1 << 64) - 1)


def _check(driver, code, operation):
    if code:
        error = C.c_char_p()
        driver.cuGetErrorString(code, C.byref(error))
        detail = error.value.decode('utf8', 'replace') if error.value else str(code)
        raise RuntimeError(f'{NOISE_MODE}: {operation} failed ({code}: {detail}). '
                           'Check the NVIDIA driver/CUDA12.8 runtime; no legacy RNG fallback is used.')


def _driver():
    global _DRIVER
    with _LOCK:
        if _DRIVER is None:
            try:
                driver = C.WinDLL('nvcuda.dll') if os.name == 'nt' else C.CDLL('libcuda.so.1')
            except OSError as exc:
                raise RuntimeError('Fixed restoration noise requires an NVIDIA CUDA driver') from exc
            signatures = {
                'cuGetErrorString': [C.c_int, C.POINTER(C.c_char_p)],
                'cuCtxGetCurrent': [C.POINTER(C.c_void_p)],
                'cuModuleLoadData': [C.POINTER(C.c_void_p), C.c_void_p],
                'cuModuleGetFunction': [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                'cuLaunchKernel': [C.c_void_p] + [C.c_uint] * 7 +
                                 [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p],
            }
            for name, arguments in signatures.items():
                function = getattr(driver, name)
                function.argtypes, function.restype = arguments, C.c_int
            _DRIVER = driver
        return _DRIVER


def _kernel(driver):
    context = C.c_void_p()
    _check(driver, driver.cuCtxGetCurrent(C.byref(context)), 'cuCtxGetCurrent')
    if not context.value:
        raise RuntimeError('PyTorch CUDA context must be initialized before noise generation')
    with _LOCK:
        if context.value not in _MODULES:
            ptx = PTX_PATH.read_bytes()
            if hashlib.sha256(ptx).hexdigest() != PTX_SHA256:
                raise RuntimeError('Fixed restoration noise PTX differs from the versioned source build')
            data = C.create_string_buffer(ptx)
            module, function = C.c_void_p(), C.c_void_p()
            _check(driver, driver.cuModuleLoadData(C.byref(module), data), 'cuModuleLoadData')
            _check(driver, driver.cuModuleGetFunction(C.byref(function), module,
                                                     b'restoration_normal'), 'cuModuleGetFunction')
            _MODULES[context.value] = (module, function)
        return _MODULES[context.value][1]


def restoration_noise(shape, seed, device, *, _launch_grid=128, _launch_block=256):
    """生成连续float32 CUDA噪声；私有launch参数只供验证执行网格不影响结果。"""
    dimensions, unsigned_seed = _shape_seed(shape, seed)
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('cuda_philox4090_v1 requires a CUDA NVIDIA GPU; no CPU/legacy fallback')
    grid, block = operator.index(_launch_grid), operator.index(_launch_block)
    if not 1 <= grid <= 65535 or not 1 <= block <= 1024:
        raise ValueError('Invalid CUDA noise launch dimensions')
    with torch.cuda.device(device):
        if torch.cuda.get_device_capability(device) < (7, 0):
            raise RuntimeError('Versioned noise PTX requires NVIDIA compute capability >= 7.0')
        out = torch.empty(dimensions, dtype=torch.float32, device=device)
        if not out.numel():
            return out
        driver = _driver()
        function = _kernel(driver)
        out_ptr = C.c_void_p(out.data_ptr())
        length, key = C.c_longlong(out.numel()), C.c_ulonglong(unsigned_seed)
        stride = C.c_longlong(reference_stride(out.numel()))
        params = (C.c_void_p * 4)(*[C.addressof(value) for value in (out_ptr, length, key, stride)])
        stream = C.c_void_p(torch.cuda.current_stream(out.device).cuda_stream)
        _check(driver, driver.cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0,
                                             stream, params, None), 'cuLaunchKernel')
        return out
