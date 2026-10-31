# -*- coding: utf-8 -*-
"""固定布局边界、显式不支持的环境及随包GPU程序完整性。"""
import hashlib

import pytest

from utils import restoration_noise as noise


@pytest.mark.parametrize('size,stride', [(0, 0), (1, 256), (256, 256), (257, 512),
                                       (196608, 196608), (3145728, 196608)])
def test_reference_layout_depends_on_tensor_size_only(size, stride):
    assert noise.reference_stride(size) == stride


@pytest.mark.parametrize('size', [-1, 1 << 63])
def test_reference_size_rejects_overflow(size):
    with pytest.raises(ValueError):
        noise.reference_stride(size)


def test_negative_seed_matches_torch_unsigned_seed_conversion():
    assert noise._shape_seed((1, 3, 1024, 1024), -1) == ((1, 3, 1024, 1024), (1 << 64)-1)
    assert noise._shape_seed((), 1) == ((), 1)


@pytest.mark.parametrize('shape,seed', [((-1,), 1), ((1,), 1 << 64), ((1,), -(1 << 63)-1)])
def test_invalid_noise_inputs_fail_before_driver_load(shape, seed):
    with pytest.raises(ValueError):
        noise.restoration_noise(shape, seed, 'cpu')


def test_cpu_is_not_a_silent_alternative_rng():
    with pytest.raises(RuntimeError, match='no CPU/legacy fallback'):
        noise.restoration_noise((1, 3, 4, 4), 314159, 'cpu')


def test_shipped_ptx_matches_version_and_contains_portable_target():
    data = noise.PTX_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == noise.PTX_SHA256
    assert b'.target sm_70' in data and b'restoration_normal' in data
    assert noise.noise_protocol()['reference_grid_limit'] == 768
