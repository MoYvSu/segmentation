# -*- coding: utf-8 -*-
"""连续概率融合的端点、坐标和错误输入防护。真实完整输出复现由GPU诊断验证。"""
import numpy as np
import pytest

from tools.probe_affinity_scale_fusion import affinity_head, fuse_probability


def test_continuous_probability_not_thresholded():
    whole = np.array([[.2, .6], [.8, .99]], np.float32)
    local = np.array([[.6, .8], [.2, .01]], np.float32)
    np.testing.assert_array_equal(fuse_probability(whole, local, 0), whole)
    np.testing.assert_array_equal(fuse_probability(whole, local, 1), local)
    np.testing.assert_allclose(fuse_probability(whole, local), [[.4, .7], [.5, .5]])


@pytest.mark.parametrize('weight', [-1, 1.01, float('nan')])
def test_invalid_weight(weight):
    with pytest.raises(ValueError, match='weight'):
        fuse_probability(np.zeros((2, 3)), np.zeros((2, 3)), weight)


def test_shape_or_nonfinite_maps_rejected():
    with pytest.raises(ValueError, match='two-dimensional'):
        fuse_probability(np.zeros((2, 3)), np.zeros((3, 2)))
    for value in (float('nan'), 1.01, -.01):
        with pytest.raises(ValueError, match='probability'):
            fuse_probability(np.zeros((2, 3)), np.full((2, 3), value))


def test_shared_model_restored_after_failed_local_inference():
    class Model:
        affinity_decoder = object()
    model = Model()
    original = model.affinity_decoder
    head = object()
    with pytest.raises(RuntimeError):
        with affinity_head(model, head):
            assert model.affinity_decoder is head
            raise RuntimeError('local failure')
    assert model.affinity_decoder is original
