# -*- coding: utf-8 -*-
"""有GPU时可在其他机器复查；不依赖私有数据或诊断原型。"""
import pytest
import torch

from utils.restoration_noise import restoration_noise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')


@pytest.mark.parametrize('seed', [314159, 314191, -1])
def test_small_noise_exactly_matches_fresh_torch_generator(seed):
    # 小张量仅用两组虚拟线程，在支持的GPU上PyTorch和固定协议布局相同。
    actual = restoration_noise((257,), seed, 'cuda')
    expected = torch.randn((257,), device='cuda', dtype=torch.float32,
                           generator=torch.Generator(device='cuda').manual_seed(seed))
    assert torch.equal(actual, expected)


def test_large_noise_is_independent_of_physical_launch_and_global_generator():
    initial_rng = torch.cuda.get_rng_state().clone()
    actual = restoration_noise((1, 3, 1024, 1024), 314159, 'cuda')
    for grid, block in [(7, 128), (156, 256), (768, 256), (129, 512)]:
        varied = restoration_noise(actual.shape, 314159, 'cuda',
                                   _launch_grid=grid, _launch_block=block)
        assert torch.equal(actual, varied)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        alternate = restoration_noise(actual.shape, 314159, 'cuda') * 2
    stream.synchronize()
    assert torch.equal(alternate, actual * 2)
    assert torch.equal(initial_rng, torch.cuda.get_rng_state())
