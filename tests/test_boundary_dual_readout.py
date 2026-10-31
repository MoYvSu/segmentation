# -*- coding: utf-8 -*-
"""双输入零/关闭路径的合成CPU最终读出合同；不读取比赛数据或GT。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch

from models.boundary_dual import BoundaryDualNAFNet
from models.boundary_rgb import BoundaryRGBNAFNet
from tools.boundary_restore_readout import evaluate_frozen_readout
from utils.final_instance_cleanup import merge_final_instances


@pytest.fixture(autouse=True)
def cpu_threads():
    old_torch, old_cv = torch.get_num_threads(), cv2.getNumThreads()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    yield
    torch.set_num_threads(old_torch)
    cv2.setNumThreads(old_cv)


def _fixture(shape):
    h, w = shape
    y, x = np.indices(shape)
    t8 = np.zeros(shape, np.float32)
    for column in (40, 80, 120):
        t8[:, column - 2:column + 3] = 1.
    for row in (40, 80):
        t8[row - 2:row + 3, :] = 1.
    continuous = (.02 + .89 * t8 + .04 * (x / w + y / h)).astype(np.float32)
    raw = np.where(x < w // 2, .8, .2).astype(np.float32)
    semantic = torch.from_numpy(np.where(x < w // 2, 2., -2.).astype(np.float32))[None, None]
    config = {'inference': dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8, marker_border_seal_width=2,
        bridge_width=1, watershed_dilate_width=1, min_instance_area=50,
        seed_min_area=50, final_min_instance_area=200,
        final_small_instance_action='neighbor_merge_v1', max_instance_id=65535,
        threshold=.5, semantic_vote_mode='probability_mean')}
    inputs = torch.from_numpy(np.stack([t8, continuous]))[None]
    kwargs = dict(width=4, encoder_blocks=(1, 1), middle_blocks=1,
                  decoder_blocks=(1, 1), seed=17)
    old = BoundaryRGBNAFNet(**kwargs, input_mode='boundary').eval()
    dual = BoundaryDualNAFNet(**kwargs).eval()
    return inputs, old, dual, semantic, raw, continuous, config


def _pipeline(probability, semantic, raw, reference_n1, config, use_s0):
    # 这里只执行纯图读出，不加载checkpoint，不修改/绕过生产S0模型白名单。
    decode_fn = None
    if use_s0:
        from tools.infer_boundary_rgb import _supported_readout

        def decode_fn(actual_semantic, actual_boundary, actual_config):
            result, _ = _supported_readout(actual_semantic,
                actual_boundary.detach().cpu()[0, 0].numpy(), reference_n1, actual_config)
            return result

    ids, classes, readout_audit = evaluate_frozen_readout(
        semantic, raw, probability, config, decode_fn=decode_fn)
    final_ids, final_classes, cleanup = merge_final_instances(ids, classes, 200)
    assert final_ids.dtype == np.uint16 and final_ids.ndim == 2
    assert int(final_ids.max(initial=0)) <= 65535
    assert set(final_classes) == {str(int(k)) for k in np.unique(final_ids) if k}
    assert cleanup['threshold'] == 200 and cleanup['final_contract_valid']
    return ids, classes, final_ids, final_classes, readout_audit, cleanup


@pytest.mark.parametrize('shape', [(120, 160), (121, 161)])
@pytest.mark.parametrize('use_s0', [False, True])
@pytest.mark.parametrize('disable_learned_branch', [False, True])
def test_random_shared_model_zero_or_disabled_branch_has_exact_final_output(
        shape, use_s0, disable_learned_branch):
    inputs, old, dual, semantic, raw, reference_n1, config = _fixture(shape)
    before_inputs, before_semantic, before_raw = inputs.clone(), semantic.clone(), raw.copy()
    before_reference, before_config = reference_n1.copy(), deepcopy(config)
    if disable_learned_branch:
        # 新分支已有非零权重时关闭，仍应保留原路径。没有改公共随机权重。
        with torch.no_grad():
            dual.continuous_intro.weight.fill_(.125)
    with torch.inference_mode():
        original_probability = old(inputs[:, :1].contiguous())[0, 0].numpy().copy()
        current_probability = dual(inputs,
            use_continuous=not disable_learned_branch)[0, 0].numpy().copy()
    assert np.array_equal(current_probability, original_probability)
    original = _pipeline(original_probability, semantic, raw, reference_n1, config, use_s0)
    current = _pipeline(current_probability, semantic, raw, reference_n1, config, use_s0)
    # 明确比较pre200、最终uint16图和类别表，避免仅测概率/残差兼容。
    assert np.array_equal(current[0], original[0]) and current[1] == original[1]
    assert np.array_equal(current[2], original[2]) and current[3] == original[3]
    assert current[4] == original[4] and current[5] == original[5]
    assert len(current[3]) >= 2  # 合成测试必须真正运行多实例读出，不能空图蒙混通过。
    assert torch.equal(inputs, before_inputs) and torch.equal(semantic, before_semantic)
    assert np.array_equal(raw, before_raw) and np.array_equal(reference_n1, before_reference)
    assert config == before_config


def test_actual_final200_absorption_is_identical_and_inherits_original_target_class():
    # 补充实际接管动作，不依赖随机NAF恰好产生一个<200的小区。
    ids = np.zeros((40, 80), np.uint16)
    ids[1:39, 1:44] = 11
    ids[14:20, 45:49] = 7
    ids[1:39, 52:79] = 30
    classes = {'11': 1, '7': 0, '30': 0}
    before = ids.copy()
    original = merge_final_instances(ids, classes, 200)
    candidate = merge_final_instances(ids.copy(), dict(classes), 200)
    assert original[2]['merged_ids'] == [7]
    assert np.array_equal(candidate[0], original[0]) and candidate[1:] == original[1:]
    assert original[1] == {'11': 1, '30': 0}
    assert np.array_equal(ids, before)
