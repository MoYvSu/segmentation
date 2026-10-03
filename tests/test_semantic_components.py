# -*- coding: utf-8 -*-
"""CPU审查真实SemanticChallenger的只读组件捕获和固定实例末端贡献诊断。"""
import ast
from copy import deepcopy
import hashlib
import inspect
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tools import probe_semantic_components as probe
from tools.probe_semantic_fixed import copy_fixed_mask, file_sha, probability_details
from tools.probe_semantic_native import verify_frozen_model, verify_whole_baseline
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import tensor_digest
from utils.affinity_deployment import crop_letterbox_output
from utils.config import load_config
from utils.semantic_challenger import SemanticChallenger


class EchoFPN(nn.Module):
    def forward(self, features):
        return features[0]


class ActualRGBDelta(nn.Module):
    """使用真实RGB/特征/粗logit的CPU小模块，外层运算来自现行SemanticChallenger。"""
    max_logit_delta = 2.

    def __init__(self):
        super().__init__()
        self.register_buffer('gain', torch.tensor(.6, dtype=torch.float32))

    def forward(self, feature, image, coarse_logits):
        coarse = F.interpolate(coarse_logits, size=image.shape[-2:], mode='bilinear', align_corners=True)
        return self.max_logit_delta*torch.tanh((image[:, :1]-.5)*self.gain+coarse*.13)


def fixture_head(grid=6, network=24):
    branch = nn.Conv2d(2, 1, 1)
    with torch.no_grad():
        branch.weight.copy_(torch.tensor([[[[1.7]], [[-.8]]]]))
        branch.bias.fill_(-.1)
    scaffold = SimpleNamespace(seg_fpn=EchoFPN(), seg_branch=branch)
    head = SemanticChallenger(scaffold, ActualRGBDelta(), 'highres_v1').eval().requires_grad_(False)
    y, x = torch.meshgrid(torch.linspace(-1, 1, grid), torch.linspace(-1, 1, grid), indexing='ij')
    features = [torch.stack((x, y))[None]]
    yy, xx = torch.meshgrid(torch.linspace(0, 1, network), torch.linspace(0, 1, network), indexing='ij')
    image = torch.stack((xx, yy, (xx+yy)/2))[None]
    return head, features, image


def _actual_functions(head, features):
    """抽出并执行新入口的真实capture/forward闭包，CPU替代encoder而非重写工具。"""
    tree = ast.parse(inspect.getsource(probe.run)).body[0]
    body = next(node for node in tree.body if isinstance(node, ast.Try)).body
    actual = [deepcopy(next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name))
              for name in ('capture', 'forward')]
    factory = ast.parse('def factory():\n    captured=[]\n    calls=0\n    return None\n').body[0]
    factory.body[2:3] = actual+[ast.parse('return capture, forward, lambda: calls, captured').body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    model = SimpleNamespace(encoder=lambda image:features, semantic_decoder=head)
    env = dict(model=model, residual=head.semantic_residual, component_grids=probe.component_grids,
               validate_native_components=probe.validate_native_components,
               logit_statistics=probe.logit_statistics,
               crop_letterbox_output=crop_letterbox_output, np=np)
    exec(compile(module, '<actual-semantic-components-capture>', 'exec'), env)
    return env['factory']()


def test_real_challenger_hook_captures_true_components_without_changing_full_output():
    head, features, image = fixture_head()
    initial = tensor_digest(head.state_dict().items())
    capture, _, _, seen = _actual_functions(head, features)
    with torch.no_grad():
        plain = head(features, image)
        hook = head.semantic_residual.register_forward_hook(capture, with_kwargs=True)
        try:
            hooked = head(features, image)
            assert len(seen) == 1
            coarse, delta = seen[0]
            actual_coarse = head.seg_branch(head.seg_fpn(features))
            actual_delta = head.semantic_residual(head.seg_fpn(features), image, coarse_logits=actual_coarse)
            assert torch.equal(coarse, actual_coarse) and torch.equal(delta, actual_delta)
            assert torch.equal(plain, hooked)
            up = probe.component_grids(coarse, delta, hooked)
            assert torch.equal(up+delta, hooked)
            assert coarse.data_ptr() != actual_coarse.data_ptr() and delta.data_ptr() != actual_delta.data_ptr()
            assert not coarse.requires_grad and not delta.requires_grad
        finally:
            hook.remove()
        unhooked = head(features, image)
        assert torch.equal(unhooked, hooked) and torch.equal(unhooked, plain)
    assert not head.semantic_residual._forward_hooks
    verify_frozen_model(head, initial)


def test_capture_hook_rejects_wrong_module_or_missing_real_kwargs():
    head, features, image = fixture_head()
    capture, _, _, seen = _actual_functions(head, features)
    for module, args, kwargs in [
        (head, (features[0], image), {'coarse_logits':torch.zeros(1,1,6,6)}),
        (head.semantic_residual, (features[0], image), {}),
        (head.semantic_residual, (features[0],), {'coarse_logits':torch.zeros(1,1,6,6)}),
    ]:
        with pytest.raises(RuntimeError):
            capture(module, args, kwargs, torch.zeros(1,1,24,24))
    assert seen == []


@pytest.mark.parametrize('mutation', ['actual_ulp', 'delta_changed', 'wrong_dtype', 'wrong_channels', 'nonfinite'])
def test_component_reconstruction_gate_requires_exact_actual_FP32_values(mutation):
    head, features, image = fixture_head()
    with torch.no_grad():
        coarse = head.seg_branch(head.seg_fpn(features))
        delta = head.semantic_residual(head.seg_fpn(features), image, coarse_logits=coarse)
        actual = head(features, image)
    if mutation == 'actual_ulp':
        actual = actual.clone()
        actual[0,0,0,0] = torch.nextafter(actual[0,0,0,0], torch.tensor(float('inf')))
    elif mutation == 'delta_changed':
        delta = delta+1.
    elif mutation == 'wrong_dtype':
        coarse = coarse.double()
    elif mutation == 'wrong_channels':
        coarse = coarse.repeat(1,2,1,1)
    else:
        delta = delta.clone(); delta[0,0,0,0] = float('nan')
    with pytest.raises(RuntimeError):
        probe.component_grids(coarse, delta, actual)


def test_actual_forward_preserves_two_stage_interpolation_and_native_component_linearity():
    head, features, image = fixture_head(grid=256, network=1024)
    initial = tensor_digest(head.state_dict().items())
    capture, forward, calls, seen = _actual_functions(head, features)
    hook = head.semantic_residual.register_forward_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            probabilities, delta_native, receipt = forward(image, 512, 0, (31,47))
            coarse, delta = seen[0]
            coarse_1024 = F.interpolate(coarse, size=(1024,1024), mode='bilinear', align_corners=True)
            expected_coarse = crop_letterbox_output(coarse_1024,1024,512,0,(31,47)).cpu().sigmoid()[0,0].numpy()
            actual_full = head(features,image)
            expected_full = crop_letterbox_output(actual_full,1024,512,0,(31,47)).cpu().sigmoid()[0,0].numpy()
            wrong_one_stage = crop_letterbox_output(coarse,1024,512,0,(31,47)).cpu().sigmoid()[0,0].numpy()
        np.testing.assert_array_equal(probabilities['coarse'], expected_coarse)
        np.testing.assert_array_equal(probabilities['full'], expected_full)
        assert np.max(np.abs(probabilities['coarse']-wrong_one_stage)) > 1e-4
        assert receipt['coarse_grid'] == [1,1,256,256] and receipt['delta_grid'] == [1,1,1024,1024]
        assert receipt['reconstructed_actual_grid_exact'] is True
        assert receipt['native_linear_reconstruction_max_error'] <= 1e-5
        assert delta_native.shape == (31,47) and delta_native.dtype == np.float32
        assert all(p.dtype == np.float32 and p.flags.c_contiguous for p in probabilities.values())
        assert calls() == 1
    finally:
        hook.remove()
    verify_frozen_model(head, initial)


def test_fixed_PNG_bytes_and_strict_mean_probability_vote_are_preserved(tmp_path):
    ids = np.array([[0,1,1,1,65535]],np.uint16)
    before = ids.copy()
    path = tmp_path/'synthetic_inst.png'; copied = tmp_path/'copy_inst.png'
    assert cv2.imwrite(str(path),ids)
    copy_fixed_mask(path,copied)
    assert file_sha(path) == file_sha(copied)
    np.testing.assert_array_equal(cv2.imread(str(copied),cv2.IMREAD_UNCHANGED),ids)
    probability = torch.tensor([[0.,8.,8.,-20.,0.]],dtype=torch.float32).sigmoid().numpy()
    classes,_ = revote_instances(ids,probability)
    assert classes == {'1':1,'65535':0}
    sha = hashlib.sha256(probability.tobytes()).hexdigest()
    assert verify_whole_baseline('synthetic',probability,sha,ids,classes)[0] == classes
    wrong = probability.copy(); wrong[0,1] = np.nextafter(wrong[0,1],np.float32(1.))
    with pytest.raises(RuntimeError,match='SHA'):
        verify_whole_baseline('synthetic',wrong,sha,ids,classes)
    with pytest.raises(RuntimeError,match='classes'):
        verify_whole_baseline('synthetic',probability,sha,ids,{'1':0,'65535':0})
    np.testing.assert_array_equal(ids,before)


def test_delta_distributions_and_per_instance_report_are_JSON_native_and_do_not_change_ids():
    ids = np.array([[0,1,1,1,65535],[0,1,1,1,65535]],np.uint16)
    before = ids.copy()
    delta = np.array([[0.,1.,-2.,2.,0.],[0.,1.,-2.,2.,0.]],np.float32)
    coarse = np.full(ids.shape,.5,np.float32)
    full = torch.from_numpy(delta).sigmoid().numpy()
    coarse_classes,coarse_scores = revote_instances(ids,coarse)
    full_classes,full_scores = revote_instances(ids,full)
    coarse_details = probability_details(ids,coarse,coarse_scores)
    full_details = probability_details(ids,full,full_scores)
    rows = probe.instance_deltas(ids,delta,coarse_details,full_details)
    stats = probe.delta_statistics(delta,2.)
    assert coarse_classes == {'1':0,'65535':0} and full_classes == {'1':1,'65535':0}
    assert rows['1']['flipped'] is True and rows['65535']['flipped'] is False
    assert rows['1']['area'] == 6 and sum(row['area'] for row in rows.values()) == int((ids>0).sum())
    assert stats['minimum'] == -2. and stats['maximum'] == 2.
    assert stats['near_bound_fraction'] == .4
    json.dumps({'distribution':stats,'instances':rows},allow_nan=False)
    np.testing.assert_array_equal(ids,before)


@pytest.mark.parametrize('mutation', ['dtype','nonfinite','overbound','nonpositive_bound'])
def test_actual_delta_bound_and_dtype_are_checked(mutation):
    delta = np.zeros((3,5),np.float32); bound = 2.
    if mutation == 'dtype': delta = delta.astype(np.float64)
    elif mutation == 'nonfinite': delta[0,0] = np.nan
    elif mutation == 'overbound': delta[0,0] = 2.1
    else: bound = 0.
    with pytest.raises((RuntimeError,ValueError)):
        probe.delta_statistics(delta,bound)


def test_plan_is_fixed_eight_and_nine_actual_forwards_without_training():
    config = load_config('config/inference/semantic_components.yaml')
    plan = probe.plan(config)
    assert plan['sources'] == list(probe.TRAIN+probe.TEST) and len(plan['sources']) == 8
    assert plan['routes'] == ['full','coarse'] and plan['semantic_forward_calls'] == 9
    assert plan['training'] is False and plan['no_optimizer_steps'] is True
    assert plan['official_submission'] is False and plan['best_deployment_changed'] is False
    assert plan['full_test_inference'] is False
    json.dumps(plan,allow_nan=False)


@pytest.mark.parametrize('key,value', [('routes',['coarse','full']),('correction_limit',1.),('correction_version','lowres_v1')])
def test_other_architecture_or_correction_recipe_is_rejected(key,value):
    config = load_config('config/inference/semantic_components.yaml')
    config['semantic_components'][key] = value
    with pytest.raises(ValueError):
        probe.plan(config)


def test_real_FP32_native_bilinear_roundoff_uses_logit_scale_without_relaxing_actual_grid():
    """高logit真实分支先精确相加，随后各自插值的合法舍入可超过固定1e-5。"""
    head, features, image = fixture_head(grid=6, network=24)
    with torch.no_grad():
        head.seg_branch.weight.mul_(100.)
        head.seg_branch.bias.mul_(100.)
        coarse = head.seg_branch(head.seg_fpn(features))
        delta = head.semantic_residual(head.seg_fpn(features),image,coarse_logits=coarse)
        actual = head(features,image)
        coarse_up = probe.component_grids(coarse,delta,actual)
        full_native = crop_letterbox_output(actual,24,8,0,(17,31))
        coarse_native = crop_letterbox_output(coarse_up,24,8,0,(17,31))
        delta_native = crop_letterbox_output(delta,24,8,0,(17,31))
    error = float((full_native-(coarse_native+delta_native)).abs().max())
    assert error > 1e-5  # 原固定门槛会误拒绝同一真实FP32前向。
    receipt = probe.validate_native_components(full_native,coarse_native,delta_native)
    scale = max(1.,float(coarse_native.abs().max())+float(delta_native.abs().max()))
    epsilon = torch.finfo(torch.float32).eps
    assert receipt['tolerance_operation_factor'] == 32
    assert receipt['fp32_epsilon'] == epsilon and receipt['tolerance_scale'] == scale
    assert receipt['native_linear_tolerance'] == 32*epsilon*scale
    assert receipt['native_linear_reconstruction_max_error'] == error
    assert error <= receipt['native_linear_tolerance']
    json.dumps(receipt,allow_nan=False)
    # 容差仅用于分开的原生插值；原始grid仍必须逐值相等。
    changed_grid = actual.clone()
    changed_grid[0,0,0,0] = torch.nextafter(changed_grid[0,0,0,0],torch.tensor(float('inf')))
    with pytest.raises(RuntimeError,match='actual logit grid'):
        probe.component_grids(coarse,delta,changed_grid)


def test_single_ULP_native_difference_is_accepted_but_material_delta_corruption_is_rejected():
    coarse = torch.full((1,1,7,11),100.,dtype=torch.float32)
    delta = torch.full_like(coarse,.5)
    actual = coarse+delta
    actual[0,0,2,3] = torch.nextafter(actual[0,0,2,3],torch.tensor(float('inf')))
    receipt = probe.validate_native_components(actual,coarse,delta)
    assert receipt['native_linear_reconstruction_max_error'] > 0
    assert receipt['native_linear_reconstruction_max_error'] <= receipt['native_linear_tolerance']
    corrupted = delta.clone(); corrupted[0,0,2,3] += .05
    with pytest.raises(RuntimeError):
        probe.validate_native_components(actual,coarse,corrupted)
    tiny = torch.zeros((1,1,3,4),dtype=torch.float32)
    with pytest.raises(RuntimeError):
        probe.validate_native_components(tiny,tiny,tiny+.001)


@pytest.mark.parametrize('mutation', ['dtype','shape','channels','nonfinite'])
def test_scale_aware_native_gate_keeps_finite_FP32_native_tensor_contract(mutation):
    actual = torch.zeros((1,1,3,4),dtype=torch.float32)
    coarse = actual.clone(); delta = actual.clone()
    if mutation == 'dtype': coarse = coarse.double()
    elif mutation == 'shape': delta = delta[:,:,:2]
    elif mutation == 'channels': actual = actual.repeat(1,2,1,1)
    else: delta[0,0,0,0] = float('nan')
    with pytest.raises(RuntimeError):
        probe.validate_native_components(actual,coarse,delta)


def test_logit_range_receipt_is_explicit_JSON_native_and_read_only():
    value = torch.tensor([[[[-30.,-10.,0.,10.,30.]]]],dtype=torch.float32)
    before = value.clone()
    receipt = probe.logit_statistics(value)
    assert receipt['shape'] == [1,1,1,5]
    assert receipt['minimum'] == -30. and receipt['maximum'] == 30.
    assert receipt['mean'] == 0. and receipt['absolute_mean'] == 16.
    assert receipt['abs_ge10_fraction'] == .8 and receipt['abs_ge20_fraction'] == .4
    json.dumps(receipt,allow_nan=False)
    assert torch.equal(value,before)
