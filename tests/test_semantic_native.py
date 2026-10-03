# -*- coding: utf-8 -*-
"""冻结 raw 语义原生视野诊断：裁块、输入布局、连续拼接及固定实例投票。"""
from copy import deepcopy
import ast
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tools import probe_semantic_native as probe
from tools.semantic_crossover import revote_instances
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.config import load_config
from utils.semantic_native import (
    NativeLogitBlend, positive_tent, prepare_native_tile, tile_boxes, tile_starts,
)
from train_backend_adaptation import tensor_digest


def coordinate_rgb(h, w):
    y, x = np.indices((h, w))
    return np.stack(((y*17+x*3) % 256, (x*11) % 256, (y*7+x*13) % 256), axis=-1).astype(np.uint8)


@pytest.mark.parametrize('length,size,overlap,expected', [
    (3, 8, .25, [0]), (8, 8, .25, [0]),
    (18, 8, .25, [0, 6, 10]), (20, 8, 0., [0, 8, 12]),
    (19, 8, .5, [0, 4, 8, 11]),
])
def test_native_windows_anchor_last_full_view_without_duplicates(length, size, overlap, expected):
    assert tile_starts(length, size, overlap) == expected
    assert len(expected) == len(set(expected))


@pytest.mark.parametrize('shape,size', [((17, 23), 8), ((3, 17), 8), ((19, 4), 8), ((4, 6), 8)])
def test_every_native_pixel_is_covered_in_row_major_order(shape, size):
    boxes = tile_boxes(shape, size, .25)
    covered = np.zeros(shape, np.uint32)
    for y0, y1, x0, x1 in boxes:
        assert 0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]
        assert y1-y0 <= size and x1-x0 <= size
        covered[y0:y1, x0:x1] += 1
    assert (covered > 0).all()
    assert boxes == sorted(boxes)
    assert len(boxes) == len(set(boxes))
    assert boxes[-1][1] == shape[0] and boxes[-1][3] == shape[1]


@pytest.mark.parametrize('length,size,overlap', [
    (0, 8, .25), (8, 0, .25), (True, 8, .25), (8, 8, -1.), (8, 8, 1.),
])
def test_bad_window_budget_is_rejected(length, size, overlap):
    with pytest.raises(ValueError):
        tile_starts(length, size, overlap)


@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (8, 1), (5, 8), (9, 11)])
def test_positive_tent_is_symmetric_finite_float32_even_at_image_frame(shape):
    weights = positive_tent(shape)
    assert weights.dtype == np.float32 and weights.flags.c_contiguous
    assert weights.shape == shape and np.isfinite(weights).all()
    assert weights.min() > 0 and weights.max() <= 1
    np.testing.assert_allclose(weights, weights[::-1, ::-1], rtol=0, atol=2e-7)


def test_rgb_is_cropped_at_native_coordinates_before_resize_and_reflect():
    rgb = coordinate_rgb(48, 73)
    before = rgb.copy()
    box = (11, 16, 29, 38)  # 真实视野5×9；网络18，按2倍放大，底部补8。
    tensor, ph, pw, meta = prepare_native_tile(rgb, box, 18, 'cpu')
    clear = rgb[11:16, 29:38]
    scaled = cv2.resize(clear, (18, 10), interpolation=cv2.INTER_LINEAR)
    expected = cv2.copyMakeBorder(scaled, 0, 8, 0, 0, cv2.BORDER_REFLECT)
    expected_tensor = torch.from_numpy(expected).permute(2, 0, 1).float().unsqueeze(0)/255.
    assert torch.equal(tensor, expected_tensor)
    assert tensor.shape == (1, 3, 18, 18) and tensor.dtype == torch.float32
    assert tensor.stride() == expected_tensor.stride() == tuple(meta['input_stride'])
    assert (ph, pw) == (8, 0)
    assert meta['native_shape'] == [5, 9] and meta['resize_scale'] == 2.
    assert meta['box'] == list(box) and meta['network_shape'] == [18, 18]
    assert meta['padding'] == 'bottom/right BORDER_REFLECT'
    np.testing.assert_array_equal(rgb, before)


@pytest.mark.parametrize('shape', [(5, 9), (9, 5), (7, 7)])
def test_whole_native_tile_matches_original_prepare_image_layout_exact(tmp_path, shape):
    rgb = coordinate_rgb(*shape)
    path = tmp_path/'synthetic.png'
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    actual_rgb, baseline, bph, bpw = prepare_image(path, 18, 'cpu')
    local, ph, pw, _ = prepare_native_tile(rgb, (0, shape[0], 0, shape[1]), 18, 'cpu')
    np.testing.assert_array_equal(actual_rgb, rgb)
    assert torch.equal(local, baseline)
    assert local.stride() == baseline.stride() and (ph, pw) == (bph, bpw)


def test_network_padding_is_removed_before_native_stitch():
    rgb = coordinate_rgb(9, 15)
    _, ph, pw, _ = prepare_native_tile(rgb, (2, 7, 3, 12), 18, 'cpu')
    logits = torch.full((1, 1, 18, 18), 999., dtype=torch.float32)
    logits[:, :, :18-ph, :18-pw] = 3.
    native = crop_letterbox_output(logits, 18, ph, pw, (5, 9))[0, 0].numpy()
    blend = NativeLogitBlend((5, 9))
    blend.add((0, 5, 0, 9), native)
    np.testing.assert_allclose(blend.finish(), 3., rtol=0, atol=3e-7)
    assert blend.receipt()['uncovered_pixels'] == 0


@pytest.mark.parametrize('constant', [0., -7., 3.25])
def test_overlap_blending_preserves_constant_logits_everywhere(constant):
    shape = (17, 23)
    blend = NativeLogitBlend(shape)
    for y0, y1, x0, x1 in tile_boxes(shape, 8, .25):
        blend.add((y0, y1, x0, x1), np.full((y1-y0, x1-x0), constant, np.float32))
    actual = blend.finish()
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_allclose(actual, constant, rtol=1e-6, atol=1e-6)
    receipt = blend.receipt()
    assert receipt['uncovered_pixels'] == 0 and receipt['minimum_denominator'] > 0
    assert receipt['tiles'] == len(tile_boxes(shape, 8, .25))
    json.dumps(receipt, allow_nan=False)


def test_coordinate_field_stitch_preserves_native_positions_at_overlaps():
    shape = (17, 23)
    y, x = np.indices(shape)
    field = (y*3-x/7).astype(np.float32)
    blend = NativeLogitBlend(shape)
    for box in tile_boxes(shape, 8, .25):
        y0, y1, x0, x1 = box
        blend.add(box, field[y0:y1, x0:x1])
    np.testing.assert_allclose(blend.finish(), field, rtol=1e-6, atol=5e-6)


def test_overlap_combines_logits_before_sigmoid_not_probability_means():
    blend = NativeLogitBlend((1, 3))
    blend.add((0, 1, 0, 2), np.array([[8., 8.]], np.float32))
    blend.add((0, 1, 1, 3), np.array([[-20., -20.]], np.float32))
    logits = blend.finish()
    assert logits[0, 1] == -6.
    p = torch.from_numpy(logits).sigmoid().numpy()
    assert p[0, 1] < .01
    incorrectly_blended_p = (torch.tensor(8.).sigmoid()+torch.tensor(-20.).sigmoid())/2
    assert float(incorrectly_blended_p) > .49


def test_missing_pixels_duplicate_tiles_or_wrong_logit_dtype_fail_closed():
    blend = NativeLogitBlend((3, 4))
    with pytest.raises(ValueError):
        blend.finish()
    blend.add((0, 2, 0, 4), np.zeros((2, 4), np.float32))
    with pytest.raises(ValueError):
        blend.finish()
    for box, value in [
        ((0, 2, 0, 4), np.zeros((2, 4), np.float32)),
        ((2, 3, 0, 4), np.zeros((1, 4), np.float64)),
        ((2, 3, 0, 4), np.full((1, 4), np.nan, np.float32)),
        ((2, 4, 0, 4), np.zeros((2, 4), np.float32)),
        ((2, 3, 0, 4), np.zeros((1, 3), np.float32)),
    ]:
        with pytest.raises(ValueError):
            blend.add(box, value)


def test_fixed_geometry_vote_uses_mean_probability_strict_threshold_and_uint16_limit():
    ids = np.array([[0, 1, 1, 1, 65535]], np.uint16)
    before = ids.copy()
    logits = torch.tensor([[0., 8., 8., -20., 0.]], dtype=torch.float32)
    probability = logits.sigmoid().numpy()
    classes, scores = revote_instances(ids, probability)
    assert classes == {'1': 1, '65535': 0}  # .5严格归珠光体。
    assert scores['1'] > .5 and logits[0, 1:4].mean() < 0
    np.testing.assert_array_equal(ids, before)
    assert ids.dtype == np.uint16 and ids.max() == 65535


def test_plan_is_bounded_frozen_and_json_serializable():
    config = load_config('config/inference/semantic_native.yaml')
    result = probe.plan(config)
    assert result['sources'] == list(probe.TRAIN+probe.TEST)
    assert len(result['sources']) == 8 and result['routes'] == ['whole', 'native1024', 'native512']
    assert result['training'] is False and result['no_optimizer_steps'] is True
    assert result['formal_submission'] is False and result['full_test_inference'] is False
    assert result['full_baseline_gate'] == 'All eight clean probability SHA and fixed classes exact'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('key,value', [
    ('routes', ['whole', 'native512']), ('train_sources', ['train_999']),
    ('test_sources', ['test_101']), ('network_size', 512), ('overlap', .5),
    ('minimum_axis_weight', 0.), ('native_sizes', [1024, 256]),
])
def test_unregistered_scale_or_cohort_recipe_is_rejected(key, value):
    cfg = deepcopy(load_config('config/inference/semantic_native.yaml')['semantic_native'])
    cfg[key] = value
    with pytest.raises(ValueError):
        probe.validate_recipe(cfg)


def _baseline():
    ids = np.array([[0, 1, 1, 65535]], np.uint16)
    probability = np.array([[.8, .6, .6, .5]], np.float32)
    classes = {'1': 1, '65535': 0}
    return ids, probability, classes, hashlib.sha256(probability.tobytes()).hexdigest()


def test_actual_whole_baseline_gate_requires_exact_probability_and_fixed_classes():
    ids, probability, classes, digest = _baseline()
    actual, _ = probe.verify_whole_baseline('synthetic', probability, digest, ids, classes)
    assert actual == classes
    altered = probability.copy()
    altered[0, 1] = np.nextafter(altered[0, 1], np.float32(1.))
    assert revote_instances(ids, altered)[0] == classes
    with pytest.raises(RuntimeError, match='SHA'):
        probe.verify_whole_baseline('synthetic', altered, digest, ids, classes)
    with pytest.raises(RuntimeError, match='classes'):
        probe.verify_whole_baseline('synthetic', probability, digest, ids, {'1': 0, '65535': 0})
    for wrong in (probability.astype(np.float64), probability[:, :3], np.full_like(probability, np.nan)):
        with pytest.raises(RuntimeError):
            probe.verify_whole_baseline('synthetic', wrong, digest, ids, classes)


@pytest.mark.parametrize('change', ['parameter', 'BN_buffer', 'requires_grad', 'grad', 'root_train', 'child_train'])
def test_actual_model_freeze_gate_checks_state_gradients_and_nonstate_eval_flags(change):
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2)).eval().requires_grad_(False)
    digest = tensor_digest(model.state_dict().items())
    probe.verify_frozen_model(model, digest)
    if change == 'parameter':
        with torch.no_grad():
            model[0].weight[0, 0, 0, 0] += 1.
    elif change == 'BN_buffer':
        model[1].running_mean[0] += 1.
    elif change == 'requires_grad':
        model[0].weight.requires_grad_(True)
    elif change == 'grad':
        model[0].weight.grad = torch.zeros_like(model[0].weight)
    elif change == 'root_train':
        model.training = True
    else:
        model[1].training = True
    with pytest.raises(RuntimeError):
        probe.verify_frozen_model(model, digest)


def _actual_run_ast():
    return ast.parse(inspect.getsource(probe.run)).body[0]


def _execute_nested_forward(environment):
    """执行入口的真实局部forward函数，使用CPU假encoder/decoder；不重写其实现。"""
    function = deepcopy(next(node for node in _actual_run_ast().body
        if isinstance(node, ast.Try)).body)
    forward = next(node for node in function if isinstance(node, ast.FunctionDef) and node.name == 'forward_logits')
    factory = ast.parse('def factory():\n    calls=0\n    return None\n').body[0]
    factory.body[1:2] = [forward, ast.parse('return forward_logits, lambda: calls').body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace = dict(environment)
    exec(compile(module, '<actual-semantic-native-forward>', 'exec'), namespace)
    return namespace['factory']()


def test_actual_forward_preserves_original_two_channels_through_native_resize_before_CPU_sigmoid():
    output = torch.tensor([[[[1., -1., 2., -2.], [3., -3., 4., -4.],
                             [9., 9., 9., 9.], [9., 9., 9., 9.]],
                            [[5., 6., 7., 8.], [9., 10., 11., 12.],
                             [13., 14., 15., 16.], [17., 18., 19., 20.]]]], dtype=torch.float64)
    observed = []
    def encoder(image):
        assert image.dtype == torch.float32
        return [image]
    model = SimpleNamespace(encoder=encoder, semantic_decoder=lambda features, image: output)
    def traced_crop(logits, *args):
        observed.append((logits.shape[1], logits.dtype))
        return crop_letterbox_output(logits, *args)
    forward, count = _execute_nested_forward({'model': model, 'crop_letterbox_output': traced_crop})
    image = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    actual = forward(image, 512, 0, (5, 9))
    expected = crop_letterbox_output(output.float(), 1024, 512, 0, (5, 9)).cpu()
    assert observed == [(2, torch.float32)] and count() == 1
    assert actual.shape == (1, 2, 5, 9) and torch.equal(actual, expected)
    actual_probability = np.ascontiguousarray(actual.sigmoid()[0, 0].numpy(), dtype=np.float32)
    assert np.array_equal(actual_probability, expected.sigmoid()[0, 0].numpy())


def test_last_whole_baseline_failure_prevents_any_native_forward_or_completed_gate():
    """执行实际入口with块；第8张SHA失败时，裁块函数必须尚未被调用。"""
    run_body = next(node for node in _actual_run_ast().body if isinstance(node, ast.Try)).body
    actual_block = deepcopy(next(node for node in run_body if isinstance(node, ast.With)))
    factory = ast.parse('def exercise():\n    expected_calls=1\n    return None\n').body[0]
    factory.body[1:2] = [actual_block]
    source = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    ids, p, classes, digest = _baseline()
    names = probe.TRAIN+probe.TEST
    cases = {name:dict(path=name, source_sha256='inputsha', fixed_dir='fixed',
        fixed_png_sha256='pngsha', fixed_classes_sha256='jsonsha') for name in names}
    rows = {name:dict(routes={'salign_raw':dict(probability_sha256=digest)}) for name in names}
    rows[names[-1]]['routes']['salign_raw']['probability_sha256'] = 'invalid'
    count = []; written = []; native_calls = []
    logits = torch.from_numpy(np.log(p/(1-p)))[None, None]
    actual_p = logits.sigmoid()[0, 0].numpy()
    actual_digest = hashlib.sha256(actual_p.tobytes()).hexdigest()
    for name in names[:-1]:
        rows[name]['routes']['salign_raw']['probability_sha256'] = actual_digest
    def forward(*args):
        count.append(1)
        return logits
    def no_native(*args):
        native_calls.append(1)
        raise AssertionError('A native crop ran before all eight baseline gates')
    env = dict(torch=torch, np=np, patch=lambda *args, **kwargs:nullcontext(), names=names,
        cases=cases, reference_rows=rows, protect=lambda path, expected=None:path,
        resolve=lambda path:Path(path), load_fixed=lambda folder, name:(ids, classes, Path('png'), Path('json')),
        prepare_image=lambda *args:(np.zeros((*ids.shape,3),np.uint8), torch.empty(0), 0, 0),
        validate_prediction=lambda *args:None, forward_logits=forward,
        verify_whole_baseline=probe.verify_whole_baseline,
        prepare_native_tile=no_native, write=lambda *args:written.append(args), out=Path('synthetic'))
    exec(compile(source, '<actual-semantic-native-baseline-order>', 'exec'), env)
    with pytest.raises(RuntimeError, match='SHA'):
        env['exercise']()
    assert len(count) == 8 and not native_calls and not written


@pytest.mark.parametrize('which', ['input', 'code'])
def test_actual_final_file_freeze_checks_reject_modified_sources(tmp_path, which):
    run_body = next(node for node in _actual_run_ast().body if isinstance(node, ast.Try)).body
    gates = [deepcopy(node) for node in run_body if isinstance(node, ast.If)
        and any(isinstance(child, ast.Constant) and child.value in ('Protected input changed', 'Protected source changed')
                for child in ast.walk(node))]
    assert len(gates) == 2
    file = tmp_path/'synthetic.txt'
    file.write_text('before', encoding='utf8')
    digest = probe.file_sha(file)
    protected = {str(file):digest} if which == 'input' else {}
    code = {file.name:digest} if which == 'code' else {}
    file.write_text('after', encoding='utf8')
    body = ast.fix_missing_locations(ast.Module(body=gates, type_ignores=[]))
    with pytest.raises(RuntimeError, match='Protected'):
        exec(compile(body, '<actual-semantic-native-file-freeze>', 'exec'),
            dict(file_sha=probe.file_sha, protected=protected, code=code, Path=Path, ROOT=tmp_path))
