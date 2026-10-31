# -*- coding: utf-8 -*-
"""多源捕获的虚构CPU检查；不读取任何真实图像、GT或权重。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import affinity_bottleneck_capture as subject
from tools.affinity_connectivity_views import decode_native
from tools.semantic_crossover import revote_instances
from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_fusion import affinity_boundary_probability
from utils.offset_letterbox import geometry_letterbox_metadata
from utils.patch_diagnostic import BlendMap, phase_description, tile_boxes


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def fictional_source(shape=(640, 1408), *, dtype=torch.float32):
    gt = np.ones(shape, np.int32)
    gt[:, shape[1]//2:] = 2
    gt[:6, :7] = 0
    config = dict(affinity_native=dict(overlap=.25), affinity_deployment=dict(
        fusion_mode='gated', short_reduction='top2', distance2_weight=.5, distance4_weight=.25),
        inference=dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
            marker_boundary_reconstruction_steps=8, marker_border_seal_width=2,
            bridge_width=1, watershed_dilate_width=1, min_instance_area=50,
            max_instance_id=65535, threshold=.5, semantic_vote_mode='probability_mean'))
    calls = dict(head=0, predict=0)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.full((1, 8, 512, 512), 8., dtype=dtype)
            self.register_buffer('logits', value.contiguous(memory_format=torch.channels_last))

        def forward(self, _input):
            calls['head'] += 1
            return dict(affinity_logits=self.logits)

    model = SimpleNamespace(affinity_decoder=Head())
    views = SimpleNamespace(decode_native=decode_native, revote_instances=revote_instances)

    def original_grid(actual_model, image, cfg):
        mode, options = subject.fusion_options(cfg)
        return affinity_boundary_probability(actual_model.affinity_decoder(image)['affinity_logits'].float(),
                                             mode=mode, **options)

    views.boundary_grid = original_grid
    views.crop_letterbox_output = crop_letterbox_output

    def original_predict(actual_model, restorer, path, cfg, device, seed, sizes):
        calls['predict'] += 1
        assert sizes == [1024]
        # 原全图affinity forward在tile之外，不能被hook误当成局部窗口。
        actual_model.affinity_decoder(None)
        geometry = geometry_letterbox_metadata(shape, 1024, 512)
        padding = (1024-geometry.resized_height, 1024-geometry.resized_width)
        semantic = views.crop_letterbox_output(torch.zeros((1, 1, 512, 512)),
            1024, *padding, shape)
        raw = views.crop_letterbox_output(torch.full((1, 1, 512, 512), 2.),
            1024, *padding, shape).sigmoid()[0, 0].numpy()
        blend = BlendMap(shape)
        boxes = tile_boxes(shape, 1024, cfg['affinity_native']['overlap'])
        for box in boxes:
            y, x, y1, x1 = box
            meta = geometry_letterbox_metadata((y1-y, x1-x), 1024, 512)
            grid = views.boundary_grid(actual_model, None, cfg)
            native = views.crop_letterbox_output(grid, 1024,
                1024-meta.resized_height, 1024-meta.resized_width, (y1-y, x1-x))
            blend.add(box, native[0, 0].numpy())
        boundary, audit = blend.finish(); audit['tiles'] = len(boxes)
        ids, _ = views.decode_native(semantic, torch.from_numpy(boundary)[None, None], cfg)
        classes, _ = views.revote_instances(ids, raw)
        value = dict(instances=ids, classes=classes, boundary=boundary,
                     blend=audit, phases=phase_description(ids, classes))
        return np.zeros((*shape, 3), np.uint8), dict(patch1024=value)

    views.predict_views = original_predict
    return gt, config, model, views, calls


def test_actual_two_tiles_transparent_once_and_original_stride_unknown_padding():
    gt, config, model, views, calls = fictional_source()
    originals = (views.boundary_grid, views.crop_letterbox_output, views.decode_native,
                 views.revote_instances)
    _, expected = views.predict_views(model, None, 'fictional.png', config, 'cpu', 31, [1024])
    calls.update(head=0, predict=0)
    logits_before = model.affinity_decoder.logits.clone()
    gt_before = gt.copy()
    rgb, captured, tiles, audit = subject.capture_native_tiles(
        views, model, None, 'fictional.png', config, gt, 'cpu', 31)
    assert calls == dict(head=3, predict=1)
    assert audit['tile_count'] == 2 and audit['skipped_non_tile_crops'] == 2
    assert audit['empty_mask_grid_native_and_stitch_exact']
    assert audit['baseline_blend'] == captured['baseline']['blend'] == audit['empty_blend']
    assert rgb.shape[:2] == gt.shape
    assert np.array_equal(captured['baseline']['instances'], expected['patch1024']['instances'])
    assert captured['baseline']['classes'] == expected['patch1024']['classes']
    assert (views.boundary_grid, views.crop_letterbox_output, views.decode_native,
            views.revote_instances) == originals
    assert not model.affinity_decoder._forward_hooks
    assert np.array_equal(gt, gt_before) and torch.equal(model.affinity_decoder.logits, logits_before)
    for index, tile in enumerate(tiles):
        assert tile['index'] == index and tile['box'] == list(tile_boxes(gt.shape, 1024, .25)[index])
        assert tile['seed'] == 31+100000+1024*1000+index
        assert tile['logits'].data_ptr() == model.affinity_decoder.logits.data_ptr()
        assert tile['logits_stride'] == list(model.affinity_decoder.logits.stride())
        assert not tile['logits'].requires_grad
        assert tile['grid'].shape == (1, 1, 512, 512)
        assert tile['native'].shape == (1, 1, 640, 1024)
        assert np.array_equal(tile['native'], tile['empty_native'])
        assert tile['empty_grid_exact'] and tile['empty_native_exact']
        assert tile['pad_h'] == 384 and tile['pad_w'] == 0
        assert not tile['valid'][320:].any() and not tile['grid_gt'][320:].any()
    assert tiles[0]['valid'][0, 0] and tiles[0]['grid_gt'][0, 0] == 0


def test_different_shapes_and_repeated_sources_do_not_leak_pending_state():
    for shape in ((128, 176), (1072, 128)):
        gt, config, model, views, calls = fictional_source(shape)
        for seed in (5, 19):
            _, _, tiles, audit = subject.capture_native_tiles(
                views, model, None, 'fictional.png', config, gt, 'cpu', seed)
            assert len(tiles) == len(tile_boxes(shape, 1024, .25))
            assert [tile['index'] for tile in tiles] == list(range(len(tiles)))
            assert audit['seed'] == seed and audit['output_shape'] == list(shape)
            assert not model.affinity_decoder._forward_hooks
        assert calls['predict'] == 2


def test_empty_fusion_mismatch_fails_and_releases_every_hook(monkeypatch):
    gt, config, model, views, _ = fictional_source((128, 176))
    original = (views.boundary_grid, views.crop_letterbox_output, views.decode_native, views.revote_instances)
    fusion = subject.affinity_boundary_probability
    monkeypatch.setattr(subject, 'affinity_boundary_probability', lambda *a, **k: fusion(*a, **k)+.001)
    with pytest.raises(RuntimeError, match='empty-mask fusion'):
        subject.capture_native_tiles(views, model, None, 'fictional.png', config, gt, 'cpu', 3)
    assert (views.boundary_grid, views.crop_letterbox_output, views.decode_native,
            views.revote_instances) == original
    assert not model.affinity_decoder._forward_hooks


def test_empty_native_mismatch_fails_and_releases_hook():
    gt, config, model, views, _ = fictional_source((128, 176))
    real_crop = views.crop_letterbox_output
    count = 0

    def corrupt_empty(*args, **kwargs):
        nonlocal count
        count += 1
        value = real_crop(*args, **kwargs)
        # 两个全图crop之后为实际tile crop，再之后为空干预重放crop。
        return value+.001 if count == 4 else value

    views.crop_letterbox_output = corrupt_empty
    with pytest.raises(RuntimeError, match='empty-mask native crop'):
        subject.capture_native_tiles(views, model, None, 'fictional.png', config, gt, 'cpu', 3)
    assert views.crop_letterbox_output is corrupt_empty
    assert not model.affinity_decoder._forward_hooks


def test_wrong_dtype_is_not_silently_converted_to_fp32():
    gt, config, model, views, _ = fictional_source((128, 176), dtype=torch.float64)
    with pytest.raises(RuntimeError, match='finite FP32'):
        subject.capture_native_tiles(views, model, None, 'fictional.png', config, gt, 'cpu', 3)
    assert not model.affinity_decoder._forward_hooks


def test_multiple_head_calls_inside_tile_are_rejected():
    gt, config, model, views, _ = fictional_source((128, 176))
    real_grid = views.boundary_grid

    def repeated_head(actual_model, *args, **kwargs):
        actual_model.affinity_decoder(None)
        return real_grid(actual_model, *args, **kwargs)

    views.boundary_grid = repeated_head
    with pytest.raises(RuntimeError, match='More than one affinity head'):
        subject.capture_native_tiles(views, model, None, 'fictional.png', config, gt, 'cpu', 3)
    assert views.boundary_grid is repeated_head and not model.affinity_decoder._forward_hooks


def test_original_blend_audit_corruption_cannot_pass():
    gt, config, model, views, _ = fictional_source((128, 176))
    original_predict = views.predict_views

    def corrupted(*args, **kwargs):
        rgb, outputs = original_predict(*args, **kwargs)
        outputs['patch1024']['blend']['tiles_max_overlap'] += 1
        return rgb, outputs

    views.predict_views = corrupted
    with pytest.raises(RuntimeError, match='original-order native BlendMap'):
        subject.capture_native_tiles(views, model, None, 'fictional.png', config, gt, 'cpu', 3)
    assert not model.affinity_decoder._forward_hooks


@pytest.mark.parametrize('invalid_gt', [np.ones((3, 4), np.float32),
    np.ones((3, 4), bool), np.full((3, 4), -1, np.int32), np.zeros((0, 4), np.int32)])
def test_invalid_gt_fails_before_forward(invalid_gt):
    _, config, model, views, calls = fictional_source((128, 176))
    with pytest.raises(ValueError, match='gt must'):
        subject.capture_native_tiles(views, model, None, 'fictional.png', config, invalid_gt, 'cpu', 3)
    assert calls == dict(head=0, predict=0) and not model.affinity_decoder._forward_hooks


def test_changed_overlap_requires_new_explicit_contract():
    gt, config, model, views, calls = fictional_source((128, 176))
    config['affinity_native']['overlap'] = .5
    with pytest.raises(ValueError, match='overlap .25'):
        subject.capture_native_tiles(views, model, None, 'fictional.png', config, gt, 'cpu', 3)
    assert calls == dict(head=0, predict=0)
