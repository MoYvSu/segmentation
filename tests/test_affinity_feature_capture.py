# -*- coding: utf-8 -*-
"""冻结特征透明捕获与真实head重放的合成CPU合同；无赛题数据。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import affinity_feature_capture as subject
from tools.affinity_connectivity_views import decode_native
from tools.semantic_crossover import revote_instances
from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_fusion import affinity_boundary_probability
from utils.offset_letterbox import geometry_letterbox_metadata
from utils.patch_diagnostic import BlendMap, phase_description, tile_boxes


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def fictional_source(shape=(128, 176), *, feature_count=3):
    gt = np.ones(shape, np.int32)
    gt[:, shape[1] // 2:] = 2
    config = dict(affinity_native=dict(overlap=.25), affinity_deployment=dict(
        fusion_mode='gated', short_reduction='top2', distance2_weight=.5, distance4_weight=.25),
        inference=dict(boundary_threshold=.65, marker_boundary_low_threshold=.45,
            marker_boundary_reconstruction_steps=8, marker_border_seal_width=2,
            bridge_width=1, watershed_dilate_width=1, min_instance_area=50,
            max_instance_id=65535, threshold=.5, semantic_vote_mode='probability_mean'))
    calls = dict(head=0, encoder=0, predict=0)
    channels = list(range(2, feature_count + 2))

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i, ch in enumerate(channels):
                shape = (1, ch, max(2, 32 // 2 ** i), max(2, 48 // 2 ** i))
                value = torch.full(shape, float(i + 1)).contiguous(memory_format=torch.channels_last)
                self.register_buffer('stage_' + str(i), value)

        def forward(self, image):
            calls['encoder'] += 1
            return [getattr(self, 'stage_' + str(i)) for i in range(feature_count)]

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            lateral = torch.nn.ModuleList([torch.nn.Conv2d(ch, 2, 1) for ch in channels])
            self.geometry_fpn = torch.nn.Module()
            self.geometry_fpn.lateral_convs = lateral
            self.geometry_fpn.num_stages = feature_count
            self.bias = torch.nn.Parameter(torch.zeros(()))
            self.register_buffer('logits', torch.full((1, 8, 512, 512), 4.).contiguous(memory_format=torch.channels_last))
            self.eval().requires_grad_(False)

        def forward(self, features):
            calls['head'] += 1
            return dict(affinity_logits=self.logits + self.bias)

    model = SimpleNamespace(encoder=Encoder(), affinity_decoder=Head())
    views = SimpleNamespace(decode_native=decode_native, revote_instances=revote_instances,
                            crop_letterbox_output=crop_letterbox_output)

    def original_grid(actual_model, image, cfg):
        features = actual_model.encoder(image)
        value = actual_model.affinity_decoder(features)['affinity_logits']
        mode, options = subject.fusion_options(cfg)
        return affinity_boundary_probability(value, mode=mode, **options)

    views.boundary_grid = original_grid

    def original_predict(actual_model, restorer, path, cfg, device, seed, sizes):
        calls['predict'] += 1
        assert sizes == [1024]
        # tile之外的head输入不合法也不得误捕获为冻结encoder特征。
        actual_model.affinity_decoder(None)
        metadata = geometry_letterbox_metadata(shape, 1024, 512)
        semantic = views.crop_letterbox_output(torch.zeros((1, 1, 512, 512)),
            1024, 1024-metadata.resized_height, 1024-metadata.resized_width, shape)
        raw = views.crop_letterbox_output(torch.ones((1, 1, 512, 512)),
            1024, 1024-metadata.resized_height, 1024-metadata.resized_width, shape).sigmoid()[0, 0].numpy()
        image = torch.zeros((1, 3, 1024, 1024))
        blend = BlendMap(shape)
        boxes = tile_boxes(shape, 1024, .25)
        for box in boxes:
            y, x, y1, x1 = box
            meta = geometry_letterbox_metadata((y1-y, x1-x), 1024, 512)
            grid = views.boundary_grid(actual_model, image, cfg)
            native = views.crop_letterbox_output(grid, 1024,
                1024-meta.resized_height, 1024-meta.resized_width, (y1-y, x1-x))
            blend.add(box, native[0, 0].numpy())
        boundary, audit = blend.finish()
        audit['tiles'] = len(boxes)
        ids, _ = views.decode_native(semantic, torch.from_numpy(boundary)[None, None], cfg)
        classes, _ = views.revote_instances(ids, raw)
        return np.zeros((*shape, 3), np.uint8), dict(patch1024=dict(
            instances=ids, classes=classes, boundary=boundary, blend=audit,
            phases=phase_description(ids, classes)))

    views.predict_views = original_predict
    return gt, config, model, views, calls


def capture(fixture):
    gt, config, model, views, _ = fixture
    return subject.capture_frozen_features(views, model, None, 'synthetic.png', config, gt, 'cpu', 9)


def test_transparent_two_tile_feature_input_storage_stride_and_three_stage_interface():
    fixture = fictional_source((640, 1408))
    gt, config, model, views, calls = fixture
    before = (views.boundary_grid, views.crop_letterbox_output, views.decode_native, views.revote_instances)
    expected = views.predict_views(model, None, 'synthetic.png', config, 'cpu', 9, [1024])[1]
    calls.update(head=0, encoder=0, predict=0)
    rgb, captured, tiles, audit = capture(fixture)
    assert calls == dict(head=3, encoder=2, predict=1)
    assert rgb.shape[:2] == gt.shape
    assert audit['actual_feature_counts'] == [3, 3]
    assert audit['affinity_FPN_input_channels'] == [2, 3, 4]
    assert audit['explicit_image_tensor_checks'] == 2
    assert audit['frozen_encoder_features_exact']
    assert np.array_equal(captured['baseline']['instances'], expected['patch1024']['instances'])
    assert captured['baseline']['classes'] == expected['patch1024']['classes']
    assert (views.boundary_grid, views.crop_letterbox_output, views.decode_native, views.revote_instances) == before
    assert not model.affinity_decoder._forward_hooks and not model.affinity_decoder._forward_pre_hooks
    for tile in tiles:
        for i, (value, meta) in enumerate(zip(tile['features'], tile['features_metadata'])):
            actual = getattr(model.encoder, 'stage_' + str(i))
            assert value.data_ptr() == actual.data_ptr()
            assert value.stride() == actual.stride()
            assert not value.requires_grad
            assert meta['shape'] == list(actual.shape) and meta['device'] == 'cpu'


@pytest.mark.parametrize('count', [1, 4])
def test_feature_level_count_follows_real_fpn_instead_of_assuming_sam2(count):
    _, _, tiles, audit = capture(fictional_source(feature_count=count))
    assert len(tiles[0]['features']) == count
    assert audit['actual_feature_counts'] == [count]


def test_empty_replay_actually_recomputes_every_head_fusion_crop_and_whole_boundary():
    fixture = fictional_source((640, 1408))
    _, config, model, views, calls = fixture
    _, captured, tiles, _ = capture(fixture)
    heads = calls['head']
    saved = [[v.clone() for v in tile['features']] for tile in tiles]
    old_native = [tile['native'].copy() for tile in tiles]
    boundary, audit = subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, (640, 1408), empty_exact=True)
    assert calls['head'] == heads + len(tiles)
    assert np.array_equal(boundary, captured['baseline']['boundary'])
    assert audit['whole_baseline_exact'] and audit['no_old_native_reuse']
    assert all(all(row['empty_checks'].values()) and row['actual_head_recomputed'] for row in audit['tiles'])
    for tile, features, native in zip(tiles, saved, old_native):
        assert np.array_equal(tile['native'], native)
        assert all(torch.equal(a, b) for a, b in zip(tile['features'], features))


def test_new_head_boundary_is_recomputed_and_empty_gate_rejects_changed_head():
    fixture = fictional_source()
    gt, config, model, views, _ = fixture
    _, captured, tiles, _ = capture(fixture)
    candidate = deepcopy(model.affinity_decoder)
    with torch.no_grad():
        candidate.bias.fill_(-4.)
    boundary, audit = subject.replay_head_boundary(views, tiles, candidate, config, gt.shape)
    assert not np.array_equal(boundary, captured['baseline']['boundary'])
    assert audit['all_tiles_actual_head_fusion_crop']
    with pytest.raises(RuntimeError, match='empty head replay'):
        subject.replay_head_boundary(views, tiles, candidate, config, gt.shape, empty_exact=True)


def test_multiple_head_inputs_rejected_and_hooks_patches_restored():
    fixture = fictional_source()
    _, _, model, views, _ = fixture
    real = views.boundary_grid

    def repeated(actual_model, image, cfg):
        actual_model.affinity_decoder(actual_model.encoder(image))
        return real(actual_model, image, cfg)

    views.boundary_grid = repeated
    before = (views.boundary_grid, views.crop_letterbox_output, views.decode_native, views.revote_instances)
    with pytest.raises(RuntimeError, match='More than one affinity head'):
        capture(fixture)
    assert (views.boundary_grid, views.crop_letterbox_output, views.decode_native, views.revote_instances) == before
    assert not model.affinity_decoder._forward_pre_hooks and not model.affinity_decoder._forward_hooks


def test_grid_exception_releases_pre_hook_and_existing_capture_patches():
    fixture = fictional_source()
    _, _, model, views, _ = fixture
    real = views.boundary_grid

    def fail(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError('synthetic failure')

    views.boundary_grid = fail
    before = (views.boundary_grid, views.crop_letterbox_output, views.decode_native, views.revote_instances)
    with pytest.raises(RuntimeError, match='synthetic failure'):
        capture(fixture)
    assert (views.boundary_grid, views.crop_letterbox_output, views.decode_native, views.revote_instances) == before
    assert not model.affinity_decoder._forward_pre_hooks and not model.affinity_decoder._forward_hooks


@pytest.mark.parametrize('failure', ['dtype', 'grad', 'shape', 'channel', 'count', 'nan'])
def test_invalid_actual_feature_input_cannot_be_hidden_by_detach(failure):
    fixture = fictional_source()
    _, _, model, _, _ = fixture
    if failure == 'dtype': model.encoder.stage_0 = model.encoder.stage_0.double()
    if failure == 'grad': model.encoder.stage_0.requires_grad_(True)
    if failure == 'shape': model.encoder.stage_0 = torch.ones((2, 2, 32, 48))
    if failure == 'channel': model.encoder.stage_0 = torch.ones((1, 5, 32, 48))
    if failure == 'count': model.affinity_decoder.geometry_fpn.num_stages = 4
    if failure == 'nan': model.encoder.stage_0[0, 0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='features|feature|FPN'):
        capture(fixture)
    assert not model.affinity_decoder._forward_pre_hooks and not model.affinity_decoder._forward_hooks


def test_feature_cache_mutation_and_train_mode_replay_fail_closed():
    fixture = fictional_source()
    gt, config, model, views, _ = fixture
    _, _, tiles, _ = capture(fixture)
    model.affinity_decoder.train()
    with pytest.raises(ValueError, match='eval mode'):
        subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, gt.shape)
    model.affinity_decoder.eval()
    tiles[0]['features'][0][0, 0, 0, 0] += 1.
    with pytest.raises(RuntimeError, match='cache differs'):
        subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, gt.shape)


def test_shape_and_tile_order_and_grad_illegal_replay_inputs():
    fixture = fictional_source()
    gt, config, model, views, _ = fixture
    _, _, tiles, _ = capture(fixture)
    with pytest.raises(ValueError, match='Complete original-order'):
        subject.replay_head_boundary(views, [], model.affinity_decoder, config, gt.shape)
    with pytest.raises(ValueError, match='Native shape'):
        subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, (0, 10))
    tiles[0]['index'] = 1
    with pytest.raises(RuntimeError, match='tile order'):
        subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, gt.shape)
    tiles[0]['index'] = 0
    tiles[0]['features'][0].requires_grad_(True)
    with pytest.raises(RuntimeError, match='requires_grad=False'):
        subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, gt.shape)


def test_named_image_tensor_checks_rgb_shape_dtype_and_frozen_input():
    for mode in ('shape', 'dtype', 'grad'):
        fixture = fictional_source()
        _, _, model, views, _ = fixture
        # 参数检查针对明确image签名的入口；在predict调用前改传入Tensor。
        old_predict = views.predict_views
        def bad_predict(*args, **kwargs):
            with pytest.MonkeyPatch.context() as inner:
                grid = views.boundary_grid
                def bad(actual_model, image, cfg):
                    if mode == 'shape': image = image[:, :2]
                    if mode == 'dtype': image = image.double()
                    if mode == 'grad': image = image.detach().requires_grad_(True)
                    return grid(actual_model, image, cfg)
                inner.setattr(views, 'boundary_grid', bad)
                return old_predict(*args, **kwargs)
        views.predict_views = bad_predict
        with pytest.raises(RuntimeError, match='Actual tile RGB'):
            capture(fixture)
        assert not model.affinity_decoder._forward_pre_hooks and not model.affinity_decoder._forward_hooks


def test_unknown_image_argument_name_is_not_guessed_and_feature_capture_stays_transparent():
    fixture = fictional_source()
    _, _, _, views, _ = fixture
    real = views.boundary_grid

    def alternate(actual_model, rgb_input, cfg):
        return real(actual_model, rgb_input, cfg)

    views.boundary_grid = alternate
    _, _, tiles, audit = capture(fixture)
    assert audit['explicit_image_tensor_checks'] == 0
    assert len(tiles[0]['features']) == 3
    assert views.boundary_grid is alternate


def test_mutating_candidate_head_is_detected_instead_of_silently_reusing_poisoned_features():
    fixture = fictional_source()
    gt, config, model, views, _ = fixture
    _, _, tiles, _ = capture(fixture)
    normal = model.affinity_decoder.forward

    def bad(features):
        features[0][0, 0, 0, 0] += 1.
        return normal(features)

    model.affinity_decoder.forward = bad
    with pytest.raises(RuntimeError, match='mutated frozen feature cache'):
        subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, gt.shape)


def test_original_native_cache_corruption_cannot_pass_actual_empty_replay():
    fixture = fictional_source()
    gt, config, model, views, _ = fixture
    _, _, tiles, _ = capture(fixture)
    tiles[0]['native'][0, 0, 0, 0] += .01
    with pytest.raises(RuntimeError, match='empty head replay'):
        subject.replay_head_boundary(views, tiles, model.affinity_decoder, config, gt.shape, empty_exact=True)


def test_unfrozen_original_head_is_rejected_before_capture_forward():
    fixture = fictional_source()
    _, _, model, _, calls = fixture
    model.affinity_decoder.bias.requires_grad_(True)
    with pytest.raises(RuntimeError, match='parameters must be frozen'):
        capture(fixture)
    assert calls == dict(head=0, encoder=0, predict=0)
    assert not model.affinity_decoder._forward_pre_hooks


def test_public_feature_validator_uses_the_same_capture_metadata_digest():
    fixture = fictional_source()
    _, _, model, _, _ = fixture
    _, _, tiles, _ = capture(fixture)
    tile = tiles[0]
    assert subject.validate_frozen_tile_features(tile) == tile['features_sha256']
    assert subject.validate_frozen_tile_features(tile, model.affinity_decoder) == tile['features_sha256']
    tile['features_metadata'][0]['stride'][0] += 1
    with pytest.raises(RuntimeError, match='cache differs'):
        subject.validate_frozen_tile_features(tile)
