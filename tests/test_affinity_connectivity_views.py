# -*- coding: utf-8 -*-
"""原部署解码、增强坐标还原及交叉分类的针对性回归。"""
import random

import cv2
import numpy as np
import pytest
import torch
from torch import nn

import tools.affinity_connectivity_views as views
from data.direct_dual_head_dataset import _spatial_transform
from utils.affinity_deployment import postprocess, probability_to_logit, crop_letterbox_output
from utils.offset_letterbox import geometry_letterbox_metadata, letterbox_instance_geometry


def _config():
    return {'inference': {
        'threshold': .5, 'boundary_threshold': .65, 'min_instance_area': 8,
        'max_instance_id': 65535, 'watershed_dilate_width': 1, 'bridge_width': 1,
        'marker_border_seal_width': 2, 'marker_boundary_low_threshold': .45,
        'marker_boundary_reconstruction_steps': 8,
        'semantic_vote_mode': 'probability_mean', 'semantic_vote_erode_width': 0,
        'semantic_vote_threshold': .5,
    }, 'affinity_deployment': {'fusion_mode': 'gated', 'short_reduction': 'top2'}}


def _maps():
    semantic = torch.full((1, 1, 61, 79), -2.)
    semantic[:, :, :35, :40] = 2.
    semantic[:, :, 37:, 43:] = .1
    boundary = torch.full_like(semantic, .03)
    boundary[:, :, :, 39:41] = .9
    boundary[:, :, 28:30, :] = .95
    boundary[:, :, 26:33, 37:44] = .52
    boundary[:, :, 24:26, 37:44] = .9
    boundary[:, :, 10:13, 10:13] = 1.
    boundary[:, :, 11, 11] = 0.
    return semantic, boundary


@pytest.mark.parametrize('variant', ['base', 'no_reconstruction', 'core'])
def test_decode_matches_actual_file_writing_postprocess(tmp_path, variant):
    config = _config()
    if variant == 'no_reconstruction':
        config['inference'].update(marker_boundary_reconstruction_steps=0, marker_border_seal_width=0)
    elif variant == 'core':
        config['inference'].update(semantic_vote_mode='adaptive_core', semantic_vote_core_fraction=.6,
                                   semantic_vote_core_min_pixels=3, semantic_vote_core_distance_power=1.5)
    semantic, boundary = _maps()
    expected_input = torch.cat([semantic, probability_to_logit(boundary)], dim=1)
    _, expected_ids, expected_classes = postprocess(
        expected_input, semantic.shape[-2:], tmp_path, variant, config['inference'],
        config['inference']['boundary_threshold'], False)
    actual_ids, actual_classes = views.decode_native(semantic, boundary, config)
    assert actual_ids.dtype == np.uint16
    assert np.array_equal(actual_ids, expected_ids)
    assert actual_classes == expected_classes
    assert len(actual_classes) >= 2


@pytest.mark.parametrize('shape,hflip,vflip,turns', [((47, 63), True, False, 1),
                                                    ((65, 43), False, True, 3),
                                                    ((53, 71), True, True, 2)])
def test_training_partition_fuses_before_resize_and_replays_canonical_coordinates(monkeypatch, shape, hflip, vflip, turns):
    config = _config()
    yy = torch.linspace(-3., 3., 512)[:, None]
    xx = torch.linspace(-2., 2., 512)[None, :]
    canonical = torch.stack([yy + xx + channel * .4 for channel in range(8)])[None]
    semantic = (yy - xx)[None, None]
    augmented = _spatial_transform(canonical, hflip, vflip, turns)
    augmented_sem = _spatial_transform(semantic, hflip, vflip, turns)
    native_ids = np.zeros(shape, np.uint16)
    native_ids[:shape[0] // 2, :shape[1] // 3] = 17
    native_ids[shape[0] // 2:, shape[1] // 3:] = 29
    captured = {}
    original_fuse = views.affinity_boundary_probability

    def fuse(logits, **kwargs):
        captured['fusion_shape'] = tuple(logits.shape)
        assert torch.equal(logits, augmented)
        return original_fuse(logits, **kwargs)

    def decode(sem, edge, _config):
        captured['semantic'], captured['boundary'] = sem, edge
        return native_ids, {17: 0, 29: 1}

    monkeypatch.setattr(views, 'affinity_boundary_probability', fuse)
    monkeypatch.setattr(views, 'decode_native', decode)
    batch = {'native_shape': torch.tensor([shape]), 'horizontal_flip': torch.tensor([hflip]),
             'vertical_flip': torch.tensor([vflip]), 'rotation_k': torch.tensor([turns])}
    output = {'semantic_logits': augmented_sem, 'affinity_logits': augmented}
    actual = views.training_partition(output, batch, config)
    metadata = geometry_letterbox_metadata(shape, 1024, 512)
    ph, pw = 1024 - metadata.resized_height, 1024 - metadata.resized_width
    mode, kwargs = views.fusion_options(config)
    fused = original_fuse(augmented, mode=mode, **kwargs)
    canonical_edge = views.undo_spatial(fused, hflip, vflip, turns)
    expected_edge = crop_letterbox_output(canonical_edge, 1024, ph, pw, shape)
    expected_semantic = crop_letterbox_output(semantic, 1024, ph, pw, shape)
    assert captured['fusion_shape'] == (1, 8, 512, 512)
    assert torch.equal(captured['boundary'], expected_edge)
    assert torch.equal(captured['semantic'], expected_semantic)
    expected_grid, _, _ = letterbox_instance_geometry(native_ids, input_size=1024, output_grid=512)
    expected = _spatial_transform(torch.from_numpy(expected_grid.astype(np.int64)), hflip, vflip, turns)[None]
    assert actual.shape == (1, 512, 512) and actual.dtype == torch.long
    assert torch.equal(actual, expected)
    assert set(actual.unique().tolist()) == {0, 17, 29}


def test_training_partition_rejects_batched_output():
    with pytest.raises(ValueError, match='affinity'):
        views.training_partition({'affinity_logits': torch.zeros(2, 8, 512, 512),
                                  'semantic_logits': torch.zeros(2, 1, 256, 256)}, {}, _config())


class _Semantic(nn.Module):
    def forward(self, features, image):
        return image[:, :1] + 1.


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.semantic_decoder = _Semantic()


def test_predict_cross_keeps_restored_partition_raw_vote_and_external_state(monkeypatch):
    config = _config()
    semantic, boundary = _maps()
    image = np.full((*semantic.shape[-2:], 3), 110, dtype=np.uint8)
    expected_ids, _ = views.decode_native(semantic, boundary, config)
    model, restorer = _Model(), nn.Sequential(nn.Dropout(.5))
    model.train()
    model.encoder.eval()
    restorer.train()
    modes = [module.training for obj in (model, restorer) for module in obj.modules()]
    torch_state, np_state, py_state = torch.get_rng_state(), np.random.get_state(), random.getstate()

    def predict(*args):
        assert not model.training and not restorer.training
        torch.rand(4)
        np.random.random()
        random.random()
        return image, semantic, boundary

    monkeypatch.setattr(views, 'prediction_maps', predict)
    monkeypatch.setattr(views, 'prepare_image', lambda *args: (image, torch.zeros(1, 3, 1024, 1024), 0, 0))
    result = views.predict_cross(model, restorer, 'unused', config, 'cpu', 314167)
    assert np.array_equal(result['instances'], expected_ids)
    assert set(result['classes'].values()) == {1}
    assert result['raw_probability'].dtype == np.float32
    assert np.allclose(result['raw_probability'], torch.sigmoid(torch.tensor(1.)).item())
    assert torch.equal(result['boundary'], boundary)
    assert modes == [module.training for obj in (model, restorer) for module in obj.modules()]
    assert torch.equal(torch.get_rng_state(), torch_state)
    after_np = np.random.get_state()
    assert after_np[0] == np_state[0] and np.array_equal(after_np[1], np_state[1]) and after_np[2:] == np_state[2:]
    assert random.getstate() == py_state


def test_monitor_uses_full_cohort_indices_for_seeds_and_small_artifacts(monkeypatch, tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    for index in range(4):
        cv2.imwrite(str(images / f'test_{index:03}.png'), np.full((31, 45, 3), 120, np.uint8))
    config = _config()
    config.update(paths={'project_root': str(tmp_path)}, backend_adaptation={'restoration': {'inference_seed': 314159}})
    config['inference']['test_dir'] = 'images'
    calls = []

    def predict(_model, _restorer, path, _config, _device, seed):
        calls.append((path.stem, seed))
        return {'image_rgb': np.full((31, 45, 3), 120, np.uint8),
                'instances': np.ones((31, 45), np.uint16), 'classes': {'1': 1},
                'raw_probability': np.full((31, 45), .7, np.float32),
                'boundary': torch.full((1, 1, 31, 45), .2)}

    monkeypatch.setattr(views, 'predict_cross', predict)
    summary = views.save_monitor(None, None, config, 'cpu', tmp_path / 'out', 5, ['test_003', 'test_001'])
    assert calls == [('test_001', 314160), ('test_003', 314162)]
    assert summary['source_count'] == 4 and len(summary['images']) == 2
    target = tmp_path / 'out/monitor/epoch_005'
    assert {path.name for path in target.iterdir()} == {'test_001.png', 'test_003.png', 'summary.json'}
    thumbnail = cv2.imread(str(target / 'test_001.png'))
    assert thumbnail.shape[1] == 960
    assert thumbnail.shape[0] < 350
