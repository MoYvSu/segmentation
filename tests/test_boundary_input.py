# -*- coding: utf-8 -*-
"""受控线宽表示的CPU合同：无GT输入、整图后裁、旧行为和诊断集合保留。"""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
from PIL import Image
import pytest
from skimage.morphology import skeletonize
import torch

from data.boundary_rgb import BoundaryRGBDataset, collate_boundary_rgb
from tools.boundary_naf_monitor import real_pair_metrics
import tools.boundary_rgb_monitor as monitor
from tools.boundary_restore_readout import evaluate_frozen_readout
from utils.boundary_input import input_normalization_settings, normalize_boundary_input

from test_boundary_rgb import _fixture, _force
from test_boundary_rgb_monitor import PointwiseRGB, rgb_for, settings, frozen_config


NORM = {'mode': 'skeleton_dilate', 'kernel_size': 8}


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_none_is_exact_threshold_and_does_not_thin(monkeypatch):
    import utils.boundary_input as module
    boundary = np.random.default_rng(1).random((33, 47), dtype=np.float32)
    original = boundary.copy()
    monkeypatch.setattr(module, 'skeletonize', lambda *_: (_ for _ in ()).throw(AssertionError('none must not thin')))
    for cfg in (None, {'mode': 'none'}, {'mode': 'none', 'kernel_size': 8}):
        result, receipt = normalize_boundary_input(boundary, settings=cfg)
        np.testing.assert_array_equal(result, boundary >= .5)
        assert result.dtype == bool and receipt['mode'] == 'none'
        assert receipt['raw_binary_sha256'] == receipt['output_binary_sha256']
    np.testing.assert_array_equal(boundary, original)


def test_ellipse8_matches_author_polarity_anchor_and_native_shape():
    boundary = np.zeros((64, 80), np.float32)
    boundary[10:52, 24:35] = .8
    boundary[20:24, 7:62] = .6
    boundary[14:18, 24:35] = .49
    actual, receipt = normalize_boundary_input(boundary, settings=NORM)
    expected = cv2.dilate(skeletonize(boundary >= .5).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)), iterations=1) > 0
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == boundary.shape and actual.dtype == bool
    assert receipt['resolved_anchor_xy'] == [4, 4]
    assert receipt['opencv_anchor_argument_xy'] == [-1, -1]
    assert receipt['even_kernel_anchor_offset_from_center_xy'] == [.5, .5]
    assert receipt['polarity'] == 'high_is_boundary' and not receipt['GT_used']
    assert receipt['whole_image_before_crop'] and not receipt['topology_preserved']


@pytest.mark.parametrize('fill', [0., 1.])
@pytest.mark.parametrize('shape', [(1, 3), (19, 11), (64, 80)])
def test_empty_or_full_sources_are_valid_and_match_author_algorithm(fill, shape):
    source = np.full(shape, fill, np.float32)
    actual, _ = normalize_boundary_input(source, settings=NORM)
    expected = cv2.dilate(skeletonize(source >= .5).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))) > 0
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize('config', [[], {'mode': 'wrong'}, {'mode': 'skeleton_dilate', 'kernel_size': 0},
    {'mode': 'skeleton_dilate', 'kernel_size': True}, {'mode': 'none', 'unexpected': 1}])
def test_bad_representation_settings_rejected(config):
    with pytest.raises(ValueError):
        input_normalization_settings(config)


@pytest.mark.parametrize('source', [np.zeros((0, 4)), np.zeros((4, 4, 1)),
    np.full((4, 4), np.nan), np.full((4, 4), 1.1), np.full((4, 4), -.1)])
def test_invalid_input_rejected(source):
    with pytest.raises(ValueError):
        normalize_boundary_input(source, settings=NORM)


def test_fixed_threshold_and_whole_image_before_crop_are_explicit():
    image = np.zeros((80, 80), np.float32)
    image[10:70, 10:70] = 1
    full, _ = normalize_boundary_input(image, settings=NORM)
    cropped_first, _ = normalize_boundary_input(image[:40], settings=NORM)
    assert not np.array_equal(full[:40], cropped_first)
    with pytest.raises(ValueError, match='remain 0.5'):
        normalize_boundary_input(image, threshold=.65, settings=NORM)


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in range(4) for f in (False, True)])
def test_dataset_normalizes_full_native_T_then_shares_D4_and_raw_gap(tmp_path, rotation, flip):
    manifest, _, cfg, rgb, boundary, ids = _fixture(tmp_path)
    original = BoundaryRGBDataset(manifest, cfg)
    normalized_cfg = deepcopy(cfg)
    normalized_cfg['boundary_naf']['input_normalization'] = dict(NORM)
    ds = BoundaryRGBDataset(manifest, normalized_cfg)
    before = _force(original, rotation=rotation, flip=flip)
    sample = _force(ds, rotation=rotation, flip=flip)
    full, _ = normalize_boundary_input(boundary, settings=NORM)
    expected = np.rot90(full[8:72, 8:72], rotation)
    if flip:
        expected = expected[:, ::-1]
    np.testing.assert_array_equal(sample['input_boundary'][0].numpy(), expected)
    for key in ('target', 'valid', 'rgb'):
        assert torch.equal(sample[key], before[key])
    assert torch.equal(sample['raw_input_boundary'], before['input_boundary'])
    assert torch.equal(sample['raw_gap_mask'], before['gap_mask'])
    assert ds.plan_receipt() == original.plan_receipt()
    assert sample['meta']['input_normalization'] == NORM
    assert sample['meta']['input_normalization_anchor_xy'] == [4, 4]
    assert ds.identity['format'] == 'boundary_rgb_data_v2'
    ds.verify_sources()


@pytest.mark.parametrize('mode', ['boundary', 'rgb_boundary'])
def test_dataset_unknown_never_masks_normalized_input_and_none_identity_is_compatible(tmp_path, mode):
    manifest, _, cfg, _, _, ids = _fixture(tmp_path)
    cfg['boundary_naf']['input_mode'] = mode
    old = BoundaryRGBDataset(manifest, cfg)
    explicit_off = deepcopy(cfg)
    explicit_off['boundary_naf']['input_normalization'] = {'mode': 'none', 'kernel_size': 8}
    off = BoundaryRGBDataset(manifest, explicit_off)
    assert old.identity == off.identity
    on = deepcopy(cfg)
    on['boundary_naf']['input_normalization'] = dict(NORM)
    ds = BoundaryRGBDataset(manifest, on)
    sample = _force(ds, box=(0, 0, 64, 64))
    unknown = ids[:64, :64] == 0
    assert sample['input_boundary'][0].numpy()[unknown].any()
    assert sample['raw_input_boundary'][0].numpy()[unknown].any()
    assert not sample['valid'][0].numpy()[unknown].any()
    assert sample['input'].shape[0] == (1 if mode == 'boundary' else 4)
    batch = collate_boundary_rgb([sample, sample])
    assert torch.equal(batch['raw_gap_mask'][0], sample['raw_gap_mask'])
    assert batch['raw_input_boundary'].shape == (2, 1, 64, 64)


@pytest.mark.parametrize('mode', ['boundary', 'rgb_boundary'])
def test_inference_uses_same_full_normalization_before_padding_and_tiles(mode):
    boundary = np.zeros((73, 91), np.float32)
    boundary[2:66, 2:84] = .8
    boundary[31:38, 41:48] = 0
    rgb = rgb_for(boundary.shape)
    model = PointwiseRGB(mode)
    cfg = settings(mode, input_normalization=dict(NORM))
    actual, receipt = monitor.predict_boundary_rgb_tiled(boundary, rgb, model, cfg)
    expected, _ = normalize_boundary_input(boundary, settings=NORM)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    first = model.seen[0][0].numpy()
    np.testing.assert_array_equal(first[-1], np.pad(expected, 3, mode='reflect')[:32, :32])
    if mode == 'rgb_boundary':
        image = np.pad(rgb, ((3, 3), (3, 3), (0, 0)), mode='reflect')[:32, :32]
        np.testing.assert_array_equal(first[:3], image.transpose(2, 0, 1).astype(np.float32)/255.)
    assert receipt['input_normalization']['resolved_anchor_xy'] == [4, 4]


def test_normalized_monitor_keeps_raw_missing_definition_and_extra_adapter_is_once(tmp_path, monkeypatch):
    # 配对图用rawT集合，标准化集合另列；不能用更窄/更宽输入改掉原161点的分母。
    side = 64
    target = torch.zeros(1, side, side)
    target[:, :, 31:33] = 1
    raw = target.clone()
    raw[:, 20:35] = 0
    standardized, _ = normalize_boundary_input(raw[0].numpy(), settings=NORM)
    actual_input = torch.tensor(standardized, dtype=torch.float32)[None]
    rgb = torch.from_numpy(rgb_for((side, side)).transpose(2, 0, 1).copy()).float()/255.
    sample = dict(input=actual_input, input_boundary=actual_input, raw_input_boundary=raw, rgb=rgb,
        target=target, valid=torch.ones_like(target), gap_mask=target*(actual_input == 0),
        raw_gap_mask=target*(raw == 0), source='train_001', meta={})
    cfg = dict(boundary_naf=dict(input_normalization=dict(NORM),
        inference=settings('boundary', input_normalization=dict(NORM)), monitor=dict(pair_indices=[0],
        thumbnail_width=64, source_names=[], full_readout_steps=[])))
    monkeypatch.setattr(monitor, 'original_view', lambda *_: np.ones((side, side), bool))
    result = monitor.save_monitor(PointwiseRGB('boundary'), cfg, tmp_path/'pair', 7, {0: sample})
    expected, _ = real_pair_metrics(raw[0].numpy(), target[0].numpy(), standardized.astype(np.float32),
                                    np.ones((side, side), bool), np.ones((side, side), bool))
    row = result['rows'][0]
    assert row['natural_missing_gt_pixels'] == expected['natural_missing_gt_pixels']
    assert 'normalized_natural_missing_gt_pixels' in row
    assert Image.open(tmp_path/'pair/monitor/s000007/pair_000.jpg').size[0] == 5*64

    # 最终读出只首次增加一个normT-only适配控制，后续只重算候选，不复用旧seed。
    boundary = np.zeros((64, 80), np.float32)
    boundary[:, 34:47] = .9
    image = rgb_for(boundary.shape)
    semantic = np.zeros(boundary.shape, np.float32)
    raw_probability = np.full(boundary.shape, .8, np.float32)
    frozen = frozen_config()
    ids, classes, _ = evaluate_frozen_readout(semantic, raw_probability, boundary, frozen)
    cache = tmp_path/'test_101.npz'
    np.savez_compressed(cache, boundary=boundary, rgb=image, semantic=semantic,
                        raw_probability=raw_probability, instances=ids)
    class_path = tmp_path/'classes.json'
    class_path.write_text(json.dumps(classes))
    control = dict(source='test_101', path=str(cache), control_classes=str(class_path),
                   sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    monkeypatch.setattr(monitor, '_fixed_controls', lambda *_: [(control, frozen)])
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs)
        return evaluate_frozen_readout(*args, **kwargs)
    monkeypatch.setattr(monitor, 'evaluate_frozen_readout', record)
    cfg['boundary_naf']['monitor'].update(pair_indices=[], source_names=['test_101'], full_readout_steps=[0, 1000])
    first = monitor.save_monitor(PointwiseRGB('boundary'), cfg, tmp_path/'full', 0, {})
    assert len(calls) == 4
    assert first['rows'][0]['adapter_only_instances'] >= 0
    assert 'normalized_adapter_only_instances' in first['rows'][0]
    monitor.save_monitor(PointwiseRGB('boundary'), cfg, tmp_path/'full', 1000, {})
    assert len(calls) == 5 and 'reference' not in calls[-1]


def test_monitor_requires_train_inference_normalization_parity(tmp_path):
    cfg = dict(boundary_naf=dict(input_normalization=dict(NORM), inference=settings(), monitor={}))
    with pytest.raises(ValueError, match='must match'):
        monitor.save_monitor(PointwiseRGB(), cfg, tmp_path, 0, {})
