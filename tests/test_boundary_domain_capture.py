# -*- coding: utf-8 -*-
"""合成图验证单窗采集的原生像素、历史seed与实际512网格契约。"""
import json

import numpy as np
import pytest
import torch

from tools.capture_boundary_domain import (
    RoiBlendMap, capture_one, load_samples, roi_window_plan, seed_for, select_window,
)


def test_original_test_seed_is_not_subset_index():
    row = dict(source='test_100', domain='test', source_index=31)
    assert seed_for(row, 4)['local_seed'] == 314159+31+100000+1024*1000+4
    with pytest.raises(ValueError, match='original'):
        seed_for({**row, 'source_index': 0}, 4)


def test_training_namespaces_stable_and_independent_of_group():
    row = dict(source='synthetic', domain='train_gt', source_index=37)
    seeds = seed_for(row, 2)
    assert seeds['train_namespace_offset'] == 2000000
    assert seeds == seed_for({**row, 'domain': 'train_extra'}, 2)
    assert seeds['local_seed'] == 314159+2000000+37+100000+1024*1000+2


def test_selected_window_is_true_native_deployment_box():
    index, box = select_window((2048, 2048), {})
    assert index == 4 and box == (768, 768, 1792, 1792)
    assert select_window((2048, 2048), {'window': index, 'box': list(box)}) == (index, box)
    with pytest.raises(ValueError):
        select_window((2048, 2048), {'window': index, 'box': [0, 0, 1024, 1024]})
    with pytest.raises(ValueError, match='requires'):
        select_window((900, 1100), {})


def test_rgb_and_native_probability_contract_without_gpu():
    from tools.probe_affinity_patch import tensor_from_rgb
    from utils.affinity_deployment import crop_letterbox_output
    rgb = np.zeros((1100, 1200, 3), np.uint8)
    rgb[..., 0] = 61
    rgb[..., 1] = 123
    rgb[..., 2] = 207
    box = (11, 17, 1035, 1041)
    seeds = []
    def restore(_model, image, seed):
        seeds.append(seed)
        assert torch.equal(image[:, 0], torch.full_like(image[:, 0], 61/255))
        return image
    def grid(_model, image, _config):
        value = torch.linspace(0, 1, 512, dtype=torch.float32)
        return value[None, None, None].expand(1, 1, 512, 512).contiguous()
    arrays, audit = capture_one(None, None, rgb, box, {}, 12345, device='cpu',
        operators=(tensor_from_rgb, restore, grid, crop_letterbox_output))
    assert seeds == [12345] and np.array_equal(arrays['rgb1024'], rgb[11:1035, 17:1041])
    assert arrays['rgb1024'].dtype == np.uint8
    assert arrays['grid512'].shape == (512, 512) and arrays['boundary1024'].shape == (1024, 1024)
    assert arrays['boundary1024'].dtype == np.float32
    expected = crop_letterbox_output(torch.from_numpy(arrays['grid512'])[None, None], 1024, 0, 0, (1024, 1024))
    assert np.array_equal(expected[0, 0].numpy(), arrays['boundary1024'])
    assert audit['crop512_in_window_xyxy'] == [256, 256, 768, 768]


def test_wrong_grid_cannot_silently_be_upsampled():
    from tools.probe_affinity_patch import tensor_from_rgb
    from utils.affinity_deployment import crop_letterbox_output
    with pytest.raises(RuntimeError, match='actual512'):
        capture_one(None, None, np.zeros((1024, 1024, 3), np.uint8), (0, 0, 1024, 1024), {}, 1,
            device='cpu', operators=(tensor_from_rgb, lambda m, x, s:x,
            lambda m, x, c:torch.zeros((1, 1, 256, 256)), crop_letterbox_output))


def test_manifest_requires_complete_unique_frozen_sources(tmp_path):
    row = dict(source='synthetic', path='/private/synthetic.png', input_sha256='a'*64,
               source_index=0, domain='train_gt')
    manifest = dict(format='boundary_domain_samples_v1', complete=True, files=[row])
    path = tmp_path/'samples.json'
    path.write_text(json.dumps(manifest))
    assert load_samples(path)['files'] == [row]
    path.write_text(json.dumps({**manifest, 'files': [row, row]}))
    with pytest.raises(ValueError, match='unique'):
        load_samples(path)


def test_roi_blend_is_exact_full_blend_crop_in_original_order():
    from utils.patch_diagnostic import BlendMap, tile_boxes
    shape = (2100, 2600)
    primary, box, roi, selected = roi_window_plan(shape, {})
    assert primary in [index for index, _ in selected]
    assert [index for index, _ in selected] == sorted(index for index, _ in selected)
    full, cropped = BlendMap(shape), RoiBlendMap(shape, roi)
    rng = np.random.default_rng(19)
    for index, tile in enumerate(tile_boxes(shape, 1024, .25)):
        value = rng.random((1024, 1024), dtype=np.float32)
        full.add(tile, value)
        if index in [i for i, _ in selected]:
            cropped.add(tile, value)
    expected, _ = full.finish()
    actual, audit = cropped.finish()
    y, x, y1, x1 = roi
    assert np.array_equal(actual, expected[y:y1, x:x1])
    assert audit['tiles_min_overlap'] >= 1


def test_roi_explicit_coordinate_convention_and_original_indices():
    shape = (1936, 2584)
    index, box, roi, selected = roi_window_plan(shape, {})
    assert len(selected) < len(__import__('utils.patch_diagnostic', fromlist=['tile_boxes']).tile_boxes(shape, 1024))
    assert any(i != relative for relative, (i, _) in enumerate(selected))
    assert roi_window_plan(shape, dict(roi_box=list(roi), roi_box_format='yxyx'))[2] == roi
    y, x, y1, x1 = roi
    assert roi_window_plan(shape, dict(roi_box=[x, y, x1, y1], roi_box_format='xyxy'))[2] == roi
    with pytest.raises(ValueError, match='explicitly'):
        roi_window_plan(shape, dict(roi_box=list(roi)))
    with pytest.raises(ValueError, match='differs'):
        roi_window_plan(shape, dict(source_shape=[1937, 2584]))
    with pytest.raises(ValueError, match='differs'):
        roi_window_plan(shape, dict(roi_box=[y, x+1, y1, x1+1], roi_box_format='yxyx'))


def test_roi_invalid_or_uncovered_region_is_rejected():
    with pytest.raises(ValueError, match='outside'):
        RoiBlendMap((100, 100), (-1, 0, 80, 80))
    blend = RoiBlendMap((100, 100), (20, 30, 80, 90))
    with pytest.raises(ValueError, match='Uncovered'):
        blend.finish()


def test_manifest_keeps_anchor_outside_statistics(tmp_path):
    row = dict(source='synthetic', path='/private/train.png', file_stem='train',
               input_sha256='a'*64, source_index=0, domain='train_gt')
    anchor = dict(source='test_100', path='/private/test_100.png', input_sha256='b'*64,
                  source_index=31, domain='test', outside_statistical_sample=True)
    manifest = dict(format='boundary_domain_samples_v1', complete=True, files=[row], anchors=[anchor])
    path = tmp_path/'samples.json'
    path.write_text(json.dumps(manifest))
    assert load_samples(path)['anchors'] == [anchor]
    del anchor['outside_statistical_sample']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='excluded'):
        load_samples(path)


def test_same_sample_can_also_be_an_exact_reference_anchor(tmp_path):
    row = dict(source='test_100', path='/private/test_100.png', input_sha256='b'*64,
               source_index=31, domain='test')
    manifest = dict(format='boundary_domain_samples_v1', complete=True, files=[row],
                    anchors=[{**row, 'outside_statistical_sample': True}])
    path = tmp_path/'samples.json'
    path.write_text(json.dumps(manifest))
    assert len(load_samples(path)['files']) == 1
    manifest['anchors'][0]['input_sha256'] = 'c'*64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='identities'):
        load_samples(path)
