# -*- coding: utf-8 -*-
"""RGB条件补边的合成CPU合同：不读取赛题数据或训练权重。"""
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize
import torch

from data.boundary_rgb import BoundaryRGBDataset, collate_boundary_rgb
from models.boundary_naf import BoundaryNAFNet
from models.boundary_rgb import BoundaryRGBNAFNet
from utils.image_io import write_image


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value), encoding='utf-8')


def _fixture(tmp_path, shape=(80, 96)):
    y, x = np.indices(shape)
    rgb = np.stack([y % 256, x % 256, (y * 3 + x * 2) % 256], axis=-1).astype(np.uint8)
    image = tmp_path / 'train_001.png'
    write_image(image, rgb[..., ::-1])
    ids = np.ones(shape, np.uint16)
    ids[:, shape[1] // 2:] = 2
    ids[12:20, 12:20] = 0
    original = ids > 0
    original[32:40, shape[1] // 2 - 3:shape[1] // 2 + 3] = False
    gt = tmp_path / 'train_001_gt.npz'
    np.savez_compressed(gt, instance_map=ids, original_covered=original,
                        filled=(ids > 0) & ~original, residual_unknown=ids == 0)
    classes = tmp_path / 'train_001_class.json'
    _json(classes, {'1': 0, '2': 1})
    row = dict(source='train_001', image=str(image), gt=str(gt), classes=str(classes),
               hashes={key + '_sha256': _sha(value) for key, value in
                       dict(image=image, gt=gt, classes=classes).items()}, shape=list(shape))
    processed = tmp_path / 'processed.json'
    _json(processed, {'rows': [row]})
    boundary = np.zeros(shape, np.float32)
    boundary[:, 13:18] = .8
    boundary[30:36, :] = .6
    boundary[:, shape[1] // 2] = .49  # 低于阈值的线不能偷偷保留成连续B输入。
    heat = tmp_path / 'train_001_heat.npz'
    np.savez_compressed(heat, boundary=boundary)
    heat_manifest = tmp_path / 'real.json'
    _json(heat_manifest, dict(format='boundary_naf_real_heatmaps_v1', complete=True,
        no_test_training=True, local_restoration={'mode': 'zero', 'model': 'restorer'},
        models={'backend': '1' * 64, 'restorer': '2' * 64}, rows=[dict(source='train_001',
        heat_path=heat.name, heat_sha256=_sha(heat), **row['hashes'], shape=list(shape))]))
    config = dict(boundary_naf=dict(input_mode='rgb_boundary', seed=20261026,
        real_input_models={'backend': '1' * 64, 'restorer': '2' * 64},
        real_input_manifest=str(heat_manifest), data=dict(expected_sources=1,
        crop_size=64, stride=32, augmentations=8, samples_per_epoch=16,
        valid_margin=4, crop_margin=4)))
    return processed, heat_manifest, config, rgb, boundary, ids


def _force(ds, box=(8, 8, 64, 64), rotation=0, flip=False):
    ds.plan[0] = dict(ds.plan[0], box=list(box), rotation=rotation, flip=flip,
                      augmentation=rotation + 4 * int(flip))
    return ds[0]


def _small(mode='rgb_boundary', seed=13):
    return BoundaryRGBNAFNet(width=4, encoder_blocks=(1, 1), middle_blocks=1,
        decoder_blocks=(1, 1), input_mode=mode, seed=seed)


@pytest.mark.parametrize('shape', [(2, 4, 32, 32), (1, 4, 31, 47), (1, 4, 1, 3)])
def test_one_channel_output_and_boundary_only_global_skip(shape):
    model = _small()
    inputs = torch.rand(shape)
    inputs[:, 3:4] = (inputs[:, 3:4] >= .5).float()
    output = model(inputs)
    assert output.shape == (shape[0], 1, *shape[-2:])
    assert torch.equal(output, model.forward_logits(inputs).sigmoid())
    torch.testing.assert_close(model.residual(inputs), output - inputs[:, 3:4], rtol=0, atol=0)
    with torch.no_grad():
        model.ending.weight.zero_()
        model.ending.bias.zero_()
    assert torch.equal(model.forward_logits(inputs), inputs[:, 3:4])


def test_common_initialization_rgb_zero_but_learnable_and_rng_local():
    before = torch.get_rng_state()
    rgb_model, boundary_model = _small(), _small('boundary')
    assert torch.equal(before, torch.get_rng_state())
    rgb_state, boundary_state = rgb_model.state_dict(), boundary_model.state_dict()
    assert not rgb_state['intro.weight'][:, :3].any()
    assert torch.equal(rgb_state['intro.weight'][:, 3:4], boundary_state['intro.weight'])
    for key in boundary_state:
        if key != 'intro.weight':
            assert torch.equal(rgb_state[key], boundary_state[key])
    value = torch.rand(2, 4, 32, 32)
    value[:, 3:4] = (value[:, 3:4] >= .5).float()
    torch.testing.assert_close(rgb_model(value), boundary_model(value[:, 3:4]), rtol=1e-6, atol=1e-7)
    rgb_model(value).mean().backward()
    assert rgb_model.intro.weight.grad[:, :3].abs().sum() > 0


def test_checkpoint_strict_reload_and_old_contract_is_unchanged():
    model = _small()
    clone = BoundaryRGBNAFNet(**model.architecture)
    clone.load_state_dict(model.state_dict(), strict=True)
    value = torch.rand(1, 4, 32, 32)
    value[:, 3:4] = (value[:, 3:4] >= .5).float()
    assert torch.equal(model(value), clone(value))
    with pytest.raises(RuntimeError):
        _small('boundary').load_state_dict(model.state_dict(), strict=True)
    damaged = copy.deepcopy(model.state_dict())
    damaged['extra'] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        clone.load_state_dict(damaged, strict=True)
    assert model.checkpoint_format == 'boundary_rgb_naf_v1'
    assert BoundaryNAFNet.checkpoint_format == 'boundary_naf_v1'


@pytest.mark.parametrize('value', [torch.zeros(1, 3, 8, 8), torch.zeros(1, 4, 8, 8, dtype=torch.long),
    torch.full((1, 4, 8, 8), float('nan')), torch.full((1, 4, 8, 8), 1.1),
    torch.full((1, 4, 8, 8), .3)])
def test_model_rejects_invalid_rgb_or_nonbinary_T(value):
    with pytest.raises(ValueError):
        _small()(value)


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in range(4) for f in (False, True)])
def test_rgb_T_target_exactly_share_native_D4_geometry(tmp_path, rotation, flip):
    manifest, _, cfg, rgb, boundary, ids = _fixture(tmp_path)
    ds = BoundaryRGBDataset(manifest, cfg)
    sample = _force(ds, rotation=rotation, flip=flip)
    expected_rgb = np.rot90(rgb[8:72, 8:72], rotation)
    expected_T = np.rot90(boundary[8:72, 8:72] >= .5, rotation)
    expected_gt = np.rot90(ds.cache['train_001'][0][8:72, 8:72], rotation)
    if flip:
        expected_rgb, expected_T, expected_gt = [v[:, ::-1] for v in (expected_rgb, expected_T, expected_gt)]
    np.testing.assert_array_equal(sample['rgb'].numpy(), expected_rgb.transpose(2, 0, 1).astype(np.float32) / 255.)
    np.testing.assert_array_equal(sample['input_boundary'][0].numpy(), expected_T)
    np.testing.assert_array_equal(sample['target'][0].numpy(), expected_gt)
    assert torch.equal(sample['input'][:3], sample['rgb'])
    assert torch.equal(sample['input'][3:4], sample['input_boundary'])
    assert sample['meta']['resize_scale'] == 1.
    assert not sample['meta']['input_skeletonized']


def test_T_is_not_skeletonized_and_unknown_never_masks_model_inputs(tmp_path):
    manifest, _, cfg, rgb, boundary, ids = _fixture(tmp_path)
    ds = BoundaryRGBDataset(manifest, cfg)
    sample = _force(ds, box=(0, 0, 64, 64))
    binary = boundary[:64, :64] >= .5
    assert binary.sum() > skeletonize(binary).sum() * 2
    np.testing.assert_array_equal(sample['input_boundary'][0].numpy(), binary)
    unknown = ids[:64, :64] == 0
    assert sample['input_boundary'][0].numpy()[unknown].any()
    assert not sample['valid'][0].numpy()[unknown].any()
    np.testing.assert_array_equal(sample['rgb'].numpy(), rgb[:64, :64].transpose(2, 0, 1).astype(np.float32) / 255.)
    assert sample['gap_mask'].bool().any()
    assert not (sample['gap_mask'].bool() & ~sample['valid'].bool()).any()


def test_small_native_source_rgb_and_T_reflect_together_padding_ignored(tmp_path):
    manifest, _, cfg, rgb, boundary, _ = _fixture(tmp_path, shape=(40, 48))
    ds = BoundaryRGBDataset(manifest, cfg)
    sample = _force(ds, box=(0, 0, 40, 48), rotation=1, flip=True)
    expected_rgb = np.rot90(rgb, 1)[:, ::-1]
    expected_T = np.rot90(boundary >= .5, 1)[:, ::-1]
    expected_rgb = np.pad(expected_rgb, ((0, 16), (0, 24), (0, 0)), mode='reflect')
    expected_T = np.pad(expected_T, ((0, 16), (0, 24)), mode='reflect')
    np.testing.assert_array_equal(sample['rgb'].numpy(), expected_rgb.transpose(2, 0, 1).astype(np.float32) / 255.)
    np.testing.assert_array_equal(sample['input_boundary'][0].numpy(), expected_T)
    assert not sample['valid'][0, 48:].any()
    assert not sample['valid'][0, :, 40:].any()


def test_T_only_control_has_same_plan_target_RGB_and_identity_binds_mode(tmp_path):
    manifest, _, cfg, _, _, _ = _fixture(tmp_path)
    rgb = BoundaryRGBDataset(manifest, cfg)
    cfg2 = copy.deepcopy(cfg)
    cfg2['boundary_naf']['input_mode'] = 'boundary'
    boundary = BoundaryRGBDataset(manifest, cfg2)
    assert rgb.plan_receipt() == boundary.plan_receipt()
    assert rgb.verify_sources() != boundary.verify_sources()
    for index in range(len(rgb)):
        a, b = rgb[index], boundary[index]
        for key in ('target', 'valid', 'input_boundary', 'rgb', 'gap_mask'):
            assert torch.equal(a[key], b[key])
        assert torch.equal(b['input'], a['input_boundary'])
    batch = collate_boundary_rgb([rgb[0], rgb[1]])
    assert batch['input'].shape == (2, 4, 64, 64)
    assert batch['rgb'].shape == (2, 3, 64, 64)
    assert batch['target'].shape == (2, 1, 64, 64)
    json.dumps(rgb.identity, allow_nan=False)


def test_cached_training_step_does_not_reload_or_mutate_maps(tmp_path, monkeypatch):
    manifest, _, cfg, _, _, _ = _fixture(tmp_path)
    ds = BoundaryRGBDataset(manifest, cfg)
    identity = ds.verify_sources()
    with monkeypatch.context() as patch:
        patch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('No hot-step NPZ read')))
        for i in range(len(ds)):
            sample = ds[i]
            sample['input'].zero_()
            sample['target'].zero_()
    assert ds.verify_sources() == identity
    ds.input_boundaries['train_001'][2, 2] = ~ds.input_boundaries['train_001'][2, 2]
    with pytest.raises(RuntimeError, match='unthinned T changed'):
        ds.verify_sources()
