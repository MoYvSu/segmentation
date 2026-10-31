# -*- coding: utf-8 -*-
"""RGB＋填谷＋零GAN联合合同；仅合成CPU图，不读取赛题数据。"""
from copy import deepcopy

import numpy as np
import pytest
import torch

import train_boundary_rgb as trainer
from data.boundary_rgb import BoundaryRGBDataset
from models.boundary_restore import state_sha256
from test_boundary_erasure import _dataset_config
from test_boundary_rgb import _force, _sha
from test_boundary_rgb_restored import _restored
from test_boundary_resize_training import small_models, synthetic_batch
from test_boundary_ridge_training import tree_equal
from utils.boundary_erasure import DEFAULTS
from utils.boundary_rgb_inputs import clean_model_input
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def color_config():
    return load_config('config/train/boundary_color.yaml')


def test_color_changes_only_declared_conditioning_fields():
    current = color_config()
    trainer.validate_config(current)
    a, b = deepcopy(current['boundary_naf']), load_config('config/train/boundary_nogan.yaml')['boundary_naf']
    assert a['ridge_supervision'] == b['ridge_supervision'] and a['training']['adversarial_weight'] == 0
    a['output_dir'] = b['output_dir']
    a['input_mode'] = b['input_mode']
    a.pop('restored_rgb_manifest')
    a['monitor'].pop('restored_rgb_manifest')
    a['inference']['input_mode'] = b['inference']['input_mode']
    a['inference'].pop('rgb_input_format')
    assert a == b


@pytest.mark.parametrize('change', [
    lambda c: c.pop('restored_rgb_manifest'),
    lambda c: c['monitor'].pop('restored_rgb_manifest'),
    lambda c: c.update(erasure={'enabled': False}),
    lambda c: c['model'].update(upsampling='pixelshuffle'),
    lambda c: c['rgb_loss'].update(coverage_mode='coherent3'),
])
def test_color_does_not_relax_data_or_supervision_gates(change):
    current = color_config()
    change(current['boundary_naf'])
    with pytest.raises(ValueError):
        trainer.validate_config(current)


def test_T_only_trainer_keeps_legacy_architecture_validation():
    # 数据层既有bilinear约束另有专门测试；trainer不额外收紧旧单T架构门槛。
    current = load_config('config/train/boundary_nogan.yaml')
    current['boundary_naf']['model']['upsampling'] = 'pixelshuffle'
    trainer.validate_config(current)


@pytest.mark.parametrize('rotation,flip', [(r, f) for r in range(4) for f in (False, True)])
def test_RGB_GT_instances_erasure_and_padding_share_native_coordinates(tmp_path, rotation, flip):
    manifest, cfg = _dataset_config(tmp_path)
    cfg['boundary_naf'].update(model={'upsampling': 'project_bilinear_v1'},
        erasure=dict(DEFAULTS, probability=.5), ridge_supervision={'enabled': True})
    control = BoundaryRGBDataset(manifest, cfg)
    raw = control.rgb_images['train_001']
    path, _, restored = _restored(tmp_path, raw, image_sha=_sha(tmp_path/'train_001.png'))
    rgb_cfg = deepcopy(cfg)
    rgb_cfg['boundary_naf'].update(input_mode='rgb_boundary', restored_rgb_manifest=str(path))
    candidate = BoundaryRGBDataset(manifest, rgb_cfg)
    assert candidate.plan_receipt() == control.plan_receipt()
    a = _force(control, box=(0, 0, 112, 96), rotation=rotation, flip=flip)
    b = _force(candidate, box=(0, 0, 112, 96), rotation=rotation, flip=flip)
    for key in ('target', 'valid', 'ridge_instances', 'fully_known_mask', 'input_boundary',
                'clean_input_boundary', 'erasure_mask', 'artificial_gap_mask', 'image'):
        assert torch.equal(a[key], b[key]), key
    expected = np.rot90(restored[:112, :96], rotation)
    if flip:
        expected = expected[:, ::-1]
    expected = np.pad(expected, ((0, 128-expected.shape[0]), (0, 128-expected.shape[1]), (0, 0)), mode='reflect')
    np.testing.assert_array_equal(b['rgb'].numpy(), expected.transpose(2, 0, 1))
    assert torch.equal(b['input'][:3], b['rgb']) and torch.equal(b['input'][3:], a['input'])
    clean = clean_model_input(b, input_mode='rgb_boundary')
    assert torch.equal(clean[:3], b['rgb']) and torch.equal(clean[3:], b['clean_input_boundary'])
    assert not b['valid'][b['ridge_instances'] == 0].any()
    assert candidate.verify_sources()


def _batch():
    value = synthetic_batch()
    value['ridge_instances'] = torch.ones_like(value['target'], dtype=torch.int64)
    value['ridge_instances'][..., 48:] = 2
    rgb = torch.rand(1, 3, 96, 96, generator=torch.Generator().manual_seed(71))
    value['input'] = torch.cat([rgb, value['input']], dim=1)
    value['rgb'] = rgb
    return value


def test_actual_NAF_Adam_zero_GAN_ridge_RGB_update_and_strict_resume(tmp_path, monkeypatch):
    cfg, g, d, go, do = small_models(color_config()['boundary_naf'])
    sample = _batch()
    old_g, old_d, old_do = deepcopy(g.state_dict()), deepcopy(d.state_dict()), deepcopy(do.state_dict())
    assert not torch.count_nonzero(g.intro.weight[:, :3])
    def forbidden(*args, **kwargs):
        raise AssertionError('Zero GAN must never forward D')
    monkeypatch.setattr(d, 'forward', forbidden)
    stat = trainer.update(g, d, go, do, sample, cfg, 0)
    assert stat['generator_updated'] and not stat['discriminator_updated']
    assert stat['ridge_eligible_points'] > 0 and stat['ridge_positions'] > 0
    assert stat['ridge_loss'] > 0
    assert torch.linalg.vector_norm(g.intro.weight.grad[:, :3]) > 0
    assert torch.count_nonzero(g.intro.weight[:, :3])
    assert not tree_equal(old_g, g.state_dict()) and tree_equal(old_d, d.state_dict())
    assert tree_equal(old_do, do.state_dict()) and not do.state
    identity = dict(recipe=deepcopy(cfg), test='RGB_zeroGAN_ridge_joint')
    path = tmp_path/'latest.pt'
    trainer.save_checkpoint(path, g, d, go, do, cfg, identity, 1, [0, 1])
    _, fresh, fresh_d, fresh_go, fresh_do = small_models(cfg)
    assert trainer.load_checkpoint(path, fresh, fresh_d, fresh_go, fresh_do, cfg, identity) == (1, [0, 1])
    monkeypatch.setattr(fresh_d, 'forward', forbidden)
    before_rng = trainer.capture_rng()
    a = trainer.update(g, d, go, do, sample, cfg, 1)
    after_rng = trainer.capture_rng()
    trainer.restore_rng(before_rng)
    b = trainer.update(fresh, fresh_d, fresh_go, fresh_do, sample, cfg, 1)
    actual_rng = trainer.capture_rng()
    assert a == b and after_rng.keys() == actual_rng.keys()
    for key, value in after_rng.items():
        if key == 'numpy':
            assert value[0] == actual_rng[key][0] and value[2:] == actual_rng[key][2:]
            np.testing.assert_array_equal(value[1], actual_rng[key][1])
        else:
            assert tree_equal(value, actual_rng[key]), key
    assert state_sha256(g) == state_sha256(fresh) and state_sha256(d) == state_sha256(fresh_d)
    assert tree_equal(go.state_dict(), fresh_go.state_dict())
    assert tree_equal(do.state_dict(), fresh_do.state_dict())
