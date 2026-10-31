# -*- coding: utf-8 -*-
"""上采样结构对照的训练合同；仅用合成CPU数据，不加载赛题图或GT。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import train_boundary_rgb as trainer
from models.boundary_restore import state_sha256
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return load_config('config/train/boundary_resize.yaml')


def small_models(cfg):
    """仅缩小测试模型，保留新上采样模式及真实优化器/检查点入口。"""
    cfg = deepcopy(cfg)
    mode = cfg['model'].get('upsampling')
    cfg['model'] = dict(width=4, encoder_blocks=[1, 1], middle_blocks=1,
                        decoder_blocks=[1, 1])
    if mode is not None:
        cfg['model']['upsampling'] = mode
    cfg['discriminator'] = dict(width=2)
    cfg['training']['micro_batch_size'] = 1
    cfg['training']['minimum_free_gib'] = .001
    g, d = trainer.build_models(cfg, 'cpu')
    go, do = trainer.optimizers(g, d, cfg)
    return cfg, g, d, go, do


def synthetic_batch():
    target = torch.zeros(1, 1, 96, 96)
    target[..., 8:88, 48] = 1
    valid = torch.ones_like(target)
    valid[..., :4, :] = valid[..., -4:, :] = 0
    valid[..., :, :4] = valid[..., :, -4:] = 0
    clean = torch.zeros_like(target)
    clean[..., 8:88, 44:52] = 1
    clean[..., 18:26, :] = 0  # 保留一处天然缺口，与人工擦断分开。
    request = torch.zeros_like(target)
    request[..., 40:56, 40:57] = 1
    damaged = clean.masked_fill(request.bool(), 0)
    near = lambda value: torch.nn.functional.max_pool2d(value, 5, 1, 2) > .5
    artificial = target.bool() & valid.bool() & near(clean) & ~near(damaged)
    meta = dict(requested=True, applied=True, reason='applied', requested_count=1,
        applied_count=1, deleted_input_pixels=int((clean.bool() & request.bool()).sum()),
        artificial_gap_gt_points=int(artificial.sum()), segments=[dict(provenance='B')])
    return dict(input=damaged, input_boundary=damaged, clean_input_boundary=clean,
        erasure_mask=request, artificial_gap_mask=artificial.float(), target=target,
        valid=valid, gap_mask=(target.bool() & ~near(clean)).float(), meta=[{'erasure': meta}])


def test_recipe_changes_only_up_blocks_and_private_output():
    current = config()
    trainer.validate_config(current)
    actual = deepcopy(current['boundary_naf'])
    control = load_config('config/train/boundary_erasure.yaml')['boundary_naf']
    assert actual['model'].pop('upsampling') == 'project_bilinear_v1'
    actual['output_dir'] = control['output_dir']
    assert actual == control
    assert actual['input_mode'] == actual['inference']['input_mode'] == 'boundary'
    assert 'shift_consistency' not in actual
    assert 'restored_rgb_manifest' not in actual
    assert actual['training']['main_updates'] == 1000
    assert actual['training']['effective_batch_size'] == actual['training']['micro_batch_size'] == 16
    assert actual['monitor']['pair_indices'] == [0, 5, 17, 29, 41, 63, 9921]
    assert actual['monitor']['full_readout_steps'] == [0, 1000]


def test_shared_backbone_and_discriminator_still_have_common_random_tensors():
    cfg, resized, d, _, _ = small_models(config()['boundary_naf'])
    _, control, old_d, _, _ = small_models(load_config('config/train/boundary_erasure.yaml')['boundary_naf'])
    assert resized.architecture['upsampling'] == 'project_bilinear_v1'
    assert 'upsampling' not in control.architecture
    common = {key: value for key, value in control.state_dict().items() if not key.startswith('ups.')}
    assert common
    assert all(torch.equal(value, resized.state_dict()[key]) for key, value in common.items())
    assert state_sha256(d) == state_sha256(old_d)
    assert state_sha256(resized) != state_sha256(control)
    assert sum(p.numel() for p in resized.parameters()) < sum(p.numel() for p in control.parameters())


def test_existing_source_identity_records_new_architecture_without_new_data_dependencies(monkeypatch):
    resized_config = config()
    control_config = load_config('config/train/boundary_erasure.yaml')
    _, g, d, _, _ = small_models(resized_config['boundary_naf'])
    _, old_g, old_d, _, _ = small_models(control_config['boundary_naf'])
    monkeypatch.setattr(trainer, 'file_sha', lambda path: str(path))
    data = SimpleNamespace(identity={'synthetic': True})
    current = trainer.source_identity(resized_config, data, g, d)
    old = trainer.source_identity(control_config, data, old_g, old_d)
    assert current['architecture']['upsampling'] == 'project_bilinear_v1'
    assert current['dataset'] == old['dataset']
    assert current['assets_sha256'] == old['assets_sha256']
    assert current['source_sha256'] == old['source_sha256']
    assert {'models/boundary_naf.py', 'models/boundary_rgb.py'} <= set(current['source_sha256'])
    assert current['initial_model_sha256'] != old['initial_model_sha256']
    assert current['initial_discriminator_sha256'] == old['initial_discriminator_sha256']


def test_new_generator_really_updates_and_strict_resume_repeats_next_G_D_step(tmp_path):
    cfg, g, d, go, do = small_models(config()['boundary_naf'])
    sample = synthetic_batch()
    before = state_sha256(g)
    stat = trainer.update(g, d, go, do, sample, cfg, 64)
    assert stat['generator_updated'] and stat['discriminator_updated']
    assert state_sha256(g) != before
    assert stat['grad_norm_g'] > 0
    assert stat['erasure_applied_samples'] == 1
    assert 'shift_consistency_enabled' not in stat
    assert all(block[0].weight.grad is not None and bool(block[0].weight.grad.abs().sum())
               for block in g.ups)
    identity = dict(recipe=deepcopy(cfg), test='resize_single_variable')
    checkpoint = tmp_path/'latest.pt'
    trainer.save_checkpoint(checkpoint, g, d, go, do, cfg, identity, 65, [0, 1])
    _, fresh, fresh_d, fresh_go, fresh_do = small_models(cfg)
    assert trainer.load_checkpoint(checkpoint, fresh, fresh_d, fresh_go, fresh_do,
                                   cfg, identity) == (65, [0, 1])
    assert state_sha256(g) == state_sha256(fresh)
    assert state_sha256(d) == state_sha256(fresh_d)
    expected = trainer.update(g, d, go, do, sample, cfg, 65)
    actual = trainer.update(fresh, fresh_d, fresh_go, fresh_do, sample, cfg, 65)
    assert expected == actual
    assert state_sha256(g) == state_sha256(fresh)
    assert state_sha256(d) == state_sha256(fresh_d)


def test_trainer_rejects_checkpoint_with_other_up_architecture_before_weight_loading(tmp_path):
    cfg, g, d, go, do = small_models(config()['boundary_naf'])
    trainer.update(g, d, go, do, synthetic_batch(), cfg, 0)
    identity = dict(recipe=deepcopy(cfg), test='resize_strict_architecture')
    checkpoint = tmp_path/'latest.pt'
    trainer.save_checkpoint(checkpoint, g, d, go, do, cfg, identity, 1, [0])
    _, original, old_d, old_go, old_do = small_models(
        load_config('config/train/boundary_erasure.yaml')['boundary_naf'])
    before = state_sha256(original)
    with pytest.raises(ValueError, match='架构'):
        trainer.load_checkpoint(checkpoint, original, old_d, old_go, old_do, cfg, identity)
    assert state_sha256(original) == before


def test_old_default_checkpoint_still_uses_legacy_architecture_and_strict_loader(tmp_path):
    cfg, g, d, go, do = small_models(load_config('config/train/boundary_erasure.yaml')['boundary_naf'])
    assert 'upsampling' not in g.architecture
    trainer.update(g, d, go, do, synthetic_batch(), cfg, 0)
    identity = dict(recipe=deepcopy(cfg), test='unchanged_default_strict_checkpoint')
    checkpoint = tmp_path/'original.pt'
    trainer.save_checkpoint(checkpoint, g, d, go, do, cfg, identity, 1, [0])
    _, fresh, fresh_d, fresh_go, fresh_do = small_models(cfg)
    assert trainer.load_checkpoint(checkpoint, fresh, fresh_d, fresh_go, fresh_do,
                                   cfg, identity) == (1, [0])
    assert state_sha256(g) == state_sha256(fresh)
    assert state_sha256(d) == state_sha256(fresh_d)
