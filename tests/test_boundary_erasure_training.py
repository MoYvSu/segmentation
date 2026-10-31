# -*- coding: utf-8 -*-
"""擦断任务的训练入口合同：额外日志不改变真实loss/更新，保存严格可续。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import train_boundary_rgb as module
from models.boundary_restore import state_sha256
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def config():
    return load_config('config/train/boundary_erasure.yaml')


def batch():
    target = torch.zeros(1, 1, 96, 96)
    target[..., 8:88, 48] = 1
    valid = torch.ones_like(target)
    clean = torch.zeros_like(target)
    clean[..., 8:88, 44:52] = 1
    clean[..., 18:26, :] = 0  # 固定已有天然缺口。
    request = torch.zeros_like(target)
    request[..., 40:56, 40:57] = 1
    damaged = clean.masked_fill(request.bool(), 0)
    near = lambda x: torch.nn.functional.max_pool2d(x, 5, 1, 2) > .5
    artificial = target.bool() & valid.bool() & near(clean) & ~near(damaged)
    meta = dict(requested=True, applied=True, reason='applied', requested_count=1, applied_count=1,
        deleted_input_pixels=int((clean.bool() & request.bool()).sum()),
        artificial_gap_gt_points=int(artificial.sum()), segments=[dict(provenance='B')])
    return dict(input=damaged, input_boundary=damaged, clean_input_boundary=clean,
        erasure_mask=request, artificial_gap_mask=artificial.float(), target=target, valid=valid,
        gap_mask=(target.bool() & ~near(clean)).float(), meta=[{'erasure': meta}])


def small_models(cfg):
    cfg = deepcopy(cfg)
    cfg['model'] = dict(width=2, encoder_blocks=[1], middle_blocks=1, decoder_blocks=[1])
    cfg['discriminator'] = dict(width=2)
    cfg['training']['micro_batch_size'] = 1
    cfg['training']['minimum_free_gib'] = .001
    g, d = module.build_models(cfg, 'cpu')
    go, do = module.optimizers(g, d, cfg)
    return cfg, g, d, go, do


def test_recipe_keeps_control_except_erasure_output_and_receipted_monitor_draw():
    current, control = config()['boundary_naf'], load_config('config/train/boundary_width.yaml')['boundary_naf']
    module.validate_config({'boundary_naf': current})
    current = deepcopy(current)
    current.pop('erasure')
    current['output_dir'] = control['output_dir']
    assert current['monitor'].pop('erasure_pair_indices') == []
    # 实际8核控制运行已额外固定9921，公开父YAML只保留前六个。
    assert current['monitor']['pair_indices'] == [0, 5, 17, 29, 41, 63, 9921]
    assert current['monitor']['pair_indices'][:-1] == control['monitor']['pair_indices']
    current['monitor']['pair_indices'] = control['monitor']['pair_indices']
    assert current == control
    assert current['training']['main_updates'] == 1000
    assert current['training']['effective_batch_size'] == 16
    assert current['monitor']['full_readout_steps'] == [0, 1000]


@pytest.mark.parametrize('mutate', [
    lambda c: c['boundary_naf']['erasure'].update(enabled='true'),
    lambda c: c['boundary_naf']['erasure'].update(version='old'),
    lambda c: c['boundary_naf']['rgb_loss'].update(coverage_mode='coherent3'),
    lambda c: c['boundary_naf']['input_normalization'].update(kernel_size=6),
])
def test_erasure_cannot_silently_enter_with_bad_contract(mutate):
    cfg = config()
    mutate(cfg)
    with pytest.raises(ValueError):
        module.validate_config(cfg)


def test_disabled_batch_adds_no_audit_and_mixed_recording_rejected():
    assert module.erasure_batch_audit({'meta': [{}, {}]}) == {}
    with pytest.raises(ValueError, match='回执缺失'):
        module.erasure_batch_audit({'meta': [{'erasure': {}}, {}]})


def test_source_identity_only_extends_enabled_dependency_set(monkeypatch):
    enabled = config()
    disabled = load_config('config/train/boundary_width.yaml')
    cfg, g, d, _, _ = small_models(enabled['boundary_naf'])
    # 只验证源码依赖和配方身份，不需读取真实赛题资产。
    monkeypatch.setattr(module, 'file_sha', lambda path: str(path))
    data = SimpleNamespace(identity={'synthetic': True})
    old = module.source_identity(disabled, data, g, d)
    new = module.source_identity(enabled, data, g, d)
    added = set(new['source_sha256']) - set(old['source_sha256'])
    assert len(old['source_sha256']) == 16
    assert added == {'utils/boundary_erasure.py', 'utils/boundary_core_supervision.py',
                     'utils/boundary_rgb_inputs.py'}
    assert 'erasure' not in old['recipe'] and 'erasure_pair_indices' not in old['recipe']['monitor']
    assert new['recipe']['monitor']['erasure_pair_indices'] == []
    assert new['initial_model_sha256'] == old['initial_model_sha256']
    assert new['initial_discriminator_sha256'] == old['initial_discriminator_sha256']


def test_actual_loss_and_gradient_update_unchanged_by_added_metadata_then_strict_reload(tmp_path):
    cfg, g, d, go, do = small_models(config()['boundary_naf'])
    _, plain_g, plain_d, plain_go, plain_do = small_models(cfg)
    before = state_sha256(g)
    sample = batch()
    plain = {k: v for k, v in sample.items() if k not in
             {'meta', 'clean_input_boundary', 'erasure_mask', 'artificial_gap_mask'}}
    actual = module.update(g, d, go, do, sample, cfg, 64)
    expected = module.update(plain_g, plain_d, plain_go, plain_do, plain, cfg, 64)
    assert state_sha256(g) == state_sha256(plain_g) != before
    assert state_sha256(d) == state_sha256(plain_d)
    assert actual['discriminator_updated'] and actual['erasure_applied_samples'] == 1
    assert actual['erasure_segments_by_provenance'] == {'B': 1}
    assert all(actual[key] == value for key, value in expected.items())
    assert actual['grad_norm_g'] > 0
    identity = {'recipe': deepcopy(cfg), 'data_contract': 'synthetic_erasure_v1'}
    module.save_checkpoint(tmp_path/'latest.pt', g, d, go, do, cfg, identity, 65, [0, 1])
    _, fresh_g, fresh_d, fresh_go, fresh_do = small_models(cfg)
    assert module.load_checkpoint(tmp_path/'latest.pt', fresh_g, fresh_d, fresh_go, fresh_do,
                                  cfg, identity) == (65, [0, 1])
    assert state_sha256(fresh_g) == state_sha256(g)
    assert len(fresh_go.state) == len(go.state) > 0
    with pytest.raises(ValueError, match='来源或源码'):
        module.load_checkpoint(tmp_path/'latest.pt', fresh_g, fresh_d, fresh_go, fresh_do,
                               cfg, {'different': True})


def test_artificial_and_natural_missing_are_fixed_separate_cohorts():
    cfg, sample = config()['boundary_naf'], batch()
    logits = torch.full_like(sample['target'], -8.)
    logits[sample['target'].bool()] = 8
    metrics = module.reconstruction_metrics(logits, sample, cfg, clean_logits=logits)
    assert metrics['artificial_missing_points'] > 0 and metrics['natural_missing_points'] > 0
    assert metrics['missing_points'] == metrics['artificial_missing_points'] + metrics['natural_missing_points']
    assert metrics['artificial_missing_above065'] == 1
    assert metrics['clean_output_original_supported_above065'] == 1


def test_smoke_cannot_pass_when_artificial_gap_learning_is_hidden_by_aggregate():
    before = dict(loss=1., missing_points=100, missing_probability=.4,
        supported_points=100, supported_probability=.9, recorded_interior_input_points=0,
        recorded_interior_input_probability=None, erasure_enabled=True,
        natural_missing_points=10, artificial_missing_points=90, artificial_missing_probability=.2,
        clean_output_original_supported_probability=.9)
    after = dict(before, loss=.1, missing_probability=.9, supported_probability=.95,
                 artificial_missing_probability=.3)
    checks = module.smoke_checks(before, after, True, 64)
    assert checks['reconstruction_reduced'] and checks['missing_progress']
    assert not checks['artificial_missing_progress']
