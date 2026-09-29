# -*- coding: utf-8 -*-
"""尺度候选只能同步裁切既有目标，并继续严格复用原控制。"""
from copy import deepcopy
import json

import pytest
import torch

from data.semantic_scale import apply_scale_image, apply_scale_target
from train_semantic_consistency import gray_unlabelled_loss, verify_lora_input
from tools import run_semantic_scale as runner
from utils.config import load_config


def test_scale_config_changes_only_views_and_output():
    old = load_config('config/train/semantic_lora.yaml')
    new = load_config('config/train/semantic_scale.yaml')
    assert runner.base_config(old) == runner.base_config(new)
    assert old['direct_semantic_affinity']['semantic_loss'] == new['direct_semantic_affinity']['semantic_loss']
    assert old['inference'] == new['inference']


def test_formal_scale_pipeline_reuses_control_and_runs_only_candidate(tmp_path, monkeypatch):
    config = load_config('config/train/semantic_scale.yaml')
    config['paths']['project_root'] = str(tmp_path)
    output = tmp_path / config['semantic_adaptation']['output_dir']
    gate = output.with_name(output.name + '_smoke')
    gate.mkdir(parents=True)
    (gate / 'pipeline_status.json').write_text(json.dumps({'status': 'completed', 'config': config}))
    control = tmp_path / config['semantic_scale']['control_dir']
    monkeypatch.setattr(runner, 'load_config', lambda path: deepcopy(config))
    monkeypatch.setattr(runner, 'validate_control', lambda cfg: {'directory': str(control), 'passed': True})
    monkeypatch.setattr(runner, 'snapshot_sources', lambda directory: None)
    monkeypatch.setattr(runner, 'verify_run', lambda *a, **kw: {'passed': True})
    commands = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda cmd, **kw: commands.append(cmd))
    runner.run_pipeline('loaded.yaml')
    assert [c[2] for c in commands] == ['train_semantic_consistency.py', 'train_semantic_d5a.py', 'tools/render_backend_ablation.py']
    assert '--smoke' not in commands[0] and '--freeze-semantic-lora' not in commands[0]
    assert 'LoRA control e20=' + str(control / 'deployment') in commands[-1]


def test_prior_is_built_before_crop_and_loss_receives_synchronized_targets(monkeypatch):
    from data import semantic_gray, semantic_hard
    target = torch.full((1, 1, 8, 8), .01)
    target[..., :4] = .99
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[..., :2, :] = False
    counts = dict(accepted_pixels=int(mask.sum()), accepted_ferrite=int((mask & (target > .5)).sum()),
        accepted_pearlite=int((mask & (target < .5)).sum()), valid_pixels=64, known_gt_excluded_pixels=0)
    clean = torch.linspace(0, 1, 32*32).reshape(1, 1, 32, 32).repeat(1, 3, 1, 1)
    raw = dict(weak_image=clean, strong_image=clean.flip(-1), valid_content=torch.ones(1, 1, 32, 32, dtype=torch.bool),
        known_gt_mask=torch.zeros(1, 1, 32, 32, dtype=torch.bool), name=['train_only.png'])
    for key in ('global_draw', 'source_index', 'profile', 'is_spatial', 'endpoint_applied', 'noise_sigma',
                'horizontal_flip', 'vertical_flip', 'rotation_k'):
        raw[key] = torch.tensor([0])
    cfg = dict(semantic_consistency={}, semantic_gray={'comparison_grid': 8}, semantic_hard={'enabled': True},
        semantic_scale={'enabled': True, 'local_probability': 1., 'crop_size': 16, 'alignment': 4})
    calls = []
    def build(image, valid, known, *args, **kwargs):
        assert torch.equal(image, clean)
        calls.append('build_full_prior')
        return target.clone(), mask.clone(), deepcopy(counts)
    def forward(model, restorer, batch, q, accepted, config, seed, replay=None, scale_spec=None):
        assert calls == ['build_full_prior'] and torch.equal(q, target) and torch.equal(accepted, mask)
        assert scale_spec['mode'] == 'local'
        calls.append('restore_select_then_crop')
        return torch.zeros_like(q, requires_grad=True), apply_scale_image(batch['strong_image'], scale_spec), {}, {'selected_index': 0}
    monkeypatch.setattr(semantic_gray, 'build_gray_targets', build)
    monkeypatch.setattr(semantic_hard, 'hard_gray_forward', forward)
    loss, info = gray_unlabelled_loss(None, None, raw, cfg, torch.device('cpu'), 42)
    expected_mask = apply_scale_target(mask, info['scale_view'])
    expected_target = apply_scale_target(target, info['scale_view'])
    assert info['accepted_pixels'] == int(expected_mask.sum())
    assert info['accepted_ferrite'] == int((expected_mask & (expected_target > .5)).sum())
    assert info['scale_parent_target'] == counts
    assert torch.isfinite(loss) and loss.requires_grad
    reference = deepcopy(info)
    reference.pop('scale_parent_target')
    reference.update(counts)
    verify_lora_input(reference, info, unlabeled=True)
    info['scale_parent_target']['accepted_pixels'] += 1
    with pytest.raises(RuntimeError, match='accepted_pixels'):
        verify_lora_input(reference, info, unlabeled=True)
