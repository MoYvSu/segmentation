# -*- coding: utf-8 -*-
"""定时关机专用恢复：保护原始prefix、来源反传系数和20轮学习率日程。"""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from tools import resume_affinity_source as resume
from train_affinity_sweep import capture_rng_state
from utils.config import load_config


def config():
    return load_config('config/train/affinity_source_p025.yaml')


def test_prefix_retains_original_bytes_and_discards_unsaved_updates(tmp_path):
    original = b'{"a": 1}\r\n{"a":2.000}\n{"unsaved":3}\n'
    path = tmp_path / 'steps.jsonl'
    path.write_bytes(original)
    assert resume.retained_bytes(path, 2) == (b'{"a": 1}\r\n{"a":2.000}\n', 1)
    with pytest.raises(RuntimeError, match='Incomplete'):
        resume.retained_bytes(path, 4)
    path.write_bytes(b'{"unfinished":1}')
    with pytest.raises(RuntimeError, match='Incomplete'):
        resume.retained_bytes(path, 1)


@pytest.mark.parametrize('pseudo,coefficient', [(False, 1.2), (True, .3)])
def test_resume_loss_preserves_original_scalar_hook_order_and_sidecar(monkeypatch, tmp_path, pseudo, coefficient):
    cfg = config()
    logits = torch.tensor([.1, -.2, .5], requires_grad=True)
    monkeypatch.setattr(resume.native, 'compute_affinity_loss',
        lambda value, batch, loss_cfg, **kwargs: (value.square().mean(), {'diagnostic': 1}))
    batch = {'name': ['source_1']}
    loss, details = resume.weighted_loss(logits, batch, {}, pseudo=pseudo, config=cfg,
        output=tmp_path, update=1025, epoch=17, step=1)
    assert loss.item() == pytest.approx(logits.detach().square().mean().item())
    assert details == {'diagnostic': 1}
    (loss * (.25 if pseudo else 1)).backward()
    assert torch.allclose(logits.grad, 2 * logits.detach() / 3 * coefficient)
    receipt = json.loads((tmp_path / resume.SIDECAR).read_text())
    assert receipt['call'] == 2049 + int(pseudo)
    assert receipt['update'] == 1025 and receipt['epoch'] == 17 and receipt['step'] == 1
    assert receipt['source_kind'] == ('sam2' if pseudo else 'manual')
    assert receipt['source_coefficient'] == coefficient
    assert receipt['weighted_logits_grad_norm'] == pytest.approx(coefficient * receipt['raw_logits_grad_norm'])


def test_restore_rng_restores_all_cpu_streams(monkeypatch):
    random.seed(14)
    np.random.seed(14)
    torch.manual_seed(14)
    saved = capture_rng_state()
    saved['torch_cuda'] = []
    expected = (random.random(), np.random.random(), torch.rand(4))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda state: None)
    resume.restore_rng(saved)
    actual = (random.random(), np.random.random(), torch.rand(4))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    saved['torch_cuda'] = [torch.ones(4, dtype=torch.uint8)]
    with pytest.raises(RuntimeError, match='device count'):
        resume.restore_rng(saved)


def make_head():
    cfg = config()
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter], lr=cfg['backend_adaptation']['learning_rates']['affinity'],
        weight_decay=cfg['backend_adaptation']['weight_decay'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda epoch: resume.expected_lr(cfg, epoch) / cfg['backend_adaptation']['learning_rates']['affinity'])
    for epoch in range(16):
        for _ in range(64):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()
        scheduler.step()
    return cfg, dict(format='affinity_native_head_v1', epoch=16, size=1024, config=cfg,
        geometry_state_dict={'weight': parameter.detach()}, optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), **resume.checkpoint_extras(cfg, 16))


def test_registered_head_and_original_twenty_epoch_cosine_are_required():
    cfg, head = make_head()
    resume.check_head(cfg, head)
    assert head['optimizer']['param_groups'][0]['lr'] == pytest.approx(3.7188470506254746e-06, rel=1e-12)
    for key, value in [('epoch', 15), ('size', 512), ('format', 'unknown')]:
        changed = deepcopy(head)
        changed[key] = value
        with pytest.raises(RuntimeError):
            resume.check_head(cfg, changed)
    changed = deepcopy(head)
    changed['scheduler']['last_epoch'] = 0
    with pytest.raises(RuntimeError, match='scheduler'):
        resume.check_head(cfg, changed)
    changed = deepcopy(head)
    changed['optimizer']['state'][0]['step'] = torch.tensor(1025.)
    with pytest.raises(RuntimeError, match='AdamW'):
        resume.check_head(cfg, changed)
    changed = deepcopy(head)
    changed['geometry_state_dict']['weight'].fill_(float('nan'))
    with pytest.raises(RuntimeError, match='Nonfinite'):
        resume.check_head(cfg, changed)


def test_original_partial_step_replay_matches_draw_and_finite_fp32(monkeypatch):
    row = dict(epoch=17, step=1, updates=1025, seed=17, manual=['m'], pseudo=['s'],
        manual_draw={'source': 1}, pseudo_draw={'source': 2}, manual_crop=[[1]], pseudo_crop=[[2]],
        crop_attempts=[[1], [1]], training_view='whole', manual_bce=.2, pseudo_bce=.1,
        grad_norm=.3, supervision=[{'positive_edges': 2, 'negative_edges': 3}] * 2)
    monkeypatch.setattr(resume, '_rows', lambda path: [None] * 1024 + [row])
    assert resume.compare_interrupted_step(deepcopy(row))['passed']
    changed = deepcopy(row)
    changed['manual_bce'] = .4
    with pytest.raises(RuntimeError, match='numerical replay'):
        resume.compare_interrupted_step(changed)
    changed = deepcopy(row)
    changed['manual_crop'] = [[9]]
    with pytest.raises(RuntimeError, match='draw'):
        resume.compare_interrupted_step(changed)


def test_smoke_gate_rejects_source_or_identity_change(monkeypatch, tmp_path):
    monkeypatch.setattr(resume, 'ROOT', tmp_path)
    folder = tmp_path / 'outputs/affinity_source_resume_smoke'
    (folder / 'p025').mkdir(parents=True)
    identity = dict(original_source_sha256={'old.py': 'old-sha'}, resume_source_sha256={'new.py': 'new-sha'})
    (folder / 'pipeline_status.json').write_text(json.dumps(dict(status='complete', smoke=True,
        sources=identity['original_source_sha256'], resume_sources=identity['resume_source_sha256'])))
    (folder / 'resume_receipt.json').write_text(json.dumps({'identity': {'changed': True}}))
    (folder / 'p025/smoke_receipt.json').write_text(json.dumps({'passed': True}))
    with pytest.raises(RuntimeError, match='restoration smoke'):
        resume.smoke_gate(config(), identity)
