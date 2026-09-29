# -*- coding: utf-8 -*-
"""正式只训练候选；复用gray4时必须保留输入和冻结回放验证。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import run_semantic_lora as runner
from tools.run_semantic_consistency import COMPLETION_FLAGS, STREAM_FIELDS
from train_semantic_consistency import verify_lora_input
from utils.config import load_config


def test_formal_pipeline_never_retrains_gray4_or_freeze_replay(tmp_path, monkeypatch):
    config = load_config('config/train/semantic_lora.yaml')
    config['paths']['project_root'] = str(tmp_path)
    output = tmp_path / config['semantic_adaptation']['output_dir']
    gate = output.with_name(output.name + '_smoke')
    gate.mkdir(parents=True)
    (gate / 'pipeline_status.json').write_text(json.dumps({'status': 'completed', 'config': config}))
    reference = tmp_path / config['semantic_lora']['reference_dir']
    monkeypatch.setattr(runner, 'load_config', lambda path: deepcopy(config))
    monkeypatch.setattr(runner, 'validate_reference', lambda cfg: {'directory': str(reference), 'passed': True})
    monkeypatch.setattr(runner, 'snapshot_sources', lambda directory: None)
    monkeypatch.setattr(runner, 'verify_run', lambda *args, **kwargs: {'passed': True})
    commands = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    runner.run_pipeline('loaded.yaml')
    assert [command[2] for command in commands] == [
        'train_semantic_consistency.py', 'train_semantic_d5a.py', 'tools/render_backend_ablation.py']
    assert '--freeze-semantic-lora' not in commands[0]
    assert 'Gray4 e20=' + str(reference / 'deployment') in commands[-1]
    status = json.loads((output / 'pipeline_status.json').read_text())
    assert status['completed'] == ['train_candidate', 'infer_candidate', 'render']


def test_unlabeled_replay_rejects_a_different_selected_view():
    fields = ('name', 'global_draw', 'source_index', 'profile', 'is_spatial', 'endpoint_applied',
              'noise_sigma', 'horizontal_flip', 'vertical_flip', 'rotation_k', 'appearance',
              'strong_restoration_seed', 'accepted_pixels', 'accepted_ferrite', 'accepted_pearlite',
              'valid_pixels', 'known_gt_excluded_pixels')
    row = {key: 1 for key in fields}
    row['hard_views'] = {'selected_index': 0}
    verify_lora_input(row, row, unlabeled=True)
    changed = deepcopy(row)
    changed['hard_views']['selected_index'] = 2
    with pytest.raises(RuntimeError, match='selected view'):
        verify_lora_input(row, changed, unlabeled=True)
