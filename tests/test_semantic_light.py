# -*- coding: utf-8 -*-
"""光照实验只改变输入光照，不能悄悄改变初始化、GT、损失或预算。"""
from copy import deepcopy
import json

import pytest

from tools.run_semantic_light import comparison_config, verify_control
from utils.config import load_config


def test_light_config_retains_the_complete_simple_recipe():
    baseline = load_config('config/train/semantic_d5a.yaml')
    light = load_config('config/train/semantic_light.yaml')
    assert comparison_config(light) == comparison_config(baseline)
    assert light['semantic_adaptation']['seed'] == 20260925
    assert light['semantic_adaptation']['illumination']['probability'] == .5
    assert baseline['semantic_adaptation'].get('illumination') is None


@pytest.mark.parametrize('changed', ['seed', 'epochs', 'manual_repeats', 'completed_gt_dir'])
def test_comparison_keeps_nonlight_training_factors(changed):
    baseline = load_config('config/train/semantic_d5a.yaml')
    altered = deepcopy(baseline)
    altered['semantic_adaptation'][changed] = 'unexpected_change'
    assert comparison_config(altered) != comparison_config(baseline)


def test_control_rejects_changed_loss_and_incomplete_training(tmp_path):
    config = load_config('config/train/semantic_light.yaml')
    config['semantic_light']['control_dir'] = str(tmp_path)
    status = {'status': 'completed', 'arm': 'simple', 'updates': 3840, 'failed_updates': 0,
              'config': load_config('config/train/semantic_d5a.yaml')}
    for key in ('frozen_weights_unchanged', 'd5a_unchanged', 'frozen_affinity_output_equal',
                'strict_reload_passed', 'semantic_weights_changed'):
        status[key] = True
    path = tmp_path / 'status.json'
    path.write_text(json.dumps(status), encoding='utf-8')
    assert verify_control(config) == tmp_path
    status['config']['direct_semantic_affinity']['semantic_loss']['dice_weight'] = .5
    path.write_text(json.dumps(status), encoding='utf-8')
    with pytest.raises(RuntimeError, match='beyond illumination'):
        verify_control(config)
    status['updates'] = 64
    path.write_text(json.dumps(status), encoding='utf-8')
    with pytest.raises(RuntimeError, match='incomplete'):
        verify_control(config)
