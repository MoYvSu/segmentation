# -*- coding: utf-8 -*-
"""短跑不能隐式占用全量推理预算。"""
from pathlib import Path

from tools.run_affinity_gt_source_queue import build_stages


def test_short_run_stops_after_paired_training():
    stages = build_stages(Path('config.yml'), Path('private'))
    assert [name for name, _ in stages] == ['train_manual', 'train_teacher']
    command = stages[1][1]
    assert command[command.index('--paired-steps')+1] == str(Path('private/manual/steps.jsonl'))


def test_full_evaluation_requires_explicit_selection():
    stages = build_stages(Path('config.yml'), Path('private'), evaluate=True)
    assert [name for name, _ in stages] == [
        'train_manual', 'train_teacher', 'evaluate_manual', 'evaluate_teacher']
