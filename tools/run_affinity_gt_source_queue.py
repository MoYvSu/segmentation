# -*- coding: utf-8 -*-
"""同源GT短跑默认仅训练；显式--evaluate才接全量评估，失败即停止。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_affinity_gt_source import load_targets, validate_config
from utils.config import load_config, project_path


def build_stages(config_path, base, *, evaluate=False):
    stages = []
    for arm in ('manual', 'teacher'):
        command = [sys.executable, '-u', str(ROOT/'train_affinity_gt_source.py'),
                   '--config', str(config_path), '--arm', arm, '--output', str(base/arm)]
        if arm == 'teacher':
            command += ['--paired-steps', str(base/'manual/steps.jsonl')]
        stages.append(('train_'+arm, command))
    if evaluate:
        for arm in ('manual', 'teacher'):
            stages.append(('evaluate_'+arm, [sys.executable, '-u',
                str(ROOT/'tools/evaluate_affinity_gt_source.py'), '--config', str(config_path),
                '--arm', arm, '--checkpoint', str(base/arm/'final.pt')]))
    return stages


def run(config_path, *, evaluate=False):
    config_path = Path(config_path).resolve()
    config = load_config(str(config_path))
    validate_config(config)
    # 在A开始前核对真实B确认，避免控制组完成后才发现候选不合格。
    load_targets(config, 'teacher')
    base = Path(project_path(config, config['gt_source']['output_dir'])).resolve()
    folder = base/'queue'
    if folder.exists() or any((base/arm).exists() for arm in ('manual', 'teacher')):
        raise FileExistsError('Queue and both arms require fresh output directories')
    folder.mkdir(parents=True)
    stages = build_stages(config_path, base, evaluate=evaluate)
    receipt = dict(status='running', config=str(config_path), started=time.time(),
                   selection='fixed_e5', training_updates_per_arm=320,
                   test_images_per_arm=100 if evaluate else 0,
                   full_evaluation_requested=evaluate, stages=[], official_submission=False)

    def save():
        pending = folder/'status.tmp'
        pending.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        pending.replace(folder/'status.json')

    save()
    for name, command in stages:
        stage = dict(name=name, command=command, status='running', started=time.time())
        receipt['stages'].append(stage)
        save()
        try:
            with (folder/(name+'.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            stage.update(returncode=result.returncode, completed=time.time(),
                         status='complete' if result.returncode == 0 else 'failed')
            if result.returncode:
                receipt.update(status='failed', failed_stage=name)
                save()
                raise RuntimeError('Queue stopped: '+name)
            save()
        except Exception as error:
            receipt.update(status='failed', failed_stage=name, error=repr(error))
            stage.update(status='failed')
            save()
            raise
    receipt.update(status='complete', completed=time.time())
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_gt_source.yaml')
    parser.add_argument('--evaluate', action='store_true', help='显式选择训练后全100图推理；短跑默认跳过')
    args = parser.parse_args()
    run(args.config, evaluate=args.evaluate)
