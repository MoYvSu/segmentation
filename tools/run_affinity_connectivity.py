# -*- coding: utf-8 -*-
"""先短测，后顺序训练唯一一组control/candidate；完成后各作100图原部署推理。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, project_path

SOURCES = ('train_affinity_connectivity.py', 'tools/run_affinity_connectivity.py',
           'tools/affinity_connectivity_views.py', 'data/affinity_connectivity.py',
           'utils/affinity_connectivity.py', 'config/train/affinity_connectivity.yaml',
           'train_backend_adaptation.py', 'data/backend_adaptation.py',
           'models/backend_adaptation.py', 'models/fused_deployment.py',
           'utils/affinity_deployment.py', 'utils/post_process.py',
           'utils/semantic_challenger.py', 'tools/semantic_crossover.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_pair(root, expected):
    arms = [read(root / arm / 'status.json') for arm in ('control', 'candidate')]
    for value in arms:
        if value['status'] != 'complete' or value['updates'] != expected:
            raise RuntimeError('Incomplete affinity training')
        for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'):
            if value.get(key) is not True:
                raise RuntimeError(f'Training audit failed: {key}')
    for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256', 'data'):
        if arms[0][key] != arms[1][key]:
            raise RuntimeError(f'Unmatched control/candidate dependency: {key}')
    logs = [[json.loads(line) for line in (root / arm / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]
            for arm in ('control', 'candidate')]
    if any(len(rows) != expected for rows in logs):
        raise RuntimeError('Missing paired step receipts')
    for left, right in zip(*logs):
        for key in ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw'):
            if left[key] != right[key]:
                raise RuntimeError(f'Paired training draws differ: {key}')
        if left['auxiliary_weight'] != 0:
            raise RuntimeError('Control has auxiliary gradient')
    if not arms[1]['both_directions_active'] or not any(r['auxiliary_weight'] > 0 for r in logs[1]):
        raise RuntimeError('Candidate must exercise both supervision directions')
    return dict(passed=True, paired_updates=expected, active_edges=arms[1]['active_edges'])


def run(config_path, smoke=False):
    config = load_config(config_path)
    root = Path(project_path(config, config['backend_adaptation']['output_dir'] + ('_smoke' if smoke else '')))
    if not smoke:
        gate_root = Path(str(root) + '_smoke')
        gate = read(gate_root / 'pipeline_status.json')
        # JSON把类别映射的整数键转成字符串；比较同一序列化口径。
        if gate['status'] != 'complete' or gate['config'] != json.loads(json.dumps(config)):
            raise RuntimeError('This exact configuration must pass smoke first')
        for name in SOURCES:
            if name == 'tools/run_affinity_connectivity.py':
                # 调度器本身保留快照供审阅；短测锁定实际训练/部署源码和配置。
                continue
            if (gate_root / 'source' / name).read_bytes() != (ROOT / name).read_bytes():
                raise RuntimeError(f'Source changed since smoke: {name}')
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError('Need at least 3 GiB free for bundles and snapshots')
    root.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        target = root / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    status = dict(status='running', stage='preflight', config=config, smoke=smoke,
                  automatic_extension=False, competition_submission=False, completed=[])
    def save():
        (root / 'pipeline_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    def execute(stage, args):
        status['stage'] = stage
        save()
        with (root / f'{stage}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'train_affinity_connectivity.py'),
                            '--config', str(config_path), *args], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        status['completed'].append(stage)
        save()
    save()
    try:
        for arm in ('control', 'candidate'):
            execute(arm, ['--arm', arm, '--output', str(root / arm)] + (['--smoke'] if smoke else []))
        expected = (config['affinity_connectivity']['smoke_updates'] if smoke else
                    config['backend_adaptation']['epochs'] * 64)
        status['paired_audit'] = verify_pair(root, expected)
        if not smoke:
            for arm in ('control', 'candidate'):
                execute(arm + '_inference', ['--arm', arm, '--infer',
                    '--checkpoint', str(root / arm / 'final.pt'), '--output', str(root / arm / 'deployment')])
        status.update(status='complete', stage='done')
        save()
    except Exception as error:
        status.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train/affinity_connectivity.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, args.smoke)
