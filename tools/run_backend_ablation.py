# -*- coding: utf-8 -*-
"""串行微调两组后端，再自动推理、打包和渲染；无需持续人工盯训。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, project_path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_pair(root, smoke, epochs):
    a, b = [read(root / arm / 'status.json') for arm in ('d4', 'raw')]
    expected = 2 if smoke else epochs * 64
    for status in (a, b):
        if status['status'] != 'completed' or status['updates'] != expected or status['failed_updates']:
            raise RuntimeError('Incomplete paired training; do not produce a comparison')
        if not status['initial_final_output_equal'] or not status['frozen_encoder_unchanged'] or not status['d4_unchanged']:
            raise RuntimeError('Initial/frozen model contract failed')
    for field in ('initial_state_sha256', 'frozen_encoder_sha256', 'data'):
        if a['sources'][field] != b['sources'][field]:
            raise RuntimeError(f'Unmatched paired setup: {field}')
    if a['config'] != b['config']:
        raise RuntimeError('Paired training configurations differ')
    logs = [[json.loads(line) for line in (root / arm / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]
            for arm in ('d4', 'raw')]
    if len(logs[0]) != expected or len(logs[1]) != expected:
        raise RuntimeError('Incomplete step logs')
    for left, right in zip(*logs):
        for key in ('epoch', 'step', 'updates', 'manual', 'pseudo', 'seed'):
            if left[key] != right[key]:
                raise RuntimeError(f'Unpaired sample/random stream: {key}')
    report = {'same_initial_weights': True, 'same_data_recipe': True, 'same_sample_sequence': True,
              'same_step_seeds': True, 'updates_per_arm': expected,
              'comparison': 'frozen_D4_first_x0 versus no_D4; both jointly finetuned LoRA and task heads',
              'official_scores': None, 'smoke': smoke}
    (root / 'pair_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/backend_d4.yaml')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--postprocess-only', action='store_true', help='训练已有完整产物时，只复核并生成推理与图')
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config['backend_adaptation']
    suffix = '_smoke' if args.smoke else ''
    output = Path(project_path(config, cfg['output_dir'] + suffix))
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'pipeline_status.json'
    if status_path.exists() and not args.postprocess_only:
        raise FileExistsError('Existing pipeline; use a distinct output directory')
    snapshot = output / 'config.yaml'
    if snapshot.exists():
        config = load_config(str(snapshot))
        cfg = config['backend_adaptation']
    else:
        snapshot.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    run_config = str(snapshot)
    status = {'status': 'running', 'completed': [], 'smoke': args.smoke,
              'config': args.config, 'automatic_final_inference_and_render': True}

    def write():
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')

    def execute(name, command):
        status['stage'] = name
        write()
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['completed'].append(name)
        write()

    try:
        if not args.postprocess_only:
            for arm in ('d4', 'raw'):
                execute('train_' + arm, ['train_backend_adaptation.py', '--config', run_config,
                    '--arm', arm, '--output-dir', str(output / arm), *(['--smoke'] if args.smoke else [])])
        status['pair_check'] = verify_pair(output, args.smoke, cfg['epochs'])
        epoch = 1 if args.smoke else cfg['epochs']
        for arm in ('d4', 'raw'):
            prediction_dir = output / arm / 'deployment'
            execute('infer_' + arm, ['train_backend_adaptation.py', '--config', run_config,
                '--arm', arm, '--checkpoint', str(output / arm / f'epoch_{epoch:03d}.pt'),
                '--output-dir', str(prediction_dir), *(['--smoke'] if args.smoke else [])])
            if not args.smoke:
                execute('package_' + arm, ['tools/package_submission.py', '--prediction-dir', str(prediction_dir),
                    '--test-dir', project_path(config, config['inference']['test_dir']),
                    '--output', str(output / f'submission_{arm}.zip')])
        render = ['tools/render_backend_ablation.py', '--image-dir', project_path(config, config['inference']['test_dir']),
            '--prediction', 'Baseline=' + project_path(config, cfg['baseline_prediction_dir']),
            '--prediction', 'D4 original=' + project_path(config, cfg['d4_prediction_dir']),
            '--prediction', 'D4 finetuned=' + str(output / 'd4' / 'deployment'),
            '--prediction', 'Raw finetuned=' + str(output / 'raw' / 'deployment'),
            '--images', *cfg['monitor']['images'], '--output-dir', str(output / 'comparison')]
        execute('render', render)
        status.update(status='completed', stage='completed', comparison_dir=str(output / 'comparison'),
                      competition_submission=False)
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
