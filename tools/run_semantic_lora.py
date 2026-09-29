# -*- coding: utf-8 -*-
"""复用完整gray4，只训练语义专用LoRA候选；短测含6步冻结回放。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.mim_dataset import list_images
from train_semantic_consistency import load_lora_reference, verify_lora_input
from tools.run_semantic_consistency import (COMPLETION_FLAGS, read, rows, same_config,
                                           snapshot_sources as snapshot_common)
from utils.config import load_config, project_path


EXTRA_SOURCES = ('models/fused_deployment.py', 'models/semantic_lora.py', 'models/lora.py',
                 'models/sam2_encoder.py', 'utils/semantic_challenger.py', 'tools/run_semantic_lora.py')


def snapshot_sources(output):
    snapshot_common(output)
    for name in EXTRA_SOURCES:
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


def validate_reference(config):
    status, steps = load_lora_reference(config)
    if status is None or any(status.get(key) is not True for key in COMPLETION_FLAGS):
        raise RuntimeError('Gray4 did not pass its completed training contract')
    directory = Path(project_path(config, config['semantic_lora']['reference_dir']))
    manifest = read(directory / 'deployment/manifest.json')
    restore = config['semantic_adaptation']['restoration']
    if (manifest['checkpoint_sha256'] != config['semantic_lora']['reference_sha256']
            or manifest['epoch'] != config['semantic_adaptation']['epochs']
            or manifest['d5a_sha256'] != restore['checkpoint_sha256']
            or manifest['sampling_mode'] != restore['mode'] or manifest['precision'] != 'FP32'):
        raise RuntimeError('Gray4 deployment checkpoint/restoration differs')
    names = [Path(path).name for path in list_images(project_path(config, config['inference']['test_dir']))]
    previous = [row['image'] for row in manifest['images']]
    if not names or len(set(previous)) != len(previous) or set(names) != set(previous):
        raise RuntimeError('Gray4 full deployment image set differs')
    for name in names:
        for suffix in ('_inst.png', '_class.json', '_class_confidence.json'):
            if not (directory / 'deployment' / (Path(name).stem + suffix)).is_file():
                raise RuntimeError('Gray4 deployment file is missing')
    return {'directory': str(directory), 'checkpoint': status['final_checkpoint'],
            'checkpoint_sha256': manifest['checkpoint_sha256'], 'images': len(names),
            'updates': len(steps), 'passed': True}


def verify_run(reference, run, config, *, smoke=False, frozen=False):
    old, new = read(reference / 'status.json'), read(run / 'status.json')
    count = config['semantic_consistency']['smoke_steps'] if smoke else old['updates']
    if (new['status'] != 'completed' or new['updates'] != count or new['failed_updates']
            or any(new.get(key) is not True for key in COMPLETION_FLAGS)
            or new.get('gray4_input_replay_passed') is not True
            or new['semantic_lora_changed'] == frozen):
        raise RuntimeError('Incomplete independent-LoRA training/freeze contract')
    for key in ('sources', 'data', 'frozen_state_sha256', 'd5a_state_sha256', 'initial_semantic_sha256'):
        if old[key] != new[key]:
            raise RuntimeError(f'Gray4 shared dependency differs: {key}')
    left, right = rows(reference), rows(run)
    if len(right) != count:
        raise RuntimeError('Incomplete independent-LoRA step receipts')
    max_difference = 0.
    for x, y in zip(left, right):
        verify_lora_input(x, y)
        verify_lora_input(x['unlabeled'], y['unlabeled'], unlabeled=True)
        if frozen:
            for key in ('semantic_loss', 'unlabeled_loss', 'unlabeled_weight', 'grad_norm'):
                difference = abs(x[key] - y[key])
                max_difference = max(max_difference, difference)
                if difference != 0:
                    raise RuntimeError(f'Frozen private LoRA does not exactly replay gray4: {key}, {difference}')
    if not frozen and not any(row['semantic_lora_supervised_grad_norm'] > 0 for row in right):
        raise RuntimeError('No measured semantic-LoRA gradients')
    return {'passed': True, 'updates': count, 'reused_gray4': True,
            'same_gt_and_selected_unlabeled_views': True, 'frozen_replay': frozen,
            'frozen_replay_max_difference': max_difference if frozen else None,
            'semantic_lora_max_delta': new['semantic_lora_max_delta'], 'official_score': None}


def run_pipeline(config_path, smoke=False):
    config = load_config(config_path)
    cfg = config['semantic_adaptation']
    reference = validate_reference(config)
    reference_dir = Path(reference['directory'])
    output = Path(project_path(config, cfg['output_dir'] + ('_smoke' if smoke else '')))
    if not smoke:
        gate_dir = Path(project_path(config, cfg['output_dir'] + '_smoke'))
        gate = read(gate_dir / 'pipeline_status.json')
        if gate['status'] != 'completed' or not same_config(gate['config'], config):
            raise RuntimeError('Run the exact configuration short test before formal training')
        for source in (gate_dir / 'source').rglob('*.py'):
            if source.read_bytes() != (ROOT / source.relative_to(gate_dir / 'source')).read_bytes():
                raise RuntimeError(f'Source changed after the accepted smoke: {source.name}')
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output)
    snapshot = output / 'config.yaml'
    snapshot.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    status = {'status': 'running', 'stage': 'preflight', 'smoke': smoke, 'config': config,
              'completed': [], 'reused_gray4': reference, 'competition_submission': False,
              'automatic_extension': False}

    def write():
        (output / 'pipeline_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')

    def execute(name, command):
        status['stage'] = name
        write()
        with (output / f'{name}.log').open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status['completed'].append(name)
        write()

    try:
        if smoke:
            execute('frozen_replay', ['train_semantic_consistency.py', '--config', str(snapshot),
                '--arm', 'prior', '--output-dir', str(output / 'frozen'), '--smoke', '--freeze-semantic-lora'])
            status['frozen_replay'] = verify_run(reference_dir, output / 'frozen', config, smoke=True, frozen=True)
        flags = ['--smoke'] if smoke else []
        execute('train_candidate', ['train_semantic_consistency.py', '--config', str(snapshot),
            '--arm', 'prior', '--output-dir', str(output / 'candidate'), *flags])
        status['pair_check'] = verify_run(reference_dir, output / 'candidate', config, smoke=smoke)
        epoch = 1 if smoke else cfg['epochs']
        execute('infer_candidate', ['train_semantic_d5a.py', '--mode', 'infer', '--arm', 'simple',
            '--config', str(snapshot), '--output-dir', str(output / 'candidate'),
            '--prediction-dir', str(output / 'candidate/deployment'),
            '--checkpoint', str(output / 'candidate' / f'epoch_{epoch:03d}.pt'), *flags])
        execute('render', ['tools/render_backend_ablation.py', '--image-dir', project_path(config, config['inference']['test_dir']),
            '--prediction', 'Original simple=' + project_path(config, config['semantic_consistency']['original_prediction_dir']),
            '--prediction', 'Gray4 e20=' + str(reference_dir / 'deployment'),
            '--prediction', ('Private LoRA smoke e1=' if smoke else 'Private LoRA e20=') + str(output / 'candidate/deployment'),
            '--images', *cfg['monitor']['images'], '--output-dir', str(output / 'comparison')])
        status.update(status='completed', stage='completed')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_lora.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_pipeline(args.config, args.smoke)


if __name__ == '__main__':
    main()
