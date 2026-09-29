# -*- coding: utf-8 -*-
"""复用已完成私有LoRA控制，仅训练20轮尺度候选并沿原全图部署推理。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from train_backend_adaptation import sha
from train_semantic_consistency import verify_lora_input
from tools.run_semantic_consistency import COMPLETION_FLAGS, read, rows, same_config
from tools.run_semantic_lora import snapshot_sources as snapshot_lora
from utils.config import load_config, project_path


def base_config(config):
    value = deepcopy(config)
    value.pop('semantic_scale', None)
    value['semantic_adaptation'].pop('output_dir', None)
    return value


def snapshot_sources(output):
    snapshot_lora(output)
    for name in ('data/semantic_scale.py', 'tools/semantic_scale_monitor.py', 'tools/run_semantic_scale.py'):
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


def validate_control(config):
    cfg, scfg = config['semantic_adaptation'], config['semantic_scale']
    if not scfg['enabled'] or cfg['epochs'] != 20:
        raise ValueError('This experiment requires enabled scale views and the matched 20-epoch budget')
    control = Path(project_path(config, scfg['control_dir']))
    old = read(control / 'status.json')
    count = cfg['epochs'] * cfg['expected_manual_sources'] * cfg['manual_repeats']
    if (old['status'] != 'completed' or old['smoke'] or old['arm'] != 'prior'
            or old['updates'] != count or old['failed_updates']
            or any(old.get(key) is not True for key in COMPLETION_FLAGS)
            or old.get('semantic_lora_changed') is not True):
        raise RuntimeError('Private-LoRA control is incomplete')
    if not same_config(base_config(old['config']), base_config(config)):
        raise RuntimeError('Scale experiment changed a variable beyond views/output')
    manifest = read(control / 'deployment/manifest.json')
    restoration = cfg['restoration']
    if (manifest['checkpoint_sha256'] != scfg['control_sha256']
            or sha(Path(old['final_checkpoint'])) != scfg['control_sha256']
            or manifest['epoch'] != cfg['epochs']
            or manifest['d5a_sha256'] != restoration['checkpoint_sha256']
            or manifest['sampling_mode'] != restoration['mode'] or manifest['precision'] != 'FP32'):
        raise RuntimeError('Control checkpoint or restoration identity changed')
    names = [Path(p).name for p in list_images(project_path(config, config['inference']['test_dir']))]
    if len(names) != 100 or sorted(r['image'] for r in manifest['images']) != sorted(names):
        raise RuntimeError('Control deployment does not contain the full test cohort')
    for name in names:
        for suffix in ('_inst.png', '_class.json', '_class_confidence.json'):
            if not (control / 'deployment' / (Path(name).stem + suffix)).is_file():
                raise RuntimeError('Missing control deployment pair')
    if len(rows(control)) != count:
        raise RuntimeError('Control step receipts are incomplete')
    return dict(directory=str(control), checkpoint=old['final_checkpoint'],
        checkpoint_sha256=scfg['control_sha256'], images=len(names), updates=count, passed=True)


def verify_run(control, run, config, *, smoke=False, replay=False):
    old, new = read(control / 'status.json'), read(run / 'status.json')
    count = config['semantic_consistency']['smoke_steps'] if smoke else old['updates']
    if (new['status'] != 'completed' or new['updates'] != count or new['failed_updates']
            or any(new.get(key) is not True for key in COMPLETION_FLAGS)
            or new.get('semantic_lora_changed') is not True):
        raise RuntimeError('Scale candidate did not complete its training/freeze checks')
    for key in ('sources', 'data', 'frozen_state_sha256', 'd5a_state_sha256', 'initial_semantic_sha256'):
        if old[key] != new[key]:
            raise RuntimeError(f'Scale/control shared dependency differs: {key}')
    left, right = rows(control), rows(run)
    if len(right) != count:
        raise RuntimeError('Incomplete scale training receipts')
    seen = {'supervised': set(), 'unlabeled': set()}
    for x, y in zip(left, right):
        verify_lora_input(x, y)
        verify_lora_input(x['unlabeled'], y['unlabeled'], unlabeled=True)
        if replay:
            for key in ('semantic_loss', 'unlabeled_loss', 'unlabeled_weight', 'grad_norm'):
                if x[key] != y[key]:
                    raise RuntimeError(f'Disabled scale does not replay existing LoRA: {key}')
        else:
            for stream, receipt in [('supervised', y), ('unlabeled', y['unlabeled'])]:
                seen[stream].add(receipt['scale_view']['mode'])
    if not replay and any(value != {'full', 'local'} for value in seen.values()):
        raise RuntimeError('Short/formal scale run must exercise both full and local views in both streams')
    return dict(passed=True, updates=count, original_input_and_prior_replayed=True,
        disabled_replay_exact=replay, view_modes={k: sorted(v) for k, v in seen.items()},
        same_pool_loss=True, official_score=None)


def run_pipeline(config_path, smoke=False):
    config = load_config(config_path)
    cfg = config['semantic_adaptation']
    reference = validate_control(config)
    control = Path(reference['directory'])
    output = Path(project_path(config, cfg['output_dir'] + ('_smoke' if smoke else '')))
    if not smoke:
        gate_dir = Path(project_path(config, cfg['output_dir'] + '_smoke'))
        gate = read(gate_dir / 'pipeline_status.json')
        if gate['status'] != 'completed' or not same_config(gate['config'], config):
            raise RuntimeError('Run this exact scale configuration smoke first')
        for source in (gate_dir / 'source').rglob('*.py'):
            if source.read_bytes() != (ROOT / source.relative_to(gate_dir / 'source')).read_bytes():
                raise RuntimeError(f'Source changed after accepted smoke: {source.name}')
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output)
    snapshot = output / 'config.yaml'
    snapshot.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    status = dict(status='running', stage='preflight', smoke=smoke, config=config, completed=[],
        reused_control=reference, competition_submission=False, automatic_extension=False)

    def write():
        (output / 'pipeline_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')

    def execute(name, args):
        status['stage'] = name
        write()
        with (output / f'{name}.log').open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-u', *args], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        status['completed'].append(name)
        write()

    try:
        if smoke:
            disabled = deepcopy(config)
            disabled['semantic_scale']['enabled'] = False
            disabled_path = output / 'disabled.yaml'
            disabled_path.write_text(yaml.safe_dump(disabled, allow_unicode=True), encoding='utf-8')
            execute('disabled_replay', ['train_semantic_consistency.py', '--config', str(disabled_path),
                '--arm', 'prior', '--output-dir', str(output / 'disabled'), '--smoke'])
            status['disabled_replay'] = verify_run(control, output / 'disabled', config, smoke=True, replay=True)
        flags = ['--smoke'] if smoke else []
        execute('train_candidate', ['train_semantic_consistency.py', '--config', str(snapshot),
            '--arm', 'prior', '--output-dir', str(output / 'candidate'), *flags])
        status['paired_inputs'] = verify_run(control, output / 'candidate', config, smoke=smoke)
        epoch = 1 if smoke else cfg['epochs']
        execute('infer_candidate', ['train_semantic_d5a.py', '--mode', 'infer', '--arm', 'simple',
            '--config', str(snapshot), '--output-dir', str(output / 'candidate'),
            '--prediction-dir', str(output / 'candidate/deployment'),
            '--checkpoint', str(output / 'candidate' / f'epoch_{epoch:03d}.pt'), *flags])
        execute('render', ['tools/render_backend_ablation.py', '--image-dir', project_path(config, config['inference']['test_dir']),
            '--prediction', 'LoRA control e20=' + str(control / 'deployment'),
            '--prediction', ('Scale smoke e1=' if smoke else 'Scale e20=') + str(output / 'candidate/deployment'),
            '--images', *cfg['monitor']['images'], '--output-dir', str(output / 'comparison')])
        status.update(status='completed', stage='completed')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_scale.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_pipeline(args.config, args.smoke)
