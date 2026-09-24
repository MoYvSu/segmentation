# -*- coding: utf-8 -*-
"""D5b从零短链训练；正式参照已完成D5a，工程短测独立运行开/关两组。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from models.rgb_restoration import build_rgb_restorer
from train_rgb_restoration import build_datasets, write_json
from utils.config import load_config, project_path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def equal_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_tree(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def check_recipes(candidate, control):
    left, right = copy.deepcopy(candidate), copy.deepcopy(control)
    left.pop('output_dir', None)
    right.pop('output_dir', None)
    chain = left['train'].pop('short_chain', None)
    display = left['monitor'].pop('trajectory_steps', None)
    if (chain != dict(enabled=True, probability=.25, extra_steps=2, loss_weight=.25)
            or display != [1, 2, 3, 4, 8, 16] or left != right):
        raise ValueError('D5b differs from D5a outside short_chain/output_dir/monitor display steps')
    if left['split_policy'] != 'all_train' or left.get('holdout_manifest') is not None:
        raise ValueError('D5b requires all sources without holdout')
    if left['train']['epochs'] != 60:
        raise ValueError('D5b retains the 60-epoch learning-rate schedule')
    return chain


def audit_inputs(candidate, control):
    sets = [build_datasets(config) for config in (candidate, control)]
    if any(heldout is not None for _, heldout in sets):
        raise ValueError('unexpected holdout')
    a, b = [item[0] for item in sets]
    if a.samples != b.samples or len(a) != candidate['rgb_restoration']['expected_train_sources']:
        raise ValueError('unpaired source pool')
    indices = np.linspace(0, len(a)-1, min(64, len(a)), dtype=int)
    for index in indices:
        left, right = a[int(index)], b[int(index)]
        if not equal_tree(left, right):
            raise AssertionError(f'D5b altered a training input/target/metadata at source index {index}')
    return dict(training_sources=len(a), audited_sources=len(indices), inputs_targets_metadata_equal=True,
                scope='training_input_engineering_audit_not_validation')


def check_reference(reference, candidate_config):
    summary, run = read(reference/'summary.json'), read(reference/'run_config.json')
    cfg = candidate_config['rgb_restoration']
    check_recipes(cfg, run['experiment_config'])
    expected = cfg['train']['epochs'] * ((cfg['expected_train_sources'] + cfg['train']['batch_size'] - 1)
                                      // cfg['train']['batch_size'])
    if (summary['status'] != 'completed' or summary['updates'] != expected
            or summary['epoch'] != 60 or summary['failed_updates'] or run.get('continuation')):
        raise ValueError('D5a reference must be a complete zero-failure scratch run')
    sources = list_images(project_path(candidate_config, cfg['data_dir']))
    if [Path(path).name for path in sources] != run['train_sources']:
        raise ValueError('current training pool differs from completed D5a')
    versions = read(reference.parent/'source_versions.json')
    # 训练循环和显示步数是本轮有意改动；数据与网络实现必须与历史D5a一致。
    fixed_code = ['train_rgb_restoration.py', 'data/rgb_restoration_dataset.py',
                  'data/rgb_spatial_blur.py', 'models/rgb_diffusion_restoration.py',
                  'models/rgb_restoration.py', 'tools/rgb_spatial_monitor.py']
    for name in fixed_code:
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != versions[name]:
            raise ValueError(f'reference implementation changed: {name}')
    start = torch.load(reference/'epoch_000.pt', map_location='cpu', weights_only=True)
    if start['epoch'] or start['updates'] or start.get('continuation'):
        raise ValueError('D5a reference did not start from scratch')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg['seed']))
        initial_model = build_rgb_restorer(cfg['model'])
    if not equal_tree(initial_model.state_dict(), start['model']):
        raise ValueError('fresh D5b initialization differs from D5a reference')
    return dict(reference_dir=str(reference), reference_updates=expected, seed=cfg['seed'],
                same_source_pool=True, same_initial_model=True, unchanged_data_and_model_code=fixed_code,
                reference_is_loaded_for_comparison_only=True, checkpoint_initialization=None)


def verify_pair(candidate, reference, *, smoke=False):
    summaries = [read(path/'summary.json') for path in (candidate, reference)]
    expected = 16 if smoke else 15000
    expected_epoch, status = (1, 'segment_completed') if smoke else (60, 'completed')
    for row in summaries:
        if (row['updates'] != expected or row['epoch'] != expected_epoch
                or row['failed_updates'] or row['status'] != status):
            raise RuntimeError('incomplete or failed D5b/reference run')
    starts = [torch.load(path/'epoch_000.pt', map_location='cpu', weights_only=True)
              for path in (candidate, reference)]
    for start in starts:
        if start['epoch'] or start['updates'] or start.get('continuation'):
            raise RuntimeError('D5b and reference must independently start from scratch')
    for key in ('model', 'optimizer', 'scheduler', 'scaler'):
        if not equal_tree(starts[0][key], starts[1][key]):
            raise RuntimeError(f'initial states differ: {key}')
    for key in ('noise_generator', 'loader_generator', 'torch_cpu', 'torch_cuda'):
        if not equal_tree(starts[0]['rng'][key], starts[1]['rng'][key]):
            raise RuntimeError(f'initial main RNG differs: {key}')
    fixed_files = ['fixed_samples.pt', 'spatial_fixed_samples.pt',
                   'endpoint_fixed_samples.pt', 'real_fixed_samples.pt']
    for filename in fixed_files:
        pair = [torch.load(path/filename, map_location='cpu', weights_only=True)
                for path in (candidate, reference)]
        if not equal_tree(*pair):
            raise RuntimeError(f'fixed observation inputs differ: {filename}')
    logs = [[json.loads(line) for line in (path/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            for path in (candidate, reference)]
    if any(len(rows) != expected for rows in logs):
        raise RuntimeError('update log length differs')
    keys = ('sources', 'profiles', 't_histogram', 'learning_rate', 'epoch', 'update', 'crop_blur')
    for left, right in zip(*logs):
        for key in keys:
            if left.get(key) != right.get(key):
                raise RuntimeError(f'paired main input/step sequence differs: {key}')
    last = [torch.load(path/'last.pt', map_location='cpu', weights_only=True)
            for path in (candidate, reference)]
    for key in ('noise_generator', 'loader_generator'):
        if not equal_tree(last[0]['rng'][key], last[1]['rng'][key]):
            raise RuntimeError(f'final main RNG differs: {key}')
    chain_rows = [row['short_chain'] for row in logs[0]]
    extra_calls = sum(row['network_calls'] for row in chain_rows)
    if extra_calls <= 0:
        raise RuntimeError('short-chain branch was never exercised')
    report = dict(from_scratch=True, same_initial_weights_optimizer_scheduler_scaler=True,
                  same_main_rng=True, paired_sources_profiles_timesteps_lr_and_blur=True,
                  fixed_inputs_equal=fixed_files, updates=expected, reference=str(reference),
                  extra_network_calls=extra_calls, equal_compute=False,
                  official_scores=None, comparison='D5a recipe versus D5a plus short-chain supervision')
    write_json(candidate.parent/'pair_check.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='64 sources, first epoch, enabled/disabled pair')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--output-dir', help='fresh private run directory')
    parser.add_argument('--reference-dir', default=str(ROOT/'outputs/d5a/candidate'))
    args = parser.parse_args()
    configurations = {arm: load_config(str(ROOT/'config/train'/filename)) for arm, filename in (
        ('candidate', 'rgb_diffusion_d5b.yaml'), ('control', 'rgb_diffusion_d5a.yaml'))}
    check_recipes(*[configurations[arm]['rgb_restoration'] for arm in ('candidate', 'control')])
    output = Path(args.output_dir or ROOT/'outputs'/('d5b_smoke' if args.smoke else
                  'd5b_preflight' if args.preflight_only else 'd5b')).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output must be new: {output}')
    reference = Path(args.reference_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    status = dict(status='running', stage='preflight', completed=[], smoke=args.smoke,
                  from_scratch=True, competition_submission=False)
    def save_status():
        write_json(output/'pipeline_status.json', status)
    save_status()
    try:
        if args.smoke:
            cfg = configurations['candidate']['rgb_restoration']
            sources = list_images(project_path(configurations['candidate'], cfg['data_dir']))
            if len(sources) != cfg['expected_train_sources']:
                raise ValueError('unexpected full training pool before smoke')
            destination = output/'sources'
            destination.mkdir()
            for index in np.linspace(0, len(sources)-1, 64, dtype=int):
                source = Path(sources[int(index)]).resolve()
                (destination/source.name).symlink_to(source)
            for config in configurations.values():
                cfg = config['rgb_restoration']
                cfg.update(data_dir=str(destination), expected_train_sources=64)
                cfg['monitor']['samples'] = 1
                for name in ('spatial_blur', 'endpoint_blur'):
                    cfg['monitor'][name]['samples'] = 1
                cfg['monitor']['real']['images'] = ['test_101']
            reference = output/'control'
        else:
            report = check_reference(reference, configurations['candidate'])
            write_json(output/'reference_check.json', report)
        write_json(output/'input_audit.json', audit_inputs(configurations['candidate'], configurations['control']))
        for arm, config in configurations.items():
            config['rgb_restoration']['output_dir'] = str(output/arm)
            (output/f'{arm}.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        files = ['train_rgb_diffusion_restoration.py', 'train_rgb_restoration.py',
                 'data/rgb_restoration_dataset.py', 'data/rgb_spatial_blur.py',
                 'models/rgb_diffusion_restoration.py', 'models/rgb_restoration.py',
                 'tools/rgb_spatial_monitor.py', 'tools/rgb_endpoint_monitor.py', 'tools/run_rgb_d5b.py',
                 'config/train/rgb_diffusion_d5b.yaml']
        write_json(output/'source_versions.json', {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                                 for name in files})
        status['completed'].append('preflight')
        if not args.preflight_only:
            for arm in (('candidate', 'control') if args.smoke else ('candidate',)):
                status['stage'] = arm
                command = [sys.executable, '-u', 'train_rgb_diffusion_restoration.py',
                           '--config', str(output/f'{arm}.yaml'), '--device', 'cuda']
                if args.smoke:
                    command.extend(['--stop-after-epoch', '1'])
                with (output/f'{arm}.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    status['child_pid'] = process.pid
                    save_status()
                    if process.wait() != 0:
                        raise RuntimeError(f'{arm} failed; inspect {arm}.log')
                status['completed'].append(arm)
            status['pair_check'] = verify_pair(output/'candidate', reference, smoke=args.smoke)
        status.update(status='completed', stage='completed')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        save_status()


if __name__ == '__main__':
    main()
