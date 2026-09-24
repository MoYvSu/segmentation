# -*- coding: utf-8 -*-
"""D5a与原空间增强从零训练的配对队列；不依赖历史权重，不提交黑盒。"""
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
from train_rgb_restoration import build_datasets, write_json
from utils.config import load_config, project_path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_recipes(candidate, control):
    left, right = copy.deepcopy(candidate['rgb_restoration']), copy.deepcopy(control['rgb_restoration'])
    left.pop('output_dir')
    right.pop('output_dir')
    endpoint = left['degradation']['spatial_blur'].pop('endpoint_transition')
    if left != right or not endpoint.get('enabled') or endpoint['probability'] != .5:
        raise ValueError('D5a paired recipes may differ only in endpoint_transition and output_dir')
    if left['split_policy'] != 'all_train' or left.get('holdout_manifest') is not None:
        raise ValueError('D5a must use all training sources without a held-out set')
    return endpoint


def audit_inputs(candidate, control, count=64):
    """少量真实训练源上的增强通路检查，不运行代理分割或选模型。"""
    sets = [build_datasets(cfg) for cfg in (candidate, control)]
    if any(heldout is not None for _, heldout in sets):
        raise ValueError('unexpected held-out data')
    a, b = [item[0] for item in sets]
    if a.samples != b.samples or len(a) != candidate['rgb_restoration']['expected_train_sources']:
        raise ValueError('unpaired training sources')
    indices = np.linspace(0, len(a)-1, min(count, len(a)), dtype=int)
    rows, counts = [], {arm: dict(identity=0, uniform=0, oldspatial=0, endpoint=0)
                       for arm in ('candidate', 'control')}
    for index in indices:
        left, right = a[int(index)], b[int(index)]
        if left['source'] != right['source'] or left['profile'] != right['profile']:
            raise AssertionError('unpaired source/profile')
        if not all(torch.equal(left[key], right[key]) for key in ('target', 'valid')):
            raise AssertionError('target or crop geometry changed')
        if not left['endpoint_blur_applied'] and not torch.equal(left['input'], right['input']):
            raise AssertionError('non-endpoint training input changed')
        if left['endpoint_blur_applied'] and not left['sigma_crop_coexist']:
            raise AssertionError('endpoint crop lacks simultaneous weak and strong regions')
        row = dict(source=left['source'], profile=left['profile'])
        for arm, item in (('candidate', left), ('control', right)):
            kind = ('identity' if item['profile'] == 'identity' else
                    'endpoint' if item['endpoint_blur_applied'] else
                    'oldspatial' if item['spatial_blur_applied'] else 'uniform')
            counts[arm][kind] += 1
            row[arm] = {key: item[key] for key in (
                'endpoint_blur_applied', 'sigma_crop_mean_square', 'sigma_crop_weak_fraction',
                'sigma_crop_strong_fraction', 'sigma_crop_span', 'sigma_crop_coexist')}
        rows.append(row)
    if counts['candidate']['endpoint'] == 0:
        raise AssertionError('preflight did not exercise endpoint blur')
    return dict(training_sources=len(a), audited_sources=len(rows), counts=counts, records=rows,
                target_geometry_equal=True, nonendpoint_inputs_equal=True,
                all_endpoint_crops_have_weak_and_strong=True,
                scope='augmentation_engineering_check_not_validation', nominal_sigma_only=True)


def verify_pair(output, configurations, *, smoke=False):
    cfg = configurations['candidate']['rgb_restoration']
    expected = 1 if smoke else cfg['train']['epochs']
    expected_updates = expected * ((cfg['expected_train_sources'] + cfg['train']['batch_size'] - 1)
                                   // cfg['train']['batch_size'])
    expected_status = 'segment_completed' if smoke else 'completed'
    summaries = [read(output/arm/'summary.json') for arm in ('candidate', 'control')]
    for summary in summaries:
        if (summary['status'] != expected_status or summary['epoch'] != expected or summary['failed_updates']
                or summary['updates'] != expected_updates):
            raise RuntimeError('incomplete paired training')
    starts = [torch.load(output/arm/'epoch_000.pt', map_location='cpu', weights_only=True)
              for arm in ('candidate', 'control')]
    if starts[0]['model'].keys() != starts[1]['model'].keys() or not all(
            torch.equal(value, starts[1]['model'][key]) for key, value in starts[0]['model'].items()):
        raise RuntimeError('scratch initial models differ')
    if any(cp['epoch'] or cp['updates'] or cp.get('continuation') for cp in starts):
        raise RuntimeError('D5a must start from random initialization with zero counters')
    fixed_equal = []
    for filename in ('fixed_samples.pt', 'spatial_fixed_samples.pt', 'endpoint_fixed_samples.pt', 'real_fixed_samples.pt'):
        pair = [torch.load(output/arm/filename, map_location='cpu', weights_only=True)
                for arm in ('candidate', 'control')]
        if pair[0].keys() != pair[1].keys() or any(
                not (torch.equal(value, pair[1][key]) if torch.is_tensor(value) else value == pair[1][key])
                for key, value in pair[0].items()):
            raise RuntimeError(f'fixed monitor inputs differ: {filename}')
        fixed_equal.append(filename)
    logs = [[json.loads(line) for line in (output/arm/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            for arm in ('candidate', 'control')]
    if len(logs[0]) != len(logs[1]) or len(logs[0]) != expected_updates:
        raise RuntimeError('incomplete update logs')
    for a, b in zip(*logs):
        for key in ('sources', 'profiles', 't_histogram', 'learning_rate', 'epoch', 'update'):
            if a[key] != b[key]:
                raise RuntimeError(f'paired sequence differs: {key}')
    report = dict(from_scratch=True, same_initial_weights=True, paired_samples_profiles_timesteps_lr=True,
                  same_fixed_inputs=fixed_equal, updates_per_arm=len(logs[0]), official_scores=None,
                  comparison='endpoint-transition versus original-spatial recipe; both from epoch zero')
    write_json(output/'pair_check.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='64 training-source engineering run, one epoch per arm')
    parser.add_argument('--preflight-only', action='store_true', help='audit paired inputs without starting training')
    parser.add_argument('--output-dir', help='new private queue directory')
    args = parser.parse_args()
    configurations = {arm: load_config(str(ROOT/'config'/'train'/filename)) for arm, filename in (
        ('candidate', 'rgb_diffusion_d5a.yaml'), ('control', 'rgb_diffusion_d5a_control.yaml'))}
    check_recipes(configurations['candidate'], configurations['control'])
    suffix = 'd5a_smoke' if args.smoke else 'd5a_preflight' if args.preflight_only else 'd5a'
    output = Path(args.output_dir or ROOT/'outputs'/suffix).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'queue directory must be new: {output}')
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        cfg = configurations['candidate']['rgb_restoration']
        sources = list_images(project_path(configurations['candidate'], cfg['data_dir']))
        if len(sources) != cfg['expected_train_sources']:
            raise ValueError('unexpected formal training pool before smoke')
        destination = output/'sources'
        destination.mkdir()
        for position in np.linspace(0, len(sources)-1, 64, dtype=int):
            source = Path(sources[int(position)]).resolve()
            (destination/source.name).symlink_to(source)
        for config in configurations.values():
            setting = config['rgb_restoration']
            setting.update(data_dir=str(destination), expected_train_sources=64)
            # 保持60轮日程和原mask warmup，仅让真实训练入口在首轮结束后停下。
            setting['monitor'].update(samples=1)
            for name in ('spatial_blur', 'endpoint_blur'):
                setting['monitor'][name]['samples'] = 1
            setting['monitor']['real']['images'] = ['test_101']
    for arm, config in configurations.items():
        config['rgb_restoration']['output_dir'] = str(output/arm)
        (output/f'{arm}.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    provenance = {}
    source_files = ['train_rgb_diffusion_restoration.py', 'train_rgb_restoration.py',
                    'data/rgb_restoration_dataset.py', 'data/rgb_spatial_blur.py',
                    'models/rgb_diffusion_restoration.py', 'models/rgb_restoration.py',
                    'tools/rgb_endpoint_monitor.py', 'tools/rgb_spatial_monitor.py', 'tools/run_rgb_d5a.py']
    for name in source_files:
        provenance[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    write_json(output/'source_versions.json', provenance)
    status = dict(status='running', stage='preflight', completed=[], from_scratch=True,
                  smoke=args.smoke, competition_submission=False)
    def save_status():
        write_json(output/'pipeline_status.json', status)
    save_status()
    try:
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        report = audit_inputs(configurations['candidate'], configurations['control'])
        write_json(output/'input_audit.json', report)
        status['completed'].append('preflight')
        if not args.preflight_only:
            for arm in ('candidate', 'control'):
                status['stage'] = arm
                save_status()
                command = [sys.executable, '-u', 'train_rgb_diffusion_restoration.py', '--config',
                           str(output/f'{arm}.yaml'), '--device', 'cuda']
                if args.smoke:
                    command.extend(['--stop-after-epoch', '1'])
                with (output/f'{arm}.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    status['child_pid'] = process.pid
                    save_status()
                    if process.wait() != 0:
                        raise RuntimeError(f'{arm} training failed; inspect {arm}.log')
                status['completed'].append(arm)
            status['pair_check'] = verify_pair(output, configurations, smoke=args.smoke)
        status.update(status='completed', stage='completed')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        save_status()


if __name__ == '__main__':
    main()
