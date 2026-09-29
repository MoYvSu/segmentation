# -*- coding: utf-8 -*-
"""先1024再512，独立同初始化；训练完后自动生成全图与本尺度的完整对照。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_affinity_connectivity import SOURCES as BASE_SOURCES
from train_backend_adaptation import sha, write_json
from utils.config import load_config, project_path

SOURCES = tuple(dict.fromkeys((*BASE_SOURCES,
    'config/train/affinity_native.yaml', 'data/affinity_native.py', 'train_affinity_native.py',
    'tools/run_affinity_native.py', 'tools/affinity_native_views.py', 'tools/probe_affinity_patch.py',
    'tools/analyze_affinity_connectivity.py', 'tools/render_backend_ablation.py',
    'data/dataset.py', 'data/semantic_targets.py', 'data/offset_geometry_dataset.py',
    'data/sam2_geometry_dataset.py', 'data/direct_dual_head_dataset.py', 'data/mim_dataset.py',
    'data/rgb_restoration_dataset.py', 'data/rgb_spatial_blur.py',
    'train_direct_semantic_affinity.py', 'utils/affinity_loss.py', 'utils/affinity_fusion.py',
    'utils/patch_diagnostic.py', 'utils/offset_letterbox.py', 'utils/config.py',
    'models/rgb_restoration.py', 'models/rgb_diffusion_restoration.py')))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def verify_run(root, expected):
    states = []
    for size in (1024, 512):
        folder = root/f'p{size}'
        state = read(folder/'status.json')
        if state['status'] != 'complete' or state['updates'] != expected or state['paired_control_updates'] != expected:
            raise RuntimeError(f'Incomplete native{size} training')
        for key in ('reused_control_verified','frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
            if state.get(key) is not True:
                raise RuntimeError('Audit failed: ' + key)
        lines = (folder/'steps.jsonl').read_text(encoding='utf8').splitlines()
        if len(lines) != expected:
            raise RuntimeError('Missing native training receipts')
        states.append(state)
    for key in ('initial_state_sha256','frozen_state_sha256','restorer_state_sha256'):
        if states[0][key] != states[1][key]:
            raise RuntimeError('Native scales must share independent initialization: '+key)
    if states[0]['data']['parent_data'] != states[1]['data']['parent_data']:
        raise RuntimeError('Native scales use different source cohorts')
    return dict(passed=True, independent_initialization=True, control_retrained=False,
                updates_per_arm=expected)


def render_final(config, root):
    from data.mim_dataset import list_images
    from tools.analyze_affinity_connectivity import load_directory, render
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    for size in config['affinity_native']['sizes']:
        folder = root/f'p{size}'/'gallery'
        folder.mkdir(parents=True, exist_ok=False)
        for path in files:
            predictions = [load_directory(project_path(config, config['affinity_native']['reused_control'], 'deployment'), path.stem),
                           load_directory(root/'control_views'/f'patch{size}', path.stem),
                           load_directory(root/f'p{size}'/'deployment'/'global', path.stem),
                           load_directory(root/f'p{size}'/'deployment'/f'patch{size}', path.stem)]
            render(path, predictions, ['control whole', f'control native{size}', 'trained whole', f'trained native{size}'],
                   folder/(path.stem+'.png'), path.stem, width=320)


def run(args):
    config = load_config(args.config)
    opt = config['affinity_native']
    if opt['sizes'] != [1024, 512] or config['backend_adaptation']['epochs'] != 20:
        raise ValueError('This run is the authorized independent 1024 -> 512, 20 epochs each')
    root = Path(project_path(config, args.output or (config['backend_adaptation']['output_dir']+('_smoke' if args.smoke else ''))))
    if shutil.disk_usage(ROOT).free < opt['minimum_free_gib']*1024**3:
        raise RuntimeError('Insufficient fast-disk space for both independent runs')
    if not args.smoke:
        gate = Path(project_path(config, args.gate))
        state = read(gate/'pipeline_status.json')
        if state['status'] != 'complete' or state['config'] != json.loads(json.dumps(config)):
            raise RuntimeError('Exact configuration must pass two-arm smoke')
        for name in SOURCES:
            if (gate/'source'/name).read_bytes() != (ROOT/name).read_bytes():
                raise RuntimeError('Source changed after smoke: '+name)
        if not read(gate/'control_audit/report.json')['passed']:
            raise RuntimeError('Original full and native control output parity is required')
    root.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        target = root/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    status = dict(status='running', pid=os.getpid(), smoke=args.smoke, config=config,
                  stage='preflight', completed=[], control_retrained=False,
                  independent_initialization=True, automatic_extension=False, competition_submission=False,
                  sources={name:sha(ROOT/name) for name in SOURCES})
    def execute(stage, arguments):
        status['stage'] = stage
        write_json(root/'pipeline_status.json', status)
        with (root/(stage+'.log')).open('w', encoding='utf8') as log:
            subprocess.run([sys.executable, '-u', str(ROOT/'train_affinity_native.py'),
                            '--config', args.config, *arguments], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status['completed'].append(stage)
        write_json(root/'pipeline_status.json', status)
    write_json(root/'pipeline_status.json', status)
    try:
        if args.smoke:
            execute('control_audit', ['--mode','audit','--output',str(root/'control_audit')])
        # 两轮先接续训练，全部训练完成后才启动全量推理。
        for size in opt['sizes']:
            execute(f'p{size}', ['--size',str(size),'--output',str(root/f'p{size}')] + (['--smoke'] if args.smoke else []))
        status['paired_audit'] = verify_run(root, opt['smoke_updates'] if args.smoke else 1280)
        if not args.smoke:
            execute('control_infer', ['--mode','infer','--checkpoint',project_path(config,opt['reused_control'],'final.pt'),
                    '--output',str(root/'control_views'),'--views','1024','512'])
            for size in opt['sizes']:
                execute(f'p{size}_infer', ['--mode','infer','--checkpoint',str(root/f'p{size}'/'final.pt'),
                        '--output',str(root/f'p{size}'/'deployment'),'--views','0',str(size)])
            status['stage'] = 'render'
            write_json(root/'pipeline_status.json', status)
            render_final(config, root)
        status.update(status='complete', stage='done')
        write_json(root/'pipeline_status.json', status)
    except Exception as error:
        status.update(status='failed', error=repr(error))
        write_json(root/'pipeline_status.json', status)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/train/affinity_native.yaml')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--output')
    p.add_argument('--gate', default='outputs/affinity_native_smoke')
    run(p.parse_args())
