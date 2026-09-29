# -*- coding: utf-8 -*-
"""短测通过后顺序运行20轮A/B，保存过程图，完成后推理并停止。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import load_config, project_path
from train_semantic_consistency import candidate_arm
from train_backend_adaptation import sha
from data.mim_dataset import list_images


STREAM_FIELDS = ('epoch', 'step', 'updates', 'name', 'seed', 'draw_index', 'input_sha256',
                 'appearance_input_sha256', 'appearance', 'restoration_seed', 'learning_rate')
COMPLETION_FLAGS = ('initial_final_output_equal', 'frozen_weights_unchanged', 'd5a_unchanged',
                    'fixed_teacher_unchanged', 'frozen_affinity_output_equal',
                    'semantic_weights_changed', 'strict_reload_passed')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def same_config(left, right):
    """JSON回执会将类别的整数键变成字符串；统一表示后核对真实配置值。"""
    return json.loads(json.dumps(left, allow_nan=False)) == json.loads(json.dumps(right, allow_nan=False))


def control_config(config):
    """只忽略控制组不执行的损失参数和输出／复用位置，其余配置必须相同。"""
    result = deepcopy(config)
    result['semantic_adaptation'].pop('output_dir', None)
    result['semantic_consistency'].pop('reuse_control_dir', None)
    result['semantic_consistency'].pop('maximum_weight', None)
    # 此配置仅在额外无标签前向与候选过程图中使用，控制的GT计算完全不读取。
    result.pop('semantic_hard', None)
    for key in ('loss_mode', 'minimum_confidence', 'maximum_normalization_gain'):
        result.get('semantic_gray', {}).pop(key, None)
    return result


def verify_control_prefix(old, short):
    """新代码短测必须复现旧完整控制的相同前缀，不能仅凭配置或seed复用。"""
    for key in ('sources', 'data', 'frozen_state_sha256', 'd5a_state_sha256', 'initial_semantic_sha256'):
        if old[key] != short[key]:
            raise RuntimeError(f'Reused control dependency differs: {key}')
    if not same_config(control_config(old['config']), control_config(short['config'])):
        raise RuntimeError('Reused control training configuration differs')


def validate_reused_control(control, config, short_control):
    """只读核验旧完整训练、部署与新代码短测；不复制或改写旧产物。"""
    control, short_control = Path(control), Path(short_control)
    old, short = read(control / 'status.json'), read(short_control / 'status.json')
    cfg = config['semantic_adaptation']
    expected = cfg['epochs'] * cfg['expected_manual_sources'] * cfg['manual_repeats']
    if (old['status'] != 'completed' or old.get('smoke') or old['arm'] != 'control'
            or old['epoch'] != cfg['epochs'] or old['updates'] != expected or old['failed_updates']):
        raise RuntimeError('Reused control is not a complete matched training run')
    if any(old.get(key) is not True for key in COMPLETION_FLAGS):
        raise RuntimeError('Reused control failed its training contract')
    if not same_config(control_config(old['config']), control_config(config)):
        raise RuntimeError('Reused control training configuration differs')
    if (short['status'] != 'completed' or not short.get('smoke') or short['arm'] != 'control'
            or short['updates'] != config['semantic_consistency']['smoke_steps']):
        raise RuntimeError('Current control short test is incomplete')
    verify_control_prefix(old, short)
    left, right = rows(control), rows(short_control)
    if len(left) != expected or len(right) != short['updates']:
        raise RuntimeError('Incomplete reused-control step receipts')
    for x, y in zip(left, right):
        for key in (*STREAM_FIELDS, 'semantic_loss'):
            if x[key] != y[key]:
                raise RuntimeError(f'Current short test does not reproduce reused control: {key}')
        if x['unlabeled'] is not None or y['unlabeled'] is not None:
            raise RuntimeError('Reused control unexpectedly used unlabeled targets')
    checkpoint = Path(old['final_checkpoint'])
    manifest = read(control / 'deployment/manifest.json')
    if not checkpoint.is_file() or manifest['checkpoint_sha256'] != sha(checkpoint):
        raise RuntimeError('Reused control checkpoint is missing or changed')
    if manifest['epoch'] != cfg['epochs'] or not same_config(control_config(manifest['config']), control_config(config)):
        raise RuntimeError('Reused control deployment contract differs')
    restoration = cfg['restoration']
    if (manifest['d5a_sha256'] != restoration['checkpoint_sha256']
            or manifest['sampling_mode'] != restoration['mode'] or manifest['precision'] != 'FP32'):
        raise RuntimeError('Reused control restoration or precision differs')
    names = [Path(p).name for p in list_images(project_path(config, config['inference']['test_dir']))]
    previous_names = [r['image'] for r in manifest['images']]
    if not names or len(set(previous_names)) != len(previous_names) or set(names) != set(previous_names):
        raise RuntimeError('Reused control deployment image set is incomplete or different')
    for name in names:
        for suffix in ('_inst.png', '_class.json', '_class_confidence.json'):
            if not (control / 'deployment' / (Path(name).stem + suffix)).is_file():
                raise RuntimeError('Reused control deployment file is missing')
    return {'passed': True, 'directory': str(control), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': manifest['checkpoint_sha256'], 'deployment': str(control / 'deployment'),
            'images': len(names), 'prefix_replayed_steps': len(right), 'prefix_max_loss_difference': 0.,
            'same_training_configuration_and_dependencies': True}


def rows(directory):
    return [json.loads(line) for line in (Path(directory) / 'steps.jsonl').read_text(encoding='utf-8').splitlines()]


def verify_pair(control, candidate, config, smoke=False, zero=False, reuse_control=False):
    a, b = read(control / 'status.json'), read(candidate / 'status.json')
    cfg, ucfg = config['semantic_adaptation'], config['semantic_consistency']
    expected = ucfg['smoke_steps'] if smoke else cfg['epochs'] * cfg['expected_manual_sources'] * cfg['manual_repeats']
    for result in (a, b):
        if result['status'] != 'completed' or result['updates'] != expected or result['failed_updates']:
            raise RuntimeError('Incomplete training comparison')
        for key in COMPLETION_FLAGS:
            if result.get(key) is not True:
                raise RuntimeError(f'Training contract failed: {key}')
    for key in ('sources', 'data', 'frozen_state_sha256', 'd5a_state_sha256', 'initial_semantic_sha256'):
        if a[key] != b[key]:
            raise RuntimeError(f'A/B shared contract differs: {key}')
    configs = [control_config(r['config']) if reuse_control else r['config'] for r in (a, b)]
    if not same_config(*configs):
        raise RuntimeError('A/B shared contract differs: config')
    mode = candidate_arm(config)
    if mode == 'consistency' and b['teacher_state_sha256'] != a['initial_semantic_sha256']:
        raise RuntimeError('Teacher is not the frozen initial simple head')
    if mode == 'prior' and b['teacher_state_sha256'] is not None:
        raise RuntimeError('Gray prior must not depend on a neural teacher')
    left, right = rows(control), rows(candidate)
    if len(left) != expected or len(right) != expected:
        raise RuntimeError('Incomplete step receipts')
    max_loss_difference = 0.
    for x, y in zip(left, right):
        for key in STREAM_FIELDS:
            if x[key] != y[key]:
                raise RuntimeError(f'Supervised stream differs across arms: {key}')
        if x['unlabeled'] is not None or y['unlabeled'] is None:
            raise RuntimeError('Unexpected unlabeled branch routing')
        if zero:
            if y['unlabeled_weight'] != 0 or not np.isclose(x['semantic_loss'], y['semantic_loss'], rtol=1e-3, atol=1e-6):
                raise RuntimeError('Zero-weight replay perturbed supervised training')
            max_loss_difference = max(max_loss_difference, abs(x['semantic_loss'] - y['semantic_loss']))
    if b['visited_sources'] != min(expected, ucfg['expected_sources']):
        raise RuntimeError('Full-cycle source coverage failed')
    if min(b['accepted_ferrite_pixels'], b['accepted_pearlite_pixels']) <= 0:
        raise RuntimeError('Reliable targets did not cover both predicted classes')
    if not zero and not any(r['unlabeled_grad_norm'] is not None and r['unlabeled_grad_norm'] > 0 for r in right):
        raise RuntimeError('No measured nonzero unlabeled gradient')
    max_weight_difference = None
    if zero:
        sa = torch.load(a['final_checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
        sb = torch.load(b['final_checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
        max_weight_difference = max(float((sa[k].float() - sb[k].float()).abs().max())
                                    for k in sa if k.startswith('semantic_decoder.'))
        if max_weight_difference > 2e-6:
            raise RuntimeError(f'Zero-weight replay changed semantic weights: {max_weight_difference}')
    return {'passed': True, 'updates_per_arm': expected, 'reused_control': reuse_control,
            'same_supervised_inputs_and_seeds': True,
            'same_initial_and_frozen_weights': True, 'zero_weight_replay': zero,
            'zero_replay_max_loss_difference': max_loss_difference if zero else None,
            'zero_replay_max_weight_difference': max_weight_difference,
            'visited_unlabeled_sources': b['visited_sources'], 'effective_unlabeled_sources': b['effective_sources'],
            'changed_factor': ('additional_independent_gray_prior_on_allowed_training_pool' if mode == 'prior'
                               else 'additional_reliable_semantic_consistency_on_allowed_training_pool'),
            'official_score': None}


def snapshot_sources(directory):
    """只备份本轮实际入口与依赖代码；数据及模型不会被复制。"""
    files = ['train_semantic_consistency.py', 'tools/run_semantic_consistency.py',
             'data/semantic_consistency.py', 'data/semantic_illumination.py', 'data/backend_adaptation.py', 'data/rgb_restoration_dataset.py',
             'data/rgb_spatial_blur.py', 'train_semantic_d5a.py', 'train_backend_adaptation.py',
             'train_direct_semantic_affinity.py', 'models/backend_adaptation.py',
             'tools/render_backend_ablation.py', 'utils/config.py', 'data/semantic_gray.py',
             'tools/run_semantic_gray.py', 'tools/analyze_semantic_domain.py',
             'data/semantic_hard.py', 'tools/probe_semantic_hard.py']
    for name in files:
        target = directory / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)


def run_pipeline(config_path, smoke=False):
    config = load_config(config_path)
    candidate = candidate_arm(config)
    cfg = config['semantic_adaptation']
    output = Path(project_path(config, cfg['output_dir'] + ('_smoke' if smoke else '')))
    reuse_path = config['semantic_consistency'].get('reuse_control_dir')
    reused_control = Path(project_path(config, reuse_path)) if reuse_path else None
    reference_check = None
    if not smoke:
        gate_dir = Path(project_path(config, cfg['output_dir'] + '_smoke'))
        gate = read(gate_dir / 'pipeline_status.json')
        if gate.get('status') != 'completed' or not same_config(gate['config'], config):
            raise RuntimeError('Run a successful short test with the exact same configuration first')
        # 短测后若入口被改动，必须重新短测，不能用旧回执放行新实现。
        for source in (gate_dir / 'source').rglob('*.py'):
            # 协调器的日志/回执读取修正不改变已短测的训练入口与模型计算。
            if source.relative_to(gate_dir / 'source').as_posix() == 'tools/run_semantic_consistency.py':
                continue
            if source.read_bytes() != (ROOT / source.relative_to(gate_dir / 'source')).read_bytes():
                raise RuntimeError(f'Source changed after the accepted smoke: {source.name}')
        if reused_control is not None:
            reference_check = validate_reused_control(reused_control, config, gate_dir / 'control')
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(output)
    snapshot = output / 'config.yaml'
    snapshot.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    status = {'status': 'running', 'stage': 'preflight', 'completed': [], 'smoke': smoke,
              'config': config, 'competition_submission': False, 'automatic_extension': False}
    control = reused_control if reused_control is not None and not smoke else output / 'control'
    if reference_check is not None:
        status['reused_control'] = reference_check
        (output / 'control_reference.json').write_text(json.dumps(reference_check, indent=2), encoding='utf-8')
    def write():
        (output / 'pipeline_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    def execute(name, command):
        status['stage'] = name
        write()
        with (output / f'{name}.log').open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, '-u', *command], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status['completed'].append(name)
        write()
    flags = ['--smoke'] if smoke else []
    try:
        arms = ('control', 'zero', candidate) if smoke else ((candidate,) if reused_control is not None else ('control', candidate))
        for arm in arms:
            execute('train_' + arm, ['train_semantic_consistency.py', '--config', str(snapshot),
                '--arm', candidate if arm == 'zero' else arm, '--output-dir', str(output / arm),
                *flags, *(['--zero-weight'] if arm == 'zero' else [])])
        if smoke:
            status['zero_replay'] = verify_pair(output / 'control', output / 'zero', config, smoke=True, zero=True)
            if reused_control is not None:
                status['reused_control'] = validate_reused_control(reused_control, config, output / 'control')
        status['pair_check'] = verify_pair(control, output / candidate, config, smoke=smoke,
                                           reuse_control=reused_control is not None and not smoke)
        epoch = 1 if smoke else cfg['epochs']
        for arm in (('control', candidate) if smoke or reused_control is None else (candidate,)):
            run = output / arm
            execute('infer_' + arm, ['train_semantic_d5a.py', '--mode', 'infer', '--arm', 'simple',
                '--config', str(snapshot), '--output-dir', str(run), '--prediction-dir', str(run / 'deployment'),
                '--checkpoint', str(run / f'epoch_{epoch:03d}.pt'), *flags])
        execute('render', ['tools/render_backend_ablation.py', '--image-dir', project_path(config, config['inference']['test_dir']),
            '--prediction', 'Original simple=' + project_path(config, config['semantic_consistency']['original_prediction_dir']),
            '--prediction', 'Control=' + str(control / 'deployment'),
            '--prediction', ('Gray prior=' if candidate == 'prior' else 'Consistency=') + str(output / candidate / 'deployment'),
            '--images', *cfg['monitor']['images'], '--output-dir', str(output / 'comparison')])
        status.update(status='completed', stage='completed')
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_consistency.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_pipeline(args.config, args.smoke)


if __name__ == '__main__':
    main()
