# -*- coding: utf-8 -*-
"""全图与局部D5a均显式零噪声；固定1024/旧20a59/S0/.65/final200。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_backend_adaptation as backend_entry
from models import backend_adaptation
from tools import infer_boundary_rgb as naf_entry
from tools import infer_semifinal as entry
from utils import restoration_noise as noise_module
from utils import restoration_probe
from utils.instance_area_policy import resolve_instance_area_policy

FORMAT = 'all_zero_restoration_inference_v1'
ZERO_MODE = 'explicit_zero_v1'
FINAL_POLICY = dict(seed_min_area=50, final_min_instance_area=200,
                    final_small_instance_action='neighbor_merge_v1', pre_vote_final_min_area=0)


def zero_contract():
    return dict(format=FORMAT, restoration_noise=ZERO_MODE, global_noise='zero', local_noise='zero',
        native_sizes=[1024], network_input_size=1024, naf_tile_size=256,
        naf_model_sha256=naf_entry.EXPECTED_MODEL_SHA, seed_support='original-zero',
        final_instance_area_policy=deepcopy(FINAL_POLICY), random_noise_generated=False,
        prediction_noise_generators_forbidden=True,
        initial_state='q_sample(condition, condition, terminal_step, noise=zeros_like(condition))')


def validate_baseline(config):
    entry.validate_config(config)
    if entry.local_restoration(config) != dict(mode='zero', model='restorer', candidate='N1'):
        raise ValueError('All-zero inference requires the N1 baseline')
    if config['submission']['models'].get('terminal_semantic') is not None:
        raise ValueError('All-zero inference keeps the original semantic head')
    if resolve_instance_area_policy(config['inference']) != FINAL_POLICY:
        raise ValueError('All-zero inference requires seed50 and final200 neighbor_merge_v1')


def execution_config(config):
    validate_baseline(config)
    effective = deepcopy(config)
    effective['submission']['restoration_noise'] = ZERO_MODE
    effective['submission']['global_restoration'] = dict(mode='zero', model='restorer')
    effective['all_zero_restoration'] = zero_contract()
    return effective


def forbidden_noise(*args, **kwargs):
    raise RuntimeError('Random restoration noise generation is forbidden during all-zero prediction')


def zero_predict(original_predict, records):
    """只在实际预测期间拦截噪声；模型构造和加载不改变。"""
    original_probe = restoration_probe.restore_probe

    def predict(model, restorer, terminal, path, config, device, *, local_restorer=None,
                fixed_audit=None, local_audit=None, analysis_callback=None):
        if terminal is not None or local_restorer is not None:
            raise ValueError('All-zero inference does not replace semantic/local models')
        effective = execution_config(config)
        calls = []
        windows = [] if local_audit is None else local_audit

        def restore(actual_restorer, image, seed, branch, audit=None):
            if actual_restorer is not restorer:
                raise RuntimeError('All-zero branch used an unexpected restorer')
            proof = {}
            value, raw = original_probe(actual_restorer, image, seed, 'zero', audit=proof)
            if (proof.get('noise_nonzero') != 0 or proof.get('initial_equals_condition') is not True
                    or proof.get('q_sample_calls') != 1 or proof.get('initial_max_abs') != 0.0):
                raise RuntimeError('D5a did not receive an explicit zero-noise condition state')
            if audit is not None:
                audit.update(proof)
            calls.append(dict(branch=branch, mode='zero', shape=list(image.shape),
                              seed_ignored_for_noise=True, **proof))
            return value, raw

        def global_restore(actual_restorer, image, seed, strength=1.0):
            if float(strength) != 1.0:
                raise ValueError('All-zero inference keeps full D5a first-step restoration')
            return restore(actual_restorer, image, seed, 'global')[0]

        def local_restore(actual_restorer, image, seed, mode='single', *, positive_raw=None, audit=None):
            if mode != 'zero' or positive_raw is not None:
                raise ValueError('Local D5a must use explicit zero noise without cached alternative states')
            return restore(actual_restorer, image, seed, 'local', audit)

        with ExitStack() as stack:
            stack.enter_context(patch.object(backend_entry, 'restore_first', side_effect=global_restore))
            stack.enter_context(patch.object(entry.views, 'restore_probe', side_effect=local_restore))
            # 任何意外绕回旧single/固定Philox路径均失败，禁止静默生成再乘零。
            for module, name in ((torch, 'randn'), (torch, 'randn_like'), (torch, 'normal'),
                    (backend_adaptation, 'restoration_noise'), (noise_module, 'restoration_noise'),
                    (restoration_probe, 'restoration_noise'), (entry.views, 'restore_first')):
                stack.enter_context(patch.object(module, name, side_effect=forbidden_noise))
            rgb, result = original_predict(model, restorer, terminal, path, effective, device,
                local_restorer=None, fixed_audit=fixed_audit, local_audit=windows,
                analysis_callback=analysis_callback)
        global_calls = [row for row in calls if row['branch'] == 'global']
        local_calls = [row for row in calls if row['branch'] == 'local']
        if len(global_calls) != 1 or not windows or len(local_calls) != len(windows):
            raise RuntimeError('Expected one global zero restoration and one per actual local window')
        if any(row.get('size') != 1024 or row.get('noise_mode') != 'zero'
                or row.get('noise_nonzero') != 0 or row.get('initial_equals_condition') is not True for row in windows):
            raise RuntimeError('Actual local windows violate the native1024 zero-noise contract')
        records.append(dict(source=path.stem, global_zero_calls=1, local_zero_calls=len(local_calls),
                            random_noise_generated=False, calls=calls))
        return rgb, result

    return predict


def audit_document(document, records, source_sha):
    value = deepcopy(document)
    value['legacy_format'] = value.get('legacy_format', value.get('format'))
    value['format'] = FORMAT
    value['all_zero_restoration'] = zero_contract()
    value['zero_entry_sha256'] = source_sha
    value['zero_noise_images'] = deepcopy(records)
    value['global_zero_calls'] = sum(row['global_zero_calls'] for row in records)
    value['local_zero_calls'] = sum(row['local_zero_calls'] for row in records)
    if 'config' in value:
        baseline = deepcopy(value.get('baseline_config', value['config']))
        value['baseline_config'] = baseline
        value['config'] = execution_config(baseline)
        value['config_role'] = 'effective all-zero execution; baseline_config preserves historical validation settings'
    if 'runtime' in value:
        value['baseline_runtime'] = deepcopy(value.get('baseline_runtime', value['runtime']))
        value['runtime']['restoration_noise'] = dict(name=ZERO_MODE, global_noise='zero', local_noise='zero',
            random_noise_generated=False, historical_restorer_noise_attribute_unused=True)
    for key in ('geometry', 'upstream_geometry'):
        if isinstance(value.get(key), str) and not value[key].startswith('global/local D5a explicit zero; '):
            value[key] = 'global/local D5a explicit zero; ' + value[key]
    return value


def run(args, *, device='cuda'):
    args = argparse.Namespace(**vars(args))
    if getattr(args, 'global_noise', 'zero') != 'zero':
        raise ValueError('This independent entry requires global-noise zero')
    if getattr(args, 'expected_model_sha', naf_entry.EXPECTED_MODEL_SHA) != naf_entry.EXPECTED_MODEL_SHA:
        raise ValueError('All-zero inference is fixed to old 20a59 NAF')
    if getattr(args, 'seed_support', 'original-zero') != 'original-zero':
        raise ValueError('All-zero inference requires original-zero S0')
    args.expected_model_sha, args.seed_support = naf_entry.EXPECTED_MODEL_SHA, 'original-zero'
    baseline = entry.load_config(str(Path(args.config).resolve()))
    if getattr(args, 'dataset', None):
        baseline['submission']['dataset'] = args.dataset
    validate_baseline(baseline)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh all-zero inference directory')
    source_sha = entry.sha(__file__)
    original_write, original_predict = entry.write, entry.predict
    original_sources = entry.local_module_sources
    records = []

    def write(path, value):
        path = Path(path)
        if path.parent == output / 'audit' and path.name in (
                'report.json', 'status.json', 'naf_report.json', 'naf_failed.json'):
            value = audit_document(value, records, source_sha)
        original_write(path, value)

    try:
        with patch.object(entry, 'predict', side_effect=zero_predict(original_predict, records)), \
                patch.object(entry, 'write', side_effect=write), \
                patch.object(entry, 'local_module_sources',
                             new=lambda: original_sources(include_noise_resources=False)):
            receipt = naf_entry.run(args, device=device)
        if entry.sha(__file__) != source_sha or [r['source'] for r in records] != [r['source'] for r in receipt['images']]:
            raise RuntimeError('All-zero entry/source invocation changed')
        result = audit_document(receipt, records, source_sha)
        original_write(output / 'audit/zero_report.json', result)
        return result
    except Exception as exc:
        if (output / 'audit').is_dir():
            original_write(output / 'audit/zero_failed.json', dict(complete=False, format=FORMAT,
                error=repr(exc), zero_noise_images=records))
            for name in ('report.json', 'status.json'):
                path = output / 'audit' / name
                if path.is_file():
                    value = json.loads(path.read_text(encoding='utf-8'))
                    value.update(complete=False, zero_complete=False, zero_error=repr(exc))
                    if name == 'status.json':
                        value['status'] = 'failed'
                    original_write(path, value)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default=str(ROOT / 'config/inference/area_floor_200.yaml'))
    parser.add_argument('--model-dir')
    parser.add_argument('--dataset', choices=('quarter_final_100', 'semifinal_150'))
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--naf-checkpoint', required=True)
    parser.add_argument('--capture-names', nargs='+')
    parser.add_argument('--global-noise', choices=('zero',), default='zero')
    result = run(parser.parse_args(argv))
    print(json.dumps(dict(complete=True, format=FORMAT, images=len(result['images']),
                         global_zero_calls=result['global_zero_calls'], local_zero_calls=result['local_zero_calls'])), flush=True)


if __name__ == '__main__':
    main()
