# -*- coding: utf-8 -*-
"""单原生视野研究入口：旧N1/20a59/S0/final200固定，显式比较1024与768。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import infer_boundary_rgb as naf_entry
from tools import infer_semifinal as entry
from utils.instance_area_policy import resolve_instance_area_policy

FORMAT = 'native_scale_research_v1'
SIZES = (1024, 768)
FINAL_POLICY = dict(seed_min_area=50, final_min_instance_area=200,
                    final_small_instance_action='neighbor_merge_v1', pre_vote_final_min_area=0)


def research_contract(native_size):
    if type(native_size) is not int or native_size not in SIZES:
        raise ValueError('Research native size must be exactly 1024 or 768')
    return dict(format=FORMAT, native_size=native_size, native_sizes=[native_size],
        baseline_native_sizes=[1024], network_input_size=1024, affinity_grid=512,
        overlap=.25, local_restoration='zero', global_restoration_noise=entry.NOISE_MODE,
        naf_model_sha256=naf_entry.EXPECTED_MODEL_SHA, naf_tile_size=256,
        seed_support='original-zero', support_reference=f'own_N1_{native_size}_pre_NAF',
        final_instance_area_policy=deepcopy(FINAL_POLICY),
        source_configuration_role='validated legacy1024 baseline; scale override is explicit research only')


def validate_baseline(config):
    """不放宽正式入口；研究只允许改变局部原生视野。"""
    entry.validate_config(config)
    if entry.local_restoration(config) != dict(mode='zero', model='restorer', candidate='N1'):
        raise ValueError('Scale research requires the unchanged N1 zero-local restoration')
    if config['submission']['models'].get('terminal_semantic') is not None:
        raise ValueError('Scale research keeps the original semantic head')
    if resolve_instance_area_policy(config['inference']) != FINAL_POLICY:
        raise ValueError('Scale research requires seed50 and final200 neighbor_merge_v1')


def execution_config(baseline, native_size):
    validate_baseline(baseline)
    effective = deepcopy(baseline)
    effective['submission']['native_sizes'] = [native_size]
    effective['native_scale_research'] = research_contract(native_size)
    return effective


def scale_predict(native_size, records):
    """沿用正式predict的算术/调用顺序，仅显式选择一个patch<size>。"""
    research_contract(native_size)

    def predict(model, restorer, terminal, path, config, device, *, local_restorer=None,
                fixed_audit=None, local_audit=None, analysis_callback=None):
        if terminal is not None or local_restorer is not None:
            raise ValueError('Scale research does not use replacement semantic/local models')
        effective = execution_config(config, native_size)
        policy = entry.local_restoration(effective)
        original_decode = entry.views.decode_native

        def reflected_decode(semantic, boundary, actual_config, *args, **kwargs):
            if args or kwargs:
                raise RuntimeError('Unexpected native geometry decode arguments')
            return entry.decode_reflect(semantic, boundary, actual_config, original_decode=original_decode)

        with torch.autocast(device_type=torch.device(device).type, enabled=False), \
                patch.object(entry.views, 'decode_native', side_effect=reflected_decode):
            rgb, predictions = entry.views.predict_views(
                model, restorer, path, effective, device,
                entry.source_seed(path.stem, effective['submission'].get('dataset')), [native_size],
                local_restorer=local_restorer, local_noise_mode=policy['mode'],
                fixed_audit=fixed_audit, local_audit=local_audit, analysis_callback=analysis_callback)
            if set(predictions) != {f'patch{native_size}'}:
                raise RuntimeError('Single-scale research returned unexpected view keys')
            result = entry.finalize_instance_area(predictions[f'patch{native_size}'], effective)
        entry.verify_prediction(result['instances'], result['classes'], rgb.shape[:2])
        windows = local_audit if local_audit is not None else []
        if any(row.get('size') != native_size or row.get('noise_mode') != 'zero' for row in windows):
            raise RuntimeError('Actual local windows differ from the research scale/zero-noise contract')
        records.append(dict(source=path.stem, native_size=native_size, view_key=f'patch{native_size}',
                            local_window_count=len(windows), single_decode_and_raw_vote='enforced_by_NAF_wrapper'))
        return rgb, result

    return predict


def audit_document(document, native_size, source_sha):
    """源配置与实际执行配置分开保存，运行中回执也不冒称native1024。"""
    value = deepcopy(document)
    value['legacy_format'] = value.get('legacy_format', value.get('format'))
    value['format'] = FORMAT
    value['native_scale_research'] = research_contract(native_size)
    value['research_entry_sha256'] = source_sha
    if 'config' in value:
        baseline = deepcopy(value.get('baseline_config', value['config']))
        value['baseline_config'] = baseline
        value['config'] = execution_config(baseline, native_size)
        value['config_role'] = 'effective execution configuration; baseline_config retains validated source settings'
    for key in ('geometry', 'upstream_geometry'):
        if isinstance(value.get(key), str):
            value[key] = value[key].replace('native1024', f'native{native_size}')
    return value


def run(args, *, device='cuda'):
    args = argparse.Namespace(**vars(args))
    native_size = args.native_size
    contract = research_contract(native_size)
    if getattr(args, 'expected_model_sha', naf_entry.EXPECTED_MODEL_SHA) != naf_entry.EXPECTED_MODEL_SHA:
        raise ValueError('Scale research is fixed to the old 20a59 NAF model')
    if getattr(args, 'seed_support', 'original-zero') != 'original-zero':
        raise ValueError('Scale research requires original-zero S0')
    args.expected_model_sha = naf_entry.EXPECTED_MODEL_SHA
    args.seed_support = 'original-zero'
    baseline = entry.load_config(str(Path(args.config).resolve()))
    if getattr(args, 'dataset', None):
        baseline['submission']['dataset'] = args.dataset
    validate_baseline(baseline)
    source = Path(__file__).resolve()
    source_sha = entry.sha(source)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh scale research output directory')
    original_write = entry.write
    records = []

    def write(path, document):
        path = Path(path)
        if path.parent == output / 'audit' and path.name in (
                'report.json', 'status.json', 'naf_report.json', 'naf_failed.json'):
            document = audit_document(document, native_size, source_sha)
        return original_write(path, document)

    with patch.object(entry, 'predict', side_effect=scale_predict(native_size, records)), \
            patch.object(entry, 'write', side_effect=write):
        receipt = naf_entry.run(args, device=device)
    if entry.sha(source) != source_sha:
        raise RuntimeError('Scale research entry changed during inference')
    if [row['source'] for row in records] != [row['source'] for row in receipt['images']]:
        raise RuntimeError('Scale research source invocation count/order differs')
    result = audit_document(receipt, native_size, source_sha)
    result['scale_images'] = records
    original_write(output / 'audit/scale_report.json', result)
    return result


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
    parser.add_argument('--native-size', type=int, choices=SIZES, required=True)
    result = run(parser.parse_args(argv))
    print(json.dumps(dict(complete=True, format=FORMAT, native_size=result['native_scale_research']['native_size'],
                         images=len(result['images']), parameter_count=result['parameter_count'])), flush=True)


if __name__ == '__main__':
    main()
