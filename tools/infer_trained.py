# -*- coding: utf-8 -*-
"""训练导出清单的显式Torch推理桥：固定零噪C算法，不冒称旧C权重身份。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import infer_submission as submission
from tools.build_semifinal_bundle import state_digest

compact, entry, zero, flat, naf = (submission.compact, submission.entry, submission.zero,
                                  submission.flat, submission.naf)
FORMAT = 'trained_export_C_inference_v1'
TRAINING_FORMAT = 'current_training_export_manifest_v1'
MODEL_FORMATS = dict(backend='phase_affinity_fused_v1', restorer='rgb_restoration_diffusion_v1',
                     naf_export='boundary_rgb_training_v1')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _positive_integer(value):
    return type(value) is int and value > 0


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate manifest key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_manifest(path):
    """只读CPU核验真实导出状态；不构建SAM2/NAF模型或加载CUDA。"""
    path = Path(path).resolve(strict=True)
    document = _json(path)
    if (not isinstance(document, dict) or document.get('format') != TRAINING_FORMAT
            or document.get('parameter_limit_passed') is not True
            or document.get('inference_noise') != 'explicit_zero'
            or document.get('GT_modified') is not False
            or not isinstance(document.get('models'), dict)
            or set(document['models']) != set(MODEL_FORMATS)):
        raise ValueError('Expected the complete three-model current training export manifest')
    total = document.get('state_elements_parameter_upper_bound')
    if not _positive_integer(total) or total >= 500_000_000:
        raise ValueError('Training export parameter upper bound must be below 500M')
    run_root = path.parent.parent
    files, records, measured_total = {}, {}, 0
    for role, expected_format in MODEL_FORMATS.items():
        record = document['models'][role]
        if (not isinstance(record, dict) or record.get('format') != expected_format
                or not isinstance(record.get('path'), str) or not _digest(record.get('file_sha256'))
                or not _positive_integer(record.get('state_tensors'))
                or not _positive_integer(record.get('state_elements_parameter_upper_bound'))):
            raise ValueError('Invalid training export model record: ' + role)
        declared = Path(record['path'])
        file = (declared if declared.is_absolute() else run_root / declared).resolve(strict=True)
        if not file.is_relative_to(run_root) or not file.is_file() or entry.sha(file) != record['file_sha256']:
            raise ValueError('Training export model path/file SHA differs: ' + role)
        payload = torch.load(file, map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or payload.get('format') != expected_format:
            raise ValueError('Checkpoint format differs from its export role: ' + role)
        key = 'model' if role == 'restorer' else 'model_state_dict'
        state = payload.get(key)
        # 复用正式导出的FP32/稠密命名tensor合同和同一state摘要算法。
        digest = state_digest(state)
        if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError('Nonfinite checkpoint state: ' + role)
        elements = sum(value.numel() for value in state.values())
        if (len(state) != record['state_tensors']
                or elements != record['state_elements_parameter_upper_bound']):
            raise ValueError('Actual checkpoint tensor counts differ from manifest: ' + role)
        if payload.get('state_sha256') is not None and payload['state_sha256'] != digest:
            raise ValueError('Exported checkpoint state SHA differs: ' + role)
        extra = {}
        if role == 'backend' and not isinstance(payload.get('architecture'), dict):
            raise ValueError('Complete backend architecture is missing')
        if role == 'restorer':
            cfg = payload.get('model_config', {})
            if (not isinstance(cfg, dict) or cfg.get('architecture') != 'pixel_diffusion_nafnet'
                    or cfg.get('num_steps') != 16 or cfg.get('kappa') != .03):
                raise ValueError('Expected D5a first-x0 16steps/kappa.03 architecture')
        if role == 'naf_export':
            if type(payload.get('global_step')) is not int or payload['global_step'] != 1000:
                raise ValueError('Expected formal NAF step1000 export')
            naf.validate_input_contract(payload.get('input_contract'))
            numerical = payload.get('identity')
            if (not isinstance(numerical, dict)
                    or not naf._exact_contract(numerical.get('numerical_contract'), naf.NUMERICAL_CONTRACT)):
                raise ValueError('NAF numerical contract differs')
            # 此算法包含dtype/shape，与entry.tensor_digest并非同一摘要，不能混用。
            boundary_digest = naf.state_sha256(SimpleNamespace(state_dict=lambda: state))
            if not _digest(payload.get('model_sha256')) or payload['model_sha256'] != boundary_digest:
                raise ValueError('NAF checkpoint model/state SHA differs')
            extra['state_sha256'] = boundary_digest
        files[role] = file
        records[role] = dict(file=file.relative_to(run_root).as_posix(), sha256=record['file_sha256'],
                             format=expected_format, state_tensors=len(state),
                             state_elements_parameter_upper_bound=elements, entry_state_sha256=digest, **extra)
        measured_total += elements
        del payload, state
    if len(set(files.values())) != 3 or measured_total != total:
        raise ValueError('Require distinct three-model files and an exact aggregate parameter upper bound')
    manifest_sha = entry.sha(path)
    identity = dict(format='trained_export_C_model_identity_v1', algorithm=submission.ALGORITHM,
        identity='current_training_export:' + manifest_sha, training_manifest_sha256=manifest_sha,
        state_elements_parameter_upper_bound=total,
        models=dict(backend=records['backend'], restorer=records['restorer'], boundary_naf=records['naf_export']),
        original_model_identity_claimed=False, full_cold_start_claimed=False,
        training_manifest_is_identity_not_score_evidence=True)
    return document, identity, files, run_root


def execution_config(path, identity, dataset=None, *, loader=None):
    loader = entry.load_config if loader is None else loader
    config = deepcopy(loader(str(path)))
    source = config['submission']
    if (source.get('semantic_head_delta') is not None
            or any(source.get('models', {}).get(key) is not None for key in ('terminal_semantic', 'local_restorer'))):
        raise ValueError('Training manifest supplies complete backend/shared D5a; extra head/local model is forbidden')
    config['paths']['project_root'] = str(entry.ROOT)
    source.update(restoration_noise=zero.ZERO_MODE,
        models={key: {field: identity['models'][key][field] for field in ('file', 'sha256')}
                for key in ('backend', 'restorer')})
    source['models'].update(terminal_semantic=None, local_restorer=None)
    if dataset is not None:
        source['dataset'] = dataset
    original_validate = entry.validate_config
    with patch.object(entry, 'validate_config', new=lambda value: submission.zero_validate(original_validate, value)):
        zero.validate_baseline(config)
    return config


def safe_output(path, input_dir, run_root):
    output = Path(path).resolve()
    if (not any(output != base and output.is_relative_to(base) for base in (ROOT / 'output', ROOT / 'outputs'))
            or output.exists()):
        raise ValueError('Use a fresh independent project output/ or outputs/ subdirectory')
    for protected in (Path(input_dir).resolve(), run_root):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('Prediction output must not overlap input or source training run')
    return output


def run(args, *, device='cuda'):
    args = argparse.Namespace(**vars(args))
    if getattr(args, 'geometry_mode', 'C') != 'C' or getattr(args, 'backend', 'torch') != 'torch':
        raise ValueError('Training exports currently support the explicit Torch C route only')
    manifest_path = Path(args.training_manifest).resolve(strict=True)
    _, identity, files, run_root = validate_manifest(manifest_path)
    output = safe_output(args.output_dir, args.input_dir, run_root)
    config_path = Path(args.config).resolve(strict=True)
    original_load, original_validate = entry.load_config, entry.validate_config
    original_paths, original_runtime, original_write = entry.model_paths, entry.runtime_record, entry.write
    effective = execution_config(config_path, identity, getattr(args, 'dataset', None), loader=original_load)
    config_sha, effective_sha = entry.sha(config_path), canonical_sha(effective)
    identity_sha, source_sha = canonical_sha(identity), entry.sha(__file__)
    models = dict(backend=files['backend'], restorer=files['restorer'])

    def load(path):
        if Path(path).resolve() != config_path or entry.sha(path) != config_sha:
            raise ValueError('Unexpected or changed inference config')
        return deepcopy(effective)

    def paths(config, model_dir):
        if Path(model_dir).resolve() != run_root or config['submission']['models'] != effective['submission']['models']:
            raise ValueError('Unexpected model root/config for training exports')
        return dict(models)

    def annotate(document):
        value = submission.audit_document(document, identity, identity_sha, config_sha,
            dict(scope='local workspace; no standalone code/model package created'))
        value.pop('portable_config_sha256', None)
        value.update(format=FORMAT, inference_entry='tools/infer_trained.py', no_training=True,
            config_file_sha256=config_sha, effective_config_sha256=effective_sha,
            trained_entry_sha256=source_sha, model_identity_digest_kind='canonical_json_v1',
            training_manifest=dict(path=str(manifest_path), sha256=identity['training_manifest_sha256'],
                                   format=TRAINING_FORMAT), original_model_identity_claimed=False)
        if 'config_role' in value:
            value['config_role'] = 'effective zero-C algorithm with verified training exports; source config may use _base'
        return value

    def write(path, document):
        path = Path(path)
        if path.parent.resolve() == output / 'audit' and path.suffix == '.json':
            document = annotate(document)
        return original_write(path, document)

    args.config, args.output_dir, args.model_dir = str(config_path), str(output), str(run_root)
    args.naf_checkpoint = str(files['naf_export'])
    args.expected_model_sha = identity['models']['boundary_naf']['state_sha256']
    args.global_noise, args.geometry_mode, args.seed_support = 'zero', 'C', 'original-zero'
    args.capture_names = getattr(args, 'capture_names', None) or []
    try:
        with ExitStack() as stack:
            for owner, name, replacement in (
                (naf, 'EXPECTED_MODEL_SHA', args.expected_model_sha),
                (flat, 'BACKEND_SHA', identity['models']['backend']['sha256']),
                (flat, 'RESTORER_SHA', identity['models']['restorer']['sha256']),
                (entry, 'load_config', load), (entry, 'model_paths', paths),
                (entry, 'validate_config', lambda value: submission.zero_validate(original_validate, value)),
                (entry, 'configure_restoration_noise', submission.zero_configure),
                (entry, 'runtime_record', lambda **kw: submission.zero_runtime(original_runtime, **kw)),
                (entry, 'write', write)):
                stack.enter_context(patch.object(owner, name, new=replacement))
            result = compact.run(args, device=device)
        _, after, _, _ = validate_manifest(manifest_path)
        if (after != identity or entry.sha(config_path) != config_sha or entry.sha(__file__) != source_sha
                or canonical_sha(execution_config(config_path, identity, getattr(args, 'dataset', None), loader=original_load)) != effective_sha):
            raise RuntimeError('Training export/config/entry identity changed during inference')
        result = annotate(result)
        original_write(output / 'audit/trained_report.json', result)
        return result
    except Exception as exc:
        audit = output / 'audit'
        if audit.is_dir():
            for path in sorted(audit.glob('*.json')):
                document = _json(path)
                document.update(complete=False, trained_inference_error=repr(exc))
                if path.name == 'status.json': document['status'] = 'failed'
                original_write(path, annotate(document))
            original_write(audit / 'trained_failed.json', annotate(dict(complete=False, error=repr(exc))))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--training-manifest', required=True)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default=str(ROOT / 'config/inference/area_floor_200.yaml'))
    parser.add_argument('--dataset', choices=('quarter_final_100', 'semifinal_150'))
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--capture-names', nargs='+')
    return parser.parse_args(argv)


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(dict(complete=bool(result.get('complete')), format=FORMAT,
        identity=result['model_identity']['identity'], images=len(result['images'])), ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
