# -*- coding: utf-8 -*-
"""旧C全五网络TensorRT与CPU整数读出候选；原入口和模型对象保持不变。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import infer_fast_readout as fast
from utils.tensorrt_runtime import Engine
from utils import tensorrt_runtime

entry, naf = fast.entry, fast.compact.naf
FORMAT = 'C_TensorRT_accelerated_inference_v1'
MANIFEST_FORMAT = 'C_TensorRT_acceleration_v2'
ROLES = ('trunk', 'semantic', 'affinity', 'naf', 'd5a')
FEATURES = ('f0', 'f1', 'f2', 'f3')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate manifest key: ' + key)
        result[key] = value
    return result


def load_manifest(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique,
                       parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    if (not isinstance(value, dict) or value.get('format') != MANIFEST_FORMAT
            or value.get('cpu_integer_readout') is not True or value.get('precision') != 'float32'
            or not isinstance(value.get('trt_flags'), dict)
            or set(value['trt_flags']) != {'TF32', 'FP16', 'INT8'}
            or any(flag is not False for flag in value['trt_flags'].values())):
        raise ValueError('Expected explicit FP32/no-TF32/FP16/INT8 and CPU integer readout manifest')
    if not isinstance(value.get('engines'), dict) or set(value['engines']) != set(ROLES):
        raise ValueError('All five engine roles are required; no partial or fallback mode')
    files = {}
    for role in ROLES:
        row = value['engines'][role]
        if not isinstance(row, dict):
            raise ValueError('Invalid engine record: ' + role)
        name, digest = row.get('file'), row.get('sha256')
        if (not isinstance(name, str) or not name or '\\' in name
                or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
                or '..' in PurePosixPath(name).parts):
            raise ValueError('Engine paths must be portable relative paths without parent traversal')
        target = (path.parent / name).resolve(strict=True)
        if not target.is_relative_to(path.parent) or not target.is_file():
            raise ValueError('Engine file escapes manifest directory or is not a file')
        if (not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None
                or entry.sha(target) != digest):
            raise ValueError('Engine file SHA256 differs: ' + role)
        source_digest = row.get('source_state_sha256')
        if (not isinstance(source_digest, str)
                or re.fullmatch(r'[0-9a-f]{64}', source_digest) is None):
            raise ValueError('Engine source_state_sha256 is required: ' + role)
        files[role] = target
    return value, files


def validate_engine(role, engine):
    inputs, outputs = set(engine.input_names), set(engine.output_names)
    expected_in = dict(trunk={'image'}, semantic=set(FEATURES), affinity=set(FEATURES),
                       naf={'inputs'}, d5a={'xt', 'condition', 't_index'})[role]
    if role == 'semantic' and 'image' in inputs:
        expected_in.add('image')
    required_out = dict(trunk=set(FEATURES), semantic={'semantic_logits'},
                        affinity={'affinity_logits', 'affinity_feature'}, naf={'logits'}, d5a={'x0'})[role]
    allowed_out = required_out | ({'semantic_feature'} if role == 'semantic' else set())
    if inputs != expected_in or not required_out <= outputs or not outputs <= allowed_out:
        raise ValueError('Engine I/O names differ from the selected role: ' + role)
    for name, row in engine.signature.items():
        dtype = 'int64' if role == 'd5a' and name == 't_index' else 'float32'
        if row['dtype'] != dtype:
            raise ValueError('Engine runtime I/O dtype differs: ' + role + '.' + name)


def naf_input_checks(model, inputs):
    """逐项保留BoundaryRGBNAFNet.forward_logits的入口检查，不跑原网络。"""
    if (not torch.is_tensor(inputs) or inputs.ndim != 4
            or inputs.shape[1] != model.input_channels or min(inputs.shape) < 1
            or not inputs.is_floating_point()):
        raise ValueError(f'Expected nonempty floating B{model.input_channels}HW input')
    if not bool(torch.isfinite(inputs).all()) or bool(((inputs < 0) | (inputs > 1)).any()):
        raise ValueError('RGB and boundary inputs must be finite and within [0,1]')
    boundary = inputs[:, model.boundary_channel:model.boundary_channel + 1]
    if bool(((boundary != 0) & (boundary != 1)).any()):
        raise ValueError('Boundary input must be binary T, not continuous affinity or skeleton adapter')
    if inputs.dtype != torch.float32:
        raise TypeError('NAF TensorRT input must be FP32')


def audit_document(document, manifest, manifest_sha, source_sha, helper_sha, engines, attempts, verified_sources):
    value = deepcopy(document)
    value.setdefault('accelerated_base_format', value.get('format'))
    rows = {}
    for role in ROLES:
        engine = engines.get(role)
        rows[role] = dict(manifest['engines'][role], loaded=engine is not None,
            attempts=attempts[role], actual_calls=engine.execution_count if engine is not None else 0,
            verified_source_state_sha256=verified_sources.get(role),
            source_state_verified=role in verified_sources,
            io_signature=deepcopy(engine.signature) if engine is not None else None,
            last_execution=deepcopy(engine.last_execution) if engine is not None else None)
    contract = dict(format=MANIFEST_FORMAT, geometry_mode='C', all_five_engines_required=True,
        engines=rows, engine_manifest_sha256=manifest_sha, precision=manifest['precision'],
        declared_trt_flags=deepcopy(manifest['trt_flags']), runtime_IO_dtypes_verified=set(engines) == set(ROLES),
        engine_flags_source='declared build contract bound to engine file SHA256',
        cpu_integer_readout=True, original_weights_and_module_objects_unchanged=True,
        all_engine_source_states_verified=set(verified_sources) == set(ROLES),
        original_NAF_Torch_sigmoid=True, original_D5a_input_checks_and_finish=True,
        raw_vote_and_final200_unchanged=True, restoration_noise='explicit_zero_v1',
        fallback_calls=0, no_fallback=True, no_training=True)
    value.update(format=FORMAT, training=False, no_training=True, accelerated_runtime=contract,
        accelerated_entry_sha256=source_sha, tensorrt_runtime_sha256=helper_sha)
    # 内层fast/compact的“前向实现未变”只适用于其单独运行；组合入口明确改口。
    for container in (value, value.get('config'), value.get('baseline_config')):
        if not isinstance(container, dict):
            continue
        for key in ('fast_readout', 'compact_readout'):
            if isinstance(container.get(key), dict):
                container[key]['model_forward_unchanged'] = False
                container[key]['model_weights_unchanged'] = True
                container[key]['network_execution_backend'] = 'TensorRT via scoped method adapters'
    if isinstance(value.get('config'), dict):
        value['config']['accelerated_runtime'] = deepcopy(contract)
    return value


def run(args, *, device='cuda'):
    if getattr(args, 'geometry_mode', 'C') != 'C':
        raise ValueError('This all-engine entry only supports original C geometry')
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh accelerated inference output')
    manifest_path = Path(args.engine_manifest).resolve(strict=True)
    manifest, files = load_manifest(manifest_path)
    manifest_sha = entry.sha(manifest_path)
    source_sha, helper_sha = entry.sha(__file__), entry.sha(tensorrt_runtime.__file__)
    original_load, original_naf, original_write = entry.load_models, naf.load_naf_checkpoint, entry.write
    engines, models, verified_sources = {}, {}, {}
    attempts = dict.fromkeys(ROLES, 0)

    def load_engine(role, source_module):
        if role in engines:
            raise RuntimeError('Unexpected repeated engine load: ' + role)
        # 先核原始严格加载模块，再创建引擎和替换方法；文件SHA不能代替此绑定。
        source_digest = entry.tensor_digest(source_module.state_dict().items())
        if source_digest != manifest['engines'][role]['source_state_sha256']:
            raise ValueError('Engine source module state SHA256 differs: ' + role)
        verified_sources[role] = source_digest
        engine = Engine(files[role], expected_sha256=manifest['engines'][role]['sha256'])
        validate_engine(role, engine)
        engines[role] = engine
        return engine

    def invoke(role, **inputs):
        attempts[role] += 1
        return engines[role](**inputs)

    def feature_inputs(features):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError('Expected exactly four ordered encoder feature tensors')
        return dict(zip(FEATURES, features))

    with ExitStack() as stack:
        def load(*values, **options):
            model, restorer, terminal = original_load(*values, **options)
            if models or terminal is not None:
                raise ValueError('Expected one unchanged backend/restorer and no terminal semantic model')
            if (getattr(model, 'semantic_lora', None) is not None
                    or getattr(model, 'geometry_feature_adapter', None) is not None
                    or getattr(model, 'geometry_highres_refiner', None) is not None):
                raise ValueError('All-engine C requires the unchanged shared encoder/two-head architecture')
            models.update(model=model, restorer=restorer)
            source_modules = dict(trunk=model.encoder.trunk, semantic=model.semantic_decoder,
                                  affinity=model.affinity_decoder, d5a=restorer)
            for role in ('trunk', 'semantic', 'affinity', 'd5a'):
                load_engine(role, source_modules[role])

            def trunk(image):
                result = invoke('trunk', image=image)
                return [result[name] for name in FEATURES]

            def semantic(features, image=None, *, return_features=False):
                inputs = feature_inputs(features)
                if 'image' in engines['semantic'].input_names:
                    if image is None:
                        raise ValueError('Semantic engine requires its original RGB input')
                    inputs['image'] = image
                if return_features and 'semantic_feature' not in engines['semantic'].output_names:
                    raise ValueError('Semantic engine lacks the requested semantic_feature output')
                result = invoke('semantic', **inputs)
                return ((result['semantic_logits'], result['semantic_feature'])
                        if return_features else result['semantic_logits'])

            def affinity(features):
                return invoke('affinity', **feature_inputs(features))

            def denoise(xt, condition, t_index):
                restorer._check_inputs(xt, condition, t_index)
                return invoke('d5a', xt=xt, condition=condition, t_index=t_index)['x0']

            for owner, name, replacement in ((model.encoder.trunk, 'forward', trunk),
                    (model.semantic_decoder, 'forward', semantic),
                    (model.affinity_decoder, 'forward', affinity), (restorer, 'denoise', denoise)):
                stack.enter_context(patch.object(owner, name, new=replacement))
            return model, restorer, terminal

        def load_naf(*values, **options):
            model, provenance = original_naf(*values, **options)
            load_engine('naf', model)

            def logits(inputs):
                naf_input_checks(model, inputs)
                return invoke('naf', inputs=inputs)['logits']

            stack.enter_context(patch.object(model, 'forward_logits', new=logits))
            return model, provenance

        def annotate(document):
            return audit_document(document, manifest, manifest_sha, source_sha, helper_sha,
                                  engines, attempts, verified_sources)

        def write(path, document):
            path = Path(path)
            if path.parent.resolve() == output / 'audit' and path.suffix == '.json':
                document = annotate(document)
            return original_write(path, document)

        try:
            for owner, name, replacement in ((entry, 'load_models', load),
                    (naf, 'load_naf_checkpoint', load_naf), (entry, 'write', write)):
                stack.enter_context(patch.object(owner, name, new=replacement))
            result = fast.run(args, device=device)
            n, local = len(result['images']), result['local_zero_calls']
            if result['global_zero_calls'] != n or set(engines) != set(ROLES):
                raise RuntimeError('Expected one global restoration per source and all five engines')
            expected = dict(trunk=2 * n + local, semantic=2 * n, affinity=n + local, d5a=n + local,
                naf=sum((row['inference']['tiles'] + row['inference']['batch_size'] - 1)
                        // row['inference']['batch_size'] for row in result['images']))
            if any(engines[role].execution_count != expected[role]
                   or attempts[role] != expected[role] or expected[role] < 1 for role in ROLES):
                raise RuntimeError('Actual engine calls differ from unchanged C global/local/tile paths')
            final_manifest, _ = load_manifest(manifest_path)
            if (final_manifest != manifest or entry.sha(manifest_path) != manifest_sha
                    or entry.sha(__file__) != source_sha or entry.sha(tensorrt_runtime.__file__) != helper_sha):
                raise RuntimeError('Accelerated manifest/source/engine identity changed during inference')
            result = annotate(result)
            original_write(output / 'audit/accelerated_report.json', result)
            return result
        except Exception as exc:
            audit = output / 'audit'
            if audit.is_dir():
                for path in sorted(audit.glob('*.json')):
                    document = json.loads(path.read_text(encoding='utf-8'))
                    document.update(complete=False, accelerated_error=repr(exc))
                    if path.name == 'status.json':
                        document['status'] = 'failed'
                    original_write(path, annotate(document))
                original_write(audit / 'accelerated_failed.json', annotate(dict(complete=False, error=repr(exc))))
            raise


def parse_args(argv=None):
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
    parser.add_argument('--geometry-mode', choices=('C',), default='C')
    parser.add_argument('--engine-manifest', required=True)
    return parser.parse_args(argv)


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(dict(complete=bool(result.get('complete')), format=FORMAT,
        images=len(result['images']), engines=result['accelerated_runtime']['engines']),
        ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
