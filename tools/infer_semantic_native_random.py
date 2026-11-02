# -*- coding: utf-8 -*-
"""正式native-random语义增量部署；复用统一1024零噪双头推理，不进行训练。"""
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

from tools import infer_unified as unified
from tools import train_semantic_native_random as random_trainer
trainer = random_trainer.base

entry = unified.entry
FORMAT = 'trained_semantic_native_random_inference_v1'
SEMANTIC_ROLE = 'random_retrained_native1024_zeroD5a_local_logits_tent_then_CPU_sigmoid'
EXPECTED_HEAD_PARAMETERS = trainer.HEAD_PARAMETERS
AUDIT_FILES = frozenset(('report.json', 'status.json', 'naf_report.json', 'naf_failed.json',
                         'unified_report.json', 'unified_failed.json'))


def inspect_checkpoint(path):
    """正式入口拒绝初态、smoke和未完成的接续训练增量。"""
    path = Path(path).resolve()
    checkpoint_sha = trainer.file_sha(path)
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle.get('format') != random_trainer.FORMAT:
        raise ValueError('Expected semantic_native_random_delta_v1')
    identity, progress = bundle['identity'], bundle['progress']
    if (random_trainer.normalize_identity(deepcopy(identity)) != identity
            or identity.get('semantic_initialization') != 'random_full_v1'
            or identity.get('initialization_seed') != 1101):
        raise ValueError('Explicit original/random semantic initialization identity is required')
    initial = identity['init_receipt']
    if (identity['base_semantic_sha256'] != identity['base_original_semantic_sha256']
            or identity['random_initial_semantic_sha256'] == identity['base_original_semantic_sha256']
            or initial.get('previous_trained_delta_loaded') is not False
            or initial.get('torch_rng_preserved') is not True
            or initial['nonsemantic_before_sha256'] != initial['nonsemantic_after_sha256']
            or initial['nonsemantic_after_sha256'] != identity['frozen']['frozen_backend_sha256']):
        raise ValueError('Random delta must preserve the original frozen base and distinguish its random initial head')
    if (initial.get('initialization') != 'random_full_v1' or initial.get('initialization_seed') != 1101
            or initial.get('parameter_count') != EXPECTED_HEAD_PARAMETERS):
        raise ValueError('Random initialization receipt differs from the complete-head recipe')
    if (identity.get('base_backend_file_sha256') != trainer.BACKEND_SHA
            or identity.get('base_restorer_file_sha256') != trainer.RESTORER_SHA):
        raise ValueError('Semantic delta must bind the fixed 13b3 backend and 9315 D5a')
    if (identity.get('total_updates') != 2560 or progress.get('completed_updates') != 2560
            or progress.get('next_epoch') != 21 or progress.get('next_index') != 0):
        raise ValueError('Formal inference requires all 2560 updates; initial/smoke is forbidden')
    if (any(progress.get(key) != 2560 for key in (
            'training_zero_calls', 'training_encoder_calls', 'training_semantic_calls'))
            or progress.get('affinity_calls') != 0):
        raise ValueError('Training forward-count contract differs')
    recipe = identity['recipe']
    if any(recipe.get(key) != expected for key, expected in random_trainer.FIXED.items()):
        raise ValueError('Semantic delta differs from the fixed native-random training recipe')
    if (identity['frozen'].get('trainable_parameters') != EXPECTED_HEAD_PARAMETERS
            or sum(value.numel() for value in bundle['semantic_decoder'].values()) != EXPECTED_HEAD_PARAMETERS):
        raise ValueError('Expected the complete 6302010-parameter semantic head')
    head_sha = trainer.tensor_sha(bundle['semantic_decoder'].items())
    if head_sha != bundle.get('semantic_sha256') or trainer.file_sha(path) != checkpoint_sha:
        raise ValueError('Semantic delta file/tensor seal differs')
    return dict(file=str(path), sha256=checkpoint_sha, checkpoint_format=random_trainer.FORMAT,
                completed_updates=2560, epoch=20, head_sha256=head_sha,
                head_parameters=EXPECTED_HEAD_PARAMETERS,
                base_backend_file_sha256=trainer.BACKEND_SHA,
                base_restorer_file_sha256=trainer.RESTORER_SHA,
                base_semantic_sha256=identity['base_semantic_sha256'],
                base_original_semantic_sha256=identity['base_original_semantic_sha256'],
                random_initial_semantic_sha256=identity['random_initial_semantic_sha256'],
                initialization='random_full_v1', initialization_seed=1101,
                semantic_architecture=deepcopy(identity['semantic_architecture']),
                training_identity_sha256=trainer.json_sha(identity),
                training_data_source_seal_sha256=identity.get('data', {}).get('source_seal_sha256'),
                checkpoint_role='trained_semantic_head_delta_on_immutable_backend',
                optimizer_or_rng_restored=False, training_performed_in_this_run=False)


def delta_loader(original, checkpoint, state):
    """沿原三元组模型接口，只替换已核验基座中的语义头。"""
    def load(config, paths, device):
        if state.get('loads', 0):
            raise RuntimeError('Expected exactly one semantic-delta model load')
        if (trainer.file_sha(paths['backend']) != trainer.BACKEND_SHA
                or trainer.file_sha(paths['restorer']) != trainer.RESTORER_SHA):
            raise ValueError('Actual immutable base files differ from delta identity')
        if trainer.file_sha(checkpoint['file']) != checkpoint['sha256']:
            raise ValueError('Semantic delta changed before model loading')
        model, restorer, terminal = original(config, paths, device)
        if terminal is not None:
            raise ValueError('Trained native-random inference has one semantic head, no terminal model')
        unified.check_model(model)
        if (sum(p.numel() for p in model.semantic_decoder.parameters()) != EXPECTED_HEAD_PARAMETERS
                or trainer.tensor_sha(model.semantic_decoder.state_dict().items()) != checkpoint['base_semantic_sha256']):
            raise ValueError('Loaded immutable base semantic head differs')
        receipt = random_trainer.load_head_delta(model, checkpoint['file'])
        if (receipt['completed_updates'] != 2560 or receipt['checkpoint_file_sha256'] != checkpoint['sha256']
                or receipt['loaded_semantic_sha256'] != checkpoint['head_sha256']
                or receipt['architecture'] != checkpoint['semantic_architecture']
                or receipt['nonsemantic_before_sha256'] != receipt['nonsemantic_after_sha256']):
            raise RuntimeError('Strict semantic-delta load receipt differs')
        if (receipt['initialization'] != checkpoint['initialization']
                or receipt['initialization_seed'] != checkpoint['initialization_seed']
                or receipt['base_original_semantic_sha256'] != checkpoint['base_original_semantic_sha256']
                or receipt['random_initial_semantic_sha256'] != checkpoint['random_initial_semantic_sha256']):
            raise RuntimeError('Random/original initialization identities differ after strict loading')
        model.eval().requires_grad_(False)
        restorer.eval().requires_grad_(False)
        state.update(loads=1, receipt=receipt, model=model)
        return model, restorer, None
    return load


def audit_document(document, checkpoint, state, source_sha):
    value = deepcopy(document)
    value.setdefault('base_inference_format', value.get('format'))
    value.update(format=FORMAT, training=False, optimizer_steps=0, no_optimizer_steps=True,
                 semantic=SEMANTIC_ROLE, semantic_delta=deepcopy(checkpoint),
                 semantic_delta_loaded=state.get('loads', 0) == 1,
                 semantic_delta_load_receipt=deepcopy(state.get('receipt')),
                 trained_semantic_entry_sha256=source_sha,
                 effective_backend_identity='immutable_13b3_base_plus_trained_semantic_delta')
    value['semantic_delta']['inference_training'] = False
    if 'unified_local1024' in value:
        value['unified_local1024'].update(geometry_semantic_role=SEMANTIC_ROLE,
            vote_probability_role=SEMANTIC_ROLE, semantic_head_sha256=checkpoint['head_sha256'],
            semantic_checkpoint_sha256=checkpoint['sha256'], training=False)
    for field in ('images', 'unified_images'):
        for row in value.get(field, []):
            row.update(semantic_head_sha256=checkpoint['head_sha256'], semantic_role=SEMANTIC_ROLE)
            for record in (row, row.get('fixed_paths', {}), row.get('boundary_naf', {})):
                if 'raw_probability_sha256' in record:
                    record['raw_probability_role'] = SEMANTIC_ROLE
                for key in ('vote_probability_role', 'geometry_semantic_role'):
                    if key in record:
                        record[key] = SEMANTIC_ROLE
    if 'config' in value:
        value['config'].setdefault('submission', {})['semantic_head_delta'] = deepcopy(checkpoint)
        value['config_role'] = 'effective immutable base plus trained semantic delta; baseline_config validates base assets only'
    if 'models' in value:
        value['models']['semantic_head_delta'] = deepcopy(checkpoint)
        if 'backend' in value['models']:
            value['models']['backend']['role'] = 'immutable_base_file; effective_semantic_replaced_by_delta'
    return value


def run(args, *, device='cuda'):
    args = argparse.Namespace(**vars(args))
    if not getattr(args, 'semantic_checkpoint', None):
        raise ValueError('--semantic-checkpoint is required')
    checkpoint = inspect_checkpoint(args.semantic_checkpoint)
    output = Path(args.output_dir).resolve()
    source_sha = entry.sha(__file__)
    original_load, original_write = entry.load_models, entry.write
    state = dict(loads=0)

    def write(path, document):
        path = Path(path)
        if path.parent == output/'audit' and path.name in AUDIT_FILES:
            document = audit_document(document, checkpoint, state, source_sha)
        return original_write(path, document)

    try:
        with patch.object(entry, 'load_models', new=delta_loader(original_load, checkpoint, state)), \
                patch.object(entry, 'write', new=write):
            result = unified.run(args, device=device)
        if state['loads'] != 1:
            raise RuntimeError('Inference did not actually load the trained semantic delta')
        if (entry.sha(__file__) != source_sha or trainer.file_sha(checkpoint['file']) != checkpoint['sha256']
                or trainer.tensor_sha(state['model'].semantic_decoder.state_dict().items()) != checkpoint['head_sha256']
                or trainer.frozen_sha(state['model']) != state['receipt']['nonsemantic_after_sha256']):
            raise RuntimeError('Trained semantic source/checkpoint/model changed during inference')
        counts = result['actual_forward_counts']
        if (counts['final200_calls'] != len(result['images'])
                or any(counts[key] != 0 for key in ('global_zero_calls', 'global_encoder_calls', 'raw_encoder_calls'))):
            raise RuntimeError('Unified single-readout/no-whole-image contract differs')
        result = audit_document(result, checkpoint, state, source_sha)
        original_write(output/'audit/semantic_native_random_report.json', result)
        return result
    except Exception as exc:
        if (output/'audit').is_dir():
            original_write(output/'audit/semantic_native_random_failed.json',
                audit_document(dict(complete=False, error=repr(exc)), checkpoint, state, source_sha))
            for name in ('report.json', 'status.json', 'naf_report.json', 'unified_report.json'):
                path = output/'audit'/name
                if path.is_file():
                    document = json.loads(path.read_text(encoding='utf-8'))
                    document.update(complete=False, trained_semantic_error=repr(exc))
                    if name == 'status.json':
                        document['status'] = 'failed'
                    original_write(path, audit_document(document, checkpoint, state, source_sha))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default=str(ROOT/'config/inference/area_floor_200.yaml'))
    parser.add_argument('--model-dir')
    parser.add_argument('--dataset', choices=('quarter_final_100', 'semifinal_150'))
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--naf-checkpoint', required=True)
    parser.add_argument('--semantic-checkpoint', required=True)
    parser.add_argument('--capture-names', nargs='+')
    parser.add_argument('--global-noise', choices=('zero',), default='zero')
    return parser.parse_args(argv)


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(dict(complete=True, format=FORMAT, images=len(result['images']),
                         semantic_checkpoint_sha256=result['semantic_delta']['sha256'],
                         actual_forward_counts=result['actual_forward_counts'])), flush=True)


if __name__ == '__main__':
    main()
