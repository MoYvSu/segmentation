# -*- coding: utf-8 -*-
"""F0：WS语义底图固定128；旧raw末端票、local零噪N1/NAF/S0/final200保持。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import infer_raw_shared as raw_entry

entry, naf_entry, zero_entry = raw_entry.entry, raw_entry.naf_entry, raw_entry.zero_entry
FORMAT = 'flat_semantic_F0_inference_v1'
GEOMETRY_ROLE = 'F0_zero_logits_constant128_WS_base_with_original_boundary_overlay'
VOTE_ROLE = 'unchanged_original_13b3_raw_global_probability'
COUNT_KEYS = raw_entry.COUNT_KEYS
BACKEND_SHA = '13b3ab1bc509fa45848e2eb8a40ee458b178fe747fc83eb65333c527a8161f04'
RESTORER_SHA = '9315da872ead66735f799e27f85eb97f8c2c265d44d64c6f8a271be50417007b'


def contract():
    value = raw_entry.contract()
    value.update(format=FORMAT, candidate='F0', geometry_semantic_role=GEOMETRY_ROLE,
        vote_probability_role=VOTE_ROLE, geometry_semantic_logit_constant=0.,
        WS_semantic_base_uint8=128, WS_original_NAF_barrier_overlay_unchanged=True,
        raw_semantic_shared_with_geometry=False, global_restoration='not_computed',
        vote='unchanged CPU sigmoid(raw whole native logits); instance probability_mean strictly >.5',
        legacy_capture_roles=dict(semantic='F0_all_zero_native_B1HW_geometry_logits',
                                  raw_probability='unchanged_original_raw_probability_HW'))
    return value


def execution_config(baseline):
    zero_entry.validate_baseline(baseline)
    if baseline['submission'].get('semantic_head_delta') is not None:
        raise ValueError('F0 uses the original 13b3 head; trained deltas are forbidden')
    for name, expected in (('backend',BACKEND_SHA),('restorer',RESTORER_SHA)):
        if baseline['submission']['models'][name]['sha256'] != expected:
            raise ValueError('F0 requires the original 13b3 backend and 9315 D5a')
    cfg = deepcopy(baseline)
    cfg['submission']['restoration_noise'] = zero_entry.ZERO_MODE
    cfg['submission']['global_restoration'] = dict(mode='not_computed')
    cfg['submission']['semantic_route'] = 'F0_flat_WS_base_original_raw_vote'
    cfg['flat_semantic'] = contract()
    return cfg


def flat_geometry(semantic):
    if (not torch.is_tensor(semantic) or semantic.dtype != torch.float32 or semantic.ndim != 4
            or tuple(semantic.shape[:2]) != (1,1)):
        raise ValueError('F0 requires native FP32 B1HW geometry logits')
    return torch.zeros_like(semantic)


def expected_WS_image(after, config):
    stages = naf_entry.marker_stages(after, config)
    image = np.full((*after.shape,3),128,np.uint8)
    overlay = image.copy()
    overlay[stages['barrier_belt'] > 0] = 255
    return cv2.addWeighted(image,.7,overlay,.3,0), stages


def flat_supported_readout(semantic, after, reference, config, *, original_readout=None):
    """在现有唯一WS后核实际输入SHA；本函数也供私有CPU桥接调用。"""
    if torch.count_nonzero(semantic) != 0 or semantic.dtype != torch.float32:
        raise ValueError('F0 readout must receive exact zero geometry logits')
    expected, stages = expected_WS_image(after, config)
    readout = naf_entry._supported_readout if original_readout is None else original_readout
    result, audit = readout(semantic, after, reference, config)
    capture = audit['readout_audit']
    digest = naf_entry.array_sha(expected)
    if (audit['readout_hashes']['elevation_sha256'] != digest
            or capture.get('actual_watershed_calls') != 1 or not capture.get('original_elevation_exact')):
        raise RuntimeError('Actual WS input differs from constant128 plus original NAF barrier')
    proof = dict(candidate='F0', semantic_base_uint8=128, geometry_semantic_logit_constant=0.,
        geometry_semantic_logits_zero=True, expected_WS_elevation_sha256=digest,
        actual_WS_elevation_sha256=audit['readout_hashes']['elevation_sha256'],
        actual_WS_elevation_exact=True, actual_WS_calls=1,
        original_barrier_belt_sha256=naf_entry.array_sha(stages['barrier_belt']),
        barrier_overlay='cv2.addWeighted(base128,.7,base128_with_original_barrier255,.3,0)')
    audit['flat_semantic_proof'] = proof
    return result, audit


def flat_predict(records):
    """沿已验证的raw-only全图路径；在NAF实际读出前仅替换几何语义。"""
    upstream_rows = []
    original_predict = raw_entry.raw_shared_predict(upstream_rows)

    def predict(model, restorer, terminal, path, config, device, **kwargs):
        actual_decode = entry.decode_reflect  # 正式运行时已是NAF/S0包装，必须在其前替换。
        original_supported = naf_entry._supported_readout
        observed = dict(calls=0, supported_calls=0)

        def supported(semantic, after, reference, cfg):
            observed['supported_calls'] += 1
            result, audit = flat_supported_readout(semantic,after,reference,cfg,original_readout=original_supported)
            observed['flat_WS_audit'] = deepcopy(audit['flat_semantic_proof'])
            return result, audit

        def decode(semantic, boundary, cfg, **options):
            if observed['calls']:
                raise RuntimeError('Expected exactly one F0 geometry substitution per source')
            observed['calls'] += 1
            flat = flat_geometry(semantic)
            observed.update(geometry_semantic_sha256=naf_entry.array_sha(flat.cpu().contiguous().numpy()),
                            raw_semantic_logits_sha256=naf_entry.array_sha(semantic.cpu().contiguous().numpy()))
            return actual_decode(flat, boundary, cfg, **options)

        if upstream_rows:
            raise RuntimeError('Unconsumed upstream audit rows')
        with patch.object(raw_entry, 'execution_config', new=execution_config), \
                patch.object(entry, 'decode_reflect', new=decode), \
                patch.object(naf_entry, '_supported_readout', new=supported):
            rgb, result = original_predict(model, restorer, terminal, path, config, device, **kwargs)
        if observed['calls'] != 1 or observed['supported_calls'] != 1 or len(upstream_rows) != 1:
            raise RuntimeError('F0 must execute exactly one flat geometry readout')
        row = upstream_rows.pop()
        if row['raw_semantic_logits_sha256'] != observed['raw_semantic_logits_sha256']:
            raise RuntimeError('Raw vote semantic provenance differs from actual native logits')
        row.update(candidate='F0', geometry_semantic_sha256=observed['geometry_semantic_sha256'],
                   geometry_semantic_role=GEOMETRY_ROLE, vote_probability_role=VOTE_ROLE,
                   geometry_semantic_logit_constant=0., WS_semantic_base_uint8=128,
                   flat_WS_audit=observed['flat_WS_audit'],
                   global_restored_semantic_computed=False)
        audit = kwargs.get('fixed_audit')
        if audit is not None:
            audit.update(geometry_semantic_sha256=row['geometry_semantic_sha256'],
                         geometry_semantic_role=GEOMETRY_ROLE, vote_probability_role=VOTE_ROLE,
                         raw_semantic_logits_sha256=row['raw_semantic_logits_sha256'],
                         global_restored_semantic_computed=False)
        records.append(row)
        return rgb, result
    return predict


def audit_document(document, records, source_sha):
    value = deepcopy(document)
    value['legacy_format'] = value.get('legacy_format', value.get('format'))
    value.update(format=FORMAT, flat_semantic=contract(), flat_semantic_images=deepcopy(records),
                 flat_semantic_entry_sha256=source_sha, semantic=VOTE_ROLE, training=False)
    value['actual_forward_counts'] = {key:sum(row[key] for row in records) for key in COUNT_KEYS}
    if 'config' in value:
        base = deepcopy(value.get('baseline_config', value['config']))
        value['baseline_config'], value['config'] = base, execution_config(base)
        value['config_role'] = 'effective F0 flat geometry with original raw vote; baseline_config validates base assets'
    if 'runtime' in value:
        value['baseline_runtime'] = deepcopy(value.get('baseline_runtime', value['runtime']))
        value['runtime']['restoration_noise'] = dict(name=zero_entry.ZERO_MODE,
            global_restoration='not_computed', local_noise='zero', random_noise_generated=False)
    if 'geometry' in value:
        value['geometry'] = 'F0 constant128 WS semantic base plus original NAF barrier; unchanged N1/20a59/S0; original raw vote then final200 once'
    for image in value.get('images', []):
        for record in (image, image.get('fixed_paths', {}), image.get('boundary_naf', {})):
            if 'raw_probability_sha256' in record:
                record['raw_probability_role'] = VOTE_ROLE
            record['geometry_semantic_role'] = GEOMETRY_ROLE
    return value


def run(args, *, device='cuda'):
    args = argparse.Namespace(**vars(args))
    if getattr(args, 'semantic_checkpoint', None):
        raise ValueError('F0 must use the original 13b3 semantic head')
    if (getattr(args, 'global_noise', 'zero') != 'zero'
            or getattr(args, 'expected_model_sha', naf_entry.EXPECTED_MODEL_SHA) != naf_entry.EXPECTED_MODEL_SHA
            or getattr(args, 'seed_support', 'original-zero') != 'original-zero'):
        raise ValueError('F0 fixes local zero noise, old20a59 NAF and original-zero S0')
    args.expected_model_sha, args.seed_support = naf_entry.EXPECTED_MODEL_SHA, 'original-zero'
    baseline = entry.load_config(str(Path(args.config).resolve()))
    if getattr(args, 'dataset', None):
        baseline['submission']['dataset'] = args.dataset
    execution_config(baseline)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh F0 output')
    source_sha, original_write, records = entry.sha(__file__), entry.write, []

    def write(path, document):
        path = Path(path)
        if path.parent == output/'audit' and path.name in ('report.json','status.json','naf_report.json','naf_failed.json'):
            document = audit_document(document, records, source_sha)
        return original_write(path, document)

    try:
        with patch.object(naf_entry, 'patch', new=raw_entry.PlainPatch), \
                patch.object(entry, 'predict', new=flat_predict(records)), patch.object(entry, 'write', new=write):
            receipt = naf_entry.run(args, device=device)
        if entry.sha(__file__) != source_sha or [row['source'] for row in records] != [row['source'] for row in receipt['images']]:
            raise RuntimeError('F0 source or invocation order changed')
        for row, actual in zip(records, receipt['images']):
            if (row['geometry_semantic_sha256'] != actual['semantic_sha256']
                    or row['raw_shared_probability_sha256'] != actual['raw_probability_sha256']
                    or row['input_boundary_sha256'] != actual['input_boundary_sha256']):
                raise RuntimeError('Actual F0 geometry/raw vote/NAF provenance differs')
        result = audit_document(receipt, records, source_sha)
        original_write(output/'audit/flat_semantic_report.json', result)
        return result
    except Exception as exc:
        if (output/'audit').is_dir():
            original_write(output/'audit/flat_semantic_failed.json', dict(complete=False,format=FORMAT,
                error=repr(exc),flat_semantic_images=records))
            for name in ('report.json','status.json','naf_report.json'):
                path = output/'audit'/name
                if path.is_file():
                    value = json.loads(path.read_text(encoding='utf-8'))
                    value.update(complete=False,flat_semantic_error=repr(exc))
                    if name == 'status.json':
                        value['status'] = 'failed'
                    original_write(path, value)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config', default=str(ROOT/'config/inference/area_floor_200.yaml'))
    parser.add_argument('--model-dir')
    parser.add_argument('--dataset', choices=('quarter_final_100','semifinal_150'))
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--naf-checkpoint', required=True)
    parser.add_argument('--capture-names', nargs='+')
    parser.add_argument('--global-noise', choices=('zero',), default='zero')
    result = run(parser.parse_args(argv))
    print(json.dumps(dict(complete=True,format=FORMAT,images=len(result['images']),
                         actual_forward_counts=result['actual_forward_counts'])),flush=True)


if __name__ == '__main__':
    main()
