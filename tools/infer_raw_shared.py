# -*- coding: utf-8 -*-
"""原图全图语义一次同时供WS和实例票；无全图恢复，局部零噪边界保持。"""
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
from tools import infer_zero as zero_entry
from tools.affinity_connectivity_views import _inference_state
from tools.probe_affinity_patch import boundary_grid, tensor_from_rgb
from utils import restoration_noise as noise_module
from utils import restoration_probe
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.patch_diagnostic import BlendMap, phase_description, tile_boxes

FORMAT = 'raw_shared_semantic_inference_v1'
PROBABILITY_ROLE = 'raw_shared_probability'
COUNT_KEYS = ('encoder_calls', 'global_encoder_calls', 'local_encoder_calls', 'raw_encoder_calls',
              'global_zero_calls', 'local_zero_calls', 'semantic_head_calls', 'affinity_head_calls',
              'global_model_calls', 'final200_calls')


class PlainPatch:
    """计数由旧NAF运行器显式维护；替换函数不积存数组调用历史。"""
    @staticmethod
    def object(target, name, *, side_effect):
        return patch.object(target, name, new=side_effect)


def contract():
    return dict(format=FORMAT, native_sizes=[1024], network_input_size=1024, overlap=.25,
        restoration_noise=zero_entry.ZERO_MODE, global_restoration='not_computed', local_restoration='zero',
        geometry_semantic_role='raw_original_logits_for_geometry_and_vote', vote_probability_role=PROBABILITY_ROLE,
        raw_encoder_calls=1, raw_image_semantic_forward=True, raw_semantic_shared_with_geometry=True,
        global_affinity_computed=False, global_affinity_used_for_geometry=False,
        local_affinity_blend='original BlendMap probabilities and order',
        vote='CPU sigmoid(raw whole native logits), shared WS and instance probability_mean strictly >.5',
        naf_model_sha256=naf_entry.EXPECTED_MODEL_SHA, seed_support='original-zero',
        final_instance_area_policy=deepcopy(zero_entry.FINAL_POLICY), final200_calls_per_source=1,
        legacy_capture_roles=dict(semantic='raw_original_shared_semantic_logits_B1HW',
                                  raw_probability='raw_original_shared_semantic_probability_HW'),
        random_noise_generated=False, training=False)


def execution_config(baseline):
    zero_entry.validate_baseline(baseline)
    cfg = deepcopy(baseline)
    cfg['submission']['restoration_noise'] = zero_entry.ZERO_MODE
    cfg['submission']['global_restoration'] = dict(mode='not_computed')
    cfg['submission']['semantic_route'] = PROBABILITY_ROLE
    cfg['raw_shared'] = contract()
    return cfg


def raw_shared_predict(records):
    original_probe = restoration_probe.restore_probe

    @torch.no_grad()
    def predict(model, restorer, terminal, path, config, device, *, local_restorer=None,
                fixed_audit=None, local_audit=None, analysis_callback=None):
        if terminal is not None or local_restorer is not None:
            raise ValueError('Raw shared keeps the original global/local models')
        if any(getattr(model, key, None) is not None for key in (
                'semantic_lora', 'geometry_feature_adapter', 'geometry_highres_refiner')):
            raise ValueError('Raw shared requires the verified unadapted shared backend')
        cfg = execution_config(config)
        counts, phase = {key: 0 for key in COUNT_KEYS}, dict(name='raw')
        proofs = []
        window_rows = [] if local_audit is None else local_audit

        def counted(key):
            def hook(module, args, output):
                counts[key] += 1
                if key == 'encoder_calls':
                    counts[phase['name'] + '_encoder_calls'] += 1
            return hook

        def restore(image, seed, branch, audit=None):
            if branch != 'local':
                raise RuntimeError('Global restoration is forbidden in raw-shared inference')
            proof = {}
            result, _ = original_probe(restorer, image, seed, 'zero', audit=proof)
            if (proof.get('noise_nonzero') != 0 or proof.get('initial_equals_condition') is not True
                    or proof.get('q_sample_calls') != 1 or proof.get('initial_max_abs') != 0.0):
                raise RuntimeError('D5a did not receive the explicit zero-noise condition')
            counts[branch + '_zero_calls'] += 1
            proofs.append(dict(branch=branch, **proof))
            if audit is not None:
                audit.update(proof)
            return result

        with ExitStack() as stack:
            stack.enter_context(_inference_state(model, restorer, device))
            for key, module in (('encoder_calls', model.encoder), ('semantic_head_calls', model.semantic_decoder),
                                ('affinity_head_calls', model.affinity_decoder), ('global_model_calls', model)):
                handle = module.register_forward_hook(counted(key))
                stack.callback(handle.remove)
            stack.enter_context(patch.object(backend_entry, 'restore_first', new=zero_entry.forbidden_noise))
            stack.enter_context(patch.object(backend_adaptation, 'restore_first', new=zero_entry.forbidden_noise))
            for module, name in ((torch, 'randn'), (torch, 'randn_like'), (torch, 'normal'),
                    (backend_adaptation, 'restoration_noise'), (noise_module, 'restoration_noise'),
                    (restoration_probe, 'restoration_noise'), (entry.views, 'restore_first')):
                stack.enter_context(patch.object(module, name, new=zero_entry.forbidden_noise))
            seed = entry.source_seed(path.stem, cfg['submission'].get('dataset'))
            # 原图全图只读一次encoder/语义头；不调用全图恢复、model.forward或affinity头。
            rgb, image, ph, pw = prepare_image(path, 1024, device)
            raw = model.semantic_decoder(model.encoder(image.float()), image.float())
            if isinstance(raw, dict):
                raw = raw['semantic_logits']
            semantic = crop_letterbox_output(raw.float(), 1024, ph, pw, rgb.shape[:2]).cpu()
            if semantic.ndim != 4 or tuple(semantic.shape[:2]) != (1, 1) or semantic.dtype != torch.float32:
                raise RuntimeError('Expected native FP32 B1HW raw semantic logits')
            # 与旧C的raw票概率相同的CPU sigmoid，现在同图也供WS几何。
            probability = semantic.sigmoid()[0, 0].numpy()
            semantic_sha = naf_entry.array_sha(semantic.contiguous().numpy())
            probability_sha = naf_entry.array_sha(probability)
            semantic_mask_sha = naf_entry.array_sha(probability > .5)
            if fixed_audit is not None:
                fixed_audit.update(geometry_semantic_sha256=semantic_sha,
                    raw_probability_sha256=probability_sha, vote_probability_role=PROBABILITY_ROLE,
                    global_boundary_computed=False)
            phase['name'] = 'local'
            blend = BlendMap(rgb.shape[:2])
            boxes = tile_boxes(rgb.shape[:2], 1024, .25)
            for index, box in enumerate(boxes):
                y, x, y1, x1 = box
                image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], device)
                local_seed = seed + 100000 + 1024 * 1000 + index
                proof = {}
                restored = restore(image, local_seed, 'local', proof)
                grid = boundary_grid(model, restored, cfg)
                native = crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
                blend.add(box, native)
                window_rows.append(dict(size=1024, window=index, box=list(box), seed=local_seed,
                    noise_mode='zero', **proof, input_sha256=naf_entry.array_sha(image.cpu().contiguous().numpy()),
                    restored_sha256=naf_entry.array_sha(restored.cpu().contiguous().numpy())))
                if analysis_callback is not None:
                    analysis_callback(source_shape=rgb.shape[:2], box=box, image=image, restored=restored,
                                      boundary=native, ph=ph, pw=pw, size=1024, window=index)
            expected = dict(encoder_calls=1+len(boxes), global_encoder_calls=0, local_encoder_calls=len(boxes),
                raw_encoder_calls=1, global_zero_calls=0, local_zero_calls=len(boxes), semantic_head_calls=1,
                affinity_head_calls=len(boxes), global_model_calls=0, final200_calls=0)
            if counts != expected:
                raise RuntimeError('Raw-shared forward counts differ from global plus local-only contract')
            boundary, blend_audit = blend.finish()
            blend_audit['tiles'] = len(boxes)
            ids, _ = entry.decode_reflect(semantic, torch.from_numpy(boundary)[None, None], cfg,
                                         original_decode=entry.views.decode_native)
            classes, _ = entry.views.revote_instances(ids, probability)
            result = dict(instances=ids, classes=classes, boundary=boundary,
                          phases=phase_description(ids, classes), blend=blend_audit)
            result = entry.finalize_instance_area(result, cfg)
            counts['final200_calls'] += 1
        entry.verify_prediction(result['instances'], result['classes'], rgb.shape[:2])
        if naf_entry.array_sha(semantic.contiguous().numpy()) != semantic_sha or naf_entry.array_sha(probability) != probability_sha:
            raise RuntimeError('Readout mutated the shared raw semantic logits/probability')
        records.append(dict(source=path.stem, windows=len(boxes), **counts,
            raw_semantic_logits_sha256=semantic_sha, raw_shared_probability_sha256=probability_sha,
            raw_semantic_mask_sha256=semantic_mask_sha,
            raw_semantic_mask_role='native HW bool (raw shared probability > .5); same bytes as uint8 0/1',
            input_boundary_sha256=naf_entry.array_sha(boundary), zero_noise_calls=proofs,
            vote_probability_role=PROBABILITY_ROLE, geometry_semantic_role='raw_original_logits_for_geometry_and_vote'))
        return rgb, result
    return predict


def audit_document(document, records, source_sha):
    value = deepcopy(document)
    value['legacy_format'] = value.get('legacy_format', value.get('format'))
    value.update(format=FORMAT, raw_shared=contract(), raw_shared_images=deepcopy(records),
                 raw_shared_entry_sha256=source_sha, semantic=PROBABILITY_ROLE)
    value['actual_forward_counts'] = {key: sum(row[key] for row in records) for key in COUNT_KEYS}
    if 'config' in value:
        base = deepcopy(value.get('baseline_config', value['config']))
        value['baseline_config'], value['config'] = base, execution_config(base)
        value['config_role'] = 'effective raw-shared configuration; baseline_config is validation identity only'
    if 'runtime' in value:
        value['baseline_runtime'] = deepcopy(value.get('baseline_runtime', value['runtime']))
        value['runtime']['restoration_noise'] = dict(name=zero_entry.ZERO_MODE, global_restoration='not_computed', local_noise='zero',
                                                     random_noise_generated=False)
    if 'geometry' in value:
        value['geometry'] = 'raw whole semantic for WS and instance vote; unchanged local N1/20a59/S0; final200 once'
    for image in value.get('images', []):
        record = image.get('boundary_naf', image)
        if 'raw_probability_sha256' in record:
            record['raw_probability_role'] = PROBABILITY_ROLE
            record['geometry_semantic_role'] = 'raw_original_logits_for_geometry_and_vote'
    return value


def run(args, *, device='cuda'):
    args = argparse.Namespace(**vars(args))
    if getattr(args, 'global_noise', 'zero') != 'zero':
        raise ValueError('Raw-shared inference requires local zero noise and no global restoration')
    if getattr(args, 'expected_model_sha', naf_entry.EXPECTED_MODEL_SHA) != naf_entry.EXPECTED_MODEL_SHA:
        raise ValueError('Raw shared is fixed to old 20a59 NAF')
    if getattr(args, 'seed_support', 'original-zero') != 'original-zero':
        raise ValueError('Raw shared requires original-zero S0')
    args.expected_model_sha, args.seed_support = naf_entry.EXPECTED_MODEL_SHA, 'original-zero'
    baseline = entry.load_config(str(Path(args.config).resolve()))
    if getattr(args, 'dataset', None):
        baseline['submission']['dataset'] = args.dataset
    zero_entry.validate_baseline(baseline)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh raw-shared output')
    source_sha, original_write, records = entry.sha(__file__), entry.write, []

    def write(path, document):
        path = Path(path)
        if path.parent == output / 'audit' and path.name in ('report.json', 'status.json', 'naf_report.json', 'naf_failed.json'):
            document = audit_document(document, records, source_sha)
        return original_write(path, document)

    try:
        with patch.object(naf_entry, 'patch', new=PlainPatch), \
                patch.object(entry, 'predict', new=raw_shared_predict(records)), \
                patch.object(entry, 'write', new=write):
            receipt = naf_entry.run(args, device=device)
        if entry.sha(__file__) != source_sha or [row['source'] for row in records] != [row['source'] for row in receipt['images']]:
            raise RuntimeError('Raw-shared source or invocation order changed')
        for row, naf in zip(records, receipt['images']):
            if (row['raw_semantic_logits_sha256'] != naf['semantic_sha256']
                    or row['raw_shared_probability_sha256'] != naf['raw_probability_sha256']
                    or row['input_boundary_sha256'] != naf['input_boundary_sha256']):
                raise RuntimeError('Actual raw-shared maps differ from NAF readout provenance')
        result = audit_document(receipt, records, source_sha)
        original_write(output / 'audit/raw_shared_report.json', result)
        return result
    except Exception as exc:
        if (output / 'audit').is_dir():
            original_write(output / 'audit/raw_shared_failed.json', dict(complete=False, format=FORMAT,
                error=repr(exc), raw_shared_images=records))
            for name in ('report.json', 'status.json'):
                path = output / 'audit' / name
                if path.is_file():
                    value = json.loads(path.read_text(encoding='utf-8'))
                    value.update(complete=False, raw_shared_complete=False, raw_shared_error=repr(exc))
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
                         actual_forward_counts=result['actual_forward_counts'])), flush=True)


if __name__ == '__main__':
    main()
