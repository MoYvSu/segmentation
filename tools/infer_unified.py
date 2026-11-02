# -*- coding: utf-8 -*-
"""统一原生1024滑窗：零噪D5a、一次encoder共享两头、整图NAF/S0/final200。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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

from models import backend_adaptation
from tools import infer_boundary_rgb as naf_entry
from tools import infer_semifinal as entry
from tools import infer_zero as zero_entry
from tools.affinity_connectivity_views import _inference_state
from tools.probe_affinity_patch import tensor_from_rgb
from train_backend_adaptation import fusion_options
from utils import restoration_noise as noise_module
from utils import restoration_probe
from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_fusion import affinity_boundary_probability
from utils.patch_diagnostic import BlendMap, phase_description, tile_boxes
from utils.semantic_native import NativeLogitBlend

FORMAT = 'unified_local1024_inference_v1'
SEMANTIC_ROLE = 'local_restored_semantic_logits_tent_then_CPU_sigmoid'


class PlainPatch:
    """旧NAF运行器的计数由自身维护；普通替换函数不保留大数组调用历史。"""
    @staticmethod
    def object(target, name, *, side_effect):
        return patch.object(target, name, new=side_effect)


def contract():
    return dict(format=FORMAT, native_sizes=[1024], network_input_size=1024, overlap=.25,
        restoration_noise=zero_entry.ZERO_MODE, global_restoration='not_computed',
        global_encoder_calls=0, raw_encoder_calls=0, local_restorer_calls_per_window=1,
        local_encoder_calls_per_window=1, shared_encoder_features=True,
        affinity_blend='unchanged BlendMap probability float32; original order/weights',
        semantic_blend='NativeLogitBlend float32 logits, axis minimum .001; CPU sigmoid after stitching',
        geometry_semantic_role=SEMANTIC_ROLE, vote_probability_role=SEMANTIC_ROLE,
        legacy_capture_roles=dict(semantic='local_restored_semantic_logits_B1HW',
                                  raw_probability='local_restored_semantic_probability_HW_not_raw_RGB'),
        naf_model_sha256=naf_entry.EXPECTED_MODEL_SHA, naf_input='T8_only', seed_support='original-zero',
        final_instance_area_policy=deepcopy(zero_entry.FINAL_POLICY), final200_calls_per_source=1,
        training=False, random_noise_generated=False)


def execution_config(baseline):
    zero_entry.validate_baseline(baseline)
    cfg = deepcopy(baseline)
    cfg['submission']['restoration_noise'] = zero_entry.ZERO_MODE
    cfg['submission']['global_restoration'] = dict(mode='not_computed')
    cfg['submission']['semantic_route'] = 'local_restored_shared_features'
    cfg['unified_local1024'] = contract()
    return cfg


def check_model(model):
    if any(getattr(model, name, None) is not None for name in (
            'semantic_lora', 'geometry_feature_adapter', 'geometry_highres_refiner')):
        raise ValueError('Unified experiment requires the unadapted shared-feature backend architecture')


def shared_tile(model, restored, config):
    """先保留旧affinity运算顺序，再用同一features调用语义头。"""
    features = model.encoder(restored)
    affinity = model.affinity_decoder(features)['affinity_logits'].float()
    mode, options = fusion_options(config)
    boundary = affinity_boundary_probability(affinity, mode=mode, **options)
    semantic = model.semantic_decoder(features, restored)
    if isinstance(semantic, dict):
        semantic = semantic['semantic_logits']
    if (semantic.ndim != 4 or tuple(semantic.shape[:2]) != (1, 1)
            or not bool(torch.isfinite(semantic).all())):
        raise RuntimeError('Unified semantic head must output finite B1HW logits')
    return boundary, semantic.float()


def unified_predict(records):
    @torch.no_grad()
    def predict(model, restorer, terminal, path, config, device, *, local_restorer=None,
                fixed_audit=None, local_audit=None, analysis_callback=None):
        if terminal is not None or local_restorer is not None:
            raise ValueError('Unified experiment has no replacement/raw/global models')
        check_model(model)
        cfg = execution_config(config)
        bgr = entry.read_image(path, cv2.IMREAD_COLOR)
        if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError('Native RGB image cannot be read')
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        boxes = tile_boxes(rgb.shape[:2], 1024, .25)
        boundary_blend = BlendMap(rgb.shape[:2])
        semantic_blend = NativeLogitBlend(rgb.shape[:2], .001)
        counts = dict(encoder_calls=0, semantic_head_calls=0, affinity_head_calls=0, local_zero_calls=0,
                      global_zero_calls=0, global_encoder_calls=0, raw_encoder_calls=0, final200_calls=0)
        window_rows = [] if local_audit is None else local_audit

        def counted(name):
            def hook(module, args, output):
                counts[name] += 1
            return hook

        with ExitStack() as stack:
            stack.enter_context(_inference_state(model, restorer, device))
            for name, module in (('encoder_calls', model.encoder), ('semantic_head_calls', model.semantic_decoder),
                                 ('affinity_head_calls', model.affinity_decoder)):
                handle = module.register_forward_hook(counted(name))
                stack.callback(handle.remove)
            for module, name in ((torch, 'randn'), (torch, 'randn_like'), (torch, 'normal'),
                    (backend_adaptation, 'restoration_noise'), (noise_module, 'restoration_noise'),
                    (restoration_probe, 'restoration_noise'), (entry.views, 'restore_first')):
                stack.enter_context(patch.object(module, name, new=zero_entry.forbidden_noise))
            source_seed = entry.source_seed(path.stem, cfg['submission'].get('dataset'))
            for index, box in enumerate(boxes):
                y, x, y1, x1 = box
                image, ph, pw = tensor_from_rgb(rgb[y:y1, x:x1], device)
                seed = source_seed + 100000 + 1024 * 1000 + index
                proof = {}
                restored, _ = restoration_probe.restore_probe(restorer, image, seed, 'zero', audit=proof)
                if (proof.get('noise_nonzero') != 0 or proof.get('initial_equals_condition') is not True
                        or proof.get('q_sample_calls') != 1 or proof.get('initial_max_abs') != 0.0):
                    raise RuntimeError('Unified D5a did not receive an explicit zero-noise state')
                counts['local_zero_calls'] += 1
                grid, semantic = shared_tile(model, restored, cfg)
                native_boundary = crop_letterbox_output(grid, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
                native_semantic = crop_letterbox_output(semantic, 1024, ph, pw, (y1-y, x1-x)).cpu()[0, 0].numpy()
                boundary_blend.add(box, native_boundary)
                # 两个既有工具的box顺序不同，必须显式转换。
                semantic_blend.add((int(y), int(y1), int(x), int(x1)), native_semantic)
                window_rows.append(dict(size=1024, window=index, box=list(box), seed=seed, noise_mode='zero',
                    **proof, input_sha256=naf_entry.array_sha(image.cpu().contiguous().numpy()),
                    restored_sha256=naf_entry.array_sha(restored.cpu().contiguous().numpy()),
                    local_semantic_logits_sha256=naf_entry.array_sha(native_semantic)))
                if analysis_callback is not None:
                    analysis_callback(source_shape=rgb.shape[:2], box=box, image=image, restored=restored,
                                      boundary=native_boundary, ph=ph, pw=pw, size=1024, window=index)
            if any(counts[name] != len(boxes) for name in ('encoder_calls', 'semantic_head_calls',
                                                          'affinity_head_calls', 'local_zero_calls')):
                raise RuntimeError('Unified branch did not execute exactly once per local window')
            boundary, blend = boundary_blend.finish()
            blend['tiles'] = len(boxes)
            semantic = semantic_blend.finish()
            semantic_tensor = torch.from_numpy(semantic)[None, None]
            probability = semantic_tensor.sigmoid()[0, 0].numpy()
            semantic_sha, probability_sha = naf_entry.array_sha(semantic), naf_entry.array_sha(probability)
            if fixed_audit is not None:
                fixed_audit.update(geometry_semantic_sha256=semantic_sha,
                    raw_probability_sha256=probability_sha, vote_probability_role=SEMANTIC_ROLE,
                    global_boundary_computed=False)
            # 外层NAF运行器在此接入唯一NAF/S0读出；末端票也走其唯一调用审计。
            ids, _ = entry.decode_reflect(semantic_tensor, torch.from_numpy(boundary)[None, None], cfg,
                                         original_decode=entry.views.decode_native)
            classes, _ = entry.views.revote_instances(ids, probability)
            result = dict(instances=ids, classes=classes, boundary=boundary,
                          phases=phase_description(ids, classes), blend=blend)
            result = entry.finalize_instance_area(result, cfg)
            counts['final200_calls'] += 1
        entry.verify_prediction(result['instances'], result['classes'], rgb.shape[:2])
        if naf_entry.array_sha(semantic) != semantic_sha or naf_entry.array_sha(probability) != probability_sha:
            raise RuntimeError('Readout mutated unified semantic maps')
        records.append(dict(source=path.stem, **counts, windows=len(boxes),
            input_boundary_sha256=naf_entry.array_sha(boundary), semantic_logits_sha256=semantic_sha,
            semantic_probability_sha256=probability_sha, semantic_blend=semantic_blend.receipt(),
            geometry_semantic_role=SEMANTIC_ROLE, vote_probability_role=SEMANTIC_ROLE,
            random_noise_generated=False))
        return rgb, result
    return predict


def audit_document(document, records, source_sha):
    value = deepcopy(document)
    value['legacy_format'] = value.get('legacy_format', value.get('format'))
    value.update(format=FORMAT, unified_local1024=contract(), unified_images=deepcopy(records),
                 unified_entry_sha256=source_sha, semantic=SEMANTIC_ROLE)
    value['actual_forward_counts'] = {name: sum(row[name] for row in records) for name in (
        'encoder_calls', 'semantic_head_calls', 'affinity_head_calls', 'local_zero_calls',
        'global_zero_calls', 'global_encoder_calls', 'raw_encoder_calls', 'final200_calls')}
    if 'config' in value:
        base = deepcopy(value.get('baseline_config', value['config']))
        value['baseline_config'], value['config'] = base, execution_config(base)
        value['config_role'] = 'effective unified-local execution; baseline_config is validation identity only'
    if 'runtime' in value:
        value['baseline_runtime'] = deepcopy(value.get('baseline_runtime', value['runtime']))
        value['runtime']['restoration_noise'] = dict(name=zero_entry.ZERO_MODE, local_noise='zero',
                                                     global_restoration='not_computed', random_noise_generated=False)
    if 'geometry' in value:
        value['geometry'] = 'local shared encoder semantic/affinity -> independent native blends -> T8/20a59/S0 -> WS/local semantic vote -> final200 once'
    for row in value.get('images', []):
        if 'raw_probability_sha256' in row:
            row['raw_probability_role'] = SEMANTIC_ROLE
        if 'boundary_naf' in row:
            row['boundary_naf']['raw_probability_role'] = SEMANTIC_ROLE
            row['boundary_naf']['geometry_semantic_role'] = SEMANTIC_ROLE
    return value


def run(args, *, device='cuda'):
    args = argparse.Namespace(**vars(args))
    if getattr(args, 'global_noise', 'zero') != 'zero':
        raise ValueError('Unified inference has explicit zero local noise and no global restoration')
    if getattr(args, 'expected_model_sha', naf_entry.EXPECTED_MODEL_SHA) != naf_entry.EXPECTED_MODEL_SHA:
        raise ValueError('Unified inference is fixed to old 20a59 NAF')
    if getattr(args, 'seed_support', 'original-zero') != 'original-zero':
        raise ValueError('Unified inference requires original-zero S0')
    args.expected_model_sha, args.seed_support = naf_entry.EXPECTED_MODEL_SHA, 'original-zero'
    baseline = entry.load_config(str(Path(args.config).resolve()))
    if getattr(args, 'dataset', None):
        baseline['submission']['dataset'] = args.dataset
    zero_entry.validate_baseline(baseline)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh unified inference output')
    source_sha, original_write, records = entry.sha(__file__), entry.write, []

    def write(path, document):
        path = Path(path)
        if path.parent == output / 'audit' and path.name in ('report.json', 'status.json', 'naf_report.json', 'naf_failed.json'):
            document = audit_document(document, records, source_sha)
        return original_write(path, document)

    try:
        with patch.object(naf_entry, 'patch', new=PlainPatch), \
                patch.object(entry, 'predict', new=unified_predict(records)), \
                patch.object(entry, 'write', new=write):
            receipt = naf_entry.run(args, device=device)
        if entry.sha(__file__) != source_sha or [r['source'] for r in records] != [r['source'] for r in receipt['images']]:
            raise RuntimeError('Unified source invocation order or source changed')
        result = audit_document(receipt, records, source_sha)
        original_write(output / 'audit/unified_report.json', result)
        return result
    except Exception as exc:
        if (output / 'audit').is_dir():
            original_write(output / 'audit/unified_failed.json', dict(complete=False, format=FORMAT,
                error=repr(exc), unified_images=records))
            for name in ('report.json', 'status.json'):
                path = output / 'audit' / name
                if path.is_file():
                    value = json.loads(path.read_text(encoding='utf-8'))
                    value.update(complete=False, unified_complete=False, unified_error=repr(exc))
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
