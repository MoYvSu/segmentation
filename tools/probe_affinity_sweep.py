# -*- coding: utf-8 -*-
"""同一D5a/encoder输入上的三头参考loss与等召回诊断；已见GT，不训练或选epoch。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_affinity_equal_recall import (
    LAYERS, make_cases, prepared_view, score_histogram, fused_targets, summarize, curve_metrics, equal_recall)
from models.backend_adaptation import load_backend, restore_first
from train_affinity_connectivity import get_restorer, frozen_digest
from train_affinity_sweep import verify_run
from train_backend_adaptation import sha, tensor_digest, fusion_options, write_json
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_loss import balanced_affinity_loss
from utils.config import load_config, project_path

PATHS = {'control': 'outputs/affinity_balance/mixed/final.pt',
         'n075': 'outputs/affinity_sweep/n075/final.pt',
         'n125': 'outputs/affinity_sweep/n125/final.pt'}
HASHES = {'control': '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434',
          'n075': 'e6ef317af4b4f86427807143cd5fa426c5b40bedb64e5b0d16b1fdff3bad4394',
          'n125': 'e20aac61eb1e45bc0f253ce999691a3ae60ad4fd7e7fe4d4e3fb7c1a4a2fb138'}


def tensor_items(value, prefix='features'):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from tensor_items(item, prefix+'/'+str(name))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_items(item, prefix+'/'+str(index))


def run(output, seed=20261001):
    if not torch.cuda.is_available():
        raise RuntimeError('Use server sam2_env')
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = load_config('config/train/affinity_balance_mixed.yaml')
    root = ROOT/'outputs/affinity_sweep'
    pipeline = json.loads((root/'pipeline_status.json').read_text())
    if pipeline['status'] != 'complete':
        raise RuntimeError('Both training and complete inference must finish first')
    for name, expected in pipeline['sources'].items():
        if sha(ROOT/name) != expected or sha(root/'source'/name) != expected:
            raise RuntimeError('Trained source differs: '+name)
    for name in ('n075', 'n125'):
        verify_run(load_config(f'config/train/affinity_sweep_{name}.yaml'), root/name, 1280)
    out = Path(project_path(config, output))
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    cases = make_cases(config, 'all', seed)
    if len(cases) != 32:
        raise RuntimeError('Expected the same all-32 seen manual sources')
    write_json(out/'cases.json', cases)
    models, before, checkpoints = {}, {}, {}
    shared = None
    for name, path in PATHS.items():
        absolute = project_path(config, path)
        if sha(absolute) != HASHES[name]:
            raise RuntimeError('Checkpoint differs: '+name)
        model, bundle = load_backend(absolute, config, 'cuda')
        model.eval().requires_grad_(False)
        frozen = frozen_digest(model)
        if shared is not None and frozen != shared:
            raise RuntimeError('Encoder/LoRA/semantic weights differ')
        shared = frozen
        models[name], before[name] = model, tensor_digest(model.state_dict().items())
        checkpoints[name] = dict(path=path, sha256=HASHES[name], epoch=bundle['epoch'],
                                 shared_state_sha256=frozen)
        if bundle['epoch'] != 20:
            raise RuntimeError('Only fixed e20 endpoints are compared')
    restorer = get_restorer(config, 'cuda')
    restorer_before = tensor_digest(restorer.state_dict().items())
    mode, fusion = fusion_options(config)
    reference = {'negative_weight': 1., 'hard_negative_weight': 1., 'hard_negative_gamma': 2.,
                 'normalize_edge_weights': True}
    records, losses, receipts = [], [], []
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        for case in cases:
            for profile in ('clean', 'blur'):
                image, target, valid, receipt = prepared_view(case, profile, config, seed)
                tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()[None].to('cuda')
                restored = restore_first(restorer, tensor, receipt['restoration_seed'])
                features = models['control'].encoder(restored)
                receipt['restored_sha256'] = tensor_digest([('image', restored)])
                receipt['shared_features_sha256'] = tensor_digest(tensor_items(features))
                if not receipts:
                    for name in ('n075', 'n125'):
                        other = models[name].encoder(restored)
                        if tensor_digest(tensor_items(other)) != receipt['shared_features_sha256']:
                            raise RuntimeError('Shared encoder features do not reproduce '+name)
                receipts.append(receipt)
                target_t = torch.from_numpy(target)[None].to('cuda', dtype=torch.float32)
                valid_t = torch.from_numpy(valid)[None].to('cuda')
                for name, model in models.items():
                    logits = model.affinity_decoder(features)['affinity_logits'].float()
                    if tuple(logits.shape) != (1, 8, 512, 512):
                        raise RuntimeError('Unexpected affinity grid')
                    loss, metrics = balanced_affinity_loss(logits, target_t, valid_t, **reference)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite reference loss')
                    losses.append(dict(model=name, source=case['name'], profile=profile,
                        reference_loss=float(loss), connection_recall=metrics['recall'],
                        separation_specificity=metrics['specificity']))
                    scores = (1-logits.sigmoid())[0].cpu().numpy()
                    fused = affinity_boundary_probability(logits, mode=mode, **fusion)[0, 0].cpu().numpy()
                    histograms = {}
                    for layer, selection in [('relations_all', slice(None)), ('relations_short', slice(0, 4)),
                        ('relations_distance2', slice(4, 6)), ('relations_distance4', slice(6, 8))]:
                        histograms[layer] = score_histogram(scores[selection], ~target[selection], valid[selection])
                    for layer, strict in [('fused_boundary', False), ('fused_deep_interior', True)]:
                        boundary, eligible = fused_targets(target, valid, strict_interior=strict)
                        histograms[layer] = score_histogram(fused, boundary, eligible)
                    records.append(dict(model=name, source=case['name'], profile=profile, histograms=histograms))
                print(case['name'], profile, 'completed', len(receipts), flush=True)
    if any(tensor_digest(model.state_dict().items()) != before[name] for name, model in models.items()):
        raise RuntimeError('Diagnostic changed a backend')
    if tensor_digest(restorer.state_dict().items()) != restorer_before:
        raise RuntimeError('Diagnostic changed D5a')
    comparisons = {}
    for candidate in ('n075', 'n125'):
        pair = [{**record, 'model': 'control' if record['model'] == 'control' else 'balance'}
                for record in records if record['model'] in ('control', candidate)]
        pooled, paired = summarize(pair, [.90, .95, .97, .98])
        comparisons[candidate] = dict(pooled=pooled, source_paired=paired)
    loss_summary = {profile: {name: {key: float(np.mean([row[key] for row in losses
        if row['model'] == name and row['profile'] == profile]))
        for key in ('reference_loss', 'connection_recall', 'separation_specificity')}
        for name in models} for profile in ('clean', 'blur')}
    np.savez_compressed(out/'histograms.npz', **{
        record['model']+'/'+record['source']+'/'+record['profile']+'/'+layer: histogram
        for record in records for layer, histogram in record['histograms'].items()})
    report = dict(complete=True, training=False,
        scope='Same 32 SEEN manual sources, originally trusted pixels; no holdout or official accuracy',
        sources=32, input_views=64, affinity_head_views=len(records), shared_encoder_views=64,
        extra_encoder_parity_views=2, encoder_parity_exact=True, frozen_exact=True, d5a_frozen_exact=True,
        checkpoints=checkpoints, seed=seed, source_sha256=sha(__file__),
        reference_loss_formula=reference, loss_summary=loss_summary, loss_rows=losses,
        comparisons=comparisons, receipts=receipts, elapsed_seconds=time.time()-started,
        caveats=['Original GT is incomplete and imperfect; unknown pixels remain ignored.',
                 'Equal-recall thresholds are diagnostic and never alter deployment.',
                 'Fused pixel boundary labels diagnose a mechanism and are not final-instance scores.',
                 'Reference loss is not a held-out metric or checkpoint selector.'])
    write_json(out/'report.json', report)
    print('COMPLETE', json.dumps(loss_summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/affinity_sweep/analysis/probe')
    args = parser.parse_args()
    run(args.output)
