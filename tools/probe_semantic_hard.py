# -*- coding: utf-8 -*-
"""固定64张非人工标注训练源，比较原增强与全局/局部困难光度视图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate

from data.semantic_consistency import apply_acquisition_appearance
from data.semantic_gray import GrayPriorDataset, build_gray_targets, save_prior_preview
from data.semantic_hard import gray_regions, hard_appearance, region_diagnostics
from models.backend_adaptation import load_backend, restore_first
from train_backend_adaptation import write_json
from train_semantic_consistency import verify_contract
from train_semantic_d5a import build_dataset, get_restorer
from utils.config import load_config, project_path


@torch.inference_mode()
def run(config_path, output_dir):
    config = load_config(config_path)
    cfg, ucfg, hcfg = config['semantic_adaptation'], config['semantic_consistency'], config['semantic_hard']
    output = Path(project_path(config, output_dir))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = torch.device('cuda')
    model, saved = load_backend(project_path(config, ucfg['initial_checkpoint']), config, device)
    verify_contract(config, saved)
    model.eval().requires_grad_(False)
    restorer = get_restorer(config, device)
    manual, info = build_dataset(config)
    dataset = GrayPriorDataset(project_path(config, ucfg['unlabeled_dir']), info['degradation'], cfg['noise'],
        cfg['seed'] + 271, draws_per_epoch=1000, expected_sources=1000, image_size=1024,
        weak_config=ucfg['weak_check'], manual_dataset=manual.base_dataset)
    excluded = {r['pool_name'] for r in dataset.manual_mapping}
    indices = [i for i in range(1000) if dataset.samples[dataset.source_index_for_draw(i)].name not in excluded]
    selected = np.random.default_rng(20260927).choice(indices, int(hcfg['probe_sources']), replace=False).tolist()
    rows, started = [], time.time()
    for n, index in enumerate(selected):
        for view in range(int(hcfg['probe_views'])):
            # 全1000源跨周期顺序可变；重复视图保持源，直接改PairedDegradationDataset退化随机流。
            raw = dataset[index]
            if view:
                dataset.paired.set_epoch(index + 100000)
                extra = dataset.paired[raw['source_index']]
                from data.direct_dual_head_dataset import _spatial_transform
                geometry = (extra['horizontal_flip'], extra['vertical_flip'], extra['rotation_k'])
                raw.update(strong_image=extra['image'], valid_content=extra['valid_content'],
                           weak_image=_spatial_transform(extra['clean_image'], *geometry))
                # 本诊断仅排除了人工源，known mask全零，几何改变不影响排除契约。
                raw['known_gt_mask'] = torch.zeros_like(raw['valid_content'], dtype=torch.bool)
            batch = default_collate([raw])
            target, mask, _ = build_gray_targets(batch['weak_image'], batch['valid_content'],
                batch['known_gt_mask'], config['semantic_gray'])
            target, mask = target.to(device), mask.to(device)
            regions, classes = gray_regions(target, mask, hcfg['minimum_region_pixels'])
            seed = cfg['seed'] + 10000 + index + view * 100000
            strong, valid = batch['strong_image'].to(device), batch['valid_content'].to(device)
            base, appearance = apply_acquisition_appearance(strong, valid, ucfg['appearance'], seed + 3000000)
            entry = dict(name=raw['name'], draw=index, view=view, conditions={})
            for mode in ['clean', *hcfg['views']]:
                transformed, stats = (batch['weak_image'].to(device), {}) if mode == 'clean' else (base, appearance)
                if mode in ('global', 'local'):
                    transformed, stats = hard_appearance(base, valid, hcfg, seed + 7000000, mode)
                restored = restore_first(restorer, transformed, seed + 4000000)
                logits = model.semantic_decoder(model.encoder(restored.float()), restored)
                measured = region_diagnostics(logits, target, mask, regions, classes)
                entry['conditions'][mode] = dict(**measured, appearance=stats)
                if view == 0:
                    save_prior_preview(output / 'preview' / f'{n:03d}_{mode}.png', batch['weak_image'],
                        restored, target, mask, logits, batch['valid_content'])
            rows.append(entry)
            with (output / 'rows.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        if (n + 1) % 8 == 0:
            print(json.dumps({'sources_done': n + 1, 'elapsed': time.time() - started}), flush=True)
    summary = {'sources': len(selected), 'views_per_source': int(hcfg['probe_views']),
        'checkpoint': ucfg['initial_checkpoint'], 'checkpoint_sha256': ucfg['initial_sha256'],
        'config': config, 'training_sources_only': True, 'measures_prior_disagreement_not_accuracy': True,
        'elapsed_seconds': time.time() - started, 'conditions': {}}
    for mode in ['clean', *hcfg['views']]:
        values = [r['conditions'][mode] for r in rows]
        numeric = ['accepted', 'disagree_pixels', 'ferrite_to_pearlite_pixels', 'pearlite_to_ferrite_pixels',
                   'regions', 'disagree_regions', 'ferrite_to_pearlite_regions', 'pearlite_to_ferrite_regions']
        total = {key: sum(v[key] for v in values) for key in numeric}
        total.update(views_with_disagreement=sum(v['disagree_regions'] > 0 for v in values),
            views_with_large_conflict=sum(v['maximum_wrong_component'] >= 64 for v in values),
            maximum_wrong_component=max(v['maximum_wrong_component'] for v in values),
            class_disagreement_fraction=total['disagree_pixels'] / max(1, total['accepted']))
        summary['conditions'][mode] = total
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary['conditions'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_gray4.yaml')
    parser.add_argument('--output-dir', default='outputs/semantic_gray4_probe')
    args = parser.parse_args()
    run(args.config, args.output_dir)
