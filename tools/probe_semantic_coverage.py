# -*- coding: utf-8 -*-
"""仅训练源：核查新增灰度目标、已有GT上的规则冲突及固定候选的实际学习信号。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from data.semantic_consistency import apply_acquisition_appearance, full_content_mask
from data.semantic_gray import GrayPriorDataset, build_gray_targets, save_coverage_preview
from data.semantic_hard import gray_regions, hard_appearance, region_diagnostics
from models.backend_adaptation import load_backend, restore_first
from models.semantic_lora import semantic_features
from train_backend_adaptation import sha, write_json
from train_semantic_consistency import unbatch_value, verify_contract
from train_semantic_d5a import build_dataset, get_restorer
from utils.config import load_config, project_path


def paired_targets(clean, valid, known, config):
    base = deepcopy(config)
    base.pop('coverage')
    old_target, old_mask, old_info = build_gray_targets(clean, valid, known, base)
    target, mask, info = build_gray_targets(clean, valid, known, config)
    if not torch.all(mask[old_mask]) or not torch.equal(old_target[old_mask], target[old_mask]):
        raise RuntimeError('Coverage expansion changed an existing target')
    return target, old_mask, mask, old_info, info


def check_gt(manual, config, output):
    """只校验固定规则，不训练模型、不从GT拟合灰度阈值。"""
    rows = []
    for i in range(len(manual)):
        raw = default_collate([manual[i]])
        for key in ['horizontal_flip', 'vertical_flip', 'rotation_k']:
            raw[key] = torch.tensor([0])
        valid = full_content_mask(raw, raw['image'])
        target, old, mask, _, info = paired_targets(raw['image'], valid, torch.zeros_like(valid), config)
        gt = F.interpolate(raw['semantic_target'].float(), target.shape[-2:], mode='area')
        interior = (F.interpolate(raw['semantic_valid_content'].float(), target.shape[-2:], mode='area') >= 1)
        interior &= F.interpolate(raw['semantic_boundary'].float(), target.shape[-2:], mode='area') == 0
        interior &= (gt == 0) | (gt == 1)
        record = dict(name=raw['name'][0], coverage=info['coverage'], groups={})
        for name, selection in [('base',old), ('added',mask & ~old)]:
            selection = selection & interior
            wrong = selection & ((target > .5) != (gt > .5))
            record['groups'][name] = dict(pixels=int(selection.sum()), wrong=int(wrong.sum()),
                true_ferrite=int((selection & (gt > .5)).sum()), true_pearlite=int((selection & (gt <= .5)).sum()),
                ferrite_to_pearlite=int((wrong & (gt > .5)).sum()),
                pearlite_to_ferrite=int((wrong & (gt <= .5)).sum()))
        if info['coverage']['added_pixels']:
            save_coverage_preview(output/'gt_preview'/f'{i:03d}.png', raw['image'], raw['image'],
                target, old, mask, None, valid)
        rows.append(record)
    write_json(output/'gt_rows.json', rows)
    return {group: {key:sum(r['groups'][group][key] for r in rows)
                    for key in rows[0]['groups'][group]} for group in ['base','added']}


@torch.inference_mode()
def run(config_path, output_dir=None):
    config = load_config(config_path)
    cfg, ucfg, ccfg = config['semantic_adaptation'], config['semantic_consistency'], config['semantic_coverage']
    output = Path(project_path(config, output_dir or ccfg['probe_dir']))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    reference = Path(project_path(config, ccfg['reference_dir']))
    status = json.loads((reference/'status.json').read_text())
    checkpoint = Path(status['final_checkpoint'])
    if status['status'] != 'completed' or status['updates'] != 1280 or sha(checkpoint) != ccfg['reference_sha256']:
        raise RuntimeError('Expected complete independent-LoRA control')
    replay = [json.loads(line) for line in (reference/'steps.jsonl').read_text().splitlines()]
    if len(replay) != status['updates']:
        raise RuntimeError('Incomplete reference input records')
    manual, data = build_dataset(config)
    gt_summary = check_gt(manual.base_dataset, config['semantic_gray'], output)
    print(json.dumps({'gt_rule_check':gt_summary}), flush=True)
    device = torch.device('cuda')
    model, saved = load_backend(str(checkpoint), config, device)
    verify_contract(config, saved)
    model.eval().requires_grad_(False)
    if model.semantic_lora is None:
        raise RuntimeError('Probe must use independent semantic LoRA')
    restorer = get_restorer(config, device)
    dataset = GrayPriorDataset(project_path(config, ucfg['unlabeled_dir']), data['degradation'], cfg['noise'],
        cfg['seed'] + 271, draws_per_epoch=data['updates_per_epoch'], expected_sources=1000,
        image_size=1024, weak_config=ucfg['weak_check'], manual_dataset=manual.base_dataset)
    rows, previews, largest, started = [], [], -1, time.time()
    for n, reference_row in enumerate(replay):
        old_info = reference_row['unlabeled']
        draw = int(old_info['global_draw'])
        dataset.set_epoch(draw // len(dataset) + 1)
        raw = default_collate([dataset[draw % len(dataset)]])
        for key in ['name','global_draw','source_index','profile','is_spatial','endpoint_applied',
                    'noise_sigma','horizontal_flip','vertical_flip','rotation_k']:
            if unbatch_value(raw[key]) != old_info[key]:
                raise RuntimeError(f'Replayed source/geometry differs: {key}')
        target, old_mask, mask, base_info, info = paired_targets(raw['weak_image'], raw['valid_content'],
            raw['known_gt_mask'], config['semantic_gray'])
        for key in ['accepted_pixels','accepted_ferrite','accepted_pearlite','valid_pixels','known_gt_excluded_pixels']:
            if base_info[key] != old_info[key]:
                raise RuntimeError(f'Legacy prior differs from completed training: {key}')
        added = mask & ~old_mask
        row = dict(draw=draw, name=old_info['name'], valid=info['valid_pixels'], base=base_info['accepted_pixels'],
                   **info['coverage'], active_pixels=0, disagree_pixels=0, disagree_regions=0,
                   maximum_wrong_component=0, ferrite_to_pearlite_pixels=0, pearlite_to_ferrite_pixels=0)
        if added.any():
            seed = reference_row['seed']
            base, appearance = apply_acquisition_appearance(raw['strong_image'].to(device),
                raw['valid_content'].to(device), ucfg['appearance'], seed + 3000000)
            h = old_info['hard_views']
            selected = h['selected_index']
            image, stats = (base,appearance) if selected == 0 else hard_appearance(base,
                raw['valid_content'].to(device), config['semantic_hard'],
                seed + 7000000 + (selected-1)*100003, h['selected_mode'])
            if stats != h['views'][selected]['appearance'] or appearance != old_info['appearance']:
                raise RuntimeError('Selected appearance replay differs')
            restored = restore_first(restorer, image, old_info['strong_restoration_seed'])
            logits = model.semantic_decoder(semantic_features(model, restored.float()), restored)
            t, m = target.to(device), added.to(device)
            regions, classes = gray_regions(t, m, config['semantic_hard']['minimum_region_pixels'])
            diagnostics = region_diagnostics(logits, t, m, regions, classes)
            diagnostics.pop('region_scores')
            row.update(diagnostics)
            p = logits.float().sigmoid()
            row['active_pixels'] = int((m & (torch.where(t > .5, p, 1-p) < config['semantic_gray']['minimum_confidence'])).sum())
            if len(previews) < 8 or draw in [3,221,227,509]:
                filename = f'draw_{draw:04d}.png'
                save_coverage_preview(output/'preview'/filename, raw['weak_image'], restored,
                    target, old_mask, mask, logits, raw['valid_content'])
                previews.append(dict(file=filename, draw=draw, name=old_info['name']))
            if row['disagree_pixels'] > largest:
                largest = row['disagree_pixels']
                save_coverage_preview(output/'preview/largest_disagreement.png', raw['weak_image'], restored,
                    target, old_mask, mask, logits, raw['valid_content'])
                write_json(output/'largest_disagreement.json', row)
        rows.append(row)
        with (output/'rows.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        if (n+1)%100 == 0:
            print(json.dumps({'draws':n+1,'added_sources':len({r['name'] for r in rows if r['added_pixels']}),
                'disagree_pixels':sum(r['disagree_pixels'] for r in rows),'elapsed':time.time()-started}), flush=True)
    totals = {k:sum(r[k] for r in rows) for k in ['valid','base','unsupported_pixels','added_pixels',
        'added_ferrite','added_pearlite','active_pixels','disagree_pixels','disagree_regions',
        'ferrite_to_pearlite_pixels','pearlite_to_ferrite_pixels']}
    summary = dict(status='completed', training_sources_only=True, no_optimization=True,
        legacy_targets_unchanged=True, selected_view_replay_passed=True, checkpoint=str(checkpoint),
        checkpoint_sha256=ccfg['reference_sha256'], config=config, draws=len(rows),
        sources=len({r['name'] for r in rows}), sources_with_added=len({r['name'] for r in rows if r['added_pixels']}),
        sources_with_conflict=len({r['name'] for r in rows if r['disagree_pixels']}),
        sources_with_component_ge32=len({r['name'] for r in rows if r['maximum_wrong_component']>=32}),
        totals=totals, gt_rule_check=gt_summary, previews=previews, elapsed_seconds=time.time()-started,
        decision='pending_review_of_signal_and_target_quality', official_score=None)
    write_json(output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config','previews')},ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/semantic_coverage.yaml')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    run(args.config, args.output_dir)
