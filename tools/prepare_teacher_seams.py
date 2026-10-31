# -*- coding: utf-8 -*-
"""只读原32源教师配对，生成独立窄零缝候选及CPU关系/缩略图预检。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.semantic_targets import load_completed_semantic_source
from data.teacher_seam_manifest import (FORMAT, POLICY, CODE_PATHS, file_sha, load_seam_manifest,
                                        relative_file, seam_options, validate_seam_arrays)
from data.rgb_restoration_dataset import read_rgb
from utils.config import load_config
from utils.teacher_seam_support import repair_teacher_zero_seams, resize_seam_pair


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _relative(config, path):
    return Path(path).resolve().relative_to(Path(config['paths']['project_root']).resolve()).as_posix()


def _output_directory(config):
    opts = seam_options(config)
    value = opts['manifest']
    root = Path(config['paths']['project_root']).resolve()
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Output manifest must be project-relative')
    target = (root / path).resolve()
    if (not target.is_relative_to(root) or not path.parts
            or path.parts[0] not in ('output', 'outputs') or target.name != 'manifest.json'):
        raise ValueError('Private output(s)/.../manifest.json required')
    if target.parent.exists():
        raise FileExistsError('Fresh derivative directory required; refusing overwrite: ' + str(target.parent))
    return target.parent, target


def _colors(ids):
    value = ids.astype(np.uint32)
    color = np.stack(((value * 97 + 37) % 220 + 25,
                      (value * 57 + 103) % 220 + 25,
                      (value * 137 + 11) % 220 + 25), axis=-1).astype(np.uint8)
    color[value == 0] = 0
    return color


def _panel(rgb, label, width=240, height=190):
    sy = min(width / rgb.shape[1], (height - 22) / rgb.shape[0])
    view = cv2.resize(rgb, (max(1, round(rgb.shape[1] * sy)), max(1, round(rgb.shape[0] * sy))), interpolation=cv2.INTER_AREA)
    panel = np.zeros((height, width, 3), np.uint8)
    x, y = (width-view.shape[1])//2, 22+(height-22-view.shape[0])//2
    panel[y:y+view.shape[0], x:x+view.shape[1]] = view
    cv2.putText(panel, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (240, 240, 240), 1, cv2.LINE_AA)
    return panel


def _save_rgb(path, value):
    if not cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR)):
        raise RuntimeError('Failed to write private gallery')


def save_gallery(directory, source, rgb, manual, old_teacher, new_teacher, assigned):
    overlay = rgb.copy()
    overlay[assigned] = (255, 30, 220)
    labels = ('raw', 'processed manual', 'old teacher', 'repaired teacher', 'assigned seams')
    images = (rgb, _colors(manual), _colors(old_teacher), _colors(new_teacher), overlay)
    _save_rgb(directory / (source + '.jpg'), np.concatenate([_panel(im, title) for im, title in zip(images, labels)], axis=1))
    positions = np.argwhere(assigned)
    center = positions[len(positions)//2] if len(positions) else np.array(manual.shape)//2
    height, width = manual.shape
    y, x = [int(value) for value in center]
    y0, y1 = max(0, y-96), min(height, y+96)
    x0, x1 = max(0, x-96), min(width, x+96)
    _save_rgb(directory / (source + '_local.jpg'), np.concatenate([_panel(im[y0:y1, x0:x1], title) for im, title in zip(images, labels)], axis=1))
    card = np.concatenate([_panel(im, title, 160, 138) for im, title in zip(images[2:], labels[2:])], axis=1)
    cv2.putText(card, source, (4, 134), cv2.FONT_HERSHEY_SIMPLEX, .4, (240, 240, 240), 1)
    return card, [y0, x0, y1, x1]


def _summarize(rows):
    totals = {key: sum(row['audit']['pixels'][key] for row in rows) for key in rows[0]['audit']['pixels']}
    grid_keys = ('old_legal_edges', 'new_legal_edges', 'added_legal_edges', 'full_manual_legal_edges')
    grid = {key: sum(row['counts'][key] for row in rows) for key in grid_keys}
    strata = {}
    relation_keys = ('selected_edges', 'manual_same_teacher_same', 'manual_same_teacher_different',
                     'manual_different_teacher_same', 'manual_different_teacher_different')
    for name in rows[0]['counts']['strata']:
        values = {key: sum(row['counts']['strata'][name][key] for row in rows) for key in relation_keys}
        values['per_channel'] = [dict(channel=channel, **{key: sum(row['counts']['strata'][name]['per_channel'][channel][key] for row in rows) for key in relation_keys}) for channel in range(8)]
        strata[name] = values
    grid['strata'] = strata
    grid['restored_manual_different_teacher_different'] = strata['all_added']['manual_different_teacher_different']
    grid['new_manual_same_teacher_different'] = strata['all_added']['manual_same_teacher_different']
    return {'native_pixels': totals, 'whole1024_grid512': grid,
            'all_existing_positive_ids_preserved': all(row['audit']['existing_positive_ids_preserved'] for row in rows),
            'all_instance_sets_preserved': all(row['audit']['known_instance_id_set_preserved'] for row in rows),
            'all_old_legal_edges_and_answers_preserved': all(row['counts']['old_legal_subset_verified'] and row['counts']['old_teacher_answers_unchanged'] for row in rows),
            'limits': 'Full-view512 audit only; actual native crop/augmentation exposure is measured in training. Manual comparisons are mechanism diagnostics on seen, incomplete GT, not validation accuracy.'}


def prepare(config):
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    # 原教师loader的确认和SHA门禁完全执行；不假造新的人工确认。
    from train_affinity_gt_source import load_targets
    parent, protected = load_targets(config, 'teacher')
    parent_path = relative_file(config, config['gt_source']['manifest'])
    approval = relative_file(config, config['gt_source']['approval_file'])
    parent_sha, approval_sha = file_sha(parent_path), file_sha(approval)
    code_hashes = {value: file_sha(relative_file(config, value)) for value in CODE_PATHS}
    protected.update({str(relative_file(config, value)): digest for value, digest in code_hashes.items()})
    directory, manifest_path = _output_directory(config)
    directory.mkdir(parents=True)
    (directory / 'pairs').mkdir()
    (directory / 'gallery').mkdir()
    receipt = dict(format=FORMAT, status='preparing', parent_manifest_sha256=parent_sha,
                   parent_approval_sha256=approval_sha, policy=POLICY,
                   user_authorized_transform=True, user_authorization='User authorized the proposed seam repair experiment in current conversation',
                   derived_targets_individually_human_confirmed=False,
                   original_user_confirmation_preserved=True,
                   semantic_labels_used_for_assignment=False, semantic_labels_used_for_training=False,
                   test_training=False, original_files_modified=False,
                   protected_code=code_hashes,
                   automatic_best_promotion=False, rows=[])
    write_json(directory / 'preflight_status.json', receipt)
    cards = []
    try:
        for row in sorted(parent['rows'], key=lambda value: value['source']):
            source = row['source']
            image_path = relative_file(config, row['image'])
            manual_path = relative_file(config, row['manual_gt'])
            old_pair_path = relative_file(config, row['pair_npz'])
            teacher_path = parent_path.parent / 'teacher' / (source + '_inst.png')
            teacher_sha = row['prediction_hashes']['png_sha256']
            if not teacher_path.is_file() or file_sha(teacher_path) != teacher_sha:
                raise RuntimeError('Original teacher PNG SHA differs: ' + source)
            protected[str(teacher_path)] = teacher_sha
            raw = read_rgb(str(image_path))
            teacher = cv2.imread(str(teacher_path), cv2.IMREAD_UNCHANGED)
            with np.load(manual_path, allow_pickle=False) as gt, np.load(old_pair_path, allow_pickle=False) as old_blob:
                manual = gt['instance_map'].copy()
                pair, audit = repair_teacher_zero_seams(manual, teacher, gt['original_covered'],
                    source_name=source, allowed_training_sources=parent['allowed_training_sources'], **POLICY)
                arrays = {'teacher_ids': pair['teacher_masked'], 'assigned_mask': pair['assigned_mask'].astype(np.uint8)}
                validate_seam_arrays(arrays, old_blob, manual, teacher)
            semantic_instances, lookup = load_completed_semantic_source(manual_path, manual.shape)
            if not np.array_equal(semantic_instances, manual):
                raise ValueError('Semantic audit differs from original processed instance map')
            semantic = lookup[manual].astype(np.int16)
            class_path = manual_path.with_name(source + '_class.json')
            class_sha = file_sha(class_path)
            protected[str(class_path)] = class_sha
            _, counts = resize_seam_pair(pair, input_size=1024, output_grid=512, semantic=semantic)
            path = directory / 'pairs' / (source + '.npz')
            np.savez_compressed(path, **arrays)
            card, crop = save_gallery(directory / 'gallery', source, raw, manual,
                pair['old_teacher_masked'], pair['teacher_masked'], pair['assigned_mask'])
            cards.append(card)
            unique, areas = np.unique(pair['old_teacher_masked'][pair['old_common_known']], return_counts=True)
            new_unique, new_areas = np.unique(pair['teacher_masked'][pair['common_known']], return_counts=True)
            receipt['rows'].append(dict(source=source, pair_npz=_relative(config, path), pair_sha256=file_sha(path),
                old_pair_sha256=row['pair_sha256'], image_sha256=row['image_sha256'], manual_gt_sha256=row['manual_gt_sha256'],
                teacher_png=_relative(config, teacher_path), teacher_png_sha256=teacher_sha,
                semantic_class_json=_relative(config, class_path), semantic_class_sha256=class_sha,
                audit=audit, counts=counts, gallery_local_native_box=crop,
                instance_retention=dict(old_count=int(unique.size), new_count=int(new_unique.size),
                    tiny_native_lt200_old=int((areas < 200).sum()), tiny_native_lt200_new=int((new_areas < 200).sum()),
                    retained_all_old_ids=bool(np.array_equal(unique, new_unique)))))
        # 在完整计算后复验源，不因长预检期间的文件变化生成貌似完备回执。
        for path, digest in protected.items():
            if file_sha(path) != digest:
                raise RuntimeError('Protected source changed during preflight: ' + path)
        receipt['status'] = 'complete'
        receipt['summary'] = _summarize(receipt['rows'])
        write_json(manifest_path, receipt)
        bound = deepcopy(config)
        bound['gt_source']['seam_repair']['manifest_sha256'] = file_sha(manifest_path)
        _, checked = load_seam_manifest(bound, parent, parent_sha)
        contact = np.concatenate([np.concatenate(cards[index:index+2], axis=1) for index in range(0, len(cards), 2)], axis=0)
        _save_rgb(directory / 'contact32.jpg', contact)
        write_json(directory / 'preflight_status.json', {'status': 'complete', 'manifest_sha256': file_sha(manifest_path),
            'protected_input_count': len(checked), 'summary': receipt['summary'],
            'training_started': False, 'user_authorized_transform': True,
            'derived_targets_individually_human_confirmed': False})
        return receipt, file_sha(manifest_path)
    except Exception as error:
        if manifest_path.exists():
            receipt['status'] = 'failed'
            receipt['error'] = str(error)
            write_json(manifest_path, receipt)
        write_json(directory / 'preflight_status.json', {'status': 'failed', 'error': str(error),
                   'training_started': False, 'original_files_modified': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/teacher_seams.yaml')
    arguments = parser.parse_args()
    result, digest = prepare(load_config(arguments.config))
    print(json.dumps({'status': result['status'], 'sources': len(result['rows']),
                      'manifest_sha256': digest, 'summary': result['summary']}, ensure_ascii=False))
