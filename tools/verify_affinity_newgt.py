# -*- coding: utf-8 -*-
"""只读绑定32张处理后新GT；逐图哈希只写入忽略目录的私有回执。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.backend_adaptation import CanonicalBackendDataset
from data.rgb_restoration_dataset import read_rgb
from data.semantic_targets import load_completed_semantic_source
from utils.config import load_config, project_path

REGISTERED_NPZ_COHORT_SHA256 = 'e7e34544fbb792c0e300d086689d28739882edd4b7aecd847f9e07499b49fe06'
REGISTERED_CLASS_COHORT_SHA256 = '1905ee57eec60c930fb75a5c1a8a605816a682262fa88e2c979d9a4fb87ce020'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_path(config, path):
    # 保留项目内合法共享链接的逻辑相对目录，不将服务器实际链接位置写死。
    base = Path(os.path.abspath(config['paths']['project_root']))
    actual = Path(os.path.abspath(path))
    try:
        return actual.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError('Audit inputs must use a project-relative location') from exc


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key in new-GT receipt/source')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf8'), object_pairs_hook=unique)


def audit_new_gt(config):
    """规范instance_map是唯一目标；provenance仅审计，补区与原覆盖等权。"""
    options = config['backend_adaptation']
    if options['expected_manual_sources'] != 32:
        raise ValueError('Exactly 32 allowed manual sources are required')
    dataset = CanonicalBackendDataset(project_path(config, config['paths']['raw_data_dir']),
        project_path(config, options['completed_gt_dir']), image_size=1024, affinity_grid=512)
    if len(dataset) != 32:
        raise ValueError('Actual canonical manual source count is not 32')
    directory = dataset.completed_gt_dir
    gt_relative = relative_path(config, directory)
    manifest_path = directory / 'manifest.json'
    manifest_sha = file_sha256(manifest_path)
    manifest = read_json(manifest_path)
    stems = [p.stem for p in dataset.samples]
    if (manifest.get('format') != 'manual_target_v2' or manifest.get('version') != 1
            or manifest.get('sample_count') != 32 or sorted(manifest.get('samples', [])) != sorted(stems)
            or manifest.get('max_fill_distance_native') != 8. or manifest.get('blur_sigma') != 1.2):
        raise ValueError('Approved manual_target_v2/v1 radius8 completion manifest differs')
    for suffix in ('_gt.npz', '_class.json'):
        if {p.name for p in directory.glob('*' + suffix)} != {stem + suffix for stem in stems}:
            raise ValueError('New-GT file cohort differs from the exact canonical source set')
    totals = dict(pixels=0, completed_known=0, original_covered=0, filled=0,
                  residual_unknown=0, instances=0, completion_only_sources=0)
    npz_cohort, class_cohort, rows = hashlib.sha256(), hashlib.sha256(), []
    for image_path in dataset.samples:
        label_path = directory / (image_path.stem + '_gt.npz')
        class_path = directory / (image_path.stem + '_class.json')
        identity_path = image_path.with_suffix('.json')
        paths = (image_path, identity_path, label_path, class_path)
        before = {relative_path(config, p): file_sha256(p) for p in paths}
        shape = read_rgb(str(image_path)).shape[:2]
        ids, lookup = load_completed_semantic_source(label_path, shape)
        classes = read_json(class_path)
        positive = ids > 0
        present = {str(int(i)) for i in np.unique(ids[positive])}
        if (set(classes) != present or any(type(v) is not int or v not in (0, 1) for v in classes.values())):
            raise ValueError('New-GT positive IDs require an exact integer 0/1 class mapping')
        with np.load(label_path, allow_pickle=False) as blob:
            required = {'instance_map', 'original_covered', 'filled', 'residual_unknown'}
            if not required.issubset(blob.files):
                raise ValueError('New-GT completion provenance fields are incomplete')
            masks = {}
            for key in ('original_covered', 'filled', 'residual_unknown'):
                value = blob[key]
                if value.shape != ids.shape or not np.all((value == 0) | (value == 1)):
                    raise ValueError('New-GT provenance mask must be binary and match the target shape')
                masks[key] = value.astype(bool)
            original, filled, unknown = (masks[k] for k in ('original_covered', 'filled', 'residual_unknown'))
            if (not np.array_equal(unknown, ~positive) or (original & ~positive).any()
                    or not np.array_equal(filled, positive & ~original)):
                raise ValueError('New-GT filled/unknown relationship is inconsistent; no automatic correction')
        for path in paths:
            if file_sha256(path) != before[relative_path(config, path)]:
                raise RuntimeError('New-GT or source identity changed during audit')
        for path, cohort in ((label_path, npz_cohort), (class_path, class_cohort)):
            cohort.update(path.name.encode('utf8'))
            cohort.update(bytes.fromhex(before[relative_path(config, path)]))
        stats = dict(pixels=ids.size, completed_known=int(positive.sum()), original_covered=int(original.sum()),
            filled=int(filled.sum()), residual_unknown=int(unknown.sum()), instances=len(present))
        for key, value in stats.items():
            totals[key] += value
        totals['completion_only_sources'] += int(filled.any())
        rows.append(dict(source=image_path.stem, shape=list(shape), files_sha256=before, counts=stats))
    if file_sha256(manifest_path) != manifest_sha:
        raise RuntimeError('Completion manifest changed during audit')
    summary = manifest.get('summary', {})
    expected = dict(num_images=32, total_pixels=totals['pixels'], filled_pixels=totals['filled'],
        residual_unknown_pixels=totals['residual_unknown'],
        original_uncovered_pixels=totals['pixels']-totals['original_covered'])
    if any(summary.get(k) != value for k, value in expected.items()):
        raise ValueError('Completion manifest summary no longer matches the 32 target files')
    for key, value in [('original_coverage', totals['original_covered']/totals['pixels']),
                       ('completed_coverage', totals['completed_known']/totals['pixels'])]:
        if not isinstance(summary.get(key), (int, float)) or not math.isclose(summary[key], value, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Completion manifest coverage does not match actual labels')
    return dict(passed=True, format='affinity_newgt_audit_v1', manual_sources=32,
        raw_data_dir=relative_path(config, dataset.data_dir), completed_gt_dir=gt_relative,
        canonical_target='*_gt.npz.instance_map + adjacent *_class.json',
        source_identity='CanonicalBackendDataset image/LabelMe pairs; original polygons not read as targets',
        supervision_support='after existing crop/letterbox/augmentation: affinity_valid_content & (affinity_instance_map>0)',
        original_covered_is_supervision_mask=False, filled_and_original_equal_weight=True,
        unknown_rule='residual_unknown == (instance_map==0); padding/unknown/outside pairs ignored',
        manifest_sha256=manifest_sha,
        completion_recipe={k: manifest[k] for k in ('format', 'version', 'source', 'max_fill_distance_native', 'blur_sigma')},
        cohort_hash_algorithm='sorted canonical source order; UTF8 file basename followed by binary file SHA256',
        npz_cohort_sha256=npz_cohort.hexdigest(), class_cohort_sha256=class_cohort.hexdigest(),
        totals=totals, completed_coverage=totals['completed_known']/totals['pixels'],
        original_coverage=totals['original_covered']/totals['pixels'], samples=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train/affinity_balance_mixed.yaml')
    parser.add_argument('--output', required=True, help='Private receipt inside ignored output/ or outputs/')
    args = parser.parse_args()
    config = load_config(args.config)
    target = Path(project_path(config, args.output))
    relative = Path(relative_path(config, target))
    if relative.parts[0] not in {'output', 'outputs'} or target.exists():
        raise ValueError('Write a fresh receipt only inside ignored output/ or outputs/')
    receipt = audit_new_gt(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({k: receipt[k] for k in ('passed', 'manual_sources', 'completed_gt_dir',
        'npz_cohort_sha256', 'class_cohort_sha256', 'totals')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
