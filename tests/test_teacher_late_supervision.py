# -*- coding: utf-8 -*-
"""教师32源后期兼容：只用合成小数组，验证unknown和来源不冒充人工。"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.backend_adaptation import CanonicalBackendDataset
from data.boundary_naf import load_processed_binary
from data.boundary_restore import known_instance_boundary, reliable_region
from data.semantic_targets import load_completed_semantic_source, semantic_targets_from_instances
from repro import current_naf
from utils.affinity_loss import build_affinity_targets_torch


def make_cohort(tmp_path, count=1, *, zero_seam=False):
    raw, target = tmp_path / 'raw', tmp_path / 'teacher'
    raw.mkdir(); target.mkdir()
    rows = []
    for i in range(count):
        name = f'train_{i:03d}'
        image = raw / (name + '.png')
        assert cv2.imwrite(str(image), np.full((32, 48, 3), 110, np.uint8))
        image.with_suffix('.json').write_text('{}', encoding='utf-8')
        ids = np.ones((32, 48), np.uint16); ids[:, 24:] = 2
        if zero_seam:
            width = 2 if zero_seam is True else int(zero_seam)
            ids[:, 24:24+width] = 0
        else:
            ids[8:11, 8:11] = 0
        gt, classes = target / (name + '_gt.npz'), target / (name + '_class.json')
        np.savez_compressed(gt, instance_map=ids, teacher_known=ids > 0, teacher_unknown=ids == 0)
        classes.write_text('{"1":0,"2":1}', encoding='utf-8')
        rows.append(dict(source=name, image=str(image), gt=str(gt), classes=str(classes), shape=list(ids.shape),
            hashes={key + '_sha256': current_naf.file_sha(path) for key, path in
                    [('image', image), ('gt', gt), ('classes', classes)]}))
    current_naf.write(target / 'manifest.json', dict(format='teacher_supervision_v1', source_kind='model_teacher',
        sample_count=count, samples=[row['source'] for row in rows], rows=rows))
    return raw, target, rows


def context_for(tmp_path, raw, target):
    run = tmp_path / 'run'; run.mkdir()
    artifacts = {name: str(run / name) for name in ('backend', 'restorer', 'n1_cache', 'naf')}
    artifacts['naf_export'] = str(run / 'export/model/naf.pt')
    for name in ('backend', 'restorer'):
        Path(artifacts[name]).write_bytes(('student_' + name).encode())
    return dict(format='current_training_context_v1', project_root=str(current_naf.ROOT), run_dir=str(run),
        settings={}, assets=dict(raw_dir=str(raw), completed_gt_dir=str(target),
            completed_gt_manifest=str(target / 'manifest.json'), naf_target_manifest=None,
            sam2_repo=str(tmp_path / 'sam2')), artifacts=artifacts)


def add_student_cache(context, rows, target_manifest):
    directory = Path(context['artifacts']['n1_cache']); directory.mkdir()
    files = []
    for row in rows:
        path = directory / (row['source'] + '.npz')
        # 故意与teacher边界不同，确保目标不被用来替代学生输入。
        boundary = np.full(row['shape'], .8, np.float32)
        np.savez_compressed(path, boundary=boundary)
        files.append(dict(source=row['source'], shape=row['shape'], heat_path=path.name,
                          heat_sha256=current_naf.file_sha(path), **row['hashes']))
    current_naf.write(directory / 'manifest.json', dict(format='boundary_naf_real_heatmaps_v1',
        complete=True, no_test_training=True, frozen_models_unchanged=True,
        local_restoration=dict(mode='zero', model='restorer'), models=current_naf.model_files(context),
        processed_gt_manifest_sha256=current_naf.file_sha(target_manifest), files=files))


def test_teacher_semantic_readout_keeps_ID0_ignore_and_legacy_core_default(tmp_path):
    _, target, rows = make_cohort(tmp_path)
    ids, lookup = load_completed_semantic_source(rows[0]['gt'], (32, 48))
    assert ids.dtype == np.int32 and lookup.dtype == np.float32
    semantic, boundary = semantic_targets_from_instances(ids, lookup, known_boundaries_only=True)
    assert np.array_equal(semantic < 0, ids == 0)
    np.testing.assert_array_equal(boundary > 0, known_instance_boundary(ids))
    small = np.array([[1, 1, 0, 2]], np.int32)
    _, old = semantic_targets_from_instances(small, lookup)
    _, teacher = semantic_targets_from_instances(small, lookup, known_boundaries_only=True)
    assert old.any() and not teacher.any()
    with np.load(rows[0]['gt']) as source:
        assert not {'original_covered', 'filled', 'residual_unknown'} & set(source.files)


def test_R1_teacher_targets_override_original_annotation_and_zero_pairs_ignore(tmp_path):
    raw, target, _ = make_cohort(tmp_path)
    dataset = CanonicalBackendDataset(raw, target, image_size=48, affinity_grid=48)
    sample = dataset[0]
    assert dataset.supervision_source_kind == sample['supervision_source_kind'] == 'model_teacher'
    ids = sample['semantic_instance_map']
    assert torch.equal(sample['semantic_target'][0] < 0, ids == 0)
    assert not sample['uncovered_boundary_source']
    affinity_ids = sample['instance_map']
    target_edges, valid, uncovered = build_affinity_targets_torch(affinity_ids[None],
        sample['valid_content'][None], offsets=[(0, 1)],
        uncovered_as_boundary=sample['uncovered_boundary_source'].reshape(1), return_uncovered_mask=True)
    expected = (affinity_ids[:, :-1] > 0) & (affinity_ids[:, 1:] > 0)
    assert torch.equal(valid[0, 0, :, :-1], expected) and not uncovered.any()
    assert not target_edges[0, 0][valid[0, 0] & (affinity_ids == 0)].any()


def test_NAF_teacher_boundary_and_provenance_do_not_fabricate_manual_pixels(tmp_path):
    _, _, rows = make_cohort(tmp_path)
    with np.load(rows[0]['gt']) as source:
        ids = source['instance_map']
    boundary, valid, metadata = load_processed_binary(rows[0]['gt'], rows[0]['classes'], valid_margin=4)
    np.testing.assert_array_equal(boundary, known_instance_boundary(ids))
    np.testing.assert_array_equal(valid, reliable_region(ids, 4))
    assert not valid[ids == 0].any()
    assert metadata['source_kind'] == 'model_teacher' and metadata['boundary_pixels'] > 0
    assert metadata['boundary_pixels'] + metadata['valid_negative_pixels'] == metadata['valid_pixels']
    assert not {'original_pixels', 'filled_pixels', 'original_covered'} & set(metadata)


def test_teacher_zero_seams_produce_no_invented_NAF_boundary_and_block_training(tmp_path):
    raw, target, rows = make_cohort(tmp_path, 32, zero_seam=True)
    context = context_for(tmp_path, raw, target)
    path, wrapper = current_naf.prepare_targets(context)
    stats = wrapper['target_statistics']
    assert stats['source_kind'] == 'model_teacher'
    assert stats['valid_positive_pixels'] == 0 and stats['valid_negative_pixels'] > 0
    assert path.is_file() and all(row['provenance']['raw_boundary_pixels'] == 0 for row in wrapper['rows'])
    with pytest.raises(ValueError, match='全部有效正边界为0'):
        current_naf.training_config(context)


def test_teacher_NAF_native_T8_dataset_binds_student_cache_and_teacher_targets(tmp_path):
    raw, target, rows = make_cohort(tmp_path, 32)
    context = context_for(tmp_path, raw, target)
    # early和late均显式优先teacher资产，不能因旧completed路径仍在context而混回r8。
    context['assets'].update(teacher_gt_dir=str(target), teacher_gt_manifest=str(target / 'manifest.json'),
                             completed_gt_dir=str(tmp_path / 'old_r8_unused'),
                             completed_gt_manifest=str(tmp_path / 'old_r8_unused/manifest.json'))
    original = {row['source']: current_naf.file_sha(row['gt']) for row in rows}
    path, wrapper = current_naf.prepare_targets(context)
    assert wrapper['parent_format'] == 'teacher_supervision_v1'
    add_student_cache(context, rows, path)
    cfg = current_naf.training_config(context)
    from train_boundary_rgb import dataset_for
    dataset = dataset_for(cfg, 1)
    assert dataset.identity['model_predictions_as_target'] is True
    assert dataset.identity['target_source_kind'] == 'model_teacher'
    assert dataset.target_identity['model_predictions_as_target'] is True
    assert dataset.identity['real_input_models']['backend'] == current_naf.file_sha(context['artifacts']['backend'])
    assert dataset.input_boundaries['train_000'].dtype == bool
    # 全.8学生缓存经整图细化/T8的结果，不是teacher的两粒交界目标。
    assert not np.array_equal(dataset.input_boundaries['train_000'], dataset.cache['train_000'][0])
    assert all(current_naf.file_sha(row['gt']) == original[row['source']] for row in rows)
    dataset.verify_sources()


def test_teacher_partial_or_mixed_provenance_never_falls_back_to_manual(tmp_path):
    raw, target, rows = make_cohort(tmp_path)
    p = Path(rows[0]['gt'])
    with np.load(p) as saved:
        ids = saved['instance_map'].copy()
    np.savez_compressed(p, instance_map=ids, teacher_known=ids > 0)
    with pytest.raises(ValueError, match='teacher target requires'):
        load_processed_binary(p, rows[0]['classes'])
    np.savez_compressed(p, instance_map=ids, teacher_known=ids > 0, teacher_unknown=ids == 0,
                        original_covered=ids > 0)
    with pytest.raises(ValueError, match='must not claim'):
        load_completed_semantic_source(p, ids.shape)


def test_R1_config_explicit_teacher_assets_replace_old_r8_target_paths(tmp_path):
    raw, target, _ = make_cohort(tmp_path, 32)
    context = context_for(tmp_path, raw, target)
    run = Path(context['run_dir'])
    for key in ('joint', 'semantic', 'affinity', 'd5a'):
        p = run / key / ('epoch_060.pt' if key == 'd5a' else 'final.pt')
        p.parent.mkdir(); p.write_bytes(key.encode())
        context['artifacts'][key] = str(p)
    context['artifacts']['r1'] = str(run / 'r1/final.pt')
    sam, repo = tmp_path / 'sam64', tmp_path / 'sam2'
    sam.mkdir(); repo.mkdir()
    (sam / 'manifest.jsonl').write_text('{}\n', encoding='utf-8')
    for key in ('backbone', 'sam2_authorization_doc'):
        p = tmp_path / key; p.write_text('{}', encoding='utf-8')
        context['assets'][key] = str(p)
    context['assets'].update(sam2_geometry_dir=str(sam), sam2_source_dir=str(raw),
        sam2_repo=str(repo), monitor_dir=str(raw), teacher_gt_dir=str(target),
        teacher_gt_manifest=str(target / 'manifest.json'),
        completed_gt_dir=str(tmp_path / 'old_r8_unused'),
        completed_gt_manifest=str(tmp_path / 'old_r8_unused/manifest.json'))
    (run / 'configs').mkdir()
    (run / 'configs/d5a.yaml').write_text('rgb_restoration: {}\n', encoding='utf-8')
    from repro.current_backend import prepare_r1_config
    cfg = prepare_r1_config(context)['backend_adaptation']
    assert cfg['completed_gt_dir'] == str(target)
    assert cfg['supervision_source']['manual_GT_targets_used'] is False
    assert cfg['supervision_source']['manifest_sha256'] == current_naf.file_sha(target / 'manifest.json')


def test_explicit_separator_NAF_uses_geometry_domain_while_semantic_and_affinity_ignore_zero(tmp_path):
    raw, target, rows = make_cohort(tmp_path, zero_seam=1)
    p = rows[0]['gt']; before = current_naf.file_sha(p)
    ids, lookup = load_completed_semantic_source(p, (32, 48))
    boundary, valid, meta = load_processed_binary(p, rows[0]['classes'],
        teacher_boundary_policy='explicit_separators_v1')
    assert boundary[:, 24].all() and valid[4:-4, 24].all()
    assert meta['separator_pixels'] == 32 and meta['valid_boundary_pixels'] > 0
    assert meta['boundary_policy'] == 'explicit_separators_v1'
    semantic, _ = semantic_targets_from_instances(ids, lookup, known_boundaries_only=True)
    assert np.all(semantic[:, 24] == -1)
    _, affinity_valid = build_affinity_targets_torch(torch.from_numpy(ids.astype(np.int64))[None],
        torch.ones((1, 1, 32, 48), dtype=torch.bool), offsets=[(0, 1)])
    assert not affinity_valid[0, 0, :, 23:25].any()
    assert current_naf.file_sha(p) == before


def test_explicit_separator_does_not_supervise_wide_unknown_or_same_ID_hole(tmp_path):
    _, _, rows = make_cohort(tmp_path, zero_seam=3)
    boundary, valid, meta = load_processed_binary(rows[0]['gt'], rows[0]['classes'],
        teacher_boundary_policy='explicit_separators_v1')
    assert not boundary.any() and not valid[:, 24:27].any() and meta['separator_pixels'] == 0
    p = Path(rows[0]['gt'])
    with np.load(p) as source:
        ids = source['instance_map'].copy()
    ids[ids > 0] = 1
    ids[:, 24:27] = 1
    ids[10:20, 10] = 0  # 两侧同ID的窄unknown洞，不是分界。
    np.savez_compressed(p, instance_map=ids, teacher_known=ids > 0, teacher_unknown=ids == 0)
    Path(rows[0]['classes']).write_text('{"1":0}', encoding='utf-8')
    boundary, valid, meta = load_processed_binary(p, rows[0]['classes'],
        teacher_boundary_policy='explicit_separators_v1')
    assert not boundary.any() and meta['separator_pixels'] == 0 and not valid[10:20, 10].any()


def test_explicit_separator_policy_reaches_real_width_dataset_and_tolerant_loss_gradient(tmp_path):
    raw, target, rows = make_cohort(tmp_path, 32, zero_seam=1)
    context = context_for(tmp_path, raw, target)
    context['settings']['teacher_boundary_policy'] = 'explicit_separators_v1'
    path, wrapper = current_naf.prepare_targets(context)
    assert wrapper['teacher_boundary_policy'] == 'explicit_separators_v1'
    assert wrapper['target_statistics']['valid_positive_pixels'] > 0
    assert wrapper['target_statistics']['separator_pixels'] == 32*32
    add_student_cache(context, rows, path)
    cfg = current_naf.training_config(context)
    assert cfg['boundary_naf']['data']['teacher_boundary_policy'] == 'explicit_separators_v1'
    from train_boundary_rgb import dataset_for
    dataset = dataset_for(cfg, 1)
    assert dataset.identity['teacher_boundary_policy'] == 'explicit_separators_v1'
    assert 'separator' in dataset.identity['policy'] and 'affinity' in dataset.identity['policy']
    assert dataset.identity['target_valid_boundary_pixels'] > 0
    # 使用真实NAF架构的小宽度微批，检查原容差loss再侵蚀3之后仍有正例与有限梯度。
    from models.boundary_rgb import BoundaryRGBNAFNet
    from utils.boundary_rgb_training import tolerant_boundary_loss
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = BoundaryRGBNAFNet(width=4, encoder_blocks=[1, 1], middle_blocks=1,
                                 decoder_blocks=[1, 1], input_mode='boundary')
        binary, valid = dataset.cache['train_000']
        gt = torch.from_numpy(binary.copy())[None, None].float()
        valid = torch.from_numpy(valid.copy())[None, None]
        inputs = torch.zeros_like(gt); inputs[..., 10] = 1
        logits = model.forward_logits(inputs)
        loss, stats = tolerant_boundary_loss(logits, gt, valid, tolerance=2)
        assert stats['positive_points'] > 0 and stats['negative_pixels'] > 0
        coverage_gradient = torch.autograd.grad(stats['coverage'], logits, retain_graph=True)[0]
        assert torch.isfinite(coverage_gradient).all() and coverage_gradient.abs().sum() > 0
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    finally:
        torch.set_num_threads(old_threads)
