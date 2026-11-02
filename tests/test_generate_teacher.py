# -*- coding: utf-8 -*-
"""教师生成adapter的CPU合同；合成32源，不加载真实模型或赛题数据。"""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from repro import generate_teacher as teacher


@pytest.fixture
def context(tmp_path):
    raw, original, models = (tmp_path / name for name in ('raw', 'original', 'models'))
    for path in (raw, original, models):
        path.mkdir()
    rows = []
    for index in range(32):
        name = f'train_{index:03d}'
        image = raw / (name + '.png')
        Image.fromarray(np.full((40, 64, 3), 128, np.uint8)).save(image)
        labelme = image.with_suffix('.json')
        labelme.write_text(json.dumps(dict(imageHeight=40, imageWidth=64)), encoding='utf-8')
        ids = np.ones((40, 64), np.uint16)
        ids[:, 32:] = 2
        gt, classes = original / (name + '_gt.npz'), original / (name + '_class.json')
        np.savez_compressed(gt, instance_map=ids, original_covered=np.ones(ids.shape, bool),
                            filled=np.zeros(ids.shape, bool), residual_unknown=np.zeros(ids.shape, bool))
        classes.write_text('{"1":0,"2":1}', encoding='utf-8')
        rows.append(dict(source=name, image_sha256=teacher.sha(image), labelme_sha256=teacher.sha(labelme),
                         gt_sha256=teacher.sha(gt), classes_sha256=teacher.sha(classes)))
    whitelist = tmp_path / 'whitelist.json'
    teacher.write(whitelist, dict(format=teacher.WHITELIST_FORMAT, rows=rows))
    naf = tmp_path / 'naf.pt'
    naf.write_bytes(b'CPU-test-no-model-loaded')
    return dict(format=teacher.CONTEXT_FORMAT, project_root=str(teacher.ROOT), output_dir=str(tmp_path / 'teacher'),
        raw_dir=str(raw), original_gt_dir=str(original), manual_whitelist=str(whitelist),
        manual_whitelist_sha256=teacher.sha(whitelist), naf_checkpoint=str(naf), naf_checkpoint_sha256=teacher.sha(naf),
        model_dir=str(models), config=str(teacher.ROOT / 'config/inference/area_floor_200.yaml'))


def fake_C_run(context):
    """只验证adapter接线；C真实算术由test_infer_compact独立覆盖。"""
    def run(args):
        from tools import infer_compact as compact
        assert args.geometry_mode == 'C' and args.global_noise == 'zero' and not args.capture_names
        rows = teacher.source_rows(context)
        files = compact.entry.input_files(args.input_dir, args.only, args.dataset)
        assert [p.stem for p in files] == [row['source'] for row in rows]
        output = Path(args.output_dir)
        (output / 'results').mkdir(parents=True)
        cfg = compact.entry.load_config(args.config)
        cfg['submission']['dataset'] = teacher.DATASET
        cfg = compact.zero.execution_config(cfg)
        images, zero = [], []
        for index, row in enumerate(rows):
            name = row['source']
            assert compact.entry.source_seed(name, teacher.DATASET) == 314159 + index
            ids = np.ones(row['shape'], np.uint16)
            ids[:, 32:] = 2
            ids[:, 32] = 0
            png = output / 'results' / (name + '_inst.png')
            assert cv2.imwrite(str(png), ids)
            classes = output / 'results' / (name + '_class.json')
            teacher.write(classes, {'1': 1, '2': 0})
            images.append(dict(source=name, shape=row['shape'], input_sha256=row['hashes']['image_sha256'],
                full_source_index=index, seed=314159 + index, png_sha256=teacher.sha(png), classes_sha256=teacher.sha(classes)))
            zero.append(dict(source=name, global_zero_calls=1, local_zero_calls=1, random_noise_generated=False,
                calls=[dict(q_sample_calls=1, noise_nonzero=0, initial_equals_condition=True)]))
        counts = dict(marker_stage_calls=32, reflect_marker_calls=32, S0_filter_calls=32, actual_WS_calls=32,
                      temporary_semantic_vote_calls=0, raw_revote_calls=32, final200_calls=32)
        report = dict(complete=True, frozen_models_unchanged=True, naf_complete=True, naf_frozen_unchanged=True,
            parameter_count=102583318, no_optimizer_steps=True,
            geometry_mode='C', dataset=teacher.DATASET, dataset_count=32, config=cfg, config_sha256=teacher.sha(args.config),
            models=dict(backend={'sha256': teacher.BACKEND_SHA}, restorer={'sha256': teacher.RESTORER_SHA},
                boundary_naf={'model_sha256': teacher.NAF_STATE_SHA, 'checkpoint_sha256': context['naf_checkpoint_sha256']}),
            model_states={}, local_restoration=dict(mode='zero', model='restorer', candidate='N1'),
            seed_support='original-zero', global_zero_calls=32, actual_compact_counts=counts, images=images, zero_noise_images=zero)
        teacher.write(output / 'audit/report.json', report)
        teacher.write(output / 'audit/compact_report.json', report)
        return report
    return run


def generate_fake(context, monkeypatch):
    from tools import infer_compact
    monkeypatch.setattr(teacher, 'validate_teacher_models', lambda context: None)
    monkeypatch.setattr(infer_compact, 'run', fake_C_run(context))
    return teacher.generate(context)


def test_full_generate_convert_check_preserves_old_GT_and_teacher_schema(context, monkeypatch):
    source = teacher.source_rows(context)
    result = generate_fake(context, monkeypatch)
    assert result['complete'] and result['samples'] == 32 and result['npz_prediction_arrays_exact']
    assert result['naf_positive_supervision_available'] is False  # 原C零缝不会偷偷填成NAF正例。
    assert teacher.source_rows(context) == source
    manifest = teacher.read(result['manifest'])
    assert manifest['format'] == 'teacher_supervision_v1' and manifest['source_kind'] == 'model_teacher'
    assert manifest['sample_count'] == 32 and len(manifest['samples']) == 32
    first = manifest['rows'][0]
    with np.load(first['gt']) as values:
        assert set(values.files) == {'instance_map', 'teacher_known', 'teacher_unknown'}
        assert values['instance_map'].dtype == np.uint16
        assert np.array_equal(values['teacher_known'], values['instance_map'] > 0)
        assert np.array_equal(values['teacher_unknown'], ~values['teacher_known'])
    summary = teacher.read(Path(context['output_dir']) / 'diagnostic_summary.json')
    assert summary['native_totals']['zero_four_neighbors_two_distinct_IDs'] > 0
    assert summary['images'][0]['affinity512']['teacher_valid_edges'] > 0
    assert summary['images'][0]['affinity512']['teacher_different_edges'] == 0  # 合成小图被放大后，0缝也可能隔开长程边。
    assert summary['images'][0]['phase']['P_to_F_pixels'] > 0
    assert summary['images'][0]['phase']['F_to_P_pixels'] > 0
    assert len(list(Path(context['output_dir'], 'gallery').glob('*.png'))) == 4
    assert teacher.check(context)['manifest_sha256'] == result['manifest_sha256']
    with pytest.raises(ValueError, match='fresh output'):
        teacher.generate(context)
    with pytest.raises(ValueError, match='already exist'):
        teacher.convert(context)


def test_zero_seams_are_diagnostic_only_and_NAF_known_pairs_can_be_empty():
    ids = np.ones((40, 64), np.uint16)
    ids[:, 32:] = 2
    direct = teacher.native_topology(ids)
    ids[:, 32] = 0
    zero = teacher.native_topology(ids)
    assert direct['native_four_neighbor_different_pairs'] == 40
    assert direct['naf_valid_boundary_pixels'] > 0
    assert zero['native_four_neighbor_different_pairs'] == zero['naf_valid_boundary_pixels'] == 0
    assert zero['zero_four_neighbors_two_distinct_IDs'] == 40
    assert np.all(ids[:, 32] == 0)


def test_training_dispatch_is_scoped_and_legacy_test_contract_unchanged(context):
    from tools import infer_semifinal as entry
    rows = teacher.source_rows(context)
    original_inputs, original_contract = entry.input_files, entry.dataset_contract
    with pytest.raises(ValueError):
        entry.source_index(rows[0]['source'], 'semifinal_150')
    with teacher.training_contract(entry, context, rows):
        assert entry.source_index(rows[0]['source'], teacher.DATASET) == 0
        assert entry.source_index('test_169', 'semifinal_150') == 0
        with pytest.raises(ValueError, match='Unwhitelisted'):
            entry.source_index('test_169', teacher.DATASET)
        with pytest.raises(ValueError, match='complete sorted32'):
            entry.input_files(context['raw_dir'], [rows[0]['source']], teacher.DATASET)
    assert entry.input_files is original_inputs and entry.dataset_contract is original_contract


@pytest.mark.parametrize('kind', ['test_name', 'image_bytes', 'labelme_bytes', 'extra_test'])
def test_whitelist_rejects_test_and_changed_sources(context, kind):
    path = Path(context['manual_whitelist'])
    if kind == 'test_name':
        document = teacher.read(path)
        document['rows'][0]['source'] = 'test_169'
        teacher.write(path, document, exclusive=False)
        context['manual_whitelist_sha256'] = teacher.sha(path)
    elif kind == 'extra_test':
        Path(context['raw_dir'], 'test_169.png').write_bytes(b'never-read')
    else:
        suffix = '.png' if kind == 'image_bytes' else '.json'
        Path(context['raw_dir'], 'train_000' + suffix).write_bytes(b'changed')
    with pytest.raises(ValueError):
        teacher.source_rows(teacher.checked_context(context))


def test_manifest_and_target_tampering_is_rejected(context, monkeypatch):
    result = generate_fake(context, monkeypatch)
    manifest = teacher.read(result['manifest'])
    labels = Path(manifest['rows'][0]['classes'])
    labels.write_text('{"1":0,"2":0}', encoding='utf-8')
    with pytest.raises(ValueError, match='asset changed'):
        teacher.check(context)


def test_prediction_class_contract_disallows_unknown_or_bool_classes():
    ids = np.ones((20, 30), np.uint16)
    for classes in ({}, {'1': True}, {'1': 1, '2': 0}, {'01': 1}):
        with pytest.raises(ValueError):
            teacher.prediction(ids, classes, ids.shape)
