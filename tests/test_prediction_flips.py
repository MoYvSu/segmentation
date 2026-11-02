# -*- coding: utf-8 -*-
"""仅用合成uint16预测及真实ZIP验证翻转、几何诊断与组包合同。"""
import hashlib
import json
from pathlib import Path
import stat
import struct
import warnings
import zipfile

import cv2
import numpy as np
import pytest

from tools import analyze_prediction_flips as module


def result(root, source, ids, classes, *, class_bytes=None):
    root.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode('.png', np.asarray(ids))
    assert ok
    (root / (source + '_inst.png')).write_bytes(png.tobytes())
    data = json.dumps(classes).encode() if class_bytes is None else class_bytes
    (root / (source + '_class.json')).write_bytes(data)


def archive(path, folder, extra=None):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as z:
        for file in sorted(folder.iterdir()):
            z.writestr('results/' + file.name, file.read_bytes())
        if extra is not None:
            z.writestr(*extra)
    return path


def test_F_and_P_direction_common_foreground_and_background_are_separate(tmp_path):
    old = np.array([[1, 1, 2, 2, 0], [1, 1, 2, 2, 0]], np.uint16)
    new = np.array([[3, 3, 4, 4, 5], [3, 0, 4, 4, 5]], np.uint16)
    result(tmp_path / 'old', 'test_169', old, {'1': 1, '2': 0})
    result(tmp_path / 'new', 'test_169', new, {'3': 0, '4': 1, '5': 1})
    summary = module.analyze_predictions(tmp_path / 'old', tmp_path / 'new', tmp_path / 'analysis', 1)
    counts = summary['aggregate']['counts']
    assert counts['total_pixels'] == 10 and counts['common_foreground_pixels'] == 7
    assert counts['F_to_P_pixels'] == 3 and counts['P_to_F_pixels'] == 4
    assert counts['positive_to_zero_pixels'] == 1 and counts['zero_to_positive_pixels'] == 2
    assert summary['aggregate']['pixel_weighted_rates']['all_pixels']['any_FP_flip_pixels'] == .7
    assert summary['aggregate']['pixel_weighted_rates']['common_foreground']['any_FP_flip_pixels'] == 1
    assert summary['aggregate']['pixel_weighted_rates']['conditional_on_old_phase'] == {'F_to_P': .75, 'P_to_F': 1}
    assert (tmp_path / 'analysis/per_source/test_169.json').is_file()
    assert (tmp_path / 'analysis/per_source.csv').read_text().count('\n') == 2
    assert summary['no_GT'] and summary['predictions_unchanged']


def test_id_permutation_and_max_uint16_id_do_not_create_geometry_or_phase_changes(tmp_path):
    old = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], np.uint16)
    new = np.where(old == 1, 65535, 32000).astype(np.uint16)
    row = module.compare_predictions(old, {'1': 1, '2': 0}, new, {'65535': 1, '32000': 0})
    assert row['flip']['counts']['any_FP_flip_pixels'] == 0
    assert row['geometry']['partition_equivalent_up_to_relabeling']
    assert row['geometry']['boundary_symmetric_difference_pixels'] == 0
    assert row['geometry']['stable_correspondences']['count'] == 2


def test_equal_geometry_whole_region_class_flips_are_counted_in_both_directions():
    ids = np.array([[1, 1, 2, 2]], np.uint16)
    row = module.compare_predictions(ids, {'1': 1, '2': 0}, ids, {'1': 0, '2': 1})
    stable = row['geometry']['stable_correspondences']
    assert stable['count'] == 2 and stable['F_to_P'] == stable['P_to_F'] == 1
    assert stable['flip_rate'] == 1


def test_absorbed_whole_region_is_not_hidden_by_empty_stable_subset():
    old = np.ones((16, 16), np.uint16)
    old[:, 8:] = 2
    new = np.full_like(old, 9)
    row = module.compare_predictions(old, {'1': 1, '2': 0}, new, {'9': 0})
    assert row['flip']['counts']['F_to_P_pixels'] == 128
    assert row['geometry']['stable_correspondences']['count'] == 0
    assert row['geometry']['stable_correspondences']['flip_rate'] is None
    excluded = row['geometry']['excluded_correspondences']['old']
    assert excluded['count'] == excluded['absorbed_into_larger_region'] == 2
    assert excluded['dominant_phase_changed'] == 1
    assert row['geometry']['merged_new_regions'] == 1 and row['geometry']['split_old_regions'] == 0
    reverse = module.compare_predictions(new, {'9': 0}, old, {'1': 1, '2': 0})
    assert reverse['geometry']['split_old_regions'] == 1


def test_macro_and_pixel_weighted_denominators_are_distinct(tmp_path):
    for source, width, changed in [('test_169', 1, True), ('test_170', 3, False)]:
        ids = np.ones((1, width), np.uint16)
        result(tmp_path / 'old', source, ids, {'1': 1})
        result(tmp_path / 'new', source, ids, {'1': 0 if changed else 1})
    summary = module.analyze_predictions(tmp_path / 'old', tmp_path / 'new', tmp_path / 'analysis', 2)
    assert summary['aggregate']['macro_image_rates']['all_pixels']['any_FP_flip_pixels'] == .5
    assert summary['aggregate']['pixel_weighted_rates']['all_pixels']['any_FP_flip_pixels'] == .25


def test_all_background_has_no_false_P_class_and_undefined_stable_denominator(tmp_path):
    ids = np.zeros((3, 4), np.uint16)
    result(tmp_path / 'old', 'test_169', ids, {})
    result(tmp_path / 'new', 'test_169', ids, {})
    summary = module.analyze_predictions(tmp_path / 'old', tmp_path / 'new', tmp_path / 'analysis', 1)
    aggregate = summary['aggregate']
    assert aggregate['counts']['any_FP_flip_pixels'] == 0
    assert aggregate['composition']['baseline']['P']['pixels'] == 0
    assert aggregate['pixel_weighted_rates']['common_foreground']['any_FP_flip_pixels'] is None
    assert aggregate['macro_rate_valid_images']['common_foreground']['any_FP_flip_pixels'] == 0
    assert aggregate['stable_correspondences']['flip_rate'] is None


@pytest.mark.parametrize('kind', ['shape', 'source', 'uint8', 'RGB', 'missing_class', 'extra_class', 'bool_class', 'noncanonical_ID', 'duplicate_JSON'])
def test_invalid_prediction_contracts_rejected(tmp_path, kind):
    ids = np.ones((4, 5), np.uint16)
    result(tmp_path / 'old', 'test_169', ids, {'1': 1})
    new, name, classes, raw = ids.copy(), 'test_169', {'1': 1}, None
    if kind == 'shape': new = np.ones((5, 4), np.uint16)
    elif kind == 'source': name = 'test_170'
    elif kind == 'uint8': new = ids.astype(np.uint8)
    elif kind == 'RGB': new = np.repeat(ids[..., None], 3, axis=2)
    elif kind == 'missing_class': classes = {}
    elif kind == 'extra_class': classes['2'] = 0
    elif kind == 'bool_class': classes = {'1': True}
    elif kind == 'noncanonical_ID': classes = {'01': 1}
    elif kind == 'duplicate_JSON': raw = b'{"1":1,"1":0}'
    result(tmp_path / 'new', name, new, classes, class_bytes=raw)
    with pytest.raises(ValueError):
        module.analyze_predictions(tmp_path / 'old', tmp_path / 'new', tmp_path / 'analysis', 1)


@pytest.mark.parametrize('extra', [('audit.json', b'{}'), ('results/../audit.json', b'{}'), ('results/subdir/', b''), ('results/test_170_class.json', b'{}')])
def test_extra_or_unsafe_zip_members_rejected(tmp_path, extra):
    result(tmp_path / 'pred', 'test_169', np.ones((4, 5), np.uint16), {'1': 1})
    path = archive(tmp_path / 'bad.zip', tmp_path / 'pred', extra)
    with pytest.raises(ValueError):
        module.PredictionStore(path, 1)


def test_duplicate_and_symlink_zip_entries_rejected(tmp_path):
    result(tmp_path / 'pred', 'test_169', np.ones((4, 5), np.uint16), {'1': 1})
    duplicate = tmp_path / 'duplicate.zip'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        archive(duplicate, tmp_path / 'pred', ('results/test_169_class.json', b'{"1":1}'))
    with pytest.raises(ValueError, match='unique'):
        module.PredictionStore(duplicate, 1)
    info = zipfile.ZipInfo('results/test_169_inst.png')
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    symlink = tmp_path / 'symlink.zip'
    with zipfile.ZipFile(symlink, 'w') as z:
        z.writestr(info, b'target')
        z.writestr('results/test_169_class.json', b'{"1":1}')
    with pytest.raises(ValueError, match='regular'):
        module.PredictionStore(symlink, 1)


def test_zip_CRC_corruption_rejected_before_publishing_summary(tmp_path):
    result(tmp_path / 'pred', 'test_169', np.ones((4, 5), np.uint16), {'1': 1})
    path = archive(tmp_path / 'bad.zip', tmp_path / 'pred')
    with zipfile.ZipFile(path) as z:
        info = z.getinfo('results/test_169_class.json')
        start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    data = bytearray(path.read_bytes()); data[start] ^= 1; path.write_bytes(data)
    with module.PredictionStore(path, 1) as source:
        with pytest.raises(ValueError, match='CRC'):
            source.prediction('test_169')


def test_pack_copies_original_bytes_ignores_external_audit_and_checks_baseline_SHA(tmp_path):
    folder = tmp_path / 'candidate/results'
    result(folder, 'test_169', np.array([[1, 0, 65535]], np.uint16), {'1': 1, '65535': 0})
    (tmp_path / 'candidate/audit.json').write_text('{}')
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    path = tmp_path / 'submission.zip'
    receipt = module.pack_results(tmp_path / 'candidate', path, 1)
    assert receipt['complete'] and receipt['members'] == 2 and receipt['CRC_all_passed']
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist()) == {'results/' + k for k in before}
        assert all(z.read('results/' + name) == data for name, data in before.items())
        assert z.testzip() is None
    summary = module.analyze_predictions(path, folder, tmp_path / 'analysis', 1, receipt['sha256'])
    assert summary['aggregate']['counts']['any_FP_flip_pixels'] == 0
    with pytest.raises(ValueError, match='SHA256'):
        module.analyze_predictions(path, folder, tmp_path / 'wrong', 1, '0' * 64)
    with pytest.raises(FileExistsError):
        module.pack_results(folder, path, 1)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_fresh_outputs_and_prediction_directory_boundaries_enforced(tmp_path):
    folder = tmp_path / 'pred'
    result(folder, 'test_169', np.ones((3, 4), np.uint16), {'1': 1})
    with pytest.raises(ValueError, match='outside'):
        module.pack_results(folder, folder / 'submission.zip', 1)
    with pytest.raises(ValueError, match='outside'):
        module.analyze_predictions(folder, folder, folder / 'analysis', 1)
    with pytest.raises(ValueError, match='outside'):
        module.main(['--candidate', str(folder), '--expected-count', '1',
                     '--output-dir', str(folder / 'analysis'), '--pack-zip', str(tmp_path / 'pack.zip')])
    assert not (folder / 'analysis').exists() and not (tmp_path / 'pack.zip').exists()
    (folder / 'audit.json').write_text('{}')
    with pytest.raises(ValueError, match='extra'):
        module.PredictionStore(folder, 1)


def test_truncated_or_oversize_PNG_header_rejected_before_decode():
    with pytest.raises(ValueError, match='bounded PNG'):
        module.decode_prediction(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR', b'{}')
    header = bytearray(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' + struct.pack('>II', module.MAX_PIXELS, 2)
                       + bytes((16, 0, 0, 0, 0)) + b'0000')
    with pytest.raises(ValueError, match='pixel bound'):
        module.decode_prediction(header, b'{}')


def test_actual_150_pair_uint16_ZIP_and_default_CLI_count(tmp_path):
    folder = tmp_path / 'pred'
    ids = np.array([[0, 1, 65535]], np.uint16)
    for index in range(169, 319):
        result(folder, f'test_{index:03d}', ids, {'1': 1, '65535': 0})
    baseline = tmp_path / 'baseline.zip'
    receipt = module.pack_results(folder, baseline)
    assert receipt['source_count'] == 150 and receipt['members'] == 300
    candidate = tmp_path / 'candidate.zip'
    output = tmp_path / 'analysis'
    assert module.main(['--baseline', str(baseline), '--candidate', str(folder),
        '--baseline-sha256', receipt['sha256'], '--output-dir', str(output), '--pack-zip', str(candidate)]) == 0
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['source_count'] == 150 and summary['aggregate']['counts']['total_pixels'] == 450
    assert summary['aggregate']['counts']['any_FP_flip_pixels'] == 0
    assert len(list((output / 'per_source').glob('*.json'))) == 150
    packed = json.loads((output / 'package_audit.json').read_text(encoding='utf-8'))
    assert packed['members'] == 300 and packed['sha256'] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    with zipfile.ZipFile(candidate) as z:
        assert len(z.infolist()) == 300 and z.testzip() is None
        assert all(name.startswith('results/') and not name.endswith('/') for name in z.namelist())
