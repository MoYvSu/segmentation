# -*- coding: utf-8 -*-
"""缓存 ZIP 改末端面积的安全契约；全部为合成图和合成来源快照。"""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from tools.package_final_area import (SOURCES, archive_members, decode_prediction,
    package_archive, sha, write_json)


@pytest.fixture
def cached(tmp_path):
    config_dir = tmp_path / 'snapshot' / 'config' / 'inference'
    config_dir.mkdir(parents=True)
    base = config_dir / 'base.yaml'
    base.write_text('inference:\n  min_instance_area: 50\n', encoding='utf8')
    config = config_dir / 'source.yaml'
    config.write_text('_base: base.yaml\n', encoding='utf8')
    manifest = tmp_path / 'source.json'
    write_json(manifest, {p.relative_to(tmp_path / 'snapshot').as_posix(): sha(p) for p in (base, config)})
    ids = np.zeros((30, 44), np.uint16)
    ids[1:29, 1:25] = 100
    ids[8:18, 26:32] = 5  # 60px：源50阈值下保留，候选100阈值下合并。
    ids[1:29, 35:43] = 65535
    _, png_array = cv2.imencode('.png', ids)
    png = png_array.tobytes()
    labels = b'{"100":1,"5":0,"65535":0}\n'
    input_zip = tmp_path / 'source.zip'
    with zipfile.ZipFile(input_zip, 'x') as archive:
        for source in SOURCES:
            archive.writestr('results/' + source + '_inst.png', png)
            archive.writestr('results/' + source + '_class.json', labels)
    report = tmp_path / 'package_audit.json'
    write_json(report, dict(complete=True, sha256=sha(input_zip), pairs=150, files=300, only_results=True,
        CRC_all_passed=True, member_sha256_exact=True, native_sizes_uint16_and_ID_class_pairs_exact=True,
        files_sha=[dict(source=s, shape=list(ids.shape), png_sha256=hashlib.sha256(png).hexdigest(),
                        classes_sha256=hashlib.sha256(labels).hexdigest()) for s in SOURCES]))
    candidate = config_dir / 'candidate.yaml'
    candidate.write_text('_base: source.yaml\ninference:\n  seed_min_area: 50\n'
                         '  final_min_instance_area: 100\n  final_small_instance_action: neighbor_merge_v1\n', encoding='utf8')
    return dict(input_zip=input_zip, source_report=report, source_config=config, source_manifest=manifest,
                candidate_config=candidate, output_zip=tmp_path / 'candidate.zip', audit_dir=tmp_path / 'audit')


def test_complete_150_cached_candidate_validates_original_sizes_classes_and_external_audit(cached):
    original_sha = sha(cached['input_zip'])
    receipt = package_archive(**cached, seed_min_area=50, expected_input_sha256=original_sha)
    assert receipt['complete'] and receipt['pairs'] == 150 and receipt['files'] == 300
    assert receipt['seed_min_area'] == 50 and receipt['final_min_instance_area'] == 100
    assert receipt['source_area_policy']['final_min_instance_area'] == 50
    assert receipt['cpu_only'] and not receipt['model_forward_performed'] and not receipt['cached_seed_changed']
    assert receipt['merged_instances'] == 150 and sha(cached['input_zip']) == original_sha
    assert (cached['audit_dir'] / 'complete.json').exists()
    with zipfile.ZipFile(cached['output_zip']) as archive:
        members = archive_members(archive)
        assert len(members) == 300 and all(name.startswith('results/') for name in members.values())
        assert not any('audit' in name or 'complete' in name for name in archive.namelist())
        ids, classes = decode_prediction(archive.read(members['test_169_inst.png']),
            archive.read(members['test_169_class.json']), [30, 44], 'test_169')
        assert ids.dtype == np.uint16 and classes == {'100': 1, '65535': 0}
        assert np.all(ids[8:18, 26:32] == 100)


def test_50_identity_preserves_all_original_member_bytes_and_refuses_overwrite(cached):
    args = dict(cached)
    args.pop('candidate_config')
    receipt = package_archive(**args, final_min_instance_area=50)
    assert receipt['merged_instances'] == receipt['changed_pixels'] == 0
    with zipfile.ZipFile(cached['input_zip']) as before, zipfile.ZipFile(cached['output_zip']) as after:
        assert before.namelist() == after.namelist()
        assert all(before.read(name) == after.read(name) for name in before.namelist())
    with pytest.raises(ValueError, match='overwrite'):
        package_archive(**args, final_min_instance_area=50)


def test_wrong_source_sha_or_cached_seed_claim_is_rejected_before_output(cached):
    with pytest.raises(ValueError, match='SHA256'):
        package_archive(**cached, expected_input_sha256='0' * 64)
    with pytest.raises(ValueError, match='cannot change'):
        package_archive(**cached, seed_min_area=100)
    assert not cached['output_zip'].exists() and not cached['audit_dir'].exists()


def test_modified_source_config_and_candidate_seed_changes_are_rejected(cached):
    old = cached['source_config'].read_text(encoding='utf8')
    cached['source_config'].write_text(old + '# changed\n', encoding='utf8')
    with pytest.raises(ValueError, match='frozen config SHA'):
        package_archive(**cached)
    cached['source_config'].write_text(old, encoding='utf8')
    text = cached['candidate_config'].read_text(encoding='utf8').replace('seed_min_area: 50', 'seed_min_area: 75')
    cached['candidate_config'].write_text(text, encoding='utf8')
    with pytest.raises(ValueError, match='changes cached source seed'):
        package_archive(**cached)


@pytest.mark.parametrize('extra', ['  boundary_threshold: 0.9\n',
    'submission:\n  models:\n    backend:\n      file: changed.pt\n'])
def test_candidate_cannot_claim_unexecuted_upstream_threshold_or_model_changes(cached, extra):
    with cached['candidate_config'].open('a', encoding='utf8') as stream:
        stream.write(extra)
    with pytest.raises(ValueError, match='changes upstream settings'):
        package_archive(**cached)
    assert not cached['output_zip'].exists() and not cached['audit_dir'].exists()


@pytest.mark.parametrize('bad_name', ['../evil.txt', 'results/../../evil.txt', '/evil.txt',
                                   'results\\evil.txt', 'results/extra.json', 'results/test_319_inst.png'])
def test_extra_or_traversal_members_are_rejected_without_final_zip(cached, bad_name):
    with zipfile.ZipFile(cached['input_zip'], 'a') as archive:
        archive.writestr(bad_name, b'unused')
    value = json.loads(cached['source_report'].read_text(encoding='utf8'))
    value['sha256'] = sha(cached['input_zip'])
    write_json(cached['source_report'], value)
    with pytest.raises(ValueError):
        package_archive(**cached)
    assert not cached['output_zip'].exists() and not cached['output_zip'].with_suffix('.zip.partial').exists()
    assert (cached['audit_dir'] / 'failed.json').exists()


def test_wrong_native_shape_member_hash_or_class_contract_is_rejected(cached):
    report = json.loads(cached['source_report'].read_text(encoding='utf8'))
    report['files_sha'][0]['shape'] = [29, 44]
    write_json(cached['source_report'], report)
    with pytest.raises(ValueError, match='original dimensions'):
        package_archive(**cached)
    assert not cached['output_zip'].exists()


@pytest.mark.parametrize('labels', [b'{"1":0,"1":1}', b'{"1":true}', b'{"01":0}', b'{"2":0}'])
def test_duplicate_invalid_or_missing_ids_in_json_rejected(labels):
    ids = np.zeros((4, 5), np.uint16)
    ids[1, 1] = 1
    _, png = cv2.imencode('.png', ids)
    with pytest.raises(ValueError):
        decode_prediction(png.tobytes(), labels, [4, 5], 'synthetic')


def test_png_must_be_native_16_bit_grayscale():
    for image in (np.zeros((4, 5), np.uint8), np.zeros((4, 5, 3), np.uint16)):
        _, png = cv2.imencode('.png', image)
        with pytest.raises(ValueError, match='16-bit single-channel'):
            decode_prediction(png.tobytes(), b'{}', [4, 5], 'synthetic')
