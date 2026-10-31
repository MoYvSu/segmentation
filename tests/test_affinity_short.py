# -*- coding: utf-8 -*-
"""仅融合方式消融的配置隔离、全图面积归因与既有控制缓存契约。"""
from copy import deepcopy
import json
import zipfile

import cv2
import numpy as np
import pytest

from tools import run_affinity_short as runner
from tools.analyze_affinity_connectivity import relation
from tools.analyze_affinity_sweep_test import changes, pixel_class_changes
from train_backend_adaptation import sha
from utils.patch_diagnostic import phase_description


def config_fixture():
    return dict(affinity_deployment=dict(fusion_mode='gated', short_reduction='top2',
                                       d2_weight=.25, boundary_threshold=.5),
                affinity_native=dict(overlap=.25),
                backend_adaptation=dict(restoration=dict(inference_seed=123)),
                inference=dict(semantic_vote_threshold=.5))


def test_short_config_changes_only_one_key_without_mutating_original():
    original = config_fixture()
    before = deepcopy(original)
    candidate = runner.short_config(original)
    assert original == before
    assert candidate['affinity_deployment']['fusion_mode'] == 'short'
    candidate['affinity_deployment']['fusion_mode'] = 'gated'
    assert candidate == original
    candidate['backend_adaptation']['restoration']['inference_seed'] += 1
    assert original == before


@pytest.mark.parametrize('section,key,value', [
    ('affinity_deployment', 'boundary_threshold', .6),
    ('affinity_deployment', 'short_reduction', 'mean'),
    ('affinity_deployment', 'd2_weight', .3),
    ('affinity_native', 'overlap', .5),
    ('inference', 'semantic_vote_threshold', .6)])
def test_short_config_rejects_any_extra_deployment_change(section, key, value):
    original = config_fixture()
    candidate = runner.short_config(original)
    candidate[section][key] = value
    with pytest.raises(RuntimeError, match='Only .*fusion_mode'):
        runner.require_short_config(original, candidate)


@pytest.mark.parametrize('section,key,value', [
    ('affinity_deployment', 'fusion_mode', 'short'),
    ('affinity_deployment', 'short_reduction', 'mean'),
    ('affinity_native', 'overlap', .5)])
def test_unregistered_control_fusion_or_window_is_rejected(section, key, value):
    config = config_fixture()
    config[section][key] = value
    with pytest.raises(RuntimeError, match='Registered gated'):
        runner.short_config(config)


def tiny_split():
    old = np.ones((40, 50), np.uint16)
    new = np.full(old.shape, 5, np.uint16)
    new[0] = 9
    return old, new, {'1': 1}, {'5': 1, '9': 1}


def test_tiny_child_trace_uses_new_area_dominance_and_count_closes():
    old, new, old_classes, new_classes = tiny_split()
    result = runner.overlap_mapping(old, new, old_classes, new_classes, 2000.)
    child = next(v for v in result['new_ferrite_regions'] if v['new_id'] == 9)
    assert child['area'] == 50
    assert child['origin'] == 'old_ferrite'
    assert child['dominant_old_id'] == 1
    assert child['maximum_old_overlap_fraction'] == 1
    assert child['matched_old_iou50_id'] is None
    assert child['under200'] and child['below_control_median_five_percent']
    assert not child['below_control_median_one_percent']
    parent = result['old_ferrite_parents'][0]
    assert parent['largest_child'] == 5
    assert parent['additional_children'] == parent['additional_under200'] == 1
    assert parent['additional_child_pixels'] == parent['additional_under200_pixels'] == 50
    decomposition = result['ferrite_count_decomposition']
    assert decomposition['delta'] == decomposition['reconstructed_delta'] == 1
    assert decomposition['old_ferrite_children_minus_parents'] == 1
    assert decomposition['new_origins'] == dict(old_ferrite=2, old_pearlite=0, background=0, ambiguous=0)
    assert decomposition['exact'] is True


def test_ferrite_mean_area_uses_pixels_per_instance_not_mean_of_old_children():
    old, new, oc, nc = tiny_split()
    before = runner.area_description(old, oc, 2000.)
    after = runner.area_description(new, nc, 2000.)
    assert before['ferrite']['mean_area'] == 2000.
    assert after['ferrite']['pixels'] == 2000
    assert after['ferrite']['instances'] == 2
    assert after['ferrite']['mean_area'] == 1000.
    assert after['ferrite']['tiny_under200_pixels'] == 50
    assert changes(before, after)['ferrite']['mean_area_percent'] == -50.


def test_no_ferrite_and_background_only_predictions_have_no_nan_mean():
    for ids, classes in [(np.ones((2, 3), np.uint16), {'1': 0}),
                         (np.zeros((2, 3), np.uint16), {})]:
        result = runner.area_description(ids, classes, None)
        assert result['ferrite']['instances'] == result['ferrite']['pixels'] == 0
        assert result['ferrite']['mean_area'] is None
        assert result['ferrite']['area_quantiles'] is None
        assert result['ferrite']['relative_tails'] is None
        mapping = runner.overlap_mapping(ids, ids, classes, classes, None)
        assert mapping['ferrite_count_decomposition']['delta'] == 0
        assert mapping['ferrite_count_decomposition']['exact']


def test_id_relabeling_preserves_geometry_correspondence_and_origin():
    old = np.array([[1, 1, 2], [1, 0, 2]], np.uint16)
    new = np.array([[31, 31, 14], [31, 0, 14]], np.uint16)
    result = runner.overlap_mapping(old, new, {'1': 1, '2': 0}, {'31': 1, '14': 0}, 3.)
    row = result['new_ferrite_regions'][0]
    assert row['new_id'] == 31 and row['dominant_old_id'] == 1
    assert row['matched_old_iou50_id'] == 1 and row['matched_iou50'] == 1.
    assert row['origin'] == 'old_ferrite'
    assert all(result['unmatched_iou50'][arm][p]['instances'] == 0
               for arm in ('old', 'new') for p in runner.PHASES)
    assert result['coverage'] == dict(old_background_to_new_covered=0, old_covered_to_new_background=0)
    assert result['ferrite_count_decomposition']['delta'] == 0


def test_background_phase_flip_and_ambiguous_regions_are_separate_from_splits():
    old = np.array([[1, 1, 2, 2, 0, 0, 3, 4]], np.uint16)
    new = np.array([[0, 0, 10, 10, 11, 11, 12, 12]], np.uint16)
    result = runner.overlap_mapping(old, new, {'1': 1, '2': 0, '3': 0, '4': 1},
                                    {'10': 1, '11': 1, '12': 1}, 2.)
    rows = {r['new_id']: r for r in result['new_ferrite_regions']}
    assert rows[10]['origin'] == 'old_pearlite' and rows[10]['matched_old_iou50_id'] == 2
    assert rows[11]['origin'] == 'background' and rows[11]['old_background_fraction'] == 1
    assert rows[12]['origin'] == 'ambiguous' and rows[12]['dominant_old_id'] is None
    assert rows[12]['maximum_old_overlap_fraction'] == .5
    d = result['ferrite_count_decomposition']
    assert d['old_ferrite'] == 2 and d['new_ferrite'] == 3
    assert d['old_ferrite_children_minus_parents'] == -2
    assert d['delta'] == d['reconstructed_delta'] == 1
    assert d['new_origins'] == dict(old_ferrite=0, old_pearlite=1, background=1, ambiguous=1)
    assert result['coverage'] == dict(old_background_to_new_covered=2, old_covered_to_new_background=2)


def test_whole_area_iou_includes_new_coverage_outside_old_masks():
    old = np.array([[1, 0, 0, 0]], np.uint16)
    new = np.full((1, 4), 7, np.uint16)
    result = runner.overlap_mapping(old, new, {'1': 1}, {'7': 1}, 1.)
    row = result['new_ferrite_regions'][0]
    assert row['origin'] == 'background'
    assert row['old_background_fraction'] == .75
    assert row['matched_old_iou50_id'] is None
    assert result['unmatched_iou50']['old']['ferrite']['pixels'] == 1
    assert result['unmatched_iou50']['new']['ferrite']['pixels'] == 4


def test_shape_mismatch_is_not_silently_flattened():
    with pytest.raises(ValueError, match='shapes differ'):
        runner.overlap_mapping(np.ones((2, 3), np.uint16), np.ones((3, 2), np.uint16),
                               {'1': 1}, {'1': 1}, 6.)


def json_write(path, value):
    path.write_text(json.dumps(value), encoding='utf8')


def package_fixture(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    files, records, entries = [], [], []
    for index in range(2):
        stem = 'synthetic_' + str(index)
        source = tmp_path / (stem + '.png')
        assert cv2.imwrite(str(source), np.full((4, 5, 3), index, np.uint8))
        files.append(source)
        ids = np.ones((4, 5), np.uint16)
        ids[:, 3:] = 2
        classes = {'1': 1, '2': 0}
        png = stem + '_inst.png'
        labels = stem + '_class.json'
        assert cv2.imwrite(str(cache / png), ids)
        json_write(cache / labels, classes)
        entries.extend([(png, (cache / png).read_bytes()), (labels, (cache / labels).read_bytes())])
        records.append(dict(source=stem, view='patch1024', phases=phase_description(ids, classes)))
    archive = tmp_path / 'control.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    receipt = tmp_path / 'receipt.json'
    manifest = dict(complete=True, epoch=20, precision='FP32', checkpoint_sha256=runner.CONTROL_SHA,
                    sizes=[0, 1024], images=records)
    value = dict(sha256=sha(archive), images=2, files=4, epoch=20, checkpoint_sha256=runner.CONTROL_SHA,
                 inference='native1024', flat=True, crc_ok=True)
    json_write(receipt, value)
    return dict(cache=cache, archive=archive, receipt=receipt, files=files, manifest=manifest,
                receipt_value=value, entries=entries)


def verify_fixture(f):
    return runner.verify_package(f['cache'], f['archive'], f['receipt'], f['files'],
                                 f['manifest'], expected_sha=sha(f['archive']), expected_images=2)


def test_existing_package_cache_binding_checks_all_native_files(tmp_path):
    f = package_fixture(tmp_path)
    result = verify_fixture(f)
    assert result['passed'] and result['png_bytes_exact'] and result['class_dicts_exact']
    assert len(result['images']) == 2
    assert all(r['shape'] == [4, 5] and r['class_json_bytes_exact'] for r in result['images'])


def test_json_formatting_can_differ_but_class_values_cannot(tmp_path):
    f = package_fixture(tmp_path)
    labels = f['cache'] / 'synthetic_0_class.json'
    labels.write_text('{"2": 0,\n "1": 1}\n', encoding='utf8')
    result = verify_fixture(f)
    assert not result['images'][0]['class_json_bytes_exact']
    json_write(labels, {'1': 0, '2': 0})
    with pytest.raises(RuntimeError, match='Control cache differs'):
        verify_fixture(f)


@pytest.mark.parametrize('mutation', ['png_bytes', 'uint8', 'wrong_shape', 'classes_bool', 'extra_file'])
def test_cache_contract_rejects_changed_encoding_dtype_shape_or_labels(tmp_path, mutation):
    f = package_fixture(tmp_path)
    path = f['cache'] / 'synthetic_0_inst.png'
    ids = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mutation == 'png_bytes':
        # 相同像素重新编码也不能冒充既有官方包的原PNG字节。
        assert cv2.imwrite(str(path), ids, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        assert sha(path) != __import__('hashlib').sha256(f['entries'][0][1]).hexdigest()
    elif mutation == 'uint8':
        assert cv2.imwrite(str(path), ids.astype(np.uint8))
    elif mutation == 'wrong_shape':
        assert cv2.imwrite(str(path), ids[:2])
    elif mutation == 'classes_bool':
        json_write(f['cache'] / 'synthetic_0_class.json', {'1': True, '2': 0})
    else:
        (f['cache'] / 'unexpected.txt').write_text('extra', encoding='utf8')
    with pytest.raises((RuntimeError, ValueError)):
        verify_fixture(f)


@pytest.mark.parametrize('entry', ['../synthetic_0_inst.png', 'nested/synthetic_0_inst.png',
                                  r'nested\synthetic_0_inst.png', 'extra.txt', 'synthetic_0_inst.png'])
def test_archive_rejects_paths_extras_and_duplicate_entries(tmp_path, entry):
    f = package_fixture(tmp_path)
    with zipfile.ZipFile(f['archive'], 'a') as z:
        z.writestr(entry, b'unexpected')
    f['receipt_value']['sha256'] = sha(f['archive'])
    json_write(f['receipt'], f['receipt_value'])
    with pytest.raises(ValueError, match='ZIP|duplicate'):
        verify_fixture(f)


def test_final_candidate_roundtrip_preserves_uint16_max_id_contract(tmp_path):
    ids = np.array([[0, 65535], [65535, 7]], np.uint16)
    classes = {'7': 0, '65535': 1}
    receipt = runner.save_candidate(tmp_path, 'synthetic', dict(instances=ids, classes=classes), (2, 2))
    actual, labels = runner.load_prediction(tmp_path, 'synthetic', (2, 2))
    assert np.array_equal(actual, ids) and labels == classes
    assert receipt['png_sha256'] == sha(tmp_path / 'synthetic_inst.png')


def test_source_snapshot_rejects_mutation_and_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    (tmp_path / 'new.py').write_text('value = 1\n', encoding='utf8')
    output = tmp_path / 'output'
    (output / 'source').mkdir(parents=True)
    (output / 'source' / 'new.py').write_bytes((tmp_path / 'new.py').read_bytes())
    sources = {'new.py': sha(tmp_path / 'new.py')}
    runner.verify_new_sources(output, sources)
    for key in ('../new.py', str((tmp_path / 'new.py').resolve())):
        with pytest.raises(RuntimeError, match='source changed'):
            runner.verify_new_sources(output, {key: sources['new.py']})
    (output / 'source' / 'new.py').write_text('value = 2\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='source changed'):
        runner.verify_new_sources(output, sources)
