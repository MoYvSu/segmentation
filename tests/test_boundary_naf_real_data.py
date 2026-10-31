# -*- coding: utf-8 -*-
"""真实输入配对的CPU契约；全部使用独立构造图，不读取赛题数据。"""
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize
import torch

from data.boundary_naf import BoundaryNAFDataset, collate_boundary_naf
from data.boundary_naf_real import BoundaryNAFRealDataset


MODELS = {'backend': '1' * 64, 'restorer': '2' * 64}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, document):
    Path(path).write_text(json.dumps(document), encoding='utf-8')


def fixture_files(tmp_path, sources=2, shape=(96, 128), boundary=None):
    rows, heat_rows = [], []
    ids = np.ones(shape, np.int32)
    ids[:, shape[1] // 2:] = 2
    ids[12:20, 10:18] = 0
    original = ids > 0
    original[48:60, shape[1]//2-4:shape[1]//2+4] = False
    if boundary is None:
        boundary = np.zeros(shape, np.float32)
        boundary[32:48, :] = 1.
        boundary[:, 12:17] = 1.
    for i in range(1, sources + 1):
        name = f'train_{i:03d}'
        image = tmp_path / f'{name}.jpg'
        image.write_bytes(b'image identity only; deliberately not a decodable RGB file')
        gt = tmp_path / f'{name}_gt.npz'
        np.savez_compressed(gt, instance_map=ids, original_covered=original,
                            filled=(ids > 0) & ~original, residual_unknown=ids == 0)
        classes = tmp_path / f'{name}_class.json'
        write(classes, {'1': 0, '2': 1})
        paths = dict(image=image, gt=gt, classes=classes)
        row = dict(source=name, **{k: str(v) for k, v in paths.items()},
                   hashes={k + '_sha256': sha(v) for k, v in paths.items()}, shape=list(shape))
        heat = tmp_path / f'{name}_heat.npz'
        np.savez_compressed(heat, boundary=boundary)
        rows.append(row)
        heat_rows.append(dict(source=name, heat_path=heat.name, heat_sha256=sha(heat),
                              **row['hashes'], shape=list(shape)))
    gt_manifest = tmp_path / 'processed.json'
    write(gt_manifest, {'rows': rows})
    heat_manifest = tmp_path / 'real.json'
    write(heat_manifest, dict(format='boundary_naf_real_heatmaps_v1', complete=True,
        no_test_training=True, local_restoration={'mode': 'zero', 'model': 'restorer'},
        models={key: dict(file=key + '.pt', sha256=value) for key, value in MODELS.items()},
        rows=heat_rows))
    return gt_manifest, heat_manifest, rows, boundary, ids


def config(heat_manifest, sources=2, **data):
    return dict(boundary_naf=dict(seed=20261026,
        real_input_manifest=str(heat_manifest), real_input_models=MODELS.copy(),
        data=dict(expected_sources=sources, crop_size=64, stride=32, augmentations=8,
                  samples_per_epoch=32, valid_margin=4, crop_margin=4,
                  identity_probability=.2, **data)))


def forced_view(dataset, box=(0, 0, 64, 64), rotation=0, flip=False):
    item = dict(dataset.plan[0], source='train_001', box=list(box), rotation=rotation,
                flip=flip, augmentation=rotation + 4 * int(flip))
    dataset.plan[0] = item
    return dataset[0]


def test_same_gt_sampling_and_transform_plan_as_existing_control(tmp_path):
    gt, heat, _, _, _ = fixture_files(tmp_path)
    cfg = config(heat)
    old = BoundaryNAFDataset(gt, cfg, epoch=1)
    real = BoundaryNAFRealDataset(gt, cfg, epoch=1)
    assert real.plan == old.plan
    assert real.plan_receipt() == old.plan_receipt()
    for index in range(16):
        a, b = old[index], real[index]
        assert torch.equal(a['target'], b['target'])
        assert torch.equal(a['valid'], b['valid'])
        assert a['meta']['geometry'] == b['meta']['geometry']
    first = real.plan_receipt()
    real.set_epoch(2)
    assert real.plan_receipt()['plan_sha256'] != first['plan_sha256']
    real.set_epoch(1)
    assert real.plan_receipt() == first


def test_input_is_skeletonized_before_native_crop_not_after(tmp_path):
    gt, heat, _, boundary, _ = fixture_files(tmp_path)
    ds = BoundaryNAFRealDataset(gt, config(heat), epoch=0)
    item = forced_view(ds, box=(24, 32, 64, 64))
    expected = skeletonize(boundary >= .5)[24:88, 32:96]
    cropped_first = skeletonize(boundary[24:88, 32:96] >= .5)
    assert not np.array_equal(expected, cropped_first), 'Fixture must expose crop-edge thinning artifacts'
    np.testing.assert_array_equal(item['input'][0].numpy(), expected)
    assert item['meta']['geometry']['box'] == [24, 32, 64, 64]
    assert item['meta']['no_artificial_corruption']


@pytest.mark.parametrize('rotation,flip', [(0, False), (1, False), (2, True), (3, True)])
def test_input_follows_exact_same_d4_transform(tmp_path, rotation, flip):
    gt, heat, _, boundary, _ = fixture_files(tmp_path)
    ds = BoundaryNAFRealDataset(gt, config(heat))
    item = forced_view(ds, box=(24, 32, 64, 64), rotation=rotation, flip=flip)
    expected = np.rot90(skeletonize(boundary >= .5)[24:88, 32:96], rotation)
    if flip:
        expected = expected[:, ::-1]
    np.testing.assert_array_equal(item['input'][0].numpy(), expected)


def test_gt_unknown_never_masks_prediction_input(tmp_path):
    gt, heat, _, _, ids = fixture_files(tmp_path)
    ds = BoundaryNAFRealDataset(gt, config(heat))
    item = forced_view(ds)
    unknown = ids[:64, :64] == 0
    assert item['input'][0].numpy()[unknown].any()
    assert not item['valid'][0].numpy()[unknown].any()
    assert torch.equal(item['valid'], item['fully_known_mask'])
    assert not torch.any(item['gap_mask'].bool() & ~item['valid'].bool())
    assert torch.all(item['target'][item['gap_mask'].bool()] == 1)


def test_natural_gap_is_nonempty_diagnostic_not_an_input_erasure(tmp_path):
    gt, heat, _, boundary, _ = fixture_files(tmp_path)
    ds = BoundaryNAFRealDataset(gt, config(heat))
    item = forced_view(ds, box=(0, 32, 64, 64))
    source = skeletonize(boundary >= .5)[:64, 32:96]
    np.testing.assert_array_equal(item['input'][0].numpy(), source)
    gap = item['gap_mask'][0].numpy().astype(bool)
    assert gap.any()
    assert np.all(item['target'][0].numpy()[gap] == 1)
    assert not np.any(gap & source)
    assert item['meta']['gap_kind'].startswith('natural_')
    assert item['meta']['gaps'] == []


def test_gt_content_changes_target_but_not_real_input(tmp_path):
    gt, heat, rows, _, ids = fixture_files(tmp_path)
    cfg = config(heat)
    first = BoundaryNAFRealDataset(gt, cfg)
    first_view = forced_view(first, box=(0, 32, 64, 64))
    ids[:, 56:] = 2
    ids[:, :56] = np.where(ids[:, :56] == 0, 0, 1)
    gt_file = Path(rows[0]['gt'])
    np.savez_compressed(gt_file, instance_map=ids, original_covered=ids > 0,
                        filled=np.zeros(ids.shape, bool), residual_unknown=ids == 0)
    document = json.loads(gt.read_text())
    document['rows'][0]['hashes']['gt_sha256'] = sha(gt_file)
    write(gt, document)
    heat_doc = json.loads(heat.read_text())
    heat_doc['rows'][0]['gt_sha256'] = sha(gt_file)
    write(heat, heat_doc)
    second = BoundaryNAFRealDataset(gt, cfg)
    second_view = forced_view(second, box=(0, 32, 64, 64))
    assert not torch.equal(first_view['target'], second_view['target'])
    assert torch.equal(first_view['input'], second_view['input'])


def test_empty_real_input_marks_all_valid_target_positives_as_natural_missing(tmp_path):
    gt, heat, _, _, _ = fixture_files(tmp_path, boundary=np.zeros((96, 128), np.float32))
    ds = BoundaryNAFRealDataset(gt, config(heat))
    item = forced_view(ds, box=(0, 32, 64, 64))
    assert not item['input'].any()
    expected = item['target'].bool() & item['valid'].bool()
    assert expected.any() and torch.equal(item['gap_mask'].bool(), expected)
    assert not item['identity'].item()


def test_legacy_identity_and_gap_parameters_do_not_corrupt_real_input(tmp_path):
    gt, heat, _, _, _ = fixture_files(tmp_path)
    a = config(heat)
    b = copy.deepcopy(a)
    a['boundary_naf']['data']['identity_probability'] = 0.
    b['boundary_naf']['data']['identity_probability'] = 1.
    b['boundary_naf']['data']['gap_count_range'] = [4, 4]
    da = BoundaryNAFRealDataset(gt, a)
    db = BoundaryNAFRealDataset(gt, b)
    for index in range(16):
        assert torch.equal(da[index]['input'], db[index]['input'])
        assert da[index]['identity'].item() == torch.equal(da[index]['input'], da[index]['target'])
        assert db[index]['meta']['gaps'] == []
    assert da.identity['identity_probability'] is None
    assert not da.identity['artificial_corruption']
    assert da.identity['legacy_synthetic_settings_ignored_for_real_input']['identity_probability'] == 0.
    assert db.identity['legacy_synthetic_settings_ignored_for_real_input']['identity_probability'] == 1.


def test_cached_input_does_not_reload_or_mutate_source_maps(tmp_path, monkeypatch):
    gt, heat, _, _, _ = fixture_files(tmp_path)
    ds = BoundaryNAFRealDataset(gt, config(heat))
    old_input = {k: v.copy() for k, v in ds.input_skeletons.items()}
    old_target = {k: [v.copy() for v in values] for k, values in ds.cache.items()}
    with monkeypatch.context() as patch:
        patch.setattr(np, 'load', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('No hot-step NPZ read')))
        for i in range(len(ds)):
            ds[i]
    for key in old_input:
        assert ds.input_skeletons[key].dtype == bool
        np.testing.assert_array_equal(ds.input_skeletons[key], old_input[key])
        for current, old in zip(ds.cache[key], old_target[key]):
            np.testing.assert_array_equal(current, old)
    ds.verify_sources()
    json.dumps(ds.identity, allow_nan=False)
    batch = collate_boundary_naf([ds[0], ds[1]])
    assert batch['input'].shape == (2, 1, 64, 64)
    assert batch['identity'].shape == (2,)


def test_small_source_reflect_padding_is_not_supervised(tmp_path):
    gt, heat, _, boundary, _ = fixture_files(tmp_path, shape=(48, 56))
    ds = BoundaryNAFRealDataset(gt, config(heat), epoch=0)
    item = forced_view(ds, box=(0, 0, 48, 56))
    expected = np.pad(skeletonize(boundary >= .5), ((0, 16), (0, 8)), mode='reflect')
    np.testing.assert_array_equal(item['input'][0].numpy(), expected)
    assert not item['valid'][0, 48:].any()
    assert not item['valid'][0, :, 56:].any()


@pytest.mark.parametrize('mutation,error', [
    (lambda d: d.update(no_test_training=False), 'no-test'),
    (lambda d: d.update(complete=False), 'complete'),
    (lambda d: d.update(local_restoration={'mode': 'single', 'model': 'restorer'}), 'zero/restorer'),
    (lambda d: d['models']['backend'].update(sha256='3'*64), 'model SHA256'),
    (lambda d: d['rows'][0].update(source='test_169'), 'training sources'),
    (lambda d: d['rows'][0].update(image_sha256='3'*64), 'source binding'),
    (lambda d: d['rows'][0].update(gt_sha256='3'*64), 'source binding'),
    (lambda d: d['rows'][0].update(classes_sha256='3'*64), 'source binding'),
    (lambda d: d['rows'][0].update(heat_sha256='3'*64), 'file SHA256'),
])
def test_manifest_rejects_wrong_mode_model_or_source_binding(tmp_path, mutation, error):
    gt, heat, _, _, _ = fixture_files(tmp_path)
    document = json.loads(heat.read_text())
    mutation(document)
    write(heat, document)
    with pytest.raises(ValueError, match=error):
        BoundaryNAFRealDataset(gt, config(heat))


def test_expected_model_identities_must_not_be_inferred_from_manifest(tmp_path):
    gt, heat, _, _, _ = fixture_files(tmp_path)
    cfg = config(heat)
    del cfg['boundary_naf']['real_input_models']
    with pytest.raises(ValueError, match='explicitly bind'):
        BoundaryNAFRealDataset(gt, cfg)


def test_top_level_gt_manifest_binding_is_checked_when_declared(tmp_path):
    gt, heat, _, _, _ = fixture_files(tmp_path)
    document = json.loads(heat.read_text())
    document['processed_gt_manifest_sha256'] = sha(gt)
    write(heat, document)
    BoundaryNAFRealDataset(gt, config(heat)).verify_sources()
    document['processed_gt_manifest_sha256'] = '3' * 64
    write(heat, document)
    with pytest.raises(ValueError, match='processed-GT manifest SHA256'):
        BoundaryNAFRealDataset(gt, config(heat))


@pytest.mark.parametrize('kind', ['nan', 'range', 'dtype', 'shape'])
def test_native_probability_array_contract_is_checked(tmp_path, kind):
    gt, heat, _, boundary, _ = fixture_files(tmp_path)
    if kind == 'nan': boundary[0, 0] = np.nan
    elif kind == 'range': boundary[0, 0] = 2.
    elif kind == 'dtype': boundary = boundary.astype(np.uint8)
    else: boundary = boundary[:-1]
    document = json.loads(heat.read_text())
    p = tmp_path / document['rows'][0]['heat_path']
    np.savez_compressed(p, boundary=boundary)
    document['rows'][0]['heat_sha256'] = sha(p)
    write(heat, document)
    with pytest.raises(ValueError, match='finite native float32'):
        BoundaryNAFRealDataset(gt, config(heat))


@pytest.mark.parametrize('changed', ['manifest', 'heat', 'skeleton'])
def test_verify_sources_rejects_changed_manifest_heat_or_ram_cache(tmp_path, changed):
    gt, heat, _, _, _ = fixture_files(tmp_path)
    ds = BoundaryNAFRealDataset(gt, config(heat))
    if changed == 'manifest': heat.write_text(heat.read_text()+'\n')
    elif changed == 'heat':
        p = Path(ds.real_input_records[0]['heat_path'])
        p.write_bytes(p.read_bytes()+b'changed')
    else: ds.input_skeletons['train_001'][0, 0] = ~ds.input_skeletons['train_001'][0, 0]
    with pytest.raises(RuntimeError, match='changed'):
        ds.verify_sources()
