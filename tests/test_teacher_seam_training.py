# -*- coding: utf-8 -*-
"""虚构图检查零缝实验控制隔离、增强重放及旧合法监督的保持。"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from utils.affinity_loss import build_affinity_targets_torch
from utils.offset_letterbox import letterbox_instance_geometry
from utils.teacher_seam_training import (
    audit_repaired_batch, compare_repaired_rows, validate_repaired_peer,
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def tensor_sha(value):
    digest = hashlib.sha256(b'value')
    digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(autouse=True)
def cpu_threads():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


@pytest.fixture
def recipes():
    original = dict(paths=dict(project_root='synthetic-root'),
                    gt_source=dict(epochs=5, learning_rate=2e-5, output_dir='outputs/teacher', seed=42),
                    backend_adaptation=dict(pseudo_weight=.5), sam2=dict(freeze=True))
    repaired = deepcopy(original)
    repaired['gt_source']['output_dir'] = 'outputs/seam'
    repaired['gt_source']['seam_repair'] = dict(format='teacher_seam_support_v1',
        manifest_sha256='a'*64, baseline_steps_sha256='b'*64, baseline_status_sha256='c'*64)
    old_contract = dict(format='affinity_gt_source_v1', recipe_sha256=canonical_sha(original),
        manifest_sha256='parent', checkpoint_sha256='r1', restorer_checkpoint_sha256='d5a',
        frozen_state_sha256='frozen', restorer_state_sha256='restorer', initial_head_sha256='head',
        training_code_sha256='old-code')
    new_contract = deepcopy(old_contract)
    new_contract.update(recipe_sha256=canonical_sha(repaired), training_code_sha256='new-code',
        variant='teacher_zero_seam_v1', seam_manifest_sha256='a'*64,
        baseline_steps_sha256='b'*64, baseline_status_sha256='c'*64)
    status = dict(format='affinity_gt_source_v1', arm='teacher', status='complete', epoch=5,
                  epochs_requested=5, updates=320, frozen_unchanged=True,
                  restorer_unchanged=True, strict_reload_equal=True, config=original, contract=old_contract)
    return status, new_contract, repaired


def test_control_reuse_allows_only_repair_output_and_new_code(recipes):
    status, contract, config = recipes
    assert validate_repaired_peer(status, contract, config, 5)['passed']


def test_json_roundtrip_integer_class_keys_preserves_recipe_but_value_change_rejected(recipes):
    status, contract, config = recipes
    classes = {0: 'pearlite', 1: 'ferrite'}
    config['data'] = dict(classes=classes)
    # 旧status从JSON读取，YAML的整数键自然变成字符串；两者JSON配方相同。
    status['config']['data'] = json.loads(json.dumps(dict(classes=classes)))
    status['contract']['recipe_sha256'] = canonical_sha(status['config'])
    contract['recipe_sha256'] = canonical_sha(config)
    assert validate_repaired_peer(status, contract, config, 5)['passed']
    config['data']['classes'][0] = 'ferrite'
    contract['recipe_sha256'] = canonical_sha(config)
    with pytest.raises(ValueError, match='outside explicit seam repair'):
        validate_repaired_peer(status, contract, config, 5)


@pytest.mark.parametrize('area,field,value', [
    ('gt_source', 'seed', 43), ('gt_source', 'learning_rate', 3e-5),
    ('backend_adaptation', 'pseudo_weight', .4), ('sam2', 'freeze', False),
    ('paths', 'project_root', 'other-root'),
])
def test_recipe_drift_is_rejected_even_with_updated_recipe_hash(recipes, area, field, value):
    status, contract, config = recipes
    config[area][field] = value
    contract['recipe_sha256'] = canonical_sha(config)
    with pytest.raises(ValueError, match='outside explicit seam repair'):
        validate_repaired_peer(status, contract, config, 5)


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'restorer_checkpoint_sha256',
                                 'initial_head_sha256', 'frozen_state_sha256', 'manifest_sha256'])
def test_changed_initialization_or_parent_manifest_rejected(recipes, field):
    status, contract, config = recipes
    contract[field] = 'changed'
    with pytest.raises(ValueError, match='initialization or frozen contract'):
        validate_repaired_peer(status, contract, config, 5)


def test_incomplete_teacher_and_wrong_repair_bindings_rejected(recipes):
    status, contract, config = recipes
    status['updates'] = 319
    with pytest.raises(ValueError, match='same-budget'):
        validate_repaired_peer(status, contract, config, 5)
    status['updates'] = 320
    contract['baseline_status_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='SHA bindings'):
        validate_repaired_peer(status, contract, config, 5)
    contract['baseline_status_sha256'] = 'c'*64
    status['contract']['recipe_sha256'] = 'stale'
    with pytest.raises(ValueError, match='outside explicit seam repair'):
        validate_repaired_peer(status, contract, config, 5)


def test_unknown_contract_fields_and_missing_freeze_checks_rejected(recipes):
    status, contract, config = recipes
    contract['extra_change'] = True
    with pytest.raises(ValueError, match='explicit repair contracts'):
        validate_repaired_peer(status, contract, config, 5)
    contract.pop('extra_change')
    status['frozen_unchanged'] = 1
    with pytest.raises(ValueError, match='strictly reloaded'):
        validate_repaired_peer(status, contract, config, 5)


@pytest.fixture
def native_pair(tmp_path):
    processed = np.ones((12, 16), np.uint16)
    processed[:, 9:] = 2
    processed[:2, :2] = 0
    teacher = np.full_like(processed, 7)
    teacher[:, 9:] = 8
    teacher[:, 8] = 0
    teacher[processed == 0] = 0
    repaired = teacher.copy()
    repaired[:, 8] = 8
    pair = tmp_path/'pair.npz'
    gt = tmp_path/'gt.npz'
    np.savez(pair, teacher_ids=teacher, common_known=teacher > 0)
    np.savez(gt, instance_map=processed)
    rows = dict(train_001=dict(source='train_001', pair_npz='pair.npz', manual_gt='gt.npz'))
    config = dict(paths=dict(project_root=str(tmp_path)), data=dict(image_size=16))
    return config, rows, teacher, repaired, processed


def make_batch(native_ids, *, crop=None, horizontal=False, vertical=False, turns=0):
    crop = crop or [0, 0, *native_ids.shape]
    y, x, y1, x1 = crop
    grid, content, _ = letterbox_instance_geometry(native_ids[y:y1, x:x1], 16, 16)
    ids = torch.from_numpy(grid.astype(np.int64))[None]
    pixels = torch.from_numpy(content & (grid > 0))[None, None]
    # 与原在线增强执行顺序相同，测试覆盖非对称裁块及奇数旋转。
    if horizontal:
        ids, pixels = ids.flip(-1), pixels.flip(-1)
    if vertical:
        ids, pixels = ids.flip(-2), pixels.flip(-2)
    ids, pixels = torch.rot90(ids, turns, (-2, -1)), torch.rot90(pixels, turns, (-2, -1))
    return dict(name=['train_001.jpg'], crop_box=torch.tensor([crop], dtype=torch.int32),
        horizontal_flip=torch.tensor([horizontal]), vertical_flip=torch.tensor([vertical]),
        rotation_k=torch.tensor([turns]), affinity_instance_map=ids,
        affinity_valid_content=pixels)


@pytest.mark.parametrize('horizontal,vertical,turns,crop', [
    (False, False, 0, None), (True, False, 1, [1, 2, 10, 14]),
    (False, True, 3, [0, 3, 12, 12]), (True, True, 2, None),
])
def test_actual_crop_flip_rotation_expands_support_and_preserves_answers(native_pair, horizontal, vertical, turns, crop):
    config, rows, teacher, repaired, _ = native_pair
    batch = make_batch(repaired, crop=crop, horizontal=horizontal, vertical=vertical, turns=turns)
    old_batch = make_batch(teacher, crop=crop, horizontal=horizontal, vertical=vertical, turns=turns)
    audit = audit_repaired_batch(batch, config, rows)
    _, old_edges = build_affinity_targets_torch(old_batch['affinity_instance_map'], old_batch['affinity_valid_content'])
    assert audit['old_valid_pixels_sha256'] == tensor_sha(old_batch['affinity_valid_content'])
    assert audit['old_valid_edges_sha256'] == tensor_sha(old_edges)
    assert audit['old_targets_unchanged'] and audit['old_target_changed_edges'] == 0
    assert audit['added_edges'] > 0 and audit['added_different'] > 0 and audit['added_same'] > 0
    assert audit['added_edges'] == audit['added_same'] + audit['added_different']
    assert audit['new_legal_edges'] == audit['old_legal_edges'] + audit['added_edges']


def test_old_nonzero_id_change_rejected(native_pair):
    config, rows, _, repaired, _ = native_pair
    repaired[4, 4] = 8
    with pytest.raises(RuntimeError, match='nonzero teacher ownership'):
        audit_repaired_batch(make_batch(repaired), config, rows)


def test_unknown_filling_or_old_support_removal_rejected(native_pair):
    config, rows, _, repaired, _ = native_pair
    unknown = repaired.copy()
    unknown[0, 0] = 7
    with pytest.raises(RuntimeError, match='processed unknown or padding'):
        audit_repaired_batch(make_batch(unknown), config, rows)
    repaired[4, 4] = 0
    with pytest.raises(RuntimeError, match='removed original teacher support'):
        audit_repaired_batch(make_batch(repaired), config, rows)


@pytest.mark.parametrize('field,value', [('name', ['test_101.jpg']), ('horizontal_flip', True),
                                       ('rotation_k', torch.tensor([4])),
                                       ('crop_box', torch.tensor([[0, 0, 12, 17]]))])
def test_draw_identity_shape_or_transform_errors_rejected(native_pair, field, value):
    config, rows, _, repaired, _ = native_pair
    batch = make_batch(repaired)
    batch[field] = value
    with pytest.raises(ValueError):
        audit_repaired_batch(batch, config, rows)


@pytest.fixture
def paired_rows(native_pair):
    config, rows, teacher, repaired, _ = native_pair
    actual_batch = make_batch(repaired, horizontal=True, turns=1)
    old_batch = make_batch(teacher, horizontal=True, turns=1)
    audit = audit_repaired_batch(actual_batch, config, rows)
    def signature(batch):
        target, edges = build_affinity_targets_torch(batch['affinity_instance_map'], batch['affinity_valid_content'])
        return dict(image='image', clean_image='clean', restored='restored',
                    valid_pixels=tensor_sha(batch['affinity_valid_content']), valid_edges=tensor_sha(edges),
                    shape=list(target.shape))
    old = dict(epoch=1, step=1, seed=42, manual=['train_001.jpg'], pseudo=['train_002.jpg'],
               manual_draw=dict(horizontal_flip=[True], vertical_flip=[False], rotation_k=[1]),
               pseudo_draw={}, manual_crop=[[0, 0, 12, 16]], pseudo_crop=[[0, 0, 12, 16]],
               training_view='whole', inputs=[signature(old_batch), signature(old_batch)])
    new = deepcopy(old)
    new['inputs'][0] = signature(actual_batch)
    new['seam_support'] = audit
    return new, old


def test_actual_control_comparison_allows_only_audited_manual_domain(paired_rows):
    actual, old = paired_rows
    compare_repaired_rows(actual, old)


@pytest.mark.parametrize('change', ['crop', 'manual_image', 'sam2_domain', 'old_hash', 'new_hash', 'old_answers'])
def test_control_comparison_rejects_unrelated_or_unaudited_changes(paired_rows, change):
    actual, old = paired_rows
    if change == 'crop':
        actual['manual_crop'][0][0] = 1
    elif change == 'manual_image':
        actual['inputs'][0]['restored'] = 'different'
    elif change == 'sam2_domain':
        actual['inputs'][1]['valid_edges'] = 'different'
    elif change == 'old_hash':
        actual['seam_support']['old_valid_edges_sha256'] = 'different'
    elif change == 'new_hash':
        actual['seam_support']['new_valid_edges_sha256'] = 'different'
    else:
        actual['seam_support']['old_target_changed_edges'] = 1
    with pytest.raises(RuntimeError):
        compare_repaired_rows(actual, old)


def test_hash_is_compatible_with_existing_receipts():
    from train_backend_adaptation import tensor_digest
    value = torch.tensor([[[[True, False], [False, True]]]])
    assert tensor_sha(value) == tensor_digest([('value', value)])


def test_trainer_native_crop_keeps_old_common_support_after_repaired_ids(native_pair, tmp_path):
    import cv2
    from train_affinity_gt_source import PairGeometrySource
    config, rows, teacher, repaired, _ = native_pair
    config['affinity_native'] = dict(crop_attempts=32, minimum_known_pairs=1)
    rgb = np.indices(teacher.shape).sum(axis=0).astype(np.uint8)
    rgb = np.stack([rgb, rgb * 2, rgb * 3], -1)
    cv2.imwrite(str(tmp_path/'train_001.png'), rgb)
    # 实际old pair还有manual等字段，本test仅读取teacher/common供teacher分支使用。
    seam_path = tmp_path/'seam.npz'
    np.savez(seam_path, teacher_ids=repaired)
    old_row = dict(rows['train_001'], image='train_001.png')
    repaired_row = dict(old_row, seam_pair_npz='seam.npz')
    old_ds = PairGeometrySource(config, [old_row], 'teacher', image_size=16, grid=16, seed=43)
    new_ds = PairGeometrySource(config, [repaired_row], 'teacher', image_size=16, grid=16, seed=43)
    for draw in range(16):
        for dataset in (old_ds, new_ds):
            dataset.size, dataset.epoch, dataset.draw = 8, 2, draw
        old, new = old_ds[0], new_ds[0]
        assert torch.equal(old['crop_box'], new['crop_box'])
        assert old['crop_attempt'] == new['crop_attempt']
        assert torch.equal(old['image'], new['image'])
        old_t, old_v = build_affinity_targets_torch(old['instance_map'][None], old['valid_content'][None])
        new_t, new_v = build_affinity_targets_torch(new['instance_map'][None], new['valid_content'][None])
        assert not (old_v & ~new_v).any()
        assert torch.equal(old_t[old_v], new_t[old_v])


def test_same_run_resume_rejects_changed_seam_contract(recipes):
    from types import SimpleNamespace
    from train_affinity_gt_source import checkpoint_payload, strict_resume
    status, contract, config = recipes
    head = torch.nn.Conv2d(2, 8, 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    payload = checkpoint_payload(SimpleNamespace(affinity_decoder=head), optimizer,
                                 scheduler, config, 'teacher', 1, 64, contract)
    assert strict_resume(payload, head, optimizer, scheduler, contract, 'teacher') == (1, 64)
    changed = deepcopy(contract)
    changed['seam_manifest_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='Same-arm versioned'):
        strict_resume(payload, head, optimizer, scheduler, changed, 'teacher')
