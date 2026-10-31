# -*- coding: utf-8 -*-
"""虚构同源图经真实裁窗／退化／翻转后，修补只能扩展人工监督。"""
from copy import deepcopy

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.backend_adaptation import CanonicalBackendDataset, PairedDegradationDataset
from train_affinity_connectivity import draw_receipt
from train_affinity_gt_source import IndependentMixedDataset, PairGeometrySource, batch_signature
from utils.teacher_seam_support import repair_teacher_zero_seams
from utils.teacher_seam_training import audit_repaired_batch, compare_repaired_rows


@pytest.fixture
def synthetic_pair(tmp_path):
    raw, gt, pairs = (tmp_path/name for name in ('raw', 'gt', 'pairs'))
    for folder in (raw, gt, pairs):
        folder.mkdir()
    rows = []
    allowlist = ['train_%03d' % i for i in range(32)]
    for index in range(2):
        stem = allowlist[index]
        y, x = np.mgrid[:48, :64]
        rgb = np.stack((x*3, y*4, (x+y)*2), -1).astype(np.uint8)
        image = raw/(stem+'.png')
        cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        image.with_suffix('.json').write_text('{}')
        manual = np.where(x < 32, 1, 2).astype(np.uint16)
        manual[:3, :3] = 0
        teacher = np.where(x < 32, 7, 9).astype(np.uint16)
        teacher[:, 32] = 0
        original = manual > 0
        full = gt/(stem+'_gt.npz')
        np.savez(full, instance_map=manual, original_covered=original,
                 filled=np.zeros_like(original))
        (gt/(stem+'_class.json')).write_text('{"1":1,"2":0}')
        common = original & (teacher > 0)
        old = pairs/(stem+'.npz')
        np.savez(old, manual_ids=np.where(common, manual, 0),
                 teacher_ids=np.where(common, teacher, 0), common_known=common,
                 original_covered=original, filled=np.zeros_like(original))
        repaired, _ = repair_teacher_zero_seams(manual, teacher, original,
            source_name=stem, allowed_training_sources=allowlist)
        new = pairs/(stem+'_seam.npz')
        np.savez(new, teacher_ids=repaired['teacher_masked'])
        rows.append(dict(source=stem, image=image.relative_to(tmp_path).as_posix(),
                         manual_gt=full.relative_to(tmp_path).as_posix(),
                         pair_npz=old.relative_to(tmp_path).as_posix(),
                         seam_pair_npz=new.relative_to(tmp_path).as_posix()))
    config = dict(paths=dict(project_root=str(tmp_path)), data=dict(image_size=32),
                  gt_source=dict(seed=20261004),
                  affinity_native=dict(crop_attempts=8, minimum_known_pairs=4))
    base = CanonicalBackendDataset(raw, gt, image_size=32, affinity_grid=16)
    parent = PairedDegradationDataset(base,
        dict(profile_probabilities=[.2,.8,0,0], blur_sigma=[.4,1.5],
             resample_probability=0., resample_scale=[.5,.9]),
        dict(enabled=True, probability=.5, sigma=[0,.006]), seed=42, repeats=2)
    return config, rows, parent


def test_actual_augmented_dataset_preserves_control_input_and_old_answers(synthetic_pair):
    config, rows, parent = synthetic_pair
    legacy = [{k:v for k,v in row.items() if k != 'seam_pair_npz'} for row in rows]
    old = IndependentMixedDataset(parent, 'manual', config, pair_rows=legacy, arm='teacher')
    new = IndependentMixedDataset(parent, 'manual', config, pair_rows=rows, arm='teacher')
    added = 0
    for epoch in (1, 2):
        old.set_epoch(epoch); new.set_epoch(epoch)
        old_loader = DataLoader(old, batch_size=1)
        new_loader = DataLoader(new, batch_size=1)
        for step, (before, after) in enumerate(zip(old_loader, new_loader), start=1):
            audit = audit_repaired_batch(after, config, {r['source']:r for r in rows})
            original = dict(epoch=epoch, step=step, seed=42, manual=before['name'],
                pseudo=['train_999.png'], manual_draw=draw_receipt(before), pseudo_draw={},
                manual_crop=before['crop_box'].tolist(), pseudo_crop=[[0,0,48,64]],
                training_view=before['training_view'][0],
                inputs=[batch_signature(before,before['image']), batch_signature(before,before['image'])])
            variant = deepcopy(original)
            variant.update(manual=after['name'], manual_draw=draw_receipt(after),
                manual_crop=after['crop_box'].tolist(), training_view=after['training_view'][0],
                seam_support=audit)
            variant['inputs'][0] = batch_signature(after,after['image'])
            compare_repaired_rows(variant, original)
            added += audit['added_edges']
    assert added > 0


def test_native_crop_position_still_depends_on_old_shared_support(synthetic_pair):
    config, rows, _ = synthetic_pair
    legacy = [{k:v for k,v in row.items() if k != 'seam_pair_npz'} for row in rows]
    old = PairGeometrySource(config, legacy, 'teacher', image_size=32, grid=16, seed=42)
    new = PairGeometrySource(config, rows, 'teacher', image_size=32, grid=16, seed=42)
    for draw in range(8):
        for dataset in (old, new):
            dataset.size=16; dataset.epoch=2; dataset.draw=draw
        before, after = old[0], new[0]
        assert torch.equal(before['crop_box'], after['crop_box'])
        assert before['crop_attempt'] == after['crop_attempt']
        assert torch.equal(before['image'], after['image'])
